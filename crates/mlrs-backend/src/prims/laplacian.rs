//! `laplacian` — normalized graph-Laplacian primitive (PRIM-09).
//!
//! Given a ready affinity matrix `A` (`n×n` row-major, RESEARCH Open Q2 — the
//! ESTIMATOR builds the affinity via `kernel_matrix(Rbf)` or the kNN-connectivity
//! graph), this prim returns the pair `(L, dd)` of the symmetric normalized
//! Laplacian
//!   `L = I − D^-1/2 · A · D^-1/2`
//! and the degree-normalization vector `dd[i] = sqrt(degree_i)` (with a typed-zero
//! guard `dd[i] = 1` for an isolated zero-degree node so the recovery never
//! divides by zero). The estimator divides each recovered eigenvector by `dd` for
//! the `D^-1/2` diffusion-map recovery (D-07), which is why `dd` is returned
//! alongside `L`.
//!
//! ## The 4-step scipy `_laplacian_dense` host orchestration (RESEARCH Pattern 1)
//!   1. zero the diagonal of `A` (an isolated/self edge is excluded from the
//!      degree — scipy `fill_diagonal(m, 0)` BEFORE the degree reduction).
//!   2. `w = row_reduce(A, n, n, ScalarOp::Sum)` — the degree vector, a SINGLE
//!      single-owner row reduction (GATHER, no scatter / no atomics — PRIM-09).
//!   3. `dd[i] = if w[i] == 0 { 1 } else { sqrt(w[i]) }` — the typed-zero guard
//!      that replaces the would-be `1/sqrt(0)` infinite value.
//!   4. `L[i,j] = -A[i,j] / (dd[i]·dd[j])`; `L[i,i] = if w[i] == 0 { 0 } else { 1 }`
//!      — ONE [`laplacian_map`] pass over the affinity buffer (the diagonal of an
//!      isolated node is `0`, the "no NaN / no infinite value on zero-degree
//!      nodes" success criterion).
//!
//! ## Composition (the kernel_matrix.rs base-op → in-place-map idiom)
//! Like [`crate::prims::kernel_matrix`], `laplacian` is a thin host orchestration
//! over already-validated v1 base ops ([`row_reduce`]) plus one shared-memory-free
//! per-element map kernel ([`laplacian_map`], added in `mlrs-kernels`). The dense
//! `n×n` `L` stays in GLOBAL memory (no LDS tile; gfx1100 LDS ≤ 65536 B).
//!
//! ## Validate-before-launch (ASVS V5)
//! The geometry guard (`a.len() == n*n`, `n != 0`) runs BEFORE any launch,
//! returning a typed [`PrimError`]. The `n ≤ 64` MAX_DIM cap is the ESTIMATOR's
//! job (D-06 — [`mlrs-algos`] `AlgoError::NSamplesExceedsMaxDim`), NOT the prim;
//! `laplacian.rs` stays cap-agnostic exactly like `kernel_matrix.rs`.
//!
//! ## Status (Wave-1 / plan 09-02 — FILLED)
//! The compute path is implemented: a `zero_diag_copy` pass (step 1), the
//! `row_reduce(Sum)` degree GATHER (step 2), the `degree_guard` typed-zero `dd`
//! (step 3), and the single `laplacian_map` build pass (step 4). The geometry
//! guard runs before any launch; the `n ≤ 64` MAX_DIM cap remains the ESTIMATOR's
//! job (D-06), NOT this prim. Both new map kernels are shared-memory-free,
//! atomics-free, and free of the infinite-value constant (the cpu-MLIR-safe
//! profile).
//!
//! Tests live in `crates/mlrs-backend/tests/laplacian_test.rs` (AGENTS.md §2 —
//! no in-source `#[cfg(test)] mod tests`).

use std::mem::size_of;

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::{degree_guard, laplacian_map, zero_diag_copy};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::reduce::{row_reduce, ReducePath, ScalarOp};
use crate::runtime::ActiveRuntime;

/// Compute the symmetric normalized Laplacian of a ready affinity matrix
/// (PRIM-09). Returns the pair `(L, dd)`:
///
/// - `L` is the `n×n` row-major normalized Laplacian `I − D^-1/2 A D^-1/2`,
///   device-resident (acquired from `pool`).
/// - `dd` is the length-`n` degree-normalization vector `dd[i] = sqrt(degree_i)`
///   (or `1` for an isolated zero-degree node — the typed-zero guard, NO infinite
///   value), device-resident. The caller divides each recovered eigenvector by
///   `dd` for the `D^-1/2` recovery (D-07).
///
/// `a` is the ready `n×n` affinity (the estimator builds it via
/// `kernel_matrix(Rbf)` or the kNN-connectivity graph). Geometry is validated
/// against `a.len()` and `n` **before** any launch (`a.len() == n*n`, `n != 0`);
/// a mismatch returns a typed [`PrimError`] (ASVS V5). The `n ≤ 64` MAX_DIM cap
/// is the ESTIMATOR's job (D-06), NOT this prim.
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
///
/// The 4-step path is: [`zero_diag_copy`] → [`row_reduce`]`(Sum)` degree →
/// [`degree_guard`] typed-zero `dd` → [`laplacian_map`] build. The result stays
/// device-resident; the transient diagonal-zeroed working buffer and the degree
/// vector are released back to `pool`, while `dd` is returned alongside `L`.
pub fn laplacian<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    n: usize,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    ),
    PrimError,
>
where
    F: Float + CubeElement + Pod,
{
    // --- Geometry guard (ASVS V5): validate BEFORE any unsafe launch so a wrong
    //     shape is a recoverable typed error, not an out-of-bounds device read.
    //     Mirrors kernel_matrix.rs:133-142 / eig.rs squareness guard. The n ≤ 64
    //     MAX_DIM cap is the ESTIMATOR's job (D-06), NOT this prim. ---
    validate_geometry(a.len(), n)?;

    let nn = n * n;
    let elem = size_of::<F>();
    let client = pool.client().clone();

    // --- Step 1: zero the diagonal of A into a fresh working buffer `m`
    //     (scipy `np.fill_diagonal(m, 0)` BEFORE the degree — RESEARCH "Affinity
    //     diagonal handling"). Non-in-place so the caller's `a` is never mutated.
    //     The dense n×n `m` stays in GLOBAL memory (no LDS tile). ---
    let m_handle = pool.acquire(nn * elem);
    {
        let (count, dim) = launch_dims_1d(nn);
        // SAFETY: `nn == a.len()` (validated geometry); the kernel bounds-checks
        // `tid < a.len()`. Distinct in/out handles (non-in-place copy).
        let a_arg = unsafe { ArrayArg::from_raw_parts(a.handle().clone(), nn) };
        let m_arg = unsafe { ArrayArg::from_raw_parts(m_handle.clone(), nn) };
        zero_diag_copy::launch::<F, ActiveRuntime>(&client, count, dim, a_arg, m_arg, n as u32);
    }
    let m: DeviceArray<ActiveRuntime, F> = DeviceArray::from_raw(m_handle, nn);

    // --- Step 2: degree vector `w = row_reduce(m, Sum)` — the single-owner GATHER
    //     row reduction (no scatter, no atomics). Force the always-portable Shared
    //     path (never plane-gated to None on a non-subgroup adapter like cpu). ---
    let w = row_reduce::<F>(pool, &m, n, n, ScalarOp::Sum, ReducePath::Shared)?
        .expect("shared path is never plane-gated to None");

    // --- Step 3: `dd[i] = if w[i] == 0 { 1 } else { sqrt(w[i]) }` — the typed-zero
    //     guard (NO infinite value). Device-resident length-n output. ---
    let dd_handle = pool.acquire(n * elem);
    {
        let (count, dim) = launch_dims_1d(n);
        // SAFETY: `w.len() == n`; the kernel bounds-checks `tid < w.len()`.
        let w_arg = unsafe { ArrayArg::from_raw_parts(w.handle().clone(), n) };
        let dd_arg = unsafe { ArrayArg::from_raw_parts(dd_handle.clone(), n) };
        degree_guard::launch::<F, ActiveRuntime>(&client, count, dim, w_arg, dd_arg);
    }
    let dd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_raw(dd_handle, n);

    // --- Step 4: `L = laplacian_map(m, w, dd, n)` — off-diagonal -m/(dd_i·dd_j),
    //     diagonal `1 - isolated` (`w[i] == 0 ? 0 : 1`). The dd divisor is GATHERed
    //     by row/column index (no scatter/atomics); the typed-zero guard on `dd`
    //     makes the division infinity-free. Fresh n×n output buffer. ---
    let l_handle = pool.acquire(nn * elem);
    {
        let (count, dim) = launch_dims_1d(nn);
        // SAFETY: all lengths are the carried element counts (validated); the
        // kernel bounds-checks `tid < a.len()`. `m`/`w`/`dd` are read-only inputs,
        // `l_handle` is the distinct output.
        let m_arg = unsafe { ArrayArg::from_raw_parts(m.handle().clone(), nn) };
        let w_arg = unsafe { ArrayArg::from_raw_parts(w.handle().clone(), n) };
        let dd_arg = unsafe { ArrayArg::from_raw_parts(dd.handle().clone(), n) };
        let l_arg = unsafe { ArrayArg::from_raw_parts(l_handle.clone(), nn) };
        laplacian_map::launch::<F, ActiveRuntime>(
            &client, count, dim, m_arg, w_arg, dd_arg, l_arg, n as u32,
        );
    }
    let l: DeviceArray<ActiveRuntime, F> = DeviceArray::from_raw(l_handle, nn);

    // --- The diagonal-zeroed working `m` and the degree `w` are TRANSIENT scratch
    //     — both consumed by the laplacian_map launch above and never read again.
    //     Release each at its TRUE byte size so `live_bytes` conserves across
    //     repeated calls (the PoolStats memory gate). `dd` is RETURNED alongside
    //     `L` (the estimator divides each recovered eigenvector by `dd` — D-07). ---
    m.release_into(pool);
    w.release_into(pool);

    Ok((l, dd))
}

/// Standard ceiling-division 1D launch config for the per-element passes (the
/// `elementwise` per-element launch idiom; mirrors `kernel_matrix.rs`).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256usize;
    let cubes = u32::try_from(n.div_ceil(block))
        .expect("element count exceeds u32 launch-grid limit");
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim {
            x: block as u32,
            y: 1,
            z: 1,
        },
    )
}

/// Validate the affinity geometry (ASVS V5): `a` must be a well-formed `n×n`
/// matrix (`n*n == a.len()`) and non-empty (`n != 0`). Empty / non-square
/// geometry is rejected at the boundary (no well-defined Laplacian).
fn validate_geometry(a_len: usize, n: usize) -> Result<(), PrimError> {
    if n == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "a",
            rows: n,
            cols: n,
            len: a_len,
        });
    }
    if n.checked_mul(n).map(|v| v != a_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "a",
            rows: n,
            cols: n,
            len: a_len,
        });
    }
    Ok(())
}
