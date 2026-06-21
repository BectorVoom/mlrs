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
//! ## Wave-0 scaffold status
//! This file is the 09-01 Wave-0 COMPILING STUB: it defines the public surface
//! (the `laplacian` host-fn signature returning `(L, dd)`) that the Wave-0 test
//! scaffold compiles against, with the geometry validation REAL (so the signature
//! and error type are real) but the compute path left as `todo!()` for the Wave-1
//! plan (09-02) to fill (it adds the diagonal-zero + degree-reduce + `dd` guard +
//! `laplacian_map` orchestration). Do NOT write that compute here — it is Wave 1.
//!
//! Tests live in `crates/mlrs-backend/tests/laplacian_test.rs` (AGENTS.md §2 —
//! no in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
#[allow(unused_imports)] // Wave-1 (09-02) consumes the map kernel; the stub
                         // references it in the module docs so the seam is fixed.
use mlrs_kernels::laplacian_map;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
#[allow(unused_imports)] // Wave-1 (09-02) consumes the degree row reduction.
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
/// **Wave-0 stub:** geometry validation is real; the compute path is `todo!()`
/// pending the Wave-1 plan (09-02), which adds the diagonal-zero + degree-reduce
/// + `dd` typed-zero guard + [`laplacian_map`] orchestration.
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

    // --- Compute path (Wave-1 / plan 09-02): zero the diagonal of A, reduce the
    //     degree (row_reduce(Sum)), build dd with the typed-zero guard, then the
    //     single laplacian_map pass over the affinity buffer. Left as a stub here
    //     (Wave-0 lands only the validated signature + the (L, dd) return type).
    //     `pool` is the buffer source the filled path acquires L / dd from. ---
    let _ = pool;
    todo!("laplacian compute is filled by the Wave-1 plan 09-02 (PRIM-09)")
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
