//! `kernel_matrix` — pairwise kernel-matrix primitive (PRIM-08).
//!
//! Computes the general kernel matrix `K(X, Y)` (an `rows_x × rows_y` matrix,
//! D-02) for the four KernelRidge kernel families (D-01):
//!   - `linear`:  `K = X·Yᵀ` (the GEMM base op, identity map).
//!   - `rbf`:     `K = exp(-γ·‖xᵢ − yⱼ‖²)` (squared-euclidean distance base op,
//!                then `exp(-γ··)` map).
//!   - `poly`:    `K = (γ·⟨xᵢ, yⱼ⟩ + coef0)^degree` (GEMM base op, then powf map).
//!   - `sigmoid`: `K = tanh(γ·⟨xᵢ, yⱼ⟩ + coef0)` (GEMM base op, then tanh map).
//!
//! ## Composition (the covariance.rs base-op → in-place-map idiom)
//! Like [`crate::prims::covariance`], `kernel_matrix` is a thin host
//! orchestration over already-validated v1 base ops plus one SharedMemory-free
//! per-element map kernel (added in `mlrs-kernels` by the Wave-1 plan):
//!   - `linear`/`poly`/`sigmoid` use [`crate::prims::gemm::gemm`] (`transb =
//!     true`, the `X·Yᵀ` base) as the base op (`gemm.rs:54`).
//!   - `rbf` uses [`crate::prims::distance::distance`] (`sqrt = false`, the
//!     squared-euclidean base, `distance.rs:79`).
//! The per-kernel map then runs IN PLACE over the base buffer (input handle ==
//! output handle), exactly the covariance `scale`-in-place idiom
//! (`covariance.rs:190-204`); the result IS the base buffer, mapped in place
//! (D-02/D-03 single code path). `linear` is the identity map — it skips the map
//! launch and returns the GEMM buffer directly.
//!
//! ## Validate-before-launch (ASVS V5 / T-08-01-01)
//! The geometry guard (`rows_x·cols == x.len()`, `rows_y·cols == y.len()`,
//! reject empty geometry, `out` len == `rows_x·rows_y`) runs BEFORE any `unsafe`
//! kernel launch, returning a typed [`PrimError`], never an out-of-bounds device
//! read — the same contract as `covariance.rs:212-262` / `gemm.rs`.
//!
//! ## Wave-0 scaffold status
//! This file is the 08-01 Wave-0 COMPILING STUB: it defines the public surface
//! (`Kernel<F>` enum + the `kernel_matrix` host-fn signature) that the Wave-0
//! test scaffold compiles against, with the geometry validation REAL (so the
//! signature and error type are real) but the compute path left as `todo!()` for
//! the Wave-1 plan (08-02) to fill (it adds the `mlrs-kernels` map kernel + the
//! base-op dispatch). Do NOT write the map kernel here — that is Wave 1.
//!
//! Tests live in `crates/mlrs-backend/tests/kernel_matrix_test.rs` (AGENTS.md §2
//! — no in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::{poly_map, rbf_map, sigmoid_map};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::distance::distance;
use crate::prims::gemm::gemm;
use crate::runtime::ActiveRuntime;

/// The typed kernel-family selector (D-01) `kernel_matrix` matches on to pick the
/// base op + per-element map. Generic over the float element type `F` so the
/// kernel hyperparameters (`gamma` / `degree` / `coef0`) carry the same precision
/// as the data, matching the `<F: Float + CubeElement + Pod>` bound the prim
/// functions use (`covariance.rs`).
///
/// `degree` is stored as `F` (not an integer) because sklearn's poly kernel takes
/// a real degree (`Interval(Real, 1, None)`) and the map uses `F::powf` — the
/// sklearn-faithful real-exponent form (RESEARCH §kernel_matrix.rs / Pitfall 7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel<F>
where
    F: Float + CubeElement + Pod,
{
    /// Linear kernel `K = X·Yᵀ` — the GEMM base op, identity map (no map launch).
    Linear,
    /// RBF (Gaussian) kernel `K = exp(-γ·‖xᵢ − yⱼ‖²)` — squared-euclidean
    /// distance base op, then the `exp(-γ··)` map.
    Rbf {
        /// Kernel coefficient `γ` (resolved to `1/n_features` by the caller when
        /// sklearn's `gamma=None` default is requested — D-05).
        gamma: F,
    },
    /// Polynomial kernel `K = (γ·⟨xᵢ, yⱼ⟩ + coef0)^degree` — GEMM base op, then
    /// the `powf(γ·g + coef0, degree)` map.
    Poly {
        /// Kernel coefficient `γ`.
        gamma: F,
        /// Polynomial degree (real, `≥ 1` — validated by the estimator).
        degree: F,
        /// Independent term `coef0`.
        coef0: F,
    },
    /// Sigmoid kernel `K = tanh(γ·⟨xᵢ, yⱼ⟩ + coef0)` — GEMM base op, then the
    /// `tanh(γ·g + coef0)` map.
    Sigmoid {
        /// Kernel coefficient `γ`.
        gamma: F,
        /// Independent term `coef0`.
        coef0: F,
    },
}

/// Compute the general kernel matrix `K(X, Y)` (D-02): an `rows_x × rows_y`
/// row-major matrix whose `(i, j)` entry is the chosen [`Kernel`] applied to the
/// `i`-th row of `X` and the `j`-th row of `Y`. Both operands are row-major
/// `(rows, cols)` device buffers sharing the feature dimension `cols`.
///
/// - `x` is the `rows_x × cols` left operand; `y` is the `rows_y × cols` right
///   operand (for the symmetric training Gram `K(X, X)` the caller passes `y =
///   x`, D-02).
/// - Geometry is validated against the operand element counts **before** any
///   launch (`rows_x·cols == x.len()`, `rows_y·cols == y.len()`, non-empty,
///   `out` len == `rows_x·rows_y`); a mismatch returns
///   [`PrimError::ShapeMismatch`] / [`PrimError::DimMismatch`] (ASVS V5 /
///   T-08-01-01).
/// - The `rows_x × rows_y` output is acquired from `pool` when `out` is `None`,
///   else the supplied buffer is reused (D-11). The result stays device-resident
///   (D-05) — NO host round-trip inside this API.
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
///
/// **Wave-0 stub:** geometry validation is real; the compute path is `todo!()`
/// pending the Wave-1 plan (08-02), which adds the `mlrs-kernels` map kernel and
/// the base-op (`gemm`/`distance`) dispatch.
#[allow(clippy::too_many_arguments)]
pub fn kernel_matrix<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    (rows_x, cols): (usize, usize),
    y: &DeviceArray<ActiveRuntime, F>,
    (rows_y, cols_y): (usize, usize),
    kernel: Kernel<F>,
    out: Option<DeviceArray<ActiveRuntime, F>>,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- Geometry guard (ASVS V5 / T-08-01-01): validate BEFORE any unsafe
    //     launch so a wrong shape is a recoverable typed error, not an
    //     out-of-bounds device read. Mirrors covariance.rs:212-262 / gemm.rs. ---
    validate_geometry(
        x.len(),
        (rows_x, cols),
        y.len(),
        (rows_y, cols_y),
        out.as_ref().map(DeviceArray::len),
    )?;

    // --- Base-op dispatch + in-place map (the covariance.rs:151-204 idiom).
    //       linear  → gemm(x, y, transb=true), return directly (identity map).
    //       rbf     → distance(x, y, sqrt=false), then exp(-gamma··) map in place.
    //       poly    → gemm(x, y, transb=true), then powf(gamma·g+coef0, degree).
    //       sigmoid → gemm(x, y, transb=true), then tanh(gamma·g+coef0).
    //     The map runs IN PLACE over the base buffer (covariance.rs:190-204) —
    //     the result IS the base buffer, mapped in place (D-02/D-03). The full
    //     general rows_x × rows_y K(X, Y) is always computed (D-02 — no symmetry
    //     special-case). The n×n operand stays in GLOBAL memory (no SharedMemory
    //     tile; gfx1100 LDS ≤ 65536 B — T-08-02-02). ---
    match kernel {
        // Linear: K = X·Yᵀ. The GEMM buffer IS the kernel matrix (identity map),
        // so we return it directly with NO map launch.
        Kernel::Linear => {
            // logical lhs (m, k) = (rows_x, cols); rhs (k, n) = (cols, rows_y),
            // transb=true ⇒ stored y is (rows_y, cols) = `y`'s layout.
            gemm::<F>(pool, x, (rows_x, cols), y, (cols, rows_y), false, true, out)
        }
        // RBF: squared-euclidean base (sqrt=false ⇒ ‖xᵢ − yⱼ‖²), then the
        // exp(-γ··) map in place over that base buffer (D-03 / Pitfall 4).
        Kernel::Rbf { gamma } => {
            let base = distance::<F>(pool, x, (rows_x, cols), y, (rows_y, cols_y), false, out)?;
            launch_map_in_place(pool, &base, rows_x * rows_y, |client, count, dim, in_arg, out_arg| {
                rbf_map::launch::<F, ActiveRuntime>(client, count, dim, in_arg, out_arg, gamma);
            });
            Ok(base)
        }
        // Poly: XYᵀ Gram base, then powf(γ·g + coef0, degree) map in place.
        Kernel::Poly { gamma, degree, coef0 } => {
            let base =
                gemm::<F>(pool, x, (rows_x, cols), y, (cols, rows_y), false, true, out)?;
            launch_map_in_place(pool, &base, rows_x * rows_y, |client, count, dim, in_arg, out_arg| {
                poly_map::launch::<F, ActiveRuntime>(
                    client, count, dim, in_arg, out_arg, gamma, coef0, degree,
                );
            });
            Ok(base)
        }
        // Sigmoid: XYᵀ Gram base, then tanh(γ·g + coef0) map in place.
        Kernel::Sigmoid { gamma, coef0 } => {
            let base =
                gemm::<F>(pool, x, (rows_x, cols), y, (cols, rows_y), false, true, out)?;
            launch_map_in_place(pool, &base, rows_x * rows_y, |client, count, dim, in_arg, out_arg| {
                sigmoid_map::launch::<F, ActiveRuntime>(
                    client, count, dim, in_arg, out_arg, gamma, coef0,
                );
            });
            Ok(base)
        }
    }
}

/// Launch a per-element map IN PLACE over `base` (input handle == output handle),
/// the covariance.rs:190-204 scale-in-place idiom. `n` is the element count
/// (`rows_x · rows_y`); the closure receives the client + launch dims + the
/// in/out `ArrayArg`s (both wrapping the SAME `base` handle) so the map rewrites
/// the base buffer with no parallel allocation (T-08-02-02 — the in-place map
/// reuses the base buffer).
fn launch_map_in_place<F, L>(
    pool: &mut BufferPool<ActiveRuntime>,
    base: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    launch: L,
) where
    F: Float + CubeElement + Pod,
    L: FnOnce(
        &cubecl::client::ComputeClient<ActiveRuntime>,
        CubeCount,
        CubeDim,
        ArrayArg<ActiveRuntime>,
        ArrayArg<ActiveRuntime>,
    ),
{
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(n);
    // SAFETY: `n` is the carried base-op output element count (rows_x · rows_y,
    // itself derived from the validated geometry); each map kernel bounds-checks
    // `tid < input.len()` (T-08-02-01). input and output are the SAME handle so
    // the map is applied in place over the reused base buffer (no parallel
    // allocation — T-08-02-02).
    let in_arg = unsafe { ArrayArg::from_raw_parts(base.handle().clone(), n) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(base.handle().clone(), n) };
    launch(&client, count, dim, in_arg, out_arg);
}

/// Standard ceiling-division 1D launch config for the in-place map pass (the
/// `elementwise` per-element launch idiom, copied from covariance.rs:266-273).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256usize;
    // Compute the cube count in `usize` and check the `u32` launch-grid cast
    // (WR-02): an unchecked `n as u32` silently wraps for `n > u32::MAX`,
    // under-provisioning threads so trailing elements are never mapped — a silent
    // wrong-result. The kernel-matrix problem sizes are small today, but the guard
    // turns the overflow into a loud panic instead.
    let cubes = u32::try_from((n + block - 1) / block)
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

/// Validate the kernel-matrix operand geometry (ASVS V5 / T-08-01-01). `x` is
/// `rows_x × cols`, `y` is `rows_y × cols` (the SHARED feature dimension must
/// agree); the output (if supplied) must be the `rows_x × rows_y` kernel matrix.
/// Empty geometry is rejected at the boundary (a 0-row / 0-col operand has no
/// well-defined kernel matrix).
fn validate_geometry(
    x_len: usize,
    (rows_x, cols): (usize, usize),
    y_len: usize,
    (rows_y, cols_y): (usize, usize),
    out_len: Option<usize>,
) -> Result<(), PrimError> {
    // x must be a well-formed rows_x × cols.
    if rows_x
        .checked_mul(cols)
        .map(|v| v != x_len)
        .unwrap_or(true)
    {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: rows_x,
            cols,
            len: x_len,
        });
    }
    // y must be a well-formed rows_y × cols_y.
    if rows_y
        .checked_mul(cols_y)
        .map(|v| v != y_len)
        .unwrap_or(true)
    {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: rows_y,
            cols: cols_y,
            len: y_len,
        });
    }
    // The shared feature dimension must agree (K(X, Y) is only defined when X and
    // Y live in the same feature space).
    if cols != cols_y {
        return Err(PrimError::DimMismatch {
            dim: "n_features",
            lhs: cols,
            rhs: cols_y,
        });
    }
    // Reject empty geometry at the boundary (no well-defined kernel matrix).
    if rows_x == 0 || rows_y == 0 || cols == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: rows_x,
            cols,
            len: x_len,
        });
    }
    // The output (if supplied) must be the rows_x × rows_y kernel matrix.
    if let Some(o) = out_len {
        let expect = rows_x * rows_y;
        if o != expect {
            return Err(PrimError::ShapeMismatch {
                operand: "out",
                rows: rows_x,
                cols: rows_y,
                len: o,
            });
        }
    }
    Ok(())
}
