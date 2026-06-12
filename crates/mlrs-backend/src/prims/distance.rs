//! Pairwise squared-Euclidean distance host API (PRIM-03) — the GEMM-expansion
//! `‖x_i‖² + ‖y_j‖² − 2·XYᵀ` with an unconditional `max(d², 0)` clamp and an
//! optional sqrt boundary, composing the Plan-01 GEMM and the Plan-02 row
//! squared-norm reduction.
//!
//! ## Why GEMM-expansion (D-07)
//! For `X` (`rows_x × cols`) and `Y` (`rows_y × cols`), the squared Euclidean
//! distance `d²(x_i, y_j) = ‖x_i‖² + ‖y_j‖² − 2·x_i·y_j`. The cross term is the
//! whole `XYᵀ` matrix (one GEMM with `transb=true`); the two norm terms are
//! per-row squared norms `‖x_i‖² = Σ_k X[i,k]²` (the Plan-02 row reduction with
//! [`ScalarOp::SumSq`] — the SQUARED norm, no sqrt). Reusing the validated GEMM
//! and reduction is the single-validated-kernel mandate (one distance serves
//! KMeans, DBSCAN, KNN).
//!
//! ## The clamp produces NO negative distances (Criterion 3 / Pitfall 5)
//! In f32, `‖x_i‖² + ‖y_j‖² − 2·x_i·y_j` for near-identical rows is a
//! catastrophic cancellation that can land slightly negative. The
//! `dist_combine_clamp` kernel applies `max(d², 0)` (STATEMENT form) UNCONDITIONALLY,
//! so the squared distance is never negative and the optional sqrt never sees a
//! negative argument (T-0203-03). The `distance_min_nonnegative` property test
//! pins this on a deliberate cancellation case.
//!
//! ## Squared is the core output; sqrt is the boundary (D-08)
//! [`distance`] returns the clamped SQUARED distance by default; passing
//! `sqrt = true` applies [`sqrt_elem`] in place at the boundary so KNN gets true
//! Euclidean distances. Squaring is the cheaper, sufficient form for the
//! distance-comparison consumers (KMeans/DBSCAN), so sqrt is opt-in.
//!
//! ## Device residency (D-05 / D-10 gate 2)
//! Inputs and every intermediate (`XYᵀ`, the two norm vectors, the clamped
//! output, the optional in-place sqrt) stay on the device as [`DeviceArray`]s.
//! This module performs NO host read-back between stages (the device-residency
//! grep gate over this file is `0`). Scratch + the output buffer are drawn from
//! the [`BufferPool`] (D-11).
//!
//! Tests live in `crates/mlrs-backend/tests/distance_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::{dist_combine_clamp, sqrt_elem};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::gemm::gemm;
use crate::prims::reduce::{row_reduce, ReducePath, ScalarOp};
use crate::runtime::ActiveRuntime;

/// Compute the pairwise squared-Euclidean distance matrix `D` (`rows_x ×
/// rows_y`) between the rows of `x` (`rows_x × cols`) and `y` (`rows_y × cols`)
/// via the GEMM-expansion `‖x_i‖² + ‖y_j‖² − 2·XYᵀ`, clamped to `max(d², 0)`.
///
/// - `x` is the row-major `rows_x × cols` left operand; `y` is `rows_y × cols`.
///   Both share the feature dimension `cols`.
/// - Shapes are validated (`rows_x*cols == x.len()`, `rows_y*cols == y.len()`)
///   BEFORE any launch (D-04 / T-0203-02); a mismatch returns
///   [`PrimError::ShapeMismatch`].
/// - `sqrt = true` applies the optional Euclidean sqrt at the boundary (D-08);
///   `sqrt = false` returns the squared distance (the core output).
/// - The `rows_x × rows_y` result is acquired from `pool` when `out` is `None`,
///   else the supplied buffer is reused (D-11). The result stays device-resident
///   (D-05) — NO host round-trip inside this API.
/// - `path` selects the reduction kernel family for the norm terms
///   ([`ReducePath::Shared`] is always portable; [`ReducePath::Plane`] is the
///   subgroup fast path where supported).
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
#[allow(clippy::too_many_arguments)]
pub fn distance<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    (rows_x, cols): (usize, usize),
    y: &DeviceArray<ActiveRuntime, F>,
    (rows_y, cols_y): (usize, usize),
    sqrt: bool,
    out: Option<DeviceArray<ActiveRuntime, F>>,
    path: ReducePath,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- D-04 / T-0203-02: validate geometry BEFORE any unsafe launch. ---
    validate_geometry(x.len(), (rows_x, cols), y.len(), (rows_y, cols_y), out.as_ref().map(DeviceArray::len))?;

    // --- 1. XYᵀ via GEMM(transb=true): (rows_x × cols)·(cols × rows_y) →
    //        rows_x × rows_y. `y` is stored (rows_y × cols); transb reads it as
    //        its transpose (cols × rows_y) with no transpose buffer (D-06). ---
    let xy = gemm::<F>(
        pool,
        x,
        (rows_x, cols),
        y,
        // logical rhs shape (k, n) = (cols, rows_y); transb=true ⇒ stored (rows_y, cols).
        (cols, rows_y),
        false,
        true,
        None,
    )?;

    // --- 2. Per-row SQUARED norms ‖x_i‖² (len rows_x) and ‖y_j‖² (len rows_y)
    //        via the Plan-02 row reduction with SumSq (NO sqrt — distance needs
    //        the squared norm directly). Device-resident outputs. ---
    let xnorm = row_reduce::<F>(pool, x, rows_x, cols, ScalarOp::SumSq, path)?
        .expect("row SumSq reduction is shared-path-backed (never plane-gated to None)");
    let ynorm = row_reduce::<F>(pool, y, rows_y, cols, ScalarOp::SumSq, path)?
        .expect("row SumSq reduction is shared-path-backed (never plane-gated to None)");

    // --- 3. Combine + clamp: out[i,j] = max(‖x_i‖² + ‖y_j‖² − 2·XYᵀ[i,j], 0).
    //        Device-resident; the clamp guarantees no negative squared distance
    //        (Criterion 3). Output reuses the caller's buffer (D-11) or a pool
    //        acquisition. ---
    let out_len = rows_x * rows_y;
    let elem = size_of::<F>();
    let out_handle = match &out {
        Some(o) => o.handle().clone(),
        None => pool.acquire(out_len * elem),
    };

    let client = pool.client().clone();
    let (count, dim) = launch_dims_2d(rows_x, rows_y);

    // SAFETY: lengths are the carried DeviceArray element counts (themselves
    // derived from validated host slices); the kernel bounds-checks
    // `i < rows && j < cols` (mitigates T-0203-01).
    let xy_arg = unsafe { ArrayArg::from_raw_parts(xy.handle().clone(), out_len) };
    let xn_arg = unsafe { ArrayArg::from_raw_parts(xnorm.handle().clone(), rows_x) };
    let yn_arg = unsafe { ArrayArg::from_raw_parts(ynorm.handle().clone(), rows_y) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };

    dist_combine_clamp::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        xy_arg,
        xn_arg,
        yn_arg,
        out_arg,
        // Scalar args are passed by value in cubecl 0.10 (no ScalarArg wrapper —
        // see spike_test.rs), like `saxpy_kernel`'s `a: F`.
        rows_x as u32,
        rows_y as u32,
    );

    // --- 4. Optional Euclidean sqrt at the boundary (D-08), in place over the
    //        already-clamped (non-negative) buffer — so sqrt never sees a
    //        negative argument. Still device-resident. ---
    if sqrt {
        let (scount, sdim) = launch_dims_1d(out_len);
        let in_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };
        let sout_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };
        sqrt_elem::launch::<F, ActiveRuntime>(&client, scount, sdim, in_arg, sout_arg);
    }

    // The result stays device-resident (D-05); the caller reads it back via the
    // DeviceArray read-back methods at the boundary when needed.
    Ok(DeviceArray::from_raw(out_handle, out_len))
}

/// Validate distance operand geometry (D-04 / T-0203-02). `x` is `rows_x ×
/// cols`, `y` is `rows_y × cols_y`; the two feature dimensions must agree, and
/// each `rows*cols == len`. The output (if supplied) must be `rows_x × rows_y`.
fn validate_geometry(
    x_len: usize,
    (rows_x, cols): (usize, usize),
    y_len: usize,
    (rows_y, cols_y): (usize, usize),
    out_len: Option<usize>,
) -> Result<(), PrimError> {
    if rows_x.checked_mul(cols).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: rows_x,
            cols,
            len: x_len,
        });
    }
    if rows_y.checked_mul(cols_y).map(|v| v != y_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: rows_y,
            cols: cols_y,
            len: y_len,
        });
    }
    if cols != cols_y {
        return Err(PrimError::DimMismatch {
            dim: "cols",
            lhs: cols,
            rhs: cols_y,
        });
    }
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

/// 2D launch config for the `dist_combine_clamp` kernel: one unit per output
/// element `(i, j)`, `i` on `ABSOLUTE_POS_X` (rows), `j` on `ABSOLUTE_POS_Y`
/// (cols). Ceiling-division over a 16×16 cube so over-provisioned threads are
/// bounds-checked away in the kernel.
fn launch_dims_2d(rows: usize, cols: usize) -> (CubeCount, CubeDim) {
    let bx = 16u32;
    let by = 16u32;
    let cx = ((rows as u32) + bx - 1) / bx;
    let cy = ((cols as u32) + by - 1) / by;
    (
        CubeCount::Static(cx.max(1), cy.max(1), 1),
        CubeDim { x: bx, y: by, z: 1 },
    )
}

/// Standard ceiling-division 1D launch config for the in-place sqrt pass.
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = ((n as u32) + block - 1) / block;
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}
