//! Dense linear-model inference host API (LINEAR-01/02 predict perf lever) —
//! `y = X·coef + intercept`, device-resident, single kernel launch, no host
//! round-trip.
//!
//! ## Why this exists (see `mlrs_kernels::linear_predict` module docs)
//! Every dense linear regressor (`LinearRegression`, `Ridge`, `Lasso`,
//! `ElasticNet`) shared ONE predict body: `raw = gemm(X, coef)` (a skinny
//! `m×1` output) followed by a HOST intercept broadcast — `intercept.to_host()`
//! (blocking scalar readback) + `raw.to_host()` (`m`-length device→host) +
//! an element-wise host loop + `DeviceArray::from_host()` (`m`-length host→
//! device, only for the PyO3 boundary to read it back to host AGAIN). On a
//! discrete GPU those crossings — not the FLOPs — dominate `predict` (the same
//! host-sync pathology `center`/`gram` fixed for the fit path). [`linear_predict`]
//! replaces the whole dance with a single [`linear_predict_bias`] launch that
//! computes `y[r] = Σ_c X[r,c]·coef[c] + bias` fully on device; the caller's
//! own terminal readback is then the ONLY host↔device crossing.
//!
//! ## Portability (no cpu fallback needed)
//! Unlike the `SharedMemory` `gram`/`colmean` perf kernels, the fused matvec is
//! GATHER-only (`mlrs_kernels::linear_predict` module docs) — MLIR-safe on the
//! cpu backend too — so this prim launches the SAME kernel on every backend
//! with no dispatch branch.
//!
//! Tests live in `crates/mlrs-backend/tests/linear_predict_test.rs`
//! (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
// The canonical per-dimension CubeCL grid cap (`65_535`) — the same const
// `prims::center` imports for its identical row-block grid fold; reused here
// (not redefined) so a future adjustment to the launch grid math lives in one
// place. The launch folds the row-block count across the X/Y axes so an
// arbitrarily large `m` (`> 65535·256 ≈ 16.7M` predict rows) never overflows a
// single grid dimension and silently drops tail rows.
use mlrs_kernels::colmean::MAX_GRID_DIM;
use mlrs_kernels::linear_predict_bias;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Compute `y = X·coef + intercept` for the `m × n` row-major test matrix `x`,
/// the length-`n` fitted `coef`, and the length-1 device-resident `bias`
/// (the intercept; a real `0`-valued length-1 buffer for the no-intercept
/// case). Returns the length-`m` device-resident predictions — NO host
/// round-trip (D-05).
///
/// - Shapes are validated (`m * n == x.len()`, `coef.len() == n`,
///   `bias.len() >= 1`, both dims non-zero) BEFORE the launch; a mismatch
///   returns [`PrimError::ShapeMismatch`] / [`PrimError::DimMismatch`].
/// - A SINGLE [`linear_predict_bias`] launch (one unit per output row, grid
///   folded across X/Y for large `m`) does the fused dot-product-plus-bias on
///   device.
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
pub fn linear_predict<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    coef: &DeviceArray<ActiveRuntime, F>,
    bias: &DeviceArray<ActiveRuntime, F>,
    (m, n): (usize, usize),
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry(x.len(), (m, n), coef.len(), bias.len())?;

    let elem = size_of::<F>();
    let out_handle = pool.acquire(m * elem);
    let client = pool.client().clone();
    let (ccount, cdim) = launch_dims_1d(m);

    // SAFETY: `x.len()`/`coef.len()`/`bias.len()`/`m` are the carried/validated
    // element counts; the kernel bounds-checks `r < m` (masking slack lanes)
    // and reads only `x[r*n + c]` for `c < n` and `bias[0]` (both in range by
    // the geometry validation above).
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let coef_arg = unsafe { ArrayArg::from_raw_parts(coef.handle().clone(), coef.len()) };
    let bias_arg = unsafe { ArrayArg::from_raw_parts(bias.handle().clone(), bias.len()) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), m) };
    linear_predict_bias::launch::<F, ActiveRuntime>(
        &client, ccount, cdim, x_arg, coef_arg, bias_arg, out_arg, m as u32, n as u32,
    );

    Ok(DeviceArray::from_raw(out_handle, m))
}

/// Validate the inference operand geometry. `x` is `m × n` row-major; `coef`
/// is length `n`; `bias` holds at least the length-1 intercept scalar. Both
/// dims non-zero (an empty test batch / feature axis has no prediction).
fn validate_geometry(
    x_len: usize,
    (m, n): (usize, usize),
    coef_len: usize,
    bias_len: usize,
) -> Result<(), PrimError> {
    if m == 0 || n == 0 || m.checked_mul(n).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: m,
            cols: n,
            len: x_len,
        });
    }
    if coef_len != n {
        return Err(PrimError::DimMismatch {
            dim: "n_features",
            lhs: coef_len,
            rhs: n,
        });
    }
    if bias_len == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "bias",
            rows: 1,
            cols: 1,
            len: bias_len,
        });
    }
    Ok(())
}

/// Ceiling-division per-row launch config, FOLDED across the X/Y grid axes so
/// the cube count never exceeds `MAX_GRID_DIM` in any single dimension. The
/// kernel addresses its row via the flattened `ABSOLUTE_POS` (which linearizes
/// contiguously across a multi-axis grid — cube `(x, y)` covers rows
/// `[(y·CUBE_COUNT_X + x)·block, +block)`) and bounds-checks `r < m`, so the
/// 2D fold is transparent to it (the `prims::center::launch_dims_1d`
/// precedent, which the large-`m` predict hot path likewise requires).
fn launch_dims_1d(m: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = ((m as u32) + block - 1) / block;
    let x = cubes.min(MAX_GRID_DIM).max(1);
    let y = cubes.div_ceil(x).max(1);
    (
        CubeCount::Static(x, y, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}
