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
//! ## Two kernels: coalesced shared-tile (wgpu perf lever) + GATHER (default)
//! The GATHER kernel ([`linear_predict_bias`]) is thread-per-row over the
//! ROW-MAJOR `x`, so a warp's rows stride by `n` — an UNCOALESCED read. On
//! THIS ENV's wgpu adapter (AMD iGPU) that measurably starves the
//! bandwidth-bound matvec (worst at `n = 64`: ~10× the per-element cost of a
//! coalesced read). [`linear_predict_bias_shared`]
//! (`mlrs_kernels::linear_predict` docs) fixes that by staging each row block
//! into `SharedMemory` with a fully COALESCED load, then doing the per-row dot
//! from a bank-conflict-free padded tile — confirmed a 2.5–4× wgpu win in its
//! band (`PREDICT_SHARED_MIN_FEATURES ≤ n ≤ PREDICT_MAX_FEATURES`).
//!
//! **On a real Tesla T4 (CUDA) this does NOT hold**: a within-session A/B (5
//! reps each, `LR_PREDICT_GATHER` toggle, `n = 64` / `m = 100000`, both
//! LinearRegression and Ridge) measured GATHER ~15–28% FASTER than the shared
//! kernel, consistently across all 10 comparisons with zero overlap — the
//! T4's large L2 (4 MiB) apparently already absorbs the strided-read cost that
//! hurts on this env's iGPU, so the extra `SharedMemory` barriers and halved
//! block occupancy (64 vs 256 threads) are pure overhead there. [`linear_predict`]
//! reserves the shared kernel for [`use_shared_predict`]'s wgpu-only gate;
//! cuda/rocm/cpu always use GATHER (already validated to beat cuML/sklearn
//! substantially on the T4 — see `mlrs-linear-predict-coalesced` project
//! memory). The shared kernel stays compiled and tested (not dead code) in
//! case a future backend/GPU generation profile differs; `LR_PREDICT_GATHER`
//! remains available to force GATHER on wgpu too for A/B re-verification.
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
use mlrs_kernels::{linear_predict_bias, linear_predict_bias_shared, PREDICT_ROWS_PER_BLOCK};
// The shared-tile winning-band bounds are only consulted by `use_shared_predict`'s
// wgpu arm (the ONE backend where the kernel measurably wins — module docs
// above); importing them unconditionally would warn `unused` on cuda/rocm/cpu.
#[cfg(feature = "wgpu")]
use mlrs_kernels::{PREDICT_MAX_FEATURES, PREDICT_SHARED_ELEMS, PREDICT_SHARED_MIN_FEATURES};

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

    // SAFETY (both paths): `x.len()`/`coef.len()`/`bias.len()`/`m`/`n` are the
    // carried/validated element counts; every kernel bounds-checks its row id
    // and reads only `x[r*n + c]` for `c < n`, `coef[c]`, and `bias[0]` — all in
    // range by the geometry validation above.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let coef_arg = unsafe { ArrayArg::from_raw_parts(coef.handle().clone(), coef.len()) };
    let bias_arg = unsafe { ArrayArg::from_raw_parts(bias.handle().clone(), bias.len()) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), m) };

    if use_shared_predict(n, elem) {
        // Coalesced shared-tile path (the wgpu perf lever): one
        // `PREDICT_ROWS_PER_BLOCK`-thread cube per row block, grid folded across
        // X/Y so `nblocks` never overflows a single dimension's cap.
        let b_rows = PREDICT_ROWS_PER_BLOCK as usize;
        let nblocks = m.div_ceil(b_rows);
        let (ccount, cdim) = launch_cubes_block(nblocks);
        linear_predict_bias_shared::launch::<F, ActiveRuntime>(
            &client, ccount, cdim, x_arg, coef_arg, bias_arg, out_arg, m as u32, n as u32,
            nblocks as u32,
        );
    } else {
        // GATHER thread-per-row default: every backend except wgpu-in-its-band
        // (cpu, cuda, rocm always; wgpu outside `[MIN_FEATURES, MAX_FEATURES]`
        // or under the `LR_PREDICT_GATHER` A/B hatch — see `use_shared_predict`).
        // The kernel bounds-checks `r < m`, masking the slack lanes of the
        // final block.
        let (ccount, cdim) = launch_dims_1d(m);
        linear_predict_bias::launch::<F, ActiveRuntime>(
            &client, ccount, cdim, x_arg, coef_arg, bias_arg, out_arg, m as u32, n as u32,
        );
    }

    Ok(DeviceArray::from_raw(out_handle, m))
}

/// Route `predict` to the coalesced shared-tile kernel
/// ([`linear_predict_bias_shared`]) only on the ONE backend where it is
/// confirmed to win: wgpu, in its measured band. Every other backend keeps the
/// GATHER kernel:
/// - **cpu**: its MLIR lowering rejects the `SharedMemory` kernel (the
///   `prims::gram::use_shared_gram` `#[cfg(feature = "cpu")]` precedent).
/// - **cuda / rocm**: a within-session Tesla T4 A/B (5 reps each, both
///   LinearRegression and Ridge, `n = 64`/`m = 100000`) measured GATHER
///   ~15–28% FASTER than the shared kernel, consistently, with zero overlap
///   across 10 comparisons — the opposite of the wgpu result (see the module
///   docs above). rocm is untested but treated the same conservative way
///   pending its own measurement.
/// - **wgpu, `n < PREDICT_SHARED_MIN_FEATURES`**: GATHER's short row stride
///   still coalesces there, and the shared cube's fixed cost would regress it
///   (the measured wgpu crossover).
/// - **`n > PREDICT_MAX_FEATURES`**: the padded shared tile is a comptime
///   `64·65` allocation; a fitted dense linear model never exceeds
///   `GRAM_EIG_MAX_FEATURES = 64`, so this is a defensive bound.
/// - **the tile would not FIT the adapter's shared-memory budget**: the tile is
///   sized against the CUDA 48 KiB budget, but a wgpu adapter can advertise as
///   little as 16 KiB (`f32` tile ≈ 16.5 KiB, `f64` ≈ 33 KiB). Launching a
///   `SharedMemory` kernel larger than `max_shared_memory_size` fails pipeline
///   creation, so we fall back to GATHER (which uses no shared memory) rather
///   than break `predict` where it previously worked. `elem = size_of::<F>()`.
/// - **`LR_PREDICT_GATHER` set**: A/B escape hatch (mirrors `LR_GRAM_GEMM` /
///   `KM_SUMS_GATHER`), also usable on wgpu to re-verify the win. Read once and
///   cached (an inference loop calls `predict` repeatedly; the toggle never
///   changes within a process).
fn use_shared_predict(n: usize, elem: usize) -> bool {
    #[cfg(feature = "wgpu")]
    {
        use std::sync::OnceLock;
        static FORCE_GATHER: OnceLock<bool> = OnceLock::new();
        if *FORCE_GATHER.get_or_init(|| std::env::var("LR_PREDICT_GATHER").is_ok()) {
            return false;
        }
        n >= PREDICT_SHARED_MIN_FEATURES as usize
            && n <= PREDICT_MAX_FEATURES as usize
            && PREDICT_SHARED_ELEMS * elem <= crate::capability::active_max_shared_memory()
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let (_, _) = (n, elem);
        false
    }
}

/// `PREDICT_ROWS_PER_BLOCK`-thread workgroup grid for the shared-tile predict
/// kernel: one cube per row block, folded across the X/Y grid axes so the cube
/// count never exceeds `MAX_GRID_DIM` in any single dimension (the slack cubes
/// are guarded in-kernel by `b < nblocks`). Mirrors `prims::gram::launch_cubes_64`.
fn launch_cubes_block(nblocks: usize) -> (CubeCount, CubeDim) {
    let c = (nblocks as u32).max(1);
    let y = c.div_ceil(MAX_GRID_DIM);
    let x = c.div_ceil(y);
    (
        CubeCount::Static(x, y, 1),
        CubeDim { x: PREDICT_ROWS_PER_BLOCK, y: 1, z: 1 },
    )
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
