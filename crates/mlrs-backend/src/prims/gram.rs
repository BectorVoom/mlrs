//! Gram/Xty host API (LINEAR-01 perf lever, D-02) — `gram = XᵀX` (`d×d`),
//! `xty = Xᵀy` (`d×1`), device-resident, no host round-trip.
//!
//! ## Why this exists (see `mlrs_kernels::gram` module docs)
//! `LinearRegression`'s large-`n_samples` Gram+eig path (`fit_gram_eig`)
//! originally formed `G`/`c` via two `gemm` calls: `M=d, K=n_samples, N=d` — a
//! SKINNY `d×d` output over a HUGE `K` reduction. `cubek-matmul` has no
//! split-K, so this shape starves the GPU of independent output tiles no
//! matter how large `n_samples` is — the EXACT pathology that made KMeans'
//! `onehotᵀX` GEMM-sums "catastrophic" (`prims::kmeans` module docs). This
//! prim applies the SAME fix: [`gram_xty`] dispatches to a row-blocked
//! shared-memory accumulation ([`mlrs_kernels::gram::gram_xty_shared`] +
//! [`mlrs_kernels::gram::gram_xty_reduce_partials`]) on every backend except
//! cpu (whose MLIR lowering rejects `SharedMemory` — the `use_shared_sums`
//! precedent), falling back to the original two-`gemm` formation there (and
//! whenever `d² > 4096`, though the caller's `GRAM_EIG_MAX_FEATURES = 64` cap
//! means that never happens in practice today).
//!
//! Tests live in `crates/mlrs-backend/tests/gram_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::gram::{gram_xty_reduce_partials, gram_xty_shared};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::gemm::gemm;
use crate::runtime::ActiveRuntime;

/// Compute `gram = XᵀX` (`d×d` row-major) and `xty = Xᵀy` (`d×1`) for the
/// `n × d` row-major `x` and length-`n` `y`, device-resident (no host
/// round-trip).
///
/// - Shapes are validated (`n * d == x.len()`, `y.len() == n`, both dims
///   non-zero) BEFORE any launch; a mismatch returns
///   [`PrimError::ShapeMismatch`].
/// - Dispatches to the row-blocked shared-memory kernels
///   ([`use_shared_gram`]) or the `gemm`-based fallback (cpu backend, or
///   `d² > 4096`, or the `LR_GRAM_GEMM` A/B env override).
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
pub fn gram_xty<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
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
    validate_geometry(x.len(), (n, d), y.len())?;

    if use_shared_gram(d * d) {
        gram_xty_shared_impl::<F>(pool, x, y, n, d)
    } else {
        gram_xty_gemm_fallback::<F>(pool, x, y, n, d)
    }
}

/// Whether to use the row-blocked shared-memory Gram kernels. `false` on the
/// cpu backend (MLIR rejects `SharedMemory` — the `use_shared_sums`
/// precedent in `prims::kmeans`) and whenever the `d × d` Gram would exceed
/// the fixed 4096-slot `SharedMemory` budget (never happens under the
/// caller's `GRAM_EIG_MAX_FEATURES = 64` cap, but kept as a defensive bound
/// rather than an assert). `LR_GRAM_GEMM=1` forces the `gemm` fallback
/// everywhere, for A/B benchmarking (mirrors `KM_SUMS_GATHER`).
fn use_shared_gram(dd: usize) -> bool {
    #[cfg(feature = "cpu")]
    {
        let _ = dd;
        false
    }
    #[cfg(not(feature = "cpu"))]
    {
        if std::env::var("LR_GRAM_GEMM").is_ok() {
            return false;
        }
        dd <= 4096
    }
}

/// Row-blocked shared-memory Gram/Xty formation — the LINEAR-01 perf path.
/// Mirrors `prims::kmeans::centroid_sums_shared`'s row-block sizing: caps the
/// per-block partial buffer at an ~8M-element budget, then folds the
/// (small, capped) `nblocks` partials with a single `gram_xty_reduce_partials`
/// launch.
fn gram_xty_shared_impl<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
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
    let dd = d * d;
    debug_assert!(dd <= 4096, "shared Gram caller must gate d*d <= 4096");
    let nb_cap = ((8usize << 20) / dd.max(1)).max(64);
    let nb = n.div_ceil(256).clamp(1, nb_cap);
    let rpb = n.div_ceil(nb);
    let nb = n.div_ceil(rpb);

    let pgram_len = nb * dd;
    let pxty_len = nb * d;
    let pgram = pool.acquire(pgram_len * size_of::<F>());
    let pxty = pool.acquire(pxty_len * size_of::<F>());
    let gram = pool.acquire(dd * size_of::<F>());
    let xty = pool.acquire(d * size_of::<F>());

    let client = pool.client().clone();

    // Stage 1: shared-memory row-blocked partial Gram + Xty (one 64-thread
    // cube per row block).
    // SAFETY: validated element counts (caller); the kernel bounds-checks
    // unit ids and clamps each block's row range to `n`.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let y_arg = unsafe { ArrayArg::from_raw_parts(y.handle().clone(), y.len()) };
    let pg_arg = unsafe { ArrayArg::from_raw_parts(pgram.clone(), pgram_len) };
    let px_arg = unsafe { ArrayArg::from_raw_parts(pxty.clone(), pxty_len) };
    let (cc, cd) = launch_cubes_64(nb);
    gram_xty_shared::launch::<F, ActiveRuntime>(
        &client, cc, cd, x_arg, y_arg, pg_arg, px_arg, n as u32, d as u32, nb as u32, rpb as u32,
    );

    // Stage 2: fold the (small, capped) nblocks partials.
    let pg_arg2 = unsafe { ArrayArg::from_raw_parts(pgram.clone(), pgram_len) };
    let px_arg2 = unsafe { ArrayArg::from_raw_parts(pxty.clone(), pxty_len) };
    let g_arg = unsafe { ArrayArg::from_raw_parts(gram.clone(), dd) };
    let xt_arg = unsafe { ArrayArg::from_raw_parts(xty.clone(), d) };
    let (c2, d2) = launch_dims_1d(dd);
    gram_xty_reduce_partials::launch::<F, ActiveRuntime>(
        &client, c2, d2, pg_arg2, px_arg2, g_arg, xt_arg, d as u32, nb as u32,
    );

    pool.release(pgram, pgram_len * size_of::<F>());
    pool.release(pxty, pxty_len * size_of::<F>());

    Ok((
        DeviceArray::from_raw(gram, dd),
        DeviceArray::from_raw(xty, d),
    ))
}

/// The original two-`gemm` Gram/Xty formation (cpu backend fallback, and the
/// `LR_GRAM_GEMM`/`d² > 4096` A/B escape hatches). `gram = XᵀX` via
/// `gemm(x, x, transa=true)`; `xty = Xᵀy` the same way against `y`.
fn gram_xty_gemm_fallback<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
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
    let gram = gemm::<F>(pool, x, (d, n), x, (n, d), true, false, None)?;
    let xty = gemm::<F>(pool, x, (d, n), y, (n, 1), true, false, None)?;
    Ok((gram, xty))
}

/// Validate the Gram/Xty operand geometry. `x` is `n × d` row-major; `y` is
/// length `n`. Both dims non-zero (an empty axis has no well-defined Gram).
fn validate_geometry(x_len: usize, (n, d): (usize, usize), y_len: usize) -> Result<(), PrimError> {
    if n == 0 || d == 0 || n.checked_mul(d).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    if y_len != n {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: n,
            cols: 1,
            len: y_len,
        });
    }
    Ok(())
}

/// 64-thread workgroup grid for a CUBE-addressed kernel (one cube per row
/// block; folds past the per-dimension dispatch limit; slack cubes are
/// guarded in-kernel — mirrors `prims::kmeans::launch_cubes_64`).
fn launch_cubes_64(cubes: usize) -> (CubeCount, CubeDim) {
    const MAX_DIM: u32 = 65_535;
    let c = (cubes as u32).max(1);
    let y = c.div_ceil(MAX_DIM);
    let x = c.div_ceil(y);
    (
        CubeCount::Static(x, y, 1),
        CubeDim { x: 64, y: 1, z: 1 },
    )
}

/// Standard ceiling-division 1D launch config (the `elementwise`/`center`
/// per-element launch idiom).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = ((n as u32) + block - 1) / block;
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}
