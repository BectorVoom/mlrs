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
//! prim applies the SAME fix: split `n_samples` into row BLOCKS so the
//! parallelism is `nblocks`-way rather than `d×d`-way, accumulate a private
//! partial Gram per block, and fold the (small, capped) partials
//! ([`mlrs_kernels::gram::gram_xty_reduce_partials`]).
//!
//! ## Three formations, and why the default changed (RIDGE-POS-PERF)
//! - [`GramPath::Blocked`] — the DEFAULT
//!   ([`mlrs_kernels::gram::gram_xty_blocked`]): register-resident
//!   accumulators, no shared memory, no barriers, only the lower triangle
//!   computed, and centering fusable into the same pass
//!   ([`gram_xty_centered`]).
//! - [`GramPath::Shared`] — the previous default
//!   ([`mlrs_kernels::gram::gram_xty_shared`]), kept as the `LR_GRAM_SHARED=1`
//!   A/B arm. It stages one row at a time into `SharedMemory` and pays TWO
//!   `sync_cube` barriers PER ROW; measured at `n=100 000, d=64` on the local
//!   wgpu adapter it took 40 ms against the blocked kernel's 4.8 ms.
//! - [`GramPath::Gemm`] — the original two-`gemm` formation. Still the cpu
//!   backend's arm (whose MLIR lowering rejects `SharedMemory` — the
//!   `use_shared_sums` precedent — and where `prims::gram_host` carries the
//!   `Ridge` fit anyway) and the arm above [`BLOCKED_MAX_DD`].
//!
//! The register accumulator also lifted the `d ≤ 64` ceiling the shared
//! kernel's fixed 4096-slot `SharedMemory` budget imposed: `64 < d ≤ 256` used
//! to fall into the starved-GEMM formation and now does not.
//!
//! Tests live in `crates/mlrs-backend/tests/gram_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_core::f64_to_host;
use mlrs_kernels::gram::{
    center_scale_rows, col_sums_blocked, col_sums_reduce, gram_xty_blocked,
    gram_xty_reduce_partials, gram_xty_shared, gram_xty_shared_tiled, gram_xty_tiled, transpose_2d,
    xty_multi_blocked, xty_multi_reduce, GRAM_BLK, GRAM_CHUNK_ROWS, GRAM_REG_TILE,
    GRAM_SHARED_UNITS, GRAM_TILE, XTY_MULTI_MAX_TARGETS,
};

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
/// - Dispatches per [`gram_path`]: the register-blocked kernels by default, the
///   previous shared-memory pair under `LR_GRAM_SHARED=1`, or the `gemm`
///   fallback (cpu backend, `d² > `[`BLOCKED_MAX_DD`], or `LR_GRAM_GEMM=1`).
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

    match gram_path::<F>(d * d) {
        GramPath::SharedTiled | GramPath::Tiled | GramPath::Blocked => {
            gram_xty_blocked_impl::<F>(pool, x, y, None, n, d)
        }
        GramPath::Shared => gram_xty_shared_impl::<F>(pool, x, y, n, d),
        GramPath::Gemm => gram_xty_gemm_fallback::<F>(pool, x, y, n, d),
    }
}

/// [`gram_xty`] of the CENTERED design, without ever materializing it:
/// `gram[i,j] = Σ_r (x[r,i] − x̄_i)(x[r,j] − x̄_j)` and
/// `xty[i] = Σ_r (x[r,i] − x̄_i)(y_r − ȳ)`.
///
/// `means` is the `(x̄, ȳ)` pair [`column_means`] produces, both device-resident
/// (length `d` and length 1). The subtraction is fused into the accumulation
/// kernel, so against the `center_columns` → [`gram_xty`] composition this
/// saves a full `n × d` allocation, its write, and its re-read — the second
/// largest device cost of a `Ridge` fit after the design upload itself.
///
/// Falls back to `center_columns` + the unfused formation on the paths that
/// have no fused kernel (the cpu backend and the over-cap `d`, see
/// [`gram_path`]), so the result is identical on every backend.
pub fn gram_xty_centered<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    means: (
        &DeviceArray<ActiveRuntime, F>,
        &DeviceArray<ActiveRuntime, F>,
    ),
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
    if means.0.len() != d || means.1.len() != 1 {
        return Err(PrimError::ShapeMismatch {
            operand: "gram.xmean",
            rows: d,
            cols: 1,
            len: means.0.len(),
        });
    }

    match gram_path::<F>(d * d) {
        GramPath::SharedTiled | GramPath::Tiled | GramPath::Blocked => {
            gram_xty_blocked_impl::<F>(pool, x, y, Some(means), n, d)
        }
        // No fused kernel on these arms: center first, then form the Gram of
        // the centered copy — the pre-fusion composition, same answer.
        GramPath::Shared | GramPath::Gemm => {
            let (xc, _) = crate::prims::center::center_columns::<F>(pool, x, (n, d))?;
            let (yc, _) = crate::prims::center::center_columns::<F>(pool, y, (n, 1))?;
            let out = gram_xty::<F>(pool, &xc, &yc, n, d);
            xc.release_into(pool);
            yc.release_into(pool);
            out
        }
    }
}

/// Does the FUSED centering route actually fuse for this `(F, d)`?
///
/// [`gram_xty_centered`] and [`column_means`] are correct on every backend, but
/// on the arms with no fused kernel ([`GramPath::Shared`] / [`GramPath::Gemm`] —
/// the cpu backend and the over-cap `d`) they both fall back to
/// `center_columns`, so the "fused" composition centers the design TWICE: once
/// inside `column_means`, whose centered copy is thrown away, and once inside
/// `gram_xty_centered`. That is strictly worse than centering once and forming
/// the Gram of the result.
///
/// A caller choosing between the fused and unfused compositions (`Ridge`) must
/// therefore ask whether the fusion is real, not merely whether it is available.
/// `false` means "take the `center_columns` → [`gram_xty`] route".
pub fn fused_centering_available<F>(d: usize) -> bool
where
    F: Float + CubeElement + Pod,
{
    matches!(
        gram_path::<F>(d * d),
        GramPath::SharedTiled | GramPath::Tiled | GramPath::Blocked
    )
}

/// Device-resident column means of the `n × d` row-major `x` and the mean of
/// the length-`n` `y`, returned as `(x̄, ȳ)` device buffers.
///
/// The row-blocked coalesced pass (see
/// [`mlrs_kernels::gram::col_sums_blocked`]) — adjacent units read adjacent
/// addresses, where `prims::center`'s column-mean walks one column at a time
/// with a `d`-element stride. On the paths with no blocked kernel
/// ([`gram_path`]) it defers to `center_columns`, discarding the centered copy,
/// so every backend gets the same means.
pub fn column_means<F>(
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

    if !fused_centering_available::<F>(d) {
        let (xc, xm) = crate::prims::center::center_columns::<F>(pool, x, (n, d))?;
        let (yc, ym) = crate::prims::center::center_columns::<F>(pool, y, (n, 1))?;
        xc.release_into(pool);
        yc.release_into(pool);
        return Ok((xm, ym));
    }

    column_means_multi::<F>(pool, x, y, n, d, 1, None)
}

/// Multi-target twin of [`column_means`]: the column means of the `n × d`
/// row-major `x` and the `k` column means of the `n × k` row-major `y`,
/// optionally WEIGHTED — `(x̄` length `d`, `ȳ` length `k`)`, device-resident.
///
/// `weights` is sklearn's `sample_weight` (already multiplied by any
/// `class_weight`), device-resident and length `n`. `Some` switches both means
/// to `Σwᵢ·vᵢ / Σwᵢ` (`_preprocess_data`'s `np.average(..., weights=sw)`), with
/// `Σw` folded ON-DEVICE, so the weighted arm needs no host round-trip either —
/// where `Ridge`'s weighted arm reads the whole `n × d` design back to compute
/// exactly these means.
///
/// `k = 1, weights = None` is what [`column_means`] delegates to, and is
/// bit-for-bit the pre-generalization pass.
///
/// This does NOT defer to `center_columns` on the non-fused paths the way
/// [`column_means`] does: it has no centered copy to discard (there is no fused
/// Gram pass to pair it with — see [`gram_xty_multi`]) and both `col_sums`
/// kernels are GATHER-only, so running them everywhere is correct on every
/// backend and strictly less work than centering twice.
pub fn column_means_multi<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    k: usize,
    weights: Option<&DeviceArray<ActiveRuntime, F>>,
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
    validate_geometry_multi(x.len(), (n, d), y.len(), k)?;
    if let Some(w) = weights {
        if w.len() != n {
            return Err(PrimError::ShapeMismatch {
                operand: "gram.sample_weight",
                rows: n,
                cols: 1,
                len: w.len(),
            });
        }
    }

    let (nb, rpb) = row_blocking(n, d);
    let psum_len = nb * d;
    let pysum_len = nb * k;
    let psum = pool.acquire(psum_len * size_of::<F>());
    let pysum = pool.acquire(pysum_len * size_of::<F>());
    // The `weighted == 0` arm never indexes `w`/`pwsum`, but a launch still
    // needs real buffers to bind: one element each, never read on that arm.
    let pwsum_len = if weights.is_some() { nb } else { 1 };
    let pwsum = pool.acquire(pwsum_len * size_of::<F>());
    let xmean = pool.acquire(d * size_of::<F>());
    let ymean = pool.acquire(k * size_of::<F>());

    let dummy_w: Option<DeviceArray<ActiveRuntime, F>> = if weights.is_none() {
        Some(DeviceArray::from_host(pool, &[f64_to_host::<F>(0.0)]))
    } else {
        None
    };
    let w_ref = match (weights, &dummy_w) {
        (Some(w), _) => w,
        (None, Some(dw)) => dw,
        (None, None) => unreachable!("dummy_w is Some whenever weights is None"),
    };
    let weighted = u32::from(weights.is_some());
    let client = pool.client().clone();

    // SAFETY: validated element counts (above); the kernel bounds-checks the
    // cube id against `nblocks`, clamps each block's row range to `n`, and walks
    // columns only while `c < d` / targets while `j < k`.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let y_arg = unsafe { ArrayArg::from_raw_parts(y.handle().clone(), y.len()) };
    let w_arg = unsafe { ArrayArg::from_raw_parts(w_ref.handle().clone(), w_ref.len()) };
    let ps_arg = unsafe { ArrayArg::from_raw_parts(psum.clone(), psum_len) };
    let py_arg = unsafe { ArrayArg::from_raw_parts(pysum.clone(), pysum_len) };
    let pw_arg = unsafe { ArrayArg::from_raw_parts(pwsum.clone(), pwsum_len) };
    let (cc, cd) = launch_cubes(nb, BLOCKED_CUBE_DIM);
    col_sums_blocked::launch::<F, ActiveRuntime>(
        &client,
        cc,
        cd,
        x_arg,
        y_arg,
        w_arg,
        ps_arg,
        py_arg,
        pw_arg,
        n as u32,
        d as u32,
        k as u32,
        nb as u32,
        rpb as u32,
        weighted,
    );

    let ps_arg2 = unsafe { ArrayArg::from_raw_parts(psum.clone(), psum_len) };
    let py_arg2 = unsafe { ArrayArg::from_raw_parts(pysum.clone(), pysum_len) };
    let pw_arg2 = unsafe { ArrayArg::from_raw_parts(pwsum.clone(), pwsum_len) };
    let xm_arg = unsafe { ArrayArg::from_raw_parts(xmean.clone(), d) };
    let ym_arg = unsafe { ArrayArg::from_raw_parts(ymean.clone(), k) };
    // The fold covers BOTH outputs, so it is launched over `max(d, k)`: a `k`
    // wider than `d` would otherwise leave the tail target means unwritten.
    let (c2, d2) = super::launch_dims_1d_folded(d.max(k), crate::capability::gather_launch_width());
    col_sums_reduce::launch::<F, ActiveRuntime>(
        &client,
        c2,
        d2,
        ps_arg2,
        py_arg2,
        pw_arg2,
        xm_arg,
        ym_arg,
        d as u32,
        k as u32,
        nb as u32,
        f64_to_host::<F>(1.0 / n as f64),
        weighted,
    );

    pool.release(psum, psum_len * size_of::<F>());
    pool.release(pysum, pysum_len * size_of::<F>());
    pool.release(pwsum, pwsum_len * size_of::<F>());
    if let Some(dw) = dummy_w {
        dw.release_into(pool);
    }

    Ok((
        DeviceArray::from_raw(xmean, d),
        DeviceArray::from_raw(ymean, k),
    ))
}

/// The normal equations of a MULTI-TARGET fit: `gram = XᵀX` (`d × d`) and
/// `xty = XᵀY` (`d × k` row-major, i.e. FEATURE-major), both device-resident,
/// with the centering fused wherever the single-target path fuses it.
///
/// `means` is the `(x̄, ȳ)` pair [`column_means_multi`] produces (`ȳ` length
/// `k`). `None` forms the RAW normal equations of `x`/`y` as given — which is
/// what the WEIGHTED arm wants, having pre-applied both the centering and the
/// `√w` scale through [`center_scale`].
///
/// ## The Gram is formed ONCE, and the `k` `Xᵀy` columns in one extra pass
/// This is the whole reason a `RidgeClassifier` is not a `Ridge`-per-class
/// loop: `XᵀX` does not depend on `y`, so `k` independent single-target fits
/// re-walk the design's `O(n·d²)` reduction `k` times for zero mathematical
/// benefit. Here the Gram runs on the SAME (bitwise unchanged) kernel the
/// single-target path uses, and [`xty_multi_blocked`] adds all `k` target
/// columns in one further `O(n·d·k)` pass — so the fit costs
/// `O(n·d² + n·d·k)`, not `O(n·d²·k)`.
///
/// ## Why the `Xᵀy` the Gram launch produces is discarded for `k > 1`
/// The three Gram kernels carry a single-target `Xᵀy` on whichever units
/// already hold the column in registers. Widening that to `k` runtime targets
/// would need `4·k` live registers per unit, which CubeCL cannot express and a
/// GPU could not hold — hence the separate pass, which costs one extra
/// `O(n·d)` read against the Gram's `O(n·d²)` of work. The `y` operand bound to
/// the Gram launch is a length-`n` prefix VIEW of the `n × k` target matrix, so
/// its `Xᵀy` output is meaningless; `gram` itself never reads `y` at any point,
/// so the discarded column changes nothing about the value this returns.
/// `k == 1` keeps that output and runs no second pass at all — a binary
/// `RidgeClassifier` is byte-for-byte the shipped `Ridge` formation.
#[allow(clippy::type_complexity)]
pub fn gram_xty_multi<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    means: Option<(
        &DeviceArray<ActiveRuntime, F>,
        &DeviceArray<ActiveRuntime, F>,
    )>,
    n: usize,
    d: usize,
    k: usize,
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
    validate_geometry_multi(x.len(), (n, d), y.len(), k)?;
    if let Some((xm, ym)) = means {
        if xm.len() != d || ym.len() != k {
            return Err(PrimError::ShapeMismatch {
                operand: "gram.xmean",
                rows: d,
                cols: k,
                len: xm.len(),
            });
        }
    }

    if k == 1 {
        return match means {
            Some(m) => gram_xty_centered::<F>(pool, x, y, m, n, d),
            None => gram_xty::<F>(pool, x, y, n, d),
        };
    }

    let y_view = DeviceArray::<ActiveRuntime, F>::from_raw(y.handle().clone(), n);
    let (gram, junk_xty) = match means {
        Some((xm, ym)) => {
            let ym_view = DeviceArray::<ActiveRuntime, F>::from_raw(ym.handle().clone(), 1);
            let out = gram_xty_centered::<F>(pool, x, &y_view, (xm, &ym_view), n, d);
            drop(ym_view);
            out?
        }
        None => gram_xty::<F>(pool, x, &y_view, n, d)?,
    };
    drop(y_view);
    junk_xty.release_into(pool);

    let xty = xty_multi::<F>(pool, x, y, means, n, d, k)?;
    Ok((gram, xty))
}

/// `XᵀY` (`d × k` row-major) of the centered design against the centered
/// targets — [`gram_xty_multi`]'s second pass.
///
/// Blocked independently of the Gram (its partial is `nblocks · d · k`, not
/// `nblocks · d²`) and chunked over targets in windows of
/// [`XTY_MULTI_MAX_TARGETS`] so the kernel's per-unit accumulator stays a
/// comptime-sized register array. Every realistic class count is ONE window.
fn xty_multi<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    means: Option<(
        &DeviceArray<ActiveRuntime, F>,
        &DeviceArray<ActiveRuntime, F>,
    )>,
    n: usize,
    d: usize,
    k: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // The RAW `XᵀY` is the fused kernel with all-zero means — one `d + k`
    // element buffer instead of a second kernel (the `gram_xty_blocked_impl`
    // precedent).
    let zeros: Option<(
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    )> = if means.is_none() {
        let z = f64_to_host::<F>(0.0);
        Some((
            DeviceArray::from_host(pool, &vec![z; d]),
            DeviceArray::from_host(pool, &vec![z; k]),
        ))
    } else {
        None
    };
    let (xmean, ymean) = match (&means, &zeros) {
        (Some((xm, ym)), _) => (*xm, *ym),
        (None, Some((xm, ym))) => (xm, ym),
        (None, None) => unreachable!("zeros is Some whenever means is None"),
    };

    let (nb, rpb) = row_blocking_multi(n, d, k);
    let dk = d * k;
    let pxty_len = nb * dk;
    let pxty = pool.acquire(pxty_len * size_of::<F>());
    let xty = pool.acquire(dk * size_of::<F>());
    let client = pool.client().clone();

    let mut t0 = 0usize;
    while t0 < k {
        let tcount = (k - t0).min(XTY_MULTI_MAX_TARGETS as usize);
        // SAFETY: validated element counts (caller); the kernel bounds-checks
        // the cube id against `nblocks`, clamps its row range to `n`, walks
        // columns while `c < d` and targets while `j < tcount ≤ k − t0`.
        let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
        let y_arg = unsafe { ArrayArg::from_raw_parts(y.handle().clone(), y.len()) };
        let xm_arg = unsafe { ArrayArg::from_raw_parts(xmean.handle().clone(), d) };
        let ym_arg = unsafe { ArrayArg::from_raw_parts(ymean.handle().clone(), k) };
        let px_arg = unsafe { ArrayArg::from_raw_parts(pxty.clone(), pxty_len) };
        let (cc, cd) = launch_cubes(nb, BLOCKED_CUBE_DIM);
        xty_multi_blocked::launch::<F, ActiveRuntime>(
            &client,
            cc,
            cd,
            x_arg,
            y_arg,
            xm_arg,
            ym_arg,
            px_arg,
            n as u32,
            d as u32,
            k as u32,
            t0 as u32,
            tcount as u32,
            nb as u32,
            rpb as u32,
        );
        t0 += tcount;
    }

    let px_arg2 = unsafe { ArrayArg::from_raw_parts(pxty.clone(), pxty_len) };
    let xt_arg = unsafe { ArrayArg::from_raw_parts(xty.clone(), dk) };
    let (c2, d2) = super::launch_dims_1d_folded(dk, crate::capability::gather_launch_width());
    xty_multi_reduce::launch::<F, ActiveRuntime>(
        &client, c2, d2, px_arg2, xt_arg, dk as u32, nb as u32,
    );

    pool.release(pxty, pxty_len * size_of::<F>());
    if let Some((xm, ym)) = zeros {
        xm.release_into(pool);
        ym.release_into(pool);
    }
    Ok(DeviceArray::from_raw(xty, dk))
}

/// `out[r,c] = √wᵣ · (x[r,c] − mean[c])` for an `n × d` row-major `x`, in ONE
/// device pass — sklearn's `_preprocess_data` centering composed with
/// `_rescale_data`'s `√w` row scale.
///
/// `sqrt_w` is the length-`n` ALREADY-square-rooted weight vector: the caller
/// holds the weights on the host, where the `sqrt` is one cheap pass and stays
/// off the `f64` transcendental path some wgpu adapters lack (see the
/// `mlrs-wgpu-f64-eig-broken` finding).
///
/// The weighted fit arm cannot fuse its centering into the Gram accumulation
/// the way the unweighted one does — `√w` multiplies the OPERANDS, so it has to
/// exist before the reduction — which is exactly the `n × d` materialization
/// [`gram_xty_multi`]'s `means` route avoids. It is still strictly better than
/// the host preprocessing pass it replaces (`Ridge::preprocess`'s weighted
/// arm): the design never leaves the device.
pub fn center_scale<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    mean: &DeviceArray<ActiveRuntime, F>,
    sqrt_w: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if n == 0 || d == 0 || x.len() != n * d {
        return Err(PrimError::ShapeMismatch {
            operand: "center_scale.x",
            rows: n,
            cols: d,
            len: x.len(),
        });
    }
    if mean.len() != d || sqrt_w.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "center_scale.mean",
            rows: d,
            cols: 1,
            len: mean.len(),
        });
    }

    let len = n * d;
    let out = pool.acquire(len * size_of::<F>());
    let client = pool.client().clone();

    // SAFETY: validated element counts above; the kernel guards `idx < n*d` and
    // derives `r < n` / `c < d` from it.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), len) };
    let m_arg = unsafe { ArrayArg::from_raw_parts(mean.handle().clone(), d) };
    let w_arg = unsafe { ArrayArg::from_raw_parts(sqrt_w.handle().clone(), n) };
    let o_arg = unsafe { ArrayArg::from_raw_parts(out.clone(), len) };
    let (cc, cd) = super::launch_dims_1d_folded(len, crate::capability::gather_launch_width());
    center_scale_rows::launch::<F, ActiveRuntime>(
        &client, cc, cd, x_arg, m_arg, w_arg, o_arg, n as u32, d as u32,
    );
    Ok(DeviceArray::from_raw(out, len))
}

/// Transpose a `rows × cols` row-major device matrix into its `cols × rows`
/// row-major twin, in one launch and with no host round-trip.
///
/// `RidgeClassifier`'s device fit produces `coef` FEATURE-major (`d × k`, the
/// layout `cholesky_solve_reg` returns and `linear_predict_bias_multi` /
/// `linear_predict_classify` consume) while sklearn's `coef_` attribute is
/// `k × d`. Both are device-resident (D-03) and `d · k` is a few thousand
/// elements, so this replaces a blocking read-back + host transpose +
/// re-upload in the middle of an otherwise round-trip-free fit.
pub fn transpose<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    src: &DeviceArray<ActiveRuntime, F>,
    rows: usize,
    cols: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if rows == 0 || cols == 0 || src.len() != rows * cols {
        return Err(PrimError::ShapeMismatch {
            operand: "transpose.src",
            rows,
            cols,
            len: src.len(),
        });
    }
    let len = rows * cols;
    let out = pool.acquire(len * size_of::<F>());
    let client = pool.client().clone();

    // SAFETY: validated element counts above; the kernel guards `idx < len` and
    // writes `dst[j*rows + i]` with `i < rows`, `j < cols`.
    let s_arg = unsafe { ArrayArg::from_raw_parts(src.handle().clone(), len) };
    let d_arg = unsafe { ArrayArg::from_raw_parts(out.clone(), len) };
    let (cc, cd) = super::launch_dims_1d_folded(len, crate::capability::gather_launch_width());
    transpose_2d::launch::<F, ActiveRuntime>(
        &client,
        cc,
        cd,
        s_arg,
        d_arg,
        rows as u32,
        cols as u32,
    );
    Ok(DeviceArray::from_raw(out, len))
}

/// `(nblocks, rows_per_block)` for the [`xty_multi_blocked`] pass.
///
/// Deliberately NOT [`row_blocking`]: that cap is sized for a `nblocks · d²`
/// partial, and this pass's is `nblocks · d · k`. Sharing it would let a small
/// `d` with a large `k` allocate a partial buffer far past the ~8 M-element
/// budget every other row-blocked prim here respects (`d = 8, k = 32,
/// n = 10⁷` would ask for 32 M elements).
fn row_blocking_multi(n: usize, d: usize, k: usize) -> (usize, usize) {
    let nb_cap = ((8usize << 20) / (d * k).max(1)).max(1);
    let nb = n.div_ceil(256).clamp(1, nb_cap);
    let rpb = n.div_ceil(nb);
    (n.div_ceil(rpb), rpb)
}

/// [`validate_geometry`] for a `k`-target `y`: `x` is `n × d` row-major, `y` is
/// `n × k` row-major. All three dims non-zero.
fn validate_geometry_multi(
    x_len: usize,
    (n, d): (usize, usize),
    y_len: usize,
    k: usize,
) -> Result<(), PrimError> {
    if n == 0 || d == 0 || n.checked_mul(d).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    if k == 0 || n.checked_mul(k).map(|v| v != y_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: n,
            cols: k,
            len: y_len,
        });
    }
    Ok(())
}

/// Why [`column_means`] does NOT get its own, wider row blocking.
///
/// It shares the Gram's [`row_blocking`], whose `nblocks` cap is sized for the
/// `nblocks · d²` partial buffer — a factor of `d` stricter than this pass
/// needs, since its partial is only `nblocks · d`. At `n = 100 000, d = 256`
/// that pins the mean pass to 128 cubes × 64 units = 8192 threads, ~20% of a
/// T4's 40 960, and the drained laps show it costing 4.2 ms of a 13.2 ms
/// on-device fit. That looks exactly like an occupancy bug.
///
/// **It is not, and the obvious fix was measured and reverted.** Giving the
/// pass its own cap (`8M/d`, 128 rows per block → 782 cubes, ~50 000 threads)
/// moved the isolated `column_means` + `gram_xty_centered` phase on a T4 from
/// 4.06 ms to 3.90/4.32 ms across two min-of-9 runs — a wash — while the
/// whole-fit attribution pass got slightly WORSE at all five rungs (4.2 → 5.4 ms
/// at `d = 256`). Six times the threads bought nothing, so the pass is not
/// thread-starved.
///
/// It is also not bandwidth-bound: 102 MiB in ~4 ms is ~25 GB/s against the
/// card's ~320. Whatever the real limiter is, it has not been identified, and
/// the next attempt should start by finding it rather than by adding
/// parallelism that has already been shown not to help.
///
/// `(nblocks, rows_per_block)` for the cube-per-row-block kernels.
///
/// 256 rows per block is the sizing `gram_xty_shared_impl` established; the cap
/// keeps the `nblocks · d²` partial buffer inside an ~8 M-element budget
/// (`prims::kmeans::centroid_sums_shared`'s precedent).
pub(crate) fn row_blocking(n: usize, d: usize) -> (usize, usize) {
    let nb_cap = ((8usize << 20) / (d * d).max(1)).max(1);
    let nb = n.div_ceil(256).clamp(1, nb_cap);
    let rpb = n.div_ceil(nb);
    (n.div_ceil(rpb), rpb)
}

/// Which Gram/Xty formation [`gram_xty`] runs.
///
/// `Blocked` / `Shared` are unreachable under `--features cpu` (whose
/// [`gram_path`] short-circuits to `Gemm`), hence the cfg'd `dead_code`
/// allowance rather than a cfg'd enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "cpu", allow(dead_code))]
enum GramPath {
    /// [`mlrs_kernels::gram::gram_xty_shared_tiled`] — shared-memory row
    /// staging, `O(d)` global loads per row.
    SharedTiled,
    /// [`mlrs_kernels::gram::gram_xty_tiled`] — the 2-D register-tiled,
    /// barrier-free arm.
    Tiled,
    /// [`mlrs_kernels::gram::gram_xty_blocked`] — the previous 1×8 tile, kept
    /// as the `LR_GRAM_BLOCKED=1` A/B arm.
    Blocked,
    /// [`gram_xty_shared_impl`] — the previous shared-memory kernel, kept as
    /// the `LR_GRAM_SHARED=1` A/B arm.
    Shared,
    /// [`gram_xty_gemm_fallback`] — the cpu backend and the over-cap `d`.
    Gemm,
}

/// Pick the Gram/Xty formation for a `d × d` output.
///
/// The register-blocked kernel is the default everywhere except the cpu backend
/// — where every launch is an OS thread spawn and the `gemm` fallback is
/// already validated (the `use_shared_sums` precedent) — and above
/// [`BLOCKED_MAX_DD`], where the `nblocks · d²` partial buffer stops being
/// worth its footprint.
///
/// Two A/B escape hatches, read through [`crate::abflag`]: `LR_GRAM_SHARED=1`
/// forces the previous shared-memory kernel, `LR_GRAM_GEMM=1` the original
/// two-`gemm` formation (the pre-existing knob, unchanged).
fn gram_path<F>(dd: usize) -> GramPath {
    #[cfg(feature = "cpu")]
    {
        let _ = dd;
        GramPath::Gemm
    }
    #[cfg(not(feature = "cpu"))]
    {
        if crate::abflag::var("LR_GRAM_GEMM").is_some() {
            return GramPath::Gemm;
        }
        if crate::abflag::var("LR_GRAM_SHARED").is_some() {
            // The shared kernel's accumulator is a fixed 4096-slot
            // `SharedMemory`, so the A/B arm only exists below that cap.
            return if dd <= 4096 {
                GramPath::Shared
            } else {
                GramPath::Gemm
            };
        }
        if dd > BLOCKED_MAX_DD {
            return GramPath::Gemm;
        }
        if crate::abflag::var("LR_GRAM_BLOCKED").is_some() {
            return GramPath::Blocked;
        }
        // Every FORCE is honoured before any size gate. Ordering these after
        // the `TILED_MIN_DD` early return silently disarms them below the
        // threshold — which made both bitwise A/B tests compare the blocked
        // kernel against itself and pass green over two deliberately broken
        // kernels. A knob that cannot reach the arm it names is worse than no
        // knob.
        if crate::abflag::var("LR_GRAM_TILED").is_some() {
            return GramPath::Tiled;
        }
        if crate::abflag::var("LR_GRAM_SHARED_TILED").is_some() {
            return GramPath::SharedTiled;
        }
        // The shared-staged kernel is a strict traffic reduction over BOTH
        // register kernels, so it takes precedence wherever it applies. It
        // needs a shared-memory budget they do not, and an overrun surfaces as
        // a SILENT all-zero Gram (the `prims::eig` finding), so the budget is
        // checked, never assumed.
        if dd >= SHARED_TILED_MIN_DD && shared_tiled_fits::<F>() {
            return GramPath::SharedTiled;
        }
        // Fallback ladder for adapters without the shared budget.
        if dd >= TILED_MIN_DD {
            return GramPath::Tiled;
        }
        GramPath::Blocked
    }
}

/// Largest `d × d` Gram the register-blocked path forms.
///
/// The partial buffer is `nblocks · d²` elements and `nblocks` is floored at 1,
/// so past this point one partial alone is 512 KiB (`f32`) and the fold stops
/// paying for itself. `d = 256` — well past the `d ≤ 64` the shared kernel
/// could reach, which is the point: the shapes between 64 and 256 used to fall
/// into the starved-GEMM formation.
#[cfg_attr(feature = "cpu", allow(dead_code))]
const BLOCKED_MAX_DD: usize = 256 * 256;


/// `d²` at or above which [`GramPath::SharedTiled`] is used.
///
/// The shared-staged kernel reads each design element from global memory ONCE
/// per cube (`O(d)` loads per row) where both register kernels re-read it once
/// per output tile (`O(d²)` per row). That is a strictly larger win the larger
/// `d` is, so it displaces [`GramPath::Tiled`] everywhere the latter applied
/// and reaches further down — its crossover sits a full octave lower, because
/// staging also fixes the parallelism starvation that made the register tile
/// lose at small `d` (a cube covers a whole `64 × 64` block with 256 units
/// rather than handing out `d²/32` tiles).
///
/// Measured on the local wgpu adapter (`gram_perf_test.rs`, `n = 100 000`,
/// min-of-9, three runs, shared-tiled ÷ blocked):
///
/// | `d` | run 1 | run 2 | run 3 | verdict |
/// |---|---|---|---|---|
/// | 16 | 0.72 | 0.72 | 0.79 | loses |
/// | 32 | 0.99 | 0.92 | 0.92 | wash |
/// | 64 | 2.11 | 1.33 | 2.19 | **wins** |
/// | 128 | 3.63 | 5.06 | 4.03 | **wins** |
/// | 256 | 6.28–8.63 | | | **wins** |
///
/// Crossover between 32 and 64, hence `d = 64`.
///
/// **Confirmed on a Colab Tesla T4** (same probe, min-of-9) — independently,
/// and by a far larger margin than the wgpu sweep suggested:
///
/// | `d` | blocked | shared-staged | |
/// |---|---|---|---|
/// | 16 | 0.556 ms | 0.698 ms | 0.80× |
/// | 32 | 1.178 ms | 1.176 ms | 1.00× |
/// | 64 | 3.446 ms | 0.738 ms | **4.67×** |
/// | 128 | 7.892 ms | 1.446 ms | **5.46×** |
/// | 256 | 56.367 ms | 4.056 ms | **13.90×** |
///
/// The crossover lands between 32 and 64 on BOTH adapters, so this threshold is
/// measured on two independent devices rather than transferred from one. (The
/// register tile's crossover did NOT transfer that cleanly, which is why each
/// arm gets its own swept constant.) `LR_GRAM_SHARED_TILED=1` forces the arm at
/// any size for re-sweeping on a new backend.
#[cfg_attr(feature = "cpu", allow(dead_code))]
const SHARED_TILED_MIN_DD: usize = 64 * 64;

/// `d²` at or above which [`GramPath::Tiled`] replaces [`GramPath::Blocked`].
///
/// The two kernels trade the SAME two quantities in opposite directions, and
/// which one binds depends only on `d`:
///
/// | | loads issued | work items (parallelism) |
/// |---|---|---|
/// | `Blocked` (1×8) | `≈ 0.5625 · d²` per row | `≈ d²/16` groups |
/// | `Tiled` (4×4) | `≈ 0.25 · d²` per row | `≈ d²/32` tiles |
///
/// So the tile issues 2.25× fewer loads for HALF the work items. Below the
/// crossover the reduction is not bandwidth-bound at all — a `d = 16` Gram
/// hands out ten tiles, nowhere near enough to hide memory latency — and the
/// coarser granularity loses; above it the load count is the whole cost and the
/// tile wins by very close to that 2.25×.
///
/// Measured on the local wgpu adapter (`gram_perf_test.rs`, `n = 100 000`,
/// min-of-9, THREE independent runs — this adapter swings ±30% between runs, so
/// a single sweep is not enough to place a threshold):
///
/// | `d` | tiled ÷ blocked (3 runs) | verdict |
/// |---|---|---|
/// | 16 | 0.86 / 1.10 / 1.10 | wash |
/// | 32 | 0.77 / 1.05 / 0.88 | wash |
/// | 64 | 0.66 / 0.61 / 0.60 | **loses, consistently** |
/// | 128 | 1.34 / 1.20 / 1.21 | **wins, consistently** |
/// | 256 | 2.12 / 1.42 / 1.76 | **wins, consistently** |
///
/// The crossover sits between 64 and 128, hence `d = 128`. (A first single-run
/// sweep read `d = 128` at 1.52× and a second at 0.98×; neither was
/// reproducible on its own, which is why the table above is the one to trust.)
///
/// **This constant is calibrated on ONE adapter.** The mechanism
/// (parallelism-bound below, load-bound above) is architectural, but the `d`
/// where they cross is not — a device with more SMs starves at a larger `d`,
/// one with more bandwidth crosses later. `LR_GRAM_TILED=1` / `LR_GRAM_BLOCKED=1`
/// force either arm so the crossover can be re-swept per backend
/// (`gram_perf_test.rs` does exactly that), and until a cuda sweep exists this
/// threshold is deliberately conservative: it only diverges from the previously
/// shipped behaviour where a measurement supports it.
#[cfg_attr(feature = "cpu", allow(dead_code))]
const TILED_MIN_DD: usize = 128 * 128;


/// Does [`mlrs_kernels::gram::gram_xty_shared_tiled`]'s staging budget fit this
/// adapter's per-workgroup shared memory?
///
/// The kernel stages two `GRAM_CHUNK_ROWS × GRAM_BLK` tiles plus the target
/// column, at the COMPTIME size regardless of the runtime `d`. This is checked
/// rather than assumed because a shared-memory overrun does not fail loudly —
/// it produces a silently all-zero result (the `prims::eig` finding, which cost
/// a whole debugging cycle). At 8.3 KiB (`f32`) / 16.5 KiB (`f64`) it should
/// never bind, and the register-tiled arm carries any adapter where it does.
#[cfg_attr(feature = "cpu", allow(dead_code))]
fn shared_tiled_fits<F>() -> bool {
    let staged = 2 * (GRAM_CHUNK_ROWS * GRAM_BLK) as usize + GRAM_CHUNK_ROWS as usize;
    staged * std::mem::size_of::<F>() <= crate::capability::active_max_shared_memory()
}

/// `(lower-triangle tile count, units per cube)` for
/// [`mlrs_kernels::gram::gram_xty_tiled`] at this `d`.
///
/// The cube is sized to the WORK, not to a fixed width: the kernel's cost is
/// `ceil(ntiles / CUBE_DIM)` re-streams of the row block, so a cube wider than
/// `ntiles` buys nothing and only idles units (`d = 16` has ten tiles — a
/// 256-unit cube would leave 246 of them with no tile at all). Rounded up to a
/// 32-unit warp multiple and capped at 256, which is where a `d = 256` fit's
/// 2080 tiles already fold to 9 passes.
/// `MLRS_GRAM_TILE_DIM` overrides the width for on-target A/B — the cube shape
/// is the one parameter here whose best value is a property of the ADAPTER
/// (warp/wavefront width, register file, scheduler) rather than of the problem,
/// so it must be swept on the machine that will run it rather than derived.
fn tile_launch(d: usize) -> (u32, u32) {
    let t = (d as u32).div_ceil(GRAM_TILE);
    let ntiles = t * (t + 1) / 2;
    if let Some(v) = crate::abflag::var("MLRS_GRAM_TILE_DIM") {
        if let Ok(dim) = v.parse::<u32>() {
            if dim > 0 {
                return (ntiles, dim);
            }
        }
    }
    let dim = ntiles.min(256).div_ceil(32) * 32;
    (ntiles, dim.max(32))
}

/// Units per cube of the register-blocked kernel.
///
/// 64 keeps the previous kernel's cube shape (and so its row-block sizing and
/// grid folding), which matters because the parallelism that actually fills the
/// device here is the ROW-BLOCK count, not the unit count: at `d = 16` there
/// are only `16 · 2 = 32` slot groups to hand out, so a wider cube would idle
/// half of itself either way.
pub(crate) const BLOCKED_CUBE_DIM: u32 = 64;

/// Register-blocked, barrier-free Gram/Xty formation — the default path (see
/// [`mlrs_kernels::gram::gram_xty_blocked`]). Same row-block sizing and
/// two-stage fold as [`gram_xty_shared_impl`]; the difference is entirely
/// inside the stage-1 kernel.
fn gram_xty_blocked_impl<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    means: Option<(
        &DeviceArray<ActiveRuntime, F>,
        &DeviceArray<ActiveRuntime, F>,
    )>,
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
    let (nb, rpb) = row_blocking(n, d);

    // The RAW Gram is the fused kernel with an all-zero mean — one `d + 1`
    // element buffer instead of a second kernel.
    let zeros: Option<(
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    )> = if means.is_none() {
        let z = f64_to_host::<F>(0.0);
        Some((
            DeviceArray::from_host(pool, &vec![z; d]),
            DeviceArray::from_host(pool, &[z]),
        ))
    } else {
        None
    };
    let (xmean, ymean) = match (&means, &zeros) {
        (Some((xm, ym)), _) => (*xm, *ym),
        (None, Some((xm, ym))) => (xm, ym),
        (None, None) => unreachable!("zeros is Some whenever means is None"),
    };

    let pgram_len = nb * dd;
    let pxty_len = nb * d;
    let pgram = pool.acquire(pgram_len * size_of::<F>());
    let pxty = pool.acquire(pxty_len * size_of::<F>());
    let gram = pool.acquire(dd * size_of::<F>());
    let xty = pool.acquire(d * size_of::<F>());

    let client = pool.client().clone();

    // Stage 1: one cube per row block, register accumulators, no barriers.
    // SAFETY: validated element counts (caller); both kernels bounds-check the
    // cube id against `nblocks`, clamp each block's row range to `n`, and walk
    // only their own slot/tile space.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let y_arg = unsafe { ArrayArg::from_raw_parts(y.handle().clone(), y.len()) };
    let xm_arg = unsafe { ArrayArg::from_raw_parts(xmean.handle().clone(), d) };
    let ym_arg = unsafe { ArrayArg::from_raw_parts(ymean.handle().clone(), 1) };
    let pg_arg = unsafe { ArrayArg::from_raw_parts(pgram.clone(), pgram_len) };
    let px_arg = unsafe { ArrayArg::from_raw_parts(pxty.clone(), pxty_len) };
    let path = gram_path::<F>(dd);
    if path == GramPath::SharedTiled {
        let tb = (d as u32).div_ceil(GRAM_BLK);
        let ntb = tb * (tb + 1) / 2;
        let (cc, cd) = launch_cubes(nb * ntb as usize, GRAM_SHARED_UNITS);
        gram_xty_shared_tiled::launch::<F, ActiveRuntime>(
            &client,
            cc,
            cd,
            x_arg,
            y_arg,
            xm_arg,
            ym_arg,
            pg_arg,
            px_arg,
            n as u32,
            d as u32,
            nb as u32,
            rpb as u32,
            ntb,
        );
    } else if path == GramPath::Blocked {
        let groups_per_row = (d as u32).div_ceil(GRAM_REG_TILE);
        let (cc, cd) = launch_cubes(nb, BLOCKED_CUBE_DIM);
        gram_xty_blocked::launch::<F, ActiveRuntime>(
            &client,
            cc,
            cd,
            x_arg,
            y_arg,
            xm_arg,
            ym_arg,
            pg_arg,
            px_arg,
            n as u32,
            d as u32,
            nb as u32,
            rpb as u32,
            groups_per_row,
        );
    } else {
        let (ntiles, dim) = tile_launch(d);
        let (cc, cd) = launch_cubes(nb, dim);
        gram_xty_tiled::launch::<F, ActiveRuntime>(
            &client,
            cc,
            cd,
            x_arg,
            y_arg,
            xm_arg,
            ym_arg,
            pg_arg,
            px_arg,
            n as u32,
            d as u32,
            nb as u32,
            rpb as u32,
            ntiles,
        );
    }

    // Stage 2: fold the (small, capped) nblocks partials.
    let pg_arg2 = unsafe { ArrayArg::from_raw_parts(pgram.clone(), pgram_len) };
    let px_arg2 = unsafe { ArrayArg::from_raw_parts(pxty.clone(), pxty_len) };
    let g_arg = unsafe { ArrayArg::from_raw_parts(gram.clone(), dd) };
    let xt_arg = unsafe { ArrayArg::from_raw_parts(xty.clone(), d) };
    let (c2, d2) = super::launch_dims_1d_folded(dd, crate::capability::gather_launch_width());
    gram_xty_reduce_partials::launch::<F, ActiveRuntime>(
        &client, c2, d2, pg_arg2, px_arg2, g_arg, xt_arg, d as u32, nb as u32, 1u32,
    );

    pool.release(pgram, pgram_len * size_of::<F>());
    pool.release(pxty, pxty_len * size_of::<F>());
    if let Some((xm, ym)) = zeros {
        xm.release_into(pool);
        ym.release_into(pool);
    }

    Ok((
        DeviceArray::from_raw(gram, dd),
        DeviceArray::from_raw(xty, d),
    ))
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
    let (c2, d2) = super::launch_dims_1d(dd, crate::capability::gather_launch_width());
    gram_xty_reduce_partials::launch::<F, ActiveRuntime>(
        &client, c2, d2, pg_arg2, px_arg2, g_arg, xt_arg, d as u32, nb as u32, 0u32,
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
    launch_cubes(cubes, 64)
}

/// [`launch_cubes_64`] with an explicit cube width, for the register-blocked
/// path's [`BLOCKED_CUBE_DIM`].
pub(crate) fn launch_cubes(cubes: usize, dim_x: u32) -> (CubeCount, CubeDim) {
    const MAX_DIM: u32 = 65_535;
    let c = (cubes as u32).max(1);
    let y = c.div_ceil(MAX_DIM);
    let x = c.div_ceil(y);
    (
        CubeCount::Static(x, y, 1),
        CubeDim {
            x: dim_x,
            y: 1,
            z: 1,
        },
    )
}
