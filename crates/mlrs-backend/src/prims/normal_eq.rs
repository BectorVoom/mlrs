//! Device-resident normal-equation assembly in `f64` — the GPU twin of
//! [`gram_host::centered_gram_xty`].
//!
//! [`centered_gram_xty_device`] returns exactly what the host sweep returns —
//! `(x̄, ȳ, G, Xᵀy)` as host `f64` — but forms the `O(n·d²)` reduction on the
//! DEVICE and reads back only the `d² + d + 1` scalars the caller actually
//! consumes. For `BayesianRidge` that is the whole point of the arm: everything
//! after this reduction is `O(d³)` eigenwork and an `O(d)` iteration, so once
//! the Gram is off the host the design never has to reach the host at all.
//!
//! ## Why this is not just `gram_xty_centered`
//! Two constraints separate it from the `Ridge`/`LinearRegression` device Gram:
//!
//! 1. **`f64` accumulation is mandatory, whatever the design's width.**
//!    `BayesianRidge` consumes the Gram through the residual identity
//!    `sse = yᵀy − 2wᵀXᵀy + wᵀGw`, whose ABSOLUTE error is `~ε·yᵀy` however
//!    small `sse` is — so the RELATIVE error in `sse` is amplified by
//!    `yᵀy/sse`, and `sse` then drives `alpha_` and every subsequent
//!    iteration's shrinkage. Measured (see `bayesian_ridge.rs`), an `f32` Gram
//!    on a design explaining all but `3e-4` of the variance returned
//!    `alpha_ = 762` against sklearn's `254`. The Gram kernels accumulate in
//!    their element type, so this module runs the ENTIRE assembly at `f64`,
//!    widening an `f32` design on the device first
//!    ([`mlrs_kernels::elementwise::widen_elem`]). The widening pass is
//!    `O(n·d)` of device bandwidth against the `O(n·d²)` it protects.
//!
//! 2. **`sample_weight` has to be honoured.** sklearn's `BayesianRidge` applies
//!    `_rescale_data` unconditionally (it has no SAG-family exception), so a
//!    weighted fit centres by the WEIGHTED column means and then scales row `r`
//!    by `√wᵣ`. The fused Gram kernels subtract a constant per-column mean and
//!    know nothing about a per-row scale, so the weighted arm materializes the
//!    preprocessed design once through
//!    [`mlrs_kernels::elementwise::row_scale_center`] and forms the RAW Gram of
//!    it. That costs one `n · d` scratch buffer and one extra streaming pass;
//!    the unweighted arm — the common one, and the one the perf work targets —
//!    keeps the fully fused `column_means` + `gram_xty_centered` route and
//!    allocates nothing extra.
//!
//! ## Equivalence with the host sweep
//! Both arms compute the same mathematical object,
//! `G = Σᵣ wᵣ(xᵣ − x̄)(xᵣ − x̄)ᵀ` and `Xᵀy = Σᵣ wᵣ(xᵣ − x̄)(yᵣ − ȳ)`, in `f64`,
//! from the same `f64`-widened inputs. They differ only in SUMMATION ORDER (the
//! device sums per row block and then folds; the host sums per thread chunk and
//! then folds), which moves the result by `~ε` — orders below the `1e-5` oracle
//! contract. `bayesian_ridge_test.rs`'s host/device agreement test is the gate.
//!
//! Note what that gate does NOT cover: it is reached through
//! [`device_gram_applicable`], which is `false` on the cpu backend, so under
//! `--features cpu` both of its arms are the host sweep and the device path
//! here never runs at all. `normal_eq_test.rs` calls
//! [`centered_gram_xty_device`] directly for exactly that reason — it is what
//! keeps this module's launch sites executable in cpu CI rather than only
//! type-checked, which is the gap a stale `col_sums_*` argument list slipped
//! through.
//!
//! Tests live in `crates/mlrs-backend/tests/normal_eq_test.rs` and
//! `crates/mlrs-algos/tests/bayesian_ridge_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::elementwise::{row_scale_center, widen_elem};
use mlrs_kernels::gram::{col_sums_blocked, col_sums_reduce};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::gram::{
    fused_centering_available, gram_xty, gram_xty_centered, launch_cubes, row_blocking,
    BLOCKED_CUBE_DIM,
};
use crate::runtime::ActiveRuntime;

/// The `(x̄, ȳ, G, Xᵀy)` a caller needs to run a centered normal-equation solve,
/// all host-resident `f64` — the same tuple
/// [`gram_host::centered_gram_xty`](crate::prims::gram_host::centered_gram_xty)
/// returns, so a caller can swap arms without touching anything downstream.
pub type NormalEquations = (Vec<f64>, f64, Vec<f64>, Vec<f64>);

/// Is the device arm of [`centered_gram_xty_device`] available AND worth taking
/// for this shape?
///
/// Three gates, in increasing order of how much they cost to evaluate:
///
/// - **`cpu` backend → never.** The cpu runtime spawns one OS thread per unit
///   and JITs at `-O0`; a GPU-shaped reduction is pathological there (the
///   measured lesson behind every host arm in this crate). The host sweep is
///   already the right code on that backend.
/// - **No `f64` KERNELS on the adapter → never.** The whole arm is an `f64`
///   accumulation, so a backend that cannot run an `f64` kernel cannot run it
///   at all. This is a CORRECTNESS gate, not a perf one. The question is
///   [`capability::f64_device_kernels_available`](crate::capability::f64_device_kernels_available)
///   and deliberately NOT `supports_type(F64)`: on cuda the advertised flag is
///   `false` because of an upstream MATMUL workaround, and reading it here
///   would disable this arm on exactly the hardware with fast `f64`.
/// - **No fused Gram kernel for this `d` → never.** Above the `gram_path` cap
///   the Gram falls back to a `gemm` composition that materializes intermediate
///   `n × d` products, which gives back more than the arm wins.
///
/// `MLRS_BAYES_GRAM_DEVICE` overrides the verdict for A/B on the target
/// hardware (`1` forces the device arm where it is LEGAL — the capability gates
/// above still bind — and `0` forces the host arm). Read through
/// [`crate::abflag`] rather than `std::env` so a test can scope the override
/// without an environment data race.
///
/// Note what is deliberately NOT gated here: `n` and `d` thresholds. Unlike
/// `gram_host_applicable`, both arms of this decision produce bit-comparable
/// answers, so the choice is purely about throughput and the caller
/// ([`crate::prims::gram_host::gram_host_applicable`]) has already decided
/// whether the design is even large enough to leave the host. This predicate
/// answers only "CAN the reduction run on the device, and would the fused path
/// be taken if it did".
pub fn device_gram_applicable<F>(d: usize) -> bool
where
    F: Float + CubeElement + Pod,
{
    if crate::capability::active_backend_name() == "cpu" {
        return false;
    }
    if !crate::capability::f64_device_kernels_available() {
        return false;
    }
    if !fused_centering_available::<f64>(d) {
        return false;
    }
    if let Some(v) = crate::abflag::var("MLRS_BAYES_GRAM_DEVICE") {
        return v != "0";
    }
    true
}

/// Should a caller UPLOAD an `n × d` HOST-resident design to reach the device
/// arm, or form the normal equations on the host and never upload at all?
///
/// **`false` by default on every backend measured so far.** That is a statement
/// about transfers, not about the arm: see the tables below.
///
/// This is the INGRESS decision only. A caller who ALREADY holds the design on
/// the device reaches [`Fit::fit`](mlrs_algos) without consulting this
/// predicate at all, and takes the device Gram — where it belongs, since there
/// is no transfer to amortize.
///
/// ## What the measurement says
/// The device arm buys exactly one thing, the `O(n·d²)` reduction, and must pay
/// an `O(n·d)` upload for it. The ratio of the two is `d/2` multiply-adds per
/// element transferred, so the break-even is a threshold on `d`:
///
/// ```text
/// d_crossover = 16 / ( bandwidth · (1/host_rate − 1/device_rate) )
/// ```
///
/// **Tesla P100-PCIE-16GB, Kaggle 4-vCPU VM, `f64`, min-of-5, upload inside the
/// timer** (`bayesian_ridge_perf_test`'s forced-arm ladder):
///
/// | `n` | `d` | host arm | device arm | |
/// |---|---|---|---|---|
/// | 1 000 | 8 | 0.09 ms | 0.64 ms | 0.14× |
/// | 10 000 | 64 | 5.00 ms | 8.49 ms | 0.59× |
/// | 100 000 | 16 | 5.17 ms | 22.9 ms | 0.23× |
/// | 100 000 | 64 | 31.8 ms | 128.2 ms | 0.25× |
/// | 100 000 | 128 | 112.2 ms | 391.0 ms | 0.29× |
/// | 100 000 | 256 | 430.7 ms | 803.2 ms | 0.54× |
/// | 200 000 | 256 | 807.3 ms | 1 578.8 ms | 0.51× |
///
/// The ratio does improve with `d` exactly as the model says — and still does
/// not reach 1. Backing the numbers out: the `d = 256` device fit spends
/// ~733 ms of its 803 ms moving 204.8 MB, i.e. **~0.28 GB/s host→device**, which
/// is 92% of the fit and about 2% of what a PCIe 3.0 ×16 link can carry. That
/// matches what `Ridge` measured on Kaggle/Colab independently
/// ([[mlrs-ridge-positive-cuda]]: 85–91% of every device fit, 0.33 GB/s), so it
/// is a property of these VMs' transfer path, not of this estimator or the card.
/// Substituting the measured rates into the formula puts the crossover at
/// `d ≈ 520` on that VM.
///
/// wgpu is worse and for a different reason — an integrated adapter shares the
/// host's memory bus, so the "transfer" is a copy the host could have skipped:
///
/// | `n` | `d` | host arm | device arm | |
/// |---|---|---|---|---|
/// | 1 000 | 8 | 0.23 ms | 1.57 ms | 0.15× |
/// | 10 000 | 64 | 7.24 ms | 33.0 ms | 0.22× |
/// | 100 000 | 16 | 2.18 ms | 24.4 ms | 0.09× |
/// | 100 000 | 64 | 13.5 ms | 123.3 ms | 0.11× |
/// | 100 000 | 128 | 35.1 ms | 215.3 ms | 0.16× |
/// | 100 000 | 256 | 128.8 ms | 512.0 ms | 0.25× |
/// | 200 000 | 256 | 217.9 ms | 839.5 ms | 0.26× |
///
/// ## Why the default is `false` rather than a `d ≥ 520` threshold
/// Because 520 is a DERIVED number and nothing has run there. Shipping a gate
/// that switches arms at a size no one has measured is precisely the
/// extrapolation `mlrs-feedback-verify-on-target-hardware` was written down to
/// prevent — and the derivation is machine-specific twice over (it moves with
/// the host's core count AND the bus). A host with real PCIe bandwidth would
/// flip this at a far smaller `d`; the way to find out is to run the probe
/// there, not to guess here.
///
/// `MLRS_BAYES_FIT_HOST=0` forces the device ingress at any size, which is how
/// that measurement is taken; `=1` pins the host ingress.
pub fn device_fit_preferred<F>(n: usize, d: usize) -> bool
where
    F: Float + CubeElement + Pod,
{
    if !device_gram_applicable::<F>(d) {
        return false;
    }
    if let Some(v) = crate::abflag::var("MLRS_BAYES_FIT_HOST") {
        return v == "0";
    }
    let _ = n;
    false
}

/// Column means, target mean, centered Gram and centered `Xᵀy` from a
/// DEVICE-resident design, accumulated in `f64` and read back as host `f64`.
///
/// `x` is the `n × d` row-major design and `y` the length-`n` target, both in
/// the estimator's own width `F`. `sample_weight`, when given, is a length-`n`
/// host slice of non-negative `f64` weights (already validated by the caller);
/// `fit_intercept` selects whether the design and target are centered before
/// the Gram is formed.
///
/// Returns `(x̄, ȳ, G, Xᵀy)` with `x̄`/`G`/`Xᵀy` of length `d`/`d²`/`d`. With
/// `fit_intercept = false` the means are zeros and `0.0` — reported rather than
/// omitted, so the caller's `intercept_` arithmetic is uniform.
///
/// The caller must have validated geometry; `gram_xty` re-checks the element
/// counts and surfaces a [`PrimError`] rather than launching on a bad shape.
pub fn centered_gram_xty_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
) -> Result<NormalEquations, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- Widen to f64 if the estimator is narrower (module docs, constraint 1).
    //     `Owned` carries buffers this function allocated and must free;
    //     `Borrowed` is a second typed VIEW of the caller's handles, which it
    //     must NOT free. Handles are ref-counted, so the view is a cheap clone
    //     of the same allocation. ---
    let widened = size_of::<F>() != size_of::<f64>();
    let (xd, yd) = if widened {
        (widen_to_f64::<F>(pool, x), widen_to_f64::<F>(pool, y))
    } else {
        (
            DeviceArray::<ActiveRuntime, f64>::from_raw(x.handle().clone(), x.len()),
            DeviceArray::<ActiveRuntime, f64>::from_raw(y.handle().clone(), y.len()),
        )
    };

    let result = assemble_f64(pool, &xd, &yd, n, d, sample_weight, fit_intercept);

    // Only the buffers this function allocated are returned to the pool. A
    // `Borrowed` view aliases the caller's handle and releasing it would put a
    // still-live allocation on the free list.
    if widened {
        xd.release_into(pool);
        yd.release_into(pool);
    }
    result
}

/// [`centered_gram_xty_device`]'s body, once both operands are known to be
/// `f64`. Split out so the widening bookkeeping above has exactly one exit path
/// for the scratch buffers.
fn assemble_f64(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, f64>,
    y: &DeviceArray<ActiveRuntime, f64>,
    n: usize,
    d: usize,
    sample_weight: Option<&[f64]>,
    fit_intercept: bool,
) -> Result<NormalEquations, PrimError> {
    match sample_weight {
        // --- Unweighted: the fully fused route. Centering is folded into the
        //     Gram kernel's row tile, so the centered n × d design is never
        //     materialized and nothing beyond the d² + d outputs is allocated.
        None => {
            let (x_mean, y_mean, gram, xty) = if fit_intercept {
                let (xm, ym) = crate::prims::gram::column_means::<f64>(pool, x, y, n, d)?;
                let (g, b) = gram_xty_centered::<f64>(pool, x, y, (&xm, &ym), n, d)?;
                let x_mean = xm.to_host(pool);
                let y_mean = ym.to_host(pool)[0];
                xm.release_into(pool);
                ym.release_into(pool);
                (x_mean, y_mean, g, b)
            } else {
                let (g, b) = gram_xty::<f64>(pool, x, y, n, d)?;
                (vec![0.0f64; d], 0.0f64, g, b)
            };
            let out = (x_mean, y_mean, gram.to_host(pool), xty.to_host(pool));
            gram.release_into(pool);
            xty.release_into(pool);
            Ok(out)
        }

        // --- Weighted: sklearn centres by the WEIGHTED means and then scales
        //     row r by √wᵣ, which no fused Gram kernel expresses. Materialize
        //     the preprocessed design once and take the RAW Gram of it.
        Some(w) => weighted(pool, x, y, n, d, w, fit_intercept),
    }
}

/// The `sample_weight` arm: weighted means, then `xs = (x − x̄)·√w` /
/// `ys = (y − ȳ)·√w`, then the raw Gram of the pair.
///
/// The identity that makes the raw Gram correct is
/// `Σᵣ wᵣ(xᵣ − x̄)(xᵣ − x̄)ᵀ = Σᵣ uᵣuᵣᵀ` with `uᵣ = √wᵣ(xᵣ − x̄)` — i.e. the
/// weight is absorbed into the design exactly once, which is also how sklearn's
/// `_rescale_data` spells it.
///
/// ONE `n · d` scratch buffer serves both preprocessing passes: it first holds
/// `w·x` (whose column sums give the weighted means) and is then OVERWRITTEN
/// with `(x − x̄)·√w`. The two uses never overlap in time, so a second
/// allocation would only raise peak footprint.
fn weighted(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, f64>,
    y: &DeviceArray<ActiveRuntime, f64>,
    n: usize,
    d: usize,
    w: &[f64],
    fit_intercept: bool,
) -> Result<NormalEquations, PrimError> {
    let w_sum: f64 = w.iter().sum();
    let sqrt_w: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
    let w_dev = DeviceArray::<ActiveRuntime, f64>::from_host(pool, w);
    let sqw_dev = DeviceArray::<ActiveRuntime, f64>::from_host(pool, &sqrt_w);
    // The unused-operand placeholder (module docs on `row_scale_center`): a
    // `#[cube(launch)]` signature has no optional arrays, so a switched-off
    // branch still needs a real buffer behind it.
    //
    // It is sized to `d` — the full length the ENABLED branch would index —
    // rather than to one element. The `use_mean` guard means the load is never
    // semantically executed, but a shader compiler is free to hoist or
    // speculate a load out of a branch, and a one-element buffer indexed by
    // `tid % d` would then be an out-of-bounds read. Paying `d` elements to
    // make the buffer safe under speculation costs nothing next to the `n · d`
    // pass it accompanies, and it does not depend on being right about any
    // particular backend's optimizer.
    let zeros_d = DeviceArray::<ActiveRuntime, f64>::from_host(pool, &vec![0.0f64; d.max(1)]);

    let xs = DeviceArray::<ActiveRuntime, f64>::from_raw(pool.acquire(n * d * 8), n * d);
    let ys = DeviceArray::<ActiveRuntime, f64>::from_raw(pool.acquire(n * 8), n);

    // --- Weighted means, when centring. `x̄ = Σwᵣxᵣ / Σw`, so the column-sum
    //     pass runs over `w·x` and the reduce divides by `Σw` instead of `n`
    //     (the same two kernels `column_means` uses, with the one scalar
    //     changed). ---
    let (x_mean, y_mean) = if fit_intercept {
        scale_center(pool, x, &zeros_d, &w_dev, &xs, d, false, true);
        scale_center(pool, y, &zeros_d, &w_dev, &ys, 1, false, true);
        let inv = if w_sum > 0.0 { 1.0 / w_sum } else { 0.0 };
        let (xm, ym) = column_sums_scaled(pool, &xs, &ys, n, d, inv);
        let xh = xm.to_host(pool);
        let yh = ym.to_host(pool)[0];
        xm.release_into(pool);
        ym.release_into(pool);
        (xh, yh)
    } else {
        (vec![0.0f64; d], 0.0f64)
    };

    let xmean_dev = DeviceArray::<ActiveRuntime, f64>::from_host(pool, &x_mean);
    let ymean_dev = DeviceArray::<ActiveRuntime, f64>::from_host(pool, &[y_mean]);
    scale_center(pool, x, &xmean_dev, &sqw_dev, &xs, d, fit_intercept, true);
    scale_center(pool, y, &ymean_dev, &sqw_dev, &ys, 1, fit_intercept, true);

    let (gram, xty) = gram_xty::<f64>(pool, &xs, &ys, n, d)?;
    let out = (x_mean, y_mean, gram.to_host(pool), xty.to_host(pool));

    gram.release_into(pool);
    xty.release_into(pool);
    xs.release_into(pool);
    ys.release_into(pool);
    w_dev.release_into(pool);
    sqw_dev.release_into(pool);
    zeros_d.release_into(pool);
    xmean_dev.release_into(pool);
    ymean_dev.release_into(pool);
    Ok(out)
}

/// One [`row_scale_center`] launch: `out[r·cols + c] = (a[r·cols + c] − mean[c])
/// · scale[r]`, with either half switchable off.
fn scale_center(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, f64>,
    mean: &DeviceArray<ActiveRuntime, f64>,
    scale: &DeviceArray<ActiveRuntime, f64>,
    out: &DeviceArray<ActiveRuntime, f64>,
    cols: usize,
    use_mean: bool,
    use_scale: bool,
) {
    let client = pool.client().clone();
    let len = a.len();
    let (cc, cd) = super::launch_dims_1d_folded(len, crate::capability::gather_launch_width());
    // SAFETY: every length is the carried element count of a live DeviceArray;
    // the kernel bounds-checks `tid < a.len()` and reads `mean`/`scale` only
    // under the flags whose buffers the caller sized to `cols`/`rows`.
    let a_arg = unsafe { ArrayArg::from_raw_parts(a.handle().clone(), len) };
    let m_arg = unsafe { ArrayArg::from_raw_parts(mean.handle().clone(), mean.len()) };
    let s_arg = unsafe { ArrayArg::from_raw_parts(scale.handle().clone(), scale.len()) };
    let o_arg = unsafe { ArrayArg::from_raw_parts(out.handle().clone(), out.len()) };
    row_scale_center::launch::<f64, ActiveRuntime>(
        &client,
        cc,
        cd,
        a_arg,
        m_arg,
        s_arg,
        o_arg,
        cols as u32,
        u32::from(use_mean),
        u32::from(use_scale),
    );
}

/// `column_means` with the normalizing factor supplied by the caller rather
/// than fixed at `1/n` — the weighted arm needs `1/Σw`.
///
/// Same two kernels and the same row blocking as
/// [`crate::prims::gram::column_means`]; only the scalar differs. The caller
/// has already gated `fused_centering_available` through
/// [`device_gram_applicable`], so the unfused fallback that function carries is
/// unreachable here.
///
/// `col_sums_blocked`/`col_sums_reduce` gained a device-side WEIGHTED arm and a
/// multi-TARGET `k` (`crate::prims::gram::column_means`'s weighted/multi-target
/// fit); this caller wants neither — the scale here is a HOST-supplied `inv`
/// (module docs, constraint 2: sklearn's `_rescale_data`-style `1/Σw`, already
/// folded by the caller) rather than a device weight vector, and there is
/// exactly one target column (`y`). So this always launches the `weighted = 0,
/// k = 1` arm: a length-1 dummy `w`/`pwsum` buffer that the kernel's
/// `weighted == 0` branch never indexes (see `column_means`'s identical dummy),
/// and `k = 1` matches the single `ymean` slot this function already allocates.
fn column_sums_scaled(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, f64>,
    y: &DeviceArray<ActiveRuntime, f64>,
    n: usize,
    d: usize,
    inv: f64,
) -> (
    DeviceArray<ActiveRuntime, f64>,
    DeviceArray<ActiveRuntime, f64>,
) {
    const K: usize = 1;
    let (nb, rpb) = row_blocking(n, d);
    let psum_len = nb * d;
    let pysum_len = nb * K;
    let psum = pool.acquire(psum_len * 8);
    let pysum = pool.acquire(pysum_len * 8);
    // `weighted == 0` never indexes `w`/`pwsum`, but the launch still needs a
    // real buffer to bind — one element, never read on this arm (the
    // `column_means` precedent).
    let pwsum_len = 1;
    let pwsum = pool.acquire(pwsum_len * 8);
    let dummy_w: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(pool, &[0.0f64]);
    let xmean = pool.acquire(d * 8);
    let ymean = pool.acquire(8);
    let client = pool.client().clone();

    // SAFETY: validated element counts; the kernel bounds-checks the cube id
    // against `nblocks`, clamps each block's row range to `n`, and walks
    // columns only while `c < d` (the `column_means` launch, verbatim).
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let y_arg = unsafe { ArrayArg::from_raw_parts(y.handle().clone(), y.len()) };
    let w_arg = unsafe { ArrayArg::from_raw_parts(dummy_w.handle().clone(), dummy_w.len()) };
    let ps_arg = unsafe { ArrayArg::from_raw_parts(psum.clone(), psum_len) };
    let py_arg = unsafe { ArrayArg::from_raw_parts(pysum.clone(), pysum_len) };
    let pw_arg = unsafe { ArrayArg::from_raw_parts(pwsum.clone(), pwsum_len) };
    let (cc, cd) = launch_cubes(nb, BLOCKED_CUBE_DIM);
    col_sums_blocked::launch::<f64, ActiveRuntime>(
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
        K as u32,
        nb as u32,
        rpb as u32,
        0, // weighted
    );

    let ps_arg2 = unsafe { ArrayArg::from_raw_parts(psum.clone(), psum_len) };
    let py_arg2 = unsafe { ArrayArg::from_raw_parts(pysum.clone(), pysum_len) };
    let pw_arg2 = unsafe { ArrayArg::from_raw_parts(pwsum.clone(), pwsum_len) };
    let xm_arg = unsafe { ArrayArg::from_raw_parts(xmean.clone(), d) };
    let ym_arg = unsafe { ArrayArg::from_raw_parts(ymean.clone(), K) };
    // `d.max(k)`, the `column_means` rule (a `k` wider than `d` would otherwise
    // leave the tail target means unwritten) — here `k = 1 <= d` always, so this
    // is just `d`, but written the general way so a future `k > 1` here does not
    // silently truncate.
    let (c2, d2) =
        super::launch_dims_1d_folded(d.max(K), crate::capability::gather_launch_width());
    col_sums_reduce::launch::<f64, ActiveRuntime>(
        &client,
        c2,
        d2,
        ps_arg2,
        py_arg2,
        pw_arg2,
        xm_arg,
        ym_arg,
        d as u32,
        K as u32,
        nb as u32,
        inv,
        0, // weighted
    );

    pool.release(psum, psum_len * 8);
    pool.release(pysum, pysum_len * 8);
    pool.release(pwsum, pwsum_len * 8);
    dummy_w.release_into(pool);
    (
        DeviceArray::from_raw(xmean, d),
        DeviceArray::from_raw(ymean, K),
    )
}

/// One [`widen_elem`] launch producing a fresh `f64` copy of an `F`-width
/// operand (module docs, constraint 1).
fn widen_to_f64<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
) -> DeviceArray<ActiveRuntime, f64>
where
    F: Float + CubeElement + Pod,
{
    let len = a.len();
    let out = pool.acquire(len * 8);
    let client = pool.client().clone();
    let (cc, cd) = super::launch_dims_1d_folded(len, crate::capability::gather_launch_width());
    // SAFETY: `len` is the carried element count of a live DeviceArray and the
    // output was just acquired at `len` f64 slots; the kernel bounds-checks
    // `tid < input.len()`.
    let a_arg = unsafe { ArrayArg::from_raw_parts(a.handle().clone(), len) };
    let o_arg = unsafe { ArrayArg::from_raw_parts(out.clone(), len) };
    widen_elem::launch::<F, f64, ActiveRuntime>(&client, cc, cd, a_arg, o_arg);
    DeviceArray::from_raw(out, len)
}
