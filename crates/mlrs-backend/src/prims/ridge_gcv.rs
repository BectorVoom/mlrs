//! `RidgeCV`'s generalized-CV engine on the device (RIDGECV-02) — the GPU twin
//! of `ridge_cv.rs::gcv_cov`.
//!
//! [`GcvDevice`] owns the uploaded design for the whole fit and exposes the two
//! phases that are worth having on a device:
//!
//! | phase | cost | who runs it |
//! |---|---|---|
//! | weighted means + `X̃ᵀX̃` + `X̃ᵀ[ỹ ǀ √w]` | `O(n·d² + n·d·n_y)` | [`GcvDevice::normal_equations`] |
//! | `sym_eig` + per-alpha spectral weights + coefficients | `O(d³ + n_alphas·d²·n_y)` | the HOST — see below |
//! | the streaming sweep | `O(n·d² + n·n_alphas·d·(n_y+2))` | [`GcvDevice::sweep`] |
//!
//! The middle row stays on the host on purpose and not for lack of a kernel: at
//! the shapes this route is taken at (`n > d`, and `d ≤ `[`GCV_MAX_D`]) it is
//! three to five orders below the two device phases, and it is a `d × d`
//! eigendecomposition — a serial scalar recurrence, the one shape a GPU is worst
//! at. `bayesian_ridge.rs` reaches the same split for the same reason.
//!
//! ## Everything is `f64`, whatever the estimator's `F`
//! The LOO denominator is `1 − diag(X̃·Hinv·X̃ᵀ)`, a CANCELLATION that approaches
//! zero exactly where the alpha grid is interesting (small `α`, `q → 1`), and
//! the result is then DIVIDED by it. The host engine therefore accumulates in
//! `f64` at any `F` (`ridge_cv.rs::Elem`), and so does this one: an `f32` design
//! is widened ON DEVICE (`elementwise::widen_elem`) so the upload still moves
//! only `4·n·d` bytes, and every kernel after that is monomorphized on `f64`.
//!
//! That makes `capability::f64_device_kernels_available` a CORRECTNESS gate for
//! this arm, not a perf one — see [`gcv_device_possible`], and note the
//! deliberate choice of that predicate over `supports_type(F64)` (which is
//! `false` on cuda for an unrelated matmul workaround).
//!
//! ## Equivalence with the host arm
//! Both arms evaluate the same expressions in `f64` from the same inputs and
//! differ only in SUMMATION ORDER (the device folds per row block, the host per
//! thread chunk), so they agree to ~`ε·κ` rather than bit-for-bit — the same
//! contract `normal_eq.rs` states. `ridge_cv_device_test.rs` is the gate, and it
//! compares the SCORES, the COEFFICIENTS and the per-alpha `cv_values`, because
//! a sweep that got the denominator right and the prediction rescale wrong would
//! pass a scores-only check on an unweighted fixture.
//!
//! Tests live in `crates/mlrs-backend/tests/ridge_gcv_test.rs` and
//! `crates/mlrs-algos/tests/ridge_cv_device_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::ridge_gcv::{gcv_cov_sweep, GCV_CUBE_DIM, GCV_MAX_D, GCV_ROW_TILE};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::gram::{
    center_scale, column_means_multi, fused_centering_available, gram_xty_multi, launch_cubes,
    xty_multi,
};
use crate::prims::normal_eq::widen_to_f64;
use crate::runtime::ActiveRuntime;

/// The normal equations of the preprocessed design, host-resident `f64` — what
/// [`GcvDevice::normal_equations`] hands back to the eigensolver.
///
/// `(x̄, ȳ, G, X̃ᵀỹ, X̃ᵀ√w)` with lengths `d`, `n_y`, `d²`, `d·n_y`, `d`. The last
/// is the operand of the LOO intercept correction; it is zeros when
/// `fit_intercept` is off (the correction does not apply) and is genuinely
/// ACCUMULATED otherwise, even though weighted centering makes it analytically
/// zero — the host arm makes the same choice, for the same reason
/// (`ridge_cv.rs` module docs).
pub type GcvNormalEquations = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// What [`GcvDevice::sweep`] produces.
pub struct GcvSweepOut {
    /// `n_alphas × n_y` row-major SUMS of `looe²` (not yet negated or divided by
    /// `n` — the caller does that, so the two arms share one reduction).
    /// Empty when the caller asked for predictions.
    pub score_sums: Vec<f64>,
    /// `n × n_alphas × n_y` row-major squared errors or rescaled LOO
    /// predictions. Empty unless `emit_values`.
    pub cv_values: Vec<f64>,
}

/// CAN the device GCV sweep run at all, on this backend and at this `d`?
///
/// A capability question only — nothing here is about speed, and
/// `device = "gpu"` may NOT override it (the `mlrs-device-param` rule: an
/// override of a capability gate is a crash, not a slowdown).
///
/// - **No `f64` device kernels → no.** The whole engine is an `f64`
///   accumulation (module docs). Asked through
///   [`f64_device_kernels_available`](crate::capability::f64_device_kernels_available),
///   deliberately not `supports_type(F64)`.
/// - **`d` above [`GCV_MAX_D`] → no.** The sweep's shared tiles are sized at
///   compile time; a wider design has no kernel.
/// - **Not enough shared memory on the adapter → no.**
/// - **No fused Gram kernel for this `d` → no.** The normal-equation phase is
///   `gram_xty_multi`, whose fused-centering route is what makes it worth
///   leaving the host; below that it composes `center_columns` + a starved
///   `gemm`, which gives back more than the arm wins. This is what excludes the
///   cpu backend, whose `gram_path` is a hard-wired `Gemm`.
pub fn gcv_device_possible(d: usize) -> bool {
    if !crate::capability::f64_device_kernels_available() {
        return false;
    }
    if d == 0 || d > GCV_MAX_D as usize {
        return false;
    }
    if !fused_centering_available::<f64>(d) {
        return false;
    }
    let need = 2 * (GCV_ROW_TILE as usize) * (GCV_MAX_D as usize) * size_of::<f64>();
    crate::capability::active_max_shared_memory() >= need
}

/// Should a HOST-resident design be UPLOADED to reach the device engine?
///
/// The ingress decision, and the only part of the routing a `device`
/// hyperparameter or an `MLRS_RIDGECV_DEVICE` A/B flag may move.
///
/// ## The arithmetic the default rests on
/// The device arm buys two `O(n·d²)` passes (the Gram and the eigenbasis
/// projection) plus the `O(n·n_alphas·d·(n_y+2))` alpha contraction, and must
/// pay an `O(n·d)` upload for them. Per element transferred that is
///
/// ```text
/// 2·d + n_alphas·(n_y + 2)     multiply-adds
/// ```
///
/// which is roughly twice `BayesianRidge`'s `d/2` before the alpha grid is
/// counted at all, and grows with `len(alphas)` — the parameter this estimator
/// is USED to sweep. That is why the gate reads `n_alphas`, and why it is not
/// simply inherited from `normal_eq::device_fit_preferred`.
///
/// ## What the measurement says — and why the default is still `false`
/// **AMD Radeon 860M (gfx1151, INTEGRATED), rocm, `f64`, min-of-5, interleaved,
/// upload inside the device timer** (`ridge_cv_device_perf_test.rs`).
///
/// PHASE ATTRIBUTION first, because it is the number that decides the default —
/// and it is the one table here taken with the contention banner CLEAR (6.8%
/// foreign cpu):
///
/// | `n` | `d` | upload | normal eq | sweep | whole fit |
/// |---|---|---|---|---|---|
/// | 100 000 | 64 | 47.1 ms (**1.09 GB/s**) | 18.3 ms | 64.0 ms | 151 ms |
/// | 100 000 | 128 | 93.2 ms (1.10 GB/s) | 57.6 ms | 180.7 ms | 355 ms |
/// | 50 000 | 256 | 96.4 ms (1.06 GB/s) | 78.8 ms | 250.1 ms | 479 ms |
///
/// The upload is a THIRD of the device fit, and on its own is HALF of what the
/// host arm spends on the entire thing (91 ms at that rung). That is the same
/// story `BayesianRidge` and `Ridge` told on their hardware, in the form an
/// INTEGRATED adapter tells it: there is no bus to cross, so the "transfer" is
/// a copy the host could have skipped.
///
/// The two A/B ladders, at the LOWEST foreign-cpu share this machine allowed
/// (34% and 38% — see the caveat below):
///
/// | `n` | `d` | 30 alphas, host | device | |
/// |---|---|---|---|---|
/// | 10 000 | 16 | 2.94 ms | 7.06 ms | 0.42× |
/// | 10 000 | 64 | 8.96 ms | 18.72 ms | 0.48× |
/// | 100 000 | 16 | 21.94 ms | 59.49 ms | 0.37× |
/// | 100 000 | 64 | 96.61 ms | 177.06 ms | 0.55× |
/// | 100 000 | 128 | 235.44 ms | 343.16 ms | 0.69× |
/// | 50 000 | 256 | 321.67 ms | 488.40 ms | 0.66× |
/// | 200 000 | 64 | 197.37 ms | 308.35 ms | 0.64× |
///
/// The alpha grid is where the model above says the arm should turn, and it
/// does — at `n = 100 000, d = 64`:
///
/// | `len(alphas)` | host | device | |
/// |---|---|---|---|
/// | 1 | 31.2 ms | 159.4 ms | 0.20× |
/// | 3 | 36.1 ms | 179.5 ms | 0.20× |
/// | 10 | 50.5 ms | 157.0 ms | 0.32× |
/// | 30 | 95.9 ms | 150.3 ms | 0.64× |
/// | 100 | 255.4 ms | 234.0 ms | **1.09×** |
/// | 200 | 486.9 ms | 338.6 ms | **1.44×** |
///
/// **Caveat on the two ladders, stated because it has a SIGN.** Another process
/// held this machine in bursts for the whole campaign; 40 attempts did not land
/// a ladder under the 10% foreign-cpu limit, and the two above are the quietest
/// of them. Contention hurts the 16-thread HOST arm far more than the
/// one-thread-plus-GPU device arm, so these ratios FLATTER the device — the true
/// losses are at least this large and the crossover is at least this late. What
/// makes them usable anyway is that the verdict did not move across six runs
/// spanning 12% to 98% foreign cpu: the shape ladder stayed in 0.36–0.69× and
/// the alpha crossover stayed at ~100 (0.96–1.09×) with 200 alphas at
/// 1.23–1.49×. Re-run on a quiet machine before tightening any of it.
///
/// So a crossover exists and has been run through — and the default is still
/// `false`, for three reasons rather than one:
///
/// 1. It arrives at `len(alphas) ≈ 100`, and the fixed cost below it is a 5×
///    loss. A gate keyed on the alpha count would make `RidgeCV(alphas=...)`
///    change execution arm on a parameter users vary casually.
/// 2. The best it buys on this adapter is ~1.4× — measured on a machine that was
///    biased IN ITS FAVOUR — against a host arm already ~11× sklearn at that
///    rung. There is very little to win and a whole class of surprise to lose.
/// 3. gfx1151 is ONE integrated GPU. A discrete card changes both terms of the
///    trade at once (real bus bandwidth, and `f64` throughput that is not a
///    fraction of an iGPU's), so tuning a threshold to this box is exactly the
///    extrapolation `mlrs-feedback-verify-on-target-hardware` warns about.
///
/// `device='gpu'` is how a caller who knows their hardware takes the arm, and it
/// is measured, tested and supported — it is just not guessed at.
///
/// ## The device arm vs SKLEARN, which is a different question
/// Losing to mlrs's own host arm is not the same as being slow. Each library in
/// its own process, `scripts/bench_ridge_cv.py --device gpu --dtype float32`,
/// min-of-5, foreign cpu **4.8% / 1.7%** — a CLEAN run, unlike the two ladders
/// above:
///
/// | `n = 100 000, d = 64` | sklearn | mlrs `device='gpu'` | |
/// |---|---|---|---|
/// | 1 alpha | 83.7 ms | 116.8 ms | 0.72× |
/// | 3 alphas (the default) | 143.4 ms | 123.4 ms | 1.16× |
/// | 10 alphas | 333.0 ms | 111.0 ms | 3.00× |
/// | 30 alphas | 880.2 ms | 115.7 ms | 7.61× |
/// | 100 alphas | 2 389 ms | 207.2 ms | 11.5× |
/// | 200 alphas | 4 872 ms | 290.1 ms | **16.8×** |
///
/// The device arm's time is nearly FLAT to 30 alphas (111–123 ms) because the
/// upload and the Gram dominate there and the sweep absorbs the grid — which is
/// the shape that makes it interesting despite losing the host A/B. It still
/// loses to sklearn on the small-grid, small-design rungs (0.33–0.9× across the
/// `shape` ladder), which is the same fixed cost the host A/B shows and another
/// reason `auto` does not pick it.
///
/// `MLRS_RIDGECV_DEVICE=1` forces the device ingress wherever it is LEGAL (the
/// capability gate above still binds), `=0` pins the host. Read through
/// [`crate::abflag`] so a test can scope the override without an environment
/// data race.
pub fn gcv_device_preferred(n: usize, d: usize, n_alphas: usize, n_y: usize) -> bool {
    if !gcv_device_possible(d) {
        return false;
    }
    if let Some(v) = crate::abflag::var("MLRS_RIDGECV_DEVICE") {
        return v == "1";
    }
    let _ = (n, n_alphas, n_y);
    false
}

/// The design, target and `√w`, uploaded ONCE in `f64` and held for the whole
/// fit — both device phases read them, and the fit would otherwise upload the
/// `n × d` design twice.
pub struct GcvDevice {
    x: DeviceArray<ActiveRuntime, f64>,
    y: DeviceArray<ActiveRuntime, f64>,
    /// `√wᵢ`, or all ones. Always materialized: the LOO intercept correction
    /// reads it independently of any weighting, so a switched-off branch would
    /// still need a bound buffer, and `n` elements is nothing beside `n · d`.
    sqrt_sw: DeviceArray<ActiveRuntime, f64>,
    n: usize,
    d: usize,
    n_y: usize,
    weighted: bool,
}

impl GcvDevice {
    /// Upload an `F`-width host design and target, widening to `f64` ON THE
    /// DEVICE so the transfer stays at the estimator's own width.
    ///
    /// `sqrt_sw` is the length-`n` `√w` the caller already formed on the host
    /// (`Vec` of ones when unweighted); `weighted` records whether the weights
    /// were real, because sklearn's prediction rescale branches on it.
    pub fn from_host<F>(
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        y: &[F],
        n: usize,
        d: usize,
        n_y: usize,
        sqrt_sw: &[f64],
        weighted: bool,
    ) -> Result<Self, PrimError>
    where
        F: Float + CubeElement + Pod,
    {
        if n == 0 || d == 0 || n_y == 0 || x.len() != n * d || y.len() != n * n_y {
            return Err(PrimError::ShapeMismatch {
                operand: "ridge_gcv.x",
                rows: n,
                cols: d,
                len: x.len(),
            });
        }
        if sqrt_sw.len() != n {
            return Err(PrimError::ShapeMismatch {
                operand: "ridge_gcv.sqrt_sw",
                rows: n,
                cols: 1,
                len: sqrt_sw.len(),
            });
        }
        guard_u32("n", n)?;
        guard_u32("d", d)?;
        guard_u32("n_y", n_y)?;

        let (xd, yd) = if size_of::<F>() == size_of::<f64>() {
            (
                DeviceArray::<ActiveRuntime, f64>::from_host(pool, bytemuck::cast_slice(x)),
                DeviceArray::<ActiveRuntime, f64>::from_host(pool, bytemuck::cast_slice(y)),
            )
        } else {
            let xn = DeviceArray::<ActiveRuntime, F>::from_host(pool, x);
            let yn = DeviceArray::<ActiveRuntime, F>::from_host(pool, y);
            let xw = widen_to_f64::<F>(pool, &xn);
            let yw = widen_to_f64::<F>(pool, &yn);
            xn.release_into(pool);
            yn.release_into(pool);
            (xw, yw)
        };

        Ok(Self {
            x: xd,
            y: yd,
            sqrt_sw: DeviceArray::<ActiveRuntime, f64>::from_host(pool, sqrt_sw),
            n,
            d,
            n_y,
            weighted,
        })
    }

    /// Return every buffer this holds to the pool. Call exactly once, after the
    /// fit is done with it (every `DeviceArray`-owning prim's convention).
    pub fn release_into(self, pool: &mut BufferPool<ActiveRuntime>) {
        self.x.release_into(pool);
        self.y.release_into(pool);
        self.sqrt_sw.release_into(pool);
    }

    /// Weighted means, the `d × d` Gram of the preprocessed design, its
    /// `d × n_y` `X̃ᵀỹ`, and the `d` `X̃ᵀ√w` — the whole eigensolver input, read
    /// back as host `f64`.
    ///
    /// Two arms, the same split `normal_eq.rs` makes and for the same reason:
    ///
    /// - **Unweighted** takes the FULLY FUSED route — the centering is folded
    ///   into the Gram kernel's row tile, so the centered `n × d` design is
    ///   never materialized and nothing beyond the `d² + d·n_y` outputs is
    ///   allocated.
    /// - **Weighted** cannot fuse: `√w` multiplies the OPERANDS, so the
    ///   preprocessed design has to exist before the reduction. It is
    ///   materialized once through [`center_scale`], consumed by both the Gram
    ///   and the `X̃ᵀ√w` pass, and released before [`GcvDevice::sweep`] runs —
    ///   so the peak footprint is `n · d` `f64` on that arm and zero on the
    ///   other.
    pub fn normal_equations(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        fit_intercept: bool,
    ) -> Result<GcvNormalEquations, PrimError> {
        let (n, d, n_y) = (self.n, self.d, self.n_y);

        // `w = (√w)²`, device-resident — `column_means_multi`'s weighted arm
        // folds `Σw` on the device, so the means need no host round-trip.
        let w_dev = if self.weighted {
            let w: Vec<f64> = self.sqrt_sw.to_host(pool).iter().map(|s| s * s).collect();
            Some(DeviceArray::<ActiveRuntime, f64>::from_host(pool, &w))
        } else {
            None
        };

        let means = if fit_intercept {
            Some(column_means_multi::<f64>(
                pool,
                &self.x,
                &self.y,
                n,
                d,
                n_y,
                w_dev.as_ref(),
            )?)
        } else {
            None
        };
        let zeros_d = DeviceArray::<ActiveRuntime, f64>::from_host(pool, &vec![0.0f64; d]);
        let zeros_y = DeviceArray::<ActiveRuntime, f64>::from_host(pool, &vec![0.0f64; n_y]);
        let (xm_dev, ym_dev) = match &means {
            Some((xm, ym)) => (xm, ym),
            None => (&zeros_d, &zeros_y),
        };
        let x_offset = xm_dev.to_host(pool);
        let y_offset = ym_dev.to_host(pool);

        // `X̃ᵀ√w` rides the SAME multi-target `Xᵀy` kernel as `X̃ᵀỹ`, with `√w`
        // bound as a one-column target and a zero target-mean — so the term is
        // an accumulated reduction over the centered design rather than the
        // analytic zero it is under weighted centering (module docs).
        let ones_or_sqw = DeviceArray::<ActiveRuntime, f64>::from_raw(
            self.sqrt_sw.handle().clone(),
            self.sqrt_sw.len(),
        );
        let zero_1 = DeviceArray::<ActiveRuntime, f64>::from_host(pool, &[0.0f64]);

        let (gram_dev, xty_dev, xtsw_dev) = if self.weighted {
            let xs = center_scale::<f64>(pool, &self.x, xm_dev, &self.sqrt_sw, n, d)?;
            let ys = center_scale::<f64>(pool, &self.y, ym_dev, &self.sqrt_sw, n, n_y)?;
            let (g, b) = gram_xty_multi::<f64>(pool, &xs, &ys, None, n, d, n_y)?;
            let sw = if fit_intercept {
                Some(xty_multi::<f64>(pool, &xs, &ones_or_sqw, None, n, d, 1)?)
            } else {
                None
            };
            xs.release_into(pool);
            ys.release_into(pool);
            (g, b, sw)
        } else {
            let (g, b) = gram_xty_multi::<f64>(
                pool,
                &self.x,
                &self.y,
                means.as_ref().map(|(a, b)| (a, b)),
                n,
                d,
                n_y,
            )?;
            let sw = if fit_intercept {
                Some(xty_multi::<f64>(
                    pool,
                    &self.x,
                    &ones_or_sqw,
                    Some((xm_dev, &zero_1)),
                    n,
                    d,
                    1,
                )?)
            } else {
                None
            };
            (g, b, sw)
        };

        let gram = gram_dev.to_host(pool);
        let xty = xty_dev.to_host(pool);
        let xtsw = match &xtsw_dev {
            Some(a) => a.to_host(pool),
            None => vec![0.0f64; d],
        };

        gram_dev.release_into(pool);
        xty_dev.release_into(pool);
        if let Some(a) = xtsw_dev {
            a.release_into(pool);
        }
        drop(ones_or_sqw);
        zero_1.release_into(pool);
        zeros_d.release_into(pool);
        zeros_y.release_into(pool);
        if let Some((xm, ym)) = means {
            xm.release_into(pool);
            ym.release_into(pool);
        }
        if let Some(w) = w_dev {
            w.release_into(pool);
        }

        Ok((x_offset, y_offset, gram, xty, xtsw))
    }

    /// The streaming sweep: one [`gcv_cov_sweep`] launch, then the per-block
    /// fold.
    ///
    /// `v` is the `d × d` eigenvector matrix (`v[j·d + k]`), and `g`/`gz`/`gzsw`
    /// the per-alpha spectral weights the caller formed from the spectrum
    /// (`n_alphas × d`, `n_alphas × d × n_y`, `n_alphas × d`). `x_offset` /
    /// `y_offset` are the means the sweep re-applies as it stages each row.
    ///
    /// The `nblocks × n_alphas × n_y` partial is folded HERE rather than in a
    /// second kernel: it is at most a few hundred thousand `f64` against the
    /// `n · d²` the launch just did, and folding it on the host keeps the
    /// summation order the same shape as the host arm's per-worker fold.
    #[allow(clippy::too_many_arguments)]
    pub fn sweep(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x_offset: &[f64],
        y_offset: &[f64],
        v: &[f64],
        g: &[f64],
        gz: &[f64],
        gzsw: &[f64],
        n_alphas: usize,
        sw_sum: f64,
        fit_intercept: bool,
        want_predictions: bool,
        emit_values: bool,
    ) -> Result<GcvSweepOut, PrimError> {
        let (n, d, n_y) = (self.n, self.d, self.n_y);
        if v.len() != d * d
            || g.len() != n_alphas * d
            || gz.len() != n_alphas * d * n_y
            || gzsw.len() != n_alphas * d
            || x_offset.len() != d
            || y_offset.len() != n_y
        {
            return Err(PrimError::ShapeMismatch {
                operand: "ridge_gcv.sweep operands",
                rows: d,
                cols: n_alphas,
                len: v.len(),
            });
        }
        guard_u32("n_alphas", n_alphas)?;

        let (nb, rpb) = sweep_blocking(n, n_alphas, n_y);
        let part_len = nb * n_alphas * n_y;
        let cv_len = if emit_values { n * n_alphas * n_y } else { 1 };

        let xm = DeviceArray::<ActiveRuntime, f64>::from_host(pool, x_offset);
        let ym = DeviceArray::<ActiveRuntime, f64>::from_host(pool, y_offset);
        let v_dev = DeviceArray::<ActiveRuntime, f64>::from_host(pool, v);
        let g_dev = DeviceArray::<ActiveRuntime, f64>::from_host(pool, g);
        let gz_dev = DeviceArray::<ActiveRuntime, f64>::from_host(pool, gz);
        let gzsw_dev = DeviceArray::<ActiveRuntime, f64>::from_host(pool, gzsw);
        let part = pool.acquire(part_len * 8);
        let cv = pool.acquire(cv_len * 8);
        let client = pool.client().clone();

        // SAFETY: every length above is either the carried element count of a
        // live DeviceArray or a freshly acquired allocation of exactly that many
        // f64 slots. The kernel bounds-checks its cube id against `nblocks`,
        // clamps each block's row range to `n`, and walks `d` / `n_y` /
        // `n_alphas` only while below the scalars passed here.
        let x_arg = unsafe { ArrayArg::from_raw_parts(self.x.handle().clone(), self.x.len()) };
        let y_arg = unsafe { ArrayArg::from_raw_parts(self.y.handle().clone(), self.y.len()) };
        let xm_arg = unsafe { ArrayArg::from_raw_parts(xm.handle().clone(), d) };
        let ym_arg = unsafe { ArrayArg::from_raw_parts(ym.handle().clone(), n_y) };
        let sw_arg =
            unsafe { ArrayArg::from_raw_parts(self.sqrt_sw.handle().clone(), self.sqrt_sw.len()) };
        let v_arg = unsafe { ArrayArg::from_raw_parts(v_dev.handle().clone(), d * d) };
        let g_arg = unsafe { ArrayArg::from_raw_parts(g_dev.handle().clone(), n_alphas * d) };
        let gz_arg =
            unsafe { ArrayArg::from_raw_parts(gz_dev.handle().clone(), n_alphas * d * n_y) };
        let gzsw_arg = unsafe { ArrayArg::from_raw_parts(gzsw_dev.handle().clone(), n_alphas * d) };
        let p_arg = unsafe { ArrayArg::from_raw_parts(part.clone(), part_len) };
        let cv_arg = unsafe { ArrayArg::from_raw_parts(cv.clone(), cv_len) };

        // 64 units, ALWAYS. The kernel's two phases want different widths —
        // the projection is `d`-wide, the contraction `n_alphas`-wide — so a
        // 30-alpha grid leaves half a 64-wide cube idle through the second
        // phase, and narrowing to 32 looks like free utilization. It is not:
        // MEASURED on gfx1151 at `n = 100 000`, 30 alphas, a 32-wide cube took
        // the sweep from 61 ms to 101 ms at `d = 64` and from 245 ms to 389 ms
        // at `d = 256`. Halving the cube halves the projection's per-cube
        // parallelism and the latency hiding that pays for its `V` reads, which
        // costs more than the idle lanes save. Do not re-add it without a
        // measurement that says otherwise.
        let (cc, cd) = launch_cubes(nb, GCV_CUBE_DIM);
        gcv_cov_sweep::launch::<f64, ActiveRuntime>(
            &client,
            cc,
            cd,
            x_arg,
            y_arg,
            xm_arg,
            ym_arg,
            sw_arg,
            v_arg,
            g_arg,
            gz_arg,
            gzsw_arg,
            p_arg,
            cv_arg,
            n as u32,
            d as u32,
            n_y as u32,
            n_alphas as u32,
            nb as u32,
            rpb as u32,
            sw_sum,
            u32::from(fit_intercept),
            u32::from(self.weighted),
            u32::from(want_predictions),
            u32::from(emit_values),
        );

        let part_dev = DeviceArray::<ActiveRuntime, f64>::from_raw(part, part_len);
        let cv_dev = DeviceArray::<ActiveRuntime, f64>::from_raw(cv, cv_len);
        let part_host = part_dev.to_host(pool);
        let cv_values = if emit_values {
            cv_dev.to_host(pool)
        } else {
            Vec::new()
        };
        part_dev.release_into(pool);
        cv_dev.release_into(pool);
        xm.release_into(pool);
        ym.release_into(pool);
        v_dev.release_into(pool);
        g_dev.release_into(pool);
        gz_dev.release_into(pool);
        gzsw_dev.release_into(pool);

        let score_sums = if want_predictions {
            Vec::new()
        } else {
            let mut acc = vec![0.0f64; n_alphas * n_y];
            for block in part_host.chunks(n_alphas * n_y) {
                for (dst, src) in acc.iter_mut().zip(block.iter()) {
                    *dst += src;
                }
            }
            acc
        };

        Ok(GcvSweepOut {
            score_sums,
            cv_values,
        })
    }
}

/// `(nblocks, rows_per_block)` for [`gcv_cov_sweep`].
///
/// Deliberately not `gram::row_blocking`: that cap sizes a `nblocks · d²`
/// partial and this one is `nblocks · n_alphas · n_y`, which a 200-alpha
/// multi-target fit would blow through at the same block count. Rows per block
/// are also rounded UP to a whole [`GCV_ROW_TILE`], so no block wastes a partial
/// tile.
fn sweep_blocking(n: usize, n_alphas: usize, n_y: usize) -> (usize, usize) {
    let nb_cap = ((4usize << 20) / n_alphas.saturating_mul(n_y).max(1)).max(1);
    let nb = n.div_ceil(256).clamp(1, nb_cap);
    let tile = GCV_ROW_TILE as usize;
    let rpb = n.div_ceil(nb).div_ceil(tile) * tile;
    (n.div_ceil(rpb), rpb)
}

/// WR-03: reject a `usize` dimension that does not fit the kernel-launch `u32`
/// (an unguarded `as u32` truncation becomes an out-of-bounds device read).
fn guard_u32(operand: &'static str, dim: usize) -> Result<(), PrimError> {
    if dim > u32::MAX as usize {
        return Err(PrimError::ShapeMismatch {
            operand,
            rows: dim,
            cols: 0,
            len: u32::MAX as usize,
        });
    }
    Ok(())
}
