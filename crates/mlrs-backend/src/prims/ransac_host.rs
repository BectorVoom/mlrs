//! `ransac_host` — the host compute engine behind `RANSACRegressor` (RANSAC-01).
//!
//! RANSAC is a *driver*: it draws `max_trials` random sub-samples, fits a base
//! model to each, scores every training row against it, and keeps the model with
//! the largest consensus set. This module owns the arithmetic; the loop,
//! the numpy-MT19937 draw sequence and the stop bookkeeping live in
//! [`mlrs_algos::linear::ransac`], which is where they can reach the
//! `model_selection` RNG.
//!
//! ## Why this engine is the DEFAULT on every backend
//! Two independent reasons, both measured elsewhere in this crate:
//!
//! 1. **The per-trial pass is a single `n × d` matvec fused with an `O(n)`
//!    threshold.** That is the exact shape [`linear_predict`](super::linear_predict)
//!    already found to be memory-bandwidth-bound on the host and *launch*-bound
//!    on a device — and RANSAC runs it a hundred times with a different `coef`
//!    each time, so a per-trial device arm would pay a hundred launches plus a
//!    hundred `coef` uploads and a hundred host syncs to read back the inlier
//!    count the NEXT draw's stopping rule needs. The sync is unavoidable:
//!    `max_trials` shrinks as a function of the best consensus found so far.
//! 2. **The sub-sample fit is a `min_samples × d` least-squares solve**, with
//!    `min_samples = d + 1` by default. A `d`-column Jacobi rotation sweep on
//!    eleven rows is nothing but launch overhead on a GPU
//!    ([[mlrs-gpu-perf-root-cause]]).
//!
//! So this is the `gmm_host` / `hgb_host` situation again, and it takes the same
//! shape: one persistent [`WorkerPool`] spawned per fit, every trial dispatched
//! as one pass over fixed row blocks.
//!
//! [`ransac_device`](super::ransac_device) (RANSAC-02) answers reason 1 — and
//! only reason 1 — by scanning a BATCH of mutually independent trials per
//! launch, so the launch and the stall are paid per batch rather than per trial.
//! Reason 2 stands unchanged: the sub-sample solve stays here on both arms, and
//! [`subset_lstsq_batch`](RansacHostEngine::subset_lstsq_batch) is where a batch
//! of them is spread across this pool. The device arm is opt-in
//! (`RANSAC_DEVICE_MIN_WORK`); this one runs by default everywhere.
//!
//! ## Determinism of the parallel reductions
//! The row axis is split into fixed-size [`SCAN_BLOCK`] blocks whose boundaries
//! depend only on `n`, never on the worker count. Each block's partial sums are
//! written to its own slot and folded **in block order** by the driver, so the
//! floating-point result of a scan is identical at one worker and at sixteen.
//! That property is load-bearing: the consensus tie-break compares two R² values
//! for equality-of-inlier-count, and a reduction that reassociated with the
//! thread count could pick a different final model on a different machine.
//!
//! ## What it does NOT reproduce bit-for-bit
//! `numpy`'s `X @ coef` is a BLAS `gemv` whose blocking mlrs does not model, so
//! a residual can differ from sklearn's in the last ulps and a row sitting
//! *exactly* on `residual_threshold` can land on the other side of it. That is
//! the same accumulation-order caveat [`linear_predict_host`] documents, and it
//! is why the oracle fixture is generated with a clear inlier/outlier margin.
//!
//! Tests live in `crates/mlrs-backend/tests/ransac_host_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_core::PrimError;

use crate::capability;
use crate::prims::host_pool::{Shared, WorkerPool};
#[cfg(target_arch = "x86_64")]
use crate::prims::host_simd::avx2_available;

/// Rows per scan block. Fixed so the reduction order is independent of the
/// worker count (module docs).
///
/// It also caps the achievable width — a pool never gets more units than there
/// are blocks to hand out — which is what sets the value. At 8192 a 100k-row
/// design has only 13 blocks, and that rung measured 8 GB/s against the 28 GB/s
/// the 200k-row rung reached; at 4096 it has 25, and the same rung saturates.
const SCAN_BLOCK: usize = 4096;

/// Row-block-work below which the pool is not worth entering.
///
/// Deliberately an eighth of [`linear_predict`](super::linear_predict)'s
/// `HOST_ELEMS_PER_UNIT`, which sizes a ONE-SHOT predict that must pay a thread
/// SPAWN per call. Here the pool is persistent for the whole fit and a pass
/// costs two barrier crossings (single-digit microseconds), so the width worth
/// entering is far smaller — and a fit makes a hundred of these passes, not
/// one.
const MIN_WORK_PER_UNIT: usize = 1 << 15;

/// Independent accumulator lanes in the row dot product — the
/// [`linear_predict`](super::linear_predict) `HOST_DOT_LANES` constant,
/// re-affirmed locally (the `spectral_clustering` precedent) because it is a
/// property of THIS loop's vectorization, not a shared contract.
const DOT_LANES: usize = 8;

/// Jacobi sweeps the sub-sample SVD will run before giving up.
///
/// A one-sided sweep is quadratically convergent and a well-scaled system is
/// done in six to ten; the cap is set far above that because this is the
/// FALLBACK path, taken only for the rank-deficient and under-determined draws
/// the QR declines, and those are exactly the ones that converge slowest. It
/// exists so a pathological input cannot spin, not as a tuning knob.
const MAX_JACOBI_SWEEPS: usize = 30;

/// Relative off-diagonality below which a Jacobi rotation is skipped.
const JACOBI_TOL: f64 = 1e-15;

/// `np.spacing(1)` — sklearn `_ransac._EPSILON`, the floor both terms of the
/// dynamic-trial formula are clamped to.
const NP_SPACING_ONE: f64 = 2.220446049250313e-16;

/// sklearn's `RANSACRegressor(loss=...)`, the estimator's ONE string-valued
/// parameter.
///
/// Both variants reduce a row's per-target error to a single scalar the
/// `residual_threshold` is compared against: sklearn sums over the target axis
/// for a 2-D `y` (`np.sum(..., axis=1)`) and takes the bare value for a 1-D one,
/// which is the `n_targets == 1` case of the same sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RansacLoss {
    /// sklearn `"absolute_error"` (the default) — `Σ_k |y_k − ŷ_k|`.
    AbsoluteError,
    /// sklearn `"squared_error"` — `Σ_k (y_k − ŷ_k)²`.
    SquaredError,
}

impl RansacLoss {
    /// The sklearn spelling, for round-tripping a fitted estimator's report.
    pub fn as_str(self) -> &'static str {
        match self {
            RansacLoss::AbsoluteError => "absolute_error",
            RansacLoss::SquaredError => "squared_error",
        }
    }
}

/// What one trial's scan of the FULL design produced.
#[derive(Debug, Clone)]
pub struct TrialScan {
    /// Rows whose loss is `<= residual_threshold`. sklearn's comparison is
    /// non-strict ("Points whose residuals are strictly equal to the threshold
    /// are considered as inliers").
    pub n_inliers: usize,
    /// `Σ_inliers (y_k − ŷ_k)²` per target — the numerator of the R² the
    /// consensus tie-break needs, accumulated during the same pass rather than
    /// in a second walk over the design.
    pub sq_err: Vec<f64>,
}

/// The host engine: borrows the design for the length of one fit and owns the
/// worker pool every trial dispatches through.
pub struct RansacHostEngine<'a, F> {
    x: &'a [F],
    y: &'a [F],
    n: usize,
    d: usize,
    t: usize,
    units: usize,
    pool: WorkerPool,
    /// `(lo, hi)` of every fixed row block (module docs).
    blocks: Vec<(usize, usize)>,
}

impl<'a, F> RansacHostEngine<'a, F>
where
    F: Float + CubeElement + Pod,
{
    /// Borrow an `n × d` row-major design and its `n × t` row-major targets.
    ///
    /// Spawns the pool, so build it ONCE per fit and drive every trial through
    /// it — that is the whole reason this is a struct rather than a free
    /// function ([`WorkerPool`] docs).
    ///
    /// `work_per_pass` is the element count ONE dispatched pass reads, and it is
    /// the caller's to supply because the arms do genuinely different passes:
    ///
    /// | arm | dispatched pass | `work_per_pass` |
    /// |---|---|---|
    /// | native, host scan | [`scan`](Self::scan) over the design | `n · d · t` |
    /// | native, device scan | [`subset_lstsq_batch`](Self::subset_lstsq_batch) | `batch · m · d` |
    /// | arbitrary base | [`scan_pred`](Self::scan_pred) over the targets | `n · t` |
    ///
    /// ## Why this is not derived from `(n, d, t)` here
    /// It used to be, and the arbitrary-base arm paid 2.7× for it. That arm's
    /// pass has NO `d` factor — it reads `y` and the caller's predictions and
    /// nothing else — so an `n · d`-sized pool spawned fifteen threads to split
    /// half a megabyte, and then crossed two barriers per trial around a gap
    /// filled by the caller's Python `fit`/`predict`. The barrier's spin-then-
    /// yield backoff ([`host_pool`](super::host_pool)) is tuned for
    /// back-to-back passes; across a millisecond-long foreign call it burns
    /// cores that numpy's own BLAS pool is simultaneously trying to spin on.
    /// Measured at `n = 50 000, d = 32` with a `Ridge` base: 542 ms at sixteen
    /// units, 247 ms at four, **197 ms at one**. Sized from `n · t` the planner
    /// picks one unit there by itself, and only widens where a pass is actually
    /// big enough to split.
    pub fn new(
        x: &'a [F],
        y: &'a [F],
        n: usize,
        d: usize,
        t: usize,
        work_per_pass: usize,
    ) -> Result<Self, PrimError> {
        if x.len() != n * d {
            return Err(PrimError::ShapeMismatch {
                operand: "ransac_x",
                rows: n,
                cols: d,
                len: x.len(),
            });
        }
        if y.len() != n * t {
            return Err(PrimError::ShapeMismatch {
                operand: "ransac_y",
                rows: n,
                cols: t,
                len: y.len(),
            });
        }
        let blocks: Vec<(usize, usize)> = (0..n.div_ceil(SCAN_BLOCK.max(1)))
            .map(|b| (b * SCAN_BLOCK, ((b + 1) * SCAN_BLOCK).min(n)))
            .collect();
        let units = plan_units(work_per_pass, blocks.len());
        Ok(Self {
            x,
            y,
            n,
            d,
            t,
            units,
            pool: WorkerPool::new(units),
            blocks,
        })
    }

    /// Rows in the design.
    pub fn n_samples(&self) -> usize {
        self.n
    }
    /// Columns in the design.
    pub fn n_features(&self) -> usize {
        self.d
    }
    /// Target columns (`1` for a 1-D `y`).
    pub fn n_targets(&self) -> usize {
        self.t
    }
    /// Pool participants, driver included. Reported by the perf probe.
    pub fn units(&self) -> usize {
        self.units
    }

    /// sklearn's default `residual_threshold`: `np.median(np.abs(y − np.median(y)))`
    /// over the FLATTENED targets (numpy's `median` with no `axis`).
    pub fn target_mad(&self) -> f64 {
        let mut v: Vec<f64> = self.y.iter().map(|&e| to_f64(e)).collect();
        let med = median_in_place(&mut v);
        for e in v.iter_mut() {
            *e = (*e - med).abs();
        }
        median_in_place(&mut v)
    }

    /// Score every row against `(coef, intercept)`, write the inlier mask and
    /// report the consensus size plus the R² numerator.
    ///
    /// `coef` is `t × d` row-major and `intercept` is length `t`, both `f64`
    /// (the sub-sample solve runs in `f64` whatever the design's width — the
    /// `bayesian_ridge`/`huber` precision precedent). They are narrowed to the
    /// design's width once, here, because sklearn's `estimator.predict` runs in
    /// the design's dtype and the residual comparison must too.
    ///
    /// The one-trial case of [`scan_batch`](Self::scan_batch), spelled out
    /// because it is what the host arm runs: the host engine deliberately does
    /// NOT speculate a batch of trials (that exists to amortize a device
    /// launch — see [`scan_batch`](Self::scan_batch)), so this is the shape it
    /// actually takes.
    pub fn scan(
        &self,
        coef: &[f64],
        intercept: &[f64],
        loss: RansacLoss,
        threshold: f64,
        mask: &mut [bool],
    ) -> TrialScan {
        let mut out = self.scan_batch(coef, intercept, 1, loss, threshold, mask);
        out.pop().expect("scan_batch(batch = 1) yields one scan")
    }

    /// Score every row against EACH of `batch` candidate models in one pass over
    /// the design, writing `batch` consecutive inlier masks.
    ///
    /// `coef` is `batch` consecutive `t × d` row-major blocks, `intercept`
    /// `batch` consecutive length-`t` blocks, `mask` is `batch · n` long. The
    /// returned scans are in trial order.
    ///
    /// ## Why a host batch exists at all
    /// The device arm ([`ransac_device`](super::ransac_device)) speculates a
    /// batch of trials to amortize its launch and its host stall; this is the
    /// same computation on the host, which the arms-agree tests A/B against it,
    /// and it is what a caller gets when the device arm is unavailable but the
    /// batched shape has already been planned. The host arm's own driver uses
    /// `batch = 1`: with no launch to amortize, speculation buys nothing and the
    /// surplus trials a stop rule discards would be pure waste.
    ///
    /// The parallel work item is a `(trial, row-block)` PAIR, so a batch of one
    /// trial still spreads across the pool exactly as the unbatched scan did,
    /// and the fold is over `(trial, block)` in that order — independent of the
    /// worker count, which is the determinism property the module docs pin.
    pub fn scan_batch(
        &self,
        coef: &[f64],
        intercept: &[f64],
        batch: usize,
        loss: RansacLoss,
        threshold: f64,
        mask: &mut [bool],
    ) -> Vec<TrialScan> {
        debug_assert_eq!(coef.len(), batch * self.t * self.d);
        debug_assert_eq!(intercept.len(), batch * self.t);
        debug_assert_eq!(mask.len(), batch * self.n);

        // Per-(trial, block) partials, folded in that order by the driver
        // (module docs).
        let nb = self.blocks.len();
        let items = batch * nb;
        let mut part_n = vec![0usize; items];
        let mut part_sq = vec![0.0f64; items * self.t];

        {
            let sh_mask = Shared::new(mask);
            let sh_n = Shared::new(&mut part_n);
            let sh_sq = Shared::new(&mut part_sq);
            let (n, d, t, units) = (self.n, self.d, self.t, self.units);
            let blocks = &self.blocks;
            let (x, y) = (self.x, self.y);
            let pass = move |u: usize| {
                // Item `it` is owned by exactly one unit, so the `Shared`
                // writes below are disjoint by construction.
                let mut it = u;
                while it < items {
                    let (trial, b) = (it / nb, it % nb);
                    let (lo, hi) = blocks[b];
                    // SAFETY: unit `u` owns item `it` alone (the stride above),
                    // and each item writes only `mask[trial*n + lo .. +hi]`,
                    // `part_n[it]` and `part_sq[it*t..(it+1)*t]`.
                    let m = unsafe { &mut sh_mask.get_mut()[trial * n + lo..trial * n + hi] };
                    let nslot = unsafe { &mut sh_n.get_mut()[it..it + 1] };
                    let sslot = unsafe { &mut sh_sq.get_mut()[it * t..(it + 1) * t] };
                    let cnt = scan_block_dispatch::<F>(
                        &x[lo * d..hi * d],
                        &y[lo * t..hi * t],
                        &coef[trial * t * d..(trial + 1) * t * d],
                        &intercept[trial * t..(trial + 1) * t],
                        d,
                        t,
                        loss,
                        threshold,
                        m,
                        sslot,
                    );
                    nslot[0] = cnt;
                    it += units;
                }
            };
            self.pool.run(&pass);
        }

        (0..batch)
            .map(|trial| {
                let mut n_inliers = 0usize;
                let mut sq_err = vec![0.0f64; self.t];
                for b in 0..nb {
                    let it = trial * nb + b;
                    n_inliers += part_n[it];
                    for k in 0..self.t {
                        sq_err[k] += part_sq[it * self.t + k];
                    }
                }
                TrialScan { n_inliers, sq_err }
            })
            .collect()
    }

    /// The same verdict as [`scan`](Self::scan), for a candidate model whose
    /// predictions were computed ELSEWHERE — the arbitrary-base-estimator path.
    ///
    /// `y_pred` is `n × t` row-major `f64`, exactly what
    /// `estimator.predict(X)` produced for the whole training design.
    /// `resid` is `None` for the two string losses (this pass forms them from
    /// `y_pred`) and `Some(n)` when the caller's `loss` is a CALLABLE, whose
    /// answer only Python can produce and which is then compared against the
    /// threshold as sklearn compares it.
    ///
    /// ## Why the predictions are narrowed to `F` first
    /// sklearn's residual is `y − estimator.predict(X)` evaluated in the
    /// DESIGN's dtype: on an `f32` design its `y_pred` is `f32` and the
    /// subtraction, the absolute value and the threshold comparison all happen
    /// there. The bridge hands predictions over as `f64` because that is the one
    /// width the boundary speaks, so this narrows them back before subtracting —
    /// otherwise a row within an `f32` ulp of `residual_threshold` could be
    /// classified differently here than in sklearn, which is a CONTROL-FLOW
    /// difference and not a rounding one.
    ///
    /// The squared errors are accumulated in `f64` from those same `F`
    /// differences, matching [`scan`](Self::scan)'s own bookkeeping.
    pub fn scan_pred(
        &self,
        y_pred: &[f64],
        resid: Option<&[f64]>,
        loss: RansacLoss,
        threshold: f64,
        mask: &mut [bool],
    ) -> TrialScan {
        debug_assert_eq!(y_pred.len(), self.n * self.t);
        debug_assert_eq!(mask.len(), self.n);

        let nb = self.blocks.len();
        let mut part_n = vec![0usize; nb];
        let mut part_sq = vec![0.0f64; nb * self.t];

        {
            let sh_mask = Shared::new(mask);
            let sh_n = Shared::new(&mut part_n);
            let sh_sq = Shared::new(&mut part_sq);
            let (t, units) = (self.t, self.units);
            let blocks = &self.blocks;
            let y = self.y;
            let pass = move |u: usize| {
                let mut b = u;
                while b < blocks.len() {
                    let (lo, hi) = blocks[b];
                    // SAFETY: unit `u` owns block `b` alone (the stride below),
                    // and each block writes only its own disjoint slices.
                    let m = unsafe { &mut sh_mask.get_mut()[lo..hi] };
                    let nslot = unsafe { &mut sh_n.get_mut()[b..b + 1] };
                    let sslot = unsafe { &mut sh_sq.get_mut()[b * t..(b + 1) * t] };
                    let cnt = pred_block_dispatch::<F>(
                        &y[lo * t..hi * t],
                        &y_pred[lo * t..hi * t],
                        resid.map(|r| &r[lo..hi]),
                        t,
                        loss,
                        threshold,
                        m,
                        sslot,
                    );
                    nslot[0] = cnt;
                    b += units;
                }
            };
            self.pool.run(&pass);
        }

        let mut n_inliers = 0usize;
        let mut sq_err = vec![0.0f64; self.t];
        for b in 0..nb {
            n_inliers += part_n[b];
            for k in 0..self.t {
                sq_err[k] += part_sq[b * self.t + k];
            }
        }
        TrialScan { n_inliers, sq_err }
    }

    /// sklearn's `RegressorMixin.score` on the consensus set: `r2_score` with
    /// `multioutput="uniform_average"`, evaluated on the inlier rows with the
    /// residuals [`scan`](Self::scan) already produced.
    ///
    /// ## Why the predictions are not recomputed
    /// sklearn calls `estimator.score(X[inliers], y[inliers])`, which runs a
    /// SECOND `predict` over the consensus set — and over a fancy-indexed COPY
    /// of it. A linear prediction is a per-row dot product with no cross-row
    /// term, so `predict(X)[i] == predict(X[inliers])[j]` for the row those
    /// index the same, and the numerator is exactly the `sq_err` the scan
    /// already summed. The copy and the second matvec are pure overhead and are
    /// simply not performed.
    ///
    /// The denominator DOES need a walk: `Σ (y − ȳ_inliers)²` is a two-pass
    /// quantity and the one-pass `Σy² − n·ȳ²` identity is not numerically the
    /// same sum. That walk is `O(n · t)` with no `d` factor, and it only runs on
    /// a trial that has already beaten the incumbent.
    pub fn r2_on_mask(&self, mask: &[bool], n_inliers: usize, sq_err: &[f64]) -> f64 {
        // sklearn's `r2_score` warns and returns NaN below two samples.
        if n_inliers < 2 {
            return f64::NAN;
        }
        let t = self.t;
        let mut mean = vec![0.0f64; t];
        for (i, &m) in mask.iter().enumerate() {
            if m {
                for (k, acc) in mean.iter_mut().enumerate() {
                    *acc += to_f64(self.y[i * t + k]);
                }
            }
        }
        for m in mean.iter_mut() {
            *m /= n_inliers as f64;
        }
        let mut den = vec![0.0f64; t];
        for (i, &m) in mask.iter().enumerate() {
            if m {
                for k in 0..t {
                    let dv = to_f64(self.y[i * t + k]) - mean[k];
                    den[k] += dv * dv;
                }
            }
        }
        self.r2_from_sums(n_inliers, sq_err, &den)
    }

    /// sklearn's `r2_score` verdict from the two sums that define it, for a
    /// caller that already has them.
    ///
    /// Split out of [`r2_on_mask`](Self::r2_on_mask) so the device arm — whose
    /// denominator is formed by
    /// [`ransac_den_block`](mlrs_kernels::ransac::ransac_den_block) against the
    /// mask it kept resident, precisely so no `n`-length array crosses the bus —
    /// scores through the SAME arithmetic as the host arm instead of a second
    /// transcription of sklearn's per-output branch.
    pub fn r2_from_sums(&self, n_inliers: usize, sq_err: &[f64], den: &[f64]) -> f64 {
        // sklearn's `r2_score` warns and returns NaN below two samples.
        if n_inliers < 2 {
            return f64::NAN;
        }
        // sklearn's per-output verdict: a zero denominator scores 1.0 when the
        // numerator is zero too (a constant target predicted exactly) and 0.0
        // otherwise; everything else is the ordinary `1 − num/den`.
        let mut acc = 0.0f64;
        for k in 0..self.t {
            acc += match (den[k] != 0.0, sq_err[k] != 0.0) {
                (true, true) => 1.0 - sq_err[k] / den[k],
                (true, false) => 1.0,
                (false, true) => 0.0,
                (false, false) => 1.0,
            };
        }
        acc / self.t as f64
    }

    /// Least-squares fit of the base `LinearRegression` to the rows `idxs`
    /// selects, returning `(coef[t·d], intercept[t])` in `f64`.
    ///
    /// This is sklearn's `LinearRegression.fit` on `X[idxs], y[idxs]`, in order:
    /// weighted column means removed when `fit_intercept`
    /// (`_preprocess_data`), rows scaled by `√w` when weights are given
    /// (`_rescale_data`), then `scipy.linalg.lstsq` — the minimum-norm solution
    /// through the SVD with singular values `σ ≤ eps·σ_max` dropped, `eps` being
    /// the DESIGN dtype's machine epsilon, which is what `linalg.lstsq(a, b)`
    /// passes as `cond` when the caller gives none.
    ///
    /// ## Why the cutoff is `eps` here and `1e-6` in `linear_regression.rs`
    /// The standalone estimator pins `RCOND = 1e-6` because that is what
    /// reproduces sklearn on its near-collinear oracle fixture. RANSAC cannot
    /// use a truncation that aggressive: it deliberately draws `d + 1` rows at
    /// random, so a rank-deficient or badly-conditioned sub-sample is a NORMAL
    /// event that happens several times in a hundred trials, and whether the
    /// solve keeps or drops that direction decides how many inliers the trial
    /// reports — i.e. it changes the CONTROL FLOW, not just a coefficient. The
    /// faithful reproduction is scipy's own default, so that is what runs.
    /// A blown-up solution from a near-singular draw simply scores zero inliers
    /// and is skipped, exactly as it is in sklearn.
    ///
    /// The solve itself is a one-sided Jacobi SVD ([`jacobi_lstsq`]): the
    /// systems are `min_samples × d` with `min_samples = d + 1` by default, and
    /// at that size a rotation sweep is both the accurate choice and cheaper
    /// than anything with a blocking strategy.
    pub fn subset_lstsq(
        &self,
        idxs: &[i64],
        fit_intercept: bool,
        sample_weight: Option<&[f64]>,
    ) -> (Vec<f64>, Vec<f64>) {
        Self::subset_lstsq_impl(self.x, self.y, self.d, self.t, idxs, fit_intercept, sample_weight)
    }

    /// The body of [`subset_lstsq`](Self::subset_lstsq), as an associated
    /// function over the borrowed design rather than a method.
    ///
    /// It is spelled this way for one reason:
    /// [`subset_lstsq_batch`](Self::subset_lstsq_batch) runs these solves
    /// concurrently, and [`WorkerPool`] is deliberately `!Sync` — so the pass
    /// closure cannot capture `&self`. It captures the two design slices (which
    /// ARE `Sync`) and calls this instead.
    #[allow(clippy::too_many_arguments)]
    fn subset_lstsq_impl(
        x: &[F],
        y: &[F],
        d: usize,
        t: usize,
        idxs: &[i64],
        fit_intercept: bool,
        sample_weight: Option<&[f64]>,
    ) -> (Vec<f64>, Vec<f64>) {
        let m = idxs.len();
        // COLUMN-major (`a[c·m + r]`, `b[k·m + r]`) for the whole solve. Every
        // consumer wants it that way: the centering and the `√w` rescale are
        // per-column passes, the Householder reflectors walk a column, and the
        // Jacobi fallback orthogonalizes columns. The design is row-major, so
        // the transpose happens once here, during the gather that had to touch
        // every element anyway. Measured at `d = 128`, `m = 129` (isolated by
        // shrinking `n` until only the solve is left) this took the sub-sample
        // solve from 1.33 ms to 0.58 ms per trial, and the `50k x 128` fit from
        // 0.40 s to 0.20 s — the strided 1 KiB-apart loads of the row-major
        // form were the whole difference.
        let mut a = vec![0.0f64; m * d];
        let mut b = vec![0.0f64; m * t];
        for (r, &gi) in idxs.iter().enumerate() {
            let g = gi as usize;
            for c in 0..d {
                a[c * m + r] = to_f64(x[g * d + c]);
            }
            for k in 0..t {
                b[k * m + r] = to_f64(y[g * t + k]);
            }
        }
        let w: Option<Vec<f64>> =
            sample_weight.map(|sw| idxs.iter().map(|&g| sw[g as usize]).collect());

        // `_preprocess_data`: weighted column means, removed from both sides.
        let mut x_off = vec![0.0f64; d];
        let mut y_off = vec![0.0f64; t];
        if fit_intercept {
            let wsum = w.as_ref().map_or(m as f64, |w| w.iter().sum::<f64>());
            let inv = if wsum != 0.0 { 1.0 / wsum } else { 0.0 };
            for c in 0..d {
                let col = &mut a[c * m..(c + 1) * m];
                let mut acc = 0.0f64;
                match w.as_ref() {
                    Some(w) => {
                        for (r, &v) in col.iter().enumerate() {
                            acc += w[r] * v;
                        }
                    }
                    None => {
                        for &v in col.iter() {
                            acc += v;
                        }
                    }
                }
                let mean = acc * inv;
                x_off[c] = mean;
                for v in col.iter_mut() {
                    *v -= mean;
                }
            }
            for k in 0..t {
                let col = &mut b[k * m..(k + 1) * m];
                let mut acc = 0.0f64;
                match w.as_ref() {
                    Some(w) => {
                        for (r, &v) in col.iter().enumerate() {
                            acc += w[r] * v;
                        }
                    }
                    None => {
                        for &v in col.iter() {
                            acc += v;
                        }
                    }
                }
                let mean = acc * inv;
                y_off[k] = mean;
                for v in col.iter_mut() {
                    *v -= mean;
                }
            }
        }
        // `_rescale_data`: both sides scaled by √w. Applied AFTER centering,
        // which is sklearn's order.
        if let Some(w) = w.as_ref() {
            let root: Vec<f64> = w.iter().map(|v| v.max(0.0).sqrt()).collect();
            for col in a.chunks_mut(m).chain(b.chunks_mut(m)) {
                for (r, v) in col.iter_mut().enumerate() {
                    *v *= root[r];
                }
            }
        }

        // scipy's `cond = np.finfo(a.dtype).eps` — the DESIGN's width, not the
        // f64 the solve runs in.
        let rcond = match size_of::<F>() {
            4 => f32::EPSILON as f64,
            _ => f64::EPSILON,
        };
        // The QR fast path handles every WELL-CONDITIONED over-determined
        // sub-sample, which is what a random draw almost always is; the Jacobi
        // SVD stays as the fallback for the ones it is not (see
        // [`householder_qr_lstsq`]). QR DESTROYS its operands, so the SVD's
        // copy is taken first — `m · (d + t)` doubles, against the
        // `O(m·d²)` the solve itself costs.
        let coef = {
            let mut aq = a.clone();
            let mut bq = b.clone();
            match householder_qr_lstsq(&mut aq, &mut bq, m, d, t) {
                Some(c) => c,
                None => jacobi_lstsq(&a, &b, m, d, t, rcond),
            }
        };

        let mut intercept = vec![0.0f64; t];
        if fit_intercept {
            for k in 0..t {
                let mut dot = 0.0f64;
                for c in 0..d {
                    dot += x_off[c] * coef[k * d + c];
                }
                intercept[k] = y_off[k] - dot;
            }
        }
        (coef, intercept)
    }

    /// [`subset_lstsq`](Self::subset_lstsq) for `batch` sub-samples at once,
    /// solved CONCURRENTLY across the pool.
    ///
    /// `idxs` is `batch · m` row indices (trial-major); the results are `batch`
    /// consecutive `t × d` coefficient blocks and `batch` length-`t` intercept
    /// blocks — exactly the layout
    /// [`scan_batch`](Self::scan_batch) and
    /// [`ransac_scan_batch`](mlrs_kernels::ransac::ransac_scan_batch) consume.
    ///
    /// Each solve is independent and writes only its own output slice, so the
    /// answer is bit-identical to solving them one at a time in any order — the
    /// parallelism here cannot reassociate anything, unlike the row reductions
    /// this module is careful about.
    ///
    /// This is where the `d ≥ 64` batch actually pays: with
    /// `min_samples = d + 1` a trial's solve is `O(m·d²)` and DOMINATES the fit
    /// ([[mlrs-ransac]]), and it is the one part of a trial the device arm
    /// deliberately does not take (a `d`-column Householder QR on `d + 1` rows is
    /// launch overhead, and the QR's rank verdict decides control flow, so it
    /// stays exactly where sklearn's `scipy.linalg.lstsq` semantics are already
    /// reproduced).
    pub fn subset_lstsq_batch(
        &self,
        idxs: &[i64],
        m: usize,
        batch: usize,
        fit_intercept: bool,
        sample_weight: Option<&[f64]>,
    ) -> (Vec<f64>, Vec<f64>) {
        debug_assert_eq!(idxs.len(), batch * m);
        let (d, t) = (self.d, self.t);
        let mut coef = vec![0.0f64; batch * t * d];
        let mut icept = vec![0.0f64; batch * t];
        if batch == 1 {
            let (c, b) = self.subset_lstsq(idxs, fit_intercept, sample_weight);
            coef.copy_from_slice(&c);
            icept.copy_from_slice(&b);
            return (coef, icept);
        }
        {
            let sh_c = Shared::new(&mut coef);
            let sh_b = Shared::new(&mut icept);
            let units = self.units;
            let (x, y) = (self.x, self.y);
            let pass = move |u: usize| {
                let mut trial = u;
                while trial < batch {
                    let (c, b) = Self::subset_lstsq_impl(
                        x,
                        y,
                        d,
                        t,
                        &idxs[trial * m..(trial + 1) * m],
                        fit_intercept,
                        sample_weight,
                    );
                    // SAFETY: unit `u` owns trial `trial` alone (the stride
                    // below), and writes only that trial's own output block.
                    let cs = unsafe { &mut sh_c.get_mut()[trial * t * d..(trial + 1) * t * d] };
                    let bs = unsafe { &mut sh_b.get_mut()[trial * t..(trial + 1) * t] };
                    cs.copy_from_slice(&c);
                    bs.copy_from_slice(&b);
                    trial += units;
                }
            };
            self.pool.run(&pass);
        }
        (coef, icept)
    }

    /// Gather `X[idxs]` (row-major `m × d`) — what a caller-supplied
    /// `is_data_valid` / `is_model_valid` predicate is handed.
    pub fn gather_x(&self, idxs: &[i64]) -> Vec<F> {
        let d = self.d;
        let mut out = Vec::with_capacity(idxs.len() * d);
        for &g in idxs {
            out.extend_from_slice(&self.x[g as usize * d..(g as usize + 1) * d]);
        }
        out
    }

    /// Gather `y[idxs]` (row-major `m × t`).
    pub fn gather_y(&self, idxs: &[i64]) -> Vec<F> {
        let t = self.t;
        let mut out = Vec::with_capacity(idxs.len() * t);
        for &g in idxs {
            out.extend_from_slice(&self.y[g as usize * t..(g as usize + 1) * t]);
        }
        out
    }
}

/// sklearn `_ransac._dynamic_max_trials` — how many draws it takes to see one
/// outlier-free sub-sample with probability `probability`, given the best
/// inlier ratio found so far.
///
/// Returned as `f64` because sklearn's is: the `denom == 1` branch yields
/// `inf`, and the caller's `max_trials` is `min`'d against it as a float before
/// the loop compares it to an integer trial counter.
pub fn dynamic_max_trials(
    n_inliers: usize,
    n_samples: usize,
    min_samples: usize,
    probability: f64,
) -> f64 {
    let inlier_ratio = n_inliers as f64 / n_samples as f64;
    let nom = NP_SPACING_ONE.max(1.0 - probability);
    let denom = NP_SPACING_ONE.max(1.0 - inlier_ratio.powi(min_samples as i32));
    if nom == 1.0 {
        return 0.0;
    }
    if denom == 1.0 {
        return f64::INFINITY;
    }
    (nom.ln() / denom.ln()).ceil().abs()
}

// --------------------------------------------------------------------------- //
// The scan kernel
// --------------------------------------------------------------------------- //

/// Size-dispatch onto the two concrete float monomorphizations. `F`'s own
/// arithmetic is CubeCL *kernel* arithmetic, so the host math runs on the
/// bytemuck-cast primitive view (the `linear_predict_host` idiom).
#[allow(clippy::too_many_arguments)]
fn scan_block_dispatch<F: Pod>(
    x: &[F],
    y: &[F],
    coef: &[f64],
    intercept: &[f64],
    d: usize,
    t: usize,
    loss: RansacLoss,
    threshold: f64,
    mask: &mut [bool],
    sq_err: &mut [f64],
) -> usize {
    match size_of::<F>() {
        4 => {
            // Narrowed per BLOCK rather than once per trial: `t * d + t` values
            // against the block's `SCAN_BLOCK * d` multiply-adds, and doing it
            // here keeps the narrowed operands thread-local instead of shared.
            let c: Vec<f32> = coef.iter().map(|&v| v as f32).collect();
            let b: Vec<f32> = intercept.iter().map(|&v| v as f32).collect();
            scan_block::<f32>(
                bytemuck::cast_slice(x),
                bytemuck::cast_slice(y),
                &c,
                &b,
                d,
                t,
                loss,
                threshold,
                mask,
                sq_err,
            )
        }
        8 => scan_block::<f64>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            coef,
            intercept,
            d,
            t,
            loss,
            threshold,
            mask,
            sq_err,
        ),
        other => unreachable!("ransac_host is f32/f64 only, got a {other}-byte element"),
    }
}

/// One block's rows, run on the machine's REAL vector unit.
///
/// The `t == 1` case — every 1-D `y`, i.e. essentially every RANSAC — gets its
/// own body so the inner loop is the bare dot product with no target axis to
/// carry, which is what LLVM will vectorize.
#[allow(clippy::too_many_arguments)]
#[inline]
fn scan_block<T: HostFloat>(
    x: &[T],
    y: &[T],
    coef: &[T],
    intercept: &[T],
    d: usize,
    t: usize,
    loss: RansacLoss,
    threshold: f64,
    mask: &mut [bool],
    sq_err: &mut [f64],
) -> usize {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: guarded by the runtime detection this branch tests; the body
        // is the ordinary row loop and contains nothing unsafe.
        return unsafe {
            scan_block_avx2(x, y, coef, intercept, d, t, loss, threshold, mask, sq_err)
        };
    }
    scan_block_inner(x, y, coef, intercept, d, t, loss, threshold, mask, sq_err)
}

/// [`scan_block_inner`] compiled for AVX2 + FMA. Widening the accumulator lanes
/// reassociates nothing — each lane still sums its own subsequence in its own
/// order — so this cannot change a result ([`host_simd`](super::host_simd)).
///
/// # Safety
/// The caller must have established that the CPU supports `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn scan_block_avx2<T: HostFloat>(
    x: &[T],
    y: &[T],
    coef: &[T],
    intercept: &[T],
    d: usize,
    t: usize,
    loss: RansacLoss,
    threshold: f64,
    mask: &mut [bool],
    sq_err: &mut [f64],
) -> usize {
    scan_block_inner(x, y, coef, intercept, d, t, loss, threshold, mask, sq_err)
}

/// The body both [`scan_block`] arms share.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn scan_block_inner<T: HostFloat>(
    x: &[T],
    y: &[T],
    coef: &[T],
    intercept: &[T],
    d: usize,
    t: usize,
    loss: RansacLoss,
    threshold: f64,
    mask: &mut [bool],
    sq_err: &mut [f64],
) -> usize {
    let mut n_in = 0usize;
    if t == 1 {
        let b0 = intercept[0];
        let mut acc = 0.0f64;
        for (r, m) in mask.iter_mut().enumerate() {
            let diff = y[r] - (host_dot(&x[r * d..(r + 1) * d], coef) + b0);
            let sq = diff * diff;
            let resid = match loss {
                RansacLoss::AbsoluteError => diff.abs(),
                RansacLoss::SquaredError => sq,
            };
            // Branch-free so the loop stays vectorizable: the mask store and
            // both accumulations are predicated, not jumped over.
            let inl = resid.to_f64() <= threshold;
            *m = inl;
            n_in += usize::from(inl);
            acc += if inl { sq.to_f64() } else { 0.0 };
        }
        sq_err[0] = acc;
        return n_in;
    }
    for (r, m) in mask.iter_mut().enumerate() {
        let row = &x[r * d..(r + 1) * d];
        let mut resid = 0.0f64;
        // Two passes over `t` (which is tiny) rather than one, so the residual
        // is complete before the mask decides whether to bank the squares.
        let mut sq = [0.0f64; SMALL_T];
        let mut spill: Vec<f64> = Vec::new();
        if t > SMALL_T {
            spill = vec![0.0; t];
        }
        for k in 0..t {
            let diff = y[r * t + k] - (host_dot(row, &coef[k * d..(k + 1) * d]) + intercept[k]);
            let s = (diff * diff).to_f64();
            resid += match loss {
                RansacLoss::AbsoluteError => diff.abs().to_f64(),
                RansacLoss::SquaredError => s,
            };
            if t > SMALL_T {
                spill[k] = s;
            } else {
                sq[k] = s;
            }
        }
        let inl = resid <= threshold;
        *m = inl;
        if inl {
            n_in += 1;
            for k in 0..t {
                sq_err[k] += if t > SMALL_T { spill[k] } else { sq[k] };
            }
        }
    }
    n_in
}

/// Target columns the multi-output scan keeps on the stack. Past this it spills
/// to a heap vector; `t > 8` regressions are not a shape RANSAC is used for.
const SMALL_T: usize = 8;

/// [`scan_block_dispatch`] for a block whose predictions were computed
/// elsewhere — the arbitrary-base-estimator path
/// ([`RansacHostEngine::scan_pred`]).
#[allow(clippy::too_many_arguments)]
fn pred_block_dispatch<F: Pod>(
    y: &[F],
    y_pred: &[f64],
    resid: Option<&[f64]>,
    t: usize,
    loss: RansacLoss,
    threshold: f64,
    mask: &mut [bool],
    sq_err: &mut [f64],
) -> usize {
    /// [`pred_block`] on the machine's REAL vector unit, the
    /// [`scan_block`] arrangement.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn go<T: HostFloat>(
        y: &[T],
        p: &[T],
        resid: Option<&[f64]>,
        t: usize,
        loss: RansacLoss,
        threshold: f64,
        mask: &mut [bool],
        sq_err: &mut [f64],
    ) -> usize {
        #[cfg(target_arch = "x86_64")]
        if avx2_available() {
            // SAFETY: guarded by the runtime detection this branch tests; the
            // body is the ordinary row loop and contains nothing unsafe.
            return unsafe { pred_block_avx2(y, p, resid, t, loss, threshold, mask, sq_err) };
        }
        pred_block(y, p, resid, t, loss, threshold, mask, sq_err)
    }
    match size_of::<F>() {
        // The predictions are narrowed to the DESIGN's width before the
        // subtraction, so the residual is the one sklearn would have formed
        // (`scan_pred`'s docs).
        4 => {
            let p: Vec<f32> = y_pred.iter().map(|&v| v as f32).collect();
            go::<f32>(
                bytemuck::cast_slice(y),
                &p,
                resid,
                t,
                loss,
                threshold,
                mask,
                sq_err,
            )
        }
        8 => go::<f64>(
            bytemuck::cast_slice(y),
            y_pred,
            resid,
            t,
            loss,
            threshold,
            mask,
            sq_err,
        ),
        other => unreachable!("ransac_host is f32/f64 only, got a {other}-byte element"),
    }
}

/// [`pred_block`] compiled for AVX2 + FMA. Widening the accumulator lanes
/// reassociates nothing, so this cannot change a result
/// ([`host_simd`](super::host_simd)).
///
/// # Safety
/// The caller must have established that the CPU supports `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn pred_block_avx2<T: HostFloat>(
    y: &[T],
    y_pred: &[T],
    resid: Option<&[f64]>,
    t: usize,
    loss: RansacLoss,
    threshold: f64,
    mask: &mut [bool],
    sq_err: &mut [f64],
) -> usize {
    pred_block(y, y_pred, resid, t, loss, threshold, mask, sq_err)
}

/// One block of rows classified against predictions the caller supplied.
///
/// `resid` short-circuits the loss: when the caller's `loss` is a CALLABLE it
/// has already produced the per-row residual, and sklearn compares exactly that
/// against the threshold. The squared errors still come from `y − y_pred`,
/// because the R² tie-break is `r2_score`'s and not the loss's.
///
/// The `t == 1` case — every 1-D `y`, i.e. essentially every RANSAC — gets its
/// own branch-free body for the same reason [`scan_block_inner`] does, and it
/// matters MORE here: this pass is the only per-trial work the arbitrary-base
/// arm does in Rust, so it is measured directly against numpy's own vectorized
/// `abs`/`<=`/`sum` on the other side of the comparison. The general arm below
/// keeps its per-target array and cannot vectorize; at `t == 1` there is nothing
/// to keep.
#[allow(clippy::too_many_arguments)]
#[inline]
fn pred_block<T: HostFloat>(
    y: &[T],
    y_pred: &[T],
    resid: Option<&[f64]>,
    t: usize,
    loss: RansacLoss,
    threshold: f64,
    mask: &mut [bool],
    sq_err: &mut [f64],
) -> usize {
    let mut n_in = 0usize;
    if t == 1 {
        let mut acc = 0.0f64;
        for (r, m) in mask.iter_mut().enumerate() {
            let diff = y[r] - y_pred[r];
            let sq = diff * diff;
            // Branch-free: the mask store and both accumulations are
            // predicated, not jumped over.
            let e = match loss {
                RansacLoss::AbsoluteError => diff.abs().to_f64(),
                RansacLoss::SquaredError => sq.to_f64(),
            };
            let inl = match resid {
                Some(rv) => rv[r] <= threshold,
                None => e <= threshold,
            };
            *m = inl;
            n_in += usize::from(inl);
            acc += if inl { sq.to_f64() } else { 0.0 };
        }
        sq_err[0] = acc;
        return n_in;
    }
    for (r, m) in mask.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        let mut sq = [0.0f64; SMALL_T];
        let mut spill: Vec<f64> = Vec::new();
        if t > SMALL_T {
            spill = vec![0.0; t];
        }
        for k in 0..t {
            let diff = y[r * t + k] - y_pred[r * t + k];
            let s = (diff * diff).to_f64();
            acc += match loss {
                RansacLoss::AbsoluteError => diff.abs().to_f64(),
                RansacLoss::SquaredError => s,
            };
            if t > SMALL_T {
                spill[k] = s;
            } else {
                sq[k] = s;
            }
        }
        let inl = match resid {
            Some(rv) => rv[r] <= threshold,
            None => acc <= threshold,
        };
        *m = inl;
        if inl {
            n_in += 1;
            for k in 0..t {
                sq_err[k] += if t > SMALL_T { spill[k] } else { sq[k] };
            }
        }
    }
    n_in
}

/// `Σ row[c]·coef[c]` split across [`DOT_LANES`] INDEPENDENT accumulators —
/// which is what lets LLVM vectorize it without any fast-math permission
/// (each lane sums its own subsequence in its own order).
///
/// Deliberately no `mul_add`: without hardware FMA `f32::mul_add` lowers to a
/// library call an order of magnitude slower than the `mul`+`add` pair
/// ([`linear_predict`](super::linear_predict) measured this).
#[inline(always)]
fn host_dot<T: HostFloat>(row: &[T], coef: &[T]) -> T {
    let mut lanes = [T::ZERO; DOT_LANES];
    let mut c = 0;
    while c + DOT_LANES <= row.len() {
        for (l, lane) in lanes.iter_mut().enumerate() {
            *lane = *lane + row[c + l] * coef[c + l];
        }
        c += DOT_LANES;
    }
    let mut tail = T::ZERO;
    while c < row.len() {
        tail = tail + row[c] * coef[c];
        c += 1;
    }
    let mut acc = T::ZERO;
    for lane in lanes {
        acc = acc + lane;
    }
    acc + tail
}

/// The host arithmetic surface both float widths provide.
trait HostFloat:
    Copy
    + Send
    + Sync
    + bytemuck::Pod
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
{
    /// The additive identity the accumulators start from.
    const ZERO: Self;
    /// `|self|`.
    fn abs(self) -> Self;
    /// Widen for the `f64` bookkeeping (thresholds, R² sums).
    fn to_f64(self) -> f64;
}

impl HostFloat for f32 {
    const ZERO: f32 = 0.0;
    #[inline(always)]
    fn abs(self) -> f32 {
        f32::abs(self)
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl HostFloat for f64 {
    const ZERO: f64 = 0.0;
    #[inline(always)]
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
}

// --------------------------------------------------------------------------- //
// Small helpers
// --------------------------------------------------------------------------- //

/// Widen an opaque `F` element through its byte view (`F`'s own ops are CubeCL
/// kernel ops, not host ones).
#[inline]
fn to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        other => unreachable!("ransac_host is f32/f64 only, got a {other}-byte element"),
    }
}

/// `numpy.median` — the mean of the two middle order statistics for an even
/// length, the middle one for an odd length. Destroys `v`'s order.
fn median_in_place(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let n = v.len();
    let mid = n / 2;
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        v[mid]
    } else {
        // The mean of the pair, written as numpy writes it, so an overflow-free
        // midpoint is not silently substituted for numpy's answer.
        (v[mid - 1] + v[mid]) / 2.0
    }
}

/// Pool width for a dispatched pass: proportional to the work THAT pass does
/// (see [`RansacHostEngine::new`]), capped by the machine and by the number of
/// blocks there are to hand out.
///
/// `MLRS_RANSAC_UNITS` overrides it for on-target A/B, read through
/// [`abflag`](crate::abflag) so a test can scope the override to its own thread
/// rather than racing `environ` ([[mlrs-abflag-test-knobs]]).
fn plan_units(work_per_pass: usize, n_blocks: usize) -> usize {
    if let Some(v) = crate::abflag::var("MLRS_RANSAC_UNITS").and_then(|v| v.parse::<usize>().ok()) {
        return v.max(1);
    }
    let by_work = (work_per_pass / MIN_WORK_PER_UNIT).max(1);
    let hw = capability::cpu_launch_units() as usize;
    by_work.min(hw).min(n_blocks.max(1)).max(1)
}

// --------------------------------------------------------------------------- //
// The sub-sample least-squares solve
// --------------------------------------------------------------------------- //

/// `|R_ii|` ratio below which [`householder_qr_lstsq`] declines the system and
/// hands it to the SVD.
///
/// It is a SPEED knob, not a correctness one, and it is deliberately far
/// stricter than the `rcond` the SVD then applies. `R`'s diagonal bounds the
/// spectrum loosely, so a ratio above `√ε` still guarantees the condition
/// number is under `~7e7` — nine orders inside the `1/ε` at which scipy's
/// `gelsd` would begin dropping directions. Any system this rejects is
/// therefore one the SVD path answers identically, just more slowly; only how
/// OFTEN that happens is at stake, and for a random `d + 1`-row draw it is
/// rare.
const QR_RANK_GUARD: f64 = 1.4901161e-8;

/// Least squares `min ‖A·xₖ − bₖ‖` by Householder QR, or `None` when the system
/// is not one QR should answer.
///
/// This is the FAST PATH of the sub-sample solve, and it exists because the
/// default `min_samples = d + 1` puts the whole cost of a trial here once `d`
/// grows. A one-sided Jacobi sweep costs `O(sweeps · d² · m)` — measured at
/// `d = 128` it was 90 % of a fit's wall clock, and mlrs LOST to sklearn 0.53×
/// on that rung — whereas a Householder QR is a single `O(m·d² − d³/3)` pass.
/// Both are backward stable and both return the exact least-squares solution
/// for a full-rank system, so the fast path is a cost change and not an answer
/// change; the `d = 128` rung went from 0.53× to a win with the coefficient
/// agreement against scipy unchanged at `~1e-13`.
///
/// Declines (returns `None`, so the caller falls back to the SVD) when:
/// - `m < d`, an under-determined system whose MINIMUM-NORM solution QR without
///   column pivoting does not produce; or
/// - the `R` diagonal is not comfortably full rank ([`QR_RANK_GUARD`]) — the
///   case where scipy's own answer is the pseudo-inverse's, not the
///   back-substituted one.
///
/// `a` (`m × d` row-major) and `b` (`m × t` row-major) are DESTROYED: the
/// reflectors are stored in `a`'s lower triangle and applied to `b` in place.
fn householder_qr_lstsq(
    a: &mut [f64],
    b: &mut [f64],
    m: usize,
    d: usize,
    t: usize,
) -> Option<Vec<f64>> {
    if d == 0 || m < d {
        return None;
    }
    // `R`'s diagonal, which the reflectors would otherwise overwrite.
    let mut diag = vec![0.0f64; d];
    for k in 0..d {
        let mut nrm = 0.0f64;
        for i in k..m {
            let v = a[k * m + i];
            nrm += v * v;
        }
        nrm = nrm.sqrt();
        if !nrm.is_finite() || nrm == 0.0 {
            return None;
        }
        // Sign chosen away from `a_kk` so the reflector vector never cancels.
        let akk = a[k * m + k];
        let alpha = if akk >= 0.0 { -nrm } else { nrm };
        diag[k] = alpha;
        a[k * m + k] = akk - alpha;
        let mut vnorm2 = 0.0f64;
        for i in k..m {
            let v = a[k * m + i];
            vnorm2 += v * v;
        }
        if vnorm2 == 0.0 {
            continue;
        }
        let scale = 2.0 / vnorm2;
        // Apply the reflector to the trailing columns of A and to every column
        // of B. Both the reflector and the column being updated are CONTIGUOUS
        // runs, which is the whole point of the column-major layout.
        for j in (k + 1)..d {
            let mut dot = 0.0f64;
            for i in k..m {
                dot += a[k * m + i] * a[j * m + i];
            }
            let f = scale * dot;
            for i in k..m {
                a[j * m + i] -= f * a[k * m + i];
            }
        }
        for j in 0..t {
            let mut dot = 0.0f64;
            for i in k..m {
                dot += a[k * m + i] * b[j * m + i];
            }
            let f = scale * dot;
            for i in k..m {
                b[j * m + i] -= f * a[k * m + i];
            }
        }
    }

    let rmax = diag.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if rmax == 0.0 || diag.iter().any(|v| v.abs() <= QR_RANK_GUARD * rmax) {
        return None;
    }

    let mut coef = vec![0.0f64; t * d];
    for j in 0..t {
        // Back-substitute into the output row. `R`'s strict upper triangle
        // lives in `a`'s columns above the diagonal: `R[i][c] == a[c*m + i]`
        // for `c > i`.
        let row = &mut coef[j * d..(j + 1) * d];
        for i in (0..d).rev() {
            let mut acc = b[j * m + i];
            for (c, &xc) in row.iter().enumerate().skip(i + 1) {
                acc -= a[c * m + i] * xc;
            }
            row[i] = acc / diag[i];
        }
    }
    Some(coef)
}

/// Minimum-norm least squares by one-sided Jacobi SVD -- the FALLBACK the QR
/// fast path declines to (`min_samples < n_features`, or a rank-deficient draw).
///
/// `a` is `d` COLUMNS of `m` (`a[c*m + r]`) and `b` is `t` columns of `m`; the
/// result is the `t x d` ROW-major coefficient block sklearn stores as `coef_`
/// (one row per target -- its `(n_targets, n_features)` layout).
///
/// One-sided Jacobi rotates A's columns until they are mutually orthogonal, so
/// `A*V = U*S` with `sigma_j = norm((A*V)_j)`. Then
/// `x = sum_j ((A*V)_j' * b / sigma_j^2) * v_j` over the directions that survive
/// the cutoff -- no explicit `U` is ever formed.
///
/// A WIDE system (`m < d`, reachable whenever a caller sets `min_samples` below
/// `n_features`) is solved through `A'` instead: one-sided Jacobi needs the tall
/// orientation to orthogonalize columns, and `A = W*S*P'` from `A'*W = P*S`
/// gives `pinv(A) = P*pinv(S)*W'`, the same minimum-norm solution scipy's
/// `gelsd` returns. That branch is the one place the column-major operand has
/// to be transposed, and it is also the rare one.
fn jacobi_lstsq(a: &[f64], b: &[f64], m: usize, d: usize, t: usize, rcond: f64) -> Vec<f64> {
    let mut coef = vec![0.0f64; t * d];
    if m == 0 || d == 0 {
        return coef;
    }
    if m >= d {
        let mut cols = a.to_vec();
        let mut v = identity(d);
        jacobi_sweep(&mut cols, &mut v, m, d);
        let sigma: Vec<f64> = (0..d).map(|j| norm(&cols[j * m..(j + 1) * m])).collect();
        let cutoff = rcond * sigma.iter().cloned().fold(0.0f64, f64::max);
        for k in 0..t {
            for j in 0..d {
                if sigma[j] <= cutoff {
                    continue;
                }
                let mut num = 0.0f64;
                for r in 0..m {
                    num += cols[j * m + r] * b[k * m + r];
                }
                let scaled = num / (sigma[j] * sigma[j]);
                for c in 0..d {
                    coef[k * d + c] += scaled * v[j * d + c];
                }
            }
        }
        return coef;
    }
    // Wide: orthogonalize the columns of A' (d x m, tall). Column `r` of A' is
    // row `r` of A, which the column-major `a` holds strided.
    let mut cols = vec![0.0f64; m * d];
    for r in 0..m {
        for c in 0..d {
            cols[r * d + c] = a[c * m + r];
        }
    }
    let mut w = identity(m);
    jacobi_sweep(&mut cols, &mut w, d, m);
    let sigma: Vec<f64> = (0..m).map(|j| norm(&cols[j * d..(j + 1) * d])).collect();
    let cutoff = rcond * sigma.iter().cloned().fold(0.0f64, f64::max);
    for k in 0..t {
        for j in 0..m {
            if sigma[j] <= cutoff {
                continue;
            }
            // w_j' * b_k, with W's columns held as `w[j*m + r]` (see `identity`).
            let mut num = 0.0f64;
            for r in 0..m {
                num += w[j * m + r] * b[k * m + r];
            }
            let scaled = num / (sigma[j] * sigma[j]);
            for c in 0..d {
                coef[k * d + c] += scaled * cols[j * d + c];
            }
        }
    }
    coef
}

/// `k × k` identity, stored as `k` consecutive length-`k` COLUMNS — the layout
/// [`jacobi_sweep`] rotates and [`jacobi_lstsq`] reads `vⱼ` out of.
fn identity(k: usize) -> Vec<f64> {
    let mut v = vec![0.0f64; k * k];
    for j in 0..k {
        v[j * k + j] = 1.0;
    }
    v
}

/// Euclidean norm of a column.
fn norm(col: &[f64]) -> f64 {
    col.iter().map(|&e| e * e).sum::<f64>().sqrt()
}

/// One-sided Jacobi: rotate pairs of columns of `cols` (`k` columns of length
/// `rows`) until they are mutually orthogonal, applying every rotation to `v`
/// (`k` columns of length `k`) as well.
fn jacobi_sweep(cols: &mut [f64], v: &mut [f64], rows: usize, k: usize) {
    for _ in 0..MAX_JACOBI_SWEEPS {
        let mut rotated = false;
        for p in 0..k.saturating_sub(1) {
            for q in (p + 1)..k {
                let (mut alpha, mut beta, mut gamma) = (0.0f64, 0.0f64, 0.0f64);
                for r in 0..rows {
                    let cp = cols[p * rows + r];
                    let cq = cols[q * rows + r];
                    alpha += cp * cp;
                    beta += cq * cq;
                    gamma += cp * cq;
                }
                if gamma == 0.0 || gamma.abs() <= JACOBI_TOL * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let tval = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + tval * tval).sqrt();
                let s = c * tval;
                for r in 0..rows {
                    let cp = cols[p * rows + r];
                    let cq = cols[q * rows + r];
                    cols[p * rows + r] = c * cp - s * cq;
                    cols[q * rows + r] = s * cp + c * cq;
                }
                for r in 0..k {
                    let vp = v[p * k + r];
                    let vq = v[q * k + r];
                    v[p * k + r] = c * vp - s * vq;
                    v[q * k + r] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }
}
