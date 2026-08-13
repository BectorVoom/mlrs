//! `RANSACRegressor` (RANSAC-01) — RANdom SAmple Consensus, the outlier-robust
//! meta-regressor.
//!
//! RANSAC does not smooth outliers down the way [`HuberRegressor`] does; it
//! *excludes* them. Each trial draws `min_samples` rows without replacement,
//! fits the base model to just those rows, measures every training row against
//! the result, and keeps whichever model won the largest consensus set. The
//! final `coef_` is a refit on that set alone.
//!
//! ```text
//! for trial in 0..max_trials:                   # max_trials SHRINKS as the
//!     idxs   = sample_without_replacement(...)  #   consensus grows
//!     model  = base.fit(X[idxs], y[idxs])       #   (stop_probability)
//!     resid  = loss(y, model.predict(X))
//!     mask   = resid <= residual_threshold
//!     if better than incumbent: keep it
//! estimator_ = base.fit(X[best_mask], y[best_mask])
//! ```
//!
//! ## Where the work is, and who does it
//! | piece | here | [`ransac_host`](mlrs_backend::prims::ransac_host) | [`ransac_device`](mlrs_backend::prims::ransac_device) |
//! |---|---|---|---|
//! | draw sequence (numpy MT19937) | ✔ | | |
//! | skip / stop / tie-break bookkeeping | ✔ | | |
//! | `min_samples` + `residual_threshold` lowering | ✔ | | |
//! | per-trial `n × d` scan, consensus R² | | ✔ | ✔ (batched) |
//! | sub-sample least squares | | ✔ | |
//!
//! The split is forced: the draw is
//! [`sample_without_replacement`](crate::model_selection::rng::sample_without_replacement),
//! which lives in this crate's `model_selection` surface, and `mlrs-backend`
//! does not (and must not) depend on `mlrs-algos`.
//!
//! ## Any base regressor, and what each one costs (RANSAC-02)
//! sklearn's `estimator` parameter takes any duck-typed regressor. Both are
//! driven from THIS loop; they differ only in who fits the sub-sample:
//!
//! | base | sub-model fit | scan | GIL crossings per trial |
//! |---|---|---|---|
//! | `LinearRegression` ([`RansacBase::Ols`]) | [`RansacHostEngine::subset_lstsq`] | host or device | **0** |
//! | anything else ([`RansacBase::Foreign`]) | the caller's, through [`RansacTrialBridge`] | host, from its predictions | **1** |
//!
//! The native arm reproduces sklearn's `_preprocess_data` → `_rescale_data` →
//! `scipy.linalg.lstsq` chain including its singular-value cutoff; the one base
//! hyperparameter that survives into the arithmetic — `fit_intercept` — is a
//! builder setter here.
//!
//! The foreign arm is what replaced the shim's second, Python-side transcription
//! of this loop. That transcription was ~10 numpy calls per trial (a fancy-index
//! copy of the sub-sample, the estimator's `fit`, a full-design `predict`, the
//! loss, the threshold, the count, a SECOND fancy-index copy for `score`, a
//! second `predict` inside it); the bridge is ONE call, which fits the caller's
//! estimator on the drawn rows and hands its predictions to a scan that runs
//! inside the same call — so nothing of size `n` is copied in either direction.
//! Everything after that — the loss, the mask, the consensus size, the R² and
//! every stop rule — happens here, over a worker pool that never wanted the GIL.
//! One crossing per trial is also the FLOOR for a foreign estimator: its `fit`
//! is Python, so the loop cannot be ahead of it, and speculating a BATCH of them
//! (which is what the native arm does for the device) would run user code for
//! trials a stop rule then discards. See [`RansacTrialBridge`].
//!
//! **What that arm costs, honestly.** Against scikit-learn's own loop on the
//! same base estimator (cpu backend, `f64`, min-of-5, each engine in its own
//! process — `scripts/bench_ransac_cpu.py --base ridge`): 0.86× at
//! `20 000 × 16`, 1.00× at `50 000 × 32`, 0.82× at `200 000 × 32`; with a
//! `DecisionTree` or `KNeighbors` base, where the estimator's own `fit`
//! dominates, 0.82–1.07× across the same ladder. So this arm is at PARITY, not
//! a win — the per-trial cost is the caller's Python `fit`/`predict` on both
//! sides, and what is left over is one extra streaming pass over the predictions
//! (Rust's, instead of numpy's fused `abs`/`<=`/`sum`) on a memory-bound box.
//! The win it does buy is structural: one implementation instead of two, every
//! RANSAC parameter available with every base, and a loop whose bookkeeping is
//! no longer interpreted. The 4–34× of [[mlrs-ransac]] belongs to the NATIVE
//! arm, which crosses nothing at all.
//!
//! `is_data_valid` / `is_model_valid` stay native callbacks
//! ([`RansacCallbacks`]) on the OLS arm and ride inside the bridge call on the
//! foreign one, so neither costs a crossing of its own.
//!
//! ## Batched trials, and why they are sound (RANSAC-02)
//! The trials inside a batch are mutually independent: a trial's scan reads the
//! design and its own candidate model, and the sequential part — incumbent
//! comparison, skip counters, dynamic `max_trials`, stop rules — consumes those
//! scans AFTERWARDS, in order. So the device arm draws, solves and scans `B`
//! trials speculatively in one launch and replays the bookkeeping over them; if
//! a stop rule fires at trial `k < B` the surplus is discarded and the MT19937
//! stream is rewound to where trial `k` left it. The fitted answer is the
//! unbatched loop's, exactly, and `ransac_test.rs::batching_does_not_change_the_fit`
//! is what holds that.
//!
//! Speculation is confined to that arm on purpose. It is pure arithmetic there;
//! with a foreign estimator or a user predicate installed it would be user code,
//! so the batch width drops to one and nothing is ever computed for a trial the
//! loop does not reach.
//!
//! ## Parity contract
//! Given the same `random_state`, the draw sequence is IDENTICAL to sklearn's,
//! index for index — that is what
//! [`NumpyRandomState`](crate::model_selection::rng::NumpyRandomState) is for.
//! The residuals themselves are computed by a different summation order than
//! numpy's BLAS `gemv` (the `linear_predict_host` caveat), so a row sitting
//! exactly on `residual_threshold` can fall on the other side of it and, in
//! principle, change which model wins. The oracle fixture is generated with a
//! clear inlier/outlier margin so that the trial trajectory — not just the
//! final coefficients — is comparable.
//!
//! Tests live in `crates/mlrs-algos/tests/ransac_test.rs` (AGENTS.md §2).

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device::Device;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::linear_predict::{
    linear_predict_host, linear_predict_multi_host, HostPrediction,
};
use mlrs_backend::prims::ransac_device::{ransac_batch_width, ransac_device_chosen, RansacDevice};
use mlrs_backend::prims::ransac_host::{dynamic_max_trials, RansacHostEngine, TrialScan};
use mlrs_backend::runtime::ActiveRuntime;

use crate::error::{AlgoError, BuildError};
use crate::model_selection::rng::{sample_without_replacement, NumpyRandomState, SampleMethod};
use crate::typestate::{Fitted, Unfit};

pub use mlrs_backend::prims::ransac_host::RansacLoss;

/// Which estimator name the typed errors carry.
const ESTIMATOR: &str = "ransac";

/// sklearn's `min_samples`, which is three different things behind one
/// argument: `None`, an absolute count, or a fraction of `n_samples`.
///
/// The lowering is data-DEPENDENT (`None` resolves to `n_features + 1`, a
/// fraction to `ceil(frac · n_samples)`), so it happens at `fit`, not at
/// `build()` — the D-08 split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MinSamples {
    /// sklearn's `None`: `n_features + 1`, the smallest sub-sample that
    /// determines a linear model with an intercept.
    Auto,
    /// sklearn's `min_samples >= 1`: an absolute row count.
    Absolute(usize),
    /// sklearn's `0 < min_samples < 1`: `ceil(min_samples · n_samples)`.
    Fraction(f64),
}

impl Default for MinSamples {
    fn default() -> Self {
        MinSamples::Auto
    }
}

/// What a caller-supplied validity predicate answers.
///
/// The third variant is what keeps the Rust core free of any foreign error
/// type. A PyO3 wrapper whose Python callable RAISED stashes the real `PyErr`,
/// answers `Abort`, and re-raises the original exception once `fit` has unwound
/// — instead of the core having to model "a Python exception" at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RansacVerdict {
    /// Keep going with this sub-sample.
    Valid,
    /// Skip this sub-sample (sklearn increments the matching `n_skips_*`).
    Invalid,
    /// Abandon the fit — [`AlgoError::CallbackAborted`].
    Abort,
}

/// The candidate model an [`is_model_valid`](RansacCallbacks::is_model_valid)
/// predicate inspects: sklearn hands it the fitted `estimator`, and these are
/// the fitted quantities a `LinearRegression` carries.
#[derive(Debug, Clone, Copy)]
pub struct RansacModel<'m> {
    /// `n_targets × n_features` row-major — sklearn's `coef_` layout.
    pub coef: &'m [f64],
    /// Length `n_targets` — sklearn's `intercept_`.
    pub intercept: &'m [f64],
    /// Columns in the design.
    pub n_features: usize,
    /// Target columns (`1` for a 1-D `y`).
    pub n_targets: usize,
}

/// sklearn's `is_data_valid` / `is_model_valid`, as borrowed closures.
///
/// Both receive the drawn sub-sample as its ROW INDICES into the design, which
/// is what sklearn's `X_subset` / `y_subset` are a fancy-indexed copy OF.
/// [`Default`] is "neither predicate", the sklearn default.
///
/// ## Why indices and not the gathered rows
/// The caller already holds the design — it is the slice it just passed to
/// `fit` — so gathering `m × d` values here only to hand them straight back is a
/// copy that helps nobody. It hurt measurably at the PyO3 boundary, where the
/// gathered block used to be widened into a Python LIST: `m · d` boxed floats
/// per trial, against one small index buffer now. The shim's bridge does
/// `X[idx]` in numpy instead, which is the same fancy index sklearn performs.
///
/// It also drops the float parameter this type used to carry, so the predicates
/// are built ONCE at the boundary rather than once per dtype arm.
pub struct RansacCallbacks<'h> {
    /// Called with the drawn row indices BEFORE the base model is fitted.
    #[allow(clippy::type_complexity)]
    pub is_data_valid: Option<&'h dyn Fn(&[i64]) -> RansacVerdict>,
    /// Called with the fitted candidate and the rows it came from.
    #[allow(clippy::type_complexity)]
    pub is_model_valid: Option<&'h dyn Fn(RansacModel<'_>, &[i64]) -> RansacVerdict>,
}

impl Default for RansacCallbacks<'_> {
    fn default() -> Self {
        Self {
            is_data_valid: None,
            is_model_valid: None,
        }
    }
}

/// What one trial of a [`RansacTrialBridge`] produced — sklearn's three-way
/// per-trial outcome, before the consensus is even looked at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialStatus {
    /// The base estimator was fitted and its predictions are in the buffer.
    Fitted,
    /// `is_data_valid` rejected the sub-sample (sklearn
    /// `n_skips_invalid_data_`).
    InvalidData,
    /// `is_model_valid` rejected the fitted candidate (sklearn
    /// `n_skips_invalid_model_`).
    InvalidModel,
}

/// The one call a trial makes into a base estimator this crate cannot host.
///
/// ## The contract
/// [`run_trial`](Self::run_trial) receives the drawn row indices and must, in
/// sklearn's order: evaluate `is_data_valid` on the sub-sample, fit the base
/// estimator to it, evaluate `is_model_valid` on the result, and hand
/// `estimator.predict(X)` for the WHOLE design (`n × t` row-major) to the
/// `scan` callback — exactly once, before returning
/// [`TrialStatus::Fitted`]. Returning [`TrialStatus::InvalidData`] /
/// [`TrialStatus::InvalidModel`] short-circuits at the corresponding step and
/// does not call `scan` at all.
///
/// `scan`'s second argument is the per-row residual, and is `Some` only when
/// [`supplies_residual`](Self::supplies_residual) is true — sklearn's CALLABLE
/// `loss`, whose answer only the caller can produce. The two string losses are
/// formed from the predictions instead, so the common case moves no extra
/// `n`-length array.
///
/// `Err(())` aborts the fit with [`AlgoError::CallbackAborted`]. The core
/// deliberately does not model a foreign error type: a PyO3 implementation
/// stashes the real `PyErr`, answers `Err`, and re-raises it once `fit` has
/// unwound — the same shape [`RansacVerdict::Abort`] uses.
///
/// ## Why a callback and not a returned buffer
/// A trial's predictions are `n · t` doubles — the single largest thing crossing
/// this boundary. Returning them means either an allocation per trial or a copy
/// into a buffer the driver owns, and the PyO3 implementation already holds them
/// in an arrow array whose lifetime ends with the call. Handing the SCAN to
/// the data instead of the data to the scan removes that copy entirely: the
/// pass reads the caller's own buffer in place, and nothing of size `n` is
/// duplicated anywhere in a trial.
pub trait RansacTrialBridge {
    /// Run one trial's sub-sample fit and full-design predict (trait docs).
    ///
    /// `rng` is the LIVE draw stream, handed over for the duration of the call.
    /// sklearn seeds the sub-estimator with the very generator the trial draws
    /// come from (`estimator.set_params(random_state=rs)`), so a randomized base
    /// consumes words BETWEEN draws; an implementation that drives such an
    /// estimator must therefore push this state into it and pull the advanced
    /// state back, or the interleave — and with it every subsequent draw —
    /// diverges from sklearn's. An implementation whose estimator draws nothing
    /// leaves it alone.
    ///
    /// This is also the second reason the foreign arm is never batched: the
    /// estimator's draw has to land between trial `k` and trial `k + 1`, which
    /// a batch of draws-then-fits could not reproduce.
    fn run_trial(
        &self,
        idxs: &[i64],
        rng: &mut NumpyRandomState,
        scan: &mut dyn FnMut(&[f64], Option<&[f64]>),
    ) -> Result<TrialStatus, ()>;

    /// Whether [`run_trial`](Self::run_trial) fills `resid` — sklearn's callable
    /// `loss`.
    fn supplies_residual(&self) -> bool {
        false
    }

    /// Refit the base estimator on the winning consensus set. Called exactly
    /// once, after the loop, and it is what leaves the caller holding a fitted
    /// `estimator_`. `rng` is handed over on the same terms as in
    /// [`run_trial`](Self::run_trial) — sklearn's final `fit` draws from the
    /// shared generator too.
    fn refit(&self, idxs: &[i64], rng: &mut NumpyRandomState) -> Result<(), ()>;
}

/// Which base estimator a fit drives.
pub enum RansacBase<'h> {
    /// sklearn's default `LinearRegression()`, fitted natively by
    /// [`RansacHostEngine::subset_lstsq`] — no foreign call anywhere in the
    /// loop, and the only arm the device scan is available to.
    Ols,
    /// Any other regressor, driven one trial per call through the bridge
    /// (module docs).
    Foreign(&'h dyn RansacTrialBridge),
}

/// Everything the trial loop needs from outside the hyperparameters: which base
/// estimator to fit, and the two optional validity predicates.
///
/// Bundled into one borrow because the two are related — the foreign arm folds
/// the predicates into its bridge call, and only the OLS arm fires them from
/// here — and because it keeps `fit_from_host_slice`'s signature from growing a
/// tenth positional argument.
pub struct RansacDriver<'h> {
    /// The base estimator arm.
    pub base: RansacBase<'h>,
    /// sklearn's `is_data_valid` / `is_model_valid`, for [`RansacBase::Ols`].
    /// Both must be `None` on the foreign arm, which evaluates them itself.
    pub callbacks: RansacCallbacks<'h>,
}

impl Default for RansacDriver<'_> {
    fn default() -> Self {
        Self {
            base: RansacBase::Ols,
            callbacks: RansacCallbacks::default(),
        }
    }
}

impl<'h> RansacDriver<'h> {
    /// The native OLS base with the two validity predicates installed.
    pub fn with_callbacks(callbacks: RansacCallbacks<'h>) -> Self {
        Self {
            base: RansacBase::Ols,
            callbacks,
        }
    }

    /// A base estimator this crate cannot host, driven through `bridge`.
    pub fn foreign(bridge: &'h dyn RansacTrialBridge) -> Self {
        Self {
            base: RansacBase::Foreign(bridge),
            callbacks: RansacCallbacks::default(),
        }
    }

    /// Whether anything in this driver can run CALLER code during a trial —
    /// which is what pins the batch width to one (module docs).
    fn runs_foreign_code(&self) -> bool {
        matches!(self.base, RansacBase::Foreign(_))
            || self.callbacks.is_data_valid.is_some()
            || self.callbacks.is_model_valid.is_some()
    }
}

/// The lowered, validated hyperparameter bundle every arm carries.
#[derive(Debug, Clone, Copy)]
struct RansacConfig {
    min_samples: MinSamples,
    residual_threshold: Option<f64>,
    max_trials: usize,
    max_skips: f64,
    stop_n_inliers: f64,
    stop_score: f64,
    stop_probability: f64,
    loss: RansacLoss,
    base_fit_intercept: bool,
    device: Device,
}

/// RANdom SAmple Consensus robust regression (RANSAC-01).
///
/// Build with [`RansacRegressor::builder`], fit with
/// [`fit_from_host_slice`](RansacRegressor::fit_from_host_slice) — which is the
/// only fit ingress, because the engine is host-resident on every backend (see
/// [`ransac_host`](mlrs_backend::prims::ransac_host) for why). The fitted
/// accessors exist ONLY on `RansacRegressor<F, Fitted>` (the compile-time
/// typestate replaces a runtime `NotFitted` guard, D-03).
///
/// `Debug` is derived so `Result<Self, _>::expect_err` is usable in the tests —
/// the fitted state is a handful of small host vectors, with no device buffer
/// to render.
#[derive(Debug)]
pub struct RansacRegressor<F, S = Unfit> {
    config: RansacConfig,
    /// Fitted `n_targets × n_features` row-major coefficients (sklearn
    /// `estimator_.coef_`). Held in `f64` on both float widths — the sub-sample
    /// solve runs in `f64` whatever the design's width, the `bayesian_ridge` /
    /// `huber` precision precedent.
    coef_: Vec<f64>,
    /// Fitted intercepts, length `n_targets` (sklearn `estimator_.intercept_`).
    intercept_: Vec<f64>,
    /// sklearn `inlier_mask_` — the consensus set of the winning model.
    inlier_mask_: Vec<bool>,
    /// sklearn `n_trials_`.
    n_trials_: usize,
    /// sklearn `n_skips_no_inliers_`.
    n_skips_no_inliers_: usize,
    /// sklearn `n_skips_invalid_data_`.
    n_skips_invalid_data_: usize,
    /// sklearn `n_skips_invalid_model_`.
    n_skips_invalid_model_: usize,
    /// Whether the skip budget was blown even though a consensus set WAS found
    /// — sklearn's `ConvergenceWarning` case, which is a warning and not an
    /// error, so it is surfaced rather than raised (the `huber` precedent).
    exceeded_max_skips_: bool,
    /// The RESOLVED sub-sample size, for diagnosis (`None`/fraction lowered).
    min_samples_: usize,
    /// The RESOLVED inlier cut-off (the target MAD when the caller passed
    /// `None`).
    residual_threshold_: f64,
    n_features_: usize,
    n_targets_: usize,
    /// The scan arm that ACTUALLY ran, `"cpu"` or `"gpu"` (DEVICE-PARAM-01).
    device_: &'static str,
    /// Trials scanned per launch — one on every host fit, the batch width on a
    /// device one. Reported so a perf probe can tell an unhonoured `device`
    /// preference from an honoured one that simply had nothing to batch.
    batch_width_: usize,
    /// Whether the fitted state carries a LINEAR model. False on the foreign
    /// arm, where `estimator_` is the caller's own fitted object and there are
    /// no coefficients here to hand back.
    has_model_: bool,
    /// Compile-time lifecycle marker plus the float width the fitted state was
    /// produced at. `F` is not stored anywhere else — `coef_`/`intercept_` are
    /// `f64` on both arms (see their docs) — but it still parameterizes the
    /// `predict` ingress and the sub-sample solve's `rcond`, so it stays on the
    /// type rather than becoming a runtime field.
    _state: PhantomData<(F, S)>,
}

impl<F> RansacRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Start building with sklearn's defaults (D-01/D-03).
    pub fn builder() -> RansacRegressorBuilder {
        RansacRegressorBuilder::default()
    }

    /// sklearn `min_samples`, unlowered.
    pub fn min_samples(&self) -> MinSamples {
        self.config.min_samples
    }
    /// sklearn `residual_threshold` (`None` = the target MAD, resolved at fit).
    pub fn residual_threshold(&self) -> Option<f64> {
        self.config.residual_threshold
    }
    /// sklearn `max_trials`.
    pub fn max_trials(&self) -> usize {
        self.config.max_trials
    }
    /// sklearn `max_skips` (`f64::INFINITY` by default).
    pub fn max_skips(&self) -> f64 {
        self.config.max_skips
    }
    /// sklearn `stop_n_inliers` (`f64::INFINITY` by default).
    pub fn stop_n_inliers(&self) -> f64 {
        self.config.stop_n_inliers
    }
    /// sklearn `stop_score` (`f64::INFINITY` by default).
    pub fn stop_score(&self) -> f64 {
        self.config.stop_score
    }
    /// sklearn `stop_probability`.
    pub fn stop_probability(&self) -> f64 {
        self.config.stop_probability
    }
    /// sklearn `loss` — the estimator's ONE string-valued parameter.
    pub fn loss(&self) -> RansacLoss {
        self.config.loss
    }
    /// The base `LinearRegression`'s `fit_intercept`.
    pub fn base_fit_intercept(&self) -> bool {
        self.config.base_fit_intercept
    }

    /// Run the RANSAC loop over a design still in the CALLER'S memory.
    ///
    /// `x` is `n × d` row-major, `y` is `n × n_targets` row-major (pass
    /// `n_targets = 1` for a 1-D target), `sample_weight` is either `None` or
    /// length `n`. `rng` is the caller's numpy `RandomState`, advanced in place
    /// exactly as sklearn advances it — the shim writes the advanced words back
    /// into the Python object. `driver` says which base estimator to fit and
    /// carries the two validity predicates ([`RansacDriver`]).
    ///
    /// This is the only fit INGRESS on every arm: the host engine reads the
    /// caller's memory directly, and the device arm uploads from it once (never
    /// per trial), so there is no device-resident entry point to add. `pool` is
    /// untouched unless the device scan is selected.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_from_host_slice(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        y: &[F],
        shape: (usize, usize),
        n_targets: usize,
        sample_weight: Option<&[f64]>,
        rng: &mut NumpyRandomState,
        driver: &RansacDriver<'_>,
    ) -> Result<RansacRegressor<F, Fitted>, AlgoError> {
        let (n, d) = shape;
        let t = n_targets.max(1);
        let cfg = self.config;

        // --- 1. Lower `min_samples` (sklearn's three-way branch). -----------
        let min_samples = match cfg.min_samples {
            MinSamples::Auto => d + 1,
            MinSamples::Absolute(k) => k,
            MinSamples::Fraction(f) => (f * n as f64).ceil() as usize,
        };
        if min_samples > n {
            return Err(AlgoError::MinSamplesExceedsNSamples {
                estimator: ESTIMATOR,
                min_samples,
                n_samples: n,
            });
        }
        if let Some(sw) = sample_weight {
            if sw.len() != n {
                return Err(AlgoError::Prim(mlrs_core::PrimError::ShapeMismatch {
                    operand: "ransac_sample_weight",
                    rows: n,
                    cols: 1,
                    len: sw.len(),
                }));
            }
            // The repo-wide weight contract: finite and non-negative. sklearn's
            // `_check_sample_weight` lets a negative through and produces a NaN
            // fit; mlrs names the offending index instead
            // ([[mlrs-nb-sample-weight]]).
            if let Some((index, &value)) = sw
                .iter()
                .enumerate()
                .find(|(_, &w)| !w.is_finite() || w < 0.0)
            {
                return Err(AlgoError::InvalidSampleWeight {
                    estimator: ESTIMATOR,
                    index,
                    value,
                });
            }
            // All-zero weights leave nothing to fit, and the centered
            // sub-sample solve would return the all-zero coefficient vector as
            // if it were an answer. sklearn's
            // `check_all_zero_sample_weights_error` requires a `ValueError`
            // here, and its inner `LinearRegression` is what raises one.
            if sw.iter().sum::<f64>() == 0.0 {
                return Err(AlgoError::ZeroSampleWeightSum {
                    estimator: ESTIMATOR,
                });
            }
        }

        // The pool is sized for the pass this fit will ACTUALLY dispatch, which
        // differs per arm — see `RansacHostEngine::new`. The arm is not yet
        // chosen at this point (step 3 needs `n`/`d`, which is why it is cheap
        // to decide it here first).
        let foreign = match driver.base {
            RansacBase::Ols => None,
            RansacBase::Foreign(b) => Some(b),
        };
        let on_device = foreign.is_none() && ransac_device_chosen::<F>(n, d, cfg.device);
        let width = if on_device && !driver.runs_foreign_code() {
            ransac_batch_width(cfg.max_trials)
        } else {
            1
        };
        let work_per_pass = match (foreign.is_some(), on_device) {
            // The bridge's pass reads `y` and the caller's predictions — no `d`.
            (true, _) => n.saturating_mul(t),
            // The device scans; the only pass this pool dispatches is the batch
            // of sub-sample solves, whose cost is `O(m·d²)` per trial — the one
            // arm where the pass is compute-bound rather than a streaming read,
            // so the flop count is what sizes it.
            (false, true) => width
                .saturating_mul(min_samples)
                .saturating_mul(d)
                .saturating_mul(d),
            // The host scan reads the whole design, once per trial.
            (false, false) => n.saturating_mul(d.max(1)).saturating_mul(t),
        };
        let engine = RansacHostEngine::new(x, y, n, d, t, work_per_pass)?;

        // --- 2. Lower `residual_threshold` (`None` → the target MAD). -------
        let threshold = match cfg.residual_threshold {
            Some(v) => v,
            None => engine.target_mad(),
        };

        // --- 3. Build the device engine if that arm was chosen above. -------
        // Speculation is only ever over ARITHMETIC: a foreign base or an
        // installed predicate pins the batch width to one, so no caller code
        // runs for a trial the loop does not reach (module docs).
        let mut device = match on_device {
            true => Some(RansacDevice::<F>::new(pool, x, y, n, d, t, width)?),
            false => None,
        };

        // --- 4. The trial loop, statement for statement with sklearn's. -----
        // `n_inliers_best` starts at ONE, not zero: sklearn's incumbent is a
        // sentinel, so a trial that finds a single inlier does not qualify and
        // is counted as a `no_inliers` skip.
        let mut n_inliers_best = 1usize;
        let mut score_best = f64::NEG_INFINITY;
        let mut have_best = false;
        // The host arm writes `width` masks per batch; the device arm keeps its
        // masks resident and reads back only the incumbent's, so it wants none
        // of this (which at `n = 10⁶` is megabytes it would never touch).
        let mut masks = vec![false; if on_device { 0 } else { width * n }];
        let mut best_mask = vec![false; n];
        let mut n_trials = 0usize;
        let mut n_skips_no_inliers = 0usize;
        let mut n_skips_invalid_data = 0usize;
        let mut n_skips_invalid_model = 0usize;
        // sklearn's `max_trials` is shadowed by a FLOAT inside the loop, because
        // `_dynamic_max_trials` can return `inf`.
        let mut max_trials = cfg.max_trials as f64;

        let fit_result = (|| -> Result<(), AlgoError> {
            'batches: loop {
                // sklearn's per-trial preamble, for the FIRST trial of this
                // batch. It runs BEFORE the draw and before anything is
                // computed, which is what makes trial 0 non-speculative — the
                // property the width-one arms rely on.
                if (n_trials as f64) >= max_trials {
                    break 'batches;
                }
                n_trials += 1;
                let skips = n_skips_no_inliers + n_skips_invalid_data + n_skips_invalid_model;
                if (skips as f64) > cfg.max_skips {
                    break 'batches;
                }

                let remaining = (max_trials - (n_trials - 1) as f64).ceil().max(1.0);
                let bw = width.min(remaining as usize).max(1);

                // --- draw, snapshotting the stream after each trial so a stop
                //     rule firing mid-batch can rewind to exactly where the last
                //     CONSUMED trial left it.
                let start_state = rng.clone();
                let mut snaps: Vec<NumpyRandomState> = Vec::with_capacity(bw);
                let mut idxs = Vec::with_capacity(bw * min_samples);
                for _ in 0..bw {
                    idxs.extend_from_slice(
                        &sample_without_replacement(n, min_samples, SampleMethod::Auto, rng)
                            .expect("min_samples <= n_samples was checked above"),
                    );
                    snaps.push(rng.clone());
                }

                // --- run the trials: validity, sub-model, scan.
                let mut status = vec![TrialStatus::Fitted; bw];
                let mut scans: Vec<Option<TrialScan>> = vec![None; bw];
                let mut y_sums: Vec<Vec<f64>> = Vec::new();

                match foreign {
                    // The arbitrary-base arm: ONE call per trial, which fits the
                    // caller's estimator and hands back its full-design
                    // predictions (module docs). `bw == 1` here.
                    Some(bridge) => {
                        for b in 0..bw {
                            let sub = &idxs[b * min_samples..(b + 1) * min_samples];
                            // The scan runs INSIDE the bridge call, over the
                            // caller's own prediction buffer (trait docs).
                            let (engine_ref, mask) =
                                (&engine, &mut masks[b * n..(b + 1) * n]);
                            let mut scanned = None;
                            let status_b = bridge.run_trial(
                                sub,
                                rng,
                                &mut |y_pred: &[f64], resid: Option<&[f64]>| {
                                    scanned = Some(engine_ref.scan_pred(
                                        y_pred, resid, cfg.loss, threshold, mask,
                                    ));
                                },
                            );
                            status[b] = status_b.map_err(|()| AlgoError::CallbackAborted {
                                estimator: ESTIMATOR,
                                callback: "estimator.fit",
                            })?;
                            if status[b] == TrialStatus::Fitted {
                                scans[b] = Some(scanned.ok_or(AlgoError::CallbackAborted {
                                    estimator: ESTIMATOR,
                                    callback: "estimator.predict",
                                })?);
                            }
                        }
                    }
                    // The native OLS arm.
                    None => {
                        for b in 0..bw {
                            if let Some(f) = driver.callbacks.is_data_valid {
                                match f(&idxs[b * min_samples..(b + 1) * min_samples]) {
                                    RansacVerdict::Valid => {}
                                    RansacVerdict::Invalid => status[b] = TrialStatus::InvalidData,
                                    RansacVerdict::Abort => {
                                        return Err(AlgoError::CallbackAborted {
                                            estimator: ESTIMATOR,
                                            callback: "is_data_valid",
                                        })
                                    }
                                }
                            }
                        }
                        // A rejected sub-sample is only reachable at `bw == 1`
                        // (a predicate pins the width), so "solve the batch or
                        // none of it" loses nothing and keeps one solve call.
                        if status.iter().all(|s| *s == TrialStatus::Fitted) {
                            let (coef, icept) = engine.subset_lstsq_batch(
                                &idxs,
                                min_samples,
                                bw,
                                cfg.base_fit_intercept,
                                sample_weight,
                            );
                            for b in 0..bw {
                                if let Some(f) = driver.callbacks.is_model_valid {
                                    let model = RansacModel {
                                        coef: &coef[b * t * d..(b + 1) * t * d],
                                        intercept: &icept[b * t..(b + 1) * t],
                                        n_features: d,
                                        n_targets: t,
                                    };
                                    match f(model, &idxs[b * min_samples..(b + 1) * min_samples]) {
                                        RansacVerdict::Valid => {}
                                        RansacVerdict::Invalid => {
                                            status[b] = TrialStatus::InvalidModel
                                        }
                                        RansacVerdict::Abort => {
                                            return Err(AlgoError::CallbackAborted {
                                                estimator: ESTIMATOR,
                                                callback: "is_model_valid",
                                            })
                                        }
                                    }
                                }
                            }
                            let live = status.iter().all(|s| *s == TrialStatus::Fitted);
                            if live {
                                match device.as_mut() {
                                    Some(dev) => {
                                        let out = dev.scan_batch(
                                            pool, &coef, &icept, bw, cfg.loss, threshold,
                                        )?;
                                        scans = out.iter().map(|s| Some(s.as_trial_scan())).collect();
                                        y_sums = out.into_iter().map(|s| s.y_sum).collect();
                                    }
                                    None => {
                                        scans = engine
                                            .scan_batch(
                                                &coef,
                                                &icept,
                                                bw,
                                                cfg.loss,
                                                threshold,
                                                &mut masks[..bw * n],
                                            )
                                            .into_iter()
                                            .map(Some)
                                            .collect();
                                    }
                                }
                            }
                        }
                    }
                }

                // --- replay the sequential bookkeeping over the batch.
                let mut consumed = 0usize;
                let mut stop = false;
                for b in 0..bw {
                    if b > 0 {
                        // Trial 0's preamble ran before the draw; every later
                        // trial in the batch gets it here, and a break leaves
                        // its draw UNCONSUMED (sklearn draws after this check).
                        if (n_trials as f64) >= max_trials {
                            stop = true;
                            break;
                        }
                        n_trials += 1;
                        let skips =
                            n_skips_no_inliers + n_skips_invalid_data + n_skips_invalid_model;
                        if (skips as f64) > cfg.max_skips {
                            stop = true;
                            break;
                        }
                    }
                    consumed = b + 1;

                    match status[b] {
                        TrialStatus::InvalidData => {
                            n_skips_invalid_data += 1;
                            continue;
                        }
                        TrialStatus::InvalidModel => {
                            n_skips_invalid_model += 1;
                            continue;
                        }
                        TrialStatus::Fitted => {}
                    }
                    let scan = scans[b]
                        .as_ref()
                        .expect("a Fitted trial always produced a scan");

                    // Fewer inliers than the incumbent — sklearn books this under
                    // `n_skips_no_inliers_` (the name is historical; it covers
                    // "not enough", not only "none").
                    if scan.n_inliers < n_inliers_best {
                        n_skips_no_inliers += 1;
                        continue;
                    }

                    // The score is formed only HERE, for a trial that already
                    // matched the incumbent — which is why the device arm's
                    // denominator is a per-trial launch and not a per-batch one.
                    let score = match device.as_mut() {
                        Some(dev) => {
                            let mean: Vec<f64> = y_sums[b]
                                .iter()
                                .map(|s| s / scan.n_inliers as f64)
                                .collect();
                            let den = dev.r2_den(pool, b, &mean)?;
                            engine.r2_from_sums(scan.n_inliers, &scan.sq_err, &den)
                        }
                        None => engine.r2_on_mask(
                            &masks[b * n..(b + 1) * n],
                            scan.n_inliers,
                            &scan.sq_err,
                        ),
                    };

                    // Equal consensus, worse score — keep the incumbent. NaN
                    // (fewer than two inliers) compares false here, exactly as it
                    // does in Python, so it does NOT skip.
                    if scan.n_inliers == n_inliers_best && score < score_best {
                        continue;
                    }

                    n_inliers_best = scan.n_inliers;
                    score_best = score;
                    match device.as_mut() {
                        Some(dev) => dev.mask_of(pool, b, &mut best_mask),
                        None => best_mask.copy_from_slice(&masks[b * n..(b + 1) * n]),
                    }
                    have_best = true;

                    max_trials = max_trials.min(dynamic_max_trials(
                        n_inliers_best,
                        n,
                        min_samples,
                        cfg.stop_probability,
                    ));

                    if n_inliers_best as f64 >= cfg.stop_n_inliers || score_best >= cfg.stop_score {
                        stop = true;
                        break;
                    }
                }

                // Rewind the draw stream over the trials the bookkeeping never
                // reached, so the caller's `RandomState` ends exactly where an
                // unbatched loop would have left it.
                if consumed < bw {
                    *rng = match consumed {
                        0 => start_state,
                        k => snaps[k - 1].clone(),
                    };
                }
                if stop {
                    break 'batches;
                }
            }
            Ok(())
        })();
        // The design and the mask go back to the pool whichever way the loop
        // left, including an aborted bridge call.
        if let Some(dev) = device.take() {
            dev.release(pool);
        }
        fit_result?;

        let skips = n_skips_no_inliers + n_skips_invalid_data + n_skips_invalid_model;
        let exceeded = (skips as f64) > cfg.max_skips;
        if !have_best {
            return Err(AlgoError::NoValidConsensusSet {
                estimator: ESTIMATOR,
                n_trials,
                n_skips: skips,
                skipped_out: exceeded,
            });
        }

        // --- 5. Final model: the base estimator refitted on the consensus. --
        let inlier_idxs: Vec<i64> = best_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| m.then_some(i as i64))
            .collect();
        let (coef_, intercept_) = match foreign {
            Some(bridge) => {
                bridge
                    .refit(&inlier_idxs, rng)
                    .map_err(|()| AlgoError::CallbackAborted {
                        estimator: ESTIMATOR,
                        callback: "estimator.fit",
                    })?;
                (Vec::new(), Vec::new())
            }
            None => engine.subset_lstsq(&inlier_idxs, cfg.base_fit_intercept, sample_weight),
        };

        Ok(RansacRegressor {
            config: cfg,
            coef_,
            intercept_,
            inlier_mask_: best_mask,
            n_trials_: n_trials,
            n_skips_no_inliers_: n_skips_no_inliers,
            n_skips_invalid_data_: n_skips_invalid_data,
            n_skips_invalid_model_: n_skips_invalid_model,
            exceeded_max_skips_: exceeded,
            min_samples_: min_samples,
            residual_threshold_: threshold,
            n_features_: d,
            n_targets_: t,
            device_: if on_device { "gpu" } else { "cpu" },
            batch_width_: width,
            has_model_: foreign.is_none(),
            _state: PhantomData,
        })
    }
}

// LINEAR-PERSIST: the safetensors container. RANSAC is the only fully
// HOST-resident member of the family — `coef_`/`intercept_` are plain `f64`
// vectors, never device arrays — so its core is written at `f64` regardless of
// the estimator's `F`, which is a `PhantomData` marker here.
use crate::linear::linear_persist::{
    as_bools, pack_bools, read_linear_core, shape_1d, AlignedBytes, LinearCoreRef, LinearFile,
    LinearWriter, LoadModel, PersistError, SaveModel, TensorRef,
};
/// The `estimator` discriminator written into every saved file and required by
/// [`LoadModel::load`].
const PERSIST_TAG: &str = "ransac";

/// `__metadata__` key for the [`MinSamples`] enum. Its `Absolute`/`Fraction`
/// arms carry a payload, encoded into the tag itself (`"absolute:5"`) rather
/// than a companion tensor — unlike `RidgeClassifier`'s `class_weight`, the
/// payload here is a SINGLE number, so a tensor would cost ~60 bytes of header
/// for 8 bytes of data.
const KEY_MIN_SAMPLES: &str = "param:min_samples";

impl<F> SaveModel for RansacRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted model to `path`.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `coef_` | `F64` | `[n_targets_, n_features_]` |
    /// | `intercept_` | `F64` | `[n_targets_]` |
    /// | `inlier_mask_` | `BOOL` | `[n_samples]` |
    /// | the ten `param:*` and the eight fitted counters | `__metadata__` | — |
    ///
    /// `coef_` is `F64` and not the model's own width because RANSAC holds it
    /// as `Vec<f64>` on the host — the `F` parameter is a
    /// [`PhantomData`] marker for API uniformity, not a storage type, since the
    /// consensus engine is host-resident on every backend.
    ///
    /// The base estimator is NOT stored, and does not need to be: `RansacConfig`
    /// retains only `base_fit_intercept`, because a fitted RANSAC keeps the
    /// winning model's COEFFICIENTS rather than the estimator that produced
    /// them. The `Foreign` base — a borrowed `&dyn RansacTrialBridge`, typically
    /// a Python callback — exists only for the duration of `fit` and is
    /// unreachable from the fitted value, so there is nothing here that could
    /// fail to serialize.
    fn save(&self, _pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        if !self.has_model_ {
            return Err(PersistError::MissingState {
                estimator: PERSIST_TAG,
                field: "coef_ (no consensus model was found)",
            });
        }
        // `bool` is not `Pod`; bound here so the writer can borrow it.
        let mask = pack_bools(&self.inlier_mask_);
        let cfg = &self.config;

        let mut w = LinearWriter::new(PERSIST_TAG);
        match cfg.min_samples {
            MinSamples::Auto => w.scalar_str(KEY_MIN_SAMPLES, "auto"),
            MinSamples::Absolute(n) => w.scalar_str(KEY_MIN_SAMPLES, &format!("absolute:{n}")),
            MinSamples::Fraction(f) => w.scalar_str(KEY_MIN_SAMPLES, &format!("fraction:{f:?}")),
        }
        w.scalar_opt_f64("param:residual_threshold", cfg.residual_threshold);
        w.scalar_usize("param:max_trials", cfg.max_trials);
        // These three default to `f64::INFINITY`. Rust's float formatter emits
        // `inf` and its parser accepts it, so the sentinel survives the
        // round-trip without a special case.
        w.scalar_f64("param:max_skips", cfg.max_skips);
        w.scalar_f64("param:stop_n_inliers", cfg.stop_n_inliers);
        w.scalar_f64("param:stop_score", cfg.stop_score);
        w.scalar_f64("param:stop_probability", cfg.stop_probability);
        w.scalar_str("param:loss", cfg.loss.as_str());
        w.scalar_bool("param:base_fit_intercept", cfg.base_fit_intercept);
        w.scalar_str("param:device", cfg.device.name());

        w.scalar_usize("n_trials_", self.n_trials_);
        w.scalar_usize("n_skips_no_inliers_", self.n_skips_no_inliers_);
        w.scalar_usize("n_skips_invalid_data_", self.n_skips_invalid_data_);
        w.scalar_usize("n_skips_invalid_model_", self.n_skips_invalid_model_);
        w.scalar_bool("exceeded_max_skips_", self.exceeded_max_skips_);
        w.scalar_usize("min_samples_", self.min_samples_);
        w.scalar_f64("residual_threshold_", self.residual_threshold_);
        w.scalar_usize("batch_width_", self.batch_width_);
        w.scalar_str("device_", self.device_);

        w.tensor("inlier_mask_", TensorRef::bools(&mask, vec![mask.len()])?);
        LinearCoreRef::<f64> {
            coef: &self.coef_,
            intercept: &self.intercept_,
            n_targets: self.n_targets_,
            n_features: self.n_features_,
            fit_intercept: cfg.base_fit_intercept,
        }
        .write_into(&mut w)?;
        w.write(path)
    }
}

impl<F> LoadModel for RansacRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read a model back from `path`. Nothing is uploaded — the fitted state is
    /// host-resident, so `pool` is unused, exactly as it is on the save side.
    ///
    /// The core is read back at `f64` (see [`SaveModel::save`]) rather than at
    /// `F`, so the file this estimator writes is width-independent: a
    /// `RansacRegressor<f32>` and a `RansacRegressor<f64>` produce and consume
    /// byte-identical files.
    fn load(
        _pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<RansacRegressor<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = LinearFile::parse(&raw, PERSIST_TAG)?;
        let core = read_linear_core::<f64>(&file)?;

        let mask_v = file.tensor("inlier_mask_")?;
        shape_1d(&mask_v, "inlier_mask_")?;
        let inlier_mask_ = as_bools(&mask_v, "inlier_mask_")?;

        let tag = file.scalar_str(KEY_MIN_SAMPLES)?;
        let min_samples = match tag {
            "auto" => MinSamples::Auto,
            other => {
                let absolute = other
                    .strip_prefix("absolute:")
                    .and_then(|n| n.parse().ok())
                    .map(MinSamples::Absolute);
                let fraction = other
                    .strip_prefix("fraction:")
                    .and_then(|f| f.parse().ok())
                    .map(MinSamples::Fraction);
                absolute.or(fraction).ok_or(PersistError::BadMetadata {
                    key: KEY_MIN_SAMPLES,
                })?
            }
        };
        let loss = match file.scalar_str("param:loss")? {
            "absolute_error" => RansacLoss::AbsoluteError,
            "squared_error" => RansacLoss::SquaredError,
            _ => return Err(PersistError::BadMetadata { key: "param:loss" }),
        };
        let device = Device::from_name(file.scalar_str("param:device")?).ok_or(
            PersistError::BadMetadata {
                key: "param:device",
            },
        )?;
        let device_ = Device::from_name(file.scalar_str("device_")?)
            .ok_or(PersistError::BadMetadata { key: "device_" })?
            .name();

        Ok(RansacRegressor {
            config: RansacConfig {
                min_samples,
                residual_threshold: file.scalar_opt_f64("param:residual_threshold")?,
                max_trials: file.scalar_usize("param:max_trials")?,
                max_skips: file.scalar_f64("param:max_skips")?,
                stop_n_inliers: file.scalar_f64("param:stop_n_inliers")?,
                stop_score: file.scalar_f64("param:stop_score")?,
                stop_probability: file.scalar_f64("param:stop_probability")?,
                loss,
                base_fit_intercept: core.fit_intercept,
                device,
            },
            coef_: core.coef.into_owned(),
            intercept_: core.intercept.into_owned(),
            inlier_mask_,
            n_trials_: file.scalar_usize("n_trials_")?,
            n_skips_no_inliers_: file.scalar_usize("n_skips_no_inliers_")?,
            n_skips_invalid_data_: file.scalar_usize("n_skips_invalid_data_")?,
            n_skips_invalid_model_: file.scalar_usize("n_skips_invalid_model_")?,
            exceeded_max_skips_: file.scalar_bool("exceeded_max_skips_")?,
            min_samples_: file.scalar_usize("min_samples_")?,
            residual_threshold_: file.scalar_f64("residual_threshold_")?,
            n_features_: core.n_features,
            n_targets_: core.n_targets,
            device_,
            batch_width_: file.scalar_usize("batch_width_")?,
            // A file is only ever written from a fitted model that HAS one
            // (`save` refuses otherwise), so this is true by construction.
            has_model_: true,
            _state: PhantomData,
        })
    }
}

impl<F> RansacRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// sklearn `estimator_.coef_`, `n_targets × n_features` row-major.
    pub fn coef(&self) -> &[f64] {
        &self.coef_
    }
    /// sklearn `estimator_.intercept_`, length `n_targets`.
    pub fn intercept(&self) -> &[f64] {
        &self.intercept_
    }
    /// sklearn `inlier_mask_`.
    pub fn inlier_mask(&self) -> &[bool] {
        &self.inlier_mask_
    }
    /// sklearn `n_trials_`.
    pub fn n_trials(&self) -> usize {
        self.n_trials_
    }
    /// sklearn `n_skips_no_inliers_`.
    pub fn n_skips_no_inliers(&self) -> usize {
        self.n_skips_no_inliers_
    }
    /// sklearn `n_skips_invalid_data_`.
    pub fn n_skips_invalid_data(&self) -> usize {
        self.n_skips_invalid_data_
    }
    /// sklearn `n_skips_invalid_model_`.
    pub fn n_skips_invalid_model(&self) -> usize {
        self.n_skips_invalid_model_
    }
    /// Whether the skips exceeded `max_skips` even though a consensus set was
    /// found — sklearn raises a `ConvergenceWarning` in exactly this case, and
    /// the shim turns this flag into it.
    pub fn exceeded_max_skips(&self) -> bool {
        self.exceeded_max_skips_
    }
    /// The RESOLVED sub-sample size (`n_features + 1` for the `None` default).
    pub fn min_samples_used(&self) -> usize {
        self.min_samples_
    }
    /// The RESOLVED inlier cut-off (the target MAD when the caller passed
    /// `None`).
    pub fn residual_threshold_used(&self) -> f64 {
        self.residual_threshold_
    }
    /// Columns the fit saw.
    pub fn n_features(&self) -> usize {
        self.n_features_
    }
    /// Target columns the fit saw.
    pub fn n_targets(&self) -> usize {
        self.n_targets_
    }
    /// sklearn `loss`, round-tripped.
    pub fn loss(&self) -> RansacLoss {
        self.config.loss
    }

    /// The scan arm that ACTUALLY ran, `"cpu"` or `"gpu"` (DEVICE-PARAM-01).
    ///
    /// `device='gpu'` overrides the SIZE half of
    /// [`ransac_device_applicable`](mlrs_backend::prims::ransac_device::ransac_device_applicable),
    /// not its capability half, and it cannot apply at all to a base estimator
    /// this crate does not host (the foreign arm's predictions come from the
    /// caller, on the host). Both cases report `"cpu"` here rather than faking
    /// the preference.
    pub fn device_arm(&self) -> &'static str {
        self.device_
    }

    /// Trials scanned per launch — `1` on every host fit.
    ///
    /// Reported so a perf probe can distinguish "the device arm ran" from "the
    /// device arm ran with nothing to batch", which are different measurements.
    pub fn batch_width(&self) -> usize {
        self.batch_width_
    }

    /// Whether this fit carries a LINEAR model — false on the foreign base arm,
    /// where the fitted `estimator_` is the caller's own object and
    /// [`coef`](Self::coef) / [`intercept`](Self::intercept) are empty.
    pub fn has_linear_model(&self) -> bool {
        self.has_model_
    }

    /// `predict` for a test matrix still in the CALLER'S memory —
    /// `X·coef_ᵀ + intercept_`, i.e. sklearn's delegation to `estimator_`.
    ///
    /// Runs the HOST matvec on every backend. That is not a cpu-arm special
    /// case: a dense linear `predict` is one pass over the operand, and the
    /// device arm has to upload that operand first, which measured 10-26x
    /// SLOWER than the host path even on a real T4
    /// ([[mlrs-ridge-predict-cuda-vs-cpu]]).
    ///
    /// Returns the [`HostPrediction`] rather than the bare values, because that
    /// pass ALREADY reads every element of `x` and so already knows whether the
    /// operand was finite. The caller relays that verdict as `check_array`'s own
    /// `ValueError`, which is what lets the shim skip a second single-threaded
    /// scan of the whole matrix (`linear_predict` module docs) WITHOUT dropping
    /// the validation.
    pub fn predict_from_host(
        &self,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<HostPrediction<F>, AlgoError> {
        if !self.has_model_ {
            // The foreign arm's `estimator_` is the caller's fitted object and
            // its `predict` is the caller's too; there is no coefficient vector
            // here to run a matvec against, and inventing one would be worse
            // than saying so.
            return Err(AlgoError::Unsupported {
                estimator: ESTIMATOR,
                operation: "predict on a non-LinearRegression base estimator",
            });
        }
        let (m, d) = shape;
        if d != self.n_features_ {
            return Err(AlgoError::Prim(mlrs_core::PrimError::DimMismatch {
                dim: "n_features",
                lhs: self.n_features_,
                rhs: d,
            }));
        }
        if self.n_targets_ == 1 {
            let coef: Vec<F> = self.coef_.iter().map(|&v| f64_to_f::<F>(v)).collect();
            let bias = f64_to_f::<F>(self.intercept_[0]);
            return Ok(linear_predict_host::<F>(x, &coef, bias, (m, d))?);
        }
        // `linear_predict_multi_host` wants `coef` FEATURE-major (`d × t`);
        // sklearn's `coef_` is target-major (`t × d`).
        let t = self.n_targets_;
        let mut coef = vec![f64_to_f::<F>(0.0); d * t];
        for k in 0..t {
            for c in 0..d {
                coef[c * t + k] = f64_to_f::<F>(self.coef_[k * d + c]);
            }
        }
        let bias: Vec<F> = self.intercept_.iter().map(|&v| f64_to_f::<F>(v)).collect();
        Ok(linear_predict_multi_host::<F>(x, &coef, &bias, (m, d), t)?)
    }
}

/// Narrow an `f64` hyperparameter/coefficient to the design's width through its
/// byte view (`F`'s own ops are CubeCL *kernel* ops, not host ones).
fn f64_to_f<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        other => unreachable!("ransac is f32/f64 only, got a {other}-byte element"),
    }
}

/// Builder for [`RansacRegressor`] (D-01). Default field initializers encode
/// sklearn's `RANSACRegressor` defaults: `min_samples=None`,
/// `residual_threshold=None`, `max_trials=100`, `max_skips=inf`,
/// `stop_n_inliers=inf`, `stop_score=inf`, `stop_probability=0.99`,
/// `loss="absolute_error"`, plus the base `LinearRegression()`'s
/// `fit_intercept=True`.
#[derive(Debug, Clone, Copy)]
pub struct RansacRegressorBuilder {
    min_samples: MinSamples,
    residual_threshold: Option<f64>,
    max_trials: usize,
    max_skips: f64,
    stop_n_inliers: f64,
    stop_score: f64,
    stop_probability: f64,
    loss: RansacLoss,
    base_fit_intercept: bool,
    device: Device,
}

impl Default for RansacRegressorBuilder {
    fn default() -> Self {
        Self {
            min_samples: MinSamples::Auto,
            residual_threshold: None,
            max_trials: 100,
            max_skips: f64::INFINITY,
            stop_n_inliers: f64::INFINITY,
            stop_score: f64::INFINITY,
            stop_probability: 0.99,
            loss: RansacLoss::AbsoluteError,
            base_fit_intercept: true,
            device: Device::Auto,
        }
    }
}

impl RansacRegressorBuilder {
    /// sklearn `min_samples`.
    pub fn min_samples(mut self, v: MinSamples) -> Self {
        self.min_samples = v;
        self
    }
    /// sklearn `residual_threshold` (`None` = the target MAD).
    pub fn residual_threshold(mut self, v: Option<f64>) -> Self {
        self.residual_threshold = v;
        self
    }
    /// sklearn `max_trials`.
    pub fn max_trials(mut self, v: usize) -> Self {
        self.max_trials = v;
        self
    }
    /// sklearn `max_skips`.
    pub fn max_skips(mut self, v: f64) -> Self {
        self.max_skips = v;
        self
    }
    /// sklearn `stop_n_inliers`.
    pub fn stop_n_inliers(mut self, v: f64) -> Self {
        self.stop_n_inliers = v;
        self
    }
    /// sklearn `stop_score`.
    pub fn stop_score(mut self, v: f64) -> Self {
        self.stop_score = v;
        self
    }
    /// sklearn `stop_probability`.
    pub fn stop_probability(mut self, v: f64) -> Self {
        self.stop_probability = v;
        self
    }
    /// sklearn `loss` — the estimator's ONE string-valued parameter.
    pub fn loss(mut self, v: RansacLoss) -> Self {
        self.loss = v;
        self
    }
    /// sklearn `loss` from its STRING spelling, the form the PyO3 boundary
    /// receives. Unknown values become [`BuildError::UnknownLoss`], which is
    /// the single mapper the boundary already turns into a `ValueError` (D-09).
    pub fn loss_str(mut self, v: &str) -> Result<Self, BuildError> {
        self.loss = match v {
            "absolute_error" => RansacLoss::AbsoluteError,
            "squared_error" => RansacLoss::SquaredError,
            other => {
                return Err(BuildError::UnknownLoss {
                    value: other.to_string(),
                })
            }
        };
        Ok(self)
    }
    /// The base `LinearRegression`'s `fit_intercept`.
    pub fn base_fit_intercept(mut self, v: bool) -> Self {
        self.base_fit_intercept = v;
        self
    }
    /// Where the trial SCAN runs (DEVICE-PARAM-01). A preference: the foreign
    /// base arm has only a host scan and reports `device_ = "cpu"` whatever this
    /// says, rather than faking it.
    pub fn device(mut self, v: Device) -> Self {
        self.device = v;
        self
    }
    /// [`device`](Self::device) from its STRING spelling, the form the PyO3
    /// boundary receives. An unrecognised value becomes
    /// [`BuildError::UnknownDevice`], the single mapper the boundary already
    /// turns into a `ValueError` (D-09).
    pub fn device_str(mut self, v: &str) -> Result<Self, BuildError> {
        self.device = Device::from_name(v).ok_or_else(|| BuildError::UnknownDevice {
            value: v.to_string(),
        })?;
        Ok(self)
    }

    /// Validate the data-INDEPENDENT hyperparameters and construct (D-08).
    ///
    /// The bounds are sklearn's `_parameter_constraints`, with one deliberate
    /// addition: sklearn's `Interval(RealNotInt, 0, 1, closed="both")` ADMITS
    /// `min_samples=0.0`, but its `fit` then falls through both branches of the
    /// `0 < min_samples < 1` / `min_samples >= 1` chain and dies on an unbound
    /// local. `Fraction(0.0)` is rejected here instead, with a message that
    /// says what the bound is.
    pub fn build<F>(self) -> Result<RansacRegressor<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if self.max_trials < 1 {
            return Err(BuildError::InvalidHyperprior {
                estimator: ESTIMATOR,
                param: "max_trials",
                value: self.max_trials as f64,
                bound: ">= 1",
            });
        }
        if !(self.max_skips >= 0.0) {
            return Err(BuildError::InvalidHyperprior {
                estimator: ESTIMATOR,
                param: "max_skips",
                value: self.max_skips,
                bound: ">= 0",
            });
        }
        if !(self.stop_n_inliers >= 0.0) {
            return Err(BuildError::InvalidHyperprior {
                estimator: ESTIMATOR,
                param: "stop_n_inliers",
                value: self.stop_n_inliers,
                bound: ">= 0",
            });
        }
        if self.stop_score.is_nan() {
            return Err(BuildError::InvalidHyperprior {
                estimator: ESTIMATOR,
                param: "stop_score",
                value: self.stop_score,
                bound: "not NaN",
            });
        }
        if !(0.0..=1.0).contains(&self.stop_probability) {
            return Err(BuildError::InvalidHyperprior {
                estimator: ESTIMATOR,
                param: "stop_probability",
                value: self.stop_probability,
                bound: "in [0, 1]",
            });
        }
        if let Some(rt) = self.residual_threshold {
            if !rt.is_finite() || rt < 0.0 {
                return Err(BuildError::InvalidHyperprior {
                    estimator: ESTIMATOR,
                    param: "residual_threshold",
                    value: rt,
                    bound: "finite and >= 0",
                });
            }
        }
        match self.min_samples {
            MinSamples::Auto => {}
            MinSamples::Absolute(k) => {
                if k < 1 {
                    return Err(BuildError::InvalidHyperprior {
                        estimator: ESTIMATOR,
                        param: "min_samples",
                        value: k as f64,
                        bound: ">= 1",
                    });
                }
            }
            MinSamples::Fraction(f) => {
                if !(f > 0.0 && f < 1.0) {
                    return Err(BuildError::InvalidHyperprior {
                        estimator: ESTIMATOR,
                        param: "min_samples",
                        value: f,
                        bound: "in (0, 1) when fractional",
                    });
                }
            }
        }

        Ok(RansacRegressor {
            config: RansacConfig {
                min_samples: self.min_samples,
                residual_threshold: self.residual_threshold,
                max_trials: self.max_trials,
                max_skips: self.max_skips,
                stop_n_inliers: self.stop_n_inliers,
                stop_score: self.stop_score,
                stop_probability: self.stop_probability,
                loss: self.loss,
                base_fit_intercept: self.base_fit_intercept,
                device: self.device,
            },
            coef_: Vec::new(),
            intercept_: Vec::new(),
            inlier_mask_: Vec::new(),
            n_trials_: 0,
            n_skips_no_inliers_: 0,
            n_skips_invalid_data_: 0,
            n_skips_invalid_model_: 0,
            exceeded_max_skips_: false,
            min_samples_: 0,
            residual_threshold_: 0.0,
            n_features_: 0,
            n_targets_: 0,
            device_: "cpu",
            batch_width_: 1,
            has_model_: true,
            _state: PhantomData,
        })
    }
}
