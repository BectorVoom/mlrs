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
//! | piece | here | [`ransac_host`](mlrs_backend::prims::ransac_host) |
//! |---|---|---|
//! | draw sequence (numpy MT19937) | ✔ | |
//! | skip / stop / tie-break bookkeeping | ✔ | |
//! | `min_samples` + `residual_threshold` lowering | ✔ | |
//! | per-trial `n × d` scan, consensus R² | | ✔ |
//! | sub-sample least squares | | ✔ |
//!
//! The split is forced: the draw is
//! [`sample_without_replacement`](crate::model_selection::rng::sample_without_replacement),
//! which lives in this crate's `model_selection` surface, and `mlrs-backend`
//! does not (and must not) depend on `mlrs-algos`.
//!
//! ## The base estimator is a `LinearRegression`, deliberately
//! sklearn's `estimator` parameter takes any duck-typed regressor, and its
//! DEFAULT — the only value the vast majority of uses pass — is
//! `LinearRegression()`. That is what this type fits, through
//! [`RansacHostEngine::subset_lstsq`], which reproduces sklearn's
//! `_preprocess_data` → `_rescale_data` → `scipy.linalg.lstsq` chain including
//! its singular-value cutoff. The one base hyperparameter that survives into
//! the arithmetic — `fit_intercept` — is a builder setter here.
//!
//! An arbitrary Python estimator cannot be driven from this loop without
//! re-acquiring the GIL inside every trial, which is the same conclusion
//! `feature_selection`'s meta-selectors reached; the `mlrs.linear` shim
//! therefore keeps a faithful Python trial loop for that case and routes to
//! this engine for the LinearRegression base. What it does NOT do is give up
//! the parameters that are RANSAC's own — `is_data_valid` and `is_model_valid`
//! are supported here as native callbacks ([`RansacCallbacks`]), because they
//! fire at most once per trial and a hundred GIL crossings cost nothing against
//! a hundred `n × d` scans.
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

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::prims::linear_predict::{
    linear_predict_host, linear_predict_multi_host, HostPrediction,
};
use mlrs_backend::prims::ransac_host::{dynamic_max_trials, RansacHostEngine};

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
/// Both receive the SUB-SAMPLE (`m × d` row-major `x`, `m × t` row-major `y`,
/// and `m`), which is exactly what sklearn passes as `X_subset` / `y_subset`.
/// [`Default`] is "neither predicate", the sklearn default.
pub struct RansacCallbacks<'h, F> {
    /// Called with the drawn sub-sample BEFORE the base model is fitted to it.
    #[allow(clippy::type_complexity)]
    pub is_data_valid: Option<&'h dyn Fn(&[F], &[F], usize) -> RansacVerdict>,
    /// Called with the fitted candidate and the sub-sample it came from.
    #[allow(clippy::type_complexity)]
    pub is_model_valid: Option<&'h dyn Fn(RansacModel<'_>, &[F], &[F], usize) -> RansacVerdict>,
}

impl<F> Default for RansacCallbacks<'_, F> {
    fn default() -> Self {
        Self {
            is_data_valid: None,
            is_model_valid: None,
        }
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
    /// into the Python object.
    ///
    /// This is the only fit ingress: the engine reads host memory on every
    /// backend, so a device upload of the design would be pure cost (module
    /// docs).
    pub fn fit_from_host_slice(
        self,
        x: &[F],
        y: &[F],
        shape: (usize, usize),
        n_targets: usize,
        sample_weight: Option<&[f64]>,
        rng: &mut NumpyRandomState,
        callbacks: &RansacCallbacks<'_, F>,
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

        let engine = RansacHostEngine::new(x, y, n, d, t)?;

        // --- 2. Lower `residual_threshold` (`None` → the target MAD). -------
        let threshold = match cfg.residual_threshold {
            Some(v) => v,
            None => engine.target_mad(),
        };

        // --- 3. The trial loop, statement for statement with sklearn's. -----
        // `n_inliers_best` starts at ONE, not zero: sklearn's incumbent is a
        // sentinel, so a trial that finds a single inlier does not qualify and
        // is counted as a `no_inliers` skip.
        let mut n_inliers_best = 1usize;
        let mut score_best = f64::NEG_INFINITY;
        let mut have_best = false;
        let mut mask = vec![false; n];
        let mut best_mask = vec![false; n];

        let mut n_trials = 0usize;
        let mut n_skips_no_inliers = 0usize;
        let mut n_skips_invalid_data = 0usize;
        let mut n_skips_invalid_model = 0usize;
        // sklearn's `max_trials` is shadowed by a FLOAT inside the loop, because
        // `_dynamic_max_trials` can return `inf`.
        let mut max_trials = cfg.max_trials as f64;

        while (n_trials as f64) < max_trials {
            n_trials += 1;

            let skips = n_skips_no_inliers + n_skips_invalid_data + n_skips_invalid_model;
            if (skips as f64) > cfg.max_skips {
                break;
            }

            let idxs = sample_without_replacement(n, min_samples, SampleMethod::Auto, rng)
                .expect("min_samples <= n_samples was checked above");

            if let Some(f) = callbacks.is_data_valid {
                let xs = engine.gather_x(&idxs);
                let ys = engine.gather_y(&idxs);
                match f(&xs, &ys, min_samples) {
                    RansacVerdict::Valid => {}
                    RansacVerdict::Invalid => {
                        n_skips_invalid_data += 1;
                        continue;
                    }
                    RansacVerdict::Abort => {
                        return Err(AlgoError::CallbackAborted {
                            estimator: ESTIMATOR,
                            callback: "is_data_valid",
                        })
                    }
                }
            }

            let (coef, intercept) =
                engine.subset_lstsq(&idxs, cfg.base_fit_intercept, sample_weight);

            if let Some(f) = callbacks.is_model_valid {
                let xs = engine.gather_x(&idxs);
                let ys = engine.gather_y(&idxs);
                let model = RansacModel {
                    coef: &coef,
                    intercept: &intercept,
                    n_features: d,
                    n_targets: t,
                };
                match f(model, &xs, &ys, min_samples) {
                    RansacVerdict::Valid => {}
                    RansacVerdict::Invalid => {
                        n_skips_invalid_model += 1;
                        continue;
                    }
                    RansacVerdict::Abort => {
                        return Err(AlgoError::CallbackAborted {
                            estimator: ESTIMATOR,
                            callback: "is_model_valid",
                        })
                    }
                }
            }

            let scan = engine.scan(&coef, &intercept, cfg.loss, threshold, &mut mask);

            // Fewer inliers than the incumbent — sklearn books this under
            // `n_skips_no_inliers_` (the name is historical; it covers "not
            // enough", not only "none").
            if scan.n_inliers < n_inliers_best {
                n_skips_no_inliers += 1;
                continue;
            }

            let score = engine.r2_on_mask(&mask, scan.n_inliers, &scan.sq_err);

            // Equal consensus, worse score — keep the incumbent. NaN (fewer
            // than two inliers) compares false here, exactly as it does in
            // Python, so it does NOT skip.
            if scan.n_inliers == n_inliers_best && score < score_best {
                continue;
            }

            n_inliers_best = scan.n_inliers;
            score_best = score;
            best_mask.copy_from_slice(&mask);
            have_best = true;

            max_trials = max_trials.min(dynamic_max_trials(
                n_inliers_best,
                n,
                min_samples,
                cfg.stop_probability,
            ));

            if n_inliers_best as f64 >= cfg.stop_n_inliers || score_best >= cfg.stop_score {
                break;
            }
        }

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

        // --- 4. Final model: the base estimator refitted on the consensus. --
        let inlier_idxs: Vec<i64> = best_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| m.then_some(i as i64))
            .collect();
        let (coef_, intercept_) =
            engine.subset_lstsq(&inlier_idxs, cfg.base_fit_intercept, sample_weight);

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
            _state: PhantomData,
        })
    }
}
