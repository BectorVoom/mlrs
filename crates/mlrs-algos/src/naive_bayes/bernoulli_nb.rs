//! `BernoulliNB` (NB-03) — Bernoulli Naive Bayes,
//! ≈ `sklearn.naive_bayes.BernoulliNB`.
//!
//! Wave-0 SCAFFOLD: struct + [`BernoulliNBBuilder`] (D-02 sklearn defaults) +
//! `build()` (data-INDEPENDENT validation incl. the D-06 `force_alpha` clip+warn)
//! are SHIPPED; the `Fit` impl carries a REAL geometry guard but a `todo!()`
//! compute body filled in Wave 1. Analog: `multinomial_nb.rs` (discrete builder
//! shape) + the `Option<f64>` knob precedent from `density/kernel_density.rs`.
//! SEPARATE struct (D-03 — no shared base).
//!
//! The D-04 `binarize: Option<f64>` knob — `None` disables binarization
//! (assumes already-binary input); `Some(t)` thresholds `x > t → 1`. The
//! `(1 − x)·log(1 − p)` non-occurrence term folds into the Wave-1 GEMM via
//! `flp = log p − log(1 − p)` + a per-class constant `Σ_j log(1 − p_cj)`
//! (Pitfall 5) — set up there, not here.
//!
//! Tests live in `crates/mlrs-algos/tests/bernoulli_nb_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::naive_bayes::multinomial_nb::validate_discrete_alpha;
use crate::traits::Fit;

/// Bernoulli Naive Bayes (NB-03). Construct via [`BernoulliNB::builder`], then
/// [`Fit::fit`] + (Wave-1) the predict surface. Fitted `feature_log_prob_` /
/// `class_log_prior_` are device-resident / host f64 (D-03), `None` until `fit`.
pub struct BernoulliNB<F> {
    /// Additive smoothing (D-02 default `1.0`).
    alpha: f64,
    /// Keep `alpha` as-is when `< 1e-10` (D-02 default `true`); else clip (D-06).
    force_alpha: bool,
    /// Threshold for binarizing the input; `None` disables binarization (assumes
    /// already-binary), `Some(t)` maps `x > t → 1` (D-02 default `Some(0.0)`).
    binarize: Option<f64>,
    /// Learn class priors from the data (D-02 default `true`).
    fit_prior: bool,
    /// User-supplied class priors, or `None` → empirical (D-02 default `None`).
    class_prior: Option<Vec<f64>>,
    /// DISTINCT sorted class labels inferred at `fit`.
    classes_: Vec<i64>,
    /// Feature count inferred at `fit`.
    n_features: usize,
    /// Fitted `feature_log_prob_` (`n_classes × n_features`), device-resident.
    feature_log_prob_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Per-class log-prior (host f64), `None` until `fit`.
    class_log_prior_: Option<Vec<f64>>,
}

impl<F> BernoulliNB<F>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `BernoulliNB` with sklearn's defaults (D-02).
    pub fn builder() -> BernoulliNBBuilder {
        BernoulliNBBuilder::default()
    }

    /// The inferred class labels (empty until `fit`).
    pub fn classes(&self) -> &[i64] {
        &self.classes_
    }

    /// The per-class log-prior (`None` until `fit`).
    pub fn class_log_prior(&self) -> Option<&[f64]> {
        self.class_log_prior_.as_deref()
    }
}

/// Builder for [`BernoulliNB`] (D-01). Defaults (D-02): `alpha=1.0`,
/// `force_alpha=true`, `binarize=Some(0.0)`, `fit_prior=true`,
/// `class_prior=None`. Setter names mirror sklearn (D-09).
#[derive(Debug, Clone)]
pub struct BernoulliNBBuilder {
    alpha: f64,
    force_alpha: bool,
    binarize: Option<f64>,
    fit_prior: bool,
    class_prior: Option<Vec<f64>>,
}

impl Default for BernoulliNBBuilder {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            force_alpha: true,
            binarize: Some(0.0),
            fit_prior: true,
            class_prior: None,
        }
    }
}

impl BernoulliNBBuilder {
    /// Set the additive smoothing `alpha`.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
    /// Set whether to keep a tiny `alpha` as-is (else clip to `1e-10`, D-06).
    pub fn force_alpha(mut self, force_alpha: bool) -> Self {
        self.force_alpha = force_alpha;
        self
    }
    /// Set the binarization threshold (`None` disables binarization).
    pub fn binarize(mut self, binarize: Option<f64>) -> Self {
        self.binarize = binarize;
        self
    }
    /// Set whether to learn class priors from the data.
    pub fn fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }
    /// Set explicit class priors (`None` → empirical / uniform).
    pub fn class_prior(mut self, class_prior: Option<Vec<f64>>) -> Self {
        self.class_prior = class_prior;
        self
    }

    /// Build the estimator, validating the data-INDEPENDENT hyperparameters at
    /// `build()` BEFORE any data is seen (D-05): `alpha >= 0`, finite+non-negative
    /// `class_prior` entries, and the D-06 `force_alpha` clip+warn (shared
    /// [`validate_discrete_alpha`]). `binarize` needs no validation (any finite or
    /// `None` threshold is valid).
    pub fn build<F>(self) -> Result<BernoulliNB<F>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        let alpha = validate_discrete_alpha(
            "bernoulli_nb",
            self.alpha,
            self.force_alpha,
            self.class_prior.as_deref(),
        )?;
        Ok(BernoulliNB {
            alpha,
            force_alpha: self.force_alpha,
            binarize: self.binarize,
            fit_prior: self.fit_prior,
            class_prior: self.class_prior,
            classes_: Vec::new(),
            n_features: 0,
            feature_log_prob_: None,
            class_log_prior_: None,
        })
    }
}

impl<F> Fit<F> for BernoulliNB<F>
where
    F: Float + CubeElement + Pod,
{
    fn fit(
        &mut self,
        _pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
        let (n_samples, n_features) = shape;
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "bernoulli_nb",
            operation: "fit (requires y)",
        })?;
        if y.len() != n_samples {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_samples,
                cols: 1,
                len: y.len(),
            }));
        }
        let _ = (
            self.alpha,
            self.force_alpha,
            self.binarize,
            self.fit_prior,
            &self.class_prior,
        );
        let _ = (&self.feature_log_prob_, &self.class_log_prior_, &self.n_features);
        todo!("BernoulliNB::fit compute body — Wave 1 (11-03)")
    }
}
