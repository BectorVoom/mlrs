//! `MultinomialNB` (NB-02) — Multinomial Naive Bayes,
//! ≈ `sklearn.naive_bayes.MultinomialNB`.
//!
//! Wave-0 SCAFFOLD: struct + [`MultinomialNBBuilder`] (D-02 sklearn defaults) +
//! `build()` (data-INDEPENDENT validation incl. the D-06 `force_alpha` clip+warn)
//! are SHIPPED; the `Fit` impl carries a REAL geometry guard but a `todo!()`
//! compute body filled in Wave 1. Analog: `linear/mbsgd_classifier.rs` (builder +
//! GEMM joint-LL). This is a SEPARATE struct from the other variants (D-03 — no
//! shared base); do NOT copy MultinomialNB into ComplementNB (Pitfall 6).
//!
//! Tests live in `crates/mlrs-algos/tests/multinomial_nb_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::traits::Fit;

/// Multinomial Naive Bayes (NB-02). Construct via [`MultinomialNB::builder`],
/// then [`Fit::fit`] + (Wave-1) the predict surface. Fitted `feature_log_prob_`
/// (`n_classes × n_features`) / `class_log_prior_` are device-resident / host f64
/// (D-03), `None` until `fit`.
pub struct MultinomialNB<F> {
    /// Additive (Laplace/Lidstone) smoothing (D-02 default `1.0`).
    alpha: f64,
    /// Keep `alpha` as-is even when `< 1e-10` (D-02 default `true`); when `false`
    /// a tiny `alpha` is clipped to `1e-10` at `build()` with a warning (D-06).
    force_alpha: bool,
    /// Learn class priors from the data (D-02 default `true`); when `false` a
    /// uniform prior is used.
    fit_prior: bool,
    /// User-supplied class priors, or `None` → empirical (D-02 default `None`).
    class_prior: Option<Vec<f64>>,
    /// DISTINCT sorted class labels inferred at `fit`.
    classes_: Vec<i64>,
    /// Feature count inferred at `fit`.
    n_features: usize,
    /// Fitted `feature_log_prob_` (`n_classes × n_features`), device-resident,
    /// `None` until `fit`.
    feature_log_prob_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Per-class log-prior (host f64, length `n_classes`), `None` until `fit`.
    class_log_prior_: Option<Vec<f64>>,
}

impl<F> MultinomialNB<F>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `MultinomialNB` with sklearn's defaults (D-02).
    pub fn builder() -> MultinomialNBBuilder {
        MultinomialNBBuilder::default()
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

/// Builder for [`MultinomialNB`] (D-01). Defaults (D-02): `alpha=1.0`,
/// `force_alpha=true`, `fit_prior=true`, `class_prior=None`. Setter names mirror
/// sklearn (D-09).
#[derive(Debug, Clone)]
pub struct MultinomialNBBuilder {
    alpha: f64,
    force_alpha: bool,
    fit_prior: bool,
    class_prior: Option<Vec<f64>>,
}

impl Default for MultinomialNBBuilder {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            force_alpha: true,
            fit_prior: true,
            class_prior: None,
        }
    }
}

impl MultinomialNBBuilder {
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
    /// `build()` BEFORE any data is seen (D-05):
    ///
    /// - `alpha >= 0` ([`BuildError::InvalidAlpha`]).
    /// - every `class_prior` entry finite + non-negative
    ///   ([`BuildError::InvalidClassPrior`]).
    /// - the D-06 `force_alpha` clip+warn: when `force_alpha == false` and
    ///   `alpha < 1e-10` the stored `alpha` is clipped to `1e-10` with a warning
    ///   (sklearn parity depends only on the clipped numeric, A2).
    pub fn build<F>(self) -> Result<MultinomialNB<F>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        let alpha = validate_discrete_alpha(
            "multinomial_nb",
            self.alpha,
            self.force_alpha,
            self.class_prior.as_deref(),
        )?;
        Ok(MultinomialNB {
            alpha,
            force_alpha: self.force_alpha,
            fit_prior: self.fit_prior,
            class_prior: self.class_prior,
            classes_: Vec::new(),
            n_features: 0,
            feature_log_prob_: None,
            class_log_prior_: None,
        })
    }
}

impl<F> Fit<F> for MultinomialNB<F>
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
            estimator: "multinomial_nb",
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
        let _ = (self.alpha, self.force_alpha, self.fit_prior, &self.class_prior);
        let _ = (&self.feature_log_prob_, &self.class_log_prior_, &self.n_features);
        todo!("MultinomialNB::fit compute body — Wave 1 (11-03)")
    }
}

/// Shared data-INDEPENDENT alpha / class_prior validation + the D-06
/// `force_alpha` clip+warn for the four discrete NB variants (Multinomial /
/// Bernoulli / Complement / Categorical). Lives here (the first discrete variant)
/// and is `pub(crate)` so the sibling discrete builders reuse it WITHOUT a shared
/// base struct (D-03 — sharing is at the function level only). Returns the
/// possibly-clipped `alpha`.
pub(crate) fn validate_discrete_alpha(
    estimator: &'static str,
    alpha: f64,
    force_alpha: bool,
    class_prior: Option<&[f64]>,
) -> Result<f64, BuildError> {
    if !(alpha >= 0.0) {
        return Err(BuildError::InvalidAlpha { estimator, alpha });
    }
    if let Some(p) = class_prior {
        if p.iter().any(|&v| !v.is_finite() || v < 0.0) {
            return Err(BuildError::InvalidClassPrior { estimator });
        }
    }
    // D-06: sklearn clips a too-small alpha to 1e-10 (with a warning) unless
    // force_alpha. Parity depends only on the clipped numeric, not the text (A2).
    let alpha = if !force_alpha && alpha < 1e-10 {
        log::warn!(
            "estimator '{estimator}': alpha too small, setting alpha=1e-10. \
             Use force_alpha=true to keep alpha unchanged."
        );
        1e-10
    } else {
        alpha
    };
    Ok(alpha)
}
