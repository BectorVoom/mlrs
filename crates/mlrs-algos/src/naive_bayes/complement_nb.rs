//! `ComplementNB` (NB-04) — Complement Naive Bayes,
//! ≈ `sklearn.naive_bayes.ComplementNB`.
//!
//! Wave-0 SCAFFOLD: struct + [`ComplementNBBuilder`] (D-02 sklearn defaults) +
//! `build()` (data-INDEPENDENT validation incl. the D-06 `force_alpha` clip+warn)
//! are SHIPPED; the `Fit` impl carries a REAL geometry guard but a `todo!()`
//! compute body filled in Wave 1. Analog: `multinomial_nb.rs` (discrete builder
//! shape). SEPARATE struct (D-03). ComplementNB carries the extra `norm: bool`
//! knob and decodes with `argmin` INTERNALLY (D-08 — it picks the class whose
//! complement fits worst; the sign flips). Its complement-weighted
//! `feature_log_prob_` + optional L1 `norm` is a DIFFERENT formula from
//! MultinomialNB — implement it verbatim from FEATURES.md in Wave 1, do NOT copy
//! Multinomial (Pitfall 6).
//!
//! Tests live in `crates/mlrs-algos/tests/complement_nb_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::naive_bayes::multinomial_nb::validate_discrete_alpha;
use crate::traits::Fit;

/// Complement Naive Bayes (NB-04). Construct via [`ComplementNB::builder`], then
/// [`Fit::fit`] + (Wave-1) the predict surface (argmin decode internally, D-08).
/// Fitted `feature_log_prob_` / `class_log_prior_` are device-resident / host f64
/// (D-03), `None` until `fit`.
pub struct ComplementNB<F> {
    /// Additive smoothing (D-02 default `1.0`).
    alpha: f64,
    /// Keep `alpha` as-is when `< 1e-10` (D-02 default `true`); else clip (D-06).
    force_alpha: bool,
    /// Learn class priors from the data (D-02 default `true`).
    fit_prior: bool,
    /// User-supplied class priors, or `None` → empirical (D-02 default `None`).
    class_prior: Option<Vec<f64>>,
    /// Apply a second L1 normalization to the complement weights (D-02 default
    /// `false`).
    norm: bool,
    /// DISTINCT sorted class labels inferred at `fit`.
    classes_: Vec<i64>,
    /// Feature count inferred at `fit`.
    n_features: usize,
    /// Fitted complement-weighted `feature_log_prob_` (`n_classes × n_features`),
    /// device-resident, `None` until `fit`.
    feature_log_prob_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Per-class log-prior (host f64), `None` until `fit`.
    class_log_prior_: Option<Vec<f64>>,
}

impl<F> ComplementNB<F>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `ComplementNB` with sklearn's defaults (D-02).
    pub fn builder() -> ComplementNBBuilder {
        ComplementNBBuilder::default()
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

/// Builder for [`ComplementNB`] (D-01). Defaults (D-02): `alpha=1.0`,
/// `force_alpha=true`, `fit_prior=true`, `class_prior=None`, `norm=false`. Setter
/// names mirror sklearn (D-09).
#[derive(Debug, Clone)]
pub struct ComplementNBBuilder {
    alpha: f64,
    force_alpha: bool,
    fit_prior: bool,
    class_prior: Option<Vec<f64>>,
    norm: bool,
}

impl Default for ComplementNBBuilder {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            force_alpha: true,
            fit_prior: true,
            class_prior: None,
            norm: false,
        }
    }
}

impl ComplementNBBuilder {
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
    /// Set whether to apply the second L1 normalization to the weights.
    pub fn norm(mut self, norm: bool) -> Self {
        self.norm = norm;
        self
    }

    /// Build the estimator, validating the data-INDEPENDENT hyperparameters at
    /// `build()` (D-05): `alpha >= 0`, finite+non-negative `class_prior`, and the
    /// D-06 `force_alpha` clip+warn (shared [`validate_discrete_alpha`]). `norm`
    /// needs no validation.
    pub fn build<F>(self) -> Result<ComplementNB<F>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        let alpha = validate_discrete_alpha(
            "complement_nb",
            self.alpha,
            self.force_alpha,
            self.class_prior.as_deref(),
        )?;
        Ok(ComplementNB {
            alpha,
            force_alpha: self.force_alpha,
            fit_prior: self.fit_prior,
            class_prior: self.class_prior,
            norm: self.norm,
            classes_: Vec::new(),
            n_features: 0,
            feature_log_prob_: None,
            class_log_prior_: None,
        })
    }
}

impl<F> Fit<F> for ComplementNB<F>
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
            estimator: "complement_nb",
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
            self.fit_prior,
            &self.class_prior,
            self.norm,
        );
        let _ = (&self.feature_log_prob_, &self.class_log_prior_, &self.n_features);
        todo!("ComplementNB::fit compute body — Wave 1 (11-03)")
    }
}
