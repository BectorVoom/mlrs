//! `GaussianNB` (NB-01) — Gaussian Naive Bayes, ≈ `sklearn.naive_bayes.GaussianNB`.
//!
//! Wave-0 SCAFFOLD: the struct, the [`GaussianNBBuilder`] (D-02 — sklearn-default
//! field initializers), and the `build() -> Result<GaussianNB<F>, BuildError>`
//! data-INDEPENDENT validation are SHIPPED; the `Fit` impl carries a REAL
//! geometry guard but a `todo!()` compute body filled in Wave 1. The closest
//! analog is `linear/mbsgd_classifier.rs` (builder + `classes_` remap +
//! device-resident fitted state). Construction is the builder (D-01).
//!
//! Tests live in `crates/mlrs-algos/tests/gaussian_nb_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::traits::Fit;

/// Gaussian Naive Bayes (NB-01). Construct via [`GaussianNB::builder`], then
/// [`Fit::fit`] + (Wave-1) `predict_labels` / `predict_proba` /
/// `predict_log_proba`. Fitted `theta_` (means) / `var_` (variances) /
/// `class_prior_` are device-resident / host f64 small tensors (D-03), `None`
/// until `fit`.
pub struct GaussianNB<F> {
    /// User-supplied class priors, or `None` → empirical from `class_count_`.
    priors: Option<Vec<f64>>,
    /// Portion of the largest feature variance added to all variances (D-02
    /// default `1e-9`) for numerical stability.
    var_smoothing: f64,
    /// DISTINCT sorted class labels inferred at `fit` (NO 2-class restriction).
    classes_: Vec<i64>,
    /// Feature count inferred at `fit`.
    n_features: usize,
    /// Fitted per-class feature means `theta_` (`n_classes × n_features`),
    /// device-resident, `None` until `fit`.
    theta_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted per-class feature variances `var_` (`n_classes × n_features`),
    /// device-resident, `None` until `fit`.
    var_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Per-class log-prior (host f64, length `n_classes`), `None` until `fit`.
    class_log_prior_: Option<Vec<f64>>,
}

impl<F> GaussianNB<F>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `GaussianNB` with sklearn's defaults (D-02).
    pub fn builder() -> GaussianNBBuilder {
        GaussianNBBuilder::default()
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

/// Builder for [`GaussianNB`] (D-01). Default initializers encode the sklearn
/// `GaussianNB` defaults (D-02): `priors=None`, `var_smoothing=1e-9`. Setter
/// names mirror sklearn (D-09): `.priors(..)` / `.var_smoothing(..)` — NO `alpha`.
#[derive(Debug, Clone, Default)]
pub struct GaussianNBBuilder {
    priors: Option<Vec<f64>>,
    var_smoothing: Option<f64>,
}

impl GaussianNBBuilder {
    /// Set the class priors (`None` → empirical from `class_count_`).
    pub fn priors(mut self, priors: Option<Vec<f64>>) -> Self {
        self.priors = priors;
        self
    }
    /// Set `var_smoothing` (the portion of the largest variance added to all).
    pub fn var_smoothing(mut self, var_smoothing: f64) -> Self {
        self.var_smoothing = Some(var_smoothing);
        self
    }

    /// Build the estimator, validating the data-INDEPENDENT hyperparameters at
    /// `build()` BEFORE any data is seen (D-05):
    ///
    /// - `var_smoothing >= 0` ([`BuildError::InvalidVarSmoothing`]).
    /// - every `priors` entry finite + non-negative
    ///   ([`BuildError::InvalidClassPrior`]) — the data-DEPENDENT
    ///   length-`== n_classes` / sum-to-one checks stay at `fit`.
    pub fn build<F>(self) -> Result<GaussianNB<F>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        let var_smoothing = self.var_smoothing.unwrap_or(1e-9);
        if !(var_smoothing >= 0.0) {
            return Err(BuildError::InvalidVarSmoothing {
                estimator: "gaussian_nb",
                var_smoothing,
            });
        }
        if let Some(ref p) = self.priors {
            if p.iter().any(|&v| !v.is_finite() || v < 0.0) {
                return Err(BuildError::InvalidClassPrior {
                    estimator: "gaussian_nb",
                });
            }
        }
        Ok(GaussianNB {
            priors: self.priors,
            var_smoothing,
            classes_: Vec::new(),
            n_features: 0,
            theta_: None,
            var_: None,
            class_log_prior_: None,
        })
    }
}

impl<F> Fit<F> for GaussianNB<F>
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
        // Data-DEPENDENT geometry guard BEFORE any launch (T-11-02 / ASVS V5).
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "gaussian_nb",
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
        // Wave-1 (11-02): host distinct-sorted classes_, the two-GATHER
        // theta_/var_ via nb_common::class_grouped_sum / class_grouped_sumsq, the
        // var_smoothing floor, and the empirical-or-supplied class_log_prior_.
        let _ = (&self.priors, self.var_smoothing);
        let _ = (&self.theta_, &self.var_, &self.class_log_prior_, &self.n_features);
        todo!("GaussianNB::fit compute body — Wave 1 (11-02)")
    }
}
