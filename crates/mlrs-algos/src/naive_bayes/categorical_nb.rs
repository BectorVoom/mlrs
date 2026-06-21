//! `CategoricalNB` (NB-05) — Categorical Naive Bayes,
//! ≈ `sklearn.naive_bayes.CategoricalNB`.
//!
//! Wave-0 SCAFFOLD: struct + the [`MinCategories`] enum (D-04) +
//! [`CategoricalNBBuilder`] (D-02 sklearn defaults) + `build()`
//! (data-INDEPENDENT validation incl. the D-06 `force_alpha` clip+warn and the
//! per-entry `min_categories >= 0` check) are SHIPPED; the `Fit` impl carries a
//! REAL geometry guard but a `todo!()` compute body filled in Wave 1. Analog:
//! `multinomial_nb.rs` (discrete builder shape) + the `BandwidthSpec` enum
//! precedent from `density/kernel_density.rs`. SEPARATE struct (D-03).
//!
//! `feature_log_prob_` is a RAGGED `Vec<Vec<f64>>` (one matrix per feature,
//! variable category count — Pitfall 7), NOT a single tensor; the non-negative-
//! integer input validation and the predict-time category-index guard live at
//! `fit` / `predict` (data-DEPENDENT — [`AlgoError::InvalidCategoricalInput`]),
//! wired in Wave 1.
//!
//! Tests live in `crates/mlrs-algos/tests/categorical_nb_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::naive_bayes::multinomial_nb::validate_discrete_alpha;
use crate::traits::Fit;

/// The minimum-categories-per-feature specification (D-04), modeled on the
/// `BandwidthSpec` value-shaped-knob precedent. Captures sklearn's
/// scalar-vs-per-feature-vs-None `min_categories` polymorphism at the type level.
///
/// `PerFeature` carries a `Vec`, so this enum is `Clone` (NOT `Copy`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MinCategories {
    /// `None`: infer each feature's category count from the data at `fit`
    /// (`max + 1` per feature).
    Infer,
    /// A single scalar applied to EVERY feature (sklearn's `int` form): each
    /// feature's category count is at least this value.
    Uniform(usize),
    /// A per-feature vector (sklearn's array-like form): entry `j` is feature
    /// `j`'s minimum category count. Length-`== n_features` is a data-DEPENDENT
    /// check at `fit`.
    PerFeature(Vec<usize>),
}

/// Categorical Naive Bayes (NB-05). Construct via [`CategoricalNB::builder`],
/// then [`Fit::fit`] + (Wave-1) the predict surface. Fitted `feature_log_prob_`
/// is a ragged host `Vec<Vec<f64>>` (one matrix per feature); `class_log_prior_`
/// is host f64 (D-03), `None` until `fit`.
pub struct CategoricalNB<F> {
    /// Additive smoothing (D-02 default `1.0`).
    alpha: f64,
    /// Keep `alpha` as-is when `< 1e-10` (D-02 default `true`); else clip (D-06).
    force_alpha: bool,
    /// Learn class priors from the data (D-02 default `true`).
    fit_prior: bool,
    /// User-supplied class priors, or `None` → empirical (D-02 default `None`).
    class_prior: Option<Vec<f64>>,
    /// Minimum categories per feature (D-02 default `MinCategories::Infer`,
    /// i.e. sklearn `min_categories=None`).
    min_categories: MinCategories,
    /// DISTINCT sorted class labels inferred at `fit`.
    classes_: Vec<i64>,
    /// Feature count inferred at `fit`.
    n_features: usize,
    /// Per-feature category counts learned at `fit` (length `n_features`), `None`
    /// until `fit`.
    n_categories_: Option<Vec<usize>>,
    /// Ragged fitted `feature_log_prob_`: `feature_log_prob_[j]` is the
    /// `n_classes × n_categories_[j]` log-probability matrix for feature `j`
    /// (Pitfall 7). `None` until `fit`.
    feature_log_prob_: Option<Vec<Vec<f64>>>,
    /// Per-class log-prior (host f64), `None` until `fit`.
    class_log_prior_: Option<Vec<f64>>,
    /// Marker to retain the `F` type parameter (the device buffers land in Wave-1).
    _marker: std::marker::PhantomData<F>,
}

impl<F> CategoricalNB<F>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `CategoricalNB` with sklearn's defaults (D-02).
    pub fn builder() -> CategoricalNBBuilder {
        CategoricalNBBuilder::default()
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

/// Builder for [`CategoricalNB`] (D-01). Defaults (D-02): `alpha=1.0`,
/// `force_alpha=true`, `fit_prior=true`, `class_prior=None`,
/// `min_categories=Infer`. Setter names mirror sklearn (D-09).
#[derive(Debug, Clone)]
pub struct CategoricalNBBuilder {
    alpha: f64,
    force_alpha: bool,
    fit_prior: bool,
    class_prior: Option<Vec<f64>>,
    min_categories: MinCategories,
}

impl Default for CategoricalNBBuilder {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            force_alpha: true,
            fit_prior: true,
            class_prior: None,
            min_categories: MinCategories::Infer,
        }
    }
}

impl CategoricalNBBuilder {
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
    /// Set the minimum-categories-per-feature specification (D-04).
    pub fn min_categories(mut self, min_categories: MinCategories) -> Self {
        self.min_categories = min_categories;
        self
    }

    /// Build the estimator, validating the data-INDEPENDENT hyperparameters at
    /// `build()` (D-05): `alpha >= 0`, finite+non-negative `class_prior`, the D-06
    /// `force_alpha` clip+warn (shared [`validate_discrete_alpha`]). Since
    /// `MinCategories` carries `usize` entries they are non-negative by
    /// construction; the per-feature LENGTH-`== n_features` check is data-DEPENDENT
    /// and stays at `fit`. The [`BuildError::InvalidMinCategories`] variant exists
    /// for any future signed-input path (kept for the typed surface).
    pub fn build<F>(self) -> Result<CategoricalNB<F>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        let alpha = validate_discrete_alpha(
            "categorical_nb",
            self.alpha,
            self.force_alpha,
            self.class_prior.as_deref(),
        )?;
        Ok(CategoricalNB {
            alpha,
            force_alpha: self.force_alpha,
            fit_prior: self.fit_prior,
            class_prior: self.class_prior,
            min_categories: self.min_categories,
            classes_: Vec::new(),
            n_features: 0,
            n_categories_: None,
            feature_log_prob_: None,
            class_log_prior_: None,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<F> Fit<F> for CategoricalNB<F>
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
            estimator: "categorical_nb",
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
        // Wave-1 (11-04): the non-negative-integer categorical input validation
        // (AlgoError::InvalidCategoricalInput), the per-feature category-count
        // inference / MinCategories padding, the ragged feature_log_prob_, and
        // the empirical-or-supplied class_log_prior_.
        let _ = (
            self.alpha,
            self.force_alpha,
            self.fit_prior,
            &self.class_prior,
            &self.min_categories,
        );
        let _ = (
            &self.n_categories_,
            &self.feature_log_prob_,
            &self.class_log_prior_,
            &self.n_features,
        );
        todo!("CategoricalNB::fit compute body — Wave 1 (11-04)")
    }
}
