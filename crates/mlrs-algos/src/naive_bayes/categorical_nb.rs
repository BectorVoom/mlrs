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
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::naive_bayes::multinomial_nb::{decode_classes, resolve_class_log_prior, validate_discrete_alpha};
use crate::naive_bayes::nb_common::{argmax_decode, log_sum_exp_normalize};
use crate::traits::{Fit, PredictLabels, PredictLogProba, PredictProba};

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
    /// Per-class sample counts (host f64, length `n_classes`), `None` until
    /// `fit`. The empirical-prior numerator AND the per-feature smoothing
    /// denominator `class_count[c] + alpha·n_categories_j` — kept so the
    /// predict-time unseen-category fallback computes the EXACT smoothed
    /// `log(alpha / denom_cj)` (T-11-04-02) without reconstructing it from the
    /// fitted table.
    class_count_: Option<Vec<f64>>,
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

    /// The per-class sample counts (`None` until `fit`).
    pub fn class_count(&self) -> Option<&[f64]> {
        self.class_count_.as_deref()
    }

    /// The per-feature category counts `n_categories_` (length `n_features`,
    /// `None` until `fit`). Entry `j` is the padded `n_categories_j =
    /// max(observed_max+1, min_categories_j)`.
    pub fn n_categories(&self) -> Option<&[usize]> {
        self.n_categories_.as_deref()
    }

    /// The ragged fitted `feature_log_prob_` (`feature_log_prob_[j]` is the
    /// `n_classes × n_categories_[j]` row-major log-prob matrix for feature `j`),
    /// `None` until `fit`.
    pub fn feature_log_prob(&self) -> Option<&[Vec<f64>]> {
        self.feature_log_prob_.as_deref()
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
            class_count_: None,
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
        pool: &mut BufferPool<ActiveRuntime>,
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
        let _ = self.force_alpha; // fitted-config provenance (D-06 clip applied at build()).

        // --- T-11-04-01: validate X is a non-negative-INTEGER categorical encoding
        //     BEFORE any table is sized (a negative / non-integer value would later
        //     index a ragged table out of bounds). Round-to-nearest within 1e-6 to
        //     tolerate the f32/f64 round-trip of integer-encoded categories. ---
        let x_host = x.to_host(pool);
        let mut x_cat: Vec<usize> = Vec::with_capacity(n_samples * n_features);
        for &xv in x_host.iter() {
            let xf = host_to_f64(xv);
            let xr = xf.round();
            if (xr - xf).abs() > 1e-6 || xr < 0.0 {
                return Err(AlgoError::InvalidCategoricalInput {
                    estimator: "categorical_nb",
                    reason: format!("feature values must be non-negative integers (got {xf})"),
                });
            }
            x_cat.push(xr as usize);
        }

        // --- classes_ / dense per-row class index / n_classes via the shared
        //     discrete decode (integer + i32-range label guard, WR-02). ---
        let (classes_, class_of_row, n_classes) = decode_classes::<F>(pool, y, n_samples)?;

        // class_count_[c] = #rows of class c (every observed class has >= 1).
        let mut class_count_: Vec<f64> = vec![0.0; n_classes];
        for &c in &class_of_row {
            class_count_[c] += 1.0;
        }

        // --- Per-feature observed_max + the MinCategories padding (D-04, Pitfall 7):
        //     n_categories_j = max(observed_max+1, min_categories_j). The
        //     PerFeature length-`== n_features` check is data-DEPENDENT (D-05). ---
        if let MinCategories::PerFeature(v) = &self.min_categories {
            if v.len() != n_features {
                return Err(AlgoError::InvalidCategoricalInput {
                    estimator: "categorical_nb",
                    reason: format!(
                        "min_categories (per-feature) length {} != n_features {n_features}",
                        v.len()
                    ),
                });
            }
        }
        let mut n_categories_: Vec<usize> = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let mut observed_max = 0usize;
            for i in 0..n_samples {
                observed_max = observed_max.max(x_cat[i * n_features + j]);
            }
            let base = observed_max + 1;
            let min_j = match &self.min_categories {
                MinCategories::Infer => 0,
                MinCategories::Uniform(u) => *u,
                MinCategories::PerFeature(v) => v[j],
            };
            n_categories_.push(base.max(min_j));
        }

        // --- Host-tabulate category_count_[j][c, k] (one owner per
        //     (feature, class, category) — a host count, NEVER a device scatter).
        //     feature_log_prob_[j][c, k] = log((count + alpha) /
        //       (class_count[c] + alpha · n_categories_j))  (Pitfall 4 — the
        //     denominator smoothing is alpha · n_categories_j). ---
        let alpha = self.alpha;
        let mut feature_log_prob_: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let n_cat_j = n_categories_[j];
            // category_count[c * n_cat_j + k]
            let mut count = vec![0.0f64; n_classes * n_cat_j];
            for i in 0..n_samples {
                let c = class_of_row[i];
                let k = x_cat[i * n_features + j];
                // k < n_cat_j by construction (n_categories_j >= observed_max+1).
                count[c * n_cat_j + k] += 1.0;
            }
            let mut flp = vec![0.0f64; n_classes * n_cat_j];
            for c in 0..n_classes {
                let denom = class_count_[c] + alpha * n_cat_j as f64;
                for k in 0..n_cat_j {
                    flp[c * n_cat_j + k] = ((count[c * n_cat_j + k] + alpha) / denom).ln();
                }
            }
            feature_log_prob_.push(flp);
        }

        // --- class_log_prior_: supplied class_prior (length == n_classes) takes
        //     precedence; else empirical log(count_c/n) when fit_prior=true; else
        //     uniform (the shared discrete resolver, sklearn semantics). ---
        let class_log_prior_ = resolve_class_log_prior(
            "categorical_nb",
            self.fit_prior,
            &self.class_prior,
            &class_count_,
            n_classes,
        )?;

        // WR-07: the only device scratch was the host read of `x`; the ragged
        // tables are host f64. Re-fit simply overwrites the host-resident fitted
        // state (no device buffers held), so live_bytes is conserved.
        self.classes_ = classes_;
        self.n_features = n_features;
        self.n_categories_ = Some(n_categories_);
        self.feature_log_prob_ = Some(feature_log_prob_);
        self.class_log_prior_ = Some(class_log_prior_);
        self.class_count_ = Some(class_count_);
        Ok(self)
    }
}

impl<F> CategoricalNB<F>
where
    F: Float + CubeElement + Pod,
{
    /// Per-query-row joint log-likelihood matrix (`n_query × n_classes`, host
    /// f64, row-major). Shared by `predict_labels` / `predict_proba` /
    /// `predict_log_proba`. Runs the geometry guard, then evaluates
    /// `class_log_prior_[c] + Σ_j feature_log_prob_[j][c, x[i,j]]` in host f64
    /// with the per-feature lookup index GUARDED against `n_categories_[j]`
    /// (T-11-04-02): an unseen / out-of-range category index `k ≥ n_categories_j`
    /// maps to the smoothed `log(alpha / denom_cj)` rather than indexing the
    /// ragged table out of bounds.
    fn joint_log_likelihood(
        &self,
        pool: &BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<f64>, AlgoError> {
        let (n_query, n_features) = shape;
        let feature_log_prob = self.feature_log_prob_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "categorical_nb",
            operation: "predict (call fit first)",
        })?;
        let n_categories = self.n_categories_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "categorical_nb",
            operation: "predict (call fit first)",
        })?;
        let class_log_prior = self.class_log_prior_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "categorical_nb",
            operation: "predict (call fit first)",
        })?;
        let class_count = self.class_count_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "categorical_nb",
            operation: "predict (call fit first)",
        })?;
        // Geometry guard BEFORE any host work (T-11-04 / ASVS V5).
        if n_query == 0 || n_features != self.n_features || x.len() != n_query * n_features {
            return Err(AlgoError::InvalidCategoricalInput {
                estimator: "categorical_nb",
                reason: format!(
                    "predict geometry: got {n_query}x{n_features}, fitted n_features={}",
                    self.n_features
                ),
            });
        }
        let n_classes = self.classes_.len();
        let x_h = x.to_host(pool);

        let alpha = self.alpha;
        let mut jll = vec![0.0f64; n_query * n_classes];
        for r in 0..n_query {
            for c in 0..n_classes {
                let mut acc = class_log_prior[c];
                for j in 0..n_features {
                    let n_cat_j = n_categories[j];
                    let flp_j = &feature_log_prob[j];
                    let xf = host_to_f64(x_h[r * n_features + j]);
                    let xr = xf.round();
                    // T-11-04-02: clamp the lookup index against n_categories_j. A
                    // negative / non-integer / out-of-range category is treated as
                    // UNSEEN (count == 0) → the smoothed log(alpha / denom_cj) where
                    // denom_cj = class_count[c] + alpha·n_cat_j — NOT an OOB ragged-
                    // table index. An in-range category indexes the fitted cell.
                    let k = if (xr - xf).abs() <= 1e-6 && xr >= 0.0 {
                        xr as usize
                    } else {
                        usize::MAX
                    };
                    let lp = if k < n_cat_j {
                        flp_j[c * n_cat_j + k]
                    } else {
                        let denom = class_count[c] + alpha * n_cat_j as f64;
                        (alpha / denom).ln()
                    };
                    acc += lp;
                }
                jll[r * n_classes + c] = acc;
            }
        }
        Ok(jll)
    }
}

impl<F> PredictLabels<F> for CategoricalNB<F>
where
    F: Float + CubeElement + Pod,
{
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        let jll = self.joint_log_likelihood(pool, x, shape)?;
        let labels = argmax_decode(&jll, &self.classes_);
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

impl<F> PredictProba<F> for CategoricalNB<F>
where
    F: Float + CubeElement + Pod,
{
    fn predict_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, _n_features) = shape;
        let jll = self.joint_log_likelihood(pool, x, shape)?;
        let n_classes = self.classes_.len();
        let mut proba: Vec<F> = vec![f64_to_host::<F>(0.0); n_query * n_classes];
        for r in 0..n_query {
            let row = &jll[r * n_classes..(r + 1) * n_classes];
            let (p, _lp) = log_sum_exp_normalize(row, n_classes);
            for (c, &pv) in p.iter().enumerate() {
                proba[r * n_classes + c] = f64_to_host::<F>(pv);
            }
        }
        Ok(DeviceArray::from_host(pool, &proba))
    }
}

impl<F> PredictLogProba<F> for CategoricalNB<F>
where
    F: Float + CubeElement + Pod,
{
    fn predict_log_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, _n_features) = shape;
        let jll = self.joint_log_likelihood(pool, x, shape)?;
        let n_classes = self.classes_.len();
        let mut log_proba: Vec<F> = vec![f64_to_host::<F>(0.0); n_query * n_classes];
        for r in 0..n_query {
            let row = &jll[r * n_classes..(r + 1) * n_classes];
            let (_p, lp) = log_sum_exp_normalize(row, n_classes);
            for (c, &lpv) in lp.iter().enumerate() {
                log_proba[r * n_classes + c] = f64_to_host::<F>(lpv);
            }
        }
        Ok(DeviceArray::from_host(pool, &log_proba))
    }
}
