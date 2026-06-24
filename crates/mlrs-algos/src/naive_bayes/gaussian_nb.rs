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

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::naive_bayes::nb_common::{
    argmax_decode, class_grouped_sum, class_grouped_sumsq, empirical_class_log_prior,
    log_sum_exp_normalize, NB_LABEL_INT_TOL,
};
// Phase 16 (D-02 shape-B trait-swap): the pre-existing builder is UNTOUCHED; the
// estimator gains the `<F, S = Unfit>` state param and migrates from the legacy
// trait surface to the consuming-self `typestate` surface. fit/predict math is
// BYTE-IDENTICAL (D-03).
use crate::typestate::{
    validate_geometry, Fit, Fitted, PredictLabels, PredictLogProba, PredictProba, Unfit,
};

/// `ln(2π)`, the constant term of the Gaussian log-likelihood
/// `−0.5·Σ_j[log(2π·var) + (x−mean)²/var]` factored as
/// `−0.5·Σ_j[ln(2π) + ln(var) + (x−mean)²/var]`.
const LN_2PI: f64 = 1.837_877_066_409_345_6; // (2.0 * std::f64::consts::PI).ln()

/// Gaussian Naive Bayes (NB-01). Construct via [`GaussianNB::builder`], then
/// [`Fit::fit`] + (Wave-1) `predict_labels` / `predict_proba` /
/// `predict_log_proba`. Fitted `theta_` (means) / `var_` (variances) /
/// `class_prior_` are device-resident / host f64 small tensors (D-03), `None`
/// until `fit`.
pub struct GaussianNB<F, S = Unfit> {
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
    /// Per-class sample counts (host f64, length `n_classes`), `None` until
    /// `fit`. The empirical-prior numerator and the `theta_`/`var_` divisor.
    class_count_: Option<Vec<f64>>,
    /// `var_smoothing · max_j Var(X[:,j])` over the WHOLE training set
    /// (population, ddof=0) — the GLOBAL floor added to every `var_` cell
    /// (Pitfall 3), `None` until `fit`.
    epsilon_: Option<f64>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> GaussianNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `GaussianNB` with sklearn's defaults (D-02).
    pub fn builder() -> GaussianNBBuilder {
        GaussianNBBuilder::default()
    }
}

impl<F> GaussianNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
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

    /// The global variance floor `epsilon_` (`None` until `fit`).
    pub fn epsilon(&self) -> Option<f64> {
        self.epsilon_
    }

    /// Host-materialized per-class feature means `theta_` (`n_classes × n_features`
    /// row-major), `None` until `fit`.
    pub fn theta(&self, pool: &BufferPool<ActiveRuntime>) -> Option<Vec<f64>> {
        self.theta_
            .as_ref()
            .map(|t| t.to_host(pool).iter().map(|&v| host_to_f64(v)).collect())
    }

    /// Host-materialized per-class feature variances `var_` (`n_classes ×
    /// n_features` row-major, epsilon_-floored), `None` until `fit`.
    pub fn var(&self, pool: &BufferPool<ActiveRuntime>) -> Option<Vec<f64>> {
        self.var_
            .as_ref()
            .map(|t| t.to_host(pool).iter().map(|&v| host_to_f64(v)).collect())
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
    pub fn build<F>(self) -> Result<GaussianNB<F, Unfit>, BuildError>
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
            class_count_: None,
            epsilon_: None,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for GaussianNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = GaussianNB<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<GaussianNB<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        // Data-DEPENDENT geometry guard BEFORE any launch (T-11-02 / ASVS V5).
        validate_geometry(x, shape)?;
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
        // --- Pitfall 4: host distinct-sorted classes_ (no class-count
        //     restriction; GaussianNB is multiclass). Integer labels only. ---
        let y_host = y.to_host(pool);
        let mut raw_labels: Vec<i64> = Vec::with_capacity(n_samples);
        for &yv in y_host.iter() {
            let lf = host_to_f64(yv);
            let li = lf.round();
            if (li - lf).abs() > NB_LABEL_INT_TOL {
                return Err(AlgoError::InvalidLabels {
                    estimator: "gaussian_nb",
                    reason: format!("labels must be integers (got {lf})"),
                });
            }
            raw_labels.push(li as i64);
        }
        let mut classes_: Vec<i64> = raw_labels.clone();
        classes_.sort_unstable();
        classes_.dedup();
        let n_classes = classes_.len();
        // WR-02: predicted labels are emitted as `i32`; a class id outside i32
        // range would be SILENTLY truncated into a wrong label. Validate here.
        for &cls in classes_.iter() {
            if i32::try_from(cls).is_err() {
                return Err(AlgoError::InvalidLabels {
                    estimator: "gaussian_nb",
                    reason: format!(
                        "class label {cls} does not fit in i32 (predicted labels are i32)"
                    ),
                });
            }
        }
        // Dense class index per row (position in the sorted classes_ table).
        let class_of_row: Vec<usize> = raw_labels
            .iter()
            .map(|l| classes_.binary_search(l).expect("label is in classes_"))
            .collect();

        // --- Per-class sufficient statistics via the validated GATHER prims
        //     (one owner per (class, feature); NO scatter-add, NO new kernel). ---
        let sums = class_grouped_sum::<F>(pool, x, shape, &class_of_row, n_classes)?;
        let sumsqs = class_grouped_sumsq::<F>(pool, x, shape, &class_of_row, n_classes)?;

        // class_count_[c] = #rows of class c (every observed class has >= 1).
        let mut class_count_: Vec<f64> = vec![0.0; n_classes];
        for &c in &class_of_row {
            class_count_[c] += 1.0;
        }

        // theta_[c,j] = sum/n_c ; var_[c,j] = sumsq/n_c − theta_² (population,
        // ddof=0). epsilon_ is added below (a single global floor).
        let mut theta: Vec<f64> = vec![0.0; n_classes * n_features];
        let mut var: Vec<f64> = vec![0.0; n_classes * n_features];
        for c in 0..n_classes {
            let n_c = class_count_[c];
            debug_assert!(n_c > 0.0, "every observed class has at least one sample");
            for j in 0..n_features {
                let mean = sums[c][j] / n_c;
                let raw_var = sumsqs[c][j] / n_c - mean * mean;
                theta[c * n_features + j] = mean;
                // Population variance can dip slightly negative from f64 round-off
                // when the true variance is ~0 (e.g. a constant feature); clamp to
                // 0 before the epsilon_ floor so var_ stays >= epsilon_ > 0.
                var[c * n_features + j] = raw_var.max(0.0);
            }
        }

        // --- epsilon_ = var_smoothing · max_j Var(X[:,j]) over the WHOLE X
        //     (population, ddof=0), computed ONCE — GLOBAL, not per-class
        //     (Pitfall 3, FEATURES.md `var_smoothing * X.var(axis=0).max()`). ---
        let x_host = x.to_host(pool);
        let n = n_samples as f64;
        let mut max_col_var = 0.0f64;
        for j in 0..n_features {
            let mut s = 0.0f64;
            let mut ss = 0.0f64;
            for i in 0..n_samples {
                let v = host_to_f64(x_host[i * n_features + j]);
                s += v;
                ss += v * v;
            }
            let mean = s / n;
            let col_var = (ss / n - mean * mean).max(0.0);
            if col_var > max_col_var {
                max_col_var = col_var;
            }
        }
        // WR-05: floor epsilon_ to a tiny positive minimum so an all-constant
        // feature matrix (every column variance ~0 → epsilon_ == 0) cannot leave a
        // var_ cell at 0 and divide-by-zero the predict quadratic `(d*d)/v`
        // (→ +inf/NaN joint-LL in f32). The `raw_var.max(0.0)` clamp above guards
        // negatives but not this degenerate; flooring epsilon_ keeps var_ > 0.
        let epsilon_ = (self.var_smoothing * max_col_var).max(f64::MIN_POSITIVE);
        for cell in var.iter_mut() {
            *cell += epsilon_;
        }

        // --- class_log_prior_: empirical log(count_c / n) when priors=None, else
        //     the supplied priors (length == n_classes, validated as a
        //     data-DEPENDENT check here per D-05). ---
        let class_log_prior_ = match &self.priors {
            None => empirical_class_log_prior(&class_count_),
            Some(p) => {
                if p.len() != n_classes {
                    return Err(AlgoError::InvalidLabels {
                        estimator: "gaussian_nb",
                        reason: format!(
                            "priors length {} != number of classes {n_classes}",
                            p.len()
                        ),
                    });
                }
                // WR-01: sklearn requires the priors to sum to 1
                // (`ValueError("The sum of the priors should be 1.")`); a
                // non-normalized prior is silently `.ln()`-mapped here and the
                // log-sum-exp renormalization masks the divergence from the oracle.
                let prior_sum: f64 = p.iter().sum();
                if (prior_sum - 1.0).abs() > 1e-6 {
                    return Err(AlgoError::InvalidLabels {
                        estimator: "gaussian_nb",
                        reason: format!("the sum of the priors should be 1 (got {prior_sum})"),
                    });
                }
                p.iter().map(|&v| v.ln()).collect()
            }
        };

        // Store fitted state. theta_/var_ are device-resident (per the stub's
        // field types); the host materializes them at predict / accessor. The
        // consuming-self transition means there is no prior fitted state to
        // release — a freshly-built `Unfit` carries theta_/var_ = None, so the old
        // WR-07 re-fit buffer-release pass is vacuous and dropped (the
        // KernelDensity/IncrementalPCA precedent, 16-07/16-04); buffer reuse across
        // re-CONSTRUCT+fit cycles still flows through the pool free-list.
        let theta_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(
            pool,
            &theta.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<F>>(),
        );
        let var_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(
            pool,
            &var.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<F>>(),
        );

        Ok(GaussianNB {
            priors: self.priors,
            var_smoothing: self.var_smoothing,
            classes_,
            n_features,
            theta_: Some(theta_dev),
            var_: Some(var_dev),
            class_log_prior_: Some(class_log_prior_),
            class_count_: Some(class_count_),
            epsilon_: Some(epsilon_),
            _state: PhantomData,
        })
    }
}

impl<F> GaussianNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Per-query-row joint log-likelihood matrix (`n_query × n_classes`, host
    /// f64, row-major). Shared by `predict_labels` / `predict_proba` /
    /// `predict_log_proba`. Runs the geometry guard, then evaluates
    /// `class_log_prior_[c] − 0.5·Σ_j[ln(2π·var_[c,j]) + (x_j−theta_[c,j])²/var_[c,j]]`
    /// in host f64 (the var_ cells are epsilon_-floored, so no div-by-zero).
    fn joint_log_likelihood(
        &self,
        pool: &BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<f64>, AlgoError> {
        let (n_query, n_features) = shape;
        let theta = self.theta_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "gaussian_nb",
            operation: "predict (call fit first)",
        })?;
        let var = self.var_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "gaussian_nb",
            operation: "predict (call fit first)",
        })?;
        let class_log_prior = self.class_log_prior_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "gaussian_nb",
            operation: "predict (call fit first)",
        })?;
        // Geometry guard BEFORE any host work (T-11-02-04 / ASVS V5).
        if n_query == 0 || n_features != self.n_features || x.len() != n_query * n_features {
            return Err(AlgoError::InvalidLabels {
                estimator: "gaussian_nb",
                reason: format!(
                    "predict geometry: got {n_query}x{n_features}, fitted n_features={}",
                    self.n_features
                ),
            });
        }
        let n_classes = self.classes_.len();
        let theta_h: Vec<f64> = theta.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        let var_h: Vec<f64> = var.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        let x_h = x.to_host(pool);

        let mut jll = vec![0.0f64; n_query * n_classes];
        for r in 0..n_query {
            for c in 0..n_classes {
                let mut acc = class_log_prior[c];
                let mut quad = 0.0f64;
                for j in 0..n_features {
                    let cj = c * n_features + j;
                    let xv = host_to_f64(x_h[r * n_features + j]);
                    let v = var_h[cj];
                    let d = xv - theta_h[cj];
                    quad += (LN_2PI + v.ln()) + (d * d) / v;
                }
                acc -= 0.5 * quad;
                jll[r * n_classes + c] = acc;
            }
        }
        Ok(jll)
    }
}

impl<F> PredictLabels<F> for GaussianNB<F, Fitted>
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

impl<F> PredictProba<F> for GaussianNB<F, Fitted>
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

impl<F> PredictLogProba<F> for GaussianNB<F, Fitted>
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
