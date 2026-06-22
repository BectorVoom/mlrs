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
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::naive_bayes::multinomial_nb::{
    decode_classes, resolve_class_log_prior, validate_discrete_alpha, validate_non_negative_counts,
};
use crate::naive_bayes::nb_common::{argmin_decode, class_grouped_sum, log_sum_exp_normalize};
use crate::traits::{Fit, PredictLabels, PredictLogProba, PredictProba};

/// Complement Naive Bayes (NB-04). Construct via [`ComplementNB::builder`], then
/// [`Fit::fit`] + (Wave-1) the predict surface (argmin decode internally, D-08).
/// Fitted `feature_log_prob_` / `class_log_prior_` are device-resident / host f64
/// (D-03), `None` until `fit`.
pub struct ComplementNB<F> {
    /// Additive smoothing (D-02 default `1.0`).
    alpha: f64,
    /// Keep `alpha` as-is when `< 1e-10` (D-02 default `true`); else clip (D-06).
    /// Retained as fitted-config provenance (exposed via [`ComplementNB::force_alpha`]);
    /// the clip already applied at `build()` (WR-08).
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

    /// The stored `force_alpha` config provenance (WR-08). The D-06 alpha clip is
    /// already applied at `build()`; this exposes whether the clip was suppressed.
    pub fn force_alpha(&self) -> bool {
        self.force_alpha
    }

    /// The per-class log-prior (`None` until `fit`).
    pub fn class_log_prior(&self) -> Option<&[f64]> {
        self.class_log_prior_.as_deref()
    }

    /// Host-materialized complement-weighted `feature_log_prob_` (`n_classes ×
    /// n_features` row-major), `None` until `fit`. This is the sklearn weights
    /// (`-logged`, or `logged/summed` under `norm`).
    pub fn feature_log_prob(&self, pool: &BufferPool<ActiveRuntime>) -> Option<Vec<f64>> {
        self.feature_log_prob_
            .as_ref()
            .map(|t| t.to_host(pool).iter().map(|&v| host_to_f64(v)).collect())
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

        // CR-01 / T-11-02: validate X is a finite, non-negative count matrix BEFORE
        // it reaches `(cc / comp_sum).ln()` (sklearn's `check_non_negative` parity;
        // a negative count drives comp_sum / the log to NaN/-inf silently).
        let x_host: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        validate_non_negative_counts("complement_nb", &x_host)?;

        let (classes_, class_of_row, n_classes) = decode_classes("complement_nb", pool, y, n_samples)?;

        // feature_count_[c,j] via the validated GATHER.
        let feature_count = class_grouped_sum::<F>(pool, x, shape, &class_of_row, n_classes)?;

        let mut class_count_: Vec<f64> = vec![0.0; n_classes];
        for &c in &class_of_row {
            class_count_[c] += 1.0;
        }

        // --- ComplementNB weights (Pitfall 6 — DIFFERENT formula from
        //     MultinomialNB; do NOT copy it). feature_all_[j] = Σ_c count[c,j];
        //     comp_count[c,j] = feature_all_[j] + alpha − count[c,j] (sklearn folds
        //     the +alpha per-element so the row denominator already carries the
        //     alpha·n_features smoothing); logged[c,j] = log(comp_count[c,j] /
        //     Σ_j comp_count[c,j]). The stored feature_log_prob_ is sklearn's exact
        //     weights: `-logged` (default) or `logged / Σ_j logged` (norm). ---
        let alpha = self.alpha;
        let mut feature_all: Vec<f64> = vec![0.0; n_features];
        for c in 0..n_classes {
            for j in 0..n_features {
                feature_all[j] += feature_count[c][j];
            }
        }

        let mut flp: Vec<f64> = vec![0.0; n_classes * n_features];
        for c in 0..n_classes {
            // comp_count row and its sum (per-element +alpha already folded in).
            let comp: Vec<f64> = (0..n_features)
                .map(|j| feature_all[j] + alpha - feature_count[c][j])
                .collect();
            let comp_sum: f64 = comp.iter().sum();
            let logged: Vec<f64> = comp.iter().map(|&cc| (cc / comp_sum).ln()).collect();
            if self.norm {
                // Second L1 normalization: feature_log_prob_ = logged / Σ_j logged.
                let summed: f64 = logged.iter().sum();
                for j in 0..n_features {
                    flp[c * n_features + j] = logged[j] / summed;
                }
            } else {
                // feature_log_prob_ = −logged (the complement weights).
                for j in 0..n_features {
                    flp[c * n_features + j] = -logged[j];
                }
            }
        }

        // class_log_prior_ resolved as the discrete sibling (only used in the
        // single-class edge case at predict, but kept for the accessor surface).
        let class_log_prior_ = resolve_class_log_prior(
            "complement_nb",
            self.fit_prior,
            &self.class_prior,
            &class_count_,
            n_classes,
        )?;

        if let Some(old) = self.feature_log_prob_.take() {
            old.release_into(pool);
        }
        let flp_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(
            pool,
            &flp.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<F>>(),
        );

        self.classes_ = classes_;
        self.n_features = n_features;
        self.feature_log_prob_ = Some(flp_dev);
        self.class_log_prior_ = Some(class_log_prior_);
        Ok(self)
    }
}

impl<F> ComplementNB<F>
where
    F: Float + CubeElement + Pod,
{
    /// Per-query-row joint log-likelihood (`n_query × n_classes`, host f64) =
    /// `X @ feature_log_prob_.T` (+ `class_log_prior_` only in the single-class
    /// edge case, per sklearn). The device matvec is `gemm` (transb=true) over the
    /// stored `(n_classes, n_features)` weights. Labels decode with `argmin` over
    /// `−jll` (D-08 — argmax over feature_log_prob_ == argmin over `−`), proba
    /// log-sum-exp-normalizes `jll` directly (sklearn `predict_proba` convention).
    fn joint_log_likelihood(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<f64>, AlgoError> {
        let (n_query, n_features) = shape;
        let flp = self.feature_log_prob_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "complement_nb",
            operation: "predict (call fit first)",
        })?;
        let class_log_prior = self.class_log_prior_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "complement_nb",
            operation: "predict (call fit first)",
        })?;
        if n_query == 0 || n_features != self.n_features || x.len() != n_query * n_features {
            return Err(AlgoError::InvalidLabels {
                estimator: "complement_nb",
                reason: format!(
                    "predict geometry: got {n_query}x{n_features}, fitted n_features={}",
                    self.n_features
                ),
            });
        }
        // CR-01 / T-11-02: a negative / NaN query row is equally invalid for the
        // count model — reject it before the GEMM (sklearn rejects at predict too).
        let x_host: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        validate_non_negative_counts("complement_nb", &x_host)?;
        let n_classes = self.classes_.len();
        let raw = gemm::<F>(
            pool,
            x,
            (n_query, n_features),
            flp,
            (n_features, n_classes),
            false,
            true,
            None,
        )?;
        let raw_host = raw.to_host(pool);
        raw.release_into(pool);

        // sklearn adds class_log_prior_ only when there is a single class.
        let single = n_classes == 1;
        let mut jll = vec![0.0f64; n_query * n_classes];
        for i in 0..n_query {
            for c in 0..n_classes {
                let mut v = host_to_f64(raw_host[i * n_classes + c]);
                if single {
                    v += class_log_prior[c];
                }
                jll[i * n_classes + c] = v;
            }
        }
        Ok(jll)
    }
}

impl<F> PredictLabels<F> for ComplementNB<F>
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
        // sklearn predicts argmax over feature_log_prob_; that equals argmin over
        // the negated jll (D-08 — the ComplementNB internal argmin convention).
        let neg: Vec<f64> = jll.iter().map(|&v| -v).collect();
        let labels = argmin_decode(&neg, &self.classes_);
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

impl<F> PredictProba<F> for ComplementNB<F>
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

impl<F> PredictLogProba<F> for ComplementNB<F>
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
