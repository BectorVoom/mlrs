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

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::naive_bayes::nb_common::{
    argmax_decode, class_grouped_sum, empirical_class_log_prior, log_sum_exp_normalize,
    NB_LABEL_INT_TOL,
};
// Phase 16 (D-02 shape-B trait-swap): builder UNTOUCHED; `<F, S = Unfit>` state
// param + migration to the consuming-self `typestate` surface. fit/predict math
// BYTE-IDENTICAL (D-03).
use crate::typestate::{
    validate_geometry, Fit, Fitted, PredictLabels, PredictLogProba, PredictProba, Unfit,
};

/// Multinomial Naive Bayes (NB-02). Construct via [`MultinomialNB::builder`],
/// then [`Fit::fit`] + (Wave-1) the predict surface. Fitted `feature_log_prob_`
/// (`n_classes × n_features`) / `class_log_prior_` are device-resident / host f64
/// (D-03), `None` until `fit`.
pub struct MultinomialNB<F, S = Unfit> {
    /// Additive (Laplace/Lidstone) smoothing (D-02 default `1.0`).
    alpha: f64,
    /// Keep `alpha` as-is even when `< 1e-10` (D-02 default `true`); when `false`
    /// a tiny `alpha` is clipped to `1e-10` at `build()` with a warning (D-06).
    /// Retained as fitted-config provenance (exposed via [`MultinomialNB::force_alpha`]);
    /// the clip already applied at `build()` (WR-08).
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
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> MultinomialNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `MultinomialNB` with sklearn's defaults (D-02).
    pub fn builder() -> MultinomialNBBuilder {
        MultinomialNBBuilder::default()
    }
}

impl<F> MultinomialNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
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

    /// Host-materialized `feature_log_prob_` (`n_classes × n_features` row-major),
    /// `None` until `fit`.
    pub fn feature_log_prob(&self, pool: &BufferPool<ActiveRuntime>) -> Option<Vec<f64>> {
        self.feature_log_prob_
            .as_ref()
            .map(|t| t.to_host(pool).iter().map(|&v| host_to_f64(v)).collect())
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
    pub fn build<F>(self) -> Result<MultinomialNB<F, Unfit>, BuildError>
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
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for MultinomialNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = MultinomialNB<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<MultinomialNB<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        validate_geometry(x, shape)?;
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

        // --- CR-01 / T-11-02: validate X is a finite, non-negative count matrix
        //     BEFORE the GATHER reaches `((count + alpha) / denom).ln()` (sklearn's
        //     `check_non_negative` parity; a negative / NaN count otherwise yields a
        //     silent NaN feature_log_prob_). ---
        let x_host: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        validate_non_negative_counts("multinomial_nb", &x_host)?;

        // --- host distinct-sorted classes_ (multiclass, integer labels only, i32
        //     range guarded — predicted labels are emitted as i32, WR-02). ---
        let (classes_, class_of_row, n_classes) = decode_classes("multinomial_nb", pool, y, n_samples)?;

        // --- feature_count_[c,j] via the validated GATHER (one owner per
        //     (class, feature); the counts are accumulated in host f64). ---
        let feature_count = class_grouped_sum::<F>(pool, x, shape, &class_of_row, n_classes)?;

        // class_count_[c] = #rows of class c.
        let mut class_count_: Vec<f64> = vec![0.0; n_classes];
        for &c in &class_of_row {
            class_count_[c] += 1.0;
        }

        // --- feature_log_prob_[c,j] = log((count[c,j] + alpha) /
        //     (Σ_j count[c,j] + alpha·n_features)) (Pitfall 4: the denominator
        //     smoothing is alpha·n_features, NOT alpha·1). ---
        let alpha = self.alpha;
        let mut flp: Vec<f64> = vec![0.0; n_classes * n_features];
        for c in 0..n_classes {
            let row_total: f64 = feature_count[c].iter().sum();
            let denom = row_total + alpha * n_features as f64;
            for j in 0..n_features {
                flp[c * n_features + j] = ((feature_count[c][j] + alpha) / denom).ln();
            }
        }

        // --- class_log_prior_: empirical log(count_c / n) when fit_prior=true &
        //     class_prior=None; supplied class_prior (validated length); else a
        //     uniform prior when fit_prior=false (D-05 data-dependent check). ---
        let class_log_prior_ =
            resolve_class_log_prior("multinomial_nb", self.fit_prior, &self.class_prior, &class_count_, n_classes)?;

        // The consuming-self transition carries no prior fitted state — a fresh
        // `Unfit` has feature_log_prob_ = None, so the old WR-07 re-fit release is
        // vacuous and dropped (the KernelDensity/IncrementalPCA precedent); buffer
        // reuse across re-CONSTRUCT+fit cycles still flows via the pool free-list.
        let flp_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(
            pool,
            &flp.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<F>>(),
        );

        Ok(MultinomialNB {
            alpha: self.alpha,
            force_alpha: self.force_alpha,
            fit_prior: self.fit_prior,
            class_prior: self.class_prior,
            classes_,
            n_features,
            feature_log_prob_: Some(flp_dev),
            class_log_prior_: Some(class_log_prior_),
            _state: PhantomData,
        })
    }
}

impl<F> MultinomialNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Per-query-row joint log-likelihood matrix (`n_query × n_classes`, host f64,
    /// row-major). Shared by the three predict surfaces. Runs the geometry guard,
    /// computes `X @ feature_log_prob_.T` on the device via `gemm` (transb=true:
    /// the stored `(n_classes, n_features)` buffer is read as its transpose), then
    /// host-adds the `class_log_prior_[c]` bias.
    fn joint_log_likelihood(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<f64>, AlgoError> {
        let (n_query, n_features) = shape;
        let flp = self.feature_log_prob_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "multinomial_nb",
            operation: "predict (call fit first)",
        })?;
        let class_log_prior = self.class_log_prior_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "multinomial_nb",
            operation: "predict (call fit first)",
        })?;
        if n_query == 0 || n_features != self.n_features || x.len() != n_query * n_features {
            return Err(AlgoError::InvalidLabels {
                estimator: "multinomial_nb",
                reason: format!(
                    "predict geometry: got {n_query}x{n_features}, fitted n_features={}",
                    self.n_features
                ),
            });
        }
        // CR-01 / T-11-02: a negative / NaN query row is equally invalid for the
        // count model — reject it before the GEMM (sklearn rejects at predict too).
        let x_host: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        validate_non_negative_counts("multinomial_nb", &x_host)?;
        let n_classes = self.classes_.len();
        // raw[i,c] = Σ_j X[i,j] · flp[c,j] = (X @ flp.T)[i,c]. The stored flp buffer
        // is (n_classes, n_features); transb=true reads it as (n_features, n_classes).
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

        let mut jll = vec![0.0f64; n_query * n_classes];
        for i in 0..n_query {
            for c in 0..n_classes {
                jll[i * n_classes + c] =
                    class_log_prior[c] + host_to_f64(raw_host[i * n_classes + c]);
            }
        }
        Ok(jll)
    }
}

impl<F> PredictLabels<F> for MultinomialNB<F, Fitted>
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

impl<F> PredictProba<F> for MultinomialNB<F, Fitted>
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

impl<F> PredictLogProba<F> for MultinomialNB<F, Fitted>
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

/// Shared non-negativity / finiteness guard for the COUNT-based discrete NB
/// variants (Multinomial / Complement / Bernoulli, CR-01 / T-11-02). sklearn
/// rejects a negative or non-finite count matrix with
/// `check_non_negative(X, …) -> ValueError`; a negative count here flows
/// straight into `((count + alpha) / denom).ln()` (or ComplementNB's
/// `(cc / comp_sum).ln()`), producing a silent `NaN`/`-inf` `feature_log_prob_`
/// that corrupts every `predict`/`predict_proba` row with no error surfaced. A
/// `NaN` input is equally unguarded. This validates the host-read matrix `x`
/// (already on host at `fit`/`predict`) BEFORE it reaches the smoothed-log
/// formulas, mirroring sklearn's contract (D-09). `pub(crate)` so the sibling
/// count-based fits/predicts reuse it without a base struct (D-03).
pub(crate) fn validate_non_negative_counts(
    estimator: &'static str,
    x_host: &[f64],
) -> Result<(), AlgoError> {
    for &v in x_host {
        if !v.is_finite() || v < 0.0 {
            return Err(AlgoError::InvalidLabels {
                estimator,
                reason: format!("input X must be finite and non-negative (got {v})"),
            });
        }
    }
    Ok(())
}

/// Shared label decode for the discrete NB variants (D-03 — function-level
/// sharing): read `y` to host, validate integer labels in i32 range (WR-02 —
/// predicted labels are emitted as i32), and return the distinct-sorted
/// `classes_`, the dense per-row class index, and `n_classes`. `pub(crate)` so
/// the sibling discrete fits reuse it without a base struct. `estimator` is the
/// caller's name, surfaced in the user-facing label errors (IN-03 — no leaking
/// the internal `"discrete_nb"` helper name).
pub(crate) fn decode_classes<F>(
    estimator: &'static str,
    pool: &BufferPool<ActiveRuntime>,
    y: &DeviceArray<ActiveRuntime, F>,
    n_samples: usize,
) -> Result<(Vec<i64>, Vec<usize>, usize), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let y_host = y.to_host(pool);
    let mut raw_labels: Vec<i64> = Vec::with_capacity(n_samples);
    for &yv in y_host.iter() {
        let lf = host_to_f64(yv);
        let li = lf.round();
        if (li - lf).abs() > NB_LABEL_INT_TOL {
            // IN-03: name the concrete caller estimator (not the internal helper
            // name) in the user-facing error.
            return Err(AlgoError::InvalidLabels {
                estimator,
                reason: format!("labels must be integers (got {lf})"),
            });
        }
        raw_labels.push(li as i64);
    }
    let mut classes_: Vec<i64> = raw_labels.clone();
    classes_.sort_unstable();
    classes_.dedup();
    for &cls in classes_.iter() {
        if i32::try_from(cls).is_err() {
            return Err(AlgoError::InvalidLabels {
                estimator,
                reason: format!("class label {cls} does not fit in i32 (predicted labels are i32)"),
            });
        }
    }
    let n_classes = classes_.len();
    let class_of_row: Vec<usize> = raw_labels
        .iter()
        .map(|l| classes_.binary_search(l).expect("label is in classes_"))
        .collect();
    Ok((classes_, class_of_row, n_classes))
}

/// Shared `class_log_prior_` resolution for the discrete NB variants (D-03):
/// supplied `class_prior` (validated length == n_classes) takes precedence; else
/// the empirical `log(count_c / n)` when `fit_prior == true`; else a uniform
/// `log(1/n_classes)` prior when `fit_prior == false` (sklearn semantics).
pub(crate) fn resolve_class_log_prior(
    estimator: &'static str,
    fit_prior: bool,
    class_prior: &Option<Vec<f64>>,
    class_count_: &[f64],
    n_classes: usize,
) -> Result<Vec<f64>, AlgoError> {
    if let Some(p) = class_prior {
        if p.len() != n_classes {
            return Err(AlgoError::InvalidLabels {
                estimator,
                reason: format!("class_prior length {} != number of classes {n_classes}", p.len()),
            });
        }
        // WR-01: sklearn requires a normalized `class_prior` (sum to 1); a
        // non-normalized prior is otherwise silently `.ln()`-mapped here and the
        // log-sum-exp renormalization at predict masks the oracle divergence.
        let prior_sum: f64 = p.iter().sum();
        if (prior_sum - 1.0).abs() > 1e-6 {
            return Err(AlgoError::InvalidLabels {
                estimator,
                reason: format!("the sum of the priors should be 1 (got {prior_sum})"),
            });
        }
        return Ok(p.iter().map(|&v| v.ln()).collect());
    }
    if fit_prior {
        Ok(empirical_class_log_prior(class_count_))
    } else {
        let uniform = (1.0 / n_classes as f64).ln();
        Ok(vec![uniform; n_classes])
    }
}
