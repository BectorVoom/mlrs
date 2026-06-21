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
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::naive_bayes::multinomial_nb::{
    decode_classes, resolve_class_log_prior, validate_discrete_alpha,
};
use crate::naive_bayes::nb_common::{argmax_decode, class_grouped_sum, log_sum_exp_normalize};
use crate::traits::{Fit, PredictLabels, PredictLogProba, PredictProba};

/// Bernoulli Naive Bayes (NB-03). Construct via [`BernoulliNB::builder`], then
/// [`Fit::fit`] + (Wave-1) the predict surface. Fitted `feature_log_prob_` /
/// `class_log_prior_` are device-resident / host f64 (D-03), `None` until `fit`.
pub struct BernoulliNB<F> {
    /// Additive smoothing (D-02 default `1.0`).
    alpha: f64,
    /// Keep `alpha` as-is when `< 1e-10` (D-02 default `true`); else clip (D-06).
    /// Retained as fitted-config provenance; the clip already applied at `build()`.
    #[allow(dead_code)]
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
    /// This is the GEMM operand `log p − log(1 − p)` (Pitfall 5), NOT the raw
    /// `log p` — the non-occurrence term is folded in so the device matvec is a
    /// single GEMM. The raw `log p` is recoverable but never needed at predict.
    feature_log_prob_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Per-class non-occurrence constant `Σ_j log(1 − p_cj)` (sklearn `neg_prob`
    /// row-sum), host f64, length `n_classes`, `None` until `fit`. Added to the
    /// joint LL bias alongside `class_log_prior_` (Pitfall 5).
    neg_prob_sum_: Option<Vec<f64>>,
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

    /// Host-materialized GEMM operand `log p − log(1 − p)` (`n_classes ×
    /// n_features` row-major), `None` until `fit`. NOTE this is the folded
    /// operand, not the raw `feature_log_prob_` (= `log p`).
    pub fn feature_log_prob_delta(&self, pool: &BufferPool<ActiveRuntime>) -> Option<Vec<f64>> {
        self.feature_log_prob_
            .as_ref()
            .map(|t| t.to_host(pool).iter().map(|&v| host_to_f64(v)).collect())
    }
}

/// Apply the D-04 binarization to a host f64 buffer: `Some(t)` maps `x > t → 1.0`
/// else `0.0`; `None` assumes the input is already binary and passes it through.
fn binarize_host(buf: &mut [f64], binarize: Option<f64>) {
    if let Some(t) = binarize {
        for v in buf.iter_mut() {
            *v = if *v > t { 1.0 } else { 0.0 };
        }
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
            neg_prob_sum_: None,
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

        let (classes_, class_of_row, n_classes) = decode_classes(pool, y, n_samples)?;

        // --- D-04 binarize: apply x>t → 1 on a host copy BEFORE the GATHER so the
        //     per-(class, feature) counts are occurrence counts. binarize=None
        //     assumes the input is already binary (pass-through). ---
        let mut x_bin: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        binarize_host(&mut x_bin, self.binarize);
        let x_bin_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(
            pool,
            &x_bin.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<F>>(),
        );

        // feature_count_[c,j] = Σ over class-c rows of binarized x[i,j].
        let feature_count =
            class_grouped_sum::<F>(pool, &x_bin_dev, shape, &class_of_row, n_classes)?;
        x_bin_dev.release_into(pool);

        let mut class_count_: Vec<f64> = vec![0.0; n_classes];
        for &c in &class_of_row {
            class_count_[c] += 1.0;
        }

        // --- feature_log_prob_[c,j] = log((count+alpha)/(class_count[c]+2·alpha))
        //     (Pitfall 4: the Bernoulli denominator smoothing is 2·alpha). The GEMM
        //     operand is the DELTA log p − log(1−p) and the per-class const
        //     Σ_j log(1−p_cj) becomes the bias (Pitfall 5). ---
        let alpha = self.alpha;
        let mut flp_delta: Vec<f64> = vec![0.0; n_classes * n_features];
        let mut neg_prob_sum: Vec<f64> = vec![0.0; n_classes];
        for c in 0..n_classes {
            let denom = class_count_[c] + 2.0 * alpha;
            for j in 0..n_features {
                let p = (feature_count[c][j] + alpha) / denom;
                let log_p = p.ln();
                let log_1mp = (1.0 - p).ln();
                flp_delta[c * n_features + j] = log_p - log_1mp;
                neg_prob_sum[c] += log_1mp;
            }
        }

        let class_log_prior_ = resolve_class_log_prior(
            "bernoulli_nb",
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
            &flp_delta.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<F>>(),
        );

        self.classes_ = classes_;
        self.n_features = n_features;
        self.feature_log_prob_ = Some(flp_dev);
        self.neg_prob_sum_ = Some(neg_prob_sum);
        self.class_log_prior_ = Some(class_log_prior_);
        Ok(self)
    }
}

impl<F> BernoulliNB<F>
where
    F: Float + CubeElement + Pod,
{
    /// Per-query-row joint log-likelihood (`n_query × n_classes`, host f64). The
    /// query X is binarized the SAME way as fit, then
    /// `LL[i,c] = class_log_prior_[c] + Σ_j log(1−p_cj)
    ///          + Σ_j x_ij·(log p_cj − log(1−p_cj))` — the Σ_j x·delta term is the
    /// device `gemm(X_bin @ flp_delta.T)` (Pitfall 5), the rest the per-class bias.
    fn joint_log_likelihood(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<f64>, AlgoError> {
        let (n_query, n_features) = shape;
        let flp = self.feature_log_prob_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "bernoulli_nb",
            operation: "predict (call fit first)",
        })?;
        let neg_prob_sum = self.neg_prob_sum_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "bernoulli_nb",
            operation: "predict (call fit first)",
        })?;
        let class_log_prior = self.class_log_prior_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "bernoulli_nb",
            operation: "predict (call fit first)",
        })?;
        if n_query == 0 || n_features != self.n_features || x.len() != n_query * n_features {
            return Err(AlgoError::InvalidLabels {
                estimator: "bernoulli_nb",
                reason: format!(
                    "predict geometry: got {n_query}x{n_features}, fitted n_features={}",
                    self.n_features
                ),
            });
        }
        let n_classes = self.classes_.len();

        // Binarize the query the same way as fit BEFORE the GEMM.
        let mut xq_bin: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        binarize_host(&mut xq_bin, self.binarize);
        let xq_bin_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(
            pool,
            &xq_bin.iter().map(|&v| f64_to_host::<F>(v)).collect::<Vec<F>>(),
        );

        let raw = gemm::<F>(
            pool,
            &xq_bin_dev,
            (n_query, n_features),
            flp,
            (n_features, n_classes),
            false,
            true,
            None,
        )?;
        let raw_host = raw.to_host(pool);
        raw.release_into(pool);
        xq_bin_dev.release_into(pool);

        let mut jll = vec![0.0f64; n_query * n_classes];
        for i in 0..n_query {
            for c in 0..n_classes {
                jll[i * n_classes + c] = class_log_prior[c]
                    + neg_prob_sum[c]
                    + host_to_f64(raw_host[i * n_classes + c]);
            }
        }
        Ok(jll)
    }
}

impl<F> PredictLabels<F> for BernoulliNB<F>
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

impl<F> PredictProba<F> for BernoulliNB<F>
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

impl<F> PredictLogProba<F> for BernoulliNB<F>
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
