//! `MBSGDClassifier` (SGDSVM-01) — minibatch-SGD linear classifier, ≈
//! `sklearn.linear_model.SGDClassifier`.
//!
//! Phase-10 Wave-0 scaffold (plan 10-01): the struct, the
//! [`MBSGDClassifierBuilder`] (D-01/D-03 — sklearn-default field initializers),
//! and the `build() -> Result<MBSGDClassifier<F>, BuildError>` SIGNATURE are
//! final now; the validation predicates and the `fit`/`predict` bodies land in
//! the Wave-1/Wave-3 plans. The closest analog is `logistic.rs` (classifier:
//! `classes_` remap + `PredictLabels` + `PredictProba`); the construction surface
//! switches from `new()`/`with_opts()` to the builder (D-01).
//!
//! Tests live in `crates/mlrs-algos/tests/mbsgd_classifier_test.rs`
//! (AGENTS.md §2), never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::runtime::ActiveRuntime;

use crate::error::BuildError;
use crate::linear::sgd_config::{LearningRate, Loss, Penalty, SgdConfig};

/// Minibatch-SGD linear classifier (SGDSVM-01). Construct via
/// [`MBSGDClassifier::builder`].
///
/// Wave-0 scaffold: the fitted-state fields (`n_features` / `coef_` /
/// `intercept_`) are written by the Wave-1 `fit` body, hence `allow(dead_code)`
/// until then.
#[allow(dead_code)]
pub struct MBSGDClassifier<F> {
    /// The lowered, validated hyperparameter bundle (D-06).
    config: SgdConfig,
    /// DISTINCT sorted class labels inferred at `fit` (Pitfall 4 — ±1 encoding).
    classes_: Vec<i64>,
    /// Feature count inferred at `fit`.
    n_features: usize,
    /// Fitted coefficients (device-resident), `None` until `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (device-resident), `None` until `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
}

impl<F> MBSGDClassifier<F>
where
    F: Float + CubeElement + Pod,
{
    /// Start building an `MBSGDClassifier` with sklearn's `SGDClassifier`
    /// defaults (D-03).
    pub fn builder() -> MBSGDClassifierBuilder {
        MBSGDClassifierBuilder::default()
    }

    /// The lowered configuration (D-06).
    pub fn config(&self) -> &SgdConfig {
        &self.config
    }

    /// The inferred class labels (empty until `fit`).
    pub fn classes(&self) -> &[i64] {
        &self.classes_
    }
}

/// Builder for [`MBSGDClassifier`] (D-01). Default field initializers encode the
/// sklearn `SGDClassifier` defaults (D-03): `loss=hinge`, `penalty=l2`,
/// `alpha=1e-4`, `l1_ratio=0.15`, `max_iter=1000`, `tol=1e-3`,
/// `learning_rate=optimal`, `eta0=0.01`, `power_t=0.5`.
#[derive(Debug, Clone, Copy)]
pub struct MBSGDClassifierBuilder {
    loss: Loss,
    penalty: Penalty,
    alpha: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    learning_rate: LearningRate,
    eta0: f64,
    power_t: f64,
    batch_size: usize,
    shuffle: bool,
    seed: u64,
}

impl Default for MBSGDClassifierBuilder {
    fn default() -> Self {
        Self {
            loss: Loss::Hinge,
            penalty: Penalty::L2,
            alpha: 1e-4,
            l1_ratio: 0.15,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-3,
            learning_rate: LearningRate::Optimal,
            eta0: 0.01,
            power_t: 0.5,
            batch_size: 1,
            shuffle: true,
            seed: 0,
        }
    }
}

impl MBSGDClassifierBuilder {
    /// Set the loss family.
    pub fn loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }
    /// Set the penalty family.
    pub fn penalty(mut self, penalty: Penalty) -> Self {
        self.penalty = penalty;
        self
    }
    /// Set the penalty strength `alpha`.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
    /// Set the ElasticNet mixing `l1_ratio`.
    pub fn l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }
    /// Set whether to fit an intercept.
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
    /// Set the epoch cap.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    /// Set the stopping tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    /// Set the learning-rate schedule.
    pub fn learning_rate(mut self, learning_rate: LearningRate) -> Self {
        self.learning_rate = learning_rate;
        self
    }
    /// Set the initial learning rate `eta0`.
    pub fn eta0(mut self, eta0: f64) -> Self {
        self.eta0 = eta0;
        self
    }
    /// Set the inverse-scaling exponent `power_t`.
    pub fn power_t(mut self, power_t: f64) -> Self {
        self.power_t = power_t;
        self
    }
    /// Set the minibatch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    /// Set whether to shuffle each epoch.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
    /// Set the RNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Build the estimator, validating the data-INDEPENDENT hyperparameters
    /// (D-08). The SIGNATURE is final (Wave-0); the validation predicates land in
    /// Wave-1. On success the lowered [`SgdConfig`] is stored and the fitted
    /// state is `None`.
    pub fn build<F>(self) -> Result<MBSGDClassifier<F>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        // Wave-1 fills the `alpha >= 0` / `l1_ratio ∈ [0,1]` / `eta0 > 0` /
        // valid-loss-for-classifier checks here, returning the matching
        // `BuildError`. The Wave-0 stub lowers the (default-valid) params.
        let config = SgdConfig {
            loss: self.loss,
            penalty: self.penalty,
            alpha: self.alpha,
            l1_ratio: self.l1_ratio,
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
            learning_rate: self.learning_rate,
            eta0: self.eta0,
            power_t: self.power_t,
            epsilon: 0.0,
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            seed: self.seed,
        };
        Ok(MBSGDClassifier {
            config,
            classes_: Vec::new(),
            n_features: 0,
            coef_: None,
            intercept_: None,
        })
    }
}
