//! `MBSGDRegressor` (SGDSVM-02) — minibatch-SGD linear regressor, ≈
//! `sklearn.linear_model.SGDRegressor`.
//!
//! Phase-10 Wave-0 scaffold (plan 10-01): the struct, the
//! [`MBSGDRegressorBuilder`] (D-01/D-03 — sklearn-default field initializers),
//! and the `build() -> Result<MBSGDRegressor<F>, BuildError>` SIGNATURE are final
//! now; the validation predicates and the `fit`/`predict` bodies land in the
//! Wave-1/Wave-3 plans. The closest analog is `elastic_net.rs` (regressor: `Fit`
//! + `Predict`, alpha/l1_ratio penalty); the construction surface switches from
//! `new()`/`with_opts()` to the builder (D-01). Unlike the classifier, the
//! regressor exposes an `epsilon` setter (for the epsilon-insensitive losses).
//!
//! Tests live in `crates/mlrs-algos/tests/mbsgd_regressor_test.rs`
//! (AGENTS.md §2), never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::runtime::ActiveRuntime;

use crate::error::BuildError;
use crate::linear::sgd_config::{LearningRate, Loss, Penalty, SgdConfig};

/// Minibatch-SGD linear regressor (SGDSVM-02). Construct via
/// [`MBSGDRegressor::builder`].
///
/// Wave-0 scaffold: the fitted-state fields are written by the Wave-1 `fit`
/// body, hence `allow(dead_code)` until then.
#[allow(dead_code)]
pub struct MBSGDRegressor<F> {
    /// The lowered, validated hyperparameter bundle (D-06).
    config: SgdConfig,
    /// Fitted coefficients (device-resident), `None` until `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (device-resident), `None` until `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
}

impl<F> MBSGDRegressor<F>
where
    F: Float + CubeElement + Pod,
{
    /// Start building an `MBSGDRegressor` with sklearn's `SGDRegressor` defaults
    /// (D-03).
    pub fn builder() -> MBSGDRegressorBuilder {
        MBSGDRegressorBuilder::default()
    }

    /// The lowered configuration (D-06).
    pub fn config(&self) -> &SgdConfig {
        &self.config
    }
}

/// Builder for [`MBSGDRegressor`] (D-01). Default field initializers encode the
/// sklearn `SGDRegressor` defaults (D-03): `loss=squared_error`, `penalty=l2`,
/// `alpha=1e-4`, `l1_ratio=0.15`, `max_iter=1000`, `tol=1e-3`,
/// `learning_rate=invscaling`, `eta0=0.01`, `power_t=0.25`, `epsilon=0.1`.
#[derive(Debug, Clone, Copy)]
pub struct MBSGDRegressorBuilder {
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
    epsilon: f64,
    batch_size: usize,
    shuffle: bool,
    seed: u64,
}

impl Default for MBSGDRegressorBuilder {
    fn default() -> Self {
        Self {
            loss: Loss::SquaredLoss,
            penalty: Penalty::L2,
            alpha: 1e-4,
            l1_ratio: 0.15,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-3,
            learning_rate: LearningRate::InvScaling,
            eta0: 0.01,
            power_t: 0.25,
            epsilon: 0.1,
            batch_size: 1,
            shuffle: true,
            seed: 0,
        }
    }
}

impl MBSGDRegressorBuilder {
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
    /// Set the epsilon-insensitive margin (regression losses only).
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
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
    /// Wave-1.
    pub fn build<F>(self) -> Result<MBSGDRegressor<F>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
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
            epsilon: self.epsilon,
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            seed: self.seed,
        };
        Ok(MBSGDRegressor {
            config,
            coef_: None,
            intercept_: None,
        })
    }
}
