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
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::sgd::sgd_solve;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::linear::elastic_net::predict_linear;
use crate::linear::mbsgd_classifier::lower_config;
use crate::linear::sgd_config::{LearningRate, Loss, Penalty, SgdConfig};
use crate::traits::{Fit, Predict};

/// Minibatch-SGD linear regressor (SGDSVM-02). Construct via
/// [`MBSGDRegressor::builder`], then [`Fit::fit`] + [`Predict::predict`]. Fitted
/// `coef_` (length `n_features`) / `intercept_` (length 1) are device-resident
/// (D-03).
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

    /// Host copy of the fitted `coef_` (length `n_features`). Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.coef_
            .as_ref()
            .map(|c| c.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "mbsgd_regressor",
                operation: "coef_",
            })
    }

    /// Host copy of the fitted `intercept_` (scalar). Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> Result<F, AlgoError> {
        self.intercept_
            .as_ref()
            .map(|i| i.to_host(pool)[0])
            .ok_or(AlgoError::NotFitted {
                estimator: "mbsgd_regressor",
                operation: "intercept_",
            })
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
    /// (D-08, T-10-03-01) BEFORE any data is seen (the data-DEPENDENT geometry
    /// check lives in [`Fit::fit`]):
    ///
    /// - `alpha >= 0` ([`BuildError::InvalidAlpha`]).
    /// - `l1_ratio ∈ [0, 1]` ([`BuildError::InvalidL1Ratio`]) when the penalty is
    ///   `ElasticNet`.
    /// - `eta0 > 0` ([`BuildError::InvalidEta0`]) unless the schedule is `Optimal`.
    /// - `epsilon >= 0` ([`BuildError::InvalidEpsilon`]).
    /// - the loss must be valid for a REGRESSOR ({`SquaredLoss`,
    ///   `EpsilonInsensitive`, `SquaredEpsilonInsensitive`}); a classification loss
    ///   (`Hinge` / `Log` / `SquaredHinge`) is
    ///   [`BuildError::InvalidLossForEstimator`].
    pub fn build<F>(self) -> Result<MBSGDRegressor<F>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        // --- T-10-03-01 / ASVS V5: validate the data-INDEPENDENT hyperparameters
        //     at build() BEFORE any data is seen (D-08). ---
        if !(self.alpha >= 0.0) {
            return Err(BuildError::InvalidAlpha {
                estimator: "mbsgd_regressor",
                alpha: self.alpha,
            });
        }
        if self.penalty == Penalty::ElasticNet
            && !(self.l1_ratio >= 0.0 && self.l1_ratio <= 1.0)
        {
            return Err(BuildError::InvalidL1Ratio {
                estimator: "mbsgd_regressor",
                l1_ratio: self.l1_ratio,
            });
        }
        if self.learning_rate != LearningRate::Optimal && !(self.eta0 > 0.0) {
            return Err(BuildError::InvalidEta0 {
                estimator: "mbsgd_regressor",
                eta0: self.eta0,
            });
        }
        if !(self.epsilon >= 0.0) {
            return Err(BuildError::InvalidEpsilon {
                estimator: "mbsgd_regressor",
                epsilon: self.epsilon,
            });
        }
        match self.loss {
            Loss::SquaredLoss
            | Loss::EpsilonInsensitive
            | Loss::SquaredEpsilonInsensitive => {}
            other => {
                return Err(BuildError::InvalidLossForEstimator {
                    estimator: "mbsgd_regressor",
                    loss: other.name().to_string(),
                });
            }
        }
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

impl<F> Fit<F> for MBSGDRegressor<F>
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

        // --- T-10-03-02 / ASVS V5: data-DEPENDENT geometry guard BEFORE any launch
        //     (D-08 — the data-INDEPENDENT params were validated at build()). ---
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "mbsgd_regressor",
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

        // --- Lower the validated SgdConfig into the prim-local flat SgdParams
        //     (D-06; shared lowering with the classifier) and delegate to the
        //     validated PRIM-10 prim (10-02). The regressor target `y` is the raw
        //     continuous response (no ±1 remap). A device failure is a typed
        //     PrimError wrapped via `?` (never a panic — T-10-03-03). ---
        let params = lower_config(&self.config);
        let (coef, intercept) = sgd_solve::<F>(pool, x, y, shape, &params)?;

        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept);
        Ok(self)
    }
}

impl<F> Predict<F> for MBSGDRegressor<F>
where
    F: Float + CubeElement + Pod,
{
    /// Predict via the shared `X·coef_ + intercept_` path (the `elastic_net.rs`
    /// [`predict_linear`] GEMM-then-broadcast — reused so the regression predict
    /// surface is implemented once, D-03). No duplicated GEMM path.
    fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        predict_linear(
            self.coef_.as_ref(),
            self.intercept_.as_ref(),
            "mbsgd_regressor",
            pool,
            x,
            shape,
        )
    }
}
