//! `LinearSVR` (SGDSVM-04) — linear support-vector regressor, ≈
//! `sklearn.svm.LinearSVR`.
//!
//! ## Solver (Open Question Q1 RESOLVED — L-BFGS, shared with LinearSVC)
//! `LinearSVR` minimizes the L2-regularized SQUARED-EPSILON-INSENSITIVE primal
//! `½‖w‖² + C·Σᵢ max(0, |yᵢ − xᵢ·w| − ε)²`. Like the LinearSVC squared-hinge
//! objective this is SMOOTH (C¹) and CONVEX but is NOT the Lasso/ElasticNet
//! soft-threshold CD objective (Open Q1 / RESEARCH §LinearSVC), so it reuses the
//! SAME validated 05-06 L-BFGS path via the shared
//! [`svm_lbfgs_fit`](crate::linear::linear_svc::svm_lbfgs_fit) helper — only the
//! per-sample margin-loss closure differs (squared-eps-insensitive vs
//! squared-hinge). An early Python spike confirmed this reproduces sklearn's
//! `coef_`/`intercept_`/`predict` (see the 10-04 SUMMARY).
//!
//! ## Intercept + predict (shared paths)
//! The intercept is the SAME synthetic-feature `intercept_scaling` mechanism as
//! LinearSVC (`intercept_ = intercept_scaling · w_last`, Pitfall 5 — NOT
//! center-then-solve). `predict` delegates to the shared
//! [`predict_linear`](crate::linear::elastic_net::predict_linear) `X·coef_ +
//! intercept_` GEMM path (the `elastic_net.rs` regressor precedent).
//!
//! Tests live in `crates/mlrs-algos/tests/linear_svr_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::linear::elastic_net::predict_linear;
use crate::linear::linear_svc::svm_lbfgs_fit;
use crate::linear::sgd_config::{LearningRate, Loss, Penalty, SgdConfig};
use crate::traits::{Fit, Predict};

/// Linear support-vector regressor (SGDSVM-04). Construct via
/// [`LinearSVR::builder`], then [`Fit::fit`] + [`Predict::predict`]. Fitted
/// `coef_` (length `n_features`) / `intercept_` (length 1) are device-resident
/// (D-03).
pub struct LinearSVR<F> {
    /// The lowered hyperparameter bundle (D-06); the SVM-specific knobs (`c`,
    /// `intercept_scaling`) sit alongside it.
    config: SgdConfig,
    /// Inverse-regularization strength `C > 0` (sklearn `C`).
    c: f64,
    /// Synthetic-feature intercept scaling (Pitfall 5 — NOT center-then-solve).
    intercept_scaling: f64,
    /// Fitted coefficients (device-resident), `None` until `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (device-resident), `None` until `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
}

impl<F> LinearSVR<F>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `LinearSVR` with sklearn's `LinearSVR` defaults (D-03).
    pub fn builder() -> LinearSVRBuilder {
        LinearSVRBuilder::default()
    }

    /// The lowered configuration (D-06).
    pub fn config(&self) -> &SgdConfig {
        &self.config
    }

    /// The inverse-regularization strength `C`.
    pub fn c(&self) -> f64 {
        self.c
    }

    /// The synthetic-feature intercept scaling.
    pub fn intercept_scaling(&self) -> f64 {
        self.intercept_scaling
    }

    /// Host copy of the fitted `coef_` (length `n_features`). Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.coef_
            .as_ref()
            .map(|c| c.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "linear_svr",
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
                estimator: "linear_svr",
                operation: "intercept_",
            })
    }
}

/// Builder for [`LinearSVR`] (D-01). Default field initializers encode the
/// sklearn `LinearSVR` defaults (D-03): `loss=squared_epsilon_insensitive`,
/// `penalty=l2`, `c=1.0`, `epsilon=0.0`, `intercept_scaling=1.0`,
/// `max_iter=1000`, `tol=1e-4`. The CD-solved SVM has NO learning-rate schedule.
#[derive(Debug, Clone, Copy)]
pub struct LinearSVRBuilder {
    loss: Loss,
    penalty: Penalty,
    c: f64,
    epsilon: f64,
    intercept_scaling: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
}

impl Default for LinearSVRBuilder {
    fn default() -> Self {
        Self {
            loss: Loss::SquaredEpsilonInsensitive,
            penalty: Penalty::L2,
            c: 1.0,
            epsilon: 0.0,
            intercept_scaling: 1.0,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
        }
    }
}

impl LinearSVRBuilder {
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
    /// Set the inverse-regularization strength `C`.
    pub fn c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }
    /// Set the epsilon-insensitive margin.
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
    /// Set the synthetic-feature intercept scaling.
    pub fn intercept_scaling(mut self, intercept_scaling: f64) -> Self {
        self.intercept_scaling = intercept_scaling;
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

    /// Build the estimator, validating the data-INDEPENDENT hyperparameters
    /// (D-08, T-10-04-01). `C > 0` ([`BuildError::InvalidC`]), `epsilon >= 0`
    /// ([`BuildError::InvalidEpsilon`]), and the loss family must be valid for a
    /// REGRESSOR ({`EpsilonInsensitive`, `SquaredEpsilonInsensitive`} — a
    /// classifier loss like `Hinge`/`SquaredHinge`/`Log` is
    /// [`BuildError::InvalidLossForEstimator`]). Only `L1`/`L2` penalties.
    pub fn build<F>(self) -> Result<LinearSVR<F>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        // --- T-10-04-01 / ASVS V5: validate the data-INDEPENDENT hyperparameters
        //     at build() BEFORE any data is seen (D-08). ---
        if !(self.c > 0.0) {
            return Err(BuildError::InvalidC {
                estimator: "linear_svr",
                c: self.c,
            });
        }
        if !(self.epsilon >= 0.0) {
            return Err(BuildError::InvalidEpsilon {
                estimator: "linear_svr",
                epsilon: self.epsilon,
            });
        }
        match self.loss {
            Loss::EpsilonInsensitive | Loss::SquaredEpsilonInsensitive => {}
            other => {
                return Err(BuildError::InvalidLossForEstimator {
                    estimator: "linear_svr",
                    loss: other.name().to_string(),
                });
            }
        }
        match self.penalty {
            Penalty::L1 | Penalty::L2 => {}
            Penalty::ElasticNet => {
                return Err(BuildError::UnknownPenalty {
                    value: "elasticnet (LinearSVR supports only l1/l2)".to_string(),
                });
            }
        }
        let config = SgdConfig {
            loss: self.loss,
            penalty: self.penalty,
            // alpha is derived from C at fit (l2_reg = 1/(C·n)); placeholder here.
            alpha: 0.0,
            l1_ratio: 0.0,
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
            // The CD-solved SVR has no schedule; the schedule fields are inert.
            learning_rate: LearningRate::Constant,
            eta0: 0.0,
            power_t: 0.0,
            epsilon: self.epsilon,
            batch_size: 0,
            shuffle: false,
            seed: 0,
        };
        Ok(LinearSVR {
            config,
            c: self.c,
            intercept_scaling: self.intercept_scaling,
            coef_: None,
            intercept_: None,
        })
    }
}

impl<F> Fit<F> for LinearSVR<F>
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

        // --- T-10-04-02 / ASVS V5: geometry guard BEFORE any launch. ---
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "linear_svr",
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

        // The regression targets are the per-sample `y` (no ±1 remap — SVR is a
        // regressor). Reuse the SHARED svm_lbfgs_fit with the squared-eps-insensitive
        // margin loss: residual r = target − margin; viol = max(0, |r| − ε);
        //   ℓ = viol² ;  dℓ/dmargin = d(viol²)/d(−r)... = −2·sign(r)·viol·(−1) =
        //   2·sign(r)·viol·(−1)? Derive carefully: margin m = x̃·w, r = y − m, so
        //   ∂r/∂m = −1. viol = max(0,|r|−ε). ℓ = viol². dℓ/dm = 2·viol·∂viol/∂m.
        //   ∂viol/∂m = sign(r)·∂|r|/∂r·∂r/∂m... when viol>0: ∂|r|/∂r = sign(r),
        //   ∂r/∂m = −1, so ∂viol/∂m = −sign(r). Hence dℓ/dm = −2·sign(r)·viol.
        let targets = y.to_host(pool);
        let targets_f64: Vec<f64> = targets.iter().map(|&v| host_to_f64(v)).collect();
        // IN-03: `self.c` is already `f64`; use it directly (no identity cast).
        let c = self.c;
        let eps = self.config.epsilon;
        let (coef, intercept) = svm_lbfgs_fit::<F>(
            pool,
            x,
            &targets_f64,
            n_samples,
            n_features,
            c,
            self.intercept_scaling,
            self.config.fit_intercept,
            self.config.max_iter,
            self.config.tol,
            "linear_svr",
            |margin, target| {
                let r = target - margin;
                let absr = r.abs();
                let viol = absr - eps;
                if viol > 0.0 {
                    let s = if r >= 0.0 { 1.0 } else { -1.0 };
                    (viol * viol, -2.0 * s * viol) // (loss_i, dloss/dmargin)
                } else {
                    (0.0, 0.0)
                }
            },
        )?;

        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept);
        Ok(self)
    }
}

impl<F> Predict<F> for LinearSVR<F>
where
    F: Float + CubeElement + Pod,
{
    fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        predict_linear(
            self.coef_.as_ref(),
            self.intercept_.as_ref(),
            "linear_svr",
            pool,
            x,
            shape,
        )
    }
}
