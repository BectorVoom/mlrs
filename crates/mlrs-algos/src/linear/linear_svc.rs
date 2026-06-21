//! `LinearSVC` (SGDSVM-03) — linear support-vector classifier, ≈
//! `sklearn.svm.LinearSVC`.
//!
//! Phase-10 Wave-0 scaffold (plan 10-01): the struct, the [`LinearSVCBuilder`]
//! (D-01/D-03 — sklearn-default field initializers), and the
//! `build() -> Result<LinearSVC<F>, BuildError>` SIGNATURE are final now; the
//! validation predicates and the `fit`/`predict` bodies land in the
//! Wave-2/Wave-3 plans. The closest analog is `elastic_net.rs` (struct + predict)
//! over the v1 coordinate-descent solver (D-07 — `LinearSVC` is liblinear CD, NOT
//! the SGD prim). Unlike the SGD estimators, the SVM builder exposes `c` +
//! `intercept_scaling` and has NO `eta0`/`learning_rate` setter (the CD solver
//! has no schedule).
//!
//! Tests live in `crates/mlrs-algos/tests/linear_svc_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::runtime::ActiveRuntime;

use crate::error::BuildError;
use crate::linear::sgd_config::{LearningRate, Loss, Penalty, SgdConfig};

/// Linear support-vector classifier (SGDSVM-03). Construct via
/// [`LinearSVC::builder`].
///
/// Wave-0 scaffold: the fitted-state fields are written by the Wave-2 `fit`
/// body, hence `allow(dead_code)` until then.
#[allow(dead_code)]
pub struct LinearSVC<F> {
    /// The lowered hyperparameter bundle (D-06); the SVM-specific knobs (`c`,
    /// `intercept_scaling`) sit alongside it.
    config: SgdConfig,
    /// Inverse-regularization strength `C > 0` (sklearn `C`).
    c: f64,
    /// Synthetic-feature intercept scaling (Pitfall 5 — NOT center-then-solve).
    intercept_scaling: f64,
    /// DISTINCT sorted class labels inferred at `fit` (Pitfall 4 — ±1 encoding).
    classes_: Vec<i64>,
    /// Feature count inferred at `fit`.
    n_features: usize,
    /// Fitted coefficients (device-resident), `None` until `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (device-resident), `None` until `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
}

impl<F> LinearSVC<F>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `LinearSVC` with sklearn's `LinearSVC` defaults (D-03).
    pub fn builder() -> LinearSVCBuilder {
        LinearSVCBuilder::default()
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

    /// The inferred class labels (empty until `fit`).
    pub fn classes(&self) -> &[i64] {
        &self.classes_
    }
}

/// Builder for [`LinearSVC`] (D-01). Default field initializers encode the
/// sklearn `LinearSVC` defaults (D-03): `loss=squared_hinge`, `penalty=l2`,
/// `c=1.0`, `intercept_scaling=1.0`, `max_iter=1000`, `tol=1e-4`. The CD-solved
/// SVM has NO learning-rate schedule, so there is no `eta0`/`learning_rate`
/// setter.
#[derive(Debug, Clone, Copy)]
pub struct LinearSVCBuilder {
    loss: Loss,
    penalty: Penalty,
    c: f64,
    intercept_scaling: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
}

impl Default for LinearSVCBuilder {
    fn default() -> Self {
        Self {
            loss: Loss::SquaredHinge,
            penalty: Penalty::L2,
            c: 1.0,
            intercept_scaling: 1.0,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
        }
    }
}

impl LinearSVCBuilder {
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
    /// (D-08). The SIGNATURE is final (Wave-0); the validation predicates land in
    /// Wave-2. The `c`/`intercept_scaling` knobs are stored alongside the lowered
    /// [`SgdConfig`] (the CD path maps `C` → penalty internally at `fit`).
    pub fn build<F>(self) -> Result<LinearSVC<F>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        let config = SgdConfig {
            loss: self.loss,
            penalty: self.penalty,
            // alpha is derived from C at fit (l2_reg = 1/(C·n)); stored as a
            // placeholder here so the SVM path keeps the shared lowering target.
            alpha: 0.0,
            l1_ratio: 0.0,
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
            // The CD-solved SVM has no schedule; the SgdConfig schedule fields are
            // inert for LinearSVC (kept only for the shared lowering shape, D-06).
            learning_rate: LearningRate::Constant,
            eta0: 0.0,
            power_t: 0.0,
            epsilon: 0.0,
            batch_size: 0,
            shuffle: false,
            seed: 0,
        };
        Ok(LinearSVC {
            config,
            c: self.c,
            intercept_scaling: self.intercept_scaling,
            classes_: Vec::new(),
            n_features: 0,
            coef_: None,
            intercept_: None,
        })
    }
}
