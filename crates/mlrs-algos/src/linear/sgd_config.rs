//! `sgd_config` — the typed construction surface for the Phase-10 SGD /
//! linear-SVM estimators (D-04 typed enums, D-05 single-source `TryFrom`, D-06
//! shared `SgdConfig` lowering target).
//!
//! ## Typed categorical knobs (D-04 — NOT `String`)
//! sklearn exposes `loss` / `penalty` / `learning_rate` as strings; mlrs models
//! each as a `Copy` enum ([`Loss`] / [`Penalty`] / [`LearningRate`]) with a
//! single-source [`TryFrom<&str>`] that accepts the sklearn spellings AND the
//! legacy aliases (`log`/`log_loss`, `squared_error`/`squared_loss`). The
//! `String`-typed `affinity` field in `spectral_clustering.rs` is the
//! ANTI-PATTERN this module deliberately rejects (D-04). The matcher lives HERE
//! (in `mlrs-algos`), not in the PyO3 wrapper — the wrapper just calls
//! `Loss::try_from(s)` and maps the [`BuildError`] to a `ValueError` (D-09).
//!
//! ## `SgdConfig` — the shared lowering target (D-06)
//! All four builders ([`mbsgd_classifier`](crate::linear::mbsgd_classifier),
//! [`mbsgd_regressor`](crate::linear::mbsgd_regressor),
//! [`linear_svc`](crate::linear::linear_svc),
//! [`linear_svr`](crate::linear::linear_svr)) lower their validated, typed
//! hyperparameters into ONE [`SgdConfig`] that the estimator stores and the
//! (Wave-1) `sgd_solve` prim consumes. The builders themselves expose only the
//! knobs valid for their estimator (the SVMs have no `eta0`/`learning_rate`
//! setter; the regressors expose `epsilon`).
//!
//! Tests live in `crates/mlrs-algos/tests/sgd_config_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use crate::error::BuildError;

/// The SGD / linear-SVM loss family (D-04). `Copy` and stored typed in
/// [`SgdConfig`]; constructed via the single-source [`TryFrom<&str>`] that
/// accepts the sklearn spellings and the `log`/`squared_error` aliases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    /// Linear SVM hinge loss `max(0, 1 - y·p)` (classifier default).
    Hinge,
    /// Logistic (log) loss `log(1 + exp(-y·p))` — the `predict_proba` family.
    Log,
    /// Squared hinge `max(0, 1 - y·p)²` (LinearSVC default).
    SquaredHinge,
    /// Squared error `½(p - y)²` (regressor default; sklearn `squared_error`).
    SquaredLoss,
    /// Epsilon-insensitive `max(0, |y - p| - ε)` (SVR loss family).
    EpsilonInsensitive,
    /// Squared epsilon-insensitive `max(0, |y - p| - ε)²` (LinearSVR default).
    SquaredEpsilonInsensitive,
}

impl Loss {
    /// The canonical sklearn loss name (for diagnostics / round-trip).
    pub fn name(self) -> &'static str {
        match self {
            Loss::Hinge => "hinge",
            Loss::Log => "log",
            Loss::SquaredHinge => "squared_hinge",
            Loss::SquaredLoss => "squared_error",
            Loss::EpsilonInsensitive => "epsilon_insensitive",
            Loss::SquaredEpsilonInsensitive => "squared_epsilon_insensitive",
        }
    }
}

impl TryFrom<&str> for Loss {
    type Error = BuildError;

    /// Parse a sklearn loss string (D-05 single source). Accepts the legacy
    /// aliases `log`/`log_loss` and `squared_error`/`squared_loss`; any other
    /// value is [`BuildError::UnknownLoss`].
    fn try_from(s: &str) -> Result<Self, BuildError> {
        match s {
            "hinge" => Ok(Loss::Hinge),
            "log" | "log_loss" => Ok(Loss::Log),
            "squared_hinge" => Ok(Loss::SquaredHinge),
            "squared_error" | "squared_loss" => Ok(Loss::SquaredLoss),
            "epsilon_insensitive" => Ok(Loss::EpsilonInsensitive),
            "squared_epsilon_insensitive" => Ok(Loss::SquaredEpsilonInsensitive),
            other => Err(BuildError::UnknownLoss {
                value: other.to_string(),
            }),
        }
    }
}

/// The regularization penalty family (D-04).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Penalty {
    /// L1 (Lasso) penalty `α·‖w‖₁`.
    L1,
    /// L2 (Ridge) penalty `½·α·‖w‖₂²` (the default for every Phase-10 estimator).
    L2,
    /// ElasticNet `α·(l1_ratio·‖w‖₁ + ½·(1−l1_ratio)·‖w‖₂²)`.
    ElasticNet,
}

impl Penalty {
    /// The canonical sklearn penalty name.
    pub fn name(self) -> &'static str {
        match self {
            Penalty::L1 => "l1",
            Penalty::L2 => "l2",
            Penalty::ElasticNet => "elasticnet",
        }
    }
}

impl TryFrom<&str> for Penalty {
    type Error = BuildError;

    /// Parse a sklearn penalty string (D-05). Any other value is
    /// [`BuildError::UnknownPenalty`].
    fn try_from(s: &str) -> Result<Self, BuildError> {
        match s {
            "l1" => Ok(Penalty::L1),
            "l2" => Ok(Penalty::L2),
            "elasticnet" => Ok(Penalty::ElasticNet),
            other => Err(BuildError::UnknownPenalty {
                value: other.to_string(),
            }),
        }
    }
}

/// The SGD learning-rate schedule (D-04). Only the SGD estimators
/// (`MBSGDClassifier`/`MBSGDRegressor`) expose this; the CD-solved SVMs do not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearningRate {
    /// Bottou `optimal` schedule `η(t) = 1/(α·(t₀ + t − 1))` (classifier default).
    Optimal,
    /// Inverse-scaling `η(t) = eta0 / t^power_t` (regressor default).
    InvScaling,
    /// Constant `η = eta0`.
    Constant,
    /// Adaptive `η = eta0`, halved on a stalled objective.
    Adaptive,
}

impl LearningRate {
    /// The canonical sklearn learning-rate name.
    pub fn name(self) -> &'static str {
        match self {
            LearningRate::Optimal => "optimal",
            LearningRate::InvScaling => "invscaling",
            LearningRate::Constant => "constant",
            LearningRate::Adaptive => "adaptive",
        }
    }
}

impl TryFrom<&str> for LearningRate {
    type Error = BuildError;

    /// Parse a sklearn learning-rate string (D-05). Any other value is
    /// [`BuildError::UnknownLearningRate`].
    fn try_from(s: &str) -> Result<Self, BuildError> {
        match s {
            "optimal" => Ok(LearningRate::Optimal),
            "invscaling" => Ok(LearningRate::InvScaling),
            "constant" => Ok(LearningRate::Constant),
            "adaptive" => Ok(LearningRate::Adaptive),
            other => Err(BuildError::UnknownLearningRate {
                value: other.to_string(),
            }),
        }
    }
}

/// The shared, lowered hyperparameter bundle every Phase-10 builder produces
/// (D-06). The estimator stores this and the (Wave-1) `sgd_solve` prim consumes
/// it. The fields are data-INDEPENDENT and already validated at `build()` (D-08).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgdConfig {
    /// The loss family (typed, D-04).
    pub loss: Loss,
    /// The penalty family (typed, D-04).
    pub penalty: Penalty,
    /// Overall penalty strength `α ≥ 0`.
    pub alpha: f64,
    /// ElasticNet L1/L2 mixing parameter `∈ [0, 1]`.
    pub l1_ratio: f64,
    /// Whether to fit an intercept term.
    pub fit_intercept: bool,
    /// Epoch cap.
    pub max_iter: usize,
    /// Stopping tolerance (`0` ⇒ run the full `max_iter` epochs, the oracle pin).
    pub tol: f64,
    /// The learning-rate schedule (SGD estimators only; CD estimators ignore it).
    pub learning_rate: LearningRate,
    /// Initial learning rate `eta0 > 0` for the `constant`/`invscaling` schedules.
    pub eta0: f64,
    /// Inverse-scaling exponent `power_t`.
    pub power_t: f64,
    /// Epsilon-insensitive margin `ε ≥ 0` (regression losses only).
    pub epsilon: f64,
    /// Minibatch size.
    pub batch_size: usize,
    /// Whether to shuffle each epoch (`false` is the deterministic oracle pin).
    pub shuffle: bool,
    /// RNG seed (used only when `shuffle == true`).
    pub seed: u64,
}
