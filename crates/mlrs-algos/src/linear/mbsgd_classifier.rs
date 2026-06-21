//! `MBSGDClassifier` (SGDSVM-01) — minibatch-SGD linear classifier, ≈
//! `sklearn.linear_model.SGDClassifier`.
//!
//! The struct, the [`MBSGDClassifierBuilder`] (D-01/D-03 — sklearn-default field
//! initializers), the `build() -> Result<MBSGDClassifier<F>, BuildError>`
//! validation, and the `fit`/`predict` bodies are all SHIPPED: `fit` lowers the
//! validated `SgdConfig` into the flat `SgdParams` and drives the PRIM-10
//! `sgd_solve` minibatch-SGD solver; `predict_labels`/`predict_proba` run the
//! on-device decision-margin matvec. The closest analog is `logistic.rs`
//! (classifier: `classes_` remap + `PredictLabels` + `PredictProba`); the
//! construction surface switches from `new()`/`with_opts()` to the builder (D-01).
//!
//! Tests live in `crates/mlrs-algos/tests/mbsgd_classifier_test.rs`
//! (AGENTS.md §2), never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::sgd::{sgd_solve, SgdLoss, SgdParams, SgdSchedule};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::linear::sgd_config::{LearningRate, Loss, Penalty, SgdConfig};
use crate::traits::{Fit, PredictLabels, PredictProba};

/// Minibatch-SGD linear classifier (SGDSVM-01). Construct via
/// [`MBSGDClassifier::builder`], then [`Fit::fit`] +
/// [`PredictLabels::predict_labels`] / [`PredictProba::predict_proba`]. Fitted
/// `coef_` (length `n_features`) / `intercept_` (length 1) are device-resident
/// (D-03).
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

    /// Host copy of the fitted `coef_` (length `n_features`). Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.coef_
            .as_ref()
            .map(|c| c.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "mbsgd_classifier",
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
                estimator: "mbsgd_classifier",
                operation: "intercept_",
            })
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
    /// (D-08, T-10-03-01). The data-INDEPENDENT predicates are checked HERE,
    /// BEFORE any data is seen (the data-DEPENDENT geometry / label checks live in
    /// [`Fit::fit`], D-08):
    ///
    /// - `alpha >= 0` ([`BuildError::InvalidAlpha`]) — a negative penalty is
    ///   undefined.
    /// - `l1_ratio ∈ [0, 1]` ([`BuildError::InvalidL1Ratio`]) when the penalty is
    ///   `ElasticNet` (the mixing parameter blends L1/L2).
    /// - `eta0 > 0` ([`BuildError::InvalidEta0`]) unless the schedule is `Optimal`
    ///   (the Bottou schedule does not read `eta0`).
    /// - the loss must be valid for a CLASSIFIER ({`Hinge`, `Log`,
    ///   `SquaredHinge`}); a regression loss (`EpsilonInsensitive` /
    ///   `SquaredEpsilonInsensitive`) is [`BuildError::InvalidLossForEstimator`].
    ///
    /// On success the lowered [`SgdConfig`] is stored and the fitted state is
    /// `None`.
    pub fn build<F>(self) -> Result<MBSGDClassifier<F>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        // --- T-10-03-01 / ASVS V5: validate the data-INDEPENDENT hyperparameters
        //     at build() BEFORE any data is seen (D-08). ---
        if !(self.alpha >= 0.0) {
            return Err(BuildError::InvalidAlpha {
                estimator: "mbsgd_classifier",
                alpha: self.alpha,
            });
        }
        if self.penalty == Penalty::ElasticNet
            && !(self.l1_ratio >= 0.0 && self.l1_ratio <= 1.0)
        {
            return Err(BuildError::InvalidL1Ratio {
                estimator: "mbsgd_classifier",
                l1_ratio: self.l1_ratio,
            });
        }
        if self.learning_rate != LearningRate::Optimal && !(self.eta0 > 0.0) {
            return Err(BuildError::InvalidEta0 {
                estimator: "mbsgd_classifier",
                eta0: self.eta0,
            });
        }
        // WR-04: reject a non-finite `power_t` (NaN / ±inf) — it feeds the
        // `invscaling` schedule `eta0 / t^power_t` and would drive the step rate
        // to NaN/inf. A negative finite `power_t` is accepted (documented
        // divergence — it makes the rate grow with t).
        if !self.power_t.is_finite() {
            return Err(BuildError::InvalidPowerT {
                estimator: "mbsgd_classifier",
                power_t: self.power_t,
            });
        }
        match self.loss {
            Loss::Hinge | Loss::Log | Loss::SquaredHinge => {}
            other => {
                return Err(BuildError::InvalidLossForEstimator {
                    estimator: "mbsgd_classifier",
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

impl<F> Fit<F> for MBSGDClassifier<F>
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

        // --- T-10-03-02 / ASVS V5: data-DEPENDENT geometry guard BEFORE any
        //     launch (D-08 — the data-INDEPENDENT params were validated at
        //     build()). ---
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "mbsgd_classifier",
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

        // --- Pitfall 4: distinct-sorted classes_ (logistic.rs precedent), binary
        //     ±1 remap for the margin loss. Phase-10 scope is the binary linear
        //     classifier (A6); a non-binary target is rejected. ---
        let y_host = y.to_host(pool);
        let mut raw_labels: Vec<i64> = Vec::with_capacity(n_samples);
        for &yv in y_host.iter() {
            let lf = host_to_f64(yv);
            let li = lf.round();
            if (li - lf).abs() > 1e-6 {
                return Err(AlgoError::InvalidLabels {
                    estimator: "mbsgd_classifier",
                    reason: format!("labels must be integers (got {lf})"),
                });
            }
            raw_labels.push(li as i64);
        }
        let mut classes_: Vec<i64> = raw_labels.clone();
        classes_.sort_unstable();
        classes_.dedup();
        if classes_.len() != 2 {
            return Err(AlgoError::InvalidLabels {
                estimator: "mbsgd_classifier",
                reason: format!(
                    "binary classifier needs exactly 2 classes, found {}",
                    classes_.len()
                ),
            });
        }
        // WR-02: `predict_labels` emits class ids as `i32`; a class id that fits
        // an `f64` mantissa but exceeds `i32` range would be SILENTLY TRUNCATED
        // (`as i32` wraps) into a wrong predicted label. Validate the distinct
        // class ids against `i32` range at fit so an out-of-range label is a
        // typed error, not a silent wrong prediction.
        for &cls in classes_.iter() {
            if i32::try_from(cls).is_err() {
                return Err(AlgoError::InvalidLabels {
                    estimator: "mbsgd_classifier",
                    reason: format!(
                        "class label {cls} does not fit in i32 \
                         (predicted labels are i32)"
                    ),
                });
            }
        }
        // classes_[0] → −1, classes_[1] → +1 (sklearn maps the higher class to +1).
        let yp: Vec<F> = raw_labels
            .iter()
            .map(|&l| f64_to_host::<F>(if l == classes_[1] { 1.0 } else { -1.0 }))
            .collect();
        let yp_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &yp);

        // --- Lower the validated SgdConfig into the prim-local flat SgdParams
        //     (D-06; the prim cannot take the algos SgdConfig — circular
        //     dependency). The classifier never uses epsilon (regression-only). ---
        let params = lower_config(&self.config);

        // Delegate to the validated PRIM-10 prim (10-02). A device failure is a
        // typed PrimError, wrapped into AlgoError::Prim via `?` (never a panic
        // across the estimator boundary — T-10-03-03).
        let (coef, intercept) = sgd_solve::<F>(pool, x, &yp_dev, shape, &params)?;

        // The ±1 target buffer is only needed during the solve (WR-07 re-fit
        // buffer release).
        yp_dev.release_into(pool);

        self.classes_ = classes_;
        self.n_features = n_features;
        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept);
        Ok(self)
    }
}

impl<F> PredictLabels<F> for MBSGDClassifier<F>
where
    F: Float + CubeElement + Pod,
{
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        let (n_query, _n_features) = shape;

        // The signed decision margin per query row (X·coef + intercept).
        let margins = self.decision_margin(pool, x, shape)?;

        // sign of the margin selects the class: >= 0 → classes_[1] (the +1 class),
        // else classes_[0] (the −1 class) — sklearn's `decision >= 0 → +1`.
        let mut labels: Vec<i32> = vec![0i32; n_query];
        for (r, label) in labels.iter_mut().enumerate() {
            *label = if margins[r] >= 0.0 {
                self.classes_[1] as i32
            } else {
                self.classes_[0] as i32
            };
        }
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

impl<F> PredictProba<F> for MBSGDClassifier<F>
where
    F: Float + CubeElement + Pod,
{
    /// Per-class probabilities from the log-loss sigmoid `1/(1 + exp(−margin))`
    /// (sklearn's `SGDClassifier(loss="log_loss").predict_proba`). The returned
    /// matrix is `n_query × 2` (`[P(class₀), P(class₁)]` per row); `P(class₁) =
    /// σ(margin)`, `P(class₀) = 1 − σ(margin)`. For a non-log loss this sigmoid is
    /// NOT a calibrated probability (sklearn raises); mlrs returns the same sigmoid
    /// shape over the decision margin (the caller pins the log-loss fixture).
    fn predict_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, _n_features) = shape;
        let margins = self.decision_margin(pool, x, shape)?;

        let mut proba: Vec<F> = vec![F::from_int(0i64); n_query * 2];
        for (r, &m) in margins.iter().enumerate() {
            // Numerically-stable logistic sigmoid σ(m) = 1/(1 + exp(−m)).
            let p1 = if m >= 0.0 {
                1.0 / (1.0 + (-m).exp())
            } else {
                let e = m.exp();
                e / (1.0 + e)
            };
            proba[r * 2] = f64_to_host::<F>(1.0 - p1);
            proba[r * 2 + 1] = f64_to_host::<F>(p1);
        }
        Ok(DeviceArray::from_host(pool, &proba))
    }
}

impl<F> MBSGDClassifier<F>
where
    F: Float + CubeElement + Pod,
{
    /// Host-materialized signed decision margin `X·coef_ + intercept_` (length
    /// `n_query`), shared by `predict_labels` (sign) and `predict_proba`
    /// (sigmoid). Runs the on-device matvec GEMM, then broadcasts the scalar
    /// intercept host-side (the small predict geometry; the fitted state stays
    /// device-resident until here). Validates geometry / fitted-`n_features`
    /// (ASVS V5).
    fn decision_margin(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<f64>, AlgoError> {
        let (n_query, n_features) = shape;

        let coef = self.coef_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "mbsgd_classifier",
            operation: "predict",
        })?;
        let intercept = self.intercept_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "mbsgd_classifier",
            operation: "predict",
        })?;

        if n_query == 0 || n_features == 0 || x.len() != n_query * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_query,
                cols: n_features,
                len: x.len(),
            }));
        }
        if n_features != self.n_features {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: n_features,
                rhs: self.n_features,
            }));
        }

        let raw = gemm::<F>(
            pool,
            x,
            (n_query, n_features),
            coef,
            (n_features, 1),
            false,
            false,
            None,
        )?;
        let bias = host_to_f64(intercept.to_host(pool)[0]);
        let raw_host = raw.to_host(pool);
        raw.release_into(pool);

        Ok((0..n_query)
            .map(|r| host_to_f64(raw_host[r]) + bias)
            .collect())
    }
}

/// Lower a validated [`SgdConfig`] into the prim-local flat [`SgdParams`] the
/// PRIM-10 `sgd_solve` consumes (D-06; the prim cannot take the algos
/// `SgdConfig` — circular dependency, so the estimator lowers at the call site,
/// the cd_solve flat-scalar precedent). Shared by both SGD estimators.
pub(crate) fn lower_config(cfg: &SgdConfig) -> SgdParams {
    let loss = match cfg.loss {
        Loss::Hinge => SgdLoss::Hinge,
        Loss::Log => SgdLoss::Log,
        Loss::SquaredHinge => SgdLoss::SquaredHinge,
        Loss::SquaredLoss => SgdLoss::SquaredError,
        Loss::EpsilonInsensitive => SgdLoss::EpsilonInsensitive,
        Loss::SquaredEpsilonInsensitive => SgdLoss::SquaredEpsilonInsensitive,
    };
    let schedule = match cfg.learning_rate {
        LearningRate::Optimal => SgdSchedule::Optimal,
        LearningRate::InvScaling => SgdSchedule::InvScaling,
        LearningRate::Constant => SgdSchedule::Constant,
        LearningRate::Adaptive => SgdSchedule::Adaptive,
    };
    // The host applies the L1 cumulative soft-shrink only when the penalty
    // includes an L1 term (L1 or ElasticNet with l1_ratio > 0).
    let apply_l1 = match cfg.penalty {
        Penalty::L1 => true,
        Penalty::ElasticNet => true,
        Penalty::L2 => false,
    };
    // L2-only / ElasticNet lower `l1_ratio` straight through; a pure-L1 penalty is
    // the `l1_ratio = 1` case of the elastic-net shrink math the prim runs.
    let l1_ratio = match cfg.penalty {
        Penalty::L1 => 1.0,
        Penalty::L2 => 0.0,
        Penalty::ElasticNet => cfg.l1_ratio,
    };
    SgdParams {
        loss,
        schedule,
        alpha: cfg.alpha,
        l1_ratio,
        apply_l1,
        fit_intercept: cfg.fit_intercept,
        eta0: cfg.eta0,
        power_t: cfg.power_t,
        epsilon: cfg.epsilon,
        batch_size: cfg.batch_size,
        max_iter: cfg.max_iter,
        tol: cfg.tol,
    }
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine (mirrors the
/// `logistic.rs` helper).
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("mbsgd_classifier is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("mbsgd_classifier is f32/f64 only"),
    }
}
