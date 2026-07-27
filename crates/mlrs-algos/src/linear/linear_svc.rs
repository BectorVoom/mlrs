//! `LinearSVC` (SGDSVM-03) — linear support-vector classifier, ≈
//! `sklearn.svm.LinearSVC`.
//!
//! ## Solver (Open Question Q1 RESOLVED — L-BFGS, NOT `cd_fit` reuse)
//! The Wave-0 scaffold and plan 10-04 hypothesized that the Lasso/ElasticNet
//! [`cd_fit`](crate::linear::coordinate_descent::cd_fit) soft-threshold
//! coordinate-descent could express the SVM objective. **It cannot.** `cd_fit`
//! solves the SQUARED-ERROR data term `½‖y − Xβ‖²`; sklearn's `LinearSVC`
//! minimizes the L2-regularized **squared-hinge** primal
//! `½‖w‖² + C·Σᵢ max(0, 1 − yᵢ·(xᵢ·w))²` — a different per-coordinate update
//! entirely (Open Q1 / RESEARCH §LinearSVC). The squared-hinge objective is
//! SMOOTH (C¹) and CONVEX, so the natural converged-optimum solver is the
//! validated 05-06 [`lbfgs_minimize`] primitive (option (b): a thin SVM solver
//! host-orchestrated over the device matvec), EXACTLY the `LogisticRegression`
//! precedent (05-10) — not the SGD prim, not the CD prim. An early Python spike
//! against the pinned fixture confirmed this objective reproduces sklearn's
//! `coef_`/`intercept_` (and EXACT predict labels) — see the 10-04 SUMMARY.
//!
//! ## C ↔ penalty + intercept_scaling (Pitfall 5)
//! `C` is the inverse-regularization strength (the `½‖w‖²` weight is 1, the data
//! term carries `C`). When `fit_intercept`, the intercept is handled by the
//! sklearn SYNTHETIC-FEATURE mechanism (Pitfall 5 — NOT the `cd_fit`
//! center-then-solve): a constant column of value `intercept_scaling` is appended
//! to the design, the augmented weight vector is solved with NO separate bias, and
//! `intercept_ = intercept_scaling · w_last`. The synthetic column IS penalized
//! (it is just another weight in `½‖w‖²`), which is precisely why a larger
//! `intercept_scaling` reduces the penalty's effect on the intercept.
//!
//! ## dual='auto' (D-07 — internal, never a builder knob)
//! sklearn resolves `dual='auto'` at fit: `if n_samples < n_features AND the
//! (loss, penalty) is dual-supported → dual else primal`. For the Phase-10
//! fixtures (`n_samples ≥ n_features`) it resolves to PRIMAL. mlrs always solves
//! the PRIMAL squared-hinge objective (the primal optimum equals the dual optimum
//! for this convex problem), and resolves the flag INTERNALLY for diagnostics —
//! it is NEVER exposed as a builder setter (D-07).
//!
//! ## Label encoding (Pitfall 4)
//! Binary labels are remapped to ±1 for the margin loss (copying the `logistic.rs`
//! `classes_` distinct-sorted pattern); `predict_labels` maps the margin sign back
//! through `classes_` so a non-contiguous label set returns the original id.
//!
//! Tests live in `crates/mlrs-algos/tests/linear_svc_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::lbfgs::{lbfgs_minimize, LbfgsStopReason, LBFGS_FTOL, LBFGS_MAXLS};
use mlrs_backend::prims::linear_predict::HostMirror;
use mlrs_backend::prims::svm_objective::SvmObjective;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::linear::elastic_net::{predict_linear, predict_linear_from_host};
use crate::linear::sgd_config::{LearningRate, Loss, Penalty, SgdConfig};
use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, Unfit};

/// Linear support-vector classifier (SGDSVM-03). Construct via
/// [`LinearSVC::builder`], then the consuming [`Fit::fit`] (returns the
/// `Fitted`-tagged sibling) + [`PredictLabels::predict_labels`]. Fitted `coef_`
/// (length `n_features`) / `intercept_` (length 1) are device-resident (D-03);
/// the host accessors exist ONLY on `LinearSVC<F, Fitted>` (the compile-time
/// typestate replaces the old runtime `NotFitted` guard, D-03).
pub struct LinearSVC<F, S = Unfit> {
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
    /// Memoized host copy of `(coef_, intercept_)` for the host-ingress
    /// `predict` path (IN-05 `OnceLock` mirror idiom). Empty until the first
    /// `predict_from_host` on the cpu backend, and never filled at all on the
    /// device backends — see
    /// [`HostMirror`](mlrs_backend::prims::linear_predict::HostMirror) for why a
    /// 64-byte read-back is worth caching. Fresh on every `fit`.
    predict_mirror: HostMirror<F>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> LinearSVC<F, Unfit>
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
}

impl<F> LinearSVC<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The lowered configuration (D-06).
    pub fn config(&self) -> &SgdConfig {
        &self.config
    }

    /// The inferred class labels (length 2 for the binary fit).
    pub fn classes(&self) -> &[i64] {
        &self.classes_
    }

    /// Host copy of the fitted `coef_` (length `n_features`). `Some` by
    /// construction on the `Fitted` state, so no `NotFitted` branch is needed
    /// (the compile-time typestate replaces the runtime guard, D-03).
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.coef_
            .as_ref()
            .expect("coef_ is Some by construction on LinearSVC<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `intercept_` (scalar). `Some` by construction on
    /// the `Fitted` state (D-03).
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> F {
        self.intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on LinearSVC<F, Fitted>")
            .to_host(pool)[0]
    }

    /// `predict_labels` for a test matrix still in the CALLER'S memory — the
    /// ingress the Arrow/PyO3 boundary actually has.
    ///
    /// A `LinearSVC` prediction is a decision-function sign test, and the
    /// decision function is exactly the `X·coef_ + intercept_` matvec the dense
    /// linear regressors compute — so this reuses their
    /// [`predict_linear_from_host`] path for the arithmetic and only adds the
    /// `classes_` lookup on top (D-03: the matvec is implemented once). That
    /// inherits the backend routing (cpu reads the caller's buffer in place;
    /// wgpu/cuda/rocm upload and run the fused device kernel) and the fused
    /// operand-finiteness verdict.
    ///
    /// It also removes two whole-result host↔device crossings the
    /// [`PredictLabels::predict_labels`] device-ingress path has to make for a
    /// host caller: the labels are produced ON the host, so they are not
    /// uploaded into an `i32` [`DeviceArray`] only for the binding to read them
    /// straight back.
    ///
    /// `values` is EMPTY when `operand_finite` is `false` — the caller is about
    /// to reject the input, so no labels are derived (mirrors
    /// [`HostPrediction`](mlrs_backend::prims::linear_predict::HostPrediction)).
    pub fn predict_labels_from_host(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<HostLabels, AlgoError> {
        let (_, n_features) = shape;
        // Checked BEFORE the shared matvec so a feature-count disagreement
        // reports LinearSVC's own operand-vs-fitted error, identically to the
        // device-ingress path (the shared path would otherwise report the same
        // mismatch with its arguments the other way round).
        if n_features != self.n_features {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: n_features,
                rhs: self.n_features,
            }));
        }
        let pred = predict_linear_from_host(
            self.coef_.as_ref(),
            self.intercept_.as_ref(),
            &self.predict_mirror,
            "linear_svc",
            pool,
            x,
            shape,
        )?;
        if !pred.operand_finite {
            return Ok(HostLabels {
                values: Vec::new(),
                operand_finite: false,
            });
        }
        Ok(HostLabels {
            values: self.labels_from_margins(&pred.values),
            operand_finite: true,
        })
    }

    /// Map decision-function values to class ids: `>= 0` selects `classes_[1]`
    /// (the `+1` class), anything else `classes_[0]` — sklearn's
    /// `classes_[(decision > 0).astype(int)]` with its `>= 0` tie-break, through
    /// the stored `classes_` so a non-contiguous label set returns the original
    /// ids (Pitfall 4).
    ///
    /// Shared by both ingresses so the encoding is written once.
    fn labels_from_margins(&self, margins: &[F]) -> Vec<i32> {
        let neg = self.classes_[0] as i32;
        let pos = self.classes_[1] as i32;
        margins
            .iter()
            .map(|&m| if host_to_f64(m) >= 0.0 { pos } else { neg })
            .collect()
    }
}

/// What [`LinearSVC::predict_labels_from_host`] produces: the class ids, plus
/// whether every element of the operand it read was finite.
///
/// The `i32` label twin of
/// [`HostPrediction`](mlrs_backend::prims::linear_predict::HostPrediction) —
/// same contract, including that `values` is only meaningful when
/// `operand_finite` is `true`.
#[derive(Debug, Clone)]
pub struct HostLabels {
    /// The length-`n_query` predicted class ids, drawn from `classes_`.
    pub values: Vec<i32>,
    /// `false` if ANY element of `x` was NaN or ±infinity.
    pub operand_finite: bool,
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
    /// (D-08, T-10-04-01). `C > 0` ([`BuildError::InvalidC`]) and the loss family
    /// must be valid for a CLASSIFIER ({`Hinge`, `SquaredHinge`} — a regression
    /// loss like `EpsilonInsensitive` is [`BuildError::InvalidLossForEstimator`]).
    /// Only `L1`/`L2` penalties are valid (sklearn `LinearSVC` has no `elasticnet`
    /// penalty). The `c`/`intercept_scaling` knobs are stored alongside the lowered
    /// [`SgdConfig`]; the L-BFGS fit maps `C` → the data-term weight internally.
    pub fn build<F>(self) -> Result<LinearSVC<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        // --- T-10-04-01 / ASVS V5: validate the data-INDEPENDENT hyperparameters
        //     at build() BEFORE any data is seen (D-08). ---
        if !(self.c > 0.0) {
            return Err(BuildError::InvalidC {
                estimator: "linear_svc",
                c: self.c,
            });
        }
        match self.loss {
            Loss::Hinge | Loss::SquaredHinge => {}
            other => {
                return Err(BuildError::InvalidLossForEstimator {
                    estimator: "linear_svc",
                    loss: other.name().to_string(),
                });
            }
        }
        match self.penalty {
            Penalty::L1 | Penalty::L2 => {}
            Penalty::ElasticNet => {
                return Err(BuildError::UnknownPenalty {
                    value: "elasticnet (LinearSVC supports only l1/l2)".to_string(),
                });
            }
        }
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
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for LinearSVC<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = LinearSVC<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<LinearSVC<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-10-04-02 / ASVS V5: data-DEPENDENT geometry guard BEFORE any
        //     launch (the data-INDEPENDENT params were validated at build()). ---
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "linear_svc",
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
        //     ±1 remap for the margin loss. A non-binary target is out of scope for
        //     the linear-SVM binary classifier (sklearn LinearSVC is OvR multiclass;
        //     Phase-10 scope is binary — A6). ---
        let y_host = y.to_host(pool);
        let mut raw_labels: Vec<i64> = Vec::with_capacity(n_samples);
        for &yv in y_host.iter() {
            let lf = host_to_f64(yv);
            let li = lf.round();
            if (li - lf).abs() > 1e-6 {
                return Err(AlgoError::InvalidLabels {
                    estimator: "linear_svc",
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
                estimator: "linear_svc",
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
                    estimator: "linear_svc",
                    reason: format!(
                        "class label {cls} does not fit in i32 \
                         (predicted labels are i32)"
                    ),
                });
            }
        }
        // classes_[0] → −1, classes_[1] → +1 (sklearn maps the higher class to +1).
        let yp: Vec<f64> = raw_labels
            .iter()
            .map(|&l| if l == classes_[1] { 1.0 } else { -1.0 })
            .collect();

        // --- D-07: resolve dual='auto' INTERNALLY (never a builder knob). For the
        //     squared-hinge primal we always solve the primal (its optimum equals
        //     the dual's); the flag is computed only for fidelity to sklearn's
        //     resolution rule (and would route a sparse/dual path in a future
        //     extension). n_samples >= n_features → primal here. ---
        let _dual = n_samples < n_features; // false for the Phase-10 fixtures.

        // --- The L2-regularized squared-hinge primal, minimized by L-BFGS over the
        //     synthetic-feature-augmented design (Pitfall 5 intercept). The data
        //     term carries `C`; the regularizer is the plain ½‖w‖² (synthetic
        //     column included). The per-sample margin loss/grad is squared-hinge:
        //       z = 1 − yᵢ·mᵢ ;  ℓ = max(0, z)² ;  dℓ/dmᵢ = −2·yᵢ·max(0, z). ---
        // IN-03: `self.c` is already `f64`; use it directly (no identity cast).
        let c = self.c;
        let (coef, intercept) = svm_lbfgs_fit::<F>(
            pool,
            x,
            &yp,
            n_samples,
            n_features,
            c,
            self.intercept_scaling,
            self.config.fit_intercept,
            self.config.max_iter,
            self.config.tol,
            "linear_svc",
            |margin, target| {
                // target is ±1; squared-hinge in the margin m = target·pred form
                // expressed via z = 1 − target·m.
                let z = 1.0 - target * margin;
                if z > 0.0 {
                    (z * z, -2.0 * target * z) // (loss_i, dloss/dmargin)
                } else {
                    (0.0, 0.0)
                }
            },
        )?;

        Ok(LinearSVC {
            config: self.config,
            c: self.c,
            intercept_scaling: self.intercept_scaling,
            classes_,
            n_features,
            coef_: Some(coef),
            intercept_: Some(intercept),
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }
}

impl<F> PredictLabels<F> for LinearSVC<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        let (_, n_features) = shape;

        // --- ASVS V5: fitted-n_features consistency. The remaining geometry
        // checks (`n_query`/`n_features` non-zero, `x.len()` consistent) live in
        // the shared `predict_linear` below and raise the identical
        // `ShapeMismatch`; this one stays here because it compares against
        // LinearSVC's own fitted `n_features`. ---
        if n_features != self.n_features {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: n_features,
                rhs: self.n_features,
            }));
        }

        // decision = X·coef + intercept, through the SAME fused matvec+bias prim
        // the dense linear regressors predict with (D-03) — one launch, no
        // separate `intercept.to_host()` + host bias loop. `coef_`/`intercept_`
        // are `Some` by construction on the `Fitted` state (the compile-time
        // typestate replaces the old runtime `NotFitted` guard, D-03), so the
        // shared path's `NotFitted` arm is unreachable from here.
        let raw = predict_linear(
            self.coef_.as_ref(),
            self.intercept_.as_ref(),
            "linear_svc",
            pool,
            x,
            shape,
        )?;
        let raw_host = raw.to_host(pool);
        raw.release_into(pool);

        Ok(DeviceArray::from_host(
            pool,
            &self.labels_from_margins(&raw_host),
        ))
    }
}

/// Shared L-BFGS host fit for the linear-SVM primal objectives (LinearSVC
/// squared-hinge AND LinearSVR squared-epsilon-insensitive, SGDSVM-03/04). Open
/// Question Q1 RESOLUTION: the SVM losses are SMOOTH + CONVEX but NOT the
/// Lasso/ElasticNet soft-threshold CD objective, so they reuse the validated 05-06
/// [`lbfgs_minimize`] primitive (option (b) — a thin SVM solver host-orchestrated
/// over the device matvec), the `logistic.rs` L-BFGS precedent.
///
/// Minimizes `½‖w‖² + C·Σᵢ ℓ(mᵢ, tᵢ)` where `mᵢ = (x̃ᵢ·w)` is the margin on the
/// SYNTHETIC-FEATURE-augmented design `x̃ = [x | intercept_scaling]` (Pitfall 5 —
/// NOT center-then-solve) and `ℓ` is the caller's per-sample
/// `(loss_i, dloss/dmargin)` closure (squared-hinge for SVC, squared-eps for SVR).
/// `tᵢ` is the per-sample target (±1 label for SVC, the regression target for
/// SVR). Returns the device-resident `(coef_, intercept_)`: `coef_` is the
/// first `n_features` augmented weights, `intercept_ = intercept_scaling · w_last`
/// (length 1). When `fit_intercept` is false the design is NOT augmented and the
/// intercept is 0.
///
/// The per-iteration `(Σℓ, x̃ᵀg)` evaluation — the entire cost of the solve — is
/// [`SvmObjective`], which owns the backend routing: two GEMM launches with a
/// host round-trip on each side for wgpu/cuda/rocm, and ONE fused `-O3` host
/// pass over the caller's slab on cpu (where a cubecl launch costs three orders
/// of magnitude more than the matvec it performs — see the prim's module docs
/// for the measured breakdown). This function owns only the regularizer, the
/// `C` weight and the L-BFGS driver, so the objective is written once and the
/// per-backend arithmetic once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn svm_lbfgs_fit<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    targets: &[f64],
    n_samples: usize,
    n_features: usize,
    c: f64,
    intercept_scaling: f64,
    fit_intercept: bool,
    max_iter: usize,
    gtol: f64,
    estimator: &'static str,
    margin_loss: impl Fn(f64, f64) -> (f64, f64) + Sync,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    ),
    AlgoError,
>
where
    F: Float + CubeElement + Pod,
{
    // The design in the form the active backend evaluates against, prepared ONCE
    // and reused by every L-BFGS iteration and line-search step (the
    // bounded-allocation iterative-solver shape, 05-11). `d_aug` counts the
    // synthetic intercept column when there is one (Pitfall 5).
    let objective = SvmObjective::<F>::new(
        pool,
        x,
        (n_samples, n_features),
        targets.to_vec(),
        intercept_scaling,
        fit_intercept,
    )
    .map_err(AlgoError::Prim)?;
    let d_aug = objective.d_aug();

    // L-BFGS over the augmented weight vector w (length d_aug). The closure is
    // evaluated every iteration + per line-search step; an evaluation failure is
    // captured (never panics across the boundary) and surfaced after the solve.
    let mut prim_err: Option<PrimError> = None;
    let closure = |w: &[f64]| -> (f64, Vec<f64>) {
        if prim_err.is_some() {
            return (f64::MAX, vec![0.0f64; d_aug]);
        }
        let ev = match objective.eval(pool, w, &margin_loss) {
            Ok(ev) => ev,
            Err(e) => {
                prim_err = Some(e);
                return (f64::MAX, vec![0.0f64; d_aug]);
            }
        };

        // Total objective = ½‖w‖² + C·Σ ℓ ;  grad = w + C·X̃ᵀg.
        let mut reg = 0.0f64;
        for &wv in w.iter() {
            reg += wv * wv;
        }
        let loss = 0.5 * reg + c * ev.data_loss;
        let mut grad = ev.xtg;
        for (j, gj) in grad.iter_mut().enumerate() {
            *gj = w[j] + c * *gj;
        }
        (loss, grad)
    };

    let x0 = vec![0.0f64; d_aug];
    // gtol = the caller-configured `tol` (clamped, WR-01) / a generous line-search
    // budget so the convex objective reaches the
    // converged optimum the liblinear oracle compares against (the SVM objective is
    // strictly convex — a unique global minimum, like the lbfgs convex-quadratic
    // standalone gate — so a deep solve lands ON the optimum, not past it). In f64
    // gtol=1e-9 is reachable; in f32 the achievable `max|grad|` is pinned to a
    // dtype-precision FLOOR (round-off in the matvec accumulations), so gtol can
    // never fire and the strong-Wolfe line search instead BREAKS DOWN at the floor —
    // exactly the `logistic.rs` precision-floor accept (05-10). We therefore accept
    // a line-search breakdown / cap as converged when the residual `max|grad|` is at
    // or below the dtype floor `k·sqrt(eps_F)` (the smallest gradient a flat-near-
    // minimum float loss can resolve); a residual ABOVE the floor is a genuine
    // non-stationary breakdown and stays `NotConverged` (T-10-04-03 DoS signal).
    // (On the cpu backend the f32 floor is much lower than `f_epsilon::<f32>()`
    // implies, because `SvmObjective`'s host arm accumulates in f64 whatever `F`
    // is — the floor accept then simply never has to fire. It is kept as the
    // device backends' f32 path still accumulates in `F`.)
    // WR-01: thread the caller-configured L-BFGS gradient tolerance through
    // (sklearn `tol`), clamped to a sane positive floor so a `tol = 0` (the pinned
    // deterministic-epochs oracle override) still requests a deep converged solve
    // rather than gtol=0 (which can never fire).
    let gtol = gtol.max(1e-12);
    let result = lbfgs_minimize(x0, closure, gtol, LBFGS_FTOL, LBFGS_MAXLS, max_iter)?;
    if let Some(e) = prim_err {
        objective.release_into(pool);
        return Err(AlgoError::Prim(e));
    }

    // Dtype precision floor for the convex-minimum residual gradient: f32 ≈
    // 1.7e-4, f64 ≈ 7.5e-9 (the `logistic.rs` GAUGE_FLOOR_K·sqrt(eps) shape; here
    // there is no gauge null-space, just float round-off near the unique minimum).
    let floor_accept = 0.5 * f_epsilon::<F>().sqrt();
    let residual_ok = result.max_grad <= floor_accept;
    let broke = result.stop_reason == LbfgsStopReason::LineSearchFailed && !residual_ok;
    let hit_cap = result.iters >= max_iter && !result.converged && !residual_ok;
    if hit_cap || broke {
        objective.release_into(pool);
        return Err(AlgoError::NotConverged {
            estimator,
            max_iter,
        });
    }

    // Recover coef_ (first n_features augmented weights) and
    // intercept_ = intercept_scaling · w_last (Pitfall 5).
    let coef_host: Vec<F> = result.x[..n_features]
        .iter()
        .map(|&v| f64_to_host::<F>(v))
        .collect();
    let intercept = if fit_intercept {
        intercept_scaling * result.x[n_features]
    } else {
        0.0
    };
    let coef_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &coef_host);
    let intercept_dev: DeviceArray<ActiveRuntime, F> =
        DeviceArray::from_host(pool, &[f64_to_host::<F>(intercept)]);

    objective.release_into(pool);
    Ok((coef_dev, intercept_dev))
}

/// Machine epsilon of `F` (f32 / f64) as `f64`, for the convex-minimum residual
/// precision floor `k·sqrt(eps_F)` (the `logistic.rs` precision-floor helper).
fn f_epsilon<F: Pod>() -> f64 {
    match size_of::<F>() {
        4 => f32::EPSILON as f64,
        8 => f64::EPSILON,
        _ => unreachable!("linear_svc is f32/f64 only"),
    }
}
