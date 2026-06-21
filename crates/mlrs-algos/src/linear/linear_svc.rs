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

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::lbfgs::{lbfgs_minimize, LbfgsStopReason, LBFGS_FTOL, LBFGS_MAXLS};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::linear::sgd_config::{LearningRate, Loss, Penalty, SgdConfig};
use crate::traits::{Fit, PredictLabels};

/// Linear support-vector classifier (SGDSVM-03). Construct via
/// [`LinearSVC::builder`], then [`Fit::fit`] + [`PredictLabels::predict_labels`].
/// Fitted `coef_` (length `n_features`) / `intercept_` (length 1) are
/// device-resident (D-03).
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

    /// Host copy of the fitted `coef_` (length `n_features`). Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.coef_
            .as_ref()
            .map(|c| c.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "linear_svc",
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
                estimator: "linear_svc",
                operation: "intercept_",
            })
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
    /// (D-08, T-10-04-01). `C > 0` ([`BuildError::InvalidC`]) and the loss family
    /// must be valid for a CLASSIFIER ({`Hinge`, `SquaredHinge`} — a regression
    /// loss like `EpsilonInsensitive` is [`BuildError::InvalidLossForEstimator`]).
    /// Only `L1`/`L2` penalties are valid (sklearn `LinearSVC` has no `elasticnet`
    /// penalty). The `c`/`intercept_scaling` knobs are stored alongside the lowered
    /// [`SgdConfig`]; the L-BFGS fit maps `C` → the data-term weight internally.
    pub fn build<F>(self) -> Result<LinearSVC<F>, BuildError>
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
        })
    }
}

impl<F> Fit<F> for LinearSVC<F>
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
        let c = host_to_f64(self.c);
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

        self.classes_ = classes_;
        self.n_features = n_features;
        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept);
        Ok(self)
    }
}

impl<F> PredictLabels<F> for LinearSVC<F>
where
    F: Float + CubeElement + Pod,
{
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        let (n_query, n_features) = shape;

        let coef = self.coef_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "linear_svc",
            operation: "predict_labels",
        })?;
        let intercept = self.intercept_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "linear_svc",
            operation: "predict_labels",
        })?;

        // --- ASVS V5: geometry + fitted-n_features consistency. ---
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

        // margin = X·coef + intercept (the on-device matvec GEMM, then host bias).
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

        // sign of the margin selects the class: > 0 → classes_[1] (the +1 class),
        // else classes_[0] (the −1 class). Ties (margin == 0) break toward the
        // lower class, matching sklearn's `>= 0 → +1`? sklearn uses `decision >= 0`
        // → the +1 class; we mirror that with `>= 0`.
        let mut labels: Vec<i32> = vec![0i32; n_query];
        for r in 0..n_query {
            let m = host_to_f64(raw_host[r]) + bias;
            labels[r] = if m >= 0.0 {
                self.classes_[1] as i32
            } else {
                self.classes_[0] as i32
            };
        }
        Ok(DeviceArray::from_host(pool, &labels))
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
/// The per-iteration margin is the on-device `x̃ · w` matvec (GEMM); the gradient
/// `w − 2?` is `w + C·x̃ᵀ·g` with `gᵢ = dloss/dmargin` (a second GEMM, transa).
/// Both are read back so the (smooth, convex) host L-BFGS recursion can step —
/// the bounded-allocation iterative-solver shape (05-11), one metered readback per
/// eval.
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
    estimator: &'static str,
    margin_loss: impl Fn(f64, f64) -> (f64, f64),
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
    // Augmented feature count: append the synthetic constant column when fitting an
    // intercept (Pitfall 5). The augmented design lives device-resident for the
    // whole solve (reused every L-BFGS eval — bounded allocation, 05-11).
    let d_aug = if fit_intercept { n_features + 1 } else { n_features };

    let x_host = x.to_host(pool);
    let mut x_aug: Vec<F> = vec![F::from_int(0i64); n_samples * d_aug];
    for r in 0..n_samples {
        for col in 0..n_features {
            x_aug[r * d_aug + col] = x_host[r * n_features + col];
        }
        if fit_intercept {
            x_aug[r * d_aug + n_features] = f64_to_host::<F>(intercept_scaling);
        }
    }
    let x_aug_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_aug);

    // L-BFGS over the augmented weight vector w (length d_aug). The closure is
    // evaluated every iteration + per line-search step; a device GEMM failure is
    // captured (never panics across the boundary) and surfaced after the solve.
    let mut prim_err: Option<PrimError> = None;
    let closure = |w: &[f64]| -> (f64, Vec<f64>) {
        if prim_err.is_some() {
            return (f64::MAX, vec![0.0f64; d_aug]);
        }
        // margin m = X̃ · w  (n_samples × 1) via the on-device matvec.
        let w_host: Vec<F> = w.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let w_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &w_host);
        let margins = match gemm::<F>(
            pool,
            &x_aug_dev,
            (n_samples, d_aug),
            &w_dev,
            (d_aug, 1),
            false,
            false,
            None,
        ) {
            Ok(m) => m,
            Err(e) => {
                w_dev.release_into(pool);
                prim_err = Some(e);
                return (f64::MAX, vec![0.0f64; d_aug]);
            }
        };
        let margins_host = margins.to_host(pool);
        margins.release_into(pool);

        // Per-sample loss + dloss/dmargin (host — the SVM losses are cheap scalar
        // maps; the matvecs are the device work).
        let mut data_loss = 0.0f64;
        let mut g: Vec<F> = vec![F::from_int(0i64); n_samples];
        for i in 0..n_samples {
            let m = host_to_f64(margins_host[i]);
            let (li, dli) = margin_loss(m, targets[i]);
            data_loss += li;
            g[i] = f64_to_host::<F>(dli);
        }
        let g_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &g);

        // data-term gradient contribution C · X̃ᵀ · g  (d_aug × 1) via GEMM transa.
        // The LOGICAL op is (d_aug × n_samples)·(n_samples × 1); the stored X̃ is
        // (n_samples × d_aug) so transa presents the transposed view (gemm.rs:78).
        let xtg = match gemm::<F>(
            pool,
            &x_aug_dev,
            (d_aug, n_samples), // logical (m, k) AFTER the transpose
            &g_dev,
            (n_samples, 1),
            true, // transa: X̃ᵀ (d_aug × n_samples)
            false,
            None,
        ) {
            Ok(v) => v,
            Err(e) => {
                w_dev.release_into(pool);
                g_dev.release_into(pool);
                prim_err = Some(e);
                return (f64::MAX, vec![0.0f64; d_aug]);
            }
        };
        let xtg_host = xtg.to_host(pool);
        xtg.release_into(pool);
        w_dev.release_into(pool);
        g_dev.release_into(pool);

        // Total objective = ½‖w‖² + C·Σ ℓ ;  grad = w + C·X̃ᵀg.
        let mut reg = 0.0f64;
        for &wv in w.iter() {
            reg += wv * wv;
        }
        let loss = 0.5 * reg + c * data_loss;
        let mut grad = vec![0.0f64; d_aug];
        for j in 0..d_aug {
            grad[j] = w[j] + c * host_to_f64(xtg_host[j]);
        }
        (loss, grad)
    };

    let x0 = vec![0.0f64; d_aug];
    // gtol 1e-9 / a generous line-search budget so the convex objective reaches the
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
    let result = lbfgs_minimize(x0, closure, 1e-9, LBFGS_FTOL, LBFGS_MAXLS, max_iter)?;
    if let Some(e) = prim_err {
        x_aug_dev.release_into(pool);
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
        x_aug_dev.release_into(pool);
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

    x_aug_dev.release_into(pool);
    Ok((coef_dev, intercept_dev))
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine (mirrors the
/// `logistic.rs` / `elastic_net.rs` helper).
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("linear_svc is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("linear_svc is f32/f64 only"),
    }
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
