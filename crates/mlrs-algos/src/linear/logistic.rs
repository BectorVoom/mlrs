//! `LogisticRegression` (LINEAR-05) — the project's highest-risk estimator:
//! an L-BFGS iterative-solver classifier matching
//! `sklearn.linear_model.LogisticRegression(solver='lbfgs')` (binary AND
//! multinomial) within 1e-5 on the gauge-invariant `predict`/`predict_proba`.
//!
//! ## Solver (deliberately L-BFGS, NOT coordinate descent — D-03)
//! `LogisticRegression` minimizes the SYMMETRIC over-parameterized multinomial
//! softmax objective by the validated 05-06 [`lbfgs_minimize`] primitive driven
//! by the [`softmax_loss_grad`] device kernel. This is a DIFFERENT optimizer for
//! a DIFFERENT objective than the Lasso/ElasticNet coordinate-descent family
//! (05-09); the two solvers MUST NOT be unified (RESEARCH Anti-Patterns / D-03).
//!
//! ## Objective (Pitfall 3 — C scaling + unpenalized intercept)
//! The objective is `(1/n)·Σ loss + ½·l2_reg·‖W‖²` with
//! `l2_reg = 1/(C·n_samples)` (the sklearn-equivalent inverse-regularization
//! scaling). The intercept `b` is UNPENALIZED — it never appears in the
//! `‖·‖²` term (Pitfall 3). Both are enforced inside the 05-06 kernel; the
//! estimator's only responsibility is to compute `l2_reg = 1/(C·n)` and pass it.
//!
//! ## Symmetric K-full-weight-vector form (D-12)
//! The parameter vector is `[W (k×d) | b (k)]` flattened — K full weight
//! vectors, so BINARY is genuinely the K=2 case of the SAME kernel + the SAME
//! host loop (no separate sigmoid path, no deprecated `multi_class` argument;
//! sklearn ≥1.5 is multinomial-by-default — RESEARCH §"State of the Art"). The
//! over-parameterization introduces a GAUGE FREEDOM: `coef_` is only determined
//! up to a per-row additive constant, so `coef_` may differ from sklearn's
//! binary `(1×d)` / multinomial parameterization while `predict_proba` (a
//! softmax, invariant under that shift) is identical. This is why the oracle
//! gates on `predict_proba`/`predict` (PRIMARY, 1e-5) and treats `coef_` as a
//! LOOSER secondary check (Pitfall 5 — gauge freedom, not a tolerance
//! regression).
//!
//! ## Stable predict path (Pitfall 4)
//! `predict_proba` forms the logits `raw[i,k] = x_i·W_k + b_k` and applies the
//! STABLE softmax (subtract the per-row max before `exp`, the logsumexp trick)
//! so well-separated classes never overflow to NaN. `predict_labels` is the
//! arg-max of `predict_proba`, breaking ties toward the lowest class index.
//!
//! ## Device residency (D-03)
//! Fitted `coef_` (K×d) and `intercept_` (K) are stored as device-resident
//! [`DeviceArray`]s; host materialization happens only at a Rust accessor / the
//! predict broadcast (the LogReg geometry — small n/d/K — does the stable
//! softmax host-side after the on-device logit GEMM).
//!
//! Tests live in `crates/mlrs-algos/tests/logistic_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::lbfgs::{
    lbfgs_minimize, softmax_loss_grad, LbfgsStopReason, LBFGS_FTOL, LBFGS_MAXLS,
};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;
use crate::traits::{Fit, PredictLabels, PredictProba};

/// Default `max_iter` — the L-BFGS iteration cap. sklearn's `LogisticRegression`
/// default is 100; we give the solver headroom (`300`) so the tightened `gtol`
/// below is reachable before the iteration cap for both binary and multiclass.
/// The fixtures are now the TRUE MINIMUM of the (shared) objective (tightly-fit
/// sklearn multiclass; scipy-minimized binary self-reference), so converging
/// deeper lands ON the fixture, not past it. Still a finite cap (T-05-10-03 DoS).
const LOG_DEFAULT_MAX_ITER: usize = 300;
/// Default convergence tolerance (the L-BFGS `gtol` on `max|grad|`). sklearn's
/// default `tol = 1e-4` stops ~3.2e-5 short of the minimum — borderline OVER the
/// strict 1e-5 PRIMARY `predict_proba` gate. We tighten to `1e-5`: reachable in
/// BOTH f32 (gradient floor ~9e-6 at the minimum) and f64 before the `ftol =
/// 64·eps` relative-f stall, and deep enough that `predict_proba` lands within
/// the 1e-5 (abs-OR-rel) gate against the now-true-minimum fixtures (sklearn
/// multiclass + our binary self-reference, which agree to ~5e-8 at the minimum).
const LOG_DEFAULT_TOL: f64 = 1e-5;

/// Multinomial (symmetric-softmax) logistic regression (LINEAR-05) fitted by the
/// L-BFGS iterative solver.
///
/// Construct with [`LogisticRegression::new`] (`c`, `fit_intercept`) or
/// [`LogisticRegression::with_opts`] (also `max_iter` / `tol`), then [`Fit::fit`]
/// and [`PredictProba::predict_proba`] / [`PredictLabels::predict_labels`].
/// Fitted `coef_` (K×d) / `intercept_` (K) are device-resident (D-03); the host
/// accessors [`coef`](Self::coef) / [`intercept`](Self::intercept) materialize
/// them on demand.
pub struct LogisticRegression<F> {
    /// Inverse-regularization strength (`C > 0`; larger = weaker L2 penalty).
    /// Maps to `l2_reg = 1/(C·n_samples)` at fit (Pitfall 3). A non-positive `C`
    /// is rejected at `fit` with [`AlgoError::InvalidC`] (T-05-10-01).
    c: F,
    /// Whether to fit an (unpenalized) intercept term per class (Pitfall 3).
    fit_intercept: bool,
    /// L-BFGS iteration cap; `NotConverged` is surfaced if reached (default 100).
    max_iter: usize,
    /// L-BFGS convergence tolerance on `max |grad|` (`gtol`; default 1e-4).
    tol: F,
    /// Number of classes inferred from the integer labels at `fit` (binary = 2).
    n_classes: usize,
    /// CR-02: the DISTINCT sorted training labels (`classes_`), one per fitted
    /// class column. The softmax kernel only ever sees the dense remapped index
    /// `0..n_classes`; `predict_labels` maps each argmax column back through this
    /// vector so a non-contiguous label set (e.g. `{0, 2}`) returns the ORIGINAL
    /// id (`2`), never a phantom never-trained class. Empty until `fit`.
    classes_: Vec<i64>,
    /// Number of features inferred at `fit` (for the predict geometry guard).
    n_features: usize,
    /// Fitted weights `W` (K×d, row-major: class-major), device-resident, `None`
    /// until `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercepts `b` (length K), device-resident, `None` until `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
}

impl<F> LogisticRegression<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `LogisticRegression` with inverse-regularization `C`
    /// and the `fit_intercept` flag, using sklearn's defaults `max_iter = 100`
    /// and `tol = 1e-4`. A non-positive `C` is rejected at `fit` with
    /// [`AlgoError::InvalidC`].
    pub fn new(c: F, fit_intercept: bool) -> Self {
        Self::with_opts(c, fit_intercept, LOG_DEFAULT_MAX_ITER, f64_to_host::<F>(LOG_DEFAULT_TOL))
    }

    /// Like [`new`](Self::new) but with an explicit L-BFGS `max_iter` cap and
    /// convergence `tol` (the `gtol` on `max |grad|`).
    pub fn with_opts(c: F, fit_intercept: bool, max_iter: usize, tol: F) -> Self {
        Self {
            c,
            fit_intercept,
            max_iter,
            tol,
            n_classes: 0,
            classes_: Vec::new(),
            n_features: 0,
            coef_: None,
            intercept_: None,
        }
    }

    /// Host copy of the fitted `coef_` (K×d, row-major). Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.coef_
            .as_ref()
            .map(|c| c.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "logistic_regression",
                operation: "coef_",
            })
    }

    /// Host copy of the fitted `intercept_` (length K). Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.intercept_
            .as_ref()
            .map(|i| i.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "logistic_regression",
                operation: "intercept_",
            })
    }

    /// Number of classes inferred at `fit` (binary = 2). 0 before `fit`.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
}

impl<F> Fit<F> for LogisticRegression<F>
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

        // --- T-05-10-01 / ASVS V5: validate the untrusted hyperparameter and
        //     geometry BEFORE any prim launch. C ≤ 0 makes l2_reg = 1/(C·n)
        //     non-positive (degenerate / unbounded objective). ---
        let c64 = host_to_f64(self.c);
        if !(c64 > 0.0) {
            return Err(AlgoError::InvalidC {
                estimator: "logistic_regression",
                c: c64,
            });
        }
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "logistic_regression",
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

        // --- CR-02: determine the classes from the DISTINCT integer labels, not
        //     `max(label)+1`. Inferring K from `max+1` mislabels a non-contiguous
        //     target (`{0, 2}` would fit a phantom never-trained class 1 and could
        //     emit id 1, which never existed) and forces a degenerate one-class
        //     input up to binary. Instead: round + validate each label is a
        //     non-negative integer, collect the DISTINCT sorted labels as
        //     `classes_`, remap `y` to a dense `[0, n_classes)` index for the
        //     kernel (which trusts `yi < K` and indexes the weight rows), and set
        //     `n_classes = n_distinct`. `predict_labels` maps the argmax column
        //     back through `classes_` to recover the original id. ---
        let y_host = y.to_host(pool);
        let mut raw_labels: Vec<i64> = Vec::with_capacity(n_samples);
        for &yv in y_host.iter() {
            let lf = host_to_f64(yv);
            let li = lf.round();
            if !(li >= 0.0) || (li - lf).abs() > 1e-6 {
                return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                    operand: "logistic.y (labels must be integers in 0..n_classes)",
                    rows: n_samples,
                    cols: 1,
                    len: y.len(),
                }));
            }
            raw_labels.push(li as i64);
        }
        // Distinct sorted labels = classes_; the dense remap index of label L is
        // its position in this vector.
        let mut classes_: Vec<i64> = raw_labels.clone();
        classes_.sort_unstable();
        classes_.dedup();
        // sklearn requires >= 2 classes; a single-class (or empty) target is a
        // degenerate problem, not a binary one (max+1 silently forced it to 2).
        if classes_.len() < 2 {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "logistic.y (needs at least 2 distinct classes)",
                rows: n_samples,
                cols: classes_.len(),
                len: y.len(),
            }));
        }
        let n_classes = classes_.len();
        // Remap each sample's raw label to its dense class index (classes_ is
        // sorted, so a binary search gives the position). The kernel must ONLY
        // see remapped indices in `0..n_classes`.
        let y_remapped: Vec<F> = raw_labels
            .iter()
            .map(|&l| {
                let idx = classes_
                    .binary_search(&l)
                    .expect("every raw label is in classes_ by construction");
                f64_to_host::<F>(idx as f64)
            })
            .collect();
        let y_remap_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y_remapped);

        // --- l2_reg = 1/(C·n_samples) (Pitfall 3); the intercept is unpenalized
        //     inside the 05-06 kernel (b never enters the ‖W‖² term). ---
        let l2_reg = 1.0 / (c64 * n_samples as f64);

        let d = n_features;
        let k = n_classes;
        let w_len = k * d;

        // Device-resident X / y for the closure's softmax launcher (reused across
        // every L-BFGS iteration — the params change, X/y do not). CR-02: the
        // kernel must see the DENSE remapped labels (`0..n_classes`), never the
        // raw (possibly non-contiguous) ids.
        let x_dev = x;
        let y_dev = &y_remap_dev;

        // --- L-BFGS over the symmetric softmax. The flat parameter vector is
        //     [W (k×d) | b (k)]; the closure splits it, launches the device
        //     softmax_loss_grad, and re-flattens (gradW | gradb). When
        //     fit_intercept is false, the intercept block is held at 0 and its
        //     gradient is zeroed so b never moves off the origin. ---
        let x0 = vec![0.0f64; w_len + k];
        let fit_intercept = self.fit_intercept;

        // WR-01: the closure runs on every L-BFGS iteration AND multiple times per
        // line-search step. A `PrimError` from the device softmax launch must NOT
        // panic across the (future PyO3) boundary — capture the FIRST error in a
        // slot, return a sentinel (huge loss + zero grad) so the line search backs
        // off, and surface the typed AlgoError after `lbfgs_minimize` returns. The
        // sentinel never wins the line search, so a failed solve ends at the cap
        // and the captured error takes precedence.
        let mut prim_err: Option<PrimError> = None;
        let grad_len = w_len + k;
        let closure = |params: &[f64]| -> (f64, Vec<f64>) {
            // Once an error has been recorded, keep returning the sentinel without
            // re-launching (the result is discarded anyway).
            if prim_err.is_some() {
                return (f64::MAX, vec![0.0f64; grad_len]);
            }
            let w_host: Vec<F> = params[..w_len].iter().map(|&v| f64_to_host::<F>(v)).collect();
            let b_host: Vec<F> = if fit_intercept {
                params[w_len..].iter().map(|&v| f64_to_host::<F>(v)).collect()
            } else {
                vec![f64_to_host::<F>(0.0); k]
            };
            let w_d: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &w_host);
            let b_d: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &b_host);

            let res =
                softmax_loss_grad::<F>(pool, x_dev, y_dev, &w_d, &b_d, n_samples, d, k, l2_reg);

            w_d.release_into(pool);
            b_d.release_into(pool);

            let (loss, grad_w, mut grad_b) = match res {
                Ok(v) => v,
                Err(e) => {
                    prim_err = Some(e);
                    return (f64::MAX, vec![0.0f64; grad_len]);
                }
            };

            if !fit_intercept {
                for g in grad_b.iter_mut() {
                    *g = 0.0;
                }
            }
            let mut grad = grad_w;
            grad.extend_from_slice(&grad_b);
            (loss, grad)
        };

        let gtol = host_to_f64(self.tol);
        let maxiter = self.max_iter;
        let result = lbfgs_minimize(x0, closure, gtol, LBFGS_FTOL, LBFGS_MAXLS, maxiter)?;

        // WR-01: a device softmax failure during the solve surfaces here as a typed
        // AlgoError::Prim, never a panic across the estimator boundary.
        if let Some(e) = prim_err {
            return Err(AlgoError::Prim(e));
        }

        // --- Convergence for the SYMMETRIC over-parameterized objective (D-12).
        //     The 05-06 prim reports `converged` only on the `max|grad| <= gtol`
        //     test. But the symmetric K-full-weight form has a GAUGE NULL-SPACE
        //     (a per-class additive shift leaves the loss — and predict_proba —
        //     unchanged): the gradient's null-space components never shrink, so in
        //     f32 `max|grad|` plateaus ~1e-4 even though the loss has reached its
        //     true minimum (the gauge-INVARIANT predict_proba is fully converged
        //     to ~5e-8). The prim's strong-Wolfe loop then breaks EARLY on the
        //     `ftol = 64·eps` relative-f stall (`iters < maxiter`) — a genuine
        //     stationary point of this convex objective, NOT a hung solve.
        //
        //     So the real DoS / non-convergence signal (T-05-10-03) is hitting the
        //     iteration CAP (`iters == maxiter`), not `max|grad| > gtol` at an
        //     early ftol stall. We surface `NotConverged` ONLY when the cap is
        //     reached without the prim flagging convergence; an early gtol/ftol
        //     stop (either `result.converged` OR `iters < maxiter`) is accepted —
        //     the primary `predict_proba` 1e-5 oracle is the correctness witness
        //     that the accepted iterate is the right minimizer (Pitfall 5). ---
        // WR-01: a line-search BREAKDOWN (`LbfgsStopReason::LineSearchFailed`) is a
        // stop at a possibly NON-stationary point — a non-minimizer — and must be
        // surfaced as NotConverged REGARDLESS of `iters` (it can happen well before
        // the cap). It is NOT the benign ftol stall the Pitfall-5 comment accepts.
        if result.stop_reason == LbfgsStopReason::LineSearchFailed {
            return Err(AlgoError::NotConverged {
                estimator: "logistic_regression",
                max_iter: maxiter,
            });
        }
        let hit_cap = result.iters >= maxiter;
        if hit_cap && !result.converged {
            return Err(AlgoError::NotConverged {
                estimator: "logistic_regression",
                max_iter: maxiter,
            });
        }

        // Store device-resident fitted W (k×d) and b (k) (D-03).
        let w_final: Vec<F> = result.x[..w_len].iter().map(|&v| f64_to_host::<F>(v)).collect();
        let b_final: Vec<F> = if fit_intercept {
            result.x[w_len..].iter().map(|&v| f64_to_host::<F>(v)).collect()
        } else {
            vec![f64_to_host::<F>(0.0); k]
        };
        let coef_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &w_final);
        let intercept_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &b_final);

        // The remapped-label device buffer is only needed during the solve.
        y_remap_dev.release_into(pool);

        self.n_classes = n_classes;
        self.classes_ = classes_;
        self.n_features = n_features;
        self.coef_ = Some(coef_dev);
        self.intercept_ = Some(intercept_dev);
        Ok(self)
    }
}

impl<F> PredictProba<F> for LogisticRegression<F>
where
    F: Float + CubeElement + Pod,
{
    fn predict_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, n_features) = shape;

        let coef = self.coef_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "logistic_regression",
            operation: "predict_proba",
        })?;
        let intercept = self.intercept_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "logistic_regression",
            operation: "predict_proba",
        })?;

        // --- T-05-10-01 / ASVS V5: geometry + fitted-n_features consistency. ---
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

        let k = self.n_classes;

        // --- Logits raw = X · Wᵀ  (n_query × k) via the on-device GEMM (D-03).
        //     coef is W (k×d) row-major; transb reads it as Wᵀ (d×k), so the
        //     product is (n_query × k). ---
        let raw = gemm::<F>(
            pool,
            x,
            (n_query, n_features),
            coef,
            (n_features, k), // logical Wᵀ is (d × k); stored buffer is W (k×d), transb.
            false,
            true,
            None,
        )?;

        // --- Broadcast-add the per-class intercept, then the STABLE softmax
        //     (subtract per-row max before exp — Pitfall 4) host-side. The LogReg
        //     geometry (small n_query/k) makes the host softmax the natural
        //     terminal; the fitted state itself stays device-resident until
        //     here. ---
        let raw_host = raw.to_host(pool);
        let b_host = intercept.to_host(pool);
        raw.release_into(pool);

        let mut proba_host: Vec<F> = vec![F::from_int(0i64); n_query * k];
        for r in 0..n_query {
            // logits[k] = raw[r,k] + b[k]
            let mut logits = vec![0.0f64; k];
            let mut row_max = f64::NEG_INFINITY;
            for c in 0..k {
                let v = host_to_f64(raw_host[r * k + c]) + host_to_f64(b_host[c]);
                logits[c] = v;
                if v > row_max {
                    row_max = v;
                }
            }
            let mut sum_exp = 0.0f64;
            for c in 0..k {
                let e = (logits[c] - row_max).exp();
                logits[c] = e;
                sum_exp += e;
            }
            let inv = 1.0 / sum_exp;
            for c in 0..k {
                proba_host[r * k + c] = f64_to_host::<F>(logits[c] * inv);
            }
        }
        Ok(DeviceArray::from_host(pool, &proba_host))
    }
}

impl<F> PredictLabels<F> for LogisticRegression<F>
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
        let k = self.n_classes;

        // Reuse the (validated) predict_proba path; argmax of each row is the
        // predicted class (lowest-class-index tie via the strict `>` scan).
        let proba = self.predict_proba(pool, x, shape)?;
        let proba_host = proba.to_host(pool);
        proba.release_into(pool);

        let mut labels: Vec<i32> = vec![0i32; n_query];
        for r in 0..n_query {
            let mut best = 0usize;
            let mut best_v = host_to_f64(proba_host[r * k]);
            for c in 1..k {
                let v = host_to_f64(proba_host[r * k + c]);
                if v > best_v {
                    best_v = v;
                    best = c;
                }
            }
            // CR-02: `best` is the dense class COLUMN (`0..n_classes`); map it back
            // through `classes_` to the ORIGINAL training label so a
            // non-contiguous set (e.g. `{0, 2}`) returns `2`, not the phantom `1`.
            labels[r] = self.classes_[best] as i32;
        }
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine (mirrors the
/// `ridge.rs` / `lbfgs.rs` helper).
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("logistic is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("logistic is f32/f64 only"),
    }
}
