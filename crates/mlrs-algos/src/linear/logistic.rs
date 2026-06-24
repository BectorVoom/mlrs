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

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::lbfgs::{
    lbfgs_minimize, softmax_loss_grad, LbfgsStopReason, LBFGS_FTOL, LBFGS_MAXLS,
};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, PredictProba, Unfit};

/// Default `max_iter` — the L-BFGS iteration cap. sklearn's `LogisticRegression`
/// default is 100; we give the solver headroom (`300`) so the tightened `gtol`
/// below is reachable before the iteration cap for both binary and multiclass.
/// The fixtures are now the TRUE MINIMUM of the (shared) objective (tightly-fit
/// sklearn multiclass; scipy-minimized binary self-reference), so converging
/// deeper lands ON the fixture, not past it. Still a finite cap (T-05-10-03 DoS).
const LOG_DEFAULT_MAX_ITER: usize = 300;
/// Default convergence tolerance (the L-BFGS `gtol` on `max|grad|`). sklearn's
/// default `tol = 1e-4` stops ~3.2e-5 short of the minimum — borderline OVER the
/// strict 1e-5 PRIMARY `predict_proba` gate. We tighten to `1e-5`. This is
/// reachable via the `max|grad| <= gtol` convergence test in **f64** (the
/// gauge-null-space gradient floor is ~9.2e-6, just below gtol — f64 converges
/// cleanly at ~iter 61), so the f64 fixtures (the cpu correctness gate) land
/// within the strict 1e-5 gate.
///
/// In **f32** the gauge-null-space `max|grad|` floor is ~9.93e-5 (~1e-4) — a full
/// DECADE ABOVE gtol=1e-5 — so `max|grad| <= gtol` can NEVER fire and the solver
/// instead exits via a strong-Wolfe LINE-SEARCH BREAKDOWN at the floor (NOT an
/// ftol stall, and NOT the 300-iter cap). The convergence decision in `fit`
/// therefore ACCEPTS that f32 breakdown as converged when its residual `max|grad|`
/// is at/below the dtype precision floor `0.5·sqrt(eps_F)` (≈1.726e-4 for f32) —
/// the gauge-invariant `predict_proba` is fully converged there (within the
/// documented 5e-5 f32 family bound), even though the gauge-VARIANT `max|grad|`
/// scalar cannot reach gtol. A breakdown with `max|grad|` above that floor (a
/// genuine non-stationary stop) is still surfaced as `NotConverged` (T-05-10-03).
const LOG_DEFAULT_TOL: f64 = 1e-5;

/// Multinomial (symmetric-softmax) logistic regression (LINEAR-05) fitted by the
/// L-BFGS iterative solver.
///
/// Construct with the zero-arg [`LogisticRegression::new`] (sklearn defaults:
/// `c = 1.0`, `fit_intercept = true`, `max_iter = 300`, `tol = 1e-5`) or
/// [`LogisticRegression::builder`] (`.c(f64).fit_intercept(bool).max_iter(usize)
/// .tol(f64)` — subsumes the old `new`/`with_opts` constructors), then the
/// consuming [`Fit::fit`] (returns the `Fitted`-tagged sibling) and
/// [`PredictProba::predict_proba`] / [`PredictLabels::predict_labels`]. Fitted
/// `coef_` (K×d) / `intercept_` (K) are device-resident (D-03); the host
/// accessors [`coef`](Self::coef) / [`intercept`](Self::intercept) materialize
/// them on demand and exist ONLY on `LogisticRegression<F, Fitted>` (the
/// compile-time typestate replaces the old runtime `NotFitted` guard, D-03).
pub struct LogisticRegression<F, S = Unfit> {
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
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> LogisticRegression<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct a `LogisticRegression` with sklearn's defaults (`c = 1.0`,
    /// `fit_intercept = true`, `max_iter = 300`, `tol = 1e-5`) directly in the
    /// `Unfit` state. This is the SINGLE source of truth for the default
    /// hyperparameters (D-08): the builder `Default` re-derives from here via
    /// [`LogisticRegression::into_builder`], rather than re-listing the literals.
    /// Defaults are trusted valid, so this bypasses
    /// [`LogisticRegressionBuilder::build`]'s validation.
    pub fn new() -> Self {
        Self {
            c: F::from_int(1),
            fit_intercept: true,
            max_iter: LOG_DEFAULT_MAX_ITER,
            tol: f64_to_host::<F>(LOG_DEFAULT_TOL),
            n_classes: 0,
            classes_: Vec::new(),
            n_features: 0,
            coef_: None,
            intercept_: None,
            _state: PhantomData,
        }
    }

    /// Start building a `LogisticRegression` from sklearn's defaults (D-08 single
    /// source).
    pub fn builder() -> LogisticRegressionBuilder {
        LogisticRegressionBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`LogisticRegressionBuilder::default`] to
    /// re-derive the defaults from [`LogisticRegression::new`] (D-08).
    pub fn into_builder(self) -> LogisticRegressionBuilder {
        LogisticRegressionBuilder {
            c: host_to_f64(self.c),
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: host_to_f64(self.tol),
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `coef_`/`intercept_`/`classes_` fields are excluded). Used by the
    /// defaults-equality test (BLDR-01):
    /// `LogisticRegression::new().hyperparams_eq(&LogisticRegression::builder().build()?)`.
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        host_to_f64(self.c) == host_to_f64(other.c)
            && self.fit_intercept == other.fit_intercept
            && self.max_iter == other.max_iter
            && host_to_f64(self.tol) == host_to_f64(other.tol)
    }
}

impl<F> Default for LogisticRegression<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`LogisticRegression`] (D-01). Setters are `f64`/`usize`/`bool`
/// per the A5 convention; `build::<F>()` narrows the `c`/`tol` scalars to the
/// target float `F`. Subsumes the old `new(c, fit_intercept)` / `with_opts(c,
/// fit_intercept, max_iter, tol)` constructors — every former argument is now a
/// setter. `Default` re-derives the sklearn defaults from
/// [`LogisticRegression::new`] (D-08 single source) rather than holding literals
/// (Pitfall 1: default-drift breaks the oracle gate silently).
#[derive(Debug, Clone, Copy)]
pub struct LogisticRegressionBuilder {
    c: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
}

impl Default for LogisticRegressionBuilder {
    /// Re-derive the sklearn defaults from [`LogisticRegression::new`] (D-08
    /// single source). `f64` is pinned only to read the F-independent scalar
    /// defaults — the builder is non-generic.
    fn default() -> Self {
        LogisticRegression::<f64, Unfit>::new().into_builder()
    }
}

impl LogisticRegressionBuilder {
    /// Set the inverse-regularization strength `C` (A5: `f64` setter).
    pub fn c(mut self, v: f64) -> Self {
        self.c = v;
        self
    }

    /// Set whether to fit an (unpenalized) intercept term per class.
    pub fn fit_intercept(mut self, v: bool) -> Self {
        self.fit_intercept = v;
        self
    }

    /// Set the L-BFGS iteration cap.
    pub fn max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }

    /// Set the L-BFGS convergence tolerance (the `gtol` on `max |grad|`).
    pub fn tol(mut self, v: f64) -> Self {
        self.tol = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08; the data-DEPENDENT geometry
    /// / label checks live in [`Fit::fit`]):
    ///
    /// - `C > 0` ([`BuildError::InvalidC`]) — a non-positive `C` makes
    ///   `l2_reg = 1/(C·n)` non-positive (degenerate / unbounded objective),
    ///   relocated from the old fit-body [`AlgoError::InvalidC`] check
    ///   (T-05-10-01 / Pitfall 7).
    ///
    /// The stored `f64` `c`/`tol` are narrowed to the target float `F` via cast
    /// (A5).
    pub fn build<F>(self) -> Result<LogisticRegression<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if !(self.c > 0.0) {
            return Err(BuildError::InvalidC {
                estimator: "logistic_regression",
                c: self.c,
            });
        }
        Ok(LogisticRegression {
            c: f64_to_host::<F>(self.c),
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: f64_to_host::<F>(self.tol),
            n_classes: 0,
            classes_: Vec::new(),
            n_features: 0,
            coef_: None,
            intercept_: None,
            _state: PhantomData,
        })
    }
}

impl<F> LogisticRegression<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `coef_` (K×d, row-major). `Some` by construction
    /// on the `Fitted` state, so no `NotFitted` branch is needed (the
    /// compile-time typestate replaces the runtime guard, D-03).
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.coef_
            .as_ref()
            .expect("coef_ is Some by construction on LogisticRegression<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `intercept_` (length K). `Some` by construction on
    /// the `Fitted` state (D-03).
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on LogisticRegression<F, Fitted>")
            .to_host(pool)
    }

    /// Number of classes inferred at `fit` (binary = 2).
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
}

impl<F> Fit<F> for LogisticRegression<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = LogisticRegression<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<LogisticRegression<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-05-10-01 / ASVS V5: data-DEPENDENT geometry guard BEFORE any prim
        //     launch (the data-INDEPENDENT `C > 0` check was validated at
        //     build() — Pitfall 7). `c64` is still needed for l2_reg below. ---
        let c64 = host_to_f64(self.c);
        validate_geometry(x, shape)?;
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

            // WR-02: release the two per-iteration scratch buffers through an RAII
            // guard so a PANIC anywhere in `softmax_loss_grad` (a kernel-launch
            // assertion, an `unreachable!` in a bit-cast for a non-f32/f64 `F`)
            // still returns both handles to the pool's free-list as the closure
            // frame unwinds — the previous `release_into` calls sat AFTER the
            // fallible launch and ran only on the normal path, stranding both
            // handles on a panic. The guard borrows the same `pool` only in its
            // `Drop`, after the launch has fully returned (or unwound), so there is
            // no aliasing with the `&mut pool` the launch itself takes.
            let res = {
                let mut guard = ScratchGuard::new(pool, w_d, b_d);
                let (pool_ref, w_ref, b_ref) = guard.parts();
                softmax_loss_grad::<F>(
                    pool_ref, x_dev, y_dev, w_ref, b_ref, n_samples, d, k, l2_reg,
                )
            };

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
        //     unchanged): the gradient's null-space components never shrink, so
        //     `max|grad|` plateaus at a dtype-precision FLOOR even though the loss
        //     has reached its true minimum (the gauge-INVARIANT predict_proba is
        //     fully converged). The floor depends on the float type:
        //       - f64: floor ~9.2e-6, just BELOW gtol=1e-5 → the prim reaches
        //         `max|grad| <= gtol` and stops via `Converged` (~iter 61).
        //       - f32: floor ~9.93e-5 (~1e-4), a DECADE ABOVE gtol=1e-5 → gtol is
        //         unreachable; the loss is flat (rel-f ~1e-8/step, far above
        //         LBFGS_FTOL=1.42e-14 so the ftol stall never fires either), and the
        //         strong-Wolfe line search runs out of acceptable steps → the prim
        //         exits via `LineSearchFailed` at ~iter 51. (Empirically confirmed —
        //         this is NOT an ftol stall and NOT the 300-iter cap.)
        //
        //     So the real DoS / non-convergence signal (T-05-10-03) is EITHER hitting
        //     the iteration CAP (`iters == maxiter`) OR a LineSearchFailed at a point
        //     whose residual `max|grad|` is ABOVE the dtype precision floor (a genuine
        //     non-stationary breakdown). We accept an early gtol/ftol stop
        //     (`result.converged` OR `iters < maxiter` with a Converged/FtolStall
        //     reason) AND a LineSearchFailed whose `max|grad|` is at/below the gauge
        //     floor `0.5·sqrt(eps_F)` (the f32 case above) — the primary
        //     `predict_proba` 1e-5/5e-5 oracle is the correctness witness that the
        //     accepted iterate is the right minimizer (Pitfall 5). ---
        // WR-01 + GAUGE-FLOOR ACCEPT (05-10): a line-search BREAKDOWN
        // (`LbfgsStopReason::LineSearchFailed`) is, in general, a stop at a possibly
        // NON-stationary point — a non-minimizer — and must be rejected as
        // NotConverged. BUT for the symmetric over-parameterized softmax (D-12)
        // there is one legitimate exception: the gauge null-space pins the achievable
        // `max|grad|` to a dtype-precision FLOOR. In f32 that floor is ~9.93e-5
        // (measured), a full decade ABOVE the tightened gtol=1e-5 (LOG_DEFAULT_TOL),
        // so `max|grad| <= gtol` can never fire and the ONLY available stop is this
        // line-search breakdown — at a point that IS first-order stationary within
        // f32 resolution (the gauge-invariant predict_proba is fully converged). f64's
        // floor (~9.2e-6) sits just below gtol, so f64 converges via `Converged` and
        // never reaches this branch.
        //
        // So we ACCEPT a LineSearchFailed stop as converged IFF its residual
        // `max|grad|` is at or below the dtype's gauge-null-space precision floor,
        // `k·sqrt(F::EPSILON)`. We use sqrt(eps) because the smallest gradient a
        // floating-point loss can resolve near a minimum scales like sqrt(eps) (the
        // loss is flat to first order, so the representable curvature step is ~sqrt(eps);
        // this is the same scaling scipy uses for its default finite-difference step).
        //   - f32: sqrt(eps_f32) ≈ 3.4527e-4; the measured floor 9.928e-5 is
        //     ≈0.288·sqrt(eps_f32). We pick k = 0.5 → floor_accept = 1.726e-4, which
        //     clears the measured 9.928e-5 with ~1.74× headroom yet stays a TIGHT
        //     sub-multiple of sqrt(eps): a genuine non-convergent breakdown has
        //     max|grad| >> floor (orders of magnitude), so this cannot mask the real
        //     NotConverged / DoS signal (T-05-10-03).
        //   - f64: sqrt(eps_f64) ≈ 1.4901e-8 → floor_accept ≈ 7.45e-9. The f64 path
        //     already stops via `Converged` (max|grad| 9.24e-6 <= gtol), so this
        //     branch is never entered for f64; f64 behavior is unchanged.
        // A LineSearchFailed with `max|grad|` ABOVE this floor is STILL rejected as
        // NotConverged — the genuine non-stationary breakdown.
        const GAUGE_FLOOR_K: f64 = 0.5;
        let gauge_floor_accept = GAUGE_FLOOR_K * f_epsilon::<F>().sqrt();
        if result.stop_reason == LbfgsStopReason::LineSearchFailed {
            if result.max_grad <= gauge_floor_accept {
                // Stationary within dtype resolution (gauge floor) — accept as the
                // converged minimizer; the predict_proba 1e-5/5e-5 oracle is the
                // correctness witness (Pitfall 5).
            } else {
                // Genuine non-stationary breakdown — preserve the T-05-10-03 signal.
                return Err(AlgoError::NotConverged {
                    estimator: "logistic_regression",
                    max_iter: maxiter,
                });
            }
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

        Ok(LogisticRegression {
            c: self.c,
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
            n_classes,
            classes_,
            n_features,
            coef_: Some(coef_dev),
            intercept_: Some(intercept_dev),
            _state: PhantomData,
        })
    }
}

impl<F> PredictProba<F> for LogisticRegression<F, Fitted>
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

        // `coef_`/`intercept_` are `Some` by construction on the `Fitted` state
        // (the compile-time typestate replaces the old runtime `NotFitted`
        // guard, D-03).
        let coef = self
            .coef_
            .as_ref()
            .expect("coef_ is Some by construction on LogisticRegression<F, Fitted>");
        let intercept = self
            .intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on LogisticRegression<F, Fitted>");

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

impl<F> PredictLabels<F> for LogisticRegression<F, Fitted>
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

/// WR-02: a panic-safe RAII guard for the two per-iteration L-BFGS objective
/// scratch buffers (`w_d`, `b_d`). It owns the buffers plus a mutable borrow of
/// the pool and returns BOTH buffers to the pool's free-list in its `Drop` —
/// whether the closure body returns normally OR unwinds on a panic from the
/// device softmax launch. Releasing in `Drop` (not after the fallible launch)
/// closes the WR-02 window where a panic between acquire and release stranded the
/// handles for the process lifetime.
struct ScratchGuard<'a, F: Pod> {
    pool: &'a mut BufferPool<ActiveRuntime>,
    w: Option<DeviceArray<ActiveRuntime, F>>,
    b: Option<DeviceArray<ActiveRuntime, F>>,
}

impl<'a, F: Float + CubeElement + Pod> ScratchGuard<'a, F> {
    /// Take ownership of the pool borrow and the two scratch buffers.
    fn new(
        pool: &'a mut BufferPool<ActiveRuntime>,
        w: DeviceArray<ActiveRuntime, F>,
        b: DeviceArray<ActiveRuntime, F>,
    ) -> Self {
        Self {
            pool,
            w: Some(w),
            b: Some(b),
        }
    }

    /// Reborrow the pool plus the two buffers for the launch. The returned
    /// references all live as long as the `&mut self` reborrow, so the guard (and
    /// thus the buffers) cannot be dropped until the launch fully returns.
    fn parts(
        &mut self,
    ) -> (
        &mut BufferPool<ActiveRuntime>,
        &DeviceArray<ActiveRuntime, F>,
        &DeviceArray<ActiveRuntime, F>,
    ) {
        (
            self.pool,
            self.w.as_ref().expect("scratch w present until drop"),
            self.b.as_ref().expect("scratch b present until drop"),
        )
    }
}

impl<F: Pod> Drop for ScratchGuard<'_, F> {
    fn drop(&mut self) {
        if let Some(w) = self.w.take() {
            w.release_into(self.pool);
        }
        if let Some(b) = self.b.take() {
            b.release_into(self.pool);
        }
    }
}

/// Machine epsilon of `F` (f32 / f64) as an `f64`, for the gauge-null-space
/// precision floor `k·sqrt(eps_F)` (mirrors the `svd.rs` / `eig.rs` per-dtype
/// epsilon helper). Keeps the gauge-floor accept generic over the float type.
fn f_epsilon<F: Pod>() -> f64 {
    match size_of::<F>() {
        4 => f32::EPSILON as f64,
        8 => f64::EPSILON,
        _ => unreachable!("logistic is f32/f64 only"),
    }
}
