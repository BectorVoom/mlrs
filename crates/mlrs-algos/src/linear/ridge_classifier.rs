//! `RidgeClassifier` (LINEAR-07) — sklearn's `Ridge`-as-classifier: encode the
//! targets as `{-1, +1}` (one-hot per class for K>2, one column for K=2, the
//! `LabelBinarizer(neg_label=-1, pos_label=1)` convention) and solve a
//! multi-output Ridge regression, `predict` reading off the sign (binary) or
//! the `argmax` (multiclass) of the fitted decision function.
//!
//! ```text
//! RidgeClassifier(alpha=1.0, *, fit_intercept=True, copy_X=True,
//!                  max_iter=None, tol=1e-4, class_weight=None,
//!                  solver='auto', positive=False, random_state=None)
//!                  .fit(X, y, sample_weight=None)
//! ```
//!
//! Every one of those parameters is implemented, plus the fitted `coef_` /
//! `intercept_` / `n_iter_` / `classes_` / `solver_` attributes sklearn exposes.
//! `RidgeClassifier` deliberately does NOT expose `predict_proba` — neither does
//! sklearn's (only `decision_function`/`predict`, via `_RidgeClassifierMixin`).
//!
//! ## Why this is not "loop `Ridge::fit` once per class column" and stop there
//! Mathematically a per-target-column Ridge solve is exact: `alpha` is a single
//! scalar shared by every target, so the K columns are K INDEPENDENT
//! least-squares problems against the SAME (regularized) normal matrix — this
//! holds regardless of whether sklearn's own `_ridge_regression` batches the
//! solve or not (multi-RHS linear algebra is just "apply the same inverse to
//! different vectors"). But the Gram `XᵀX` costs `O(n·d²)` to form and does
//! NOT depend on `y` at all, so naively calling the single-output
//! `Ridge::fit_from_host_slice` once per class would re-walk the whole design
//! K times for zero mathematical benefit. [`centered_gram_multi_xty`] forms the
//! Gram ONCE and the K `Xᵀy` columns in the SAME pass (sharing the
//! centered-and-transposed tile every column needs anyway), so this
//! estimator's cpu fast path costs `O(n·d² + n·d·K)`, not `O(n·d²·K)` — this is
//! the whole point of a dedicated `RidgeClassifier` rather than a thin
//! `Ridge`-in-a-loop Python wrapper.
//!
//! ## Two entry points, mirroring `Ridge` exactly (D-02)
//! - [`RidgeClassifier::fit_from_host_slice`] — the no-upload HOST arm, used
//!   when [`RidgeClassifier::host_fit_applicable`] is true: `solver` resolves
//!   to `cholesky` (the `positive=False` default) or `lbfgs`
//!   (`positive=True`), AND [`gram_host_applicable`] holds for the shape
//!   (unconditionally true on the cpu backend — `gram_host.rs`'s module docs).
//!   This is the path `RidgeClassifier()` on cpu actually takes, and it is the
//!   one this module optimizes: shared Gram, per-column Cholesky solve (with
//!   sklearn's `LinAlgError → svd` retry, uniform across columns because the
//!   Gram is shared) or per-column non-negative coordinate descent.
//! - [`RidgeClassifier::fit_with_sample_weight`] — the DEVICE-array arm, used
//!   for every other `(solver, backend)` combination. It delegates to the
//!   FULLY-VALIDATED single-output [`Ridge`] estimator once per target column,
//!   which is what gives this estimator its complete 8-solver parameter
//!   surface without re-deriving `svd`/`sparse_cg`/`lsqr`/`sag`/`saga` for
//!   multi-output from scratch. The shared-Gram optimization above does not
//!   apply here (those solvers do not share a single factorable normal
//!   matrix across a device upload), so this arm is correct-by-delegation
//!   rather than independently perf-tuned — acceptable because it is reached
//!   only by an explicit non-default `solver=` choice or a non-cpu backend.
//!
//! ## `class_weight` (sklearn's `compute_sample_weight`, reproduced)
//! `class_weight='balanced'` sets `weight[c] = n_samples / (n_classes ·
//! count[c])` from the RAW (unweighted) label counts — sklearn's
//! `_RidgeClassifierMixin._prepare_data` calls `compute_sample_weight` with NO
//! `sample_weight` argument, so the balanced weights are computed from
//! class counts alone and THEN multiplied into any user-supplied
//! `sample_weight`. A `class_weight` dict maps a training label to its weight;
//! a class absent from the dict keeps weight 1.0 (sklearn's
//! `compute_class_weight`), and a dict key that names a label outside
//! `classes_` is rejected as [`AlgoError::InvalidLabels`] rather than silently
//! ignored.
//!
//! ## Label / target encoding (`LabelBinarizer(neg_label=-1, pos_label=1)`)
//! `classes_` is the DISTINCT SORTED set of training labels (CR-02 — never a
//! fabricated `0..n_classes` range, so a non-contiguous target like `{0, 2}`
//! round-trips). Binary (`len(classes_) == 2`) gets ONE target column:
//! `+1` for `classes_[1]`, `-1` for `classes_[0]`. Multiclass gets one column
//! PER class: `+1` in its own column, `-1` everywhere else. `predict` inverts
//! this exactly as sklearn's `LinearClassifierMixin.predict` does: binary
//! reads the STRICT sign (`score > 0 → classes_[1]`, matching `>`, not `>=`
//! — sklearn's `scores > 0`), multiclass takes the `argmax` column (numpy's
//! first-occurrence tie-break, i.e. `>` not `>=` when scanning for the max).
//!
//! Tests live in `crates/mlrs-algos/tests/ridge_classifier_test.rs` (AGENTS.md
//! §2), never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;
use std::sync::OnceLock;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gram_host::{centered_gram_multi_xty, gram_host_applicable};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::linear::ridge::{validate_sample_weight, Ridge, RidgeSolver};
use crate::linear::ridge_solvers;
use crate::typestate::{validate_geometry, Fitted, PredictLabels, Unfit};

/// sklearn's `Ridge`/`RidgeClassifier` default `tol` (`1e-4`).
const RIDGE_CLASSIFIER_DEFAULT_TOL: f64 = 1e-4;

/// sklearn's `class_weight` — `None` (uniform), `'balanced'`, or a
/// `{label: weight}` dict. A non-scalar hyperparameter, so the builder takes
/// this enum directly (the `RidgeSolver`/`KernelKind` precedent) rather than a
/// stringly-typed setter.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassWeight {
    /// sklearn's `None` — every class weighs 1.0.
    Uniform,
    /// sklearn's `'balanced'` — `n_samples / (n_classes · count[c])` from the
    /// raw (unweighted) training label counts.
    Balanced,
    /// sklearn's `{label: weight}` dict. A label absent from this list keeps
    /// weight 1.0; a label that is not one of the training `classes_` is
    /// rejected at `fit` ([`AlgoError::InvalidLabels`]).
    Map(Vec<(i64, f64)>),
}

impl Default for ClassWeight {
    fn default() -> Self {
        ClassWeight::Uniform
    }
}

/// `Ridge`-as-classifier (LINEAR-07) with sklearn's full parameter set. See
/// the module docs for the two fit entry points and the `class_weight` /
/// label-encoding contracts.
pub struct RidgeClassifier<F, S = Unfit> {
    /// L2 penalty strength (`alpha >= 0`), shared by every target column.
    alpha: F,
    /// Whether to center `X`/`Y` and recover a per-target bias term.
    fit_intercept: bool,
    /// sklearn's `copy_X`. Stored for parity; mlrs never writes into the
    /// caller's buffer, so this has no observable effect ([`Ridge`]'s module
    /// docs, verbatim here).
    copy_x: bool,
    /// Iteration cap for the iterative solvers (delegation arm only).
    max_iter: Option<usize>,
    /// Stopping tolerance for the iterative solvers.
    tol: f64,
    /// sklearn's `class_weight`.
    class_weight: ClassWeight,
    /// Which solver to use (`Auto` resolves at `fit`, same table as [`Ridge`]).
    solver: RidgeSolver,
    /// Constrain every target's `coef_ >= 0`. Requires the `lbfgs` solver (or
    /// `auto`, which resolves to it) — the same constraint [`Ridge`] enforces.
    positive: bool,
    /// Seed for the `sag`/`saga` sampling order (delegation arm only).
    random_state: Option<u64>,
    /// The DISTINCT sorted training labels (CR-02). Empty until `fit`.
    classes_: Vec<i64>,
    /// `1` for a binary fit, `classes_.len()` for multiclass — the number of
    /// target columns / rows of `coef_`.
    n_targets_: usize,
    /// Feature count inferred at `fit`.
    n_features_: usize,
    /// Fitted coefficients, device-resident, row-major `n_targets_ ×
    /// n_features_`. `None` until `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercepts, device-resident, length `n_targets_`. `None` until
    /// `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
    /// sklearn's `n_iter_`: `Some` (length `n_targets_`) only for the solvers
    /// that report it (`lsqr`/`sag`/`saga`, delegation arm only); `None` for
    /// every other solver, INCLUDING the cpu shared-Gram fast path (`cholesky`
    /// / `lbfgs`, which `Ridge::fit_from_host_slice` also leaves unset).
    n_iter_: Option<Vec<usize>>,
    /// sklearn's `solver_`: the solver that ACTUALLY ran (after `auto`
    /// resolution and any singular-Gram fallback). Uniform across every
    /// target column because the fallback depends only on the shared `X`
    /// Gram, never on which target column is being solved.
    solver_: Option<RidgeSolver>,
    /// Host mirror of `(coef_, intercept_)`, both flattened row-major, for the
    /// no-upload `predict`/`decision_function` host ingress (the [`Ridge`]
    /// `HostMirror` idiom, generalized to `n_targets_ > 1`). Empty until the
    /// first host-ingress predict call.
    predict_mirror: OnceLock<(Vec<F>, Vec<F>)>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> RidgeClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct a `RidgeClassifier` with sklearn's defaults (`alpha = 1.0`,
    /// `fit_intercept = true`, `copy_X = true`, `max_iter = None`,
    /// `tol = 1e-4`, `class_weight = None`, `solver = 'auto'`,
    /// `positive = false`, `random_state = None`). The single source of truth
    /// for the defaults (D-08): the builder's `Default` re-derives from here.
    pub fn new() -> Self {
        Self {
            alpha: F::from_int(1),
            fit_intercept: true,
            copy_x: true,
            max_iter: None,
            tol: RIDGE_CLASSIFIER_DEFAULT_TOL,
            class_weight: ClassWeight::Uniform,
            solver: RidgeSolver::Auto,
            positive: false,
            random_state: None,
            classes_: Vec::new(),
            n_targets_: 0,
            n_features_: 0,
            coef_: None,
            intercept_: None,
            n_iter_: None,
            solver_: None,
            predict_mirror: OnceLock::new(),
            _state: PhantomData,
        }
    }

    /// Start building a `RidgeClassifier` from sklearn's defaults (D-08).
    pub fn builder() -> RidgeClassifierBuilder {
        RidgeClassifierBuilder::default()
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (mirrors
    /// [`Ridge::hyperparams_eq`]). Used by the defaults-equality test
    /// (BLDR-01): `RidgeClassifier::new().hyperparams_eq(&RidgeClassifier::builder().build()?)`.
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        host_to_f64(self.alpha) == host_to_f64(other.alpha)
            && self.fit_intercept == other.fit_intercept
            && self.copy_x == other.copy_x
            && self.max_iter == other.max_iter
            && self.tol == other.tol
            && self.class_weight == other.class_weight
            && self.solver == other.solver
            && self.positive == other.positive
            && self.random_state == other.random_state
    }

    /// Decompose back into a builder, copying every hyperparameter (mirrors
    /// [`Ridge::into_builder`]).
    pub fn into_builder(self) -> RidgeClassifierBuilder {
        RidgeClassifierBuilder {
            alpha: host_to_f64(self.alpha),
            fit_intercept: self.fit_intercept,
            copy_x: self.copy_x,
            max_iter: self.max_iter,
            tol: self.tol,
            class_weight: self.class_weight,
            solver: self.solver,
            positive: self.positive,
            random_state: self.random_state,
        }
    }

    /// Does the shared-Gram HOST fit arm ([`RidgeClassifier::fit_from_host_slice`])
    /// apply to this configuration? Mirrors [`Ridge::host_fit_applicable`]
    /// exactly: `true` for the two normal-equations solvers (`cholesky` — the
    /// `positive = false` default — and `lbfgs`), and only where the Gram
    /// formation belongs on the host ([`gram_host_applicable`] — the cpu
    /// backend, or below the fixed dispatch-cost floor on any backend).
    pub fn host_fit_applicable(&self, shape: (usize, usize)) -> bool {
        matches!(
            self.solver.resolve(self.positive),
            RidgeSolver::Cholesky | RidgeSolver::Lbfgs
        ) && gram_host_applicable(shape.0, shape.1)
    }

    /// The no-upload HOST fit arm — the fast path this estimator exists for.
    /// See [`RidgeClassifier::host_fit_applicable`]; returns
    /// [`PrimError::UnsupportedCapability`] when it does not hold, exactly as
    /// [`Ridge::fit_from_host_slice`] does.
    ///
    /// `x` is the `n × d` row-major design and `y_labels` the length-`n` RAW
    /// class labels (float-encoded non-negative integers, the same convention
    /// `LogisticRegression::fit` uses) — NOT the `{-1,+1}` target matrix,
    /// which is derived internally.
    pub fn fit_from_host_slice(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        y_labels: &[F],
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<RidgeClassifier<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        if !self.host_fit_applicable(shape) {
            return Err(AlgoError::Prim(PrimError::UnsupportedCapability {
                operand: "ridge_classifier.fit_from_host_slice",
                capability: "the host fit arm (a normal-equations solver on a host-Gram backend)",
            }));
        }
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        if y_labels.len() != n_samples {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_samples,
                cols: 1,
                len: y_labels.len(),
            }));
        }

        let (classes_, class_idx) = decode_labels::<F>("ridge_classifier", y_labels)?;
        let n_classes = classes_.len();
        let n_targets = if n_classes == 2 { 1 } else { n_classes };
        let combined =
            combined_sample_weight::<F>(&self.class_weight, &classes_, &class_idx, sample_weight, n_samples)?;
        let y_targets = encode_targets::<F>(&class_idx, n_targets);

        let alpha64 = host_to_f64(self.alpha);
        let (x_mean, y_means, gram, xty) = centered_gram_multi_xty::<F>(
            x,
            &y_targets,
            n_samples,
            n_features,
            n_targets,
            combined.as_deref(),
            self.fit_intercept,
        );

        let mut coef = vec![0.0f64; n_targets * n_features];
        let mut solver_used = RidgeSolver::Cholesky;
        if self.positive {
            for t in 0..n_targets {
                let xty_t: Vec<f64> = (0..n_features).map(|i| xty[i * n_targets + t]).collect();
                let (w, _sweeps) = ridge_solvers::nonnegative_cd(
                    &gram, &xty_t, n_features, alpha64, self.tol, self.max_iter,
                );
                coef[t * n_features..(t + 1) * n_features].copy_from_slice(&w);
            }
            solver_used = RidgeSolver::Lbfgs;
        } else {
            for t in 0..n_targets {
                let xty_t: Vec<f64> = (0..n_features).map(|i| xty[i * n_targets + t]).collect();
                // sklearn's `except LinAlgError: solver = "svd"` retry, shared
                // across every target column because it depends only on the
                // (shared) Gram, never on `y` — either every column falls back
                // together or none does.
                let w = match ridge_solvers::cholesky_ridge(&gram, &xty_t, n_features, alpha64) {
                    Some(w) => w,
                    None => {
                        solver_used = RidgeSolver::Svd;
                        ridge_solvers::gram_eig_ridge(&gram, &xty_t, n_features, alpha64)
                    }
                };
                coef[t * n_features..(t + 1) * n_features].copy_from_slice(&w);
            }
        }

        let mut intercept = vec![0.0f64; n_targets];
        for t in 0..n_targets {
            let dot: f64 = x_mean
                .iter()
                .zip(&coef[t * n_features..(t + 1) * n_features])
                .map(|(m, c)| m * c)
                .sum();
            intercept[t] = if self.fit_intercept { y_means[t] - dot } else { 0.0 };
        }

        let coef_f: Vec<F> = coef.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let intercept_f: Vec<F> = intercept.iter().map(|&v| f64_to_host::<F>(v)).collect();

        Ok(RidgeClassifier {
            alpha: self.alpha,
            fit_intercept: self.fit_intercept,
            copy_x: self.copy_x,
            max_iter: self.max_iter,
            tol: self.tol,
            class_weight: self.class_weight,
            solver: self.solver,
            positive: self.positive,
            random_state: self.random_state,
            classes_,
            n_targets_: n_targets,
            n_features_: n_features,
            coef_: Some(DeviceArray::from_host(pool, &coef_f)),
            intercept_: Some(DeviceArray::from_host(pool, &intercept_f)),
            n_iter_: None,
            solver_: Some(solver_used),
            predict_mirror: OnceLock::new(),
            _state: PhantomData,
        })
    }

    /// The DEVICE-array fit arm — every `(solver, backend)` combination
    /// [`RidgeClassifier::fit_from_host_slice`] does not cover. Delegates to
    /// the fully-featured [`Ridge`] estimator once per target column (see the
    /// module docs for why this is correct — the columns are mathematically
    /// independent — and why it is NOT perf-specialized the way the host arm
    /// is).
    ///
    /// `y` carries the RAW class labels (float-encoded), exactly like
    /// [`RidgeClassifier::fit_from_host_slice`]'s `y_labels`.
    pub fn fit_with_sample_weight(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<RidgeClassifier<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "ridge_classifier",
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

        let y_host = y.to_host(pool);
        let (classes_, class_idx) = decode_labels::<F>("ridge_classifier", &y_host)?;
        let n_classes = classes_.len();
        let n_targets = if n_classes == 2 { 1 } else { n_classes };
        let combined64 =
            combined_sample_weight::<F>(&self.class_weight, &classes_, &class_idx, sample_weight, n_samples)?;
        let combined_f: Option<Vec<F>> = combined64
            .as_ref()
            .map(|v| v.iter().map(|&w| f64_to_host::<F>(w)).collect());
        let y_targets = encode_targets::<F>(&class_idx, n_targets);

        let mut coef_flat: Vec<F> = Vec::with_capacity(n_targets * n_features);
        let mut intercept_flat: Vec<F> = Vec::with_capacity(n_targets);
        let mut n_iters: Vec<Option<usize>> = Vec::with_capacity(n_targets);
        let mut solver_used: Option<RidgeSolver> = None;
        for t in 0..n_targets {
            let col: Vec<F> = (0..n_samples).map(|r| y_targets[r * n_targets + t]).collect();
            let y_col_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &col);
            let ridge_unfit = Ridge::<F>::builder()
                .alpha(host_to_f64(self.alpha))
                .fit_intercept(self.fit_intercept)
                .copy_x(self.copy_x)
                .max_iter(self.max_iter)
                .tol(self.tol)
                .solver(self.solver)
                .positive(self.positive)
                .random_state(self.random_state)
                .build::<F>()
                .expect("hyperparameters already validated at RidgeClassifierBuilder::build");
            let fitted =
                ridge_unfit.fit_with_sample_weight(pool, x, Some(&y_col_dev), shape, combined_f.as_deref())?;
            y_col_dev.release_into(pool);
            coef_flat.extend(fitted.coef(pool));
            intercept_flat.push(fitted.intercept(pool));
            n_iters.push(fitted.n_iter());
            solver_used = Some(fitted.solver());
        }
        // sklearn's `n_iter_` is a property of the SOLVER, not the data: either
        // every target reports an iteration count or none do.
        let n_iter_ = if n_iters.iter().all(Option::is_some) {
            Some(n_iters.into_iter().map(|v| v.expect("checked Some above")).collect())
        } else {
            None
        };

        Ok(RidgeClassifier {
            alpha: self.alpha,
            fit_intercept: self.fit_intercept,
            copy_x: self.copy_x,
            max_iter: self.max_iter,
            tol: self.tol,
            class_weight: self.class_weight,
            solver: self.solver,
            positive: self.positive,
            random_state: self.random_state,
            classes_,
            n_targets_: n_targets,
            n_features_: n_features,
            coef_: Some(DeviceArray::from_host(pool, &coef_flat)),
            intercept_: Some(DeviceArray::from_host(pool, &intercept_flat)),
            n_iter_,
            solver_: solver_used,
            predict_mirror: OnceLock::new(),
            _state: PhantomData,
        })
    }
}

impl<F> Default for RidgeClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`RidgeClassifier`] (D-01). Scalar setters are `f64`-typed
/// (A5); `build::<F>()` narrows to the target float `F`. `Default` re-derives
/// the sklearn defaults from [`RidgeClassifier::new`] (D-08).
#[derive(Debug, Clone)]
pub struct RidgeClassifierBuilder {
    alpha: f64,
    fit_intercept: bool,
    copy_x: bool,
    max_iter: Option<usize>,
    tol: f64,
    class_weight: ClassWeight,
    solver: RidgeSolver,
    positive: bool,
    random_state: Option<u64>,
}

impl Default for RidgeClassifierBuilder {
    fn default() -> Self {
        RidgeClassifier::<f64, Unfit>::new().into_builder()
    }
}

impl RidgeClassifierBuilder {
    /// Set the L2 penalty strength `alpha` (A5: `f64` setter).
    pub fn alpha(mut self, v: f64) -> Self {
        self.alpha = v;
        self
    }

    /// Set whether to center `X`/`Y` and recover a per-target bias term.
    pub fn fit_intercept(mut self, v: bool) -> Self {
        self.fit_intercept = v;
        self
    }

    /// Set sklearn's `copy_X`. Accepted for API parity; no observable effect
    /// (see the module docs).
    pub fn copy_x(mut self, v: bool) -> Self {
        self.copy_x = v;
        self
    }

    /// Set the iterative solvers' iteration cap (delegation arm only).
    pub fn max_iter(mut self, v: Option<usize>) -> Self {
        self.max_iter = v;
        self
    }

    /// Set the iterative solvers' stopping tolerance (delegation arm only).
    pub fn tol(mut self, v: f64) -> Self {
        self.tol = v;
        self
    }

    /// Set sklearn's `class_weight`.
    pub fn class_weight(mut self, v: ClassWeight) -> Self {
        self.class_weight = v;
        self
    }

    /// Set the solver (sklearn's `solver`). Takes the [`RidgeSolver`] enum
    /// directly.
    pub fn solver(mut self, v: RidgeSolver) -> Self {
        self.solver = v;
        self
    }

    /// Constrain every target's coefficients to be non-negative. Requires
    /// `solver` to be `auto` or `lbfgs`, exactly as [`Ridge`] requires.
    pub fn positive(mut self, v: bool) -> Self {
        self.positive = v;
        self
    }

    /// Seed the `sag`/`saga` sampling order (delegation arm only).
    pub fn random_state(mut self, v: Option<u64>) -> Self {
        self.random_state = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08) — identical to
    /// [`Ridge`]'s builder validation (`alpha >= 0`, `tol` finite `>= 0`,
    /// `max_iter >= 1` when given, and the `lbfgs`/`positive` compatibility
    /// pair). `class_weight` needs no build-time check: every variant of the
    /// [`ClassWeight`] enum is valid by construction, and content validation
    /// (an unknown label in a `Map`) is data-DEPENDENT, so it happens at `fit`.
    pub fn build<F>(self) -> Result<RidgeClassifier<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if !(self.alpha >= 0.0) {
            return Err(BuildError::InvalidAlpha {
                estimator: "ridge_classifier",
                alpha: self.alpha,
            });
        }
        if !self.tol.is_finite() || self.tol < 0.0 {
            return Err(BuildError::InvalidTol {
                estimator: "ridge_classifier",
                tol: self.tol,
            });
        }
        if self.max_iter == Some(0) {
            return Err(BuildError::InvalidMaxIter {
                estimator: "ridge_classifier",
                max_iter: 0,
            });
        }
        if self.solver == RidgeSolver::Lbfgs && !self.positive {
            return Err(BuildError::LbfgsRequiresPositive {
                estimator: "ridge_classifier",
            });
        }
        if self.positive && !matches!(self.solver, RidgeSolver::Auto | RidgeSolver::Lbfgs) {
            return Err(BuildError::PositiveUnsupportedSolver {
                estimator: "ridge_classifier",
                solver: self.solver.name(),
            });
        }
        Ok(RidgeClassifier {
            alpha: f64_to_host::<F>(self.alpha),
            fit_intercept: self.fit_intercept,
            copy_x: self.copy_x,
            max_iter: self.max_iter,
            tol: self.tol,
            class_weight: self.class_weight,
            solver: self.solver,
            positive: self.positive,
            random_state: self.random_state,
            classes_: Vec::new(),
            n_targets_: 0,
            n_features_: 0,
            coef_: None,
            intercept_: None,
            n_iter_: None,
            solver_: None,
            predict_mirror: OnceLock::new(),
            _state: PhantomData,
        })
    }
}

impl<F> RidgeClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The DISTINCT sorted training labels (`classes_`, CR-02).
    pub fn classes(&self) -> &[i64] {
        &self.classes_
    }

    /// `1` for a binary fit, `n_classes` for multiclass.
    pub fn n_targets(&self) -> usize {
        self.n_targets_
    }

    /// Host copy of the fitted `coef_`, row-major `n_targets × n_features`.
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.coef_
            .as_ref()
            .expect("coef_ is Some by construction on RidgeClassifier<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `intercept_`, length `n_targets`.
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on RidgeClassifier<F, Fitted>")
            .to_host(pool)
    }

    /// sklearn's `n_iter_`: `Some` (length `n_targets`) only for the solvers
    /// that report it; `None` otherwise (see the field docs).
    pub fn n_iter(&self) -> Option<Vec<usize>> {
        self.n_iter_.clone()
    }

    /// sklearn's `solver_` — the solver that ACTUALLY ran.
    pub fn solver(&self) -> RidgeSolver {
        self.solver_
            .expect("solver_ is Some by construction on RidgeClassifier<F, Fitted>")
    }

    /// The host `(coef, intercept)` mirror, reading it back on first call only
    /// (the [`Ridge`] `HostMirror` idiom, generalized to `n_targets_ > 1`).
    fn host_mirror(&self, pool: &BufferPool<ActiveRuntime>) -> &(Vec<F>, Vec<F>) {
        self.predict_mirror.get_or_init(|| {
            let coef = self
                .coef_
                .as_ref()
                .expect("coef_ is Some by construction on RidgeClassifier<F, Fitted>")
                .to_host(pool);
            let intercept = self
                .intercept_
                .as_ref()
                .expect("intercept_ is Some by construction on RidgeClassifier<F, Fitted>")
                .to_host(pool);
            (coef, intercept)
        })
    }

    /// Decision-function scores from a HOST `x` (the no-upload cpu ingress —
    /// every dense linear estimator's `predict_from_host` precedent): length
    /// `n_samples` for a binary fit, row-major `n_samples × n_targets` for
    /// multiclass, plus the operand-finiteness verdict.
    ///
    /// [`decision_multi_host`] rather than one single-column matvec call
    /// per target column: calling a single-column prim `k` times streams
    /// the WHOLE `n × d` design past the cache `k` times (and pays `k`
    /// redundant finite-checks of the same `x`) — fine for the binary/
    /// low-cardinality fit this estimator shares with `Ridge`, but it made
    /// `predict` LOSE to sklearn's one-shot `X · coef_ᵀ` GEMM as `n_classes`
    /// grew (measured: 0.32× at `k = 26`). Reading each row ONCE and taking
    /// all `k` dot products against it while it is still hot is the same
    /// `O(n·d·k)` arithmetic with `O(n·d)` traffic instead of `O(n·d·k)`.
    pub fn decision_function_from_host(
        &self,
        pool: &BufferPool<ActiveRuntime>,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<RidgeClassifierScores, AlgoError> {
        let (n_samples, n_features) = shape;
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        if n_features != self.n_features_ {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: n_features,
                rhs: self.n_features_,
            }));
        }
        let (coef_host, intercept_host) = self.host_mirror(pool);
        let k = self.n_targets_;
        let (values, operand_finite) =
            decision_multi_host::<F>(x, coef_host, intercept_host, n_samples, n_features, k);
        Ok(RidgeClassifierScores {
            values,
            n_targets: k,
            operand_finite,
        })
    }

    /// `predict` from a HOST `x`: the sign (binary, STRICT `> 0`) or `argmax`
    /// (multiclass, first-occurrence tie-break) of
    /// [`RidgeClassifier::decision_function_from_host`], mapped back through
    /// `classes_`.
    pub fn predict_labels_from_host(
        &self,
        pool: &BufferPool<ActiveRuntime>,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<RidgeClassifierPrediction, AlgoError> {
        let scores = self.decision_function_from_host(pool, x, shape)?;
        let (n_samples, _) = shape;
        let k = scores.n_targets;
        let mut labels = vec![0i32; n_samples];
        if k == 1 {
            for (r, label) in labels.iter_mut().enumerate() {
                *label = if scores.values[r] > 0.0 {
                    self.classes_[1] as i32
                } else {
                    self.classes_[0] as i32
                };
            }
        } else {
            for (r, label) in labels.iter_mut().enumerate() {
                let row = &scores.values[r * k..(r + 1) * k];
                let mut best = 0usize;
                for c in 1..k {
                    if row[c] > row[best] {
                        best = c;
                    }
                }
                *label = self.classes_[best] as i32;
            }
        }
        Ok(RidgeClassifierPrediction {
            labels,
            operand_finite: scores.operand_finite,
        })
    }
}

impl<F> PredictLabels<F> for RidgeClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The DEVICE-array trait surface: reads `x` back to the host (labels-scale
    /// output, not the perf-critical path — see
    /// [`RidgeClassifier::predict_labels_from_host`] for the no-upload cpu
    /// ingress the PyO3 boundary actually uses) and reuses the same host
    /// decision logic.
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        let x_host = x.to_host(pool);
        let pred = self.predict_labels_from_host(pool, &x_host, shape)?;
        Ok(DeviceArray::from_host(pool, &pred.labels))
    }
}

/// [`RidgeClassifier::decision_function_from_host`]'s result: `values` is
/// length `n_samples` when `n_targets == 1` (binary), else row-major
/// `n_samples × n_targets`.
#[derive(Debug, Clone)]
pub struct RidgeClassifierScores {
    /// The decision-function scores.
    pub values: Vec<f64>,
    /// `1` (binary) or `n_classes` (multiclass) — the row width of `values`.
    pub n_targets: usize,
    /// `false` if ANY element of the query `x` was NaN or ±infinity (checked
    /// once, against the first target column — see the method docs).
    pub operand_finite: bool,
}

/// [`RidgeClassifier::predict_labels_from_host`]'s result.
#[derive(Debug, Clone)]
pub struct RidgeClassifierPrediction {
    /// The predicted label per query row, mapped through `classes_`.
    pub labels: Vec<i32>,
    /// See [`RidgeClassifierScores::operand_finite`].
    pub operand_finite: bool,
}

/// Multiply-adds one worker thread must be given before spawning it pays —
/// the [`mlrs_backend::prims::gram_host`] `HOST_MACS_PER_UNIT` precedent,
/// applied to a bandwidth-bound matvec rather than a FLOP-bound Gram sweep.
const DECISION_MACS_PER_UNIT: usize = 1 << 19;

/// `scores[r, c] = Σ_j x[r,j]·coef[c,j] + bias[c]` for every query row `r` and
/// target column `c`, in ONE pass over `x` (row-blocked across
/// [`mlrs_backend::capability::cpu_launch_units`] worker threads, the
/// `gram_host` precedent): each row is read and widened to `f64` ONCE, then
/// reused for all `k` dot products while it is still hot, rather than
/// re-streaming the whole `n × d` design past the cache once per target
/// column. Returns `(scores, operand_finite)`, row-major `n × k`.
fn decision_multi_host<F>(x: &[F], coef: &[F], bias: &[F], n: usize, d: usize, k: usize) -> (Vec<f64>, bool)
where
    F: Float + CubeElement + Pod,
{
    let macs = n.saturating_mul(d).saturating_mul(k).max(1);
    let units = (macs / DECISION_MACS_PER_UNIT)
        .clamp(1, mlrs_backend::capability::cpu_launch_units().max(1) as usize)
        .min(n.max(1));

    let partials: Vec<(usize, Vec<f64>, bool)> = if units <= 1 {
        vec![decision_chunk::<F>(x, coef, bias, 0, n, d, k)]
    } else {
        let rows_per_unit = n.div_ceil(units);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..units)
                .filter_map(|u| {
                    let r0 = u * rows_per_unit;
                    if r0 >= n {
                        return None;
                    }
                    let r1 = (r0 + rows_per_unit).min(n);
                    Some(scope.spawn(move || decision_chunk::<F>(x, coef, bias, r0, r1, d, k)))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("decision_multi_host worker panicked"))
                .collect()
        })
    };

    let mut scores = vec![0.0f64; n * k];
    let mut operand_finite = true;
    for (r0, chunk, finite) in partials {
        scores[r0 * k..r0 * k + chunk.len()].copy_from_slice(&chunk);
        operand_finite &= finite;
    }
    (scores, operand_finite)
}

/// One contiguous row range of [`decision_multi_host`]'s sweep. Returns
/// `(r0, scores, finite)` so the caller can place the chunk at the right
/// offset regardless of how the row ranges were split.
fn decision_chunk<F>(
    x: &[F],
    coef: &[F],
    bias: &[F],
    r0: usize,
    r1: usize,
    d: usize,
    k: usize,
) -> (usize, Vec<f64>, bool)
where
    F: Float + CubeElement + Pod,
{
    let rows = r1 - r0;
    let mut out = vec![0.0f64; rows * k];
    let mut finite = true;
    let mut row64 = vec![0.0f64; d];
    for (lr, r) in (r0..r1).enumerate() {
        let row = &x[r * d..(r + 1) * d];
        for (dst, &v) in row64.iter_mut().zip(row.iter()) {
            let w = host_to_f64(v);
            finite &= w.is_finite();
            *dst = w;
        }
        for c in 0..k {
            let coef_c = &coef[c * d..(c + 1) * d];
            let mut acc = host_to_f64(bias[c]);
            for (&a, &cv) in row64.iter().zip(coef_c.iter()) {
                acc += a * host_to_f64(cv);
            }
            out[lr * k + c] = acc;
        }
    }
    (r0, out, finite)
}

/// Validate a length-`n` label vector and return `(classes_, class_idx)`:
/// the DISTINCT sorted training labels, and each sample's DENSE position in
/// that list. Shared by both fit entry points so they cannot drift.
///
/// Mirrors `LogisticRegression::fit`'s CR-02 validation verbatim: a label must
/// round to a non-negative integer within `1e-6` (rejected as
/// [`AlgoError::InvalidLabels`] otherwise), and the fit needs at least 2
/// distinct classes.
fn decode_labels<F>(estimator: &'static str, y_host: &[F]) -> Result<(Vec<i64>, Vec<usize>), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let mut raw_labels: Vec<i64> = Vec::with_capacity(y_host.len());
    for &yv in y_host.iter() {
        let lf = host_to_f64(yv);
        let li = lf.round();
        if !(li >= 0.0) || (li - lf).abs() > 1e-6 {
            return Err(AlgoError::InvalidLabels {
                estimator,
                reason: format!("labels must be non-negative integers in 0..n_classes (got {lf})"),
            });
        }
        raw_labels.push(li as i64);
    }
    let mut classes_: Vec<i64> = raw_labels.clone();
    classes_.sort_unstable();
    classes_.dedup();
    if classes_.len() < 2 {
        return Err(AlgoError::InvalidLabels {
            estimator,
            reason: format!(
                "binary/multiclass fit needs at least 2 distinct classes, found {}",
                classes_.len()
            ),
        });
    }
    let class_idx: Vec<usize> = raw_labels
        .iter()
        .map(|&l| {
            classes_
                .binary_search(&l)
                .expect("every raw label is in classes_ by construction")
        })
        .collect();
    Ok((classes_, class_idx))
}

/// Build the `{-1, +1}` target matrix (`n_samples × n_targets`, row-major)
/// from each sample's dense class index — sklearn's
/// `LabelBinarizer(neg_label=-1, pos_label=1)`. Binary (`n_targets == 1`) sets
/// `+1` only for class index `1` (the SECOND sorted class); multiclass sets
/// `+1` in each sample's own class column.
fn encode_targets<F>(class_idx: &[usize], n_targets: usize) -> Vec<F>
where
    F: Float + CubeElement + Pod,
{
    let neg = f64_to_host::<F>(-1.0);
    let pos = f64_to_host::<F>(1.0);
    let mut y = vec![neg; class_idx.len() * n_targets];
    if n_targets == 1 {
        for (r, &ci) in class_idx.iter().enumerate() {
            if ci == 1 {
                y[r] = pos;
            }
        }
    } else {
        for (r, &ci) in class_idx.iter().enumerate() {
            y[r * n_targets + ci] = pos;
        }
    }
    y
}

/// Combine sklearn's `class_weight` with an optional user `sample_weight`,
/// exactly as `_RidgeClassifierMixin._prepare_data` does: `'balanced'` /
/// `Map` weights are computed from the RAW (unweighted) label counts, then
/// multiplied into `sample_weight` (defaulting to all-ones if the user gave
/// none). Returns `None` only when BOTH are absent — the unweighted path,
/// which stays on the faster unweighted route in both fit arms.
///
/// The user-supplied `sample_weight` is validated BEFORE combining (the
/// [`Ridge`] non-negative/finite/non-all-zero rules), and the COMBINED weight
/// is validated again afterward — a user-supplied negative `class_weight`
/// entry would otherwise poison the `√w` rescale silently.
fn combined_sample_weight<F>(
    class_weight: &ClassWeight,
    classes: &[i64],
    class_idx: &[usize],
    sample_weight: Option<&[F]>,
    n_samples: usize,
) -> Result<Option<Vec<f64>>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let sw64 = validate_sample_weight::<F>("ridge_classifier", sample_weight, n_samples)?;

    let n_classes = classes.len();
    let cw_vec: Option<Vec<f64>> = match class_weight {
        ClassWeight::Uniform => None,
        ClassWeight::Balanced => {
            let mut counts = vec![0.0f64; n_classes];
            for &ci in class_idx {
                counts[ci] += 1.0;
            }
            let n = n_samples as f64;
            Some(
                counts
                    .iter()
                    .map(|&c| if c > 0.0 { n / (n_classes as f64 * c) } else { 0.0 })
                    .collect(),
            )
        }
        ClassWeight::Map(pairs) => {
            let mut w = vec![1.0f64; n_classes];
            for &(label, weight) in pairs {
                match classes.binary_search(&label) {
                    Ok(pos) => w[pos] = weight,
                    Err(_) => {
                        return Err(AlgoError::InvalidLabels {
                            estimator: "ridge_classifier",
                            reason: format!(
                                "class_weight contains label {label}, which is not one of the \
                                 training classes"
                            ),
                        })
                    }
                }
            }
            Some(w)
        }
    };

    if sw64.is_none() && cw_vec.is_none() {
        return Ok(None);
    }

    let mut combined = vec![1.0f64; n_samples];
    if let Some(sw) = &sw64 {
        combined.copy_from_slice(sw);
    }
    if let Some(cw) = &cw_vec {
        for (r, &ci) in class_idx.iter().enumerate() {
            combined[r] *= cw[ci];
        }
    }

    // Re-validate the COMBINED weight (a user-supplied negative/zero class
    // weight would otherwise reach the `√w` rescale unchecked).
    let combined_f: Vec<F> = combined.iter().map(|&v| f64_to_host::<F>(v)).collect();
    validate_sample_weight::<F>("ridge_classifier", Some(&combined_f), n_samples)
}
