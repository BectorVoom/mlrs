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
//! ## Three fit routes (D-02)
//! - [`RidgeClassifier::fit_from_host_slice`] — the no-upload HOST arm, used
//!   when [`RidgeClassifier::host_fit_applicable`] is true: `solver` resolves
//!   to `cholesky` (the `positive=False` default) or `lbfgs`
//!   (`positive=True`), AND [`gram_host_applicable`] holds for the shape
//!   (unconditionally true on the cpu backend — `gram_host.rs`'s module docs).
//!   This is the path `RidgeClassifier()` on cpu actually takes: shared Gram,
//!   per-column Cholesky solve (with sklearn's `LinAlgError → svd` retry,
//!   uniform across columns because the Gram is shared) or per-column
//!   non-negative coordinate descent.
//! - [`RidgeClassifier::fit_device_normal_equations`] — the FUSED, fully
//!   device-resident arm (RIDGECLF-CUDA), used for those same two solvers
//!   wherever the host arm does not apply, i.e. on cuda / rocm / wgpu above the
//!   dispatch-cost floor. It is the device twin of the host arm's shape, and it
//!   is the same idea one layer down: `column_means_multi` → `gram_xty_multi`
//!   (Gram formed ONCE with the centering fused into the accumulation, all `K`
//!   `Xᵀy` columns in one further pass) → ONE multi-RHS `cholesky_solve_reg`
//!   with `α` added in-kernel → `ridge_intercept_multi_device`. Nothing crosses
//!   the bus between the design upload and `coef_`.
//! - [`RidgeClassifier::fit_with_sample_weight`]'s DELEGATION route — every
//!   other solver. It calls the FULLY-VALIDATED single-output [`Ridge`]
//!   estimator once per target column, which is what gives this estimator its
//!   complete 8-solver parameter surface without re-deriving
//!   `svd`/`sparse_cg`/`lsqr`/`sag`/`saga` for multi-output from scratch. The
//!   shared-Gram optimization does not apply there (those solvers do not all
//!   consume a single factorable normal matrix), so it is
//!   correct-by-delegation rather than independently perf-tuned — acceptable
//!   because it is reached only by an explicit non-default `solver=`.
//!
//! ## `predict` on device, and why a classifier can win where `Ridge` could not
//! [`RidgeClassifier::predict_labels_device`] runs the whole prediction in ONE
//! `linear_predict_labels` launch: the decision function, its `argmax` (or
//! strict sign for a binary fit) and the `classes_` lookup, without ever
//! materializing the `m × K` score matrix.
//!
//! That fusion is not cosmetic. `Ridge`'s single-target device `predict`
//! measured **10–23× SLOWER** than this crate's own host matvec on a Kaggle
//! P100 (`ridge_predict_device_vs_host_perf_test.rs`), for a reason that
//! generalizes past one adapter: `predict` is `O(m·d)` of compute over the SAME
//! `O(m·d)` transfer that `fit` also pays, a strictly worse
//! compute-to-transfer ratio than `fit`'s `O(m·d²)`, so the GPU never gets
//! `fit`'s chance to pay the upload back. A `RidgeClassifier` changes exactly
//! two things about that arithmetic, and both scale with `K`: the compute
//! becomes `O(m·d·K)` over an unchanged transfer, and the fused kernel shrinks
//! the EGRESS from `m·K` floats to `m` `i32`s (a 26-class, 100 000-row query
//! returns 400 KB instead of 10.4 MB). [`RidgeClassifier::device_predict_applicable`]
//! is where those two effects are traded against the upload; the cpu backend
//! never takes the device arm at all.
//!
//! ## Measured on a Colab Tesla T4 (RIDGECLF-CUDA, f32, min-of-15, upload
//! ## INSIDE the timer, BOTH arms forced)
//!
//! `fit`, device arm against this crate's own host arm on the same VM:
//!
//! | shape | `k` | host | device | |
//! |---|---|---|---|---|
//! | 10 000 × 16 | 2 | 1.55 ms | 1.49 ms | 1.04× |
//! | 10 000 × 64 | 3 | 8.98 | 4.45 | 2.02× |
//! | 100 000 × 16 | 3 | 15.2 | 9.90 | 1.53× |
//! | 100 000 × 16 | 26 | 63.0 | 24.1 | 2.62× |
//! | 100 000 × 64 | 3 | 81.4 | 36.8 | 2.21× |
//! | 100 000 × 64 | 10 | 138 | 42.3 | 3.27× |
//! | 100 000 × 64 | 26 | 267 | 55.3 | **4.83×** |
//! | 100 000 × 128 | 10 | 365 | 120 | 3.04× |
//! | 100 000 × 128 | 26 | 629 | 109 | **5.78×** |
//! | 100 000 × 256 | 10 | 1144 | 346 | 3.31× |
//! | 100 000 × 256 | 26 | 1696 | 368 | **4.61×** |
//!
//! The device arm wins at EVERY rung, and the margin grows with `K` exactly as
//! the shared-Gram design predicts: the host arm's `O(n·d·K)` `XᵀY` term costs
//! it 267 ms at `K = 26` against 81 ms at `K = 3` (`d = 64`), where the device
//! arm moves 36.8 → 55.3 ms for the same change. That difference IS the point
//! of forming the Gram once.
//!
//! **Read the ratio carefully.** That host column is the same code the cpu
//! backend runs, but on Colab's 2-vCPU Xeon @2GHz. The RIDGE-DEFAULT-CUDA
//! campaign measured the 16-thread dev box at roughly 4× that on the same host
//! arm, so against THAT cpu these ratios shrink by about 4× — which would leave
//! the device arm winning at the high-`K`/high-`d` rungs (≈1.2–1.4× at
//! `K = 26`) and LOSING at `K ≤ 3`. That extrapolation has not been measured
//! and is not a claim; the honest headline is the same-machine comparison
//! above, and `d = 256, K = 26` is where a device fit is most clearly worth it
//! on any host.
//!
//! `predict` is a narrower win and is gated accordingly — see
//! [`RIDGECLF_DEVICE_PREDICT_MIN_TARGETS`] for that table.
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

use mlrs_backend::device::Device;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::cholesky::{cholesky_solve_reg, CHOLESKY_MAX_DIM};
use mlrs_backend::prims::gram::{center_scale, column_means_multi, gram_xty_multi, transpose};
use mlrs_backend::prims::gram_host::{centered_gram_multi_xty, gram_host_applicable};
use mlrs_backend::prims::linear_predict::{linear_predict_labels, linear_predict_multi};
use mlrs_backend::prims::nnls::ridge_intercept_multi_device;
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
    /// Where to run the heavy phase (DEVICE-PARAM-01). Covers BOTH fit and
    /// predict — see `device_predict_applicable`.
    device: Device,
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
    /// The SAME coefficients transposed — row-major `n_features_ × n_targets_`
    /// (FEATURE-major), device-resident. `None` until `fit`.
    ///
    /// Not redundant storage for its own sake: this is the layout the fused
    /// device kernels take (`cholesky_solve_reg` emits it for `rhs = k`,
    /// `linear_predict_bias_multi` and `linear_predict_classify` both index
    /// `coef[c·k + t]`), while `coef_` above is sklearn's `coef_` attribute
    /// layout. `n_features · n_targets` is a few thousand floats — one
    /// [`transpose`] launch on the fit path, against a per-`predict` transpose
    /// or a strided (uncoalesced) kernel read on every query row.
    coef_t_: Option<DeviceArray<ActiveRuntime, F>>,
    /// `classes_` as a device-resident `i32` table, so the fused classify
    /// kernel can map its `argmax`/sign straight to the training label without
    /// a host round-trip. Length `classes_.len()` — which is `2` for a binary
    /// fit (where `n_targets_ == 1`) and `n_targets_` for a multiclass one,
    /// exactly what `linear_predict_labels` validates. `None` until `fit`.
    classes_dev_: Option<DeviceArray<ActiveRuntime, i32>>,
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
    /// The execution arm that ACTUALLY ran (`"cpu"` / `"gpu"`), `None` until
    /// `fit`.
    device_: Option<&'static str>,
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
            device: Device::Auto,
            classes_: Vec::new(),
            n_targets_: 0,
            n_features_: 0,
            coef_: None,
            coef_t_: None,
            classes_dev_: None,
            intercept_: None,
            n_iter_: None,
            solver_: None,
            device_: None,
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
            device: self.device,
        }
    }

    /// Does the shared-Gram HOST fit arm ([`RidgeClassifier::fit_from_host_slice`])
    /// apply to this configuration? Mirrors [`Ridge::host_fit_applicable`]
    /// exactly: `true` for the two normal-equations solvers (`cholesky` — the
    /// `positive = false` default — and `lbfgs`), and only where the Gram
    /// formation belongs on the host ([`gram_host_applicable`] — the cpu
    /// backend, or below the fixed dispatch-cost floor on any backend).
    pub fn host_fit_applicable(&self, shape: (usize, usize)) -> bool {
        // `matches!` is a CAPABILITY gate — only the normal-equations solvers
        // have a host-slice ingress — so `device = Cpu` cannot conjure one for
        // the rest. Those keep the device route and say so through `device_`.
        matches!(
            self.solver.resolve(self.positive),
            RidgeSolver::Cholesky | RidgeSolver::Lbfgs
        ) && self
            .device
            .prefers_host(|| gram_host_applicable(shape.0, shape.1))
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
        let (coef_dev, coef_t_dev, classes_dev) =
            stage_fitted_state::<F>(pool, &coef_f, &classes_, n_targets, n_features);

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
            device: self.device,
            classes_,
            n_targets_: n_targets,
            n_features_: n_features,
            coef_: Some(coef_dev),
            coef_t_: Some(coef_t_dev),
            classes_dev_: Some(classes_dev),
            intercept_: Some(DeviceArray::from_host(pool, &intercept_f)),
            n_iter_: None,
            solver_: Some(solver_used),
            // Reached only through `host_fit_applicable`: no upload, no launch.
            device_: Some("cpu"),
            predict_mirror: OnceLock::new(),
            _state: PhantomData,
        })
    }

    /// Does the FUSED, fully device-resident fit arm
    /// ([`RidgeClassifier::fit_device_normal_equations`]) apply to this
    /// configuration?
    ///
    /// `true` for the two NORMAL-EQUATIONS solvers — `cholesky` (the
    /// `positive = false` default) and `lbfgs` (`positive = true`) — which are
    /// exactly the solvers that read only `XᵀX` / `XᵀY` and never the design
    /// itself, so a single shared Gram serves every target column. Every other
    /// `solver` keeps the per-target [`Ridge`] delegation loop, which is what
    /// gives this estimator its complete eight-solver surface without
    /// re-deriving `svd`/`sparse_cg`/`lsqr`/`sag`/`saga` for multi-output.
    ///
    /// This does NOT consult the shape: unlike the host arm's
    /// [`gram_host_applicable`] floor, there is no size below which the fused
    /// arm is the wrong choice *relative to the delegation loop* — the
    /// delegation loop forms the same Gram `n_targets` times over, so the fused
    /// arm dominates it at every shape. The host-vs-device decision is made one
    /// level up, by [`RidgeClassifier::host_fit_applicable`].
    pub fn device_fit_applicable(&self) -> bool {
        matches!(
            self.solver.resolve(self.positive),
            RidgeSolver::Cholesky | RidgeSolver::Lbfgs
        )
    }

    /// The DEVICE-array fit arm — every `(solver, backend)` combination
    /// [`RidgeClassifier::fit_from_host_slice`] does not cover.
    ///
    /// TWO routes, split by [`RidgeClassifier::device_fit_applicable`]:
    ///
    /// - the two NORMAL-EQUATIONS solvers take
    ///   [`RidgeClassifier::fit_device_normal_equations`] — one shared Gram,
    ///   one multi-RHS solve, no host round-trip;
    /// - everything else delegates to the fully-featured [`Ridge`] estimator
    ///   once per target column (correct because the columns are
    ///   mathematically independent — see the module docs — and deliberately
    ///   NOT perf-specialized, since it is reached only by an explicit
    ///   non-default `solver=`).
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
        let y_targets = encode_targets::<F>(&class_idx, n_targets);

        // --- The fused arm: form the shared Gram and all `n_targets` `Xᵀy`
        //     columns ONCE on device, solve them in ONE multi-RHS launch, and
        //     recover the intercepts on device too. `y_targets` is the only
        //     thing that has to be uploaded (the design is already resident),
        //     and it is `n · n_targets` values against the design's `n · d`. ---
        if self.device_fit_applicable() {
            let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y_targets);
            let fitted = self.fit_device_normal_equations(
                pool,
                x,
                &y_dev,
                classes_,
                n_samples,
                n_features,
                n_targets,
                combined64.as_deref(),
            );
            y_dev.release_into(pool);
            return fitted;
        }

        // Only the DELEGATION route needs the weights narrowed to `F` (the
        // fused arm consumed the `f64` form directly), so the conversion lives
        // below the early return rather than above it.
        let combined_f: Option<Vec<F>> = combined64
            .as_ref()
            .map(|v| v.iter().map(|&w| f64_to_host::<F>(w)).collect());

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

        let (coef_dev, coef_t_dev, classes_dev) =
            stage_fitted_state::<F>(pool, &coef_flat, &classes_, n_targets, n_features);

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
            device: self.device,
            classes_,
            n_targets_: n_targets,
            n_features_: n_features,
            coef_: Some(coef_dev),
            coef_t_: Some(coef_t_dev),
            classes_dev_: Some(classes_dev),
            intercept_: Some(DeviceArray::from_host(pool, &intercept_flat)),
            n_iter_,
            solver_: solver_used,
            // The device ingress: `x` arrived as a `DeviceArray`.
            device_: Some("gpu"),
            predict_mirror: OnceLock::new(),
            _state: PhantomData,
        })
    }

    /// The FUSED, fully device-resident normal-equations fit (RIDGECLF-CUDA) —
    /// the arm this estimator exists for on a GPU backend.
    ///
    /// `y_multi` is the `n × n_targets` row-major `{−1, +1}` target matrix
    /// already on the device; `weights` is the COMBINED
    /// `class_weight × sample_weight` vector (`None` for the unweighted
    /// default), still on the host because that is where it was built.
    ///
    /// ## What crosses the bus: nothing
    ///
    /// | phase | what runs | host round-trips |
    /// |---|---|---|
    /// | column means (`x̄`, `ȳ` — weighted or not) | [`column_means_multi`] | none |
    /// | `XᵀX` (`d × d`) + `XᵀY` (`d × k`) | [`gram_xty_multi`], centering FUSED into the accumulation | none |
    /// | solve `(XᵀX + αI)·W = XᵀY` | ONE multi-RHS [`cholesky_solve_reg`] launch, `α` added in-kernel | none |
    /// | `intercept_[t] = ȳ_t − x̄·W[·,t]` | [`ridge_intercept_multi_device`] | none |
    /// | `coef_` in sklearn's `k × d` layout | one [`transpose`] launch | none |
    ///
    /// Against the [`Ridge`]-per-target delegation loop it replaces, this is a
    /// factor of `n_targets` less Gram work (`O(n·d² + n·d·k)` against
    /// `O(n·d²·k)`), `n_targets` fewer design uploads on the paths that upload,
    /// and `4·n_targets` fewer blocking read-backs.
    ///
    /// ## Three places it still touches the host, and why each is bounded
    /// 1. `positive = true` reads the `d² + d·k` normal equations back and runs
    ///    [`ridge_solvers::nonnegative_cd`] per target. The device NNLS prim
    ///    ([`mlrs_backend::prims::nnls::ridge_nnls`]) is single-RHS and cannot
    ///    slice a target column out of the feature-major `XᵀY`, and the
    ///    constrained solve converges in a handful of sweeps over a `d × d`
    ///    matrix — so what would be gained is bounded by a quantity INDEPENDENT
    ///    of `n_samples`, which is the same contract `Ridge`'s `sparse_cg` arm
    ///    documents. The `O(n·d²)` reduction, which IS the fit, still runs on
    ///    device.
    /// 2. An order the device factorization cannot take — `d` past
    ///    [`CHOLESKY_MAX_DIM`], or an adapter whose shared-memory budget the
    ///    wide arm does not fit — falls back to the same read-back-and-solve
    ///    route instead of failing.
    /// 3. sklearn's `except LinAlgError: solver = "svd"` retry: a non-SPD pivot
    ///    re-solves the SAME read-back equations through
    ///    [`ridge_solvers::cholesky_ridge`] in `f64` and, if that fails too,
    ///    [`ridge_solvers::gram_eig_ridge`], reporting `solver_ = "svd"` exactly
    ///    as both `Ridge` arms do. (The `f64` retry is not redundant with the
    ///    device failure it follows: a Gram that has no `F`-precision Cholesky
    ///    can still have an `f64` one, and taking it keeps this arm's
    ///    `solver_` agreeing with the host arm's on the same data.)
    ///
    /// ## Why the Gram is NOT threaded through the factorization's `out`
    /// `Ridge` passes its Gram buffer as `cholesky_solve_reg`'s working output
    /// so the factor overwrites it in place (D-11 gate 2, no parallel `d²`
    /// allocation). This does not, deliberately. Both host fallbacks above need
    /// to READ the Gram after the device solve has been attempted, and threading
    /// it consumes the allocation whether the call succeeds or not — so the
    /// alternative is re-forming an `O(n·d²)` reduction to recover a `d²` buffer.
    /// At `d = 256` that trades 256 KiB of transient device memory against a
    /// second full pass over a design that is three orders of magnitude larger.
    #[allow(clippy::too_many_arguments)]
    fn fit_device_normal_equations(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y_multi: &DeviceArray<ActiveRuntime, F>,
        classes_: Vec<i64>,
        n_samples: usize,
        n_features: usize,
        n_targets: usize,
        weights: Option<&[f64]>,
    ) -> Result<RidgeClassifier<F, Fitted>, AlgoError> {
        let alpha64 = host_to_f64(self.alpha);
        let d = n_features;
        let k = n_targets;

        let NormalEquations {
            xmean,
            ymean,
            gram,
            xty,
        } = self.form_normal_equations(pool, x, y_multi, n_samples, d, k, weights)?;

        // --- Solve. `coef` comes back FEATURE-major (`d × k`), which is both
        //     what the multi-RHS Cholesky emits and what the fused predict
        //     kernels want. ---
        let device_cholesky = !self.positive && d <= CHOLESKY_MAX_DIM;
        let solved = if device_cholesky {
            match cholesky_solve_reg::<F>(pool, &gram, &xty, d, k, alpha64, None) {
                Ok(coef) => Some(coef),
                // A non-SPD pivot is sklearn's `except LinAlgError` (the
                // trigger depends only on `X`/`α`, never on which target column
                // is being solved, so it is uniform across all `k` of them);
                // `NotSquare` is an order this adapter's factorization cannot
                // take. Both fall through to the host route below rather than
                // failing the fit. Anything else is a real geometry bug and is
                // propagated.
                Err(PrimError::NotPositiveDefinite { .. }) | Err(PrimError::NotSquare { .. }) => {
                    None
                }
                Err(e) => {
                    gram.release_into(pool);
                    xty.release_into(pool);
                    xmean.release_into(pool);
                    ymean.release_into(pool);
                    return Err(AlgoError::Prim(e));
                }
            }
        } else {
            None
        };

        let (coef_dk, solver_used) = match solved {
            Some(coef) => {
                gram.release_into(pool);
                xty.release_into(pool);
                (coef, RidgeSolver::Cholesky)
            }
            // The host route: `positive = true`, an order the device
            // factorization cannot take, or its singular-Gram retry. All three
            // read back the SAME `d² + d·k` normal equations — a quantity
            // INDEPENDENT of `n_samples`, which is what makes this bounded (the
            // `Ridge::host_gram` contract).
            None => {
                let gram_h = to_f64(&gram.to_host(pool));
                let xty_h = to_f64(&xty.to_host(pool));
                gram.release_into(pool);
                xty.release_into(pool);
                let route = if self.positive {
                    SolveRoute::NonNegative
                } else {
                    SolveRoute::Cholesky
                };
                solve_multi_host::<F>(
                    pool,
                    &gram_h,
                    &xty_h,
                    d,
                    k,
                    alpha64,
                    route,
                    self.tol,
                    self.max_iter,
                )
            }
        };

        // --- intercept_[t] = ȳ_t − x̄·coef[·,t], on device (D-05: α is NOT
        //     applied here and neither is the `positive` bound — sklearn
        //     constrains only `coef_`). `fit_intercept = false` never launches
        //     it: the means are all-zero by construction there, so the answer
        //     is the zero vector and a launch would only confirm it. ---
        let intercept_dev = if self.fit_intercept {
            ridge_intercept_multi_device::<F>(pool, &xmean, &ymean, &coef_dk, d, k)?
        } else {
            DeviceArray::from_host(pool, &vec![f64_to_host::<F>(0.0); k])
        };
        xmean.release_into(pool);
        ymean.release_into(pool);

        let coef_kd = transpose::<F>(pool, &coef_dk, d, k)?;
        let classes_dev: DeviceArray<ActiveRuntime, i32> =
            DeviceArray::from_host(pool, &classes_as_i32(&classes_));

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
            device: self.device,
            classes_,
            n_targets_: k,
            n_features_: d,
            coef_: Some(coef_kd),
            coef_t_: Some(coef_dk),
            classes_dev_: Some(classes_dev),
            intercept_: Some(intercept_dev),
            // sklearn leaves `n_iter_` unset for BOTH normal-equations solvers
            // (the module-doc table in `ridge.rs`).
            n_iter_: None,
            solver_: Some(solver_used),
            device_: Some("gpu"),
            predict_mirror: OnceLock::new(),
            _state: PhantomData,
        })
    }

    /// Form `(x̄, ȳ, XᵀX, XᵀY)` on device for
    /// [`RidgeClassifier::fit_device_normal_equations`], honouring
    /// `fit_intercept` and the combined sample weights.
    ///
    /// THREE regimes, matching sklearn's `_preprocess_data` + `_rescale_data`
    /// split:
    ///
    /// - **unweighted, `fit_intercept`** — the fused route: only the column
    ///   means are formed, and the subtraction happens inside the accumulation
    ///   kernel, so the `n × d` centered design is never materialized.
    /// - **unweighted, no intercept** — the raw normal equations of `x`/`y`,
    ///   with all-zero means kept so the intercept recovery and the retry path
    ///   have the same operand shapes on every route.
    /// - **weighted** — `√w` multiplies the OPERANDS, not the accumulator, so
    ///   it cannot fuse: the weighted means are formed first
    ///   ([`column_means_multi`]'s `weights` arm), then [`center_scale`] writes
    ///   the `√w`-scaled centered design and targets, and the RAW normal
    ///   equations of those are formed. This is the one route that allocates an
    ///   `n × d` intermediate — and it still never leaves the device, where
    ///   `Ridge`'s weighted arm reads the whole design back to the host to do
    ///   the same thing.
    #[allow(clippy::too_many_arguments)]
    fn form_normal_equations(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y_multi: &DeviceArray<ActiveRuntime, F>,
        n: usize,
        d: usize,
        k: usize,
        weights: Option<&[f64]>,
    ) -> Result<NormalEquations<F>, AlgoError> {
        let zero = f64_to_host::<F>(0.0);

        let Some(w) = weights else {
            let (xmean, ymean) = if self.fit_intercept {
                column_means_multi::<F>(pool, x, y_multi, n, d, k, None)?
            } else {
                (
                    DeviceArray::from_host(pool, &vec![zero; d]),
                    DeviceArray::from_host(pool, &vec![zero; k]),
                )
            };
            let means = self.fit_intercept.then_some((&xmean, &ymean));
            let (gram, xty) = gram_xty_multi::<F>(pool, x, y_multi, means, n, d, k)?;
            return Ok(NormalEquations {
                xmean,
                ymean,
                gram,
                xty,
            });
        };

        // The `sqrt` runs on the HOST, over `n` values the caller already holds
        // there — one cheap pass, and it keeps the kernel off the `f64`
        // transcendental path some wgpu adapters lack entirely.
        let w_f: Vec<F> = w.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let sqrt_w_f: Vec<F> = w.iter().map(|&v| f64_to_host::<F>(v.sqrt())).collect();
        let w_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &w_f);
        let sqrt_w_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &sqrt_w_f);

        let (xmean, ymean) = if self.fit_intercept {
            column_means_multi::<F>(pool, x, y_multi, n, d, k, Some(&w_dev))?
        } else {
            (
                DeviceArray::from_host(pool, &vec![zero; d]),
                DeviceArray::from_host(pool, &vec![zero; k]),
            )
        };
        w_dev.release_into(pool);

        let xw = center_scale::<F>(pool, x, &xmean, &sqrt_w_dev, n, d)?;
        let yw = center_scale::<F>(pool, y_multi, &ymean, &sqrt_w_dev, n, k)?;
        sqrt_w_dev.release_into(pool);

        let (gram, xty) = gram_xty_multi::<F>(pool, &xw, &yw, None, n, d, k)?;
        xw.release_into(pool);
        yw.release_into(pool);

        Ok(NormalEquations {
            xmean,
            ymean,
            gram,
            xty,
        })
    }
}

/// The device-resident normal equations of a multi-target fit, plus the means
/// the intercept recovery needs — what
/// [`RidgeClassifier::form_normal_equations`] returns.
struct NormalEquations<F> {
    /// Column means of the design (length `d`); all-zero when
    /// `fit_intercept = false`.
    xmean: DeviceArray<ActiveRuntime, F>,
    /// Column means of the `{−1, +1}` targets (length `k`); all-zero when
    /// `fit_intercept = false`.
    ymean: DeviceArray<ActiveRuntime, F>,
    /// `XᵀX` (`d × d` row-major).
    gram: DeviceArray<ActiveRuntime, F>,
    /// `XᵀY` (`d × k` row-major — FEATURE-major).
    xty: DeviceArray<ActiveRuntime, F>,
}

/// Which host solver [`solve_multi_host`] runs for every target column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolveRoute {
    /// `positive = false`: [`ridge_solvers::cholesky_ridge`], with sklearn's
    /// `LinAlgError → svd` retry through [`ridge_solvers::gram_eig_ridge`].
    Cholesky,
    /// `positive = true`: [`ridge_solvers::nonnegative_cd`].
    NonNegative,
}

/// Solve `(gram + αI)·W = xty` for all `k` targets on the HOST, returning the
/// coefficients as a device-resident `d × k` (FEATURE-major) buffer — the
/// layout the device solve produces — plus the solver sklearn would report.
///
/// `xty` is `d × k` row-major, so target `t`'s right-hand side is the stride-`k`
/// column `xty[i·k + t]`. The Gram is SHARED across every column (it does not
/// depend on `y`), so a singular-Gram fallback either fires for all `k` targets
/// or for none — which is why `solver_` is one value, not `k` of them.
#[allow(clippy::too_many_arguments)]
fn solve_multi_host<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    gram: &[f64],
    xty: &[f64],
    d: usize,
    k: usize,
    alpha: f64,
    route: SolveRoute,
    tol: f64,
    max_iter: Option<usize>,
) -> (DeviceArray<ActiveRuntime, F>, RidgeSolver)
where
    F: Float + CubeElement + Pod,
{
    let mut coef_dk = vec![0.0f64; d * k];
    let mut used = match route {
        SolveRoute::Cholesky => RidgeSolver::Cholesky,
        SolveRoute::NonNegative => RidgeSolver::Lbfgs,
    };
    for t in 0..k {
        let xty_t: Vec<f64> = (0..d).map(|i| xty[i * k + t]).collect();
        let w = match route {
            SolveRoute::NonNegative => {
                ridge_solvers::nonnegative_cd(gram, &xty_t, d, alpha, tol, max_iter).0
            }
            SolveRoute::Cholesky => match ridge_solvers::cholesky_ridge(gram, &xty_t, d, alpha) {
                Some(w) => w,
                None => {
                    used = RidgeSolver::Svd;
                    ridge_solvers::gram_eig_ridge(gram, &xty_t, d, alpha)
                }
            },
        };
        for (i, &v) in w.iter().enumerate() {
            coef_dk[i * k + t] = v;
        }
    }
    let host: Vec<F> = coef_dk.iter().map(|&v| f64_to_host::<F>(v)).collect();
    (DeviceArray::from_host(pool, &host), used)
}

/// Widen a device read-back to `f64` for a host solver.
fn to_f64<F>(v: &[F]) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
{
    v.iter().map(|&x| host_to_f64(x)).collect()
}

/// `classes_` as the `i32` table [`linear_predict_labels`] indexes: length 2
/// for a binary fit (`[negative, positive]`) and length `n_classes` for a
/// multiclass one — which is the same thing, since `classes_` IS that table in
/// both cases.
///
/// The narrowing matches every other classifier's label egress in this
/// codebase (`predict_labels` returns `i32`), so a label outside `i32` was
/// already unrepresentable before reaching here.
fn classes_as_i32(classes: &[i64]) -> Vec<i32> {
    classes.iter().map(|&c| c as i32).collect()
}

/// Stage a host-solved `coef_` onto the device in BOTH layouts the estimator
/// keeps, plus the `i32` `classes_` table.
///
/// Returns `(coef_ (k × d), coef_t_ (d × k), classes_dev_)`. The transpose runs
/// on the host here because the coefficients are already there — the device
/// [`transpose`] prim exists for the arm where they are not.
fn stage_fitted_state<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    coef_kd: &[F],
    classes: &[i64],
    k: usize,
    d: usize,
) -> (
    DeviceArray<ActiveRuntime, F>,
    DeviceArray<ActiveRuntime, F>,
    DeviceArray<ActiveRuntime, i32>,
)
where
    F: Float + CubeElement + Pod,
{
    debug_assert_eq!(coef_kd.len(), k * d);
    let mut coef_dk: Vec<F> = vec![f64_to_host::<F>(0.0); d * k];
    for t in 0..k {
        for c in 0..d {
            coef_dk[c * k + t] = coef_kd[t * d + c];
        }
    }
    (
        DeviceArray::from_host(pool, coef_kd),
        DeviceArray::from_host(pool, &coef_dk),
        DeviceArray::from_host(pool, &classes_as_i32(classes)),
    )
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
    device: Device,
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
    /// Pin the execution arm (DEVICE-PARAM-01). Covers BOTH `fit` and the
    /// host-ingress `predict`; [`Device::Auto`] keeps the existing heuristics
    /// (and their `MLRS_*` A/B flags). The arm that ran is reported by
    /// [`RidgeClassifier::device`].
    pub fn device(mut self, v: Device) -> Self {
        self.device = v;
        self
    }

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
            device: self.device,
            classes_: Vec::new(),
            n_targets_: 0,
            n_features_: 0,
            coef_: None,
            coef_t_: None,
            classes_dev_: None,
            intercept_: None,
            n_iter_: None,
            solver_: None,
            device_: None,
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
    /// The execution arm that ACTUALLY ran, `"cpu"` or `"gpu"`
    /// (DEVICE-PARAM-01). A preference the configuration cannot honour — a
    /// solver with no host-slice ingress — shows up here rather than silently.
    pub fn device(&self) -> &'static str {
        self.device_
            .expect("device_ is Some by construction on RidgeClassifier<F, Fitted>")
    }

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

    /// Validate a query `x` against the fitted geometry — the guard both
    /// device ingresses share, byte-identical to
    /// [`RidgeClassifier::decision_function_from_host`]'s.
    fn check_query(
        &self,
        x_len: usize,
        (n_samples, n_features): (usize, usize),
    ) -> Result<(), AlgoError> {
        if n_samples == 0 || n_features == 0 || x_len != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x_len,
            }));
        }
        if n_features != self.n_features_ {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: n_features,
                rhs: self.n_features_,
            }));
        }
        Ok(())
    }

    /// `decision_function` from a DEVICE-resident `x` — the `n_samples ×
    /// n_targets` row-major scores, device-resident, in ONE fused
    /// [`linear_predict_multi`] launch.
    ///
    /// The scores' finiteness is NOT reported here (unlike the host twin): a
    /// device-resident operand reached the device through the PyO3 ingress
    /// validator, which already rejects NaN/±inf.
    pub fn decision_function_device(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        self.check_query(x.len(), shape)?;
        let coef_t = self
            .coef_t_
            .as_ref()
            .expect("coef_t_ is Some by construction on RidgeClassifier<F, Fitted>");
        let intercept = self
            .intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on RidgeClassifier<F, Fitted>");
        Ok(linear_predict_multi::<F>(
            pool,
            x,
            coef_t,
            intercept,
            shape,
            self.n_targets_,
        )?)
    }

    /// `predict` from a DEVICE-resident `x` — the length-`n_samples` `i32`
    /// class labels, device-resident, in ONE fused
    /// [`linear_predict_labels`] launch that computes the decision function,
    /// takes its `argmax` (or STRICT sign for a binary fit) and maps the
    /// winner through `classes_` without ever materializing the scores.
    ///
    /// This is the on-device `predict` (RIDGECLF-CUDA). Its host twin is
    /// [`RidgeClassifier::predict_labels_from_host`]; which one a caller should
    /// take is [`RidgeClassifier::device_predict_applicable`].
    pub fn predict_labels_device(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        self.check_query(x.len(), shape)?;
        let coef_t = self
            .coef_t_
            .as_ref()
            .expect("coef_t_ is Some by construction on RidgeClassifier<F, Fitted>");
        let intercept = self
            .intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on RidgeClassifier<F, Fitted>");
        let classes = self
            .classes_dev_
            .as_ref()
            .expect("classes_dev_ is Some by construction on RidgeClassifier<F, Fitted>");
        Ok(linear_predict_labels::<F>(
            pool,
            x,
            coef_t,
            intercept,
            classes,
            shape,
            self.n_targets_,
        )?)
    }

    /// Should a caller holding a HOST-resident query take
    /// [`RidgeClassifier::predict_labels_device`] (uploading `x`) rather than
    /// [`RidgeClassifier::predict_labels_from_host`]?
    ///
    /// `false` on the **cpu** backend, always: "device" memory IS host memory
    /// there, so the upload is a pure `memcpy` of the whole query and the
    /// cubecl launch spawns one OS thread per unit at `-O0`
    /// (`prims::linear_predict`'s module docs, §"The cpu backend does NOT take
    /// either kernel").
    ///
    /// On the device backends the answer is a MEASURED threshold on
    /// `n_targets`, and the reason it is not simply "always" is the finding
    /// this estimator inherits from `Ridge`: a single-target device predict
    /// measured 10–23× SLOWER than the same crate's host matvec on a P100,
    /// because `predict` is `O(m·d)` of compute over an `O(m·d)` transfer — the
    /// one linear-model operation whose compute-to-transfer ratio a GPU cannot
    /// improve. A `RidgeClassifier` changes that ratio in exactly two ways, and
    /// both scale with `n_targets`: the compute becomes `O(m·d·k)` over the
    /// same transfer, and the fused classify kernel shrinks the EGRESS from
    /// `m·k` floats to `m` `i32`s.
    ///
    /// [`RIDGECLF_DEVICE_PREDICT_MIN_TARGETS`] is where those two effects
    /// overtake the upload — see that constant for the P100 sweep it comes
    /// from. `MLRS_RIDGECLF_PREDICT_DEVICE=1`/`=0` forces either arm at any
    /// shape, which is how the threshold is A/B'd on a new backend
    /// (`ridge_classifier_cuda_perf_test.rs`).
    ///
    /// Note the asymmetry with the `PredictLabels` trait impl below, which
    /// takes the device kernel UNCONDITIONALLY: an `x` that is already
    /// device-resident has no upload left to amortize, so the gate only
    /// concerns callers whose query starts on the host.
    pub fn device_predict_applicable(&self) -> bool {
        // `device` covers PREDICT as well as fit: a caller who pinned `"cpu"`
        // to avoid an upload means it for the query matrix too, and splitting
        // the parameter across the two phases would be a surprise. `Auto` keeps
        // the original abflag-then-shape ladder verbatim.
        self.device.prefers_device(|| {
            match mlrs_backend::abflag::var("MLRS_RIDGECLF_PREDICT_DEVICE").as_deref() {
                Some("0") => return false,
                Some(_) => return true,
                None => {}
            }
            #[cfg(feature = "cpu")]
            {
                false
            }
            #[cfg(not(feature = "cpu"))]
            {
                self.n_targets_ >= RIDGECLF_DEVICE_PREDICT_MIN_TARGETS
                    && self.n_features_ <= RIDGECLF_DEVICE_PREDICT_MAX_FEATURES
            }
        })
    }
}

/// `n_targets` at or above which [`RidgeClassifier::device_predict_applicable`]
/// sends a host-resident query through the fused DEVICE predict — paired with
/// [`RIDGECLF_DEVICE_PREDICT_MAX_FEATURES`], which caps the same gate on `d`.
///
/// ## Both values are a T4 measurement, not an inference
/// `ridge_classifier_cuda_perf_test.rs` on a Colab Tesla T4, min-of-15, f32,
/// `n_query = 100 000`, upload INSIDE the timer, device ÷ host:
///
/// | `d` | `k = 2` | `k = 3` | `k = 5` | `k = 10` | `k = 26` |
/// |---|---|---|---|---|---|
/// | 16 | 0.60× | 1.22× | | | **6.31×** |
/// | 64 | | 0.66× | **1.52×** | **1.99×** | **3.35×** |
/// | 128 | | | | **1.20×** | **2.01×** |
/// | 256 | | | | 0.63× | 1.42× |
///
/// `k ≥ 5 AND d ≤ 128` is the largest rectangle containing NO measured loss:
/// every one of its six rungs wins, by 1.20–6.31×. Two wins are deliberately
/// left outside it — `(16, 3)` at 1.22× and `(256, 26)` at 1.42× — because a
/// rung immediately adjacent to each LOSES (`(64, 3)` at 0.66×, `(256, 10)` at
/// 0.63×), and shipping a gate that straddles a measured regression to capture
/// a 1.2–1.4× win is the wrong trade.
///
/// ## Why the gate needs `d` at all
/// A first-principles model says it should not: the device moves `m·d` bytes
/// and does `m·d·k` MACs, the host does `m·d·k` MACs, so the ratio is
/// `k / (a + b·k)` with `d` cancelling. The `d = 256` row is where that model
/// breaks, and it breaks for a reason this codebase has measured before — the
/// design upload's effective throughput DEGRADES as the operand grows (0.33
/// GB/s at 102 MiB against 0.81 at 6.4 MiB, `mlrs-ridge-positive-cuda`), which
/// is a per-call allocation cost rather than a link rate. So `a` grows with
/// `d` and the cancellation fails at exactly the widest shapes.
///
/// Both rungs on the `d = 256` row reproduced across two independent sweeps
/// (0.62× / 0.63× at `k = 10`), so this is not sampling noise.
///
/// `MLRS_RIDGECLF_PREDICT_DEVICE=1`/`=0` forces either arm at any shape;
/// `scripts/colab_ridge_classifier.py` §E runs that A/B, which is how these two
/// constants should be re-placed on a different adapter. They are calibrated on
/// ONE GPU (see the `mlrs-feedback-verify-on-target-hardware` project memory).
#[cfg_attr(feature = "cpu", allow(dead_code))]
const RIDGECLF_DEVICE_PREDICT_MIN_TARGETS: usize = 5;

/// `n_features` above which the fused device predict LOSES to the host matvec
/// regardless of `n_targets` — see [`RIDGECLF_DEVICE_PREDICT_MIN_TARGETS`] for
/// the T4 table both constants come from.
#[cfg_attr(feature = "cpu", allow(dead_code))]
const RIDGECLF_DEVICE_PREDICT_MAX_FEATURES: usize = 128;

impl<F> PredictLabels<F> for RidgeClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The DEVICE-array trait surface — [`RidgeClassifier::predict_labels_device`]
    /// verbatim. An `x` that is ALREADY device-resident has no upload left to
    /// amortize, so the fused kernel is unconditionally the right arm here; the
    /// [`RidgeClassifier::device_predict_applicable`] gate exists for the
    /// callers whose query starts on the host and would have to pay for the
    /// crossing.
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        self.predict_labels_device(pool, x, shape)
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
