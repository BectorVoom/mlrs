//! Linear-model `#[pyclass]` wrappers (PY-01/PY-02/PY-05): `PyLinearRegression`,
//! `PyRidge`, `PyLasso`, `PyElasticNet`, `PyLogisticRegression`.
//!
//! Each is the `Fit` + (`Predict` | `Transform` | `PredictLabels` | `PredictProba`)
//! surface of its `mlrs_algos` estimator, dtype-dispatched (D-06) through the
//! macro-emitted `Any<Name>` enum. The four regressors expose `predict`
//! ([`Predict`]); `LogisticRegression` exposes `predict` (label vote via
//! [`PredictLabels`], i32) and `predict_proba` (softmax via [`PredictProba`]) and
//! the sklearn-named hyperparameter `C` (mapped to the Rust `c` field).

use pyo3::prelude::*;

use mlrs_algos::linear::bayesian_ridge::BayesianRidge;
use mlrs_algos::linear::elastic_net::ElasticNet;
use mlrs_algos::linear::lasso::Lasso;
use mlrs_algos::linear::linear_regression::LinearRegression;
use mlrs_algos::linear::linear_svc::LinearSVC;
use mlrs_algos::linear::linear_svr::LinearSVR;
use mlrs_algos::linear::logistic::LogisticRegression;
use mlrs_algos::linear::mbsgd_classifier::MBSGDClassifier;
use mlrs_algos::linear::mbsgd_regressor::MBSGDRegressor;
use mlrs_algos::linear::ridge::{Ridge, RidgeSolver};
use mlrs_algos::linear::sgd_config::{LearningRate, Loss, Penalty};
// Phase 16 (D-01): every estimator in this file now consumes the typestate
// surface — the legacy trait glob has been removed. The typestate
// lifecycle + accessor traits are imported under disambiguating `Typestate*`
// aliases (mirrors `cluster.rs`) and called via UFCS at each fit/predict arm so
// the `fit`/`predict`/`predict_labels`/`predict_proba` method-name collisions
// across the trait family resolve unambiguously.
use mlrs_algos::error::AlgoError;
use mlrs_algos::typestate::{
    Fit as TypestateFit, Fitted as AlgoFitted, Predict as TypestatePredict,
    PredictLabels as TypestatePredictLabels, PredictProba as TypestatePredictProba,
};
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::linear_predict::HostPrediction;
use mlrs_backend::runtime::ActiveRuntime;

use crate::egress::{f32_vec_to_pyarrow, f64_vec_to_pyarrow, i32_vec_to_pyarrow};
use crate::errors::{algo_err_to_py, build_err_to_py, nonfinite_input_err, not_fitted};
use crate::ingress::{
    as_f32, as_f64, capsule_to_array, float_dtype, host_slice_f32, host_slice_f64, validated_f32,
    validated_f64, FloatDtype,
};

// ---------------------------------------------------------------------------
// Shared `predict` body for the four dense linear regressors
// ---------------------------------------------------------------------------

/// The `predict_from_host` surface `LinearRegression` / `Ridge` / `Lasso` /
/// `ElasticNet` all expose, so [`dense_predict_f32`] / [`dense_predict_f64`] can
/// be written ONCE instead of pasted into eight `#[pymethods]` bodies.
///
/// A trait rather than a `macro_rules!` because `#[pymethods]` is a proc-macro
/// attribute that reads the impl block's tokens BEFORE `macro_rules!` expansion
/// — a macro invoked inside the block would not be registered as a Python
/// method at all.
trait DensePredictHost<F>: Sync {
    /// `y = X·coef_ + intercept_` from a borrowed HOST `x`, plus the operand
    /// finiteness verdict (`mlrs_algos::linear::elastic_net::predict_linear_from_host`).
    fn predict_from_host_slice(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<HostPrediction<F>, AlgoError>;
}

/// Emit the [`DensePredictHost`] impls for each estimator, at both float widths.
///
/// Spelled at the CONCRETE `f32`/`f64` rather than generically over the algos
/// layer's `F: Float + CubeElement + Pod`, because this crate does not depend on
/// `cubecl` and so cannot name those bounds. `f32`/`f64` are the only widths the
/// PyO3 surface ever instantiates anyway — the whole binding is a two-arm dtype
/// dispatch (D-06).
macro_rules! impl_dense_predict_host {
    (@one $estimator:ident, $float:ty) => {
        impl DensePredictHost<$float> for $estimator<$float, AlgoFitted> {
            fn predict_from_host_slice(
                &self,
                pool: &mut BufferPool<ActiveRuntime>,
                x: &[$float],
                shape: (usize, usize),
            ) -> Result<HostPrediction<$float>, AlgoError> {
                self.predict_from_host(pool, x, shape)
            }
        }
    };
    ($($estimator:ident),+ $(,)?) => {
        $(
            impl_dense_predict_host!(@one $estimator, f32);
            impl_dense_predict_host!(@one $estimator, f64);
        )+
    };
}

// `LinearSVR` is in the list because its prediction IS the same
// `X·coef_ + intercept_` matvec — only its `fit` differs — so it takes the same
// no-upload / no-list `predict` body rather than a fifth hand-written one.
// `BayesianRidge` predicts through the same `X·coef_ + intercept_` matvec as
// the rest; only its `fit` differs. Its `predict(return_std=True)` second
// return value is a SEPARATE method (`predict_std_*`) because sklearn returns
// the mean whether or not `return_std` is set.
impl_dense_predict_host!(
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LinearSVR,
    BayesianRidge,
);

/// The whole `predict` body for a fitted f32 dense linear regressor: borrow the
/// validated Arrow values, predict, reject a non-finite operand, hand the result
/// back over Arrow. GIL released around the compute.
///
/// Two departures from the `predict_f32` template the rest of this file uses,
/// both measured on the cpu backend and both on the ingress/egress rather than
/// the arithmetic (see the LINEAR-PRED-CPU notes on `prims::linear_predict` and
/// [`f32_vec_to_pyarrow`]):
///
/// 1. **No upload.** [`host_slice_f32`] runs the SAME hard-reject bridge
///    validator as `validated_f32` but BORROWS the Arrow values instead of
///    copying them to a `DeviceArray`, and `predict_from_host` routes cpu to a
///    host matvec over that borrow. On cpu the upload was a plain memcpy of the
///    whole test matrix — 13.5 ms for 64 MiB, three times sklearn's entire
///    `predict`. wgpu/cuda/rocm still upload and run the fused device kernel,
///    inside `predict_from_host`.
/// 2. **No Python list.** The result goes back over the Arrow C data interface,
///    which numpy views in place, instead of being expanded into one boxed
///    `float` per row (16.2 ms for 1 000 000 rows).
///
/// It also OWNS the NaN/inf rejection that `check_array` used to perform on this
/// path — the `mlrs.linear` wrappers pass `ensure_all_finite=False` because
/// `predict_from_host` reports the same verdict from the pass it was already
/// making. [`nonfinite_input_err`] reproduces `check_array`'s exact message so
/// the contract is unchanged from Python.
fn dense_predict_f32<'py, E: DensePredictHost<f32>>(
    py: Python<'py>,
    x: &Bound<'_, PyAny>,
    (rows, cols): (usize, usize),
    est: &E,
) -> PyResult<Bound<'py, PyAny>> {
    let xa = capsule_to_array(x)?;
    let out = py.detach(|| -> PyResult<Vec<f32>> {
        let mut pool = crate::lock_pool();
        let xh = host_slice_f32(as_f32(&xa)?)?;
        let pred = est
            .predict_from_host_slice(&mut pool, xh, (rows, cols))
            .map_err(algo_err_to_py)?;
        if !pred.operand_finite {
            return Err(nonfinite_input_err(xh, "float32"));
        }
        Ok(pred.values)
    })?;
    f32_vec_to_pyarrow(py, out)
}

/// f64 twin of [`dense_predict_f32`]. No `guard_f64` here: reaching this means
/// the estimator is already in its `F64` fitted arm, which `fit` could only have
/// produced on an f64-capable backend (D-04).
fn dense_predict_f64<'py, E: DensePredictHost<f64>>(
    py: Python<'py>,
    x: &Bound<'_, PyAny>,
    (rows, cols): (usize, usize),
    est: &E,
) -> PyResult<Bound<'py, PyAny>> {
    let xa = capsule_to_array(x)?;
    let out = py.detach(|| -> PyResult<Vec<f64>> {
        let mut pool = crate::lock_pool();
        let xh = host_slice_f64(as_f64(&xa)?)?;
        let pred = est
            .predict_from_host_slice(&mut pool, xh, (rows, cols))
            .map_err(algo_err_to_py)?;
        if !pred.operand_finite {
            return Err(nonfinite_input_err(xh, "float64"));
        }
        Ok(pred.values)
    })?;
    f64_vec_to_pyarrow(py, out)
}

// ---------------------------------------------------------------------------
// LinearRegression — Fit + Predict; coef_ / intercept_
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyLinearRegression,
    algo:  mlrs_algos::linear::linear_regression::LinearRegression,
    unfit: { fit_intercept: bool },
}

/// sklearn-compatible `LinearRegression` (ordinary least squares).
#[pyclass(name = "LinearRegression")]
pub struct PyLinearRegression {
    inner: AnyLinearRegression,
}

impl PyLinearRegression {
    /// Rust-callable default constructor (for the cross-crate smoke test, which
    /// proves the macro-expanded wrapper instantiates in the `Unfit` arm without
    /// a Python interpreter). Mirrors the `#[new]` defaults.
    pub fn unfit_default() -> Self {
        Self { inner: AnyLinearRegression::Unfit { fit_intercept: true } }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyLinearRegression::Unfit { .. })
    }
}

#[pymethods]
impl PyLinearRegression {
    /// `LinearRegression(fit_intercept=True)`.
    #[new]
    #[pyo3(signature = (fit_intercept = true))]
    fn new(fit_intercept: bool) -> Self {
        Self {
            inner: AnyLinearRegression::Unfit { fit_intercept },
        }
    }

    /// Fit on `x` (`rows × cols`, row-major) and target `y`. GIL released around
    /// the device call (PY-03); f64 guarded on an f64-incapable backend (D-04).
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&xa)?;
        let fit_intercept = match &self.inner {
            AnyLinearRegression::Unfit { fit_intercept } => *fit_intercept,
            _ => true,
        };
        let fitted = py.detach(|| -> PyResult<AnyLinearRegression> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = LinearRegression::<f32>::builder()
                        .fit_intercept(fit_intercept)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyLinearRegression::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = LinearRegression::<f64>::builder()
                        .fit_intercept(fit_intercept)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyLinearRegression::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x)` → a length-`rows` **pyarrow** float array (D-03). See
    /// [`dense_predict_f32`] for the body all four dense linear regressors share
    /// and why it departs from this file's `predict_f32` template.
    fn predict_f32<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyLinearRegression::F32(est) => dense_predict_f32(py, x, (rows, cols), est),
            _ => Err(not_fitted("linear_regression", "predict (f32 path)")),
        }
    }

    fn predict_f64<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyLinearRegression::F64(est) => dense_predict_f64(py, x, (rows, cols), est),
            _ => Err(not_fitted("linear_regression", "predict (f64 path)")),
        }
    }

    /// Host `coef_` (f32 arm) or `NotFitted`.
    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearRegression::F32(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("linear_regression", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearRegression::F64(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("linear_regression", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearRegression::F32(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("linear_regression", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearRegression::F64(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("linear_regression", "intercept_ (f64)")),
        }
    }

    /// `True` once `fit` has run (either dtype arm), for the shim's fitted-check.
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyLinearRegression::Unfit { .. })
    }
    /// `"f32"`/`"f64"` of the fitted arm, or `None` before `fit`.
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyLinearRegression::Unfit { .. } => None,
            AnyLinearRegression::F32(_) => Some("f32"),
            AnyLinearRegression::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// Ridge — Fit + Predict; the FULL sklearn parameter surface
//   alpha, fit_intercept, copy_X, max_iter, tol, solver, positive, random_state
//   + fit(..., sample_weight) and the n_iter_ / solver_ fitted attributes
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyRidge,
    algo:  mlrs_algos::linear::ridge::Ridge,
    unfit: {
        alpha: f64,
        fit_intercept: bool,
        copy_x: bool,
        max_iter: Option<usize>,
        tol: f64,
        solver: String,
        positive: bool,
        random_state: Option<u64>,
    },
}

/// The verbatim ctor hyperparameters, carried from the `Unfit` arm to `fit`.
/// `solver` stays a STRING until `fit` because that is what the sklearn shim
/// passes; the `RidgeSolver` parse (and its `UnknownSolver` rejection) happens
/// with the rest of the `build()` validation (D-09).
struct RidgeParams {
    alpha: f64,
    fit_intercept: bool,
    copy_x: bool,
    max_iter: Option<usize>,
    tol: f64,
    solver: String,
    positive: bool,
    random_state: Option<u64>,
}

/// sklearn-compatible `Ridge` (L2-penalized least squares).
#[pyclass(name = "Ridge")]
pub struct PyRidge {
    inner: AnyRidge,
    /// sklearn's `n_iter_`, captured at `fit` (the fitted arms are consumed into
    /// `AnyRidge`, and a `#[pyclass]` getter cannot reach through the dtype
    /// dispatch generically, so the two scalars are mirrored here).
    n_iter: Option<usize>,
    /// sklearn's `solver_` — the solver that actually ran, after `auto`
    /// resolution and any singular-Gram fallback.
    solver_used: Option<String>,
}

impl PyRidge {
    /// Rust-callable default constructor for the smoke test. See
    /// [`PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyRidge::Unfit {
                alpha: 1.0,
                fit_intercept: true,
                copy_x: true,
                max_iter: None,
                tol: 1e-4,
                solver: "auto".to_string(),
                positive: false,
                random_state: None,
            },
            n_iter: None,
            solver_used: None,
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyRidge::Unfit { .. })
    }

    /// Read back the ctor hyperparameters (WR-02: the typestate wrapper rebuilds
    /// from these at every `fit`, so a second `fit` of the same object works).
    fn params(&self) -> RidgeParams {
        match &self.inner {
            AnyRidge::Unfit {
                alpha,
                fit_intercept,
                copy_x,
                max_iter,
                tol,
                solver,
                positive,
                random_state,
            } => RidgeParams {
                alpha: *alpha,
                fit_intercept: *fit_intercept,
                copy_x: *copy_x,
                max_iter: *max_iter,
                tol: *tol,
                solver: solver.clone(),
                positive: *positive,
                random_state: *random_state,
            },
            // Already fitted: the shim always constructs a fresh wrapper per
            // `fit`, so this arm is unreachable in practice; fall back to
            // sklearn's defaults rather than panicking.
            _ => RidgeParams {
                alpha: 1.0,
                fit_intercept: true,
                copy_x: true,
                max_iter: None,
                tol: 1e-4,
                solver: "auto".to_string(),
                positive: false,
                random_state: None,
            },
        }
    }
}

/// Build an unfit `Ridge<F>` from the ctor params. Monomorphized per float width
/// by the macro below so the `solver` parse + the eight builder setters are
/// written once.
macro_rules! ridge_build {
    ($float:ty, $p:expr) => {{
        let solver = RidgeSolver::try_from($p.solver.as_str()).map_err(build_err_to_py)?;
        Ridge::<$float>::builder()
            .alpha($p.alpha)
            .fit_intercept($p.fit_intercept)
            .copy_x($p.copy_x)
            .max_iter($p.max_iter)
            .tol($p.tol)
            .solver(solver)
            .positive($p.positive)
            .random_state($p.random_state)
            .build::<$float>()
            .map_err(build_err_to_py)?
    }};
}

/// Build the estimator and run whichever `fit` ingress its configuration, shape
/// and backend call for. Monomorphized per float width by the caller so the
/// two-arm branch is written once.
///
/// The branch has to happen HERE, before ingress, because the two entry points
/// take different operand types: `Ridge::fit_from_host_slice` borrows the Arrow
/// values directly (`host_slice_*`) and `Ridge::fit_with_sample_weight` needs a
/// device upload (`validated_*`). On the host arm — either normal-equations
/// solver (`cholesky`, i.e. the DEFAULT, or `lbfgs`) on the cpu backend, or
/// below the dispatch-cost floor on any backend — the `n·d` design is therefore
/// never copied at all. Both helpers run the SAME hard-reject
/// bridge validator, so the ingress contract is identical either way.
macro_rules! ridge_fit_dispatch {
    ($float:ty, $p:expr, $xa:expr, $ya:expr, $swa:expr, $rows:expr, $cols:expr,
     $pool:expr, $as:ident, $host_slice:ident, $validated:ident) => {{
        let est = ridge_build!($float, $p);
        let sw = match $swa.as_ref() {
            Some(a) => Some($host_slice($as(a)?)?),
            None => None,
        };
        if est.host_fit_applicable(($rows, $cols)) {
            let xh = $host_slice($as(&$xa)?)?;
            let yh = $host_slice($as(&$ya)?)?;
            est.fit_from_host_slice(&mut $pool, xh, yh, ($rows, $cols), sw)
                .map_err(algo_err_to_py)?
        } else {
            let xd = $validated($as(&$xa)?, &mut $pool)?;
            let yd = $validated($as(&$ya)?, &mut $pool)?;
            est.fit_with_sample_weight(&mut $pool, &xd, Some(&yd), ($rows, $cols), sw)
                .map_err(algo_err_to_py)?
        }
    }};
}

#[pymethods]
impl PyRidge {
    /// `Ridge(alpha=1.0, fit_intercept=True, copy_X=True, max_iter=None,
    /// tol=1e-4, solver='auto', positive=False, random_state=None)` — sklearn's
    /// signature one-for-one. `copy_X` keeps its sklearn spelling at the Python
    /// boundary and maps to the Rust `copy_x`.
    #[new]
    #[pyo3(signature = (
        alpha = 1.0,
        fit_intercept = true,
        copy_x = true,
        max_iter = None,
        tol = 1e-4,
        solver = "auto".to_string(),
        positive = false,
        random_state = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        alpha: f64,
        fit_intercept: bool,
        copy_x: bool,
        max_iter: Option<usize>,
        tol: f64,
        solver: String,
        positive: bool,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            inner: AnyRidge::Unfit {
                alpha,
                fit_intercept,
                copy_x,
                max_iter,
                tol,
                solver,
                positive,
                random_state,
            },
            n_iter: None,
            solver_used: None,
        }
    }

    /// `fit(X, y, rows, cols, sample_weight=None)`.
    ///
    /// `sample_weight` is an optional length-`rows` Arrow float array in the SAME
    /// dtype as `X` — it is borrowed as a host slice (never uploaded), because
    /// the weighted preprocessing that consumes it is a host pass anyway
    /// (`ridge.rs::preprocess`).
    #[pyo3(signature = (x, y, rows, cols, sample_weight = None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
        sample_weight: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let swa = sample_weight.map(capsule_to_array).transpose()?;
        let dt = float_dtype(&xa)?;
        let p = self.params();
        let (fitted, n_iter, solver_used) =
            py.detach(|| -> PyResult<(AnyRidge, Option<usize>, String)> {
                let mut pool = crate::lock_pool();
                match dt {
                    FloatDtype::F32 => {
                        let fitted = ridge_fit_dispatch!(
                            f32, p, xa, ya, swa, rows, cols, pool,
                            as_f32, host_slice_f32, validated_f32
                        );
                        let n_iter = fitted.n_iter();
                        let used = fitted.solver().name().to_string();
                        Ok((AnyRidge::F32(fitted), n_iter, used))
                    }
                    FloatDtype::F64 => {
                        crate::capability::guard_f64()?;
                        let fitted = ridge_fit_dispatch!(
                            f64, p, xa, ya, swa, rows, cols, pool,
                            as_f64, host_slice_f64, validated_f64
                        );
                        let n_iter = fitted.n_iter();
                        let used = fitted.solver().name().to_string();
                        Ok((AnyRidge::F64(fitted), n_iter, used))
                    }
                }
            })?;
        self.inner = fitted;
        self.n_iter = n_iter;
        self.solver_used = Some(solver_used);
        Ok(())
    }

    /// sklearn's `n_iter_` (`None` for the solvers sklearn leaves unset —
    /// `cholesky`, `svd`, `sparse_cg`, `lbfgs`).
    fn n_iter(&self) -> Option<usize> {
        self.n_iter
    }

    /// sklearn's `solver_` — the solver that actually ran.
    fn solver_used(&self) -> Option<String> {
        self.solver_used.clone()
    }

    /// `predict(x)` → a length-`rows` **pyarrow** float array (D-03). Shared
    /// body: [`dense_predict_f32`].
    fn predict_f32<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyRidge::F32(est) => dense_predict_f32(py, x, (rows, cols), est),
            _ => Err(not_fitted("ridge", "predict (f32 path)")),
        }
    }
    fn predict_f64<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyRidge::F64(est) => dense_predict_f64(py, x, (rows, cols), est),
            _ => Err(not_fitted("ridge", "predict (f64 path)")),
        }
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyRidge::F32(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("ridge", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyRidge::F64(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("ridge", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyRidge::F32(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("ridge", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyRidge::F64(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("ridge", "intercept_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyRidge::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyRidge::Unfit { .. } => None,
            AnyRidge::F32(_) => Some("f32"),
            AnyRidge::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// Lasso — Fit + Predict; alpha, fit_intercept, max_iter, tol
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyLasso,
    algo:  mlrs_algos::linear::lasso::Lasso,
    unfit: { alpha: f64, fit_intercept: bool, max_iter: usize, tol: f64 },
}

/// sklearn-compatible `Lasso` (L1-penalized least squares, coordinate descent).
#[pyclass(name = "Lasso")]
pub struct PyLasso {
    inner: AnyLasso,
}

impl PyLasso {
    /// Rust-callable default constructor for the smoke test. See
    /// [`PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self { inner: AnyLasso::Unfit { alpha: 1.0, fit_intercept: true, max_iter: 1000, tol: 1e-4 } }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyLasso::Unfit { .. })
    }
}

#[pymethods]
impl PyLasso {
    /// `Lasso(alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4)`.
    #[new]
    #[pyo3(signature = (alpha = 1.0, fit_intercept = true, max_iter = 1000, tol = 1e-4))]
    fn new(alpha: f64, fit_intercept: bool, max_iter: usize, tol: f64) -> Self {
        Self {
            inner: AnyLasso::Unfit { alpha, fit_intercept, max_iter, tol },
        }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&xa)?;
        let (alpha, fit_intercept, max_iter, tol) = match &self.inner {
            AnyLasso::Unfit { alpha, fit_intercept, max_iter, tol } => (*alpha, *fit_intercept, *max_iter, *tol),
            _ => (1.0, true, 1000, 1e-4),
        };
        let fitted = py.detach(|| -> PyResult<AnyLasso> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = Lasso::<f32>::builder()
                        .alpha(alpha)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyLasso::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = Lasso::<f64>::builder()
                        .alpha(alpha)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyLasso::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x)` → a length-`rows` **pyarrow** float array (D-03). Shared
    /// body: [`dense_predict_f32`].
    fn predict_f32<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyLasso::F32(est) => dense_predict_f32(py, x, (rows, cols), est),
            _ => Err(not_fitted("lasso", "predict (f32 path)")),
        }
    }
    fn predict_f64<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyLasso::F64(est) => dense_predict_f64(py, x, (rows, cols), est),
            _ => Err(not_fitted("lasso", "predict (f64 path)")),
        }
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLasso::F32(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("lasso", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLasso::F64(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("lasso", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLasso::F32(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("lasso", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLasso::F64(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("lasso", "intercept_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyLasso::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyLasso::Unfit { .. } => None,
            AnyLasso::F32(_) => Some("f32"),
            AnyLasso::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// ElasticNet — Fit + Predict; alpha, l1_ratio, fit_intercept, max_iter, tol
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyElasticNet,
    algo:  mlrs_algos::linear::elastic_net::ElasticNet,
    unfit: { alpha: f64, l1_ratio: f64, fit_intercept: bool, max_iter: usize, tol: f64 },
}

/// sklearn-compatible `ElasticNet` (combined L1/L2, coordinate descent).
#[pyclass(name = "ElasticNet")]
pub struct PyElasticNet {
    inner: AnyElasticNet,
}

impl PyElasticNet {
    /// Rust-callable default constructor for the smoke test. See
    /// [`PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyElasticNet::Unfit {
                alpha: 1.0,
                l1_ratio: 0.5,
                fit_intercept: true,
                max_iter: 1000,
                tol: 1e-4,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyElasticNet::Unfit { .. })
    }
}

#[pymethods]
impl PyElasticNet {
    /// `ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4)`.
    #[new]
    #[pyo3(signature = (alpha = 1.0, l1_ratio = 0.5, fit_intercept = true, max_iter = 1000, tol = 1e-4))]
    fn new(alpha: f64, l1_ratio: f64, fit_intercept: bool, max_iter: usize, tol: f64) -> Self {
        Self {
            inner: AnyElasticNet::Unfit { alpha, l1_ratio, fit_intercept, max_iter, tol },
        }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&xa)?;
        let (alpha, l1_ratio, fit_intercept, max_iter, tol) = match &self.inner {
            AnyElasticNet::Unfit { alpha, l1_ratio, fit_intercept, max_iter, tol } => {
                (*alpha, *l1_ratio, *fit_intercept, *max_iter, *tol)
            }
            _ => (1.0, 0.5, true, 1000, 1e-4),
        };
        let fitted = py.detach(|| -> PyResult<AnyElasticNet> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = ElasticNet::<f32>::builder()
                        .alpha(alpha)
                        .l1_ratio(l1_ratio)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyElasticNet::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = ElasticNet::<f64>::builder()
                        .alpha(alpha)
                        .l1_ratio(l1_ratio)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyElasticNet::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x)` → a length-`rows` **pyarrow** float array (D-03). Shared
    /// body: [`dense_predict_f32`].
    fn predict_f32<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyElasticNet::F32(est) => dense_predict_f32(py, x, (rows, cols), est),
            _ => Err(not_fitted("elastic_net", "predict (f32 path)")),
        }
    }
    fn predict_f64<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyElasticNet::F64(est) => dense_predict_f64(py, x, (rows, cols), est),
            _ => Err(not_fitted("elastic_net", "predict (f64 path)")),
        }
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyElasticNet::F32(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("elastic_net", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyElasticNet::F64(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("elastic_net", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyElasticNet::F32(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("elastic_net", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyElasticNet::F64(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("elastic_net", "intercept_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyElasticNet::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyElasticNet::Unfit { .. } => None,
            AnyElasticNet::F32(_) => Some("f32"),
            AnyElasticNet::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// LogisticRegression — Fit + PredictLabels (i32) + PredictProba; C, ...
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyLogisticRegression,
    algo:  mlrs_algos::linear::logistic::LogisticRegression,
    unfit: { c: f64, fit_intercept: bool, max_iter: usize, tol: f64 },
}

/// sklearn-compatible `LogisticRegression`. The sklearn-named inverse-regularization
/// strength `C` maps to the Rust `c` field (PY-02).
#[pyclass(name = "LogisticRegression")]
pub struct PyLogisticRegression {
    inner: AnyLogisticRegression,
}

impl PyLogisticRegression {
    /// Rust-callable default constructor for the smoke test. See
    /// [`PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyLogisticRegression::Unfit { c: 1.0, fit_intercept: true, max_iter: 100, tol: 1e-4 },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyLogisticRegression::Unfit { .. })
    }
}

#[pymethods]
impl PyLogisticRegression {
    /// `LogisticRegression(C=1.0, fit_intercept=True, max_iter=100, tol=1e-4)`.
    /// The sklearn `C` is the constructor's first positional/keyword param.
    #[new]
    #[pyo3(signature = (C = 1.0, fit_intercept = true, max_iter = 100, tol = 1e-4))]
    #[allow(non_snake_case)]
    fn new(C: f64, fit_intercept: bool, max_iter: usize, tol: f64) -> Self {
        Self {
            inner: AnyLogisticRegression::Unfit { c: C, fit_intercept, max_iter, tol },
        }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&xa)?;
        let (c, fit_intercept, max_iter, tol) = match &self.inner {
            AnyLogisticRegression::Unfit { c, fit_intercept, max_iter, tol } => (*c, *fit_intercept, *max_iter, *tol),
            _ => (1.0, true, 100, 1e-4),
        };
        let fitted = py.detach(|| -> PyResult<AnyLogisticRegression> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = LogisticRegression::<f32>::builder()
                        .c(c)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyLogisticRegression::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = LogisticRegression::<f64>::builder()
                        .c(c)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyLogisticRegression::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x)` → length-`rows` host `Vec<i32>` class labels (D-06).
    fn predict_labels(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<i32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyLogisticRegression::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestatePredictLabels::predict_labels(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                AnyLogisticRegression::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestatePredictLabels::predict_labels(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("logistic_regression", "predict")),
            }
        })
    }

    /// `predict_proba(x)` → row-major `rows × n_classes` host floats.
    fn predict_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyLogisticRegression::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestatePredictProba::predict_proba(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("logistic_regression", "predict_proba (f32 path)")),
            }
        })
    }
    fn predict_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyLogisticRegression::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestatePredictProba::predict_proba(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("logistic_regression", "predict_proba (f64 path)")),
            }
        })
    }

    /// Number of classes inferred at fit (0 before fit).
    fn n_classes(&self) -> usize {
        match &self.inner {
            AnyLogisticRegression::F32(e) => e.n_classes(),
            AnyLogisticRegression::F64(e) => e.n_classes(),
            _ => 0,
        }
    }

    /// The DISTINCT sorted training labels (`classes_`). The shim MUST use these
    /// rather than a fabricated `0..n_classes` range so a non-contiguous target
    /// (e.g. `{0, 2}`) round-trips through `predict` (WR-01).
    fn classes_(&self) -> PyResult<Vec<i64>> {
        match &self.inner {
            AnyLogisticRegression::F32(e) => Ok(e.classes().to_vec()),
            AnyLogisticRegression::F64(e) => Ok(e.classes().to_vec()),
            _ => Err(not_fitted("logistic_regression", "classes_")),
        }
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLogisticRegression::F32(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("logistic_regression", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLogisticRegression::F64(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("logistic_regression", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLogisticRegression::F32(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("logistic_regression", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLogisticRegression::F64(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("logistic_regression", "intercept_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyLogisticRegression::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyLogisticRegression::Unfit { .. } => None,
            AnyLogisticRegression::F32(_) => Some("f32"),
            AnyLogisticRegression::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// Phase-10 SGD / linear-SVM dtype-dispatch enums (SGDSVM-01..04, Wave-0 stubs).
//
// The 10-01 Wave-0 scaffold lands ONLY the `any_estimator!` Unfit{} stub blocks
// (the dtype-dispatch enum the macro emits — the macro needs NO extension,
// RESEARCH §Builder-API). Each `Unfit` arm stores the sklearn-named STRINGS +
// scalars verbatim (loss/penalty/learning_rate strings, alpha/eta0/epsilon
// scalars), exactly as `kernel.rs` stores `kernel: String`. The hand-written
// `#[pymethods]` fit bodies — `Loss::try_from(s).map_err(build_err_to_py)?` →
// `Estimator::<F>::builder()...build().map_err(build_err_to_py)?` →
// `est.fit(...).map_err(algo_err_to_py)?` — and the `#[pyclass]` registration on
// the `_mlrs` module are owned by the Wave-3 plan (so this scaffold compiles
// WITHOUT the estimator bodies). The `unfit_default_*` helpers below are the
// Wave-3 promotion seam (they exercise the `Unfit` arm exactly like
// `PyLinearRegression::unfit_default`); `#[allow(dead_code)]` until Wave 3 wires
// the pyclasses that consume the F32/F64 arms.
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyMBSGDClassifier,
    algo:  mlrs_algos::linear::mbsgd_classifier::MBSGDClassifier,
    unfit: {
        loss: String, penalty: String, alpha: f64, l1_ratio: f64,
        fit_intercept: bool, max_iter: usize, tol: f64,
        learning_rate: String, eta0: f64, power_t: f64,
        batch_size: usize, shuffle: bool, seed: u64,
    },
}

crate::any_estimator_typestate! {
    any:   AnyMBSGDRegressor,
    algo:  mlrs_algos::linear::mbsgd_regressor::MBSGDRegressor,
    unfit: {
        loss: String, penalty: String, alpha: f64, l1_ratio: f64,
        fit_intercept: bool, max_iter: usize, tol: f64,
        learning_rate: String, eta0: f64, power_t: f64, epsilon: f64,
        batch_size: usize, shuffle: bool, seed: u64,
    },
}

crate::any_estimator_typestate! {
    any:   AnyLinearSVC,
    algo:  mlrs_algos::linear::linear_svc::LinearSVC,
    unfit: {
        loss: String, penalty: String, c: f64, intercept_scaling: f64,
        fit_intercept: bool, max_iter: usize, tol: f64,
    },
}

crate::any_estimator_typestate! {
    any:   AnyLinearSVR,
    algo:  mlrs_algos::linear::linear_svr::LinearSVR,
    unfit: {
        loss: String, penalty: String, c: f64, epsilon: f64,
        intercept_scaling: f64, fit_intercept: bool, max_iter: usize, tol: f64,
    },
}

// ===========================================================================
// MBSGDClassifier — Fit (TryFrom enums + builder().build()) + PredictLabels (i32)
// + PredictProba (log-loss sigmoid); sklearn-named string knobs (SGDSVM-01).
// ===========================================================================

/// sklearn-compatible `MBSGDClassifier` (minibatch SGD classifier). The
/// sklearn-named `loss`/`penalty`/`learning_rate` STRINGS are stored verbatim in
/// the `Unfit` arm; the typed `Loss`/`Penalty`/`LearningRate` enums + the builder
/// `build()` run at the first `fit` (an unknown string / bad data-independent
/// param surfaces as a `ValueError` there, D-05/D-09).
#[pyclass(name = "MBSGDClassifier")]
pub struct PyMBSGDClassifier {
    inner: AnyMBSGDClassifier,
}

impl PyMBSGDClassifier {
    /// Rust-callable default constructor (smoke test seam — see
    /// [`PyLinearRegression::unfit_default`]).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyMBSGDClassifier::Unfit {
                loss: "hinge".to_string(),
                penalty: "l2".to_string(),
                alpha: 1e-4,
                l1_ratio: 0.15,
                fit_intercept: true,
                max_iter: 1000,
                tol: 1e-3,
                learning_rate: "optimal".to_string(),
                eta0: 0.01,
                power_t: 0.5,
                batch_size: 1,
                shuffle: true,
                seed: 0,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyMBSGDClassifier::Unfit { .. })
    }
}

#[pymethods]
impl PyMBSGDClassifier {
    /// `MBSGDClassifier(loss="hinge", penalty="l2", alpha=1e-4, l1_ratio=0.15,
    /// fit_intercept=True, max_iter=1000, tol=1e-3, learning_rate="optimal",
    /// eta0=0.01, power_t=0.5, batch_size=1, shuffle=True, seed=0)`.
    #[new]
    #[pyo3(signature = (
        loss = "hinge".to_string(), penalty = "l2".to_string(), alpha = 1e-4,
        l1_ratio = 0.15, fit_intercept = true, max_iter = 1000, tol = 1e-3,
        learning_rate = "optimal".to_string(), eta0 = 0.01, power_t = 0.5,
        batch_size = 1, shuffle = true, seed = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        loss: String,
        penalty: String,
        alpha: f64,
        l1_ratio: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        learning_rate: String,
        eta0: f64,
        power_t: f64,
        batch_size: usize,
        shuffle: bool,
        seed: u64,
    ) -> Self {
        Self {
            inner: AnyMBSGDClassifier::Unfit {
                loss,
                penalty,
                alpha,
                l1_ratio,
                fit_intercept,
                max_iter,
                tol,
                learning_rate,
                eta0,
                power_t,
                batch_size,
                shuffle,
                seed,
            },
        }
    }

    /// Fit on `x` (`rows × cols`, row-major) + label vector `y`. The sklearn enum
    /// strings are parsed (`TryFrom` → `ValueError` on a bad string, D-05) and the
    /// builder validates the data-independent params (`build()` → `ValueError`,
    /// D-09) BEFORE the device launch; GIL released (PY-03); f64 guarded (D-04).
    ///
    /// **No upload on cpu** (MBSGD-PERF-CPU). Where the SGD solve runs on the
    /// host arm anyway ([`sgd_host_available`]), the design matrix is BORROWED
    /// from the Arrow values via [`host_slice_f32`] / [`host_slice_f64`] — the
    /// same hard-reject bridge validator `validated_f32` runs, minus the copy —
    /// and handed to `fit_from_host_slice`. Routing it through a `DeviceArray`
    /// instead costs three full passes over `x` (one to upload, two more
    /// because `to_host` materializes a byte buffer and then a typed one); at
    /// `50 000 × 64` f32 that was 10 ms against 6 ms of actual solving. Real
    /// device backends keep the upload and the device epoch loop.
    ///
    /// [`sgd_host_available`]: mlrs_backend::prims::sgd::sgd_host_available
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&xa)?;
        let (
            loss_s, penalty_s, alpha, l1_ratio, fit_intercept, max_iter, tol,
            lr_s, eta0, power_t, batch_size, shuffle, seed,
        ) = match &self.inner {
            AnyMBSGDClassifier::Unfit {
                loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol,
                learning_rate, eta0, power_t, batch_size, shuffle, seed,
            } => (
                loss.clone(), penalty.clone(), *alpha, *l1_ratio, *fit_intercept,
                *max_iter, *tol, learning_rate.clone(), *eta0, *power_t,
                *batch_size, *shuffle, *seed,
            ),
            _ => return Err(not_fitted("mbsgd_classifier", "re-fit")),
        };
        // Construction-time enum-string validation (D-05 → ValueError).
        let loss = Loss::try_from(loss_s.as_str()).map_err(build_err_to_py)?;
        let penalty = Penalty::try_from(penalty_s.as_str()).map_err(build_err_to_py)?;
        let lr = LearningRate::try_from(lr_s.as_str()).map_err(build_err_to_py)?;
        let host_ingress = mlrs_backend::prims::sgd::sgd_host_available();
        let fitted = py.detach(|| -> PyResult<AnyMBSGDClassifier> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let est = MBSGDClassifier::<f32>::builder()
                        .loss(loss)
                        .penalty(penalty)
                        .alpha(alpha)
                        .l1_ratio(l1_ratio)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .learning_rate(lr)
                        .eta0(eta0)
                        .power_t(power_t)
                        .batch_size(batch_size)
                        .shuffle(shuffle)
                        .seed(seed)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = if host_ingress {
                        est.fit_from_host_slice(
                            &mut pool,
                            host_slice_f32(as_f32(&xa)?)?,
                            host_slice_f32(as_f32(&ya)?)?,
                            (rows, cols),
                        )
                    } else {
                        let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                        let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                        TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                    }
                    .map_err(algo_err_to_py)?;
                    Ok(AnyMBSGDClassifier::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let est = MBSGDClassifier::<f64>::builder()
                        .loss(loss)
                        .penalty(penalty)
                        .alpha(alpha)
                        .l1_ratio(l1_ratio)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .learning_rate(lr)
                        .eta0(eta0)
                        .power_t(power_t)
                        .batch_size(batch_size)
                        .shuffle(shuffle)
                        .seed(seed)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = if host_ingress {
                        est.fit_from_host_slice(
                            &mut pool,
                            host_slice_f64(as_f64(&xa)?)?,
                            host_slice_f64(as_f64(&ya)?)?,
                            (rows, cols),
                        )
                    } else {
                        let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                        let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                        TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                    }
                    .map_err(algo_err_to_py)?;
                    Ok(AnyMBSGDClassifier::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x)` → length-`rows` host `Vec<i32>` class labels (margin sign).
    fn predict_labels(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<i32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMBSGDClassifier::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestatePredictLabels::predict_labels(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                AnyMBSGDClassifier::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestatePredictLabels::predict_labels(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("mbsgd_classifier", "predict")),
            }
        })
    }

    /// `predict_proba(x)` → row-major `rows × 2` host floats (log-loss sigmoid;
    /// sklearn raises for a non-log loss — the caller pins the log-loss path).
    fn predict_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMBSGDClassifier::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestatePredictProba::predict_proba(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("mbsgd_classifier", "predict_proba (f32 path)")),
            }
        })
    }
    fn predict_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMBSGDClassifier::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestatePredictProba::predict_proba(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("mbsgd_classifier", "predict_proba (f64 path)")),
            }
        })
    }

    /// The inferred class labels (`classes_`, length 2 for the binary fit).
    fn classes_(&self) -> Vec<i64> {
        match &self.inner {
            AnyMBSGDClassifier::F32(e) => e.classes().to_vec(),
            AnyMBSGDClassifier::F64(e) => e.classes().to_vec(),
            _ => Vec::new(),
        }
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyMBSGDClassifier::F32(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("mbsgd_classifier", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyMBSGDClassifier::F64(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("mbsgd_classifier", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyMBSGDClassifier::F32(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("mbsgd_classifier", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyMBSGDClassifier::F64(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("mbsgd_classifier", "intercept_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyMBSGDClassifier::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyMBSGDClassifier::Unfit { .. } => None,
            AnyMBSGDClassifier::F32(_) => Some("f32"),
            AnyMBSGDClassifier::F64(_) => Some("f64"),
        }
    }
}

// ===========================================================================
// MBSGDRegressor — Fit (TryFrom enums + builder().build()) + Predict (SGDSVM-02).
// ===========================================================================

/// sklearn-compatible `MBSGDRegressor` (minibatch SGD regressor).
#[pyclass(name = "MBSGDRegressor")]
pub struct PyMBSGDRegressor {
    inner: AnyMBSGDRegressor,
}

impl PyMBSGDRegressor {
    /// Rust-callable default constructor (smoke test seam).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyMBSGDRegressor::Unfit {
                loss: "squared_error".to_string(),
                penalty: "l2".to_string(),
                alpha: 1e-4,
                l1_ratio: 0.15,
                fit_intercept: true,
                max_iter: 1000,
                tol: 1e-3,
                learning_rate: "invscaling".to_string(),
                eta0: 0.01,
                power_t: 0.25,
                epsilon: 0.1,
                batch_size: 1,
                shuffle: true,
                seed: 0,
            },
        }
    }

    /// Is this wrapper in the unfit arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyMBSGDRegressor::Unfit { .. })
    }
}

#[pymethods]
impl PyMBSGDRegressor {
    /// `MBSGDRegressor(loss="squared_error", penalty="l2", alpha=1e-4,
    /// l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3,
    /// learning_rate="invscaling", eta0=0.01, power_t=0.25, epsilon=0.1,
    /// batch_size=1, shuffle=True, seed=0)`.
    #[new]
    #[pyo3(signature = (
        loss = "squared_error".to_string(), penalty = "l2".to_string(), alpha = 1e-4,
        l1_ratio = 0.15, fit_intercept = true, max_iter = 1000, tol = 1e-3,
        learning_rate = "invscaling".to_string(), eta0 = 0.01, power_t = 0.25,
        epsilon = 0.1, batch_size = 1, shuffle = true, seed = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        loss: String,
        penalty: String,
        alpha: f64,
        l1_ratio: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        learning_rate: String,
        eta0: f64,
        power_t: f64,
        epsilon: f64,
        batch_size: usize,
        shuffle: bool,
        seed: u64,
    ) -> Self {
        Self {
            inner: AnyMBSGDRegressor::Unfit {
                loss,
                penalty,
                alpha,
                l1_ratio,
                fit_intercept,
                max_iter,
                tol,
                learning_rate,
                eta0,
                power_t,
                epsilon,
                batch_size,
                shuffle,
                seed,
            },
        }
    }

    /// Fit on `x` (`rows × cols`) + target `y`. Enum strings + builder validation
    /// → `ValueError` (D-05/D-09) before the device launch; GIL released; f64 guarded.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&xa)?;
        let (
            loss_s, penalty_s, alpha, l1_ratio, fit_intercept, max_iter, tol,
            lr_s, eta0, power_t, epsilon, batch_size, shuffle, seed,
        ) = match &self.inner {
            AnyMBSGDRegressor::Unfit {
                loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol,
                learning_rate, eta0, power_t, epsilon, batch_size, shuffle, seed,
            } => (
                loss.clone(), penalty.clone(), *alpha, *l1_ratio, *fit_intercept,
                *max_iter, *tol, learning_rate.clone(), *eta0, *power_t, *epsilon,
                *batch_size, *shuffle, *seed,
            ),
            _ => return Err(not_fitted("mbsgd_regressor", "re-fit")),
        };
        let loss = Loss::try_from(loss_s.as_str()).map_err(build_err_to_py)?;
        let penalty = Penalty::try_from(penalty_s.as_str()).map_err(build_err_to_py)?;
        let lr = LearningRate::try_from(lr_s.as_str()).map_err(build_err_to_py)?;
        let fitted = py.detach(|| -> PyResult<AnyMBSGDRegressor> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = MBSGDRegressor::<f32>::builder()
                        .loss(loss)
                        .penalty(penalty)
                        .alpha(alpha)
                        .l1_ratio(l1_ratio)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .learning_rate(lr)
                        .eta0(eta0)
                        .power_t(power_t)
                        .epsilon(epsilon)
                        .batch_size(batch_size)
                        .shuffle(shuffle)
                        .seed(seed)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyMBSGDRegressor::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = MBSGDRegressor::<f64>::builder()
                        .loss(loss)
                        .penalty(penalty)
                        .alpha(alpha)
                        .l1_ratio(l1_ratio)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .learning_rate(lr)
                        .eta0(eta0)
                        .power_t(power_t)
                        .epsilon(epsilon)
                        .batch_size(batch_size)
                        .shuffle(shuffle)
                        .seed(seed)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyMBSGDRegressor::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn predict_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMBSGDRegressor::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestatePredict::predict(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("mbsgd_regressor", "predict (f32 path)")),
            }
        })
    }
    fn predict_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMBSGDRegressor::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestatePredict::predict(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("mbsgd_regressor", "predict (f64 path)")),
            }
        })
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyMBSGDRegressor::F32(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("mbsgd_regressor", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyMBSGDRegressor::F64(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("mbsgd_regressor", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyMBSGDRegressor::F32(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("mbsgd_regressor", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyMBSGDRegressor::F64(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("mbsgd_regressor", "intercept_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyMBSGDRegressor::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyMBSGDRegressor::Unfit { .. } => None,
            AnyMBSGDRegressor::F32(_) => Some("f32"),
            AnyMBSGDRegressor::F64(_) => Some("f64"),
        }
    }
}

// ===========================================================================
// LinearSVC — Fit (TryFrom enums + builder().build()) + PredictLabels (i32);
// no learning_rate string (L-BFGS solver, SGDSVM-03).
// ===========================================================================

/// sklearn-compatible `LinearSVC` (L2-regularized squared-hinge primal).
#[pyclass(name = "LinearSVC")]
pub struct PyLinearSVC {
    inner: AnyLinearSVC,
}

impl PyLinearSVC {
    /// Rust-callable default constructor (smoke test seam).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyLinearSVC::Unfit {
                loss: "squared_hinge".to_string(),
                penalty: "l2".to_string(),
                c: 1.0,
                intercept_scaling: 1.0,
                fit_intercept: true,
                max_iter: 1000,
                tol: 1e-4,
            },
        }
    }

    /// Is this wrapper in the unfit arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyLinearSVC::Unfit { .. })
    }
}

#[pymethods]
impl PyLinearSVC {
    /// `LinearSVC(loss="squared_hinge", penalty="l2", C=1.0, intercept_scaling=1.0,
    /// fit_intercept=True, max_iter=1000, tol=1e-4)`. The sklearn-named inverse-
    /// regularization strength `C` maps to the Rust `c` field.
    #[new]
    #[pyo3(signature = (
        loss = "squared_hinge".to_string(), penalty = "l2".to_string(), C = 1.0,
        intercept_scaling = 1.0, fit_intercept = true, max_iter = 1000, tol = 1e-4,
    ))]
    #[allow(non_snake_case)]
    fn new(
        loss: String,
        penalty: String,
        C: f64,
        intercept_scaling: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            inner: AnyLinearSVC::Unfit {
                loss,
                penalty,
                c: C,
                intercept_scaling,
                fit_intercept,
                max_iter,
                tol,
            },
        }
    }

    /// Fit on `x` (`rows × cols`) + label vector `y`. Enum strings + builder
    /// validation (`C>0`) → `ValueError` (D-05/D-09); GIL released; f64 guarded.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&xa)?;
        let (loss_s, penalty_s, c, intercept_scaling, fit_intercept, max_iter, tol) = match &self.inner {
            AnyLinearSVC::Unfit {
                loss, penalty, c, intercept_scaling, fit_intercept, max_iter, tol,
            } => (
                loss.clone(), penalty.clone(), *c, *intercept_scaling,
                *fit_intercept, *max_iter, *tol,
            ),
            _ => return Err(not_fitted("linear_svc", "re-fit")),
        };
        let loss = Loss::try_from(loss_s.as_str()).map_err(build_err_to_py)?;
        let penalty = Penalty::try_from(penalty_s.as_str()).map_err(build_err_to_py)?;
        let fitted = py.detach(|| -> PyResult<AnyLinearSVC> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = LinearSVC::<f32>::builder()
                        .loss(loss)
                        .penalty(penalty)
                        .c(c)
                        .intercept_scaling(intercept_scaling)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyLinearSVC::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = LinearSVC::<f64>::builder()
                        .loss(loss)
                        .penalty(penalty)
                        .c(c)
                        .intercept_scaling(intercept_scaling)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyLinearSVC::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x)` → a length-`rows` **pyarrow** `int32` array of class ids
    /// (decision-function sign through `classes_`).
    ///
    /// The classifier twin of [`dense_predict_f32`], and it departs from this
    /// file's `predict_labels` template for exactly the two reasons documented
    /// there — plus a third that is specific to the label path:
    ///
    /// 1. **No upload.** `predict_labels_from_host` borrows the validated Arrow
    ///    values and routes cpu to the host matvec; wgpu/cuda/rocm still upload
    ///    and run the fused device kernel.
    /// 2. **No Python list.** The ids go back over Arrow, which numpy views in
    ///    place, instead of one boxed `int` per row.
    /// 3. **No round-trip for the labels themselves.** The estimator's
    ///    device-ingress `predict_labels` derives the ids on the host and then
    ///    uploads them into an `i32` `DeviceArray` to satisfy its trait
    ///    signature — which this binding would immediately read straight back.
    ///    The host-ingress path skips both crossings.
    ///
    /// Like the dense regressors it OWNS the NaN/inf rejection (`mlrs.linear`
    /// passes `ensure_all_finite=False`), reproducing `check_array`'s message
    /// via [`nonfinite_input_err`].
    fn predict_labels<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let xa = capsule_to_array(x)?;
        let out = py.detach(|| -> PyResult<Vec<i32>> {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyLinearSVC::F32(est) => {
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    let pred = est
                        .predict_labels_from_host(&mut pool, xh, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    if !pred.operand_finite {
                        return Err(nonfinite_input_err(xh, "float32"));
                    }
                    Ok(pred.values)
                }
                AnyLinearSVC::F64(est) => {
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    let pred = est
                        .predict_labels_from_host(&mut pool, xh, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    if !pred.operand_finite {
                        return Err(nonfinite_input_err(xh, "float64"));
                    }
                    Ok(pred.values)
                }
                _ => Err(not_fitted("linear_svc", "predict")),
            }
        })?;
        i32_vec_to_pyarrow(py, out)
    }

    /// The inferred class labels (`classes_`, length 2 for the binary fit).
    fn classes_(&self) -> Vec<i64> {
        match &self.inner {
            AnyLinearSVC::F32(e) => e.classes().to_vec(),
            AnyLinearSVC::F64(e) => e.classes().to_vec(),
            _ => Vec::new(),
        }
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearSVC::F32(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("linear_svc", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearSVC::F64(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("linear_svc", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearSVC::F32(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("linear_svc", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearSVC::F64(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("linear_svc", "intercept_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyLinearSVC::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyLinearSVC::Unfit { .. } => None,
            AnyLinearSVC::F32(_) => Some("f32"),
            AnyLinearSVC::F64(_) => Some("f64"),
        }
    }
}

// ===========================================================================
// LinearSVR — Fit (TryFrom enums + builder().build()) + Predict; no learning_rate
// string (L-BFGS solver, SGDSVM-04).
// ===========================================================================

/// sklearn-compatible `LinearSVR` (L2-regularized squared-eps-insensitive primal).
#[pyclass(name = "LinearSVR")]
pub struct PyLinearSVR {
    inner: AnyLinearSVR,
}

impl PyLinearSVR {
    /// Rust-callable default constructor (smoke test seam).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyLinearSVR::Unfit {
                loss: "squared_epsilon_insensitive".to_string(),
                penalty: "l2".to_string(),
                c: 1.0,
                epsilon: 0.0,
                intercept_scaling: 1.0,
                fit_intercept: true,
                max_iter: 1000,
                tol: 1e-4,
            },
        }
    }

    /// Is this wrapper in the unfit arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyLinearSVR::Unfit { .. })
    }
}

#[pymethods]
impl PyLinearSVR {
    /// `LinearSVR(loss="squared_epsilon_insensitive", penalty="l2", C=1.0,
    /// epsilon=0.0, intercept_scaling=1.0, fit_intercept=True, max_iter=1000,
    /// tol=1e-4)`. The sklearn-named `C` maps to the Rust `c` field.
    #[new]
    #[pyo3(signature = (
        loss = "squared_epsilon_insensitive".to_string(), penalty = "l2".to_string(),
        C = 1.0, epsilon = 0.0, intercept_scaling = 1.0, fit_intercept = true,
        max_iter = 1000, tol = 1e-4,
    ))]
    #[allow(non_snake_case)]
    fn new(
        loss: String,
        penalty: String,
        C: f64,
        epsilon: f64,
        intercept_scaling: f64,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            inner: AnyLinearSVR::Unfit {
                loss,
                penalty,
                c: C,
                epsilon,
                intercept_scaling,
                fit_intercept,
                max_iter,
                tol,
            },
        }
    }

    /// Fit on `x` (`rows × cols`) + target `y`. Enum strings + builder validation
    /// (`C>0`, `epsilon>=0`) → `ValueError` (D-05/D-09); GIL released; f64 guarded.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&xa)?;
        let (loss_s, penalty_s, c, epsilon, intercept_scaling, fit_intercept, max_iter, tol) = match &self.inner {
            AnyLinearSVR::Unfit {
                loss, penalty, c, epsilon, intercept_scaling, fit_intercept, max_iter, tol,
            } => (
                loss.clone(), penalty.clone(), *c, *epsilon, *intercept_scaling,
                *fit_intercept, *max_iter, *tol,
            ),
            _ => return Err(not_fitted("linear_svr", "re-fit")),
        };
        let loss = Loss::try_from(loss_s.as_str()).map_err(build_err_to_py)?;
        let penalty = Penalty::try_from(penalty_s.as_str()).map_err(build_err_to_py)?;
        let fitted = py.detach(|| -> PyResult<AnyLinearSVR> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = LinearSVR::<f32>::builder()
                        .loss(loss)
                        .penalty(penalty)
                        .c(c)
                        .epsilon(epsilon)
                        .intercept_scaling(intercept_scaling)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyLinearSVR::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = LinearSVR::<f64>::builder()
                        .loss(loss)
                        .penalty(penalty)
                        .c(c)
                        .epsilon(epsilon)
                        .intercept_scaling(intercept_scaling)
                        .fit_intercept(fit_intercept)
                        .max_iter(max_iter)
                        .tol(tol)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyLinearSVR::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x)` → a length-`rows` **pyarrow** float array (D-03). Shares
    /// [`dense_predict_f32`] with the four dense linear regressors — see its
    /// docs for the no-upload / no-Python-list ingress and egress.
    fn predict_f32<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyLinearSVR::F32(est) => dense_predict_f32(py, x, (rows, cols), est),
            _ => Err(not_fitted("linear_svr", "predict (f32 path)")),
        }
    }
    fn predict_f64<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyLinearSVR::F64(est) => dense_predict_f64(py, x, (rows, cols), est),
            _ => Err(not_fitted("linear_svr", "predict (f64 path)")),
        }
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearSVR::F32(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("linear_svr", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearSVR::F64(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("linear_svr", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearSVR::F32(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("linear_svr", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearSVR::F64(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("linear_svr", "intercept_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyLinearSVR::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyLinearSVR::Unfit { .. } => None,
            AnyLinearSVR::F32(_) => Some("f32"),
            AnyLinearSVR::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// BayesianRidge — Fit + Predict; the FULL sklearn parameter surface
//   max_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, alpha_init,
//   lambda_init, compute_score, fit_intercept, copy_X, verbose
//   + fit(..., sample_weight) and the alpha_ / lambda_ / sigma_ / scores_ /
//     n_iter_ / X_offset_ / X_scale_ fitted attributes
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyBayesianRidge,
    algo:  mlrs_algos::linear::bayesian_ridge::BayesianRidge,
    unfit: {
        max_iter: usize,
        tol: f64,
        alpha_1: f64,
        alpha_2: f64,
        lambda_1: f64,
        lambda_2: f64,
        alpha_init: Option<f64>,
        lambda_init: Option<f64>,
        compute_score: bool,
        fit_intercept: bool,
        copy_x: bool,
        verbose: bool,
    },
}

/// The verbatim ctor hyperparameters, carried from the `Unfit` arm to `fit`
/// (WR-02: the wrapper rebuilds from these at every `fit`, so a second `fit` of
/// the same object works).
#[derive(Clone, Copy)]
struct BayesianRidgeParams {
    max_iter: usize,
    tol: f64,
    alpha_1: f64,
    alpha_2: f64,
    lambda_1: f64,
    lambda_2: f64,
    alpha_init: Option<f64>,
    lambda_init: Option<f64>,
    compute_score: bool,
    fit_intercept: bool,
    copy_x: bool,
    verbose: bool,
}

/// sklearn-compatible `BayesianRidge` (evidence-maximized ridge regression).
#[pyclass(name = "BayesianRidge")]
pub struct PyBayesianRidge {
    inner: AnyBayesianRidge,
    /// The scalar/vector fitted attributes, mirrored out of the consumed
    /// `Fitted` arms at `fit`: a `#[pyclass]` getter cannot reach through the
    /// dtype dispatch generically, and all of these are `f64` on both arms
    /// anyway (the evidence iteration accumulates in `f64` whatever the design's
    /// width — see `bayesian_ridge.rs`).
    alpha: Option<f64>,
    lambda: Option<f64>,
    sigma: Option<Vec<f64>>,
    scores: Vec<f64>,
    n_iter: Option<usize>,
    x_offset: Vec<f64>,
    x_scale: Vec<f64>,
}

impl PyBayesianRidge {
    /// Rust-callable default constructor for the smoke test. See
    /// [`PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyBayesianRidge::Unfit {
                max_iter: 300,
                tol: 1e-3,
                alpha_1: 1e-6,
                alpha_2: 1e-6,
                lambda_1: 1e-6,
                lambda_2: 1e-6,
                alpha_init: None,
                lambda_init: None,
                compute_score: false,
                fit_intercept: true,
                copy_x: true,
                verbose: false,
            },
            alpha: None,
            lambda: None,
            sigma: None,
            scores: Vec::new(),
            n_iter: None,
            x_offset: Vec::new(),
            x_scale: Vec::new(),
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyBayesianRidge::Unfit { .. })
    }

    /// Read back the ctor hyperparameters.
    fn params(&self) -> BayesianRidgeParams {
        match &self.inner {
            AnyBayesianRidge::Unfit {
                max_iter,
                tol,
                alpha_1,
                alpha_2,
                lambda_1,
                lambda_2,
                alpha_init,
                lambda_init,
                compute_score,
                fit_intercept,
                copy_x,
                verbose,
            } => BayesianRidgeParams {
                max_iter: *max_iter,
                tol: *tol,
                alpha_1: *alpha_1,
                alpha_2: *alpha_2,
                lambda_1: *lambda_1,
                lambda_2: *lambda_2,
                alpha_init: *alpha_init,
                lambda_init: *lambda_init,
                compute_score: *compute_score,
                fit_intercept: *fit_intercept,
                copy_x: *copy_x,
                verbose: *verbose,
            },
            // Already fitted: the shim always constructs a fresh wrapper per
            // `fit`, so this arm is unreachable in practice; fall back to
            // sklearn's defaults rather than panicking.
            _ => BayesianRidgeParams {
                max_iter: 300,
                tol: 1e-3,
                alpha_1: 1e-6,
                alpha_2: 1e-6,
                lambda_1: 1e-6,
                lambda_2: 1e-6,
                alpha_init: None,
                lambda_init: None,
                compute_score: false,
                fit_intercept: true,
                copy_x: true,
                verbose: false,
            },
        }
    }
}

/// Build an unfit `BayesianRidge<F>` from the ctor params. Monomorphized per
/// float width by the macro below so the twelve builder setters are written
/// once.
macro_rules! bayes_build {
    ($float:ty, $p:expr) => {{
        BayesianRidge::<$float>::builder()
            .max_iter($p.max_iter)
            .tol($p.tol)
            .alpha_1($p.alpha_1)
            .alpha_2($p.alpha_2)
            .lambda_1($p.lambda_1)
            .lambda_2($p.lambda_2)
            .alpha_init($p.alpha_init)
            .lambda_init($p.lambda_init)
            .compute_score($p.compute_score)
            .fit_intercept($p.fit_intercept)
            .copy_x($p.copy_x)
            .verbose($p.verbose)
            .build::<$float>()
            .map_err(build_err_to_py)?
    }};
}

/// Build the estimator and run whichever `fit` ingress its shape and backend
/// call for — the `ridge_fit_dispatch!` shape, for the same reason.
///
/// The branch has to happen HERE, before ingress, because the two entry points
/// take different operand types: `fit_from_host_slice` borrows the Arrow values
/// directly (`host_slice_*`) and `fit_with_sample_weight` needs a device upload
/// (`validated_*`). On the host arm — the cpu backend, or below the
/// dispatch-cost floor on any backend — the `n·d` design is therefore never
/// copied at all. Both helpers run the SAME hard-reject bridge validator, so the
/// ingress contract is identical either way.
macro_rules! bayes_fit_dispatch {
    ($float:ty, $p:expr, $xa:expr, $ya:expr, $swa:expr, $rows:expr, $cols:expr,
     $pool:expr, $as:ident, $host_slice:ident, $validated:ident) => {{
        let est = bayes_build!($float, $p);
        let sw = match $swa.as_ref() {
            Some(a) => Some($host_slice($as(a)?)?),
            None => None,
        };
        if est.host_fit_applicable(($rows, $cols)) {
            let xh = $host_slice($as(&$xa)?)?;
            let yh = $host_slice($as(&$ya)?)?;
            est.fit_from_host_slice(&mut $pool, xh, yh, ($rows, $cols), sw)
                .map_err(algo_err_to_py)?
        } else {
            let xd = $validated($as(&$xa)?, &mut $pool)?;
            let yd = $validated($as(&$ya)?, &mut $pool)?;
            est.fit_with_sample_weight(&mut $pool, &xd, Some(&yd), ($rows, $cols), sw)
                .map_err(algo_err_to_py)?
        }
    }};
}

/// Snapshot the fitted attributes a `#[pyclass]` getter cannot reach through the
/// dtype dispatch. Written once, invoked on both float arms.
macro_rules! bayes_snapshot {
    ($fitted:expr) => {{
        (
            $fitted.alpha(),
            $fitted.lambda(),
            $fitted.sigma().to_vec(),
            $fitted.scores().to_vec(),
            $fitted.n_iter(),
            $fitted.x_offset().to_vec(),
            $fitted.x_scale().to_vec(),
        )
    }};
}

#[pymethods]
impl PyBayesianRidge {
    /// `BayesianRidge(max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6,
    /// lambda_1=1e-6, lambda_2=1e-6, alpha_init=None, lambda_init=None,
    /// compute_score=False, fit_intercept=True, copy_X=True, verbose=False)` —
    /// sklearn's signature one-for-one. `copy_X` keeps its sklearn spelling at
    /// the Python boundary and maps to the Rust `copy_x`.
    #[new]
    #[pyo3(signature = (
        max_iter = 300,
        tol = 1e-3,
        alpha_1 = 1e-6,
        alpha_2 = 1e-6,
        lambda_1 = 1e-6,
        lambda_2 = 1e-6,
        alpha_init = None,
        lambda_init = None,
        compute_score = false,
        fit_intercept = true,
        copy_x = true,
        verbose = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_iter: usize,
        tol: f64,
        alpha_1: f64,
        alpha_2: f64,
        lambda_1: f64,
        lambda_2: f64,
        alpha_init: Option<f64>,
        lambda_init: Option<f64>,
        compute_score: bool,
        fit_intercept: bool,
        copy_x: bool,
        verbose: bool,
    ) -> Self {
        Self {
            inner: AnyBayesianRidge::Unfit {
                max_iter,
                tol,
                alpha_1,
                alpha_2,
                lambda_1,
                lambda_2,
                alpha_init,
                lambda_init,
                compute_score,
                fit_intercept,
                copy_x,
                verbose,
            },
            alpha: None,
            lambda: None,
            sigma: None,
            scores: Vec::new(),
            n_iter: None,
            x_offset: Vec::new(),
            x_scale: Vec::new(),
        }
    }

    /// `fit(X, y, rows, cols, sample_weight=None)`.
    ///
    /// `sample_weight` is an optional length-`rows` Arrow float array in the SAME
    /// dtype as `X` — it is borrowed as a host slice (never uploaded), because
    /// the weighted preprocessing that consumes it is a host pass anyway.
    #[pyo3(signature = (x, y, rows, cols, sample_weight = None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
        sample_weight: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let swa = sample_weight.map(capsule_to_array).transpose()?;
        let dt = float_dtype(&xa)?;
        let p = self.params();
        type Snapshot = (f64, f64, Vec<f64>, Vec<f64>, usize, Vec<f64>, Vec<f64>);
        let (fitted, snap) = py.detach(|| -> PyResult<(AnyBayesianRidge, Snapshot)> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let fitted = bayes_fit_dispatch!(
                        f32, p, xa, ya, swa, rows, cols, pool,
                        as_f32, host_slice_f32, validated_f32
                    );
                    let snap = bayes_snapshot!(fitted);
                    Ok((AnyBayesianRidge::F32(fitted), snap))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let fitted = bayes_fit_dispatch!(
                        f64, p, xa, ya, swa, rows, cols, pool,
                        as_f64, host_slice_f64, validated_f64
                    );
                    let snap = bayes_snapshot!(fitted);
                    Ok((AnyBayesianRidge::F64(fitted), snap))
                }
            }
        })?;
        self.inner = fitted;
        let (alpha, lambda, sigma, scores, n_iter, x_offset, x_scale) = snap;
        self.alpha = Some(alpha);
        self.lambda = Some(lambda);
        self.sigma = Some(sigma);
        self.scores = scores;
        self.n_iter = Some(n_iter);
        self.x_offset = x_offset;
        self.x_scale = x_scale;
        Ok(())
    }

    /// `predict(x)` → a length-`rows` **pyarrow** float array (D-03). Shared
    /// body: [`dense_predict_f32`].
    fn predict_f32<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyBayesianRidge::F32(est) => dense_predict_f32(py, x, (rows, cols), est),
            _ => Err(not_fitted("bayesian_ridge", "predict (f32 path)")),
        }
    }
    fn predict_f64<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyBayesianRidge::F64(est) => dense_predict_f64(py, x, (rows, cols), est),
            _ => Err(not_fitted("bayesian_ridge", "predict (f64 path)")),
        }
    }

    /// sklearn's `predict(X, return_std=True)` second return value — the
    /// per-sample predictive standard deviation `√(xᵢ·Σ·xᵢᵀ + 1/α)`.
    ///
    /// Always `f64`: it is derived from `sigma_` and `alpha_`, both of which are
    /// `f64` on either fitted arm. Returned over Arrow, like `predict`.
    fn predict_std_f32<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        let xa = capsule_to_array(x)?;
        let out = py.detach(|| -> PyResult<Vec<f64>> {
            let xh = host_slice_f32(as_f32(&xa)?)?;
            match &self.inner {
                AnyBayesianRidge::F32(est) => est
                    .predict_std_from_host(xh, (rows, cols))
                    .map_err(algo_err_to_py),
                _ => Err(not_fitted("bayesian_ridge", "predict std (f32 path)")),
            }
        })?;
        f64_vec_to_pyarrow(py, out)
    }
    fn predict_std_f64<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        let xa = capsule_to_array(x)?;
        let out = py.detach(|| -> PyResult<Vec<f64>> {
            let xh = host_slice_f64(as_f64(&xa)?)?;
            match &self.inner {
                AnyBayesianRidge::F64(est) => est
                    .predict_std_from_host(xh, (rows, cols))
                    .map_err(algo_err_to_py),
                _ => Err(not_fitted("bayesian_ridge", "predict std (f64 path)")),
            }
        })?;
        f64_vec_to_pyarrow(py, out)
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyBayesianRidge::F32(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("bayesian_ridge", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyBayesianRidge::F64(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("bayesian_ridge", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyBayesianRidge::F32(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("bayesian_ridge", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyBayesianRidge::F64(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("bayesian_ridge", "intercept_ (f64)")),
        }
    }

    /// sklearn's `alpha_` — the estimated noise precision.
    ///
    /// Named `*_prec` rather than `alpha`/`lambda` because `lambda` is a Python
    /// KEYWORD: a `#[pyclass]` method called `lambda` registers fine but is
    /// unreachable from Python (`obj.lambda()` is a `SyntaxError`), so it would
    /// only fail at the shim. `alpha` follows the same spelling for symmetry.
    fn alpha_prec(&self) -> Option<f64> {
        self.alpha
    }
    /// sklearn's `lambda_` — the estimated weight precision. See
    /// [`PyBayesianRidge::alpha_prec`] for the name.
    fn lambda_prec(&self) -> Option<f64> {
        self.lambda
    }
    /// sklearn's `sigma_`, flattened row-major (`d × d`); the shim reshapes.
    fn sigma(&self) -> Option<Vec<f64>> {
        self.sigma.clone()
    }
    /// sklearn's `scores_` — empty unless the estimator was built with
    /// `compute_score`.
    fn scores(&self) -> Vec<f64> {
        self.scores.clone()
    }
    /// sklearn's `n_iter_` — evidence iterations actually run.
    fn n_iter(&self) -> Option<usize> {
        self.n_iter
    }
    /// sklearn's `X_offset_` — the column means removed before the fit.
    fn x_offset(&self) -> Vec<f64> {
        self.x_offset.clone()
    }
    /// sklearn's `X_scale_` — all ones (the attribute outlived `normalize`).
    fn x_scale(&self) -> Vec<f64> {
        self.x_scale.clone()
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyBayesianRidge::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyBayesianRidge::Unfit { .. } => None,
            AnyBayesianRidge::F32(_) => Some("f32"),
            AnyBayesianRidge::F64(_) => Some("f64"),
        }
    }
}
