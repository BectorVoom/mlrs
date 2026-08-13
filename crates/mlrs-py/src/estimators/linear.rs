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
use mlrs_algos::linear::huber::HuberRegressor;
use mlrs_algos::linear::lasso::Lasso;
use mlrs_algos::linear::linear_regression::LinearRegression;
use mlrs_algos::linear::linear_svc::LinearSVC;
use mlrs_algos::linear::linear_svr::LinearSVR;
use mlrs_algos::linear::logistic::LogisticRegression;
use mlrs_algos::linear::mbsgd_classifier::MBSGDClassifier;
use mlrs_algos::linear::mbsgd_regressor::MBSGDRegressor;
use mlrs_algos::linear::ridge::{Ridge, RidgeSolver};
use mlrs_backend::device::Device;
use mlrs_algos::linear::ridge_classifier::{ClassWeight, RidgeClassifier};
use mlrs_algos::linear::ridge_cv::{ridge_cv_grid, ridge_gcv, GcvFit, GcvMode, GcvRoute, GridFit};
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
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::linear_predict::{linear_predict_multi_host, HostPrediction};
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
    // `HuberRegressor` predicts through the same `X·coef_ + intercept_` matvec
    // as the rest — the robustness lives entirely in `fit`.
    HuberRegressor,
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

/// Multi-target twin of [`dense_predict_f32`] (RIDGE-MULTI-TARGET): returns an
/// `m × n_targets` row-major **pyarrow** float array. Ridge-only — the shared
/// [`DensePredictHost`] trait (and the four OTHER dense linear regressors that
/// implement it) stays single-target, so this is a free function over the
/// concrete `Ridge<f32, AlgoFitted>` rather than a trait method.
fn dense_predict_multi_f32<'py>(
    py: Python<'py>,
    x: &Bound<'_, PyAny>,
    (rows, cols): (usize, usize),
    est: &Ridge<f32, AlgoFitted>,
) -> PyResult<Bound<'py, PyAny>> {
    let xa = capsule_to_array(x)?;
    let out = py.detach(|| -> PyResult<Vec<f32>> {
        let mut pool = crate::lock_pool();
        let xh = host_slice_f32(as_f32(&xa)?)?;
        let pred = est
            .predict_multi_from_host(&mut pool, xh, (rows, cols))
            .map_err(algo_err_to_py)?;
        if !pred.operand_finite {
            return Err(nonfinite_input_err(xh, "float32"));
        }
        Ok(pred.values)
    })?;
    f32_vec_to_pyarrow(py, out)
}

/// f64 twin of [`dense_predict_multi_f32`].
fn dense_predict_multi_f64<'py>(
    py: Python<'py>,
    x: &Bound<'_, PyAny>,
    (rows, cols): (usize, usize),
    est: &Ridge<f64, AlgoFitted>,
) -> PyResult<Bound<'py, PyAny>> {
    let xa = capsule_to_array(x)?;
    let out = py.detach(|| -> PyResult<Vec<f64>> {
        let mut pool = crate::lock_pool();
        let xh = host_slice_f64(as_f64(&xa)?)?;
        let pred = est
            .predict_multi_from_host(&mut pool, xh, (rows, cols))
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

crate::impl_persistable_any! {
    any:  AnyLinearRegression,
    algo: mlrs_algos::linear::linear_regression::LinearRegression,
    name: "linear_regression",
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

    /// Serialize the fitted model to `path` (MODEL-PERSIST).
    ///
    /// `extra` carries the Python shim's own state — `get_params()`, the class
    /// name, the fitted attributes the Rust estimator does not hold — merged
    /// into the file's `__metadata__` under a `py:` prefix. The shim supplies
    /// it; a caller using this `#[pyclass]` directly can pass an empty list and
    /// gets a plain mlrs model file.
    #[pyo3(signature = (path, extra = Vec::new()))]
    fn save(&self, py: Python<'_>, path: &str, extra: Vec<(String, String)>) -> PyResult<()> {
        crate::persist::save_impl(py, &self.inner, path, extra)
    }

    /// Replace this wrapper's fitted state with the model in `path`.
    ///
    /// An instance method rather than a constructor, mirroring `fit`: the
    /// wrapper keeps its hyperparameters beside `inner`, and the Python shim has
    /// already rebuilt them from the file's `py:` metadata before calling this.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner = crate::persist::load_impl(py, path)?;
        Ok(())
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
        device: String,
    },
}

crate::impl_persistable_any! {
    any:  AnyRidge,
    algo: mlrs_algos::linear::ridge::Ridge,
    name: "ridge",
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
    /// DEVICE-PARAM-01. Stays a STRING until `fit` for the same reason `solver`
    /// does: the parse and its `UnknownDevice` rejection belong with the rest of
    /// the `build()` validation (D-09).
    device: String,
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
    /// `device_` — the execution arm that actually ran (DEVICE-PARAM-01),
    /// mirrored here for the same reason as `solver_used`: a `#[pyclass]`
    /// getter cannot reach through the dtype dispatch generically.
    device_used: Option<String>,
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
                device: "auto".to_string(),
            },
            n_iter: None,
            solver_used: None,
            device_used: None,
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
                device,
            } => RidgeParams {
                alpha: *alpha,
                fit_intercept: *fit_intercept,
                copy_x: *copy_x,
                max_iter: *max_iter,
                tol: *tol,
                solver: solver.clone(),
                positive: *positive,
                random_state: *random_state,
                device: device.clone(),
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
                device: "auto".to_string(),
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
        let device = parse_device(&$p.device)?;
        Ridge::<$float>::builder()
            .alpha($p.alpha)
            .fit_intercept($p.fit_intercept)
            .copy_x($p.copy_x)
            .max_iter($p.max_iter)
            .tol($p.tol)
            .solver(solver)
            .positive($p.positive)
            .random_state($p.random_state)
            .device(device)
            .build::<$float>()
            .map_err(build_err_to_py)?
    }};
}

/// Parse the sklearn-style `device` string into the typed [`Device`], turning
/// an unrecognised value into the same `BuildError`-shaped rejection every other
/// string hyperparameter uses (D-09).
///
/// Shared by every estimator that takes the parameter, so they cannot drift on
/// which spellings they accept.
pub(crate) fn parse_device(value: &str) -> PyResult<Device> {
    Device::from_name(value).ok_or_else(|| {
        build_err_to_py(mlrs_algos::error::BuildError::UnknownDevice {
            value: value.to_string(),
        })
    })
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

/// Multi-target twin of [`ridge_fit_dispatch`] (RIDGE-MULTI-TARGET): `$ya` is
/// `rows × n_targets` row-major (already ravelled that way by the Python
/// shim's `_normalize_y`, which accepts 2-D input via `check_array(ensure_2d=
/// False)`). Always takes the DEVICE ingress —
/// [`Ridge::fit_multi_target_with_sample_weight`] has no host-slice twin of
/// [`Ridge::fit_from_host_slice`] (fit-side perf is not this feature's target;
/// see that method's Rust doc comment for why).
macro_rules! ridge_fit_multi_dispatch {
    ($float:ty, $p:expr, $xa:expr, $ya:expr, $swa:expr, $rows:expr, $cols:expr, $n_targets:expr,
     $pool:expr, $as:ident, $host_slice:ident, $validated:ident) => {{
        let est = ridge_build!($float, $p);
        let sw = match $swa.as_ref() {
            Some(a) => Some($host_slice($as(a)?)?),
            None => None,
        };
        let xd = $validated($as(&$xa)?, &mut $pool)?;
        let yd = $validated($as(&$ya)?, &mut $pool)?;
        est.fit_multi_target_with_sample_weight(
            &mut $pool, &xd, &yd, ($rows, $cols), $n_targets, sw,
        )
        .map_err(algo_err_to_py)?
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
        device = "auto".to_string(),
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
        device: String,
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
                device,
            },
            n_iter: None,
            solver_used: None,
            device_used: None,
        }
    }

    /// `fit(X, y, rows, cols, sample_weight=None, n_targets=1)`.
    ///
    /// `sample_weight` is an optional length-`rows` Arrow float array in the SAME
    /// dtype as `X` — it is borrowed as a host slice (never uploaded), because
    /// the weighted preprocessing that consumes it is a host pass anyway
    /// (`ridge.rs::preprocess`).
    ///
    /// `n_targets` (RIDGE-MULTI-TARGET): `1` (the default) is the ORIGINAL
    /// single-target entry point, unchanged — `y` is length `rows`. `> 1` means
    /// `y` is `rows × n_targets` row-major (the Python shim's `_normalize_y`
    /// already ravels a 2-D `y` that way) and routes to
    /// [`Ridge::fit_multi_target_with_sample_weight`], which currently accepts
    /// it only for the default `cholesky`/`auto` (`positive=False`) solver —
    /// any other solver with `n_targets > 1` surfaces a typed
    /// `UnsupportedCapability` error rather than guessing.
    #[pyo3(signature = (x, y, rows, cols, sample_weight = None, n_targets = 1))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
        sample_weight: Option<&Bound<'_, PyAny>>,
        n_targets: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let swa = sample_weight.map(capsule_to_array).transpose()?;
        let dt = float_dtype(&xa)?;
        let p = self.params();
        let (fitted, n_iter, solver_used, device_used) =
            py.detach(|| -> PyResult<(AnyRidge, Option<usize>, String, String)> {
                let mut pool = crate::lock_pool();
                match dt {
                    FloatDtype::F32 => {
                        let fitted = if n_targets <= 1 {
                            ridge_fit_dispatch!(
                                f32, p, xa, ya, swa, rows, cols, pool,
                                as_f32, host_slice_f32, validated_f32
                            )
                        } else {
                            ridge_fit_multi_dispatch!(
                                f32, p, xa, ya, swa, rows, cols, n_targets, pool,
                                as_f32, host_slice_f32, validated_f32
                            )
                        };
                        let n_iter = fitted.n_iter();
                        let used = fitted.solver().name().to_string();
                        let arm = fitted.device().to_string();
                        Ok((AnyRidge::F32(fitted), n_iter, used, arm))
                    }
                    FloatDtype::F64 => {
                        crate::capability::guard_f64()?;
                        let fitted = if n_targets <= 1 {
                            ridge_fit_dispatch!(
                                f64, p, xa, ya, swa, rows, cols, pool,
                                as_f64, host_slice_f64, validated_f64
                            )
                        } else {
                            ridge_fit_multi_dispatch!(
                                f64, p, xa, ya, swa, rows, cols, n_targets, pool,
                                as_f64, host_slice_f64, validated_f64
                            )
                        };
                        let n_iter = fitted.n_iter();
                        let used = fitted.solver().name().to_string();
                        let arm = fitted.device().to_string();
                        Ok((AnyRidge::F64(fitted), n_iter, used, arm))
                    }
                }
            })?;
        self.inner = fitted;
        self.n_iter = n_iter;
        self.solver_used = Some(solver_used);
        self.device_used = Some(device_used);
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

    /// `device_` — the execution arm that actually ran (`"cpu"` / `"gpu"`).
    fn device_used(&self) -> Option<String> {
        self.device_used.clone()
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

    // -- multi-target (RIDGE-MULTI-TARGET) --------------------------------- //

    /// `predict(x)` for a multi-target fit → an `m × n_targets` row-major
    /// **pyarrow** float array. Shared body: [`dense_predict_multi_f32`].
    fn predict_multi_f32<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyRidge::F32(est) => dense_predict_multi_f32(py, x, (rows, cols), est),
            _ => Err(not_fitted("ridge", "predict (f32 multi-target path)")),
        }
    }
    fn predict_multi_f64<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyRidge::F64(est) => dense_predict_multi_f64(py, x, (rows, cols), est),
            _ => Err(not_fitted("ridge", "predict (f64 multi-target path)")),
        }
    }

    /// Flat `n_features × n_targets` row-major `coef_` (the Python shim
    /// reshapes and transposes to sklearn's `(n_targets, n_features)`).
    fn coef_multi_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyRidge::F32(e) => Ok(e.coef_multi(&pool)),
            _ => Err(not_fitted("ridge", "coef_ (f32 multi-target)")),
        }
    }
    fn coef_multi_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyRidge::F64(e) => Ok(e.coef_multi(&pool)),
            _ => Err(not_fitted("ridge", "coef_ (f64 multi-target)")),
        }
    }

    /// Length-`n_targets` `intercept_`.
    fn intercept_multi_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyRidge::F32(e) => Ok(e.intercept_multi(&pool)),
            _ => Err(not_fitted("ridge", "intercept_ (f32 multi-target)")),
        }
    }
    fn intercept_multi_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyRidge::F64(e) => Ok(e.intercept_multi(&pool)),
            _ => Err(not_fitted("ridge", "intercept_ (f64 multi-target)")),
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

    /// Serialize the fitted model to `path` (MODEL-PERSIST).
    ///
    /// `extra` carries the Python shim's own state — `get_params()`, the class
    /// name, the fitted attributes the Rust estimator does not hold — merged
    /// into the file's `__metadata__` under a `py:` prefix. The shim supplies
    /// it; a caller using this `#[pyclass]` directly can pass an empty list and
    /// gets a plain mlrs model file.
    #[pyo3(signature = (path, extra = Vec::new()))]
    fn save(&self, py: Python<'_>, path: &str, extra: Vec<(String, String)>) -> PyResult<()> {
        crate::persist::save_impl(py, &self.inner, path, extra)
    }

    /// Replace this wrapper's fitted state with the model in `path`.
    ///
    /// An instance method rather than a constructor, mirroring `fit`: the
    /// wrapper keeps its hyperparameters beside `inner`, and the Python shim has
    /// already rebuilt them from the file's `py:` metadata before calling this.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner = crate::persist::load_impl(py, path)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RidgeCV — the GCV / grid ENGINES plus a host predictor
//   alphas, fit_intercept, scoring, cv, gcv_mode, store_cv_results,
//   alpha_per_target + fit(..., sample_weight) and the alpha_ / best_score_ /
//   cv_results_ fitted attributes
// ---------------------------------------------------------------------------

/// Fitted `(coef_, intercept_)` for `RidgeCV`, at the dtype `fit` saw (D-06).
///
/// `coef` is `n_features × n_targets` ROW-MAJOR — the layout
/// [`linear_predict_multi_host`] consumes and the one `Ridge`'s multi-target arm
/// already produces, so `predict` is one fused host matvec whatever the target
/// count.
enum RidgeCVCoef {
    F32 {
        coef: Vec<f32>,
        intercept: Vec<f32>,
    },
    F64 {
        coef: Vec<f64>,
        intercept: Vec<f64>,
    },
}

/// sklearn-compatible `RidgeCV` — the compiled half.
///
/// Unlike the other wrappers in this file, this one is an ENGINE rather than a
/// one-shot `fit`: `RidgeCV`'s `scoring` may be an arbitrary Python callable and
/// its `cv` an arbitrary Python splitter, neither of which Rust can call. The
/// split is the same one `model_selection::search` makes — Rust owns the
/// `O(n·d²)` decompositions and the per-alpha algebra, the shim owns the scorer
/// and the splitter — and it costs one boundary crossing per fit, not one per
/// alpha.
///
/// The shim therefore drives three calls: [`PyRidgeCV::gcv`] or
/// [`PyRidgeCV::grid`] to run the engine, its accessors to read the per-alpha
/// results back, then [`PyRidgeCV::set_fitted`] with the winning
/// `(coef_, intercept_)` so `predict` has a home.
#[pyclass(name = "RidgeCV")]
pub struct PyRidgeCV {
    /// The penalty grid, verbatim from the ctor (validated in the engine).
    alphas: Vec<f64>,
    /// sklearn's `fit_intercept`.
    fit_intercept: bool,
    /// sklearn's `gcv_mode`, still a STRING until `gcv` (the `RidgeSolver`
    /// precedent: the parse and its `UnknownGcvMode` rejection happen with the
    /// rest of the validation, D-09).
    gcv_mode: String,
    /// The last [`ridge_gcv`] result, held so the shim can read the pieces it
    /// needs without re-running the decomposition.
    gcv: Option<GcvFit>,
    /// The last [`ridge_cv_grid`] result.
    grid: Option<GridFit>,
    /// The winning coefficients, once the shim has chosen them.
    fitted: Option<RidgeCVCoef>,
    /// `n_features` of the fitted `coef_`.
    n_features: usize,
    /// `n_targets` of the fitted `coef_`.
    n_targets: usize,
}

impl PyRidgeCV {
    /// Rust-callable default constructor for the smoke test (the
    /// [`PyLinearRegression::unfit_default`] convention).
    pub fn unfit_default() -> Self {
        Self {
            alphas: vec![0.1, 1.0, 10.0],
            fit_intercept: true,
            gcv_mode: "auto".to_string(),
            gcv: None,
            grid: None,
            fitted: None,
            n_features: 0,
            n_targets: 0,
        }
    }

    /// Is this wrapper still unfitted?
    pub fn is_unfit(&self) -> bool {
        self.fitted.is_none()
    }

    /// The stored [`GcvFit`], or sklearn's `NotFittedError` shape.
    fn gcv_ref(&self) -> PyResult<&GcvFit> {
        self.gcv
            .as_ref()
            .ok_or_else(|| not_fitted("ridge_cv", "gcv results (fit has not run)"))
    }
}

#[pymethods]
impl PyRidgeCV {
    /// `RidgeCV(alphas, fit_intercept=True, gcv_mode='auto')` — the subset of
    /// sklearn's signature the compiled half needs. `scoring`, `cv`,
    /// `store_cv_results` and `alpha_per_target` stay in the shim: they select
    /// which engine call is made and how its output is reduced, not what the
    /// engine computes.
    #[new]
    #[pyo3(signature = (alphas, fit_intercept = true, gcv_mode = "auto".to_string()))]
    fn new(alphas: Vec<f64>, fit_intercept: bool, gcv_mode: String) -> Self {
        Self {
            alphas,
            fit_intercept,
            gcv_mode,
            gcv: None,
            grid: None,
            fitted: None,
            n_features: 0,
            n_targets: 0,
        }
    }

    /// Run the generalized (leave-one-out) CV engine — sklearn's `cv=None` arm.
    ///
    /// `y` is `rows × n_targets` row-major. `want_predictions` switches the
    /// per-alpha output from squared LOO errors (which the engine reduces to
    /// scores itself) to rescaled LOO predictions, which is what a non-`None`
    /// `scoring` needs; `store_cv_values` fills the same buffer for
    /// `store_cv_results=True`.
    #[pyo3(signature = (
        x, y, rows, cols, n_targets, sample_weight = None,
        want_predictions = false, store_cv_values = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn gcv(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
        n_targets: usize,
        sample_weight: Option<&Bound<'_, PyAny>>,
        want_predictions: bool,
        store_cv_values: bool,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let swa = sample_weight.map(capsule_to_array).transpose()?;
        let dt = float_dtype(&xa)?;
        let mode = GcvMode::try_from(self.gcv_mode.as_str()).map_err(build_err_to_py)?;
        let (alphas, fit_intercept) = (self.alphas.clone(), self.fit_intercept);
        let out = py.detach(|| -> PyResult<GcvFit> {
            match dt {
                FloatDtype::F32 => {
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    let yh = host_slice_f32(as_f32(&ya)?)?;
                    let sw = sample_weight_f64(swa.as_ref(), dt)?;
                    ridge_gcv::<f32>(
                        xh,
                        yh,
                        rows,
                        cols,
                        n_targets,
                        sw.as_deref(),
                        &alphas,
                        fit_intercept,
                        mode,
                        want_predictions,
                        store_cv_values,
                    )
                    .map_err(algo_err_to_py)
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    let yh = host_slice_f64(as_f64(&ya)?)?;
                    let sw = sample_weight_f64(swa.as_ref(), dt)?;
                    ridge_gcv::<f64>(
                        xh,
                        yh,
                        rows,
                        cols,
                        n_targets,
                        sw.as_deref(),
                        &alphas,
                        fit_intercept,
                        mode,
                        want_predictions,
                        store_cv_values,
                    )
                    .map_err(algo_err_to_py)
                }
            }
        })?;
        self.gcv = Some(out);
        Ok(())
    }

    /// `n_alphas × n_targets` row-major per-target scores (`−mean(looe²)`).
    /// Empty when the engine produced predictions instead.
    fn gcv_scores(&self) -> PyResult<Vec<f64>> {
        Ok(self.gcv_ref()?.scores.clone())
    }

    /// `n_alphas × n_features × n_targets` row-major coefficients of the
    /// CENTERED problem, over Arrow (never a Python list).
    fn gcv_coefs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        f64_vec_to_pyarrow(py, self.gcv_ref()?.coefs.clone())
    }

    /// `n_samples × n_alphas × n_targets` row-major squared errors (or
    /// predictions), over Arrow — this is the one buffer that scales with
    /// `n_samples`, so it must not become a list.
    ///
    /// MOVES the buffer out rather than cloning it: at `n = 100 000` and 200
    /// alphas it is 160 MiB, and the shim reads it exactly once (for the
    /// scorer, for `cv_results_`, or for both at once from the same array).
    /// Cloning would double the peak for no reader. A second call returns an
    /// empty array, which is why the shim binds it to a local.
    fn gcv_cv_values<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let taken = std::mem::take(
            &mut self
                .gcv
                .as_mut()
                .ok_or_else(|| not_fitted("ridge_cv", "gcv results (fit has not run)"))?
                .cv_values,
        );
        f64_vec_to_pyarrow(py, taken)
    }

    /// The weighted column means (`X_offset`) the intercept is recovered from.
    fn gcv_x_offset(&self) -> PyResult<Vec<f64>> {
        Ok(self.gcv_ref()?.x_offset.clone())
    }

    /// The weighted target means (`y_offset`).
    fn gcv_y_offset(&self) -> PyResult<Vec<f64>> {
        Ok(self.gcv_ref()?.y_offset.clone())
    }

    /// Which Gram the engine decomposed — `"gram"` (`n ≤ d`) or `"cov"`
    /// (`n > d`). Reported for the perf probe and the route test, not used by
    /// the shim.
    fn gcv_route(&self) -> PyResult<&'static str> {
        Ok(match self.gcv_ref()?.route {
            GcvRoute::Gram => "gram",
            GcvRoute::Cov => "cov",
        })
    }

    /// Run the explicit-`cv` engine — sklearn's `GridSearchCV(Ridge(), {'alpha':
    /// alphas}, cv=cv)` arm, with the train Gram hoisted out of the alpha loop.
    ///
    /// `train`/`test` are the materialized fold indices, in the splitter's own
    /// order (`mlrs.model_selection.check_cv` produces sklearn-identical rows).
    #[pyo3(signature = (
        x, y, rows, cols, n_targets, train, test, sample_weight = None,
        want_predictions = false, weighted_score = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn grid(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
        n_targets: usize,
        train: Vec<Vec<usize>>,
        test: Vec<Vec<usize>>,
        sample_weight: Option<&Bound<'_, PyAny>>,
        want_predictions: bool,
        weighted_score: bool,
    ) -> PyResult<()> {
        if train.len() != test.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "RidgeCV: train and test index lists must have the same length",
            ));
        }
        let xa = capsule_to_array(x)?;
        let ya = capsule_to_array(y)?;
        let swa = sample_weight.map(capsule_to_array).transpose()?;
        let dt = float_dtype(&xa)?;
        let splits: Vec<(Vec<usize>, Vec<usize>)> =
            train.into_iter().zip(test).collect();
        let (alphas, fit_intercept) = (self.alphas.clone(), self.fit_intercept);
        let out = py.detach(|| -> PyResult<GridFit> {
            match dt {
                FloatDtype::F32 => {
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    let yh = host_slice_f32(as_f32(&ya)?)?;
                    let sw = sample_weight_f64(swa.as_ref(), dt)?;
                    ridge_cv_grid::<f32>(
                        xh,
                        yh,
                        rows,
                        cols,
                        n_targets,
                        sw.as_deref(),
                        &alphas,
                        fit_intercept,
                        &splits,
                        want_predictions,
                        weighted_score,
                    )
                    .map_err(algo_err_to_py)
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    let yh = host_slice_f64(as_f64(&ya)?)?;
                    let sw = sample_weight_f64(swa.as_ref(), dt)?;
                    ridge_cv_grid::<f64>(
                        xh,
                        yh,
                        rows,
                        cols,
                        n_targets,
                        sw.as_deref(),
                        &alphas,
                        fit_intercept,
                        &splits,
                        want_predictions,
                        weighted_score,
                    )
                    .map_err(algo_err_to_py)
                }
            }
        })?;
        self.grid = Some(out);
        Ok(())
    }

    /// `n_splits × n_alphas` row-major R² test scores (`GridSearchCV`'s default
    /// regressor scoring, computed in Rust so the common case ships nothing but
    /// the reduction back).
    fn grid_scores(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .grid
            .as_ref()
            .ok_or_else(|| not_fitted("ridge_cv", "grid results (fit has not run)"))?
            .scores
            .clone())
    }

    /// Split-major test predictions for a caller-supplied scorer, over Arrow.
    ///
    /// Moved out, not cloned — see [`PyRidgeCV::gcv_cv_values`].
    fn grid_predictions<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let taken = std::mem::take(
            &mut self
                .grid
                .as_mut()
                .ok_or_else(|| not_fitted("ridge_cv", "grid results (fit has not run)"))?
                .predictions,
        );
        f64_vec_to_pyarrow(py, taken)
    }

    /// Install the winning `(coef_, intercept_)` so `predict` works.
    ///
    /// `coef` is `n_features × n_targets` row-major and `intercept` is length
    /// `n_targets`, both in `f64`; `dtype` is the fitted arm the shim saw at
    /// ingress, and the values are narrowed to it here so `predict` runs at the
    /// same width as the design it will be handed (D-06).
    #[pyo3(signature = (coef, intercept, n_features, n_targets, dtype))]
    fn set_fitted(
        &mut self,
        coef: Vec<f64>,
        intercept: Vec<f64>,
        n_features: usize,
        n_targets: usize,
        dtype: &str,
    ) -> PyResult<()> {
        if coef.len() != n_features * n_targets || intercept.len() != n_targets {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "RidgeCV: coef_/intercept_ shape does not match (n_features, n_targets)",
            ));
        }
        self.n_features = n_features;
        self.n_targets = n_targets;
        self.fitted = Some(match dtype {
            "f32" => RidgeCVCoef::F32 {
                coef: coef.iter().map(|v| *v as f32).collect(),
                intercept: intercept.iter().map(|v| *v as f32).collect(),
            },
            "f64" => RidgeCVCoef::F64 { coef, intercept },
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "RidgeCV: unknown dtype '{other}' (expected 'f32' or 'f64')"
                )))
            }
        });
        // The per-alpha buffers are only needed until the winner is chosen; a
        // `store_cv_results` caller has already read `cv_values` back by now, so
        // holding an `n × n_alphas × n_y` array for the estimator's lifetime
        // would be pure footprint.
        if let Some(g) = self.gcv.as_mut() {
            g.cv_values = Vec::new();
        }
        if let Some(g) = self.grid.as_mut() {
            g.predictions = Vec::new();
        }
        Ok(())
    }

    /// `predict(x)` → an `m × n_targets` row-major **pyarrow** float array
    /// (`m` when `n_targets == 1`). Host matvec, no upload, no Python list —
    /// the RIDGE-PREDICT-CUDA-VS-CPU verdict applies verbatim here: `predict` is
    /// `O(n·d)` compute over an `O(n·d)` transfer, so the device arm never wins.
    fn predict_f32<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Some(RidgeCVCoef::F32 { coef, intercept }) = self.fitted.as_ref() else {
            return Err(not_fitted("ridge_cv", "predict (f32 path)"));
        };
        let xa = capsule_to_array(x)?;
        let out = py.detach(|| -> PyResult<Vec<f32>> {
            let xh = host_slice_f32(as_f32(&xa)?)?;
            let pred =
                linear_predict_multi_host::<f32>(xh, coef, intercept, (rows, cols), self.n_targets)
                    .map_err(|e| algo_err_to_py(AlgoError::Prim(e)))?;
            if !pred.operand_finite {
                return Err(nonfinite_input_err(xh, "float32"));
            }
            Ok(pred.values)
        })?;
        f32_vec_to_pyarrow(py, out)
    }

    /// f64 twin of [`PyRidgeCV::predict_f32`].
    fn predict_f64<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Some(RidgeCVCoef::F64 { coef, intercept }) = self.fitted.as_ref() else {
            return Err(not_fitted("ridge_cv", "predict (f64 path)"));
        };
        let xa = capsule_to_array(x)?;
        let out = py.detach(|| -> PyResult<Vec<f64>> {
            let xh = host_slice_f64(as_f64(&xa)?)?;
            let pred =
                linear_predict_multi_host::<f64>(xh, coef, intercept, (rows, cols), self.n_targets)
                    .map_err(|e| algo_err_to_py(AlgoError::Prim(e)))?;
            if !pred.operand_finite {
                return Err(nonfinite_input_err(xh, "float64"));
            }
            Ok(pred.values)
        })?;
        f64_vec_to_pyarrow(py, out)
    }

    /// `n_features × n_targets` row-major fitted `coef_` (the shim reshapes and
    /// transposes to sklearn's `(n_targets, n_features)`).
    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        match self.fitted.as_ref() {
            Some(RidgeCVCoef::F32 { coef, .. }) => Ok(coef.clone()),
            _ => Err(not_fitted("ridge_cv", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        match self.fitted.as_ref() {
            Some(RidgeCVCoef::F64 { coef, .. }) => Ok(coef.clone()),
            _ => Err(not_fitted("ridge_cv", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<Vec<f32>> {
        match self.fitted.as_ref() {
            Some(RidgeCVCoef::F32 { intercept, .. }) => Ok(intercept.clone()),
            _ => Err(not_fitted("ridge_cv", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<Vec<f64>> {
        match self.fitted.as_ref() {
            Some(RidgeCVCoef::F64 { intercept, .. }) => Ok(intercept.clone()),
            _ => Err(not_fitted("ridge_cv", "intercept_ (f64)")),
        }
    }

    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// The fitted dtype arm (`MlrsBase._suffix` reads this, D-06).
    fn dtype(&self) -> Option<&'static str> {
        match self.fitted.as_ref() {
            None => None,
            Some(RidgeCVCoef::F32 { .. }) => Some("f32"),
            Some(RidgeCVCoef::F64 { .. }) => Some("f64"),
        }
    }
}

/// Borrow an optional `sample_weight` Arrow array as the `f64` vector both
/// `ridge_cv` engines take.
///
/// Widened HERE rather than inside the engine because the engines are generic
/// over the DESIGN's element type while sklearn's weights are conceptually
/// `f64` whatever `X` is — and because both engines need the same widening, so
/// doing it once keeps them from drifting.
fn sample_weight_f64(
    swa: Option<&arrow::array::ArrayRef>,
    dt: FloatDtype,
) -> PyResult<Option<Vec<f64>>> {
    let Some(a) = swa else {
        return Ok(None);
    };
    let v: Vec<f64> = match dt {
        FloatDtype::F32 => host_slice_f32(as_f32(a)?)?
            .iter()
            .map(|v| *v as f64)
            .collect(),
        FloatDtype::F64 => host_slice_f64(as_f64(a)?)?.to_vec(),
    };
    if let Some(bad) = v.iter().position(|w| !w.is_finite() || *w < 0.0) {
        return Err(algo_err_to_py(AlgoError::InvalidSampleWeight {
            estimator: "ridge_cv",
            index: bad,
            value: v[bad],
        }));
    }
    Ok(Some(v))
}

// ---------------------------------------------------------------------------
// RidgeClassifier — Fit + PredictLabels + decision_function; classes_,
// coef_, intercept_, n_iter_, solver_
// ---------------------------------------------------------------------------

/// Parse sklearn's `class_weight` (`None` / `'balanced'` / `{label: weight}`)
/// into the typed [`ClassWeight`]. A non-scalar hyperparameter, so it is
/// parsed once here rather than threaded through as a Python object (the
/// `RidgeSolver` string-parse precedent, `ridge_build!`).
fn parse_class_weight(v: Option<&Bound<'_, PyAny>>) -> PyResult<ClassWeight> {
    let Some(v) = v else {
        return Ok(ClassWeight::Uniform);
    };
    if v.is_none() {
        return Ok(ClassWeight::Uniform);
    }
    if let Ok(s) = v.extract::<String>() {
        return if s == "balanced" {
            Ok(ClassWeight::Balanced)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "class_weight: unknown string '{s}' (expected 'balanced')"
            )))
        };
    }
    if let Ok(map) = v.extract::<std::collections::HashMap<i64, f64>>() {
        return Ok(ClassWeight::Map(map.into_iter().collect()));
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "class_weight must be None, 'balanced', or a {label: weight} dict",
    ))
}

/// The ctor hyperparameters, read back at every `fit` (WR-02: the typestate
/// wrapper rebuilds a fresh `Unfit` from these, so a second `fit` of the same
/// Python object works). Mirrors [`RidgeParams`]; `class_weight` is already
/// parsed into the typed enum (constructed once, at `#[new]`).
struct RidgeClassifierParams {
    alpha: f64,
    fit_intercept: bool,
    copy_x: bool,
    max_iter: Option<usize>,
    tol: f64,
    class_weight: ClassWeight,
    solver: String,
    positive: bool,
    random_state: Option<u64>,
    /// DEVICE-PARAM-01, a STRING until `fit` (D-09).
    device: String,
}

/// Dtype-dispatched fitted/unfit state (D-06) — hand-written like [`AnyRidge`]
/// rather than macro-emitted, because `class_weight` is a non-scalar field the
/// `any_estimator_typestate!` macro's scalar-only `unfit: { .. }` list cannot
/// express.
enum AnyRidgeClassifier {
    Unfit {
        alpha: f64,
        fit_intercept: bool,
        copy_x: bool,
        max_iter: Option<usize>,
        tol: f64,
        class_weight: ClassWeight,
        solver: String,
        positive: bool,
        random_state: Option<u64>,
        device: String,
    },
    F32(RidgeClassifier<f32, AlgoFitted>),
    F64(RidgeClassifier<f64, AlgoFitted>),
}

crate::impl_persistable_any! {
    any:  AnyRidgeClassifier,
    algo: mlrs_algos::linear::ridge_classifier::RidgeClassifier,
    name: "ridge_classifier",
}

/// sklearn-compatible `RidgeClassifier`.
#[pyclass(name = "RidgeClassifier")]
pub struct PyRidgeClassifier {
    inner: AnyRidgeClassifier,
    n_iter: Option<Vec<usize>>,
    solver_used: Option<String>,
}

impl PyRidgeClassifier {
    /// Rust-callable default constructor for the smoke test. See
    /// [`PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyRidgeClassifier::Unfit {
                alpha: 1.0,
                fit_intercept: true,
                copy_x: true,
                max_iter: None,
                tol: 1e-4,
                class_weight: ClassWeight::Uniform,
                solver: "auto".to_string(),
                positive: false,
                random_state: None,
                device: "auto".to_string(),
            },
            n_iter: None,
            solver_used: None,
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyRidgeClassifier::Unfit { .. })
    }

    fn params(&self) -> RidgeClassifierParams {
        match &self.inner {
            AnyRidgeClassifier::Unfit {
                alpha, fit_intercept, copy_x, max_iter, tol, class_weight, solver, positive, random_state,
                device,
            } => RidgeClassifierParams {
                device: device.clone(),
                alpha: *alpha,
                fit_intercept: *fit_intercept,
                copy_x: *copy_x,
                max_iter: *max_iter,
                tol: *tol,
                class_weight: class_weight.clone(),
                solver: solver.clone(),
                positive: *positive,
                random_state: *random_state,
            },
            // Already fitted: the shim always constructs a fresh wrapper per
            // `fit` (WR-02), so this arm is unreachable in practice.
            _ => RidgeClassifierParams {
                device: "auto".to_string(),
                alpha: 1.0,
                fit_intercept: true,
                copy_x: true,
                max_iter: None,
                tol: 1e-4,
                class_weight: ClassWeight::Uniform,
                solver: "auto".to_string(),
                positive: false,
                random_state: None,
            },
        }
    }
}

/// Build an unfit `RidgeClassifier<F>` from the ctor params (the `ridge_build!`
/// precedent).
macro_rules! ridge_classifier_build {
    ($float:ty, $p:expr) => {{
        let solver = RidgeSolver::try_from($p.solver.as_str()).map_err(build_err_to_py)?;
        let device = parse_device($p.device.as_str())?;
        RidgeClassifier::<$float>::builder()
            .alpha($p.alpha)
            .fit_intercept($p.fit_intercept)
            .copy_x($p.copy_x)
            .max_iter($p.max_iter)
            .tol($p.tol)
            .class_weight($p.class_weight)
            .solver(solver)
            .positive($p.positive)
            .random_state($p.random_state)
            .device(device)
            .build::<$float>()
            .map_err(build_err_to_py)?
    }};
}

/// The two-arm fit dispatch (the `ridge_fit_dispatch!` precedent): the
/// no-upload HOST arm on `host_fit_applicable` (`cholesky`/`lbfgs` + the cpu
/// backend, or below the dispatch-cost floor — this is the path
/// `RidgeClassifier()` takes on cpu, the estimator's whole reason for
/// existing), else the DEVICE delegation arm.
///
/// `class_weight` is consumed (not `Clone`d again) since the params struct is
/// rebuilt fresh per `fit` (WR-02) and each dispatch arm needs its own build.
macro_rules! ridge_classifier_fit_dispatch {
    ($float:ty, $p:expr, $xa:expr, $ya:expr, $swa:expr, $rows:expr, $cols:expr,
     $pool:expr, $as:ident, $host_slice:ident, $validated:ident) => {{
        let est = ridge_classifier_build!($float, $p);
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
impl PyRidgeClassifier {
    /// `RidgeClassifier(alpha=1.0, fit_intercept=True, copy_X=True,
    /// max_iter=None, tol=1e-4, class_weight=None, solver='auto',
    /// positive=False, random_state=None)` — sklearn's signature one-for-one.
    #[new]
    #[pyo3(signature = (
        alpha = 1.0,
        fit_intercept = true,
        copy_x = true,
        max_iter = None,
        tol = 1e-4,
        class_weight = None,
        solver = "auto".to_string(),
        positive = false,
        random_state = None,
        device = "auto".to_string(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        alpha: f64,
        fit_intercept: bool,
        copy_x: bool,
        max_iter: Option<usize>,
        tol: f64,
        class_weight: Option<&Bound<'_, PyAny>>,
        solver: String,
        positive: bool,
        random_state: Option<u64>,
        device: String,
    ) -> PyResult<Self> {
        let class_weight = parse_class_weight(class_weight)?;
        Ok(Self {
            inner: AnyRidgeClassifier::Unfit {
                alpha,
                fit_intercept,
                copy_x,
                max_iter,
                tol,
                class_weight,
                solver,
                device,
                positive,
                random_state,
            },
            n_iter: None,
            solver_used: None,
        })
    }

    /// `fit(X, y, rows, cols, sample_weight=None)`. `y` carries the RAW class
    /// labels (float-encoded, the `LogisticRegression` convention) — the
    /// `{-1,+1}` target encoding happens inside the estimator.
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
            py.detach(|| -> PyResult<(AnyRidgeClassifier, Option<Vec<usize>>, String)> {
                let mut pool = crate::lock_pool();
                match dt {
                    FloatDtype::F32 => {
                        let fitted = ridge_classifier_fit_dispatch!(
                            f32, p, xa, ya, swa, rows, cols, pool,
                            as_f32, host_slice_f32, validated_f32
                        );
                        let n_iter = fitted.n_iter();
                        let used = fitted.solver().name().to_string();
                        Ok((AnyRidgeClassifier::F32(fitted), n_iter, used))
                    }
                    FloatDtype::F64 => {
                        crate::capability::guard_f64()?;
                        let fitted = ridge_classifier_fit_dispatch!(
                            f64, p, xa, ya, swa, rows, cols, pool,
                            as_f64, host_slice_f64, validated_f64
                        );
                        let n_iter = fitted.n_iter();
                        let used = fitted.solver().name().to_string();
                        Ok((AnyRidgeClassifier::F64(fitted), n_iter, used))
                    }
                }
            })?;
        self.inner = fitted;
        self.n_iter = n_iter;
        self.solver_used = Some(solver_used);
        Ok(())
    }

    /// `predict(x)` → a length-`rows` **pyarrow** `int32` array of class ids
    /// (the `PyLinearSVC::predict_labels` no-upload / no-list precedent).
    ///
    /// TWO arms, chosen by `RidgeClassifier::device_predict_applicable`
    /// (RIDGECLF-CUDA): the no-upload HOST matvec — which is what the cpu
    /// backend and every low-cardinality fit take — or the fused DEVICE
    /// classify kernel, which computes the decision function, its `argmax` and
    /// the `classes_` lookup in one launch and brings back `rows` `i32`s rather
    /// than `rows × n_targets` floats. The device arm has to scan the query for
    /// NaN/±inf separately (the host arm folds that check into the matvec pass
    /// it is already making), which is the one thing it pays for the crossing
    /// beyond the upload itself.
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
                AnyRidgeClassifier::F32(est) => {
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    if est.device_predict_applicable() {
                        if xh.iter().any(|v| !v.is_finite()) {
                            return Err(nonfinite_input_err(xh, "float32"));
                        }
                        let xd = DeviceArray::from_host(&mut pool, xh);
                        let labels = est
                            .predict_labels_device(&mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?;
                        xd.release_into(&mut pool);
                        return Ok(labels.to_host_metered(&mut pool));
                    }
                    let pred = est.predict_labels_from_host(&pool, xh, (rows, cols)).map_err(algo_err_to_py)?;
                    if !pred.operand_finite {
                        return Err(nonfinite_input_err(xh, "float32"));
                    }
                    Ok(pred.labels)
                }
                AnyRidgeClassifier::F64(est) => {
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    if est.device_predict_applicable() {
                        if xh.iter().any(|v| !v.is_finite()) {
                            return Err(nonfinite_input_err(xh, "float64"));
                        }
                        let xd = DeviceArray::from_host(&mut pool, xh);
                        let labels = est
                            .predict_labels_device(&mut pool, &xd, (rows, cols))
                            .map_err(algo_err_to_py)?;
                        xd.release_into(&mut pool);
                        return Ok(labels.to_host_metered(&mut pool));
                    }
                    let pred = est.predict_labels_from_host(&pool, xh, (rows, cols)).map_err(algo_err_to_py)?;
                    if !pred.operand_finite {
                        return Err(nonfinite_input_err(xh, "float64"));
                    }
                    Ok(pred.labels)
                }
                _ => Err(not_fitted("ridge_classifier", "predict")),
            }
        })?;
        i32_vec_to_pyarrow(py, out)
    }

    /// `decision_function(x)` → row-major `rows × n_targets` **pyarrow** float
    /// array (binary squeezes to `n_targets == 1` at the Python shim, mirroring
    /// sklearn's own squeeze).
    ///
    /// Stays on the HOST arm on every backend, where `predict` above routes to
    /// the device kernel above a `n_targets` threshold. The asymmetry is not an
    /// oversight: the two effects that let a device `predict` pay back its
    /// upload are `k`× the compute AND `k`× less egress (`rows` `i32` labels
    /// instead of `rows × k` floats), and `decision_function` gets only the
    /// first of them — it has to return the full score matrix by definition.
    /// That leaves it with the same profile as `Ridge::predict_multi_from_host`,
    /// which measured a 2–3× LOSS at `n_targets = 4` on a P100. The device
    /// path exists and is gated by tests
    /// (`RidgeClassifier::decision_function_device`); it is not the default
    /// here because nothing has measured it winning.
    fn decision_function_f32<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        let xa = capsule_to_array(x)?;
        let out = py.detach(|| -> PyResult<Vec<f32>> {
            let pool = crate::lock_pool();
            match &self.inner {
                AnyRidgeClassifier::F32(est) => {
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    let scores = est.decision_function_from_host(&pool, xh, (rows, cols)).map_err(algo_err_to_py)?;
                    if !scores.operand_finite {
                        return Err(nonfinite_input_err(xh, "float32"));
                    }
                    Ok(scores.values.into_iter().map(|v| v as f32).collect())
                }
                _ => Err(not_fitted("ridge_classifier", "decision_function (f32 path)")),
            }
        })?;
        f32_vec_to_pyarrow(py, out)
    }
    fn decision_function_f64<'py>(&self, py: Python<'py>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Bound<'py, PyAny>> {
        let xa = capsule_to_array(x)?;
        let out = py.detach(|| -> PyResult<Vec<f64>> {
            let pool = crate::lock_pool();
            match &self.inner {
                AnyRidgeClassifier::F64(est) => {
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    let scores = est.decision_function_from_host(&pool, xh, (rows, cols)).map_err(algo_err_to_py)?;
                    if !scores.operand_finite {
                        return Err(nonfinite_input_err(xh, "float64"));
                    }
                    Ok(scores.values)
                }
                _ => Err(not_fitted("ridge_classifier", "decision_function (f64 path)")),
            }
        })?;
        f64_vec_to_pyarrow(py, out)
    }

    /// `1` for a binary fit, `n_classes` for multiclass — the row width of
    /// `decision_function`'s output and the row COUNT of `coef_`.
    fn n_targets(&self) -> PyResult<usize> {
        match &self.inner {
            AnyRidgeClassifier::F32(e) => Ok(e.n_targets()),
            AnyRidgeClassifier::F64(e) => Ok(e.n_targets()),
            _ => Err(not_fitted("ridge_classifier", "n_targets")),
        }
    }

    /// The DISTINCT sorted training labels (`classes_`, CR-02).
    fn classes_(&self) -> Vec<i64> {
        match &self.inner {
            AnyRidgeClassifier::F32(e) => e.classes().to_vec(),
            AnyRidgeClassifier::F64(e) => e.classes().to_vec(),
            _ => Vec::new(),
        }
    }

    /// sklearn's `n_iter_`: `None` unless the resolved solver is `lsqr` /
    /// `sag` / `saga` (in which case it is length `n_targets`).
    fn n_iter(&self) -> Option<Vec<usize>> {
        self.n_iter.clone()
    }

    /// sklearn's `solver_` — the solver that ACTUALLY ran.
    fn solver_used(&self) -> Option<String> {
        self.solver_used.clone()
    }

    /// `device_` — the execution arm that actually ran (`"cpu"` / `"gpu"`).
    ///
    /// Read off the FITTED estimator rather than mirrored at `fit`: the arm is
    /// already recorded on it, and re-deriving it here would be a second place
    /// to get the fallbacks wrong.
    fn device_used(&self) -> Option<String> {
        match &self.inner {
            AnyRidgeClassifier::F32(e) => Some(e.device().to_string()),
            AnyRidgeClassifier::F64(e) => Some(e.device().to_string()),
            _ => None,
        }
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyRidgeClassifier::F32(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("ridge_classifier", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyRidgeClassifier::F64(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("ridge_classifier", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyRidgeClassifier::F32(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("ridge_classifier", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyRidgeClassifier::F64(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("ridge_classifier", "intercept_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyRidgeClassifier::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyRidgeClassifier::Unfit { .. } => None,
            AnyRidgeClassifier::F32(_) => Some("f32"),
            AnyRidgeClassifier::F64(_) => Some("f64"),
        }
    }

    /// Serialize the fitted model to `path` (MODEL-PERSIST).
    ///
    /// `extra` carries the Python shim's own state — `get_params()`, the class
    /// name, the fitted attributes the Rust estimator does not hold — merged
    /// into the file's `__metadata__` under a `py:` prefix. The shim supplies
    /// it; a caller using this `#[pyclass]` directly can pass an empty list and
    /// gets a plain mlrs model file.
    #[pyo3(signature = (path, extra = Vec::new()))]
    fn save(&self, py: Python<'_>, path: &str, extra: Vec<(String, String)>) -> PyResult<()> {
        crate::persist::save_impl(py, &self.inner, path, extra)
    }

    /// Replace this wrapper's fitted state with the model in `path`.
    ///
    /// An instance method rather than a constructor, mirroring `fit`: the
    /// wrapper keeps its hyperparameters beside `inner`, and the Python shim has
    /// already rebuilt them from the file's `py:` metadata before calling this.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner = crate::persist::load_impl(py, path)?;
        Ok(())
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

crate::impl_persistable_any! {
    any:  AnyLasso,
    algo: mlrs_algos::linear::lasso::Lasso,
    name: "lasso",
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

    /// Serialize the fitted model to `path` (MODEL-PERSIST).
    ///
    /// `extra` carries the Python shim's own state — `get_params()`, the class
    /// name, the fitted attributes the Rust estimator does not hold — merged
    /// into the file's `__metadata__` under a `py:` prefix. The shim supplies
    /// it; a caller using this `#[pyclass]` directly can pass an empty list and
    /// gets a plain mlrs model file.
    #[pyo3(signature = (path, extra = Vec::new()))]
    fn save(&self, py: Python<'_>, path: &str, extra: Vec<(String, String)>) -> PyResult<()> {
        crate::persist::save_impl(py, &self.inner, path, extra)
    }

    /// Replace this wrapper's fitted state with the model in `path`.
    ///
    /// An instance method rather than a constructor, mirroring `fit`: the
    /// wrapper keeps its hyperparameters beside `inner`, and the Python shim has
    /// already rebuilt them from the file's `py:` metadata before calling this.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner = crate::persist::load_impl(py, path)?;
        Ok(())
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

crate::impl_persistable_any! {
    any:  AnyElasticNet,
    algo: mlrs_algos::linear::elastic_net::ElasticNet,
    name: "elastic_net",
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

    /// Serialize the fitted model to `path` (MODEL-PERSIST).
    ///
    /// `extra` carries the Python shim's own state — `get_params()`, the class
    /// name, the fitted attributes the Rust estimator does not hold — merged
    /// into the file's `__metadata__` under a `py:` prefix. The shim supplies
    /// it; a caller using this `#[pyclass]` directly can pass an empty list and
    /// gets a plain mlrs model file.
    #[pyo3(signature = (path, extra = Vec::new()))]
    fn save(&self, py: Python<'_>, path: &str, extra: Vec<(String, String)>) -> PyResult<()> {
        crate::persist::save_impl(py, &self.inner, path, extra)
    }

    /// Replace this wrapper's fitted state with the model in `path`.
    ///
    /// An instance method rather than a constructor, mirroring `fit`: the
    /// wrapper keeps its hyperparameters beside `inner`, and the Python shim has
    /// already rebuilt them from the file's `py:` metadata before calling this.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner = crate::persist::load_impl(py, path)?;
        Ok(())
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

crate::impl_persistable_any! {
    any:  AnyLogisticRegression,
    algo: mlrs_algos::linear::logistic::LogisticRegression,
    name: "logistic_regression",
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

    /// Serialize the fitted model to `path` (MODEL-PERSIST).
    ///
    /// `extra` carries the Python shim's own state — `get_params()`, the class
    /// name, the fitted attributes the Rust estimator does not hold — merged
    /// into the file's `__metadata__` under a `py:` prefix. The shim supplies
    /// it; a caller using this `#[pyclass]` directly can pass an empty list and
    /// gets a plain mlrs model file.
    #[pyo3(signature = (path, extra = Vec::new()))]
    fn save(&self, py: Python<'_>, path: &str, extra: Vec<(String, String)>) -> PyResult<()> {
        crate::persist::save_impl(py, &self.inner, path, extra)
    }

    /// Replace this wrapper's fitted state with the model in `path`.
    ///
    /// An instance method rather than a constructor, mirroring `fit`: the
    /// wrapper keeps its hyperparameters beside `inner`, and the Python shim has
    /// already rebuilt them from the file's `py:` metadata before calling this.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner = crate::persist::load_impl(py, path)?;
        Ok(())
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
        batch_size: usize, shuffle: bool, seed: u64, n_iter_no_change: usize,
        device: String,
    },
}

crate::impl_persistable_any! {
    any:  AnyMBSGDClassifier,
    algo: mlrs_algos::linear::mbsgd_classifier::MBSGDClassifier,
    name: "mbsgd_classifier",
}

crate::any_estimator_typestate! {
    any:   AnyMBSGDRegressor,
    algo:  mlrs_algos::linear::mbsgd_regressor::MBSGDRegressor,
    unfit: {
        loss: String, penalty: String, alpha: f64, l1_ratio: f64,
        fit_intercept: bool, max_iter: usize, tol: f64,
        learning_rate: String, eta0: f64, power_t: f64, epsilon: f64,
        batch_size: usize, shuffle: bool, seed: u64, n_iter_no_change: usize,
        device: String,
    },
}

crate::impl_persistable_any! {
    any:  AnyMBSGDRegressor,
    algo: mlrs_algos::linear::mbsgd_regressor::MBSGDRegressor,
    name: "mbsgd_regressor",
}

crate::any_estimator_typestate! {
    any:   AnyLinearSVC,
    algo:  mlrs_algos::linear::linear_svc::LinearSVC,
    unfit: {
        loss: String, penalty: String, c: f64, intercept_scaling: f64,
        fit_intercept: bool, max_iter: usize, tol: f64,
    },
}

crate::impl_persistable_any! {
    any:  AnyLinearSVC,
    algo: mlrs_algos::linear::linear_svc::LinearSVC,
    name: "linear_svc",
}

crate::any_estimator_typestate! {
    any:   AnyLinearSVR,
    algo:  mlrs_algos::linear::linear_svr::LinearSVR,
    unfit: {
        loss: String, penalty: String, c: f64, epsilon: f64,
        intercept_scaling: f64, fit_intercept: bool, max_iter: usize, tol: f64,
    },
}

crate::impl_persistable_any! {
    any:  AnyLinearSVR,
    algo: mlrs_algos::linear::linear_svr::LinearSVR,
    name: "linear_svr",
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
    /// `device_` — the arm that actually ran. Recorded at `fit` because the
    /// typestate transition consumes the `Unfit` variant the preference was
    /// read from, and a fitted estimator must still be able to name its arm.
    device_used: Option<String>,
}

impl PyMBSGDClassifier {
    /// Rust-callable default constructor (smoke test seam — see
    /// [`PyLinearRegression::unfit_default`]).
    pub fn unfit_default() -> Self {
        Self {
            device_used: None,
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
                n_iter_no_change: 5,
                device: "auto".to_string(),
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
    /// eta0=0.01, power_t=0.5, batch_size=1, shuffle=True, seed=0,
    /// n_iter_no_change=5)`.
    #[new]
    #[pyo3(signature = (
        loss = "hinge".to_string(), penalty = "l2".to_string(), alpha = 1e-4,
        l1_ratio = 0.15, fit_intercept = true, max_iter = 1000, tol = 1e-3,
        learning_rate = "optimal".to_string(), eta0 = 0.01, power_t = 0.5,
        batch_size = 1, shuffle = true, seed = 0, n_iter_no_change = 5,
        device = "auto".to_string(),
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
        n_iter_no_change: usize,
        device: String,
    ) -> Self {
        Self {
            device_used: None,
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
                n_iter_no_change,
                device,
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
            lr_s, eta0, power_t, batch_size, shuffle, seed, n_iter_no_change,
            device_s,
        ) = match &self.inner {
            AnyMBSGDClassifier::Unfit {
                loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol,
                learning_rate, eta0, power_t, batch_size, shuffle, seed,
                n_iter_no_change, device,
            } => (
                loss.clone(), penalty.clone(), *alpha, *l1_ratio, *fit_intercept,
                *max_iter, *tol, learning_rate.clone(), *eta0, *power_t,
                *batch_size, *shuffle, *seed, *n_iter_no_change, device.clone(),
            ),
            _ => return Err(not_fitted("mbsgd_classifier", "re-fit")),
        };
        // Construction-time enum-string validation (D-05 → ValueError).
        let loss = Loss::try_from(loss_s.as_str()).map_err(build_err_to_py)?;
        let penalty = Penalty::try_from(penalty_s.as_str()).map_err(build_err_to_py)?;
        let lr = LearningRate::try_from(lr_s.as_str()).map_err(build_err_to_py)?;
        let host_ingress =
            mlrs_backend::prims::sgd::sgd_host_available(parse_device(&device_s)?);
        let fitted = py.detach(|| -> PyResult<AnyMBSGDClassifier> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let est = MBSGDClassifier::<f32>::builder()
                        .device(parse_device(&device_s)?)
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
                        .n_iter_no_change(n_iter_no_change)
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
                        .device(parse_device(&device_s)?)
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
                        .n_iter_no_change(n_iter_no_change)
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
        self.device_used = Some(
            mlrs_backend::device::Device::resolved_name(host_ingress).to_string(),
        );
        self.inner = fitted;
        Ok(())
    }
    /// `device_` — the execution arm that actually ran (`"cpu"` / `"gpu"`).
    ///
    /// A preference the backend cannot honour is REPORTED here, not faked: the
    /// capability half of each gate still decides, and this names what carried
    /// the fit.
    fn device_used(&self) -> Option<String> {
        self.device_used.clone()
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
    fn intercept_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyMBSGDClassifier::F32(e) => Ok(e.intercepts(&pool)),
            _ => Err(not_fitted("mbsgd_classifier", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyMBSGDClassifier::F64(e) => Ok(e.intercepts(&pool)),
            _ => Err(not_fitted("mbsgd_classifier", "intercept_ (f64)")),
        }
    }
    /// Rows in `coef_`: `1` for a binary fit, `n_classes` for the one-vs-rest
    /// multiclass fit — sklearn's `coef_` shape rule. The shim reshapes the
    /// flat `coef_*` buffer with this.
    fn n_coef_rows(&self) -> PyResult<usize> {
        match &self.inner {
            AnyMBSGDClassifier::F32(e) => Ok(e.n_coef_rows()),
            AnyMBSGDClassifier::F64(e) => Ok(e.n_coef_rows()),
            _ => Err(not_fitted("mbsgd_classifier", "n_coef_rows")),
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

    /// Serialize the fitted model to `path` (MODEL-PERSIST).
    ///
    /// `extra` carries the Python shim's own state — `get_params()`, the class
    /// name, the fitted attributes the Rust estimator does not hold — merged
    /// into the file's `__metadata__` under a `py:` prefix. The shim supplies
    /// it; a caller using this `#[pyclass]` directly can pass an empty list and
    /// gets a plain mlrs model file.
    #[pyo3(signature = (path, extra = Vec::new()))]
    fn save(&self, py: Python<'_>, path: &str, extra: Vec<(String, String)>) -> PyResult<()> {
        crate::persist::save_impl(py, &self.inner, path, extra)
    }

    /// Replace this wrapper's fitted state with the model in `path`.
    ///
    /// An instance method rather than a constructor, mirroring `fit`: the
    /// wrapper keeps its hyperparameters beside `inner`, and the Python shim has
    /// already rebuilt them from the file's `py:` metadata before calling this.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner = crate::persist::load_impl(py, path)?;
        Ok(())
    }
}

// ===========================================================================
// MBSGDRegressor — Fit (TryFrom enums + builder().build()) + Predict (SGDSVM-02).
// ===========================================================================

/// sklearn-compatible `MBSGDRegressor` (minibatch SGD regressor).
#[pyclass(name = "MBSGDRegressor")]
pub struct PyMBSGDRegressor {
    inner: AnyMBSGDRegressor,
    /// `device_` — the arm that actually ran. Recorded at `fit` because the
    /// typestate transition consumes the `Unfit` variant the preference was
    /// read from, and a fitted estimator must still be able to name its arm.
    device_used: Option<String>,
}

impl PyMBSGDRegressor {
    /// Rust-callable default constructor (smoke test seam).
    pub fn unfit_default() -> Self {
        Self {
            device_used: None,
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
                n_iter_no_change: 5,
                device: "auto".to_string(),
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
    /// batch_size=1, shuffle=True, seed=0, n_iter_no_change=5)`.
    #[new]
    #[pyo3(signature = (
        loss = "squared_error".to_string(), penalty = "l2".to_string(), alpha = 1e-4,
        l1_ratio = 0.15, fit_intercept = true, max_iter = 1000, tol = 1e-3,
        learning_rate = "invscaling".to_string(), eta0 = 0.01, power_t = 0.25,
        epsilon = 0.1, batch_size = 1, shuffle = true, seed = 0,
        n_iter_no_change = 5,
        device = "auto".to_string(),
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
        n_iter_no_change: usize,
        device: String,
    ) -> Self {
        Self {
            device_used: None,
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
                n_iter_no_change,
                device,
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
            lr_s, eta0, power_t, epsilon, batch_size, shuffle, seed, n_iter_no_change,
            device_s,
        ) = match &self.inner {
            AnyMBSGDRegressor::Unfit {
                loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol,
                learning_rate, eta0, power_t, epsilon, batch_size, shuffle, seed,
                n_iter_no_change, device,
            } => (
                loss.clone(), penalty.clone(), *alpha, *l1_ratio, *fit_intercept,
                *max_iter, *tol, learning_rate.clone(), *eta0, *power_t, *epsilon,
                *batch_size, *shuffle, *seed, *n_iter_no_change, device.clone(),
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
                        .device(parse_device(&device_s)?)
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
                        .n_iter_no_change(n_iter_no_change)
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
                        .device(parse_device(&device_s)?)
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
                        .n_iter_no_change(n_iter_no_change)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, Some(&yd), (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyMBSGDRegressor::F64(fitted))
                }
            }
        })?;
        self.device_used = Some(
            mlrs_backend::device::Device::resolved_name(
                mlrs_backend::prims::sgd::sgd_host_available(parse_device(&device_s)?),
            )
            .to_string(),
        );
        self.inner = fitted;
        Ok(())
    }
    /// `device_` — the execution arm that actually ran (`"cpu"` / `"gpu"`).
    ///
    /// A preference the backend cannot honour is REPORTED here, not faked: the
    /// capability half of each gate still decides, and this names what carried
    /// the fit.
    fn device_used(&self) -> Option<String> {
        self.device_used.clone()
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

    /// Serialize the fitted model to `path` (MODEL-PERSIST).
    ///
    /// `extra` carries the Python shim's own state — `get_params()`, the class
    /// name, the fitted attributes the Rust estimator does not hold — merged
    /// into the file's `__metadata__` under a `py:` prefix. The shim supplies
    /// it; a caller using this `#[pyclass]` directly can pass an empty list and
    /// gets a plain mlrs model file.
    #[pyo3(signature = (path, extra = Vec::new()))]
    fn save(&self, py: Python<'_>, path: &str, extra: Vec<(String, String)>) -> PyResult<()> {
        crate::persist::save_impl(py, &self.inner, path, extra)
    }

    /// Replace this wrapper's fitted state with the model in `path`.
    ///
    /// An instance method rather than a constructor, mirroring `fit`: the
    /// wrapper keeps its hyperparameters beside `inner`, and the Python shim has
    /// already rebuilt them from the file's `py:` metadata before calling this.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner = crate::persist::load_impl(py, path)?;
        Ok(())
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
    ///
    /// **No upload on cpu** (SVM-FIT-CPU). Where the L-BFGS objective evaluates
    /// from host memory anyway ([`svm_host_ingress_preferred`]), the design is
    /// BORROWED from the Arrow values via [`host_slice_f32`] / [`host_slice_f64`]
    /// — the same hard-reject bridge validator `validated_f32` runs, minus the
    /// copy — and handed to `fit_from_host_slice`. Routing it through a
    /// `DeviceArray` instead costs three full passes over `x` before the first
    /// of ~30 evaluations (one to upload, two more because `to_host`
    /// materializes a byte buffer and then a typed one). Real device backends
    /// keep the upload and the two-GEMM device evaluator.
    ///
    /// [`svm_host_ingress_preferred`]: mlrs_backend::prims::svm_objective::svm_host_ingress_preferred
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
        let host_ingress =
            mlrs_backend::prims::svm_objective::svm_host_ingress_preferred();
        let fitted = py.detach(|| -> PyResult<AnyLinearSVC> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
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
                    Ok(AnyLinearSVC::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
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

    /// `decision_function(x)` → a **pyarrow** float array of `rows·K` values
    /// (row-major), `K = n_coef_rows`. Shares `predict_labels`' host ingress and
    /// its NaN/inf ownership: the shim passes `ensure_all_finite=False`, so the
    /// rejection is reproduced here via [`nonfinite_input_err`].
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let xa = capsule_to_array(x)?;
        enum Out {
            F32(Vec<f32>),
            F64(Vec<f64>),
        }
        let out = py.detach(|| -> PyResult<Out> {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyLinearSVC::F32(est) => {
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    let d = est
                        .decision_from_host(&mut pool, xh, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    if !d.operand_finite {
                        return Err(nonfinite_input_err(xh, "float32"));
                    }
                    Ok(Out::F32(d.values))
                }
                AnyLinearSVC::F64(est) => {
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    let d = est
                        .decision_from_host(&mut pool, xh, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    if !d.operand_finite {
                        return Err(nonfinite_input_err(xh, "float64"));
                    }
                    Ok(Out::F64(d.values))
                }
                _ => Err(not_fitted("linear_svc", "decision_function")),
            }
        })?;
        match out {
            Out::F32(v) => f32_vec_to_pyarrow(py, v),
            Out::F64(v) => f64_vec_to_pyarrow(py, v),
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
    fn intercept_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearSVC::F32(e) => Ok(e.intercepts(&pool)),
            _ => Err(not_fitted("linear_svc", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyLinearSVC::F64(e) => Ok(e.intercepts(&pool)),
            _ => Err(not_fitted("linear_svc", "intercept_ (f64)")),
        }
    }

    /// Rows in `coef_`: `1` for a binary fit, `n_classes` for the one-vs-rest
    /// multiclass fit — sklearn's `coef_` shape rule. The shim reshapes the flat
    /// `coef_*` buffer with this.
    fn n_coef_rows(&self) -> PyResult<usize> {
        match &self.inner {
            AnyLinearSVC::F32(e) => Ok(e.n_coef_rows()),
            AnyLinearSVC::F64(e) => Ok(e.n_coef_rows()),
            _ => Err(not_fitted("linear_svc", "n_coef_rows")),
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

    /// Serialize the fitted model to `path` (MODEL-PERSIST).
    ///
    /// `extra` carries the Python shim's own state — `get_params()`, the class
    /// name, the fitted attributes the Rust estimator does not hold — merged
    /// into the file's `__metadata__` under a `py:` prefix. The shim supplies
    /// it; a caller using this `#[pyclass]` directly can pass an empty list and
    /// gets a plain mlrs model file.
    #[pyo3(signature = (path, extra = Vec::new()))]
    fn save(&self, py: Python<'_>, path: &str, extra: Vec<(String, String)>) -> PyResult<()> {
        crate::persist::save_impl(py, &self.inner, path, extra)
    }

    /// Replace this wrapper's fitted state with the model in `path`.
    ///
    /// An instance method rather than a constructor, mirroring `fit`: the
    /// wrapper keeps its hyperparameters beside `inner`, and the Python shim has
    /// already rebuilt them from the file's `py:` metadata before calling this.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner = crate::persist::load_impl(py, path)?;
        Ok(())
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
    ///
    /// **No upload on cpu** — the `PyLinearSVC::fit` note applies verbatim; both
    /// SVMs share one objective and one solver.
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
        let host_ingress =
            mlrs_backend::prims::svm_objective::svm_host_ingress_preferred();
        let fitted = py.detach(|| -> PyResult<AnyLinearSVR> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
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
                    Ok(AnyLinearSVR::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
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

    /// Serialize the fitted model to `path` (MODEL-PERSIST).
    ///
    /// `extra` carries the Python shim's own state — `get_params()`, the class
    /// name, the fitted attributes the Rust estimator does not hold — merged
    /// into the file's `__metadata__` under a `py:` prefix. The shim supplies
    /// it; a caller using this `#[pyclass]` directly can pass an empty list and
    /// gets a plain mlrs model file.
    #[pyo3(signature = (path, extra = Vec::new()))]
    fn save(&self, py: Python<'_>, path: &str, extra: Vec<(String, String)>) -> PyResult<()> {
        crate::persist::save_impl(py, &self.inner, path, extra)
    }

    /// Replace this wrapper's fitted state with the model in `path`.
    ///
    /// An instance method rather than a constructor, mirroring `fit`: the
    /// wrapper keeps its hyperparameters beside `inner`, and the Python shim has
    /// already rebuilt them from the file's `py:` metadata before calling this.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner = crate::persist::load_impl(py, path)?;
        Ok(())
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
        device: String,
    },
}

crate::impl_persistable_any! {
    any:  AnyBayesianRidge,
    algo: mlrs_algos::linear::bayesian_ridge::BayesianRidge,
    name: "bayesian_ridge",
}

/// The verbatim ctor hyperparameters, carried from the `Unfit` arm to `fit`
/// (WR-02: the wrapper rebuilds from these at every `fit`, so a second `fit` of
/// the same object works).
#[derive(Clone)]
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
    /// DEVICE-PARAM-01, a STRING until `fit` (the D-09 parse-at-build rule).
    device: String,
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
                device: "auto".to_string(),
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
                device,
            } => BayesianRidgeParams {
                device: device.clone(),
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
                device: "auto".to_string(),
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
            .device(parse_device($p.device.as_str())?)
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
        device = "auto".to_string(),
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
        device: String,
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
                device,
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
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyBayesianRidge::F32(est) => est
                    .predict_std_from_host(&mut pool, xh, (rows, cols))
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
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyBayesianRidge::F64(est) => est
                    .predict_std_from_host(&mut pool, xh, (rows, cols))
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
    /// `device_` — the execution arm that actually ran (`"cpu"` / `"gpu"`).
    ///
    /// Read off the FITTED estimator rather than mirrored at `fit`: the arm is
    /// already recorded on it, and re-deriving it here would be a second place
    /// to get the fallbacks wrong.
    fn device_used(&self) -> Option<String> {
        match &self.inner {
            AnyBayesianRidge::F32(e) => Some(e.device().to_string()),
            AnyBayesianRidge::F64(e) => Some(e.device().to_string()),
            _ => None,
        }
    }

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

    /// Serialize the fitted model to `path` (MODEL-PERSIST).
    ///
    /// `extra` carries the Python shim's own state — `get_params()`, the class
    /// name, the fitted attributes the Rust estimator does not hold — merged
    /// into the file's `__metadata__` under a `py:` prefix. The shim supplies
    /// it; a caller using this `#[pyclass]` directly can pass an empty list and
    /// gets a plain mlrs model file.
    #[pyo3(signature = (path, extra = Vec::new()))]
    fn save(&self, py: Python<'_>, path: &str, extra: Vec<(String, String)>) -> PyResult<()> {
        crate::persist::save_impl(py, &self.inner, path, extra)
    }

    /// Replace this wrapper's fitted state with the model in `path`.
    ///
    /// An instance method rather than a constructor, mirroring `fit`: the
    /// wrapper keeps its hyperparameters beside `inner`, and the Python shim has
    /// already rebuilt them from the file's `py:` metadata before calling this.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner = crate::persist::load_impl(py, path)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HuberRegressor — Fit + Predict; the FULL sklearn parameter surface
//   epsilon, max_iter, alpha, warm_start, fit_intercept, tol
//   + fit(..., sample_weight) and the coef_ / intercept_ / scale_ / n_iter_ /
//     outliers_ fitted attributes
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyHuberRegressor,
    algo:  mlrs_algos::linear::huber::HuberRegressor,
    unfit: {
        epsilon: f64,
        max_iter: usize,
        alpha: f64,
        warm_start: bool,
        fit_intercept: bool,
        tol: f64,
    },
}

crate::impl_persistable_any! {
    any:  AnyHuberRegressor,
    algo: mlrs_algos::linear::huber::HuberRegressor,
    name: "huber",
}

/// The verbatim ctor hyperparameters, carried from the `Unfit` arm to `fit`
/// (WR-02: the wrapper rebuilds from these at every `fit`, so a second `fit` of
/// the same object works).
#[derive(Clone)]
struct HuberParams {
    /// DEVICE-PARAM-01, a STRING until `build` (D-09).
    device: String,
    epsilon: f64,
    max_iter: usize,
    alpha: f64,
    warm_start: bool,
    fit_intercept: bool,
    tol: f64,
}

/// sklearn-compatible `HuberRegressor` (robust L2-regularized linear regression).
#[pyclass(name = "HuberRegressor")]
pub struct PyHuberRegressor {
    inner: AnyHuberRegressor,
    /// `device_` — the arm that actually ran. Recorded at `fit` because the
    /// typestate transition consumes the `Unfit` variant the preference was
    /// read from, and a fitted estimator must still be able to name its arm.
    device_used: Option<String>,

    /// The ctor hyperparameters, kept OUTSIDE `inner` because a fitted arm no
    /// longer carries an `Unfit { .. }` to read them from and `warm_start`
    /// makes a second `fit` of the same object a supported operation.
    params: HuberParams,
    /// Scalar/vector fitted attributes mirrored out of the consumed `Fitted`
    /// arms at `fit`: a `#[pyclass]` getter cannot reach through the dtype
    /// dispatch generically, and all of these are `f64`/`bool` on both arms
    /// anyway (the solve runs in `f64` whatever the design's width).
    scale: Option<f64>,
    n_iter: Option<usize>,
    converged: bool,
    outliers: Vec<bool>,
    /// The packed `[coef…, intercept?, σ]` a `warm_start` refit seeds from —
    /// sklearn's `np.concatenate((self.coef_, [self.intercept_, self.scale_]))`.
    /// Held HERE rather than read back from `inner` so the seed survives the
    /// typestate `fit` consuming the previous estimator.
    warm_params: Vec<f64>,
}

impl PyHuberRegressor {
    /// Rust-callable default constructor (smoke test seam).
    pub fn unfit_default() -> Self {
        Self::new(1.35, 100, 1e-4, false, true, 1e-5, "auto".to_string())
    }

    /// Is this wrapper in the unfit arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyHuberRegressor::Unfit { .. })
    }
}

/// Build an unfit `HuberRegressor<F>` from the ctor params, seeding the
/// warm-start vector when there is one. Monomorphized per float width by the
/// macro so the six builder setters are written once.
macro_rules! huber_build {
    ($float:ty, $p:expr, $seed:expr) => {{
        let device = parse_device(&$p.device)?;
        let mut b = HuberRegressor::<$float>::builder()
            .device(device)
            .epsilon($p.epsilon)
            .max_iter($p.max_iter)
            .alpha($p.alpha)
            .warm_start($p.warm_start)
            .fit_intercept($p.fit_intercept)
            .tol($p.tol);
        // sklearn seeds the next fit only when `warm_start` is set AND the
        // estimator already has a `coef_`; an empty seed is the cold start.
        if $p.warm_start && !$seed.is_empty() {
            b = b.init_params($seed.clone());
        }
        b.build::<$float>().map_err(build_err_to_py)?
    }};
}

/// Build the estimator and run whichever `fit` ingress the backend calls for.
///
/// The branch has to happen HERE, before ingress, because the two entry points
/// take different operand types: `fit_from_host_slice` borrows the Arrow values
/// directly and `fit_with_sample_weight` needs a device upload. On the cpu
/// backend — where the objective evaluates from host memory anyway — the `n·d`
/// design is therefore never copied at all, which removes THREE full passes over
/// it from every fit (`from_host` copies twice, `to_host` once) and, unlike a
/// one-shot `predict`, that saving is not paid once but avoided on a solve that
/// re-reads the design a few dozen times.
macro_rules! huber_fit_dispatch {
    ($float:ty, $p:expr, $seed:expr, $xa:expr, $ya:expr, $swa:expr, $rows:expr, $cols:expr,
     $pool:expr, $as:ident, $host_slice:ident, $validated:ident) => {{
        let est = huber_build!($float, $p, $seed);
        let sw = match $swa.as_ref() {
            Some(a) => Some($host_slice($as(a)?)?),
            None => None,
        };
        if mlrs_backend::prims::huber_objective::huber_host_ingress_preferred(
            $rows,
            $cols,
            parse_device(&$p.device)?,
        ) {
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
macro_rules! huber_snapshot {
    ($fitted:expr) => {{
        (
            $fitted.scale(),
            $fitted.n_iter(),
            $fitted.converged(),
            $fitted.outliers().to_vec(),
            $fitted.warm_start_params().to_vec(),
            $fitted.device_arm(),
        )
    }};
}

#[pymethods]
impl PyHuberRegressor {
    /// `HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001,
    /// warm_start=False, fit_intercept=True, tol=1e-05)` — sklearn's signature
    /// one-for-one. Every parameter is a float, an int or a bool; there is no
    /// string-valued parameter on this estimator.
    #[new]
    #[pyo3(signature = (
        epsilon = 1.35,
        max_iter = 100,
        alpha = 1e-4,
        warm_start = false,
        fit_intercept = true,
        tol = 1e-5,
        device = "auto".to_string(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        epsilon: f64,
        max_iter: usize,
        alpha: f64,
        warm_start: bool,
        fit_intercept: bool,
        tol: f64,
        device: String,
    ) -> Self {
        Self {
            device_used: None,
            inner: AnyHuberRegressor::Unfit {
                epsilon,
                max_iter,
                alpha,
                warm_start,
                fit_intercept,
                tol,
            },
            params: HuberParams {
                device,
                epsilon,
                max_iter,
                alpha,
                warm_start,
                fit_intercept,
                tol,
            },
            scale: None,
            n_iter: None,
            converged: false,
            outliers: Vec::new(),
            warm_params: Vec::new(),
        }
    }

    /// `fit(X, y, rows, cols, sample_weight=None)`.
    ///
    /// `sample_weight` is an optional length-`rows` Arrow float array in the SAME
    /// dtype as `X` — borrowed as a host slice (never uploaded), because the
    /// objective reads it from the host on every backend.
    ///
    /// With `warm_start=True` a second `fit` on the same object seeds from the
    /// first's `[coef_, intercept_, scale_]`, exactly as sklearn's does. The seed
    /// lives on the wrapper because the Rust `fit` CONSUMES the estimator.
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
        let p = self.params.clone();
        let seed = self.warm_params.clone();
        type Snapshot = (f64, usize, bool, Vec<bool>, Vec<f64>, Option<&'static str>);
        let (fitted, snap) = py.detach(|| -> PyResult<(AnyHuberRegressor, Snapshot)> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let fitted = huber_fit_dispatch!(
                        f32, p, seed, xa, ya, swa, rows, cols, pool,
                        as_f32, host_slice_f32, validated_f32
                    );
                    let snap = huber_snapshot!(fitted);
                    Ok((AnyHuberRegressor::F32(fitted), snap))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let fitted = huber_fit_dispatch!(
                        f64, p, seed, xa, ya, swa, rows, cols, pool,
                        as_f64, host_slice_f64, validated_f64
                    );
                    let snap = huber_snapshot!(fitted);
                    Ok((AnyHuberRegressor::F64(fitted), snap))
                }
            }
        })?;
        self.inner = fitted;
        let (scale, n_iter, converged, outliers, warm_params, device_arm) = snap;
        self.device_used = device_arm.map(str::to_string);
        self.scale = Some(scale);
        self.n_iter = Some(n_iter);
        self.converged = converged;
        self.outliers = outliers;
        self.warm_params = warm_params;
        Ok(())
    }
    /// `device_` — the execution arm that actually ran (`"cpu"` / `"gpu"`).
    ///
    /// A preference the backend cannot honour is REPORTED here, not faked: the
    /// capability half of each gate still decides, and this names what carried
    /// the fit.
    fn device_used(&self) -> Option<String> {
        self.device_used.clone()
    }


    /// `predict(x)` → a length-`rows` **pyarrow** float array (D-03). Shares
    /// [`dense_predict_f32`] with the other dense linear regressors.
    fn predict_f32<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnyHuberRegressor::F32(est) => dense_predict_f32(py, x, (rows, cols), est),
            _ => Err(not_fitted("huber_regressor", "predict (f32 path)")),
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
            AnyHuberRegressor::F64(est) => dense_predict_f64(py, x, (rows, cols), est),
            _ => Err(not_fitted("huber_regressor", "predict (f64 path)")),
        }
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHuberRegressor::F32(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("huber_regressor", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHuberRegressor::F64(e) => Ok(e.coef(&pool)),
            _ => Err(not_fitted("huber_regressor", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHuberRegressor::F32(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("huber_regressor", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHuberRegressor::F64(e) => Ok(e.intercept(&pool)),
            _ => Err(not_fitted("huber_regressor", "intercept_ (f64)")),
        }
    }

    /// sklearn's `scale_` — the fitted `σ`. Always `f64`: the joint `(w, σ)`
    /// iteration accumulates in `f64` whatever the design's width.
    fn scale(&self) -> Option<f64> {
        self.scale
    }
    /// sklearn's `n_iter_` — L-BFGS iterations, capped at `max_iter`.
    fn n_iter(&self) -> Option<usize> {
        self.n_iter
    }
    /// Whether the solve met its stopping criterion inside `max_iter`. The shim
    /// turns a `False` here into sklearn's `ConvergenceWarning`, which is what
    /// sklearn itself raises (it does NOT error on a hit cap).
    fn converged(&self) -> bool {
        self.converged
    }
    /// sklearn's `outliers_` — the boolean mask
    /// `|yᵢ − Xᵢ·coef_ − intercept_| > scale_·epsilon` over the TRAINING rows.
    fn outliers(&self) -> Vec<bool> {
        self.outliers.clone()
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyHuberRegressor::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyHuberRegressor::Unfit { .. } => None,
            AnyHuberRegressor::F32(_) => Some("f32"),
            AnyHuberRegressor::F64(_) => Some("f64"),
        }
    }

    /// Serialize the fitted model to `path` (MODEL-PERSIST).
    ///
    /// `extra` carries the Python shim's own state — `get_params()`, the class
    /// name, the fitted attributes the Rust estimator does not hold — merged
    /// into the file's `__metadata__` under a `py:` prefix. The shim supplies
    /// it; a caller using this `#[pyclass]` directly can pass an empty list and
    /// gets a plain mlrs model file.
    #[pyo3(signature = (path, extra = Vec::new()))]
    fn save(&self, py: Python<'_>, path: &str, extra: Vec<(String, String)>) -> PyResult<()> {
        crate::persist::save_impl(py, &self.inner, path, extra)
    }

    /// Replace this wrapper's fitted state with the model in `path`.
    ///
    /// An instance method rather than a constructor, mirroring `fit`: the
    /// wrapper keeps its hyperparameters beside `inner`, and the Python shim has
    /// already rebuilt them from the file's `py:` metadata before calling this.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.inner = crate::persist::load_impl(py, path)?;
        Ok(())
    }
}
