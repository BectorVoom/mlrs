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

use mlrs_algos::linear::elastic_net::ElasticNet;
use mlrs_algos::linear::lasso::Lasso;
use mlrs_algos::linear::linear_regression::LinearRegression;
use mlrs_algos::linear::logistic::LogisticRegression;
use mlrs_algos::linear::ridge::Ridge;
use mlrs_algos::traits::{Fit, Predict, PredictLabels, PredictProba};

use crate::errors::{algo_err_to_py, not_fitted};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype};

// ---------------------------------------------------------------------------
// LinearRegression — Fit + Predict; coef_ / intercept_
// ---------------------------------------------------------------------------

crate::any_estimator! {
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
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let mut est = LinearRegression::<f32>::new(fit_intercept);
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyLinearRegression::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let mut est = LinearRegression::<f64>::new(fit_intercept);
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyLinearRegression::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x)` → length-`rows` host `Vec<f32|f64>` (D-03). GIL released.
    fn predict_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| -> PyResult<Vec<f32>> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyLinearRegression::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let out = est.predict(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(out.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("linear_regression", "predict (f32 path)")),
            }
        })
    }

    fn predict_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| -> PyResult<Vec<f64>> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyLinearRegression::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let out = est.predict(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(out.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("linear_regression", "predict (f64 path)")),
            }
        })
    }

    /// Host `coef_` (f32 arm) or `NotFitted`.
    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLinearRegression::F32(e) => e.coef(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("linear_regression", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLinearRegression::F64(e) => e.coef(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("linear_regression", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLinearRegression::F32(e) => e.intercept(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("linear_regression", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLinearRegression::F64(e) => e.intercept(&pool).map_err(algo_err_to_py),
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
// Ridge — Fit + Predict; alpha, fit_intercept
// ---------------------------------------------------------------------------

crate::any_estimator! {
    any:   AnyRidge,
    algo:  mlrs_algos::linear::ridge::Ridge,
    unfit: { alpha: f64, fit_intercept: bool },
}

/// sklearn-compatible `Ridge` (L2-penalized least squares).
#[pyclass(name = "Ridge")]
pub struct PyRidge {
    inner: AnyRidge,
}

impl PyRidge {
    /// Rust-callable default constructor for the smoke test. See
    /// [`PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self { inner: AnyRidge::Unfit { alpha: 1.0, fit_intercept: true } }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyRidge::Unfit { .. })
    }
}

#[pymethods]
impl PyRidge {
    /// `Ridge(alpha=1.0, fit_intercept=True)`.
    #[new]
    #[pyo3(signature = (alpha = 1.0, fit_intercept = true))]
    fn new(alpha: f64, fit_intercept: bool) -> Self {
        Self {
            inner: AnyRidge::Unfit { alpha, fit_intercept },
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
        let (alpha, fit_intercept) = match &self.inner {
            AnyRidge::Unfit { alpha, fit_intercept } => (*alpha, *fit_intercept),
            _ => (1.0, true),
        };
        let fitted = py.detach(|| -> PyResult<AnyRidge> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let mut est = Ridge::<f32>::new(alpha as f32, fit_intercept);
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyRidge::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let mut est = Ridge::<f64>::new(alpha, fit_intercept);
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyRidge::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn predict_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyRidge::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(est.predict(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("ridge", "predict (f32 path)")),
            }
        })
    }
    fn predict_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyRidge::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(est.predict(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("ridge", "predict (f64 path)")),
            }
        })
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyRidge::F32(e) => e.coef(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("ridge", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyRidge::F64(e) => e.coef(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("ridge", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyRidge::F32(e) => e.intercept(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("ridge", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyRidge::F64(e) => e.intercept(&pool).map_err(algo_err_to_py),
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

crate::any_estimator! {
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
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let mut est = Lasso::<f32>::with_opts(alpha as f32, fit_intercept, max_iter, tol);
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyLasso::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let mut est = Lasso::<f64>::with_opts(alpha, fit_intercept, max_iter, tol);
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyLasso::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn predict_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyLasso::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(est.predict(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("lasso", "predict (f32 path)")),
            }
        })
    }
    fn predict_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyLasso::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(est.predict(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("lasso", "predict (f64 path)")),
            }
        })
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLasso::F32(e) => e.coef(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("lasso", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLasso::F64(e) => e.coef(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("lasso", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLasso::F32(e) => e.intercept(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("lasso", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLasso::F64(e) => e.intercept(&pool).map_err(algo_err_to_py),
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

crate::any_estimator! {
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
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let mut est = ElasticNet::<f32>::with_opts(alpha as f32, l1_ratio as f32, fit_intercept, max_iter, tol);
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyElasticNet::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let mut est = ElasticNet::<f64>::with_opts(alpha, l1_ratio, fit_intercept, max_iter, tol);
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyElasticNet::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn predict_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyElasticNet::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(est.predict(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("elastic_net", "predict (f32 path)")),
            }
        })
    }
    fn predict_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyElasticNet::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(est.predict(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("elastic_net", "predict (f64 path)")),
            }
        })
    }

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyElasticNet::F32(e) => e.coef(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("elastic_net", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyElasticNet::F64(e) => e.coef(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("elastic_net", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<f32> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyElasticNet::F32(e) => e.intercept(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("elastic_net", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<f64> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyElasticNet::F64(e) => e.intercept(&pool).map_err(algo_err_to_py),
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

crate::any_estimator! {
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
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let mut est = LogisticRegression::<f32>::with_opts(c as f32, fit_intercept, max_iter, tol as f32);
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyLogisticRegression::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let mut est = LogisticRegression::<f64>::with_opts(c, fit_intercept, max_iter, tol);
                    est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyLogisticRegression::F64(est))
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
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyLogisticRegression::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(est.predict_labels(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                AnyLogisticRegression::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(est.predict_labels(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("logistic_regression", "predict")),
            }
        })
    }

    /// `predict_proba(x)` → row-major `rows × n_classes` host floats.
    fn predict_proba_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyLogisticRegression::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(est.predict_proba(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("logistic_regression", "predict_proba (f32 path)")),
            }
        })
    }
    fn predict_proba_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyLogisticRegression::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(est.predict_proba(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
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

    fn coef_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLogisticRegression::F32(e) => e.coef(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("logistic_regression", "coef_ (f32)")),
        }
    }
    fn coef_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLogisticRegression::F64(e) => e.coef(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("logistic_regression", "coef_ (f64)")),
        }
    }
    fn intercept_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLogisticRegression::F32(e) => e.intercept(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("logistic_regression", "intercept_ (f32)")),
        }
    }
    fn intercept_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLogisticRegression::F64(e) => e.intercept(&pool).map_err(algo_err_to_py),
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

crate::any_estimator! {
    any:   AnyMBSGDClassifier,
    algo:  mlrs_algos::linear::mbsgd_classifier::MBSGDClassifier,
    unfit: {
        loss: String, penalty: String, alpha: f64, l1_ratio: f64,
        fit_intercept: bool, max_iter: usize, tol: f64,
        learning_rate: String, eta0: f64, power_t: f64,
        batch_size: usize, shuffle: bool, seed: u64,
    },
}

crate::any_estimator! {
    any:   AnyMBSGDRegressor,
    algo:  mlrs_algos::linear::mbsgd_regressor::MBSGDRegressor,
    unfit: {
        loss: String, penalty: String, alpha: f64, l1_ratio: f64,
        fit_intercept: bool, max_iter: usize, tol: f64,
        learning_rate: String, eta0: f64, power_t: f64, epsilon: f64,
        batch_size: usize, shuffle: bool, seed: u64,
    },
}

crate::any_estimator! {
    any:   AnyLinearSVC,
    algo:  mlrs_algos::linear::linear_svc::LinearSVC,
    unfit: {
        loss: String, penalty: String, c: f64, intercept_scaling: f64,
        fit_intercept: bool, max_iter: usize, tol: f64,
    },
}

crate::any_estimator! {
    any:   AnyLinearSVR,
    algo:  mlrs_algos::linear::linear_svr::LinearSVR,
    unfit: {
        loss: String, penalty: String, c: f64, epsilon: f64,
        intercept_scaling: f64, fit_intercept: bool, max_iter: usize, tol: f64,
    },
}

/// Wave-3 promotion seam: construct the unfit `MBSGDClassifier` dispatch arm
/// from sklearn defaults (Wave 3 turns this into the `#[pyclass]` `#[new]`).
#[allow(dead_code)]
fn unfit_default_mbsgd_classifier() -> AnyMBSGDClassifier {
    AnyMBSGDClassifier::Unfit {
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
    }
}

/// Wave-3 promotion seam for `MBSGDRegressor`.
#[allow(dead_code)]
fn unfit_default_mbsgd_regressor() -> AnyMBSGDRegressor {
    AnyMBSGDRegressor::Unfit {
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
    }
}

/// Wave-3 promotion seam for `LinearSVC`.
#[allow(dead_code)]
fn unfit_default_linear_svc() -> AnyLinearSVC {
    AnyLinearSVC::Unfit {
        loss: "squared_hinge".to_string(),
        penalty: "l2".to_string(),
        c: 1.0,
        intercept_scaling: 1.0,
        fit_intercept: true,
        max_iter: 1000,
        tol: 1e-4,
    }
}

/// Wave-3 promotion seam for `LinearSVR`.
#[allow(dead_code)]
fn unfit_default_linear_svr() -> AnyLinearSVR {
    AnyLinearSVR::Unfit {
        loss: "squared_epsilon_insensitive".to_string(),
        penalty: "l2".to_string(),
        c: 1.0,
        epsilon: 0.0,
        intercept_scaling: 1.0,
        fit_intercept: true,
        max_iter: 1000,
        tol: 1e-4,
    }
}
