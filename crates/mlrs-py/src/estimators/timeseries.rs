//! Time-series `#[pyclass]` wrappers (TSA-01, Phase 22): `PyArima` /
//! `PyAutoArima`.
//!
//! `Arima` is HOST-side (no per-dtype device compute loop — see
//! `mlrs_algos::timeseries::arima` module docs for the full scope: zero-mean
//! only, no seasonal component), so these wrappers are simpler than the
//! typestate `any_estimator_typestate!` family: one `Unfit`/`F32`/`F64` enum
//! by hand, `fit` builds + consumes the `Unfit` estimator via
//! `mlrs_algos::timeseries::Arima::fit`, and every fitted accessor reads the
//! host-resident state directly (no `BufferPool` needed post-fit).
//!
//! Tests live in `crates/mlrs-py/tests/` (AGENTS.md §2).

use pyo3::prelude::*;

use mlrs_algos::timeseries::{Arima, AutoArima};

use crate::errors::{algo_err_to_py, build_err_to_py, not_fitted};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype};

/// Dtype-dispatched fitted/unfit ARIMA.
enum AnyArima {
    Unfit { p: usize, d: usize, q: usize },
    F32(Arima<f32, mlrs_algos::typestate::Fitted>),
    F64(Arima<f64, mlrs_algos::typestate::Fitted>),
}

/// sklearn/cuML-adjacent `ARIMA(p, d, q)` (zero-mean; see the Rust
/// `timeseries::arima` module docs for the full scope statement — no
/// trend/constant, no seasonal component).
#[pyclass(name = "ARIMA")]
pub struct PyArima {
    inner: AnyArima,
}

#[pymethods]
impl PyArima {
    /// `ARIMA(order=(0, 0, 0))` — sklearn/statsmodels-named `order` tuple.
    #[new]
    #[pyo3(signature = (order = (0usize, 0usize, 0usize)))]
    fn new(order: (usize, usize, usize)) -> PyResult<Self> {
        let (p, d, q) = order;
        // Data-independent order bound validated AT CONSTRUCTION (sklearn
        // parity: statsmodels also validates order eagerly).
        Arima::<f32>::builder().order(p, d, q).build::<f32>().map_err(build_err_to_py)?;
        Ok(Self { inner: AnyArima::Unfit { p, d, q } })
    }

    /// Fit to a single series `y` (Arrow capsule, length `n_obs`).
    fn fit(&mut self, py: Python<'_>, y: &Bound<'_, PyAny>, n_obs: usize) -> PyResult<()> {
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&ya)?;
        let (p, d, q) = match &self.inner {
            AnyArima::Unfit { p, d, q } => (*p, *d, *q),
            _ => return Err(not_fitted("arima", "re-fit")),
        };
        let fitted = py.detach(|| -> PyResult<AnyArima> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = Arima::<f32>::builder().order(p, d, q).build::<f32>().map_err(build_err_to_py)?;
                    let fitted = est.fit(&pool, &yd, n_obs).map_err(algo_err_to_py)?;
                    Ok(AnyArima::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = Arima::<f64>::builder().order(p, d, q).build::<f64>().map_err(build_err_to_py)?;
                    let fitted = est.fit(&pool, &yd, n_obs).map_err(algo_err_to_py)?;
                    Ok(AnyArima::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// Fitted AR coefficients.
    fn ar(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.ar().to_vec()),
            AnyArima::F64(e) => Ok(e.ar().to_vec()),
            AnyArima::Unfit { .. } => Err(not_fitted("arima", "ar")),
        }
    }
    /// Fitted MA coefficients.
    fn ma(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.ma().to_vec()),
            AnyArima::F64(e) => Ok(e.ma().to_vec()),
            AnyArima::Unfit { .. } => Err(not_fitted("arima", "ma")),
        }
    }
    /// Fitted concentrated innovation variance.
    fn sigma2(&self) -> PyResult<f64> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.sigma2()),
            AnyArima::F64(e) => Ok(e.sigma2()),
            AnyArima::Unfit { .. } => Err(not_fitted("arima", "sigma2")),
        }
    }
    /// Log-likelihood at the fitted parameters.
    fn loglik(&self) -> PyResult<f64> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.loglik()),
            AnyArima::F64(e) => Ok(e.loglik()),
            AnyArima::Unfit { .. } => Err(not_fitted("arima", "loglik")),
        }
    }
    /// Akaike information criterion.
    fn aic(&self) -> PyResult<f64> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.aic()),
            AnyArima::F64(e) => Ok(e.aic()),
            AnyArima::Unfit { .. } => Err(not_fitted("arima", "aic")),
        }
    }
    /// Corrected AIC.
    fn aicc(&self) -> PyResult<f64> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.aicc()),
            AnyArima::F64(e) => Ok(e.aicc()),
            AnyArima::Unfit { .. } => Err(not_fitted("arima", "aicc")),
        }
    }
    /// Bayesian information criterion.
    fn bic(&self) -> PyResult<f64> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.bic()),
            AnyArima::F64(e) => Ok(e.bic()),
            AnyArima::Unfit { .. } => Err(not_fitted("arima", "bic")),
        }
    }
    /// Whether the L-BFGS MLE reported `gtol` convergence.
    fn converged(&self) -> PyResult<bool> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.converged()),
            AnyArima::F64(e) => Ok(e.converged()),
            AnyArima::Unfit { .. } => Err(not_fitted("arima", "converged")),
        }
    }
    /// Forecast `n_periods` steps ahead (original series scale).
    fn forecast(&self, n_periods: usize) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.forecast(n_periods)),
            AnyArima::F64(e) => Ok(e.forecast(n_periods)),
            AnyArima::Unfit { .. } => Err(not_fitted("arima", "forecast")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyArima::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyArima::Unfit { .. } => None,
            AnyArima::F32(_) => Some("f32"),
            AnyArima::F64(_) => Some("f64"),
        }
    }
}

/// `AutoARIMA` (TSA-01): bounded `(p, q)` grid search over AICc at a FIXED
/// `d` — see the Rust `AutoArima` docs for the exhaustive-grid-vs-stepwise
/// scope note. Fits and returns a [`PyArima`]-shaped result directly (no
/// separate unfit state — the search itself IS the fit).
#[pyclass(name = "AutoARIMA")]
pub struct PyAutoArima {
    inner: AnyArima,
}

#[pymethods]
impl PyAutoArima {
    #[new]
    fn new() -> Self {
        Self { inner: AnyArima::Unfit { p: 0, d: 0, q: 0 } }
    }

    /// Search `p ∈ 0..=max_p`, `q ∈ 0..=max_q` at `d`, on series `y`.
    #[pyo3(signature = (y, n_obs, d, max_p=5, max_q=5))]
    fn fit(
        &mut self,
        py: Python<'_>,
        y: &Bound<'_, PyAny>,
        n_obs: usize,
        d: usize,
        max_p: usize,
        max_q: usize,
    ) -> PyResult<()> {
        let ya = capsule_to_array(y)?;
        let dt = float_dtype(&ya)?;
        let fitted = py.detach(|| -> PyResult<AnyArima> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let yd = validated_f32(as_f32(&ya)?, &mut pool)?;
                    let est = AutoArima::search::<f32>(&pool, &yd, n_obs, d, max_p, max_q)
                        .map_err(algo_err_to_py)?;
                    Ok(AnyArima::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let yd = validated_f64(as_f64(&ya)?, &mut pool)?;
                    let est = AutoArima::search::<f64>(&pool, &yd, n_obs, d, max_p, max_q)
                        .map_err(algo_err_to_py)?;
                    Ok(AnyArima::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn ar(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.ar().to_vec()),
            AnyArima::F64(e) => Ok(e.ar().to_vec()),
            AnyArima::Unfit { .. } => Err(not_fitted("auto_arima", "ar")),
        }
    }
    fn ma(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.ma().to_vec()),
            AnyArima::F64(e) => Ok(e.ma().to_vec()),
            AnyArima::Unfit { .. } => Err(not_fitted("auto_arima", "ma")),
        }
    }
    /// The `(p, d, q)` order AutoARIMA selected.
    fn order(&self) -> PyResult<(usize, usize, usize)> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.order()),
            AnyArima::F64(e) => Ok(e.order()),
            AnyArima::Unfit { .. } => Err(not_fitted("auto_arima", "order")),
        }
    }
    fn aicc(&self) -> PyResult<f64> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.aicc()),
            AnyArima::F64(e) => Ok(e.aicc()),
            AnyArima::Unfit { .. } => Err(not_fitted("auto_arima", "aicc")),
        }
    }
    fn loglik(&self) -> PyResult<f64> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.loglik()),
            AnyArima::F64(e) => Ok(e.loglik()),
            AnyArima::Unfit { .. } => Err(not_fitted("auto_arima", "loglik")),
        }
    }
    fn forecast(&self, n_periods: usize) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyArima::F32(e) => Ok(e.forecast(n_periods)),
            AnyArima::F64(e) => Ok(e.forecast(n_periods)),
            AnyArima::Unfit { .. } => Err(not_fitted("auto_arima", "forecast")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyArima::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyArima::Unfit { .. } => None,
            AnyArima::F32(_) => Some("f32"),
            AnyArima::F64(_) => Some("f64"),
        }
    }
}
