//! Covariance `#[pyclass]` wrappers (COV-01/COV-02): `PyEmpiricalCovariance`,
//! `PyLedoitWolf`.
//!
//! Both are `Fit` (unsupervised — `y = None`) surfaces over the `mlrs_algos`
//! covariance estimators, dtype-dispatched (D-06) through the macro-emitted
//! `Any<Name>` enum. Each device-compute body honors the two load-bearing
//! contracts documented on [`crate::dispatch`]: GIL release via `py.detach`
//! (PY-03) and [`crate::capability::guard_f64`]`()?` BEFORE the F64 arm (D-04).
//!
//! Fitted-attribute accessors are dtype-suffixed (`covariance_f32`/`_f64`,
//! `location_*`, `precision_*`); `shrinkage_` is a single `f64` scalar so it is
//! single-typed. The pure-Python shim (Plan 04) maps the sklearn-named attrs to
//! these.

use pyo3::prelude::*;

use mlrs_algos::covariance::empirical_covariance::EmpiricalCovariance;
use mlrs_algos::covariance::ledoit_wolf::LedoitWolf;
use mlrs_algos::typestate::Fit as TypestateFit;

use crate::errors::{algo_err_to_py, build_err_to_py, not_fitted};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype};

// ---------------------------------------------------------------------------
// EmpiricalCovariance — Fit (unsupervised)
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyEmpiricalCovariance,
    algo:  mlrs_algos::covariance::empirical_covariance::EmpiricalCovariance,
    unfit: { assume_centered: bool, store_precision: bool },
}

/// sklearn-compatible `EmpiricalCovariance` (MLE / `ddof=0` covariance, COV-01).
#[pyclass(name = "EmpiricalCovariance")]
pub struct PyEmpiricalCovariance {
    inner: AnyEmpiricalCovariance,
}

impl PyEmpiricalCovariance {
    /// Rust-callable default constructor for the smoke test (sklearn defaults:
    /// `assume_centered=false`, `store_precision=true`).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyEmpiricalCovariance::Unfit {
                assume_centered: false,
                store_precision: true,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyEmpiricalCovariance::Unfit { .. })
    }
}

#[pymethods]
impl PyEmpiricalCovariance {
    /// `EmpiricalCovariance(store_precision=True, assume_centered=False)`.
    #[new]
    #[pyo3(signature = (store_precision = true, assume_centered = false))]
    fn new(store_precision: bool, assume_centered: bool) -> Self {
        Self {
            inner: AnyEmpiricalCovariance::Unfit {
                assume_centered,
                store_precision,
            },
        }
    }

    /// Fit on `x` (`rows × cols`). Unsupervised — no `y`. GIL released (PY-03);
    /// f64 guarded on an f64-incapable backend (D-04).
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (assume_centered, store_precision) = match &self.inner {
            AnyEmpiricalCovariance::Unfit {
                assume_centered,
                store_precision,
            } => (*assume_centered, *store_precision),
            _ => (false, true),
        };
        let fitted = py.detach(|| -> PyResult<AnyEmpiricalCovariance> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = EmpiricalCovariance::<f32>::builder()
                        .assume_centered(assume_centered)
                        .store_precision(store_precision)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyEmpiricalCovariance::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = EmpiricalCovariance::<f64>::builder()
                        .assume_centered(assume_centered)
                        .store_precision(store_precision)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyEmpiricalCovariance::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn covariance_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyEmpiricalCovariance::F32(e) => Ok(e.covariance_(&pool)),
            _ => Err(not_fitted("empirical_covariance", "covariance_ (f32)")),
        }
    }
    fn covariance_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyEmpiricalCovariance::F64(e) => Ok(e.covariance_(&pool)),
            _ => Err(not_fitted("empirical_covariance", "covariance_ (f64)")),
        }
    }
    fn location_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyEmpiricalCovariance::F32(e) => Ok(e.location_(&pool)),
            _ => Err(not_fitted("empirical_covariance", "location_ (f32)")),
        }
    }
    fn location_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyEmpiricalCovariance::F64(e) => Ok(e.location_(&pool)),
            _ => Err(not_fitted("empirical_covariance", "location_ (f64)")),
        }
    }
    fn precision_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyEmpiricalCovariance::F32(e) => e.precision_(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("empirical_covariance", "precision_ (f32)")),
        }
    }
    fn precision_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyEmpiricalCovariance::F64(e) => e.precision_(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("empirical_covariance", "precision_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyEmpiricalCovariance::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyEmpiricalCovariance::Unfit { .. } => None,
            AnyEmpiricalCovariance::F32(_) => Some("f32"),
            AnyEmpiricalCovariance::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// LedoitWolf — Fit (unsupervised)
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyLedoitWolf,
    algo:  mlrs_algos::covariance::ledoit_wolf::LedoitWolf,
    unfit: { assume_centered: bool },
}

/// sklearn-compatible `LedoitWolf` (shrinkage covariance, COV-02).
#[pyclass(name = "LedoitWolf")]
pub struct PyLedoitWolf {
    inner: AnyLedoitWolf,
}

impl PyLedoitWolf {
    /// Rust-callable default constructor for the smoke test (sklearn default:
    /// `assume_centered=false`).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyLedoitWolf::Unfit { assume_centered: false },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyLedoitWolf::Unfit { .. })
    }
}

#[pymethods]
impl PyLedoitWolf {
    /// `LedoitWolf(assume_centered=False)`.
    #[new]
    #[pyo3(signature = (assume_centered = false))]
    fn new(assume_centered: bool) -> Self {
        Self {
            inner: AnyLedoitWolf::Unfit { assume_centered },
        }
    }

    /// Fit on `x` (`rows × cols`). Unsupervised — no `y`. GIL released (PY-03);
    /// f64 guarded on an f64-incapable backend (D-04).
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let assume_centered = match &self.inner {
            AnyLedoitWolf::Unfit { assume_centered } => *assume_centered,
            _ => false,
        };
        let fitted = py.detach(|| -> PyResult<AnyLedoitWolf> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = LedoitWolf::<f32>::builder()
                        .assume_centered(assume_centered)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyLedoitWolf::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = LedoitWolf::<f64>::builder()
                        .assume_centered(assume_centered)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyLedoitWolf::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn covariance_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLedoitWolf::F32(e) => Ok(e.covariance_(&pool)),
            _ => Err(not_fitted("ledoit_wolf", "covariance_ (f32)")),
        }
    }
    fn covariance_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLedoitWolf::F64(e) => Ok(e.covariance_(&pool)),
            _ => Err(not_fitted("ledoit_wolf", "covariance_ (f64)")),
        }
    }
    fn location_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLedoitWolf::F32(e) => Ok(e.location_(&pool)),
            _ => Err(not_fitted("ledoit_wolf", "location_ (f32)")),
        }
    }
    fn location_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyLedoitWolf::F64(e) => Ok(e.location_(&pool)),
            _ => Err(not_fitted("ledoit_wolf", "location_ (f64)")),
        }
    }

    /// `shrinkage_` is a single `f64` scalar (single-typed, no dtype suffix —
    /// the algos estimator keeps it in `f64` regardless of `F`).
    fn shrinkage_(&self) -> PyResult<f64> {
        match &self.inner {
            AnyLedoitWolf::F32(e) => Ok(e.shrinkage_()),
            AnyLedoitWolf::F64(e) => Ok(e.shrinkage_()),
            _ => Err(not_fitted("ledoit_wolf", "shrinkage_")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyLedoitWolf::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyLedoitWolf::Unfit { .. } => None,
            AnyLedoitWolf::F32(_) => Some("f32"),
            AnyLedoitWolf::F64(_) => Some("f64"),
        }
    }
}
