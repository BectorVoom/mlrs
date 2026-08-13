//! `preprocessing` `#[pyclass]` wrappers (PY-01/PY-02/PY-05, PREP-01): `PyStandardScaler`,
//! `PyMinMaxScaler`, `PyMaxAbsScaler`, `PyRobustScaler`, `PyNormalizer`, `PyBinarizer`.
//!
//! All six are `Fit` (unsupervised — `y = None`) + [`TypestateTransform`] over
//! `mlrs_algos::preprocessing`, dtype-dispatched (D-06) through the macro-emitted
//! `Any<Name>` enum (the `PCA`/`decomposition.rs` precedent). The four column
//! scalers (`StandardScaler`/`MinMaxScaler`/`MaxAbsScaler`/`RobustScaler`)
//! additionally implement `inverse_transform`; `Normalizer`/`Binarizer` leave the
//! typestate default (`AlgoError::Unsupported`, matching `TruncatedSVD`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mlrs_algos::preprocessing::normalizer::Norm;
use mlrs_algos::preprocessing::{
    Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler,
};
use mlrs_algos::typestate::{Fit as TypestateFit, Transform as TypestateTransform};

use crate::errors::{algo_err_to_py, build_err_to_py, not_fitted};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype};

fn parse_norm(s: &str) -> PyResult<Norm> {
    match s {
        "l1" => Ok(Norm::L1),
        "l2" => Ok(Norm::L2),
        "max" => Ok(Norm::Max),
        other => Err(PyValueError::new_err(format!(
            "Normalizer: unknown norm '{other}' (expected 'l1', 'l2', or 'max')"
        ))),
    }
}

// ---------------------------------------------------------------------------
// StandardScaler — Fit (unsupervised) + Transform + inverse_transform
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyStandardScaler,
    algo:  mlrs_algos::preprocessing::standard_scaler::StandardScaler,
    unfit: { with_mean: bool, with_std: bool },
}

crate::impl_persistable_any! {
    any:  AnyStandardScaler,
    algo: mlrs_algos::preprocessing::standard_scaler::StandardScaler,
    name: "standard_scaler",
}

/// sklearn-compatible `StandardScaler`.
///
/// The constructor's hyperparameters live on the PYCLASS, not only in
/// `Any*::Unfit`. `fit` overwrites `inner` with the fitted arm, so reading them
/// back out of the enum works exactly once: a second `fit` on the same handle
/// would fall through to the `_` arm and silently re-fit with sklearn's
/// DEFAULTS while `get_params()` still reported what the caller asked for. The
/// Python shim happens to build a fresh `_mlrs` object per `fit` today, which
/// hides it — but this is a public `#[pyclass]`, and reusing the handle across
/// fits is the obvious optimization to make later. Every scaler below keeps its
/// parameters the same way.
#[pyclass(name = "StandardScaler")]
pub struct PyStandardScaler {
    inner: AnyStandardScaler,
    with_mean: bool,
    with_std: bool,
}

#[pymethods]
impl PyStandardScaler {
    #[new]
    fn new(with_mean: bool, with_std: bool) -> Self {
        Self {
            inner: AnyStandardScaler::Unfit { with_mean, with_std },
            with_mean,
            with_std,
        }
    }

    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (with_mean, with_std) = (self.with_mean, self.with_std);
        let fitted = py.detach(|| -> PyResult<AnyStandardScaler> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = StandardScaler::<f32>::builder().with_mean(with_mean).with_std(with_std).build::<f32>();
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyStandardScaler::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = StandardScaler::<f64>::builder().with_mean(with_mean).with_std(with_std).build::<f64>();
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyStandardScaler::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn transform_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyStandardScaler::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestateTransform::transform(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("standard_scaler", "transform (f32 path)")),
            }
        })
    }
    fn transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyStandardScaler::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestateTransform::transform(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("standard_scaler", "transform (f64 path)")),
            }
        })
    }

    fn inverse_transform_f32(&self, py: Python<'_>, z: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let za = capsule_to_array(z)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyStandardScaler::F32(est) => {
                    let zd = validated_f32(as_f32(&za)?, &mut pool)?;
                    Ok(TypestateTransform::inverse_transform(est, &mut pool, &zd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("standard_scaler", "inverse_transform (f32 path)")),
            }
        })
    }
    fn inverse_transform_f64(&self, py: Python<'_>, z: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let za = capsule_to_array(z)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyStandardScaler::F64(est) => {
                    let zd = validated_f64(as_f64(&za)?, &mut pool)?;
                    Ok(TypestateTransform::inverse_transform(est, &mut pool, &zd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("standard_scaler", "inverse_transform (f64 path)")),
            }
        })
    }

    fn mean_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyStandardScaler::F32(e) => Ok(e.mean(&crate::lock_pool())),
            _ => Err(not_fitted("standard_scaler", "mean_ (f32)")),
        }
    }
    fn mean_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyStandardScaler::F64(e) => Ok(e.mean(&crate::lock_pool())),
            _ => Err(not_fitted("standard_scaler", "mean_ (f64)")),
        }
    }
    fn var_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyStandardScaler::F32(e) => Ok(e.var(&crate::lock_pool())),
            _ => Err(not_fitted("standard_scaler", "var_ (f32)")),
        }
    }
    fn var_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyStandardScaler::F64(e) => Ok(e.var(&crate::lock_pool())),
            _ => Err(not_fitted("standard_scaler", "var_ (f64)")),
        }
    }
    fn scale_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyStandardScaler::F32(e) => Ok(e.scale(&crate::lock_pool())),
            _ => Err(not_fitted("standard_scaler", "scale_ (f32)")),
        }
    }
    fn scale_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyStandardScaler::F64(e) => Ok(e.scale(&crate::lock_pool())),
            _ => Err(not_fitted("standard_scaler", "scale_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyStandardScaler::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyStandardScaler::Unfit { .. } => None,
            AnyStandardScaler::F32(_) => Some("f32"),
            AnyStandardScaler::F64(_) => Some("f64"),
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
// MinMaxScaler — Fit (unsupervised) + Transform + inverse_transform
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyMinMaxScaler,
    algo:  mlrs_algos::preprocessing::min_max_scaler::MinMaxScaler,
    unfit: { feature_min: f64, feature_max: f64, clip: bool },
}

crate::impl_persistable_any! {
    any:  AnyMinMaxScaler,
    algo: mlrs_algos::preprocessing::min_max_scaler::MinMaxScaler,
    name: "min_max_scaler",
}

/// sklearn-compatible `MinMaxScaler`.
#[pyclass(name = "MinMaxScaler")]
pub struct PyMinMaxScaler {
    inner: AnyMinMaxScaler,
    feature_min: f64,
    feature_max: f64,
    clip: bool,
}

#[pymethods]
impl PyMinMaxScaler {
    #[new]
    fn new(feature_min: f64, feature_max: f64, clip: bool) -> Self {
        Self {
            inner: AnyMinMaxScaler::Unfit { feature_min, feature_max, clip },
            feature_min,
            feature_max,
            clip,
        }
    }

    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (lo, hi, clip) = (self.feature_min, self.feature_max, self.clip);
        let fitted = py.detach(|| -> PyResult<AnyMinMaxScaler> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = MinMaxScaler::<f32>::builder().feature_range(lo, hi).clip(clip).build::<f32>().map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyMinMaxScaler::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = MinMaxScaler::<f64>::builder().feature_range(lo, hi).clip(clip).build::<f64>().map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyMinMaxScaler::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn transform_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMinMaxScaler::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestateTransform::transform(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("min_max_scaler", "transform (f32 path)")),
            }
        })
    }
    fn transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMinMaxScaler::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestateTransform::transform(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("min_max_scaler", "transform (f64 path)")),
            }
        })
    }

    fn inverse_transform_f32(&self, py: Python<'_>, z: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let za = capsule_to_array(z)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMinMaxScaler::F32(est) => {
                    let zd = validated_f32(as_f32(&za)?, &mut pool)?;
                    Ok(TypestateTransform::inverse_transform(est, &mut pool, &zd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("min_max_scaler", "inverse_transform (f32 path)")),
            }
        })
    }
    fn inverse_transform_f64(&self, py: Python<'_>, z: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let za = capsule_to_array(z)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMinMaxScaler::F64(est) => {
                    let zd = validated_f64(as_f64(&za)?, &mut pool)?;
                    Ok(TypestateTransform::inverse_transform(est, &mut pool, &zd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("min_max_scaler", "inverse_transform (f64 path)")),
            }
        })
    }

    fn data_min_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyMinMaxScaler::F32(e) => Ok(e.data_min(&crate::lock_pool())),
            _ => Err(not_fitted("min_max_scaler", "data_min_ (f32)")),
        }
    }
    fn data_min_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyMinMaxScaler::F64(e) => Ok(e.data_min(&crate::lock_pool())),
            _ => Err(not_fitted("min_max_scaler", "data_min_ (f64)")),
        }
    }
    fn data_max_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyMinMaxScaler::F32(e) => Ok(e.data_max(&crate::lock_pool())),
            _ => Err(not_fitted("min_max_scaler", "data_max_ (f32)")),
        }
    }
    fn data_max_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyMinMaxScaler::F64(e) => Ok(e.data_max(&crate::lock_pool())),
            _ => Err(not_fitted("min_max_scaler", "data_max_ (f64)")),
        }
    }
    fn scale_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyMinMaxScaler::F32(e) => Ok(e.scale(&crate::lock_pool())),
            _ => Err(not_fitted("min_max_scaler", "scale_ (f32)")),
        }
    }
    fn scale_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyMinMaxScaler::F64(e) => Ok(e.scale(&crate::lock_pool())),
            _ => Err(not_fitted("min_max_scaler", "scale_ (f64)")),
        }
    }
    fn min_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyMinMaxScaler::F32(e) => Ok(e.min(&crate::lock_pool())),
            _ => Err(not_fitted("min_max_scaler", "min_ (f32)")),
        }
    }
    fn min_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyMinMaxScaler::F64(e) => Ok(e.min(&crate::lock_pool())),
            _ => Err(not_fitted("min_max_scaler", "min_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyMinMaxScaler::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyMinMaxScaler::Unfit { .. } => None,
            AnyMinMaxScaler::F32(_) => Some("f32"),
            AnyMinMaxScaler::F64(_) => Some("f64"),
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
// MaxAbsScaler — Fit (unsupervised) + Transform + inverse_transform
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyMaxAbsScaler,
    algo:  mlrs_algos::preprocessing::max_abs_scaler::MaxAbsScaler,
    unfit: {},
}

crate::impl_persistable_any! {
    any:  AnyMaxAbsScaler,
    algo: mlrs_algos::preprocessing::max_abs_scaler::MaxAbsScaler,
    name: "max_abs_scaler",
}

/// sklearn-compatible `MaxAbsScaler`.
#[pyclass(name = "MaxAbsScaler")]
pub struct PyMaxAbsScaler {
    inner: AnyMaxAbsScaler,
}

#[pymethods]
impl PyMaxAbsScaler {
    #[new]
    fn new() -> Self {
        Self { inner: AnyMaxAbsScaler::Unfit {} }
    }

    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let fitted = py.detach(|| -> PyResult<AnyMaxAbsScaler> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = MaxAbsScaler::<f32>::new();
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyMaxAbsScaler::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = MaxAbsScaler::<f64>::new();
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyMaxAbsScaler::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn transform_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMaxAbsScaler::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestateTransform::transform(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("max_abs_scaler", "transform (f32 path)")),
            }
        })
    }
    fn transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMaxAbsScaler::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestateTransform::transform(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("max_abs_scaler", "transform (f64 path)")),
            }
        })
    }

    fn inverse_transform_f32(&self, py: Python<'_>, z: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let za = capsule_to_array(z)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMaxAbsScaler::F32(est) => {
                    let zd = validated_f32(as_f32(&za)?, &mut pool)?;
                    Ok(TypestateTransform::inverse_transform(est, &mut pool, &zd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("max_abs_scaler", "inverse_transform (f32 path)")),
            }
        })
    }
    fn inverse_transform_f64(&self, py: Python<'_>, z: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let za = capsule_to_array(z)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyMaxAbsScaler::F64(est) => {
                    let zd = validated_f64(as_f64(&za)?, &mut pool)?;
                    Ok(TypestateTransform::inverse_transform(est, &mut pool, &zd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("max_abs_scaler", "inverse_transform (f64 path)")),
            }
        })
    }

    fn max_abs_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyMaxAbsScaler::F32(e) => Ok(e.max_abs(&crate::lock_pool())),
            _ => Err(not_fitted("max_abs_scaler", "max_abs_ (f32)")),
        }
    }
    fn max_abs_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyMaxAbsScaler::F64(e) => Ok(e.max_abs(&crate::lock_pool())),
            _ => Err(not_fitted("max_abs_scaler", "max_abs_ (f64)")),
        }
    }
    fn scale_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyMaxAbsScaler::F32(e) => Ok(e.scale(&crate::lock_pool())),
            _ => Err(not_fitted("max_abs_scaler", "scale_ (f32)")),
        }
    }
    fn scale_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyMaxAbsScaler::F64(e) => Ok(e.scale(&crate::lock_pool())),
            _ => Err(not_fitted("max_abs_scaler", "scale_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyMaxAbsScaler::Unfit {})
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyMaxAbsScaler::Unfit {} => None,
            AnyMaxAbsScaler::F32(_) => Some("f32"),
            AnyMaxAbsScaler::F64(_) => Some("f64"),
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
// RobustScaler — Fit (unsupervised) + Transform + inverse_transform
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyRobustScaler,
    algo:  mlrs_algos::preprocessing::robust_scaler::RobustScaler,
    unfit: { with_centering: bool, with_scaling: bool, q_min: f64, q_max: f64, unit_variance: bool },
}

crate::impl_persistable_any! {
    any:  AnyRobustScaler,
    algo: mlrs_algos::preprocessing::robust_scaler::RobustScaler,
    name: "robust_scaler",
}

/// sklearn-compatible `RobustScaler`.
#[pyclass(name = "RobustScaler")]
pub struct PyRobustScaler {
    inner: AnyRobustScaler,
    with_centering: bool,
    with_scaling: bool,
    q_min: f64,
    q_max: f64,
    unit_variance: bool,
}

#[pymethods]
impl PyRobustScaler {
    #[new]
    fn new(with_centering: bool, with_scaling: bool, q_min: f64, q_max: f64, unit_variance: bool) -> Self {
        Self {
            inner: AnyRobustScaler::Unfit { with_centering, with_scaling, q_min, q_max, unit_variance },
            with_centering,
            with_scaling,
            q_min,
            q_max,
            unit_variance,
        }
    }

    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (with_centering, with_scaling, q_min, q_max, unit_variance) = (
            self.with_centering,
            self.with_scaling,
            self.q_min,
            self.q_max,
            self.unit_variance,
        );
        let fitted = py.detach(|| -> PyResult<AnyRobustScaler> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = RobustScaler::<f32>::builder()
                        .with_centering(with_centering)
                        .with_scaling(with_scaling)
                        .quantile_range(q_min, q_max)
                        .unit_variance(unit_variance)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyRobustScaler::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = RobustScaler::<f64>::builder()
                        .with_centering(with_centering)
                        .with_scaling(with_scaling)
                        .quantile_range(q_min, q_max)
                        .unit_variance(unit_variance)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyRobustScaler::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn transform_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyRobustScaler::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestateTransform::transform(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("robust_scaler", "transform (f32 path)")),
            }
        })
    }
    fn transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyRobustScaler::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestateTransform::transform(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("robust_scaler", "transform (f64 path)")),
            }
        })
    }

    fn inverse_transform_f32(&self, py: Python<'_>, z: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let za = capsule_to_array(z)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyRobustScaler::F32(est) => {
                    let zd = validated_f32(as_f32(&za)?, &mut pool)?;
                    Ok(TypestateTransform::inverse_transform(est, &mut pool, &zd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("robust_scaler", "inverse_transform (f32 path)")),
            }
        })
    }
    fn inverse_transform_f64(&self, py: Python<'_>, z: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let za = capsule_to_array(z)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyRobustScaler::F64(est) => {
                    let zd = validated_f64(as_f64(&za)?, &mut pool)?;
                    Ok(TypestateTransform::inverse_transform(est, &mut pool, &zd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("robust_scaler", "inverse_transform (f64 path)")),
            }
        })
    }

    fn center_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyRobustScaler::F32(e) => Ok(e.center(&crate::lock_pool())),
            _ => Err(not_fitted("robust_scaler", "center_ (f32)")),
        }
    }
    fn center_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyRobustScaler::F64(e) => Ok(e.center(&crate::lock_pool())),
            _ => Err(not_fitted("robust_scaler", "center_ (f64)")),
        }
    }
    fn scale_f32(&self) -> PyResult<Vec<f32>> {
        match &self.inner {
            AnyRobustScaler::F32(e) => Ok(e.scale(&crate::lock_pool())),
            _ => Err(not_fitted("robust_scaler", "scale_ (f32)")),
        }
    }
    fn scale_f64(&self) -> PyResult<Vec<f64>> {
        match &self.inner {
            AnyRobustScaler::F64(e) => Ok(e.scale(&crate::lock_pool())),
            _ => Err(not_fitted("robust_scaler", "scale_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyRobustScaler::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyRobustScaler::Unfit { .. } => None,
            AnyRobustScaler::F32(_) => Some("f32"),
            AnyRobustScaler::F64(_) => Some("f64"),
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
// Normalizer — Fit (no-op, unsupervised) + Transform (no inverse)
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyNormalizer,
    algo:  mlrs_algos::preprocessing::normalizer::Normalizer,
    unfit: { norm: String },
}

crate::impl_persistable_any! {
    any:  AnyNormalizer,
    algo: mlrs_algos::preprocessing::normalizer::Normalizer,
    name: "normalizer",
}

/// sklearn-compatible `Normalizer`.
#[pyclass(name = "Normalizer")]
pub struct PyNormalizer {
    inner: AnyNormalizer,
    norm: String,
}

#[pymethods]
impl PyNormalizer {
    #[new]
    fn new(norm: String) -> Self {
        Self {
            inner: AnyNormalizer::Unfit { norm: norm.clone() },
            norm,
        }
    }

    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let norm = parse_norm(&self.norm)?;
        let fitted = py.detach(|| -> PyResult<AnyNormalizer> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = Normalizer::<f32>::with_norm(norm);
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyNormalizer::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = Normalizer::<f64>::with_norm(norm);
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyNormalizer::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn transform_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyNormalizer::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestateTransform::transform(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("normalizer", "transform (f32 path)")),
            }
        })
    }
    fn transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyNormalizer::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestateTransform::transform(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("normalizer", "transform (f64 path)")),
            }
        })
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyNormalizer::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyNormalizer::Unfit { .. } => None,
            AnyNormalizer::F32(_) => Some("f32"),
            AnyNormalizer::F64(_) => Some("f64"),
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
// Binarizer — Fit (no-op, unsupervised) + Transform (no inverse)
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyBinarizer,
    algo:  mlrs_algos::preprocessing::binarizer::Binarizer,
    unfit: { threshold: f64 },
}

crate::impl_persistable_any! {
    any:  AnyBinarizer,
    algo: mlrs_algos::preprocessing::binarizer::Binarizer,
    name: "binarizer",
}

/// sklearn-compatible `Binarizer`.
#[pyclass(name = "Binarizer")]
pub struct PyBinarizer {
    inner: AnyBinarizer,
    threshold: f64,
}

#[pymethods]
impl PyBinarizer {
    #[new]
    fn new(threshold: f64) -> Self {
        Self {
            inner: AnyBinarizer::Unfit { threshold },
            threshold,
        }
    }

    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let threshold = self.threshold;
        let fitted = py.detach(|| -> PyResult<AnyBinarizer> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = Binarizer::<f32>::with_threshold(threshold);
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyBinarizer::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = Binarizer::<f64>::with_threshold(threshold);
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyBinarizer::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn transform_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyBinarizer::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(TypestateTransform::transform(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("binarizer", "transform (f32 path)")),
            }
        })
    }
    fn transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::lock_pool();
            match &self.inner {
                AnyBinarizer::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(TypestateTransform::transform(est, &mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("binarizer", "transform (f64 path)")),
            }
        })
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyBinarizer::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyBinarizer::Unfit { .. } => None,
            AnyBinarizer::F32(_) => Some("f32"),
            AnyBinarizer::F64(_) => Some("f64"),
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
