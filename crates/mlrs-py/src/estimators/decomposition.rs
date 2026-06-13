//! Decomposition `#[pyclass]` wrappers (PY-01/PY-02/PY-05): `PyPCA`,
//! `PyTruncatedSVD`.
//!
//! Both are `Fit` (unsupervised — `y = None`) + [`Transform`] surfaces over the
//! `mlrs_algos` decompositions, dtype-dispatched (D-06) through the macro-emitted
//! `Any<Name>` enum. PCA additionally exposes `inverse_transform` (the optional
//! [`Transform::inverse_transform`]); TruncatedSVD leaves it unsupported (algos
//! default → `AlgoError::Unsupported`, mapped to a clear `PyValueError`).

use pyo3::prelude::*;

use mlrs_algos::decomposition::pca::Pca;
use mlrs_algos::decomposition::truncated_svd::TruncatedSvd;
use mlrs_algos::traits::{Fit, Transform};

use crate::errors::{algo_err_to_py, not_fitted};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype};

// ---------------------------------------------------------------------------
// PCA — Fit (unsupervised) + Transform + inverse_transform
// ---------------------------------------------------------------------------

crate::any_estimator! {
    any:   AnyPca,
    algo:  mlrs_algos::decomposition::pca::Pca,
    unfit: { n_components: usize },
}

/// sklearn-compatible `PCA` (SVD-based principal component analysis).
#[pyclass(name = "PCA")]
pub struct PyPCA {
    inner: AnyPca,
}

impl PyPCA {
    /// Rust-callable default constructor for the smoke test (PCA requires an
    /// explicit `n_components`; the smoke test uses 2).
    pub fn unfit_default() -> Self {
        Self { inner: AnyPca::Unfit { n_components: 2 } }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyPca::Unfit { .. })
    }
}

#[pymethods]
impl PyPCA {
    /// `PCA(n_components)` — v1 requires an explicit int `n_components`.
    #[new]
    fn new(n_components: usize) -> Self {
        Self {
            inner: AnyPca::Unfit { n_components },
        }
    }

    /// Fit on `x` (`rows × cols`). Unsupervised — no `y`. GIL released (PY-03);
    /// f64 guarded on an f64-incapable backend (D-04).
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let n_components = match &self.inner {
            AnyPca::Unfit { n_components } => *n_components,
            _ => 0,
        };
        let fitted = py.detach(|| -> PyResult<AnyPca> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let mut est = Pca::<f32>::new(n_components);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyPca::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let mut est = Pca::<f64>::new(n_components);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyPca::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `transform(x)` → `rows × n_components` host floats.
    fn transform_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyPca::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(est.transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("pca", "transform (f32 path)")),
            }
        })
    }
    fn transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyPca::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(est.transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("pca", "transform (f64 path)")),
            }
        })
    }

    /// `inverse_transform(z)` (PCA only) — `z` is `rows × n_components`; returns
    /// `rows × n_features` host floats.
    fn inverse_transform_f32(&self, py: Python<'_>, z: &Bound<'_, PyAny>, rows: usize, k: usize) -> PyResult<Vec<f32>> {
        let za = capsule_to_array(z)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyPca::F32(est) => {
                    let zd = validated_f32(as_f32(&za)?, &mut pool)?;
                    Ok(est.inverse_transform(&mut pool, &zd, (rows, k)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("pca", "inverse_transform (f32 path)")),
            }
        })
    }
    fn inverse_transform_f64(&self, py: Python<'_>, z: &Bound<'_, PyAny>, rows: usize, k: usize) -> PyResult<Vec<f64>> {
        let za = capsule_to_array(z)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyPca::F64(est) => {
                    let zd = validated_f64(as_f64(&za)?, &mut pool)?;
                    Ok(est.inverse_transform(&mut pool, &zd, (rows, k)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("pca", "inverse_transform (f64 path)")),
            }
        })
    }

    fn components_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyPca::F32(e) => e.components(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("pca", "components_ (f32)")),
        }
    }
    fn components_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyPca::F64(e) => e.components(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("pca", "components_ (f64)")),
        }
    }
    fn mean_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyPca::F32(e) => e.mean(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("pca", "mean_ (f32)")),
        }
    }
    fn mean_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyPca::F64(e) => e.mean(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("pca", "mean_ (f64)")),
        }
    }
    fn explained_variance_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyPca::F32(e) => e.explained_variance(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("pca", "explained_variance_ (f32)")),
        }
    }
    fn explained_variance_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyPca::F64(e) => e.explained_variance(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("pca", "explained_variance_ (f64)")),
        }
    }
    fn explained_variance_ratio_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyPca::F32(e) => e.explained_variance_ratio(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("pca", "explained_variance_ratio_ (f32)")),
        }
    }
    fn explained_variance_ratio_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyPca::F64(e) => e.explained_variance_ratio(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("pca", "explained_variance_ratio_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyPca::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyPca::Unfit { .. } => None,
            AnyPca::F32(_) => Some("f32"),
            AnyPca::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// TruncatedSVD — Fit (unsupervised) + Transform
// ---------------------------------------------------------------------------

crate::any_estimator! {
    any:   AnyTruncatedSvd,
    algo:  mlrs_algos::decomposition::truncated_svd::TruncatedSvd,
    unfit: { n_components: usize },
}

/// sklearn-compatible `TruncatedSVD` (LSA-style truncated SVD).
#[pyclass(name = "TruncatedSVD")]
pub struct PyTruncatedSVD {
    inner: AnyTruncatedSvd,
}

impl PyTruncatedSVD {
    /// Rust-callable default constructor for the smoke test. See
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self { inner: AnyTruncatedSvd::Unfit { n_components: 2 } }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyTruncatedSvd::Unfit { .. })
    }
}

#[pymethods]
impl PyTruncatedSVD {
    /// `TruncatedSVD(n_components=2)`.
    #[new]
    #[pyo3(signature = (n_components = 2))]
    fn new(n_components: usize) -> Self {
        Self {
            inner: AnyTruncatedSvd::Unfit { n_components },
        }
    }

    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let n_components = match &self.inner {
            AnyTruncatedSvd::Unfit { n_components } => *n_components,
            _ => 2,
        };
        let fitted = py.detach(|| -> PyResult<AnyTruncatedSvd> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let mut est = TruncatedSvd::<f32>::new(n_components);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyTruncatedSvd::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let mut est = TruncatedSvd::<f64>::new(n_components);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyTruncatedSvd::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    fn transform_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyTruncatedSvd::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(est.transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("truncated_svd", "transform (f32 path)")),
            }
        })
    }
    fn transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyTruncatedSvd::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(est.transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("truncated_svd", "transform (f64 path)")),
            }
        })
    }

    fn components_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyTruncatedSvd::F32(e) => e.components(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("truncated_svd", "components_ (f32)")),
        }
    }
    fn components_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyTruncatedSvd::F64(e) => e.components(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("truncated_svd", "components_ (f64)")),
        }
    }
    fn singular_values_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyTruncatedSvd::F32(e) => e.singular_values(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("truncated_svd", "singular_values_ (f32)")),
        }
    }
    fn singular_values_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyTruncatedSvd::F64(e) => e.singular_values(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("truncated_svd", "singular_values_ (f64)")),
        }
    }
    fn explained_variance_ratio_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyTruncatedSvd::F32(e) => e.explained_variance_ratio(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("truncated_svd", "explained_variance_ratio_ (f32)")),
        }
    }
    fn explained_variance_ratio_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyTruncatedSvd::F64(e) => e.explained_variance_ratio(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("truncated_svd", "explained_variance_ratio_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyTruncatedSvd::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyTruncatedSvd::Unfit { .. } => None,
            AnyTruncatedSvd::F32(_) => Some("f32"),
            AnyTruncatedSvd::F64(_) => Some("f64"),
        }
    }
}
