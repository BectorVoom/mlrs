//! Decomposition `#[pyclass]` wrappers (PY-01/PY-02/PY-05): `PyPCA`,
//! `PyTruncatedSVD`.
//!
//! Both are `Fit` (unsupervised — `y = None`) + [`Transform`] surfaces over the
//! `mlrs_algos` decompositions, dtype-dispatched (D-06) through the macro-emitted
//! `Any<Name>` enum. PCA additionally exposes `inverse_transform` (the optional
//! [`Transform::inverse_transform`]); TruncatedSVD leaves it unsupported (algos
//! default → `AlgoError::Unsupported`, mapped to a clear `PyValueError`).

use pyo3::prelude::*;

use mlrs_algos::decomposition::incremental_pca::IncrementalPCA;
use mlrs_algos::decomposition::pca::Pca;
use mlrs_algos::decomposition::truncated_svd::TruncatedSvd;
use mlrs_algos::traits::{Fit, PartialFit, Transform};

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

// ---------------------------------------------------------------------------
// IncrementalPCA — Fit + partial_fit (the first v2 partial_fit) + Transform
//                  + inverse_transform
// ---------------------------------------------------------------------------

crate::any_estimator! {
    any:   AnyIncrementalPCA,
    algo:  mlrs_algos::decomposition::incremental_pca::IncrementalPCA,
    unfit: { n_components: usize, whiten: bool, batch_size: Option<usize> },
}

/// sklearn-compatible `IncrementalPCA` (streaming PCA via incremental SVD).
///
/// Exposes the first v2 `partial_fit` method: `partial_fit` constructs the
/// fitted `F32`/`F64` arm on the first call (from the stored hyperparameters)
/// and re-uses / mutates it in place on subsequent calls.
#[pyclass(name = "IncrementalPCA")]
pub struct PyIncrementalPCA {
    inner: AnyIncrementalPCA,
}

impl PyIncrementalPCA {
    /// Rust-callable default constructor for the smoke test (`n_components=2`,
    /// `whiten=False`, `batch_size=None`).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyIncrementalPCA::Unfit {
                n_components: 2,
                whiten: false,
                batch_size: None,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyIncrementalPCA::Unfit { .. })
    }

    /// The stored unfit hyperparameters (used to construct the fitted arm on the
    /// first `partial_fit`). Returns `None` once fitted.
    fn unfit_params(&self) -> Option<(usize, bool, Option<usize>)> {
        match &self.inner {
            AnyIncrementalPCA::Unfit {
                n_components,
                whiten,
                batch_size,
            } => Some((*n_components, *whiten, *batch_size)),
            _ => None,
        }
    }
}

#[pymethods]
impl PyIncrementalPCA {
    /// `IncrementalPCA(n_components, whiten=False, batch_size=None)`.
    #[new]
    #[pyo3(signature = (n_components, whiten = false, batch_size = None))]
    fn new(n_components: usize, whiten: bool, batch_size: Option<usize>) -> Self {
        Self {
            inner: AnyIncrementalPCA::Unfit {
                n_components,
                whiten,
                batch_size,
            },
        }
    }

    /// `fit(x)` — sklearn-faithful one-shot fit (RESETS state, loops the algos
    /// `partial_fit` over `gen_batches` internally). GIL released (PY-03); f64
    /// guarded on an f64-incapable backend (D-04).
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (n_components, whiten, batch_size) = match &self.inner {
            AnyIncrementalPCA::Unfit {
                n_components,
                whiten,
                batch_size,
            } => (*n_components, *whiten, *batch_size),
            // Re-fit from the already-fitted arm's hyperparameters.
            AnyIncrementalPCA::F32(e) => (e.n_components(), e.whiten(), e.batch_size()),
            AnyIncrementalPCA::F64(e) => (e.n_components(), e.whiten(), e.batch_size()),
        };
        let fitted = py.detach(|| -> PyResult<AnyIncrementalPCA> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let mut est = IncrementalPCA::<f32>::new(n_components, whiten, batch_size);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyIncrementalPCA::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let mut est = IncrementalPCA::<f64>::new(n_components, whiten, batch_size);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyIncrementalPCA::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `partial_fit(x)` — the first v2 streaming partial_fit. The first call
    /// constructs the `F32`/`F64` arm from the stored hyperparameters and merges
    /// the batch; subsequent calls match the existing arm and merge again,
    /// MUTATING it in place. Same `py.detach` + dtype-dispatch + `guard_f64`
    /// contract as `fit`.
    fn partial_fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let unfit = self.unfit_params();

        // If already fitted, the batch dtype must match the fitted arm's dtype
        // (mixing f32/f64 across a partial_fit stream is rejected).
        let next = py.detach(|| -> PyResult<Option<AnyIncrementalPCA>> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match (dt, unfit) {
                // First batch (Unfit) — construct the matching arm + merge.
                (FloatDtype::F32, Some((n_components, whiten, batch_size))) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let mut est = IncrementalPCA::<f32>::new(n_components, whiten, batch_size);
                    est.partial_fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(Some(AnyIncrementalPCA::F32(est)))
                }
                (FloatDtype::F64, Some((n_components, whiten, batch_size))) => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let mut est = IncrementalPCA::<f64>::new(n_components, whiten, batch_size);
                    est.partial_fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(Some(AnyIncrementalPCA::F64(est)))
                }
                // Already fitted — the dispatch below mutates the existing arm
                // in place; signal that with `None`.
                (_, None) => Ok(None),
            }
        })?;

        if let Some(new_inner) = next {
            self.inner = new_inner;
            return Ok(());
        }

        // Subsequent batch — mutate the already-fitted arm in place.
        py.detach(|| -> PyResult<()> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match (dt, &mut self.inner) {
                (FloatDtype::F32, AnyIncrementalPCA::F32(est)) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    est.partial_fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(())
                }
                (FloatDtype::F64, AnyIncrementalPCA::F64(est)) => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    est.partial_fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(())
                }
                // dtype of the batch disagrees with the fitted arm's dtype.
                _ => Err(crate::errors::dtype_mismatch_in_stream("incremental_pca")),
            }
        })
    }

    fn transform_f32(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyIncrementalPCA::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(est.transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("incremental_pca", "transform (f32 path)")),
            }
        })
    }
    fn transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyIncrementalPCA::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(est.transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("incremental_pca", "transform (f64 path)")),
            }
        })
    }

    fn inverse_transform_f32(&self, py: Python<'_>, z: &Bound<'_, PyAny>, rows: usize, k: usize) -> PyResult<Vec<f32>> {
        let za = capsule_to_array(z)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyIncrementalPCA::F32(est) => {
                    let zd = validated_f32(as_f32(&za)?, &mut pool)?;
                    Ok(est.inverse_transform(&mut pool, &zd, (rows, k)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("incremental_pca", "inverse_transform (f32 path)")),
            }
        })
    }
    fn inverse_transform_f64(&self, py: Python<'_>, z: &Bound<'_, PyAny>, rows: usize, k: usize) -> PyResult<Vec<f64>> {
        let za = capsule_to_array(z)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyIncrementalPCA::F64(est) => {
                    let zd = validated_f64(as_f64(&za)?, &mut pool)?;
                    Ok(est.inverse_transform(&mut pool, &zd, (rows, k)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("incremental_pca", "inverse_transform (f64 path)")),
            }
        })
    }

    fn components_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyIncrementalPCA::F32(e) => e.components(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("incremental_pca", "components_ (f32)")),
        }
    }
    fn components_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyIncrementalPCA::F64(e) => e.components(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("incremental_pca", "components_ (f64)")),
        }
    }
    fn explained_variance_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyIncrementalPCA::F32(e) => e.explained_variance(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("incremental_pca", "explained_variance_ (f32)")),
        }
    }
    fn explained_variance_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyIncrementalPCA::F64(e) => e.explained_variance(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("incremental_pca", "explained_variance_ (f64)")),
        }
    }
    fn explained_variance_ratio_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyIncrementalPCA::F32(e) => e.explained_variance_ratio(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("incremental_pca", "explained_variance_ratio_ (f32)")),
        }
    }
    fn explained_variance_ratio_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyIncrementalPCA::F64(e) => e.explained_variance_ratio(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("incremental_pca", "explained_variance_ratio_ (f64)")),
        }
    }
    fn singular_values_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyIncrementalPCA::F32(e) => e.singular_values(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("incremental_pca", "singular_values_ (f32)")),
        }
    }
    fn singular_values_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyIncrementalPCA::F64(e) => e.singular_values(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("incremental_pca", "singular_values_ (f64)")),
        }
    }
    fn mean_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyIncrementalPCA::F32(e) => e.mean(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("incremental_pca", "mean_ (f32)")),
        }
    }
    fn mean_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyIncrementalPCA::F64(e) => e.mean(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("incremental_pca", "mean_ (f64)")),
        }
    }
    fn var_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyIncrementalPCA::F32(e) => e.var(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("incremental_pca", "var_ (f32)")),
        }
    }
    fn var_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyIncrementalPCA::F64(e) => e.var(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("incremental_pca", "var_ (f64)")),
        }
    }
    /// `n_samples_seen_` — total samples merged so far (single `usize`, no dtype
    /// suffix). `0` before the first batch.
    fn n_samples_seen(&self) -> usize {
        match &self.inner {
            AnyIncrementalPCA::Unfit { .. } => 0,
            AnyIncrementalPCA::F32(e) => e.n_samples_seen(),
            AnyIncrementalPCA::F64(e) => e.n_samples_seen(),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyIncrementalPCA::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyIncrementalPCA::Unfit { .. } => None,
            AnyIncrementalPCA::F32(_) => Some("f32"),
            AnyIncrementalPCA::F64(_) => Some("f64"),
        }
    }
}
