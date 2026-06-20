//! Random-projection `#[pyclass]` wrappers (PROJ-01/PROJ-02):
//! `PyGaussianRandomProjection`, `PySparseRandomProjection`, plus the
//! [`johnson_lindenstrauss_min_dim`] `#[pyfunction]`.
//!
//! Both estimators are `Fit` + [`Transform`] surfaces over the `mlrs_algos`
//! random-projection transformers, dtype-dispatched (D-06) through the
//! macro-emitted `Any<Name>` enum. Each device-compute body honors the two
//! load-bearing contracts documented on [`crate::dispatch`]: GIL release via
//! `py.detach` (PY-03) and [`crate::capability::guard_f64`]`()?` BEFORE the F64
//! arm (D-04).
//!
//! `n_components='auto'` is mapped from the Python shim as a sentinel — the
//! wrapper takes `n_components: Option<usize>` (`None` → [`NComponents::Auto`],
//! `Some(k)` → [`NComponents::Fixed`]). `eps` and `seed` are sklearn-named ctor
//! args; Sparse adds `density: Option<f64>` (`None` → sklearn's
//! `1/sqrt(n_features)` default, resolved in the algos `fit`).
//!
//! SparseRandomProjection densifies sparse input at the Python ingress boundary
//! (PROJ-02) — the device path is dense-only; that densification is in the
//! pure-Python shim (Plan 04), not here.

use pyo3::prelude::*;

use mlrs_algos::projection::gaussian::{
    johnson_lindenstrauss_min_dim as algo_jl_min_dim, GaussianRandomProjection, NComponents,
};
use mlrs_algos::projection::sparse::SparseRandomProjection;
use mlrs_algos::traits::{Fit, Transform};

use crate::errors::{algo_err_to_py, not_fitted};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype};

/// Resolve the optional `n_components` sentinel from the Python shim into the
/// algos [`NComponents`] selector: `None` → `Auto` (JL-sized), `Some(k)` →
/// `Fixed(k)`.
fn resolve_n_components(n_components: Option<usize>) -> NComponents {
    match n_components {
        None => NComponents::Auto,
        Some(k) => NComponents::Fixed(k),
    }
}

// ---------------------------------------------------------------------------
// GaussianRandomProjection — Fit (unsupervised) + Transform
// ---------------------------------------------------------------------------

crate::any_estimator! {
    any:   AnyGaussianRandomProjection,
    algo:  mlrs_algos::projection::gaussian::GaussianRandomProjection,
    unfit: { n_components: Option<usize>, eps: f64, seed: u64 },
}

/// sklearn-compatible `GaussianRandomProjection` (PROJ-01).
#[pyclass(name = "GaussianRandomProjection")]
pub struct PyGaussianRandomProjection {
    inner: AnyGaussianRandomProjection,
}

impl PyGaussianRandomProjection {
    /// Rust-callable default constructor for the smoke test (sklearn defaults:
    /// `n_components='auto'` → `None`, `eps=0.1`, `seed=0`).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyGaussianRandomProjection::Unfit {
                n_components: None,
                eps: 0.1,
                seed: 0,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyGaussianRandomProjection::Unfit { .. })
    }
}

#[pymethods]
impl PyGaussianRandomProjection {
    /// `GaussianRandomProjection(n_components=None, eps=0.1, seed=0)` — `None`
    /// `n_components` means `'auto'` (JL-sized).
    #[new]
    #[pyo3(signature = (n_components = None, eps = 0.1, seed = 0))]
    fn new(n_components: Option<usize>, eps: f64, seed: u64) -> Self {
        Self {
            inner: AnyGaussianRandomProjection::Unfit { n_components, eps, seed },
        }
    }

    /// Fit on `x` (`rows × cols`). Unsupervised — no `y`. GIL released (PY-03);
    /// f64 guarded on an f64-incapable backend (D-04).
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (n_components, eps, seed) = match &self.inner {
            AnyGaussianRandomProjection::Unfit { n_components, eps, seed } => {
                (*n_components, *eps, *seed)
            }
            _ => (None, 0.1, 0),
        };
        let nc = resolve_n_components(n_components);
        let fitted = py.detach(|| -> PyResult<AnyGaussianRandomProjection> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let mut est = GaussianRandomProjection::<f32>::new(nc, seed, eps);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyGaussianRandomProjection::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let mut est = GaussianRandomProjection::<f64>::new(nc, seed, eps);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyGaussianRandomProjection::F64(est))
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
                AnyGaussianRandomProjection::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(est.transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("gaussian_random_projection", "transform (f32 path)")),
            }
        })
    }
    fn transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyGaussianRandomProjection::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(est.transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("gaussian_random_projection", "transform (f64 path)")),
            }
        })
    }

    fn components_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyGaussianRandomProjection::F32(e) => e.components(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("gaussian_random_projection", "components_ (f32)")),
        }
    }
    fn components_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyGaussianRandomProjection::F64(e) => e.components(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("gaussian_random_projection", "components_ (f64)")),
        }
    }
    /// Resolved embedding dimension after fit (`'auto'` → JL value); single
    /// `usize`, no dtype suffix.
    fn n_components_(&self) -> PyResult<usize> {
        match &self.inner {
            AnyGaussianRandomProjection::F32(e) => Ok(e.n_components_()),
            AnyGaussianRandomProjection::F64(e) => Ok(e.n_components_()),
            _ => Err(not_fitted("gaussian_random_projection", "n_components_")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyGaussianRandomProjection::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyGaussianRandomProjection::Unfit { .. } => None,
            AnyGaussianRandomProjection::F32(_) => Some("f32"),
            AnyGaussianRandomProjection::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// SparseRandomProjection — Fit (unsupervised) + Transform
// ---------------------------------------------------------------------------

crate::any_estimator! {
    any:   AnySparseRandomProjection,
    algo:  mlrs_algos::projection::sparse::SparseRandomProjection,
    unfit: { n_components: Option<usize>, eps: f64, seed: u64, density: Option<f64> },
}

/// sklearn-compatible `SparseRandomProjection` (PROJ-02, Achlioptas dense).
#[pyclass(name = "SparseRandomProjection")]
pub struct PySparseRandomProjection {
    inner: AnySparseRandomProjection,
}

impl PySparseRandomProjection {
    /// Rust-callable default constructor for the smoke test (sklearn defaults:
    /// `n_components='auto'` → `None`, `eps=0.1`, `seed=0`,
    /// `density='auto'` → `None`).
    pub fn unfit_default() -> Self {
        Self {
            inner: AnySparseRandomProjection::Unfit {
                n_components: None,
                eps: 0.1,
                seed: 0,
                density: None,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnySparseRandomProjection::Unfit { .. })
    }
}

#[pymethods]
impl PySparseRandomProjection {
    /// `SparseRandomProjection(n_components=None, eps=0.1, seed=0,
    /// density=None)` — `None` `n_components` means `'auto'`; `None` `density`
    /// means sklearn's `1/sqrt(n_features)` default.
    #[new]
    #[pyo3(signature = (n_components = None, eps = 0.1, seed = 0, density = None))]
    fn new(n_components: Option<usize>, eps: f64, seed: u64, density: Option<f64>) -> Self {
        Self {
            inner: AnySparseRandomProjection::Unfit { n_components, eps, seed, density },
        }
    }

    /// Fit on `x` (`rows × cols`). Unsupervised — no `y`. GIL released (PY-03);
    /// f64 guarded on an f64-incapable backend (D-04).
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (n_components, eps, seed, density) = match &self.inner {
            AnySparseRandomProjection::Unfit { n_components, eps, seed, density } => {
                (*n_components, *eps, *seed, *density)
            }
            _ => (None, 0.1, 0, None),
        };
        let nc = resolve_n_components(n_components);
        let fitted = py.detach(|| -> PyResult<AnySparseRandomProjection> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let mut est = SparseRandomProjection::<f32>::new(nc, seed, eps, density);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnySparseRandomProjection::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let mut est = SparseRandomProjection::<f64>::new(nc, seed, eps, density);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnySparseRandomProjection::F64(est))
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
                AnySparseRandomProjection::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(est.transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("sparse_random_projection", "transform (f32 path)")),
            }
        })
    }
    fn transform_f64(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<f64>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnySparseRandomProjection::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(est.transform(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("sparse_random_projection", "transform (f64 path)")),
            }
        })
    }

    fn components_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnySparseRandomProjection::F32(e) => e.components(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("sparse_random_projection", "components_ (f32)")),
        }
    }
    fn components_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnySparseRandomProjection::F64(e) => e.components(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("sparse_random_projection", "components_ (f64)")),
        }
    }
    /// Resolved embedding dimension after fit; single `usize`, no dtype suffix.
    fn n_components_(&self) -> PyResult<usize> {
        match &self.inner {
            AnySparseRandomProjection::F32(e) => Ok(e.n_components_()),
            AnySparseRandomProjection::F64(e) => Ok(e.n_components_()),
            _ => Err(not_fitted("sparse_random_projection", "n_components_")),
        }
    }
    /// Resolved density after fit (`None` → `1/sqrt(n_features)`); single `f64`,
    /// no dtype suffix.
    fn density_(&self) -> PyResult<f64> {
        match &self.inner {
            AnySparseRandomProjection::F32(e) => Ok(e.density_()),
            AnySparseRandomProjection::F64(e) => Ok(e.density_()),
            _ => Err(not_fitted("sparse_random_projection", "density_")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnySparseRandomProjection::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnySparseRandomProjection::Unfit { .. } => None,
            AnySparseRandomProjection::F32(_) => Some("f32"),
            AnySparseRandomProjection::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// johnson_lindenstrauss_min_dim — module-level #[pyfunction]
// ---------------------------------------------------------------------------

/// `johnson_lindenstrauss_min_dim(n_samples, eps) -> int` — the JL minimum safe
/// embedding dimension, value-matched to
/// `sklearn.random_projection.johnson_lindenstrauss_min_dim` (PROJ-01). `eps`
/// must lie in `(0, 1)`; an out-of-range `eps` raises a `PyValueError` (mapped
/// from the algos `InvalidEpsDistortion`).
#[pyfunction]
pub fn johnson_lindenstrauss_min_dim(n_samples: f64, eps: f64) -> PyResult<usize> {
    algo_jl_min_dim(n_samples, eps).map_err(algo_err_to_py)
}
