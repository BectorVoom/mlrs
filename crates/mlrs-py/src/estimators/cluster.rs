//! Clustering `#[pyclass]` wrappers (PY-01/PY-02/PY-05): `PyKMeans`, `PyDBSCAN`.
//!
//! `KMeans` is `Fit` + [`PredictLabels`] (i32 cluster ids) with the
//! `cluster_centers_` / `labels_` / `inertia_` fitted surface; its sklearn
//! `random_state` maps to the Rust `seed` (`None` → a fixed default). `DBSCAN` is
//! `Fit` + the `labels_` fitted attribute only — it has NO standalone `predict`
//! (algos D-08; sklearn `DBSCAN` likewise exposes only `fit_predict`/`labels_`),
//! and `eps` stays `f64` regardless of the input float dtype.

use pyo3::prelude::*;

use mlrs_algos::cluster::dbscan::DBSCAN;
use mlrs_algos::cluster::kmeans::KMeans;
use mlrs_algos::traits::{Fit, PredictLabels};

use crate::errors::{algo_err_to_py, not_fitted};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype};

/// Default seed used when sklearn `random_state` is `None` (the shim passes this
/// sentinel for `random_state=None`, giving deterministic v1 behavior).
const DEFAULT_SEED: u64 = 0;

// ---------------------------------------------------------------------------
// KMeans — Fit + PredictLabels (i32); cluster_centers_, labels_, inertia_
// ---------------------------------------------------------------------------

crate::any_estimator! {
    any:   AnyKMeans,
    algo:  mlrs_algos::cluster::kmeans::KMeans,
    unfit: { n_clusters: usize, seed: u64, max_iter: usize, tol: f64 },
}

/// sklearn-compatible `KMeans` (Lloyd's algorithm, k-means++ init).
#[pyclass(name = "KMeans")]
pub struct PyKMeans {
    inner: AnyKMeans,
}

#[pymethods]
impl PyKMeans {
    /// `KMeans(n_clusters=8, max_iter=300, tol=1e-4, random_state=None)`. The
    /// sklearn `random_state` is mapped to the Rust `seed`; `None` → a fixed
    /// default seed (deterministic v1).
    #[new]
    #[pyo3(signature = (n_clusters = 8, max_iter = 300, tol = 1e-4, random_state = None))]
    fn new(n_clusters: usize, max_iter: usize, tol: f64, random_state: Option<u64>) -> Self {
        Self {
            inner: AnyKMeans::Unfit {
                n_clusters,
                seed: random_state.unwrap_or(DEFAULT_SEED),
                max_iter,
                tol,
            },
        }
    }

    /// Fit on `x` (`rows × cols`). Unsupervised — no `y`. GIL released (PY-03);
    /// f64 guarded on an f64-incapable backend (D-04).
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (n_clusters, seed, max_iter, tol) = match &self.inner {
            AnyKMeans::Unfit { n_clusters, seed, max_iter, tol } => (*n_clusters, *seed, *max_iter, *tol),
            _ => (8, DEFAULT_SEED, 300, 1e-4),
        };
        let fitted = py.detach(|| -> PyResult<AnyKMeans> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let mut est = KMeans::<f32>::with_opts(n_clusters, seed, max_iter, tol);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyKMeans::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let mut est = KMeans::<f64>::with_opts(n_clusters, seed, max_iter, tol);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyKMeans::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `predict(x)` → length-`rows` host `Vec<i32>` cluster ids (D-06).
    fn predict_labels(&self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<i32>> {
        let xa = capsule_to_array(x)?;
        py.detach(|| {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match &self.inner {
                AnyKMeans::F32(est) => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    Ok(est.predict_labels(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                AnyKMeans::F64(est) => {
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    Ok(est.predict_labels(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool))
                }
                _ => Err(not_fitted("kmeans", "predict")),
            }
        })
    }

    fn cluster_centers_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyKMeans::F32(e) => e.cluster_centers(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("kmeans", "cluster_centers_ (f32)")),
        }
    }
    fn cluster_centers_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyKMeans::F64(e) => e.cluster_centers(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("kmeans", "cluster_centers_ (f64)")),
        }
    }
    /// Fitted `labels_` (i32), either dtype arm.
    fn labels_(&self) -> PyResult<Vec<i32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyKMeans::F32(e) => e.labels(&pool).map_err(algo_err_to_py),
            AnyKMeans::F64(e) => e.labels(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("kmeans", "labels_")),
        }
    }
    fn inertia_f32(&self) -> PyResult<f32> {
        match &self.inner {
            AnyKMeans::F32(e) => e.inertia().map_err(algo_err_to_py),
            _ => Err(not_fitted("kmeans", "inertia_ (f32)")),
        }
    }
    fn inertia_f64(&self) -> PyResult<f64> {
        match &self.inner {
            AnyKMeans::F64(e) => e.inertia().map_err(algo_err_to_py),
            _ => Err(not_fitted("kmeans", "inertia_ (f64)")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyKMeans::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyKMeans::Unfit { .. } => None,
            AnyKMeans::F32(_) => Some("f32"),
            AnyKMeans::F64(_) => Some("f64"),
        }
    }
}

// ---------------------------------------------------------------------------
// DBSCAN — Fit + labels_ ONLY (no standalone predict, algos D-08); eps is f64
// ---------------------------------------------------------------------------

crate::any_estimator! {
    any:   AnyDbscan,
    algo:  mlrs_algos::cluster::dbscan::DBSCAN,
    unfit: { eps: f64, min_samples: usize },
}

/// sklearn-compatible `DBSCAN`. `eps` stays `f64` regardless of the input float
/// dtype. DBSCAN has no standalone `predict` — only `fit` + `labels_`.
#[pyclass(name = "DBSCAN")]
pub struct PyDBSCAN {
    inner: AnyDbscan,
}

#[pymethods]
impl PyDBSCAN {
    /// `DBSCAN(eps=0.5, min_samples=5)`.
    #[new]
    #[pyo3(signature = (eps = 0.5, min_samples = 5))]
    fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            inner: AnyDbscan::Unfit { eps, min_samples },
        }
    }

    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (eps, min_samples) = match &self.inner {
            AnyDbscan::Unfit { eps, min_samples } => (*eps, *min_samples),
            _ => (0.5, 5),
        };
        let fitted = py.detach(|| -> PyResult<AnyDbscan> {
            let mut pool = crate::global_pool().lock().expect("pool mutex");
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let mut est = DBSCAN::<f32>::new(eps, min_samples);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyDbscan::F32(est))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let mut est = DBSCAN::<f64>::new(eps, min_samples);
                    est.fit(&mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyDbscan::F64(est))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// Fitted `labels_` (i32, noise = -1), either dtype arm.
    fn labels_(&self) -> PyResult<Vec<i32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyDbscan::F32(e) => e.labels(&pool).map_err(algo_err_to_py),
            AnyDbscan::F64(e) => e.labels(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("dbscan", "labels_")),
        }
    }
    /// Fitted `core_sample_indices_` (i32), either dtype arm.
    fn core_sample_indices_(&self) -> PyResult<Vec<i32>> {
        let pool = crate::global_pool().lock().expect("pool mutex");
        match &self.inner {
            AnyDbscan::F32(e) => e.core_sample_indices(&pool).map_err(algo_err_to_py),
            AnyDbscan::F64(e) => e.core_sample_indices(&pool).map_err(algo_err_to_py),
            _ => Err(not_fitted("dbscan", "core_sample_indices_")),
        }
    }
    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyDbscan::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyDbscan::Unfit { .. } => None,
            AnyDbscan::F32(_) => Some("f32"),
            AnyDbscan::F64(_) => Some("f64"),
        }
    }
}
