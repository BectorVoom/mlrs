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
use mlrs_algos::cluster::hdbscan::{ClusterSelectionMethod, Hdbscan, Metric};
use mlrs_algos::cluster::kmeans::KMeans;
// All three cluster wraps in this file (PyKMeans, PyDBSCAN, PyHDBSCAN) are now on
// the v3 typestate surface (consuming-self `Fit` returning the `Fitted` sibling;
// `PredictLabels` reads fitted state). The legacy trait glob is
// gone (KMeans migrated in Plan 06). `Fit` is aliased `TypestateFit` and
// `PredictLabels` `TypestatePredictLabels` and called via UFCS at the fit /
// predict sites.
use mlrs_algos::typestate::Fit as TypestateFit;
use mlrs_algos::typestate::PredictLabels as TypestatePredictLabels;

use crate::errors::{algo_err_to_py, build_err_to_py, not_fitted};
use crate::ingress::{as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype};

/// Default seed used when sklearn `random_state` is `None` (the shim passes this
/// sentinel for `random_state=None`, giving deterministic v1 behavior).
const DEFAULT_SEED: u64 = 0;

// ---------------------------------------------------------------------------
// KMeans — Fit + PredictLabels (i32); cluster_centers_, labels_, inertia_
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyKMeans,
    algo:  mlrs_algos::cluster::kmeans::KMeans,
    unfit: { n_clusters: usize, seed: u64, max_iter: usize, tol: f64 },
}

/// sklearn-compatible `KMeans` (Lloyd's algorithm, k-means++ init).
#[pyclass(name = "KMeans")]
pub struct PyKMeans {
    inner: AnyKMeans,
}

impl PyKMeans {
    /// Rust-callable default constructor for the smoke test. See
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self { inner: AnyKMeans::Unfit { n_clusters: 8, seed: DEFAULT_SEED, max_iter: 300, tol: 1e-4 } }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyKMeans::Unfit { .. })
    }
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
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = KMeans::<f32>::builder()
                        .n_clusters(n_clusters)
                        .seed(seed)
                        .max_iter(max_iter)
                        .tol(tol)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyKMeans::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = KMeans::<f64>::builder()
                        .n_clusters(n_clusters)
                        .seed(seed)
                        .max_iter(max_iter)
                        .tol(tol)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyKMeans::F64(fitted))
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
            let mut pool = crate::lock_pool();
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
        let pool = crate::lock_pool();
        match &self.inner {
            AnyKMeans::F32(e) => Ok(e.cluster_centers(&pool)),
            _ => Err(not_fitted("kmeans", "cluster_centers_ (f32)")),
        }
    }
    fn cluster_centers_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyKMeans::F64(e) => Ok(e.cluster_centers(&pool)),
            _ => Err(not_fitted("kmeans", "cluster_centers_ (f64)")),
        }
    }
    /// Fitted `labels_` (i32), either dtype arm.
    fn labels_(&self) -> PyResult<Vec<i32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyKMeans::F32(e) => Ok(e.labels(&pool)),
            AnyKMeans::F64(e) => Ok(e.labels(&pool)),
            _ => Err(not_fitted("kmeans", "labels_")),
        }
    }
    fn inertia_f32(&self) -> PyResult<f32> {
        match &self.inner {
            AnyKMeans::F32(e) => Ok(e.inertia()),
            _ => Err(not_fitted("kmeans", "inertia_ (f32)")),
        }
    }
    fn inertia_f64(&self) -> PyResult<f64> {
        match &self.inner {
            AnyKMeans::F64(e) => Ok(e.inertia()),
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

crate::any_estimator_typestate! {
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

impl PyDBSCAN {
    /// Rust-callable default constructor for the smoke test. See
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self { inner: AnyDbscan::Unfit { eps: 0.5, min_samples: 5 } }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyDbscan::Unfit { .. })
    }
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
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = DBSCAN::<f32>::builder()
                        .eps(eps)
                        .min_samples(min_samples)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyDbscan::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = DBSCAN::<f64>::builder()
                        .eps(eps)
                        .min_samples(min_samples)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyDbscan::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// Fitted `labels_` (i32, noise = -1), either dtype arm; the runtime
    /// [`not_fitted`] analog on the `Unfit` arm (D-13).
    fn labels_(&self) -> PyResult<Vec<i32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyDbscan::F32(e) => Ok(e.labels(&pool)),
            AnyDbscan::F64(e) => Ok(e.labels(&pool)),
            _ => Err(not_fitted("dbscan", "labels_")),
        }
    }
    /// Fitted `core_sample_indices_` (i32), either dtype arm; the runtime
    /// [`not_fitted`] analog on the `Unfit` arm (D-13).
    fn core_sample_indices_(&self) -> PyResult<Vec<i32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyDbscan::F32(e) => Ok(e.core_sample_indices(&pool)),
            AnyDbscan::F64(e) => Ok(e.core_sample_indices(&pool)),
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

// ---------------------------------------------------------------------------
// HDBSCAN — Fit + labels_ ONLY (labels-only, no standalone predict, algos D-08);
// the FIRST cluster-family PyO3 shell over a v3 TYPESTATE estimator (BLDR-04).
//
// Mirrors PyDBSCAN (labels-only) + the PyUMAP typestate template
// (estimators/manifold.rs): the consuming `typestate::Fit::fit` (aliased
// `TypestateFit`) returns the `Fitted`-tagged sibling stored in the F32/F64 arm,
// built via `Hdbscan::<F>::builder()...build().map_err(build_err_to_py)?` BEFORE
// the device upload (T-12-02); the `labels_` accessor returns the runtime
// `not_fitted` analog on the `Unfit` arm (D-13). Lives with the cluster family
// here (Open Question 3) — no `estimators/mod.rs` edit needed.
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyHdbscan,
    algo:  mlrs_algos::cluster::hdbscan::Hdbscan,
    unfit: {
        min_cluster_size: usize, min_samples: Option<usize>,
        cluster_selection_epsilon: f64, cluster_selection_method: String,
        metric: String, alpha: f64, max_cluster_size: usize,
    },
}

/// Parse the sklearn-named `metric` string into the algos [`Metric`] enum. Only
/// `"euclidean"` carries meaning in the Phase-12 shell.
fn parse_hdbscan_metric(s: &str) -> PyResult<Metric> {
    match s {
        "euclidean" => Ok(Metric::Euclidean),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hdbscan: unsupported metric {other:?}; expected \"euclidean\""
        ))),
    }
}

/// Parse the sklearn-named `cluster_selection_method` string into the algos
/// [`ClusterSelectionMethod`] enum.
fn parse_cluster_selection_method(s: &str) -> PyResult<ClusterSelectionMethod> {
    match s {
        "eom" => Ok(ClusterSelectionMethod::Eom),
        "leaf" => Ok(ClusterSelectionMethod::Leaf),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hdbscan: unsupported cluster_selection_method {other:?}; \
             expected \"eom\" or \"leaf\""
        ))),
    }
}

/// sklearn-compatible `HDBSCAN` (density-based clustering). Labels-only — `fit` +
/// `labels_`, NO standalone `predict` (algos D-08). The v3 typestate estimator
/// collapses behind the same `Unfit/F32/F64` enum the legacy shells use (BLDR-04).
#[pyclass(name = "HDBSCAN")]
pub struct PyHDBSCAN {
    inner: AnyHdbscan,
}

impl PyHDBSCAN {
    /// Rust-callable default constructor (cross-crate smoke seam). Mirrors the
    /// `#[new]` defaults (sklearn defaults). See
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyHdbscan::Unfit {
                min_cluster_size: 5,
                min_samples: None,
                cluster_selection_epsilon: 0.0,
                cluster_selection_method: "eom".to_string(),
                metric: "euclidean".to_string(),
                alpha: 1.0,
                max_cluster_size: 0,
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyHdbscan::Unfit { .. })
    }

    /// Rust-callable `labels_` accessor for the cross-crate not-fitted test (the
    /// live PyO3 boundary path runs in UAT, MEMORY). Returns the [`not_fitted`]
    /// analog on the `Unfit` arm.
    pub fn labels_for_test(&self) -> PyResult<Vec<i32>> {
        self.labels_inner()
    }

    fn labels_inner(&self) -> PyResult<Vec<i32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHdbscan::F32(e) => Ok(e.labels(&pool)),
            AnyHdbscan::F64(e) => Ok(e.labels(&pool)),
            _ => Err(not_fitted("hdbscan", "labels_")),
        }
    }
}

#[pymethods]
impl PyHDBSCAN {
    /// `HDBSCAN(min_cluster_size=5, min_samples=None,
    /// cluster_selection_epsilon=0.0, cluster_selection_method="eom",
    /// metric="euclidean", alpha=1.0, max_cluster_size=0)`.
    #[new]
    #[pyo3(signature = (
        min_cluster_size = 5, min_samples = None, cluster_selection_epsilon = 0.0,
        cluster_selection_method = "eom".to_string(), metric = "euclidean".to_string(),
        alpha = 1.0, max_cluster_size = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        min_cluster_size: usize,
        min_samples: Option<usize>,
        cluster_selection_epsilon: f64,
        cluster_selection_method: String,
        metric: String,
        alpha: f64,
        max_cluster_size: usize,
    ) -> Self {
        Self {
            inner: AnyHdbscan::Unfit {
                min_cluster_size,
                min_samples,
                cluster_selection_epsilon,
                cluster_selection_method,
                metric,
                alpha,
                max_cluster_size,
            },
        }
    }

    /// Fit on `x` (`rows × cols`, row-major). Unsupervised — no `y`. The
    /// data-INDEPENDENT hyperparameters are validated at `build()` BEFORE the
    /// device upload (`build_err_to_py` → `ValueError`, T-12-02); GIL released
    /// (PY-03); f64 guarded (D-04 / T-12-07). The consuming `TypestateFit::fit`
    /// returns the `Fitted`-tagged sibling.
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (
            min_cluster_size, min_samples, cluster_selection_epsilon,
            csm_s, metric_s, alpha, max_cluster_size,
        ) = match &self.inner {
            AnyHdbscan::Unfit {
                min_cluster_size, min_samples, cluster_selection_epsilon,
                cluster_selection_method, metric, alpha, max_cluster_size,
            } => (
                *min_cluster_size, *min_samples, *cluster_selection_epsilon,
                cluster_selection_method.clone(), metric.clone(), *alpha,
                *max_cluster_size,
            ),
            _ => return Err(not_fitted("hdbscan", "re-fit")),
        };
        // Construction-time enum-string validation (→ ValueError).
        let cluster_selection_method = parse_cluster_selection_method(&csm_s)?;
        let metric = parse_hdbscan_metric(&metric_s)?;
        let fitted = py.detach(|| -> PyResult<AnyHdbscan> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = Hdbscan::<f32>::builder()
                        .min_cluster_size(min_cluster_size)
                        .min_samples(min_samples)
                        .cluster_selection_epsilon(cluster_selection_epsilon)
                        .cluster_selection_method(cluster_selection_method)
                        .metric(metric)
                        .alpha(alpha)
                        .max_cluster_size(max_cluster_size)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyHdbscan::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = Hdbscan::<f64>::builder()
                        .min_cluster_size(min_cluster_size)
                        .min_samples(min_samples)
                        .cluster_selection_epsilon(cluster_selection_epsilon)
                        .cluster_selection_method(cluster_selection_method)
                        .metric(metric)
                        .alpha(alpha)
                        .max_cluster_size(max_cluster_size)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyHdbscan::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// `fit_predict(X)` (`rows × cols`, row-major) — fit on `x` then return the
    /// fitted `labels_` (i32, noise = -1), sklearn `ClusterMixin.fit_predict`
    /// semantics. Mutates `self` into the `Fitted` arm (so a subsequent
    /// `labels_`/`probabilities_`/`outlier_scores_` reads the same fit). Same GIL /
    /// guard / lock contract as `fit`: data-INDEPENDENT hyperparameters validated at
    /// `build()` BEFORE the device upload (T-12-02); GIL released (PY-03);
    /// `guard_f64()` BEFORE the F64 upload (T-16-GUARDF64); `lock_pool()`
    /// poison-recovering (T-16-POISON).
    fn fit_predict(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<Vec<i32>> {
        self.fit(py, x, rows, cols)?;
        py.detach(|| {
            let pool = crate::lock_pool();
            match &self.inner {
                AnyHdbscan::F32(e) => Ok(e.labels(&pool)),
                AnyHdbscan::F64(e) => Ok(e.labels(&pool)),
                _ => Err(not_fitted("hdbscan", "fit_predict")),
            }
        })
    }

    /// Fitted `labels_` (i32, noise = -1), either dtype arm; the runtime
    /// [`not_fitted`] analog on the `Unfit` arm (D-13).
    fn labels_(&self) -> PyResult<Vec<i32>> {
        self.labels_inner()
    }

    /// Fitted per-point membership `probabilities_` (f32 arm, length `n`, in
    /// `[0, 1]`). `None` until the feature-space probability front-end lands
    /// (algos plan 15-05) — surfaces as Python `None`. The runtime [`not_fitted`]
    /// analog on the `Unfit`/wrong-dtype arm (D-13).
    fn probabilities_f32(&self) -> PyResult<Option<Vec<f32>>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHdbscan::F32(e) => Ok(e.probabilities(&pool)),
            _ => Err(not_fitted("hdbscan", "probabilities_ (f32)")),
        }
    }
    /// Fitted per-point membership `probabilities_` (f64 arm) or the [`not_fitted`]
    /// analog.
    fn probabilities_f64(&self) -> PyResult<Option<Vec<f64>>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHdbscan::F64(e) => Ok(e.probabilities(&pool)),
            _ => Err(not_fitted("hdbscan", "probabilities_ (f64)")),
        }
    }

    /// Fitted per-point GLOSH `outlier_scores_` (f32 arm, length `n`, in `[0, 1]`;
    /// HDBS-03). `Some` after any successful fit; the runtime [`not_fitted`] analog
    /// on the `Unfit`/wrong-dtype arm (D-13).
    fn outlier_scores_f32(&self) -> PyResult<Option<Vec<f32>>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHdbscan::F32(e) => Ok(e.outlier_scores(&pool)),
            _ => Err(not_fitted("hdbscan", "outlier_scores_ (f32)")),
        }
    }
    /// Fitted per-point GLOSH `outlier_scores_` (f64 arm) or the [`not_fitted`]
    /// analog.
    fn outlier_scores_f64(&self) -> PyResult<Option<Vec<f64>>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHdbscan::F64(e) => Ok(e.outlier_scores(&pool)),
            _ => Err(not_fitted("hdbscan", "outlier_scores_ (f64)")),
        }
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyHdbscan::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyHdbscan::Unfit { .. } => None,
            AnyHdbscan::F32(_) => Some("f32"),
            AnyHdbscan::F64(_) => Some("f64"),
        }
    }
}
