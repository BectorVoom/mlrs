//! Clustering `#[pyclass]` wrappers (PY-01/PY-02/PY-05): `PyKMeans`, `PyDBSCAN`.
//!
//! `KMeans` is `Fit` + [`PredictLabels`] (i32 cluster ids) with the
//! `cluster_centers_` / `labels_` / `inertia_` / `n_iter_` fitted surface and
//! sklearn's FULL ctor parameter set (`init` / `n_init` / `algorithm` /
//! `verbose` / `random_state` / `copy_x`); `random_state=None` maps to the
//! algos layer's deterministic default seed. `DBSCAN` is
//! `Fit` + the `labels_` fitted attribute only — it has NO standalone `predict`
//! (algos D-08; sklearn `DBSCAN` likewise exposes only `fit_predict`/`labels_`),
//! and `eps` stays `f64` regardless of the input float dtype.

use pyo3::prelude::*;

use mlrs_algos::cluster::dbscan::DBSCAN;
use mlrs_algos::cluster::hdbscan::{
    Algorithm, ClusterSelectionMethod, Hdbscan, Metric, StoreCenters,
};
use mlrs_algos::cluster::kmeans::{KMeans, KMeansAlgorithm, KMeansInit, NInit};
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

// ---------------------------------------------------------------------------
// KMeans — Fit + PredictLabels (i32); cluster_centers_, labels_, inertia_
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyKMeans,
    algo:  mlrs_algos::cluster::kmeans::KMeans,
    unfit: { n_clusters: usize },
}

crate::impl_persistable_any! {
    any:  AnyKMeans,
    algo: mlrs_algos::cluster::kmeans::KMeans,
    name: "kmeans",
}

/// The verbatim sklearn ctor hyperparameters (WR-02: the typestate wrapper
/// rebuilds a fresh `Unfit` from THESE at every `fit`, so a second `fit` of the
/// same Python object works — reading them back out of the `Unfit` enum arm
/// would lose them the moment the first fit consumed it). Mirrors
/// [`crate::estimators::linear::RidgeParams`]; `init` / `n_init` / `algorithm`
/// are already parsed into their typed enums (once, at `#[new]`), so an
/// unrecognised string is rejected at CONSTRUCTION rather than at first fit —
/// matching sklearn, whose `StrOptions` validation also fires before any work.
#[derive(Clone)]
struct KMeansParams {
    n_clusters: usize,
    init: KMeansInit<f64>,
    n_init: NInit,
    max_iter: usize,
    tol: f64,
    verbose: bool,
    random_state: Option<u64>,
    copy_x: bool,
    algorithm: KMeansAlgorithm,
}

/// sklearn-compatible `KMeans` (Lloyd / Elkan, k-means++ / random / explicit
/// init).
#[pyclass(name = "KMeans")]
pub struct PyKMeans {
    inner: AnyKMeans,
    /// The ctor hyperparameters, re-read at every `fit` (WR-02).
    params: KMeansParams,
    /// sklearn's `n_iter_`, captured at `fit` (the fitted arms are consumed
    /// into `AnyKMeans` and a `#[pyclass]` getter cannot reach through the
    /// dtype dispatch generically, so the scalar is mirrored here — the
    /// `PyRidge::n_iter` idiom).
    n_iter: Option<usize>,
}

impl PyKMeans {
    /// Rust-callable default constructor for the smoke test. See
    /// [`crate::estimators::linear::PyLinearRegression::unfit_default`].
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyKMeans::Unfit { n_clusters: 8 },
            params: KMeansParams {
                n_clusters: 8,
                init: KMeansInit::KMeansPlusPlus,
                n_init: NInit::Auto,
                max_iter: 300,
                tol: 1e-4,
                verbose: false,
                random_state: None,
                copy_x: true,
                algorithm: KMeansAlgorithm::Lloyd,
            },
            n_iter: None,
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyKMeans::Unfit { .. })
    }
}

/// Build an unfit `KMeans<F>` from the ctor params. Monomorphized per float
/// width by the macro (the `ridge_build!` precedent — `mlrs-py` does not depend
/// on `cubecl`, so it cannot spell the `Float + CubeElement + Pod` bound a
/// generic `fn` would need), so the nine builder setters are written once.
macro_rules! kmeans_build {
    ($float:ty, $p:expr) => {
        KMeans::<$float>::builder()
            .n_clusters($p.n_clusters)
            .init_method($p.init.clone())
            .n_init($p.n_init)
            .max_iter($p.max_iter)
            .tol($p.tol)
            .verbose($p.verbose)
            .random_state($p.random_state)
            .copy_x($p.copy_x)
            .algorithm($p.algorithm)
            .build::<$float>()
            .map_err(build_err_to_py)
    };
}

#[pymethods]
impl PyKMeans {
    /// `KMeans(n_clusters=8, init='k-means++', init_array=None, n_init=None,
    /// max_iter=300, tol=1e-4, verbose=False, random_state=None, copy_x=True,
    /// algorithm='lloyd')` — sklearn's full ctor surface.
    ///
    /// The shim splits sklearn's polymorphic `init` into TWO arguments because
    /// PyO3 cannot express "a string, an array, or a callable" as one typed
    /// parameter: the string strategies arrive in `init`, and an explicit
    /// array (or the flattened result of a callable, which the shim evaluates)
    /// arrives in `init_array` — which, when present, WINS. That is exactly
    /// sklearn's own precedence: `_init_centroids` checks the array form
    /// before the strings.
    ///
    /// `n_init` is `Option<usize>`: `None` is sklearn's `'auto'`. The shim
    /// still parses the string itself so a bad one (`n_init='ten'`) is
    /// rejected with the same message the Rust `NInit::try_from` produces.
    #[new]
    #[pyo3(signature = (
        n_clusters = 8,
        init = "k-means++",
        init_array = None,
        n_init = None,
        max_iter = 300,
        tol = 1e-4,
        verbose = false,
        random_state = None,
        copy_x = true,
        algorithm = "lloyd",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: usize,
        init: &str,
        init_array: Option<Vec<f64>>,
        n_init: Option<usize>,
        max_iter: usize,
        tol: f64,
        verbose: bool,
        random_state: Option<u64>,
        copy_x: bool,
        algorithm: &str,
    ) -> PyResult<Self> {
        // An explicit array wins over the string (sklearn's precedence); only
        // when there is none does the `init` string have to parse.
        let init = match init_array {
            Some(a) => KMeansInit::Array(a),
            None => KMeansInit::try_from(init).map_err(build_err_to_py)?,
        };
        let n_init = match n_init {
            Some(v) => NInit::Fixed(v),
            None => NInit::Auto,
        };
        let algorithm = KMeansAlgorithm::try_from(algorithm).map_err(build_err_to_py)?;
        let params = KMeansParams {
            n_clusters,
            init,
            n_init,
            max_iter,
            tol,
            verbose,
            random_state,
            copy_x,
            algorithm,
        };
        // Surface a data-INDEPENDENT rejection (n_init < 1) at CONSTRUCTION,
        // not at first fit — sklearn's `Interval` validation fires there too.
        // The probe build is f32 only: the rejection does not depend on `F`.
        kmeans_build!(f32, params)?;
        Ok(Self {
            inner: AnyKMeans::Unfit { n_clusters },
            params,
            n_iter: None,
        })
    }

    /// Fit on `x` (`rows × cols`). Unsupervised — no `y`. GIL released (PY-03);
    /// f64 guarded on an f64-incapable backend (D-04).
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let params = self.params.clone();
        let (fitted, n_iter) = py.detach(|| -> PyResult<(AnyKMeans, usize)> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = kmeans_build!(f32, params)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    let n_iter = fitted.n_iter();
                    Ok((AnyKMeans::F32(fitted), n_iter))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = kmeans_build!(f64, params)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    let n_iter = fitted.n_iter();
                    Ok((AnyKMeans::F64(fitted), n_iter))
                }
            }
        })?;
        self.inner = fitted;
        self.n_iter = Some(n_iter);
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
    /// sklearn's `n_iter_` — the WINNING restart's iteration count.
    fn n_iter_(&self) -> PyResult<usize> {
        self.n_iter.ok_or_else(|| not_fitted("kmeans", "n_iter_"))
    }
    /// The `algorithm` that actually ran, after the `k == 1` elkan → lloyd
    /// degradation (mlrs's analogue of `Ridge.solver_`; sklearn keeps this in
    /// its private `_algorithm`).
    fn algorithm_used(&self) -> &'static str {
        self.params.algorithm.resolve(self.params.n_clusters).name()
    }
    /// The number of restarts that actually ran, after `n_init='auto'`
    /// resolution and the explicit-array override (sklearn's `_n_init`).
    fn n_init_used(&self) -> usize {
        self.params.n_init.resolve(&self.params.init)
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
// DBSCAN — Fit + labels_ ONLY (no standalone predict, algos D-08); eps is f64
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyDbscan,
    algo:  mlrs_algos::cluster::dbscan::DBSCAN,
    unfit: { eps: f64, min_samples: usize },
}

crate::impl_persistable_any! {
    any:  AnyDbscan,
    algo: mlrs_algos::cluster::dbscan::DBSCAN,
    name: "dbscan",
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
        minkowski_p: f64, algorithm: String, leaf_size: usize,
        n_jobs: Option<i32>, allow_single_cluster: bool,
        store_centers: Option<String>,
    },
}

crate::impl_persistable_any! {
    any:  AnyHdbscan,
    algo: mlrs_algos::cluster::hdbscan::Hdbscan,
    name: "hdbscan",
}

/// Parse the sklearn-named `metric` string (plus `minkowski_p`, which the shim
/// has already pulled out of sklearn's `metric_params` dict) into the algos
/// [`Metric`] enum.
///
/// The alias groups are sklearn's own: `l2` is `euclidean`, `l1`/`cityblock` are
/// `manhattan`, `infinity` is `chebyshev`, and `p` is `minkowski` — those pairs
/// name the same distance in `sklearn.neighbors`' metric tables, so accepting
/// them costs nothing and lets a caller paste an sklearn snippet unchanged.
///
/// sklearn's remaining `metric` options (the boolean-vector family —
/// `braycurtis`, `dice`, `jaccard`, `yule`, … — plus `mahalanobis`,
/// `seuclidean`, `correlation`, `haversine` and the `pyfunc` escape hatch) are
/// NOT part of the mlrs distance core, so they are rejected here with the
/// supported list rather than silently coerced to something else.
fn parse_hdbscan_metric(s: &str, minkowski_p: f64) -> PyResult<Metric> {
    match s {
        "euclidean" | "l2" => Ok(Metric::Euclidean),
        "manhattan" | "cityblock" | "l1" => Ok(Metric::Manhattan),
        "chebyshev" | "infinity" => Ok(Metric::Chebyshev),
        "minkowski" | "p" => Ok(Metric::Minkowski { p: minkowski_p }),
        "cosine" => Ok(Metric::Cosine),
        "precomputed" => Ok(Metric::Precomputed),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hdbscan: unsupported metric {other:?}; expected one of \"euclidean\", \
             \"l2\", \"manhattan\", \"cityblock\", \"l1\", \"chebyshev\", \
             \"infinity\", \"minkowski\", \"p\", \"cosine\", \"precomputed\""
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

/// Parse the sklearn-named `algorithm` string into the algos [`Algorithm`] enum.
///
/// This is a WALL-CLOCK knob only — see [`Algorithm`] for why every value
/// produces identical labels.
fn parse_hdbscan_algorithm(s: &str) -> PyResult<Algorithm> {
    match s {
        "auto" => Ok(Algorithm::Auto),
        "brute" => Ok(Algorithm::Brute),
        "kd_tree" => Ok(Algorithm::KdTree),
        "ball_tree" => Ok(Algorithm::BallTree),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hdbscan: unsupported algorithm {other:?}; expected \"auto\", \
             \"brute\", \"kd_tree\" or \"ball_tree\""
        ))),
    }
}

/// Parse the sklearn-named `store_centers` string (`None` = store neither) into
/// the algos [`StoreCenters`] enum.
fn parse_store_centers(s: Option<&str>) -> PyResult<Option<StoreCenters>> {
    match s {
        None => Ok(None),
        Some("centroid") => Ok(Some(StoreCenters::Centroid)),
        Some("medoid") => Ok(Some(StoreCenters::Medoid)),
        Some("both") => Ok(Some(StoreCenters::Both)),
        Some(other) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hdbscan: unsupported store_centers {other:?}; expected None, \
             \"centroid\", \"medoid\" or \"both\""
        ))),
    }
}

// NOTE — a deliberate, documented divergence (`docs/upstream-sklearn-issues.md`,
// SK-001): `algorithm='brute'` with `metric='infinity'` or `metric='p'` is
// ACCEPTED here, where sklearn 1.9.0 raises.
//
// sklearn's `metric` constraint is the UNION of the tree metrics and the
// pairwise metrics, but its brute path routes through `pairwise_distances`,
// whose table carries `chebyshev`/`minkowski` and NOT those two tree-only
// aliases — so four metrics (`infinity`, `p`, `pyfunc`, `sokalmichener`) are
// accepted by its own validation and then rejected mid-`fit` by a helper,
// with an error naming `pairwise_distances`' parameter rather than HDBSCAN's.
// Its `kd_tree`/`ball_tree` paths validate the metric properly and raise a
// clear message; only `brute` lacks the check, and `algorithm='auto'` hides it
// by routing those metrics to a tree.
//
// This shim originally mirrored the rejection for drop-in parity. That was the
// wrong call: parity is worth having with sklearn's SEMANTICS, not with a gap
// in its validation. mlrs has no such asymmetry — `infinity` and `p` resolve to
// the same `Metric` on every route, and `Algorithm` is value-neutral by
// construction (see `hdbscan::Algorithm`) — so refusing the pair would have
// meant inventing a restriction the engine does not have, purely to reproduce
// someone else's bug. Reported upstream; accepted here.
//
// The tree-metric rejections are a different matter and are KEPT (in
// `HdbscanBuilder::build`): a KD/ball box bound requires a per-axis-monotone
// distance, so `kd_tree`/`ball_tree` with `cosine`/`precomputed` is a real
// restriction of the algorithm, not a validation oversight.

/// The ctor hyperparameters, lifted out of the `Unfit` arm so `fit` reads them
/// once instead of threading a thirteen-element tuple through the dtype match.
/// Mirrors [`crate::estimators::linear::RidgeClassifierParams`].
struct HdbscanParams {
    min_cluster_size: usize,
    min_samples: Option<usize>,
    cluster_selection_epsilon: f64,
    cluster_selection_method: String,
    metric: String,
    alpha: f64,
    max_cluster_size: usize,
    minkowski_p: f64,
    algorithm: String,
    leaf_size: usize,
    n_jobs: Option<i32>,
    allow_single_cluster: bool,
    store_centers: Option<String>,
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
                minkowski_p: 2.0,
                algorithm: "auto".to_string(),
                leaf_size: 40,
                n_jobs: None,
                allow_single_cluster: false,
                store_centers: None,
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
    /// The full sklearn `HDBSCAN` hyperparameter surface, flattened for the
    /// boundary: sklearn's `metric_params` dict arrives as the single scalar the
    /// mlrs metric core reads from it (`minkowski_p`), `max_cluster_size=None`
    /// arrives as the `0`-means-unbounded sentinel, and `copy` is a pure
    /// Python-side concern the shim handles before it gets here. Everything else
    /// keeps sklearn's name, order and default.
    #[new]
    #[pyo3(signature = (
        min_cluster_size = 5, min_samples = None, cluster_selection_epsilon = 0.0,
        max_cluster_size = 0, metric = "euclidean".to_string(), minkowski_p = 2.0,
        alpha = 1.0, algorithm = "auto".to_string(), leaf_size = 40, n_jobs = None,
        cluster_selection_method = "eom".to_string(), allow_single_cluster = false,
        store_centers = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        min_cluster_size: usize,
        min_samples: Option<usize>,
        cluster_selection_epsilon: f64,
        max_cluster_size: usize,
        metric: String,
        minkowski_p: f64,
        alpha: f64,
        algorithm: String,
        leaf_size: usize,
        n_jobs: Option<i32>,
        cluster_selection_method: String,
        allow_single_cluster: bool,
        store_centers: Option<String>,
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
                minkowski_p,
                algorithm,
                leaf_size,
                n_jobs,
                allow_single_cluster,
                store_centers,
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
        let p = match &self.inner {
            AnyHdbscan::Unfit {
                min_cluster_size, min_samples, cluster_selection_epsilon,
                cluster_selection_method, metric, alpha, max_cluster_size,
                minkowski_p, algorithm, leaf_size, n_jobs, allow_single_cluster,
                store_centers,
            } => HdbscanParams {
                min_cluster_size: *min_cluster_size,
                min_samples: *min_samples,
                cluster_selection_epsilon: *cluster_selection_epsilon,
                cluster_selection_method: cluster_selection_method.clone(),
                metric: metric.clone(),
                alpha: *alpha,
                max_cluster_size: *max_cluster_size,
                minkowski_p: *minkowski_p,
                algorithm: algorithm.clone(),
                leaf_size: *leaf_size,
                n_jobs: *n_jobs,
                allow_single_cluster: *allow_single_cluster,
                store_centers: store_centers.clone(),
            },
            _ => return Err(not_fitted("hdbscan", "re-fit")),
        };
        // Construction-time enum-string validation (→ ValueError), BEFORE the
        // device upload (T-12-02). Every accepted `metric` string works on every
        // `algorithm` that has a route for the distance it resolves to — see the
        // divergence note above `parse_hdbscan_metric` for the one sklearn
        // rejection deliberately NOT reproduced here.
        let cluster_selection_method = parse_cluster_selection_method(&p.cluster_selection_method)?;
        let metric = parse_hdbscan_metric(&p.metric, p.minkowski_p)?;
        let algorithm = parse_hdbscan_algorithm(&p.algorithm)?;
        let store_centers = parse_store_centers(p.store_centers.as_deref())?;
        // One builder chain per dtype arm (the estimator is generic over F, so the
        // two monomorphizations cannot share a value). `hdbscan_build!` keeps them
        // from drifting apart as the surface grows.
        macro_rules! hdbscan_build {
            ($f:ty) => {
                Hdbscan::<$f>::builder()
                    .min_cluster_size(p.min_cluster_size)
                    .min_samples(p.min_samples)
                    .cluster_selection_epsilon(p.cluster_selection_epsilon)
                    .cluster_selection_method(cluster_selection_method)
                    .metric(metric)
                    .alpha(p.alpha)
                    .max_cluster_size(p.max_cluster_size)
                    .algorithm(algorithm)
                    .leaf_size(p.leaf_size)
                    .n_jobs(p.n_jobs)
                    .allow_single_cluster(p.allow_single_cluster)
                    .store_centers(store_centers)
                    .build::<$f>()
                    .map_err(build_err_to_py)?
            };
        }
        let fitted = py.detach(|| -> PyResult<AnyHdbscan> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = hdbscan_build!(f32);
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols)).map_err(algo_err_to_py)?;
                    Ok(AnyHdbscan::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = hdbscan_build!(f64);
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

    /// Fitted cluster `centroids_` (f32 arm) as a FLAT row-major
    /// `n_clusters × n_features` block — the shim reshapes. `None` unless
    /// `store_centers` requested centroids AND the fit produced a cluster
    /// (HDBS-04). The runtime [`not_fitted`] analog on the `Unfit`/wrong-dtype arm.
    fn centroids_f32(&self) -> PyResult<Option<Vec<f32>>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHdbscan::F32(e) => Ok(e.centroids(&pool)),
            _ => Err(not_fitted("hdbscan", "centroids_ (f32)")),
        }
    }
    /// Fitted cluster `centroids_` (f64 arm) or the [`not_fitted`] analog.
    fn centroids_f64(&self) -> PyResult<Option<Vec<f64>>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHdbscan::F64(e) => Ok(e.centroids(&pool)),
            _ => Err(not_fitted("hdbscan", "centroids_ (f64)")),
        }
    }
    /// Fitted cluster `medoids_` (f32 arm), same shape and contract as
    /// [`Self::centroids_f32`].
    fn medoids_f32(&self) -> PyResult<Option<Vec<f32>>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHdbscan::F32(e) => Ok(e.medoids(&pool)),
            _ => Err(not_fitted("hdbscan", "medoids_ (f32)")),
        }
    }
    /// Fitted cluster `medoids_` (f64 arm) or the [`not_fitted`] analog.
    fn medoids_f64(&self) -> PyResult<Option<Vec<f64>>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyHdbscan::F64(e) => Ok(e.medoids(&pool)),
            _ => Err(not_fitted("hdbscan", "medoids_ (f64)")),
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
// AgglomerativeClustering (AGGLO-01) — Fit + labels_/children_ ONLY (no
// standalone predict; sklearn's AgglomerativeClustering is likewise
// fit/fit_predict-only). Mirrors PyDBSCAN over the v3 typestate estimator.
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnyAgglomerative,
    algo:  mlrs_algos::cluster::agglomerative::AgglomerativeClustering,
    unfit: { n_clusters: usize, metric: String },
}

crate::impl_persistable_any! {
    any:  AnyAgglomerative,
    algo: mlrs_algos::cluster::agglomerative::AgglomerativeClustering,
    name: "agglomerative_clustering",
}

/// Parse the sklearn-named `metric` string into the algos agglomerative
/// [`Metric`](mlrs_algos::cluster::agglomerative::Metric) enum. The sklearn/cuML
/// aliases collapse: `'l2'` → Euclidean, `'l1'` → Manhattan.
fn parse_agglomerative_metric(
    s: &str,
) -> PyResult<mlrs_algos::cluster::agglomerative::Metric> {
    use mlrs_algos::cluster::agglomerative::Metric as AggloMetric;
    match s {
        "euclidean" | "l2" => Ok(AggloMetric::Euclidean),
        "manhattan" | "l1" | "cityblock" => Ok(AggloMetric::Manhattan),
        "cosine" => Ok(AggloMetric::Cosine),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "agglomerative_clustering: unsupported metric {other:?}; expected one of \
             \"euclidean\"/\"l2\", \"manhattan\"/\"l1\"/\"cityblock\", \"cosine\""
        ))),
    }
}

/// sklearn-compatible `AgglomerativeClustering` (single-linkage only — the cuML
/// scope). `fit` + `labels_`/`children_`, NO standalone `predict`.
#[pyclass(name = "AgglomerativeClustering")]
pub struct PyAgglomerativeClustering {
    inner: AnyAgglomerative,
}

impl PyAgglomerativeClustering {
    /// Rust-callable default constructor for the smoke test.
    pub fn unfit_default() -> Self {
        Self {
            inner: AnyAgglomerative::Unfit {
                n_clusters: 2,
                metric: "euclidean".to_string(),
            },
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnyAgglomerative::Unfit { .. })
    }
}

#[pymethods]
impl PyAgglomerativeClustering {
    /// `AgglomerativeClustering(n_clusters=2, metric="euclidean")`.
    #[new]
    #[pyo3(signature = (n_clusters = 2, metric = String::from("euclidean")))]
    fn new(n_clusters: usize, metric: String) -> PyResult<Self> {
        // Reject an unknown metric AT CONSTRUCTION (sklearn parity: the string
        // is data-independent), not first at fit.
        parse_agglomerative_metric(&metric)?;
        Ok(Self {
            inner: AnyAgglomerative::Unfit { n_clusters, metric },
        })
    }

    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        use mlrs_algos::cluster::agglomerative::AgglomerativeClustering;

        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let (n_clusters, metric_s) = match &self.inner {
            AnyAgglomerative::Unfit { n_clusters, metric } => (*n_clusters, metric.clone()),
            _ => (2, "euclidean".to_string()),
        };
        let metric = parse_agglomerative_metric(&metric_s)?;
        let fitted = py.detach(|| -> PyResult<AnyAgglomerative> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    let est = AgglomerativeClustering::<f32>::builder()
                        .n_clusters(n_clusters)
                        .metric(metric)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyAgglomerative::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = AgglomerativeClustering::<f64>::builder()
                        .n_clusters(n_clusters)
                        .metric(metric)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnyAgglomerative::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// Fitted `labels_` (i32), either dtype arm; the runtime [`not_fitted`]
    /// analog on the `Unfit` arm (D-13).
    fn labels_(&self) -> PyResult<Vec<i32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnyAgglomerative::F32(e) => Ok(e.labels(&pool)),
            AnyAgglomerative::F64(e) => Ok(e.labels(&pool)),
            _ => Err(not_fitted("agglomerative_clustering", "labels_")),
        }
    }

    /// Fitted `children_` FLATTENED row-major (`2·(n-1)` i64 — the Python shim
    /// reshapes to `(n-1, 2)`).
    fn children_(&self) -> PyResult<Vec<i64>> {
        match &self.inner {
            AnyAgglomerative::F32(e) => Ok(e.children().iter().flatten().copied().collect()),
            AnyAgglomerative::F64(e) => Ok(e.children().iter().flatten().copied().collect()),
            _ => Err(not_fitted("agglomerative_clustering", "children_")),
        }
    }

    /// Fitted `n_leaves_` (== n_samples).
    fn n_leaves_(&self) -> PyResult<usize> {
        match &self.inner {
            AnyAgglomerative::F32(e) => Ok(e.n_leaves()),
            AnyAgglomerative::F64(e) => Ok(e.n_leaves()),
            _ => Err(not_fitted("agglomerative_clustering", "n_leaves_")),
        }
    }

    /// Fitted `n_connected_components_` (always 1 — unstructured fit).
    fn n_connected_components_(&self) -> PyResult<usize> {
        match &self.inner {
            AnyAgglomerative::F32(e) => Ok(e.n_connected_components()),
            AnyAgglomerative::F64(e) => Ok(e.n_connected_components()),
            _ => Err(not_fitted("agglomerative_clustering", "n_connected_components_")),
        }
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnyAgglomerative::Unfit { .. })
    }
    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnyAgglomerative::Unfit { .. } => None,
            AnyAgglomerative::F32(_) => Some("f32"),
            AnyAgglomerative::F64(_) => Some("f64"),
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
