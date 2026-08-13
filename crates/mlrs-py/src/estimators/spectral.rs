//! Spectral-family `#[pyclass]` wrappers (SPECTRAL-01/SPECTRAL-02 — PY-06
//! incremental share): `PySpectralEmbedding` (fit/`embedding_`) and
//! `PySpectralClustering` (fit/`labels_`).
//!
//! Both reuse the shipped [`any_estimator!`](crate::any_estimator) Unfit/F32/F64
//! dtype-dispatch machinery (D-06) — v2 adds ZERO new binding infrastructure.
//! Each device-compute body honors the two load-bearing contracts documented on
//! [`crate::dispatch`]:
//!
//! 1. **GIL release (PY-03).** The `mlrs_algos` call runs inside
//!    `py.detach(|| { … })` around a lock of the process-global pool
//!    ([`crate::global_pool`]).
//! 2. **f64 guard (D-04).** On the `FloatDtype::F64` dispatch arm,
//!    [`crate::capability::guard_f64`]`()?` runs BEFORE any upload.
//!
//! ## Unfit stores the sklearn defaults verbatim (D-01 / D-04)
//! `SpectralEmbedding` and `SpectralClustering` DISAGREE on their affinity / gamma
//! defaults and we honor both (D-01): SE default `affinity="nearest_neighbors"`,
//! `gamma=None` (→ `1/n_features` at fit, D-04); SC default `affinity="rbf"`,
//! `gamma=1.0` (literal, D-04). The precision-typed `Option<F>` / `F` gamma is
//! built at `fit` once `n_features` is known.
//!
//! Fitted-attribute accessors are dtype-suffixed (`embedding_f32`/`_f64`) for the
//! float embedding; `labels_` is single-typed `Vec<i32>` (the KMeans i32 idiom).
//!
//! ## Wave-0 scaffold status
//! This is the 09-01 Wave-0 COMPILING STUB: the two `any_estimator!` enums + the
//! two `#[pyclass]` constructors carrying the sklearn defaults are real (so the
//! `_mlrs` registration + the smoke scaffold compile today), but every
//! device-compute body delegates to the algos `fit` / accessor stubs, which are
//! `todo!()` until the Wave-2/3 plans (09-03 / 09-04). Copies `kernel.rs`
//! structure verbatim.
//!
//! Tests live in `crates/mlrs-py/tests/` (AGENTS.md §2 — never an in-source
//! `#[cfg(test)] mod tests`).

use pyo3::prelude::*;

use mlrs_algos::cluster::spectral_clustering::SpectralClustering;
use mlrs_algos::cluster::spectral_embedding::SpectralEmbedding;

use crate::errors::{algo_err_to_py, build_err_to_py, not_fitted};
use crate::egress::{f32_vec_to_pyarrow, f64_vec_to_pyarrow, i32_vec_to_pyarrow};
use crate::ingress::{
    as_f32, as_f64, capsule_to_array, float_dtype, host_slice_f32, host_slice_f64, FloatDtype,
};

// ---------------------------------------------------------------------------
// SpectralEmbedding — fit (X) + embedding_ (n × n_components)
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnySpectralEmbedding,
    algo:  mlrs_algos::cluster::spectral_embedding::SpectralEmbedding,
    unfit: { n_components: usize, affinity: String, gamma: Option<f64>, n_neighbors: Option<usize> },
}

crate::impl_persistable_any! {
    any:  AnySpectralEmbedding,
    algo: mlrs_algos::cluster::spectral_embedding::SpectralEmbedding,
    name: "spectral_embedding",
}

/// sklearn-compatible `SpectralEmbedding` (graph-Laplacian spectral embedding,
/// SPECTRAL-01).
///
/// The constructor hyperparameters are persisted in [`SpectralEmbeddingParams`]
/// (NOT only in the `Unfit` enum arm — WR-02) so a SECOND `fit` of the same object
/// honors the user's params instead of silently reverting to sklearn defaults.
///
/// ## Ingress: no upload (SPECTRAL-PERF-CPU)
/// The spectral pipeline is a single HOST implementation, so `fit` borrows the
/// caller's Arrow buffer with [`host_slice_f32`] / [`host_slice_f64`] and never
/// builds a `DeviceArray` for `X` at all. The former binding uploaded `X` on
/// every fit; since the algos side then had to read it straight back, that cost
/// three full passes over the design (`from_host` copies once, `to_host` copies
/// twice) before any arithmetic happened.
#[pyclass(name = "SpectralEmbedding")]
pub struct PySpectralEmbedding {
    /// Constructor hyperparameters, persisted across fits (WR-02). Read on EVERY
    /// `fit`, so `est.fit(X1); est.fit(X2)` re-fits with the SAME params (sklearn
    /// semantics) rather than the `Unfit`-only defaults.
    params: SpectralEmbeddingParams,
    inner: AnySpectralEmbedding,
}

/// The persisted constructor hyperparameters for `SpectralEmbedding` (WR-02) —
/// sklearn 1.9's full eight-parameter surface.
///
/// Held on the `#[pyclass]` struct itself so they survive into the fitted
/// (`F32`/`F64`) arms and drive every `fit`, not just the first.
#[derive(Clone)]
struct SpectralEmbeddingParams {
    n_components: usize,
    affinity: String,
    gamma: Option<f64>,
    random_state: Option<u64>,
    eigen_solver: Option<String>,
    /// `None` is sklearn's `eigen_tol="auto"`.
    eigen_tol: Option<f64>,
    /// `None` resolves to `max(n_samples / 10, 1)` at fit — sklearn's rule, NOT
    /// a constant 10.
    n_neighbors: Option<usize>,
    n_jobs: Option<i64>,
}

/// Build the precision-typed algos estimator from the persisted params.
///
/// Shared by both dtype arms of `fit` so the two cannot drift on which
/// parameters are forwarded — the failure mode a hand-copied second arm invites
/// is a silently-ignored hyperparameter, which looks like a numerical bug.
macro_rules! se_build {
    ($float:ty, $p:expr) => {
        SpectralEmbedding::<$float>::builder()
            .n_components($p.n_components)
            .affinity($p.affinity.clone())
            .gamma($p.gamma)
            .random_state($p.random_state)
            .eigen_solver($p.eigen_solver.clone())
            .eigen_tol($p.eigen_tol)
            .n_neighbors($p.n_neighbors)
            .n_jobs($p.n_jobs)
            .build::<$float>()
            .map_err(build_err_to_py)?
    };
}

impl PySpectralEmbedding {
    /// Rust-callable default constructor for the smoke test (sklearn defaults:
    /// `n_components=2`, `affinity="nearest_neighbors"`, everything else `None`).
    pub fn unfit_default() -> Self {
        Self::new(2, "nearest_neighbors".to_string(), None, None, None, None, None, None)
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnySpectralEmbedding::Unfit { .. })
    }
}

#[pymethods]
impl PySpectralEmbedding {
    /// `SpectralEmbedding(n_components=2, *, affinity="nearest_neighbors",
    /// gamma=None, random_state=None, eigen_solver=None, eigen_tol=None,
    /// n_neighbors=None, n_jobs=None)`.
    ///
    /// `eigen_tol` arrives as `Option<f64>` rather than sklearn's
    /// `float | "auto"` union: the shim maps the string `"auto"` to `None`
    /// before crossing the boundary, so the Rust side has one type instead of a
    /// `PyAny` it would have to re-parse.
    #[new]
    #[pyo3(signature = (
        n_components = 2,
        affinity = "nearest_neighbors".to_string(),
        gamma = None,
        random_state = None,
        eigen_solver = None,
        eigen_tol = None,
        n_neighbors = None,
        n_jobs = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_components: usize,
        affinity: String,
        gamma: Option<f64>,
        random_state: Option<u64>,
        eigen_solver: Option<String>,
        eigen_tol: Option<f64>,
        n_neighbors: Option<usize>,
        n_jobs: Option<i64>,
    ) -> Self {
        let params = SpectralEmbeddingParams {
            n_components,
            affinity,
            gamma,
            random_state,
            eigen_solver,
            eigen_tol,
            n_neighbors,
            n_jobs,
        };
        Self {
            inner: AnySpectralEmbedding::Unfit {
                n_components: params.n_components,
                affinity: params.affinity.clone(),
                gamma: params.gamma,
                n_neighbors: params.n_neighbors,
            },
            params,
        }
    }

    /// Fit the embedding on `x` (`rows × cols`). Unsupervised — no `y`. GIL
    /// released (PY-03); f64 guarded on an f64-incapable backend (D-04).
    ///
    /// `x` is borrowed, never uploaded — see the struct docs.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        // WR-02: read the persisted constructor params (NOT the `Unfit` arm), so a
        // re-`fit` of an already-fitted object honors the user's hyperparameters
        // instead of reverting to sklearn defaults.
        let p = self.params.clone();
        let fitted = py.detach(|| -> PyResult<AnySpectralEmbedding> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    let est = se_build!(f32, p);
                    let fitted = est
                        .fit_from_host_slice(&mut pool, xh, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnySpectralEmbedding::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    let est = se_build!(f64, p);
                    let fitted = est
                        .fit_from_host_slice(&mut pool, xh, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnySpectralEmbedding::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// Host copy of the fitted `embedding_` (row-major `n × n_components`), f32
    /// arm, as a pyarrow array; the runtime [`not_fitted`] analog if not in the
    /// f32 arm (D-13).
    fn embedding_f32<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnySpectralEmbedding::F32(e) => f32_vec_to_pyarrow(py, e.embedding(&pool)),
            _ => Err(not_fitted("spectral_embedding", "embedding_ (f32)")),
        }
    }

    /// Host copy of the fitted `embedding_`, f64 arm.
    fn embedding_f64<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnySpectralEmbedding::F64(e) => f64_vec_to_pyarrow(py, e.embedding(&pool)),
            _ => Err(not_fitted("spectral_embedding", "embedding_ (f64)")),
        }
    }

    /// `affinity_matrix_` densified to a row-major `n × n` f64 pyarrow array.
    ///
    /// The kNN affinity is stored SPARSE, so asking for it dense materializes
    /// `n²` values — use [`Self::affinity_matrix_csr`] when the caller can
    /// consume CSR, which is what sklearn returns for that affinity.
    fn affinity_matrix_dense<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnySpectralEmbedding::F32(e) => f64_vec_to_pyarrow(py, e.affinity_matrix_dense()),
            AnySpectralEmbedding::F64(e) => f64_vec_to_pyarrow(py, e.affinity_matrix_dense()),
            _ => Err(not_fitted("spectral_embedding", "affinity_matrix_")),
        }
    }

    /// `affinity_matrix_` as the CSR triple `(indptr, indices, data)` when the
    /// affinity builder produced a sparse graph, else `None`.
    ///
    /// `indptr` / `indices` cross as `i32` — the width `scipy.sparse.csr_matrix`
    /// itself uses below the 2³¹-nonzero threshold, and the one integer egress
    /// helper this crate has.
    fn affinity_matrix_csr<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
        let csr = match &self.inner {
            AnySpectralEmbedding::F32(e) => e.affinity_matrix_sparse(),
            AnySpectralEmbedding::F64(e) => e.affinity_matrix_sparse(),
            _ => return Err(not_fitted("spectral_embedding", "affinity_matrix_")),
        };
        let Some(csr) = csr else { return Ok(None) };
        let indptr = i32_vec_to_pyarrow(py, csr.indptr.iter().map(|&v| v as i32).collect())?;
        let indices = i32_vec_to_pyarrow(py, csr.indices.iter().map(|&v| v as i32).collect())?;
        let data = f64_vec_to_pyarrow(py, csr.data.clone())?;
        Ok(Some((indptr, indices, data)))
    }

    /// sklearn's `n_neighbors_` — the RESOLVED neighbor count. `None` unless a
    /// kNN affinity was used, which is exactly when sklearn sets the attribute.
    fn n_neighbors_resolved(&self) -> PyResult<Option<usize>> {
        match &self.inner {
            AnySpectralEmbedding::F32(e) => Ok(e.n_neighbors_()),
            AnySpectralEmbedding::F64(e) => Ok(e.n_neighbors_()),
            _ => Err(not_fitted("spectral_embedding", "n_neighbors_")),
        }
    }

    /// sklearn's `gamma_` — the RESOLVED kernel coefficient. `None` unless a
    /// kernel affinity was used.
    fn gamma_resolved(&self) -> PyResult<Option<f64>> {
        match &self.inner {
            AnySpectralEmbedding::F32(e) => Ok(e.gamma_()),
            AnySpectralEmbedding::F64(e) => Ok(e.gamma_()),
            _ => Err(not_fitted("spectral_embedding", "gamma_")),
        }
    }

    /// Connected components of the fitted affinity graph. sklearn warns
    /// `"Graph is not fully connected, spectral embedding may not work as
    /// expected."` when this exceeds 1 and changes nothing else; the shim raises
    /// the same warning from this count.
    fn n_graph_components(&self) -> PyResult<usize> {
        match &self.inner {
            AnySpectralEmbedding::F32(e) => Ok(e.n_graph_components()),
            AnySpectralEmbedding::F64(e) => Ok(e.n_graph_components()),
            _ => Err(not_fitted("spectral_embedding", "n_graph_components")),
        }
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnySpectralEmbedding::Unfit { .. })
    }

    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnySpectralEmbedding::Unfit { .. } => None,
            AnySpectralEmbedding::F32(_) => Some("f32"),
            AnySpectralEmbedding::F64(_) => Some("f64"),
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
// SpectralClustering — fit (X) + labels_ (i32)
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnySpectralClustering,
    algo:  mlrs_algos::cluster::spectral_clustering::SpectralClustering,
    unfit: { n_clusters: usize, n_components: Option<usize>, affinity: String, gamma: f64, n_neighbors: usize, seed: u64 },
}

crate::impl_persistable_any! {
    any:  AnySpectralClustering,
    algo: mlrs_algos::cluster::spectral_clustering::SpectralClustering,
    name: "spectral_clustering",
}

/// sklearn-compatible `SpectralClustering` (spectral embedding → label
/// assignment, SPECTRAL-02).
///
/// The constructor hyperparameters are persisted in [`SpectralClusteringParams`]
/// (NOT only in the `Unfit` enum arm — WR-02) so a SECOND `fit` of the same
/// object honors the user's params instead of silently reverting to sklearn
/// defaults.
///
/// Ingress is the no-upload host-slice arm, exactly as for
/// [`PySpectralEmbedding`] — the pipeline is host-side, so building a
/// `DeviceArray` for `X` would cost three full passes over the design before any
/// arithmetic.
#[pyclass(name = "SpectralClustering")]
pub struct PySpectralClustering {
    /// Constructor hyperparameters, persisted across fits (WR-02). Read on EVERY
    /// `fit`, so `est.fit(X1); est.fit(X2)` re-fits with the SAME params (sklearn
    /// semantics) rather than the `Unfit`-only defaults.
    params: SpectralClusteringParams,
    inner: AnySpectralClustering,
}

/// The persisted constructor hyperparameters for `SpectralClustering` (WR-02) —
/// sklearn 1.9's surface.
///
/// `kernel_params` is deliberately absent: sklearn overwrites
/// `params["gamma"|"degree"|"coef0"]` from the estimator's own attributes for any
/// non-callable affinity and then calls `pairwise_kernels(filter_params=True)`,
/// which drops every key outside those three — so for a string affinity the
/// parameter is provably a no-op, and callable affinities are out of scope.
#[derive(Clone)]
struct SpectralClusteringParams {
    n_clusters: usize,
    eigen_solver: Option<String>,
    n_components: Option<usize>,
    random_state: Option<u64>,
    n_init: usize,
    gamma: f64,
    affinity: String,
    n_neighbors: usize,
    /// `None` is sklearn's `eigen_tol="auto"`.
    eigen_tol: Option<f64>,
    assign_labels: String,
    degree: f64,
    coef0: f64,
    n_jobs: Option<i64>,
    verbose: bool,
}

impl Default for SpectralClusteringParams {
    /// sklearn's `SpectralClustering` defaults verbatim.
    fn default() -> Self {
        Self {
            n_clusters: 8,
            eigen_solver: None,
            n_components: None,
            random_state: None,
            n_init: 10,
            gamma: 1.0,
            affinity: "rbf".to_string(),
            n_neighbors: 10,
            eigen_tol: None,
            assign_labels: "kmeans".to_string(),
            degree: 3.0,
            coef0: 1.0,
            n_jobs: None,
            verbose: false,
        }
    }
}

/// Build the precision-typed algos estimator from the persisted params.
///
/// Shared by both dtype arms of `fit` so the two cannot drift on which
/// parameters are forwarded — a hand-copied second arm invites a silently
/// dropped hyperparameter, which presents as a numerical bug.
macro_rules! sc_build {
    ($float:ty, $p:expr) => {
        SpectralClustering::<$float>::builder()
            .n_clusters($p.n_clusters)
            .eigen_solver($p.eigen_solver.clone())
            .n_components($p.n_components)
            .random_state($p.random_state)
            .n_init($p.n_init)
            .gamma($p.gamma)
            .affinity($p.affinity.clone())
            .n_neighbors($p.n_neighbors)
            .eigen_tol($p.eigen_tol)
            .assign_labels($p.assign_labels.clone())
            .degree($p.degree)
            .coef0($p.coef0)
            .n_jobs($p.n_jobs)
            .verbose($p.verbose)
            .build::<$float>()
            .map_err(build_err_to_py)?
    };
}

impl PySpectralClustering {
    /// Rust-callable default constructor for the smoke test.
    pub fn unfit_default() -> Self {
        Self::from_params(SpectralClusteringParams::default())
    }

    fn from_params(params: SpectralClusteringParams) -> Self {
        Self {
            inner: AnySpectralClustering::Unfit {
                n_clusters: params.n_clusters,
                n_components: params.n_components,
                affinity: params.affinity.clone(),
                gamma: params.gamma,
                n_neighbors: params.n_neighbors,
                seed: params.random_state.unwrap_or(0),
            },
            params,
        }
    }

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnySpectralClustering::Unfit { .. })
    }
}

#[pymethods]
impl PySpectralClustering {
    /// `SpectralClustering(n_clusters=8, *, eigen_solver=None, n_components=None,
    /// random_state=None, n_init=10, gamma=1.0, affinity="rbf", n_neighbors=10,
    /// eigen_tol=None, assign_labels="kmeans", degree=3, coef0=1, n_jobs=None,
    /// verbose=False)`.
    ///
    /// `eigen_tol` arrives as `Option<f64>` rather than sklearn's
    /// `float | "auto"` union: the shim maps `"auto"` to `None` before crossing
    /// the boundary so the Rust side has one type instead of a `PyAny` to
    /// re-parse.
    #[new]
    #[pyo3(signature = (
        n_clusters = 8,
        eigen_solver = None,
        n_components = None,
        random_state = None,
        n_init = 10,
        gamma = 1.0,
        affinity = "rbf".to_string(),
        n_neighbors = 10,
        eigen_tol = None,
        assign_labels = "kmeans".to_string(),
        degree = 3.0,
        coef0 = 1.0,
        n_jobs = None,
        verbose = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: usize,
        eigen_solver: Option<String>,
        n_components: Option<usize>,
        random_state: Option<u64>,
        n_init: usize,
        gamma: f64,
        affinity: String,
        n_neighbors: usize,
        eigen_tol: Option<f64>,
        assign_labels: String,
        degree: f64,
        coef0: f64,
        n_jobs: Option<i64>,
        verbose: bool,
    ) -> Self {
        Self::from_params(SpectralClusteringParams {
            n_clusters,
            eigen_solver,
            n_components,
            random_state,
            n_init,
            gamma,
            affinity,
            n_neighbors,
            eigen_tol,
            assign_labels,
            degree,
            coef0,
            n_jobs,
            verbose,
        })
    }

    /// Fit the clustering on `x` (`rows × cols`). Unsupervised — no `y`. GIL
    /// released (PY-03); f64 guarded on an f64-incapable backend (D-04).
    ///
    /// `x` is borrowed, never uploaded — see the struct docs.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let xa = capsule_to_array(x)?;
        let dt = float_dtype(&xa)?;
        let p = self.params.clone();
        let fitted = py.detach(|| -> PyResult<AnySpectralClustering> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xh = host_slice_f32(as_f32(&xa)?)?;
                    let est = sc_build!(f32, p);
                    let fitted = est
                        .fit_from_host_slice(&mut pool, xh, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnySpectralClustering::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xh = host_slice_f64(as_f64(&xa)?)?;
                    let est = sc_build!(f64, p);
                    let fitted = est
                        .fit_from_host_slice(&mut pool, xh, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnySpectralClustering::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// Fitted `labels_` (length `n`, `i32`) as a pyarrow array.
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pool = crate::lock_pool();
        let labels = match &self.inner {
            AnySpectralClustering::F32(e) => e.labels(&pool),
            AnySpectralClustering::F64(e) => e.labels(&pool),
            _ => return Err(not_fitted("spectral_clustering", "labels_")),
        };
        i32_vec_to_pyarrow(py, labels)
    }

    /// `affinity_matrix_` densified to a row-major `n × n` f64 pyarrow array.
    fn affinity_matrix_dense<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            AnySpectralClustering::F32(e) => f64_vec_to_pyarrow(py, e.affinity_matrix_dense()),
            AnySpectralClustering::F64(e) => f64_vec_to_pyarrow(py, e.affinity_matrix_dense()),
            _ => Err(not_fitted("spectral_clustering", "affinity_matrix_")),
        }
    }

    /// `affinity_matrix_` as the CSR triple `(indptr, indices, data)` when the
    /// affinity builder produced a sparse graph, else `None` — the same split
    /// sklearn produces (`csr_matrix` for the kNN affinities, dense `ndarray`
    /// for the kernels).
    fn affinity_matrix_csr<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
        let csr = match &self.inner {
            AnySpectralClustering::F32(e) => e.affinity_matrix_sparse(),
            AnySpectralClustering::F64(e) => e.affinity_matrix_sparse(),
            _ => return Err(not_fitted("spectral_clustering", "affinity_matrix_")),
        };
        let Some(csr) = csr else { return Ok(None) };
        let indptr = i32_vec_to_pyarrow(py, csr.indptr.iter().map(|&v| v as i32).collect())?;
        let indices = i32_vec_to_pyarrow(py, csr.indices.iter().map(|&v| v as i32).collect())?;
        let data = f64_vec_to_pyarrow(py, csr.data.clone())?;
        Ok(Some((indptr, indices, data)))
    }

    /// Connected components of the fitted affinity graph, for the shim's
    /// sklearn-parity connectivity warning.
    fn n_graph_components(&self) -> PyResult<usize> {
        match &self.inner {
            AnySpectralClustering::F32(e) => Ok(e.n_graph_components()),
            AnySpectralClustering::F64(e) => Ok(e.n_graph_components()),
            _ => Err(not_fitted("spectral_clustering", "n_graph_components")),
        }
    }

    fn is_fitted(&self) -> bool {
        !matches!(self.inner, AnySpectralClustering::Unfit { .. })
    }

    fn dtype(&self) -> Option<&'static str> {
        match &self.inner {
            AnySpectralClustering::Unfit { .. } => None,
            AnySpectralClustering::F32(_) => Some("f32"),
            AnySpectralClustering::F64(_) => Some("f64"),
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
