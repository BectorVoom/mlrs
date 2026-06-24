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
// The v3 typestate `Fit` (consuming-self, returns the `Fitted` sibling) is imported
// under an alias so the per-arm UFCS call disambiguates from any legacy surface and
// the multiple-method-name collision the typestate module-doc warns about.
use mlrs_algos::typestate::Fit as TypestateFit;

use crate::errors::{algo_err_to_py, build_err_to_py, not_fitted};
use crate::ingress::{
    as_f32, as_f64, capsule_to_array, float_dtype, validated_f32, validated_f64, FloatDtype,
};

// ---------------------------------------------------------------------------
// SpectralEmbedding — fit (X) + embedding_ (n × n_components)
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnySpectralEmbedding,
    algo:  mlrs_algos::cluster::spectral_embedding::SpectralEmbedding,
    unfit: { n_components: usize, affinity: String, gamma: Option<f64>, n_neighbors: usize },
}

/// sklearn-compatible `SpectralEmbedding` (graph-Laplacian spectral embedding,
/// SPECTRAL-01).
///
/// The constructor hyperparameters are persisted in [`SpectralEmbeddingParams`]
/// (NOT only in the `Unfit` enum arm — WR-02) so a SECOND `fit` of the same object
/// honors the user's params instead of silently reverting to sklearn defaults. The
/// precision-typed `SpectralEmbedding<F>` (with `gamma=None → 1/n_features`, D-04)
/// is built by the algos estimator at every `fit` once `n_features` is known.
#[pyclass(name = "SpectralEmbedding")]
pub struct PySpectralEmbedding {
    /// Constructor hyperparameters, persisted across fits (WR-02). Read on EVERY
    /// `fit`, so `est.fit(X1); est.fit(X2)` re-fits with the SAME params (sklearn
    /// semantics) rather than the `Unfit`-only defaults.
    params: SpectralEmbeddingParams,
    inner: AnySpectralEmbedding,
}

/// The persisted constructor hyperparameters for `SpectralEmbedding` (WR-02).
///
/// Held on the `#[pyclass]` struct itself so they survive into the fitted
/// (`F32`/`F64`) arms and drive every `fit`, not just the first.
#[derive(Clone)]
struct SpectralEmbeddingParams {
    n_components: usize,
    affinity: String,
    gamma: Option<f64>,
    n_neighbors: usize,
}

impl PySpectralEmbedding {
    /// Rust-callable default constructor for the smoke test (sklearn defaults:
    /// `n_components=2`, `affinity="nearest_neighbors"`, `gamma=None`,
    /// `n_neighbors=10`).
    pub fn unfit_default() -> Self {
        let params = SpectralEmbeddingParams {
            n_components: 2,
            affinity: "nearest_neighbors".to_string(),
            gamma: None,
            n_neighbors: 10,
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

    /// Is this wrapper in the unfit (constructed-but-not-fitted) arm?
    pub fn is_unfit(&self) -> bool {
        matches!(self.inner, AnySpectralEmbedding::Unfit { .. })
    }
}

#[pymethods]
impl PySpectralEmbedding {
    /// `SpectralEmbedding(n_components=2, affinity="nearest_neighbors", gamma=None, n_neighbors=10)`.
    #[new]
    #[pyo3(signature = (n_components = 2, affinity = "nearest_neighbors".to_string(), gamma = None, n_neighbors = 10))]
    fn new(
        n_components: usize,
        affinity: String,
        gamma: Option<f64>,
        n_neighbors: usize,
    ) -> Self {
        let params = SpectralEmbeddingParams {
            n_components,
            affinity,
            gamma,
            n_neighbors,
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
        let SpectralEmbeddingParams {
            n_components,
            affinity,
            gamma,
            n_neighbors,
        } = self.params.clone();
        let fitted = py.detach(|| -> PyResult<AnySpectralEmbedding> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    // Builder gamma setter is Option<f64>; drop the legacy
                    // `gamma.map(|g| g as f32)` cast (narrowed inside build::<f32>(), A5).
                    let est = SpectralEmbedding::<f32>::builder()
                        .n_components(n_components)
                        .affinity(affinity)
                        .gamma(gamma)
                        .n_neighbors(n_neighbors)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnySpectralEmbedding::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = SpectralEmbedding::<f64>::builder()
                        .n_components(n_components)
                        .affinity(affinity)
                        .gamma(gamma)
                        .n_neighbors(n_neighbors)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnySpectralEmbedding::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// Host copy of the fitted `embedding_` (row-major `n × n_components`), f32
    /// arm; the runtime [`not_fitted`] analog if not in the f32 arm (D-13).
    fn embedding_f32(&self) -> PyResult<Vec<f32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnySpectralEmbedding::F32(e) => Ok(e.embedding(&pool)),
            _ => Err(not_fitted("spectral_embedding", "embedding_ (f32)")),
        }
    }

    /// Host copy of the fitted `embedding_`, f64 arm.
    fn embedding_f64(&self) -> PyResult<Vec<f64>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnySpectralEmbedding::F64(e) => Ok(e.embedding(&pool)),
            _ => Err(not_fitted("spectral_embedding", "embedding_ (f64)")),
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
}

// ---------------------------------------------------------------------------
// SpectralClustering — fit (X) + labels_ (i32)
// ---------------------------------------------------------------------------

crate::any_estimator_typestate! {
    any:   AnySpectralClustering,
    algo:  mlrs_algos::cluster::spectral_clustering::SpectralClustering,
    unfit: { n_clusters: usize, n_components: Option<usize>, affinity: String, gamma: f64, n_neighbors: usize, seed: u64 },
}

/// sklearn-compatible `SpectralClustering` (spectral embedding → KMeans,
/// SPECTRAL-02).
///
/// The constructor hyperparameters are persisted in [`SpectralClusteringParams`]
/// (NOT only in the `Unfit` enum arm — WR-02) so a SECOND `fit` of the same object
/// honors the user's params instead of silently reverting to sklearn defaults. The
/// precision-typed `SpectralClustering<F>` (with `n_components=None → n_clusters`,
/// D-11, and the literal `gamma` default, D-04) is built by the algos estimator at
/// every `fit`.
#[pyclass(name = "SpectralClustering")]
pub struct PySpectralClustering {
    /// Constructor hyperparameters, persisted across fits (WR-02). Read on EVERY
    /// `fit`, so `est.fit(X1); est.fit(X2)` re-fits with the SAME params (sklearn
    /// semantics) rather than the `Unfit`-only defaults.
    params: SpectralClusteringParams,
    inner: AnySpectralClustering,
}

/// The persisted constructor hyperparameters for `SpectralClustering` (WR-02).
///
/// Held on the `#[pyclass]` struct itself so they survive into the fitted
/// (`F32`/`F64`) arms and drive every `fit`, not just the first.
#[derive(Clone)]
struct SpectralClusteringParams {
    n_clusters: usize,
    n_components: Option<usize>,
    affinity: String,
    gamma: f64,
    n_neighbors: usize,
    seed: u64,
}

impl PySpectralClustering {
    /// Rust-callable default constructor for the smoke test (sklearn defaults:
    /// `n_clusters=8`, `n_components=None`, `affinity="rbf"`, `gamma=1.0`,
    /// `n_neighbors=10`, `seed=0`).
    pub fn unfit_default() -> Self {
        let params = SpectralClusteringParams {
            n_clusters: 8,
            n_components: None,
            affinity: "rbf".to_string(),
            gamma: 1.0,
            n_neighbors: 10,
            seed: 0,
        };
        Self {
            inner: AnySpectralClustering::Unfit {
                n_clusters: params.n_clusters,
                n_components: params.n_components,
                affinity: params.affinity.clone(),
                gamma: params.gamma,
                n_neighbors: params.n_neighbors,
                seed: params.seed,
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
    /// `SpectralClustering(n_clusters=8, n_components=None, affinity="rbf", gamma=1.0, n_neighbors=10, seed=0)`.
    #[new]
    #[pyo3(signature = (n_clusters = 8, n_components = None, affinity = "rbf".to_string(), gamma = 1.0, n_neighbors = 10, seed = 0))]
    fn new(
        n_clusters: usize,
        n_components: Option<usize>,
        affinity: String,
        gamma: f64,
        n_neighbors: usize,
        seed: u64,
    ) -> Self {
        let params = SpectralClusteringParams {
            n_clusters,
            n_components,
            affinity,
            gamma,
            n_neighbors,
            seed,
        };
        Self {
            inner: AnySpectralClustering::Unfit {
                n_clusters: params.n_clusters,
                n_components: params.n_components,
                affinity: params.affinity.clone(),
                gamma: params.gamma,
                n_neighbors: params.n_neighbors,
                seed: params.seed,
            },
            params,
        }
    }

    /// Fit the clustering on `x` (`rows × cols`). Unsupervised — no `y`. GIL
    /// released (PY-03); f64 guarded on an f64-incapable backend (D-04).
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
        let SpectralClusteringParams {
            n_clusters,
            n_components,
            affinity,
            gamma,
            n_neighbors,
            seed,
        } = self.params.clone();
        let fitted = py.detach(|| -> PyResult<AnySpectralClustering> {
            let mut pool = crate::lock_pool();
            match dt {
                FloatDtype::F32 => {
                    let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
                    // Builder setters are f64; drop the legacy `gamma as f32` cast
                    // (the f64→F narrowing happens once inside build::<f32>(), A5).
                    let est = SpectralClustering::<f32>::builder()
                        .n_clusters(n_clusters)
                        .n_components(n_components)
                        .affinity(affinity)
                        .gamma(gamma)
                        .n_neighbors(n_neighbors)
                        .seed(seed)
                        .build::<f32>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnySpectralClustering::F32(fitted))
                }
                FloatDtype::F64 => {
                    crate::capability::guard_f64()?;
                    let xd = validated_f64(as_f64(&xa)?, &mut pool)?;
                    let est = SpectralClustering::<f64>::builder()
                        .n_clusters(n_clusters)
                        .n_components(n_components)
                        .affinity(affinity)
                        .gamma(gamma)
                        .n_neighbors(n_neighbors)
                        .seed(seed)
                        .build::<f64>()
                        .map_err(build_err_to_py)?;
                    let fitted = TypestateFit::fit(est, &mut pool, &xd, None, (rows, cols))
                        .map_err(algo_err_to_py)?;
                    Ok(AnySpectralClustering::F64(fitted))
                }
            }
        })?;
        self.inner = fitted;
        Ok(())
    }

    /// Fitted `labels_` (i32), either dtype arm; the runtime [`not_fitted`] analog
    /// on the `Unfit` arm (D-13).
    fn labels_(&self) -> PyResult<Vec<i32>> {
        let pool = crate::lock_pool();
        match &self.inner {
            AnySpectralClustering::F32(e) => Ok(e.labels(&pool)),
            AnySpectralClustering::F64(e) => Ok(e.labels(&pool)),
            _ => Err(not_fitted("spectral_clustering", "labels_")),
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
}
