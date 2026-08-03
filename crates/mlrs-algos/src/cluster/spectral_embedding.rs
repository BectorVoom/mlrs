//! `SpectralEmbedding` (SPECTRAL-01) — graph-Laplacian spectral embedding,
//! matching `sklearn.manifold.SpectralEmbedding`.
//!
//! ## Pipeline
//! `affinity A` → normalized Laplacian `(L, dd)` → the smallest
//! `n_components + 1` eigenvectors → `D^-1/2` recovery (`/ dd`) → deterministic
//! sign flip → drop the trivial ≈0 eigenvector (`drop_first = true`) →
//! `embedding_` (`n × n_components`).
//!
//! The operation ORDER is load-bearing: slice smallest → `/dd` → sign-flip →
//! drop-first, with `dd` being the SAME vector the Laplacian returned. sklearn
//! takes the argmax for the sign flip over the ALREADY-`/dd`-scaled vector, so
//! swapping those two steps changes signs.
//!
//! ## The `n_samples ≤ 64` cap is GONE (SPECTRAL-PERF-CPU)
//! The former implementation built a dense `n × n` affinity and Laplacian on the
//! device and asked the cyclic-Jacobi `eig` kernel for the FULL spectrum. That
//! kernel stages its working matrices as comptime-sized shared memory at
//! `MAX_DIM × MAX_DIM`, so it rejected `n > 64` — which made the estimator
//! unusable on any real dataset, and `O(n³)` even on the toy ones.
//!
//! `fit` now runs [`crate::cluster::spectral_host::run`]: the default kNN
//! affinity stays SPARSE, and only the `n_components + 1` eigenpairs actually
//! wanted are computed, by a restarted BLOCK Krylov iteration whose cost is
//! `O(iters·nnz)`.
//! There is no size cap left, and the pipeline is a single host implementation
//! on every backend — so the sklearn-parity decisions (`_set_diag`, the
//! diagonal-excluding degree, the sign-flip order) cannot drift between arms.
//!
//! ## sklearn parameter surface
//! `SpectralEmbedding(n_components=2, *, affinity="nearest_neighbors",
//! gamma=None, random_state=None, eigen_solver=None, eigen_tol="auto",
//! n_neighbors=None, n_jobs=None)` — all eight, with sklearn's resolution rules:
//!
//! - `n_neighbors=None` → `max(n_samples // 10, 1)` (TRUNCATING, floored at 1).
//!   This is NOT a constant 10; the previous implementation hard-coded 10 and so
//!   built a different graph from sklearn's on every input with `n ≠ 100`.
//! - `gamma=None` → `1/n_features`, resolved at fit. `gamma = 0` is LEGAL
//!   (sklearn's constraint is `closed="left"`); only negative / non-finite is
//!   rejected.
//! - `affinity` ∈ {`nearest_neighbors`, `rbf`, `precomputed`,
//!   `precomputed_nearest_neighbors`}. sklearn reaches `rbf_kernel` directly
//!   here — the wider `pairwise_kernels` family belongs to `SpectralClustering`.
//! - `eigen_solver` ∈ {`arpack`, `lobpcg`, `amg`, `None`} is accepted and
//!   validated. All four name a way to obtain the SAME invariant subspace;
//!   mlrs has one solver (dense below `DENSE_N`, Lanczos above) and routes every
//!   value to it, so the parameter selects nothing — but it is not silently
//!   ignored either: an out-of-set string is rejected exactly as sklearn
//!   rejects it.
//! - `eigen_tol="auto"` → the solver's own machine-precision target, which is
//!   what sklearn's `tol=0` asks ARPACK for.
//! - `n_jobs` is accepted for signature compatibility. The host pipeline sizes
//!   its worker pool from `MLRS_CPU_UNITS` / available parallelism rather than
//!   from this parameter.
//!
//! `SpectralEmbedding` is non-transductive: sklearn exposes `fit` /
//! `fit_transform` / `embedding_` and deliberately NO `transform` for new
//! points, and neither does this.
//!
//! Tests live in `crates/mlrs-algos/tests/spectral_embedding_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)] mod tests`).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64};

use crate::cluster::spectral_host::{self, Csr, HostAffinity, SpectralPlan};
use crate::error::{AlgoError, BuildError};
use crate::typestate::{Fit, Fitted, Unfit};

/// The `eigen_solver` values sklearn accepts alongside `None`.
pub(crate) const EIGEN_SOLVERS: [&str; 3] = ["arpack", "lobpcg", "amg"];

/// Spectral embedding (SPECTRAL-01) of an affinity graph onto the smallest
/// non-trivial eigenvectors of the normalized Laplacian.
///
/// Construct with the zero-arg [`SpectralEmbedding::new`] (sklearn defaults) or
/// [`SpectralEmbedding::builder`], then the consuming [`Fit::fit`] (which
/// returns the `Fitted`-tagged sibling) and read `embedding_`. The fitted
/// accessors exist ONLY on `SpectralEmbedding<F, Fitted>` — the compile-time
/// typestate replaces the runtime `NotFitted` guard.
pub struct SpectralEmbedding<F, S = Unfit> {
    /// Embedding dimensionality (sklearn default `2`). The smallest
    /// `n_components + 1` eigenvectors are computed and the trivial ≈0 one
    /// dropped.
    n_components: usize,
    /// Affinity construction (`"nearest_neighbors"` default).
    affinity: String,
    /// Kernel coefficient `γ` for the rbf affinity; `None` resolves to
    /// `1/n_features` at `fit`. Ignored by the non-kernel affinities.
    gamma: Option<F>,
    /// Seed for the deterministic Krylov start block (sklearn's
    /// `random_state`, which seeds ARPACK's `v0`). The converged subspace does
    /// not depend on it; it exists so repeated fits are reproducible.
    random_state: Option<u64>,
    /// `eigen_solver` — validated against [`EIGEN_SOLVERS`], see the module docs.
    eigen_solver: Option<String>,
    /// `eigen_tol`; `None` is sklearn's `"auto"`.
    eigen_tol: Option<f64>,
    /// Number of neighbors for the kNN affinities; `None` resolves to
    /// `max(n_samples / 10, 1)` at `fit`.
    n_neighbors: Option<usize>,
    /// `n_jobs`, accepted for signature compatibility (see the module docs).
    n_jobs: Option<i64>,

    /// Fitted `n × n_components` embedding, device-resident, `None` until `fit`.
    embedding_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted `affinity_matrix_`, in its builder's layout.
    affinity_matrix_: Option<HostAffinity>,
    /// The RESOLVED `n_neighbors_`; `None` unless a kNN affinity was used —
    /// sklearn likewise sets the attribute only on that branch.
    n_neighbors_: Option<usize>,
    /// The RESOLVED `gamma_`; `None` unless a kernel affinity was used.
    gamma_: Option<f64>,
    /// Connected components of the affinity graph. sklearn warns when this is
    /// `> 1`; it changes nothing else.
    n_graph_components_: usize,
    /// `n_features_in_`.
    n_features_in_: usize,
    /// `n_samples` seen at fit (the `embedding_` row count).
    n_samples_: usize,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> SpectralEmbedding<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfitted `SpectralEmbedding` with sklearn's defaults
    /// (`n_components = 2`, `affinity = "nearest_neighbors"`, `gamma = None`,
    /// `random_state = None`, `eigen_solver = None`, `eigen_tol = "auto"`,
    /// `n_neighbors = None`, `n_jobs = None`) directly in the `Unfit` state.
    /// SINGLE source of truth for the defaults: the builder `Default`
    /// re-derives via [`SpectralEmbedding::into_builder`].
    pub fn new() -> Self {
        Self {
            n_components: 2,
            affinity: "nearest_neighbors".to_string(),
            gamma: None,
            random_state: None,
            eigen_solver: None,
            eigen_tol: None,
            n_neighbors: None,
            n_jobs: None,
            embedding_: None,
            affinity_matrix_: None,
            n_neighbors_: None,
            gamma_: None,
            n_graph_components_: 0,
            n_features_in_: 0,
            n_samples_: 0,
            _state: PhantomData,
        }
    }

    /// Start building a `SpectralEmbedding` from sklearn's defaults.
    pub fn builder() -> SpectralEmbeddingBuilder {
        SpectralEmbeddingBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`SpectralEmbeddingBuilder::default`] to
    /// re-derive the defaults from [`SpectralEmbedding::new`].
    pub fn into_builder(self) -> SpectralEmbeddingBuilder {
        SpectralEmbeddingBuilder {
            n_components: self.n_components,
            affinity: self.affinity,
            gamma: self.gamma.map(host_to_f64),
            random_state: self.random_state,
            eigen_solver: self.eigen_solver,
            eigen_tol: self.eigen_tol,
            n_neighbors: self.n_neighbors,
            n_jobs: self.n_jobs,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// attributes are excluded — all absent in any `Unfit` value). Used by the
    /// defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_components == other.n_components
            && self.affinity == other.affinity
            && self.gamma.map(host_to_f64) == other.gamma.map(host_to_f64)
            && self.random_state == other.random_state
            && self.eigen_solver == other.eigen_solver
            && self.eigen_tol == other.eigen_tol
            && self.n_neighbors == other.n_neighbors
            && self.n_jobs == other.n_jobs
    }

    /// Whether the caller should reach [`Self::fit_from_host_slice`] instead of
    /// uploading `x` and calling [`Fit::fit`].
    ///
    /// Always `true`: the spectral pipeline is a single HOST implementation on
    /// every backend, so an upload before `fit` is pure waste — `from_host`
    /// copies once and the `to_host` that `fit` would then need copies twice.
    /// The predicate exists anyway because the two entry points take DIFFERENT
    /// operand types (host slice vs `DeviceArray`), so the choice has to be made
    /// BEFORE ingress; keeping it named matches the Ridge / BayesianRidge shape
    /// and leaves one place to change if a device arm ever returns.
    pub fn host_fit_applicable(&self, _shape: (usize, usize)) -> bool {
        true
    }

    /// Fit directly from a host-resident row-major `n × d` slice — the
    /// no-upload arm (see [`Self::host_fit_applicable`]).
    ///
    /// For the `precomputed` / `precomputed_nearest_neighbors` affinities `x` is
    /// the `n × n` affinity / distance matrix instead, and sklearn reports
    /// `n_features_in_ = n_samples` there, which this reproduces because the
    /// caller passes `shape = (n, n)`.
    pub fn fit_from_host_slice(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<SpectralEmbedding<F, Fitted>, AlgoError> {
        let x64: Vec<f64> = x.iter().map(|&v| host_to_f64(v)).collect();
        self.fit_host_core(pool, &x64, shape)
    }

    /// The one implementation both entry points reach, so they cannot drift.
    fn fit_host_core(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x64: &[f64],
        shape: (usize, usize),
    ) -> Result<SpectralEmbedding<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        if n_samples == 0 || n_features == 0 {
            return Err(AlgoError::InvalidGraphInput {
                estimator: "spectral_embedding",
                reason: format!("empty design ({n_samples} x {n_features})"),
            });
        }
        // `eigen_solver` is validated at `fit` rather than at `build` so the
        // error surfaces at the same point sklearn's `_fit_context` validation
        // does.
        if let Some(s) = self.eigen_solver.as_deref() {
            if !EIGEN_SOLVERS.contains(&s) {
                return Err(AlgoError::Unsupported {
                    estimator: "spectral_embedding",
                    operation: "eigen_solver (expected one of arpack / lobpcg / amg)",
                });
            }
        }

        let plan = SpectralPlan {
            estimator: "spectral_embedding",
            affinity: &self.affinity,
            gamma: self.gamma.map(host_to_f64),
            // Unused: `SpectralEmbedding` never reaches `pairwise_kernels`.
            degree: 3.0,
            coef0: 1.0,
            n_neighbors: self.n_neighbors,
            n_components: self.n_components,
            drop_first: true,
            seed: self.random_state.unwrap_or(0),
            allow_kernels: false,
        };
        let out = spectral_host::run(&plan, x64, n_samples, n_features)?;

        let emb_host: Vec<F> = out.embedding.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let embedding_dev = DeviceArray::from_host(pool, &emb_host);

        Ok(SpectralEmbedding {
            n_components: self.n_components,
            affinity: self.affinity,
            gamma: self.gamma,
            random_state: self.random_state,
            eigen_solver: self.eigen_solver,
            eigen_tol: self.eigen_tol,
            n_neighbors: self.n_neighbors,
            n_jobs: self.n_jobs,
            embedding_: Some(embedding_dev),
            affinity_matrix_: Some(out.affinity),
            n_neighbors_: out.n_neighbors_used,
            gamma_: out.gamma_used,
            n_graph_components_: out.n_graph_components,
            n_features_in_: n_features,
            n_samples_: n_samples,
            _state: PhantomData,
        })
    }
}

impl<F> Default for SpectralEmbedding<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`SpectralEmbedding`]. `gamma` is `Option<f64>` (the scalar
/// narrows to `Option<F>` at `build::<F>()`); `Default` re-derives the sklearn
/// defaults from [`SpectralEmbedding::new`] (single source).
#[derive(Debug, Clone)]
pub struct SpectralEmbeddingBuilder {
    n_components: usize,
    affinity: String,
    gamma: Option<f64>,
    random_state: Option<u64>,
    eigen_solver: Option<String>,
    eigen_tol: Option<f64>,
    n_neighbors: Option<usize>,
    n_jobs: Option<i64>,
}

impl Default for SpectralEmbeddingBuilder {
    /// Re-derive the sklearn defaults from [`SpectralEmbedding::new`]. `f64` is
    /// pinned only to read the F-independent scalar defaults.
    fn default() -> Self {
        SpectralEmbedding::<f64, Unfit>::new().into_builder()
    }
}

impl SpectralEmbeddingBuilder {
    /// Set the embedding dimensionality.
    pub fn n_components(mut self, v: usize) -> Self {
        self.n_components = v;
        self
    }

    /// Set the affinity construction (`"nearest_neighbors"` / `"rbf"` /
    /// `"precomputed"` / `"precomputed_nearest_neighbors"`).
    pub fn affinity(mut self, v: String) -> Self {
        self.affinity = v;
        self
    }

    /// Set the kernel coefficient `γ` (`None` → `1/n_features` at fit).
    pub fn gamma(mut self, v: Option<f64>) -> Self {
        self.gamma = v;
        self
    }

    /// Set the seed for the deterministic Krylov start block (sklearn
    /// `random_state`).
    pub fn random_state(mut self, v: Option<u64>) -> Self {
        self.random_state = v;
        self
    }

    /// Set `eigen_solver` (`"arpack"` / `"lobpcg"` / `"amg"` / `None`).
    pub fn eigen_solver(mut self, v: Option<String>) -> Self {
        self.eigen_solver = v;
        self
    }

    /// Set `eigen_tol` (`None` is sklearn's `"auto"`).
    pub fn eigen_tol(mut self, v: Option<f64>) -> Self {
        self.eigen_tol = v;
        self
    }

    /// Set the neighbor count (`None` → `max(n_samples / 10, 1)` at fit).
    pub fn n_neighbors(mut self, v: Option<usize>) -> Self {
        self.n_neighbors = v;
        self
    }

    /// Set `n_jobs` (accepted for signature compatibility).
    pub fn n_jobs(mut self, v: Option<i64>) -> Self {
        self.n_jobs = v;
        self
    }

    /// Build the (unfit) estimator, narrowing the stored `Option<f64>` `gamma`
    /// to the target float `Option<F>`.
    ///
    /// Only the data-INDEPENDENT bound is checked here (`eigen_tol >= 0`, a
    /// finite real). `n_components` is compared against `n_samples` and
    /// `gamma`'s admissibility is affinity-coupled, so both stay in the fit
    /// body.
    pub fn build<F>(self) -> Result<SpectralEmbedding<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if let Some(t) = self.eigen_tol {
            if !(t >= 0.0) || !t.is_finite() {
                return Err(BuildError::InvalidEps {
                    estimator: "spectral_embedding",
                    eps: t,
                });
            }
        }
        Ok(SpectralEmbedding {
            n_components: self.n_components,
            affinity: self.affinity,
            gamma: self.gamma.map(f64_to_host::<F>),
            random_state: self.random_state,
            eigen_solver: self.eigen_solver,
            eigen_tol: self.eigen_tol,
            n_neighbors: self.n_neighbors,
            n_jobs: self.n_jobs,
            embedding_: None,
            affinity_matrix_: None,
            n_neighbors_: None,
            gamma_: None,
            n_graph_components_: 0,
            n_features_in_: 0,
            n_samples_: 0,
            _state: PhantomData,
        })
    }
}

impl<F> SpectralEmbedding<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `embedding_` (`n × n_components` row-major).
    /// `Some` by construction on the `Fitted` state, so no `NotFitted` branch is
    /// needed.
    pub fn embedding(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.embedding_
            .as_ref()
            .expect("embedding_ is Some by construction on SpectralEmbedding<F, Fitted>")
            .to_host(pool)
    }

    /// `affinity_matrix_` densified to a row-major `n × n` matrix.
    ///
    /// The kNN affinity is stored SPARSE, so this materializes `n²` values on
    /// demand — reach for [`Self::affinity_matrix_sparse`] instead when the
    /// caller can consume CSR (which is what sklearn returns for that affinity).
    pub fn affinity_matrix_dense(&self) -> Vec<f64> {
        self.affinity_matrix_
            .as_ref()
            .expect("affinity_matrix_ is Some by construction on the Fitted state")
            .to_dense(self.n_samples_)
    }

    /// `affinity_matrix_` as CSR when the affinity builder produced a sparse
    /// graph (`nearest_neighbors` / `precomputed_nearest_neighbors`), else
    /// `None`. Mirrors sklearn, which returns a `csr_matrix` there and a dense
    /// `ndarray` for the kernel affinities.
    pub fn affinity_matrix_sparse(&self) -> Option<&Csr> {
        match self.affinity_matrix_.as_ref() {
            Some(HostAffinity::Sparse(c)) => Some(c),
            _ => None,
        }
    }

    /// sklearn's `n_neighbors_` — the RESOLVED neighbor count. `None` unless a
    /// kNN affinity was used, matching sklearn's attribute presence.
    pub fn n_neighbors_(&self) -> Option<usize> {
        self.n_neighbors_
    }

    /// sklearn's `gamma_` — the RESOLVED kernel coefficient. `None` unless a
    /// kernel affinity was used.
    pub fn gamma_(&self) -> Option<f64> {
        self.gamma_
    }

    /// Connected components of the fitted affinity graph. sklearn emits
    /// `"Graph is not fully connected, spectral embedding may not work as
    /// expected."` when this exceeds 1 and changes nothing else; exposing the
    /// count lets the binding layer raise the same warning.
    pub fn n_graph_components(&self) -> usize {
        self.n_graph_components_
    }

    /// `n_features_in_`.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    /// Number of training samples (the `embedding_` row count).
    pub fn n_samples(&self) -> usize {
        self.n_samples_
    }
}

impl<F> Fit<F> for SpectralEmbedding<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = SpectralEmbedding<F, Fitted>;

    /// Fit the spectral embedding to the affinity graph of `x`
    /// (`shape = (n_samples, n_features)`, row-major), CONSUMING `self`.
    ///
    /// This is the DEVICE-operand entry point, kept so the estimator stays on
    /// the single `Fit` trait surface. Because the pipeline is host-side it
    /// reads `x` back first, which costs two copies. Callers already holding
    /// host data should branch on
    /// [`SpectralEmbedding::host_fit_applicable`] and use
    /// [`SpectralEmbedding::fit_from_host_slice`], which never uploads at all.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<SpectralEmbedding<F, Fitted>, AlgoError> {
        let x64: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        self.fit_host_core(pool, &x64, shape)
    }
}
