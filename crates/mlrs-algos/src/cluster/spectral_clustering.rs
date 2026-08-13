//! `SpectralClustering` (SPECTRAL-02) — spectral embedding of an affinity graph
//! followed by a discrete label assignment, matching
//! `sklearn.cluster.SpectralClustering`.
//!
//! ## Pipeline
//! `affinity A` → normalized Laplacian `(L, dd)` → the smallest `n_components`
//! eigenvectors (`drop_first = FALSE` — the trivial ≈0 eigenvector is KEPT, and
//! sklearn is explicit that spectral CLUSTERING wants it) → `D^-1/2` recovery
//! (`/dd`) → deterministic sign flip → `maps` (`n × n_components`) →
//! `assign_labels` → `labels_`.
//!
//! Everything up to `maps` is [`crate::cluster::spectral_host::run`], shared
//! verbatim with `SpectralEmbedding`, so the two estimators cannot drift on the
//! Laplacian convention, the `_set_diag` fix-up, or the sign-flip ORDER (sklearn
//! takes the argmax over the ALREADY-`/dd`-scaled vector).
//!
//! ## The `n_samples <= 64` cap is GONE (SPECTRAL-PERF-CPU)
//! The former implementation built a dense `n × n` affinity and Laplacian on the
//! device and asked the cyclic-Jacobi `eig` kernel for the FULL spectrum. That
//! kernel stages its working matrices as comptime-sized shared memory at
//! `MAX_DIM × MAX_DIM`, so it rejected `n > 64` — which made the estimator
//! unusable on any real dataset, and `O(n³)` even on the toy ones. The host
//! pipeline has no size cap: the `nearest_neighbors` affinity stays SPARSE and
//! only the wanted eigenpairs are computed.
//!
//! The label-assignment stage moved to the host with it. The device
//! [`KMeans`](crate::cluster::kmeans::KMeans) has no `n_init` (sklearn runs 10
//! restarts and keeps the best inertia) and, on the `cpu` backend, every Lloyd
//! step is a cubecl launch onto a runtime that spawns one OS thread per unit and
//! JITs at `-O0` — pathological for the `d = n_components` geometry a spectral
//! embedding produces. [`spectral_host::host_kmeans`] replaces it.
//!
//! ## sklearn parameter surface
//! `SpectralClustering(n_clusters=8, *, eigen_solver=None, n_components=None,
//! random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10,
//! eigen_tol='auto', assign_labels='kmeans', degree=3, coef0=1,
//! kernel_params=None, n_jobs=None, verbose=False)` — every parameter is present
//! except `kernel_params`, with these resolution rules (all verified against the
//! installed `sklearn/cluster/_spectral.py`, 1.9.0):
//!
//! - `gamma` defaults to the LITERAL `1.0`, NOT `SpectralEmbedding`'s
//!   `1/n_features`. The constraint is `Interval(Real, 0, None, closed="left")`,
//!   so `gamma = 0` is LEGAL here (it yields a constant all-ones affinity); the
//!   pre-rewrite code rejected it, citing a `closed="neither"` interval that 1.9
//!   does not have.
//! - `n_neighbors` defaults to the INT `10`, not `SpectralEmbedding`'s
//!   `None → max(n_samples // 10, 1)`. The `None` branch is unreachable here, so
//!   the plan always carries `Some(n_neighbors)`.
//! - `n_components = None` → `n_clusters`.
//! - `affinity` accepts everything `pairwise_kernels` does (`linear`, `poly` /
//!   `polynomial`, `sigmoid`, `rbf`, `laplacian`, `cosine`, `chi2`,
//!   `additive_chi2` — i.e. exactly `KERNEL_PARAMS`) plus `nearest_neighbors`,
//!   `precomputed` and `precomputed_nearest_neighbors`. A CALLABLE affinity is
//!   not supported; there is no Rust analogue of handing sklearn a Python
//!   function, and the string set is the whole non-callable surface.
//! - `kernel_params` is DELIBERATELY ABSENT, and this is a parity decision, not
//!   an omission. sklearn merges the dict and then executes
//!   `params["gamma"] = self.gamma; params["degree"] = self.degree;
//!   params["coef0"] = self.coef0` whenever `affinity` is not callable —
//!   OVERWRITING any of those three the dict supplied — and calls
//!   `pairwise_kernels(..., filter_params=True)`, which DROPS every key outside
//!   `KERNEL_PARAMS[metric]`. Since `KERNEL_PARAMS` contains only `gamma`,
//!   `degree` and `coef0`, a `kernel_params` dict is provably a no-op for every
//!   string affinity. It can only matter for a callable affinity, which is out
//!   of scope, so accepting it would be accepting a parameter that could never
//!   change a result.
//! - `eigen_solver` ∈ {`arpack`, `lobpcg`, `amg`, `None`} is accepted and
//!   VALIDATED against the same set `SpectralEmbedding` uses. All four name a
//!   way to reach the SAME invariant subspace; mlrs has one solver (dense below
//!   `DENSE_N`, a restarted block Krylov iteration above) and routes every value
//!   to it, so the
//!   parameter selects nothing — but an out-of-set string is rejected exactly as
//!   sklearn rejects it rather than silently ignored.
//! - `eigen_tol = "auto"` (`None` here) is the solver's own machine-precision
//!   target, which is what sklearn's `tol=0` asks ARPACK for.
//! - `n_jobs` is accepted for signature compatibility. The host pipeline sizes
//!   its worker pool from `MLRS_CPU_UNITS` / available parallelism rather than
//!   from this parameter.
//! - `verbose` is accepted and stored. sklearn's only use of it is a
//!   `print(f"Computing label assignment using {self.assign_labels}")` plus
//!   forwarding to `k_means`; a library crate must not write to stdout, so it is
//!   exposed via [`SpectralClustering::verbose`] for a binding layer to act on.
//! - `random_state` seeds BOTH the deterministic Krylov start block and the label
//!   assignment (k-means++ draws / the `discretize` initial rotation), mirroring
//!   sklearn threading one `random_state` through both stages.
//!
//! ## Fitted attributes
//! sklearn sets `affinity_matrix_`, `labels_` and `n_features_in_`; all three
//! are exposed, plus the connected-component count behind the `"Graph is not
//! fully connected"` warning sklearn emits.
//!
//! Tests live in `crates/mlrs-algos/tests/spectral_clustering_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)] mod tests`).

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64};

use crate::cluster::spectral_embedding::EIGEN_SOLVERS;
use crate::cluster::spectral_host::{self, AssignLabels, Csr, HostAffinity, SpectralPlan};
use crate::cluster::cluster_persist::{
    read_affinity, read_labels, widen_labels, write_labels, AffinityStaging, AlignedBytes,
    ClusterFile, ClusterWriter, LoadModel, PersistError, SaveModel,
};
use crate::error::{AlgoError, BuildError};
use crate::typestate::{Fit, Fitted, Unfit};

/// The `estimator` discriminator written into every `SpectralClustering` file.
const PERSIST_TAG: &str = "spectral_clustering";

/// Spectral clustering (SPECTRAL-02): the spectral embedding of an affinity
/// graph, discretized into `n_clusters` labels.
///
/// Construct with the zero-arg [`SpectralClustering::new`] (sklearn defaults) or
/// [`SpectralClustering::builder`], then the consuming [`Fit::fit`] (or the
/// no-upload [`SpectralClustering::fit_from_host_slice`]) and read `labels_`.
/// The fitted accessors exist ONLY on `SpectralClustering<F, Fitted>` — the
/// compile-time typestate replaces the runtime `NotFitted` guard.
pub struct SpectralClustering<F, S = Unfit> {
    /// Number of clusters `k` (sklearn default `8`). `1 <= k` is checked at
    /// `build`; the data-DEPENDENT `k <= n_samples` at `fit`.
    n_clusters: usize,
    /// `eigen_solver` — validated against [`EIGEN_SOLVERS`], see the module docs.
    eigen_solver: Option<String>,
    /// Embedding dimensionality; `None` resolves to `n_clusters` at `fit`.
    n_components: Option<usize>,
    /// Seed for the deterministic Krylov start block AND the label assignment
    /// (sklearn's `random_state`, which likewise serves both).
    random_state: Option<u64>,
    /// Number of k-means restarts, lowest inertia winning (sklearn default
    /// `10`). Used only by `assign_labels = "kmeans"`.
    n_init: usize,
    /// Kernel coefficient `γ` (sklearn default `1.0` LITERAL — not
    /// `SpectralEmbedding`'s `1/n_features`). Ignored by the non-kernel
    /// affinities.
    gamma: F,
    /// Affinity construction (`"rbf"` default).
    affinity: String,
    /// Neighbor count for the kNN affinities (sklearn default `10`).
    n_neighbors: usize,
    /// `eigen_tol`; `None` is sklearn's `"auto"`.
    eigen_tol: Option<f64>,
    /// Label-assignment strategy (`"kmeans"` / `"discretize"` / `"cluster_qr"`),
    /// kept as the raw string so an invalid value is rejected at `fit` where
    /// sklearn's `StrOptions` rejects it.
    assign_labels: String,
    /// `pairwise_kernels` polynomial degree (sklearn default `3`).
    degree: f64,
    /// `pairwise_kernels` independent term (sklearn default `1`).
    coef0: f64,
    /// `n_jobs`, accepted for signature compatibility (see the module docs).
    n_jobs: Option<i64>,
    /// `verbose`, accepted and stored (see the module docs).
    verbose: bool,

    /// Fitted length-`n` integer labels (`i32`, the KMeans idiom),
    /// device-resident, `None` until `fit`.
    labels_: Option<DeviceArray<ActiveRuntime, i32>>,
    /// The host copy of `labels_` produced by `fit`, kept so `fit_predict` can
    /// build its returned device buffer WITHOUT the extra device→host read-back
    /// that calling `self.labels(pool)` would incur.
    labels_host_: Option<Vec<i32>>,
    /// Fitted `affinity_matrix_`, in its builder's layout.
    affinity_matrix_: Option<HostAffinity>,
    /// Connected components of the affinity graph. sklearn warns when this is
    /// `> 1`; it changes nothing else.
    n_graph_components_: usize,
    /// `n_features_in_`.
    n_features_in_: usize,
    /// `n_samples` seen at fit (the `labels_` length).
    n_samples_: usize,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> SpectralClustering<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfitted `SpectralClustering` with sklearn's defaults
    /// (`n_clusters = 8`, `eigen_solver = None`, `n_components = None`,
    /// `random_state = None`, `n_init = 10`, `gamma = 1.0`, `affinity = "rbf"`,
    /// `n_neighbors = 10`, `eigen_tol = "auto"`, `assign_labels = "kmeans"`,
    /// `degree = 3`, `coef0 = 1`, `n_jobs = None`, `verbose = false`) directly
    /// in the `Unfit` state. SINGLE source of truth for the defaults: the
    /// builder `Default` re-derives via [`SpectralClustering::into_builder`].
    pub fn new() -> Self {
        Self {
            n_clusters: 8,
            eigen_solver: None,
            n_components: None,
            random_state: None,
            n_init: 10,
            gamma: F::from_int(1),
            affinity: "rbf".to_string(),
            n_neighbors: 10,
            eigen_tol: None,
            assign_labels: "kmeans".to_string(),
            degree: 3.0,
            coef0: 1.0,
            n_jobs: None,
            verbose: false,
            labels_: None,
            labels_host_: None,
            affinity_matrix_: None,
            n_graph_components_: 0,
            n_features_in_: 0,
            n_samples_: 0,
            _state: PhantomData,
        }
    }

    /// Start building a `SpectralClustering` from sklearn's defaults.
    pub fn builder() -> SpectralClusteringBuilder {
        SpectralClusteringBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`SpectralClusteringBuilder::default`] to
    /// re-derive the defaults from [`SpectralClustering::new`].
    pub fn into_builder(self) -> SpectralClusteringBuilder {
        SpectralClusteringBuilder {
            n_clusters: self.n_clusters,
            eigen_solver: self.eigen_solver,
            n_components: self.n_components,
            random_state: self.random_state,
            n_init: self.n_init,
            gamma: host_to_f64(self.gamma),
            affinity: self.affinity,
            n_neighbors: self.n_neighbors,
            eigen_tol: self.eigen_tol,
            assign_labels: self.assign_labels,
            degree: self.degree,
            coef0: self.coef0,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// attributes are excluded — all absent in any `Unfit` value). Used by the
    /// defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_clusters == other.n_clusters
            && self.eigen_solver == other.eigen_solver
            && self.n_components == other.n_components
            && self.random_state == other.random_state
            && self.n_init == other.n_init
            && host_to_f64(self.gamma) == host_to_f64(other.gamma)
            && self.affinity == other.affinity
            && self.n_neighbors == other.n_neighbors
            && self.eigen_tol == other.eigen_tol
            && self.assign_labels == other.assign_labels
            && self.degree == other.degree
            && self.coef0 == other.coef0
            && self.n_jobs == other.n_jobs
            && self.verbose == other.verbose
    }

    /// Whether the caller should reach [`Self::fit_from_host_slice`] instead of
    /// uploading `x` and calling [`Fit::fit`].
    ///
    /// Always `true`: the spectral pipeline and all three label-assignment
    /// strategies are a single HOST implementation on every backend, so an
    /// upload before `fit` is pure waste — `from_host` copies once and the
    /// `to_host` that `fit` would then need copies twice. The predicate exists
    /// anyway because the two entry points take DIFFERENT operand types (host
    /// slice vs `DeviceArray`), so the choice has to be made BEFORE ingress;
    /// keeping it named matches the `SpectralEmbedding` / Ridge shape and leaves
    /// one place to change if a device arm ever returns.
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
    ) -> Result<SpectralClustering<F, Fitted>, AlgoError> {
        let x64: Vec<f64> = x.iter().map(|&v| host_to_f64(v)).collect();
        self.fit_host_core(pool, &x64, shape)
    }

    /// The one implementation both entry points reach, so they cannot drift.
    fn fit_host_core(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x64: &[f64],
        shape: (usize, usize),
    ) -> Result<SpectralClustering<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        if n_samples == 0 || n_features == 0 {
            return Err(AlgoError::InvalidGraphInput {
                estimator: "spectral_clustering",
                reason: format!("empty design ({n_samples} x {n_features})"),
            });
        }
        // `eigen_solver` and `assign_labels` are validated at `fit` rather than
        // at `build` so the error surfaces at the same point sklearn's
        // `_fit_context` `StrOptions` validation does.
        if let Some(s) = self.eigen_solver.as_deref() {
            if !EIGEN_SOLVERS.contains(&s) {
                return Err(AlgoError::Unsupported {
                    estimator: "spectral_clustering",
                    operation: "eigen_solver (expected one of arpack / lobpcg / amg)",
                });
            }
        }
        let assign = AssignLabels::parse(&self.assign_labels).ok_or(AlgoError::Unsupported {
            estimator: "spectral_clustering",
            operation: "assign_labels (expected one of kmeans / discretize / cluster_qr)",
        })?;
        // The `1 <= n_clusters <= n_samples` upper half is data-DEPENDENT: you
        // cannot ask for more clusters than there are samples. sklearn surfaces
        // the same condition from inside `k_means`.
        if self.n_clusters < 1 || self.n_clusters > n_samples {
            return Err(AlgoError::InvalidK {
                estimator: "spectral_clustering",
                k: self.n_clusters,
                n_samples,
            });
        }
        let n_components = self.n_components.unwrap_or(self.n_clusters);
        let seed = self.random_state.unwrap_or(0);

        let plan = SpectralPlan {
            estimator: "spectral_clustering",
            affinity: &self.affinity,
            // The LITERAL default (module docs): unlike `SpectralEmbedding`,
            // sklearn's `SpectralClustering.gamma` is never `None`, so the
            // `1/n_features` fork in the shared plan is never taken.
            gamma: Some(host_to_f64(self.gamma)),
            degree: self.degree,
            coef0: self.coef0,
            // Always `Some`: sklearn's default is the int 10, so the plan's
            // `None → max(n_samples / 10, 1)` branch belongs to
            // `SpectralEmbedding` alone.
            n_neighbors: Some(self.n_neighbors),
            n_components,
            // KEEP the trivial ≈0 eigenvector — sklearn's comment is explicit
            // that spectral clustering wants it as a feature.
            drop_first: false,
            seed,
            // The full `pairwise_kernels` family is in scope here.
            allow_kernels: true,
        };
        let out = spectral_host::run(&plan, x64, n_samples, n_features)?;

        // `maps` is the `n × n_components` embedding; every assignment strategy
        // consumes it unchanged.
        let maps = &out.embedding;
        let labels_host: Vec<i32> = match assign {
            AssignLabels::KMeans => {
                spectral_host::host_kmeans(
                    maps,
                    n_samples,
                    n_components,
                    self.n_clusters,
                    self.n_init,
                    seed,
                )
                .labels
            }
            AssignLabels::ClusterQr => {
                // sklearn's `cluster_qr` derives the cluster count from the
                // embedding WIDTH, not from `n_clusters` — it returns an argmax
                // over `n_components` columns. With the default
                // `n_components = None → n_clusters` the two coincide; when they
                // do not, sklearn's label range is `n_components`, and so is
                // this one.
                spectral_host::cluster_qr_labels(maps, n_samples, n_components)
            }
            AssignLabels::Discretize => {
                // Same width-not-`n_clusters` remark as `cluster_qr`: sklearn's
                // `discretize` builds an `n_components`-column partition matrix.
                spectral_host::discretize_labels(maps, n_samples, n_components, seed)?
            }
        };
        let labels_dev: DeviceArray<ActiveRuntime, i32> =
            DeviceArray::from_host(pool, &labels_host);

        Ok(SpectralClustering {
            n_clusters: self.n_clusters,
            eigen_solver: self.eigen_solver,
            n_components: self.n_components,
            random_state: self.random_state,
            n_init: self.n_init,
            gamma: self.gamma,
            affinity: self.affinity,
            n_neighbors: self.n_neighbors,
            eigen_tol: self.eigen_tol,
            assign_labels: self.assign_labels,
            degree: self.degree,
            coef0: self.coef0,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
            labels_: Some(labels_dev),
            labels_host_: Some(labels_host),
            affinity_matrix_: Some(out.affinity),
            n_graph_components_: out.n_graph_components,
            n_features_in_: n_features,
            n_samples_: n_samples,
            _state: PhantomData,
        })
    }

    /// Convenience `fit_predict` (sklearn `ClusterMixin`): fit to `x` then
    /// return the fitted `labels_` as a fresh device-resident `i32` buffer.
    /// CONSUMES `self` (the typestate `fit` transition) and returns BOTH the
    /// `Fitted`-tagged estimator and the labels buffer.
    ///
    /// The returned buffer is built directly from the host labels `fit` just
    /// materialized, avoiding the device→host→device round trip a fresh
    /// read-back of `labels_` would incur. It is an INDEPENDENT device
    /// allocation — it does not alias `labels_`, so the caller may
    /// `release_into` it freely.
    pub fn fit_predict(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<(SpectralClustering<F, Fitted>, DeviceArray<ActiveRuntime, i32>), AlgoError> {
        let fitted = self.fit(pool, x, None, shape)?;
        // `fit` always sets `labels_host_` on success; the `expect` is a
        // defensive fallback that cannot trigger on the post-`fit` path.
        let labels = fitted
            .labels_host_
            .as_ref()
            .expect("labels_host_ is Some by construction after fit");
        let labels_dev = DeviceArray::from_host(pool, labels);
        Ok((fitted, labels_dev))
    }
}

impl<F> Default for SpectralClustering<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`SpectralClustering`] — one setter per sklearn parameter.
/// `gamma` / `degree` / `coef0` are `f64`-typed (the scalar `gamma` narrows to
/// `F` at `build::<F>()`); `Default` re-derives the sklearn defaults from
/// [`SpectralClustering::new`] (single source).
#[derive(Debug, Clone)]
pub struct SpectralClusteringBuilder {
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
}

impl Default for SpectralClusteringBuilder {
    /// Re-derive the sklearn defaults from [`SpectralClustering::new`]. `f64` is
    /// pinned only to read the F-independent scalar defaults — the builder is
    /// non-generic, so the choice of `F` is irrelevant.
    fn default() -> Self {
        SpectralClustering::<f64, Unfit>::new().into_builder()
    }
}

impl SpectralClusteringBuilder {
    /// Set the number of clusters `k`.
    pub fn n_clusters(mut self, v: usize) -> Self {
        self.n_clusters = v;
        self
    }

    /// Set `eigen_solver` (`"arpack"` / `"lobpcg"` / `"amg"` / `None`), which
    /// names a route to the same invariant subspace rather than selecting one
    /// (module docs).
    pub fn eigen_solver(mut self, v: Option<String>) -> Self {
        self.eigen_solver = v;
        self
    }

    /// Set the embedding dimensionality (`None` → `n_clusters` at fit).
    pub fn n_components(mut self, v: Option<usize>) -> Self {
        self.n_components = v;
        self
    }

    /// Set sklearn's `random_state`, which seeds the deterministic Krylov start
    /// AND the label assignment.
    pub fn random_state(mut self, v: Option<u64>) -> Self {
        self.random_state = v;
        self
    }

    /// Set the seed. Retained spelling of [`Self::random_state`] for callers
    /// that predate the sklearn-named setter; `seed(v)` is exactly
    /// `random_state(Some(v))`.
    pub fn seed(mut self, v: u64) -> Self {
        self.random_state = Some(v);
        self
    }

    /// Set the number of k-means restarts (lowest inertia wins). Used only by
    /// `assign_labels = "kmeans"`, exactly as in sklearn.
    pub fn n_init(mut self, v: usize) -> Self {
        self.n_init = v;
        self
    }

    /// Set the kernel coefficient `γ` (narrowed to `F` at `build::<F>()`).
    pub fn gamma(mut self, v: f64) -> Self {
        self.gamma = v;
        self
    }

    /// Set the affinity construction — any `pairwise_kernels` metric plus
    /// `"nearest_neighbors"` / `"precomputed"` /
    /// `"precomputed_nearest_neighbors"` (module docs).
    pub fn affinity(mut self, v: String) -> Self {
        self.affinity = v;
        self
    }

    /// Set the neighbor count for the kNN affinities.
    pub fn n_neighbors(mut self, v: usize) -> Self {
        self.n_neighbors = v;
        self
    }

    /// Set `eigen_tol` (`None` is sklearn's `"auto"`).
    pub fn eigen_tol(mut self, v: Option<f64>) -> Self {
        self.eigen_tol = v;
        self
    }

    /// Set the label-assignment strategy (`"kmeans"` / `"discretize"` /
    /// `"cluster_qr"`).
    pub fn assign_labels(mut self, v: String) -> Self {
        self.assign_labels = v;
        self
    }

    /// Set the polynomial-kernel degree.
    pub fn degree(mut self, v: f64) -> Self {
        self.degree = v;
        self
    }

    /// Set the polynomial / sigmoid independent term.
    pub fn coef0(mut self, v: f64) -> Self {
        self.coef0 = v;
        self
    }

    /// Set `n_jobs` (accepted for signature compatibility).
    pub fn n_jobs(mut self, v: Option<i64>) -> Self {
        self.n_jobs = v;
        self
    }

    /// Set `verbose` (accepted and stored; a library crate must not print).
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Build the (unfit) estimator, narrowing the stored `f64` `gamma` to the
    /// target float `F`.
    ///
    /// Only the data-INDEPENDENT bounds are checked here, and each mirrors the
    /// matching sklearn `_parameter_constraints` entry:
    /// `n_clusters >= 1`, `n_components >= 1`, `n_init >= 1`, `n_neighbors >= 1`
    /// (`Interval(Integral, 1, None, closed="left")`), `eigen_tol >= 0` finite
    /// and `degree >= 0` finite (`Interval(Real, 0, None, closed="left")`), and
    /// `coef0` finite (`Interval(Real, None, None, closed="neither")`).
    ///
    /// `gamma`'s admissibility is affinity-coupled (only the kernel affinities
    /// consume it) and `n_clusters <= n_samples` needs the data, so both stay in
    /// the fit body. `gamma >= 0` — the `closed="left"` bound, which ADMITS
    /// zero — is enforced there by the shared plan.
    pub fn build<F>(self) -> Result<SpectralClustering<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if self.n_clusters < 1 {
            return Err(BuildError::InvalidNClusters {
                estimator: "spectral_clustering",
                n_clusters: self.n_clusters,
            });
        }
        if let Some(c) = self.n_components {
            if c < 1 {
                return Err(BuildError::InvalidNComponents {
                    estimator: "spectral_clustering",
                    param: "n_components",
                    value: c,
                });
            }
        }
        if self.n_init < 1 {
            return Err(BuildError::InvalidNComponents {
                estimator: "spectral_clustering",
                param: "n_init",
                value: self.n_init,
            });
        }
        if self.n_neighbors < 1 {
            return Err(BuildError::InvalidNNeighbors {
                estimator: "spectral_clustering",
                n_neighbors: self.n_neighbors,
            });
        }
        if let Some(t) = self.eigen_tol {
            if !(t >= 0.0) || !t.is_finite() {
                return Err(BuildError::InvalidEps {
                    estimator: "spectral_clustering",
                    eps: t,
                });
            }
        }
        if !(self.degree >= 0.0) || !self.degree.is_finite() {
            return Err(BuildError::InvalidHyperprior {
                estimator: "spectral_clustering",
                param: "degree",
                value: self.degree,
                bound: ">= 0",
            });
        }
        if !self.coef0.is_finite() {
            return Err(BuildError::InvalidHyperprior {
                estimator: "spectral_clustering",
                param: "coef0",
                value: self.coef0,
                bound: "a real number",
            });
        }
        Ok(SpectralClustering {
            n_clusters: self.n_clusters,
            eigen_solver: self.eigen_solver,
            n_components: self.n_components,
            random_state: self.random_state,
            n_init: self.n_init,
            gamma: f64_to_host::<F>(self.gamma),
            affinity: self.affinity,
            n_neighbors: self.n_neighbors,
            eigen_tol: self.eigen_tol,
            assign_labels: self.assign_labels,
            degree: self.degree,
            coef0: self.coef0,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
            labels_: None,
            labels_host_: None,
            affinity_matrix_: None,
            n_graph_components_: 0,
            n_features_in_: 0,
            n_samples_: 0,
            _state: PhantomData,
        })
    }
}

impl<F> SpectralClustering<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `labels_` (length `n`, `i32`). `Some` by
    /// construction on the `Fitted` state, so no `NotFitted` branch is needed.
    pub fn labels(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<i32> {
        self.labels_
            .as_ref()
            .expect("labels_ is Some by construction on SpectralClustering<F, Fitted>")
            .to_host(pool)
    }

    /// `affinity_matrix_` densified to a row-major `n × n` matrix.
    ///
    /// The kNN affinities are stored SPARSE, so this materializes `n²` values on
    /// demand — reach for [`Self::affinity_matrix_sparse`] instead when the
    /// caller can consume CSR (which is what sklearn returns there).
    pub fn affinity_matrix_dense(&self) -> Vec<f64> {
        self.affinity_matrix_
            .as_ref()
            .expect("affinity_matrix_ is Some by construction on the Fitted state")
            .to_dense(self.n_samples_)
    }

    /// `affinity_matrix_` as CSR when the affinity builder produced a sparse
    /// graph (`nearest_neighbors` / `precomputed_nearest_neighbors`), else
    /// `None`. Mirrors sklearn, which returns a `csr_matrix` there and a dense
    /// `ndarray` for the kernel and `precomputed` affinities.
    pub fn affinity_matrix_sparse(&self) -> Option<&Csr> {
        match self.affinity_matrix_.as_ref() {
            Some(HostAffinity::Sparse(c)) => Some(c),
            _ => None,
        }
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

    /// Number of training samples (the `labels_` length).
    pub fn n_samples(&self) -> usize {
        self.n_samples_
    }

    /// The stored `verbose` flag. sklearn prints
    /// `"Computing label assignment using <assign_labels>"` when it is set; a
    /// library crate must not write to stdout, so the flag is surfaced for a
    /// binding layer to act on instead.
    pub fn verbose(&self) -> bool {
        self.verbose
    }
}

impl<F> SaveModel for SpectralClustering<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted clustering to `path` as a safetensors file.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `labels_` | `I64` | `[n_samples]` |
    /// | the affinity graph | `F64` (+ `U64` indices) | see [`AffinityStaging`] |
    /// | `n_graph_components_` / `n_features_in_` | `__metadata__` scalar | — |
    /// | fourteen `param:*` scalars | `__metadata__` | — |
    ///
    /// The affinity graph is the substantial part of this file and is stored in
    /// WHICHEVER layout the fit produced — dense for a kernel affinity, CSR for
    /// a neighborhood graph. Those are different models of the same data, not
    /// two encodings of one, so the layout round-trips alongside the values;
    /// [`AffinityStaging::write_into`] carries the reasoning.
    ///
    /// `n_graph_components_` is a fitted attribute worth storing rather than
    /// recomputing: it is the connected-component count of the affinity graph,
    /// and while it IS derivable from the stored graph, re-deriving it means a
    /// union-find sweep over every edge on a load path that is otherwise one
    /// sequential read. It is also the diagnostic that explains a degenerate
    /// embedding, so a reload that silently recomputed it could disagree with
    /// what the fit reported.
    ///
    /// `pool` is unused: this estimator keeps a host mirror of its labels
    /// (`labels_host_`) precisely so the accessor path needs no readback, and
    /// `save` reads that. The parameter is present because [`SaveModel`] is one
    /// signature for every estimator.
    fn save(&self, _pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let absent = |field| PersistError::MissingState {
            estimator: PERSIST_TAG,
            field,
        };
        let affinity = self
            .affinity_matrix_
            .as_ref()
            .ok_or_else(|| absent("affinity_matrix_"))?;
        // Bound BEFORE the writer, which borrows every payload. `AffinityStaging`
        // exists for exactly this: the CSR index arrays widen `u32 → u64`, and
        // the widened copies must outlive the writer.
        let labels = widen_labels(
            self.labels_host_
                .as_ref()
                .ok_or_else(|| absent("labels_"))?,
        );
        let staging = AffinityStaging::prepare(affinity);

        let mut w = ClusterWriter::new(PERSIST_TAG);
        w.scalar_usize("param:n_clusters", self.n_clusters);
        w.scalar_str("param:eigen_solver", self.eigen_solver.as_deref().unwrap_or("auto"));
        w.scalar_opt_usize("param:n_components", self.n_components);
        w.scalar_opt_u64("param:random_state", self.random_state);
        w.scalar_usize("param:n_init", self.n_init);
        w.scalar_f64("param:gamma", host_to_f64(self.gamma));
        w.scalar_str("param:affinity", &self.affinity);
        w.scalar_usize("param:n_neighbors", self.n_neighbors);
        w.scalar_opt_f64("param:eigen_tol", self.eigen_tol);
        w.scalar_str("param:assign_labels", &self.assign_labels);
        w.scalar_f64("param:degree", self.degree);
        w.scalar_f64("param:coef0", self.coef0);
        w.scalar_bool("param:verbose", self.verbose);
        if let Some(j) = self.n_jobs {
            w.scalar_str("param:n_jobs", &j.to_string());
        }
        w.scalar_usize("n_graph_components_", self.n_graph_components_);
        w.scalar_usize("n_features_in_", self.n_features_in_);

        write_labels(&mut w, &labels)?;
        staging.write_into(&mut w, affinity, self.n_samples_)?;
        w.write(path)
    }
}

impl<F> LoadModel for SpectralClustering<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read the clustering back from `path`, re-uploading the label vector to
    /// `pool`.
    ///
    /// The affinity graph's every CSR invariant is validated by
    /// [`read_affinity`] against the sample count `labels_` establishes — the
    /// file is untrusted input (T-04-01-01), and a malformed `indptr` would
    /// index out of bounds inside the Lanczos matvec rather than report a bad
    /// file.
    ///
    /// `labels_host_` is repopulated alongside the device copy, because it is
    /// not a memo here: the accessor reads it directly, so a load that left it
    /// `None` would produce a model whose labels are unreachable without a
    /// readback the estimator is built to avoid.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<SpectralClustering<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = ClusterFile::parse(&raw, PERSIST_TAG)?;
        let labels = read_labels(&file)?;
        let n_samples = labels.len();
        let affinity = read_affinity(&file, n_samples)?;

        // `n_jobs` is `Option<i64>` and rides as a string, so absence is a
        // meaningful `None` (sklearn's "unset") rather than a sentinel.
        let n_jobs = match file.metadata().get("param:n_jobs") {
            None => None,
            Some(s) => Some(s.parse::<i64>().map_err(|_| PersistError::BadMetadata {
                key: "param:n_jobs",
            })?),
        };
        // `eigen_solver` is `Option<String>` whose `None` MEANS "auto"; the
        // string is written unconditionally, so `"auto"` reads back as `None`
        // and every other value as itself.
        let eigen_solver = match file.scalar_str("param:eigen_solver")? {
            "auto" => None,
            other => Some(other.to_string()),
        };

        Ok(SpectralClustering {
            n_clusters: file.scalar_usize("param:n_clusters")?,
            eigen_solver,
            n_components: file.scalar_opt_usize("param:n_components")?,
            random_state: file.scalar_opt_u64("param:random_state")?,
            n_init: file.scalar_usize("param:n_init")?,
            gamma: f64_to_host::<F>(file.scalar_f64("param:gamma")?),
            affinity: file.scalar_str("param:affinity")?.to_string(),
            n_neighbors: file.scalar_usize("param:n_neighbors")?,
            eigen_tol: file.scalar_opt_f64("param:eigen_tol")?,
            assign_labels: file.scalar_str("param:assign_labels")?.to_string(),
            degree: file.scalar_f64("param:degree")?,
            coef0: file.scalar_f64("param:coef0")?,
            n_jobs,
            verbose: file.scalar_bool("param:verbose")?,
            labels_: Some(DeviceArray::from_host(pool, &labels)),
            labels_host_: Some(labels),
            affinity_matrix_: Some(affinity),
            n_graph_components_: file.scalar_usize("n_graph_components_")?,
            n_features_in_: file.scalar_usize("n_features_in_")?,
            n_samples_: n_samples,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for SpectralClustering<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = SpectralClustering<F, Fitted>;

    /// Fit the spectral clustering to the affinity graph of `x`
    /// (`shape = (n_samples, n_features)`, row-major), CONSUMING `self`.
    ///
    /// This is the DEVICE-operand entry point, kept so the estimator stays on
    /// the single `Fit` trait surface. Because the pipeline is host-side it
    /// reads `x` back first, which costs two copies. Callers already holding
    /// host data should branch on
    /// [`SpectralClustering::host_fit_applicable`] and use
    /// [`SpectralClustering::fit_from_host_slice`], which never uploads at all.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<SpectralClustering<F, Fitted>, AlgoError> {
        let x64: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        self.fit_host_core(pool, &x64, shape)
    }
}
