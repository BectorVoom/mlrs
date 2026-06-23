//! `Umap` (UMAP-01) — convention-foundation SHELL for the v3 builder +
//! typestate API (Phase 12).
//!
//! This file demonstrates the Phase-12 estimator convention END-TO-END but
//! contains NO algorithm: the [`Fit::fit`] body is a NON-algorithmic trivial fit
//! that allocates an all-zeros `embedding_` of shape `(n, n_components)` — no
//! kernel, no compute. The real UMAP algorithm (k-NN graph → fuzzy simplicial
//! set → spectral init → SGD layout) lands in Phase 14; until then this shell
//! gives Phases 14–15 a born-builder-fronted, typestate-correct surface to fill.
//!
//! ## The convention this shell embodies
//! - `Umap<F, S = Unfit>` carries its lifecycle state as a `PhantomData<S>`
//!   type parameter (D-01) so `transform`-before-`fit` is a COMPILE error.
//! - Construction is builder-only: [`Umap::builder`] →
//!   [`UmapBuilder::build`] (`-> Result<Umap<F, Unfit>, BuildError>`), with the
//!   data-INDEPENDENT hyperparameter validation up front (D-08 — `min_dist`
//!   finite and `<= spread`).
//! - [`Umap::new`] is the SINGLE source of the sklearn defaults (D-08); the
//!   builder `Default` re-derives via `new().into_builder()` rather than
//!   re-listing literals.
//! - [`Fit::fit`] CONSUMES `self` and returns `Umap<F, Fitted>` (D-02).
//! - The `embedding`/`n_features_in` accessors and [`Transform`] exist ONLY on
//!   `impl Umap<F, Fitted>` — that is what makes `transform`-before-`fit` a
//!   compile error rather than a runtime [`AlgoError::NotFitted`].
//!
//! Tests live in `crates/mlrs-algos/tests/umap_test.rs` (AGENTS.md §2).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{ArrayArg, CubeCount, CubeDim, CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::knn_graph::{self, knn_graph};
use mlrs_backend::prims::rng::SplitMix64;
use mlrs_backend::prims::topk::top_k;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};
use mlrs_kernels::{chebyshev_dist, manhattan_dist, minkowski_dist, umap_layout_step};

use crate::error::{AlgoError, BuildError};
use crate::manifold::{umap_init, umap_internals};
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

/// Distance metric for the UMAP neighbor graph (UMAP-01, full set — Phase 14).
///
/// This MIRRORS `mlrs_backend::prims::knn_graph::Metric` EXACTLY (same variants,
/// same `Minkowski { p: f64 }` payload — D-01 / PATTERNS Pitfall 4) so UMAP's
/// `metric=`/`p` map straight onto the KNN-graph prim with no lossy conversion.
/// The `Metric → knn_graph::Metric` mapping fn is added at the KNN call site in
/// Plan 04 (not here).
///
/// `Minkowski { p: f64 }` carries a non-`Eq` `f64`, so this enum derives
/// `PartialEq` but NOT `Eq`; `hyperparams_eq` compares via `PartialEq`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    /// L2 (Euclidean) — umap-learn's `metric='euclidean'` default.
    Euclidean,
    /// L1 (Manhattan) — umap-learn's `metric='manhattan'`.
    Manhattan,
    /// Cosine distance `1 − x̂·ŷ` — umap-learn's `metric='cosine'`.
    Cosine,
    /// L∞ (Chebyshev) — umap-learn's `metric='chebyshev'`.
    Chebyshev,
    /// Minkowski-`p` — umap-learn's `metric='minkowski'` with `p=` exponent
    /// (validated `>= 1` host-side at the KNN call site in Plan 04).
    Minkowski {
        /// The Minkowski exponent (matches `knn_graph::Metric::Minkowski.p`).
        p: f64,
    },
}

/// Initialization strategy for the low-dimensional embedding (UMAP-01 subset).
/// Ignored by the Phase-12 trivial fit (which always emits zeros); retained so
/// the builder surface matches umap-learn's `init=` parameter for Phase 14.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Init {
    /// Spectral embedding of the fuzzy graph — umap-learn's `init='spectral'`
    /// default.
    Spectral,
    /// Uniform-random init in the embedding box — umap-learn's `init='random'`.
    Random,
}

/// UMAP manifold-learning estimator shell (UMAP-01). Construct via
/// [`Umap::builder`], then [`Fit::fit`] (which CONSUMES `self`, returning
/// `Umap<F, Fitted>`). The fitted `embedding_` (shape `(n, n_components)`,
/// device-resident) is reachable only through [`Umap<F, Fitted>::embedding`].
///
/// The `S` type parameter is the compile-time lifecycle state
/// ([`Unfit`](crate::typestate::Unfit) /
/// [`Fitted`](crate::typestate::Fitted)); it defaults to `Unfit` so
/// `Umap::<F>::new()` lands in the freshly-built state.
pub struct Umap<F, S = Unfit> {
    // --- UMAP-01 hyperparameter surface (data-independent) ---
    /// Local neighborhood size (`n_neighbors`, default 15).
    n_neighbors: usize,
    /// Embedding dimensionality (`n_components`, default 2).
    n_components: usize,
    /// Minimum inter-point distance in the embedding (`min_dist`, default 0.1).
    min_dist: f64,
    /// Effective scale of embedded points (`spread`, default 1.0).
    spread: f64,
    /// Neighbor-graph distance metric (`metric`, default Euclidean).
    metric: Metric,
    /// Optimization epoch count (`n_epochs`, default `None` = auto).
    n_epochs: Option<usize>,
    /// Embedding initialization strategy (`init`, default Spectral).
    init: Init,
    /// RNG seed (`random_state`, default `None`).
    random_state: Option<u64>,
    /// Initial optimization learning rate (`learning_rate`, default 1.0).
    learning_rate: f64,
    /// Fuzzy-set union/intersection mix (`set_op_mix_ratio`, default 1.0).
    set_op_mix_ratio: f64,
    /// Assumed local connectivity (`local_connectivity`, default 1.0).
    local_connectivity: f64,
    /// Repulsion weight in the layout (`repulsion_strength`, default 1.0).
    repulsion_strength: f64,
    /// Negative-sample rate per positive edge (`negative_sample_rate`, default 5).
    negative_sample_rate: usize,
    /// `a` curve-fit override (`a`, default `None` = derived from min_dist/spread).
    a: Option<f64>,
    /// `b` curve-fit override (`b`, default `None`).
    b: Option<f64>,

    // --- fitted state (None / 0 until fit; Some on Fitted by construction) ---
    /// Fitted embedding `(n, n_components)`, device-resident. `None` until fit.
    embedding_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Retained training design rows `(n, n_features_in_)`, device-resident, kept
    /// for `transform`'s query-vs-train KNN (new→train runs in the ORIGINAL
    /// feature space; the fitted embedding alone is insufficient). `None` until
    /// fit (UMAP-04 / D-03 — umap-learn likewise retains the training data / KNN
    /// search index for `transform`).
    x_train_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Number of features seen at fit (`n_features_in_`). `0` until fit.
    n_features_in_: usize,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> Umap<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an `Umap` with umap-learn's defaults directly in the `Unfit`
    /// state (UMAP-01). This is the SINGLE source of truth for the default
    /// hyperparameters (D-08): the builder `Default` re-derives from here via
    /// [`Umap::into_builder`], rather than re-listing the literals. Defaults are
    /// trusted valid, so this bypasses [`UmapBuilder::build`]'s validation.
    pub fn new() -> Self {
        Self {
            n_neighbors: 15,
            n_components: 2,
            min_dist: 0.1,
            spread: 1.0,
            metric: Metric::Euclidean,
            n_epochs: None,
            init: Init::Spectral,
            random_state: None,
            learning_rate: 1.0,
            set_op_mix_ratio: 1.0,
            local_connectivity: 1.0,
            repulsion_strength: 1.0,
            negative_sample_rate: 5,
            a: None,
            b: None,
            embedding_: None,
            x_train_: None,
            n_features_in_: 0,
            _state: PhantomData,
        }
    }

    /// Start building an `Umap` from umap-learn's defaults (D-08 single source).
    pub fn builder() -> UmapBuilder {
        UmapBuilder::default()
    }

    /// `fit_transform` (UMAP-01): fit to `x` and return the fitted embedding host
    /// buffer in one call — umap-learn's `UMAP.fit_transform`. CONSUMES `self`
    /// (the `Fit::fit` contract). The returned `Vec<F>` is the row-major
    /// `(n, n_components)` embedding; the fitted estimator is dropped (callers who
    /// need the estimator should `fit` then `embedding`). `y` is ignored.
    pub fn fit_transform(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<F>, AlgoError> {
        let fitted = self.fit(pool, x, None, shape)?;
        Ok(fitted.embedding(pool))
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `embedding_`/`n_features_in_` fields are excluded — both are `None`/`0`
    /// in any `Unfit` value). Used by the defaults-equality test (BLDR-01):
    /// `Umap::new().hyperparams_eq(&Umap::builder().build()?)`.
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_neighbors == other.n_neighbors
            && self.n_components == other.n_components
            && self.min_dist == other.min_dist
            && self.spread == other.spread
            && self.metric == other.metric
            && self.n_epochs == other.n_epochs
            && self.init == other.init
            && self.random_state == other.random_state
            && self.learning_rate == other.learning_rate
            && self.set_op_mix_ratio == other.set_op_mix_ratio
            && self.local_connectivity == other.local_connectivity
            && self.repulsion_strength == other.repulsion_strength
            && self.negative_sample_rate == other.negative_sample_rate
            && self.a == other.a
            && self.b == other.b
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by `UmapBuilder::default` to re-derive the defaults
    /// from [`Umap::new`] (D-08) and available to callers who want to tweak a
    /// constructed estimator before fitting.
    pub fn into_builder(self) -> UmapBuilder {
        UmapBuilder {
            n_neighbors: self.n_neighbors,
            n_components: self.n_components,
            min_dist: self.min_dist,
            spread: self.spread,
            metric: self.metric,
            n_epochs: self.n_epochs,
            init: self.init,
            random_state: self.random_state,
            learning_rate: self.learning_rate,
            set_op_mix_ratio: self.set_op_mix_ratio,
            local_connectivity: self.local_connectivity,
            repulsion_strength: self.repulsion_strength,
            negative_sample_rate: self.negative_sample_rate,
            a: self.a,
            b: self.b,
        }
    }
}

impl<F> Default for Umap<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`Umap`] (D-01). Owned chained setters mirror
/// `mbsgd_regressor.rs`'s template; the ONLY deviation is that `Default`
/// re-derives the umap-learn defaults from [`Umap::new`] (D-08) rather than
/// holding literals.
#[derive(Debug, Clone, Copy)]
pub struct UmapBuilder {
    n_neighbors: usize,
    n_components: usize,
    min_dist: f64,
    spread: f64,
    metric: Metric,
    n_epochs: Option<usize>,
    init: Init,
    random_state: Option<u64>,
    learning_rate: f64,
    set_op_mix_ratio: f64,
    local_connectivity: f64,
    repulsion_strength: f64,
    negative_sample_rate: usize,
    a: Option<f64>,
    b: Option<f64>,
}

impl Default for UmapBuilder {
    /// Re-derive the umap-learn defaults from [`Umap::new`] (D-08 single source).
    /// `f64` is pinned only to read the F-independent scalar defaults — the
    /// builder is non-generic, so the choice of `F` here is irrelevant (Pitfall
    /// 4: do NOT hand-write literal defaults here).
    fn default() -> Self {
        Umap::<f64, Unfit>::new().into_builder()
    }
}

impl UmapBuilder {
    /// Set the local neighborhood size `n_neighbors`.
    pub fn n_neighbors(mut self, v: usize) -> Self {
        self.n_neighbors = v;
        self
    }
    /// Set the embedding dimensionality `n_components`.
    pub fn n_components(mut self, v: usize) -> Self {
        self.n_components = v;
        self
    }
    /// Set the minimum inter-point distance `min_dist`.
    pub fn min_dist(mut self, v: f64) -> Self {
        self.min_dist = v;
        self
    }
    /// Set the embedded-point scale `spread`.
    pub fn spread(mut self, v: f64) -> Self {
        self.spread = v;
        self
    }
    /// Set the neighbor-graph distance `metric`.
    pub fn metric(mut self, v: Metric) -> Self {
        self.metric = v;
        self
    }
    /// Set the optimization epoch count `n_epochs`.
    pub fn n_epochs(mut self, v: Option<usize>) -> Self {
        self.n_epochs = v;
        self
    }
    /// Set the embedding initialization strategy `init`.
    pub fn init(mut self, v: Init) -> Self {
        self.init = v;
        self
    }
    /// Set the RNG seed `random_state`.
    pub fn random_state(mut self, v: Option<u64>) -> Self {
        self.random_state = v;
        self
    }
    /// Set the initial optimization `learning_rate`.
    pub fn learning_rate(mut self, v: f64) -> Self {
        self.learning_rate = v;
        self
    }
    /// Set the fuzzy-set mix `set_op_mix_ratio`.
    pub fn set_op_mix_ratio(mut self, v: f64) -> Self {
        self.set_op_mix_ratio = v;
        self
    }
    /// Set the assumed `local_connectivity`.
    pub fn local_connectivity(mut self, v: f64) -> Self {
        self.local_connectivity = v;
        self
    }
    /// Set the layout `repulsion_strength`.
    pub fn repulsion_strength(mut self, v: f64) -> Self {
        self.repulsion_strength = v;
        self
    }
    /// Set the `negative_sample_rate`.
    pub fn negative_sample_rate(mut self, v: usize) -> Self {
        self.negative_sample_rate = v;
        self
    }
    /// Set the `a` curve-fit override.
    pub fn a(mut self, v: Option<f64>) -> Self {
        self.a = v;
        self
    }
    /// Set the `b` curve-fit override.
    pub fn b(mut self, v: Option<f64>) -> Self {
        self.b = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08; the data-DEPENDENT
    /// geometry check lives in [`Fit::fit`]):
    ///
    /// - `min_dist` must be finite and `<= spread`
    ///   ([`BuildError::InvalidMinDist`]).
    /// - `n_components >= 1` and `n_neighbors >= 1`
    ///   ([`BuildError::InvalidNComponents`]); umap-learn rejects 0 for both, and
    ///   `n_components = 0` would otherwise yield a silently-empty embedding.
    pub fn build<F>(self) -> Result<Umap<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if !self.min_dist.is_finite() || self.min_dist > self.spread {
            return Err(BuildError::InvalidMinDist {
                estimator: "umap",
                min_dist: self.min_dist,
            });
        }
        if self.n_components == 0 {
            return Err(BuildError::InvalidNComponents {
                estimator: "umap",
                param: "n_components",
                value: self.n_components,
            });
        }
        if self.n_neighbors == 0 {
            return Err(BuildError::InvalidNComponents {
                estimator: "umap",
                param: "n_neighbors",
                value: self.n_neighbors,
            });
        }
        Ok(Umap {
            n_neighbors: self.n_neighbors,
            n_components: self.n_components,
            min_dist: self.min_dist,
            spread: self.spread,
            metric: self.metric,
            n_epochs: self.n_epochs,
            init: self.init,
            random_state: self.random_state,
            learning_rate: self.learning_rate,
            set_op_mix_ratio: self.set_op_mix_ratio,
            local_connectivity: self.local_connectivity,
            repulsion_strength: self.repulsion_strength,
            negative_sample_rate: self.negative_sample_rate,
            a: self.a,
            b: self.b,
            embedding_: None,
            x_train_: None,
            n_features_in_: 0,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for Umap<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = Umap<F, Fitted>;

    /// Real UMAP fit (UMAP-01/03, Phase 14): runs the full deterministic pipeline
    /// KNN graph → smooth-kNN ρ/σ → membership → t-conorm union → a/b LM fit →
    /// spectral/random init → the stochastic SGD layout (host epoch driver +
    /// `umap_layout_step` kernel), producing the real `(n, n_components)`
    /// `embedding_`. CONSUMES `self`, returning the `Fitted`-tagged sibling
    /// (D-02). `y` is ignored (UMAP is unsupervised).
    ///
    /// All randomness is HOST-drawn `SplitMix64` keyed as a pure function of
    /// `(random_state, epoch, edge)` (D-05) so two same-`random_state` fits are
    /// byte-identical per (backend, dtype). Geometry is validated BEFORE any
    /// device launch (ASVS V5).
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Umap<F, Fitted>, AlgoError> {
        let (n, p) = shape;

        // Data-DEPENDENT geometry guard BEFORE any launch (shared helper, the
        // data-INDEPENDENT params were validated at build()).
        validate_geometry(x, shape)?;

        // Data-DEPENDENT `n_components < n` guard mirroring the sibling
        // `SpectralEmbedding::fit` (spectral_embedding.rs:155). With the default
        // `Init::Spectral`, `run_umap_layout → spectral_init → spectral::recover`
        // needs the smallest `n_components + 1` eigenvectors (drop_first) and
        // computes `col = n - 1 - r` with `r` up to `n_components`; when
        // `n_components + 1 > n` (i.e. `n_components >= n`) that underflows
        // `usize` → panic (debug) / OOB device read (release). Reject the bad
        // input BEFORE any device launch as a typed error (CR-02 / ASVS V5).
        // The lower bound (`n_components >= 1`) is already enforced at build().
        if self.n_components >= n {
            return Err(AlgoError::InvalidNComponents {
                estimator: "umap",
                requested: self.n_components,
                max: n.saturating_sub(1),
            });
        }

        let embedding_host = run_umap_layout::<F>(pool, x, n, p, &self)?;
        let embedding = DeviceArray::from_host(pool, &embedding_host);

        // Retain the training design rows for `transform`'s query-vs-train KNN
        // (D-03 — new→train runs in the original feature space). A host round-trip
        // re-uploads `x` into an owned device buffer the fitted estimator keeps.
        let x_train_host: Vec<F> = x.to_host(pool);
        let x_train = DeviceArray::from_host(pool, &x_train_host);

        Ok(Umap {
            n_neighbors: self.n_neighbors,
            n_components: self.n_components,
            min_dist: self.min_dist,
            spread: self.spread,
            metric: self.metric,
            n_epochs: self.n_epochs,
            init: self.init,
            random_state: self.random_state,
            learning_rate: self.learning_rate,
            set_op_mix_ratio: self.set_op_mix_ratio,
            local_connectivity: self.local_connectivity,
            repulsion_strength: self.repulsion_strength,
            negative_sample_rate: self.negative_sample_rate,
            a: self.a,
            b: self.b,
            embedding_: Some(embedding),
            x_train_: Some(x_train),
            n_features_in_: p,
            _state: PhantomData,
        })
    }
}

impl<F> Umap<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `embedding_` (length `n * n_components`,
    /// row-major). `Some` by construction on the `Fitted` state, so no
    /// `NotFitted` branch is needed (the compile-time typestate replaces the
    /// runtime guard, D-02).
    pub fn embedding(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.embedding_
            .as_ref()
            .expect("embedding_ is Some by construction on Umap<F, Fitted>")
            .to_host(pool)
    }

    /// Number of features seen at fit (`n_features_in_`).
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }
}

impl<F> Transform<F> for Umap<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Real UMAP `transform(X_new)` (UMAP-04, D-03) — embed `m` NEW points against
    /// the fitted fuzzy graph via the umap-learn frozen-subset path: query-vs-train
    /// KNN(new→train) → membership of the new points → `init_graph_transform`
    /// neighbor-weighted-average init → reduced-epoch SGD optimizing ONLY the new
    /// points with the training embedding FROZEN (read-only GATHER targets),
    /// driving the SAME [`umap_layout_step`] kernel as `fit` with
    /// `owners = m new points`, `move_other = 0`. Exists ONLY on `Umap<F, Fitted>`,
    /// so `transform`-before-`fit` is a compile error (D-02).
    ///
    /// All randomness is HOST-drawn `SplitMix64` keyed as a pure function of
    /// `(random_state, epoch, edge)` (D-05) so two same-`random_state` transforms
    /// are byte-identical per (backend, dtype). The fit-time `n_features_in_`
    /// defines the contract: a `X_new` whose column count differs returns the typed
    /// [`PrimError::ShapeMismatch`] BEFORE any launch (T-14-15), never a wrong-shape
    /// device read.
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (m, p) = shape;
        validate_geometry(x, shape)?;
        // The fitted `n_features_in_` defines the transform contract (typestate
        // doc: "Errors if the geometry disagrees with the fitted `n_features`"):
        // a matrix whose column count differs from the fit-time feature count
        // cannot be projected onto the fitted components (WR-02 / T-14-15).
        if p != self.n_features_in_ {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: m,
                cols: p,
                len: x.len(),
            }));
        }

        let new_host = transform_new_points::<F>(pool, self, x, m, p)?;
        Ok(DeviceArray::from_host(pool, &new_host))
    }
}

/// The frozen-subset transform driver (UMAP-04, D-03). Returns the row-major
/// `(m, n_components)` host buffer of the `m` new points' embedding coordinates.
/// Pure host orchestration over the Task-1 query-vs-train KNN, the Plan-02
/// membership stage, `init_graph_transform`, and the Plan-04 `umap_layout_step`
/// kernel (driven owner-only / `move_other = 0` so the training coords stay
/// frozen — proven by the `umap_layout_step_launches_f64_owner_only` smoke test).
fn transform_new_points<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    cfg: &Umap<F, Fitted>,
    x_new: &DeviceArray<ActiveRuntime, F>,
    m: usize,
    d: usize,
) -> Result<Vec<F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let n_components = cfg.n_components;

    // Defensive boundary check (WR-02): `fit` guards `n_components >= 1` and the
    // embedding-buffer shape, so this is currently unreachable, but the transform
    // path must not divide by `n_components` (below) or index
    // `embedding_train[col * n_components + d]` (in `init_graph_transform`) on a
    // partially-built `Fitted` shell. Surface a typed error rather than panicking
    // deep inside transform.
    if n_components == 0 {
        return Err(AlgoError::InvalidGraphInput {
            estimator: "umap",
            reason: "n_components is 0 on the fitted estimator".to_string(),
        });
    }

    // The FROZEN training embedding (host f64, row-major (n, n_components)).
    let embedding_train: Vec<f64> = cfg
        .embedding_
        .as_ref()
        .expect("embedding_ is Some by construction on Umap<F, Fitted>")
        .to_host(pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    if embedding_train.len() % n_components != 0 {
        return Err(AlgoError::InvalidGraphInput {
            estimator: "umap",
            reason: format!(
                "fitted embedding length {} is not a multiple of n_components {}",
                embedding_train.len(),
                n_components
            ),
        });
    }
    let n = embedding_train.len() / n_components;

    // Host f64 copies of the query (new) and the training design rows. The
    // query-vs-train KNN runs new-vs-train in the ORIGINAL feature space, which
    // requires the training X, so the estimator retains the training design rows
    // on the fitted shell (`x_train_`, read below) alongside the frozen embedding.
    let x_new_host: Vec<f64> = x_new.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let x_train_host: Vec<f64> = cfg
        .x_train_
        .as_ref()
        .expect("x_train_ is Some by construction on Umap<F, Fitted>")
        .to_host(pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();

    // --- (1) Query-vs-train KNN (new→train), no self-drop (new ≠ train). ---
    let k = cfg.n_neighbors.min(n.saturating_sub(1)).max(1);
    let (knn_idx_host, knn_dist_host) =
        query_train_knn::<F>(pool, &x_new_host, &x_train_host, m, n, d, k, cfg.metric)?;

    // --- (2) membership of the NEW points against the fitted graph: smooth-kNN
    //     ρ/σ on the new points' OWN knn distances + the directed membership exp
    //     (same constants as fit). NO t-conorm union — the transform graph is the
    //     directed bipartite (m new rows × n train cols), not symmetrized. ---
    let (sigmas, rhos) = umap_internals::smooth_knn_dist(
        &knn_dist_host,
        m,
        k,
        cfg.n_neighbors,
        cfg.local_connectivity,
    );
    let (t_rows, t_cols, t_vals) = umap_internals::compute_membership_strengths(
        &knn_idx_host,
        &knn_dist_host,
        &rhos,
        &sigmas,
        m,
        k,
    );

    // --- (3) init_graph_transform: neighbor-weighted-average init for new pts. ---
    let init_new = umap_internals::init_graph_transform(
        &t_rows,
        &t_cols,
        &t_vals,
        &embedding_train,
        m,
        n,
        n_components,
    );

    // --- (4) a/b curve parameters (same as fit). ---
    let (a, b) = match (cfg.a, cfg.b) {
        (Some(a), Some(b)) => (a, b),
        _ => umap_init::fit_ab(cfg.min_dist, cfg.spread)?,
    };

    // --- (5) build the combined embedding [n frozen train rows][m new rows] and
    //     the per-new-point CSR edges into the frozen train block. Owners are the
    //     m NEW rows placed CONTIGUOUSLY AFTER the n training rows (D-03). ---
    let mut combined: Vec<f64> = Vec::with_capacity((n + m) * n_components);
    combined.extend_from_slice(&embedding_train);
    combined.extend_from_slice(&init_new);

    // Per-new-point positive edges (target = train col index, GLOBAL row index is
    // unchanged since train rows occupy 0..n). Weights are the transform
    // membership values; the layout schedule samples them as in fit.
    // head/tail are GLOBAL vertex indices in the combined buffer: owner = n + new_i.
    let mut head: Vec<usize> = Vec::with_capacity(t_vals.len());
    let mut tail: Vec<usize> = Vec::with_capacity(t_vals.len());
    let mut weights: Vec<f64> = Vec::with_capacity(t_vals.len());
    for e in 0..t_vals.len() {
        if t_vals[e] > 0.0 {
            head.push(n + t_rows[e]); // owner = the new point (after the train block)
            tail.push(t_cols[e]); // target = the frozen training vertex
            weights.push(t_vals[e]);
        }
    }

    // --- (6) reduced-epoch SGD: n_epochs = 100 when fit-time was None (RESEARCH
    //     Pattern 7 step 4), else the fit-time count. Drive the SAME kernel with
    //     owners = m (the new rows) starting at offset n, move_other = 0. ---
    let n_epochs = cfg.n_epochs.unwrap_or(100);
    let eps = make_epochs_per_sample(&weights, n_epochs)?;
    let seed = cfg.random_state.unwrap_or(42);

    transform_epoch_driver::<F>(
        pool,
        &mut combined,
        &head,
        &tail,
        &eps,
        n,
        m,
        n_components,
        a,
        b,
        cfg.repulsion_strength,
        cfg.learning_rate,
        cfg.negative_sample_rate,
        n_epochs,
        seed,
    );

    // Extract the m new-point rows (the tail of the combined buffer).
    let new_coords: Vec<F> = combined[n * n_components..(n + m) * n_components]
        .iter()
        .map(|&v| f64_to_host::<F>(v))
        .collect();
    Ok(new_coords)
}

/// Frozen-subset host epoch driver for `transform` (UMAP-04, D-03). Drives the
/// SAME [`umap_layout_step`] kernel as the fit-path [`host_epoch_driver`] but with
/// `move_other = 0` (the training coords are read-only GATHER targets) and only
/// the `m` NEW vertices as owners. The owners occupy the contiguous rows
/// `n..n+m` of `combined`, but `umap_layout_step` treats rows `0..n_owners` as the
/// owners — so the CSR is keyed by the OWNER-LOCAL index `0..m` while the edge
/// targets are GLOBAL vertex indices `< n + m`. To reuse the kernel's
/// `embedding[owner_local]` write semantics with owners placed at the END, the
/// driver passes a VIEW whose owner rows are the new points: it launches over the
/// full `(n + m)` buffer with `n_owners = n + m` would move the train rows too —
/// instead it builds the CSR so ONLY the `m` new owners have edges and launches
/// with `n_owners = n + m`, `move_other = 0`; train owners (rows `0..n`) get empty
/// CSR ranges and are never written. Negative samples are drawn over the whole
/// combined vertex set `0..n_vertices` host-side per `(seed, epoch, edge)` (D-05)
/// so the transform is byte-identical (matching umap-learn's `optimize_layout`,
/// which samples negatives over the full `head_embedding` vertex count).
#[allow(clippy::too_many_arguments)]
fn transform_epoch_driver<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    combined: &mut [f64],
    head: &[usize],
    tail: &[usize],
    epochs_per_sample: &[f64],
    n_train: usize,
    m_new: usize,
    dim: usize,
    a: f64,
    b: f64,
    gamma: f64,
    initial_alpha: f64,
    negative_sample_rate: usize,
    n_epochs: usize,
    seed: u64,
) where
    F: Float + CubeElement + Pod,
{
    let n_vertices = n_train + m_new;
    let n_edges = epochs_per_sample.len();
    let mut next_sample: Vec<f64> = epochs_per_sample.to_vec();
    let epochs_per_negative: Vec<f64> = epochs_per_sample
        .iter()
        .map(|&e| if e > 0.0 { e / negative_sample_rate as f64 } else { -1.0 })
        .collect();
    let mut next_negative: Vec<f64> = epochs_per_negative.clone();

    let client = pool.client().clone();

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f64 / n_epochs as f64);

        // Per-owner CSR over ALL n_vertices owners; only the m new owners (rows
        // n_train..n_vertices) ever receive edges, so the train owners stay frozen
        // (empty ranges) AND move_other = 0 prevents any train coordinate write.
        let mut pos_per_owner: Vec<Vec<u32>> = vec![Vec::new(); n_vertices];
        let mut neg_per_owner: Vec<Vec<u32>> = vec![Vec::new(); n_vertices];

        for e in 0..n_edges {
            let eps_e = epochs_per_sample[e];
            if eps_e <= 0.0 {
                continue;
            }
            if next_sample[e] > epoch as f64 {
                continue;
            }
            let owner = head[e]; // GLOBAL owner index (a new vertex, ≥ n_train)
            pos_per_owner[owner].push(tail[e] as u32);

            let epn = epochs_per_negative[e];
            let n_neg = if epn > 0.0 {
                ((epoch as f64 - next_negative[e]) / epn).floor() as i64
            } else {
                0
            };
            if n_neg > 0 {
                let sub_seed = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((epoch as u64).wrapping_mul(0x1000_0001))
                    .wrapping_add(e as u64);
                let mut rng = SplitMix64::new(sub_seed);
                for _ in 0..n_neg {
                    // Negatives are drawn over the WHOLE combined vertex set
                    // `0..n_vertices` (train ∪ new) — the new point is repelled from
                    // random vertices of the combined embedding, matching umap-learn's
                    // `optimize_layout` which samples `tail` over `n_vertices =
                    // head_embedding.shape[0]`. (Restricting to train-only measurably
                    // REGRESSED the structural gate for euclidean+cosine, so the
                    // combined-set draw is the correct, calibrated behaviour.)
                    let kk = rng.next_below(n_vertices as u64) as u32;
                    neg_per_owner[owner].push(kk);
                }
                next_negative[e] += n_neg as f64 * epn;
            }
            next_sample[e] += eps_e;
        }

        // Flatten per-owner buckets into CSR offsets + tail/neg arrays over the
        // full n_vertices owner set (train owners have empty ranges).
        let mut pos_offsets: Vec<u32> = Vec::with_capacity(n_vertices + 1);
        let mut pos_tail: Vec<u32> = Vec::new();
        let mut neg_offsets: Vec<u32> = Vec::with_capacity(n_vertices + 1);
        let mut neg_idx: Vec<u32> = Vec::new();
        pos_offsets.push(0);
        neg_offsets.push(0);
        for o in 0..n_vertices {
            pos_tail.extend_from_slice(&pos_per_owner[o]);
            pos_offsets.push(pos_tail.len() as u32);
            neg_idx.extend_from_slice(&neg_per_owner[o]);
            neg_offsets.push(neg_idx.len() as u32);
        }

        if pos_tail.is_empty() && neg_idx.is_empty() {
            continue;
        }
        if pos_tail.is_empty() {
            pos_tail.push(0);
        }
        if neg_idx.is_empty() {
            neg_idx.push(0);
        }

        let emb_f: Vec<F> = combined.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let emb_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &emb_f);
        let pos_off_dev = DeviceArray::<ActiveRuntime, u32>::from_host(pool, &pos_offsets);
        let pos_tail_dev = DeviceArray::<ActiveRuntime, u32>::from_host(pool, &pos_tail);
        let neg_off_dev = DeviceArray::<ActiveRuntime, u32>::from_host(pool, &neg_offsets);
        let neg_idx_dev = DeviceArray::<ActiveRuntime, u32>::from_host(pool, &neg_idx);

        let count = CubeCount::Static(n_vertices as u32, 1, 1);
        let cube_dim = CubeDim { x: 1, y: 1, z: 1 };
        let emb_arg =
            unsafe { ArrayArg::from_raw_parts(emb_dev.handle().clone(), n_vertices * dim) };
        let pos_off_arg =
            unsafe { ArrayArg::from_raw_parts(pos_off_dev.handle().clone(), pos_offsets.len()) };
        let pos_tail_arg =
            unsafe { ArrayArg::from_raw_parts(pos_tail_dev.handle().clone(), pos_tail.len()) };
        let neg_off_arg =
            unsafe { ArrayArg::from_raw_parts(neg_off_dev.handle().clone(), neg_offsets.len()) };
        let neg_idx_arg =
            unsafe { ArrayArg::from_raw_parts(neg_idx_dev.handle().clone(), neg_idx.len()) };

        umap_layout_step::launch::<F, ActiveRuntime>(
            &client,
            count,
            cube_dim,
            emb_arg,
            pos_off_arg,
            pos_tail_arg,
            neg_off_arg,
            neg_idx_arg,
            f64_to_host::<F>(a),
            f64_to_host::<F>(b),
            f64_to_host::<F>(gamma),
            f64_to_host::<F>(alpha),
            dim as u32,
            n_vertices as u32, // n_owners = all vertices; train owners have empty CSR
            n_vertices as u32, // n_vertices bound for the GATHER index check
            0u32,              // move_other = 0 (frozen-subset transform path, D-03)
        );

        let updated: Vec<f64> =
            emb_dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        combined.copy_from_slice(&updated);

        emb_dev.release_into(pool);
        pos_off_dev.release_into(pool);
        pos_tail_dev.release_into(pool);
        neg_off_dev.release_into(pool);
        neg_idx_dev.release_into(pool);
    }
}

// ===========================================================================
// Real UMAP fit pipeline (Phase 14, Plan 04) — host orchestration over the
// validated deterministic stages + the stochastic SGD layout.
// ===========================================================================

/// Map the estimator's [`Metric`] onto the Phase-13 [`knn_graph::Metric`]
/// (Pitfall 4 — the enums mirror each other EXACTLY, so the mapping is total and
/// lossless). Returns `(metric, p)` where `p` is the Minkowski exponent passed to
/// `knn_graph` (ignored by the non-Minkowski variants).
fn map_metric(metric: Metric) -> (knn_graph::Metric, f64) {
    match metric {
        Metric::Euclidean => (knn_graph::Metric::Euclidean, 2.0),
        Metric::Manhattan => (knn_graph::Metric::Manhattan, 1.0),
        Metric::Cosine => (knn_graph::Metric::Cosine, 2.0),
        Metric::Chebyshev => (knn_graph::Metric::Chebyshev, 2.0),
        Metric::Minkowski { p } => (knn_graph::Metric::Minkowski { p }, p),
    }
}

/// L2-normalise each row of a row-major `r × d` host matrix
/// (`x̂_i = x_i / ‖x_i‖₂`, zero-norm rows stay zero) — the Cosine pre-step before
/// the GEMM `distance` path (mirrors `knn_graph::l2_normalize_rows`, kept private
/// here so the transform query-vs-train composition is self-contained).
fn l2_normalize_rows(x: &[f64], r: usize, d: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(r * d);
    for i in 0..r {
        let row = &x[i * d..(i + 1) * d];
        let norm = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for &v in row {
            out.push(v * inv);
        }
    }
    out
}

/// Query-vs-train directed KNN for the transform path (UMAP-04, RESEARCH Q2/A2 —
/// the resolution of Pitfall 5). Composes `distance(X_new, X_train)` + `top_k(k)`
/// directly (Euclidean/Cosine → the GEMM `distance` fast path; Manhattan/
/// Chebyshev/Minkowski → the direct pairwise kernels), with NO self-drop (the new
/// points are NOT in the training set, so a query is never its own neighbour).
///
/// `x_new_host` is the row-major `(m, d)` query block (host f64); `x_train_host`
/// is the row-major `(n, d)` training block. Returns `(knn_idx, knn_dist)` as
/// row-major `(m, k)` host `f64` (indices float-encoded for the membership
/// stages, ascending per row, lowest-index tie-break — the mlrs convention).
///
/// All geometry is validated inside the composed `distance`/`top_k` prims BEFORE
/// any launch (T-14-15/T-14-16). The `(m × n)` distance block is small at
/// transform scale, so no query-axis tiling is needed (single block).
fn query_train_knn<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_new_host: &[f64],
    x_train_host: &[f64],
    m: usize,
    n: usize,
    d: usize,
    k: usize,
    metric: Metric,
) -> Result<(Vec<f64>, Vec<f64>), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let (knn_metric, p) = map_metric(metric);

    // Euclidean uses GEMM squared distance with a top_k boundary sqrt; Cosine
    // uses GEMM on L2-normalised rows (its returned squared value `2(1−cos)` is
    // order-preserving with cosine distance, so the SELECTED indices are correct).
    // The direct kernels emit true distance.
    let needs_sqrt = matches!(knn_metric, knn_graph::Metric::Euclidean);
    // Cosine post-scale: `1 − cos = ‖x̂ − ŷ‖² / 2`. The fit path's `knn_graph`
    // halves the GEMM `2(1−cos)` back to the true cosine distance `1−cos`
    // (knn_graph.rs:212-219). The transform KNN distances feed the SAME
    // membership stage (`smooth_knn_dist`), which is NOT purely scale-invariant
    // (the σ floor `MIN_K_DIST_SCALE * mean` and the ρ≤0 global fallback are
    // absolute floors against the mean), so a 2×-inflated transform distance
    // matrix would build a membership graph that does not match the one `fit`
    // builds for the same neighbour geometry (UMAP-04 / D-03 require identical
    // membership). Mirror the halving here so transform is on the `1−cos` scale.
    let cosine_halve = matches!(knn_metric, knn_graph::Metric::Cosine);

    // Build the (normalised, for Cosine) device operands.
    let (q_host, t_host): (Vec<f64>, Vec<f64>) = match metric {
        Metric::Cosine => (
            l2_normalize_rows(x_new_host, m, d),
            l2_normalize_rows(x_train_host, n, d),
        ),
        _ => (x_new_host.to_vec(), x_train_host.to_vec()),
    };
    let q_f: Vec<F> = q_host.iter().map(|&v| f64_to_host::<F>(v)).collect();
    let t_f: Vec<F> = t_host.iter().map(|&v| f64_to_host::<F>(v)).collect();
    let q_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &q_f);
    let t_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &t_f);

    // --- distance(X_new, X_train) → (m × n). ---
    let dist: DeviceArray<ActiveRuntime, F> = match knn_metric {
        knn_graph::Metric::Euclidean | knn_graph::Metric::Cosine => {
            distance::<F>(pool, &q_dev, (m, d), &t_dev, (n, d), false, None)?
        }
        knn_graph::Metric::Manhattan
        | knn_graph::Metric::Chebyshev
        | knn_graph::Metric::Minkowski { .. } => {
            let out_len = m * n;
            let out_handle = pool.acquire(out_len * std::mem::size_of::<F>());
            let client = pool.client().clone();
            let bx = 16u32;
            let count = CubeCount::Static(
                ((m as u32) + bx - 1) / bx,
                ((n as u32) + bx - 1) / bx,
                1,
            );
            let cube_dim = CubeDim { x: bx, y: bx, z: 1 };
            // SAFETY: lengths are validated element counts (m*d, n*d, m*n); the
            // direct kernels bounds-check i<m && j<n and the feature loop kk<d.
            let q_arg = unsafe { ArrayArg::from_raw_parts(q_dev.handle().clone(), q_dev.len()) };
            let t_arg = unsafe { ArrayArg::from_raw_parts(t_dev.handle().clone(), t_dev.len()) };
            let o_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };
            match knn_metric {
                knn_graph::Metric::Manhattan => manhattan_dist::launch::<F, ActiveRuntime>(
                    &client, count, cube_dim, q_arg, t_arg, o_arg, m as u32, n as u32, d as u32,
                ),
                knn_graph::Metric::Chebyshev => chebyshev_dist::launch::<F, ActiveRuntime>(
                    &client, count, cube_dim, q_arg, t_arg, o_arg, m as u32, n as u32, d as u32,
                ),
                knn_graph::Metric::Minkowski { p: mp } => minkowski_dist::launch::<F, ActiveRuntime>(
                    &client, count, cube_dim, q_arg, t_arg, o_arg, m as u32, n as u32, d as u32,
                    f64_to_host::<F>(mp),
                ),
                _ => unreachable!("outer match restricts to the direct-kernel metrics"),
            }
            DeviceArray::from_raw(out_handle, out_len)
        }
    };
    let _ = p; // p flows through the Minkowski enum payload (single source of truth)
    q_dev.release_into(pool);
    t_dev.release_into(pool);

    // --- top_k(k) over the m query rows; ascending (val, idx), lowest-index
    //     tie-break. NO self-drop (new ≠ train). sqrt boundary for Euclidean. ---
    let (tk_val, tk_idx) = top_k::<F>(pool, &dist, m, n, k, needs_sqrt, None, None)?;
    dist.release_into(pool);

    let knn_idx_host: Vec<f64> = tk_idx.to_host(pool).iter().map(|&v| v as f64).collect();
    let knn_dist_host: Vec<f64> = tk_val
        .to_host(pool)
        .iter()
        .map(|&v| {
            let d = host_to_f64(v);
            // Mirror knn_graph's cosine halving so transform distances are on the
            // same `1−cos` scale as fit (CR-01).
            if cosine_halve {
                0.5 * d
            } else {
                d
            }
        })
        .collect();
    tk_idx.release_into(pool);
    tk_val.release_into(pool);

    Ok((knn_idx_host, knn_dist_host))
}

/// umap-learn `make_epochs_per_sample` (verified): for each edge weight, the
/// number of epochs between successive positive samples of that edge is
/// `n_epochs / (n_epochs · w/w_max) = w_max / w`. Edges whose scaled sample
/// count is ≤ 0 are never sampled (sentinel `-1.0`). Mirrors umap's
/// `result[n_samples > 0] = n_epochs / n_samples[n_samples > 0]`.
///
/// WR-05: validates every weight is finite BEFORE the `w_max` reduction.
/// `f64::max(0.0, NaN)` returns `0.0`, so a NaN weight would otherwise be
/// silently folded to 0 and corrupt `w_max`, yielding a wrong per-edge sampling
/// schedule for every edge with no error. Membership values come from `exp` of
/// finite inputs so a NaN should not arise today, but the failure mode (silent
/// schedule corruption) is invisible, so it is rejected as a typed error.
fn make_epochs_per_sample(weights: &[f64], n_epochs: usize) -> Result<Vec<f64>, AlgoError> {
    if let Some(bad) = weights.iter().position(|w| !w.is_finite()) {
        return Err(AlgoError::InvalidGraphInput {
            estimator: "umap",
            reason: format!(
                "fuzzy-graph edge weight {} is non-finite at index {}",
                weights[bad], bad
            ),
        });
    }
    let w_max = weights.iter().cloned().fold(0.0_f64, f64::max);
    Ok(weights
        .iter()
        .map(|&w| {
            // n_samples = n_epochs * (w / w_max); epochs_per_sample = n_epochs / n_samples
            let n_samples = if w_max > 0.0 {
                n_epochs as f64 * (w / w_max)
            } else {
                0.0
            };
            if n_samples > 0.0 {
                n_epochs as f64 / n_samples
            } else {
                -1.0
            }
        })
        .collect())
}

/// Run the full UMAP pipeline and return the row-major `(n, n_components)`
/// embedding host buffer. Pure host orchestration over the Phase-13 KNN prim,
/// the Plan-02/03 host stages, and the Plan-04 `umap_layout_step` kernel.
fn run_umap_layout<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    cfg: &Umap<F, Unfit>,
) -> Result<Vec<F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let n_components = cfg.n_components;

    // --- (1) Directed KNN graph (UMAP path: include_self = false). ---
    let (knn_metric, p) = map_metric(cfg.metric);
    // umap clamps n_neighbors to n-1 (can't have more neighbours than points-1).
    let k = cfg.n_neighbors.min(n.saturating_sub(1)).max(1);
    let (knn_idx_dev, knn_dist_dev) =
        knn_graph::<F>(pool, x, (n, d), k, knn_metric, false, p)?;
    let knn_idx_host: Vec<f64> = knn_idx_dev
        .to_host(pool)
        .iter()
        .map(|&v| v as f64)
        .collect();
    let knn_dist_host: Vec<f64> = knn_dist_dev
        .to_host(pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    knn_idx_dev.release_into(pool);
    knn_dist_dev.release_into(pool);

    // --- (2) smooth-kNN ρ/σ → membership → t-conorm union (host f64). ---
    let (sigmas, rhos) = umap_internals::smooth_knn_dist(
        &knn_dist_host,
        n,
        k,
        cfg.n_neighbors,
        cfg.local_connectivity,
    );
    let (m_rows, m_cols, m_vals) = umap_internals::compute_membership_strengths(
        &knn_idx_host,
        &knn_dist_host,
        &rhos,
        &sigmas,
        n,
        k,
    );
    let (g_rows, g_cols, g_vals) =
        umap_internals::fuzzy_union(&m_rows, &m_cols, &m_vals, n, cfg.set_op_mix_ratio);

    // --- (3) a/b curve parameters (LM fit unless overridden). ---
    let (a, b) = match (cfg.a, cfg.b) {
        (Some(a), Some(b)) => (a, b),
        _ => umap_init::fit_ab(cfg.min_dist, cfg.spread)?,
    };

    // --- (4) init: spectral (falls back to random above n=64) or random. ---
    let seed = cfg.random_state.unwrap_or(42);
    let mut init: Vec<f64> = match cfg.init {
        Init::Spectral => {
            // Build the dense n×n symmetric affinity from the fuzzy COO and drive
            // the shared spectral_init (reuses laplacian → eig → recover). It
            // internally falls back to random_init for n > MAX_DIM (umap's own
            // behaviour) — no error.
            let mut affinity = vec![0.0f64; n * n];
            for e in 0..g_vals.len() {
                // Bounds check at the COO-consumption boundary (WR-01): the
                // Phase-13 prim guarantees `cols < n`, but that is an unchecked
                // cross-module invariant carried across a host round-trip. A
                // regression (or a NaN float-encoded index) would otherwise be a
                // silent OOB write; surface it as a typed error instead.
                if g_rows[e] >= n || g_cols[e] >= n {
                    return Err(AlgoError::InvalidGraphInput {
                        estimator: "umap",
                        reason: format!(
                            "fuzzy-graph edge ({}, {}) out of range for n_samples {}",
                            g_rows[e], g_cols[e], n
                        ),
                    });
                }
                affinity[g_rows[e] * n + g_cols[e]] = g_vals[e];
            }
            let aff_f: Vec<F> = affinity.iter().map(|&v| f64_to_host::<F>(v)).collect();
            let aff_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &aff_f);
            let coords = umap_init::spectral_init::<F>(pool, &aff_dev, n, n_components, seed)?;
            aff_dev.release_into(pool);
            coords.iter().map(|&v| host_to_f64(v)).collect()
        }
        Init::Random => {
            let coords = umap_init::random_init::<F>(n, n_components, seed);
            coords.iter().map(|&v| host_to_f64(v)).collect()
        }
    };
    // umap's separate post-init stage: scale to max-coord 10 + tiny noise.
    // Keyed off `seed` so the init is byte-deterministic (D-05).
    umap_init::noisy_scale_coords(&mut init, n, n_components, 10.0, 1e-4, seed ^ 0x5350_4543);

    // --- (5) epochs_per_sample + n_epochs resolution (A4: 500 small / 200 large). ---
    let n_epochs = cfg.n_epochs.unwrap_or(if n <= 10_000 { 500 } else { 200 });
    let eps = make_epochs_per_sample(&g_vals, n_epochs)?;

    // --- (6) host epoch driver: per epoch, select active edges by umap's
    //     epoch_of_next_sample schedule, draw host negative samples, launch
    //     umap_layout_step over owners = all n, OWNER-ONLY (move_other = 0,
    //     via FIT_MOVE_OTHER): the symmetric COO covers both endpoints of each
    //     undirected pair (once per direction) with no foreign-vertex write. ---
    host_epoch_driver::<F>(
        pool,
        &mut init,
        &g_rows,
        &g_cols,
        &eps,
        n,
        n_components,
        a,
        b,
        cfg.repulsion_strength,
        cfg.learning_rate,
        cfg.negative_sample_rate,
        n_epochs,
        seed,
    );

    Ok(init.iter().map(|&v| f64_to_host::<F>(v)).collect())
}

/// Single source of truth for the fit-path `move_other` flag passed to
/// `umap_layout_step`. **Owner-only (0)** by design (REVIEW CR-01 option b):
/// each owner-cube writes ONLY its own slot-disjoint coordinates
/// (`row*dim..(row+1)*dim`) and never the foreign vertex's slots, so two
/// concurrently-scheduled cubes can never write the same `embedding` slot —
/// the move_other==1 cross-cube WRITE-WRITE race (umap_layout.rs:155-157) is
/// eliminated and the D-05 byte-identical contract holds on ANY parallel
/// backend (wgpu/rocm/cuda), not just the sequential cpu-MLIR gate. Because the
/// fuzzy graph is symmetric (the COO carries both (r,c) and (c,r)), owner-only
/// still covers BOTH endpoints of every undirected pair — once per direction —
/// matching umap-learn's single head/tail force pass and removing the prior
/// CR-03 ~2-4× double-count. A regression to `1` reintroduces both hazards and
/// is caught by the executable `fit_move_other_is_zero` invariant test.
pub(crate) const FIT_MOVE_OTHER: u32 = 0;

/// Test-reachable accessor for [`FIT_MOVE_OTHER`] (the fit-path `move_other`
/// flag). Exposed so the `fit_move_other_is_zero` invariant test can assert the
/// fit path drives the kernel owner-only without reaching into a `::launch`
/// argument; a future regression to `move_other=1` is then a single-constant
/// change the test fails on.
pub fn fit_move_other() -> u32 {
    FIT_MOVE_OTHER
}

/// The UMAP SGD layout host epoch driver (mirrors `prims::sgd::sgd_solve`'s
/// validate→epoch-loop→per-step launch→readback shape). Replicates umap-learn's
/// per-edge sampling schedule:
///   - an edge is positively sampled at epoch `n` iff
///     `epoch_of_next_sample[e] <= n`; after sampling it advances by
///     `epochs_per_sample[e]`.
///   - each sampled edge draws `n_neg = floor((n − epoch_of_next_negative[e]) /
///     epochs_per_negative[e])` host negative samples, then advances the negative
///     clock by `n_neg · epochs_per_negative[e]`.
/// All negative-sample indices are drawn host-side with `SplitMix64::next_below`
/// keyed as a pure function of `(seed, epoch, edge)` (D-05, unbiased — NEVER
/// `% n`), packed into a per-epoch `neg_idx` device buffer the kernel GATHERs.
#[allow(clippy::too_many_arguments)]
fn host_epoch_driver<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    embedding: &mut [f64],
    head: &[usize],
    tail: &[usize],
    epochs_per_sample: &[f64],
    n: usize,
    dim: usize,
    a: f64,
    b: f64,
    gamma: f64,
    initial_alpha: f64,
    negative_sample_rate: usize,
    n_epochs: usize,
    seed: u64,
) where
    F: Float + CubeElement + Pod,
{
    let n_edges = epochs_per_sample.len();
    // Per-edge sample clocks (umap's epoch_of_next_sample / _negative_sample).
    let mut next_sample: Vec<f64> = epochs_per_sample.to_vec();
    let epochs_per_negative: Vec<f64> = epochs_per_sample
        .iter()
        .map(|&e| if e > 0.0 { e / negative_sample_rate as f64 } else { -1.0 })
        .collect();
    let mut next_negative: Vec<f64> = epochs_per_negative.clone();

    let client = pool.client().clone();

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f64 / n_epochs as f64);

        // Per-owner CSR builders for THIS epoch's active positive edges + their
        // host-drawn negative samples.
        let mut pos_per_owner: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut neg_per_owner: Vec<Vec<u32>> = vec![Vec::new(); n];

        for e in 0..n_edges {
            let eps_e = epochs_per_sample[e];
            if eps_e <= 0.0 {
                continue; // never-sampled edge (zero weight)
            }
            if next_sample[e] > epoch as f64 {
                continue; // not due this epoch
            }
            let owner = head[e];
            let other = tail[e];
            pos_per_owner[owner].push(other as u32);

            // How many negative samples this edge draws this epoch.
            let epn = epochs_per_negative[e];
            let n_neg = if epn > 0.0 {
                ((epoch as f64 - next_negative[e]) / epn).floor() as i64
            } else {
                0
            };
            if n_neg > 0 {
                // Deterministic per-(seed, epoch, edge) substream (D-05).
                let sub_seed = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((epoch as u64).wrapping_mul(0x1000_0001))
                    .wrapping_add(e as u64);
                let mut rng = SplitMix64::new(sub_seed);
                for _ in 0..n_neg {
                    let k = rng.next_below(n as u64) as u32;
                    neg_per_owner[owner].push(k);
                }
                next_negative[e] += n_neg as f64 * epn;
            }
            // Advance the positive clock.
            next_sample[e] += eps_e;
        }

        // Flatten the per-owner buckets into CSR offsets + tail/neg arrays.
        let mut pos_offsets: Vec<u32> = Vec::with_capacity(n + 1);
        let mut pos_tail: Vec<u32> = Vec::new();
        let mut neg_offsets: Vec<u32> = Vec::with_capacity(n + 1);
        let mut neg_idx: Vec<u32> = Vec::new();
        pos_offsets.push(0);
        neg_offsets.push(0);
        for o in 0..n {
            pos_tail.extend_from_slice(&pos_per_owner[o]);
            pos_offsets.push(pos_tail.len() as u32);
            neg_idx.extend_from_slice(&neg_per_owner[o]);
            neg_offsets.push(neg_idx.len() as u32);
        }

        // Nothing to do this epoch (no active edges) — skip the launch.
        if pos_tail.is_empty() && neg_idx.is_empty() {
            continue;
        }
        // The kernel indexes neg_idx/pos_tail by CSR ranges; an empty array would
        // give a zero-length ArrayArg. Pad with a single sentinel that the CSR
        // ranges never address (every owner range stays empty, offsets valid).
        if pos_tail.is_empty() {
            pos_tail.push(0);
        }
        if neg_idx.is_empty() {
            neg_idx.push(0);
        }

        // Upload this epoch's coordinates + CSR buffers, launch, read back.
        let emb_f: Vec<F> = embedding.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let emb_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &emb_f);
        let pos_off_dev = DeviceArray::<ActiveRuntime, u32>::from_host(pool, &pos_offsets);
        let pos_tail_dev = DeviceArray::<ActiveRuntime, u32>::from_host(pool, &pos_tail);
        let neg_off_dev = DeviceArray::<ActiveRuntime, u32>::from_host(pool, &neg_offsets);
        let neg_idx_dev = DeviceArray::<ActiveRuntime, u32>::from_host(pool, &neg_idx);

        let count = CubeCount::Static(n as u32, 1, 1);
        let cube_dim = CubeDim { x: 1, y: 1, z: 1 };
        let emb_arg = unsafe { ArrayArg::from_raw_parts(emb_dev.handle().clone(), n * dim) };
        let pos_off_arg =
            unsafe { ArrayArg::from_raw_parts(pos_off_dev.handle().clone(), pos_offsets.len()) };
        let pos_tail_arg =
            unsafe { ArrayArg::from_raw_parts(pos_tail_dev.handle().clone(), pos_tail.len()) };
        let neg_off_arg =
            unsafe { ArrayArg::from_raw_parts(neg_off_dev.handle().clone(), neg_offsets.len()) };
        let neg_idx_arg =
            unsafe { ArrayArg::from_raw_parts(neg_idx_dev.handle().clone(), neg_idx.len()) };

        umap_layout_step::launch::<F, ActiveRuntime>(
            &client,
            count,
            cube_dim,
            emb_arg,
            pos_off_arg,
            pos_tail_arg,
            neg_off_arg,
            neg_idx_arg,
            f64_to_host::<F>(a),
            f64_to_host::<F>(b),
            f64_to_host::<F>(gamma),
            f64_to_host::<F>(alpha),
            dim as u32,
            n as u32,
            n as u32,
            // Fit path is now OWNER-ONLY over the already-symmetric COO. Each
            // undirected pair is covered by BOTH its (r,c) and (c,r) owner-edges
            // with NO foreign-vertex write, so: (a) no owner-cube writes another
            // vertex's slots → the CR-01 cross-cube WRITE-WRITE race is gone
            // (D-05 holds on any parallel backend), and (b) each pair is
            // processed exactly once per direction → the CR-03 ~2-4× double-count
            // is gone (matches umap-learn's single head/tail force pass). The CSR
            // build above is intentionally left over the symmetric COO — that is
            // precisely what makes owner-only cover both endpoints. Routed through
            // FIT_MOVE_OTHER so the fit-path flag has ONE source of truth the
            // `fit_move_other_is_zero` invariant test reads (regression to 1 fails).
            FIT_MOVE_OTHER, // move_other = 0 (owner-only fit path)
        );

        // Read the updated coordinates back into the host buffer.
        let updated: Vec<f64> = emb_dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        embedding.copy_from_slice(&updated);

        emb_dev.release_into(pool);
        pos_off_dev.release_into(pool);
        pos_tail_dev.release_into(pool);
        neg_off_dev.release_into(pool);
        neg_idx_dev.release_into(pool);
    }
}
