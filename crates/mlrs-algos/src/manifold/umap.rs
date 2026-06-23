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
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
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
            n_features_in_: 0,
            _state: PhantomData,
        }
    }

    /// Start building an `Umap` from umap-learn's defaults (D-08 single source).
    pub fn builder() -> UmapBuilder {
        UmapBuilder::default()
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

    /// NON-algorithmic trivial fit (Phase-12 shell — real UMAP lands in Phase
    /// 14). Validates the data-DEPENDENT geometry (D-08), then allocates an
    /// all-zeros `embedding_` of shape `(n, n_components)` — NO kernel, NO
    /// compute. CONSUMES `self`, returning the `Fitted`-tagged sibling (D-02).
    /// `y` is ignored (UMAP is unsupervised).
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Umap<F, Fitted>, AlgoError> {
        let (n, p) = shape;

        // Data-DEPENDENT geometry guard BEFORE the allocation (shared helper, the
        // data-INDEPENDENT params were validated at build()).
        validate_geometry(x, shape)?;

        // Trivial non-algorithmic fit: zeros embedding. NO kernel, NO compute.
        let zeros = vec![F::from_int(0i64); n * self.n_components];
        let embedding = DeviceArray::from_host(pool, &zeros);

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
    /// NON-algorithmic trivial transform (Phase-12 shell — real UMAP transform
    /// lands in Phase 14). Re-emits an all-zeros `(rows, n_components)` buffer
    /// from the passed shape — NO kernel, NO compute. Exists ONLY on
    /// `Umap<F, Fitted>`, so `transform`-before-`fit` is a compile error (D-02).
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n, p) = shape;
        validate_geometry(x, shape)?;
        // The fitted `n_features_in_` defines the transform contract (typestate
        // doc: "Errors if the geometry disagrees with the fitted `n_features`"):
        // a matrix whose column count differs from the fit-time feature count
        // cannot be projected onto the fitted components (WR-02). Surface the
        // same typed `ShapeMismatch` rather than silently emitting a wrong-shape
        // all-zeros buffer.
        if p != self.n_features_in_ {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n,
                cols: p,
                len: x.len(),
            }));
        }
        let zeros = vec![F::from_int(0i64); n * self.n_components];
        Ok(DeviceArray::from_host(pool, &zeros))
    }
}
