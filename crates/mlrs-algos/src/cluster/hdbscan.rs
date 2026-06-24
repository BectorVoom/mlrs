//! `Hdbscan` (HDBS-01) — convention-foundation SHELL for the v3 builder +
//! typestate API (Phase 12).
//!
//! This file demonstrates the Phase-12 estimator convention END-TO-END but
//! contains NO algorithm: the [`Fit::fit`] body is a NON-algorithmic trivial fit
//! that allocates an all-`-1` (noise sentinel) `labels_` vector of length `n` —
//! no kernel, no compute. The real HDBSCAN algorithm (core distances →
//! mutual-reachability → MST → single-linkage → condensed tree → EoM/leaf
//! stability extraction) lands in Phase 15; until then this shell gives Phase 15
//! a born-builder-fronted, typestate-correct surface to fill.
//!
//! Like DBSCAN, HDBSCAN is a labels-only estimator: it exposes `labels_` and has
//! NO standalone `predict`/`transform` (so neither [`Predict`] nor [`Transform`]
//! is implemented). The `-1` noise sentinel matches `cluster/mod.rs`'s contract.
//!
//! ## The convention this shell embodies
//! - `Hdbscan<F, S = Unfit>` carries its lifecycle state as `PhantomData<S>`
//!   (D-01) so `labels`-before-`fit` is a COMPILE error.
//! - Construction is builder-only: [`Hdbscan::builder`] →
//!   [`HdbscanBuilder::build`] with data-INDEPENDENT validation up front (D-08 —
//!   `min_cluster_size >= 2`).
//! - [`Hdbscan::new`] is the SINGLE source of the sklearn defaults (D-08); the
//!   builder `Default` re-derives via `new().into_builder()`.
//! - [`Fit::fit`] CONSUMES `self` and returns `Hdbscan<F, Fitted>` (D-02).
//! - The `labels`/`n_features_in` accessors exist ONLY on
//!   `impl Hdbscan<F, Fitted>`.
//!
//! Tests live in `crates/mlrs-algos/tests/hdbscan_test.rs` (AGENTS.md §2).
//!
//! [`Predict`]: crate::typestate::Predict
//! [`Transform`]: crate::typestate::Transform

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, Unfit};

// Host back-end submodules (HDBS-02, plan 15-03). Pure scalar Rust — the
// deliberate GPU-tree-atomics dodge (RESEARCH). `mst` holds both oracle Prim
// variants + argsort-by-weight; `single_linkage` holds the UnionFind +
// make_single_linkage that the Wave-3 condense/select stage (plan 15-04)
// consumes.
pub mod condense;
pub mod mst;
pub mod single_linkage;
pub mod stability;

/// Distance metric for the HDBSCAN neighbor graph (HDBS-01, D-01). The five
/// feature-space metrics mirror [`mlrs_backend::prims::knn_graph::Metric`]
/// (consumed via the Phase-13 KNN prim with `include_self=true`); `Precomputed`
/// (D-02) is the new variant where `fit` interprets `X` as a square `n×n`
/// distance matrix and skips the device distance front-end.
///
/// NOTE: the `Minkowski { p: f64 }` variant carries an `f64`, which is NOT
/// `Eq` (no total order on floats), so this enum derives `PartialEq` ONLY (the
/// Phase-12 shell's `Eq` is dropped — see `hyperparams_eq`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    /// Euclidean (L2) distance — sklearn's `metric='euclidean'` default.
    Euclidean,
    /// L1 (Manhattan) distance — sklearn's `metric='manhattan'`.
    Manhattan,
    /// Cosine distance `1 − x̂·ŷ` — sklearn's `metric='cosine'` (routes to the
    /// dense brute MST variant, NOT a `FAST_METRIC`).
    Cosine,
    /// L∞ (Chebyshev) distance — sklearn's `metric='chebyshev'`.
    Chebyshev,
    /// Minkowski-`p` distance — sklearn's `metric='minkowski'` with `p`. The
    /// exponent is validated `>= 1` host-side at [`HdbscanBuilder::build`]
    /// (knn_graph precedent).
    Minkowski {
        /// The Minkowski exponent (validated `>= 1`).
        p: f64,
    },
    /// Precomputed distance matrix (D-02). `X` is interpreted as a square `n×n`
    /// distance matrix; the device distance front-end is skipped and the dense
    /// brute MST variant is used. `fit` validates squareness host-side.
    Precomputed,
}

/// Which cluster centers to compute and store (`store_centers`, HDBS-04 / D-08).
/// `None` on the estimator stores neither; the actual centroid/medoid compute
/// lands in plan 15-06 — the field is wired now so the builder surface is
/// complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreCenters {
    /// `'centroid'` — weighted mean per cluster → `centroids_`.
    Centroid,
    /// `'medoid'` — min-weighted-total-distance member per cluster → `medoids_`.
    Medoid,
    /// `'both'` — compute and store BOTH `centroids_` and `medoids_`.
    Both,
}

/// Cluster-selection method for the condensed-tree extraction (HDBS-01 subset).
/// Ignored by the Phase-12 trivial fit (which always emits all-`-1`); retained
/// so the builder surface matches sklearn's `cluster_selection_method=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterSelectionMethod {
    /// Excess-of-mass — sklearn's `'eom'` default.
    Eom,
    /// Leaf-cluster selection — sklearn's `'leaf'`.
    Leaf,
}

/// HDBSCAN density-based clustering estimator shell (HDBS-01). Construct via
/// [`Hdbscan::builder`], then [`Fit::fit`] (which CONSUMES `self`, returning
/// `Hdbscan<F, Fitted>`). The fitted `labels_` (length `n`, `-1` = noise,
/// device-resident `i32`) is reachable only through
/// [`Hdbscan<F, Fitted>::labels`].
///
/// The `S` type parameter is the compile-time lifecycle state
/// ([`Unfit`](crate::typestate::Unfit) /
/// [`Fitted`](crate::typestate::Fitted)); it defaults to `Unfit`.
pub struct Hdbscan<F, S = Unfit> {
    // --- HDBS-01 hyperparameter surface (data-independent) ---
    /// Minimum cluster size (`min_cluster_size`, default 5).
    min_cluster_size: usize,
    /// Core-distance smoothing (`min_samples`, default `None` → resolved to
    /// `min_cluster_size` in `new()`/`build()`). Stored verbatim; its semantic
    /// validation is deferred to Phase 15.
    min_samples: Option<usize>,
    /// Cluster-merge distance threshold (`cluster_selection_epsilon`, default
    /// 0.0).
    cluster_selection_epsilon: f64,
    /// Condensed-tree extraction method (`cluster_selection_method`, default
    /// Eom).
    cluster_selection_method: ClusterSelectionMethod,
    /// Neighbor-graph distance metric (`metric`, default Euclidean).
    metric: Metric,
    /// Robust-single-linkage distance scaling (`alpha`, default 1.0).
    alpha: f64,
    /// Maximum cluster size, `0` = unbounded (`max_cluster_size`, default 0).
    max_cluster_size: usize,
    /// Which cluster centers to compute (`store_centers`, default `None`). The
    /// compute lands in plan 15-06; wired here so the surface is complete
    /// (HDBS-04 / D-08).
    store_centers: Option<StoreCenters>,

    // --- fitted state (None / 0 until fit; Some on Fitted by construction) ---
    /// Fitted labels (length `n`, `-1` = noise), device-resident `i32`. `None`
    /// until fit.
    labels_: Option<DeviceArray<ActiveRuntime, i32>>,
    /// The single-linkage hierarchy from the MST → `make_single_linkage` pass
    /// (HDBS-02, plan 15-03). `Some` after a precomputed fit, `None` otherwise
    /// (the feature-metric device front-end lands in plan 15-05). Stored host-side
    /// for the Wave-3 condense/select stage (plan 15-04) — NOT device-resident
    /// (the back-end is pure host).
    single_linkage_: Option<Vec<single_linkage::SingleLinkageEdge>>,
    /// Number of features seen at fit (`n_features_in_`). `0` until fit.
    n_features_in_: usize,
    /// Phantom over the float type (the shell stores no `F` until Phase 15's real
    /// fit; the type parameter is carried for API uniformity with UMAP).
    _float: PhantomData<F>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> Hdbscan<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an `Hdbscan` with sklearn's defaults directly in the `Unfit`
    /// state (HDBS-01). This is the SINGLE source of truth for the default
    /// hyperparameters (D-08): the builder `Default` re-derives from here via
    /// [`Hdbscan::into_builder`]. `min_samples=None` is resolved to
    /// `min_cluster_size` here. Defaults are trusted valid, so this bypasses
    /// [`HdbscanBuilder::build`]'s validation.
    pub fn new() -> Self {
        let min_cluster_size = 5;
        Self {
            min_cluster_size,
            // None → resolved to min_cluster_size (HDBS-01 default rule).
            min_samples: Some(min_cluster_size),
            cluster_selection_epsilon: 0.0,
            cluster_selection_method: ClusterSelectionMethod::Eom,
            metric: Metric::Euclidean,
            alpha: 1.0,
            max_cluster_size: 0,
            store_centers: None,
            labels_: None,
            single_linkage_: None,
            n_features_in_: 0,
            _float: PhantomData,
            _state: PhantomData,
        }
    }

    /// Start building an `Hdbscan` from sklearn's defaults (D-08 single source).
    pub fn builder() -> HdbscanBuilder {
        HdbscanBuilder::default()
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `labels_`/`n_features_in_` fields are excluded). Used by the
    /// defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.min_cluster_size == other.min_cluster_size
            && self.min_samples == other.min_samples
            && self.cluster_selection_epsilon == other.cluster_selection_epsilon
            && self.cluster_selection_method == other.cluster_selection_method
            && self.metric == other.metric
            && self.alpha == other.alpha
            && self.max_cluster_size == other.max_cluster_size
            && self.store_centers == other.store_centers
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by `HdbscanBuilder::default` to re-derive the
    /// defaults from [`Hdbscan::new`] (D-08) and available to callers who want to
    /// tweak a constructed estimator before fitting.
    pub fn into_builder(self) -> HdbscanBuilder {
        HdbscanBuilder {
            min_cluster_size: self.min_cluster_size,
            min_samples: self.min_samples,
            cluster_selection_epsilon: self.cluster_selection_epsilon,
            cluster_selection_method: self.cluster_selection_method,
            metric: self.metric,
            alpha: self.alpha,
            max_cluster_size: self.max_cluster_size,
            store_centers: self.store_centers,
        }
    }
}

impl<F> Default for Hdbscan<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`Hdbscan`] (D-01). Owned chained setters mirror
/// `mbsgd_regressor.rs`'s template; `Default` re-derives the sklearn defaults
/// from [`Hdbscan::new`] (D-08) rather than holding literals.
#[derive(Debug, Clone, Copy)]
pub struct HdbscanBuilder {
    min_cluster_size: usize,
    min_samples: Option<usize>,
    cluster_selection_epsilon: f64,
    cluster_selection_method: ClusterSelectionMethod,
    metric: Metric,
    alpha: f64,
    max_cluster_size: usize,
    store_centers: Option<StoreCenters>,
}

impl Default for HdbscanBuilder {
    /// Re-derive the sklearn defaults from [`Hdbscan::new`] (D-08 single source).
    /// `f64` is pinned only to read the F-independent scalar defaults — the
    /// builder is non-generic (Pitfall 4: do NOT hand-write literal defaults).
    fn default() -> Self {
        Hdbscan::<f64, Unfit>::new().into_builder()
    }
}

impl HdbscanBuilder {
    /// Set the minimum cluster size `min_cluster_size`.
    pub fn min_cluster_size(mut self, v: usize) -> Self {
        self.min_cluster_size = v;
        self
    }
    /// Set the core-distance smoothing `min_samples` (`None` → resolved to
    /// `min_cluster_size` at build).
    pub fn min_samples(mut self, v: Option<usize>) -> Self {
        self.min_samples = v;
        self
    }
    /// Set the cluster-merge threshold `cluster_selection_epsilon`.
    pub fn cluster_selection_epsilon(mut self, v: f64) -> Self {
        self.cluster_selection_epsilon = v;
        self
    }
    /// Set the condensed-tree extraction `cluster_selection_method`.
    pub fn cluster_selection_method(mut self, v: ClusterSelectionMethod) -> Self {
        self.cluster_selection_method = v;
        self
    }
    /// Set the neighbor-graph distance `metric`.
    pub fn metric(mut self, v: Metric) -> Self {
        self.metric = v;
        self
    }
    /// Set the robust-single-linkage scaling `alpha`.
    pub fn alpha(mut self, v: f64) -> Self {
        self.alpha = v;
        self
    }
    /// Set the maximum cluster size `max_cluster_size` (`0` = unbounded).
    pub fn max_cluster_size(mut self, v: usize) -> Self {
        self.max_cluster_size = v;
        self
    }
    /// Set which cluster centers to compute `store_centers` (`None` = neither).
    pub fn store_centers(mut self, v: Option<StoreCenters>) -> Self {
        self.store_centers = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08; the data-DEPENDENT
    /// geometry check lives in [`Fit::fit`]):
    ///
    /// - `min_cluster_size >= 2` ([`BuildError::InvalidMinClusterSize`]).
    /// - `min_samples >= 1` when `Some` ([`BuildError::InvalidMinSamples`]).
    /// - `max_cluster_size == 0` (unbounded) or `>= min_cluster_size`
    ///   ([`BuildError::InvalidMaxClusterSize`]).
    /// - `alpha > 0` ([`BuildError::InvalidAlphaHdbscan`]).
    /// - `Metric::Minkowski { p }` requires `p >= 1`
    ///   ([`BuildError::InvalidMinkowskiP`], knn_graph precedent).
    ///
    /// All checks run BEFORE the estimator is constructed (T-15-03-V5b / ASVS V5
    /// — an untrusted hyperparameter becomes a typed error, never a device
    /// fault). `min_samples=None` is resolved to `min_cluster_size`.
    pub fn build<F>(self) -> Result<Hdbscan<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if self.min_cluster_size < 2 {
            return Err(BuildError::InvalidMinClusterSize {
                estimator: "hdbscan",
                min_cluster_size: self.min_cluster_size,
            });
        }
        // min_samples >= 1 when explicitly Some (None resolves to
        // min_cluster_size, which is already >= 2). Resolves the shell's deferred
        // validation TODO (D-09 / T-15-03-V5b).
        if let Some(ms) = self.min_samples {
            if ms < 1 {
                return Err(BuildError::InvalidMinSamples {
                    estimator: "hdbscan",
                    min_samples: ms,
                });
            }
        }
        // max_cluster_size: 0 = unbounded; otherwise it must not be smaller than
        // min_cluster_size (a finite bound below the floor is contradictory).
        if self.max_cluster_size != 0 && self.max_cluster_size < self.min_cluster_size {
            return Err(BuildError::InvalidMaxClusterSize {
                estimator: "hdbscan",
                max_cluster_size: self.max_cluster_size,
                min_cluster_size: self.min_cluster_size,
            });
        }
        // alpha > 0 (it divides pairwise distances in the MST; 0 → div-by-zero,
        // negative → flipped distances).
        if !(self.alpha > 0.0) {
            return Err(BuildError::InvalidAlphaHdbscan {
                estimator: "hdbscan",
                alpha: self.alpha,
            });
        }
        // Minkowski p >= 1 (proper distance; knn_graph precedent).
        if let Metric::Minkowski { p } = self.metric {
            if !(p >= 1.0) {
                return Err(BuildError::InvalidMinkowskiP {
                    estimator: "hdbscan",
                    p,
                });
            }
        }
        let min_samples = Some(self.min_samples.unwrap_or(self.min_cluster_size));
        Ok(Hdbscan {
            min_cluster_size: self.min_cluster_size,
            min_samples,
            cluster_selection_epsilon: self.cluster_selection_epsilon,
            cluster_selection_method: self.cluster_selection_method,
            metric: self.metric,
            alpha: self.alpha,
            max_cluster_size: self.max_cluster_size,
            store_centers: self.store_centers,
            labels_: None,
            single_linkage_: None,
            n_features_in_: 0,
            _float: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for Hdbscan<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = Hdbscan<F, Fitted>;

    /// Fit HDBSCAN (HDBS-02, plan 15-03 slice). Validates the data-DEPENDENT
    /// geometry (D-08), then runs the per-metric pipeline. CONSUMES `self`,
    /// returning the `Fitted`-tagged sibling (D-02). `y` is ignored (unsupervised).
    ///
    /// As of plan 15-03 the **precomputed** path (`Metric::Precomputed`) runs the
    /// exact host back-end up to the single-linkage hierarchy: validate `X` is a
    /// square `n×n` distance matrix, divide by `alpha`, compute core distances
    /// (the `(min_samples-1)`-th smallest per row), build the dense
    /// mutual-reachability, run the dense Variant-A Prim's MST, argsort by weight,
    /// and fold into `make_single_linkage`. The hierarchy is stored on
    /// `single_linkage_` for the Wave-3 condense/select stage (plan 15-04); until
    /// that wires the tree, `labels_` stays all-`-1` (the shell contract holds so
    /// the estimator still fits/compiles). The five feature-space metrics keep the
    /// trivial all-`-1` fit until the device front-end lands in plan 15-05.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Hdbscan<F, Fitted>, AlgoError> {
        let (n, p) = shape;

        // Data-DEPENDENT geometry guard BEFORE any compute (shared helper).
        validate_geometry(x, shape)?;

        // The single-linkage hierarchy is produced only by the precomputed path in
        // this slice; the feature-metric device front-end (plan 15-05) fills it
        // later. Labels stay all-`-1` until plan 15-04 wires the condensed tree.
        let single_linkage_ = if self.metric == Metric::Precomputed {
            Some(self.precomputed_single_linkage(pool, x, n, p)?)
        } else {
            None
        };

        // Labels-only contract holds until 15-04: all-`-1` (noise sentinel). NO
        // kernel for the precomputed path (pure host MST); the feature metrics
        // keep the trivial fit until 15-05.
        let labels = vec![-1_i32; n];
        let labels_dev = DeviceArray::from_host(pool, &labels);

        Ok(Hdbscan {
            min_cluster_size: self.min_cluster_size,
            min_samples: self.min_samples,
            cluster_selection_epsilon: self.cluster_selection_epsilon,
            cluster_selection_method: self.cluster_selection_method,
            metric: self.metric,
            alpha: self.alpha,
            max_cluster_size: self.max_cluster_size,
            store_centers: self.store_centers,
            labels_: Some(labels_dev),
            single_linkage_,
            n_features_in_: p,
            _float: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<F> Hdbscan<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Precomputed-path host back-end (D-02): `X` is interpreted as a square
    /// `n×n` distance matrix. Validates squareness (typed error — the device
    /// never sees a malformed shape, T-15-03-V5a), reads `X` to host, scales by
    /// `alpha` (the Variant-A placement: the WHOLE matrix BEFORE core distances),
    /// computes core distances, builds the dense mutual-reachability, runs the
    /// dense Prim's MST → argsort → single-linkage. Returns the hierarchy.
    ///
    /// NOTE (D-02): sklearn additionally requires the precomputed matrix to be
    /// SYMMETRIC (`np.allclose(X, X.T)`). We document that expectation here; the
    /// dense Variant-A MST reads `mr[current_node][..]` rows, so an asymmetric
    /// input would silently use the upper-triangle reading — callers must supply
    /// a symmetric matrix (the committed fixtures are `pairwise_distances`, which
    /// is symmetric by construction).
    fn precomputed_single_linkage(
        &self,
        pool: &BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        n: usize,
        p: usize,
    ) -> Result<Vec<single_linkage::SingleLinkageEdge>, AlgoError> {
        // T-15-03-V5a: a precomputed matrix MUST be square (n == p). Reject with a
        // typed PrimError before reading anything to host.
        if n != p {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "precomputed_distance_matrix",
                rows: n,
                cols: p,
                len: n * p,
            }));
        }

        // Read the dense matrix to host f64 (the shared bridging idiom).
        let dist_raw: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();

        // Variant-A alpha placement: divide the WHOLE matrix by alpha BEFORE core
        // distances (sklearn `_hdbscan_brute`).
        let alpha = self.alpha;
        let dist: Vec<f64> = dist_raw.iter().map(|&d| d / alpha).collect();

        // Core distance = (min_samples-1)-th smallest per row (incl. self-zero).
        let min_samples = self.min_samples.unwrap_or(self.min_cluster_size);
        let core = mst::core_distances_dense(&dist, n, min_samples);

        // Dense mutual-reachability + Variant-A Prim's MST → argsort → linkage.
        let mr = mst::mutual_reachability_dense(&dist, &core, n);
        let edges = mst::mst_from_mutual_reachability(&mr, n);
        let sorted = mst::argsort_by_weight(&edges);
        Ok(single_linkage::make_single_linkage(&sorted, n))
    }
}

impl<F> Hdbscan<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `labels_` (length `n`, `-1` = noise). `Some` by
    /// construction on the `Fitted` state, so no `NotFitted` branch is needed
    /// (the compile-time typestate replaces the runtime guard, D-02).
    pub fn labels(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<i32> {
        self.labels_
            .as_ref()
            .expect("labels_ is Some by construction on Hdbscan<F, Fitted>")
            .to_host(pool)
    }

    /// Number of features seen at fit (`n_features_in_`).
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    /// The single-linkage hierarchy produced by the MST → `make_single_linkage`
    /// pass (HDBS-02). `Some` after a precomputed fit (plan 15-03); `None` for the
    /// feature-space metrics until the device front-end lands (plan 15-05). The
    /// Wave-3 condense/select stage (plan 15-04) consumes this to drive labelling.
    pub fn single_linkage(&self) -> Option<&[single_linkage::SingleLinkageEdge]> {
        self.single_linkage_.as_deref()
    }
}
