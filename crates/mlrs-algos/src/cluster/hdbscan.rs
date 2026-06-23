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
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::typestate::{Fit, Fitted, Unfit};

/// Distance metric for the HDBSCAN neighbor graph (HDBS-01 subset). Only
/// `Euclidean` carries meaning in the Phase-12 shell (the trivial fit ignores
/// the metric); the full metric set is filled in Phase 15.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean (L2) distance — sklearn's `metric='euclidean'` default.
    Euclidean,
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

    // --- fitted state (None / 0 until fit; Some on Fitted by construction) ---
    /// Fitted labels (length `n`, `-1` = noise), device-resident `i32`. `None`
    /// until fit.
    labels_: Option<DeviceArray<ActiveRuntime, i32>>,
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
            labels_: None,
            n_features_in_: 0,
            _float: PhantomData,
            _state: PhantomData,
        }
    }

    /// Start building an `Hdbscan` from sklearn's defaults (D-08 single source).
    pub fn builder() -> HdbscanBuilder {
        HdbscanBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by `HdbscanBuilder::default` to re-derive the
    /// defaults from [`Hdbscan::new`] (D-08).
    pub fn into_builder(self) -> HdbscanBuilder {
        HdbscanBuilder {
            min_cluster_size: self.min_cluster_size,
            min_samples: self.min_samples,
            cluster_selection_epsilon: self.cluster_selection_epsilon,
            cluster_selection_method: self.cluster_selection_method,
            metric: self.metric,
            alpha: self.alpha,
            max_cluster_size: self.max_cluster_size,
        }
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

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08; the data-DEPENDENT
    /// geometry check lives in [`Fit::fit`]):
    ///
    /// - `min_cluster_size >= 2` ([`BuildError::InvalidMinClusterSize`]).
    ///
    /// `min_samples` is stored verbatim and is NOT validated in Phase 12 — its
    /// semantic validation is deferred to Phase 15 (the real HDBSCAN compute).
    /// `min_samples=None` is resolved to `min_cluster_size`.
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
        let min_samples = Some(self.min_samples.unwrap_or(self.min_cluster_size));
        Ok(Hdbscan {
            min_cluster_size: self.min_cluster_size,
            min_samples,
            cluster_selection_epsilon: self.cluster_selection_epsilon,
            cluster_selection_method: self.cluster_selection_method,
            metric: self.metric,
            alpha: self.alpha,
            max_cluster_size: self.max_cluster_size,
            labels_: None,
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

    /// NON-algorithmic trivial fit (Phase-12 shell — real HDBSCAN lands in Phase
    /// 15). Validates the data-DEPENDENT geometry (D-08), then allocates an
    /// all-`-1` (noise sentinel) `labels_` of length `n` — NO kernel, NO
    /// compute. CONSUMES `self`, returning the `Fitted`-tagged sibling (D-02).
    /// `y` is ignored (HDBSCAN is unsupervised).
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Hdbscan<F, Fitted>, AlgoError> {
        let (n, p) = shape;

        // Data-DEPENDENT geometry guard BEFORE the allocation (mirrors
        // mbsgd_regressor.rs:303-312).
        if n == 0 || p == 0 || x.len() != n * p {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n,
                cols: p,
                len: x.len(),
            }));
        }

        // Trivial non-algorithmic fit: all-`-1` labels. NO kernel, NO compute.
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
            labels_: Some(labels_dev),
            n_features_in_: p,
            _float: PhantomData,
            _state: PhantomData,
        })
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
}
