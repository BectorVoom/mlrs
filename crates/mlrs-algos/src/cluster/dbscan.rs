//! `DBSCAN` (CLUSTER-02) — density-based clustering via the device eps-core mask
//! + the HOST index-ordered DFS, matching `sklearn.cluster.DBSCAN` up to a label
//! permutation (D-04 / Pitfall 7).
//!
//! ## Device core mask, HOST graph walk (D-04)
//! The n²-heavy work — the pairwise squared-distance matrix, the `<= eps²`
//! threshold, and each point's self-inclusive eps-neighbor count — runs on the
//! device via the validated Phase-5 [`eps_core_mask`] prim, which reads back the
//! host `is_core` mask + the `n × n` adjacency (the SINGLE documented D-04
//! round-trip). The cluster expansion is an inherently SEQUENTIAL graph
//! traversal, so it runs on the HOST (D-04): this estimator owns the DFS, the
//! prim does NOT expand clusters.
//!
//! ## The DFS is EXACTLY `_dbscan_inner.pyx` (index-ordered LIFO, Pitfall 7)
//! `labels` init to `-1` (noise). For each seed `i in 0..n` IN INDEX ORDER, skip
//! if `i` is already labeled or not a core point; otherwise start a new cluster
//! `label_num`, push `i` onto a LIFO stack, and while the stack is non-empty pop
//! `v`, label it `label_num` if still unlabeled, and — only if `v` is itself a
//! core point — push every still-unlabeled eps-neighbor of `v`. Increment
//! `label_num` per seed. A border point therefore joins the cluster of the FIRST
//! core point that reaches it IN INDEX ORDER — order-dependent but DETERMINISTIC
//! (Pitfall 7), reproducing sklearn bit-for-bit (up to a cluster-label
//! permutation).
//!
//! ## Stored fitted state (device-resident, D-03 / D-06)
//! `labels_` (length `n`, `i32` — noise = `-1`, directly representable, D-06) and
//! `core_sample_indices_` (the ascending `i32` indices of the core points). Both
//! are materialized to a device-resident [`DeviceArray`] after the host DFS.
//!
//! ## NO standalone `predict` (D-08)
//! DBSCAN is NON-transductive: sklearn provides no `predict` for new points (a
//! query point's cluster is not well-defined without re-running the density
//! estimation). This estimator therefore implements [`Fit`] + [`fit_predict`]
//! only and DOES NOT implement
//! [`PredictLabels`](crate::typestate::PredictLabels) /
//! [`Predict`](crate::typestate::Predict) (D-08).
//!
//! ## Validate the untrusted hyperparameters at the RIGHT layer (ASVS V5)
//! The data-INDEPENDENT hyperparameter validation — `eps >= 0`
//! ([`BuildError::InvalidEps`]) and `min_samples >= 1`
//! ([`BuildError::InvalidMinSamples`]) — is performed in
//! [`DbscanBuilder::build`] BEFORE any data is seen (D-08 split, T-05-07-01). The
//! data-DEPENDENT geometry guard ([`validate_geometry`]) stays at the TOP of
//! [`Fit::fit`] before the prim launch — a tampered hyperparameter / geometry
//! becomes a typed error, not an out-of-bounds read.
//!
//! Tests live in `crates/mlrs-algos/tests/dbscan_test.rs` (AGENTS.md §2), never
//! an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::dbscan::eps_core_mask;
use mlrs_backend::runtime::ActiveRuntime;

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, Unfit};

/// Density-based spatial clustering (CLUSTER-02) via the device eps-core mask +
/// the host index-ordered DFS.
///
/// Construct with the zero-arg [`DBSCAN::new`] (sklearn defaults: `eps = 0.5`,
/// `min_samples = 5`) or [`DBSCAN::builder`], then the consuming [`Fit::fit`]
/// (returns the `Fitted`-tagged sibling) or
/// [`fit_predict`](DBSCAN::fit_predict). Fitted `labels_` (noise = `-1`) /
/// `core_sample_indices_` are device-resident (D-03); the host accessors
/// materialize them on demand and exist ONLY on `DBSCAN<F, Fitted>` (the
/// compile-time typestate replaces the old runtime `NotFitted` guard, D-03).
/// DBSCAN is non-transductive: there is NO standalone `predict` (D-08).
pub struct DBSCAN<F, S = Unfit> {
    /// Neighborhood radius `eps` (Euclidean; the prim thresholds the SQUARED
    /// distance at `eps²`). Validated `eps >= 0` at [`DbscanBuilder::build`] →
    /// [`BuildError::InvalidEps`] BEFORE any data is seen (T-05-07-01).
    eps: f64,
    /// Core-point threshold: a point is core iff its self-inclusive eps-neighbor
    /// count `>= min_samples`. Validated `min_samples >= 1` at
    /// [`DbscanBuilder::build`] → [`BuildError::InvalidMinSamples`].
    min_samples: usize,
    /// Fitted length-`n` integer labels (`i32`, noise = `-1`, D-06),
    /// device-resident, `None` until `fit`.
    labels_: Option<DeviceArray<ActiveRuntime, i32>>,
    /// Fitted ascending core-point indices (`i32`), device-resident, `None`
    /// until `fit`.
    core_sample_indices_: Option<DeviceArray<ActiveRuntime, i32>>,
    /// `PhantomData`-free float binding: `F` is the element type of the input
    /// `x` the prim consumes; DBSCAN keeps no `F`-typed fitted state (labels are
    /// integer), so a zero-sized marker carries the type parameter.
    _marker: PhantomData<F>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> DBSCAN<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfitted `DBSCAN` with sklearn's `DBSCAN` defaults
    /// (`eps = 0.5`, `min_samples = 5`) directly in the `Unfit` state. This is
    /// the SINGLE source of truth for the default hyperparameters (D-08): the
    /// builder `Default` re-derives from here via [`DBSCAN::into_builder`],
    /// rather than re-listing the literals. Defaults are trusted valid, so this
    /// bypasses [`DbscanBuilder::build`]'s validation.
    pub fn new() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
            labels_: None,
            core_sample_indices_: None,
            _marker: PhantomData,
            _state: PhantomData,
        }
    }

    /// Start building a `DBSCAN` from sklearn's defaults (D-08 single source).
    pub fn builder() -> DbscanBuilder {
        DbscanBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`DbscanBuilder::default`] to re-derive the
    /// defaults from [`DBSCAN::new`] (D-08), and available to callers who want
    /// to tweak a constructed estimator before fitting.
    pub fn into_builder(self) -> DbscanBuilder {
        DbscanBuilder {
            eps: self.eps,
            min_samples: self.min_samples,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `labels_`/`core_sample_indices_` fields are excluded — both are `None` in
    /// any `Unfit` value). Used by the defaults-equality test (BLDR-01):
    /// `DBSCAN::new().hyperparams_eq(&DBSCAN::builder().build()?)`.
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.eps == other.eps && self.min_samples == other.min_samples
    }
}

impl<F> Default for DBSCAN<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`DBSCAN`] (D-01). `Default` re-derives the sklearn defaults from
/// [`DBSCAN::new`] (D-08 single source) rather than holding literals (Pitfall 1:
/// default-drift breaks the oracle gate silently).
#[derive(Debug, Clone, Copy)]
pub struct DbscanBuilder {
    eps: f64,
    min_samples: usize,
}

impl Default for DbscanBuilder {
    /// Re-derive the sklearn defaults from [`DBSCAN::new`] (D-08 single source).
    /// `f64` is pinned only to read the F-independent scalar defaults — the
    /// builder is non-generic, so the choice of `F` here is irrelevant.
    fn default() -> Self {
        DBSCAN::<f64, Unfit>::new().into_builder()
    }
}

impl DbscanBuilder {
    /// Set the neighborhood radius `eps`.
    pub fn eps(mut self, v: f64) -> Self {
        self.eps = v;
        self
    }

    /// Set the core-point threshold `min_samples`.
    pub fn min_samples(mut self, v: usize) -> Self {
        self.min_samples = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08; the data-DEPENDENT
    /// geometry check lives in [`Fit::fit`]):
    ///
    /// - `eps >= 0` and finite ([`BuildError::InvalidEps`]) — a negative or
    ///   non-finite radius is geometrically meaningless (relocated from the old
    ///   fit-body check, T-05-07-01 / Pitfall 7).
    /// - `min_samples >= 1` ([`BuildError::InvalidMinSamples`]) — a core point
    ///   requires at least itself in its eps-neighborhood.
    pub fn build<F>(self) -> Result<DBSCAN<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if !(self.eps >= 0.0) || !self.eps.is_finite() {
            return Err(BuildError::InvalidEps {
                estimator: "dbscan",
                eps: self.eps,
            });
        }
        if self.min_samples < 1 {
            return Err(BuildError::InvalidMinSamples {
                estimator: "dbscan",
                min_samples: self.min_samples,
            });
        }
        Ok(DBSCAN {
            eps: self.eps,
            min_samples: self.min_samples,
            labels_: None,
            core_sample_indices_: None,
            _marker: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<F> DBSCAN<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `labels_` (length `n`, `i32`, noise = `-1`).
    /// `Some` by construction on the `Fitted` state, so no `NotFitted` branch is
    /// needed (the compile-time typestate replaces the runtime guard, D-03).
    pub fn labels(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<i32> {
        self.labels_
            .as_ref()
            .expect("labels_ is Some by construction on DBSCAN<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `core_sample_indices_` (ascending `i32` indices of
    /// the core points). `Some` by construction on the `Fitted` state (D-03).
    pub fn core_sample_indices(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<i32> {
        self.core_sample_indices_
            .as_ref()
            .expect("core_sample_indices_ is Some by construction on DBSCAN<F, Fitted>")
            .to_host(pool)
    }
}

impl<F> DBSCAN<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Convenience `fit_predict` (sklearn `ClusterMixin`): fit to `x` then return
    /// the fitted `labels_` as a fresh device-resident `i32` buffer (noise =
    /// `-1`). CONSUMES `self` (the typestate `fit` transition) and returns BOTH
    /// the `Fitted`-tagged estimator and the labels buffer.
    pub fn fit_predict(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<(DBSCAN<F, Fitted>, DeviceArray<ActiveRuntime, i32>), AlgoError> {
        let fitted = self.fit(pool, x, None, shape)?;
        let labels = fitted.labels(pool);
        let labels_dev = DeviceArray::from_host(pool, &labels);
        Ok((fitted, labels_dev))
    }
}

impl<F> Fit<F> for DBSCAN<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = DBSCAN<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<DBSCAN<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-05-07-01 / ASVS V5: data-DEPENDENT geometry guard BEFORE the prim
        //     launch (the data-INDEPENDENT `eps >= 0` / `min_samples >= 1` checks
        //     were validated at build() — Pitfall 7). ---
        validate_geometry(x, shape)?;

        // --- Device core mask + adjacency (D-04): the n² distance + <= eps²
        //     threshold + per-row self-inclusive count run on the device; the host
        //     reads back is_core + the n × n adjacency (the single documented
        //     round-trip). ---
        let mask = eps_core_mask::<F>(
            pool,
            x,
            n_samples,
            n_features,
            self.eps,
            self.min_samples as u32,
        )?;

        // --- HOST index-ordered LIFO DFS EXACTLY per _dbscan_inner.pyx (Pitfall 7).
        //     labels init -1 (noise); seeds scanned in 0..n index order; a border
        //     point joins the FIRST core point reaching it in that order. ---
        let n = n_samples;
        let mut labels: Vec<i32> = vec![-1; n];
        let mut label_num: i32 = 0;
        let mut stack: Vec<usize> = Vec::new();

        for i in 0..n {
            // Skip already-labeled points and non-core seeds (a cluster only ever
            // STARTS from a core point; border points are absorbed during a walk).
            if labels[i] != -1 || !mask.is_core[i] {
                continue;
            }
            stack.clear();
            stack.push(i);
            while let Some(v) = stack.pop() {
                // LIFO pop; label v if still unlabeled.
                if labels[v] == -1 {
                    labels[v] = label_num;
                    // Only EXPAND from a core point: push its still-unlabeled
                    // eps-neighbors (ascending index order — neighbors(v) is
                    // ascending). Border points are labeled but never expanded.
                    if mask.is_core[v] {
                        for u in mask.neighbors(v) {
                            if labels[u] == -1 {
                                stack.push(u);
                            }
                        }
                    }
                }
            }
            label_num += 1;
        }

        // --- core_sample_indices_ = { i : is_core[i] } in ascending order (noise
        //     stays -1). The DFS already used the SAME is_core mask, so the cluster
        //     assignment and the core set are consistent. ---
        let core_indices: Vec<i32> = (0..n)
            .filter(|&i| mask.is_core[i])
            .map(|i| i as i32)
            .collect();

        // --- Materialize the fitted state device-resident (D-03 / D-06: i32
        //     labels with the -1 noise sentinel ride the byte-keyed pool with no
        //     pool/bridge changes). ---
        let labels_dev: DeviceArray<ActiveRuntime, i32> = DeviceArray::from_host(pool, &labels);
        let core_dev: DeviceArray<ActiveRuntime, i32> = DeviceArray::from_host(pool, &core_indices);

        Ok(DBSCAN {
            eps: self.eps,
            min_samples: self.min_samples,
            labels_: Some(labels_dev),
            core_sample_indices_: Some(core_dev),
            _marker: PhantomData,
            _state: PhantomData,
        })
    }
}
