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
//! estimation). This estimator therefore implements `Fit` + [`fit_predict`] only
//! and DOES NOT implement [`PredictLabels`](crate::traits::PredictLabels) /
//! [`Predict`](crate::traits::Predict) (D-08).
//!
//! ## Validate the untrusted hyperparameters BEFORE any launch (ASVS V5)
//! `fit` rejects `eps < 0` ([`AlgoError::InvalidEps`]) and `min_samples < 1`
//! ([`AlgoError::InvalidMinSamples`]) BEFORE the prim launch (T-05-07-01) — a
//! tampered hyperparameter becomes a typed error, not an out-of-bounds read.
//!
//! Tests live in `crates/mlrs-algos/tests/dbscan_test.rs` (AGENTS.md §2), never
//! an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::dbscan::eps_core_mask;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;
use crate::traits::Fit;

/// Density-based spatial clustering (CLUSTER-02) via the device eps-core mask +
/// the host index-ordered DFS.
///
/// Construct with [`DBSCAN::new`] (`eps`, `min_samples`), then [`Fit::fit`] or
/// [`fit_predict`](DBSCAN::fit_predict). Fitted `labels_` (noise = `-1`) /
/// `core_sample_indices_` are device-resident (D-03); the host accessors
/// materialize them on demand. DBSCAN is non-transductive: there is NO
/// standalone `predict` (D-08).
pub struct DBSCAN<F> {
    /// Neighborhood radius `eps` (Euclidean; the prim thresholds the SQUARED
    /// distance at `eps²`). Validated `eps >= 0` at `fit` → [`AlgoError::InvalidEps`]
    /// BEFORE any launch (T-05-07-01).
    eps: f64,
    /// Core-point threshold: a point is core iff its self-inclusive eps-neighbor
    /// count `>= min_samples`. Validated `min_samples >= 1` → [`AlgoError::InvalidMinSamples`].
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
    _marker: std::marker::PhantomData<F>,
}

impl<F> DBSCAN<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `DBSCAN` with neighborhood radius `eps` and core
    /// threshold `min_samples`. A bad `eps` (`< 0`) or `min_samples` (`< 1`) is
    /// rejected at `fit` time ([`AlgoError::InvalidEps`] / [`AlgoError::InvalidMinSamples`]).
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            labels_: None,
            core_sample_indices_: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Host copy of the fitted `labels_` (length `n`, `i32`, noise = `-1`).
    /// Errors with [`AlgoError::NotFitted`] before `fit`.
    pub fn labels(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<i32>, AlgoError> {
        self.labels_
            .as_ref()
            .map(|l| l.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "dbscan",
                operation: "labels_",
            })
    }

    /// Host copy of the fitted `core_sample_indices_` (ascending `i32` indices of
    /// the core points). Errors with [`AlgoError::NotFitted`] before `fit`.
    pub fn core_sample_indices(
        &self,
        pool: &BufferPool<ActiveRuntime>,
    ) -> Result<Vec<i32>, AlgoError> {
        self.core_sample_indices_
            .as_ref()
            .map(|c| c.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "dbscan",
                operation: "core_sample_indices_",
            })
    }

    /// Convenience `fit_predict` (sklearn `ClusterMixin`): fit to `x` then return
    /// the fitted `labels_` as a fresh device-resident `i32` buffer (noise = `-1`).
    pub fn fit_predict(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        self.fit(pool, x, None, shape)?;
        let labels = self.labels(pool)?;
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

impl<F> Fit<F> for DBSCAN<F>
where
    F: Float + CubeElement + Pod,
{
    fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-05-07-01 / ASVS V5: validate the untrusted hyperparameters +
        //     geometry BEFORE the prim launch. (The prim re-validates eps/min_samples
        //     as a ShapeMismatch, but the estimator surfaces the precise typed
        //     AlgoError so the host→estimator boundary contract is explicit.) ---
        if !(self.eps >= 0.0) || !self.eps.is_finite() {
            return Err(AlgoError::InvalidEps {
                estimator: "dbscan",
                eps: self.eps,
            });
        }
        if self.min_samples < 1 {
            return Err(AlgoError::InvalidMinSamples {
                estimator: "dbscan",
                min_samples: self.min_samples,
            });
        }
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }

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

        self.labels_ = Some(labels_dev);
        self.core_sample_indices_ = Some(core_dev);
        Ok(self)
    }
}
