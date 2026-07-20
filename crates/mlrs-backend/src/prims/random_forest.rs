//! `random_forest` — host orchestration for the Random Forest primitive
//! (ENSEMBLE-01): batched level-wise growth of ALL trees at once over the
//! `mlrs-kernels::tree` device kernels, plus forest inference.
//!
//! ## Launch-only fit loop (the `sgd_solve` perf lesson)
//! The fit loop performs ZERO device→host readbacks: after ONE initial `x`
//! readback (host quantile bin edges — the same single-sync concession
//! `kmeanspp_sample` makes) every level is a fixed launch sequence
//! (histogram → cumulative → node stats → split scores → best split →
//! child ranges → partition) with only small host→device uploads (the
//! per-level host-drawn feature samples, D-05: no device RNG). Host syncs are
//! what made earlier mlrs fits latency-bound vs cuML, so the level loop is
//! deliberately free of them.
//!
//! ## Memory
//! The transient per-level histogram/score buffers are chunked over trees to a
//! fixed byte budget ([`RF_HIST_BUDGET_BYTES`]), so deep levels never allocate
//! `O(trees · 2^depth · features · bins)` at once. Persistent model arrays are
//! the complete-tree `n_trees × (2^(max_depth+1) − 1)` node arrays.
//!
//! ## Validate before any unsafe launch (T-05-03-01 / ASVS V5)
//! All geometry and hyperparameter ranges are validated (including every
//! `u32` kernel-index product) BEFORE the first `unsafe` `ArrayArg` is built,
//! surfacing typed [`PrimError`]s, never device OOB.
//!
//! Tests live in `crates/mlrs-backend/tests/random_forest_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)]` module).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use mlrs_kernels::tree::{
    rf_best_split, rf_bin_features_t, rf_bin_features_t_packed, rf_child_ranges,
    rf_count_left_blocks, rf_hist_class_atomic, rf_hist_class_part, rf_hist_cum,
    rf_hist_cum_u32, rf_hist_reduce, rf_hist_reg_part, rf_hist_zero_u32, rf_mean_reg,
    rf_node_stats, rf_order_iota, rf_partition_blocks, rf_predict_leaf, rf_root_ranges,
    rf_split_scores_class, rf_split_scores_reg, rf_vote_class,
};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::rng::SplitMix64;
use crate::runtime::ActiveRuntime;

/// Byte budget for the transient per-level histogram + score + staging
/// buffers. Levels whose full-width buffers would exceed it are processed in
/// tree chunks (extra launches, still zero readbacks). 64 MiB keeps single
/// allocations comfortably under wgpu's default max-buffer-size.
const RF_HIST_BUDGET_BYTES: usize = 64 << 20;

/// Target parallel-unit count for the row-blocked kernels. Shallow levels
/// have far fewer `tree × node × feature` columns than a GPU has resident
/// threads (level 0 of the cuML-comparison ladder is ~128 — a big GPU sits
/// >99% idle), so the row scans are split into up to [`RF_MAX_ROW_BLOCKS`]
/// per-column blocks until the launch reaches this many units. 64Ki covers a
/// T4-class device (40 SMs × 1024 resident threads) with headroom.
const RF_TARGET_UNITS: usize = 128 * 1024;

/// Cap on row blocks per histogram/partition column (bounds the partial-
/// histogram memory multiplier and the per-node prefix loop length).
const RF_MAX_ROW_BLOCKS: usize = 512;

/// Histogram row-block count for one level chunk (see [`RF_TARGET_UNITS`] /
/// [`RF_MAX_ROW_BLOCKS`]): reach the launch target, but never blow the
/// partial-buffer budget and never drop below ~one `nb·ncs` slice of row work
/// per block (the zero+reduce overhead ceiling).
fn hist_blocks(cols: usize, hist_bytes: usize, n: usize, nodes: usize, slice: usize) -> usize {
    RF_TARGET_UNITS
        .div_ceil(cols.max(1))
        .min(RF_HIST_BUDGET_BYTES / hist_bytes.max(1))
        .min((n / nodes.max(1) / slice.max(1)).max(1))
        .clamp(1, RF_MAX_ROW_BLOCKS)
}

/// The cpu backend's MLIR path miscompiles SharedMemory/atomic kernels
/// (spike findings 001–003), so the shared-atomic classifier histogram is a
/// GPU-backend path; cpu keeps the row-blocked gather. Integer atomics make
/// both paths produce bitwise-identical histograms, and each backend's oracle
/// suite runs its own path.
#[cfg(feature = "cpu")]
const RF_ATOMIC_HIST: bool = false;
#[cfg(not(feature = "cpu"))]
const RF_ATOMIC_HIST: bool = true;

/// Shared-memory histogram slot budget of [`rf_hist_class_atomic`] (fixed
/// comptime allocation of 4096 × u32 = 16 KiB per cube).
const RF_ATOMIC_SHARED_SLOTS: usize = 4096;

/// Workgroup (row-block) count per level for the atomic histogram: enough
/// 256-thread cubes to saturate a T4-class device, split over the level's
/// `(tree, node)` columns.
fn atomic_hist_blocks(cols: usize) -> usize {
    512usize.div_ceil(cols.max(1)).clamp(1, 64)
}

/// Partition row-block count for one level (same launch target; the floor is
/// ~16 rows of scan work per block). Shared with
/// `prims::hist_gradient_boosting` (the blocked GBT partition).
pub(crate) fn partition_blocks(units: usize, n: usize, nodes: usize) -> usize {
    RF_TARGET_UNITS
        .div_ceil(units.max(1))
        .min((n / nodes.max(1) / 16).max(1))
        .clamp(1, RF_MAX_ROW_BLOCKS)
}

/// `RF_PROFILE=1` phase profiler: device-syncs at each `lap` and records the
/// wall time since the previous one, dumping a table at the end of the fit.
/// With the env var unset every call is a no-op (ZERO added syncs — the
/// launch-only fit loop stays launch-only; profiling totals are inflated by
/// the per-phase pipeline drains and are for ATTRIBUTION, not headlines).
struct RfProf {
    on: bool,
    t: std::time::Instant,
    rows: Vec<(String, f64)>,
}

impl RfProf {
    fn new() -> Self {
        Self {
            on: std::env::var_os("RF_PROFILE").is_some(),
            t: std::time::Instant::now(),
            rows: Vec::new(),
        }
    }

    fn lap(&mut self, client: &crate::runtime::Client, label: impl Into<String>) {
        if self.on {
            let _ = cubecl::future::block_on(client.sync());
            self.rows.push((label.into(), self.t.elapsed().as_secs_f64()));
            self.t = std::time::Instant::now();
        }
    }

    fn dump(&self) {
        if self.on {
            eprintln!("=== RF_PROFILE (sync per phase; totals inflated by the syncs) ===");
            for (l, s) in &self.rows {
                eprintln!("{l:>22}: {:9.3} ms", s * 1e3);
            }
            let tot: f64 = self.rows.iter().map(|r| r.1).sum();
            eprintln!("{:>22}: {:9.3} ms", "TOTAL", tot * 1e3);
        }
    }
}

/// Hard cap on `max_depth` (complete-tree layout: node arrays grow as
/// `2^(max_depth+1)`; 16 gives 131 071 nodes/tree — the cuML default depth).
pub const RF_MAX_DEPTH_CAP: usize = 16;

/// Random Forest fit hyperparameters (the prim-level, already-resolved form:
/// `max_features` is a concrete count, not the sklearn string policy — the
/// estimator layer resolves `sqrt`/`log2`/fractions before calling in).
#[derive(Debug, Clone, Copy)]
pub struct RfParams {
    /// Number of trees `n_estimators ≥ 1`.
    pub n_trees: usize,
    /// Tree depth `1 ..= RF_MAX_DEPTH_CAP`; every leaf is forced at this depth
    /// at the latest (complete-tree layout).
    pub max_depth: usize,
    /// Histogram bins per feature (`2 ..= 256`); candidate thresholds per
    /// feature = `n_bins − 1`. When a feature has fewer distinct values than
    /// bins, the candidate set equals sklearn's exact midpoint set.
    pub n_bins: usize,
    /// Features sampled per node (`1 ..= n_features`), without replacement.
    pub max_features: usize,
    /// Minimum weighted samples to SPLIT a node (sklearn `min_samples_split`;
    /// bootstrap duplicates count individually, matching sklearn's resampled
    /// view).
    pub min_samples_split: f64,
    /// Minimum weighted samples in each CHILD (sklearn `min_samples_leaf`).
    pub min_samples_leaf: f64,
    /// Draw a with-replacement bootstrap sample per tree (host `SplitMix64`,
    /// seeded, ASVS V6 — never `OsRng`). `false` = every tree sees all rows.
    pub bootstrap: bool,
    /// Seed for the single host RNG stream (bootstrap draws, then per-level
    /// feature subsamples, in a fixed consumption order — reproducible).
    pub seed: u64,
    /// RF-OOB-01: compute `RfFitOutcome::oob_score` at fit time. `false`
    /// (default) performs ZERO extra device/host work beyond the ordinary
    /// fit. `true` re-derives the bootstrap weight mask (a second,
    /// identically-seeded `SplitMix64` + [`bootstrap_weights`] pass) and
    /// scores each training row using ONLY the trees for which it was
    /// out-of-bag (never drawn): accuracy for a classifier, R² for a
    /// regressor. Rows with zero out-of-bag trees are excluded from the
    /// aggregate (sklearn parity) and reported once via `log::warn!`.
    pub oob_score: bool,
}

/// A fitted, device-resident forest (complete-tree layout, `n_trees ×
/// total_nodes` per array). `n_values = n_classes` for a classifier
/// (`leaf_dist` rows are normalized class distributions) or `1` for a
/// regressor (`leaf_dist` is the leaf mean target).
pub struct RfModel<F>
where
    F: Float + CubeElement + Pod,
{
    split_feature: DeviceArray<ActiveRuntime, u32>,
    threshold: DeviceArray<ActiveRuntime, F>,
    is_leaf: DeviceArray<ActiveRuntime, u32>,
    leaf_dist: DeviceArray<ActiveRuntime, F>,
    /// Per-node weighted impurity decrease (RF-IMP-01), `0` on leaves.
    /// Device-resident for parity with the other model arrays; the
    /// Python/algos-facing `feature_importances_` accessor reads the
    /// already-reduced, already-normalized host `Vec<F>` on
    /// [`RfFitOutcome`] instead (computed once at fit time), not a lazy
    /// re-reduction of this field.
    node_decrease: DeviceArray<ActiveRuntime, F>,
    n_trees: usize,
    max_depth: usize,
    total_nodes: usize,
    n_features: usize,
    n_values: usize,
}

impl<F> RfModel<F>
where
    F: Float + CubeElement + Pod,
{
    /// Assemble a device-resident forest from HOST complete-layout arrays
    /// (FIL-01 — the ForestInference import path; the fit path never uses
    /// this). Each array is `n_trees × total_nodes` (× `n_values` for
    /// `leaf_dist`) with `total_nodes = 2^(max_depth+1) − 1`; the traversal
    /// contract is the fitted-forest one (`x < threshold → 2i+1`, bounded
    /// `max_depth` walk, `is_leaf != 0` stops advancement — see
    /// [`mlrs_kernels::rf_predict_leaf`]). `node_decrease` is zero-filled
    /// (impurity decrease is a FIT product; an imported forest has none).
    ///
    /// Geometry is validated host-side: `total_nodes` must equal
    /// `2^(max_depth+1) − 1` and every array length must match. Violations
    /// return [`PrimError::ShapeMismatch`] BEFORE any upload.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        pool: &mut BufferPool<ActiveRuntime>,
        split_feature: &[u32],
        threshold: &[F],
        is_leaf: &[u32],
        leaf_dist: &[F],
        n_trees: usize,
        max_depth: usize,
        n_features: usize,
        n_values: usize,
    ) -> Result<Self, PrimError> {
        let total_nodes = (1usize << (max_depth + 1)) - 1;
        let tn = n_trees.checked_mul(total_nodes).ok_or(PrimError::Overflow {
            operand: "total_nodes",
            lhs: n_trees,
            rhs: total_nodes,
        })?;
        for (operand, len, expect) in [
            ("split_feature", split_feature.len(), tn),
            ("threshold", threshold.len(), tn),
            ("is_leaf", is_leaf.len(), tn),
            ("leaf_dist", leaf_dist.len(), tn * n_values),
        ] {
            if len != expect {
                return Err(PrimError::ShapeMismatch {
                    operand,
                    rows: n_trees,
                    cols: total_nodes,
                    len,
                });
            }
        }
        fits_u32(tn, "total_nodes")?;
        let zeros: Vec<F> = vec![F::new(0.0); tn];
        Ok(Self {
            split_feature: DeviceArray::from_host(pool, split_feature),
            threshold: DeviceArray::from_host(pool, threshold),
            is_leaf: DeviceArray::from_host(pool, is_leaf),
            leaf_dist: DeviceArray::from_host(pool, leaf_dist),
            node_decrease: DeviceArray::from_host(pool, &zeros),
            n_trees,
            max_depth,
            total_nodes,
            n_features,
            n_values,
        })
    }

    /// Number of trees in the forest.
    pub fn n_trees(&self) -> usize {
        self.n_trees
    }

    /// The bounded traversal depth (complete-layout `max_depth`).
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Fitted feature count (predict geometry is validated against it).
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Values per leaf (`n_classes` classifier / `1` regressor).
    pub fn n_values(&self) -> usize {
        self.n_values
    }

    /// Complete-tree node count per tree (`2^(max_depth+1) − 1`).
    pub fn total_nodes(&self) -> usize {
        self.total_nodes
    }

    /// Host copy of the per-node leaf flags (debug/tests).
    pub fn is_leaf_host(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<u32> {
        self.is_leaf.to_host(pool)
    }

    /// Host copy of the per-node raw split feature ids (debug/tests;
    /// `u32::MAX` on leaves).
    pub fn split_feature_host(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<u32> {
        self.split_feature.to_host(pool)
    }

    /// Host copy of the per-node split thresholds (debug/tests).
    pub fn threshold_host(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.threshold.to_host(pool)
    }

    /// Host copy of the per-node leaf values (debug/tests).
    pub fn leaf_dist_host(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.leaf_dist.to_host(pool)
    }

    /// Host copy of the per-node weighted impurity decrease (RF-IMP-01,
    /// debug/tests; `0` on leaves). The normalized `feature_importances_`
    /// vector is computed ONCE at fit time and returned on
    /// [`RfFitOutcome::feature_importances`], not re-derived from this field.
    pub fn node_decrease_host(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.node_decrease.to_host(pool)
    }
}

/// The full output of a Random Forest fit (RF-IMP-01 / RF-OOB-01): the
/// fitted forest, the normalized (sums to `1.0`, all-zero in the degenerate
/// all-leaf-forest case) length-`n_features` `feature_importances_` vector,
/// and the out-of-bag score (`None` unless `params.oob_score == true` —
/// always `None` as of RF-IMP-01; RF-OOB-01/TASK-04 populates `Some`).
pub struct RfFitOutcome<F>
where
    F: Float + CubeElement + Pod,
{
    pub model: RfModel<F>,
    pub feature_importances: Vec<F>,
    pub oob_score: Option<F>,
}

/// Fit a Random Forest CLASSIFIER. `y_idx` are DENSE class indices
/// (`0 .. n_classes`, the estimator layer maps raw labels), length `n`.
pub fn rf_fit_class<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    y_idx: &[u32],
    n_classes: usize,
    params: &RfParams,
) -> Result<RfFitOutcome<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let (n, _d) = shape;
    if y_idx.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "y_idx",
            rows: n,
            cols: 1,
            len: y_idx.len(),
        });
    }
    if n_classes < 2 || n_classes > 1024 {
        return Err(PrimError::ShapeMismatch {
            operand: "n_classes",
            rows: n_classes,
            cols: 1,
            len: n_classes,
        });
    }
    for (i, &c) in y_idx.iter().enumerate() {
        if (c as usize) >= n_classes {
            return Err(PrimError::ShapeMismatch {
                operand: "y_idx",
                rows: i,
                cols: c as usize,
                len: n_classes,
            });
        }
    }
    let y_dev: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(pool, y_idx);
    rf_fit_impl::<F>(pool, x, shape, RfTarget::Class(&y_dev, n_classes), params)
}

/// Fit a Random Forest REGRESSOR. `y` is the length-`n` device target.
pub fn rf_fit_reg<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    y: &DeviceArray<ActiveRuntime, F>,
    params: &RfParams,
) -> Result<RfFitOutcome<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let (n, _d) = shape;
    if y.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: n,
            cols: 1,
            len: y.len(),
        });
    }
    rf_fit_impl::<F>(pool, x, shape, RfTarget::Reg(y), params)
}

/// The two fit targets (classifier: dense class indices + class count;
/// regressor: raw device target).
enum RfTarget<'a, F>
where
    F: Float + CubeElement + Pod,
{
    Class(&'a DeviceArray<ActiveRuntime, u32>, usize),
    Reg(&'a DeviceArray<ActiveRuntime, F>),
}

/// Standard ceiling-division 1D launch config (the `distance.rs` /
/// `kmeans.rs` shape). Shared with `prims::hist_gradient_boosting`.
///
/// Unit counts whose cube count exceeds the per-dimension dispatch limit
/// (65535 on wgpu — e.g. boosted-ensemble traversal at `n_trees × q` units)
/// fold the overflow into the Y dimension: `ABSOLUTE_POS` linearizes over the
/// whole cube grid and every kernel carries an `if tid < total` guard, so the
/// slack cubes are harmless.
pub(crate) fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    const MAX_DIM: u32 = 65_535;
    let block = 256u32;
    let cubes = (((n as u32) + block - 1) / block).max(1);
    let y = cubes.div_ceil(MAX_DIM);
    let x = cubes.div_ceil(y);
    (
        CubeCount::Static(x, y, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}

/// 256-thread workgroup grid for a CUBE-addressed kernel (folds past the
/// per-dimension dispatch limit like [`launch_dims_1d`]; slack cubes are
/// guarded in-kernel).
pub(crate) fn launch_cubes_256(cubes: usize) -> (CubeCount, CubeDim) {
    const MAX_DIM: u32 = 65_535;
    let c = (cubes as u32).max(1);
    let y = c.div_ceil(MAX_DIM);
    let x = c.div_ceil(y);
    (
        CubeCount::Static(x, y, 1),
        CubeDim { x: 256, y: 1, z: 1 },
    )
}

/// WR-03: reject a kernel-index product that does not fit `u32` BEFORE any
/// launch (a truncated index would read/write out of bounds on device).
/// Shared with `prims::hist_gradient_boosting`.
pub(crate) fn fits_u32(value: usize, operand: &'static str) -> Result<u32, PrimError> {
    u32::try_from(value).map_err(|_| PrimError::ShapeMismatch {
        operand,
        rows: value,
        cols: 0,
        len: u32::MAX as usize,
    })
}

/// Validate the shared fit geometry + hyperparameters (ASVS V5, before any
/// allocation or launch).
fn validate_fit(
    x_len: usize,
    n: usize,
    d: usize,
    params: &RfParams,
) -> Result<(), PrimError> {
    if n == 0 || d == 0 || x_len != n * d {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    if params.n_trees == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "n_trees",
            rows: 0,
            cols: 0,
            len: 0,
        });
    }
    if params.max_depth == 0 || params.max_depth > RF_MAX_DEPTH_CAP {
        return Err(PrimError::ShapeMismatch {
            operand: "max_depth",
            rows: params.max_depth,
            cols: 1,
            len: RF_MAX_DEPTH_CAP,
        });
    }
    if params.n_bins < 2 || params.n_bins > 256 {
        return Err(PrimError::ShapeMismatch {
            operand: "n_bins",
            rows: params.n_bins,
            cols: 2,
            len: 256,
        });
    }
    if params.max_features == 0 || params.max_features > d {
        return Err(PrimError::ShapeMismatch {
            operand: "max_features",
            rows: params.max_features,
            cols: 1,
            len: d,
        });
    }
    if !(params.min_samples_split.is_finite() && params.min_samples_split >= 0.0)
        || !(params.min_samples_leaf.is_finite() && params.min_samples_leaf >= 0.0)
    {
        return Err(PrimError::ShapeMismatch {
            operand: "min_samples",
            rows: 0,
            cols: 0,
            len: 0,
        });
    }
    Ok(())
}

/// Host quantile-midpoint bin edges: `d × (n_bins − 1)` ascending, padded
/// past-max so unused candidate slots stay empty. When a feature has
/// `≤ n_bins − 1` candidate midpoints, the set is EXACTLY sklearn's midpoint
/// candidate set (edges fall strictly between adjacent distinct values, so
/// the device `x < edge` rule matches sklearn's `x ≤ midpoint`).
/// Shared with `prims::hist_gradient_boosting` (sklearn's `_BinMapper`
/// midpoint rule is identical for HistGradientBoosting).
pub(crate) fn compute_edges<F>(x_host: &[F], n: usize, d: usize, n_bins: usize) -> Vec<F>
where
    F: Float + CubeElement + Pod,
{
    let n_edges = n_bins - 1;
    let mut edges = vec![0f64; d * n_edges];
    // `F` is `Pod`, so the host slice reinterprets as the concrete float via
    // bytemuck — the workers then borrow a plain `&[f32]`/`&[f64]` (Send +
    // Sync) and each extracts + sorts its OWN feature columns (no serial
    // transpose pass, and the f32 path radix-sorts 4×u8 passes on 32-bit
    // keys). Sorted order and every candidate float op are bit-identical to
    // the previous convert-then-sort (f32→f64 is exact and monotone).
    let bytes: &[u8] = bytemuck::cast_slice(x_host);
    let workers = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .clamp(1, d);
    let per = d.div_ceil(workers);
    if size_of::<F>() == 4 {
        let xf: &[f32] = bytemuck::cast_slice(bytes);
        std::thread::scope(|scope| {
            for (g, edge_group) in edges.chunks_mut(n_edges * per).enumerate() {
                scope.spawn(move || {
                    let mut col32: Vec<f32> = vec![0f32; n];
                    let mut col: Vec<f64> = vec![0f64; n];
                    let mut keys: Vec<u32> = Vec::with_capacity(n);
                    let mut tmp: Vec<u32> = Vec::with_capacity(n);
                    for (jj, out) in edge_group.chunks_mut(n_edges).enumerate() {
                        let j = g * per + jj;
                        for (i, dst) in col32.iter_mut().enumerate() {
                            *dst = xf[i * d + j];
                        }
                        radix_sort_total_f32(&mut col32, &mut keys, &mut tmp);
                        for (dst, &v) in col.iter_mut().zip(col32.iter()) {
                            *dst = v as f64;
                        }
                        edges_from_sorted(&col, n_bins, out);
                    }
                });
            }
        });
    } else {
        let xd: &[f64] = bytemuck::cast_slice(bytes);
        std::thread::scope(|scope| {
            for (g, edge_group) in edges.chunks_mut(n_edges * per).enumerate() {
                scope.spawn(move || {
                    let mut col: Vec<f64> = vec![0f64; n];
                    let mut keys: Vec<u64> = Vec::with_capacity(n);
                    let mut tmp: Vec<u64> = Vec::with_capacity(n);
                    for (jj, out) in edge_group.chunks_mut(n_edges).enumerate() {
                        let j = g * per + jj;
                        for (i, dst) in col.iter_mut().enumerate() {
                            *dst = xd[i * d + j];
                        }
                        radix_sort_total(&mut col, &mut keys, &mut tmp);
                        edges_from_sorted(&col, n_bins, out);
                    }
                });
            }
        });
    }
    edges.into_iter().map(f64_to_host::<F>).collect()
}

/// IEEE-754 totalOrder key for `f32` (the 32-bit analogue of
/// [`f64_total_key`]).
#[inline]
fn f32_total_key(v: f32) -> u32 {
    let b = v.to_bits();
    if b >> 31 == 1 { !b } else { b | (1u32 << 31) }
}

/// Inverse of [`f32_total_key`].
#[inline]
fn f32_from_total_key(k: u32) -> f32 {
    let b = if k >> 31 == 1 { k & !(1u32 << 31) } else { !k };
    f32::from_bits(b)
}

/// 4×8-bit LSB counting radix sort over f32 totalOrder keys (even pass count
/// ⇒ result lands back in `keys`).
fn radix_sort_total_f32(col: &mut [f32], keys: &mut Vec<u32>, tmp: &mut Vec<u32>) {
    let n = col.len();
    keys.clear();
    keys.extend(col.iter().map(|&v| f32_total_key(v)));
    tmp.resize(n, 0);
    for pass in 0..4u32 {
        let shift = pass * 8;
        let mut cnt = [0usize; 256];
        for &k in keys.iter() {
            cnt[((k >> shift) & 255) as usize] += 1;
        }
        if cnt.iter().any(|&c| c == n) {
            continue;
        }
        let mut pos = [0usize; 256];
        let mut acc = 0usize;
        for b in 0..256 {
            pos[b] = acc;
            acc += cnt[b];
        }
        for &k in keys.iter() {
            let b = ((k >> shift) & 255) as usize;
            tmp[pos[b]] = k;
            pos[b] += 1;
        }
        std::mem::swap(keys, tmp);
    }
    for (dst, &k) in col.iter_mut().zip(keys.iter()) {
        *dst = f32_from_total_key(k);
    }
}

/// IEEE-754 totalOrder key for `f64`: a monotone `u64` whose unsigned order
/// equals `f64::total_cmp` order (sign-magnitude flip). Bijective, so the
/// radix-sorted array is BIT-IDENTICAL to a `sort_unstable_by(total_cmp)`
/// result (equal values have equal bits under totalOrder — the sort output
/// is unique regardless of stability).
#[inline]
fn f64_total_key(v: f64) -> u64 {
    let b = v.to_bits();
    if b >> 63 == 1 { !b } else { b | (1u64 << 63) }
}

/// Inverse of [`f64_total_key`].
#[inline]
fn f64_from_total_key(k: u64) -> f64 {
    let b = if k >> 63 == 1 { k & !(1u64 << 63) } else { !k };
    f64::from_bits(b)
}

/// LSB-first 8×8-bit counting radix sort over totalOrder keys — a linear-time
/// replacement for the per-feature comparison sort (the quantile-edge sorts
/// were the single biggest host cost of the fit on the Kaggle T4 profile).
/// Even pass count ⇒ result lands back in `keys`.
fn radix_sort_total(col: &mut [f64], keys: &mut Vec<u64>, tmp: &mut Vec<u64>) {
    let n = col.len();
    keys.clear();
    keys.extend(col.iter().map(|&v| f64_total_key(v)));
    tmp.resize(n, 0);
    for pass in 0..8u32 {
        let shift = pass * 8;
        let mut cnt = [0usize; 256];
        for &k in keys.iter() {
            cnt[((k >> shift) & 255) as usize] += 1;
        }
        // Early-out: all keys share this byte — pass is the identity.
        if cnt.iter().any(|&c| c == n) {
            continue;
        }
        let mut pos = [0usize; 256];
        let mut acc = 0usize;
        for b in 0..256 {
            pos[b] = acc;
            acc += cnt[b];
        }
        for &k in keys.iter() {
            let b = ((k >> shift) & 255) as usize;
            tmp[pos[b]] = k;
            pos[b] += 1;
        }
        std::mem::swap(keys, tmp);
    }
    for (dst, &k) in col.iter_mut().zip(keys.iter()) {
        *dst = f64_from_total_key(k);
    }
}

/// One feature's edge computation over its (unsorted) f64 column: sorts in
/// place (linear-time radix, totalOrder — bit-identical to the old
/// comparison sort), then emits the candidate midpoints — the exact
/// per-feature logic (and float operations) `compute_edges` always used,
/// factored out so the features can run on separate threads.
fn edges_from_sorted(col: &[f64], n_bins: usize, out: &mut [f64]) {
    let n = col.len();
    let n_edges = n_bins - 1;
    let vmax = col[n - 1];
    let pad = if vmax.is_finite() { vmax.abs() + vmax.abs() * 1e-6 + 1.0 } else { 1.0 };

    // Distinct consecutive values (the sklearn candidate midpoints).
    let mut distinct: Vec<f64> = Vec::with_capacity(n.min(n_bins * 4));
    for &v in col.iter() {
        if distinct.last().is_none_or(|&last| v > last) {
            distinct.push(v);
        }
    }

    let mut cand: Vec<f64> = Vec::with_capacity(n_edges);
    if distinct.len().saturating_sub(1) <= n_edges {
        for w in distinct.windows(2) {
            cand.push(0.5 * (w[0] + w[1]));
        }
    } else {
        // Quantile midpoints: only strictly-increasing adjacent pairs so
        // every edge falls strictly between two data values.
        for k in 1..n_bins {
            let i = (k * n) / n_bins;
            if i >= 1 && col[i - 1] < col[i] {
                let m = 0.5 * (col[i - 1] + col[i]);
                if cand.last().is_none_or(|&last| m > last) {
                    cand.push(m);
                }
            }
        }
    }
    cand.resize(n_edges, vmax + pad);
    out.copy_from_slice(&cand);
}

/// Per-tree bootstrap draw counts (`n_trees × n`; `n` with-replacement draws
/// per tree). Each tree runs on its OWN SplitMix64 sub-stream whose seed was
/// drawn SEQUENTIALLY from the master stream (so the whole scheme stays a
/// pure function of the fit seed) — that independence is what lets the trees
/// generate on parallel host threads (the profiled Kaggle T4 run spent 52 ms
/// at 32 trees / 169 ms at 100 trees on the old single-threaded draw loop).
/// `bootstrap = false` never calls this (and consumes NO RNG — the
/// deterministic oracle tier is stream-independent).
fn bootstrap_counts(sub_seeds: &[u64], n: usize) -> Vec<u32> {
    let n_trees = sub_seeds.len();
    let mut counts = vec![0u32; n_trees * n];
    let workers = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .clamp(1, n_trees);
    let per = n_trees.div_ceil(workers);
    std::thread::scope(|scope| {
        for (g, chunk) in counts.chunks_mut(per * n).enumerate() {
            let seeds = &sub_seeds[g * per..];
            scope.spawn(move || {
                for (k, tree_counts) in chunk.chunks_mut(n).enumerate() {
                    let mut r = SplitMix64::new(seeds[k]);
                    for _ in 0..n {
                        tree_counts[r.next_below(n as u64) as usize] += 1;
                    }
                }
            });
        }
    });
    counts
}

/// Expand per-tree draw counts into the per-tree ASCENDING row list the fit
/// consumes as its level-0 `order` (row `i` appears `counts[i]` times;
/// out-of-bag rows are simply absent). The histogram gather then adds `1` per
/// visit — the resulting weighted counts are EXACTLY the old `Σ w` integers,
/// with no weight array, no weight gather, and no dead `w = 0` row scans.
fn expand_order(counts: &[u32], n_trees: usize, n: usize) -> Vec<u32> {
    let mut order = vec![0u32; n_trees * n];
    let workers = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .clamp(1, n_trees);
    let per = n_trees.div_ceil(workers);
    std::thread::scope(|scope| {
        for (g, chunk) in order.chunks_mut(per * n).enumerate() {
            let counts = &counts[g * per * n..];
            scope.spawn(move || {
                for (k, tree_order) in chunk.chunks_mut(n).enumerate() {
                    let mut w = 0usize;
                    for (i, &c) in counts[k * n..(k + 1) * n].iter().enumerate() {
                        for _ in 0..c {
                            tree_order[w] = i as u32;
                            w += 1;
                        }
                    }
                    debug_assert_eq!(w, n, "bootstrap draws must total n");
                }
            });
        }
    });
    order
}

/// All-level feature subsample table on per-level sub-streams (seeds drawn
/// sequentially from the master stream by the caller), generated on parallel
/// host threads. `mf == d` short-circuits to the identity per level and
/// consumes NO sub-seed randomness (the deterministic oracle tier). Returns
/// the concatenated table plus each level's element offset.
fn sample_features_all(
    level_seeds: &[u64],
    n_trees: usize,
    depth: usize,
    mf: usize,
    d: usize,
) -> (Vec<u32>, Vec<usize>) {
    let mut offs = Vec::with_capacity(depth + 1);
    let mut total = 0usize;
    for l in 0..=depth {
        offs.push(total);
        total += n_trees * (1usize << l) * mf;
    }
    let mut table = vec![0u32; total];
    let workers = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .clamp(1, depth + 1);
    std::thread::scope(|scope| {
        let mut rest: &mut [u32] = &mut table;
        let mut spawned = 0usize;
        for l in 0..=depth {
            let (head, tail) = rest.split_at_mut(n_trees * (1usize << l) * mf);
            rest = tail;
            let seed = level_seeds[l];
            scope.spawn(move || {
                let mut r = SplitMix64::new(seed);
                let out = sample_features(&mut r, n_trees, 1usize << l, mf, d);
                head.copy_from_slice(&out);
            });
            spawned += 1;
            // Crude thread cap: join in waves by scope end; the level count is
            // small (≤ 17) so oversubscription beyond `workers` is harmless.
            let _ = (spawned, workers);
        }
    });
    (table, offs)
}

/// Host per-node feature subsample (`n_trees × nodes × mf` raw ids, WITHOUT
/// replacement via partial Fisher–Yates on `SplitMix64::next_below` —
/// unbiased, never `% d`). `mf == d` short-circuits to the identity (no RNG
/// consumed) so full-feature forests are RNG-independent.
fn sample_features(
    rng: &mut SplitMix64,
    n_trees: usize,
    nodes: usize,
    mf: usize,
    d: usize,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(n_trees * nodes * mf);
    if mf == d {
        for _ in 0..n_trees * nodes {
            out.extend(0..d as u32);
        }
        return out;
    }
    let mut scratch: Vec<u32> = (0..d as u32).collect();
    for _ in 0..n_trees * nodes {
        for k in 0..mf {
            let r = k + rng.next_below((d - k) as u64) as usize;
            scratch.swap(k, r);
        }
        out.extend_from_slice(&scratch[..mf]);
        scratch.sort_unstable(); // restore identity for the next node
    }
    out
}

/// The shared launch-only fit driver (classifier / regressor).
fn rf_fit_impl<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    target: RfTarget<'_, F>,
    params: &RfParams,
) -> Result<RfFitOutcome<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let (n, d) = shape;
    validate_fit(x.len(), n, d, params)?;

    let t = params.n_trees;
    let depth = params.max_depth;
    let nb = params.n_bins;
    let mf = params.max_features;
    let total_nodes = (1usize << (depth + 1)) - 1;
    let max_nodes_level = 1usize << depth;
    let (mode_class, ncs, nc_out) = match &target {
        RfTarget::Class(_, nc) => (1u32, *nc, *nc),
        RfTarget::Reg(_) => (0u32, 2usize, 1usize),
    };

    // WR-03: every flat kernel index must fit u32 (validated up front).
    fits_u32(n * d, "n*d")?;
    fits_u32(t * n, "n_trees*n")?;
    fits_u32(t * total_nodes * nc_out, "n_trees*total_nodes*n_values")?;
    fits_u32(t * max_nodes_level * mf * nb * ncs, "level_hist")?;

    let mut prof = RfProf::new();
    let client = pool.client().clone();

    // --- Host RNG sub-seed derivation (sequential, cheap, a pure function of
    // the fit seed): per-tree bootstrap sub-seeds first, then per-level
    // feature sub-seeds. `bootstrap = false` and `mf == d` consume NOTHING,
    // keeping the deterministic oracle tier stream-independent.
    let mut rng = SplitMix64::new(params.seed);
    let boot_seeds: Option<Vec<u64>> = if params.bootstrap {
        Some((0..t).map(|_| rng.next_u64()).collect())
    } else {
        None
    };
    let feat_seeds: Option<Vec<u64>> = if mf < d {
        Some((0..=depth).map(|_| rng.next_u64()).collect())
    } else {
        None
    };
    let total_feat_len: usize = (0..=depth).map(|l| t * (1usize << l) * mf).sum();
    let pregen_feats = total_feat_len * size_of::<u32>() <= RF_HIST_BUDGET_BYTES;

    // --- Worker thread: bootstrap draw expansion + the whole-forest feature
    // table, OVERLAPPED with the main thread's quantile-edge sorts below. ---
    let worker = std::thread::spawn({
        let boot_seeds = boot_seeds.clone();
        let feat_seeds = feat_seeds.clone();
        move || {
            let boot_order: Option<Vec<u32>> = boot_seeds.as_deref().map(|seeds| {
                let counts = bootstrap_counts(seeds, n);
                expand_order(&counts, t, n)
            });
            let feats: Option<(Vec<u32>, Vec<usize>)> = if !pregen_feats {
                None
            } else if let Some(seeds) = &feat_seeds {
                Some(sample_features_all(seeds, t, depth, mf, d))
            } else {
                // mf == d: identity table (no RNG consumed).
                let mut dummy = SplitMix64::new(0);
                let mut offs = Vec::with_capacity(depth + 1);
                let mut table: Vec<u32> = Vec::with_capacity(total_feat_len);
                for l in 0..=depth {
                    offs.push(table.len());
                    table.extend(sample_features(&mut dummy, t, 1usize << l, mf, d));
                }
                Some((table, offs))
            };
            (boot_order, feats)
        }
    });

    // --- ONE host readback: quantile bin edges (host-side, like kmeans++;
    // per-feature sorts thread-parallel, overlapped with the worker). ---
    let x_host = x.to_host(pool);
    let edges_host = compute_edges::<F>(&x_host, n, d, nb);
    drop(x_host);
    let edges_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &edges_host);
    prof.lap(&client, "edges(host)");

    let (boot_order, pregen_table) = worker.join().expect("rf host worker panicked");
    prof.lap(&client, "boot+feat(host,overlap)");

    // --- Bin the features once on device, TRANSPOSED to d × n (feature-
    // major) so the blocked row scans below read consecutive addresses. The
    // classifier PACKS the row's class into the high bits (one load per
    // histogram visit); the partition kernels mask bins with `% 65536`
    // (a no-op on the regressor's unpacked layout). ---
    let binned_handle = pool.acquire(n * d * size_of::<u32>());
    {
        let (count, dim) = launch_dims_1d(n * d);
        match &target {
            RfTarget::Class(y_dev, _) => {
                rf_bin_features_t_packed::launch::<F, ActiveRuntime>(
                    &client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(x.handle().clone(), n * d) },
                    unsafe {
                        ArrayArg::from_raw_parts(edges_dev.handle().clone(), edges_host.len())
                    },
                    unsafe { ArrayArg::from_raw_parts(y_dev.handle().clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(binned_handle.clone(), n * d) },
                    n as u32,
                    d as u32,
                    (nb - 1) as u32,
                );
            }
            RfTarget::Reg(_) => {
                rf_bin_features_t::launch::<F, ActiveRuntime>(
                    &client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(x.handle().clone(), n * d) },
                    unsafe {
                        ArrayArg::from_raw_parts(edges_dev.handle().clone(), edges_host.len())
                    },
                    unsafe { ArrayArg::from_raw_parts(binned_handle.clone(), n * d) },
                    n as u32,
                    d as u32,
                    (nb - 1) as u32,
                );
            }
        }
    }
    let binned_t = DeviceArray::<ActiveRuntime, u32>::from_raw(binned_handle, n * d);

    // --- Row order (identity per tree) + level-0 ranges, ping-pong pairs —
    // both filled ON DEVICE (zero uploads; the ranges buffer beyond the roots
    // is written level-by-level by rf_child_ranges before it is ever read). ---
    let mut order_a: DeviceArray<ActiveRuntime, u32> = match &boot_order {
        // bootstrap: the expanded per-tree draw list (ONE upload).
        Some(order_host) => DeviceArray::from_host(pool, order_host),
        // no bootstrap: identity rows, filled on device (zero uploads).
        None => {
            let order_a_handle = pool.acquire(t * n * size_of::<u32>());
            let (count, dim) = launch_dims_1d(t * n);
            rf_order_iota::launch::<ActiveRuntime>(
                &client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(order_a_handle.clone(), t * n) },
                n as u32,
                t as u32,
            );
            DeviceArray::from_raw(order_a_handle, t * n)
        }
    };
    drop(boot_order);
    let order_b_handle = pool.acquire(t * n * size_of::<u32>());
    let mut order_b = DeviceArray::<ActiveRuntime, u32>::from_raw(order_b_handle, t * n);

    let ranges_len = t * max_nodes_level * 2;
    let ranges_a_handle = pool.acquire(ranges_len * size_of::<u32>());
    {
        let (count, dim) = launch_dims_1d(t);
        rf_root_ranges::launch::<ActiveRuntime>(
            &client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(ranges_a_handle.clone(), ranges_len) },
            n as u32,
            t as u32,
        );
    }
    let mut ranges_a = DeviceArray::<ActiveRuntime, u32>::from_raw(ranges_a_handle, ranges_len);
    let ranges_b_handle = pool.acquire(ranges_len * size_of::<u32>());
    let mut ranges_b = DeviceArray::<ActiveRuntime, u32>::from_raw(ranges_b_handle, ranges_len);

    // --- Whole-forest feature table (worker-generated, ONE upload; the
    // level loop performs no host→device transfers at all). Very deep trees
    // exceed the budget guard and regenerate per level from their sub-seed
    // instead. ---
    let feat_all: Option<(DeviceArray<ActiveRuntime, u32>, Vec<usize>)> =
        match pregen_table {
            Some((host, offs)) => {
                fits_u32(total_feat_len, "feat_ids_all")?;
                Some((DeviceArray::from_host(pool, &host), offs))
            }
            None => None,
        };

    // Shared-atomic classifier histogram path (GPU backends): decided once —
    // eligibility is level-independent (`mf·nb·ncs` is constant).
    let use_atomic = RF_ATOMIC_HIST && mode_class == 1 && mf * nb * ncs <= RF_ATOMIC_SHARED_SLOTS;

    // --- Transient level buffers: acquired ONCE at their maximum per-level
    // sizes and reused across every level/chunk (per-level acquires would
    // miss the exact-size free-list at every level on the first fit). ---
    let mut max_hist = 0usize;
    let mut max_part = 0usize;
    let mut max_scores = 0usize;
    let mut max_stats = 0usize;
    let mut max_blk = 0usize;
    for level in 0..=depth {
        let nodes = 1usize << level;
        let per_tree_bytes = nodes * mf * nb * ncs * size_of::<F>()
            + nodes * mf * (nb - 1) * size_of::<F>()
            + 3 * nodes * size_of::<F>();
        let chunk_t = (RF_HIST_BUDGET_BYTES / per_tree_bytes.max(1)).clamp(1, t);
        // Both chunk widths that can occur at this level (full + tail).
        for tc in [chunk_t.min(t), t % chunk_t] {
            if tc == 0 {
                continue;
            }
            let hist_len = tc * nodes * mf * nb * ncs;
            let bh = hist_blocks(tc * nodes * mf, hist_len * size_of::<F>(), n, nodes, nb * ncs);
            max_hist = max_hist.max(hist_len);
            if bh > 1 && !use_atomic {
                max_part = max_part.max(hist_len * bh);
            }
            max_scores = max_scores.max(tc * nodes * mf * (nb - 1));
            max_stats = max_stats.max(tc * nodes);
        }
        if level < depth {
            let units = t * nodes;
            max_blk = max_blk.max(units * partition_blocks(units, n, nodes));
        }
    }
    fits_u32(max_part.max(max_hist), "level_hist_part")?;
    fits_u32(max_blk, "partition_units")?;
    let hist_h = pool.acquire(max_hist * size_of::<F>());
    let scores_h = pool.acquire(max_scores * size_of::<F>());
    let ntot_h = pool.acquire(max_stats * size_of::<F>());
    let nmax_h = pool.acquire(max_stats * size_of::<F>());
    let nsq_h = pool.acquire(max_stats * size_of::<F>());
    let part_h = if max_part > 0 {
        Some(pool.acquire(max_part * size_of::<F>()))
    } else {
        None
    };
    let histu_h = if use_atomic {
        Some(pool.acquire(max_hist * size_of::<u32>()))
    } else {
        None
    };
    let blk_h = if max_blk > 0 {
        Some(pool.acquire(max_blk * size_of::<u32>()))
    } else {
        None
    };
    prof.lap(&client, "init(bin+order+feat)");

    // --- Persistent model arrays (complete-tree layout). ---
    let split_feature_h = pool.acquire(t * total_nodes * size_of::<u32>());
    let split_bin_h = pool.acquire(t * total_nodes * size_of::<u32>());
    let threshold_h = pool.acquire(t * total_nodes * size_of::<F>());
    let is_leaf_h = pool.acquire(t * total_nodes * size_of::<u32>());
    let leaf_dist_h = pool.acquire(t * total_nodes * nc_out * size_of::<F>());
    let node_decrease_h = pool.acquire(t * total_nodes * size_of::<F>());
    let split_feature = DeviceArray::<ActiveRuntime, u32>::from_raw(split_feature_h, t * total_nodes);
    let split_bin = DeviceArray::<ActiveRuntime, u32>::from_raw(split_bin_h, t * total_nodes);
    let threshold = DeviceArray::<ActiveRuntime, F>::from_raw(threshold_h, t * total_nodes);
    let is_leaf = DeviceArray::<ActiveRuntime, u32>::from_raw(is_leaf_h, t * total_nodes);
    let leaf_dist =
        DeviceArray::<ActiveRuntime, F>::from_raw(leaf_dist_h, t * total_nodes * nc_out);
    let node_decrease =
        DeviceArray::<ActiveRuntime, F>::from_raw(node_decrease_h, t * total_nodes);

    let min_split_f = f64_to_host::<F>(params.min_samples_split);
    let min_leaf_f = f64_to_host::<F>(params.min_samples_leaf);

    // =====================================================================
    // Level loop — LAUNCH-ONLY (no readbacks; one small upload per level).
    // =====================================================================
    for level in 0..=depth {
        let nodes = 1usize << level;
        let level_base = (nodes - 1) as u32;
        let force_leaf = if level == depth { 1u32 } else { 0u32 };

        // Per-node feature subsample: the pre-uploaded whole-forest table
        // (offset per level, zero mid-loop uploads), or the per-level upload
        // fallback for very deep trees.
        let feat_tmp: Option<DeviceArray<ActiveRuntime, u32>> = if feat_all.is_none() {
            let mut level_rng = match &feat_seeds {
                Some(seeds) => SplitMix64::new(seeds[level]),
                None => SplitMix64::new(0), // mf == d: identity, RNG unused
            };
            let feat_host = sample_features(&mut level_rng, t, nodes, mf, d);
            Some(DeviceArray::from_host(pool, &feat_host))
        } else {
            None
        };
        let (feat_handle, feat_arr_len, feat_base) = match (&feat_all, &feat_tmp) {
            (Some((arr, offs)), _) => (arr.handle().clone(), arr.len(), offs[level] as u32),
            (_, Some(arr)) => (arr.handle().clone(), arr.len(), 0u32),
            _ => unreachable!("feat_all xor feat_tmp always set"),
        };

        // Tree chunking to the transient-buffer byte budget (3 stats-shaped
        // buffers now: node_total, node_max, node_sq).
        let per_tree_bytes = nodes * mf * nb * ncs * size_of::<F>()
            + nodes * mf * (nb - 1) * size_of::<F>()
            + 3 * nodes * size_of::<F>();
        let chunk_t = (RF_HIST_BUDGET_BYTES / per_tree_bytes.max(1)).clamp(1, t);

        let mut tree_base = 0usize;
        while tree_base < t {
            let tc = chunk_t.min(t - tree_base);
            let hist_len = tc * nodes * mf * nb * ncs;
            let scores_len = tc * nodes * mf * (nb - 1);
            let stats_len = tc * nodes;

            if use_atomic {
                let histu = histu_h.as_ref().expect("u32 hist hoisted on atomic path");
                // K1a: zero the u32 histogram (atomic flush accumulates).
                {
                    let (zc, zd) = launch_dims_1d(hist_len);
                    rf_hist_zero_u32::launch::<ActiveRuntime>(
                        &client,
                        zc,
                        zd,
                        unsafe { ArrayArg::from_raw_parts(histu.clone(), hist_len) },
                        hist_len as u32,
                    );
                }
                // K1b: shared-memory atomic gather (one 256-thread cube per
                // (tree, node, row block); integer atomics — bitwise-exact).
                {
                    let aw = atomic_hist_blocks(tc * nodes);
                    let (ac, ad) = launch_cubes_256(tc * nodes * aw);
                    rf_hist_class_atomic::launch::<ActiveRuntime>(
                        &client,
                        ac,
                        ad,
                        unsafe { ArrayArg::from_raw_parts(binned_t.handle().clone(), n * d) },
                        unsafe { ArrayArg::from_raw_parts(order_a.handle().clone(), t * n) },
                        unsafe { ArrayArg::from_raw_parts(ranges_a.handle().clone(), ranges_len) },
                        unsafe { ArrayArg::from_raw_parts(feat_handle.clone(), feat_arr_len) },
                        unsafe { ArrayArg::from_raw_parts(histu.clone(), hist_len) },
                        n as u32,
                        mf as u32,
                        nb as u32,
                        ncs as u32,
                        nodes as u32,
                        tc as u32,
                        tree_base as u32,
                        feat_base,
                        aw as u32,
                    );
                }
                // K2′: u32 → F conversion + cumulative sum in one pass.
                {
                    let (cc, cd) = launch_dims_1d(tc * nodes * mf * ncs);
                    rf_hist_cum_u32::launch::<F, ActiveRuntime>(
                        &client,
                        cc,
                        cd,
                        unsafe { ArrayArg::from_raw_parts(histu.clone(), hist_len) },
                        unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
                        mf as u32,
                        nb as u32,
                        ncs as u32,
                        nodes as u32,
                        tc as u32,
                    );
                }
            } else {
                // Row-block count for the histogram gather (bounded by the
                // hoisted partial buffer acquired before the loop).
                let cols = tc * nodes * mf;
                let slice = nb * ncs;
                let mut bh = hist_blocks(cols, hist_len * size_of::<F>(), n, nodes, slice);
                if bh > 1 {
                    bh = bh.min((max_part / hist_len).max(1));
                }

                // K1: row-blocked histogram gather (+ reduce when bh > 1; with
                // bh == 1 the partial layout equals `hist`, so gather lands in
                // `hist` directly and the reduce pass is skipped).
                {
                    let part_len = hist_len * bh;
                    let gather_h = if bh > 1 {
                        part_h
                            .as_ref()
                            .expect("partial buffer hoisted when any bh > 1")
                            .clone()
                    } else {
                        hist_h.clone()
                    };
                    let (count, dim) = launch_dims_1d(cols * bh);
                    match &target {
                        RfTarget::Class(_, _) => {
                            rf_hist_class_part::launch::<F, ActiveRuntime>(
                                &client,
                                count,
                                dim,
                                unsafe { ArrayArg::from_raw_parts(binned_t.handle().clone(), n * d) },
                                unsafe { ArrayArg::from_raw_parts(order_a.handle().clone(), t * n) },
                                unsafe {
                                    ArrayArg::from_raw_parts(ranges_a.handle().clone(), ranges_len)
                                },
                                unsafe {
                                    ArrayArg::from_raw_parts(feat_handle.clone(), feat_arr_len)
                                },
                                unsafe { ArrayArg::from_raw_parts(gather_h.clone(), part_len) },
                                n as u32,
                                mf as u32,
                                nb as u32,
                                ncs as u32,
                                nodes as u32,
                                tc as u32,
                                tree_base as u32,
                                feat_base,
                                bh as u32,
                            );
                        }
                        RfTarget::Reg(y_dev) => {
                            rf_hist_reg_part::launch::<F, ActiveRuntime>(
                                &client,
                                count,
                                dim,
                                unsafe { ArrayArg::from_raw_parts(binned_t.handle().clone(), n * d) },
                                unsafe { ArrayArg::from_raw_parts(y_dev.handle().clone(), n) },
                                unsafe { ArrayArg::from_raw_parts(order_a.handle().clone(), t * n) },
                                unsafe {
                                    ArrayArg::from_raw_parts(ranges_a.handle().clone(), ranges_len)
                                },
                                unsafe {
                                    ArrayArg::from_raw_parts(feat_handle.clone(), feat_arr_len)
                                },
                                unsafe { ArrayArg::from_raw_parts(gather_h.clone(), part_len) },
                                n as u32,
                                mf as u32,
                                nb as u32,
                                nodes as u32,
                                tc as u32,
                                tree_base as u32,
                                feat_base,
                                bh as u32,
                            );
                        }
                    }
                    if bh > 1 {
                        let (rcount, rdim) = launch_dims_1d(hist_len);
                        rf_hist_reduce::launch::<F, ActiveRuntime>(
                            &client,
                            rcount,
                            rdim,
                            unsafe { ArrayArg::from_raw_parts(gather_h.clone(), part_len) },
                            unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
                            slice as u32,
                            cols as u32,
                            bh as u32,
                        );
                    }
                }

                // K2: cumulative histogram over bins.
                {
                    let (count, dim) = launch_dims_1d(tc * nodes * mf * ncs);
                    rf_hist_cum::launch::<F, ActiveRuntime>(
                        &client,
                        count,
                        dim,
                        unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
                        mf as u32,
                        nb as u32,
                        ncs as u32,
                        nodes as u32,
                        tc as u32,
                    );
                }

            }


            // K3 (fused): per-node weighted total + max slot + sum-of-squares
            // in ONE launch. Classifier sums all classes (`nsum = ncs`);
            // regressor reads slot 0 only (`nsum = 1`).
            let nsum = if mode_class == 1 { ncs as u32 } else { 1u32 };
            {
                let (count, dim) = launch_dims_1d(stats_len);
                rf_node_stats::launch::<F, ActiveRuntime>(
                    &client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
                    unsafe { ArrayArg::from_raw_parts(ntot_h.clone(), stats_len) },
                    unsafe { ArrayArg::from_raw_parts(nmax_h.clone(), stats_len) },
                    unsafe { ArrayArg::from_raw_parts(nsq_h.clone(), stats_len) },
                    mf as u32,
                    nb as u32,
                    ncs as u32,
                    nsum,
                    nodes as u32,
                    tc as u32,
                );
            }

            // K5: split scores.
            {
                let (count, dim) = launch_dims_1d(scores_len);
                match &target {
                    RfTarget::Class(_, _) => {
                        rf_split_scores_class::launch::<F, ActiveRuntime>(
                            &client,
                            count,
                            dim,
                            unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
                            unsafe { ArrayArg::from_raw_parts(ntot_h.clone(), stats_len) },
                            unsafe { ArrayArg::from_raw_parts(scores_h.clone(), scores_len) },
                            min_leaf_f,
                            mf as u32,
                            nb as u32,
                            ncs as u32,
                            nodes as u32,
                            tc as u32,
                        );
                    }
                    RfTarget::Reg(_) => {
                        rf_split_scores_reg::launch::<F, ActiveRuntime>(
                            &client,
                            count,
                            dim,
                            unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
                            unsafe { ArrayArg::from_raw_parts(scores_h.clone(), scores_len) },
                            min_leaf_f,
                            mf as u32,
                            nb as u32,
                            nodes as u32,
                            tc as u32,
                        );
                    }
                }
            }

            // K6: per-node best split + leaf finalize (writes model arrays).
            {
                let (count, dim) = launch_dims_1d(stats_len);
                rf_best_split::launch::<F, ActiveRuntime>(
                    &client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
                    unsafe { ArrayArg::from_raw_parts(ntot_h.clone(), stats_len) },
                    unsafe { ArrayArg::from_raw_parts(nmax_h.clone(), stats_len) },
                    unsafe { ArrayArg::from_raw_parts(nsq_h.clone(), stats_len) },
                    unsafe { ArrayArg::from_raw_parts(scores_h.clone(), scores_len) },
                    unsafe { ArrayArg::from_raw_parts(feat_handle.clone(), feat_arr_len) },
                    unsafe {
                        ArrayArg::from_raw_parts(edges_dev.handle().clone(), edges_host.len())
                    },
                    unsafe {
                        ArrayArg::from_raw_parts(split_feature.handle().clone(), t * total_nodes)
                    },
                    unsafe {
                        ArrayArg::from_raw_parts(split_bin.handle().clone(), t * total_nodes)
                    },
                    unsafe {
                        ArrayArg::from_raw_parts(threshold.handle().clone(), t * total_nodes)
                    },
                    unsafe { ArrayArg::from_raw_parts(is_leaf.handle().clone(), t * total_nodes) },
                    unsafe {
                        ArrayArg::from_raw_parts(
                            leaf_dist.handle().clone(),
                            t * total_nodes * nc_out,
                        )
                    },
                    unsafe {
                        ArrayArg::from_raw_parts(node_decrease.handle().clone(), t * total_nodes)
                    },
                    min_split_f,
                    mf as u32,
                    nb as u32,
                    ncs as u32,
                    nc_out as u32,
                    nodes as u32,
                    tc as u32,
                    tree_base as u32,
                    feat_base,
                    level_base,
                    total_nodes as u32,
                    force_leaf,
                    mode_class,
                );
            }

            tree_base += tc;
        }
        prof.lap(&client, format!("L{level}/hist+split"));

        // K7 + K8 + K9: blocked child-range count → per-node prefix + child
        // ranges → blocked STABLE partition. The block count multiplies the
        // former t×nodes unit count (32 at level 0 of the comparison ladder)
        // up to the launch target; contiguous chunks + the per-block prefix
        // keep the scatter bitwise-identical to the old single-unit-per-node
        // partition.
        if level < depth {
            let units = t * nodes;
            let bp = partition_blocks(units, n, nodes);
            let blk_len = units * bp;
            let blk_hh = blk_h
                .as_ref()
                .expect("blk buffer hoisted when depth > 0")
                .clone();

            let (count, dim) = launch_dims_1d(units * bp);
            rf_count_left_blocks::launch::<ActiveRuntime>(
                &client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(binned_t.handle().clone(), n * d) },
                unsafe { ArrayArg::from_raw_parts(order_a.handle().clone(), t * n) },
                unsafe { ArrayArg::from_raw_parts(ranges_a.handle().clone(), ranges_len) },
                unsafe {
                    ArrayArg::from_raw_parts(split_feature.handle().clone(), t * total_nodes)
                },
                unsafe { ArrayArg::from_raw_parts(split_bin.handle().clone(), t * total_nodes) },
                unsafe { ArrayArg::from_raw_parts(is_leaf.handle().clone(), t * total_nodes) },
                unsafe { ArrayArg::from_raw_parts(blk_hh.clone(), blk_len) },
                n as u32,
                nodes as u32,
                t as u32,
                level_base,
                total_nodes as u32,
                bp as u32,
            );
            let (count2, dim2) = launch_dims_1d(units);
            rf_child_ranges::launch::<ActiveRuntime>(
                &client,
                count2,
                dim2,
                unsafe { ArrayArg::from_raw_parts(ranges_a.handle().clone(), ranges_len) },
                unsafe { ArrayArg::from_raw_parts(is_leaf.handle().clone(), t * total_nodes) },
                unsafe { ArrayArg::from_raw_parts(blk_hh.clone(), blk_len) },
                unsafe { ArrayArg::from_raw_parts(ranges_b.handle().clone(), ranges_len) },
                nodes as u32,
                t as u32,
                level_base,
                total_nodes as u32,
                bp as u32,
            );
            let (count3, dim3) = launch_dims_1d(units * bp);
            rf_partition_blocks::launch::<ActiveRuntime>(
                &client,
                count3,
                dim3,
                unsafe { ArrayArg::from_raw_parts(binned_t.handle().clone(), n * d) },
                unsafe { ArrayArg::from_raw_parts(order_a.handle().clone(), t * n) },
                unsafe { ArrayArg::from_raw_parts(ranges_a.handle().clone(), ranges_len) },
                unsafe { ArrayArg::from_raw_parts(ranges_b.handle().clone(), ranges_len) },
                unsafe {
                    ArrayArg::from_raw_parts(split_feature.handle().clone(), t * total_nodes)
                },
                unsafe { ArrayArg::from_raw_parts(split_bin.handle().clone(), t * total_nodes) },
                unsafe { ArrayArg::from_raw_parts(is_leaf.handle().clone(), t * total_nodes) },
                unsafe { ArrayArg::from_raw_parts(blk_hh.clone(), blk_len) },
                unsafe { ArrayArg::from_raw_parts(order_b.handle().clone(), t * n) },
                n as u32,
                nodes as u32,
                t as u32,
                level_base,
                total_nodes as u32,
                bp as u32,
            );
            std::mem::swap(&mut order_a, &mut order_b);
            std::mem::swap(&mut ranges_a, &mut ranges_b);
            prof.lap(&client, format!("L{level}/partition"));
        }

        if let Some(arr) = feat_tmp {
            arr.release_into(pool);
        }
    }

    // Fit-only scratch back to the pool (hoisted transients first).
    pool.release(hist_h, max_hist * size_of::<F>());
    pool.release(scores_h, max_scores * size_of::<F>());
    pool.release(ntot_h, max_stats * size_of::<F>());
    pool.release(nmax_h, max_stats * size_of::<F>());
    pool.release(nsq_h, max_stats * size_of::<F>());
    if let Some(h) = part_h {
        pool.release(h, max_part * size_of::<F>());
    }
    if let Some(h) = histu_h {
        pool.release(h, max_hist * size_of::<u32>());
    }
    if let Some(h) = blk_h {
        pool.release(h, max_blk * size_of::<u32>());
    }
    if let Some((arr, _)) = feat_all {
        arr.release_into(pool);
    }
    binned_t.release_into(pool);
    order_a.release_into(pool);
    order_b.release_into(pool);
    ranges_a.release_into(pool);
    ranges_b.release_into(pool);
    edges_dev.release_into(pool);
    split_bin.release_into(pool);

    // The fitted forest, assembled now (not deferred to the final `Ok(...)`)
    // so both the RF-IMP-01 reduction below and the RF-OOB-01 block can read
    // it through the same `RfModel` accessors `rf_predict_leaf`'s existing
    // callers already use (`predict_leaves`, `rf_predict_proba`,
    // `rf_predict_reg`) — reusing the predict-path traversal, not
    // reimplementing it.
    let model = RfModel {
        split_feature,
        threshold,
        is_leaf,
        leaf_dist,
        node_decrease,
        n_trees: t,
        max_depth: depth,
        total_nodes,
        n_features: d,
        n_values: nc_out,
    };

    // RF-IMP-01: ONE host reduction to feature_importances_, after the
    // level loop completes (the same "readback after launch-only compute"
    // pattern already used above for the quantile bin edges — not a
    // per-iteration host-sync regression).
    //
    // sklearn's `RandomForest.feature_importances_` (sklearn.ensemble
    // `_forest.py`) normalizes EACH tree's weighted-impurity-decrease vector
    // to sum 1 individually, then averages those per-tree vectors over the
    // trees that actually split, then renormalizes the mean to sum 1:
    //   mean_t( d_{t,f} / S_t )  (over trees with S_t > 0), then / its own sum.
    // This is NOT the same as one global normalization `Σ_t d_{t,f} / Σ_t S_t`
    // whenever the per-tree totals `S_t` differ — which they do under
    // `bootstrap=true` (the DEFAULT), where each tree sees a different
    // resample. We replicate sklearn's per-tree scheme exactly so the default
    // config matches within the oracle band, not only the deterministic tier
    // (where all trees are bit-identical and the two schemes coincide).
    let split_feature_host_imp = model.split_feature_host(pool);
    let is_leaf_host_imp = model.is_leaf_host(pool);
    let node_decrease_host_imp = model.node_decrease_host(pool);
    let mut imp = vec![0f64; d];
    let mut n_contributing = 0usize;
    for tr in 0..t {
        let base = tr * total_nodes;
        let mut imp_t = vec![0f64; d];
        for node in 0..total_nodes {
            let i = base + node;
            if is_leaf_host_imp[i] == 0 {
                let f = split_feature_host_imp[i] as usize;
                imp_t[f] += host_to_f64(node_decrease_host_imp[i]);
            }
        }
        let s_t: f64 = imp_t.iter().sum();
        if s_t > 0.0 {
            for (acc, v) in imp.iter_mut().zip(imp_t.iter()) {
                *acc += *v / s_t;
            }
            n_contributing += 1;
        }
    }
    if n_contributing > 0 {
        // Mean over the split-bearing trees, then renormalize. The mean of
        // unit-sum vectors already sums to 1; the trailing divide mirrors
        // sklearn's own `all_importances / np.sum(all_importances)` for
        // exactness under float rounding.
        for v in imp.iter_mut() {
            *v /= n_contributing as f64;
        }
        let imp_sum: f64 = imp.iter().sum();
        if imp_sum > 0.0 {
            for v in imp.iter_mut() {
                *v /= imp_sum;
            }
        }
    }
    // else: every tree is a single leaf (degenerate) — leave the all-zero
    // vector, matching sklearn's zeros return, never a divide-by-zero.
    let feature_importances: Vec<F> = imp.into_iter().map(f64_to_host::<F>).collect();
    prof.lap(&client, "importances(host)");
    prof.dump();

    // RF-OOB-01: gated OOB aggregation. `false` (default) is a no-op — zero
    // extra device/host work beyond the ordinary fit.
    let oob_score: Option<F> = if params.oob_score {
        // Rederive the bootstrap mask on a FRESH, identically-seeded stream
        // (not persisted from the level loop): `bootstrap_weights` is
        // documented as the FIRST draw on the seeded stream, so this
        // reproduces `w_host` byte-for-byte at the cost of one cheap,
        // host-only, no-device-sync pass — cheaper than retaining an extra
        // `t·n·sizeof(F)` device buffer for the common `oob_score=false`
        // case.
        let counts2: Option<Vec<u32>> = if params.bootstrap {
            let mut oob_rng = SplitMix64::new(params.seed);
            let seeds: Vec<u64> = (0..t).map(|_| oob_rng.next_u64()).collect();
            Some(bootstrap_counts(&seeds, n))
        } else {
            // bootstrap = false: no row is ever out-of-bag.
            None
        };

        // Reuse the existing predict-path leaf-traversal kernel against the
        // just-built model and the TRAINING `x` (still in scope — only
        // `x_host` was dropped above).
        let leaf_dev = predict_leaves(pool, &model, x, n);
        let leaf_host = leaf_dev.to_host(pool);
        leaf_dev.release_into(pool);
        let leaf_dist_host = model.leaf_dist_host(pool);
        let nc = model.n_values;

        let mut zero_oob_count = 0usize;
        let score: f64 = match &target {
            RfTarget::Class(y_dev, _n_classes) => {
                let y_host = y_dev.to_host(pool);
                let mut correct = 0usize;
                let mut total = 0usize;
                for i in 0..n {
                    let mut acc = vec![0f64; nc];
                    let mut cnt = 0usize;
                    for tt in 0..t {
                        if counts2.as_ref().is_some_and(|c| c[tt * n + i] == 0) {
                            let lf = leaf_host[tt * n + i] as usize;
                            for (c, slot) in acc.iter_mut().enumerate() {
                                *slot += host_to_f64(leaf_dist_host[(tt * total_nodes + lf) * nc + c]);
                            }
                            cnt += 1;
                        }
                    }
                    if cnt == 0 {
                        zero_oob_count += 1;
                        continue;
                    }
                    let mut best_c = 0usize;
                    let mut best_v = acc[0];
                    for (c, &v) in acc.iter().enumerate().skip(1) {
                        if v > best_v {
                            best_v = v;
                            best_c = c;
                        }
                    }
                    total += 1;
                    if best_c as u32 == y_host[i] {
                        correct += 1;
                    }
                }
                if total == 0 {
                    0.0
                } else {
                    correct as f64 / total as f64
                }
            }
            RfTarget::Reg(y_dev) => {
                let y_host = y_dev.to_host(pool);
                let mut preds: Vec<f64> = Vec::with_capacity(n);
                let mut truths: Vec<f64> = Vec::with_capacity(n);
                for i in 0..n {
                    let mut acc = 0f64;
                    let mut cnt = 0usize;
                    for tt in 0..t {
                        if counts2.as_ref().is_some_and(|c| c[tt * n + i] == 0) {
                            let lf = leaf_host[tt * n + i] as usize;
                            acc += host_to_f64(leaf_dist_host[tt * total_nodes + lf]);
                            cnt += 1;
                        }
                    }
                    if cnt == 0 {
                        zero_oob_count += 1;
                        continue;
                    }
                    preds.push(acc / cnt as f64);
                    truths.push(host_to_f64(y_host[i]));
                }
                if truths.is_empty() {
                    0.0
                } else {
                    let mean_t: f64 = truths.iter().sum::<f64>() / truths.len() as f64;
                    let ss_res: f64 =
                        preds.iter().zip(truths.iter()).map(|(p, tv)| (tv - p).powi(2)).sum();
                    let ss_tot: f64 = truths.iter().map(|tv| (tv - mean_t).powi(2)).sum();
                    if ss_tot > 0.0 {
                        1.0 - ss_res / ss_tot
                    } else {
                        0.0
                    }
                }
            }
        };

        if zero_oob_count > 0 {
            log::warn!(
                "random_forest: {zero_oob_count} training row(s) had zero out-of-bag trees \
                 and were excluded from oob_score_ (increase n_estimators or bootstrap variance)"
            );
        }
        Some(f64_to_host::<F>(score))
    } else {
        None
    };

    Ok(RfFitOutcome {
        model,
        feature_importances,
        oob_score,
    })
}

/// Validate the shared predict geometry.
fn validate_predict<F>(
    model: &RfModel<F>,
    xq_len: usize,
    q: usize,
    d: usize,
) -> Result<(), PrimError>
where
    F: Float + CubeElement + Pod,
{
    if q == 0 || d == 0 || xq_len != q * d {
        return Err(PrimError::ShapeMismatch {
            operand: "xq",
            rows: q,
            cols: d,
            len: xq_len,
        });
    }
    if d != model.n_features {
        return Err(PrimError::DimMismatch {
            dim: "n_features",
            lhs: model.n_features,
            rhs: d,
        });
    }
    fits_u32(model.n_trees * q, "n_trees*q")?;
    Ok(())
}

/// Traverse the forest for `xq` and return the reached leaf id per
/// `(tree, row)` as a device buffer (shared by both predict paths).
fn predict_leaves<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    model: &RfModel<F>,
    xq: &DeviceArray<ActiveRuntime, F>,
    q: usize,
) -> DeviceArray<ActiveRuntime, u32>
where
    F: Float + CubeElement + Pod,
{
    let client = pool.client().clone();
    let t = model.n_trees;
    let d = model.n_features;
    let leaf_h = pool.acquire(t * q * size_of::<u32>());
    let (count, dim) = launch_dims_1d(t * q);
    rf_predict_leaf::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(xq.handle().clone(), q * d) },
        unsafe {
            ArrayArg::from_raw_parts(model.split_feature.handle().clone(), t * model.total_nodes)
        },
        unsafe {
            ArrayArg::from_raw_parts(model.threshold.handle().clone(), t * model.total_nodes)
        },
        unsafe { ArrayArg::from_raw_parts(model.is_leaf.handle().clone(), t * model.total_nodes) },
        unsafe { ArrayArg::from_raw_parts(leaf_h.clone(), t * q) },
        q as u32,
        d as u32,
        model.max_depth as u32,
        t as u32,
        model.total_nodes as u32,
    );
    DeviceArray::from_raw(leaf_h, t * q)
}

/// Classifier inference: `q × n_classes` device probabilities (mean of the
/// reached leaves' class distributions — the sklearn `predict_proba` form).
pub fn rf_predict_proba<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    model: &RfModel<F>,
    xq: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let (q, d) = shape;
    validate_predict(model, xq.len(), q, d)?;
    if model.n_values < 2 {
        return Err(PrimError::DimMismatch {
            dim: "n_values",
            lhs: model.n_values,
            rhs: 2,
        });
    }
    let leaf = predict_leaves(pool, model, xq, q);
    let client = pool.client().clone();
    let t = model.n_trees;
    let nc = model.n_values;
    let proba_h = pool.acquire(q * nc * size_of::<F>());
    let (count, dim) = launch_dims_1d(q);
    rf_vote_class::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(leaf.handle().clone(), t * q) },
        unsafe {
            ArrayArg::from_raw_parts(
                model.leaf_dist.handle().clone(),
                t * model.total_nodes * nc,
            )
        },
        unsafe { ArrayArg::from_raw_parts(proba_h.clone(), q * nc) },
        q as u32,
        nc as u32,
        t as u32,
        model.total_nodes as u32,
    );
    leaf.release_into(pool);
    Ok(DeviceArray::from_raw(proba_h, q * nc))
}

/// Regressor inference: length-`q` device predictions (forest mean of the
/// reached leaves' stored mean targets).
pub fn rf_predict_reg<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    model: &RfModel<F>,
    xq: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let (q, d) = shape;
    validate_predict(model, xq.len(), q, d)?;
    if model.n_values != 1 {
        return Err(PrimError::DimMismatch {
            dim: "n_values",
            lhs: model.n_values,
            rhs: 1,
        });
    }
    let leaf = predict_leaves(pool, model, xq, q);
    let client = pool.client().clone();
    let t = model.n_trees;
    let out_h = pool.acquire(q * size_of::<F>());
    let (count, dim) = launch_dims_1d(q);
    rf_mean_reg::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(leaf.handle().clone(), t * q) },
        unsafe {
            ArrayArg::from_raw_parts(model.leaf_dist.handle().clone(), t * model.total_nodes)
        },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), q) },
        q as u32,
        t as u32,
        model.total_nodes as u32,
    );
    leaf.release_into(pool);
    Ok(DeviceArray::from_raw(out_h, q))
}
