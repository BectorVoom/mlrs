//! `hist_gradient_boosting` — host orchestration for the HistGradientBoosting
//! primitive (GBT-01): sequential gradient boosting whose per-iteration tree
//! growth is the batched level-wise histogram pipeline over the
//! `mlrs-kernels::gbt` device kernels (the `random_forest` chassis), plus
//! ensemble inference.
//!
//! ## Launch-only fit loop (the `sgd_solve` / `random_forest` perf lesson)
//! The whole boosting loop performs ZERO device→host readbacks: after ONE
//! initial `x` readback (host quantile bin edges — the same single-sync
//! concession the forest makes) every iteration is a fixed launch sequence
//! (gradients → per-level histogram/split/partition → raw update). The
//! per-iteration partition reset and the raw-prediction baseline are device
//! kernels, so there are no per-iteration uploads either. With no bootstrap,
//! no feature subsampling and no RNG, fits are bit-deterministic across runs.
//!
//! ## Trees per iteration
//! `K = 1` for regression and binary log-loss, `K = n_classes` for multiclass
//! (sklearn `n_trees_per_iteration_`); the `K` class trees of one iteration
//! grow SIMULTANEOUSLY (they share the row scan the way forest trees do).
//!
//! ## Memory
//! Transient per-level histogram/score buffers are chunked over NODES to a
//! fixed byte budget ([`HGB_HIST_BUDGET_BYTES`]), and the histogram gather is
//! ROW-BLOCKED ([`gbt_hist`] + [`gbt_hist_reduce`]) so shallow levels still
//! saturate the device (the RF root-level under-parallelism lesson).
//! Persistent model arrays are complete-tree `(max_iter · K) × total_nodes`.
//!
//! ## Validate before any unsafe launch (T-05-03-01 / ASVS V5)
//! All geometry and hyperparameter ranges are validated (including every
//! `u32` kernel-index product) BEFORE the first `unsafe` `ArrayArg` is built,
//! surfacing typed [`PrimError`]s, never device OOB.
//!
//! Tests live in `crates/mlrs-backend/tests/hist_gradient_boosting_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)]` module).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use mlrs_kernels::gbt::{
    gbt_best_split, gbt_count_left, gbt_grad_binary, gbt_grad_multi, gbt_grad_reg, gbt_hist,
    gbt_hist_reduce, gbt_init_partition, gbt_init_raw, gbt_partition, gbt_proba_binary,
    gbt_proba_multi, gbt_row_max, gbt_row_sumexp, gbt_split_scores, gbt_sum_raw, gbt_update_raw,
};
use mlrs_kernels::tree::{rf_bin_features, rf_hist_cum, rf_predict_leaf};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::random_forest::{compute_edges, fits_u32, launch_dims_1d, RF_MAX_DEPTH_CAP};
use crate::runtime::ActiveRuntime;

/// Byte budget for the transient per-level histogram (+ partials + scores)
/// buffers; deep levels are processed in NODE chunks under it (the
/// `RF_HIST_BUDGET_BYTES` discipline, extra launches but zero readbacks).
const HGB_HIST_BUDGET_BYTES: usize = 64 << 20;

/// Row-block unit target for the histogram gather: blocks per (tree, node,
/// feature) are chosen so a level launches roughly this many units, keeping
/// shallow levels parallel (the RF root-level lesson). Deliberately modest:
/// every extra block re-pays the per-unit `nb × 3` histogram zeroing and one
/// more reduce-pass read of the whole histogram lattice, so past a few
/// thousand units the management work dominates the row scan (measured: a
/// 32k-unit / 256-block variant was ~1.7× SLOWER end-to-end at n_bins=255).
const HGB_HIST_TARGET_UNITS: usize = 4096;

/// Hard cap on histogram row blocks (bounds the partial-histogram memory and
/// the `blk * len` u32 index product).
const HGB_MAX_BLOCKS: usize = 64;

/// sklearn `TreeGrower` internal `min_hessian_to_split` (not exposed on the
/// estimator surface there either).
const HGB_MIN_HESSIAN_TO_SPLIT: f64 = 1e-3;

/// HistGradientBoosting fit hyperparameters (prim-level, already resolved).
#[derive(Debug, Clone, Copy)]
pub struct HgbParams {
    /// Boosting iterations `max_iter ≥ 1` (sklearn `max_iter`; trees total =
    /// `max_iter · K`).
    pub max_iter: usize,
    /// Tree depth `1 ..= RF_MAX_DEPTH_CAP` (complete-tree layout; the mlrs
    /// level-wise deviation from sklearn's leaf-wise `max_leaf_nodes`).
    pub max_depth: usize,
    /// Histogram bins per feature (`2 ..= 256`; sklearn `max_bins = 255`).
    pub n_bins: usize,
    /// Shrinkage `learning_rate > 0` folded into stored leaf values.
    pub learning_rate: f64,
    /// L2 penalty `λ ≥ 0` on leaf values (sklearn `l2_regularization`).
    pub l2_regularization: f64,
    /// Minimum SAMPLES per child (`≥ 1`, sklearn `min_samples_leaf`, a count —
    /// not the forest's weighted form).
    pub min_samples_leaf: usize,
}

/// A fitted, device-resident boosted ensemble (complete-tree layout,
/// `(max_iter · K) × total_nodes` per array). `n_classes = 1` marks a
/// regressor; a binary classifier has `n_classes = 2` with `K = 1` raw score
/// column; multiclass has `K = n_classes`.
pub struct HgbModel<F>
where
    F: Float + CubeElement + Pod,
{
    split_feature: DeviceArray<ActiveRuntime, u32>,
    threshold: DeviceArray<ActiveRuntime, F>,
    is_leaf: DeviceArray<ActiveRuntime, u32>,
    leaf_value: DeviceArray<ActiveRuntime, F>,
    baseline: DeviceArray<ActiveRuntime, F>,
    n_iters: usize,
    k: usize,
    n_classes: usize,
    max_depth: usize,
    total_nodes: usize,
    n_features: usize,
}

impl<F> HgbModel<F>
where
    F: Float + CubeElement + Pod,
{
    /// Boosting iterations fitted.
    pub fn n_iters(&self) -> usize {
        self.n_iters
    }

    /// Raw-score columns (`1` regression/binary, `n_classes` multiclass).
    pub fn k(&self) -> usize {
        self.k
    }

    /// `1` for a regressor, the class count for a classifier.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Fitted feature count (predict geometry is validated against it).
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Complete-tree node count per tree (`2^(max_depth+1) − 1`).
    pub fn total_nodes(&self) -> usize {
        self.total_nodes
    }

    /// Host copy of the per-node leaf flags (debug/tests).
    pub fn is_leaf_host(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<u32> {
        self.is_leaf.to_host(pool)
    }

    /// Host copy of the per-node split feature ids (debug/tests; `u32::MAX`
    /// on leaves).
    pub fn split_feature_host(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<u32> {
        self.split_feature.to_host(pool)
    }

    /// Host copy of the per-node split thresholds (debug/tests).
    pub fn threshold_host(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.threshold.to_host(pool)
    }

    /// Host copy of the per-node SHRUNK leaf values (debug/tests).
    pub fn leaf_value_host(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.leaf_value.to_host(pool)
    }

    /// Host copy of the per-class baseline raw scores (debug/tests).
    pub fn baseline_host(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.baseline.to_host(pool)
    }
}

/// Fit a HistGradientBoosting REGRESSOR (squared error). `y` is the length-`n`
/// device target; the baseline is its mean (sklearn `fit_intercept_only`).
pub fn hgb_fit_reg<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    y: &DeviceArray<ActiveRuntime, F>,
    params: &HgbParams,
) -> Result<HgbModel<F>, PrimError>
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
    // ONE y readback for the host baseline (mean) — fit-time only.
    let y_host = y.to_host(pool);
    let mean = y_host.iter().map(|&v| host_to_f64(v)).sum::<f64>() / n as f64;
    drop(y_host);
    hgb_fit_impl::<F>(pool, x, shape, HgbTarget::Reg(y), &[mean], 1, params)
}

/// Fit a HistGradientBoosting CLASSIFIER. `y_idx` are DENSE class indices
/// (`0 .. n_classes`, the estimator layer maps raw labels), length `n`.
/// Binary (`n_classes = 2`) uses one sigmoid raw-score column with the
/// log-odds baseline; multiclass uses `n_classes` softmax columns with the
/// mean-centered log-prior baseline (both are sklearn `fit_intercept_only`).
pub fn hgb_fit_class<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    y_idx: &[u32],
    n_classes: usize,
    params: &HgbParams,
) -> Result<HgbModel<F>, PrimError>
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
    if !(2..=1024).contains(&n_classes) {
        return Err(PrimError::ShapeMismatch {
            operand: "n_classes",
            rows: n_classes,
            cols: 2,
            len: 1024,
        });
    }
    let mut counts = vec![0usize; n_classes];
    for (i, &c) in y_idx.iter().enumerate() {
        if (c as usize) >= n_classes {
            return Err(PrimError::ShapeMismatch {
                operand: "y_idx",
                rows: i,
                cols: c as usize,
                len: n_classes,
            });
        }
        counts[c as usize] += 1;
    }
    for (c, &cnt) in counts.iter().enumerate() {
        if cnt == 0 {
            // A dense class index space must be fully populated: an absent
            // class makes the log-prior baseline −inf.
            return Err(PrimError::ShapeMismatch {
                operand: "n_classes",
                rows: c,
                cols: 0,
                len: n_classes,
            });
        }
    }

    if n_classes == 2 {
        // Binary log-loss: one column, baseline = logit of the positive rate.
        let y_f: Vec<F> = y_idx.iter().map(|&c| f64_to_host::<F>(c as f64)).collect();
        let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y_f);
        let p = counts[1] as f64 / n as f64;
        let baseline = (p / (1.0 - p)).ln();
        let model = hgb_fit_impl::<F>(
            pool,
            x,
            shape,
            HgbTarget::Binary(&y_dev),
            &[baseline],
            2,
            params,
        )?;
        y_dev.release_into(pool);
        Ok(model)
    } else {
        // Multiclass log-loss: mean-centered log priors (the sklearn
        // `HalfMultinomialLoss.fit_intercept_only` symmetric parameterization).
        let y_dev: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(pool, y_idx);
        let logs: Vec<f64> = counts
            .iter()
            .map(|&cnt| (cnt as f64 / n as f64).ln())
            .collect();
        let mean_log = logs.iter().sum::<f64>() / n_classes as f64;
        let baseline: Vec<f64> = logs.iter().map(|&l| l - mean_log).collect();
        let model = hgb_fit_impl::<F>(
            pool,
            x,
            shape,
            HgbTarget::Multi(&y_dev),
            &baseline,
            n_classes,
            params,
        )?;
        y_dev.release_into(pool);
        Ok(model)
    }
}

/// The three loss targets (device arrays owned by the caller).
enum HgbTarget<'a, F>
where
    F: Float + CubeElement + Pod,
{
    Reg(&'a DeviceArray<ActiveRuntime, F>),
    Binary(&'a DeviceArray<ActiveRuntime, F>),
    Multi(&'a DeviceArray<ActiveRuntime, u32>),
}

/// Validate the shared fit geometry + hyperparameters (ASVS V5, before any
/// allocation or launch).
fn validate_fit(x_len: usize, n: usize, d: usize, params: &HgbParams) -> Result<(), PrimError> {
    if n == 0 || d == 0 || x_len != n * d {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x_len,
        });
    }
    if params.max_iter == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "max_iter",
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
    if !(params.learning_rate.is_finite() && params.learning_rate > 0.0) {
        return Err(PrimError::ShapeMismatch {
            operand: "learning_rate",
            rows: 0,
            cols: 0,
            len: 0,
        });
    }
    if !(params.l2_regularization.is_finite() && params.l2_regularization >= 0.0) {
        return Err(PrimError::ShapeMismatch {
            operand: "l2_regularization",
            rows: 0,
            cols: 0,
            len: 0,
        });
    }
    if params.min_samples_leaf == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "min_samples_leaf",
            rows: 0,
            cols: 1,
            len: 0,
        });
    }
    Ok(())
}

/// The shared launch-only fit driver. `baseline` has `K` entries.
fn hgb_fit_impl<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    target: HgbTarget<'_, F>,
    baseline: &[f64],
    n_classes: usize,
    params: &HgbParams,
) -> Result<HgbModel<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let (n, d) = shape;
    validate_fit(x.len(), n, d, params)?;

    let k = baseline.len();
    let iters = params.max_iter;
    let depth = params.max_depth;
    let nb = params.n_bins;
    let total_nodes = (1usize << (depth + 1)) - 1;
    let max_nodes_level = 1usize << depth;
    let n_trees = iters * k;

    // WR-03: every flat kernel index must fit u32 (validated up front).
    fits_u32(n * d, "n*d")?;
    fits_u32(n * k, "n*k")?;
    fits_u32(n * HGB_MAX_BLOCKS, "n*blocks")?;
    fits_u32(n_trees * total_nodes, "n_trees*total_nodes")?;
    fits_u32(k * max_nodes_level * d * nb * 3, "level_hist")?;

    // --- ONE host readback: quantile bin edges (the forest concession). ---
    let x_host = x.to_host(pool);
    let edges_host = compute_edges::<F>(&x_host, n, d, nb);
    drop(x_host);
    let edges_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &edges_host);

    let client = pool.client().clone();

    // --- Bin the features once on device (n × d u32). ---
    let binned_handle = pool.acquire(n * d * size_of::<u32>());
    {
        let (count, dim) = launch_dims_1d(n * d);
        rf_bin_features::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(x.handle().clone(), n * d) },
            unsafe { ArrayArg::from_raw_parts(edges_dev.handle().clone(), edges_host.len()) },
            unsafe { ArrayArg::from_raw_parts(binned_handle.clone(), n * d) },
            n as u32,
            d as u32,
            (nb - 1) as u32,
        );
    }
    let binned = DeviceArray::<ActiveRuntime, u32>::from_raw(binned_handle, n * d);

    // --- Baseline + raw predictions (n × k), device-initialized. ---
    let baseline_f: Vec<F> = baseline.iter().map(|&b| f64_to_host::<F>(b)).collect();
    let baseline_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &baseline_f);
    let raw_h = pool.acquire(n * k * size_of::<F>());
    {
        let (count, dim) = launch_dims_1d(n * k);
        gbt_init_raw::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(baseline_dev.handle().clone(), k) },
            unsafe { ArrayArg::from_raw_parts(raw_h.clone(), n * k) },
            n as u32,
            k as u32,
        );
    }
    let raw = DeviceArray::<ActiveRuntime, F>::from_raw(raw_h, n * k);

    // --- Per-sample gradient/hessian buffers + softmax staging (multi). ---
    let g_h = pool.acquire(n * k * size_of::<F>());
    let h_h = pool.acquire(n * k * size_of::<F>());
    let g_dev = DeviceArray::<ActiveRuntime, F>::from_raw(g_h, n * k);
    let h_dev = DeviceArray::<ActiveRuntime, F>::from_raw(h_h, n * k);
    let softmax_stage = if matches!(target, HgbTarget::Multi(_)) {
        let mx = pool.acquire(n * size_of::<F>());
        let se = pool.acquire(n * size_of::<F>());
        Some((
            DeviceArray::<ActiveRuntime, F>::from_raw(mx, n),
            DeviceArray::<ActiveRuntime, F>::from_raw(se, n),
        ))
    } else {
        None
    };

    // --- Row order + ranges ping-pong (k trees wide, reset per iteration). ---
    let order_a_h = pool.acquire(k * n * size_of::<u32>());
    let order_b_h = pool.acquire(k * n * size_of::<u32>());
    let mut order_a = DeviceArray::<ActiveRuntime, u32>::from_raw(order_a_h, k * n);
    let mut order_b = DeviceArray::<ActiveRuntime, u32>::from_raw(order_b_h, k * n);
    let ranges_len = k * max_nodes_level * 2;
    let ranges_a_h = pool.acquire(ranges_len * size_of::<u32>());
    let ranges_b_h = pool.acquire(ranges_len * size_of::<u32>());
    let mut ranges_a = DeviceArray::<ActiveRuntime, u32>::from_raw(ranges_a_h, ranges_len);
    let mut ranges_b = DeviceArray::<ActiveRuntime, u32>::from_raw(ranges_b_h, ranges_len);

    // --- Persistent model arrays (complete-tree layout, all stages). ---
    let split_feature_h = pool.acquire(n_trees * total_nodes * size_of::<u32>());
    let split_bin_h = pool.acquire(n_trees * total_nodes * size_of::<u32>());
    let threshold_h = pool.acquire(n_trees * total_nodes * size_of::<F>());
    let is_leaf_h = pool.acquire(n_trees * total_nodes * size_of::<u32>());
    let leaf_value_h = pool.acquire(n_trees * total_nodes * size_of::<F>());
    let split_feature =
        DeviceArray::<ActiveRuntime, u32>::from_raw(split_feature_h, n_trees * total_nodes);
    let split_bin = DeviceArray::<ActiveRuntime, u32>::from_raw(split_bin_h, n_trees * total_nodes);
    let threshold = DeviceArray::<ActiveRuntime, F>::from_raw(threshold_h, n_trees * total_nodes);
    let is_leaf = DeviceArray::<ActiveRuntime, u32>::from_raw(is_leaf_h, n_trees * total_nodes);
    let leaf_value = DeviceArray::<ActiveRuntime, F>::from_raw(leaf_value_h, n_trees * total_nodes);

    let lr_f = f64_to_host::<F>(params.learning_rate);
    let l2_f = f64_to_host::<F>(params.l2_regularization);
    let min_leaf_f = f64_to_host::<F>(params.min_samples_leaf as f64);
    let min_hessian_f = f64_to_host::<F>(HGB_MIN_HESSIAN_TO_SPLIT);

    // =====================================================================
    // Boosting loop — LAUNCH-ONLY (no readbacks, no uploads).
    // =====================================================================
    for iter in 0..iters {
        let tree_base = (iter * k) as u32;

        // G1: per-sample gradients/hessians from the current raw predictions.
        match &target {
            HgbTarget::Reg(y_dev) => {
                let (count, dim) = launch_dims_1d(n);
                gbt_grad_reg::launch::<F, ActiveRuntime>(
                    &client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(raw.handle().clone(), n * k) },
                    unsafe { ArrayArg::from_raw_parts(y_dev.handle().clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(g_dev.handle().clone(), n * k) },
                    unsafe { ArrayArg::from_raw_parts(h_dev.handle().clone(), n * k) },
                    n as u32,
                );
            }
            HgbTarget::Binary(y_dev) => {
                let (count, dim) = launch_dims_1d(n);
                gbt_grad_binary::launch::<F, ActiveRuntime>(
                    &client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(raw.handle().clone(), n * k) },
                    unsafe { ArrayArg::from_raw_parts(y_dev.handle().clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(g_dev.handle().clone(), n * k) },
                    unsafe { ArrayArg::from_raw_parts(h_dev.handle().clone(), n * k) },
                    n as u32,
                );
            }
            HgbTarget::Multi(y_dev) => {
                let (mx, se) = softmax_stage
                    .as_ref()
                    .expect("softmax staging buffers exist for the Multi target");
                {
                    let (count, dim) = launch_dims_1d(n);
                    gbt_row_max::launch::<F, ActiveRuntime>(
                        &client,
                        count.clone(),
                        dim,
                        unsafe { ArrayArg::from_raw_parts(raw.handle().clone(), n * k) },
                        unsafe { ArrayArg::from_raw_parts(mx.handle().clone(), n) },
                        n as u32,
                        k as u32,
                    );
                    let (count2, dim2) = launch_dims_1d(n);
                    gbt_row_sumexp::launch::<F, ActiveRuntime>(
                        &client,
                        count2,
                        dim2,
                        unsafe { ArrayArg::from_raw_parts(raw.handle().clone(), n * k) },
                        unsafe { ArrayArg::from_raw_parts(mx.handle().clone(), n) },
                        unsafe { ArrayArg::from_raw_parts(se.handle().clone(), n) },
                        n as u32,
                        k as u32,
                    );
                }
                let (count, dim) = launch_dims_1d(n * k);
                gbt_grad_multi::launch::<F, ActiveRuntime>(
                    &client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(raw.handle().clone(), n * k) },
                    unsafe { ArrayArg::from_raw_parts(mx.handle().clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(se.handle().clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(y_dev.handle().clone(), n) },
                    unsafe { ArrayArg::from_raw_parts(g_dev.handle().clone(), n * k) },
                    unsafe { ArrayArg::from_raw_parts(h_dev.handle().clone(), n * k) },
                    n as u32,
                    k as u32,
                );
            }
        }

        // G2: reset the per-iteration row partition (identity order, root
        // range [0, n) per tree) — a device kernel, zero uploads.
        {
            let (count, dim) = launch_dims_1d(k * n);
            gbt_init_partition::launch::<ActiveRuntime>(
                &client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(order_a.handle().clone(), k * n) },
                unsafe { ArrayArg::from_raw_parts(ranges_a.handle().clone(), ranges_len) },
                n as u32,
                k as u32,
            );
        }

        // Level loop — the batched histogram/split pipeline over K trees.
        for level in 0..=depth {
            let nodes = 1usize << level;
            let level_base = (nodes - 1) as u32;
            let force_leaf = if level == depth { 1u32 } else { 0u32 };
            let root_level = if level == 0 { 1u32 } else { 0u32 };

            // Node chunking to the transient byte budget (hist + one block
            // set + scores per node; blocks only shrink the chunk further).
            let per_node_bytes =
                k * d * nb * 3 * size_of::<F>() * 2 + k * d * (nb - 1) * size_of::<F>();
            let node_chunk = (HGB_HIST_BUDGET_BYTES / per_node_bytes.max(1)).clamp(1, nodes);

            let mut node_base = 0usize;
            while node_base < nodes {
                let nc_now = node_chunk.min(nodes - node_base);

                // Row blocks: aim for HGB_HIST_TARGET_UNITS units, bounded by
                // the block cap and the byte budget.
                let mut blocks =
                    (HGB_HIST_TARGET_UNITS / (k * nc_now * d).max(1)).clamp(1, HGB_MAX_BLOCKS);
                while blocks > 1
                    && k * nc_now * d * blocks * nb * 3 * size_of::<F>() > HGB_HIST_BUDGET_BYTES
                {
                    blocks /= 2;
                }

                let part_len = k * nc_now * d * blocks * nb * 3;
                let hist_len = k * nc_now * d * nb * 3;
                let scores_len = k * nc_now * d * (nb - 1);
                // With a single block the "partial" histogram IS the final
                // histogram — gather straight into it and skip the reduce.
                let part_h = if blocks > 1 {
                    Some(pool.acquire(part_len * size_of::<F>()))
                } else {
                    None
                };
                let hist_h = pool.acquire(hist_len * size_of::<F>());
                let scores_h = pool.acquire(scores_len * size_of::<F>());

                // K1: row-blocked histogram gather (count, Σg, Σh).
                {
                    let gather_h = part_h.as_ref().unwrap_or(&hist_h);
                    let gather_len = if blocks > 1 { part_len } else { hist_len };
                    let (count, dim) = launch_dims_1d(k * nc_now * d * blocks);
                    gbt_hist::launch::<F, ActiveRuntime>(
                        &client,
                        count,
                        dim,
                        unsafe { ArrayArg::from_raw_parts(binned.handle().clone(), n * d) },
                        unsafe { ArrayArg::from_raw_parts(g_dev.handle().clone(), n * k) },
                        unsafe { ArrayArg::from_raw_parts(h_dev.handle().clone(), n * k) },
                        unsafe { ArrayArg::from_raw_parts(order_a.handle().clone(), k * n) },
                        unsafe { ArrayArg::from_raw_parts(ranges_a.handle().clone(), ranges_len) },
                        unsafe { ArrayArg::from_raw_parts(gather_h.clone(), gather_len) },
                        n as u32,
                        d as u32,
                        nb as u32,
                        k as u32,
                        nodes as u32,
                        node_base as u32,
                        nc_now as u32,
                        blocks as u32,
                    );
                }

                // K2: reduce the block axis (skipped when blocks == 1).
                if let Some(part) = &part_h {
                    let (count, dim) = launch_dims_1d(hist_len);
                    gbt_hist_reduce::launch::<F, ActiveRuntime>(
                        &client,
                        count,
                        dim,
                        unsafe { ArrayArg::from_raw_parts(part.clone(), part_len) },
                        unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
                        nb as u32,
                        (k * nc_now * d) as u32,
                        blocks as u32,
                    );
                }

                // K3: cumulative histogram over bins (tree.rs kernel, ncs=3).
                {
                    let (count, dim) = launch_dims_1d(k * nc_now * d * 3);
                    rf_hist_cum::launch::<F, ActiveRuntime>(
                        &client,
                        count,
                        dim,
                        unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
                        d as u32,
                        nb as u32,
                        3u32,
                        nc_now as u32,
                        k as u32,
                    );
                }

                // K4: split gains (sklearn XGBoost-form, validity-gated).
                {
                    let (count, dim) = launch_dims_1d(scores_len);
                    gbt_split_scores::launch::<F, ActiveRuntime>(
                        &client,
                        count,
                        dim,
                        unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
                        unsafe { ArrayArg::from_raw_parts(scores_h.clone(), scores_len) },
                        min_leaf_f,
                        min_hessian_f,
                        l2_f,
                        d as u32,
                        nb as u32,
                        nc_now as u32,
                        k as u32,
                        root_level,
                    );
                }

                // K5: per-node best split + leaf finalize (model writes).
                {
                    let (count, dim) = launch_dims_1d(k * nc_now);
                    gbt_best_split::launch::<F, ActiveRuntime>(
                        &client,
                        count,
                        dim,
                        unsafe { ArrayArg::from_raw_parts(hist_h.clone(), hist_len) },
                        unsafe { ArrayArg::from_raw_parts(scores_h.clone(), scores_len) },
                        unsafe {
                            ArrayArg::from_raw_parts(edges_dev.handle().clone(), edges_host.len())
                        },
                        unsafe {
                            ArrayArg::from_raw_parts(
                                split_feature.handle().clone(),
                                n_trees * total_nodes,
                            )
                        },
                        unsafe {
                            ArrayArg::from_raw_parts(
                                split_bin.handle().clone(),
                                n_trees * total_nodes,
                            )
                        },
                        unsafe {
                            ArrayArg::from_raw_parts(
                                threshold.handle().clone(),
                                n_trees * total_nodes,
                            )
                        },
                        unsafe {
                            ArrayArg::from_raw_parts(
                                is_leaf.handle().clone(),
                                n_trees * total_nodes,
                            )
                        },
                        unsafe {
                            ArrayArg::from_raw_parts(
                                leaf_value.handle().clone(),
                                n_trees * total_nodes,
                            )
                        },
                        lr_f,
                        l2_f,
                        d as u32,
                        nb as u32,
                        nc_now as u32,
                        k as u32,
                        tree_base,
                        level_base + node_base as u32,
                        total_nodes as u32,
                        force_leaf,
                    );
                }

                if let Some(part) = part_h {
                    pool.release(part, part_len * size_of::<F>());
                }
                pool.release(hist_h, hist_len * size_of::<F>());
                pool.release(scores_h, scores_len * size_of::<F>());

                node_base += nc_now;
            }

            // K6 + K7: child ranges + stable partition (full-level launches).
            if level < depth {
                let units = k * nodes;
                let (count, dim) = launch_dims_1d(units);
                gbt_count_left::launch::<ActiveRuntime>(
                    &client,
                    count,
                    dim,
                    unsafe { ArrayArg::from_raw_parts(binned.handle().clone(), n * d) },
                    unsafe { ArrayArg::from_raw_parts(order_a.handle().clone(), k * n) },
                    unsafe { ArrayArg::from_raw_parts(ranges_a.handle().clone(), ranges_len) },
                    unsafe {
                        ArrayArg::from_raw_parts(
                            split_feature.handle().clone(),
                            n_trees * total_nodes,
                        )
                    },
                    unsafe {
                        ArrayArg::from_raw_parts(split_bin.handle().clone(), n_trees * total_nodes)
                    },
                    unsafe {
                        ArrayArg::from_raw_parts(is_leaf.handle().clone(), n_trees * total_nodes)
                    },
                    unsafe { ArrayArg::from_raw_parts(ranges_b.handle().clone(), ranges_len) },
                    n as u32,
                    d as u32,
                    nodes as u32,
                    k as u32,
                    tree_base,
                    level_base,
                    total_nodes as u32,
                );
                let (count2, dim2) = launch_dims_1d(units);
                gbt_partition::launch::<ActiveRuntime>(
                    &client,
                    count2,
                    dim2,
                    unsafe { ArrayArg::from_raw_parts(binned.handle().clone(), n * d) },
                    unsafe { ArrayArg::from_raw_parts(order_a.handle().clone(), k * n) },
                    unsafe { ArrayArg::from_raw_parts(ranges_a.handle().clone(), ranges_len) },
                    unsafe { ArrayArg::from_raw_parts(ranges_b.handle().clone(), ranges_len) },
                    unsafe {
                        ArrayArg::from_raw_parts(
                            split_feature.handle().clone(),
                            n_trees * total_nodes,
                        )
                    },
                    unsafe {
                        ArrayArg::from_raw_parts(split_bin.handle().clone(), n_trees * total_nodes)
                    },
                    unsafe {
                        ArrayArg::from_raw_parts(is_leaf.handle().clone(), n_trees * total_nodes)
                    },
                    unsafe { ArrayArg::from_raw_parts(order_b.handle().clone(), k * n) },
                    n as u32,
                    d as u32,
                    nodes as u32,
                    k as u32,
                    tree_base,
                    level_base,
                    total_nodes as u32,
                );
                std::mem::swap(&mut order_a, &mut order_b);
                std::mem::swap(&mut ranges_a, &mut ranges_b);
            }
        }

        // G3: fold this iteration's trees into the train raw predictions
        // (binned traversal — exactly the training partition rule).
        {
            let (count, dim) = launch_dims_1d(n * k);
            gbt_update_raw::launch::<F, ActiveRuntime>(
                &client,
                count,
                dim,
                unsafe { ArrayArg::from_raw_parts(binned.handle().clone(), n * d) },
                unsafe {
                    ArrayArg::from_raw_parts(split_feature.handle().clone(), n_trees * total_nodes)
                },
                unsafe {
                    ArrayArg::from_raw_parts(split_bin.handle().clone(), n_trees * total_nodes)
                },
                unsafe {
                    ArrayArg::from_raw_parts(is_leaf.handle().clone(), n_trees * total_nodes)
                },
                unsafe {
                    ArrayArg::from_raw_parts(leaf_value.handle().clone(), n_trees * total_nodes)
                },
                unsafe { ArrayArg::from_raw_parts(raw.handle().clone(), n * k) },
                n as u32,
                d as u32,
                k as u32,
                depth as u32,
                tree_base,
                total_nodes as u32,
            );
        }
    }

    // Fit-only scratch back to the pool.
    binned.release_into(pool);
    raw.release_into(pool);
    g_dev.release_into(pool);
    h_dev.release_into(pool);
    if let Some((mx, se)) = softmax_stage {
        mx.release_into(pool);
        se.release_into(pool);
    }
    order_a.release_into(pool);
    order_b.release_into(pool);
    ranges_a.release_into(pool);
    ranges_b.release_into(pool);
    edges_dev.release_into(pool);
    split_bin.release_into(pool);

    Ok(HgbModel {
        split_feature,
        threshold,
        is_leaf,
        leaf_value,
        baseline: baseline_dev,
        n_iters: iters,
        k,
        n_classes,
        max_depth: depth,
        total_nodes,
        n_features: d,
    })
}

/// Validate the shared predict geometry.
fn validate_predict<F>(model: &HgbModel<F>, xq_len: usize, q: usize, d: usize) -> Result<(), PrimError>
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
    fits_u32(model.n_iters * model.k * q, "n_trees*q")?;
    Ok(())
}

/// Raw ensemble scores for `xq`: a `q × K` device buffer
/// `baseline + Σ_iters leaf values` (the sklearn `_raw_predict`). Shared by
/// the regression and both classification predict paths.
pub fn hgb_predict_raw<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    model: &HgbModel<F>,
    xq: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let (q, d) = shape;
    validate_predict(model, xq.len(), q, d)?;
    let client = pool.client().clone();
    let n_trees = model.n_iters * model.k;

    // Traverse every stage tree (tree.rs kernel; complete-tree layout).
    let leaf_h = pool.acquire(n_trees * q * size_of::<u32>());
    {
        let (count, dim) = launch_dims_1d(n_trees * q);
        rf_predict_leaf::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(xq.handle().clone(), q * d) },
            unsafe {
                ArrayArg::from_raw_parts(
                    model.split_feature.handle().clone(),
                    n_trees * model.total_nodes,
                )
            },
            unsafe {
                ArrayArg::from_raw_parts(
                    model.threshold.handle().clone(),
                    n_trees * model.total_nodes,
                )
            },
            unsafe {
                ArrayArg::from_raw_parts(
                    model.is_leaf.handle().clone(),
                    n_trees * model.total_nodes,
                )
            },
            unsafe { ArrayArg::from_raw_parts(leaf_h.clone(), n_trees * q) },
            q as u32,
            d as u32,
            model.max_depth as u32,
            n_trees as u32,
            model.total_nodes as u32,
        );
    }
    let leaf = DeviceArray::<ActiveRuntime, u32>::from_raw(leaf_h, n_trees * q);

    let raw_h = pool.acquire(q * model.k * size_of::<F>());
    {
        let (count, dim) = launch_dims_1d(q * model.k);
        gbt_sum_raw::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(leaf.handle().clone(), n_trees * q) },
            unsafe {
                ArrayArg::from_raw_parts(
                    model.leaf_value.handle().clone(),
                    n_trees * model.total_nodes,
                )
            },
            unsafe { ArrayArg::from_raw_parts(model.baseline.handle().clone(), model.k) },
            unsafe { ArrayArg::from_raw_parts(raw_h.clone(), q * model.k) },
            q as u32,
            model.k as u32,
            model.n_iters as u32,
            model.total_nodes as u32,
        );
    }
    leaf.release_into(pool);
    Ok(DeviceArray::from_raw(raw_h, q * model.k))
}

/// Regressor inference: length-`q` device predictions (the raw scores —
/// squared error has the identity link).
pub fn hgb_predict_reg<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    model: &HgbModel<F>,
    xq: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if model.n_classes != 1 {
        return Err(PrimError::DimMismatch {
            dim: "n_classes",
            lhs: model.n_classes,
            rhs: 1,
        });
    }
    hgb_predict_raw(pool, model, xq, shape)
}

/// Classifier inference: `q × n_classes` device probabilities (sigmoid of the
/// single raw column for binary, softmax over the `K` columns for multiclass
/// — the sklearn `predict_proba` link functions).
pub fn hgb_predict_proba<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    model: &HgbModel<F>,
    xq: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let (q, _d) = shape;
    if model.n_classes < 2 {
        return Err(PrimError::DimMismatch {
            dim: "n_classes",
            lhs: model.n_classes,
            rhs: 2,
        });
    }
    let raw = hgb_predict_raw(pool, model, xq, shape)?;
    let client = pool.client().clone();
    let nc = model.n_classes;
    let proba_h = pool.acquire(q * nc * size_of::<F>());

    if model.k == 1 {
        // Binary: p = σ(raw) → [1 − p, p].
        let (count, dim) = launch_dims_1d(q);
        gbt_proba_binary::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(raw.handle().clone(), q) },
            unsafe { ArrayArg::from_raw_parts(proba_h.clone(), q * nc) },
            q as u32,
        );
    } else {
        // Multiclass: softmax with staged row max / sum-exp (FINDING 002-B).
        let mx_h = pool.acquire(q * size_of::<F>());
        let se_h = pool.acquire(q * size_of::<F>());
        {
            let (count, dim) = launch_dims_1d(q);
            gbt_row_max::launch::<F, ActiveRuntime>(
                &client,
                count.clone(),
                dim,
                unsafe { ArrayArg::from_raw_parts(raw.handle().clone(), q * nc) },
                unsafe { ArrayArg::from_raw_parts(mx_h.clone(), q) },
                q as u32,
                nc as u32,
            );
            let (count2, dim2) = launch_dims_1d(q);
            gbt_row_sumexp::launch::<F, ActiveRuntime>(
                &client,
                count2,
                dim2,
                unsafe { ArrayArg::from_raw_parts(raw.handle().clone(), q * nc) },
                unsafe { ArrayArg::from_raw_parts(mx_h.clone(), q) },
                unsafe { ArrayArg::from_raw_parts(se_h.clone(), q) },
                q as u32,
                nc as u32,
            );
        }
        let (count, dim) = launch_dims_1d(q * nc);
        gbt_proba_multi::launch::<F, ActiveRuntime>(
            &client,
            count,
            dim,
            unsafe { ArrayArg::from_raw_parts(raw.handle().clone(), q * nc) },
            unsafe { ArrayArg::from_raw_parts(mx_h.clone(), q) },
            unsafe { ArrayArg::from_raw_parts(se_h.clone(), q) },
            unsafe { ArrayArg::from_raw_parts(proba_h.clone(), q * nc) },
            q as u32,
            nc as u32,
        );
        pool.release(mx_h, q * size_of::<F>());
        pool.release(se_h, q * size_of::<F>());
    }

    raw.release_into(pool);
    Ok(DeviceArray::from_raw(proba_h, q * nc))
}
