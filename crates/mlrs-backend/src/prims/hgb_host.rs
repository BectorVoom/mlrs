//! `hgb_host` — the **host** arm of the HistGradientBoosting fit (GBT-PERF-CPU).
//!
//! ## Why a host arm exists
//! `cubecl-cpu` maps ONE OS THREAD PER UNIT (see the `capability::
//! cpu_launch_units` doc and the KNN / HDBSCAN / UMAP / ARIMA cpu campaigns),
//! so a GPU-shaped launch grid is pathological on the cpu backend. The
//! boosting fit is the worst shape in the crate: `max_iter · (max_depth + 1)`
//! LEVELS, each issuing a histogram gather over `k · nodes · d · blocks` units
//! plus three partition launches over `k · nodes · bcount` units. At the
//! probe's `50 000 × 16`, 100 iterations, depth 6 that is 700 histogram
//! launches of ~3 000 units each — the `HGB_PROFILE` attribution put 62 % of
//! the fit in `hist` and 16 % in `partition`, none of it real arithmetic.
//!
//! This module replays the SAME algorithm in native host code over a small
//! persistent worker pool: threads are spawned ONCE per fit and synchronize on
//! a spin/yield barrier, so the per-level cost is a barrier (µs) instead of a
//! thread-per-unit launch (ms).
//!
//! ## Why the result is the kernel's result (bit-identical, not merely close)
//! Gradient boosting is SEQUENTIAL — an iteration's trees are grown from the
//! previous iteration's raw predictions — so a last-ULP difference compounds
//! and can flip a split, which is a macroscopic change in the fitted model.
//! The host arm is therefore an exact replay, not an independent
//! implementation. Every float value is produced by the same operations in the
//! same association and the same summation ORDER as the device path:
//!
//! - the same `blocks` count from the same
//!   [`HGB_HIST_TARGET_UNITS`]/[`HGB_MAX_BLOCKS`]/[`HGB_HIST_BUDGET_BYTES`]
//!   rule, the same `lo = s + blk·len/blocks` stripe bounds, and the same
//!   ascending row scan inside a stripe ([`gbt_hist`]);
//! - the same ascending block reduce ([`gbt_hist_reduce`]) — SKIPPED at
//!   `blocks == 1` exactly as the host orchestration skips it — then the same
//!   ascending bin cumsum ([`rf_hist_cum`]);
//! - the same sibling SUBTRACTION eligibility (`subtract_cap`) and the same
//!   `right = parent − left` with the leaf-phantom zero ([`gbt_hist_subtract`]);
//! - the same `1e-15` epsilon, which is an **f32** literal in the kernels
//!   (`F::new(1e-15_f32)`) and therefore widens to `1.0000000036274937e-15` at
//!   f64 — [`HostFloat::lit`] reproduces that widening rather than parsing a
//!   fresh f64 literal;
//! - the same strict-`>` argmax over the flat `(feature, bin)` gain slice, so
//!   ties keep the lowest flat index ([`gbt_best_split`]).
//!
//! The row-scan LOOP NEST is interchanged (rows outer, features inner, so the
//! row-major `binned` reads are contiguous) — that is safe because each
//! histogram slot still accumulates the same rows in the same ascending order;
//! only the interleaving of INDEPENDENT slots changes.
//!
//! The partition is pure INDEX work with no float arithmetic, and
//! [`gbt_partition_blocks`] is documented to produce a `order_next`
//! bitwise-identical to the serial [`gbt_partition`]. The host arm is free to
//! pick its own row-block count there, and does (sized to the worker pool
//! rather than to a GPU launch grid).
//!
//! [`gbt_hist`]: mlrs_kernels::gbt::gbt_hist
//! [`gbt_hist_reduce`]: mlrs_kernels::gbt::gbt_hist_reduce
//! [`gbt_hist_subtract`]: mlrs_kernels::gbt::gbt_hist_subtract
//! [`gbt_best_split`]: mlrs_kernels::gbt::gbt_best_split
//! [`gbt_partition`]: mlrs_kernels::gbt::gbt_partition
//! [`gbt_partition_blocks`]: mlrs_kernels::gbt::gbt_partition_blocks
//! [`rf_hist_cum`]: mlrs_kernels::tree::rf_hist_cum
//!
//! Tests live in `crates/mlrs-backend/tests/` (AGENTS.md §2).


use bytemuck::Pod;
use cubecl::prelude::*;

use super::hist_gradient_boosting::{
    HgbParams, HGB_HIST_BUDGET_BYTES, HGB_HIST_TARGET_UNITS, HGB_MAX_BLOCKS,
    HGB_MIN_HESSIAN_TO_SPLIT,
};

// =====================================================================
// Host float
// =====================================================================

/// The concrete host float the replay runs in: `f32` mirrors an `F = f32`
/// device fit, `f64` an `F = f64` one. Every method is the host twin of the
/// CubeCL op the kernels use, so the two arms round identically.
pub(crate) trait HostFloat:
    Copy
    + Send
    + Sync
    + 'static
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + PartialOrd
    + bytemuck::Pod
{
    /// `F::new(0.0_f32)`.
    const ZERO: Self;
    /// `F::new(1.0_f32)`.
    const ONE: Self;
    /// The kernels' `F::new(v_f32)` — an **f32** literal widened to `Self`.
    /// At `Self = f64` this is deliberately NOT the f64 literal of the same
    /// decimal text (`1e-15_f32 as f64 != 1e-15_f64`).
    fn lit(v: f32) -> Self;
    /// The host `f64_to_host::<F>` narrowing used for the scalar
    /// hyperparameters (`learning_rate`, `l2`, `min_samples_leaf`).
    fn from_f64(v: f64) -> Self;
    /// `F::exp`.
    fn exp(self) -> Self;
    /// `F::abs` (the instance-form `.abs()` the kernels use).
    fn abs(self) -> Self;
    /// `F::log1p` — the instance-form `.log1p()` the [`sgd_loss`
    /// kernel](mlrs_kernels::sgd::sgd_loss) uses for the `Log` loss value.
    fn log1p(self) -> Self;
    /// Widen to `f64` — the host twin of `host_to_f64`.
    fn to_f64(self) -> f64;
}

impl HostFloat for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn lit(v: f32) -> Self {
        v
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f32::exp(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f32::abs(self)
    }
    #[inline(always)]
    fn log1p(self) -> Self {
        f32::ln_1p(self)
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl HostFloat for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn lit(v: f32) -> Self {
        v as f64
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline(always)]
    fn log1p(self) -> Self {
        f64::ln_1p(self)
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
}

// =====================================================================
// Worker pool primitives
// =====================================================================

// The barrier + disjoint-slice handle this pool synchronizes on now live in
// `prims::host_pool`, shared with the other cpu arms (the SVM objective's
// per-evaluation pass dispatches through the same primitives). They were
// written and tuned here; nothing about their behaviour changed in the move.
use super::host_pool::{Barrier, Shared};

/// This worker's contiguous half-open slice of `total` tasks.
///
/// Contiguous (not round-robin) so neighbouring tasks — which write
/// neighbouring words of the small `u32` count buffers — stay on one worker
/// and do not false-share a cache line.
#[inline(always)]
fn span(total: usize, wid: usize, workers: usize) -> (usize, usize) {
    let lo = total * wid / workers;
    let hi = total * (wid + 1) / workers;
    (lo, hi)
}

/// Worker count for the fit's pool.
///
/// `MLRS_HGB_WORKERS` overrides it for on-target A/B (`1` forces the fully
/// serial replay, which is the equivalence tests' baseline).
///
/// The default is HALF of `available_parallelism`, not all of it. Two measured
/// reasons, both pointing the same way:
///
/// - The fit is **bandwidth-bound, not core-bound**. Its dominant phase is a
///   random-access histogram gather (a permuted row order, three
///   read-modify-writes per feature), so throughput saturates well before the
///   thread count does — on the 8-core/16-thread development machine the
///   scaling curve was flat from 4 workers up. `available_parallelism`
///   counts SMT siblings, which share a core's load/store path and add
///   almost nothing to this workload.
/// - A level-synchronous pool is only as fast as its slowest worker at EVERY
///   barrier, so extra workers that cannot get a core are not free — they are
///   a straggler at each of the fit's thousands of barriers. Leaving headroom
///   is what keeps a machine that is also doing something else from falling
///   off the cliff (measured: at full width on a busy machine the fit was
///   slower than the 1-worker replay).
///
/// Raise it with the knob on a dedicated machine with more memory channels.
fn worker_count() -> usize {
    if let Some(v) = crate::abflag::var("MLRS_HGB_WORKERS").and_then(|v| v.parse::<usize>().ok()) {
        return v.max(1);
    }
    let cores = crate::capability::cpu_launch_units() as usize;
    (cores / 2).max(1)
}

// =====================================================================
// Public entry
// =====================================================================

/// The three loss targets, in host form.
pub(crate) enum HostLoss<'a> {
    /// Squared error; `y` is the length-`n` target.
    Reg,
    /// Binary log-loss; `y` is the length-`n` `{0, 1}` target.
    Binary,
    /// Multiclass log-loss; the dense class index per row.
    Multi(&'a [u32]),
}

/// The fitted complete-tree model arrays, host-side
/// (`(max_iter · k) × total_nodes` each) — exactly what the device path leaves
/// in its pool buffers, ready to upload.
pub(crate) struct HostModel<F> {
    pub split_feature: Vec<u32>,
    pub threshold: Vec<F>,
    pub is_leaf: Vec<u32>,
    pub leaf_value: Vec<F>,
}

/// Is the host arm the right arm for this fit?
///
/// `cpu` only: on wgpu/CUDA/ROCm the level pipeline is a genuine parallel
/// launch and this pool would be a large regression. `MLRS_HGB_HOST=0` forces
/// the device path back on for on-target A/B (the equivalence tests drive both
/// arms through the public prim with it).
pub(crate) fn host_fit_applicable() -> bool {
    crate::capability::active_backend_name() == "cpu"
        && crate::abflag::var("MLRS_HGB_HOST")
            .map(|v| v != "0")
            .unwrap_or(true)
}

/// Host replay of `hgb_fit_impl`'s boosting loop.
///
/// `x_host` is the `n × d` row-major feature matrix and `edges_host` the
/// `d × (n_bins − 1)` quantile-midpoint table the caller already computed
/// (`compute_edges`) — both are read exactly as `rf_bin_features` reads them.
/// `y_f` carries the `Reg`/`Binary` target (ignored for `Multi`).
pub(crate) fn hgb_fit_host<F>(
    x_host: &[F],
    edges_host: &[F],
    n: usize,
    d: usize,
    y_f: &[F],
    loss: HostLoss<'_>,
    baseline: &[f64],
    params: &HgbParams,
) -> HostModel<F>
where
    F: Float + CubeElement + Pod,
{
    if size_of::<F>() == 4 {
        let m = fit_typed::<f32>(
            bytemuck::cast_slice(x_host),
            bytemuck::cast_slice(edges_host),
            n,
            d,
            bytemuck::cast_slice(y_f),
            loss,
            baseline,
            params,
        );
        HostModel {
            split_feature: m.split_feature,
            threshold: bytemuck::cast_slice::<f32, F>(&m.threshold).to_vec(),
            is_leaf: m.is_leaf,
            leaf_value: bytemuck::cast_slice::<f32, F>(&m.leaf_value).to_vec(),
        }
    } else {
        let m = fit_typed::<f64>(
            bytemuck::cast_slice(x_host),
            bytemuck::cast_slice(edges_host),
            n,
            d,
            bytemuck::cast_slice(y_f),
            loss,
            baseline,
            params,
        );
        HostModel {
            split_feature: m.split_feature,
            threshold: bytemuck::cast_slice::<f64, F>(&m.threshold).to_vec(),
            is_leaf: m.is_leaf,
            leaf_value: bytemuck::cast_slice::<f64, F>(&m.leaf_value).to_vec(),
        }
    }
}

// =====================================================================
// The typed replay
// =====================================================================

/// Per-level plan — the host twin of the decisions `hgb_fit_impl` makes inline
/// (histogram row blocks, sibling-subtraction eligibility, node chunking).
/// Precomputed once because every worker must take the SAME branches.
#[derive(Clone, Copy)]
struct LevelPlan {
    /// Nodes at this level (`2^level`).
    nodes: usize,
    /// `2^level − 1` — the complete-layout id of node 0.
    level_base: usize,
    /// Derive this level's histogram from the retained parent by subtraction.
    subtract: bool,
    /// Nodes per histogram chunk (`nodes` on the single-shot path).
    node_chunk: usize,
    /// Keep this level's histogram for the next level's subtraction.
    retain: bool,
}

/// Everything the worker body touches, in one struct so the closure capture
/// stays readable.
struct Ctx<'a, T: HostFloat> {
    n: usize,
    d: usize,
    nb: usize,
    k: usize,
    depth: usize,
    iters: usize,
    total_nodes: usize,
    elem: usize,
    workers: usize,
    exact_blocks: bool,
    plans: &'a [LevelPlan],
    lr: T,
    l2: T,
    min_leaf: T,
    min_hessian: T,
    // Read-only inputs.
    binned: &'a [u8],
    binned_t: &'a [u8],
    edges: &'a [T],
    y_f: &'a [T],
    y_idx: &'a [u32],
    loss_kind: u8,
    baseline: &'a [T],
    // Shared mutable state.
    raw: Shared<T>,
    gh: Shared<T>,
    order_a: Shared<u32>,
    order_b: Shared<u32>,
    ranges_a: Shared<u32>,
    ranges_b: Shared<u32>,
    blk_cnt: Shared<u32>,
    partials: Shared<T>,
    hist: Shared<T>,
    hist_parent: Shared<T>,
    hist_left: Shared<T>,
    scores: Shared<T>,
    split_feature: Shared<u32>,
    split_bin: Shared<u32>,
    threshold: Shared<T>,
    is_leaf: Shared<u32>,
    leaf_value: Shared<T>,
    bar: &'a Barrier,
}

/// Loss discriminants (a `u8` so `Ctx` stays `Copy`-friendly and the worker
/// body branches on a scalar).
const LOSS_REG: u8 = 0;
const LOSS_BINARY: u8 = 1;
const LOSS_MULTI: u8 = 2;

#[allow(clippy::too_many_arguments)]
fn fit_typed<T: HostFloat>(
    x: &[T],
    edges: &[T],
    n: usize,
    d: usize,
    y_f: &[T],
    loss: HostLoss<'_>,
    baseline_f64: &[f64],
    params: &HgbParams,
) -> HostModel<T> {
    let k = baseline_f64.len();
    let iters = params.max_iter;
    let depth = params.max_depth;
    let nb = params.n_bins;
    let total_nodes = (1usize << (depth + 1)) - 1;
    let max_nodes_level = 1usize << depth;
    let n_trees = iters * k;
    let elem = d * nb * 3;

    let (loss_kind, y_idx): (u8, &[u32]) = match loss {
        HostLoss::Reg => (LOSS_REG, &[]),
        HostLoss::Binary => (LOSS_BINARY, &[]),
        HostLoss::Multi(idx) => (LOSS_MULTI, idx),
    };

    // --- Bin once, in both layouts. `nb ≤ 256` so a bin index is a `u8`;
    // the row-major copy feeds the histogram's feature-inner scan and
    // `update_raw`'s per-row descent, the feature-major copy the partition's
    // single-feature row scans (the `binned_t` split the device path makes for
    // the same reason). ---
    let (binned, binned_t) = bin_features(x, edges, n, d, nb);

    // --- Level plans (identical on every worker). ---
    let per_node_bytes = k * elem * size_of::<T>() * 2 + k * d * (nb - 1) * size_of::<T>();
    let subtract_cap = HGB_HIST_BUDGET_BYTES / per_node_bytes.max(1);
    let mut plans: Vec<LevelPlan> = Vec::with_capacity(depth + 1);
    let mut retained = false;
    for level in 0..=depth {
        let nodes = 1usize << level;
        let single_shot = nodes <= subtract_cap;
        let subtract = single_shot && retained;
        let node_chunk = if single_shot {
            nodes
        } else {
            subtract_cap.clamp(1, nodes)
        };
        let retain = single_shot && level < depth && (nodes * 2) <= subtract_cap;
        plans.push(LevelPlan {
            nodes,
            level_base: nodes - 1,
            subtract,
            node_chunk,
            retain,
        });
        retained = retain;
    }

    let workers = worker_count().clamp(1, 64);
    // Bitwise-exact replay of the device histogram grouping (see
    // [`hist_blocks`]). Default ON: gradient boosting is sequential, so a
    // last-ULP difference compounds across iterations. `MLRS_HGB_EXACT=0`
    // trades that for pool-sized stripes.
    let exact_blocks = crate::abflag::var("MLRS_HGB_EXACT")
        .map(|v| v != "0")
        .unwrap_or(true);

    // --- Buffer sizing: the maxima over the level plans (allocated once, so
    // the boosting loop never touches the allocator). ---
    let mut max_hist = 0usize;
    let mut max_partials = 0usize;
    let mut max_scores = 0usize;
    for p in &plans {
        let mut base = 0usize;
        while base < p.nodes {
            let nc_full = p.node_chunk.min(p.nodes - base);
            // The subtraction pass gathers HALF the chunk (left children only).
            let nc_gather = if p.subtract { nc_full / 2 } else { nc_full };
            let blocks = hist_blocks::<T>(exact_blocks, k, nc_gather, d, nb);
            max_partials = max_partials.max(k * nc_gather * blocks * elem);
            max_hist = max_hist.max(k * nc_full * elem);
            max_scores = max_scores.max(k * nc_full * d * (nb - 1));
            base += nc_full;
        }
    }
    let ranges_len = k * max_nodes_level * 2;

    let mut raw = vec![T::ZERO; n * k];
    // Gradient and hessian INTERLEAVED (`gh[2·(i·k + tt)]` / `+1`): the
    // histogram scan reads both for every visited row, and the row order is a
    // permutation, so pairing them halves that gather's cache lines.
    let mut gh = vec![T::ZERO; n * k * 2];
    let mut order_a = vec![0u32; k * n];
    let mut order_b = vec![0u32; k * n];
    let mut ranges_a = vec![0u32; ranges_len];
    let mut ranges_b = vec![0u32; ranges_len];
    let mut partials = vec![T::ZERO; max_partials];
    let subtracts = plans.iter().any(|p| p.retain);
    let mut hist = vec![T::ZERO; max_hist];
    let mut hist_parent = vec![T::ZERO; if subtracts { max_hist } else { 0 }];
    let mut hist_left = vec![T::ZERO; if subtracts { max_hist } else { 0 }];
    let mut scores = vec![T::ZERO; max_scores];
    let mut blk_cnt = vec![0u32; k * max_nodes_level * max_partition_blocks(workers)];

    let mut split_feature = vec![0u32; n_trees * total_nodes];
    let mut split_bin = vec![0u32; n_trees * total_nodes];
    let mut threshold = vec![T::ZERO; n_trees * total_nodes];
    let mut is_leaf = vec![0u32; n_trees * total_nodes];
    let mut leaf_value = vec![T::ZERO; n_trees * total_nodes];

    let baseline: Vec<T> = baseline_f64.iter().map(|&b| T::from_f64(b)).collect();
    let bar = Barrier::new(workers);

    let ctx = Ctx::<T> {
        n,
        d,
        nb,
        k,
        depth,
        iters,
        total_nodes,
        elem,
        workers,
        exact_blocks,
        plans: &plans,
        lr: T::from_f64(params.learning_rate),
        l2: T::from_f64(params.l2_regularization),
        min_leaf: T::from_f64(params.min_samples_leaf as f64),
        min_hessian: T::from_f64(HGB_MIN_HESSIAN_TO_SPLIT),
        binned: &binned,
        binned_t: &binned_t,
        edges,
        y_f,
        y_idx,
        loss_kind,
        baseline: &baseline,
        raw: Shared::new(&mut raw),
        gh: Shared::new(&mut gh),
        order_a: Shared::new(&mut order_a),
        order_b: Shared::new(&mut order_b),
        ranges_a: Shared::new(&mut ranges_a),
        ranges_b: Shared::new(&mut ranges_b),
        blk_cnt: Shared::new(&mut blk_cnt),
        partials: Shared::new(&mut partials),
        hist: Shared::new(&mut hist),
        hist_parent: Shared::new(&mut hist_parent),
        hist_left: Shared::new(&mut hist_left),
        scores: Shared::new(&mut scores),
        split_feature: Shared::new(&mut split_feature),
        split_bin: Shared::new(&mut split_bin),
        threshold: Shared::new(&mut threshold),
        is_leaf: Shared::new(&mut is_leaf),
        leaf_value: Shared::new(&mut leaf_value),
        bar: &bar,
    };

    // ONE scope for the whole fit: `workers − 1` threads are created here and
    // live until the last boosting iteration retires. The calling thread is
    // worker 0 and runs the identical body.
    let failure = std::thread::scope(|scope| {
        let ctx = &ctx;
        let handles: Vec<_> = (1..workers)
            .map(|wid| scope.spawn(move || run_worker(ctx, wid)))
            .collect();
        let mut failure = run_worker(ctx, 0).err();
        for h in handles {
            // A worker thread that itself panicked while unwinding (or was
            // killed) surfaces as a join error; treat it the same way.
            let r = h.join().unwrap_or(Err(()));
            failure = failure.or(r.err());
        }
        failure
    });
    if failure.is_some() {
        panic!("mlrs hgb host fit: a worker thread panicked (see the panic above)");
    }

    HostModel {
        split_feature,
        threshold,
        is_leaf,
        leaf_value,
    }
}

/// Row blocks for one histogram gather.
///
/// `exact` reproduces `gather_hist`'s verbatim rule, which is what makes the
/// replay BITWISE identical to the device arm — the block count fixes the
/// histogram's summation order. That rule targets a GPU launch grid (up to
/// [`HGB_MAX_BLOCKS`] = 64 stripes per node), and every stripe costs the host
/// a `d · nb · 3` zero-fill plus one more pass of the same size in the reduce.
///
/// `!exact` instead targets [`HGB_HOST_HIST_TASKS`] stripes for the whole
/// level — enough to fill a host pool, an order of magnitude fewer than a GPU
/// grid wants. Same algorithm, different summation grouping, so results move
/// by float-association noise the way the CUDA/ROCm shared-atomic gather
/// already does (see `gbt_hist_atomic`'s doc).
///
/// The target is a CONSTANT, deliberately not the worker count: a fit whose
/// stripe count tracked the pool would produce a different model on a machine
/// with a different core count, which is a far worse property than differing
/// from the device arm. Under either policy the fit stays deterministic and
/// worker-count independent.
fn hist_blocks<T>(exact: bool, k: usize, nc: usize, d: usize, nb: usize) -> usize {
    let mut blocks = if exact {
        (HGB_HIST_TARGET_UNITS / (k * nc * d).max(1)).clamp(1, HGB_MAX_BLOCKS)
    } else {
        HGB_HOST_HIST_TASKS
            .div_ceil((k * nc).max(1))
            .clamp(1, HGB_MAX_BLOCKS)
    };
    while blocks > 1 && k * nc * d * blocks * nb * 3 * size_of::<T>() > HGB_HIST_BUDGET_BYTES {
        blocks /= 2;
    }
    blocks
}

/// Stripe target for the non-exact host histogram policy (see
/// [`hist_blocks`]): enough tasks to fill a host pool with a little slack for
/// uneven node sizes, and small enough that the per-stripe zero-fill and
/// reduce pass stay a rounding error next to the row scan.
const HGB_HOST_HIST_TASKS: usize = 32;

/// Upper bound on [`partition_blocks_host`] (buffer sizing).
fn max_partition_blocks(workers: usize) -> usize {
    (4 * workers).max(1)
}

/// Row blocks for the partition scans.
///
/// Unlike the histogram this is NOT pinned to the device's choice: the
/// partition produces a bitwise-identical `order_next` for ANY block count
/// (the [`gbt_partition_blocks`] stability argument), so the host sizes it to
/// the worker pool instead of to a GPU launch grid — enough tasks to fill the
/// pool, never so many that a block scans fewer than [`MIN_ROWS_PER_BLOCK`]
/// rows.
///
/// [`gbt_partition_blocks`]: mlrs_kernels::gbt::gbt_partition_blocks
fn partition_blocks_host(units: usize, n: usize, nodes: usize, workers: usize) -> usize {
    max_partition_blocks(workers)
        .div_ceil(units.max(1))
        .min((n / nodes.max(1) / MIN_ROWS_PER_BLOCK).max(1))
        .clamp(1, max_partition_blocks(workers))
}

/// Fewest rows a partition row block is worth splitting down to.
const MIN_ROWS_PER_BLOCK: usize = 2048;

/// `rf_bin_features` / `rf_bin_features_t`, host side, as `u8` (`n_bins ≤ 256`
/// so a bin index always fits). Parallel over row stripes.
fn bin_features<T: HostFloat>(
    x: &[T],
    edges: &[T],
    n: usize,
    d: usize,
    nb: usize,
) -> (Vec<u8>, Vec<u8>) {
    let nbe = nb - 1;
    let mut binned = vec![0u8; n * d];
    let workers = worker_count().min(n.div_ceil(4096).max(1));
    let per = n.div_ceil(workers.max(1));
    std::thread::scope(|scope| {
        for (w, chunk) in binned.chunks_mut(per * d).enumerate() {
            scope.spawn(move || {
                let row0 = w * per;
                for (r, out) in chunk.chunks_mut(d).enumerate() {
                    let xr = &x[(row0 + r) * d..(row0 + r) * d + d];
                    for j in 0..d {
                        let v = xr[j];
                        let e = &edges[j * nbe..j * nbe + nbe];
                        let mut b = 0u32;
                        for &ev in e.iter() {
                            if v >= ev {
                                b += 1;
                            }
                        }
                        out[j] = b as u8;
                    }
                }
            });
        }
    });
    // Feature-major twin (one blocked transpose — the partition's row scans
    // read a single feature column).
    let mut binned_t = vec![0u8; n * d];
    {
        let src = &binned;
        let workers_t = worker_count().min(d).max(1);
        let per_f = d.div_ceil(workers_t);
        std::thread::scope(|scope| {
            for (w, chunk) in binned_t.chunks_mut(per_f * n).enumerate() {
                scope.spawn(move || {
                    let f0 = w * per_f;
                    for (jj, col) in chunk.chunks_mut(n).enumerate() {
                        let j = f0 + jj;
                        for (i, dst) in col.iter_mut().enumerate() {
                            *dst = src[i * d + j];
                        }
                    }
                });
            }
        });
    }
    (binned, binned_t)
}

// =====================================================================
// The worker body — every worker runs this identical control flow
// =====================================================================

/// `HGB_PROFILE=1` phase attribution for the host arm — the `HgbProf` shape,
/// AGGREGATED by label because the boosting loop repeats each phase
/// `iters × levels` times.
///
/// Only worker 0 times, and only immediately AFTER a barrier, so each figure
/// is that phase's CRITICAL PATH (the slowest worker), not one worker's share.
/// With the env var unset every call is a no-op.
struct HostProf {
    on: bool,
    t: std::time::Instant,
    rows: Vec<(&'static str, f64, usize)>,
}

impl HostProf {
    fn new(active: bool) -> Self {
        Self {
            on: active && std::env::var_os("HGB_PROFILE").is_some(),
            t: std::time::Instant::now(),
            rows: Vec::new(),
        }
    }

    #[inline(always)]
    fn lap(&mut self, label: &'static str) {
        if self.on {
            let dt = self.t.elapsed().as_secs_f64();
            match self.rows.iter_mut().find(|r| r.0 == label) {
                Some(r) => {
                    r.1 += dt;
                    r.2 += 1;
                }
                None => self.rows.push((label, dt, 1)),
            }
            self.t = std::time::Instant::now();
        }
    }

    fn dump(&self) {
        if self.on {
            eprintln!("=== HGB_PROFILE (host arm; critical path per phase) ===");
            let mut tot = 0.0;
            for (label, secs, laps) in &self.rows {
                eprintln!("{label:>14}: {:9.3} ms  ({laps:5} laps)", secs * 1e3);
                tot += secs;
            }
            eprintln!("{:>14}: {:9.3} ms", "TOTAL", tot * 1e3);
        }
    }
}

/// Run one worker, converting a panic into a POOL POISON so the survivors
/// leave their barriers instead of waiting for a thread that will never
/// arrive. The original panic message is still printed by the default hook;
/// the caller re-raises afterwards.
fn run_worker<T: HostFloat>(c: &Ctx<'_, T>, wid: usize) -> Result<(), ()> {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| boost(c, wid)));
    if r.is_err() {
        c.bar.poison();
        return Err(());
    }
    Ok(())
}

fn boost<T: HostFloat>(c: &Ctx<'_, T>, wid: usize) {
    let (n, d, nb, k) = (c.n, c.d, c.nb, c.k);
    let w = c.workers;

    // Ping-pong handles, swapped identically on every worker.
    let mut order_a = c.order_a;
    let mut order_b = c.order_b;
    let mut ranges_a = c.ranges_a;
    let mut ranges_b = c.ranges_b;
    // The retained parent level is a HANDLE swap, never a copy (what the
    // device path does with its two pool handles).
    let mut hist_cur = c.hist;
    let mut hist_par = c.hist_parent;
    let mut prof = HostProf::new(wid == 0);

    // --- Raw predictions = the per-class baseline (`gbt_init_raw`). ---
    {
        let (lo, hi) = span(n, wid, w);
        // SAFETY: disjoint row stripes; read back only after the barrier below.
        let raw = unsafe { c.raw.get_mut() };
        for i in lo..hi {
            for cc in 0..k {
                raw[i * k + cc] = c.baseline[cc];
            }
        }
    }
    if !c.bar.wait() {
        return;
    }

    for iter in 0..c.iters {
        let tree_base = iter * k;

        // --- G1: gradients / hessians from the current raw predictions. ---
        gradients(c, wid);
        if !c.bar.wait() {
            return;
        }
        prof.lap("grad");

        // --- G2: reset the row partition (identity order, root range). ---
        {
            let (lo, hi) = span(k * n, wid, w);
            // SAFETY: disjoint `(tree, row)` slots.
            let order = unsafe { order_a.get_mut() };
            for t in lo..hi {
                order[t] = (t % n) as u32;
            }
            // SAFETY: `k` root ranges, written by the worker that owns tree
            // `tt`'s first row slot — disjoint by construction.
            let ranges = unsafe { ranges_a.get_mut() };
            for tt in 0..k {
                if tt * n >= lo && tt * n < hi {
                    ranges[2 * tt] = 0;
                    ranges[2 * tt + 1] = n as u32;
                }
            }
        }
        if !c.bar.wait() {
            return;
        }
        prof.lap("init_part");

        for level in 0..=c.depth {
            let plan = c.plans[level];
            let nodes = plan.nodes;
            let force_leaf = level == c.depth;
            let root_level = level == 0;

            let mut node_base = 0usize;
            while node_base < nodes {
                let nc = plan.node_chunk.min(nodes - node_base);

                // ----- Phase A: histogram gather ------------------------
                // `subtract` gathers only the EVEN (left-child) nodes; the
                // odd siblings come from `parent − left` in phase B.
                let nc_gather = if plan.subtract { nc / 2 } else { nc };
                let node_stride = if plan.subtract { 2 } else { 1 };
                let blocks = hist_blocks::<T>(c.exact_blocks, k, nc_gather, d, nb);
                // With one block the gather writes the final histogram
                // directly (the device path skips its reduce too).
                let single_block = blocks == 1;
                // The gathered level lands in `hist_left` when a subtraction
                // will expand it, otherwise straight in the level buffer.
                let gathered = if plan.subtract { c.hist_left } else { hist_cur };
                let dst = if single_block { gathered } else { c.partials };
                gather(
                    c,
                    wid,
                    dst,
                    &order_a,
                    &ranges_a,
                    nodes,
                    node_base,
                    nc_gather,
                    node_stride,
                    blocks,
                );
                if !c.bar.wait() {
                    return;
                }
                prof.lap("hist");

                // ----- Phase B: block reduce + bin cumsum (+ the sibling
                // subtraction) and the split scan, in ONE phase.
                //
                // The task is one GATHERED node, and everything it then feeds
                // is its own: its reduce, its subtraction, and the split of
                // the one (or, under subtraction, the TWO) level nodes it
                // produces. Nothing crosses tasks, so the phases need no
                // barrier between them — which matters, because at these
                // sizes a barrier costs more than the phase does.
                reduce_split(
                    c,
                    wid,
                    hist_cur,
                    hist_par,
                    gathered,
                    nc_gather,
                    nc,
                    blocks,
                    single_block,
                    plan.subtract,
                    tree_base,
                    plan.level_base + node_base,
                    root_level,
                    force_leaf,
                );
                if !c.bar.wait() {
                    return;
                }
                prof.lap("reduce+split");

                node_base += nc;
            }

            // Retain this level's histogram as the next level's parent — a
            // handle swap, so the level buffer the subtraction writes next is
            // the (now stale) previous parent.
            if plan.retain {
                std::mem::swap(&mut hist_cur, &mut hist_par);
            }

            // ----- Phase D/E/F: partition into the next level -----------
            if level < c.depth {
                partition_level(
                    c,
                    wid,
                    &order_a,
                    &order_b,
                    &ranges_a,
                    &ranges_b,
                    nodes,
                    tree_base,
                    plan.level_base,
                );
                prof.lap("partition");
                std::mem::swap(&mut order_a, &mut order_b);
                std::mem::swap(&mut ranges_a, &mut ranges_b);
            }
        }

        // --- G3: fold this iteration's trees into the raw predictions. ---
        update_raw(c, wid, tree_base);
        if !c.bar.wait() {
            return;
        }
        prof.lap("update_raw");
    }
    prof.dump();
}

/// `gbt_grad_reg` / `gbt_grad_binary` / (`gbt_row_max`, `gbt_row_sumexp`,
/// `gbt_grad_multi`) — fused per row, which is exact: the multiclass staging
/// arrays are per-ROW values consumed only by that row's own elements, so
/// computing them in a register instead of a global array changes nothing.
fn gradients<T: HostFloat>(c: &Ctx<'_, T>, wid: usize) {
    let (n, k) = (c.n, c.k);
    let (lo, hi) = span(n, wid, c.workers);
    let raw = c.raw.get();
    // SAFETY: disjoint row stripes.
    let gh = unsafe { c.gh.get_mut() };
    match c.loss_kind {
        LOSS_REG => {
            for i in lo..hi {
                gh[2 * i] = raw[i] - c.y_f[i];
                gh[2 * i + 1] = T::ONE;
            }
        }
        LOSS_BINARY => {
            for i in lo..hi {
                let p = T::ONE / (T::ONE + (-raw[i]).exp());
                gh[2 * i] = p - c.y_f[i];
                gh[2 * i + 1] = p * (T::ONE - p);
            }
        }
        _ => {
            for i in lo..hi {
                let base = i * k;
                let mut mx = raw[base];
                for cc in 1..k {
                    let v = raw[base + cc];
                    if v > mx {
                        mx = v;
                    }
                }
                let mut se = T::ZERO;
                for cc in 0..k {
                    se += (raw[base + cc] - mx).exp();
                }
                let yi = c.y_idx[i];
                for cc in 0..k {
                    let p = (raw[base + cc] - mx).exp() / se;
                    let ind = if yi as usize == cc { T::ONE } else { T::ZERO };
                    gh[2 * (base + cc)] = p - ind;
                    gh[2 * (base + cc) + 1] = p * (T::ONE - p);
                }
            }
        }
    }
}

/// `gbt_hist`, loop-interchanged.
///
/// One task per `(tree, node_in_gather, block)`; the task zeroes its
/// `d · nb · 3` slice and scans its row stripe with the FEATURE loop inside
/// the ROW loop, so `binned`'s row is read contiguously. Each histogram slot
/// still sees exactly the device kernel's ascending row sequence.
#[allow(clippy::too_many_arguments)]
fn gather<T: HostFloat>(
    c: &Ctx<'_, T>,
    wid: usize,
    dst: Shared<T>,
    order: &Shared<u32>,
    ranges: &Shared<u32>,
    nodes_total: usize,
    node_base: usize,
    nc: usize,
    node_stride: usize,
    blocks: usize,
) {
    let (n, d, nb, k) = (c.n, c.d, c.nb, c.k);
    let elem = c.elem;
    let tasks = k * nc * blocks;
    let (lo, hi) = span(tasks, wid, c.workers);
    let order = order.get();
    let ranges = ranges.get();
    let gh_all = c.gh.get();
    // SAFETY: each task owns one `elem`-wide slice, disjoint across tasks.
    let out = unsafe { dst.get_mut() };
    for task in lo..hi {
        let blk = task % blocks;
        let tn = task / blocks;
        let node = tn % nc;
        let tt = tn / nc;
        let slice = &mut out[task * elem..task * elem + elem];
        slice.fill(T::ZERO);

        let rbase = (tt * nodes_total + node_base + node * node_stride) * 2;
        let s = ranges[rbase] as usize;
        let e = ranges[rbase + 1] as usize;
        let len = e - s;
        let r_lo = s + blk * len / blocks;
        let r_hi = s + (blk + 1) * len / blocks;
        for r in r_lo..r_hi {
            let i = order[tt * n + r] as usize;
            let gv = gh_all[2 * (i * k + tt)];
            let hv = gh_all[2 * (i * k + tt) + 1];
            let row = &c.binned[i * d..i * d + d];
            // One bounds check per (row, feature) instead of three: the
            // per-feature sub-histogram is carved out by `chunks_exact_mut`
            // (so the feature stride needs no check) and the bin's three
            // slots are taken as one length-3 sub-slice.
            for (fslice, &b) in slice.chunks_exact_mut(nb * 3).zip(row.iter()) {
                let t = &mut fslice[b as usize * 3..b as usize * 3 + 3];
                t[0] += T::ONE;
                t[1] += gv;
                t[2] += hv;
            }
        }
    }
}

/// The whole post-gather half of a level, as ONE phase.
///
/// A task is one GATHERED node column: it block-reduces and bin-cumsums that
/// column, and then — under sibling subtraction — expands it plus the retained
/// parent into the level's TWO child columns and splits both; otherwise it
/// splits its own single column. Every read and write a task makes belongs to
/// that task, so the kernel-level phase boundaries need no barrier between
/// them here (at these sizes a barrier costs more than the phase does).
#[allow(clippy::too_many_arguments)]
fn reduce_split<T: HostFloat>(
    c: &Ctx<'_, T>,
    wid: usize,
    hist_cur: Shared<T>,
    hist_par: Shared<T>,
    gathered: Shared<T>,
    nc_gather: usize,
    nc: usize,
    blocks: usize,
    single_block: bool,
    subtract: bool,
    tree_base: usize,
    level_base: usize,
    root_level: bool,
    force_leaf: bool,
) {
    let k = c.k;
    let cols = k * nc_gather;
    let (lo, hi) = span(cols, wid, c.workers);
    let parts = c.partials.get();

    // The two arms are written out separately because WITHOUT subtraction the
    // gathered buffer IS the level buffer: taking `&mut` to both at once would
    // be two aliasing mutable references to the same allocation (UB, and the
    // `noalias` LLVM emits for `&mut` makes it a real miscompilation risk),
    // even though every access is in-bounds and single-threaded per task.
    if subtract {
        // SAFETY: task `col` owns gathered column `col` and the two level
        // columns `2p` / `2p+1` of its own tree; `gathered` (`hist_left`),
        // `hist_cur` and `hist_par` are three DISTINCT allocations. Both are
        // read back only after this phase's barrier.
        let gath = unsafe { gathered.get_mut() };
        let full = unsafe { hist_cur.get_mut() };
        let parent = hist_par.get();
        let left_children = nc_gather;
        for col in lo..hi {
            reduce_cum_one(c, gath, parts, col, blocks, single_block);
            let tt = col / left_children;
            let p = col % left_children;
            subtract_one(c, full, gath, parent, col, tt, p, left_children, tree_base);
            let node = p * 2;
            let base = tt * nc + node;
            split_one(
                c, full, base, tt, node, tree_base, level_base, root_level, force_leaf,
            );
            split_one(
                c,
                full,
                base + 1,
                tt,
                node + 1,
                tree_base,
                level_base,
                root_level,
                force_leaf,
            );
        }
    } else {
        // SAFETY: task `col` owns level column `col`, disjoint across tasks;
        // `split_one` only reads it (an immutable reborrow).
        let full = unsafe { hist_cur.get_mut() };
        for col in lo..hi {
            reduce_cum_one(c, full, parts, col, blocks, single_block);
            let tt = col / nc;
            let node = col % nc;
            split_one(
                c, full, col, tt, node, tree_base, level_base, root_level, force_leaf,
            );
        }
    }
}

/// `gbt_hist_reduce` + `rf_hist_cum` for ONE column.
///
/// The block reduce is skipped at `blocks == 1` (the gather wrote the final
/// buffer directly) exactly as the device orchestration skips its launch.
#[inline]
fn reduce_cum_one<T: HostFloat>(
    c: &Ctx<'_, T>,
    out: &mut [T],
    parts: &[T],
    col: usize,
    blocks: usize,
    single_block: bool,
) {
    let nb = c.nb;
    let elem = c.elem;
    let dst = &mut out[col * elem..col * elem + elem];
    if !single_block {
        for (z, slot) in dst.iter_mut().enumerate() {
            let mut acc = T::ZERO;
            for b in 0..blocks {
                acc += parts[(col * blocks + b) * elem + z];
            }
            *slot = acc;
        }
    }
    // Cumulative over the bin axis. The three slot kinds are INDEPENDENT
    // accumulators, so running them together in one ascending sweep is the
    // same arithmetic as the kernel's three separate units — and touches the
    // feature's slice once instead of three times.
    for fslice in dst.chunks_exact_mut(nb * 3) {
        let (mut a0, mut a1, mut a2) = (T::ZERO, T::ZERO, T::ZERO);
        for bin in fslice.chunks_exact_mut(3) {
            a0 += bin[0];
            a1 += bin[1];
            a2 += bin[2];
            bin[0] = a0;
            bin[1] = a1;
            bin[2] = a2;
        }
    }
}

/// `gbt_hist_subtract` for ONE parent: copy the gathered LEFT child into the
/// even slot and derive the RIGHT sibling as `parent − left`, forced to zero
/// when the parent is a LEAF (both its children are phantom complete-tree
/// slots with an empty row range — see the kernel's doc).
#[allow(clippy::too_many_arguments)]
#[inline]
fn subtract_one<T: HostFloat>(
    c: &Ctx<'_, T>,
    full: &mut [T],
    left: &[T],
    parent: &[T],
    col: usize,
    tt: usize,
    p: usize,
    left_children: usize,
    tree_base: usize,
) {
    let elem = c.elem;
    let nodes_cur = left_children * 2;
    let parent_level_base = left_children - 1;
    let even_col = tt * nodes_cur + p * 2;
    let midx = (tree_base + tt) * c.total_nodes + parent_level_base + p;
    let parent_is_leaf = c.is_leaf.get()[midx] == 1;
    for e in 0..elem {
        let lv = left[col * elem + e];
        let pv = parent[col * elem + e];
        full[even_col * elem + e] = lv;
        full[(even_col + 1) * elem + e] = if parent_is_leaf { T::ZERO } else { pv - lv };
    }
}

/// `gbt_split_scores` + `gbt_best_split` for ONE node.
///
/// Fusing the two is exact: a node's gain slice is written and immediately
/// arg-maxed by the same task, and no other task reads it. The scan is the
/// kernel's flat `(feature, bin)` order with the same strict `>`, so ties keep
/// the lowest flat index.
#[allow(clippy::too_many_arguments)]
#[inline]
fn split_one<T: HostFloat>(
    c: &Ctx<'_, T>,
    hist: &[T],
    col: usize,
    tt: usize,
    node: usize,
    tree_base: usize,
    level_base: usize,
    root_level: bool,
    force_leaf: bool,
) {
    let (d, nb) = (c.d, c.nb);
    let elem = c.elem;
    let nsplit = nb - 1;
    // SAFETY: task-owned score slice and task-owned model-array node slot.
    let scores = unsafe { c.scores.get_mut() };
    let split_feature = unsafe { c.split_feature.get_mut() };
    let split_bin = unsafe { c.split_bin.get_mut() };
    let threshold = unsafe { c.threshold.get_mut() };
    let is_leaf = unsafe { c.is_leaf.get_mut() };
    let leaf_value = unsafe { c.leaf_value.get_mut() };

    let eps = T::lit(1e-15);
    let hbase = col * elem;
    let tbase = hbase + (nb - 1) * 3;
    let nt = hist[tbase];
    let gt = hist[tbase + 1];
    let ht = hist[tbase + 2];
    let loss_node = if root_level {
        T::ZERO
    } else {
        gt * gt / (ht + c.l2 + eps)
    };

    let sbase = col * d * nsplit;
    for f in 0..d {
        let fb = hbase + f * nb * 3;
        let sf = sbase + f * nsplit;
        for s in 0..nsplit {
            let nl = hist[fb + s * 3];
            let gl = hist[fb + s * 3 + 1];
            let hl = hist[fb + s * 3 + 2];
            let nr = nt - nl;
            let gr = gt - gl;
            let hr = ht - hl;
            let sc = if nl >= c.min_leaf
                && nr >= c.min_leaf
                && hl >= c.min_hessian
                && hr >= c.min_hessian
            {
                gl * gl / (hl + c.l2 + eps) + gr * gr / (hr + c.l2 + eps) - loss_node
            } else {
                T::lit(-1.0)
            };
            scores[sf + s] = sc;
        }
    }

    let mut best = T::lit(-1.0);
    let mut bk = 0usize;
    for i in 0..d * nsplit {
        let sc = scores[sbase + i];
        if sc > best {
            best = sc;
            bk = i;
        }
    }

    let gnode = level_base + node;
    let midx = (tree_base + tt) * c.total_nodes + gnode;
    leaf_value[midx] = -c.lr * gt / (ht + c.l2 + eps);
    let leaf = force_leaf || best <= T::ZERO;
    is_leaf[midx] = u32::from(leaf);
    if leaf {
        split_feature[midx] = u32::MAX;
        split_bin[midx] = 0;
        threshold[midx] = T::ZERO;
    } else {
        let bf = bk / nsplit;
        let bs = bk % nsplit;
        split_feature[midx] = bf as u32;
        split_bin[midx] = bs as u32;
        threshold[midx] = c.edges[bf * nsplit + bs];
    }
}

/// `gbt_count_left_blocks` → `gbt_child_ranges` → `gbt_partition_blocks`.
///
/// Pure index work — no float arithmetic — and the blocked scatter is
/// documented to reproduce the serial partition's `order_next` bitwise for any
/// block count, so the host picks `bcount` for its own pool.
#[allow(clippy::too_many_arguments)]
fn partition_level<T: HostFloat>(
    c: &Ctx<'_, T>,
    wid: usize,
    order: &Shared<u32>,
    order_next: &Shared<u32>,
    ranges: &Shared<u32>,
    ranges_next: &Shared<u32>,
    nodes: usize,
    tree_base: usize,
    level_base: usize,
) {
    let (n, k) = (c.n, c.k);
    let units = k * nodes;
    let bcount = partition_blocks_host(units, n, nodes, c.workers);
    let order_in = order.get();
    let ranges_in = ranges.get();
    let split_feature = c.split_feature.get();
    let split_bin = c.split_bin.get();
    let is_leaf = c.is_leaf.get();

    // Once the level has a node per worker the row blocking has nothing left
    // to parallelize, and a task can do its node's count, child ranges and
    // scatter back to back — ONE phase instead of three (two barriers saved
    // per level, and a barrier is the dominant cost at these sizes).
    if bcount == 1 {
        let (lo, hi) = span(units, wid, c.workers);
        // SAFETY: task `tn` owns its node's two child range slots and the
        // `[s, e)` output sub-range of `order_next` — disjoint across tasks.
        let rn = unsafe { ranges_next.get_mut() };
        let on = unsafe { order_next.get_mut() };
        for tn in lo..hi {
            let node = tn % nodes;
            let tt = tn / nodes;
            let midx = (tree_base + tt) * c.total_nodes + level_base + node;
            let lbase = (tt * nodes * 2 + node * 2) * 2;
            if is_leaf[midx] == 1 {
                rn[lbase] = 0;
                rn[lbase + 1] = 0;
                rn[lbase + 2] = 0;
                rn[lbase + 3] = 0;
                continue;
            }
            let fr = split_feature[midx] as usize;
            let bs = split_bin[midx] as u8;
            let s = ranges_in[(tt * nodes + node) * 2] as usize;
            let e = ranges_in[(tt * nodes + node) * 2 + 1] as usize;
            let colf = &c.binned_t[fr * n..fr * n + n];
            let mut cnt = 0u32;
            for r in s..e {
                if colf[order_in[tt * n + r] as usize] <= bs {
                    cnt += 1;
                }
            }
            rn[lbase] = s as u32;
            rn[lbase + 1] = s as u32 + cnt;
            rn[lbase + 2] = s as u32 + cnt;
            rn[lbase + 3] = e as u32;
            let mut li = s;
            let mut ri = s + cnt as usize;
            for r in s..e {
                let i = order_in[tt * n + r];
                if colf[i as usize] <= bs {
                    on[tt * n + li] = i;
                    li += 1;
                } else {
                    on[tt * n + ri] = i;
                    ri += 1;
                }
            }
        }
        let _ = c.bar.wait();
        return;
    }

    // Phase D: per-block left counts.
    {
        let tasks = units * bcount;
        let (lo, hi) = span(tasks, wid, c.workers);
        // SAFETY: one `u32` per task, disjoint.
        let blk_cnt = unsafe { c.blk_cnt.get_mut() };
        for task in lo..hi {
            let blk = task % bcount;
            let tn = task / bcount;
            let node = tn % nodes;
            let tt = tn / nodes;
            let midx = (tree_base + tt) * c.total_nodes + level_base + node;
            let mut cnt = 0u32;
            if is_leaf[midx] == 0 {
                let fr = split_feature[midx] as usize;
                let bs = split_bin[midx] as u8;
                let s = ranges_in[(tt * nodes + node) * 2] as usize;
                let e = ranges_in[(tt * nodes + node) * 2 + 1] as usize;
                let (r_lo, r_hi) = block_span(s, e, blk, bcount);
                let colf = &c.binned_t[fr * n..fr * n + n];
                for r in r_lo..r_hi {
                    if colf[order_in[tt * n + r] as usize] <= bs {
                        cnt += 1;
                    }
                }
            }
            blk_cnt[task] = cnt;
        }
    }
    if !c.bar.wait() {
        return;
    }

    // Phase E: in-place exclusive prefix per node + the two child ranges.
    {
        let (lo, hi) = span(units, wid, c.workers);
        // SAFETY: task `tn` owns its own `bcount` counts and its own two
        // child range slots.
        let blk_cnt = unsafe { c.blk_cnt.get_mut() };
        let rn = unsafe { ranges_next.get_mut() };
        for tn in lo..hi {
            let node = tn % nodes;
            let tt = tn / nodes;
            let midx = (tree_base + tt) * c.total_nodes + level_base + node;
            let lbase = (tt * nodes * 2 + node * 2) * 2;
            if is_leaf[midx] == 1 {
                rn[lbase] = 0;
                rn[lbase + 1] = 0;
                rn[lbase + 2] = 0;
                rn[lbase + 3] = 0;
            } else {
                let cbase = tn * bcount;
                let mut run = 0u32;
                for b in 0..bcount {
                    let v = blk_cnt[cbase + b];
                    blk_cnt[cbase + b] = run;
                    run += v;
                }
                let s = ranges_in[(tt * nodes + node) * 2];
                let e = ranges_in[(tt * nodes + node) * 2 + 1];
                rn[lbase] = s;
                rn[lbase + 1] = s + run;
                rn[lbase + 2] = s + run;
                rn[lbase + 3] = e;
            }
        }
    }
    if !c.bar.wait() {
        return;
    }

    // Phase F: stable blocked scatter.
    {
        let tasks = units * bcount;
        let (lo, hi) = span(tasks, wid, c.workers);
        let blk_cnt = c.blk_cnt.get();
        let rn = ranges_next.get();
        // SAFETY: blocks write disjoint output sub-ranges (the
        // `gbt_partition_blocks` prefix argument).
        let on = unsafe { order_next.get_mut() };
        for task in lo..hi {
            let blk = task % bcount;
            let tn = task / bcount;
            let node = tn % nodes;
            let tt = tn / nodes;
            let midx = (tree_base + tt) * c.total_nodes + level_base + node;
            if is_leaf[midx] == 1 {
                continue;
            }
            let fr = split_feature[midx] as usize;
            let bs = split_bin[midx] as u8;
            let s = ranges_in[(tt * nodes + node) * 2] as usize;
            let e = ranges_in[(tt * nodes + node) * 2 + 1] as usize;
            let (r_lo, r_hi) = block_span(s, e, blk, bcount);
            let lbase = (tt * nodes * 2 + node * 2) * 2;
            let mid = rn[lbase + 1] as usize;
            let pfx = blk_cnt[tn * bcount + blk] as usize;
            let per = (e - s).div_ceil(bcount);
            let mut li = s + pfx;
            let mut ri = mid + (blk * per - pfx);
            let colf = &c.binned_t[fr * n..fr * n + n];
            for r in r_lo..r_hi {
                let i = order_in[tt * n + r];
                if colf[i as usize] <= bs {
                    on[tt * n + li] = i;
                    li += 1;
                } else {
                    on[tt * n + ri] = i;
                    ri += 1;
                }
            }
        }
    }
    let _ = c.bar.wait();
}

/// The `ceil`-strided row block of `[s, e)` — the `gbt_count_left_blocks` /
/// `gbt_partition_blocks` bound (a trailing block can be empty).
#[inline(always)]
fn block_span(s: usize, e: usize, blk: usize, bcount: usize) -> (usize, usize) {
    let per = (e - s).div_ceil(bcount);
    let lo = s + blk * per;
    let hi = (lo + per).min(e);
    (lo, hi.max(lo))
}

/// `gbt_update_raw`: fold this iteration's `k` trees into the raw predictions
/// by walking the BINNED features (exactly the training partition rule).
fn update_raw<T: HostFloat>(c: &Ctx<'_, T>, wid: usize, tree_base: usize) {
    let (n, d, k) = (c.n, c.d, c.k);
    let (lo, hi) = span(n, wid, c.workers);
    let split_feature = c.split_feature.get();
    let split_bin = c.split_bin.get();
    let is_leaf = c.is_leaf.get();
    let leaf_value = c.leaf_value.get();
    // SAFETY: disjoint row stripes.
    let raw = unsafe { c.raw.get_mut() };
    for i in lo..hi {
        let row = &c.binned[i * d..i * d + d];
        for cc in 0..k {
            let tbase = (tree_base + cc) * c.total_nodes;
            let mut cur = 0usize;
            for _ in 0..c.depth {
                if is_leaf[tbase + cur] == 0 {
                    let fr = split_feature[tbase + cur] as usize;
                    let bs = split_bin[tbase + cur] as u8;
                    cur = if row[fr] <= bs {
                        2 * cur + 1
                    } else {
                        2 * cur + 2
                    };
                }
            }
            raw[i * k + cc] += leaf_value[tbase + cur];
        }
    }
}
