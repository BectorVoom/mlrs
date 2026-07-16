//! `gbt` — HistGradientBoosting device kernels (GBT-01): sequential gradient
//! boosting over batched level-wise histogram trees, generic over `<F: Float +
//! CubeElement>` with NO backend feature (D-13).
//!
//! ## Design (sklearn `HistGradientBoosting*`, on the `tree.rs` chassis)
//! One boosting iteration = compute per-sample gradients/hessians from the
//! current raw predictions, then grow `K` trees SIMULTANEOUSLY (`K = 1` for
//! regression / binary log-loss, `K = n_classes` for multiclass — sklearn's
//! `n_trees_per_iteration_`), then fold the shrunk leaf values back into the
//! raw predictions — all launch-only (zero device→host readbacks, the
//! `sgd_solve` / `random_forest` perf lesson).
//!
//! Split semantics are the sklearn/XGBoost histogram form (splitting.pyx):
//! - histogram slots per bin: `count`, `Σ gradient`, `Σ hessian` (`ncs = 3`);
//! - node value `v = −G/(H + λ + 1e-15)`, node loss `v·G`;
//! - split gain `= loss(node) − loss(left) − loss(right)`; the ROOT of every
//!   tree has `value = 0` ⇒ `loss = 0` (sklearn `TreeNode(value=0)`);
//! - a split is VALID iff both children have `count ≥ min_samples_leaf` and
//!   `Σ hessian ≥ min_hessian_to_split`; a node is a LEAF iff the best valid
//!   gain is `≤ 0` (sklearn `gain <= 0 → _finalize_leaf`) or the depth bound
//!   forces it;
//! - leaf value stored SHRUNK: `−learning_rate · G/(H + λ + 1e-15)`.
//!
//! Unlike the Random Forest there is NO feature subsampling, NO bootstrap and
//! NO RNG anywhere — fits are bit-deterministic across runs and backends.
//!
//! ## Root-level parallelism (the RF memory's optimization lever)
//! With only `K` trees per launch, per-(node, feature) histogram units would
//! leave the device idle at shallow levels (level 0 = `K·d` units scanning all
//! `n` rows). The gather is therefore ROW-BLOCKED: [`gbt_hist`] writes
//! `n_blocks` partial histograms per (tree, node, feature) — each unit owns
//! one block slice exclusively — and [`gbt_hist_reduce`] sums the block axis.
//!
//! ## cpu-MLIR safety (the primary correctness gate)
//! Every kernel stays inside the proven op-set (spike findings 001/002/003,
//! see `tree.rs`): bare-`ABSOLUTE_POS` 1D launches with `if tid < total`
//! guards, ascending `while` scans, ≤ 2 coupled loop-carried `F` accumulators,
//! cross-loop values staged through GLOBAL arrays (`row_max` / `row_sumexp`),
//! statement-form `if`, no `SharedMemory`/atomics/`F::INFINITY`. Each unit
//! writes only memory it exclusively owns (its own histogram-block / score /
//! node / raw slice) — race-free without atomics (the `lbfgs.rs` precedent).
//!
//! LANDMINE (wgpu): small constants MUST be `F::new(1e-15)`, never
//! `F::cast_from(1e-15)` — the f64-literal cast makes the WGSL shader fail to
//! compile SILENTLY on wgpu (the kernel launches as a no-op and every score
//! reads back 0). `F::new` takes an f32 literal and lowers cleanly.
//!
//! ## Buffer contract (row-major, indices `u32`, values `F`)
//! - `binned` — `n × d` `u32` bin index per (row, feature) (`tree.rs`
//!   [`rf_bin_features`](crate::tree::rf_bin_features) output).
//! - `raw` — `n × K` current raw predictions (baseline + Σ tree values).
//! - `g` / `h` — `n × K` per-sample gradients / hessians.
//! - `order` / `ranges` — the `tree.rs` per-tree row partition, `K` trees wide
//!   (per-ITERATION scratch, reset by [`gbt_init_partition`]).
//! - `hist_part` — `K × nodes_chunk × d × n_blocks × nb × 3` partial counts;
//!   `hist` — the block-reduced `K × nodes_chunk × d × nb × 3`, made
//!   CUMULATIVE over the bin axis by [`rf_hist_cum`](crate::tree::rf_hist_cum)
//!   (`ncs = 3`).
//! - model arrays — `(n_iters · K) × total_nodes` complete-tree layout
//!   (`tree.rs` convention); kernels take a `tree_base` (= `iter · K`) so one
//!   allocation holds every stage.
//!
//! Tests live in `crates/mlrs-backend/tests/hist_gradient_boosting_test.rs`
//! (this crate is feature-free and cannot launch; AGENTS.md §2).

use cubecl::prelude::*;

/// Squared-error gradients: `g = raw − y`, `h = 1` (sklearn
/// `HalfSquaredError`; the constant hessian makes `Σh` the sample count).
/// One unit per sample — a pure per-element map.
#[cube(launch)]
pub fn gbt_grad_reg<F: Float + CubeElement>(
    raw: &Array<F>,
    y: &Array<F>,
    g: &mut Array<F>,
    h: &mut Array<F>,
    n: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < n as usize {
        g[tid] = raw[tid] - y[tid];
        h[tid] = F::new(1.0);
    }
}

/// Binary log-loss gradients: `p = σ(raw)`, `g = p − y`, `h = p·(1 − p)`
/// (sklearn `HalfBinomialLoss`; `y ∈ {0, 1}` as `F`). One unit per sample.
#[cube(launch)]
pub fn gbt_grad_binary<F: Float + CubeElement>(
    raw: &Array<F>,
    y: &Array<F>,
    g: &mut Array<F>,
    h: &mut Array<F>,
    n: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < n as usize {
        let p = F::new(1.0) / (F::new(1.0) + F::exp(-raw[tid]));
        g[tid] = p - y[tid];
        h[tid] = p * (F::new(1.0) - p);
    }
}

/// Per-row max of the `n × k` raw scores → `row_max` (softmax staging: the
/// cross-loop value goes through GLOBAL memory — FINDING 002-B). Running-max
/// statement-form `if`, one unit per row.
#[cube(launch)]
pub fn gbt_row_max<F: Float + CubeElement>(
    raw: &Array<F>,
    row_max: &mut Array<F>,
    n: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < n as usize {
        let base = (tid as u32) * k;
        let mut mx = raw[base as usize];
        let mut c = 1u32;
        while c < k {
            let v = raw[(base + c) as usize];
            if v > mx {
                mx = v;
            }
            c += 1u32;
        }
        row_max[tid] = mx;
    }
}

/// Per-row `Σ_c exp(raw − row_max)` → `row_sumexp` (softmax staging pass 2;
/// reads the GLOBAL `row_max`, single `F` accumulator). One unit per row.
#[cube(launch)]
pub fn gbt_row_sumexp<F: Float + CubeElement>(
    raw: &Array<F>,
    row_max: &Array<F>,
    row_sumexp: &mut Array<F>,
    n: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < n as usize {
        let base = (tid as u32) * k;
        let mx = row_max[tid];
        let mut acc = F::new(0.0);
        let mut c = 0u32;
        while c < k {
            acc += F::exp(raw[(base + c) as usize] - mx);
            c += 1u32;
        }
        row_sumexp[tid] = acc;
    }
}

/// Multiclass log-loss gradients: `p = softmax(raw)` (from the staged
/// `row_max` / `row_sumexp`), `g = p − 1{y = c}`, `h = p·(1 − p)` (sklearn
/// `HalfMultinomialLoss`). One unit per `(row, class)` element — no loops.
#[cube(launch)]
pub fn gbt_grad_multi<F: Float + CubeElement>(
    raw: &Array<F>,
    row_max: &Array<F>,
    row_sumexp: &Array<F>,
    y_idx: &Array<u32>,
    g: &mut Array<F>,
    h: &mut Array<F>,
    n: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n * k;
    if tid < total as usize {
        let i = (tid as u32) / k;
        let c = (tid as u32) % k;
        let p = F::exp(raw[tid] - row_max[i as usize]) / row_sumexp[i as usize];
        let mut ind = F::new(0.0);
        if y_idx[i as usize] == c {
            ind = F::new(1.0);
        }
        g[tid] = p - ind;
        h[tid] = p * (F::new(1.0) - p);
    }
}

/// Initialize the `n × k` raw predictions to the per-class baseline (sklearn
/// `loss.fit_intercept_only`, host-computed). One unit per element.
#[cube(launch)]
pub fn gbt_init_raw<F: Float + CubeElement>(
    baseline: &Array<F>,
    raw: &mut Array<F>,
    n: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n * k;
    if tid < total as usize {
        let c = (tid as u32) % k;
        raw[tid] = baseline[c as usize];
    }
}

/// Reset the per-iteration row partition: identity `order` per tree and the
/// level-0 root range `[0, n)` per tree (zero uploads — the launch-only
/// discipline). One unit per `(tree, row)`; the first `k` units also write
/// their tree's root range (exclusively owned — no race).
#[cube(launch)]
pub fn gbt_init_partition(order: &mut Array<u32>, ranges: &mut Array<u32>, n: u32, k: u32) {
    let tid = ABSOLUTE_POS;
    let total = k * n;
    if tid < total as usize {
        order[tid] = (tid as u32) % n;
        if (tid as u32) < k {
            ranges[(2u32 * (tid as u32)) as usize] = 0u32;
            ranges[(2u32 * (tid as u32) + 1u32) as usize] = n;
        }
    }
}

/// Row-BLOCKED histogram gather: one unit per `(tree, node_in_chunk, feature,
/// block)` zeroes then accumulates its OWN `nb × 3` partial slice
/// (`count`, `Σg`, `Σh`) over its stripe of the node's row range. Exclusive
/// slice ownership keeps the three `+=` read-modify-writes race-free without
/// atomics (the `lbfgs.rs` / `rf_hist_*` precedent).
///
/// `nodes_total` is the FULL level width (ranges indexing); `node_base` +
/// `nodes_chunk` select the node chunk this launch covers (deep-level memory
/// chunking, the `RF_HIST_BUDGET_BYTES` discipline).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_hist<F: Float + CubeElement>(
    binned: &Array<u32>,
    g: &Array<F>,
    h: &Array<F>,
    order: &Array<u32>,
    ranges: &Array<u32>,
    hist_part: &mut Array<F>,
    n: u32,
    d: u32,
    nb: u32,
    k: u32,
    nodes_total: u32,
    node_base: u32,
    nodes_chunk: u32,
    n_blocks: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * nodes_chunk * d * n_blocks;
    if tid < total as usize {
        let tt = (tid as u32) / (nodes_chunk * d * n_blocks);
        let rem = (tid as u32) % (nodes_chunk * d * n_blocks);
        let node = rem / (d * n_blocks);
        let rem2 = rem % (d * n_blocks);
        let f = rem2 / n_blocks;
        let blk = rem2 % n_blocks;

        let base = (tid as u32) * (nb * 3u32);
        // Zero the exclusively-owned slice (pool buffers arrive uninitialized).
        let mut z = 0u32;
        while z < nb * 3u32 {
            hist_part[(base + z) as usize] = F::new(0.0);
            z += 1u32;
        }

        let rbase = (tt * nodes_total + node_base + node) * 2u32;
        let s = ranges[rbase as usize];
        let e = ranges[(rbase + 1u32) as usize];
        let len = e - s;
        let lo = s + (blk * len) / n_blocks;
        let hi = s + ((blk + 1u32) * len) / n_blocks;
        let mut r = lo;
        while r < hi {
            let i = order[(tt * n + r) as usize];
            let b = binned[(i * d + f) as usize];
            let slot = base + b * 3u32;
            hist_part[slot as usize] += F::new(1.0);
            hist_part[(slot + 1u32) as usize] += g[(i * k + tt) as usize];
            hist_part[(slot + 2u32) as usize] += h[(i * k + tt) as usize];
            r += 1u32;
        }
    }
}

/// Reduce the block axis of [`gbt_hist`]'s partial histograms: one unit per
/// `(tree, node_in_chunk, feature, bin, slot)` sums its `n_blocks` strided
/// partials with a single `F` accumulator.
#[cube(launch)]
pub fn gbt_hist_reduce<F: Float + CubeElement>(
    hist_part: &Array<F>,
    hist: &mut Array<F>,
    nb: u32,
    cols: u32,
    n_blocks: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = cols * nb * 3u32;
    if tid < total as usize {
        let col = (tid as u32) / (nb * 3u32); // (tt, node, f) flat index
        let within = (tid as u32) % (nb * 3u32);
        let mut acc = F::new(0.0);
        let mut b = 0u32;
        while b < n_blocks {
            acc += hist_part[((col * n_blocks + b) * (nb * 3u32) + within) as usize];
            b += 1u32;
        }
        hist[tid] = acc;
    }
}

/// Split gains over the CUMULATIVE 3-slot histogram: one unit per `(tree,
/// node_in_chunk, feature, split bin s)` writes the sklearn gain
/// `G_l²/(H_l+λ+ε) + G_r²/(H_r+λ+ε) − loss_node` (`ε = 1e-15`,
/// `loss_node = G²/(H+λ+ε)`, or `0` on the ROOT level — sklearn
/// `TreeNode(value=0)`), or `−1` when a child violates `count ≥ min_leaf` or
/// `Σh ≥ min_hessian`. Straight-line global reads — no loops at all (the
/// `rf_split_scores_reg` shape).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_split_scores<F: Float + CubeElement>(
    hist: &Array<F>,
    scores: &mut Array<F>,
    min_leaf: F,
    min_hessian: F,
    l2: F,
    d: u32,
    nb: u32,
    nodes_chunk: u32,
    k: u32,
    root_level: u32,
) {
    let tid = ABSOLUTE_POS;
    let nsplit = nb - 1u32;
    let total = k * nodes_chunk * d * nsplit;
    if tid < total as usize {
        let col = (tid as u32) / nsplit; // (tt, node, f) flat
        let s = (tid as u32) % nsplit;
        let tn = col / d; // (tt, node) flat
        let fbase = col * (nb * 3u32);
        let tbase = (tn * d) * (nb * 3u32) + (nb - 1u32) * 3u32; // feature 0 totals

        let eps = F::new(1e-15);
        let nl = hist[(fbase + s * 3u32) as usize];
        let gl = hist[(fbase + s * 3u32 + 1u32) as usize];
        let hl = hist[(fbase + s * 3u32 + 2u32) as usize];
        let nt = hist[tbase as usize];
        let gt = hist[(tbase + 1u32) as usize];
        let ht = hist[(tbase + 2u32) as usize];
        let nr = nt - nl;
        let gr = gt - gl;
        let hr = ht - hl;

        let mut loss_node = F::new(0.0);
        if root_level == 0u32 {
            loss_node = gt * gt / (ht + l2 + eps);
        }

        let mut sc = F::new(-1.0);
        if nl >= min_leaf {
            if nr >= min_leaf {
                if hl >= min_hessian {
                    if hr >= min_hessian {
                        sc = gl * gl / (hl + l2 + eps) + gr * gr / (hr + l2 + eps) - loss_node;
                    }
                }
            }
        }
        scores[tid] = sc;
    }
}

/// Per-node split finalize: one unit per `(tree, node_in_chunk)` arg-maxes its
/// `d × (nb−1)` gain slice (strict `>` → lowest-(feature, bin) tie-break, the
/// sklearn scan order), decides leaf-ness, and writes the model arrays at the
/// node's GLOBAL id:
///
/// leaf ⇔ `force_leaf` (bottom level) ∨ best gain `≤ 0` (sklearn
/// `gain <= 0 → _finalize_leaf`; all-invalid slices read `−1`).
///
/// `leaf_value` is ALWAYS written — the SHRUNK sklearn node value
/// `−lr · G/(H + λ + 1e-15)` (interior values are never read; an empty node
/// yields `0`). `level_base` must already fold in the node-chunk offset
/// (`2^L − 1 + node_base`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_best_split<F: Float + CubeElement>(
    hist: &Array<F>,
    scores: &Array<F>,
    edges: &Array<F>,
    split_feature: &mut Array<u32>,
    split_bin: &mut Array<u32>,
    threshold: &mut Array<F>,
    is_leaf: &mut Array<u32>,
    leaf_value: &mut Array<F>,
    lr: F,
    l2: F,
    d: u32,
    nb: u32,
    nodes_chunk: u32,
    k: u32,
    tree_base: u32,
    level_base: u32,
    total_nodes: u32,
    force_leaf: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * nodes_chunk;
    if tid < total as usize {
        let tt = (tid as u32) / nodes_chunk;
        let node = (tid as u32) % nodes_chunk;
        let gnode = level_base + node;
        let midx = (tree_base + tt) * total_nodes + gnode;

        // Running best over the flat (f, s) gain slice — strict `>` keeps the
        // lowest flat index on ties (the sklearn feature-then-bin scan order).
        let nsplit = nb - 1u32;
        let sbase = (tid as u32) * d * nsplit;
        let mut best = F::new(-1.0);
        let mut bk = 0u32;
        let mut i = 0u32;
        while i < d * nsplit {
            let sc = scores[(sbase + i) as usize];
            if sc > best {
                best = sc;
                bk = i;
            }
            i += 1u32;
        }

        // Node totals from the cumulative histogram (feature 0, last bin).
        let tbase = ((tid as u32) * d) * (nb * 3u32) + (nb - 1u32) * 3u32;
        let gt = hist[(tbase + 1u32) as usize];
        let ht = hist[(tbase + 2u32) as usize];
        let eps = F::new(1e-15);
        leaf_value[midx as usize] = -lr * gt / (ht + l2 + eps);

        let mut leaf = 0u32;
        if force_leaf == 1u32 {
            leaf = 1u32;
        }
        if best <= F::new(0.0) {
            leaf = 1u32;
        }
        is_leaf[midx as usize] = leaf;

        if leaf == 0u32 {
            let bf = bk / nsplit;
            let bs = bk % nsplit;
            split_feature[midx as usize] = bf;
            split_bin[midx as usize] = bs;
            threshold[midx as usize] = edges[(bf * nsplit + bs) as usize];
        } else {
            split_feature[midx as usize] = 0xFFFF_FFFFu32;
            split_bin[midx as usize] = 0u32;
            threshold[midx as usize] = F::new(0.0);
        }
    }
}

/// Children `[start, end)` ranges for the NEXT level (the `rf_count_left`
/// shape with a `tree_base` model offset): one unit per `(tree, node)` counts
/// its left rows (`bin ≤ split_bin`, single `u32` accumulator) and writes the
/// two child ranges; leaves emit two EMPTY child ranges.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_count_left(
    binned: &Array<u32>,
    order: &Array<u32>,
    ranges: &Array<u32>,
    split_feature: &Array<u32>,
    split_bin: &Array<u32>,
    is_leaf: &Array<u32>,
    ranges_next: &mut Array<u32>,
    n: u32,
    d: u32,
    nodes: u32,
    k: u32,
    tree_base: u32,
    level_base: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * nodes;
    if tid < total as usize {
        let tt = (tid as u32) / nodes;
        let node = (tid as u32) % nodes;
        let gnode = level_base + node;
        let midx = (tree_base + tt) * total_nodes + gnode;
        let next_nodes = nodes * 2u32;
        let lbase = (tt * next_nodes + node * 2u32) * 2u32;

        if is_leaf[midx as usize] == 1u32 {
            ranges_next[lbase as usize] = 0u32;
            ranges_next[(lbase + 1u32) as usize] = 0u32;
            ranges_next[(lbase + 2u32) as usize] = 0u32;
            ranges_next[(lbase + 3u32) as usize] = 0u32;
        } else {
            let fr = split_feature[midx as usize];
            let bs = split_bin[midx as usize];
            let s = ranges[((tt * nodes + node) * 2u32) as usize];
            let e = ranges[((tt * nodes + node) * 2u32 + 1u32) as usize];
            let mut cnt = 0u32;
            let mut r = s;
            while r < e {
                let i = order[(tt * n + r) as usize];
                if binned[(i * d + fr) as usize] <= bs {
                    cnt += 1u32;
                }
                r += 1u32;
            }
            ranges_next[lbase as usize] = s;
            ranges_next[(lbase + 1u32) as usize] = s + cnt;
            ranges_next[(lbase + 2u32) as usize] = s + cnt;
            ranges_next[(lbase + 3u32) as usize] = e;
        }
    }
}

/// Stable two-way partition into `order_next` (the `rf_partition` shape with a
/// `tree_base` model offset): one unit per `(tree, node)` re-scans its range
/// and scatters rows to the child ranges from [`gbt_count_left`] (read from
/// GLOBAL `ranges_next` — FINDING 002-B). Each parent owns its output range
/// exclusively — race-free without atomics.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_partition(
    binned: &Array<u32>,
    order: &Array<u32>,
    ranges: &Array<u32>,
    ranges_next: &Array<u32>,
    split_feature: &Array<u32>,
    split_bin: &Array<u32>,
    is_leaf: &Array<u32>,
    order_next: &mut Array<u32>,
    n: u32,
    d: u32,
    nodes: u32,
    k: u32,
    tree_base: u32,
    level_base: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * nodes;
    if tid < total as usize {
        let tt = (tid as u32) / nodes;
        let node = (tid as u32) % nodes;
        let gnode = level_base + node;
        let midx = (tree_base + tt) * total_nodes + gnode;

        if is_leaf[midx as usize] == 0u32 {
            let fr = split_feature[midx as usize];
            let bs = split_bin[midx as usize];
            let s = ranges[((tt * nodes + node) * 2u32) as usize];
            let e = ranges[((tt * nodes + node) * 2u32 + 1u32) as usize];
            let next_nodes = nodes * 2u32;
            let lbase = (tt * next_nodes + node * 2u32) * 2u32;
            let mut li = ranges_next[lbase as usize];
            let mut ri = ranges_next[(lbase + 2u32) as usize];
            let mut r = s;
            while r < e {
                let i = order[(tt * n + r) as usize];
                if binned[(i * d + fr) as usize] <= bs {
                    order_next[(tt * n + li) as usize] = i;
                    li += 1u32;
                } else {
                    order_next[(tt * n + ri) as usize] = i;
                    ri += 1u32;
                }
                r += 1u32;
            }
        }
    }
}

/// Fold this iteration's trees into the TRAIN raw predictions: one unit per
/// `(row, class)` walks tree `tree_base + c` on the BINNED features (bounded
/// `max_depth` descent, `bin ≤ split_bin → left` — exactly the training
/// partition rule) and adds the reached leaf's shrunk value to `raw[i, c]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_update_raw<F: Float + CubeElement>(
    binned: &Array<u32>,
    split_feature: &Array<u32>,
    split_bin: &Array<u32>,
    is_leaf: &Array<u32>,
    leaf_value: &Array<F>,
    raw: &mut Array<F>,
    n: u32,
    d: u32,
    k: u32,
    max_depth: u32,
    tree_base: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n * k;
    if tid < total as usize {
        let i = (tid as u32) / k;
        let c = (tid as u32) % k;
        let tbase = (tree_base + c) * total_nodes;
        let mut cur = 0u32;
        let mut l = 0u32;
        while l < max_depth {
            if is_leaf[(tbase + cur) as usize] == 0u32 {
                let fr = split_feature[(tbase + cur) as usize];
                let bs = split_bin[(tbase + cur) as usize];
                let mut nxt = 2u32 * cur + 2u32;
                if binned[(i * d + fr) as usize] <= bs {
                    nxt = 2u32 * cur + 1u32;
                }
                cur = nxt;
            }
            l += 1u32;
        }
        raw[tid] += leaf_value[(tbase + cur) as usize];
    }
}

/// Predict-time raw-score sum: one unit per `(query row, class)` accumulates
/// `baseline[c] + Σ_iter leaf_value[iter·k + c]` over the leaves reached by
/// [`rf_predict_leaf`](crate::tree::rf_predict_leaf) (`leaf` is `(n_iters·k) ×
/// q`, the `rf_mean_reg` shape). Single `F` accumulator.
#[cube(launch)]
pub fn gbt_sum_raw<F: Float + CubeElement>(
    leaf: &Array<u32>,
    leaf_value: &Array<F>,
    baseline: &Array<F>,
    raw_out: &mut Array<F>,
    q: u32,
    k: u32,
    n_iters: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = q * k;
    if tid < total as usize {
        let row = (tid as u32) / k;
        let c = (tid as u32) % k;
        let mut acc = baseline[c as usize];
        let mut it = 0u32;
        while it < n_iters {
            let t = it * k + c;
            let lf = leaf[(t * q + row) as usize];
            acc += leaf_value[(t * total_nodes + lf) as usize];
            it += 1u32;
        }
        raw_out[tid] = acc;
    }
}

/// Binary predict_proba: `p = σ(raw)` per query row → `[1 − p, p]` (the
/// sklearn two-column form). One unit per row.
#[cube(launch)]
pub fn gbt_proba_binary<F: Float + CubeElement>(raw: &Array<F>, proba: &mut Array<F>, q: u32) {
    let tid = ABSOLUTE_POS;
    if tid < q as usize {
        let p = F::new(1.0) / (F::new(1.0) + F::exp(-raw[tid]));
        proba[(2u32 * (tid as u32)) as usize] = F::new(1.0) - p;
        proba[(2u32 * (tid as u32) + 1u32) as usize] = p;
    }
}

/// Multiclass predict_proba: `softmax(raw)` per `(row, class)` element from
/// the staged `row_max` / `row_sumexp` ([`gbt_row_max`] / [`gbt_row_sumexp`]
/// over the `q × k` raw scores). No loops.
#[cube(launch)]
pub fn gbt_proba_multi<F: Float + CubeElement>(
    raw: &Array<F>,
    row_max: &Array<F>,
    row_sumexp: &Array<F>,
    proba: &mut Array<F>,
    q: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = q * k;
    if tid < total as usize {
        let i = (tid as u32) / k;
        proba[tid] = F::exp(raw[tid] - row_max[i as usize]) / row_sumexp[i as usize];
    }
}
