//! `tree` — Random Forest device kernels (ENSEMBLE-01): batched level-wise
//! decision-tree building + forest inference, generic over `<F: Float +
//! CubeElement>` with NO backend feature (D-13).
//!
//! ## Design (cuML-style level-wise histogram builder, mlrs-safe)
//! All `n_trees` trees are grown SIMULTANEOUSLY, one depth level per pass, so
//! the host fit loop is LAUNCH-ONLY (zero device→host readbacks inside the
//! loop — the `sgd_solve` perf lesson). Rows are kept partitioned per node via
//! a per-tree row-order array + per-node contiguous `[start, end)` ranges (the
//! cuML row-partition idea), so histogram work per level is
//! `O(n_trees · max_features · n)` regardless of the node count.
//!
//! Trees use the COMPLETE binary-tree layout: global node id `g` at depth `L`
//! is `2^L − 1 + local`, children of `g` are `2g+1` / `2g+2`. Splits are BINNED:
//! the host precomputes per-feature quantile-midpoint bin edges; a split is
//! "bin ≤ s" ⇔ "x < edges[s]" (edges never coincide with data values, so the
//! `<` rule is equivalent to sklearn's `<=`-midpoint rule).
//!
//! ## cpu-MLIR safety (the primary correctness gate)
//! Every kernel stays inside the proven op-set (spike findings 001/002/003):
//! - kmeans-style bare-`ABSOLUTE_POS` 1D launch with an `if tid < total` guard
//!   and ascending `while` scans (launch-proven by `lloyd_test.rs`);
//! - only `F`/`u32` accumulators, statement-form `if` for running best;
//! - NO `SharedMemory`, NO atomics, NO `F::INFINITY`, NO mutable `bool`, NO
//!   descending loops, NO instance `x.powf()`;
//! - at most TWO coupled loop-carried `F` accumulators per `while` (FINDING
//!   003) — third values are staged through GLOBAL scratch arrays
//!   (`node_total` / `node_max` / `scores`) written by dedicated kernels;
//! - NO local accumulator is written in one `while` and read in a SIBLING
//!   `while` (FINDING 002-B) — cross-loop values go through global memory,
//!   reading an accumulator AFTER its loop in straight-line code is the
//!   kmeans-proven form;
//! - each unit writes ONLY memory it exclusively owns (its own histogram /
//!   score / node slice), so `hist[i] += w` read-modify-writes are race-free
//!   without atomics (the `lbfgs.rs` `grad_w[..] += …` precedent).
//!
//! ## Buffer contract (all row-major, indices `u32`, values `F`)
//! - `x` — `n × d` samples (fit) / `q × d` queries (predict).
//! - `edges` — `d × (nb − 1)` ascending bin edges (padded past-max for unused
//!   slots, so padded bins stay empty and are never selected).
//! - `binned` — `n × d` `u32` bin index per (row, feature), `bin = Σ_k [x ≥ e_k]`.
//! - `w` — `n_trees × n` bootstrap sample counts as `F` (0 = out-of-bag).
//! - `order` / `order_next` — `n_trees × n` row permutation (ping-pong).
//! - `ranges` / `ranges_next` — `n_trees × nodes_level × 2` per-node
//!   `[start, end)` into `order` (ping-pong; `nodes_level = 2^L`).
//! - `feat_ids` — `n_trees × nodes_level × mf` sampled RAW feature ids
//!   (host-drawn per level, D-05: no device RNG).
//! - `hist` — `t_chunk × nodes_level × mf × nb × ncs` weighted counts; after
//!   [`rf_hist_cum`] it is CUMULATIVE over the bin axis. `ncs = n_classes`
//!   (classifier) or `2` (regressor: slot 0 = Σw, slot 1 = Σw·y).
//! - model arrays, `n_trees × total_nodes` (`total_nodes = 2^(D+1) − 1`):
//!   `split_feature` (`u32`, raw id; `u32::MAX` sentinel on leaves),
//!   `split_bin` (`u32`), `threshold` (`F`), `is_leaf` (`u32` 0/1),
//!   `leaf_dist` (`× nc`: normalized class distribution, or the leaf mean for
//!   regression with `nc = 1`), `node_decrease` (`F`, RF-IMP-01: the
//!   sklearn-equivalent weighted impurity decrease at the node, `0` on
//!   leaves — reduced host-side into `feature_importances_`).
//!
//! Tests live in `crates/mlrs-backend/tests/random_forest_test.rs` (this crate
//! is feature-free and cannot launch; AGENTS.md §2 — no in-source tests).

use cubecl::prelude::*;

/// Leaf sentinel written to `split_feature` on leaf nodes (never read by the
/// traversal, which checks `is_leaf` first).
pub const RF_NO_FEATURE: u32 = 0xFFFF_FFFF;

/// Per-(row, feature) bin index: `bin = Σ_k [x[i,j] ≥ edges[j,k]]` over the
/// `nb_edges` ascending edges of feature `j`. One unit per element of the
/// `n × d` output (`tid = i·d + j`). A pure per-element map (the `rbf_map`
/// shape) with one `u32` accumulator.
#[cube(launch)]
pub fn rf_bin_features<F: Float + CubeElement>(
    x: &Array<F>,
    edges: &Array<F>,
    out: &mut Array<u32>,
    n: u32,
    d: u32,
    nb_edges: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n * d;
    if tid < total as usize {
        let j = (tid as u32) % d;
        let v = x[tid];
        let mut b = 0u32;
        let mut k = 0u32;
        while k < nb_edges {
            if v >= edges[(j * nb_edges + k) as usize] {
                b += 1u32;
            }
            k += 1u32;
        }
        out[tid] = b;
    }
}

/// Classifier histogram GATHER: one unit per `(tree_in_chunk, node, feature
/// slot)` zeroes then accumulates its OWN `nb × nc` slice of `hist` with the
/// weighted per-(bin, class) counts of the rows in the node's range. Race-free
/// without atomics: the unit exclusively owns its slice (lbfgs `+=` precedent).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rf_hist_class<F: Float + CubeElement>(
    binned: &Array<u32>,
    y_idx: &Array<u32>,
    w: &Array<F>,
    order: &Array<u32>,
    ranges: &Array<u32>,
    feat_ids: &Array<u32>,
    hist: &mut Array<F>,
    n: u32,
    d: u32,
    mf: u32,
    nb: u32,
    nc: u32,
    nodes: u32,
    t_chunk: u32,
    tree_base: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = t_chunk * nodes * mf;
    if tid < total as usize {
        let tt = (tid as u32) / (nodes * mf);
        let rem = (tid as u32) % (nodes * mf);
        let node = rem / mf;
        let f = rem % mf;
        let gtree = tree_base + tt;

        let base = (tid as u32) * (nb * nc);
        // Zero the exclusively-owned slice (pool buffers arrive uninitialized).
        let mut k = 0u32;
        while k < nb * nc {
            hist[(base + k) as usize] = F::new(0.0);
            k += 1u32;
        }

        let feat = feat_ids[((gtree * nodes + node) * mf + f) as usize];
        let s = ranges[((gtree * nodes + node) * 2u32) as usize];
        let e = ranges[((gtree * nodes + node) * 2u32 + 1u32) as usize];
        let mut r = s;
        while r < e {
            let i = order[(gtree * n + r) as usize];
            let wi = w[(gtree * n + i) as usize];
            let b = binned[(i * d + feat) as usize];
            let c = y_idx[i as usize];
            hist[(base + b * nc + c) as usize] += wi;
            r += 1u32;
        }
    }
}

/// Regressor histogram GATHER: identical shape to [`rf_hist_class`] but with
/// TWO slots per bin (`ncs = 2`): slot 0 accumulates `Σ w`, slot 1 `Σ w·y`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rf_hist_reg<F: Float + CubeElement>(
    binned: &Array<u32>,
    y: &Array<F>,
    w: &Array<F>,
    order: &Array<u32>,
    ranges: &Array<u32>,
    feat_ids: &Array<u32>,
    hist: &mut Array<F>,
    n: u32,
    d: u32,
    mf: u32,
    nb: u32,
    nodes: u32,
    t_chunk: u32,
    tree_base: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = t_chunk * nodes * mf;
    if tid < total as usize {
        let tt = (tid as u32) / (nodes * mf);
        let rem = (tid as u32) % (nodes * mf);
        let node = rem / mf;
        let f = rem % mf;
        let gtree = tree_base + tt;

        let base = (tid as u32) * (nb * 2u32);
        let mut k = 0u32;
        while k < nb * 2u32 {
            hist[(base + k) as usize] = F::new(0.0);
            k += 1u32;
        }

        let feat = feat_ids[((gtree * nodes + node) * mf + f) as usize];
        let s = ranges[((gtree * nodes + node) * 2u32) as usize];
        let e = ranges[((gtree * nodes + node) * 2u32 + 1u32) as usize];
        let mut r = s;
        while r < e {
            let i = order[(gtree * n + r) as usize];
            let wi = w[(gtree * n + i) as usize];
            let b = binned[(i * d + feat) as usize];
            hist[(base + b * 2u32) as usize] += wi;
            hist[(base + b * 2u32 + 1u32) as usize] += wi * y[i as usize];
            r += 1u32;
        }
    }
}

/// In-place cumulative sum of `hist` along the BIN axis: one unit per
/// `(tree_in_chunk, node, feature slot, value slot)` sweeps its own `nb`-stride
/// column ascending with a single `F` accumulator. After this pass
/// `hist[.., b, c] = Σ_{b' ≤ b} counts`, so left-split sums are direct reads.
#[cube(launch)]
pub fn rf_hist_cum<F: Float + CubeElement>(
    hist: &mut Array<F>,
    mf: u32,
    nb: u32,
    ncs: u32,
    nodes: u32,
    t_chunk: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = t_chunk * nodes * mf * ncs;
    if tid < total as usize {
        let col = (tid as u32) / ncs; // (tt, node, f) flat index
        let c = (tid as u32) % ncs;
        let base = col * (nb * ncs) + c;
        let mut acc = F::new(0.0);
        let mut b = 0u32;
        while b < nb {
            acc += hist[(base + b * ncs) as usize];
            hist[(base + b * ncs) as usize] = acc;
            b += 1u32;
        }
    }
}

/// Per-node WEIGHTED total `Σ_c cum_hist[f=0, last_bin, c]` → `node_total`
/// (global staging so downstream kernels never re-derive it in a sibling loop —
/// FINDING 002-B). One unit per `(tree_in_chunk, node)`, one `F` accumulator.
#[cube(launch)]
pub fn rf_node_total<F: Float + CubeElement>(
    hist: &Array<F>,
    node_total: &mut Array<F>,
    mf: u32,
    nb: u32,
    ncs: u32,
    nsum: u32,
    nodes: u32,
    t_chunk: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = t_chunk * nodes;
    if tid < total as usize {
        let base = ((tid as u32) * mf * nb + (nb - 1u32)) * ncs;
        let mut acc = F::new(0.0);
        let mut c = 0u32;
        while c < nsum {
            acc += hist[(base + c) as usize];
            c += 1u32;
        }
        node_total[tid] = acc;
    }
}

/// Per-node MAX class count `max_c cum_hist[f=0, last_bin, c]` → `node_max`
/// (purity staging, classifier only). Running-max statement-form `if`.
#[cube(launch)]
pub fn rf_node_max<F: Float + CubeElement>(
    hist: &Array<F>,
    node_max: &mut Array<F>,
    mf: u32,
    nb: u32,
    nc: u32,
    nodes: u32,
    t_chunk: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = t_chunk * nodes;
    if tid < total as usize {
        let base = ((tid as u32) * mf * nb + (nb - 1u32)) * nc;
        let mut mx = F::new(0.0);
        let mut c = 0u32;
        while c < nc {
            let v = hist[(base + c) as usize];
            if v > mx {
                mx = v;
            }
            c += 1u32;
        }
        node_max[tid] = mx;
    }
}

/// Per-node SUM-OF-SQUARES of class counts `Σ_c cum_hist[f=0, last_bin, c]²`
/// → `node_sq` (RF-IMP-01 impurity-decrease staging, classifier only; for the
/// regressor target this kernel is still launched — reusing the SAME `hist`
/// `ncs = 2` layout as [`rf_node_max`] already does unconditionally — but the
/// resulting `n² + (Σy)²` value is nonsensical and MUST NOT be read by
/// [`rf_best_split`]'s regressor arm, which gates every `node_sq` read behind
/// `mode_class == 1u32`). Mirrors [`rf_node_max`]'s exact index arithmetic,
/// replacing the running-max with a running sum-of-squares.
#[cube(launch)]
pub fn rf_node_sqsum<F: Float + CubeElement>(
    hist: &Array<F>,
    node_sq: &mut Array<F>,
    mf: u32,
    nb: u32,
    nc: u32,
    nodes: u32,
    t_chunk: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = t_chunk * nodes;
    if tid < total as usize {
        let base = ((tid as u32) * mf * nb + (nb - 1u32)) * nc;
        let mut sq = F::new(0.0);
        let mut c = 0u32;
        while c < nc {
            let v = hist[(base + c) as usize];
            sq += v * v;
            c += 1u32;
        }
        node_sq[tid] = sq;
    }
}

/// Classifier split scores: one unit per `(tree_in_chunk, node, feature slot,
/// split bin s)` writes the gini PROXY score `Σ_c l_c²/n_l + Σ_c r_c²/n_r`
/// (maximizing it minimizes the weighted children gini) or `−1` when the split
/// is invalid (`n_l`/`n_r` below `min_leaf`). Two loops, each ≤ 2 independent
/// `F` accumulators (FINDING 003); `n_l` is computed in its OWN loop and read
/// only AFTER both loops (kmeans-proven straight-line read).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rf_split_scores_class<F: Float + CubeElement>(
    hist: &Array<F>,
    node_total: &Array<F>,
    scores: &mut Array<F>,
    min_leaf: F,
    mf: u32,
    nb: u32,
    nc: u32,
    nodes: u32,
    t_chunk: u32,
) {
    let tid = ABSOLUTE_POS;
    let nsplit = nb - 1u32;
    let total = t_chunk * nodes * mf * nsplit;
    if tid < total as usize {
        let tn = (tid as u32) / (mf * nsplit); // (tt, node) flat
        let rem = (tid as u32) % (mf * nsplit);
        let f = rem / nsplit;
        let s = rem % nsplit;
        let fbase = (tn * mf + f) * (nb * nc);

        // Pass 1: left weighted count (single accumulator).
        let mut nl = F::new(0.0);
        let mut c = 0u32;
        while c < nc {
            nl += hist[(fbase + s * nc + c) as usize];
            c += 1u32;
        }

        // Pass 2: left/right sum of squared class counts (two INDEPENDENT
        // accumulators — the proven ceiling).
        let mut sql = F::new(0.0);
        let mut sqr = F::new(0.0);
        let mut c2 = 0u32;
        while c2 < nc {
            let lc = hist[(fbase + s * nc + c2) as usize];
            let tc = hist[(fbase + (nb - 1u32) * nc + c2) as usize];
            let rc = tc - lc;
            sql += lc * lc;
            sqr += rc * rc;
            c2 += 1u32;
        }

        let tot = node_total[tn as usize];
        let nr = tot - nl;
        let mut sc = F::new(-1.0);
        if nl >= min_leaf {
            if nr >= min_leaf {
                sc = sql / nl + sqr / nr;
            }
        }
        scores[tid] = sc;
    }
}

/// Regressor split scores: variance-reduction PROXY `(Σ_l y)²/n_l +
/// (Σ_r y)²/n_r` (the sklearn MSE proxy), read DIRECTLY from the cumulative
/// two-slot histogram — no loops at all.
#[cube(launch)]
pub fn rf_split_scores_reg<F: Float + CubeElement>(
    hist: &Array<F>,
    scores: &mut Array<F>,
    min_leaf: F,
    mf: u32,
    nb: u32,
    nodes: u32,
    t_chunk: u32,
) {
    let tid = ABSOLUTE_POS;
    let nsplit = nb - 1u32;
    let total = t_chunk * nodes * mf * nsplit;
    if tid < total as usize {
        let col = (tid as u32) / nsplit; // (tt, node, f) flat
        let s = (tid as u32) % nsplit;
        let base = col * (nb * 2u32);

        let nl = hist[(base + s * 2u32) as usize];
        let syl = hist[(base + s * 2u32 + 1u32) as usize];
        let tot = hist[(base + (nb - 1u32) * 2u32) as usize];
        let syt = hist[(base + (nb - 1u32) * 2u32 + 1u32) as usize];
        let nr = tot - nl;
        let syr = syt - syl;

        let mut sc = F::new(-1.0);
        if nl >= min_leaf {
            if nr >= min_leaf {
                sc = syl * syl / nl + syr * syr / nr;
            }
        }
        scores[tid] = sc;
    }
}

/// Per-node split finalize: one unit per `(tree_in_chunk, node)` arg-maxes its
/// `mf × (nb−1)` score slice (strict `>` → lowest-(f,s) tie-break), decides
/// leaf-ness, and writes the model arrays for the node's GLOBAL id
/// (`level_base + node`):
///
/// leaf ⇔ `force_leaf` (bottom level) ∨ `total < min_split` ∨ `total ≤ 0`
///        ∨ pure (`node_max ≥ total`, classifier: pass `node_max = total`
///        disabled as `-1` for regression) ∨ no valid split (`best < 0`).
///
/// `leaf_dist` is ALWAYS written (normalized class distribution, or the mean
/// target with `nc = 1` for regression via `sy_slot = 1`): interior-node
/// distributions are simply never read by the traversal.
///
/// `node_decrease` (RF-IMP-01) is ALWAYS written too: `0` on leaf nodes, else
/// the sklearn-equivalent weighted impurity decrease `best − parent_sumsq /
/// tot` — classifier: `parent_sumsq = node_sq[tid]` (the new `rf_node_sqsum`
/// input, read ONLY here, gated behind `mode_class == 1u32`); regressor:
/// `parent_sumsq = syt²` where `syt = hist[hbase+1]` is the node's own total
/// weighted `Σy` (already read by the leaf-mean branch below, no new input
/// needed).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rf_best_split<F: Float + CubeElement>(
    hist: &Array<F>,
    node_total: &Array<F>,
    node_max: &Array<F>,
    node_sq: &Array<F>,
    scores: &Array<F>,
    feat_ids: &Array<u32>,
    edges: &Array<F>,
    split_feature: &mut Array<u32>,
    split_bin: &mut Array<u32>,
    threshold: &mut Array<F>,
    is_leaf: &mut Array<u32>,
    leaf_dist: &mut Array<F>,
    node_decrease: &mut Array<F>,
    min_split: F,
    mf: u32,
    nb: u32,
    ncs: u32,
    nc_out: u32,
    nodes: u32,
    t_chunk: u32,
    tree_base: u32,
    level_base: u32,
    total_nodes: u32,
    force_leaf: u32,
    mode_class: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = t_chunk * nodes;
    if tid < total as usize {
        let tt = (tid as u32) / nodes;
        let node = (tid as u32) % nodes;
        let gtree = tree_base + tt;
        let gnode = level_base + node;
        let midx = gtree * total_nodes + gnode;

        // Running best over the flat (f, s) score slice — the topk running-max
        // pair (F best + u32 arg), strict `>` = lowest-index tie-break.
        let nsplit = nb - 1u32;
        let sbase = (tid as u32) * mf * nsplit;
        let mut best = F::new(-1.0);
        let mut bk = 0u32;
        let mut k = 0u32;
        while k < mf * nsplit {
            let sc = scores[(sbase + k) as usize];
            if sc > best {
                best = sc;
                bk = k;
            }
            k += 1u32;
        }

        let tot = node_total[tid];
        let mx = node_max[tid];
        let mut leaf = 0u32;
        if force_leaf == 1u32 {
            leaf = 1u32;
        }
        if tot < min_split {
            leaf = 1u32;
        }
        if tot <= F::new(0.0) {
            leaf = 1u32;
        }
        if mode_class == 1u32 {
            if mx >= tot {
                leaf = 1u32;
            }
        }
        if best < F::new(0.0) {
            leaf = 1u32;
        }
        is_leaf[midx as usize] = leaf;

        // Leaf value(s): normalized class distribution from the cumulative
        // histogram's last bin of feature slot 0 (classifier), or the mean
        // target Σwy / Σw (regression, nc_out = 1, slot 1 of ncs = 2).
        let hbase = ((tid as u32) * mf * nb + (nb - 1u32)) * ncs;
        if mode_class == 1u32 {
            let mut c = 0u32;
            while c < nc_out {
                let cnt = hist[(hbase + c) as usize];
                let mut p = F::new(0.0);
                if tot > F::new(0.0) {
                    p = cnt / tot;
                }
                leaf_dist[(midx * nc_out + c) as usize] = p;
                c += 1u32;
            }
        } else {
            let sy = hist[(hbase + 1u32) as usize];
            let mut mean = F::new(0.0);
            if tot > F::new(0.0) {
                mean = sy / tot;
            }
            leaf_dist[midx as usize] = mean;
        }

        // RF-IMP-01: weighted impurity decrease `best − parent_sumsq/tot`,
        // `0` on leaf nodes. Classifier reads the new `node_sq` staging
        // input (gated behind `mode_class == 1u32`, mirroring the purity
        // check above); regressor reuses `sy = hist[hbase+1]`, already read
        // above for the leaf-mean branch (`syt² / tot`).
        if leaf == 1u32 {
            node_decrease[midx as usize] = F::new(0.0);
        } else {
            if mode_class == 1u32 {
                let sq = node_sq[tid];
                node_decrease[midx as usize] = best - sq / tot;
            } else {
                let sy = hist[(hbase + 1u32) as usize];
                node_decrease[midx as usize] = best - (sy * sy) / tot;
            }
        }

        if leaf == 0u32 {
            let bf = bk / nsplit;
            let bs = bk % nsplit;
            let fraw = feat_ids[((gtree * nodes + node) * mf + bf) as usize];
            split_feature[midx as usize] = fraw;
            split_bin[midx as usize] = bs;
            threshold[midx as usize] = edges[(fraw * nsplit + bs) as usize];
        } else {
            split_feature[midx as usize] = 0xFFFF_FFFFu32;
            split_bin[midx as usize] = 0u32;
            threshold[midx as usize] = F::new(0.0);
        }
    }
}

/// Children `[start, end)` ranges for the NEXT level: one unit per
/// `(tree, node)` counts its left rows (`bin ≤ split_bin`, single `u32`
/// accumulator — the ROW count, not the weighted count, since out-of-bag
/// `w = 0` rows travel with the partition) and writes the two child ranges.
/// Leaf nodes emit two EMPTY child ranges.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rf_count_left(
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
    n_trees: u32,
    level_base: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n_trees * nodes;
    if tid < total as usize {
        let tree = (tid as u32) / nodes;
        let node = (tid as u32) % nodes;
        let gnode = level_base + node;
        let midx = tree * total_nodes + gnode;
        let next_nodes = nodes * 2u32;
        let lbase = (tree * next_nodes + node * 2u32) * 2u32;

        if is_leaf[midx as usize] == 1u32 {
            ranges_next[lbase as usize] = 0u32;
            ranges_next[(lbase + 1u32) as usize] = 0u32;
            ranges_next[(lbase + 2u32) as usize] = 0u32;
            ranges_next[(lbase + 3u32) as usize] = 0u32;
        } else {
            let fraw = split_feature[midx as usize];
            let bs = split_bin[midx as usize];
            let s = ranges[((tree * nodes + node) * 2u32) as usize];
            let e = ranges[((tree * nodes + node) * 2u32 + 1u32) as usize];
            let mut cnt = 0u32;
            let mut r = s;
            while r < e {
                let i = order[(tree * n + r) as usize];
                if binned[(i * d + fraw) as usize] <= bs {
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

/// Stable two-way partition into `order_next`: one unit per `(tree, node)`
/// re-scans its range and scatters rows to the child ranges computed by
/// [`rf_count_left`] (read from GLOBAL `ranges_next` — never a cross-sibling-
/// loop local, FINDING 002-B). The two write cursors are `u32` accumulators
/// read and bumped within the SAME iteration. Each parent owns its output
/// range exclusively — race-free without atomics.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rf_partition(
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
    n_trees: u32,
    level_base: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n_trees * nodes;
    if tid < total as usize {
        let tree = (tid as u32) / nodes;
        let node = (tid as u32) % nodes;
        let gnode = level_base + node;
        let midx = tree * total_nodes + gnode;

        if is_leaf[midx as usize] == 0u32 {
            let fraw = split_feature[midx as usize];
            let bs = split_bin[midx as usize];
            let s = ranges[((tree * nodes + node) * 2u32) as usize];
            let e = ranges[((tree * nodes + node) * 2u32 + 1u32) as usize];
            let next_nodes = nodes * 2u32;
            let lbase = (tree * next_nodes + node * 2u32) * 2u32;
            let mut li = ranges_next[lbase as usize];
            let mut ri = ranges_next[(lbase + 2u32) as usize];
            let mut r = s;
            while r < e {
                let i = order[(tree * n + r) as usize];
                if binned[(i * d + fraw) as usize] <= bs {
                    order_next[(tree * n + li) as usize] = i;
                    li += 1u32;
                } else {
                    order_next[(tree * n + ri) as usize] = i;
                    ri += 1u32;
                }
                r += 1u32;
            }
        }
    }
}

/// Forest traversal: one unit per `(tree, query row)` walks the complete-tree
/// arrays from the root for exactly `max_depth` bounded steps (a fixed
/// ascending counter — no data-dependent `while`), advancing only while the
/// current node is interior: `x < threshold → 2g+1` else `2g+2`. Writes the
/// reached LEAF node id.
#[cube(launch)]
pub fn rf_predict_leaf<F: Float + CubeElement>(
    x: &Array<F>,
    split_feature: &Array<u32>,
    threshold: &Array<F>,
    is_leaf: &Array<u32>,
    out_leaf: &mut Array<u32>,
    q: u32,
    d: u32,
    max_depth: u32,
    n_trees: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n_trees * q;
    if tid < total as usize {
        let tree = (tid as u32) / q;
        let row = (tid as u32) % q;
        let mut cur = 0u32;
        let mut l = 0u32;
        while l < max_depth {
            if is_leaf[(tree * total_nodes + cur) as usize] == 0u32 {
                let fr = split_feature[(tree * total_nodes + cur) as usize];
                let thr = threshold[(tree * total_nodes + cur) as usize];
                let xv = x[(row * d + fr) as usize];
                let mut nxt = 2u32 * cur + 2u32;
                if xv < thr {
                    nxt = 2u32 * cur + 1u32;
                }
                cur = nxt;
            }
            l += 1u32;
        }
        out_leaf[tid] = cur;
    }
}

/// Classifier vote: one unit per query row averages the reached leaves' class
/// distributions over the forest (`proba[q, c] = Σ_t dist_t[c] / n_trees` —
/// the sklearn `predict_proba` mean-of-leaf-distributions). One fresh `F`
/// accumulator per class (re-initialized inside the consuming loop).
#[cube(launch)]
pub fn rf_vote_class<F: Float + CubeElement>(
    leaf: &Array<u32>,
    leaf_dist: &Array<F>,
    proba: &mut Array<F>,
    q: u32,
    nc: u32,
    n_trees: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < q as usize {
        let mut c = 0u32;
        while c < nc {
            let mut acc = F::new(0.0);
            let mut t = 0u32;
            while t < n_trees {
                let lf = leaf[(t * q + (tid as u32)) as usize];
                acc += leaf_dist[((t * total_nodes + lf) * nc + c) as usize];
                t += 1u32;
            }
            proba[((tid as u32) * nc + c) as usize] = acc / F::cast_from(n_trees);
            c += 1u32;
        }
    }
}

/// Regressor mean: one unit per query row averages the reached leaves' stored
/// mean targets over the forest. Single `F` accumulator.
#[cube(launch)]
pub fn rf_mean_reg<F: Float + CubeElement>(
    leaf: &Array<u32>,
    leaf_value: &Array<F>,
    out: &mut Array<F>,
    q: u32,
    n_trees: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < q as usize {
        let mut acc = F::new(0.0);
        let mut t = 0u32;
        while t < n_trees {
            let lf = leaf[(t * q + (tid as u32)) as usize];
            acc += leaf_value[(t * total_nodes + lf) as usize];
            t += 1u32;
        }
        out[tid] = acc / F::cast_from(n_trees);
    }
}
