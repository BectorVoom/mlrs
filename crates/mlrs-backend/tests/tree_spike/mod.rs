//! Shared test-support module for the Phase-17 RandomForest feasibility spike
//! (TREE-01) — the three cpu-MLIR-safe tree kernels, their host launch wrappers,
//! the `SparseTreeNode` format contract, and the host per-level `build_tree` loop
//! that composes them. A subdirectory `mod.rs` is NOT compiled as its own test
//! binary; it is pulled in by `tests/tree_spike_probes.rs` via `mod tree_spike;`.
//!
//! ## cpu-MLIR authoring contract (why these three kernels exist)
//!
//! The whole point of this spike is to prove the histogram / split-find / relabel
//! math lowers and value-correctly executes under `cubecl-cpu` (the MLIR backend,
//! the f64 correctness gate). cpu-MLIR fails LOUDLY outside its proven op-set, or
//! — worse — SILENTLY miscompiles. Each kernel here is authored strictly inside
//! the proven op-set, by concept:
//!
//! - **Single-owner GATHER histogram.** Each output cell `(node, feature, bin)`
//!   is owned by exactly one unit; that unit loops the sample range and, in the
//!   SAME loop iteration, reads both the sample's node label and its feature bin
//!   and conditionally accumulates a count and a value-sum. One writer per cell,
//!   so no contention and no scatter-add — modelled on the shipped feature-loop
//!   accumulator in `mlrs_kernels::manhattan_dist`. The per-cell launch is the
//!   2D guarded `ABSOLUTE_POS_X/Y` shape (X over `node*feature`, Y over `bin`).
//! - **Statement-form running-best split-find seeded from candidate 0.** The gain
//!   argmax seeds its running best from the first candidate and updates with a
//!   statement-form `if`, carrying `(best_gain, best_col, best_bin)` and resolving
//!   ties with `u32` admit/better flags (lowest feature index, then lowest bin).
//!   No floating sentinel init, no `if`-expression in value position, no mutable
//!   `bool` — modelled on the shipped `mlrs_kernels::select_k`.
//! - **Per-sample relabel-partition GATHER.** One cube per sample
//!   (`CUBE_POS_X` / `UNIT_POS_X == 0`): the sample reads its current node's split
//!   from per-node frontier arrays and overwrites its own label with the left
//!   child id (go-left) or `left_child + 1` (go-right, D-02). Pure self-overwrite
//!   GATHER — never a scan or compaction — modelled on `self_drop_gather`.
//!
//! Constraints honoured throughout (so the spike's verdict is valid): only `F` and
//! `u32` accumulators, statement-form mutable-`if` guards, runtime `while` loops,
//! and `as usize` casts solely at the array-index boundary. No shared-memory
//! histogram, no atomic scatter-add, no floating infinity init, no cross-sibling
//! `while` accumulator (every per-cell value is recomputed inside the consuming
//! loop). Scalar kernel args pass by value (cubecl 0.10, no `ScalarArg`).

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use mlrs_backend::runtime::{self, ActiveRuntime};
use std::mem::size_of;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 — GATHER histogram (single-owner, one unit per (node, feature, bin)).
// Model: `mlrs_kernels::manhattan_dist` same-iteration feature-loop accumulator.
// ─────────────────────────────────────────────────────────────────────────────

/// Per-cell histogram: for each `(node, feature, bin)` cell, count the samples
/// whose current node label equals the cell's node AND whose feature bin equals
/// the cell's bin, and sum their target value `y` in the SAME iteration.
///
/// 2D guarded launch (`ABSOLUTE_POS_X` = `node*n_feat + feature`,
/// `ABSOLUTE_POS_Y` = `bin`). Outputs are two `n_nodes*n_feat*n_bins` arrays:
/// `hist_count` (per-cell sample count) and `hist_vsum` (per-cell sum of `y`).
#[cube(launch)]
pub fn tree_gather_histogram<F: Float + CubeElement>(
    node_id: &Array<u32>,
    binned: &Array<u32>,
    y: &Array<F>,
    hist_count: &mut Array<F>,
    hist_vsum: &mut Array<F>,
    n_samples: u32,
    n_feat: u32,
    n_nodes: u32,
    n_bins: u32,
) {
    let nf = ABSOLUTE_POS_X; // node * n_feat + feature
    let bin = ABSOLUTE_POS_Y; // candidate bin
    if nf < n_nodes * n_feat {
        if bin < n_bins {
            let my_node = nf / n_feat;
            let my_feature = nf % n_feat;
            // Single-owner accumulators (count + value-sum). Both reads happen in
            // the SAME iteration of this one loop — never a cross-sibling-loop
            // counter (002-B silent miscompile).
            let mut cnt = F::from_int(0i64);
            let mut vsum = F::from_int(0i64);
            let mut s = 0u32;
            while s < n_samples {
                if node_id[s as usize] == my_node {
                    if binned[(s * n_feat + my_feature) as usize] == bin {
                        cnt += F::from_int(1i64);
                        vsum += y[s as usize];
                    }
                }
                s += 1u32;
            }
            let cell = nf * n_bins + bin;
            hist_count[cell as usize] = cnt;
            hist_vsum[cell as usize] = vsum;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 — seed-from-first split-find argmax (NO floating sentinel init).
// Model: `mlrs_kernels::select_k` running-best with `u32` admit/better flags.
// ─────────────────────────────────────────────────────────────────────────────

/// Per-node gain argmax: one cube per node (`CUBE_POS_X`), unit 0 scans this
/// node's `n_candidates` `(feature, bin)` gains and emits the maximum-gain split.
///
/// The running best is SEEDED from candidate 0 (no floating infinity init); ties
/// resolve to the lowest feature index, then the lowest bin, via `u32` flags
/// (never a mutable `bool`). `gain` is `n_nodes * n_candidates` row-major;
/// `col_of` / `bin_of` are the shared `n_candidates` (feature, bin) maps.
#[cube(launch)]
pub fn tree_split_find<F: Float + CubeElement>(
    gain: &Array<F>,
    col_of: &Array<u32>,
    bin_of: &Array<u32>,
    out_gain: &mut Array<F>,
    out_col: &mut Array<u32>,
    out_bin: &mut Array<u32>,
    n_nodes: u32,
    n_candidates: u32,
) {
    let node = CUBE_POS_X;
    if node < n_nodes {
        if UNIT_POS_X == 0u32 {
            let base = node * n_candidates;
            // Seed running best from candidate 0 (statement-form, no -INF init).
            let mut best_gain = gain[base as usize];
            let mut best_col = col_of[0usize];
            let mut best_bin = bin_of[0usize];
            let mut c = 1u32;
            while c < n_candidates {
                let g = gain[(base + c) as usize];
                let gc = col_of[c as usize];
                let gb = bin_of[c as usize];
                // better = (g, lower col, lower bin) beats the running best.
                let mut better: u32 = 0u32;
                if g > best_gain {
                    better = 1u32;
                } else if g == best_gain {
                    if gc < best_col {
                        better = 1u32;
                    } else if gc == best_col {
                        if gb < best_bin {
                            better = 1u32;
                        }
                    }
                }
                if better == 1u32 {
                    best_gain = g;
                    best_col = gc;
                    best_bin = gb;
                }
                c += 1u32;
            }
            out_gain[node as usize] = best_gain;
            out_col[node as usize] = best_col;
            out_bin[node as usize] = best_bin;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3 — relabel-partition (per-sample GATHER, no scan/compaction).
// Model: `mlrs_kernels::self_drop_gather` per-row `CUBE_POS_X`/`UNIT_POS_X==0`.
// ─────────────────────────────────────────────────────────────────────────────

/// Per-sample relabel: one cube per sample (`CUBE_POS_X`), unit 0 reads the
/// sample's current node, looks up that node's split from the per-node frontier
/// arrays, and overwrites its own label with the left child (go-left) or
/// `left_child + 1` (go-right, D-02). Pure self-overwrite GATHER.
///
/// `split_active[nid] == 1` marks a node that splits this level; a `0` (leaf /
/// inactive) node leaves its samples untouched. All-`u32` — no signed value casts.
#[cube(launch)]
pub fn tree_relabel_partition(
    node_id: &mut Array<u32>,
    binned: &Array<u32>,
    split_active: &Array<u32>,
    split_col: &Array<u32>,
    split_bin: &Array<u32>,
    left_child: &Array<u32>,
    n_samples: u32,
    n_feat: u32,
) {
    let s = CUBE_POS_X;
    if s < n_samples {
        if UNIT_POS_X == 0u32 {
            let nid = node_id[s as usize];
            if split_active[nid as usize] == 1u32 {
                let col = split_col[nid as usize];
                let lc = left_child[nid as usize];
                let thr = split_bin[nid as usize];
                // statement-form branch; right child = left_child + 1 (D-02).
                let mut child = lc;
                let bv = binned[(s * n_feat + col) as usize];
                if bv > thr {
                    child = lc + 1u32;
                }
                node_id[s as usize] = child;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host helpers — byte casts (never call a #[cube] fn on the host) + launch dims.
// ─────────────────────────────────────────────────────────────────────────────

/// Byte-cast an `F` (f32/f64) to host `f64` without calling any `#[cube]` fn.
pub fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("tree spike is f32/f64 only"),
    }
}

/// Build an `F` (f32/f64) from a host `f64` literal without a `#[cube]` fn.
pub fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("tree spike is f32/f64 only"),
    }
}

/// Launch-geometry validation (T-17-03): reject `u32` overflow BEFORE any
/// `unsafe { ArrayArg::from_raw_parts }`. Returns the validated element product.
fn checked_mul(a: usize, b: usize, what: &str) -> usize {
    let p = a
        .checked_mul(b)
        .unwrap_or_else(|| panic!("tree geometry overflow ({what}): {a} * {b}"));
    assert!(
        p <= u32::MAX as usize,
        "tree geometry exceeds u32 ({what}): {p} > u32::MAX"
    );
    p
}

/// Per-row launch shape: one cube per row, one selecting unit (`CUBE_POS_X` /
/// `UNIT_POS_X == 0`) — NOT a bare 1D `ABSOLUTE_POS` launch (the 002-A failure).
fn launch_dims_rows(n: usize) -> (CubeCount, CubeDim) {
    (
        CubeCount::Static(n.max(1) as u32, 1, 1),
        CubeDim { x: 1, y: 1, z: 1 },
    )
}

/// Per-cell 2D launch shape (`ABSOLUTE_POS_X/Y`), the proven `manhattan_dist`
/// geometry with ceiling-div counts and a 16x16 cube.
fn launch_dims_2d(nx: usize, ny: usize) -> (CubeCount, CubeDim) {
    let bx = 16u32;
    let by = 16u32;
    let cx = ((nx as u32) + bx - 1) / bx;
    let cy = ((ny as u32) + by - 1) / by;
    (
        CubeCount::Static(cx.max(1), cy.max(1), 1),
        CubeDim { x: bx, y: by, z: 1 },
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launch wrappers — clone `spike_test.rs` / `self_drop_gather_test.rs`
// boilerplate: active_client, Bytes::from_elems upload, client.empty, clone the
// read handle BEFORE from_raw_parts consumes it, read_one + cast_slice, scalars
// by value. Each validates launch geometry before the `unsafe` launch.
// ─────────────────────────────────────────────────────────────────────────────

/// Launch `tree_gather_histogram`; returns `(counts, vsums)`, each
/// `n_nodes*n_feat*n_bins` long, in `((node*n_feat+feature)*n_bins + bin)` order.
pub fn launch_histogram<F>(
    node_id: &[u32],
    binned: &[u32],
    y: &[F],
    n_samples: usize,
    n_feat: usize,
    n_nodes: usize,
    n_bins: usize,
) -> (Vec<F>, Vec<F>)
where
    F: Float + CubeElement + bytemuck::Pod,
{
    assert_eq!(node_id.len(), n_samples, "node_id len");
    assert_eq!(
        binned.len(),
        checked_mul(n_samples, n_feat, "n_samples*n_feat"),
        "binned len"
    );
    assert_eq!(y.len(), n_samples, "y len");
    let nf = checked_mul(n_nodes, n_feat, "n_nodes*n_feat");
    let n_cells = checked_mul(nf, n_bins, "n_nodes*n_feat*n_bins");

    let client = runtime::active_client();
    let nid_h = client.create(Bytes::from_elems(node_id.to_vec()));
    let bin_h = client.create(Bytes::from_elems(binned.to_vec()));
    let y_h = client.create(Bytes::from_elems(y.to_vec()));
    let hc_h = client.empty(n_cells * size_of::<F>());
    let hv_h = client.empty(n_cells * size_of::<F>());
    let hc_read = hc_h.clone();
    let hv_read = hv_h.clone();

    let (count, dim) = launch_dims_2d(nf, n_bins);
    let nid = unsafe { ArrayArg::from_raw_parts(nid_h, n_samples) };
    let bin = unsafe { ArrayArg::from_raw_parts(bin_h, binned.len()) };
    let ya = unsafe { ArrayArg::from_raw_parts(y_h, n_samples) };
    let hc = unsafe { ArrayArg::from_raw_parts(hc_h, n_cells) };
    let hv = unsafe { ArrayArg::from_raw_parts(hv_h, n_cells) };
    tree_gather_histogram::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        nid,
        bin,
        ya,
        hc,
        hv,
        n_samples as u32,
        n_feat as u32,
        n_nodes as u32,
        n_bins as u32,
    );

    let counts =
        bytemuck::cast_slice::<u8, F>(&client.read_one(hc_read).expect("read hist_count")).to_vec();
    let vsums =
        bytemuck::cast_slice::<u8, F>(&client.read_one(hv_read).expect("read hist_vsum")).to_vec();
    (counts, vsums)
}

/// Launch `tree_split_find`; returns `(best_gain, best_col, best_bin)`, each
/// `n_nodes` long.
pub fn launch_split_find<F>(
    gain: &[F],
    col_of: &[u32],
    bin_of: &[u32],
    n_nodes: usize,
    n_candidates: usize,
) -> (Vec<F>, Vec<u32>, Vec<u32>)
where
    F: Float + CubeElement + bytemuck::Pod,
{
    assert_eq!(
        gain.len(),
        checked_mul(n_nodes, n_candidates, "n_nodes*n_candidates"),
        "gain len"
    );
    assert_eq!(col_of.len(), n_candidates, "col_of len");
    assert_eq!(bin_of.len(), n_candidates, "bin_of len");

    let client = runtime::active_client();
    let g_h = client.create(Bytes::from_elems(gain.to_vec()));
    let c_h = client.create(Bytes::from_elems(col_of.to_vec()));
    let b_h = client.create(Bytes::from_elems(bin_of.to_vec()));
    let og_h = client.empty(n_nodes * size_of::<F>());
    let oc_h = client.empty(n_nodes * size_of::<u32>());
    let ob_h = client.empty(n_nodes * size_of::<u32>());
    let og_r = og_h.clone();
    let oc_r = oc_h.clone();
    let ob_r = ob_h.clone();

    let (count, dim) = launch_dims_rows(n_nodes);
    let g = unsafe { ArrayArg::from_raw_parts(g_h, gain.len()) };
    let c = unsafe { ArrayArg::from_raw_parts(c_h, n_candidates) };
    let b = unsafe { ArrayArg::from_raw_parts(b_h, n_candidates) };
    let og = unsafe { ArrayArg::from_raw_parts(og_h, n_nodes) };
    let oc = unsafe { ArrayArg::from_raw_parts(oc_h, n_nodes) };
    let ob = unsafe { ArrayArg::from_raw_parts(ob_h, n_nodes) };
    tree_split_find::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        g,
        c,
        b,
        og,
        oc,
        ob,
        n_nodes as u32,
        n_candidates as u32,
    );

    let bg =
        bytemuck::cast_slice::<u8, F>(&client.read_one(og_r).expect("read best_gain")).to_vec();
    let bc =
        bytemuck::cast_slice::<u8, u32>(&client.read_one(oc_r).expect("read best_col")).to_vec();
    let bb =
        bytemuck::cast_slice::<u8, u32>(&client.read_one(ob_r).expect("read best_bin")).to_vec();
    (bg, bc, bb)
}

/// Launch `tree_relabel_partition`; returns the relabeled `node_id` (length
/// `n_samples`). Frontier arrays are indexed by node id (length `n_nodes`).
pub fn launch_relabel(
    node_id: &[u32],
    binned: &[u32],
    split_active: &[u32],
    split_col: &[u32],
    split_bin: &[u32],
    left_child: &[u32],
    n_samples: usize,
    n_feat: usize,
) -> Vec<u32> {
    assert_eq!(node_id.len(), n_samples, "node_id len");
    assert_eq!(
        binned.len(),
        checked_mul(n_samples, n_feat, "n_samples*n_feat"),
        "binned len"
    );
    let n_nodes = split_active.len();
    assert_eq!(split_col.len(), n_nodes, "split_col len");
    assert_eq!(split_bin.len(), n_nodes, "split_bin len");
    assert_eq!(left_child.len(), n_nodes, "left_child len");

    let client = runtime::active_client();
    let nid_h = client.create(Bytes::from_elems(node_id.to_vec()));
    let bin_h = client.create(Bytes::from_elems(binned.to_vec()));
    let sa_h = client.create(Bytes::from_elems(split_active.to_vec()));
    let sc_h = client.create(Bytes::from_elems(split_col.to_vec()));
    let sb_h = client.create(Bytes::from_elems(split_bin.to_vec()));
    let lc_h = client.create(Bytes::from_elems(left_child.to_vec()));
    let nid_r = nid_h.clone();

    let (count, dim) = launch_dims_rows(n_samples);
    let nid = unsafe { ArrayArg::from_raw_parts(nid_h, n_samples) };
    let bin = unsafe { ArrayArg::from_raw_parts(bin_h, binned.len()) };
    let sa = unsafe { ArrayArg::from_raw_parts(sa_h, n_nodes) };
    let sc = unsafe { ArrayArg::from_raw_parts(sc_h, n_nodes) };
    let sb = unsafe { ArrayArg::from_raw_parts(sb_h, n_nodes) };
    let lc = unsafe { ArrayArg::from_raw_parts(lc_h, n_nodes) };
    tree_relabel_partition::launch::<ActiveRuntime>(
        &client,
        count,
        dim,
        nid,
        bin,
        sa,
        sc,
        sb,
        lc,
        n_samples as u32,
        n_feat as u32,
    );

    bytemuck::cast_slice::<u8, u32>(&client.read_one(nid_r).expect("read node_id")).to_vec()
}

// ─────────────────────────────────────────────────────────────────────────────
// SparseTreeNode contract (D-02 / D-03 / D-04) — the shared format Plans 03/04
// reuse. Deliberately diverges from cuML's flatnode.h leaf sentinel.
// ─────────────────────────────────────────────────────────────────────────────

/// Flat decision-tree node (TREE-01 contract).
///
/// - `colid: i32` — split feature column. SENTINEL `colid == -1` marks a LEAF
///   (D-03; treelite/FIL convention — FIL stops on `colid < 0`). This diverges
///   from cuML's `flatnode.h`, which marks a leaf with `left_child_id == -1`.
/// - `threshold: F` — the real-valued split edge for `colid`'s chosen bin cut.
/// - `left_child: i32` — index of the left child in the flat node array. The
///   RIGHT child is implicit: `right = left_child + 1` (D-02, shared with cuML).
/// - `value: i32` — NOT a scalar prediction. An OFFSET/INDEX into a shared
///   leaf-value buffer (D-04). Multiclass-uniform: binary, multiclass, and
///   regression leaves all index a side buffer through this one field.
#[derive(Clone, Copy, Debug)]
pub struct SparseTreeNode<F> {
    pub colid: i32,
    pub threshold: F,
    pub left_child: i32,
    pub value: i32,
}

/// Binary-classification Gini impurity from a positive-count `pos` over total
/// `cnt`: `2 p (1 - p)` with `p = pos / cnt`.
fn gini(pos: f64, cnt: f64) -> f64 {
    if cnt <= 0.0 {
        return 0.0;
    }
    let p = pos / cnt;
    2.0 * p * (1.0 - p)
}

/// Host per-level tree build loop composing the three device kernels (D-01).
///
/// Binary-classification builder (target `y` in `{0, 1}`; the histogram's
/// value-sum is the per-cell positive count, so Gini gain is computable from
/// counts + value-sums alone). Kernels cannot recurse, so the host drives a
/// per-level `while` bounded by `max_depth`: each level launches histogram →
/// split-find → relabel, appends internal nodes (children adjacent, D-02), and
/// marks a node a leaf (`colid = -1`, `value` = leaf-buffer offset) when it is
/// pure, hits `max_depth`, or has fewer than `min_samples`.
///
/// `bin_edges[f]` holds feature `f`'s `n_bins - 1` real-valued split thresholds
/// (host-precomputed quantile edges — D-10, NO on-device sort). `n_bins` is a
/// parameter so Plan 04 can drive 64 vs 128. Returns the flat node array and the
/// shared leaf-value buffer.
pub fn build_tree<F>(
    binned: &[u32],
    y: &[F],
    bin_edges: &[Vec<f64>],
    n_samples: usize,
    n_feat: usize,
    n_bins: usize,
    max_depth: usize,
    min_samples: usize,
) -> (Vec<SparseTreeNode<F>>, Vec<f64>)
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let n_cand = n_feat * (n_bins - 1);
    let cell = move |nid: usize, f: usize, b: usize| (nid * n_feat + f) * n_bins + b;

    // Criterion-specific level math: binary-Gini gain per (node, feature,
    // split-after-bin) plus a per-node purity flag. The shared frontier /
    // adjacency / relabel skeleton lives once in `build_tree_with` (WR-02).
    let gini_gain = move |_node_id: &[u32],
                          counts: &[F],
                          vsums: &[F],
                          frontier: &[u32],
                          n_nodes_total: usize|
          -> (Vec<f64>, Vec<bool>) {
        let mut gain_h = vec![0.0f64; n_nodes_total * n_cand];
        let mut pure = vec![false; n_nodes_total];
        for &nid_u in frontier {
            let nid = nid_u as usize;
            // Purity from feature 0 (count/positive-count are feature-invariant):
            // a single-class node (pos == 0 or pos == tot) cannot be split.
            let mut tot0 = 0.0f64;
            let mut pos0 = 0.0f64;
            for b in 0..n_bins {
                tot0 += host_to_f64(counts[cell(nid, 0, b)]);
                pos0 += host_to_f64(vsums[cell(nid, 0, b)]);
            }
            pure[nid] = pos0 == 0.0 || pos0 == tot0;
            for f in 0..n_feat {
                let mut tot = 0.0f64;
                let mut pos = 0.0f64;
                for b in 0..n_bins {
                    tot += host_to_f64(counts[cell(nid, f, b)]);
                    pos += host_to_f64(vsums[cell(nid, f, b)]);
                }
                let parent = gini(pos, tot);
                let mut lc = 0.0f64;
                let mut lp = 0.0f64;
                for b in 0..(n_bins - 1) {
                    lc += host_to_f64(counts[cell(nid, f, b)]);
                    lp += host_to_f64(vsums[cell(nid, f, b)]);
                    let rc = tot - lc;
                    let rp = pos - lp;
                    let g = if tot > 0.0 {
                        parent - (lc / tot) * gini(lp, lc) - (rc / tot) * gini(rp, rc)
                    } else {
                        0.0
                    };
                    gain_h[nid * n_cand + (f * (n_bins - 1) + b)] = g;
                }
            }
        }
        (gain_h, pure)
    };

    // Classifier leaf value = positive-class probability pos/tot (D-09).
    let leaf_prob = |sum_y: f64, tot: f64| if tot > 0.0 { sum_y / tot } else { 0.0 };

    build_tree_with::<F, _, _>(
        binned, y, bin_edges, n_samples, n_feat, n_bins, max_depth, min_samples, gini_gain,
        leaf_prob,
    )
}

/// Shared per-level frontier DRIVER (D-01) — the single copy of the
/// histogram → split-find → relabel skeleton, parameterized by a criterion so
/// the classifier and regressor builders no longer duplicate it (WR-02).
///
/// `level_gain(node_id, counts, vsums, frontier, n_nodes_total)` returns this
/// level's per-`(node, feature, split-after-bin)` gain (row-major
/// `n_nodes_total * n_cand`) AND a per-node purity flag (`true` ⇒ the node
/// cannot be split and must become a leaf). It receives the histogram on `y`
/// (count + value-sum) and may launch its own auxiliary histograms (the
/// regressor launches a second one on `y^2` for sum-of-squares).
/// `leaf_value(sum_y, total)` maps a leaf node's feature-0 totals to its stored
/// value (classifier ⇒ `pos/tot` probability; regressor ⇒ `sum_y/tot` mean).
/// Adjacency (D-02), leaf sentinel (D-03/D-04), `max_depth`/`min_samples`
/// termination, and relabel all live here exactly once.
#[allow(clippy::too_many_arguments)]
pub fn build_tree_with<F, G, L>(
    binned: &[u32],
    y: &[F],
    bin_edges: &[Vec<f64>],
    n_samples: usize,
    n_feat: usize,
    n_bins: usize,
    max_depth: usize,
    min_samples: usize,
    mut level_gain: G,
    leaf_value: L,
) -> (Vec<SparseTreeNode<F>>, Vec<f64>)
where
    F: Float + CubeElement + bytemuck::Pod,
    G: FnMut(&[u32], &[F], &[F], &[u32], usize) -> (Vec<f64>, Vec<bool>),
    L: Fn(f64, f64) -> f64,
{
    assert!(n_bins >= 2, "build_tree needs n_bins >= 2");
    assert_eq!(binned.len(), n_samples * n_feat, "binned shape");
    assert_eq!(y.len(), n_samples, "y shape");
    assert_eq!(bin_edges.len(), n_feat, "bin_edges must have one row per feature");

    let n_cand = n_feat * (n_bins - 1);
    // Shared candidate (feature, bin) maps: candidate c splits feature f after bin b.
    let mut col_of = vec![0u32; n_cand];
    let mut bin_of = vec![0u32; n_cand];
    for c in 0..n_cand {
        col_of[c] = (c / (n_bins - 1)) as u32;
        bin_of[c] = (c % (n_bins - 1)) as u32;
    }

    let leaf_placeholder = SparseTreeNode::<F> {
        colid: -1,
        threshold: from_f64::<F>(0.0),
        left_child: -1,
        value: -1,
    };
    let mut nodes: Vec<SparseTreeNode<F>> = vec![leaf_placeholder];
    let mut leaf_buffer: Vec<f64> = Vec::new();
    let mut node_id: Vec<u32> = vec![0u32; n_samples];
    let mut frontier: Vec<u32> = vec![0u32];
    let mut depth = 0usize;

    // (node, feature, bin) -> flat histogram cell index.
    let cell = |nid: usize, f: usize, b: usize| (nid * n_feat + f) * n_bins + b;

    // Convert a leaf node in place: colid=-1, left_child=-1, value=offset (D-04).
    let make_leaf = |nodes: &mut Vec<SparseTreeNode<F>>,
                     leaf_buffer: &mut Vec<f64>,
                     nid: usize,
                     sum_y: f64,
                     tot: f64| {
        let v = leaf_value(sum_y, tot);
        let off = leaf_buffer.len() as i32;
        leaf_buffer.push(v);
        nodes[nid].colid = -1;
        nodes[nid].left_child = -1;
        nodes[nid].value = off;
        nodes[nid].threshold = from_f64::<F>(0.0);
    };

    while !frontier.is_empty() && depth < max_depth {
        let n_nodes_total = nodes.len();

        // 1) device histogram for every current node.
        let (counts, vsums) =
            launch_histogram::<F>(&node_id, binned, y, n_samples, n_feat, n_nodes_total, n_bins);

        // 2) criterion-specific gain + per-node purity for this level.
        let (gain_h, pure) =
            level_gain(&node_id, &counts, &vsums, &frontier, n_nodes_total);

        // 3) device split-find argmax per node.
        let gain_f: Vec<F> = gain_h.iter().map(|&g| from_f64::<F>(g)).collect();
        let (best_gain, best_col, best_bin) =
            launch_split_find::<F>(&gain_f, &col_of, &bin_of, n_nodes_total, n_cand);

        // 4) host leaf/internal decision + per-node frontier arrays for relabel.
        let mut split_active = vec![0u32; n_nodes_total];
        let mut split_col = vec![0u32; n_nodes_total];
        let mut split_bin = vec![0u32; n_nodes_total];
        let mut left_child = vec![0u32; n_nodes_total];
        let mut next_frontier: Vec<u32> = Vec::new();

        for &nid_u in &frontier {
            let nid = nid_u as usize;
            let mut tot = 0.0f64;
            let mut sum_y = 0.0f64;
            for b in 0..n_bins {
                tot += host_to_f64(counts[cell(nid, 0, b)]);
                sum_y += host_to_f64(vsums[cell(nid, 0, b)]);
            }
            let g = host_to_f64(best_gain[nid]);
            let can_split = g > 0.0 && !pure[nid] && (tot as usize) >= min_samples;
            if can_split {
                let f = best_col[nid] as usize;
                let b = best_bin[nid] as usize;
                let lc = nodes.len() as i32;
                nodes[nid].colid = f as i32;
                nodes[nid].threshold = from_f64::<F>(bin_edges[f][b]);
                nodes[nid].left_child = lc;
                nodes[nid].value = -1;
                // Adjacent children (D-02): left = lc, right = lc + 1.
                nodes.push(leaf_placeholder);
                nodes.push(leaf_placeholder);
                split_active[nid] = 1;
                split_col[nid] = f as u32;
                split_bin[nid] = b as u32;
                left_child[nid] = lc as u32;
                next_frontier.push(lc as u32);
                next_frontier.push((lc + 1) as u32);
            } else {
                make_leaf(&mut nodes, &mut leaf_buffer, nid, sum_y, tot);
            }
        }

        // 5) device relabel: move samples into their child nodes.
        node_id = launch_relabel(
            &node_id,
            binned,
            &split_active,
            &split_col,
            &split_bin,
            &left_child,
            n_samples,
            n_feat,
        );

        frontier = next_frontier;
        depth += 1;
    }

    // Remaining frontier nodes (hit max_depth) become leaves.
    if !frontier.is_empty() {
        let n_nodes_total = nodes.len();
        let (counts, vsums) =
            launch_histogram::<F>(&node_id, binned, y, n_samples, n_feat, n_nodes_total, n_bins);
        for &nid_u in &frontier {
            let nid = nid_u as usize;
            let mut tot = 0.0f64;
            let mut sum_y = 0.0f64;
            for b in 0..n_bins {
                tot += host_to_f64(counts[cell(nid, 0, b)]);
                sum_y += host_to_f64(vsums[cell(nid, 0, b)]);
            }
            make_leaf(&mut nodes, &mut leaf_buffer, nid, sum_y, tot);
        }
    }

    (nodes, leaf_buffer)
}
