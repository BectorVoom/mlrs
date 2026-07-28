//! Host-side directed k-NN graph for the UMAP fit front-end (UMAP-PERF-CPU).
//!
//! ## Why this exists (the measurement)
//! [`run_umap_layout`](super::umap) sources its neighbour graph from the
//! Phase-13 device prim
//! [`knn_graph`](mlrs_backend::prims::knn_graph::knn_graph), which composes
//! `distance → top_k` QUERY-AXIS TILED at `QUERY_TILE = 8` rows. On a GPU that
//! tiling is the whole point (it keeps the `n×n` distance block off the device).
//! On `cubecl-cpu` it is pathological — one OS thread per unit, LLVM `-O0`, and
//! `2·n/8` kernel launches whose per-launch cost dwarfs their work. This is the
//! SAME pathology `cluster::hdbscan::host_core` was written for, measured there
//! at 234 s for a 234.3 s fit. For UMAP, on this 16-core host at `n = 500,
//! d = 8, k = 15` (`umap_perf_test::umap_fit_stage_breakdown`):
//!
//! ```text
//!   fit @ 1 epoch (graph + init)     5.4759 s   <-- almost all of it knn_graph
//! ```
//!
//! umap-learn builds the same graph in single-digit milliseconds.
//!
//! [`host_knn`] replaces it on the cpu backend with a direct host scan. It is NOT
//! a different algorithm: it returns the same `(n, k)` self-dropped ascending
//! neighbour lists the prim does, under the mlrs tie-break convention (equal
//! distances ordered by LOWEST row index), so every downstream stage
//! (`smooth_knn_dist` → `compute_membership_strengths` → `fuzzy_union`) sees the
//! same input. Gated by `umap_host_knn_test::host_knn_matches_device`.
//!
//! ## What makes the host scan fast
//! - **A KD-tree, when it prunes.** The levers below make each PAIR cheaper;
//!   [`kdtree`](crate::cluster::hdbscan::kdtree) instead stops most pairs from
//!   being evaluated at all, which is what lets the exact graph keep pace with
//!   umap-learn's APPROXIMATE NN-descent as `n` grows. The tree is shared with
//!   HDBSCAN (same structure, same accumulators, same conservative bound), and
//!   each worker measures its actual pruning on its first few rows and drops
//!   back to the scan if it is not paying off — see that module for why the
//!   decision is made from data rather than from `d`.
//! - **Bounded insertion list.** Only the `k` smallest `(distance, index)` pairs
//!   per row are kept, in a sorted `k`-element array; a candidate beyond the
//!   current `k`-th is rejected in one compare, so a row never sorts its `n`
//!   distances.
//! - **Partial-distance early exit.** The feature loop bails out once the running
//!   accumulator passes the current `k`-th, screened once per
//!   [`SCREEN_BLOCK`](crate::cluster::hdbscan::distance::SCREEN_BLOCK) features
//!   (per-feature screening was a measured 30% REGRESSION in the HDBSCAN pass —
//!   see that constant). The screened accumulators are shared with HDBSCAN
//!   rather than re-derived, so the two back-ends cannot drift.
//! - **Row-parallel.** Rows are independent, so the scan splits over
//!   [`cpu_launch_units`](mlrs_backend::capability::cpu_launch_units) scoped
//!   threads with no shared mutable state and no barrier.
//!
//! Screening happens in the metric's MONOTONE pre-image (squared Euclidean, the
//! un-rooted Minkowski power sum) while the list itself holds FINAL distances —
//! the `host_core::scan` shape. Keeping the bound in the aggregate domain is
//! what makes the early exit free of any effect on the result; keeping the list
//! in the distance domain is what lets the brute and tree routes share one
//! comparison and produce identical lists.
//!
//! All scalar math is `f64` (the host bridging domain), matching every other
//! UMAP host stage.
//!
//! Tests live in `crates/mlrs-algos/tests/umap_host_knn_test.rs` (AGENTS.md §2).

use mlrs_backend::capability;

use crate::cluster::hdbscan::distance::{
    chebyshev_screened, manhattan_screened, minkowski_screened, sq_euclidean_screened,
};
use crate::cluster::hdbscan::kdtree;

use super::umap::Metric;

/// Smallest row block worth its own thread. Below this the spawn costs more than
/// the work (the `host_core::par_row_chunks` precedent).
const MIN_ROWS_PER_THREAD: usize = 16;

/// Should the host scan serve the UMAP neighbour graph on this backend?
///
/// True on `cpu` only. On a GPU backend the device `knn_graph` prim is both
/// faster and the shape the memory gate is written against, so the host scan
/// never takes over there — a perf path is gated on the target it was MEASURED
/// on, never extrapolated onto another backend (the
/// `mlrs-feedback-verify-on-target-hardware` rule).
///
/// `MLRS_UMAP_HOST_KNN=0` forces the device prim back on for on-target A/B; `=1`
/// cannot force the host scan onto a non-cpu backend.
pub fn host_knn_applicable() -> bool {
    capability::active_backend_name() == "cpu"
        && mlrs_backend::abflag::var("MLRS_UMAP_HOST_KNN")
            .map(|v| v != "0")
            .unwrap_or(true)
}

/// A sorted, bounded "`k` smallest `(distance, index)` so far" list.
///
/// `d[0..len]` is ascending and ties are ordered by ASCENDING index — the mlrs
/// `top_k` tie-break convention, which the device prim also emits. [`worst`] is
/// the rejection threshold the feature loop prunes against.
///
/// `skip` is the query's own row: UMAP's graph is self-dropped
/// (`include_self = false`), and rejecting it HERE rather than in the scan loop
/// is what lets the KD-tree traversal — which has no notion of a forbidden
/// index — be reused unchanged.
struct KNearest {
    d: Vec<f64>,
    i: Vec<u32>,
    k: usize,
    skip: u32,
}

impl kdtree::Bounded for KNearest {
    #[inline]
    fn worst(&self) -> f64 {
        KNearest::worst(self)
    }
    #[inline]
    fn offer(&mut self, d: f64, idx: u32) {
        self.push(d, idx);
    }
}

impl KNearest {
    fn new(k: usize, skip: u32) -> Self {
        Self {
            d: Vec::with_capacity(k),
            i: Vec::with_capacity(k),
            k,
            skip,
        }
    }

    /// The current `k`-th smallest distance, or `+inf` while the list is still
    /// filling. Any candidate strictly greater cannot enter, so this doubles as
    /// the partial-distance early-exit threshold. A candidate EQUAL to it may
    /// still enter (on a lower index), which is why the screen uses `>=` on the
    /// accumulator only when the list is full and the caller re-tests the tie.
    #[inline]
    fn worst(&self) -> f64 {
        if self.d.len() < self.k {
            f64::INFINITY
        } else {
            self.d[self.k - 1]
        }
    }

    /// Insert `(v, idx)` if it belongs in the `k` smallest, keeping the list
    /// ascending by `(distance, index)`. NaN never enters, nor does the query's
    /// own row.
    #[inline]
    fn push(&mut self, v: f64, idx: u32) {
        if v.is_nan() || idx == self.skip {
            return;
        }
        let full = self.d.len() == self.k;
        if full {
            let (wd, wi) = (self.d[self.k - 1], self.i[self.k - 1]);
            if v > wd || (v == wd && idx >= wi) {
                return;
            }
        }
        // Find the insertion point: after every entry that sorts strictly before
        // `(v, idx)` under the (distance, index) order.
        let mut pos = self.d.len();
        while pos > 0 {
            let (pd, pi) = (self.d[pos - 1], self.i[pos - 1]);
            if pd < v || (pd == v && pi < idx) {
                break;
            }
            pos -= 1;
        }
        if full {
            self.d.pop();
            self.i.pop();
        }
        self.d.insert(pos, v);
        self.i.insert(pos, idx);
    }
}

/// Directed k-NN graph of the row-major `n × d` host matrix `x` under `metric`,
/// SELF-DROPPED (a row is never its own neighbour — UMAP's `include_self =
/// false`).
///
/// Returns `(knn_idx, knn_dist)`, both row-major `(n, k)` host `f64`, ascending
/// per row with the lowest-index tie-break. Indices are FLOAT-ENCODED to match
/// what the device prim's `to_host` + cast produces at this call site, so the
/// membership stages consume an identical buffer either way.
///
/// `k` must satisfy `1 <= k <= n - 1`; the caller (`run_umap_layout`) already
/// clamps it that way.
pub fn host_knn(
    x: &[f64],
    n: usize,
    d: usize,
    k: usize,
    metric: Metric,
) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x.len(), n * d, "x must be a dense n×d matrix");
    let k = k.min(n.saturating_sub(1)).max(1);

    // Cosine runs on L2-NORMALISED rows and reports `‖x̂ − ŷ‖² / 2`, which is the
    // exact value the device path emits (its GEMM expansion yields `2(1−cos)`
    // and `knn_graph` halves it). Going through the norm difference rather than
    // `1 − x̂·ŷ` keeps the ZERO-ROW case identical to the device path too — the
    // two forms disagree there (0.5 vs 1.0), a landmine the HDBSCAN pass hit.
    let normalized;
    let data: &[f64] = match metric {
        Metric::Cosine => {
            normalized = l2_normalize_rows(x, n, d);
            &normalized
        }
        _ => x,
    };

    let mut idx_out = vec![0.0f64; n * k];
    let mut dist_out = vec![0.0f64; n * k];
    if n == 0 || k == 0 {
        return (idx_out, dist_out);
    }

    // A KD-tree stops most pairs from being evaluated at all, which is the only
    // lever left once the pair itself is at its scalar roofline — it is what
    // takes the graph from `O(n²·p)` to something that keeps up with
    // umap-learn's NN-descent at n ≥ 10_000. Whether the BUILT tree is worth
    // USING is decided per worker from its measured pruning, not from `p`; see
    // `kdtree::brute_is_cheaper` for the measurements behind that.
    //
    // The route is shared verbatim with HDBSCAN's core-distance scan, INCLUDING
    // its A/B knob: `MLRS_HDBSCAN_CORE_KD` forces the tree on (`=1`) or off
    // (`=0`) for BOTH back-ends. One knob for one shared decision is deliberate —
    // the measurements that set `BRUTE_RATIO` are the same measurements either
    // caller would take, and a second name would invite the two to drift.
    let tree = if kdtree::kd_applicable(n, d) {
        Some(kdtree::build_tree(data, n, d))
    } else {
        None
    };
    let tree = tree.as_ref();
    let forced = kdtree::kd_forced();

    // Rows are independent: split the OUTPUT rows over scoped threads, each
    // block owning a disjoint slice. Splitting never changes a value.
    let units = capability::cpu_launch_units().max(1) as usize;
    let rows_per_chunk = n.div_ceil(units).max(MIN_ROWS_PER_THREAD);
    if rows_per_chunk >= n || units == 1 {
        scan_rows(data, n, d, k, metric, 0, tree, forced, &mut idx_out, &mut dist_out);
    } else {
        std::thread::scope(|scope| {
            let mut i_rest: &mut [f64] = &mut idx_out;
            let mut d_rest: &mut [f64] = &mut dist_out;
            let mut row0 = 0usize;
            while row0 < n {
                let rows = rows_per_chunk.min(n - row0);
                let (i_blk, i_tail) = i_rest.split_at_mut(rows * k);
                let (d_blk, d_tail) = d_rest.split_at_mut(rows * k);
                i_rest = i_tail;
                d_rest = d_tail;
                let start = row0;
                scope.spawn(move || {
                    scan_rows(data, n, d, k, metric, start, tree, forced, i_blk, d_blk)
                });
                row0 += rows;
            }
        });
    }

    (idx_out, dist_out)
}

/// Scan the contiguous rows `row0 .. row0 + idx_blk.len()/k` of `data` against
/// the whole set, writing this block's `(index, distance)` lists.
///
/// The `match` on `metric` is hoisted OUT of the `O(n²·p)` scan (the
/// `host_core::rows_for_metric` shape) so the inner feature loop monomorphizes
/// to a straight-line accumulate instead of re-branching `n²` times. Each arm
/// supplies the four metric operations the shared bounded scan and the KD-tree
/// box bound need — see [`kdtree::MetricOps`].
#[allow(clippy::too_many_arguments)]
fn scan_rows(
    data: &[f64],
    n: usize,
    d: usize,
    k: usize,
    metric: Metric,
    row0: usize,
    tree: Option<&kdtree::KdTree>,
    forced: bool,
    idx_blk: &mut [f64],
    dist_blk: &mut [f64],
) {
    // SLACK on the Euclidean/Minkowski thresholds (and only those): their `fin`
    // is a root, so `aggregate >= pre(w)` does not imply `fin(aggregate) >= w`
    // to the last ULP. Widening by a few epsilons lets a candidate within
    // rounding distance of the current k-th fall through to the exact compare
    // instead of being pruned on a rounded inequality. Manhattan/Chebyshev need
    // no slack (`fin` is the identity), and Cosine's `fin`/`pre` are exact
    // powers of two.
    const SLACK: f64 = 1.0 + 4.0 * f64::EPSILON;
    match metric {
        Metric::Euclidean => scan(
            data,
            n,
            d,
            k,
            row0,
            tree,
            forced,
            idx_blk,
            dist_blk,
            sq_euclidean_screened,
            f64::sqrt,
            |t| t * t * SLACK,
            |s: f64, o: f64| s + o * o,
        ),
        // Cosine runs the Euclidean accumulate over the ALREADY-NORMALISED rows
        // and closes with `‖x̂ − ŷ‖² / 2` — the device `knn_graph` cosine scale.
        Metric::Cosine => scan(
            data,
            n,
            d,
            k,
            row0,
            tree,
            forced,
            idx_blk,
            dist_blk,
            sq_euclidean_screened,
            |s| 0.5 * s,
            |t| 2.0 * t,
            |s: f64, o: f64| s + o * o,
        ),
        Metric::Manhattan => scan(
            data,
            n,
            d,
            k,
            row0,
            tree,
            forced,
            idx_blk,
            dist_blk,
            manhattan_screened,
            |s| s,
            |t| t,
            |s: f64, o: f64| s + o,
        ),
        Metric::Chebyshev => scan(
            data,
            n,
            d,
            k,
            row0,
            tree,
            forced,
            idx_blk,
            dist_blk,
            chebyshev_screened,
            |s| s,
            |t| t,
            |s: f64, o: f64| if o > s { o } else { s },
        ),
        Metric::Minkowski { p: pp } => {
            let acc = move |a: &[f64], b: &[f64], bound: f64| minkowski_screened(a, b, bound, pp);
            let fin = move |s: f64| s.powf(1.0 / pp);
            let pre = move |t: f64| t.powf(pp) * SLACK;
            let axis = move |s: f64, o: f64| s + o.powf(pp);
            scan(
                data, n, d, k, row0, tree, forced, idx_blk, dist_blk, acc, fin, pre, axis,
            )
        }
    }
}

/// The shared bounded-insertion row scan, with the KD-tree route and its
/// per-worker calibration.
///
/// `acc(a, b, bound)` accumulates the metric's PRE-final aggregate and may bail
/// out early returning `+inf` once the running value reaches `bound`; `fin`
/// turns that aggregate into the true distance; `pre` maps a true distance back
/// into the aggregate domain so the current `k`-th can be used as `bound`;
/// `axis_agg` folds one axis offset into a box bound (the same aggregation, one
/// axis at a time — what makes the tree's box bound valid).
///
/// Because the early exit lives entirely in the aggregate domain, a bail-out can
/// only happen when the true distance is already `>= worst()`, which by
/// definition cannot enter the `k` smallest — so the result is identical to
/// scanning every feature of every row.
///
/// The tree is ABANDONED mid-block when its measured pruning does not pay off
/// (`kdtree::brute_is_cheaper` over the block's first `CALIB_ROWS` rows), which
/// is a wall-clock decision only: both routes evaluate the same candidates
/// against the same bound and produce identical lists.
#[allow(clippy::too_many_arguments)]
fn scan<A, Fin, Pre, Ax>(
    x: &[f64],
    n: usize,
    p: usize,
    k: usize,
    row0: usize,
    tree: Option<&kdtree::KdTree>,
    forced: bool,
    idx_blk: &mut [f64],
    dist_blk: &mut [f64],
    acc: A,
    fin: Fin,
    pre: Pre,
    axis_agg: Ax,
) where
    A: Fn(&[f64], &[f64], f64) -> f64,
    Fin: Fn(f64) -> f64,
    Pre: Fn(f64) -> f64,
    Ax: Fn(f64, f64) -> f64,
{
    let rows = idx_blk.len() / k;
    let ops = kdtree::MetricOps {
        acc,
        fin,
        pre,
        axis_agg,
    };

    let emit = |r: usize, best: &KNearest, idx_blk: &mut [f64], dist_blk: &mut [f64]| {
        for c in 0..k {
            // A short list is unreachable for `k <= n-1` (the caller clamps it);
            // it would mean fewer than `k` finite candidates existed.
            let (dist, ix) = if c < best.d.len() {
                (best.d[c], best.i[c])
            } else {
                (0.0, 0)
            };
            idx_blk[r * k + c] = ix as f64;
            dist_blk[r * k + c] = dist;
        }
    };

    // --- KD-tree route, with the per-worker calibration on its first rows. ---
    let mut done = 0usize;
    if let Some(tree) = tree {
        let mut visited = 0usize;
        for r in 0..rows {
            let i = row0 + r;
            let mut best = KNearest::new(k, i as u32);
            visited += kdtree::query(tree, x, &x[i * p..(i + 1) * p], &ops, &mut best);
            emit(r, &best, idx_blk, dist_blk);
            done = r + 1;
            if !forced && done == kdtree::CALIB_ROWS && kdtree::brute_is_cheaper(visited, done, n) {
                break;
            }
        }
        if done == rows {
            return;
        }
    }

    // --- Brute route: the remaining rows (all of them when no tree was built,
    //     or the tail after the calibration dropped it). ---
    for r in done..rows {
        let i = row0 + r;
        let qi = &x[i * p..(i + 1) * p];
        let mut best = KNearest::new(k, i as u32);
        for j in 0..n {
            if j == i {
                continue; // self-drop (UMAP's include_self = false)
            }
            let w = best.worst();
            let bound = if w.is_finite() {
                (ops.pre)(w)
            } else {
                f64::INFINITY
            };
            let a = (ops.acc)(qi, &x[j * p..(j + 1) * p], bound);
            if a.is_finite() {
                best.push((ops.fin)(a), j as u32);
            }
        }
        emit(r, &best, idx_blk, dist_blk);
    }
}

/// L2-normalise each row of a row-major `r × d` host matrix (`x̂ = x / ‖x‖₂`,
/// zero-norm rows stay zero) — the Cosine pre-step, mirroring
/// `knn_graph::l2_normalize_rows`.
fn l2_normalize_rows(x: &[f64], r: usize, d: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(r * d);
    for i in 0..r {
        let row = &x[i * d..(i + 1) * d];
        let norm = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for &v in row {
            out.push(v * inv);
        }
    }
    out
}
