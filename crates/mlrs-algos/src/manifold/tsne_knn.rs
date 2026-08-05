//! Barnes-Hut front-end (TSNE-PARAMS): the k-nearest-neighbour graph and the
//! SPARSE joint-probability matrix `P` it feeds.
//!
//! `TSNE(method='barnes_hut')` never forms an `n × n` matrix. sklearn's `_fit`
//! takes `n_neighbors = min(n − 1, int(3·perplexity + 1))`, asks
//! `NearestNeighbors` for that graph, squares its distances, and hands them to
//! `_joint_probabilities_nn`. That is the whole difference between the two
//! methods on the input side, and it is what makes the method `O(u·n)` rather
//! than `O(n²)` in both time and memory.
//!
//! ## Three things that look like details and are not
//!
//! 1. **The perplexity search runs over `k` entries, not `n`.** sklearn's
//!    `_binary_search_perplexity` takes a `using_neighbors = n_neighbors <
//!    n_samples` flag and, when set, drops the `j != i` self-skip entirely —
//!    the query is not in its own neighbour list, so there is nothing to skip.
//!    Keeping the dense skip here would silently zero one real neighbour per
//!    row.
//! 2. **`P + Pᵀ` is a STRUCTURAL union, not an elementwise add.** The kNN graph
//!    is directed: `j` can be among `i`'s neighbours while `i` is not among
//!    `j`'s. The symmetrized matrix therefore has entries the conditional
//!    matrix does not, and building it as "add the transpose into the same
//!    sparsity pattern" drops exactly those. [`joint_probabilities_nn`] unions
//!    the patterns.
//! 3. **`P` is normalized by its FULL sum, not per row.** `sum_P = max(P.sum(),
//!    MACHINE_EPSILON)` then `P /= sum_P`, and — unlike the dense path — there
//!    is NO `max(·, MACHINE_EPSILON)` clamp on the entries afterwards. The
//!    dense path clamps because its zeros are real matrix cells; here a missing
//!    cell is structurally absent and never reaches the gradient.
//!
//! ## Where the neighbours come from
//! Axis-separable metrics ([`TsneMetric::is_axis_separable`]) go through the
//! shared KD-tree ([`crate::manifold::umap_host_knn::host_knn`]) that HDBSCAN
//! and UMAP already use, so the graph is built by pruning rather than by
//! scanning. Everything else — the normalized, count-based, and covariance-
//! mixing metrics — has no box bound to prune against and takes a blocked
//! parallel scan that keeps only `k` candidates per row, so it never
//! materializes a row of `n` distances either.
//!
//! Tests live in `crates/mlrs-algos/tests/tsne_params_test.rs` (AGENTS.md §2).

use crate::error::AlgoError;
use crate::manifold::tsne_metric::{
    pair_distance, prepare_rows, validate_metric_geometry, MetricError, ResolvedMetricParams,
    TsneMetric,
};
use crate::manifold::umap::Metric as UmapMetric;
use crate::manifold::umap_host_knn::host_knn;

/// sklearn `MACHINE_EPSILON` (`np.finfo(np.double).eps`).
const MACHINE_EPSILON: f64 = 2.220_446_049_250_313e-16;

/// sklearn's neighbour count for the Barnes-Hut input graph:
/// `min(n_samples − 1, int(3·perplexity + 1))`.
///
/// The `int()` TRUNCATES (Python semantics), so `perplexity = 30` gives 91, not
/// 92. Floored at 1 so a two-point design still has a graph.
pub fn bh_n_neighbors(n: usize, perplexity: f64) -> usize {
    let raw = (3.0 * perplexity + 1.0).trunc();
    let raw = if raw.is_finite() && raw >= 1.0 {
        raw as usize
    } else {
        1
    };
    raw.min(n.saturating_sub(1)).max(1)
}

/// A directed k-NN graph in the shape the sparse-`P` stage consumes: for every
/// row, its `k` nearest neighbours (self excluded) with their SQUARED
/// distances, ascending.
pub struct KnnGraph {
    /// Neighbour indices, row-major `(n, k)`.
    pub indices: Vec<u32>,
    /// SQUARED neighbour distances, row-major `(n, k)` — sklearn's
    /// `distances_nn.data **= 2`.
    pub sq_distances: Vec<f64>,
    /// Neighbours per row.
    pub k: usize,
}

/// Build the directed k-NN graph under `metric`.
///
/// `k` must be `1..=n-1`; the caller ([`bh_n_neighbors`]) already clamps it.
pub fn knn_graph(
    x: &[f64],
    n: usize,
    d: usize,
    k: usize,
    metric: TsneMetric,
    rp: &ResolvedMetricParams,
    threads: usize,
) -> Result<KnnGraph, AlgoError> {
    validate_metric_geometry(n, d, metric)?;
    let k = k.min(n.saturating_sub(1)).max(1);

    // The KD-tree route: only for metrics that aggregate monotonely over
    // independent feature axes, which is exactly what a box bound can prune.
    if let Some(um) = umap_metric_for(metric, rp) {
        let (idx_f, dist) = host_knn(x, n, d, k, um);
        let mut indices = vec![0u32; n * k];
        let mut sq_distances = vec![0.0f64; n * k];
        for t in 0..n * k {
            indices[t] = idx_f[t] as u32;
            // `host_knn` reports the metric's own value; t-SNE wants it squared.
            // `SqEuclidean` is requested as Euclidean and squared TWICE, which
            // is what sklearn does too (it squares whatever the metric emits).
            let v = dist[t];
            sq_distances[t] = if metric == TsneMetric::SqEuclidean {
                (v * v) * (v * v)
            } else {
                v * v
            };
        }
        return Ok(KnnGraph {
            indices,
            sq_distances,
            k,
        });
    }

    scan_knn(x, n, d, k, metric, rp, threads)
}

/// Map a t-SNE metric onto the shared KD-tree's metric enum, when one exists.
/// `None` routes to the blocked scan.
fn umap_metric_for(metric: TsneMetric, rp: &ResolvedMetricParams) -> Option<UmapMetric> {
    if !metric.is_axis_separable() {
        return None;
    }
    Some(match metric {
        // Squared Euclidean is monotone in Euclidean, so the same tree and the
        // same neighbour ORDER serve both; only the reported value differs.
        TsneMetric::Euclidean | TsneMetric::SqEuclidean => UmapMetric::Euclidean,
        TsneMetric::Manhattan => UmapMetric::Manhattan,
        TsneMetric::Chebyshev => UmapMetric::Chebyshev,
        TsneMetric::Minkowski => {
            // The tree's bound assumes `p >= 1`; below that Minkowski is not a
            // metric and the box bound is not valid, so fall back to the scan.
            if rp.p >= 1.0 && rp.p.is_finite() {
                UmapMetric::Minkowski { p: rp.p }
            } else {
                return None;
            }
        }
        _ => return None,
    })
}

/// Smallest row block worth its own thread.
const MIN_ROWS_PER_THREAD: usize = 8;

/// The general route: a parallel row scan that keeps a bounded, sorted `k`-best
/// list per row. Never materializes a row of `n` distances.
fn scan_knn(
    x: &[f64],
    n: usize,
    d: usize,
    k: usize,
    metric: TsneMetric,
    rp: &ResolvedMetricParams,
    threads: usize,
) -> Result<KnnGraph, AlgoError> {
    let prep = prepare_rows(x, n, d, metric);
    let mut indices = vec![0u32; n * k];
    let mut sq_distances = vec![0.0f64; n * k];
    let err = std::sync::atomic::AtomicUsize::new(usize::MAX);

    let units = threads.max(1).min(n.div_ceil(MIN_ROWS_PER_THREAD).max(1));
    {
        let prep = &prep;
        let err = &err;
        // Rows are disjoint output blocks, so each unit is handed its own
        // contiguous `(rows × k)` slice rather than a shared pointer.
        let mut i_rest: &mut [u32] = &mut indices;
        let mut d_rest: &mut [f64] = &mut sq_distances;
        let mut blocks: Vec<(usize, &mut [u32], &mut [f64])> = Vec::with_capacity(units);
        let rows_per = n.div_ceil(units);
        let mut row0 = 0usize;
        while row0 < n {
            let rows = rows_per.min(n - row0);
            let (i_blk, i_tail) = i_rest.split_at_mut(rows * k);
            let (d_blk, d_tail) = d_rest.split_at_mut(rows * k);
            i_rest = i_tail;
            d_rest = d_tail;
            blocks.push((row0, i_blk, d_blk));
            row0 += rows;
        }

        let run = |row0: usize, i_blk: &mut [u32], d_blk: &mut [f64]| {
            let rows = i_blk.len() / k;
            let mut best_d = vec![f64::INFINITY; k];
            let mut best_i = vec![0u32; k];
            for r in 0..rows {
                let i = row0 + r;
                best_d.iter_mut().for_each(|v| *v = f64::INFINITY);
                best_i.iter_mut().for_each(|v| *v = 0);
                let mut worst = f64::INFINITY;
                let mut filled = 0usize;
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let dv = match pair_distance(prep, d, i, j, metric, rp) {
                        Ok(v) => v,
                        Err(e) => {
                            err.store(e as usize, std::sync::atomic::Ordering::Relaxed);
                            0.0
                        }
                    };
                    // A NaN distance is UNDEFINED, not far away: `nan_euclidean`
                    // between two rows with no coordinate present in both, or
                    // `dice` on two all-zero rows. It must be dropped before it
                    // reaches the list, not merely lose comparisons — while the
                    // list is still filling, nothing rejects it (every compare
                    // against NaN is false), so it would settle into the tail
                    // and BECOME `worst`. From then on `dv < worst` is false for
                    // every remaining candidate and the row's neighbours freeze
                    // at whatever the first `k` happened to be.
                    if dv.is_nan() {
                        continue;
                    }
                    if filled == k && !(dv < worst) {
                        continue;
                    }
                    // Insertion into the sorted prefix; ties keep the LOWER
                    // index, the mlrs `top_k` convention the KD-tree route also
                    // emits, so the two routes agree exactly.
                    let mut pos = filled.min(k - 1);
                    while pos > 0 && best_d[pos - 1] > dv {
                        best_d[pos] = best_d[pos - 1];
                        best_i[pos] = best_i[pos - 1];
                        pos -= 1;
                    }
                    best_d[pos] = dv;
                    best_i[pos] = j as u32;
                    if filled < k {
                        filled += 1;
                    }
                    if filled == k {
                        worst = best_d[k - 1];
                    }
                }
                for t in 0..k {
                    let v = best_d[t];
                    i_blk[r * k + t] = best_i[t];
                    d_blk[r * k + t] = if v.is_finite() { v * v } else { v };
                }
            }
        };

        if blocks.len() <= 1 {
            for (row0, i_blk, d_blk) in blocks {
                run(row0, i_blk, d_blk);
            }
        } else {
            std::thread::scope(|scope| {
                let run = &run;
                let mut iter = blocks.into_iter();
                let first = iter.next();
                for (row0, i_blk, d_blk) in iter {
                    scope.spawn(move || run(row0, i_blk, d_blk));
                }
                if let Some((row0, i_blk, d_blk)) = first {
                    run(row0, i_blk, d_blk);
                }
            });
        }
    }

    if let Some(e) = decode(err.load(std::sync::atomic::Ordering::Relaxed)) {
        return Err(e.into_algo_error());
    }
    Ok(KnnGraph {
        indices,
        sq_distances,
        k,
    })
}

fn decode(code: usize) -> Option<MetricError> {
    Some(match code {
        0 => MetricError::HaversineNot2D,
        1 => MetricError::WMinkowskiRemoved,
        2 => MetricError::SokalSneathAllZero,
        3 => MetricError::BadMetricParamShape,
        4 => MetricError::PrecomputedNotSquare,
        5 => MetricError::NegativeDistance,
        _ => return None,
    })
}

// ===========================================================================
// Sparse joint probabilities
// ===========================================================================

/// The symmetrized, normalized sparse `P`, in CSR with ascending column
/// indices per row — the layout sklearn's `_barnes_hut_tsne.gradient` walks.
pub struct SparseP {
    /// Row offsets, length `n + 1`.
    pub indptr: Vec<usize>,
    /// Column indices, ascending within each row.
    pub indices: Vec<u32>,
    /// Values.
    pub data: Vec<f64>,
}

impl SparseP {
    /// Stored entries.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

/// sklearn `_joint_probabilities_nn`: per-row perplexity search over the `k`
/// neighbour distances, then `P + Pᵀ` as a STRUCTURAL union, then a single
/// global normalization (see the module docs for why each of those three is not
/// the dense path's version).
pub fn joint_probabilities_nn(
    graph: &KnnGraph,
    n: usize,
    perplexity: f64,
    threads: usize,
) -> SparseP {
    let k = graph.k;
    let cond = binary_search_perplexity_nn(&graph.sq_distances, n, k, perplexity, threads);

    // --- P + Pᵀ over the UNION of the two sparsity patterns. Built as a
    //     per-row bucket of (column, value) contributions: row `i` receives
    //     `cond[i][t]` at column `j`, and row `j` receives the same value at
    //     column `i`. ---
    let mut counts = vec![0usize; n];
    for i in 0..n {
        for t in 0..k {
            counts[i] += 1;
            counts[graph.indices[i * k + t] as usize] += 1;
        }
    }
    let mut indptr = vec![0usize; n + 1];
    for i in 0..n {
        indptr[i + 1] = indptr[i] + counts[i];
    }
    let total = indptr[n];
    let mut cols = vec![0u32; total];
    let mut vals = vec![0.0f64; total];
    let mut cursor = indptr[..n].to_vec();
    for i in 0..n {
        for t in 0..k {
            let j = graph.indices[i * k + t] as usize;
            let v = cond[i * k + t];
            let ci = cursor[i];
            cols[ci] = j as u32;
            vals[ci] = v;
            cursor[i] = ci + 1;
            let cj = cursor[j];
            cols[cj] = i as u32;
            vals[cj] = v;
            cursor[j] = cj + 1;
        }
    }

    // --- Coalesce duplicates (a mutual neighbour pair contributes twice to the
    //     same cell, which is exactly the `+ Pᵀ` addition) and sort columns. ---
    let mut out_indptr = vec![0usize; n + 1];
    let mut out_cols: Vec<u32> = Vec::with_capacity(total);
    let mut out_vals: Vec<f64> = Vec::with_capacity(total);
    let mut order: Vec<u32> = Vec::new();
    let mut sum_p = 0.0f64;
    for i in 0..n {
        let lo = indptr[i];
        let hi = indptr[i + 1];
        order.clear();
        order.extend(lo as u32..hi as u32);
        order.sort_unstable_by_key(|&t| cols[t as usize]);
        let mut last: Option<u32> = None;
        for &t in &order {
            let c = cols[t as usize];
            let v = vals[t as usize];
            if last == Some(c) {
                let n_out = out_vals.len();
                out_vals[n_out - 1] += v;
            } else {
                out_cols.push(c);
                out_vals.push(v);
                last = Some(c);
            }
            sum_p += v;
        }
        out_indptr[i + 1] = out_cols.len();
    }

    let sum_p = sum_p.max(MACHINE_EPSILON);
    for v in out_vals.iter_mut() {
        *v /= sum_p;
    }
    SparseP {
        indptr: out_indptr,
        indices: out_cols,
        data: out_vals,
    }
}

/// sklearn `_utils._binary_search_perplexity` in its `using_neighbors = True`
/// mode: the same 100-step bisection on `beta` as the dense path, but over `k`
/// entries and with NO self-skip (the query is not in its own list).
///
/// The distances arrive as `f64` and are rounded through `f32` here, because
/// sklearn hands the Cython routine a `float32` view
/// (`distances_data.astype(np.float32)`) while doing the search itself in
/// `f64`. Skipping the rounding changes `beta` in the fourth decimal and moves
/// the embedding.
///
/// Rows are INDEPENDENT — each bisects its own `beta` against its own `k`
/// distances — so the pass is split over `threads` scoped workers on disjoint
/// output blocks. sklearn's is a serial Cython loop, and at `n = 5000,
/// perplexity = 30` (so `k = 91`) it is the single largest term in the
/// Barnes-Hut setup: up to `100 · k` calls to `exp` per row. Splitting it
/// cannot change a value.
fn binary_search_perplexity_nn(
    sq: &[f64],
    n: usize,
    k: usize,
    perplexity: f64,
    threads: usize,
) -> Vec<f64> {
    let mut p = vec![0.0f64; n * k];
    let desired_entropy = perplexity.ln();
    let units = threads.max(1).min(n.div_ceil(MIN_ROWS_PER_THREAD).max(1));

    let run = |row0: usize, block: &mut [f64]| {
        for (r, out) in block.chunks_exact_mut(k).enumerate() {
            let row = &sq[(row0 + r) * k..(row0 + r) * k + k];
            search_one_row(row, out, k, desired_entropy);
        }
    };

    if units <= 1 {
        run(0, &mut p);
        return p;
    }
    let rows_per = n.div_ceil(units);
    std::thread::scope(|scope| {
        let run = &run;
        let mut rest: &mut [f64] = &mut p;
        let mut row0 = 0usize;
        let mut first: Option<(usize, &mut [f64])> = None;
        while row0 < n {
            let rows = rows_per.min(n - row0);
            let (blk, tail) = rest.split_at_mut(rows * k);
            rest = tail;
            if first.is_none() {
                first = Some((row0, blk));
            } else {
                scope.spawn(move || run(row0, blk));
            }
            row0 += rows;
        }
        if let Some((r0, blk)) = first {
            run(r0, blk);
        }
    });
    p
}

/// One row's `beta` bisection (sklearn's inner `for l in range(n_steps)`).
#[inline]
fn search_one_row(row: &[f64], out: &mut [f64], k: usize, desired_entropy: f64) {
    const EPSILON_DBL: f64 = 1e-8;
    const PERPLEXITY_TOLERANCE: f64 = 1e-5;
    const N_STEPS: usize = 100;

    let mut beta_min = f64::NEG_INFINITY;
    let mut beta_max = f64::INFINITY;
    let mut beta = 1.0f64;

    for _ in 0..N_STEPS {
        let mut sum_pi = 0.0f64;
        for t in 0..k {
            let dv = row[t] as f32 as f64;
            let v = (-dv * beta).exp();
            out[t] = v;
            sum_pi += v;
        }
        if sum_pi == 0.0 {
            sum_pi = EPSILON_DBL;
        }
        let mut sum_disti_pi = 0.0f64;
        for t in 0..k {
            out[t] /= sum_pi;
            sum_disti_pi += (row[t] as f32 as f64) * out[t];
        }
        let entropy_diff = sum_pi.ln() + beta * sum_disti_pi - desired_entropy;
        if entropy_diff.abs() <= PERPLEXITY_TOLERANCE {
            break;
        }
        if entropy_diff > 0.0 {
            beta_min = beta;
            if beta_max == f64::INFINITY {
                beta *= 2.0;
            } else {
                beta = (beta + beta_max) / 2.0;
            }
        } else {
            beta_max = beta;
            if beta_min == f64::NEG_INFINITY {
                beta /= 2.0;
            } else {
                beta = (beta + beta_min) / 2.0;
            }
        }
    }
}
