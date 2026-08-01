//! Host-side core-distance scan for the HDBSCAN feature-metric front-end
//! (HDBS-PERF-CPU).
//!
//! ## Why this exists (the 234-second measurement)
//! `feature_metric_single_linkage` sourced its core distances from the Phase-13
//! device prim [`knn_graph`](mlrs_backend::prims::knn_graph::knn_graph), which
//! composes `distance → top_k` QUERY-AXIS TILED at `QUERY_TILE = 8` rows. On a
//! GPU that tiling is the whole point (it keeps the `n×n` distance block off the
//! device — the memory gate). On `cubecl-cpu` it is pathological: that backend
//! maps one OS THREAD per unit and JITs at LLVM `-O0`, so `n/8` tiles become
//! `2·n/8` kernel launches whose per-launch cost dwarfs their work. Measured on
//! this 16-core host at `n = 1000, d = 8` (`hdbscan_perf_test::
//! hdbscan_fit_stage_breakdown`):
//!
//! ```text
//!   knn_graph (device)           234.1541 s     <-- 99.98% of the fit
//!   mst_from_data_matrix           0.0041 s
//!   everything else                0.0312 s
//! ```
//!
//! sklearn fits the same problem in 0.0145 s. The core distances are the ONLY
//! device work on the euclidean path — every later stage (MST, single linkage,
//! condense, select) is already pure host scalar Rust — so on the cpu backend the
//! device round-trip buys nothing and costs four orders of magnitude.
//!
//! [`core_distances_host`] replaces it there with a direct host scan. It is NOT a
//! different algorithm: it computes the same value the KNN prim does — the
//! `(k-1)`-th smallest distance from each row to the whole set, self-zero
//! included — so labels are bit-identical to the device path (gated by
//! `hdbscan_test::core_distances_host_matches_device`). It serves the four
//! Variant-B FAST metrics; cosine and precomputed take their core distances from
//! a dense matrix instead ([`super::mst::core_distances_dense`]).
//!
//! ## What makes the host scan fast
//! - **A KD-tree, when it prunes (HDBS-PRED-CPU).** The levers below all make the
//!   `O(n²)` scan cheaper per pair; [`super::kdtree`] instead stops evaluating most
//!   of the pairs, which is worth 1.3-20× on the stage depending on `p` and how
//!   much density structure the data has. It computes the same distances with the
//!   same accumulator, so the core distances are bit-identical either way, and each
//!   worker measures the tree's actual pruning on its first few rows and falls back
//!   to the brute scan below if it is not paying off — see that module for the
//!   measurements and for why the decision is made from data rather than from `p`.
//! - **Bounded insertion list.** Only the `k` smallest distances per row are
//!   kept (`k = min_samples`, typically 5-25), in a sorted `k`-element array. A
//!   candidate beyond the current `k`-th is rejected in one compare, so the row
//!   never sorts its `n` distances (the `O(n log n)`-per-row full sort in
//!   [`super::mst::core_distances_dense`] is what the dense paths still pay).
//! - **Partial-distance early exit.** The feature loop bails out once the running
//!   accumulator passes the current `k`-th distance, so a far-away point costs a
//!   block of features instead of `d` (and, for Euclidean, no `sqrt`). The bound
//!   is screened once per `distance::SCREEN_BLOCK` features, NOT per feature —
//!   see that constant for the measurement that settled the granularity (per
//!   feature was a 30% regression).
//! - **Row-parallel.** Rows are independent, so the scan splits over
//!   [`cpu_launch_units`](mlrs_backend::capability::cpu_launch_units) scoped
//!   threads with no shared mutable state and no barrier — the
//!   `prims::linear_predict` / `prims::random_forest` `std::thread::scope`
//!   precedent.
//!
//! All scalar math is `f64` (the host bridging domain), matching every other
//! HDBSCAN host stage.
//!
//! Tests live in `crates/mlrs-algos/tests/hdbscan_test.rs` (AGENTS.md §2).

use mlrs_backend::capability;

use super::distance::{
    chebyshev_screened, manhattan_screened, minkowski_screened, sq_euclidean_screened,
};
use super::kdtree;
use super::Metric;

/// Should the host scan serve the core distances on this backend?
///
/// True on `cpu` only. On a GPU backend the device `knn_graph` prim is both
/// faster and the shape the memory gate is written against, so the host scan
/// never takes over there — a perf path is gated on the target it was MEASURED
/// on, never extrapolated onto another backend.
///
/// `MLRS_HDBSCAN_HOST_CORE=0` forces the device prim back on for on-target A/B;
/// `=1` cannot force the host scan onto a non-cpu backend.

/// Additionally forced when the element type is `f64` and the backend cannot
/// evaluate f64 transcendentals: the device path this replaces evaluates
/// the `knn_graph` prim's minkowski `powf`, and on such a backend that does not fail at launch — the
/// driver's shader compiler SEGFAULTS (measured: `ACO ERROR: Unimplemented NIR
/// instr bit size: 64 fexp2` → `signal: 11`). This arm is the only way the f64
/// path can run there at all, so the capability check is NOT overridable by the
/// `MLRS_HDBSCAN_HOST_CORE` A/B knob, which stays a pure perf switch. See
/// `mlrs_backend::capability::f64_transcendental_supported`.
pub fn host_core_applicable<F>() -> bool {
    if std::mem::size_of::<F>() == 8 && !capability::f64_transcendental_supported() {
        return true;
    }
    capability::active_backend_name() == "cpu"
        && mlrs_backend::abflag::var("MLRS_HDBSCAN_HOST_CORE")
            .map(|v| v != "0")
            .unwrap_or(true)
}

/// Split `out` into contiguous row blocks and run `f(row0, block)` on each over
/// scoped threads (HDBS-PERF-CPU).
///
/// The shared shape behind every row-independent host pass in this back-end:
/// core distances, the dense distance matrix, the dense mutual-reachability. Each
/// block owns a disjoint output slice, so there is no locking and no barrier —
/// the `prims::linear_predict` / `prims::random_forest` `std::thread::scope`
/// precedent.
///
/// `out` is `rows × row_width` row-major; blocks are always cut on a ROW
/// boundary, and `f` receives the block's first row index. `min_rows` is the
/// smallest block worth a thread; below it (or on a single-unit machine) the
/// whole range runs inline, because spawning costs more than the work.
///
/// Splitting NEVER changes a value — every block computes the same rows it would
/// serially, in the same order — so this is wall clock only.
pub(super) fn par_row_chunks<T, F>(out: &mut [T], row_width: usize, min_rows: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    debug_assert!(row_width >= 1, "row_width must be positive");
    debug_assert_eq!(
        out.len() % row_width,
        0,
        "out must be a whole number of rows"
    );
    let rows = out.len() / row_width;
    if rows == 0 {
        return;
    }
    let units = capability::cpu_launch_units().max(1) as usize;
    let rows_per_chunk = rows.div_ceil(units).max(min_rows);
    if rows_per_chunk >= rows || units == 1 {
        f(0, out);
        return;
    }
    let f = &f;
    std::thread::scope(|scope| {
        for (c, block) in out.chunks_mut(rows_per_chunk * row_width).enumerate() {
            let row0 = c * rows_per_chunk;
            scope.spawn(move || f(row0, block));
        }
    });
}

/// A sorted, bounded "`k` smallest so far" list. `vals[0..len]` is ascending and
/// `len <= k`; [`worst`] is the rejection threshold the feature loop prunes
/// against.
struct KSmallest {
    vals: Vec<f64>,
    k: usize,
}

impl super::kdtree::Bounded for KSmallest {
    #[inline]
    fn worst(&self) -> f64 {
        KSmallest::worst(self)
    }
    #[inline]
    fn offer(&mut self, d: f64, _idx: u32) {
        // Core distances are a pure distance quantity — the neighbour identity
        // the traversal now reports is UMAP's `host_knn` requirement, not this
        // one's.
        self.push(d);
    }
}

impl KSmallest {
    fn new(k: usize) -> Self {
        Self {
            vals: Vec::with_capacity(k),
            k,
        }
    }

    /// The current `k`-th smallest, or `+inf` while the list is still filling.
    /// Any candidate `>= worst()` cannot enter, so this doubles as the
    /// partial-distance early-exit threshold.
    #[inline]
    fn worst(&self) -> f64 {
        if self.vals.len() < self.k {
            f64::INFINITY
        } else {
            self.vals[self.k - 1]
        }
    }

    /// Insert `v` if it belongs in the `k` smallest (sorted insert, drop the
    /// tail). NaN never enters (`total_cmp` would order it last anyway, and the
    /// `< worst()` guard at the call site rejects it).
    #[inline]
    fn push(&mut self, v: f64) {
        let mut pos = self.vals.len();
        while pos > 0 && self.vals[pos - 1] > v {
            pos -= 1;
        }
        if self.vals.len() < self.k {
            self.vals.insert(pos, v);
        } else if pos < self.k {
            self.vals.pop();
            self.vals.insert(pos, v);
        }
    }
}

/// `core[i]` = the `(k-1)`-th smallest distance from row `i` to the whole set,
/// INCLUDING the self-zero at position 0 — i.e. exactly what
/// `knn_graph(include_self = true)` returns in column `k-1`.
///
/// `x` is the row-major `n × p` host design matrix; `metric` is one of the four
/// Variant-B FAST metrics (euclidean / manhattan / chebyshev / minkowski —
/// cosine and precomputed take the dense route and never reach here). `k` must be
/// in `1..=n`; the caller clamps it exactly as the device path does.
///
/// Rows are computed independently across scoped threads (chunked so each thread
/// owns a contiguous output slice — no locking, no false sharing beyond the chunk
/// boundary).
pub fn core_distances_host(x: &[f64], n: usize, p: usize, metric: Metric, k: usize) -> Vec<f64> {
    debug_assert_eq!(x.len(), n * p, "x must be a dense n×p matrix");
    debug_assert!(k >= 1 && k <= n, "k must be clamped to 1..=n by the caller");
    let mut core = vec![0.0f64; n];
    if n == 0 {
        return core;
    }

    // One tree for the whole scan, built BEFORE the row split (it is read-only and
    // shared by every worker). `None` falls through to the brute scan.
    let tree = if kdtree::kd_applicable(n, p) {
        Some(kdtree::build_tree(x, n, p))
    } else {
        None
    };
    let tree = tree.as_ref();
    // Resolved HERE, on the calling thread: `abflag` overrides are THREAD-LOCAL,
    // so a worker spawned below would not see a test's forced value and would
    // silently re-enable the calibration the test is trying to switch off.
    let forced = kdtree::kd_forced();

    par_row_chunks(&mut core, 1, 64, |row0, out| {
        // The visited count is a probe-only quantity here (the route calibration
        // consumes it inside the block); nothing in `fit` reads it.
        let _ = core_rows(x, n, p, metric, k, row0, out, tree, forced);
    });
    core
}

/// Core distances for the contiguous row block `row0..row0 + out.len()`.
#[allow(clippy::too_many_arguments)]
fn core_rows(
    x: &[f64],
    n: usize,
    p: usize,
    metric: Metric,
    k: usize,
    row0: usize,
    out: &mut [f64],
    tree: Option<&kdtree::KdTree>,
    forced: bool,
) -> usize {
    // Monomorphize the feature loop per metric: the `match` is hoisted OUT of
    // the O(n²·p) scan so the inner loop is a straight-line accumulate. A single
    // `host_pairwise`-style match inside the scan would re-branch n² times.
    match metric {
        // SLACK on the Euclidean/Minkowski thresholds (and only those): their
        // `fin` is a root, so `aggregate >= pre(w)` does not imply
        // `fin(aggregate) >= w` to the last ULP — `sqrt(fl(w²))` can land one ULP
        // BELOW `w`. Widening the bail-out threshold by a few epsilons means a
        // candidate within rounding distance of the current k-th falls through to
        // the exact `d < w` compare instead of being pruned on a rounded
        // inequality, so the host scan agrees with the device prim bit for bit.
        // Manhattan/Chebyshev need no slack: their `fin` is the identity, so the
        // aggregate comparison IS the distance comparison.
        //
        // `axis_agg` is the SAME aggregation one axis at a time, which is what
        // makes the KD-tree box bound a valid lower bound (see `kdtree`).
        Metric::Euclidean => scan(
            x,
            n,
            p,
            k,
            row0,
            out,
            tree,
            forced,
            sq_euclidean_screened,
            f64::sqrt,
            |t| t * t * (1.0 + 4.0 * f64::EPSILON),
            |s: f64, o: f64| s + o * o,
        ),
        Metric::Manhattan => scan(
            x,
            n,
            p,
            k,
            row0,
            out,
            tree,
            forced,
            manhattan_screened,
            |s| s,
            |t| t,
            |s: f64, o: f64| s + o,
        ),
        Metric::Chebyshev => scan(
            x,
            n,
            p,
            k,
            row0,
            out,
            tree,
            forced,
            chebyshev_screened,
            |s| s,
            |t| t,
            |s: f64, o: f64| if o > s { o } else { s },
        ),
        Metric::Minkowski { p: pp } => {
            // Accumulate Σ|Δ|^pp, finish with the 1/pp root; the early-exit
            // threshold is therefore the raw threshold raised to pp.
            let acc = move |a: &[f64], b: &[f64], bound: f64| minkowski_screened(a, b, bound, pp);
            let fin = move |s: f64| s.powf(1.0 / pp);
            let pre = move |t: f64| t.powf(pp) * (1.0 + 4.0 * f64::EPSILON);
            let axis = move |s: f64, o: f64| s + o.powf(pp);
            scan(x, n, p, k, row0, out, tree, forced, acc, fin, pre, axis)
        }
        // Cosine (Variant A) and Precomputed both derive their core distances
        // from a dense `n×n` matrix via `mst::core_distances_dense`, so neither
        // reaches the kNN-shaped scan. Cosine deliberately so: `knn_graph`'s
        // cosine uses the GEMM expansion `‖x̂ − ŷ‖²/2`, which disagrees with
        // `cosine_distance_matrix`'s `1 − x̂·ŷ` on a zero row — and the MR kernel
        // needs core distances from the matrix it actually reduces.
        Metric::Cosine | Metric::Precomputed => unreachable!(
            "core_distances_host serves the Variant-B FAST metrics only; cosine and \
             precomputed take their core distances from mst::core_distances_dense"
        ),
    }
}

/// The shared bounded-insertion row scan.
///
/// `acc(a, b, bound)` accumulates the metric's PRE-final aggregate (squared sum
/// for Euclidean, Σ|Δ| for Manhattan, …) and may bail out early returning
/// `+inf` once the running value reaches `bound`; `fin` turns that aggregate
/// into the true distance; `pre` maps a true distance back into the aggregate
/// domain so the current `k`-th distance can be used as `bound`.
///
/// Because the early exit is expressed entirely in the aggregate domain, a bail
/// out can only happen when the true distance is already `>= worst()`, which by
/// definition cannot enter the `k` smallest. The result is identical to scanning
/// every feature.
#[allow(clippy::too_many_arguments)]
#[inline]
fn scan<A, Fin, Pre, Ax>(
    x: &[f64],
    n: usize,
    p: usize,
    k: usize,
    row0: usize,
    out: &mut [f64],
    tree: Option<&kdtree::KdTree>,
    forced: bool,
    acc: A,
    fin: Fin,
    pre: Pre,
    axis_agg: Ax,
) -> usize
where
    A: Fn(&[f64], &[f64], f64) -> f64,
    Fin: Fn(f64) -> f64,
    Pre: Fn(f64) -> f64,
    Ax: Fn(f64, f64) -> f64,
{
    let mut best = KSmallest::new(k);

    // KD-tree route: same per-pair arithmetic, most pairs never evaluated. The
    // `k`-th smallest of the evaluated multiset is the `k`-th smallest overall
    // because the prune only skips points already `>=` the running `k`-th (see
    // `kdtree`), so this yields the identical core distances.
    //
    // The first `CALIB_ROWS` rows double as the route calibration: if the tree
    // turns out not to be pruning on THIS data, the loop breaks and the remaining
    // rows fall through to the brute scan below. Nothing is recomputed — the rows
    // already done hold the same values either route would have produced.
    let ops = kdtree::MetricOps {
        acc,
        fin,
        pre,
        axis_agg,
    };
    let mut done = 0usize;
    let mut visited = 0usize;
    if let Some(tree) = tree {
        for (r, slot) in out.iter_mut().enumerate() {
            let i = row0 + r;
            best.vals.clear();
            visited += kdtree::query(tree, x, &x[i * p..(i + 1) * p], &ops, &mut best);
            *slot = best.vals.get(k - 1).copied().unwrap_or(f64::INFINITY);
            done = r + 1;
            if !forced && done == kdtree::CALIB_ROWS && kdtree::brute_is_cheaper(visited, done, n) {
                break;
            }
        }
        if done == out.len() {
            return visited;
        }
    }

    // Brute scan — the whole block when there is no tree, or the tail after the
    // calibration abandoned one.
    for (r, slot) in out.iter_mut().enumerate().skip(done) {
        let i = row0 + r;
        best.vals.clear();
        let xi = &x[i * p..(i + 1) * p];
        for j in 0..n {
            let w = best.worst();
            // `bound` in the aggregate domain; `+inf` maps to `+inf` under every
            // `pre` here (x², |x|, x^pp all preserve infinity).
            let bound = if w.is_finite() {
                (ops.pre)(w)
            } else {
                f64::INFINITY
            };
            let xj = &x[j * p..(j + 1) * p];
            let a = (ops.acc)(xi, xj, bound);
            if a.is_finite() {
                let d = (ops.fin)(a);
                if d < w {
                    best.push(d);
                }
            }
        }
        // `k <= n` is a caller invariant and every finite candidate is admitted
        // while the list fills, so index `k-1` normally exists. `get` rather than
        // `[]` because a NaN feature makes `d < w` false for EVERY candidate: the
        // list would stay short and a direct index would panic. A degenerate
        // `+inf` core distance is the better failure — it propagates as an
        // unclusterable point instead of tearing down the caller.
        *slot = best.vals.get(k - 1).copied().unwrap_or(f64::INFINITY);
    }
    visited
}

/// The fraction of `n` that ONE KD-tree query evaluates, averaged over a strided
/// sample of rows — the quantity [`kdtree::brute_is_cheaper`] thresholds on.
///
/// Exposed for `hdbscan_perf_test::hdbscan_core_distance_sweep`, which reports it
/// alongside the measured speedup so [`kdtree`]'s `BRUTE_RATIO` is set from
/// observed pruning rather than from a guess. Not used by `fit`.
pub fn kd_visited_fraction_probe(
    x: &[f64],
    n: usize,
    p: usize,
    metric: Metric,
    k: usize,
) -> f64 {
    if n == 0 || !kdtree::kd_applicable(n, p) {
        return 1.0;
    }
    let tree = kdtree::build_tree(x, n, p);
    let rows = 64usize.min(n);
    let stride = (n / rows).max(1);
    let mut visited = 0usize;
    let mut counted = 0usize;
    for s in 0..rows {
        let i = s * stride;
        if i >= n {
            break;
        }
        // One row block of length 1 per sample, so `scan` runs the tree route on
        // exactly that row; `forced = true` keeps the calibration from switching
        // routes underneath the measurement.
        let mut slot = [0.0f64; 1];
        visited += core_rows(x, n, p, metric, k, i, &mut slot, Some(&tree), true);
        counted += 1;
    }
    if counted == 0 {
        return 1.0;
    }
    (visited as f64 / counted as f64) / n as f64
}
