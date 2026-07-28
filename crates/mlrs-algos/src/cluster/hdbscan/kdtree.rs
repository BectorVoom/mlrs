//! A static KD-tree for the HDBSCAN core-distance scan (HDBS-PRED-CPU).
//!
//! ## Why this exists
//! [`super::host_core::core_distances_host`] answers "the `(k-1)`-th smallest
//! distance from row `i` to the whole set" by scanning ALL `n` rows per query —
//! `O(n²·p)` distance work. That brute scan is already at its scalar roofline:
//! MEASURED at `n = 10_000, d = 16` on this 16-unit host it runs 4 ns/pair
//! single-threaded, and two attempts to make the pair itself cheaper both
//! REGRESSED (a 4-pair interleaved accumulator gained 1.10× because the
//! out-of-order engine already overlaps consecutive pairs; an 8-row cache tile
//! LOST 18% because it doubles the per-pair load count, and the scan is L1
//! load-throughput-bound rather than L3-bandwidth-bound). The only lever left is
//! to stop computing most of the pairs.
//!
//! A KD-tree does exactly that: a query descends to its own leaf and then prunes
//! every subtree whose bounding box is already farther than the `k`-th distance
//! found so far, so a query touches `O(log n + k)` points instead of `n`. This is
//! the same structure `sklearn.cluster.HDBSCAN` reaches for (`algorithm='kd_tree'`
//! → `KDTree.query`), which is why its core-distance step is not its bottleneck.
//!
//! ## Why the core distances stay bit-identical
//! The tree changes WHICH pairs are evaluated, never HOW. Every distance that is
//! computed goes through the same `acc` accumulator over the same features in the
//! same order as the brute scan, and the answer is the `k`-th smallest of the
//! evaluated multiset — order-independent, so the traversal order cannot move it.
//!
//! Pruning is CONSERVATIVE. A subtree is skipped only when its box lower bound is
//! at or above the current `k`-th distance, so the points it hides are all `>=` a
//! value already held in the list and cannot lower the `k`-th. The bound is
//! widened by a few epsilons before the test (the [`super::host_core`] /
//! [`super::distance`] slack idiom) so a subtree that is within FP rounding of the
//! threshold is visited rather than pruned on a rounded inequality.
//!
//! `hdbscan_test::core_distances_host_matches_device` gates the result against the
//! device `knn_graph` prim on a coarse integer lattice with a duplicated row —
//! geometry chosen so many pairwise distances are exactly equal and a
//! tie-sensitive prune would show up immediately.
//!
//! ## Metric support
//! The four Variant-B FAST metrics only. Each is an aggregate over PER-AXIS
//! offsets that is monotone in every offset, which is what makes the box bound
//! valid: the smallest possible aggregate from a box is the aggregate of each
//! axis's shortest offset to that box. Cosine and precomputed take their core
//! distances from a dense matrix and never reach here.
//!
//! Tests live in `crates/mlrs-algos/tests/hdbscan_test.rs` (AGENTS.md §2).

/// Points per leaf. A leaf is scanned linearly, so this trades traversal
/// bookkeeping (small leaves) against wasted distance work (large leaves). 32
/// keeps a leaf's coordinates inside a few cache lines at the `p` that shows up
/// here (8-64) while amortizing the box tests over enough points.
const LEAF_SIZE: usize = 32;

/// One node of the implicit tree. Leaves have `left == right == NONE`.
struct Node {
    /// The node's points are `perm[lo..hi]`.
    lo: u32,
    hi: u32,
    /// Child node ids, or [`NONE`] on a leaf.
    left: u32,
    right: u32,
}

const NONE: u32 = u32::MAX;

/// A static, median-split KD-tree over the rows of a row-major `n×p` matrix.
pub(crate) struct KdTree {
    /// Row indices, permuted so every node owns a contiguous range.
    perm: Vec<u32>,
    nodes: Vec<Node>,
    /// Per-node axis-aligned bounding box, `2·p` values per node: `p` minima
    /// followed by `p` maxima. A TIGHT box (the actual extent of the node's
    /// points) prunes far better than the split-plane half-spaces would.
    bounds: Vec<f64>,
    p: usize,
}

impl KdTree {
    /// Build the tree over all `n` rows of `x` (`n × p`, row-major).
    ///
    /// Splitting recurses on the axis of widest extent at the node's own median
    /// (`select_nth_unstable_by` — a partition, not a sort), so the tree is
    /// balanced at depth `log2(n / LEAF_SIZE)` regardless of how the data is
    /// ordered on input.
    pub(crate) fn build(x: &[f64], n: usize, p: usize) -> Self {
        debug_assert_eq!(x.len(), n * p, "x must be a dense n×p matrix");
        let mut tree = Self {
            perm: (0..n as u32).collect(),
            nodes: Vec::with_capacity(2 * n.div_ceil(LEAF_SIZE) + 1),
            bounds: Vec::with_capacity(2 * p * (2 * n.div_ceil(LEAF_SIZE) + 1)),
            p,
        };
        if n > 0 {
            tree.split(x, 0, n);
        }
        tree
    }

    /// Recursively build the node covering `perm[lo..hi]`; returns its node id.
    fn split(&mut self, x: &[f64], lo: usize, hi: usize) -> u32 {
        let p = self.p;
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            lo: lo as u32,
            hi: hi as u32,
            left: NONE,
            right: NONE,
        });

        // Tight bounding box over this node's points.
        let base = self.bounds.len();
        self.bounds.extend(
            std::iter::repeat_n(f64::INFINITY, p).chain(std::iter::repeat_n(f64::NEG_INFINITY, p)),
        );
        for &r in &self.perm[lo..hi] {
            let row = &x[r as usize * p..(r as usize + 1) * p];
            for d in 0..p {
                let v = row[d];
                if v < self.bounds[base + d] {
                    self.bounds[base + d] = v;
                }
                if v > self.bounds[base + p + d] {
                    self.bounds[base + p + d] = v;
                }
            }
        }

        if hi - lo <= LEAF_SIZE {
            return id;
        }

        // Split on the widest axis of the box just computed.
        let mut axis = 0usize;
        let mut widest = f64::NEG_INFINITY;
        for d in 0..p {
            let w = self.bounds[base + p + d] - self.bounds[base + d];
            if w > widest {
                widest = w;
                axis = d;
            }
        }
        // A zero-extent box means every point in the range is identical on every
        // axis; there is nothing left to separate, so stop here rather than
        // recursing forever on an unsplittable range.
        if !(widest > 0.0) {
            return id;
        }

        let mid = (lo + hi) / 2;
        self.perm[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
            x[a as usize * p + axis].total_cmp(&x[b as usize * p + axis])
        });

        let left = self.split(x, lo, mid);
        let right = self.split(x, mid, hi);
        self.nodes[id as usize].left = left;
        self.nodes[id as usize].right = right;
        id
    }

    /// The per-axis offsets from `q` to node `id`'s box, aggregated by `axis_agg`.
    ///
    /// `axis_agg(running, offset)` folds one axis's shortest offset into the
    /// running aggregate — `s + o*o` for Euclidean, `s + o` for Manhattan,
    /// `max(s, o)` for Chebyshev, `s + o^pp` for Minkowski. An offset is 0 on any
    /// axis where `q` lies inside the box's extent, so a query inside the box
    /// aggregates to 0 and the node is never pruned.
    ///
    /// The result is a LOWER BOUND on the aggregate distance from `q` to any point
    /// in the node, because each metric here is monotone in every axis offset.
    #[inline]
    fn box_bound<Ax>(&self, id: u32, q: &[f64], axis_agg: &Ax) -> f64
    where
        Ax: Fn(f64, f64) -> f64,
    {
        let p = self.p;
        let base = id as usize * 2 * p;
        let mut s = 0.0f64;
        for d in 0..p {
            let lo = self.bounds[base + d];
            let hi = self.bounds[base + p + d];
            let v = q[d];
            let o = if v < lo {
                lo - v
            } else if v > hi {
                v - hi
            } else {
                0.0
            };
            s = axis_agg(s, o);
        }
        s
    }
}

/// Everything the traversal needs to know about one metric, in the same
/// aggregate/finalize/pre-image terms [`super::host_core`] uses.
///
/// `acc` accumulates the pre-final aggregate for a point pair (and may bail out
/// early), `fin` turns an aggregate into a distance, `pre` maps a distance
/// threshold back into the aggregate domain, and `axis_agg` folds one box offset
/// into a box bound — the SAME aggregation `acc` performs, one axis at a time.
pub(crate) struct MetricOps<A, Fin, Pre, Ax> {
    pub acc: A,
    pub fin: Fin,
    pub pre: Pre,
    pub axis_agg: Ax,
}

/// The bounded "`k` smallest so far" list the caller maintains per query. Mirrors
/// `host_core::KSmallest`; kept behind this trait-free shape so the two modules
/// share the traversal without exposing either's internals.
pub(crate) trait Bounded {
    /// The current `k`-th smallest distance, or `+inf` while still filling.
    fn worst(&self) -> f64;
    /// Offer a candidate distance, tagged with the row index it came from.
    ///
    /// The index lets a caller that needs the NEIGHBOUR IDENTITIES (UMAP's
    /// `host_knn`, which returns `(index, distance)` lists) reuse this traversal,
    /// and lets a caller drop the query itself. Implementations that only need
    /// the distances ignore it.
    fn offer(&mut self, d: f64, idx: u32);
}

/// Answer one query: fold every point whose distance can still enter the `k`
/// smallest into `best`.
///
/// `q` is the query row (`p` values). The traversal visits the NEARER child first
/// so `best` tightens as early as possible, which is what makes the far child's
/// bound test succeed. Both children re-test their bound against the CURRENT
/// `best`, so the pruning strengthens as the descent proceeds.
/// Returns how many POINTS the traversal actually evaluated — the quantity the
/// caller's route calibration keys on (see [`brute_is_cheaper`]).
pub(crate) fn query<A, Fin, Pre, Ax, B>(
    tree: &KdTree,
    x: &[f64],
    q: &[f64],
    ops: &MetricOps<A, Fin, Pre, Ax>,
    best: &mut B,
) -> usize
where
    A: Fn(&[f64], &[f64], f64) -> f64,
    Fin: Fn(f64) -> f64,
    Pre: Fn(f64) -> f64,
    Ax: Fn(f64, f64) -> f64,
    B: Bounded,
{
    if tree.nodes.is_empty() {
        return 0;
    }
    let mut visited = 0usize;
    descend(tree, x, q, ops, best, 0, &mut visited);
    visited
}

/// Visit node `id` unless its box is already too far.
fn descend<A, Fin, Pre, Ax, B>(
    tree: &KdTree,
    x: &[f64],
    q: &[f64],
    ops: &MetricOps<A, Fin, Pre, Ax>,
    best: &mut B,
    id: u32,
    visited: &mut usize,
) where
    A: Fn(&[f64], &[f64], f64) -> f64,
    Fin: Fn(f64) -> f64,
    Pre: Fn(f64) -> f64,
    Ax: Fn(f64, f64) -> f64,
    B: Bounded,
{
    let p = tree.p;
    let node = &tree.nodes[id as usize];

    if node.left == NONE {
        // Leaf: the same bounded-insertion scan the brute path runs, over this
        // leaf's points only. Identical per-pair arithmetic, identical bound.
        *visited += (node.hi - node.lo) as usize;
        for &r in &tree.perm[node.lo as usize..node.hi as usize] {
            let w = best.worst();
            let bound = if w.is_finite() {
                (ops.pre)(w)
            } else {
                f64::INFINITY
            };
            let a = (ops.acc)(q, &x[r as usize * p..(r as usize + 1) * p], bound);
            if a.is_finite() {
                let d = (ops.fin)(a);
                if d < w {
                    best.offer(d, r);
                }
            }
        }
        return;
    }

    // Nearer child first: its points tighten `best` before the far child's box
    // bound is tested, which is where the pruning comes from.
    let (l, r) = (node.left, node.right);
    let bl = tree.box_bound(l, q, &ops.axis_agg);
    let br = tree.box_bound(r, q, &ops.axis_agg);
    let (first, second, second_bound) = if bl <= br { (l, r, br) } else { (r, l, bl) };

    descend(tree, x, q, ops, best, first, visited);

    // CONSERVATIVE prune: widen the threshold by a few epsilons so a box within
    // FP rounding of the current k-th is visited, not pruned (the `host_core`
    // slack idiom — `pre` is a power for Euclidean/Minkowski, so the mapped
    // threshold is not exact to the last ULP).
    let w = best.worst();
    if w.is_finite() {
        let thresh = (ops.pre)(w) * (1.0 + 4.0 * f64::EPSILON);
        if second_bound >= thresh {
            return;
        }
    }
    descend(tree, x, q, ops, best, second, visited);
}

/// Is the tree worth BUILDING at this geometry at all?
///
/// A cheap structural gate only — whether the built tree is worth USING is
/// decided from its measured pruning ([`brute_is_cheaper`]), because that depends
/// on the DATA and not just its shape. Below [`KD_MIN_ROWS`] the brute scan is
/// already microseconds and the `O(n log n)` build cannot pay for itself.
///
/// `MLRS_HDBSCAN_CORE_KD=0` forces the brute scan, `=1` forces the tree (skipping
/// the calibration too), for on-target A/B.
pub(crate) fn kd_applicable(n: usize, _p: usize) -> bool {
    match mlrs_backend::abflag::var("MLRS_HDBSCAN_CORE_KD").as_deref() {
        Some("0") => return false,
        Some("1") => return true,
        _ => {}
    }
    n >= KD_MIN_ROWS
}

/// Was the route FORCED on, so the calibration should not second-guess it?
pub(crate) fn kd_forced() -> bool {
    mlrs_backend::abflag::var("MLRS_HDBSCAN_CORE_KD").as_deref() == Some("1")
}

/// Below this many rows the brute scan is already microseconds and the build is
/// not worth it.
const KD_MIN_ROWS: usize = 512;

/// Should the caller ABANDON the tree, given that its first `rows` queries
/// evaluated `visited` points in total out of `n` candidates each?
///
/// ## Why this is measured per-DATA rather than gated on `d`
/// A box bound loses its power as `d` grows — every axis contributes an offset, so
/// in high dimension the boxes overlap the query ball almost everywhere (the curse
/// of dimensionality) and the traversal evaluates nearly every point anyway, on
/// top of paying for the descent. But the crossover is a property of the DATA, not
/// of `d`. MEASURED at `n = 10_000`, best of 3, core-distance stage only
/// (`hdbscan_perf_test::hdbscan_core_distance_sweep`, tree ÷ brute speedup):
///
/// ```text
///           8 well-separated blobs        uniform noise
///    d      speedup    visited/query    speedup   visited/query
///    2       20.0×         0.55%         18.3×        0.55%
///    4       14.1×         1.64%         12.4×        1.98%
///    8        4.4×         7.83%          2.4×       13.04%
///   16        2.1×        12.77%          0.53×      93.47%   <-- diverges here
///   24        1.8×        12.69%          0.51×     100.00%
///   32        1.5×        12.64%          0.56×     100.00%
///   64        1.3×        12.71%          0.61×     100.00%
/// ```
///
/// A static `d` gate has to choose one of those columns and be wrong on the other:
/// cut at `d <= 8` and clustered 16-64-dimensional data forfeits a real win; cut
/// at `d <= 64` and unstructured data at `d >= 16` runs ~2× SLOWER than the scan it
/// replaced. Since HDBSCAN's own premise is that the input has density structure,
/// neither column is the "typical" one.
///
/// The visited count settles it directly, and note how cleanly it separates the two
/// columns where the speedup diverges: at `d >= 16` the clustered data holds a flat
/// ~12.7% while the uniform data saturates at ~100%. That is the same quantity in
/// both cases, it needs no per-machine tuning, and it is DETERMINISTIC — unlike a
/// sampled wall-clock A/B, which would have to distinguish these two regimes
/// through timing noise.
pub(crate) fn brute_is_cheaper(visited: usize, rows: usize, n: usize) -> bool {
    if rows == 0 || n == 0 {
        return false;
    }
    let per_query = visited as f64 / rows as f64;
    per_query > BRUTE_RATIO * n as f64
}

/// Fraction of `n` a query may evaluate before the tree stops paying off.
///
/// Derived from the table above rather than picked: at `visited = 100%` the tree
/// runs 0.51-0.61× the brute speed, so its per-evaluated-point cost (box tests, the
/// `perm` indirection, and a leaf scan that is not one long sequential stream) is
/// roughly 1.6-2.0× the brute loop's — putting true break-even near 50-60% of the
/// set. Half sits at the low (safe) end of that range.
///
/// What matters more than the exact value is that both observed regimes are FAR
/// from it — 12.7% keeps the tree, 93.5% drops it — so the decision does not turn
/// on where in that gap the threshold lands, and a machine with a different
/// tree-to-scan cost ratio would still classify both the same way.
const BRUTE_RATIO: f64 = 0.5;

/// How many rows a worker runs through the tree before it judges the route.
///
/// The calibration is nearly free: these rows are real output rows, computed by
/// the tree and KEPT (both routes produce identical core distances), so a worker
/// that switches to the brute scan has wasted only the DIFFERENCE in cost on this
/// handful of rows. Each worker decides independently on its own first rows, which
/// also lets a locally dense region keep the tree where a sparse one drops it.
///
/// Four rows, not more: the visited fraction the decision reads separates the two
/// regimes by 12.7% against 93-100% (see [`brute_is_cheaper`]), so a handful of
/// rows already resolves it, and every extra row is paid by all
/// [`cpu_launch_units`](mlrs_backend::capability::cpu_launch_units) workers on the
/// data where the tree loses.
///
/// The count is NOT load-bearing for wall clock: measured on the uniform ladder
/// (best of 5, `n = 10_000`) the whole adaptive route costs 1.02-1.03× the brute
/// scan at `d = 24..64` and 1.03× at `d = 16` with 4 rows, against 1.03-1.08× with
/// 16 — inside run-to-run noise. 4 is kept because there is no reason to pay for
/// rows that change nothing, not because 16 measured worse.
///
/// What the calibration IS worth is the difference from forcing the tree on: on
/// that same uniform data the forced tree runs 0.51-0.61× the brute speed (up to
/// ~2× SLOWER), and the calibration turns that into a ≤3% overhead.
pub(crate) const CALIB_ROWS: usize = 4;

/// Build the tree over all `n` rows.
///
/// Exposed for [`super::host_core`], which owns the metric monomorphization and
/// the row-parallel split.
pub(crate) fn build_tree(x: &[f64], n: usize, p: usize) -> KdTree {
    KdTree::build(x, n, p)
}
