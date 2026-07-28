//! Prim's MST over the mutual-reachability graph — BOTH oracle variants
//! (HDBS-02 / D-04, plan 15-03).
//!
//! sklearn dispatches to TWO DIFFERENT MST algorithms under `algorithm='auto'`
//! (the default mlrs must match), and they resolve weight ties differently:
//!
//!   - **Variant A** [`mst_from_mutual_reachability`] — the dense Prim used for
//!     `cosine` + `precomputed` (NOT `FAST_METRICS`). Prim from node 0, the next
//!     node is the FIRST `argmin` of the running min-reachability (first-min on
//!     ties), via a shrinking `current_labels` index remap. Alpha placement:
//!     the WHOLE distance matrix is divided by alpha BEFORE core distances (done
//!     by the caller), so the mutual-reachability fed here already carries
//!     `d_ij/alpha` AND `core` recomputed from the scaled matrix.
//!
//!   - **Variant B** [`mst_from_data_matrix`] — the source-tracking Prim used for
//!     `euclidean`/`l1`/`l2`/`chebyshev`/`minkowski` (`FAST_METRICS`). It tracks a
//!     per-node `current_sources[]` and uses STRICT `<` comparisons so on a tie
//!     the FIRST-scanned `j` wins (lowest index, since `j` scans `0..n`). Alpha
//!     placement: `pair_distance /= alpha` with RAW (unscaled) core distances —
//!     a DIFFERENT placement from Variant A (RESEARCH Pattern 2 / Pitfall 2).
//!
//! After either variant, [`argsort_by_weight`] orders the `n-1` edges by ascending
//! weight to feed `make_single_linkage`. The gate fixtures use DISTINCT MST edge
//! weights so this sort is tie-free and oracle-equal under any deterministic rule
//! (RESEARCH Pitfall 1, option 2 — the tie-heavy fixture is the characterization
//! gate, not a band). All scalar math is done in `f64` via the shared
//! `mlrs_core::{host_to_f64, f64_to_host}` bridging idiom (`spectral_embedding.rs`
//! precedent).
//!
//! Tests live in `crates/mlrs-algos/tests/hdbscan_test.rs` (AGENTS.md §2).

/// One Prim's-MST edge `(u, v, weight)` over the mutual-reachability graph. `u`
/// and `v` are point indices in `0..n`; `weight` is the mutual-reachability of
/// the edge (in `f64`, the host scalar domain).
pub type MstEdge = (usize, usize, f64);

/// Variant A — dense `mst_from_mutual_reachability` (cosine + precomputed).
///
/// `mr` is the DENSE row-major `n×n` mutual-reachability matrix
/// (`mr[i*n + j] = max(core_i, core_j, d_ij/alpha)`, symmetric), already built by
/// the caller from the alpha-scaled distance matrix. Prim's grows the tree from
/// node 0; the next node is the FIRST minimum (`argmin`) of the running
/// min-reachability over the not-yet-added nodes, replicating sklearn's
/// `np.argmin` first-min tie-break through a shrinking `current_labels` index
/// remap.
///
/// Returns `n - 1` edges. `n` must be `>= 1`; an `n == 1` graph yields no edges.
pub fn mst_from_mutual_reachability(mr: &[f64], n: usize) -> Vec<MstEdge> {
    debug_assert_eq!(mr.len(), n * n, "mr must be a dense n×n matrix");
    if n <= 1 {
        return Vec::new();
    }

    let mut current_node: usize = 0;
    // `min_reachability[k]` tracks the best known reachability to the (remaining)
    // node `current_labels[k]`. `current_labels` starts as `0..n` and shrinks by
    // one (the chosen node) each step — mirroring sklearn's boolean `label_filter`
    // applied to `current_labels` BEFORE indexing `min_reachability`.
    let mut current_labels: Vec<usize> = (0..n).collect();
    let mut min_reachability: Vec<f64> = vec![f64::INFINITY; n];
    let mut mst: Vec<MstEdge> = Vec::with_capacity(n - 1);

    for _ in 0..(n - 1) {
        // label_filter = current_labels != current_node; drop current_node from
        // BOTH current_labels and the aligned min_reachability (the two stay
        // index-aligned, exactly as sklearn's `min_reachability[label_filter]`).
        let mut next_labels: Vec<usize> = Vec::with_capacity(current_labels.len());
        let mut left: Vec<f64> = Vec::with_capacity(current_labels.len());
        for (k, &lbl) in current_labels.iter().enumerate() {
            if lbl != current_node {
                next_labels.push(lbl);
                left.push(min_reachability[k]);
            }
        }
        current_labels = next_labels;

        // right = mr[current_node][current_labels]; min_reachability =
        // minimum(left, right). Recompute min_reachability ALIGNED to the new
        // (shrunk) current_labels.
        min_reachability = Vec::with_capacity(current_labels.len());
        for (k, &lbl) in current_labels.iter().enumerate() {
            let right = mr[current_node * n + lbl];
            let m = if left[k] < right { left[k] } else { right };
            min_reachability.push(m);
        }

        // new_node_index = argmin(min_reachability) — FIRST minimum on ties
        // (strict `<` keeps the earliest index, matching np.argmin).
        let mut new_node_index = 0usize;
        let mut best = min_reachability[0];
        for (k, &v) in min_reachability.iter().enumerate().skip(1) {
            if v < best {
                best = v;
                new_node_index = k;
            }
        }
        let new_node = current_labels[new_node_index];
        mst.push((current_node, new_node, min_reachability[new_node_index]));
        current_node = new_node;
    }

    mst
}

/// Variant B — source-tracking `mst_from_data_matrix` (euclidean / l1 / l2 /
/// chebyshev / minkowski — the `FAST_METRICS`).
///
/// Instead of a dense `n×n` mutual-reachability matrix, this recomputes the
/// pairwise distance each step via the supplied `pairwise` closure
/// (`pairwise(i, j)` = the RAW, unscaled distance `d(i,j)`) and divides it by
/// `alpha` — RAW `core` distances, `pair_distance /= alpha` (the Variant-B alpha
/// placement, DISTINCT from Variant A). It tracks a per-node `current_sources[]`
/// so the tree records the actual source of each chosen edge, and uses STRICT
/// `<` comparisons throughout so ties resolve to the LOWEST `j` (since `j` scans
/// `0..n` ascending).
///
/// `core[i]` is the (unscaled) core distance of point `i`. Returns `n - 1` edges.
pub fn mst_from_data_matrix<DistFn>(
    core: &[f64],
    n: usize,
    alpha: f64,
    mut pairwise: DistFn,
) -> Vec<MstEdge>
where
    DistFn: FnMut(usize, usize) -> f64,
{
    debug_assert_eq!(core.len(), n, "core must have one distance per point");
    if n <= 1 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut min_reachability = vec![f64::INFINITY; n];
    let mut current_sources = vec![0usize; n];
    let mut mst: Vec<MstEdge> = Vec::with_capacity(n - 1);

    let mut current_node: usize = 0;
    for _ in 0..(n - 1) {
        in_tree[current_node] = true;

        let mut source_node = current_node;
        let mut new_node = current_node;
        let mut new_reachability = f64::INFINITY;

        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            let pair_distance = pairwise(current_node, j) / alpha;
            // mr = max(core[current_node], core[j], pair_distance).
            let mut mr = core[current_node];
            if core[j] > mr {
                mr = core[j];
            }
            if pair_distance > mr {
                mr = pair_distance;
            }

            if mr < min_reachability[j] {
                min_reachability[j] = mr;
                current_sources[j] = current_node;
                if mr < new_reachability {
                    new_reachability = mr;
                    source_node = current_node;
                    new_node = j;
                }
            } else if min_reachability[j] < new_reachability {
                new_reachability = min_reachability[j];
                source_node = current_sources[j];
                new_node = j;
            }
        }

        mst.push((source_node, new_node, new_reachability));
        current_node = new_node;
    }

    mst
}

/// Variant B, SPECIALIZED — same Prim, same tie-order, same edges as
/// [`mst_from_data_matrix`], but with the metric monomorphized into the inner
/// loop and two exact prunes applied (HDBS-PERF-CPU).
///
/// This is the hot loop of a feature-metric fit: `n-1` steps each scanning every
/// not-yet-added node and computing a `p`-feature distance, i.e. `O(n²·p)` — the
/// same asymptotics as sklearn's Cython `mst_from_data_matrix`, which is what a
/// fit is benchmarked against. Three things make this version faster WITHOUT
/// changing a single edge it emits:
///
/// 1. **The metric is a type, not a branch.** `mst_from_data_matrix` receives an
///    `FnMut(i, j)` closure that re-`match`es the metric on every one of the
///    `n²/2` calls and re-slices both rows. Here the metric picks the
///    instantiation ONCE and the feature loop is straight-line.
///
/// 2. **Core-distance prune (exact).** `mr = max(core_i, core_j, d) >=
///    max(core_i, core_j)`, so when `max(core_i, core_j) >= min_reachability[j]`
///    the `mr < min_reachability[j]` test is ALREADY false — the branch outcome
///    is decided without knowing `d`. Those nodes skip the distance entirely and
///    fall straight to the `elif`. This is not an approximation: it is the same
///    comparison, evaluated from a bound that is tight enough to settle it.
///
/// 3. **Partial-distance early exit (exact).** When the distance IS needed, the
///    feature loop aborts once the running aggregate passes the threshold that
///    would make `d` too large to matter — for Euclidean that also skips the
///    `sqrt`. The Euclidean/Minkowski thresholds carry the same few-epsilon
///    slack as [`super::host_core`], so a candidate within rounding distance of
///    the threshold falls through to the exact compare rather than being pruned
///    on a rounded inequality.
///
/// `core[i]` is the RAW (unscaled) core distance; `alpha` divides the pair
/// distance (the Variant-B placement). `Metric::Precomputed` never reaches here
/// (the precomputed path runs Variant A); it falls back to the generic closure
/// form, which panics on it exactly as before.
pub fn mst_from_data_matrix_metric(
    x: &[f64],
    n: usize,
    p: usize,
    core: &[f64],
    alpha: f64,
    metric: super::Metric,
) -> Vec<MstEdge> {
    use super::distance::{
        chebyshev_screened, manhattan_screened, minkowski_screened, sq_euclidean_screened,
    };
    use super::Metric;
    match metric {
        Metric::Euclidean => prim_specialized(
            x,
            n,
            p,
            core,
            alpha,
            sq_euclidean_screened,
            f64::sqrt,
            |t| t * t * (1.0 + 4.0 * f64::EPSILON),
        ),
        Metric::Manhattan => {
            prim_specialized(x, n, p, core, alpha, manhattan_screened, |s| s, |t| t)
        }
        Metric::Chebyshev => {
            prim_specialized(x, n, p, core, alpha, chebyshev_screened, |s| s, |t| t)
        }
        Metric::Minkowski { p: pp } => prim_specialized(
            x,
            n,
            p,
            core,
            alpha,
            move |a: &[f64], b: &[f64], bound: f64| minkowski_screened(a, b, bound, pp),
            move |s: f64| s.powf(1.0 / pp),
            move |t: f64| t.powf(pp) * (1.0 + 4.0 * f64::EPSILON),
        ),
        // Cosine routes to Variant A (dense) and Precomputed never reaches the
        // Variant-B Prim; both keep the generic closure form so the metric
        // surface here stays exactly the FAST-metric set.
        Metric::Cosine | Metric::Precomputed => {
            let m = metric;
            mst_from_data_matrix(core, n, alpha, |i, j| {
                super::distance::host_pairwise(x, p, m, i, j)
            })
        }
    }
}

/// Rows below which the Variant-B Prim stays single-threaded (HDBS-PERF-CPU).
///
/// Prim is sequential ACROSS steps — only the scan WITHIN a step parallelizes —
/// so every one of the `n-1` steps pays a barrier. That is worth it only once a
/// step's scan is itself many microseconds. MEASURED on this 16-unit host at
/// `d = 8`, best of three (`MLRS_HDBSCAN_MST_PAR` sweep, seconds):
///
/// ```text
///   n      serial   parallel
///   300    0.0004   0.0010     <-- barrier-bound, and erratic
///   500    0.0009   0.0015
///   750    0.0021   0.0028
///  1000    0.0039   0.0024     <-- crossover
///  2000    0.0168   0.0069
/// ```
///
/// Below the crossover the spin barrier also gets NOISY (0.0015-0.0099 s at
/// `n = 500`) because a step is shorter than a scheduler quantum, which is a
/// second reason to keep small fits on the deterministic serial body.
const PAR_MIN_ROWS: usize = 1_024;

/// Elements between two workers' partial slots — 8 × 8 bytes = one 64-byte cache
/// line, so a worker's per-step store never invalidates a neighbour's line.
const SLOT_STRIDE: usize = 8;

/// Worker count for the parallel Prim: three quarters of the machine's launch
/// units, floor 1.
///
/// This is the one place in the codebase that deliberately does NOT take all the
/// cores. The workers SPIN at the per-step barrier, so a worker that loses its
/// core to another runnable thread stalls all the others for a scheduler
/// quantum instead of a few microseconds. Leaving a quarter of the machine free
/// for the caller's thread, the runtime's, and the harness's is worth more than
/// the extra scan width. MEASURED at `n = 10_000, d = 16` on this 16-unit host
/// (`MLRS_HDBSCAN_MST_UNITS` sweep):
///
/// ```text
///   4 units  0.196 s     12 units  0.098 s   <-- 3/4 of 16
///   8 units  0.122 s     14 units  0.188 s
///  10 units  0.106 s     16 units  0.194 s   <-- all cores: 2x WORSE than 12
/// ```
///
/// `MLRS_HDBSCAN_MST_UNITS` overrides it for on-target sweeps; the edge list is
/// identical at every count, so this only ever trades wall clock.
fn mst_units() -> usize {
    mlrs_backend::abflag::var("MLRS_HDBSCAN_MST_UNITS")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or_else(|| {
            let all = mlrs_backend::capability::cpu_launch_units().max(1) as usize;
            (all * 3 / 4).max(1)
        })
}

/// A sense-reversing SPIN barrier for the per-step Prim rendezvous.
///
/// `std::sync::Barrier` is a mutex + condvar: it parks, which costs tens of
/// microseconds to wake 16 threads. A Prim step's scan is a few microseconds of
/// work, and there are `n-1` steps, so parking once per step would cost more
/// than the scan it synchronizes. Spinning is the right trade precisely here —
/// the wait is always short and bounded by one scan — and it degrades to
/// `yield_now` if a thread is descheduled so an oversubscribed box still makes
/// progress.
struct SpinBarrier {
    /// Threads that have arrived in the CURRENT generation.
    count: std::sync::atomic::AtomicUsize,
    /// Bumped by the last arriver; the released threads watch it change.
    generation: std::sync::atomic::AtomicUsize,
    /// Participant count.
    parties: usize,
}

impl SpinBarrier {
    fn new(parties: usize) -> Self {
        Self {
            count: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicUsize::new(0),
            parties,
        }
    }

    fn wait(&self) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
        let gen = self.generation.load(Acquire);
        if self.count.fetch_add(1, AcqRel) == self.parties - 1 {
            self.count.store(0, Relaxed);
            self.generation.fetch_add(1, Release);
            return;
        }
        let mut spins = 0u32;
        while self.generation.load(Acquire) == gen {
            spins += 1;
            if spins < 256 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// Variant B, PARALLEL — the same Prim, the same edges, the scan within each
/// step split over the machine's cores (HDBS-PERF-CPU).
///
/// ## Why this is edge-for-edge identical to the serial Prim
/// Each worker owns a CONTIGUOUS, ASCENDING range of node indices, and owns the
/// `min_reachability` / `current_sources` / `in_tree` entries for exactly that
/// range — no entry is written by two workers, so there is no data race and no
/// ordering ambiguity in the state. Within its range each worker keeps the FIRST
/// strict minimum (the serial rule); the cross-worker reduction then walks the
/// per-worker partials in ASCENDING RANGE ORDER, again keeping the first strict
/// minimum. First-strict-minimum over ascending blocks of an ascending scan IS
/// the first strict minimum of the whole scan, so the tie-break the D-04 gate
/// pins is preserved exactly.
///
/// Every worker runs the identical reduction over the identical partials, so all
/// of them derive the same next `current_node` without a second broadcast; only
/// worker 0 records the edge. The partial slots are DOUBLE-BUFFERED by step
/// parity, which is what lets one barrier per step suffice: a worker racing
/// ahead to step `i+1` writes the other buffer, and it cannot reach step `i+2`
/// (which reuses this one) until everyone has passed the step-`i+1` barrier and
/// therefore finished reading it.
#[allow(clippy::too_many_arguments)]
fn prim_specialized_parallel<A, Fin, Pre>(
    x: &[f64],
    n: usize,
    p: usize,
    core: &[f64],
    alpha: f64,
    acc: A,
    fin: Fin,
    pre: Pre,
    units: usize,
) -> Vec<MstEdge>
where
    A: Fn(&[f64], &[f64], f64) -> f64 + Sync,
    Fin: Fn(f64) -> f64 + Sync,
    Pre: Fn(f64) -> f64 + Sync,
{
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    let mut in_tree = vec![false; n];
    let mut min_reachability = vec![f64::INFINITY; n];
    let mut current_sources = vec![0usize; n];

    // The three owned-state vectors are split into the SAME contiguous ranges, so
    // a worker's `(lo, tree, reach, sources)` all describe one index window.
    let chunk = n.div_ceil(units);
    let parts: Vec<(&mut [bool], &mut [f64], &mut [usize])> = in_tree
        .chunks_mut(chunk)
        .zip(min_reachability.chunks_mut(chunk))
        .zip(current_sources.chunks_mut(chunk))
        .map(|((t, r), s)| (t, r, s))
        .collect();
    // `chunks_mut` can yield FEWER than `units` chunks (a short tail rounds the
    // count down). The barrier party count and the slot array must follow the
    // ACTUAL worker count, or the missing workers never arrive and every step
    // deadlocks — `units` is only the target.
    let units = parts.len();

    // Per-(buffer, worker) partials: the best (reachability, source, node) that
    // worker found in its own range this step. STRIDED to one cache line each —
    // packed adjacently, all `units` workers store into the same two lines every
    // step, and the resulting ping-pong showed up as most of the barrier cost.
    let slots = 2 * units * SLOT_STRIDE;
    let slot_reach: Vec<AtomicU64> = (0..slots)
        .map(|_| AtomicU64::new(f64::INFINITY.to_bits()))
        .collect();
    let slot_source: Vec<AtomicUsize> = (0..slots).map(|_| AtomicUsize::new(0)).collect();
    let slot_node: Vec<AtomicUsize> = (0..slots).map(|_| AtomicUsize::new(0)).collect();

    let barrier = SpinBarrier::new(units);
    let mut mst: Vec<MstEdge> = Vec::with_capacity(n - 1);

    std::thread::scope(|scope| {
        let (acc, fin, pre) = (&acc, &fin, &pre);
        let (slot_reach, slot_source, slot_node) = (&slot_reach, &slot_source, &slot_node);
        let barrier = &barrier;
        let mut handles = Vec::with_capacity(units);
        for (w, (tree, reach, sources)) in parts.into_iter().enumerate() {
            let lo = w * chunk;
            let hi = lo + tree.len();
            handles.push(scope.spawn(move || {
                let mut current_node = 0usize;
                let mut edges: Vec<MstEdge> = if w == 0 {
                    Vec::with_capacity(n - 1)
                } else {
                    Vec::new()
                };
                for step in 0..(n - 1) {
                    let buf = (step & 1) * units * SLOT_STRIDE + w * SLOT_STRIDE;
                    if (lo..hi).contains(&current_node) {
                        tree[current_node - lo] = true;
                    }

                    let core_i = core[current_node];
                    let xi = &x[current_node * p..(current_node + 1) * p];
                    let mut best_reach = f64::INFINITY;
                    let mut best_source = current_node;
                    let mut best_node = current_node;

                    for j in lo..hi {
                        let local = j - lo;
                        if tree[local] {
                            continue;
                        }
                        let next_min_reach = reach[local];
                        let base = if core_i > core[j] { core_i } else { core[j] };
                        if base >= next_min_reach {
                            if next_min_reach < best_reach {
                                best_reach = next_min_reach;
                                best_source = sources[local];
                                best_node = j;
                            }
                            continue;
                        }
                        let thresh = next_min_reach * alpha;
                        let bound = if thresh.is_finite() {
                            pre(thresh)
                        } else {
                            f64::INFINITY
                        };
                        let a = acc(xi, &x[j * p..(j + 1) * p], bound);
                        let pair_distance = if a.is_finite() {
                            fin(a) / alpha
                        } else {
                            f64::INFINITY
                        };
                        let mr = if pair_distance > base {
                            pair_distance
                        } else {
                            base
                        };
                        if mr < next_min_reach {
                            reach[local] = mr;
                            sources[local] = current_node;
                            if mr < best_reach {
                                best_reach = mr;
                                best_source = current_node;
                                best_node = j;
                            }
                        } else if next_min_reach < best_reach {
                            best_reach = next_min_reach;
                            best_source = sources[local];
                            best_node = j;
                        }
                    }

                    slot_reach[buf].store(best_reach.to_bits(), Ordering::Relaxed);
                    slot_source[buf].store(best_source, Ordering::Relaxed);
                    slot_node[buf].store(best_node, Ordering::Relaxed);
                    barrier.wait();

                    // Identical reduction on every worker: ascending range order,
                    // strict `<`, seeded with the serial loop's own "found
                    // nothing" state so an all-in-tree scan yields the same edge.
                    let mut reach_best = f64::INFINITY;
                    let mut source_best = current_node;
                    let mut node_best = current_node;
                    let base = (step & 1) * units * SLOT_STRIDE;
                    for t in 0..units {
                        let s = base + t * SLOT_STRIDE;
                        let r = f64::from_bits(slot_reach[s].load(Ordering::Relaxed));
                        if r < reach_best {
                            reach_best = r;
                            source_best = slot_source[s].load(Ordering::Relaxed);
                            node_best = slot_node[s].load(Ordering::Relaxed);
                        }
                    }
                    if w == 0 {
                        edges.push((source_best, node_best, reach_best));
                    }
                    current_node = node_best;
                }
                edges
            }));
        }
        // Worker 0 is the one that recorded the edge list; the rest return empty.
        let mut handles = handles.into_iter();
        let first = handles.next().expect("at least one worker");
        mst = first.join().expect("prim worker panicked");
        for h in handles {
            h.join().expect("prim worker panicked");
        }
    });

    mst
}

/// The specialized Variant-B Prim body shared by every FAST metric.
///
/// `acc(a, b, bound)` accumulates the metric's pre-final aggregate and returns
/// `+inf` once it passes `bound`; `fin` turns the aggregate into the distance;
/// `pre` maps a distance threshold into the aggregate domain. Mirrors the
/// `super::host_core::scan` contract.
///
/// The comparison chain, the `min_reachability`/`current_sources` updates and
/// the STRICT `<` tie-break are transcribed unchanged from
/// [`mst_from_data_matrix`] — only the paths that reach them are cheaper.
///
/// Dispatches to [`prim_specialized_parallel`] once the problem is big enough to
/// amortize a barrier per step ([`PAR_MIN_ROWS`]); `MLRS_HDBSCAN_MST_PAR=0`
/// forces the serial body for on-target A/B, and the two are gated
/// edge-for-edge identical by `hdbscan_test::mst_specialized_matches_generic`.
#[inline]
fn prim_specialized<A, Fin, Pre>(
    x: &[f64],
    n: usize,
    p: usize,
    core: &[f64],
    alpha: f64,
    acc: A,
    fin: Fin,
    pre: Pre,
) -> Vec<MstEdge>
where
    A: Fn(&[f64], &[f64], f64) -> f64 + Sync,
    Fin: Fn(f64) -> f64 + Sync,
    Pre: Fn(f64) -> f64 + Sync,
{
    debug_assert_eq!(core.len(), n, "core must have one distance per point");
    debug_assert_eq!(x.len(), n * p, "x must be a dense n×p matrix");
    if n <= 1 {
        return Vec::new();
    }

    let units = mst_units();
    let parallel = n >= PAR_MIN_ROWS
        && units > 1
        && mlrs_backend::abflag::var("MLRS_HDBSCAN_MST_PAR")
            .map(|v| v != "0")
            .unwrap_or(true);
    if parallel {
        return prim_specialized_parallel(x, n, p, core, alpha, acc, fin, pre, units);
    }

    let mut in_tree = vec![false; n];
    let mut min_reachability = vec![f64::INFINITY; n];
    let mut current_sources = vec![0usize; n];
    let mut mst: Vec<MstEdge> = Vec::with_capacity(n - 1);

    let mut current_node: usize = 0;
    for _ in 0..(n - 1) {
        in_tree[current_node] = true;

        let mut source_node = current_node;
        let mut new_node = current_node;
        let mut new_reachability = f64::INFINITY;
        let core_i = core[current_node];
        let xi = &x[current_node * p..(current_node + 1) * p];

        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            let next_min_reach = min_reachability[j];

            // Prune 2: `mr >= max(core_i, core_j)`, so a base at or above
            // `next_min_reach` settles `mr < next_min_reach` as false with no
            // distance computed. `j` then only matters through the `elif`.
            let base = if core_i > core[j] { core_i } else { core[j] };
            if base >= next_min_reach {
                if next_min_reach < new_reachability {
                    new_reachability = next_min_reach;
                    source_node = current_sources[j];
                    new_node = j;
                }
                continue;
            }

            // `base < next_min_reach`, so `mr < next_min_reach` iff
            // `pair_distance < next_min_reach`. Bound the aggregate accordingly
            // (the `/alpha` is folded into the threshold so the accumulator stays
            // division-free).
            let thresh = next_min_reach * alpha;
            let bound = if thresh.is_finite() {
                pre(thresh)
            } else {
                f64::INFINITY
            };
            let xj = &x[j * p..(j + 1) * p];
            let a = acc(xi, xj, bound);
            let pair_distance = if a.is_finite() {
                fin(a) / alpha
            } else {
                f64::INFINITY
            };

            let mr = if pair_distance > base {
                pair_distance
            } else {
                base
            };

            if mr < next_min_reach {
                min_reachability[j] = mr;
                current_sources[j] = current_node;
                if mr < new_reachability {
                    new_reachability = mr;
                    source_node = current_node;
                    new_node = j;
                }
            } else if next_min_reach < new_reachability {
                new_reachability = next_min_reach;
                source_node = current_sources[j];
                new_node = j;
            }
        }

        mst.push((source_node, new_node, new_reachability));
        current_node = new_node;
    }

    mst
}

/// Order the `n-1` MST edges by ascending weight, replicating the oracle's
/// `np.argsort(min_spanning_tree["distance"])` ordering. The gate fixtures use
/// DISTINCT edge weights so this sort is tie-free — under distinct weights ANY
/// deterministic order is oracle-equal (RESEARCH Pitfall 1, option 2). On the
/// adversarial tie-heavy characterization fixture the ordering is the documented
/// D-04 gate, NOT a band.
///
/// We use a STABLE total-order sort on the `f64` weights via
/// [`f64::total_cmp`]; on the distinct-weight gate fixtures stability is moot
/// (no ties), and `total_cmp` gives a well-defined deterministic order even in
/// the tie-heavy case. Returns a NEW `Vec` (the input is left untouched).
pub fn argsort_by_weight(mst: &[MstEdge]) -> Vec<MstEdge> {
    let mut out = mst.to_vec();
    out.sort_by(|a, b| a.2.total_cmp(&b.2));
    out
}

/// Compute per-row core distances from a DENSE row-major `n×n` distance matrix
/// (the precomputed / dense-cosine path): `core[i]` is the
/// `(min_samples-1)`-th smallest distance in row `i` INCLUDING the self-zero
/// (sklearn `np.partition(row, k)[k]`, equivalent to the kth-smallest value).
///
/// `min_samples` is clamped to `1..=n` so the index `min_samples-1` is always in
/// range (a caller that resolved `min_samples=None → min_cluster_size` may exceed
/// `n` on a tiny input; sklearn's `np.partition` would clamp similarly). The dense
/// matrix is assumed already alpha-scaled by the caller (Variant-A placement).
pub fn core_distances_dense(dist: &[f64], n: usize, min_samples: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    debug_assert_eq!(dist.len(), n * n, "dist must be a dense n×n matrix");
    let k = min_samples.clamp(1, n.max(1)) - 1;
    let mut core = vec![0.0f64; n];
    // `select_nth_unstable_by` is np.partition itself: it places the k-th
    // smallest AT index k in O(n) and leaves the rest merely partitioned. The
    // full sort this replaces was O(n log n) per row for a single index — the
    // value at `k` is the same either way (equal elements are indistinguishable
    // here, so "unstable" costs nothing), and only that value is read.
    super::host_core::par_row_chunks(&mut core, 1, 64, |row0, out| {
        let mut row: Vec<f64> = vec![0.0; n];
        for (r, slot) in out.iter_mut().enumerate() {
            let i = row0 + r;
            row.copy_from_slice(&dist[i * n..(i + 1) * n]);
            let (_, nth, _) = row.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
            *slot = *nth;
        }
    });
    core
}

/// Build the DENSE row-major `n×n` mutual-reachability matrix from an
/// (alpha-scaled) distance matrix and per-row core distances:
/// `mr[i*n + j] = max(core[i], core[j], dist[i*n + j])`. The Variant-A input.
pub fn mutual_reachability_dense(dist: &[f64], core: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(dist.len(), n * n);
    debug_assert_eq!(core.len(), n);
    let mut mr = vec![0.0f64; n * n];
    // Row-parallel: each output row depends only on its own input row plus the
    // shared read-only `core`, so the split is value-neutral.
    super::host_core::par_row_chunks(&mut mr, n, 64, |row0, out| {
        for (r, row) in out.chunks_mut(n).enumerate() {
            let core_i = core[row0 + r];
            let src = &dist[(row0 + r) * n..(row0 + r + 1) * n];
            for j in 0..n {
                let mut m = src[j];
                if core_i > m {
                    m = core_i;
                }
                if core[j] > m {
                    m = core[j];
                }
                row[j] = m;
            }
        }
    });
    mr
}
