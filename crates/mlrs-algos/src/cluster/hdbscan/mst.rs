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
    debug_assert_eq!(dist.len(), n * n, "dist must be a dense n×n matrix");
    let k = min_samples.clamp(1, n) - 1;
    let mut core = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<f64> = dist[i * n..(i + 1) * n].to_vec();
        // kth-smallest value: a full sort is the simplest exact equivalent of
        // np.partition(row, k)[k] for a single index (the value is identical).
        row.sort_by(|a, b| a.total_cmp(b));
        core.push(row[k]);
    }
    core
}

/// Build the DENSE row-major `n×n` mutual-reachability matrix from an
/// (alpha-scaled) distance matrix and per-row core distances:
/// `mr[i*n + j] = max(core[i], core[j], dist[i*n + j])`. The Variant-A input.
pub fn mutual_reachability_dense(dist: &[f64], core: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(dist.len(), n * n);
    debug_assert_eq!(core.len(), n);
    let mut mr = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut m = dist[i * n + j];
            if core[i] > m {
                m = core[i];
            }
            if core[j] > m {
                m = core[j];
            }
            mr[i * n + j] = m;
        }
    }
    mr
}
