//! GLOSH `outlier_scores_` over the condensed tree (HDBS-03, plan 15-06).
//!
//! A line-for-line host port of the `hdbscan` library's
//! `_hdbscan_tree.pyx::outlier_scores` (GLOSH — Global-Local Outlier Score from
//! Hierarchies). sklearn's `HDBSCAN` has NO GLOSH, so the oracle for this score
//! is the `hdbscan` 0.8.44 library (D-07); the committed `outlier_scores` fixture
//! array is the real ≤1e-5 gate (RESEARCH Assumption A1).
//!
//! ## The algorithm (RESEARCH Pattern 7, verbatim)
//! ```text
//! deaths = max_lambdas(tree)              # per-parent "death" lambda (stability.rs)
//! root   = parent_array.min()             # the root cluster id
//! # UPWARD death-propagation reverse pass — a child's death floods its parent if larger:
//! for n in range(len(tree)-1, -1, -1):
//!     cluster, parent = child_array[n], parent_array[n]
//!     if deaths[cluster] > deaths[parent]: deaths[parent] = deaths[cluster]
//! # per-point score:
//! for n in range(len(tree)):
//!     point = child_array[n]
//!     if point >= root: continue
//!     lambda_max = deaths[parent_array[n]]
//!     if lambda_max == 0.0 or not isfinite(lambda_array[n]): result[point] = 0.0
//!     else: result[point] = (lambda_max - lambda_array[n]) / lambda_max
//! ```
//!
//! ## Why this is NOT `get_probabilities` (the key difference)
//! GLOSH indexes `deaths` by the point's PARENT (not its assigned cluster), runs
//! the upward death-propagation reverse pass FIRST (so a deeply-nested point sees
//! the maximum death lambda of its whole ancestor chain), and uses
//! `(lambda_max − lambda)/lambda_max`. `select::get_probabilities` indexes the
//! point's cluster death and uses `min(lambda, lambda_max)/lambda_max` — the
//! opposite ratio with no propagation. Conflating them silently mis-scores.
//!
//! All scalar math is `f64` (the host scalar domain), mirroring `stability.rs`.
//!
//! ## Why GLOSH needs its OWN tree (the hdbscan-convention pipeline)
//! GLOSH `outlier_scores_` is gated vs the `hdbscan` 0.8.44 library (D-07), which
//! builds a STRUCTURALLY DIFFERENT condensed tree than the sklearn pipeline that
//! produces `labels_`/`probabilities_`. Two differences are BOTH required to
//! reproduce the committed fixture at 0.0 diff (verified in a `/tmp` venv):
//!
//!   1. **Core distance at index `min_samples`** (NOT `min_samples-1`). hdbscan's
//!      `mutual_reachability` uses `np.partition(D, min_points, axis=0)[min_points]`
//!      — the `min_samples`-th smallest per column (symmetric ⇒ same as per-row).
//!      The sklearn `labels_` pipeline uses the `(min_samples-1)`-th
//!      ([`super::mst::core_distances_dense`]). [VERIFIED: hdbscan
//!      `_hdbscan_reachability.pyx::mutual_reachability`].
//!   2. **hdbscan's `mst_linkage_core` tie-order** ([`mst_linkage_core`] below): a
//!      dense Prim that tracks per-node `current_distances`/`current_sources` with
//!      NON-strict `d < current_distances[j]` updates and strict
//!      `current_distances[j] < new_distance` selection — a DIFFERENT tie
//!      resolution than the sklearn dense `argmin` Prim
//!      ([`super::mst::mst_from_mutual_reachability`]). [VERIFIED: hdbscan
//!      `_hdbscan_linkage.pyx::mst_linkage_core`].
//!
//! So `labels_`/`probabilities_` keep the sklearn-exact tree (34 tests pass), and
//! ONLY `outlier_scores_` runs over this parallel hdbscan-convention tree built by
//! [`hdbscan_outlier_scores`]. This honors D-07 (the GLOSH oracle is the hdbscan
//! library) without disturbing the sklearn-gated labels.
//!
//! Tests live in `crates/mlrs-algos/tests/hdbscan_test.rs` (AGENTS.md §2).

use super::condense::condense_tree;
use super::condense::CondensedNode;
use super::mst::{argsort_by_weight, MstEdge};
use super::single_linkage::make_single_linkage;
use super::stability::max_lambdas;

/// Compute the GLOSH `outlier_scores` for every point `0..n_samples` from the
/// `condensed_tree` (hdbscan `outlier_scores`). Returns a `Vec<f64>` of length
/// `n_samples` (each entry in `[0, 1]`); a point that never appears as a child
/// below the root keeps `0.0`.
///
/// `condensed_tree` is the output of [`super::condense::condense_tree`]. An empty
/// tree (degenerate all-noise input) yields an all-`0.0` vector of length
/// `n_samples`.
pub fn outlier_scores(condensed_tree: &[CondensedNode], n_samples: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; n_samples];
    if condensed_tree.is_empty() {
        return result;
    }

    // deaths[cluster] = per-parent max lambda (stability.rs::max_lambdas), indexed
    // by cluster id 0..=max(parent). The reverse pass below floods each parent
    // with its children's larger deaths (upward propagation).
    let mut deaths = max_lambdas(condensed_tree);
    // The root cluster id is min(parent) (sklearn `parent_array.min()`).
    let root = condensed_tree
        .iter()
        .map(|r| r.parent)
        .min()
        .expect("non-empty condensed tree has a parent");

    // UPWARD death-propagation reverse pass: iterate the condensed rows in REVERSE
    // (the rows are in topological parent order, so reverse is leaf→root). A
    // child's death floods its parent if larger. `deaths` is indexed by cluster id;
    // a child id may exceed `deaths.len()` (a singleton point child whose id is not
    // a parent) — those have no death entry and never flood (treated as 0).
    for r in condensed_tree.iter().rev() {
        let cluster = r.child;
        let parent = r.parent;
        let cluster_death = if cluster < deaths.len() {
            deaths[cluster]
        } else {
            0.0
        };
        // `parent` is always a valid index (deaths is sized max(parent)+1).
        if cluster_death > deaths[parent] {
            deaths[parent] = cluster_death;
        }
    }

    // Per-point score: index the death by the point's PARENT (the GLOSH/probabilities
    // difference), use `(lambda_max − lambda)/lambda_max`, 0.0 on `lambda_max == 0`
    // or non-finite point lambda.
    for r in condensed_tree {
        let point = r.child;
        // Skip internal cluster children (>= root); only genuine points score.
        if point >= root {
            continue;
        }
        let lambda_max = deaths[r.parent];
        if lambda_max == 0.0 || !r.lambda.is_finite() {
            // A point under a zero-death parent or born at an infinite lambda
            // (distance-0 merge) is, by GLOSH definition, not an outlier (0.0).
            result[point] = 0.0;
        } else {
            result[point] = (lambda_max - r.lambda) / lambda_max;
        }
    }

    result
}

/// Per-column core distances for the hdbscan-convention tree: `core[i]` is the
/// `min_samples`-th smallest distance in column `i` of the dense `n×n` distance
/// matrix (sklearn-pipeline core is the `(min_samples-1)`-th — see the module
/// doc). hdbscan's `mutual_reachability` does
/// `np.partition(D, min_points, axis=0)[min_points]`; we read per-ROW because the
/// distance matrix is symmetric (`D == D.T`, verified), so column-`i` and row-`i`
/// order are identical and a row read avoids a transpose.
///
/// `min_samples` is first capped to `min(n-1, min_samples)` (hdbscan
/// `mutual_reachability`'s `min_points = min(size-1, min_points)`), then the index
/// `min_samples` selects the `min_samples`-th smallest (0-indexed, self-zero at 0).
/// On a degenerate tiny input the cap keeps the index in range.
fn core_distances_hdbscan(dist: &[f64], n: usize, min_samples: usize) -> Vec<f64> {
    debug_assert_eq!(dist.len(), n * n, "dist must be a dense n×n matrix");
    // hdbscan caps min_points to n-1 BEFORE indexing (so index <= n-1).
    let mp = min_samples.min(n.saturating_sub(1));
    let mut core = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<f64> = dist[i * n..(i + 1) * n].to_vec();
        row.sort_by(|a, b| a.total_cmp(b));
        // `mp` is in `0..=n-1`; index directly (np.partition(...)[mp]).
        core.push(row[mp]);
    }
    core
}

/// hdbscan's dense `mst_linkage_core` Prim over the mutual-reachability matrix
/// `mr` (row-major `n×n`). A verbatim port of
/// `hdbscan/_hdbscan_linkage.pyx::mst_linkage_core` — DISTINCT from the sklearn
/// dense `argmin` Prim ([`super::mst::mst_from_mutual_reachability`]) in its tie
/// resolution: it tracks per-node `current_distances`/`current_sources`, updates
/// them on NON-strict improvement (`d < current_distances[j]`), and selects the
/// next node on strict improvement (`current_distances[j] < new_distance`). This
/// tie-order is one of the two differences required to reproduce the hdbscan
/// 0.8.44 GLOSH fixture (the other is [`core_distances_hdbscan`]).
///
/// Returns `n - 1` edges `(source, new_node, weight)`. `n >= 1`; `n == 1` yields
/// no edges.
fn mst_linkage_core(mr: &[f64], n: usize) -> Vec<MstEdge> {
    debug_assert_eq!(mr.len(), n * n, "mr must be a dense n×n matrix");
    if n <= 1 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut current_distances = vec![f64::MAX; n];
    let mut current_sources = vec![0usize; n];
    let mut result: Vec<MstEdge> = Vec::with_capacity(n - 1);

    let mut current_node: usize = 0;
    for _ in 1..n {
        in_tree[current_node] = true;
        let mut new_distance = f64::MAX;
        let mut new_node = 0usize;
        let mut source_node = 0usize;

        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            let d = mr[current_node * n + j];
            // NON-strict update: a strictly-smaller `d` claims node `j` for the
            // current node (so on a tie the EXISTING source is kept — hdbscan's
            // `d < current_distances[j]`).
            if d < current_distances[j] {
                current_distances[j] = d;
                current_sources[j] = current_node;
            }
            // Strict selection: the FIRST minimum of `current_distances` over the
            // not-yet-added nodes wins (lowest `j` on a tie, `j` scans ascending).
            if current_distances[j] < new_distance {
                new_distance = current_distances[j];
                source_node = current_sources[j];
                new_node = j;
            }
        }

        result.push((source_node, new_node, new_distance));
        current_node = new_node;
    }

    result
}

/// Compute the GLOSH `outlier_scores_` for every point `0..n` over the
/// **hdbscan-convention** condensed tree built from the dense `n×n` distance
/// matrix `dist` (already alpha-scaled by the caller, Variant-A placement). This
/// is the GLOSH-only parallel pipeline (D-07, Option A): core distances at index
/// `min_samples` ([`core_distances_hdbscan`]) → mutual-reachability →
/// [`mst_linkage_core`] (hdbscan tie-order) → argsort → single-linkage →
/// [`condense_tree`] by `min_cluster_size` → [`outlier_scores`]. It does NOT touch
/// the sklearn tree that produces `labels_`/`probabilities_`.
///
/// Returns a `Vec<f64>` of length `n` (each in `[0, 1]`). A `dist` for `n < 2`
/// yields all-`0.0` (no tree can form).
pub fn hdbscan_outlier_scores(
    dist: &[f64],
    n: usize,
    min_samples: usize,
    min_cluster_size: usize,
) -> Vec<f64> {
    debug_assert_eq!(dist.len(), n * n, "dist must be a dense n×n matrix");
    if n < 2 {
        return vec![0.0f64; n];
    }

    // hdbscan-convention core distance (index `min_samples`, not min_samples-1).
    let core = core_distances_hdbscan(dist, n, min_samples);
    // Mutual reachability `max(core_i, core_j, d_ij)` (dist already /alpha by the
    // caller). Reuse the shared dense builder — the MR formula is identical; only
    // the CORE-distance index differs from the sklearn path.
    let mr = super::mst::mutual_reachability_dense(dist, &core, n);
    // hdbscan dense Prim (its tie-order), argsort by weight, single linkage.
    let edges = mst_linkage_core(&mr, n);
    let sorted = argsort_by_weight(&edges);
    let hierarchy = make_single_linkage(&sorted, n);
    // Condense by min_cluster_size (NOT min_samples, Pitfall 4) and run GLOSH.
    let condensed = condense_tree(&hierarchy, min_cluster_size);
    outlier_scores(&condensed, n)
}
