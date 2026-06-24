//! Cluster stability + per-parent max-lambda over the condensed tree (HDBS-02,
//! plan 15-04).
//!
//! Line-for-line host ports of sklearn's `_hdbscan/_tree.pyx::_compute_stability`
//! and `max_lambdas`. Stability is the Excess-of-Mass coherence score the EoM
//! selection maximises; `max_lambdas` (the per-parent "death" lambda) feeds the
//! membership-probability computation.
//!
//! ## `_compute_stability` (verbatim)
//! For each condensed row `(parent, child, lambda, cluster_size)`:
//! `births[child] = lambda`; then `births[smallest_cluster] = 0` (the root has no
//! birth); finally `stability[parent] += (lambda - births[parent]) * cluster_size`
//! accumulated over every row. `smallest_cluster` is `min(parent)` — the root
//! cluster id (`n_samples`). The result is a dense `HashMap<usize, f64>` keyed by
//! cluster id, exactly sklearn's `stability_dict`.
//!
//! ## `max_lambdas` (verbatim)
//! sklearn relies on the condensed rows being grouped by parent (the topological
//! order `_condense_tree` produces); it sweeps rows tracking the running max
//! lambda per `current_parent`, flushing when the parent id changes. We replicate
//! that exact sweep (NOT a global group-by) so the result is bit-identical even on
//! the degenerate grouping sklearn assumes.
//!
//! All scalar math is `f64` (the host scalar domain); no `min_samples` is used
//! here (Pitfall 4 — stability/selection key off `min_cluster_size` upstream).
//!
//! Tests live in `crates/mlrs-algos/tests/hdbscan_test.rs` (AGENTS.md §2).

use std::collections::HashMap;

use super::condense::CondensedNode;

/// Compute the Excess-of-Mass stability per cluster id (sklearn
/// `_compute_stability`). Returns a map `cluster_id -> stability` covering every
/// cluster id in `[min(parent), max(parent)]` (the dense `stability_dict`).
///
/// `condensed_tree` is the output of [`super::condense::condense_tree`]. Empty
/// input yields an empty map (a degenerate tree with no internal clusters).
pub fn compute_stability(condensed_tree: &[CondensedNode]) -> HashMap<usize, f64> {
    if condensed_tree.is_empty() {
        return HashMap::new();
    }

    let smallest_cluster = condensed_tree
        .iter()
        .map(|r| r.parent)
        .min()
        .expect("non-empty condensed tree has a parent");
    let largest_parent = condensed_tree
        .iter()
        .map(|r| r.parent)
        .max()
        .expect("non-empty condensed tree has a parent");
    let largest_child = condensed_tree
        .iter()
        .map(|r| r.child)
        .max()
        .expect("non-empty condensed tree has a child");
    // sklearn: largest_child = max(largest_child, smallest_cluster).
    let largest_child = largest_child.max(smallest_cluster);

    // births[node] = the lambda at which `node` is born (the row whose child==node).
    // NaN = "unset" (a node that is never a child — only the root, fixed to 0 below).
    let mut births = vec![f64::NAN; largest_child + 1];
    for r in condensed_tree {
        births[r.child] = r.lambda;
    }
    births[smallest_cluster] = 0.0;

    let num_clusters = largest_parent - smallest_cluster + 1;
    let mut result = vec![0.0f64; num_clusters];
    for r in condensed_tree {
        let parent = r.parent;
        let lambda_val = r.lambda;
        let cluster_size = r.cluster_size as f64;
        let result_index = parent - smallest_cluster;
        result[result_index] += (lambda_val - births[parent]) * cluster_size;
    }

    let mut stability = HashMap::with_capacity(num_clusters);
    for (idx, &s) in result.iter().enumerate() {
        stability.insert(idx + smallest_cluster, s);
    }
    stability
}

/// Per-parent maximum lambda ("death" lambda), sklearn `max_lambdas`. Returns a
/// dense `Vec<f64>` indexed by cluster id `0..=max(parent)` (entries for ids that
/// are never a parent stay `0.0`).
///
/// Replicates sklearn's single sweep that ASSUMES the condensed rows are grouped
/// by parent: it tracks the running `max_lambda` for `current_parent`, flushing
/// `deaths[current_parent]` when the parent id changes, then flushes the final
/// parent. Empty input yields an empty `Vec`.
pub fn max_lambdas(condensed_tree: &[CondensedNode]) -> Vec<f64> {
    if condensed_tree.is_empty() {
        return Vec::new();
    }

    let largest_parent = condensed_tree
        .iter()
        .map(|r| r.parent)
        .max()
        .expect("non-empty condensed tree has a parent");
    let mut deaths = vec![0.0f64; largest_parent + 1];

    let mut current_parent = condensed_tree[0].parent;
    let mut max_lambda = condensed_tree[0].lambda;

    for r in &condensed_tree[1..] {
        let parent = r.parent;
        let lambda_val = r.lambda;
        if parent == current_parent {
            if lambda_val > max_lambda {
                max_lambda = lambda_val;
            }
        } else {
            deaths[current_parent] = max_lambda;
            current_parent = parent;
            max_lambda = lambda_val;
        }
    }
    deaths[current_parent] = max_lambda; // flush the last parent
    deaths
}
