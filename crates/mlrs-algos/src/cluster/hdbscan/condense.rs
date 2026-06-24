//! Condense the single-linkage hierarchy by `min_cluster_size` (HDBS-02, plan
//! 15-04).
//!
//! A line-for-line host port of sklearn's `_hdbscan/_tree.pyx::_condense_tree`
//! (driven by `bfs_from_hierarchy`). The single-linkage hierarchy from 15-03
//! (`single_linkage::make_single_linkage`, the `2N-1`-node dendrogram) is
//! "runt-pruned" by `min_cluster_size`: a genuine split (both children at least
//! `min_cluster_size`) keeps both children with FRESH labels; otherwise the runt
//! side's points "fall out" at `lambda = 1/distance` (or `INFTY` when
//! `distance == 0`). The result is the CONDENSED tree — a list of
//! `(parent, child, lambda, child_size)` rows that the stability + selection
//! stages consume.
//!
//! ## Why `min_cluster_size`, not `min_samples` (Pitfall 4)
//! `_condense_tree` prunes by the MINIMUM-CLUSTER-SIZE hyperparameter. The
//! core-distance smoothing `min_samples` was already consumed upstream (in the
//! 15-03 mutual-reachability core distances); swapping the two here silently
//! mislabels. This module takes `min_cluster_size` ONLY and never sees
//! `min_samples` (threat T-15-04-MIS).
//!
//! ## Node id convention (mirrors sklearn)
//! The hierarchy has `N - 1` rows over `N` singleton points; internal node `i`
//! (id `N + i`) is row `i`. The BFS root is `2*(N-1)` (the last merge). A child
//! id `< N` is a singleton point; `>= N` is an internal cluster whose
//! `cluster_size` is read from its hierarchy row.
//!
//! Tests live in `crates/mlrs-algos/tests/hdbscan_test.rs` (AGENTS.md §2).

use super::single_linkage::SingleLinkageEdge;

/// One row of the condensed tree (sklearn `CONDENSED_dtype`): a `parent` cluster
/// id, a `child` (a point `< n_samples` or a sub-cluster id), the `lambda` value
/// (`1/distance`, or `INFTY` for a distance-0 merge) at which the child departs,
/// and the `cluster_size` of that child.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CondensedNode {
    /// The parent cluster id (always `>= n_samples`).
    pub parent: usize,
    /// The child: a point index `< n_samples`, or a sub-cluster id `>= n_samples`.
    pub child: usize,
    /// The lambda value `1/distance` (or `f64::INFINITY` when `distance == 0`).
    pub lambda: f64,
    /// The number of points in the child cluster (`1` for a singleton point).
    pub cluster_size: usize,
}

/// Breadth-first node list of the hierarchy from `bfs_root`, in sklearn's
/// `bfs_from_hierarchy` order. Each internal node `>= n_samples` expands to its
/// `(left, right)` children (read from hierarchy row `node - n_samples`); points
/// `< n_samples` are leaves.
///
/// `n_samples == hierarchy.len() + 1`. The returned list contains every node
/// reachable from `bfs_root` (internal nodes and the leaf points beneath them),
/// in level order — exactly the traversal `_condense_tree` walks.
pub fn bfs_from_hierarchy(hierarchy: &[SingleLinkageEdge], bfs_root: usize) -> Vec<usize> {
    let n_samples = hierarchy.len() + 1;
    let mut process_queue: Vec<usize> = vec![bfs_root];
    let mut result: Vec<usize> = Vec::new();

    while !process_queue.is_empty() {
        result.extend(process_queue.iter().copied());
        // By construction node `x` (>= n_samples) is the union of
        // hierarchy[x - n_samples].{left,right}. Drop the leaf points (< n_samples)
        // and map the internal nodes to their hierarchy rows.
        let internal: Vec<usize> = process_queue
            .iter()
            .copied()
            .filter(|&x| x >= n_samples)
            .map(|x| x - n_samples)
            .collect();
        if internal.is_empty() {
            break;
        }
        let mut next_queue: Vec<usize> = Vec::with_capacity(internal.len() * 2);
        for node in internal {
            next_queue.push(hierarchy[node].left);
            next_queue.push(hierarchy[node].right);
        }
        process_queue = next_queue;
    }
    result
}

/// Condense the single-linkage `hierarchy` by `min_cluster_size` (runt-pruning),
/// returning the condensed tree as a list of [`CondensedNode`] rows. Verbatim
/// port of sklearn `_condense_tree`.
///
/// `hierarchy` has `N - 1` rows over `N` singleton points (the 15-03
/// `make_single_linkage` output). The root is `2*(N-1)`; relabeling starts at
/// `next_label = N + 1` (the root relabels to `N`). For each internal node in BFS
/// order (skipping ignored points and singleton leaves):
///
/// - genuine split (`left_count >= mcs && right_count >= mcs`) → both children get
///   fresh labels and a `(parent, child, lambda, count)` row each;
/// - both runt (`< mcs`) → every point under both subtrees falls out at `lambda`;
/// - one runt → the kept child inherits the parent's label, the runt's points
///   fall out.
///
/// `min_cluster_size` (NOT `min_samples`, Pitfall 4) is the prune threshold.
pub fn condense_tree(hierarchy: &[SingleLinkageEdge], min_cluster_size: usize) -> Vec<CondensedNode> {
    let n_samples = hierarchy.len() + 1;
    // The single-linkage root is the last merged node: 2*(n_samples-1) == 2*rows.
    let root = 2 * hierarchy.len();
    let mut next_label = n_samples + 1;

    let node_list = bfs_from_hierarchy(hierarchy, root);

    // relabel has room for every node id 0..=root; relabel[root] = n_samples.
    let mut relabel = vec![0usize; root + 1];
    relabel[root] = n_samples;
    // ignore[node] marks points that have already fallen out (so a later BFS visit
    // skips them). Sized to the max node id seen in node_list (== root for a full
    // tree, but BFS may not touch every id — size to root+1 to be safe).
    let mut ignore = vec![false; root + 1];

    let mut result: Vec<CondensedNode> = Vec::new();

    for &node in &node_list {
        if ignore[node] || node < n_samples {
            continue;
        }

        let children = hierarchy[node - n_samples];
        let left = children.left;
        let right = children.right;
        let distance = children.distance;
        let lambda_value = if distance > 0.0 {
            1.0 / distance
        } else {
            f64::INFINITY
        };

        let left_count = if left >= n_samples {
            hierarchy[left - n_samples].size
        } else {
            1
        };
        let right_count = if right >= n_samples {
            hierarchy[right - n_samples].size
        } else {
            1
        };

        if left_count >= min_cluster_size && right_count >= min_cluster_size {
            // Genuine split: both children keep, each gets a fresh label.
            relabel[left] = next_label;
            next_label += 1;
            result.push(CondensedNode {
                parent: relabel[node],
                child: relabel[left],
                lambda: lambda_value,
                cluster_size: left_count,
            });

            relabel[right] = next_label;
            next_label += 1;
            result.push(CondensedNode {
                parent: relabel[node],
                child: relabel[right],
                lambda: lambda_value,
                cluster_size: right_count,
            });
        } else if left_count < min_cluster_size && right_count < min_cluster_size {
            // Both runt: every point under both subtrees falls out at lambda.
            for sub_node in bfs_from_hierarchy(hierarchy, left) {
                if sub_node < n_samples {
                    result.push(CondensedNode {
                        parent: relabel[node],
                        child: sub_node,
                        lambda: lambda_value,
                        cluster_size: 1,
                    });
                }
                ignore[sub_node] = true;
            }
            for sub_node in bfs_from_hierarchy(hierarchy, right) {
                if sub_node < n_samples {
                    result.push(CondensedNode {
                        parent: relabel[node],
                        child: sub_node,
                        lambda: lambda_value,
                        cluster_size: 1,
                    });
                }
                ignore[sub_node] = true;
            }
        } else if left_count < min_cluster_size {
            // Left runt: right keeps the parent label, left's points fall out.
            relabel[right] = relabel[node];
            for sub_node in bfs_from_hierarchy(hierarchy, left) {
                if sub_node < n_samples {
                    result.push(CondensedNode {
                        parent: relabel[node],
                        child: sub_node,
                        lambda: lambda_value,
                        cluster_size: 1,
                    });
                }
                ignore[sub_node] = true;
            }
        } else {
            // Right runt: left keeps the parent label, right's points fall out.
            relabel[left] = relabel[node];
            for sub_node in bfs_from_hierarchy(hierarchy, right) {
                if sub_node < n_samples {
                    result.push(CondensedNode {
                        parent: relabel[node],
                        child: sub_node,
                        lambda: lambda_value,
                        cluster_size: 1,
                    });
                }
                ignore[sub_node] = true;
            }
        }
    }

    result
}
