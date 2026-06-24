//! Cluster selection (EoM / leaf / epsilon / max_cluster_size), point labelling,
//! and membership probabilities over the condensed tree (HDBS-01/02, plan 15-04).
//!
//! Line-for-line host ports of sklearn's `_hdbscan/_tree.pyx`: `_get_clusters`
//! (both the Excess-of-Mass and leaf traversals, the `epsilon_search` merge for
//! `cluster_selection_epsilon > 0`, and the `max_cluster_size` deselect bound),
//! `_do_labelling` (a `TreeUnionFind` over the non-cluster edges mapping each
//! point's root to its cluster label, else `NOISE = -1`), `get_probabilities`
//! (`min(lambda_n, max_lambda)/max_lambda`, `1.0` when `max_lambda == 0` or
//! `lambda` is non-finite), and the `TreeUnionFind` itself (union-by-rank with
//! path compression — a DIFFERENT union-find from the fresh-label one in
//! `single_linkage.rs`).
//!
//! ## Output convention
//! [`get_clusters`] returns `(labels, probabilities)`: `labels[i]` is the integer
//! cluster id (`0..k`) or `-1` (noise); `probabilities[i] in [0, 1]` is the
//! membership strength. The cluster ids are assigned by sorting the selected
//! cluster nodes ascending (`cluster_map`), exactly as sklearn — so the result is
//! a `-1`-pinned permutation of sklearn's labels under the oracle gate.
//!
//! ## min_cluster_size, not min_samples (Pitfall 4)
//! Selection keys entirely off the condensed tree (built with `min_cluster_size`);
//! `min_samples` never appears here.
//!
//! Tests live in `crates/mlrs-algos/tests/hdbscan_test.rs` (AGENTS.md §2).

use std::collections::{BTreeSet, HashMap};

use super::condense::CondensedNode;
use super::stability::max_lambdas;

/// The noise label (sklearn `NOISE = -1`).
pub const NOISE: i64 = -1;

/// Union-find with union-by-rank + path compression (sklearn `TreeUnionFind`).
/// Distinct from `single_linkage::UnionFind` (which mints fresh labels per merge):
/// this one keeps a fixed id space `0..size` and tracks an `is_component` flag.
struct TreeUnionFind {
    /// `parent[i]` = the representative of `i` (starts as `i`).
    parent: Vec<usize>,
    /// `rank[i]` = the union-by-rank tie-breaker.
    rank: Vec<usize>,
}

impl TreeUnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    /// Find with path compression (sklearn `find`). Iterative two-pass (walk to
    /// root, then re-point) to avoid O(chain) recursion / stack overflow on a
    /// freshly-built union-find before compression (WR-05; sklearn's `find` is
    /// iterative for the same reason).
    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut x = x;
        while self.parent[x] != root {
            let next = self.parent[x];
            self.parent[x] = root;
            x = next;
        }
        root
    }

    /// Union by rank (sklearn `union`): the lower-rank root attaches to the
    /// higher; on a tie `y_root` attaches to `x_root` and `x_root`'s rank grows.
    fn union(&mut self, x: usize, y: usize) {
        let x_root = self.find(x);
        let y_root = self.find(y);
        if x_root == y_root {
            return;
        }
        if self.rank[x_root] < self.rank[y_root] {
            self.parent[x_root] = y_root;
        } else if self.rank[x_root] > self.rank[y_root] {
            self.parent[y_root] = x_root;
        } else {
            self.parent[y_root] = x_root;
            self.rank[x_root] += 1;
        }
    }
}

/// BFS over the CLUSTER tree (rows with `cluster_size > 1`) from `bfs_root`,
/// collecting every reachable node (sklearn `bfs_from_cluster_tree`). Used by the
/// EoM subtree-deselect and the epsilon merge.
fn bfs_from_cluster_tree(cluster_tree: &[CondensedNode], bfs_root: usize) -> Vec<usize> {
    let mut result: Vec<usize> = Vec::new();
    let mut process_queue: Vec<usize> = vec![bfs_root];
    while !process_queue.is_empty() {
        result.extend(process_queue.iter().copied());
        // children whose parent is in the current queue.
        let mut next: Vec<usize> = Vec::new();
        for r in cluster_tree {
            if process_queue.contains(&r.parent) {
                next.push(r.child);
            }
        }
        process_queue = next;
    }
    result
}

/// DFS leaves of the cluster subtree rooted at `current_node` (sklearn
/// `recurse_leaf_dfs`): a node with no children in the cluster tree is a leaf.
/// Explicit-stack iteration (WR-05) to avoid O(tree-height) recursion / stack
/// overflow on a deeply nested cluster tree. Children are pushed in reverse so
/// they pop in original (left-to-right) order, preserving the recursive
/// version's leaf ordering.
fn recurse_leaf_dfs(cluster_tree: &[CondensedNode], current_node: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut stack: Vec<usize> = vec![current_node];
    while let Some(node) = stack.pop() {
        let children: Vec<usize> = cluster_tree
            .iter()
            .filter(|r| r.parent == node)
            .map(|r| r.child)
            .collect();
        if children.is_empty() {
            out.push(node);
        } else {
            for child in children.into_iter().rev() {
                stack.push(child);
            }
        }
    }
    out
}

/// All leaves of the cluster tree (sklearn `get_cluster_tree_leaves`); empty when
/// the cluster tree is empty.
fn get_cluster_tree_leaves(cluster_tree: &[CondensedNode]) -> Vec<usize> {
    if cluster_tree.is_empty() {
        return Vec::new();
    }
    let root = cluster_tree
        .iter()
        .map(|r| r.parent)
        .min()
        .expect("non-empty cluster tree has a parent");
    recurse_leaf_dfs(cluster_tree, root)
}

/// Walk up the cluster tree from `leaf` until a parent whose `1/value` exceeds
/// `cluster_selection_epsilon` (sklearn `traverse_upwards`). On reaching the root,
/// returns the root if `allow_single_cluster` else the node closest to root.
fn traverse_upwards(
    cluster_tree: &[CondensedNode],
    cluster_selection_epsilon: f64,
    leaf: usize,
    allow_single_cluster: bool,
) -> usize {
    let root = cluster_tree
        .iter()
        .map(|r| r.parent)
        .min()
        .expect("non-empty cluster tree has a parent");
    // Iterative ascent (WR-05): the recursion was tail-recursive (it returned the
    // recursive call directly), so a loop over the current node climbs each
    // ancestor without O(tree-height) stack growth.
    let mut leaf = leaf;
    loop {
        let parent = cluster_tree
            .iter()
            .find(|r| r.child == leaf)
            .map(|r| r.parent)
            .expect("leaf must have a parent edge in the cluster tree");
        if parent == root {
            return if allow_single_cluster { parent } else { leaf };
        }
        let parent_value = cluster_tree
            .iter()
            .find(|r| r.child == parent)
            .map(|r| r.lambda)
            .expect("parent must have an incoming edge");
        let parent_eps = 1.0 / parent_value;
        if parent_eps > cluster_selection_epsilon {
            return parent;
        }
        leaf = parent;
    }
}

/// The `epsilon_search` merge (sklearn): for each leaf whose `1/value` is below
/// `cluster_selection_epsilon`, climb to the epsilon-stable ancestor and mark its
/// subtree processed; leaves at/above epsilon stay selected. Returns the selected
/// cluster node set.
fn epsilon_search(
    leaves: &BTreeSet<usize>,
    cluster_tree: &[CondensedNode],
    cluster_selection_epsilon: f64,
    allow_single_cluster: bool,
) -> BTreeSet<usize> {
    let mut selected_clusters: Vec<usize> = Vec::new();
    let mut processed: Vec<usize> = Vec::new();

    for &leaf in leaves {
        // eps = 1 / (the incoming edge value of this leaf).
        let leaf_value = cluster_tree
            .iter()
            .find(|r| r.child == leaf)
            .map(|r| r.lambda)
            .expect("leaf must have an incoming edge");
        let eps = 1.0 / leaf_value;
        if eps < cluster_selection_epsilon {
            if !processed.contains(&leaf) {
                let epsilon_child = traverse_upwards(
                    cluster_tree,
                    cluster_selection_epsilon,
                    leaf,
                    allow_single_cluster,
                );
                selected_clusters.push(epsilon_child);
                for sub_node in bfs_from_cluster_tree(cluster_tree, epsilon_child) {
                    if sub_node != epsilon_child {
                        processed.push(sub_node);
                    }
                }
            }
        } else {
            selected_clusters.push(leaf);
        }
    }

    selected_clusters.into_iter().collect()
}

/// Run the cluster selection over the `condensed_tree` + `stability` map and
/// produce `(labels, probabilities)` (sklearn `_get_clusters`).
///
/// `method` is `"eom"` or `"leaf"`. `allow_single_cluster` permits the EoM root
/// to be selected (needed for the homogeneous-single-blob case).
/// `cluster_selection_epsilon` (`> 0`) triggers the epsilon merge.
/// `max_cluster_size` (`0` = unbounded) deselects EoM clusters larger than it.
///
/// `n_samples` is the true point count (passed explicitly by the caller, as
/// sklearn does), used to size the labels/probabilities output. It must NOT be
/// reconstructed from the tree's singleton rows — a degenerate condensed tree
/// with no singleton child rows would otherwise infer `n_samples = 0` and emit a
/// mis-sized labels vector (WR-02).
///
/// Returns `labels` (length `n_samples`, `-1` = noise) and `probabilities`
/// (length `n_samples`, in `[0, 1]`).
pub fn get_clusters(
    condensed_tree: &[CondensedNode],
    stability: &HashMap<usize, f64>,
    method: SelectionMethod,
    allow_single_cluster: bool,
    cluster_selection_epsilon: f64,
    max_cluster_size: usize,
    n_samples: usize,
) -> (Vec<i64>, Vec<f64>) {
    // node_list = sorted(stability.keys(), reverse=True), excluding root unless
    // allow_single_cluster.
    let mut node_list: Vec<usize> = stability.keys().copied().collect();
    node_list.sort_unstable();
    node_list.reverse();
    if !allow_single_cluster && !node_list.is_empty() {
        node_list.pop(); // drop the smallest id (the root) — last after reverse-sort.
    }

    // cluster_tree = rows with cluster_size > 1.
    let cluster_tree: Vec<CondensedNode> = condensed_tree
        .iter()
        .copied()
        .filter(|r| r.cluster_size > 1)
        .collect();

    // is_cluster keyed by node id; default true for each node in node_list.
    let mut is_cluster: HashMap<usize, bool> = node_list.iter().map(|&n| (n, true)).collect();

    // n_samples is the true point count, supplied explicitly by the caller
    // (WR-02). It MUST NOT be inferred from the tree's singleton rows: a
    // degenerate condensed tree with no `cluster_size == 1` children would yield
    // `n_samples = 0` and a mis-sized labels/probabilities vector.
    debug_assert!(
        {
            let inferred = condensed_tree
                .iter()
                .filter(|r| r.cluster_size == 1)
                .map(|r| r.child)
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);
            // The explicit count must be at least the highest singleton child id;
            // sklearn passes `num_points` and never under-counts.
            inferred <= n_samples
        },
        "n_samples must cover every singleton child id in the condensed tree",
    );

    // max_cluster_size: 0 (unbounded) → a sentinel that never triggers.
    let max_cluster_size = if max_cluster_size == 0 {
        n_samples + 1
    } else {
        max_cluster_size
    };

    // cluster_sizes: child -> cluster_size over the cluster tree.
    let mut cluster_sizes: HashMap<usize, usize> = HashMap::new();
    for r in &cluster_tree {
        cluster_sizes.insert(r.child, r.cluster_size);
    }
    if allow_single_cluster {
        // Root cluster size = sum of its children's cluster sizes.
        if let Some(&root) = node_list.last() {
            let total: usize = cluster_tree
                .iter()
                .filter(|r| r.parent == root)
                .map(|r| r.cluster_size)
                .sum();
            cluster_sizes.insert(root, total);
        }
    }

    // A local mutable copy of stability so the EoM push-up can mutate it.
    let mut stability = stability.clone();

    match method {
        SelectionMethod::Eom => {
            for &node in &node_list {
                let subtree_stability: f64 = cluster_tree
                    .iter()
                    .filter(|r| r.parent == node)
                    .map(|r| *stability.get(&r.child).unwrap_or(&0.0))
                    .sum();
                let node_size = *cluster_sizes.get(&node).unwrap_or(&0);
                if subtree_stability > stability[&node] || node_size > max_cluster_size {
                    is_cluster.insert(node, false);
                    stability.insert(node, subtree_stability);
                } else {
                    for sub_node in bfs_from_cluster_tree(&cluster_tree, node) {
                        if sub_node != node {
                            is_cluster.insert(sub_node, false);
                        }
                    }
                }
            }

            if cluster_selection_epsilon != 0.0 && !cluster_tree.is_empty() {
                let eom_clusters: Vec<usize> = is_cluster
                    .iter()
                    .filter(|(_, &v)| v)
                    .map(|(&c, _)| c)
                    .collect();
                let cluster_root = cluster_tree
                    .iter()
                    .map(|r| r.parent)
                    .min()
                    .expect("non-empty cluster tree has a parent");
                let selected: BTreeSet<usize> =
                    if eom_clusters.len() == 1 && eom_clusters[0] == cluster_root {
                        if allow_single_cluster {
                            eom_clusters.into_iter().collect()
                        } else {
                            BTreeSet::new()
                        }
                    } else {
                        epsilon_search(
                            &eom_clusters.into_iter().collect(),
                            &cluster_tree,
                            cluster_selection_epsilon,
                            allow_single_cluster,
                        )
                    };
                let keys: Vec<usize> = is_cluster.keys().copied().collect();
                for c in keys {
                    is_cluster.insert(c, selected.contains(&c));
                }
            }
        }
        SelectionMethod::Leaf => {
            let leaves: BTreeSet<usize> =
                get_cluster_tree_leaves(&cluster_tree).into_iter().collect();
            if leaves.is_empty() {
                let keys: Vec<usize> = is_cluster.keys().copied().collect();
                for c in keys {
                    is_cluster.insert(c, false);
                }
                if let Some(root) = condensed_tree.iter().map(|r| r.parent).min() {
                    is_cluster.insert(root, true);
                }
            }

            let selected_clusters: BTreeSet<usize> = if cluster_selection_epsilon != 0.0 {
                epsilon_search(
                    &leaves,
                    &cluster_tree,
                    cluster_selection_epsilon,
                    allow_single_cluster,
                )
            } else {
                leaves
            };

            let keys: Vec<usize> = is_cluster.keys().copied().collect();
            for c in keys {
                is_cluster.insert(c, selected_clusters.contains(&c));
            }
        }
    }

    // clusters = {c : is_cluster[c]}, sorted ascending → cluster_map id 0..k.
    let clusters: BTreeSet<usize> = is_cluster
        .iter()
        .filter(|(_, &v)| v)
        .map(|(&c, _)| c)
        .collect();
    let cluster_map: HashMap<usize, i64> = clusters
        .iter()
        .enumerate()
        .map(|(n, &c)| (c, n as i64))
        .collect();
    let reverse_cluster_map: HashMap<i64, usize> =
        cluster_map.iter().map(|(&c, &n)| (n, c)).collect();

    let labels = do_labelling(
        condensed_tree,
        &clusters,
        &cluster_map,
        allow_single_cluster,
        cluster_selection_epsilon,
        n_samples,
    );
    let probs = get_probabilities(condensed_tree, &reverse_cluster_map, &labels);

    (labels, probs)
}

/// Map each point to its cluster label via a `TreeUnionFind` over the
/// NON-cluster edges (sklearn `_do_labelling`). A point whose root is the
/// `root_cluster` is noise (`-1`) unless the single-cluster allowance applies.
fn do_labelling(
    condensed_tree: &[CondensedNode],
    clusters: &BTreeSet<usize>,
    cluster_label_map: &HashMap<usize, i64>,
    allow_single_cluster: bool,
    cluster_selection_epsilon: f64,
    n_samples: usize,
) -> Vec<i64> {
    let root_cluster = condensed_tree
        .iter()
        .map(|r| r.parent)
        .min()
        .expect("non-empty condensed tree has a parent");
    let max_parent = condensed_tree
        .iter()
        .map(|r| r.parent)
        .max()
        .expect("non-empty condensed tree has a parent");

    let mut union_find = TreeUnionFind::new(max_parent + 1);
    for r in condensed_tree {
        if !clusters.contains(&r.child) {
            union_find.union(r.parent, r.child);
        }
    }

    let mut result = vec![NOISE; root_cluster.max(n_samples)];
    for n in 0..root_cluster {
        let cluster = union_find.find(n);
        let mut label = NOISE;
        if cluster != root_cluster {
            label = *cluster_label_map
                .get(&cluster)
                .expect("a non-root union root must be a selected cluster");
        } else if clusters.len() == 1 && allow_single_cluster {
            // The single point's incoming-edge lambda (a unique scalar).
            let parent_lambda = condensed_tree
                .iter()
                .find(|r| r.child == n)
                .map(|r| r.lambda);
            if let Some(parent_lambda) = parent_lambda {
                let threshold = if cluster_selection_epsilon != 0.0 {
                    1.0 / cluster_selection_epsilon
                } else {
                    // The largest lambda of any sibling under the root cluster.
                    condensed_tree
                        .iter()
                        .filter(|r| r.parent == cluster)
                        .map(|r| r.lambda)
                        .fold(f64::NEG_INFINITY, f64::max)
                };
                if parent_lambda >= threshold {
                    label = *cluster_label_map
                        .get(&cluster)
                        .expect("the single allowed cluster must be in the map");
                }
            }
        }
        result[n] = label;
    }

    // Trim to n_samples (root_cluster == n_samples when every point fell out, but
    // result was sized root_cluster.max(n_samples) defensively).
    result.truncate(root_cluster);
    result
}

/// Per-point membership probability (sklearn `get_probabilities`):
/// `min(lambda_n, max_lambda)/max_lambda`, or `1.0` when `max_lambda == 0` or the
/// point's lambda is non-finite. Noise points (`label == -1`) stay `0.0`.
fn get_probabilities(
    condensed_tree: &[CondensedNode],
    reverse_cluster_map: &HashMap<i64, usize>,
    labels: &[i64],
) -> Vec<f64> {
    let mut result = vec![0.0f64; labels.len()];
    let deaths = max_lambdas(condensed_tree);
    let root_cluster = condensed_tree
        .iter()
        .map(|r| r.parent)
        .min()
        .expect("non-empty condensed tree has a parent");

    for r in condensed_tree {
        let point = r.child;
        if point >= root_cluster {
            continue;
        }
        let cluster_num = labels[point];
        if cluster_num == NOISE {
            continue;
        }
        let cluster = reverse_cluster_map[&cluster_num];
        let max_lambda = deaths[cluster];
        if max_lambda == 0.0 || !r.lambda.is_finite() {
            result[point] = 1.0;
        } else {
            let lambda_val = r.lambda.min(max_lambda);
            result[point] = lambda_val / max_lambda;
        }
    }

    result
}

/// Cluster-selection method (mirrors the estimator's `ClusterSelectionMethod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMethod {
    /// Excess-of-Mass (sklearn `'eom'`).
    Eom,
    /// Leaf-cluster selection (sklearn `'leaf'`).
    Leaf,
}
