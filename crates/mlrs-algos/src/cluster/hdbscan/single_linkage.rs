//! Single-linkage dendrogram from an argsorted MST (HDBS-02, plan 15-03).
//!
//! A line-for-line host port of sklearn's `_hdbscan/_linkage.pyx::make_single_linkage`
//! (driven by the `UnionFind` from `_hierarchical_fast.pyx`). The MST edges have
//! ALREADY been ordered by ascending weight (see [`super::mst::argsort_by_weight`]);
//! this module folds them into the `2N-1`-node single-linkage hierarchy that the
//! Wave-3 condense/select stage (plan 15-04) consumes.
//!
//! ## Why a fresh-label UnionFind (D-04)
//! `UnionFind::union` always mints a NEW label `N + i` per merge and `fast_find`
//! returns the CURRENT root, so the single-linkage `(left, right)` columns are the
//! current root labels — meaning the MERGE ORDER (fixed by the argsort) directly
//! determines the dendrogram node ids → the condensed tree → the labels. This is
//! exactly why the oracle's argsort tie-order is the D-04 exactness crux; the
//! gate fixtures use DISTINCT MST edge weights so the order is tie-free and
//! oracle-equal under any deterministic sort (RESEARCH Pitfall 1, option 2).
//!
//! Tests live in `crates/mlrs-algos/tests/hdbscan_test.rs` (AGENTS.md §2).

/// One row of the single-linkage hierarchy: a merge of cluster `left` and cluster
/// `right` at `distance`, producing a new cluster of `size` members. Mirrors
/// sklearn's `HIERARCHY_dtype` `(left_node, right_node, value, cluster_size)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SingleLinkageEdge {
    /// The (current-root) label of the left child cluster.
    pub left: usize,
    /// The (current-root) label of the right child cluster.
    pub right: usize,
    /// The merge distance (the MST edge weight).
    pub distance: f64,
    /// The total member count of the merged cluster (`size[left] + size[right]`).
    pub size: usize,
}

/// The union-find used by single linkage — a verbatim port of sklearn's
/// `_hierarchical_fast.pyx::UnionFind`. Unlike a classic disjoint-set, every
/// `union` mints a NEW label (`N`, `N+1`, …): `parent` has room for all `2N-1`
/// dendrogram nodes, and `size` accumulates per merged node. `fast_find`
/// path-compresses on the way up.
pub struct UnionFind {
    /// `parent[node]` = the node it was merged into (`usize::MAX` sentinel = a
    /// root with no parent yet). Length `2N-1`.
    parent: Vec<usize>,
    /// The next fresh label to mint on the next `union` (starts at `N`).
    next_label: usize,
    /// `size[node]` = member count of the cluster rooted at `node`. The `N`
    /// singleton points start at `1`; the `N-1` internal nodes start at `0` and
    /// are filled as they are minted.
    size: Vec<usize>,
}

/// Sentinel for "no parent yet" (sklearn uses `-1` in an `intp` array; we use a
/// `usize` array so the sentinel is `usize::MAX`).
const NO_PARENT: usize = usize::MAX;

impl UnionFind {
    /// Construct the union-find over `n` singleton points (sklearn `UnionFind(n)`):
    /// `parent = full(2n-1, NO_PARENT)`, `next_label = n`,
    /// `size = [1]*n + [0]*(n-1)`.
    pub fn new(n: usize) -> Self {
        let total = 2 * n - 1;
        let mut size = vec![0usize; total];
        for s in size.iter_mut().take(n) {
            *s = 1;
        }
        Self {
            parent: vec![NO_PARENT; total],
            next_label: n,
            size,
        }
    }

    /// Find the current root of `node` (walk parents to the root), then
    /// path-compress every node on the walk to point straight at the root.
    /// Verbatim port of sklearn `fast_find`.
    pub fn fast_find(&mut self, mut node: usize) -> usize {
        let mut root = node;
        // Walk to the root.
        while self.parent[root] != NO_PARENT {
            root = self.parent[root];
        }
        // Path-compress: re-point every node on the path to `root`.
        while self.parent[node] != NO_PARENT {
            let next = self.parent[node];
            self.parent[node] = root;
            node = next;
        }
        root
    }

    /// Merge roots `m` and `n` under a freshly-minted label: both get
    /// `parent = next_label`, the new node's `size` is the sum, then
    /// `next_label += 1`. Verbatim port of sklearn `union`.
    pub fn union(&mut self, m: usize, n: usize) {
        let new_label = self.next_label;
        self.parent[m] = new_label;
        self.parent[n] = new_label;
        self.size[new_label] = self.size[m] + self.size[n];
        self.next_label += 1;
    }

    /// Read the size of `node` (used to record `size[a] + size[b]` BEFORE the
    /// union mutates `next_label`).
    pub fn size_of(&self, node: usize) -> usize {
        self.size[node]
    }
}

/// Fold the argsorted MST edges (`(u, v, weight)`, already ascending by weight)
/// into the single-linkage hierarchy. Verbatim port of sklearn
/// `make_single_linkage`: for each edge, `a = find(u)`, `b = find(v)`, record
/// `(a, b, weight, size[a]+size[b])`, then `union(a, b)`.
///
/// `mst` must have exactly `n - 1` edges (a spanning tree over `n` points); the
/// returned hierarchy has `n - 1` rows.
pub fn make_single_linkage(mst: &[(usize, usize, f64)], n: usize) -> Vec<SingleLinkageEdge> {
    debug_assert_eq!(mst.len(), n - 1, "an MST over n points has exactly n-1 edges");
    let mut uf = UnionFind::new(n);
    let mut out = Vec::with_capacity(n - 1);
    for &(u, v, distance) in mst {
        let a = uf.fast_find(u);
        let b = uf.fast_find(v);
        let size = uf.size_of(a) + uf.size_of(b);
        out.push(SingleLinkageEdge {
            left: a,
            right: b,
            distance,
            size,
        });
        uf.union(a, b);
    }
    out
}
