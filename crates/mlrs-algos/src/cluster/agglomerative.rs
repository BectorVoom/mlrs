//! `AgglomerativeClustering` (AGGLO-01) — single-linkage hierarchical
//! clustering, sklearn/cuML-parity.
//!
//! cuML's `AgglomerativeClustering` supports `linkage='single'` only; mlrs
//! mirrors that scope. The pipeline is a line-exact port of sklearn 1.9.0's
//! unstructured (`connectivity=None`) single-linkage path:
//!
//! 1. **Pairwise distances** — Euclidean runs on the DEVICE via the Phase-2
//!    `distance` prim (GEMM expansion + sqrt, the O(n²p) bulk of the fit);
//!    Manhattan/Cosine build the dense matrix host-side (the HDBSCAN
//!    `feature_metric_dense_distances` precedent — sklearn's own fast path is
//!    also scalar host code for these).
//! 2. **MST-LINKAGE-CORE** (Müllner Fig. 6) — the chain-recording Prim walk
//!    sklearn's `_hierarchical_fast.pyx::mst_linkage_core` performs. NOTE: this
//!    is DISTINCT from HDBSCAN's source-tracking `mst_from_data_matrix`; the
//!    recorded edge source is the PREVIOUS chain node, not the true nearest
//!    tree node — the downstream union-find labelling is defined over exactly
//!    this output, so the port preserves it.
//! 3. **Stable argsort by weight** (`np.argsort(kind='mergesort')` /
//!    scipy `kind='stable'` — Rust `sort_by` is stable, so the tie order
//!    matches both).
//! 4. **Labelling** — two oracle-exact variants (sklearn dispatches by metric):
//!    - Euclidean/Manhattan (`METRIC_MAPPING64` fast path):
//!      `single_linkage_label` = the fresh-label `UnionFind` already ported in
//!      [`super::hdbscan::single_linkage::make_single_linkage`], rows recorded
//!      as `(find(u), find(v))` in RAW find order.
//!    - Cosine (sklearn falls back to `scipy.cluster.hierarchy.linkage`):
//!      scipy's `label` pass — same fresh-label union-find but each row is
//!      recorded as `(min(root_u, root_v), max(root_u, root_v))`. Verified
//!      line-exact vs scipy 1.x over randomized trials at design time.
//! 5. **`_hc_cut`** — sklearn's heapq-driven tree cut. The label ids depend on
//!    the FINAL HEAP ARRAY ORDER, so the Python `heapq` siftdown/siftup pair is
//!    ported verbatim ([`heappush`] / [`heappushpop`]).
//!
//! The fitted surface is sklearn's: `labels_` (device-resident `i32`),
//! `children_` (host `(n-1)×2`), `n_clusters_`, `n_leaves_`,
//! `n_connected_components_` (always `1` — unstructured fit is one component).
//!
//! Tests live in `crates/mlrs-algos/tests/agglomerative_test.rs` (AGENTS.md §2)
//! against `agglomerative_*_seed42.npz` fixtures (sklearn 1.9.0, EXACT label +
//! children equality — the port is deterministic, no permutation matching).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::tsne::squared_distance;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, State, Unfit};

use super::hdbscan::single_linkage;

/// The pairwise metric for the single-linkage merge (cuML's supported set,
/// deduplicated: cuML/sklearn `'l2'` is an alias of `'euclidean'` and `'l1'` of
/// `'manhattan'` — the Python shim maps the aliases onto these three variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// L2 (Euclidean) — sklearn's `metric='euclidean'` default. Device path.
    Euclidean,
    /// L1 (Manhattan / cityblock) — sklearn's `metric='manhattan'`/`'l1'`.
    Manhattan,
    /// Cosine distance `1 − x̂·ŷ` — sklearn's `metric='cosine'` (routes through
    /// the scipy labelling convention, see the module docs).
    Cosine,
}

/// Single-linkage agglomerative clustering (AGGLO-01), builder-fronted +
/// typestate (`AgglomerativeClustering<F, S = Unfit>`). No `Debug` derive —
/// `DeviceArray` is not `Debug` (the DBSCAN/HDBSCAN precedent).
pub struct AgglomerativeClustering<F, S = Unfit>
where
    S: State,
{
    /// The number of flat clusters to cut the dendrogram into (sklearn
    /// `n_clusters`, default `2`). Validated `>= 1` at build; `<= n_samples`
    /// at fit (data-dependent).
    n_clusters: usize,
    /// The pairwise metric (sklearn `metric`, default Euclidean).
    metric: Metric,
    /// Fitted flat cluster labels (length `n`, device-resident). `Some` by
    /// construction on `Fitted`.
    labels_: Option<DeviceArray<ActiveRuntime, i32>>,
    /// Fitted dendrogram children (`(n-1) × 2`, host — sklearn `children_`).
    /// Node ids `< n` are leaves; node `n + i` is the merge recorded in row `i`.
    children_: Option<Vec<[i64; 2]>>,
    /// Number of leaves (`n_samples`), sklearn `n_leaves_`.
    n_leaves_: usize,
    /// Number of features seen at fit (`n_features_in_`).
    n_features_in_: usize,
    _float: PhantomData<F>,
    _state: PhantomData<S>,
}

impl<F> AgglomerativeClustering<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// sklearn defaults: `n_clusters = 2`, `metric = 'euclidean'` (D-08 single
    /// source — the builder re-derives its defaults from here).
    pub fn new() -> Self {
        Self {
            n_clusters: 2,
            metric: Metric::Euclidean,
            labels_: None,
            children_: None,
            n_leaves_: 0,
            n_features_in_: 0,
            _float: PhantomData,
            _state: PhantomData,
        }
    }

    /// Start building from sklearn's defaults (D-08 single source).
    pub fn builder() -> AgglomerativeClusteringBuilder {
        AgglomerativeClusteringBuilder::default()
    }

    /// Fold this (unfit) estimator back into a builder (round-trip surface).
    pub fn into_builder(self) -> AgglomerativeClusteringBuilder {
        AgglomerativeClusteringBuilder {
            n_clusters: self.n_clusters,
            metric: self.metric,
        }
    }
}

impl<F> Default for AgglomerativeClustering<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`AgglomerativeClustering`] (data-INDEPENDENT validation at
/// `build`, D-08).
#[derive(Debug, Clone, Copy)]
pub struct AgglomerativeClusteringBuilder {
    n_clusters: usize,
    metric: Metric,
}

impl Default for AgglomerativeClusteringBuilder {
    /// Re-derive the sklearn defaults from [`AgglomerativeClustering::new`]
    /// (D-08 single source; `f64` pinned only to read F-independent scalars).
    fn default() -> Self {
        AgglomerativeClustering::<f64, Unfit>::new().into_builder()
    }
}

impl AgglomerativeClusteringBuilder {
    /// Set the flat cluster count `n_clusters`.
    pub fn n_clusters(mut self, v: usize) -> Self {
        self.n_clusters = v;
        self
    }

    /// Set the pairwise metric.
    pub fn metric(mut self, v: Metric) -> Self {
        self.metric = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08):
    /// - `n_clusters >= 1` ([`BuildError::InvalidNClusters`]) — sklearn raises
    ///   on `n_clusters <= 0`. The data-DEPENDENT `n_clusters <= n_samples`
    ///   bound is a fit-body check ([`AlgoError::InvalidK`]).
    pub fn build<F>(self) -> Result<AgglomerativeClustering<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if self.n_clusters < 1 {
            return Err(BuildError::InvalidNClusters {
                estimator: "agglomerative_clustering",
                n_clusters: self.n_clusters,
            });
        }
        Ok(AgglomerativeClustering {
            n_clusters: self.n_clusters,
            metric: self.metric,
            labels_: None,
            children_: None,
            n_leaves_: 0,
            n_features_in_: 0,
            _float: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for AgglomerativeClustering<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = AgglomerativeClustering<F, Fitted>;

    /// Fit: dense pairwise distances (Euclidean on device) → chain-Prim MST →
    /// stable weight sort → metric-routed union-find labelling → `_hc_cut`.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<AgglomerativeClustering<F, Fitted>, AlgoError> {
        let (n, p) = shape;
        validate_geometry(x, shape)?;

        // sklearn requires >= 2 samples (`ensure_min_samples=2`): a 1-point
        // dendrogram has no merge row and `_hc_cut` reads `children[-1]`.
        // Surface the same contract as a typed shape error BEFORE any compute.
        if n < 2 {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x (agglomerative clustering requires >= 2 samples)",
                rows: n,
                cols: p,
                len: x.len(),
            }));
        }

        // Data-DEPENDENT hyperparameter bound (sklearn `_hc_cut` raises when
        // `n_clusters > n_leaves`): typed error BEFORE any launch.
        if self.n_clusters > n {
            return Err(AlgoError::InvalidK {
                estimator: "agglomerative_clustering",
                k: self.n_clusters,
                n_samples: n,
            });
        }

        // --- 1. Dense n×n pairwise distance matrix (f64 host copy). ---
        let nn = n
            .checked_mul(n)
            .ok_or(AlgoError::Prim(PrimError::Overflow {
                operand: "agglomerative_distance_matrix",
                lhs: n,
                rhs: n,
            }))?;
        let dist: Vec<f64> = match self.metric {
            Metric::Euclidean => {
                // Device path: the direct-GATHER SQUARED-distance prim (the
                // O(n²p) bulk on device), single read-back. NOT the
                // GEMM-expansion `distance` prim: its `row_reduce(Shared)` norm
                // term is pathologically slow under PyO3 (see
                // `mlrs_backend::prims::tsne::squared_distance` docs). Single-
                // linkage `children_`/`labels_` are INVARIANT under the
                // monotone `x ↦ √x`, so squared distances give byte-identical
                // output — the boundary sqrt is pure wasted work here, dropped.
                let dmat = squared_distance::<F>(pool, x, n, p);
                let d: Vec<f64> = dmat.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
                dmat.release_into(pool);
                d
            }
            Metric::Manhattan => {
                let x_host: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
                let mut d = vec![0.0f64; nn];
                for i in 0..n {
                    for j in (i + 1)..n {
                        let mut acc = 0.0f64;
                        for k in 0..p {
                            acc += (x_host[i * p + k] - x_host[j * p + k]).abs();
                        }
                        d[i * n + j] = acc;
                        d[j * n + i] = acc;
                    }
                }
                d
            }
            Metric::Cosine => {
                let x_host: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
                cosine_distance_matrix(&x_host, n, p)
            }
        };

        // --- 2–3. Chain-Prim MST (sklearn `mst_linkage_core`) + stable sort. ---
        let mut mst = mst_linkage_core_dense(&dist, n);
        mst.sort_by(|a, b| a.2.total_cmp(&b.2)); // stable == np mergesort/scipy stable

        // --- 4. Metric-routed union-find labelling → children_. ---
        let children: Vec<[i64; 2]> = match self.metric {
            Metric::Euclidean | Metric::Manhattan => {
                // sklearn fast path: `single_linkage_label` — raw (find(u), find(v))
                // row order, the fresh-label UnionFind already ported for HDBSCAN.
                single_linkage::make_single_linkage(&mst, n)
                    .iter()
                    .map(|e| [e.left as i64, e.right as i64])
                    .collect()
            }
            Metric::Cosine => {
                // scipy path: same fresh-label union-find, rows as (min, max).
                scipy_single_linkage_children(&mst, n)
            }
        };

        // --- 5. `_hc_cut` → flat labels. ---
        let labels = hc_cut(self.n_clusters, &children, n);
        let labels_dev = DeviceArray::from_host(pool, &labels);

        Ok(AgglomerativeClustering {
            n_clusters: self.n_clusters,
            metric: self.metric,
            labels_: Some(labels_dev),
            children_: Some(children),
            n_leaves_: n,
            n_features_in_: p,
            _float: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<F> AgglomerativeClustering<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `labels_` (length `n`, `i32`). `Some` by
    /// construction on the `Fitted` state (D-03).
    pub fn labels(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<i32> {
        self.labels_
            .as_ref()
            .expect("labels_ is Some by construction on AgglomerativeClustering<F, Fitted>")
            .to_host(pool)
    }

    /// The fitted dendrogram `children_` (`(n-1) × 2`, host). Node ids `< n`
    /// are leaves; node `n + i` is the merge recorded in row `i`.
    pub fn children(&self) -> &[[i64; 2]] {
        self.children_
            .as_deref()
            .expect("children_ is Some by construction on AgglomerativeClustering<F, Fitted>")
    }

    /// The flat cluster count the dendrogram was cut into (`n_clusters_`).
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Number of dendrogram leaves (`n_leaves_` == `n_samples`).
    pub fn n_leaves(&self) -> usize {
        self.n_leaves_
    }

    /// Number of connected components (`n_connected_components_`). The
    /// unstructured (connectivity-free) fit always yields `1`.
    pub fn n_connected_components(&self) -> usize {
        1
    }

    /// Number of features seen at fit (`n_features_in_`).
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }
}

// ===========================================================================
// Host pipeline stages (line-exact sklearn/scipy ports)
// ===========================================================================

/// sklearn `_hierarchical_fast.pyx::mst_linkage_core` over a DENSE distance
/// matrix (Müllner MST-LINKAGE-CORE, chain-recording Prim). The recorded edge
/// source is the PREVIOUS chain node (`current_node`) — NOT the true nearest
/// in-tree node — which is exactly what the downstream labelling consumes.
fn mst_linkage_core_dense(dist: &[f64], n: usize) -> Vec<(usize, usize, f64)> {
    debug_assert_eq!(dist.len(), n * n);
    let mut in_tree = vec![false; n];
    let mut current_distances = vec![f64::INFINITY; n];
    let mut result = Vec::with_capacity(n - 1);
    let mut current_node = 0usize;
    for _ in 0..(n - 1) {
        in_tree[current_node] = true;
        let mut new_distance = f64::INFINITY;
        let mut new_node = 0usize;
        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            let left_value = dist[current_node * n + j];
            if left_value < current_distances[j] {
                current_distances[j] = left_value;
            }
            if current_distances[j] < new_distance {
                new_distance = current_distances[j];
                new_node = j;
            }
        }
        result.push((current_node, new_node, new_distance));
        current_node = new_node;
    }
    result
}

/// scipy `_hierarchy.pyx::label` over the stable-sorted MST edges: the same
/// fresh-label union-find as sklearn's `single_linkage_label`, but each row is
/// recorded as `(min(root_u, root_v), max(root_u, root_v))`. Verified exact vs
/// `scipy.cluster.hierarchy.linkage(method='single')` at design time.
fn scipy_single_linkage_children(sorted_mst: &[(usize, usize, f64)], n: usize) -> Vec<[i64; 2]> {
    let total = 2 * n - 1;
    let mut parent: Vec<usize> = (0..total).collect();
    let mut next_label = n;
    let mut children = Vec::with_capacity(n - 1);

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        while parent[x] != root {
            let nxt = parent[x];
            parent[x] = root;
            x = nxt;
        }
        root
    }

    for &(u, v, _w) in sorted_mst {
        let xr = find(&mut parent, u);
        let yr = find(&mut parent, v);
        let (lo, hi) = if xr < yr { (xr, yr) } else { (yr, xr) };
        children.push([lo as i64, hi as i64]);
        parent[xr] = next_label;
        parent[yr] = next_label;
        next_label += 1;
    }
    children
}

/// sklearn `_agglomerative.py::_hc_cut` — cut the dendrogram into `n_clusters`
/// flat labels. The label ids follow the FINAL HEAP ARRAY ORDER (sklearn
/// enumerates the raw heap list), so [`heappush`]/[`heappushpop`] are verbatim
/// Python-`heapq` ports — do NOT swap in `BinaryHeap` (its layout differs).
///
/// Caller guarantees `2 <= n_leaves` and `1 <= n_clusters <= n_leaves`.
fn hc_cut(n_clusters: usize, children: &[[i64; 2]], n_leaves: usize) -> Vec<i32> {
    debug_assert!(!children.is_empty());
    debug_assert!(n_clusters >= 1 && n_clusters <= n_leaves);

    // Negated ids: Python heapq is a min-heap, sklearn wants max-by-id.
    let root = children[children.len() - 1]
        .iter()
        .copied()
        .max()
        .expect("2 children")
        + 1;
    let mut nodes: Vec<i64> = vec![-root];
    for _ in 0..(n_clusters - 1) {
        // nodes[0] is the largest remaining node id — an internal node by
        // construction (leaves outrank nothing; see the loop-bound argument in
        // the module docs of the test file). Replace it with its two children.
        let idx = (-nodes[0]) as usize - n_leaves;
        let these_children = children[idx];
        heappush(&mut nodes, -these_children[0]);
        heappushpop(&mut nodes, -these_children[1]);
    }
    let mut label = vec![0i32; n_leaves];
    for (i, &node) in nodes.iter().enumerate() {
        for leaf in hc_get_descendent((-node) as usize, children, n_leaves) {
            label[leaf] = i as i32;
        }
    }
    label
}

/// sklearn `_hierarchical_fast.pyx::_hc_get_descendent` — collect the leaf ids
/// under `node` (iterative stack walk; visit order is irrelevant to the caller).
fn hc_get_descendent(node: usize, children: &[[i64; 2]], n_leaves: usize) -> Vec<usize> {
    if node < n_leaves {
        return vec![node];
    }
    let mut ind = vec![node];
    let mut descendent = Vec::new();
    while let Some(i) = ind.pop() {
        if i < n_leaves {
            descendent.push(i);
        } else {
            let c = children[i - n_leaves];
            ind.push(c[0] as usize);
            ind.push(c[1] as usize);
        }
    }
    descendent
}

// --- Verbatim Python-`heapq` ports (min-heap over i64). The heap ARRAY layout
//     is load-bearing for `hc_cut`'s label ids — keep sift semantics exact. ---

/// Python `heapq.heappush`: append then `_siftdown(heap, 0, len-1)`.
fn heappush(heap: &mut Vec<i64>, item: i64) {
    heap.push(item);
    let last = heap.len() - 1;
    siftdown(heap, 0, last);
}

/// Python `heapq.heappushpop`: if the root is smaller than `item`, swap and
/// `_siftup(heap, 0)`; return the displaced value (unused by `hc_cut`).
fn heappushpop(heap: &mut [i64], mut item: i64) -> i64 {
    if !heap.is_empty() && heap[0] < item {
        std::mem::swap(&mut item, &mut heap[0]);
        siftup(heap, 0);
    }
    item
}

/// Python `heapq._siftdown` (bubble `heap[pos]` toward the root).
fn siftdown(heap: &mut [i64], startpos: usize, mut pos: usize) {
    let newitem = heap[pos];
    while pos > startpos {
        let parentpos = (pos - 1) >> 1;
        let parent = heap[parentpos];
        if newitem < parent {
            heap[pos] = parent;
            pos = parentpos;
            continue;
        }
        break;
    }
    heap[pos] = newitem;
}

/// Python `heapq._siftup` (sink the root to a leaf, then `_siftdown` back).
fn siftup(heap: &mut [i64], mut pos: usize) {
    let endpos = heap.len();
    let startpos = pos;
    let newitem = heap[pos];
    let mut childpos = 2 * pos + 1;
    while childpos < endpos {
        let rightpos = childpos + 1;
        if rightpos < endpos && !(heap[childpos] < heap[rightpos]) {
            childpos = rightpos;
        }
        heap[pos] = heap[childpos];
        pos = childpos;
        childpos = 2 * pos + 1;
    }
    heap[pos] = newitem;
    siftdown(heap, startpos, pos);
}

/// Dense row-major `n×n` cosine distance matrix `1 − x̂·ŷ` (the HDBSCAN
/// `cosine_distance_matrix` helper, duplicated file-locally — the original is
/// private to `hdbscan.rs`). Rows L2-normalised once (zero row → all-zeros →
/// distance `1` everywhere); result clamped `>= 0`.
fn cosine_distance_matrix(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut xhat = vec![0.0f64; n * p];
    for i in 0..n {
        let row = &x[i * p..(i + 1) * p];
        let norm = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for k in 0..p {
            xhat[i * p + k] = row[k] * inv;
        }
    }
    // The cosine distance is symmetric, so compute the upper triangle once and
    // mirror it (the Manhattan path above uses the same pattern) — half the
    // O(n²p) dot-product work. The diagonal is `1 − ‖x̂‖²`: exactly `0` for a
    // unit-normalised row, or `1` for a zero row (inv = 0 → x̂ = 0 → dot = 0).
    let mut dist = vec![0.0f64; n * n];
    let clamp = |dot: f64| {
        let d = 1.0 - dot;
        if d > 0.0 {
            d
        } else {
            0.0
        }
    };
    for i in 0..n {
        // Diagonal: dot with self = ‖x̂_i‖² (1 for a non-zero row, 0 for zero).
        let self_dot: f64 = (0..p).map(|k| xhat[i * p + k] * xhat[i * p + k]).sum();
        dist[i * n + i] = clamp(self_dot);
        for j in (i + 1)..n {
            let mut dot = 0.0f64;
            for k in 0..p {
                dot += xhat[i * p + k] * xhat[j * p + k];
            }
            let d = clamp(dot);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}
