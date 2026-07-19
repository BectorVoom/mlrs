//! Path-dependent TreeSHAP (SHAP-01, Phase 21) — a host port of the Lundberg
//! `tree_path_dependent` algorithm (the `shap` package's C `tree_shap`
//! kernel), specialized to the mlrs COMPLETE-tree layout.
//!
//! Given per-node covers (`node_sample_weight` — the training-sample weight
//! reaching each node), the algorithm attributes a tree's prediction for a
//! query `x` across features by walking every root→leaf path once while
//! maintaining the "unique path" of (feature, zero_fraction, one_fraction,
//! pweight) elements: `extend_path` folds a new split into the permutation
//! weights, `unwind_path` removes a repeated feature before re-extending,
//! and `unwound_path_sum` produces each feature's Shapley weight at a leaf.
//! The three weight routines are line-for-line ports of the C reference
//! (`shap/cext/tree_shap.h`); exactness is gated ≤1e-5 against
//! `shap.TreeExplainer` on the SAME imported forest in
//! `crates/mlrs-algos/tests/tree_shap_test.rs` plus an EXACT
//! additive-efficiency check (`Σ_f φ + E[f] == prediction`).
//!
//! Complete-layout specifics: children of interior `i` are `2i+1`/`2i+2`,
//! `is_leaf != 0` marks leaves, thresholds are the import-time
//! `next_up`-bumped values so the HOT child test `x < threshold → left`
//! reproduces sklearn's `x <= t_orig → left` exactly. The per-tree expected
//! value is the cover-weighted leaf mean (== sklearn's root `value`).
//!
//! ## Two cover sources ([`forest_shap_values`] / [`native_forest_shap_values`])
//! Path-dependent SHAP needs a per-node cover the recursion can read; mlrs
//! offers it from two places:
//! - **[`ForestInference`](super::forest_inference::ForestInference)** (an
//!   imported sklearn forest) carries the EXACT `tree_.weighted_n_node_samples`
//!   from the source model — [`forest_shap_values`] uses it directly, so this
//!   path is ≤1e-5 GATED against a real `shap.TreeExplainer` on the same
//!   sklearn model (the primary oracle test).
//! - **A native mlrs `RandomForest{Classifier,Regressor}`** has no such array
//!   (the fit histogram/split kernels never track it) — [`native_forest_shap_values`]
//!   derives a self-consistent cover by re-routing a caller-supplied reference
//!   dataset through the ALREADY-FITTED tree host-side ([`compute_cover_from_data`]).
//!   This path has no external oracle (no library reproduces mlrs's own split
//!   policy), so it is gated by the EXACT additive-efficiency identity only.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::random_forest::RfModel;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::host_to_f64;

/// One element of the SHAP "unique path".
#[derive(Clone, Copy, Default)]
struct PathElement {
    /// The feature this path element splits on (`-1` for the root sentinel).
    feature_index: i64,
    /// Fraction of "zero" (background) paths flowing through this split.
    zero_fraction: f64,
    /// `1` if `x` follows this split, else `0` (possibly merged).
    one_fraction: f64,
    /// The permutation weight.
    pweight: f64,
}

/// One tree's complete-layout view (borrowed slices over the forest arrays).
pub(crate) struct TreeView<'a> {
    /// Per-node split feature (`total_nodes`; ignored on leaves).
    pub split_feature: &'a [u32],
    /// Per-node BUMPED threshold (`x < t → left` == sklearn `<=` original).
    pub threshold: &'a [f64],
    /// Per-node leaf flag.
    pub is_leaf: &'a [u32],
    /// Per-node values (`total_nodes × n_values`; meaningful on leaves).
    pub values: &'a [f64],
    /// Per-node cover (training-sample weight reaching the node).
    pub cover: &'a [f64],
    /// Values per node.
    pub n_values: usize,
}

impl TreeView<'_> {
    fn is_leaf_node(&self, i: usize) -> bool {
        self.is_leaf[i] != 0
    }
}

/// `extend_path` — fold a new split into the permutation weights
/// (line-for-line from `shap/cext/tree_shap.h`).
fn extend_path(
    path: &mut [PathElement],
    unique_depth: usize,
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: i64,
) {
    path[unique_depth] = PathElement {
        feature_index,
        zero_fraction,
        one_fraction,
        pweight: if unique_depth == 0 { 1.0 } else { 0.0 },
    };
    let ud = unique_depth as f64;
    for i in (0..unique_depth).rev() {
        path[i + 1].pweight += one_fraction * path[i].pweight * (i as f64 + 1.0) / (ud + 1.0);
        path[i].pweight = zero_fraction * path[i].pweight * (ud - i as f64) / (ud + 1.0);
    }
}

/// `unwind_path` — remove the split at `path_index` (inverse of extend).
fn unwind_path(path: &mut [PathElement], unique_depth: usize, path_index: usize) {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[unique_depth].pweight;
    let ud = unique_depth as f64;

    if one_fraction != 0.0 {
        for i in (0..unique_depth).rev() {
            let tmp = path[i].pweight;
            path[i].pweight = next_one_portion * (ud + 1.0) / ((i as f64 + 1.0) * one_fraction);
            next_one_portion =
                tmp - path[i].pweight * zero_fraction * (ud - i as f64) / (ud + 1.0);
        }
    } else {
        for i in (0..unique_depth).rev() {
            path[i].pweight = (path[i].pweight * (ud + 1.0)) / (zero_fraction * (ud - i as f64));
        }
    }
    for i in path_index..unique_depth {
        path[i].feature_index = path[i + 1].feature_index;
        path[i].zero_fraction = path[i + 1].zero_fraction;
        path[i].one_fraction = path[i + 1].one_fraction;
    }
}

/// `unwound_path_sum` — the total permutation weight of the path with the
/// element at `path_index` unwound (read-only).
fn unwound_path_sum(path: &[PathElement], unique_depth: usize, path_index: usize) -> f64 {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[unique_depth].pweight;
    let ud = unique_depth as f64;
    let mut total = 0.0;

    if one_fraction != 0.0 {
        for i in (0..unique_depth).rev() {
            let tmp = next_one_portion / ((i as f64 + 1.0) * one_fraction);
            total += tmp;
            next_one_portion = path[i].pweight - tmp * zero_fraction * (ud - i as f64);
        }
    } else {
        for i in (0..unique_depth).rev() {
            total += path[i].pweight / (zero_fraction * (ud - i as f64));
        }
    }
    total * (ud + 1.0)
}

/// The recursive walk (`tree_shap_recursive`, condition-free form). `phi` is
/// `n_features × n_values`, accumulated in place.
#[allow(clippy::too_many_arguments)]
fn recurse(
    tree: &TreeView<'_>,
    x: &[f64],
    phi: &mut [f64],
    node: usize,
    unique_depth: usize,
    parent_path: &[PathElement],
    parent_zero_fraction: f64,
    parent_one_fraction: f64,
    parent_feature_index: i64,
) {
    // Copy the parent path (the C code offsets into one big buffer; a clone
    // per level is equivalent and depth-bounded).
    let mut path: Vec<PathElement> = parent_path[..unique_depth + 1].to_vec();
    path.push(PathElement::default());
    extend_path(
        &mut path,
        unique_depth,
        parent_zero_fraction,
        parent_one_fraction,
        parent_feature_index,
    );

    if tree.is_leaf_node(node) {
        for i in 1..=unique_depth {
            let w = unwound_path_sum(&path, unique_depth, i);
            let el = &path[i];
            let phi_offset = el.feature_index as usize * tree.n_values;
            let values_offset = node * tree.n_values;
            for j in 0..tree.n_values {
                phi[phi_offset + j] +=
                    w * (el.one_fraction - el.zero_fraction) * tree.values[values_offset + j];
            }
        }
        return;
    }

    // Interior: hot child = where x goes (`x < bumped_t → left` == sklearn `<=`).
    let split_index = tree.split_feature[node] as usize;
    let (hot, cold) = if x[split_index] < tree.threshold[node] {
        (2 * node + 1, 2 * node + 2)
    } else {
        (2 * node + 2, 2 * node + 1)
    };
    // A reached interior node's cover is the sum of its children's covers, so
    // `w == 0` means NO training weight flows through this whole subtree. The
    // child zero-fractions `cover[child]/w` are then 0/0, and worse, feeding a
    // resulting 0 zero-fraction into the downstream permutation-weight
    // recursion (`unwind_path`/`unwound_path_sum` divide BY the zero-fraction)
    // produces NaN. Since the subtree carries no training weight, its leaves
    // contribute nothing to the cover-weighted attribution anyway (exactly as
    // `tree_expected_value` already EXCLUDES zero-cover leaves), so skip the
    // whole subtree rather than descend into the 0/0. Fitted sklearn forests
    // never hit this (interior `weighted_n_node_samples > 0`); an imported
    // forest with a zero-sample-weight branch can.
    let w = tree.cover[node];
    if w <= 0.0 {
        return;
    }
    let hot_zero_fraction = tree.cover[hot] / w;
    let cold_zero_fraction = tree.cover[cold] / w;
    let mut incoming_zero_fraction = 1.0;
    let mut incoming_one_fraction = 1.0;
    let mut unique_depth = unique_depth;

    // A repeated feature on the path is unwound and merged.
    let mut path_index = 0usize;
    while path_index <= unique_depth {
        if path[path_index].feature_index == split_index as i64 {
            break;
        }
        path_index += 1;
    }
    if path_index != unique_depth + 1 {
        incoming_zero_fraction = path[path_index].zero_fraction;
        incoming_one_fraction = path[path_index].one_fraction;
        unwind_path(&mut path, unique_depth, path_index);
        unique_depth -= 1;
    }

    recurse(
        tree,
        x,
        phi,
        hot,
        unique_depth + 1,
        &path,
        hot_zero_fraction * incoming_zero_fraction,
        incoming_one_fraction,
        split_index as i64,
    );
    recurse(
        tree,
        x,
        phi,
        cold,
        unique_depth + 1,
        &path,
        cold_zero_fraction * incoming_zero_fraction,
        0.0,
        split_index as i64,
    );
}

/// SHAP values for ONE query `x` over ONE tree: accumulates
/// `n_features × n_values` attributions into `phi` (caller-zeroed or
/// accumulated across trees).
pub(crate) fn tree_shap(tree: &TreeView<'_>, x: &[f64], phi: &mut [f64]) {
    let root_path: [PathElement; 1] = [PathElement::default()];
    // The C entry calls recursive(root, depth 0, zero=1, one=1, feature=-1);
    // the first extend then installs the -1 sentinel with weight 1.
    recurse(tree, x, phi, 0, 0, &root_path, 1.0, 1.0, -1);
}

/// The per-tree expected value: the cover-weighted leaf mean (== sklearn's
/// root `value` row for a fitted tree). `out` is `n_values`, accumulated.
pub(crate) fn tree_expected_value(tree: &TreeView<'_>, total_nodes: usize, out: &mut [f64]) {
    let root_cover = tree.cover[0];
    if root_cover <= 0.0 {
        return;
    }
    for node in 0..total_nodes {
        if tree.is_leaf_node(node) && tree.cover[node] > 0.0 && reachable(tree, node) {
            let wfrac = tree.cover[node] / root_cover;
            for j in 0..tree.n_values {
                out[j] += wfrac * tree.values[node * tree.n_values + j];
            }
        }
    }
}

/// Is `node` reachable from the root through interior nodes? (Unvisited
/// complete-layout slots are `is_leaf = 1` with zero cover — the zero-cover
/// guard in the caller already excludes them; this walk guards the corner
/// where an imported cover is legitimately 0 on a REAL leaf.)
fn reachable(tree: &TreeView<'_>, mut node: usize) -> bool {
    while node != 0 {
        let parent = (node - 1) / 2;
        if tree.is_leaf_node(parent) {
            return false; // slot under a leaf — an unvisited padding slot
        }
        node = parent;
    }
    true
}

/// Route every row of `x_host` (`n × d`) through each tree's complete-layout
/// arrays (root to leaf, incrementing every node ON the path — not just the
/// leaf), producing a self-consistent per-node cover
/// (`n_trees × total_nodes`). Terminates within `<= max_depth + 1` steps per
/// row/tree by construction: the routed path only ever visits REAL nodes
/// (interior `is_leaf == 0` correctly guides children; the walk stops the
/// instant it lands on a real leaf), so it never wanders into an unvisited
/// `is_leaf == 1` padding slot mid-route.
pub fn compute_cover_from_data(
    split_feature: &[u32],
    threshold: &[f64],
    is_leaf: &[u32],
    n_trees: usize,
    total_nodes: usize,
    x_host: &[f64],
    n: usize,
    d: usize,
) -> Vec<f64> {
    let mut cover = vec![0.0f64; n_trees * total_nodes];
    for row in 0..n {
        let xr = &x_host[row * d..(row + 1) * d];
        for t in 0..n_trees {
            let base = t * total_nodes;
            let mut cur = 0usize;
            loop {
                cover[base + cur] += 1.0;
                if is_leaf[base + cur] != 0 {
                    break;
                }
                let f = split_feature[base + cur] as usize;
                let thr = threshold[base + cur];
                cur = if xr[f] < thr { 2 * cur + 1 } else { 2 * cur + 2 };
            }
        }
    }
    cover
}

/// Shared driver: given complete-layout host arrays (already f64) + a cover
/// array, compute `(phi, expected_value)` over `n_query` rows. `phi` is
/// `n_query × n_features × n_values` row-major; `expected_value` is length
/// `n_values`. `Σ_f phi[q, f, :] + expected_value == prediction[q]` exactly
/// (the additive-efficiency identity every caller gates on).
#[allow(clippy::too_many_arguments)]
fn shap_from_arrays(
    split_feature: &[u32],
    threshold: &[f64],
    is_leaf: &[u32],
    values: &[f64],
    cover: &[f64],
    n_trees: usize,
    total_nodes: usize,
    n_values: usize,
    query_host: &[f64],
    n_query: usize,
    n_features: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut expected_value = vec![0.0f64; n_values];
    let mut phi = vec![0.0f64; n_query * n_features * n_values];
    for t in 0..n_trees {
        let base = t * total_nodes;
        let tv = TreeView {
            split_feature: &split_feature[base..base + total_nodes],
            threshold: &threshold[base..base + total_nodes],
            is_leaf: &is_leaf[base..base + total_nodes],
            values: &values[base * n_values..(base + total_nodes) * n_values],
            cover: &cover[base..base + total_nodes],
            n_values,
        };
        tree_expected_value(&tv, total_nodes, &mut expected_value);
        for q in 0..n_query {
            let x_row = &query_host[q * n_features..(q + 1) * n_features];
            let phi_row = &mut phi[q * n_features * n_values..(q + 1) * n_features * n_values];
            tree_shap(&tv, x_row, phi_row);
        }
    }
    // The forest predicts the MEAN over trees (rf_predict_proba/rf_predict_reg),
    // and Shapley attribution is linear in the model output — so the forest's
    // phi/expected_value are the per-tree MEAN, not the sum accumulated above.
    let inv_t = 1.0 / n_trees as f64;
    for v in expected_value.iter_mut() {
        *v *= inv_t;
    }
    for v in phi.iter_mut() {
        *v *= inv_t;
    }
    (phi, expected_value)
}

/// SHAP values over an IMPORTED forest's OWN cover (exact — see the module
/// docs' "two cover sources"). `cover` is `n_trees × total_nodes` (the
/// caller's [`ForestInference`](super::forest_inference::ForestInference)
/// import-time array). Returns `(phi, expected_value)`.
pub fn forest_shap_values<F>(
    pool: &BufferPool<ActiveRuntime>,
    model: &RfModel<F>,
    cover: &[f64],
    query_host: &[f64],
    n_query: usize,
) -> (Vec<f64>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let n_features = model.n_features();
    let n_trees = model.n_trees();
    let total_nodes = model.total_nodes();
    let n_values = model.n_values();
    let split_feature = model.split_feature_host(pool);
    let threshold: Vec<f64> = model.threshold_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let is_leaf = model.is_leaf_host(pool);
    let values: Vec<f64> = model.leaf_dist_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    shap_from_arrays(
        &split_feature, &threshold, &is_leaf, &values, cover, n_trees, total_nodes, n_values,
        query_host, n_query, n_features,
    )
}

/// SHAP values over a NATIVE mlrs forest, deriving cover by re-routing
/// `x_train_host` (`n_train × n_features` — typically the training set the
/// caller fit on) through the ALREADY-FITTED tree (self-consistency-gated
/// only — see the module docs). Returns `(phi, expected_value)`.
pub fn native_forest_shap_values<F>(
    pool: &BufferPool<ActiveRuntime>,
    model: &RfModel<F>,
    x_train_host: &[f64],
    n_train: usize,
    query_host: &[f64],
    n_query: usize,
) -> (Vec<f64>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let n_features = model.n_features();
    let n_trees = model.n_trees();
    let total_nodes = model.total_nodes();
    let n_values = model.n_values();
    let split_feature = model.split_feature_host(pool);
    let threshold: Vec<f64> = model.threshold_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let is_leaf = model.is_leaf_host(pool);
    let values: Vec<f64> = model.leaf_dist_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let cover = compute_cover_from_data(
        &split_feature, &threshold, &is_leaf, n_trees, total_nodes, x_train_host, n_train, n_features,
    );
    shap_from_arrays(
        &split_feature, &threshold, &is_leaf, &values, &cover, n_trees, total_nodes, n_values,
        query_host, n_query, n_features,
    )
}
