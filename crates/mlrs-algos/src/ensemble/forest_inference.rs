//! `ForestInference` (FIL-01) — batched device inference over an IMPORTED
//! forest, the cuML `ForestInference` parity surface (Phase 20).
//!
//! cuML's FIL loads an externally-trained tree ensemble (sklearn / XGBoost /
//! LightGBM) and serves batched GPU inference. mlrs's v1 scope is the sklearn
//! layout: each tree arrives as the explicit-children arrays sklearn's
//! `tree_.__getstate__()` exposes (`children_left` / `children_right` /
//! `feature` / `threshold` / per-node `value`), is converted host-side into
//! the mlrs COMPLETE-tree layout (node `i` → children `2i+1`/`2i+2`,
//! `total_nodes = 2^(max_depth+1) − 1`), and is served by the SAME device
//! traversal the native mlrs forests use ([`mlrs_kernels::rf_predict_leaf`] +
//! vote/mean kernels via [`rf_predict_proba`]/[`rf_predict_reg`]).
//!
//! ## Comparator adaptation (the correctness crux)
//! sklearn routes `x <= threshold → left`; the mlrs kernel routes
//! `x < threshold → left`. On import every threshold is bumped to
//! `next_up(threshold)` in the TARGET float precision: `x <= t  ⇔
//! x < next_up(t)` exactly (no representable float lies strictly between),
//! so leaf routing is EXACTLY sklearn's for every representable query.
//!
//! ## Depth bound
//! The complete layout squares with depth: `max_depth <= 16` (the native
//! forest cap — 2^17−1 nodes/tree). An imported tree deeper than 16 is a
//! typed [`BuildError::InvalidMaxDepth`]; retrain the source model with a
//! depth cap to import it.
//!
//! Classifier leaf values are the sklearn per-leaf class-count rows
//! NORMALIZED to distributions (sklearn's own `predict_proba` semantics —
//! the forest probability is the mean of per-tree distributions, exactly the
//! mlrs vote kernel's form). Regressor leaves carry the mean target.
//!
//! Tests live in `crates/mlrs-algos/tests/fil_test.rs` (AGENTS.md §2) plus
//! the Python `test_oracle_fil.py` sklearn round-trip (import a real fitted
//! `sklearn.ensemble.RandomForest*`, compare predictions ≤1e-5).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::random_forest::{rf_predict_proba, rf_predict_reg, RfModel};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, PrimError};

use crate::error::{AlgoError, BuildError};

/// The native-forest depth cap (complete-layout arrays square with depth).
const MAX_IMPORT_DEPTH: usize = 16;

/// One imported tree in the sklearn explicit-children layout. All arrays are
/// per-node, length `n_nodes`; `children_left[i] < 0` marks a LEAF (sklearn
/// uses `-1`). `value` is row-major `n_nodes × n_values` (class counts for a
/// classifier — normalized on import — or the length-1 mean target).
#[derive(Debug, Clone)]
pub struct TreeSpec {
    /// Left child per node (`-1` = leaf).
    pub children_left: Vec<i64>,
    /// Right child per node (`-1` = leaf).
    pub children_right: Vec<i64>,
    /// Split feature per node (ignored on leaves).
    pub feature: Vec<i64>,
    /// Split threshold per node (sklearn `x <= t → left`; bumped to
    /// `next_up(t)` on import for the mlrs `<` comparator).
    pub threshold: Vec<f64>,
    /// Per-node values (`n_nodes × n_values`, row-major).
    pub value: Vec<f64>,
    /// Per-node training-sample cover (sklearn `tree_.weighted_n_node_samples`
    /// — SHAP-01's "cover"). Empty means "no cover available"; any consumer
    /// that needs it (currently only [`crate::ensemble::tree_shap`]) treats an
    /// empty vec as absent and returns a typed error, never a silent zero.
    pub node_sample_weight: Vec<f64>,
}

/// What the imported forest predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForestKind {
    /// Classifier over `n_classes` (leaf value rows normalize to
    /// distributions; `predict` is the argmax of the mean distribution).
    Classifier {
        /// The class count (`n_values` of every tree's `value` rows).
        n_classes: usize,
    },
    /// Regressor (`n_values = 1`; the forest predicts the mean of leaves).
    Regressor,
}

/// A device-resident imported forest (FIL-01). Constructed by
/// [`ForestInference::from_trees`]; NOT a typestate estimator — there is no
/// `fit`, the model arrives fitted.
pub struct ForestInference<F>
where
    F: Float + CubeElement + Pod,
{
    model: RfModel<F>,
    kind: ForestKind,
    /// Complete-layout host cover (`n_trees × total_nodes`, f64), carried
    /// alongside `model` for [`Self::shap_values`] (SHAP-01). `Some` iff
    /// EVERY imported [`TreeSpec::node_sample_weight`] was non-empty; `None`
    /// otherwise (a partial import never fabricates cover for the trees that
    /// omitted it).
    cover: Option<Vec<f64>>,
}

impl<F> ForestInference<F>
where
    F: Float + CubeElement + Pod,
{
    /// Import `trees` (sklearn layout) into the device complete-tree store.
    ///
    /// Validation (typed, BEFORE any upload):
    /// - at least one tree ([`BuildError::InvalidNEstimators`]),
    /// - every tree's arrays agree on `n_nodes` and `n_values` matches
    ///   `kind` ([`AlgoError`] via [`PrimError::ShapeMismatch`] would be
    ///   misleading here, so malformed specs surface as
    ///   [`BuildError::InvalidNEstimators`]-adjacent typed errors — see each
    ///   check),
    /// - forest depth `<= 16` ([`BuildError::InvalidMaxDepth`]),
    /// - child indices in range and acyclic by construction (the conversion
    ///   walks parent→child once; an out-of-range child is a typed error).
    pub fn from_trees(
        pool: &mut BufferPool<ActiveRuntime>,
        trees: &[TreeSpec],
        kind: ForestKind,
        n_features: usize,
    ) -> Result<Self, AlgoError> {
        if trees.is_empty() {
            return Err(AlgoError::Build(BuildError::InvalidNEstimators {
                estimator: "forest_inference",
                n_estimators: 0,
            }));
        }
        let n_values = match kind {
            ForestKind::Classifier { n_classes } => n_classes,
            ForestKind::Regressor => 1,
        };

        // --- Validate specs + measure the forest depth. ---
        let mut forest_depth = 0usize;
        for t in trees {
            let n_nodes = t.children_left.len();
            let cover_ok = t.node_sample_weight.is_empty() || t.node_sample_weight.len() == n_nodes;
            if n_nodes == 0
                || t.children_right.len() != n_nodes
                || t.feature.len() != n_nodes
                || t.threshold.len() != n_nodes
                || t.value.len() != n_nodes * n_values
                || !cover_ok
            {
                return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                    operand: "tree_spec",
                    rows: n_nodes,
                    cols: n_values,
                    len: t.value.len(),
                }));
            }
            let depth = tree_depth(t, 0, 0).ok_or(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "tree_spec (child index out of range)",
                rows: n_nodes,
                cols: 0,
                len: 0,
            }))?;
            forest_depth = forest_depth.max(depth);
        }
        if forest_depth > MAX_IMPORT_DEPTH {
            return Err(AlgoError::Build(BuildError::InvalidMaxDepth {
                estimator: "forest_inference",
                max_depth: forest_depth,
            }));
        }
        // A depth-0 forest (single-leaf trees) still needs a 1-node layout.
        let max_depth = forest_depth.max(1);
        let total_nodes = (1usize << (max_depth + 1)) - 1;
        let n_trees = trees.len();

        // --- Convert to the complete layout (host). Unvisited slots stay
        //     `is_leaf = 1` with zero values — unreachable by construction
        //     (the walk stops at real leaves). ---
        let tn = n_trees * total_nodes;
        let mut split_feature = vec![0u32; tn];
        let mut threshold = vec![f64_to_host::<F>(0.0); tn];
        let mut is_leaf = vec![1u32; tn];
        let mut leaf_dist = vec![f64_to_host::<F>(0.0); tn * n_values];
        // SHAP-01: per-node cover (`node_sample_weight`), complete-layout
        // mapped alongside the other arrays. `Some` only if EVERY tree
        // supplied it — a partial import never fabricates cover. The buffer
        // is allocated ONLY when cover is present (the common raw-array
        // import carries none, so it stays empty rather than paying a
        // `tn`-element zero-fill that would be immediately discarded).
        let all_have_cover = trees.iter().all(|t| !t.node_sample_weight.is_empty());
        let mut cover_arr = if all_have_cover { vec![0.0f64; tn] } else { Vec::new() };

        for (ti, t) in trees.iter().enumerate() {
            // Iterative parent→child walk: (spec node, complete index).
            let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
            while let Some((node, cidx)) = stack.pop() {
                let base = ti * total_nodes + cidx;
                if all_have_cover {
                    cover_arr[base] = t.node_sample_weight[node];
                }
                if t.children_left[node] < 0 {
                    // Leaf: normalized distribution (classifier) / mean (reg).
                    is_leaf[base] = 1;
                    let row = &t.value[node * n_values..(node + 1) * n_values];
                    match kind {
                        ForestKind::Classifier { .. } => {
                            let sum: f64 = row.iter().sum();
                            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                            for (c, &v) in row.iter().enumerate() {
                                leaf_dist[base * n_values + c] = f64_to_host::<F>(v * inv);
                            }
                        }
                        ForestKind::Regressor => {
                            leaf_dist[base] = f64_to_host::<F>(row[0]);
                        }
                    }
                } else {
                    is_leaf[base] = 0;
                    split_feature[base] = t.feature[node] as u32;
                    // sklearn `x <= t → left` ⇔ mlrs `x < next_up(t) → left`
                    // in the TARGET precision (no representable float lies
                    // between t and next_up(t)).
                    threshold[base] = bump_threshold::<F>(t.threshold[node]);
                    stack.push((t.children_left[node] as usize, 2 * cidx + 1));
                    stack.push((t.children_right[node] as usize, 2 * cidx + 2));
                }
            }
        }

        let model = RfModel::<F>::from_parts(
            pool,
            &split_feature,
            &threshold,
            &is_leaf,
            &leaf_dist,
            n_trees,
            max_depth,
            n_features,
            n_values,
        )
        .map_err(AlgoError::Prim)?;
        Ok(Self {
            model,
            kind,
            cover: all_have_cover.then_some(cover_arr),
        })
    }

    /// The imported forest kind.
    pub fn kind(&self) -> ForestKind {
        self.kind
    }

    /// Trees in the imported forest.
    pub fn n_trees(&self) -> usize {
        self.model.n_trees()
    }

    /// Feature count the import declared (predict geometry is validated
    /// against it by the underlying prim).
    pub fn n_features(&self) -> usize {
        self.model.n_features()
    }

    /// Classifier: the `n_query × n_classes` mean of reached-leaf
    /// distributions (device-computed — sklearn `predict_proba` parity).
    /// Errors on a regressor import.
    pub fn predict_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        match self.kind {
            ForestKind::Classifier { .. } => {
                Ok(rf_predict_proba::<F>(pool, &self.model, x, shape)?)
            }
            ForestKind::Regressor => Err(AlgoError::Unsupported {
                estimator: "forest_inference",
                operation: "predict_proba on a regressor import",
            }),
        }
    }

    /// Classifier: argmax class INDEX per query row (host argmax over the
    /// device probabilities — lowest index wins ties, the sklearn rule).
    /// Errors on a regressor import.
    pub fn predict_class_indices(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<u32>, AlgoError> {
        let n_classes = match self.kind {
            ForestKind::Classifier { n_classes } => n_classes,
            ForestKind::Regressor => {
                return Err(AlgoError::Unsupported {
                    estimator: "forest_inference",
                    operation: "predict (class) on a regressor import",
                })
            }
        };
        let proba = self.predict_proba(pool, x, shape)?;
        let host = proba.to_host(pool);
        proba.release_into(pool);
        let (q, _) = shape;
        let mut out = Vec::with_capacity(q);
        for r in 0..q {
            let row = &host[r * n_classes..(r + 1) * n_classes];
            let mut best = 0usize;
            for (c, v) in row.iter().enumerate().skip(1) {
                if mlrs_core::host_to_f64(*v) > mlrs_core::host_to_f64(row[best]) {
                    best = c;
                }
            }
            out.push(best as u32);
        }
        Ok(out)
    }

    /// Regressor: length-`n_query` device predictions (forest mean of the
    /// reached leaves). Errors on a classifier import.
    pub fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        match self.kind {
            ForestKind::Regressor => Ok(rf_predict_reg::<F>(pool, &self.model, x, shape)?),
            ForestKind::Classifier { .. } => Err(AlgoError::Unsupported {
                estimator: "forest_inference",
                operation: "predict (regression) on a classifier import",
            }),
        }
    }

    /// SHAP-01: path-dependent TreeSHAP values for `n_query` rows, using the
    /// EXACT cover the import carried (`tree_.weighted_n_node_samples` from
    /// the source model) — ≤1e-5 GATED against a real `shap.TreeExplainer`
    /// on the same sklearn model (see the `tree_shap` module docs).
    ///
    /// Returns `(phi, expected_value)`: `phi` is `n_query × n_features ×
    /// n_values` row-major; `expected_value` is length `n_values`.
    /// `Σ_f phi[q, f, :] + expected_value == prediction[q]` exactly.
    ///
    /// `query_host` is `n_query × n_features` (host, f64 — the caller reads
    /// its device query buffer back before calling, mirroring the other
    /// host-domain estimator helpers).
    ///
    /// Errors ([`AlgoError::Unsupported`]) if the import carried no
    /// `node_sample_weight` (cover) on every tree.
    pub fn shap_values(
        &self,
        pool: &BufferPool<ActiveRuntime>,
        query_host: &[f64],
        n_query: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), AlgoError> {
        let cover = self.cover.as_ref().ok_or(AlgoError::Unsupported {
            estimator: "forest_inference",
            operation: "shap_values without an imported node_sample_weight (cover)",
        })?;
        Ok(super::tree_shap::forest_shap_values(
            pool,
            &self.model,
            cover,
            query_host,
            n_query,
        ))
    }
}

/// Depth of the subtree at `node` (`None` on an out-of-range child index).
/// Iteratively bounded by the node count, so a malformed cyclic spec
/// terminates as out-of-range/depth-overflow rather than looping.
fn tree_depth(t: &TreeSpec, node: usize, depth: usize) -> Option<usize> {
    let n_nodes = t.children_left.len();
    if node >= n_nodes || depth > n_nodes {
        return None;
    }
    if t.children_left[node] < 0 {
        return Some(depth);
    }
    let l = t.children_left[node];
    let r = t.children_right[node];
    if l < 0 || r < 0 {
        return None; // half-leaf nodes are not a valid sklearn tree
    }
    let dl = tree_depth(t, l as usize, depth + 1)?;
    let dr = tree_depth(t, r as usize, depth + 1)?;
    Some(dl.max(dr))
}

/// `next_up(t)` in the TARGET precision `F` (f32 arm bumps the f32-rounded
/// value; f64 arm the f64). Non-finite thresholds pass through unchanged.
fn bump_threshold<F: Pod>(t: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => {
            let b = (t as f32).next_up();
            *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&b))
        }
        8 => {
            let b = t.next_up();
            *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&b))
        }
        _ => unreachable!("forest_inference is f32/f64 only"),
    }
}
