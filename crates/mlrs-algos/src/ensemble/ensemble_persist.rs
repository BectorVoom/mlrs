//! `ensemble_persist` (ENSEMBLE-PERSIST, prototype) — the `mlrs-ensemble` half
//! of the mlrs model file format: the container discriminator, the aliases the
//! four tree ensembles write and read through, and the node-table layout they
//! share.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! 8-byte-aligned zero-copy read path, the typed tensor accessors, the error
//! type and the `SaveModel` / `LoadModel` surface — lives in [`crate::persist`]
//! and is estimator-agnostic. Read that module's docs for the four decisions
//! that make the files small and the loads fast.
//!
//! ## The trees are COMPLETE, and that is why this family is simple
//!
//! A tree ensemble is the case where a model format usually gets complicated:
//! trees are ragged, so the file needs an offset table, a child-pointer scheme,
//! or a nested encoding. mlrs sidesteps all of it. Both
//! [`RfModel`](mlrs_backend::prims::random_forest::RfModel) and
//! [`HgbModel`](mlrs_backend::prims::hist_gradient_boosting::HgbModel) hold
//! every tree in a COMPLETE layout — `total_nodes = 2^(max_depth+1) − 1` slots
//! per tree, children at `2i+1` / `2i+2`, `is_leaf` marking where a walk stops
//! — because that is what makes the traversal kernel branch-free.
//!
//! So the file is four flat arrays of identical length and nothing else:
//!
//! | name | dtype | shape |
//! |---|---|---|
//! | `split_feature` | `U64` | `[n_trees, total_nodes]` |
//! | `threshold` | `F` (`F32`/`F64`) | `[n_trees, total_nodes]` |
//! | `is_leaf` | `U64` | `[n_trees, total_nodes]` |
//! | `leaf_dist` / `leaf_value` | `F` | `[n_trees, total_nodes, n_values]` / `[n_trees, total_nodes]` |
//!
//! `n_trees` and `total_nodes` come off the shapes, and `max_depth` is recovered
//! from `total_nodes` by [`depth_from_total_nodes`] — which also REJECTS a node
//! count that is not `2^(d+1) − 1`, since the traversal's child arithmetic is
//! only in bounds for a complete layout.
//!
//! ## The size trade this makes, stated plainly
//!
//! A complete layout stores every slot a tree of that depth COULD have, so a
//! shallow or unbalanced forest wastes the difference — at `max_depth = 12`
//! that is 8191 slots per tree whether or not the fit used them. A ragged
//! encoding would be smaller, sometimes by a lot.
//!
//! It is not done, for the reason this format makes every other layout
//! decision: the file follows the COMPUTE path. mlrs's traversal indexes
//! `2i+1`/`2i+2` directly with no child pointers to load, so a ragged file would
//! have to be expanded into exactly this layout on every load — an
//! `O(n_trees · 2^max_depth)` scatter — to be usable at all. The waste is real
//! and bounded by a hyperparameter the caller chose; the expansion would be
//! unbounded work on the read path.
//!
//! `split_feature` and `is_leaf` are `u32` in memory and `U64` on disk, which
//! doubles the two integer arrays. `TensorRef` has no `u32` constructor, and the
//! widening keeps the file readable on a host whose forests outgrow a `u32`
//! without a format change.
//!
//! Tests live in `crates/mlrs-algos/tests/ensemble_persist_test.rs`
//! (AGENTS.md §2).

use bytemuck::Pod;

// The container is shared with every other family; only the discriminator and
// the node-table helpers below are local. Re-exported (not just imported) so
// `ensemble::ensemble_persist::{AlignedBytes, SaveModel, …}` is the single
// import path for an ensemble's `save`/`load`.
pub use crate::persist::{
    as_f64, as_floats, as_i64, as_usizes, expect_len, shape_1d, shape_2d, AlignedBytes, Container,
    LoadModel, ModelFile, ModelWriter, PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

/// The ensemble container discriminator (`format = "mlrs-ensemble"`).
pub struct EnsembleContainer;

impl Container for EnsembleContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`.
pub const FORMAT_ID: &str = "mlrs-ensemble";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`EnsembleFile::parse`] rejects anything else outright.
pub const FORMAT_VERSION: &str = "1";

/// The tensor holding each node's split feature index.
pub const SPLIT_FEATURE_NAME: &str = "split_feature";
/// The tensor holding each node's split threshold.
pub const THRESHOLD_NAME: &str = "threshold";
/// The tensor holding each node's leaf flag (non-zero stops the walk).
pub const IS_LEAF_NAME: &str = "is_leaf";
/// The tensor holding a random forest's per-leaf class distribution or
/// regression value, `[n_trees, total_nodes, n_values]`.
pub const LEAF_DIST_NAME: &str = "leaf_dist";
/// The tensor holding a booster's per-leaf raw-score contribution,
/// `[n_trees, total_nodes]`.
pub const LEAF_VALUE_NAME: &str = "leaf_value";
/// The tensor holding a random forest's per-node weighted impurity decrease.
pub const NODE_DECREASE_NAME: &str = "node_decrease";
/// The tensor holding a booster's per-column baseline raw score, `[k]` — the
/// constant every tree's contribution is added to.
pub const BASELINE_NAME: &str = "baseline";

/// The ensemble writer: [`ModelWriter`] pinned to the `mlrs-ensemble`
/// container.
pub type EnsembleWriter<'a> = ModelWriter<'a, EnsembleContainer>;

/// The ensemble reader: [`ModelFile`] pinned to the `mlrs-ensemble` container.
pub type EnsembleFile<'a> = ModelFile<'a, EnsembleContainer>;

/// Widen a `u32` node array to the `u64` the file stores.
///
/// A copy, and an unavoidable one — [`TensorRef`] has no `u32` constructor.
/// Callers must bind the result before constructing the writer, since the writer
/// borrows it.
pub fn widen_nodes(values: &[u32]) -> Vec<u64> {
    values.iter().map(|&v| u64::from(v)).collect()
}

/// Recover `max_depth` from a complete tree's node count, rejecting any count
/// that is not `2^(d+1) − 1`.
///
/// This is the load-bearing check of the whole family. The traversal kernel
/// indexes children at `2i+1` and `2i+2` with no bound of its own beyond the
/// depth counter, so it is in range ONLY for a complete layout. A `total_nodes`
/// of, say, 100 is a perfectly plausible-looking header value and would make
/// every walk that reached the last level read past the end of the node table.
///
/// Storing `max_depth` as a scalar instead would not help: it would be a second
/// copy of the same fact, and a hand-edited header could make the two disagree.
/// Deriving it and rejecting the impossible is the encoding that cannot be
/// internally inconsistent.
pub fn depth_from_total_nodes(total_nodes: usize) -> Result<usize, PersistError> {
    // `2^(d+1) − 1` is exactly the numbers whose binary form is all ones.
    if total_nodes == 0 || !(total_nodes + 1).is_power_of_two() {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "a complete tree has 2^(max_depth+1) − 1 nodes, but the node tables \
                 declare {total_nodes}"
            ),
        });
    }
    Ok((total_nodes + 1).trailing_zeros() as usize - 1)
}

/// Stage the three node tables every tree ensemble shares.
///
/// All three are the same length by construction, and that is checked HERE
/// rather than left to the reader: a mismatch on the save side is a bug in the
/// estimator, and catching it before the bytes reach disk is the difference
/// between a failed save and a corrupt file.
pub fn write_node_tables<'a, F: Pod>(
    w: &mut EnsembleWriter<'a>,
    split_feature: &'a [u64],
    threshold: &'a [F],
    is_leaf: &'a [u64],
    n_trees: usize,
    total_nodes: usize,
) -> Result<(), PersistError> {
    if n_trees == 0 || total_nodes == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "the node tables would be [{n_trees}, {total_nodes}]; a fitted ensemble \
                 has at least one tree with at least one node"
            ),
        });
    }
    let expected = n_trees * total_nodes;
    expect_len(SPLIT_FEATURE_NAME, split_feature.len(), expected, "nodes")?;
    expect_len(THRESHOLD_NAME, threshold.len(), expected, "nodes")?;
    expect_len(IS_LEAF_NAME, is_leaf.len(), expected, "nodes")?;

    let shape = vec![n_trees, total_nodes];
    w.tensor(
        SPLIT_FEATURE_NAME,
        TensorRef::u64s(split_feature, shape.clone())?,
    );
    w.tensor(THRESHOLD_NAME, TensorRef::floats(threshold, shape.clone())?);
    w.tensor(IS_LEAF_NAME, TensorRef::u64s(is_leaf, shape)?);
    Ok(())
}

/// The three node tables recovered from a file, with the geometry they imply.
pub struct NodeTables<F> {
    /// Per-node split feature index, narrowed back to `u32`.
    pub split_feature: Vec<u32>,
    /// Per-node split threshold.
    pub threshold: Vec<F>,
    /// Per-node leaf flag, narrowed back to `u32`.
    pub is_leaf: Vec<u32>,
    /// Tree count, from the tables' row extent.
    pub n_trees: usize,
    /// Slots per tree, from the tables' column extent.
    pub total_nodes: usize,
    /// Recovered from `total_nodes` by [`depth_from_total_nodes`].
    pub max_depth: usize,
}

/// Read back what [`write_node_tables`] staged, validating the complete-layout
/// invariant and every cross-table extent.
///
/// The file is UNTRUSTED input (T-04-01-01), and this is the family where that
/// matters most: the traversal kernel walks `2i+1`/`2i+2` with no bound beyond
/// the depth counter, and it reads `split_feature[node]` to index the FEATURE
/// axis of the query matrix. So both the node count's completeness and every
/// split feature's range have to be established before the model exists — which
/// is why `n_features` is a parameter here rather than something the caller
/// checks afterwards.
pub fn read_node_tables<F: Pod>(
    file: &EnsembleFile<'_>,
    n_features: usize,
) -> Result<NodeTables<F>, PersistError> {
    let split_v = file.tensor(SPLIT_FEATURE_NAME)?;
    let (n_trees, total_nodes) = shape_2d(&split_v, SPLIT_FEATURE_NAME)?;
    if n_trees == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{SPLIT_FEATURE_NAME}' declares 0 trees; a fitted ensemble has at \
                 least one"
            ),
        });
    }
    let max_depth = depth_from_total_nodes(total_nodes)?;

    let threshold_v = file.tensor(THRESHOLD_NAME)?;
    let (t_rows, t_cols) = shape_2d(&threshold_v, THRESHOLD_NAME)?;
    if (t_rows, t_cols) != (n_trees, total_nodes) {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{THRESHOLD_NAME}' declares shape [{t_rows}, {t_cols}], but \
                 '{SPLIT_FEATURE_NAME}' implies [{n_trees}, {total_nodes}]"
            ),
        });
    }
    let leaf_v = file.tensor(IS_LEAF_NAME)?;
    let (l_rows, l_cols) = shape_2d(&leaf_v, IS_LEAF_NAME)?;
    if (l_rows, l_cols) != (n_trees, total_nodes) {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{IS_LEAF_NAME}' declares shape [{l_rows}, {l_cols}], but \
                 '{SPLIT_FEATURE_NAME}' implies [{n_trees}, {total_nodes}]"
            ),
        });
    }

    let is_leaf: Vec<u32> = as_usizes(&leaf_v, IS_LEAF_NAME)?
        .into_iter()
        .map(|v| {
            u32::try_from(v).map_err(|_| PersistError::InconsistentGeometry {
                reason: format!(
                    "tensor '{IS_LEAF_NAME}' holds {v}, which does not fit the u32 the \
                     traversal kernel consumes"
                ),
            })
        })
        .collect::<Result<_, _>>()?;

    // Every INTERNAL node's split feature indexes the query matrix's FEATURE
    // axis, so its range is a cross-model invariant rather than a formality: an
    // out-of-range index reads past the end of a query row on the first
    // prediction.
    //
    // The check is conditioned on `is_leaf` because that is exactly the
    // condition the traversal kernel applies —
    // [`rf_predict_leaf`](mlrs_kernels::rf_predict_leaf) reads `split_feature`
    // only inside its `is_leaf == 0` branch. A LEAF's slot is never
    // dereferenced, and the fit leaves a sentinel there rather than a valid
    // index, so validating it unconditionally would reject every real forest.
    // Checking exactly what the kernel reads is both the correct guard and the
    // only one that is not vacuous.
    let split_feature: Vec<u32> = as_usizes(&split_v, SPLIT_FEATURE_NAME)?
        .into_iter()
        .zip(is_leaf.iter())
        .map(|(v, &leaf)| {
            if leaf == 0 && v >= n_features {
                return Err(PersistError::InconsistentGeometry {
                    reason: format!(
                        "tensor '{SPLIT_FEATURE_NAME}' holds the feature index {v} on an \
                         internal node, out of range for a model with {n_features} features"
                    ),
                });
            }
            // A leaf's sentinel is deliberately allowed through, but it still
            // has to FIT the `u32` the kernel's array is typed as.
            u32::try_from(v).map_err(|_| PersistError::InconsistentGeometry {
                reason: format!(
                    "tensor '{SPLIT_FEATURE_NAME}' holds {v}, which does not fit the u32 the \
                     traversal kernel consumes"
                ),
            })
        })
        .collect::<Result<_, _>>()?;

    Ok(NodeTables {
        split_feature,
        threshold: as_floats::<F>(&threshold_v, THRESHOLD_NAME)?.into_owned(),
        is_leaf,
        n_trees,
        total_nodes,
        max_depth,
    })
}

/// Read a per-node float table (`leaf_dist`, `leaf_value`, `node_decrease`) and
/// check its length against the node geometry.
///
/// `values_per_node` is 1 for every table but a random forest's `leaf_dist`,
/// which holds `n_values` entries per node (the class distribution, or the
/// single regression value).
pub fn read_leaf_table<F: Pod>(
    file: &EnsembleFile<'_>,
    name: &'static str,
    n_trees: usize,
    total_nodes: usize,
    values_per_node: usize,
) -> Result<Vec<F>, PersistError> {
    let view = file.tensor(name)?;
    let len: usize = view.shape().iter().product();
    expect_len(
        name,
        len,
        n_trees * total_nodes * values_per_node,
        "entries",
    )?;
    Ok(as_floats::<F>(&view, name)?.into_owned())
}
