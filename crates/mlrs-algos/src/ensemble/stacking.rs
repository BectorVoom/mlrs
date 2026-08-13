//! `stacking` — the structural core of the stacked-generalization meta-estimator
//! (STACK-01), shared by `mlrs.ensemble.StackingRegressor`.
//!
//! Stacking is a *composition*: the arithmetic that matters (each base
//! estimator's `fit`/`predict`, the final estimator's `fit`) already runs on the
//! device inside the composed estimators. What the meta-estimator itself owns is
//! **structure** — which entries are dropped, how the per-estimator prediction
//! blocks lay out into one meta-feature matrix, what those meta columns are
//! called, and whether `cv` selected the cross-fold or the `"prefit"` route.
//! That structural half is exactly what lives here, mirroring the split already
//! established by [`crate::model_selection`]: index/layout logic in Rust, the
//! array gathers and the estimator calls in Python.
//!
//! ## The meta-feature copy itself (STACK-META-01)
//!
//! [`concatenate_predictions`] performs the copy on the host, and
//! [`concatenate_meta`] dispatches between it and the CubeCL scatter in
//! [`mlrs_backend::prims::stacking_meta`]. Neither is the DEFAULT: the shim
//! still reaches for `np.hstack` unless `MLRS_STACK_META_ENGINE` says otherwise,
//! because the operation carries no arithmetic and both Rust arms therefore
//! start an Arrow capsule round-trip in debt. The ladder that settles it is in
//! `docs/stacking.md`; [`MetaLayout`] is what all three arms agree on.
//!
//! ## sklearn parity
//!
//! Every rule reproduced here is a rule the caller can observe as an exception
//! message or a `get_feature_names_out()` string, so each is matched verbatim
//! against `sklearn.ensemble._stacking` / `sklearn.utils.metaestimators`:
//!
//! - [`validate_names`] — `_BaseComposition._validate_names`
//! - [`kept_indices`] — the structural half of
//!   `_BaseHeterogeneousEnsemble._validate_estimators`
//! - [`MetaLayout`] — `_BaseStacking._concatenate_predictions`'s
//!   `_n_feature_outs` + column order
//! - [`meta_feature_names`] — `_BaseStacking.get_feature_names_out`
//! - [`CvRoute`] — the `cv == "prefit"` branch of `_BaseStacking.fit`
//!
//! Tests live in `crates/mlrs-algos/tests/stacking_test.rs` (AGENTS.md §2 — never
//! an in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};
use thiserror::Error;

use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::stacking_meta::{concat_meta_device, MetaEngine};
use mlrs_backend::runtime::ActiveRuntime;

/// The string sentinel a caller puts in `estimators` to disable an entry
/// (`set_params(lr="drop")`). sklearn compares against this literal, and so
/// does every rule in this module.
pub const DROP: &str = "drop";

/// The `cv="prefit"` sentinel. sklearn declares it as the sole member of
/// `StrOptions({"prefit"})` on the `cv` constraint.
pub const PREFIT: &str = "prefit";

/// A stacking-composition failure.
///
/// Single-variant on purpose: every condition sklearn's stacking layer rejects
/// — a duplicate name, a name colliding with a constructor argument, a name
/// containing `__`, an all-`"drop"` list, a mis-shaped prediction block — is a
/// plain `ValueError` there. Collapsing them into one variant keeps the Python
/// bridge from having to invent an exception class sklearn never raises.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StackingError {
    /// Maps to `ValueError` at the Python boundary, message verbatim.
    #[error("{0}")]
    Value(String),
}

type Result<T> = std::result::Result<T, StackingError>;

fn value_err(msg: impl Into<String>) -> StackingError {
    StackingError::Value(msg.into())
}

/// Render a name list the way Python's `repr(list_of_str)` does, since these
/// strings land inside error messages that are compared against sklearn's.
///
/// Python quotes with `'` and escapes an embedded `'` as `\'`; a backslash
/// doubles. Nothing else in a legal estimator name needs escaping.
fn py_repr_list(names: &[String]) -> String {
    let inner: Vec<String> = names
        .iter()
        .map(|n| format!("'{}'", n.replace('\\', "\\\\").replace('\'', "\\'")))
        .collect();
    format!("[{}]", inner.join(", "))
}

/// sklearn `_BaseComposition._validate_names`: reject duplicate names, names
/// that collide with the meta-estimator's own constructor arguments, and names
/// containing `__`.
///
/// `ctor_params` is the caller's `get_params(deep=False)` key set — passed in
/// rather than hard-coded because it differs between `StackingRegressor` and
/// any future sibling, and because the shim is the only layer that can read it.
///
/// The three checks run in sklearn's order, which is observable: a list that is
/// *both* duplicated and colliding reports the duplication.
pub fn validate_names(names: &[String], ctor_params: &[String]) -> Result<()> {
    let mut seen = std::collections::HashSet::with_capacity(names.len());
    let unique = names.iter().all(|n| seen.insert(n.as_str()));
    if !unique {
        return Err(value_err(format!(
            "Names provided are not unique: {}",
            py_repr_list(names)
        )));
    }

    let ctor: std::collections::HashSet<&str> = ctor_params.iter().map(String::as_str).collect();
    let mut invalid: Vec<String> = names
        .iter()
        .filter(|n| ctor.contains(n.as_str()))
        .cloned()
        .collect();
    if !invalid.is_empty() {
        // sklearn sorts this one (`sorted(invalid_names)`) and not the next.
        invalid.sort();
        return Err(value_err(format!(
            "Estimator names conflict with constructor arguments: {}",
            py_repr_list(&invalid)
        )));
    }

    let dunder: Vec<String> = names.iter().filter(|n| n.contains("__")).cloned().collect();
    if !dunder.is_empty() {
        return Err(value_err(format!(
            "Estimator names must not contain __: got {}",
            py_repr_list(&dunder)
        )));
    }

    Ok(())
}

/// The positions of the entries that are NOT `"drop"`, in list order.
///
/// `is_drop[i]` is the shim's answer to `estimators[i][1] == "drop"` — the
/// comparison itself has to happen in Python (the value is an arbitrary
/// estimator object, and `==` on one may be overloaded), but the *consequences*
/// of it are all here.
///
/// Errors when the list is empty or every entry is dropped, with sklearn's own
/// two messages.
pub fn kept_indices(is_drop: &[bool]) -> Result<Vec<usize>> {
    if is_drop.is_empty() {
        return Err(value_err(
            "Invalid 'estimators' attribute, 'estimators' should be a \
             non-empty list of (string, estimator) tuples.",
        ));
    }
    let kept: Vec<usize> = is_drop
        .iter()
        .enumerate()
        .filter(|(_, &d)| !d)
        .map(|(i, _)| i)
        .collect();
    if kept.is_empty() {
        return Err(value_err(
            "All estimators are dropped. At least one is required to be an estimator.",
        ));
    }
    Ok(kept)
}

/// Which route `fit` takes, decided by the `cv` argument's *string* value.
///
/// The numeric / splitter / iterable forms of `cv` are all `CrossVal` here —
/// resolving them is [`crate::model_selection::split::check_cv`]'s job, not
/// this module's. The only thing stacking adds on top of `check_cv` is the
/// `"prefit"` sentinel, which sklearn handles *before* `check_cv` is ever
/// reached and which changes the semantics completely: base estimators are not
/// cloned, not refitted, and their predictions on the FULL training set (not
/// out-of-fold predictions) become the meta features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvRoute {
    /// `cv="prefit"` — assume the given estimators are already fitted.
    Prefit,
    /// Every other `cv` — build out-of-fold predictions with `cross_val_predict`.
    CrossVal,
}

/// Classify a `cv` argument that arrived as a *string*.
///
/// `"prefit"` is the only accepted string; anything else is the
/// `InvalidParameterError` sklearn's `StrOptions({"prefit"})` constraint
/// produces. A non-string `cv` never reaches here — the shim keeps it in
/// Python and hands it to `check_cv`.
pub fn cv_route_from_str(cv: &str) -> Result<CvRoute> {
    if cv == PREFIT {
        Ok(CvRoute::Prefit)
    } else {
        Err(value_err(format!(
            "The 'cv' parameter of StackingRegressor must be an int in the range \
             [2, inf), an object implementing 'split' and 'get_n_splits', an \
             iterable or None or a str among {{'prefit'}}. Got '{cv}' instead."
        )))
    }
}

/// The column layout of the meta-feature matrix handed to `final_estimator_`.
///
/// Reproduces the order `_BaseStacking._concatenate_predictions` builds: every
/// kept estimator's prediction block in `estimators` order, then — when
/// `passthrough` is set — the original `X` columns appended last.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaLayout {
    /// sklearn's `_n_feature_outs`: columns contributed by each kept
    /// estimator, in kept order. A regressor's 1-D `predict` output counts as
    /// one column (the shim reshapes `(n,)` to `(n, 1)`, as sklearn does).
    pub n_feature_outs: Vec<usize>,
    /// Start column of each kept estimator's block in the meta matrix.
    pub offsets: Vec<usize>,
    /// Total columns contributed by the estimators alone — where the
    /// passthrough block begins.
    pub n_meta: usize,
    /// Total meta-matrix width: `n_meta`, plus `n_features` when `passthrough`.
    pub width: usize,
}

/// Build the [`MetaLayout`] for `pred_cols` prediction blocks.
///
/// `pred_cols[i]` is the column count of the `i`-th KEPT estimator's block
/// ([`kept_indices`] order). `n_features` is `X`'s width, consulted only when
/// `passthrough` is true.
///
/// A zero-column block is rejected: it cannot come from a well-behaved
/// estimator, and letting it through would silently shift every downstream
/// `get_feature_names_out` name.
pub fn meta_layout(
    pred_cols: &[usize],
    n_features: usize,
    passthrough: bool,
) -> Result<MetaLayout> {
    if pred_cols.is_empty() {
        return Err(value_err(
            "All estimators are dropped. At least one is required to be an estimator.",
        ));
    }
    let mut offsets = Vec::with_capacity(pred_cols.len());
    let mut n_meta = 0usize;
    for (i, &c) in pred_cols.iter().enumerate() {
        if c == 0 {
            return Err(value_err(format!(
                "estimator at position {i} produced a prediction block with 0 columns"
            )));
        }
        offsets.push(n_meta);
        n_meta += c;
    }
    let width = if passthrough {
        n_meta + n_features
    } else {
        n_meta
    };
    Ok(MetaLayout {
        n_feature_outs: pred_cols.to_vec(),
        offsets,
        n_meta,
        width,
    })
}

/// sklearn `_BaseStacking.get_feature_names_out`.
///
/// `class_name` is the lower-cased meta-estimator class name
/// (`"stackingregressor"`); `kept_names` are the non-dropped entry names in
/// list order; `n_feature_outs` comes from [`MetaLayout`]. `input_features` is
/// `Some` exactly when `passthrough` is set — sklearn only generates/validates
/// input names on that branch.
///
/// A single-column block is named `"{class}_{name}"`; a multi-column one is
/// suffixed with the within-block index, `"{class}_{name}{i}"`. Note there is
/// no separator before the index — `stackingclassifier_lr0`, not `..._lr_0`.
pub fn meta_feature_names(
    class_name: &str,
    kept_names: &[String],
    n_feature_outs: &[usize],
    input_features: Option<&[String]>,
) -> Result<Vec<String>> {
    if kept_names.len() != n_feature_outs.len() {
        return Err(value_err(format!(
            "have {} kept estimator names but {} feature-out counts",
            kept_names.len(),
            n_feature_outs.len()
        )));
    }
    let mut out = Vec::new();
    for (name, &n_out) in kept_names.iter().zip(n_feature_outs) {
        if n_out == 1 {
            out.push(format!("{class_name}_{name}"));
        } else {
            for i in 0..n_out {
                out.push(format!("{class_name}_{name}{i}"));
            }
        }
    }
    if let Some(inputs) = input_features {
        out.extend(inputs.iter().cloned());
    }
    Ok(out)
}

/// Copy the per-estimator prediction blocks (and, when `passthrough`, `X`) into
/// one row-major meta-feature matrix.
///
/// This is the Rust-native path used by the Rust test suite and by any
/// Rust-side caller composing stacked estimators without Python; the Python
/// shim performs the identical copy with `np.hstack`, driven by the same
/// [`MetaLayout`] (see the module docs for why it is not routed back through
/// the FFI). Keeping both means the layout contract is executable, not just
/// described.
///
/// `blocks[i]` is `(data, n_cols)` with `data.len() == n_rows * n_cols`, in
/// [`kept_indices`] order. `x` is `(data, n_features)` and is required exactly
/// when `layout.width > layout.n_meta`.
pub fn concatenate_predictions<F: Copy + Default>(
    layout: &MetaLayout,
    blocks: &[(&[F], usize)],
    n_rows: usize,
    x: Option<(&[F], usize)>,
) -> Result<Vec<F>> {
    check_blocks(layout, blocks, n_rows, x)?;
    let passthrough = layout.width > layout.n_meta;

    let mut out = vec![F::default(); n_rows * layout.width];
    for r in 0..n_rows {
        let row = &mut out[r * layout.width..(r + 1) * layout.width];
        for (b, (data, cols)) in blocks.iter().enumerate() {
            let off = layout.offsets[b];
            row[off..off + cols].copy_from_slice(&data[r * cols..(r + 1) * cols]);
        }
        if let (true, Some((data, cols))) = (passthrough, x) {
            row[layout.n_meta..].copy_from_slice(&data[r * cols..(r + 1) * cols]);
        }
    }
    Ok(out)
}

/// The shape contract [`concatenate_predictions`] and [`concatenate_meta`]'s
/// device arm both hold callers to.
///
/// Factored out so the two arms reject identically: a kernel cannot return an
/// error, so the device route has to be told "no" here or not at all, and if it
/// were told by a different validator it would report a different message for
/// the same mistake.
fn check_blocks<F>(
    layout: &MetaLayout,
    blocks: &[(&[F], usize)],
    n_rows: usize,
    x: Option<(&[F], usize)>,
) -> Result<()> {
    if blocks.len() != layout.n_feature_outs.len() {
        return Err(value_err(format!(
            "layout describes {} blocks but {} were given",
            layout.n_feature_outs.len(),
            blocks.len()
        )));
    }
    let passthrough = layout.width > layout.n_meta;
    let n_features = layout.width - layout.n_meta;
    for (i, (data, cols)) in blocks.iter().enumerate() {
        if *cols != layout.n_feature_outs[i] {
            return Err(value_err(format!(
                "block {i} has {cols} columns, layout expects {}",
                layout.n_feature_outs[i]
            )));
        }
        if data.len() != n_rows * cols {
            return Err(value_err(format!(
                "block {i} has {} elements, expected n_rows * n_cols = {}",
                data.len(),
                n_rows * cols
            )));
        }
    }
    match (passthrough, x) {
        (true, None) => {
            return Err(value_err(
                "passthrough layout requires X, but none was given",
            ))
        }
        (true, Some((data, cols))) => {
            if cols != n_features {
                return Err(value_err(format!(
                    "X has {cols} columns, layout expects {n_features}"
                )));
            }
            if data.len() != n_rows * cols {
                return Err(value_err(format!(
                    "X has {} elements, expected n_rows * n_features = {}",
                    data.len(),
                    n_rows * cols
                )));
            }
        }
        (false, _) => {}
    }
    Ok(())
}

/// Assemble the meta matrix on the arm `engine` names (STACK-META-01).
///
/// [`MetaEngine::Host`] runs [`concatenate_predictions`] above;
/// [`MetaEngine::Device`] runs the CubeCL scatter in
/// [`mlrs_backend::prims::stacking_meta`]. [`MetaEngine::Numpy`] never reaches
/// here — it is the shim's own `np.hstack` and is resolved before the FFI
/// boundary is crossed at all — so it is treated as `Host`, which is what a
/// Rust-native caller asking for "not the device" means.
///
/// Both arms produce the same bytes: the device kernel writes each block to the
/// same offsets this host loop copies them to, and neither performs arithmetic,
/// so the equality is bit-exact rather than within a tolerance
/// (`crates/mlrs-backend/tests/stacking_meta_test.rs` asserts exactly that).
pub fn concatenate_meta<F>(
    engine: MetaEngine,
    pool: &mut BufferPool<ActiveRuntime>,
    layout: &MetaLayout,
    blocks: &[(&[F], usize)],
    n_rows: usize,
    x: Option<(&[F], usize)>,
) -> Result<Vec<F>>
where
    F: Float + CubeElement + Pod + Default,
{
    match engine {
        MetaEngine::Numpy | MetaEngine::Host => {
            concatenate_predictions(layout, blocks, n_rows, x)
        }
        MetaEngine::Device => {
            // The host arm's validation runs FIRST even on the device route, so
            // a mis-shaped block reports the same sklearn-shaped `ValueError`
            // text on both arms instead of a `PrimError` on one of them.
            check_blocks(layout, blocks, n_rows, x)?;
            concat_meta_device(
                pool,
                blocks,
                &layout.offsets,
                n_rows,
                layout.n_meta,
                layout.width,
                x,
            )
            .map_err(|e| value_err(e.to_string()))
        }
    }
}
