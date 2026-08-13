//! `stacking` — the structural core of the stacked-generalization meta-estimator
//! (STACK-01), shared by `mlrs.ensemble.StackingRegressor` and
//! `mlrs.ensemble.StackingClassifier` (STACK-CLF-01).
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
//! - [`stack_method_request`] / [`resolve_stack_method`] —
//!   `StackingClassifier`'s `stack_method` constraint and
//!   `_BaseStacking._method_name`'s `"auto"` fallback chain
//! - [`classifier_meta_slices`] — the *column-dropping* half of
//!   `_BaseStacking._concatenate_predictions`, which is what makes the
//!   classifier's layout differ from the regressor's
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
///
/// `owner` is the meta-estimator's class name, which sklearn interpolates into
/// the message (`StackingRegressor` / `StackingClassifier`); it is a parameter
/// rather than a constant because both classes share this rule and a caller
/// grepping the message must see its OWN class named.
pub fn cv_route_from_str(cv: &str, owner: &str) -> Result<CvRoute> {
    if cv == PREFIT {
        Ok(CvRoute::Prefit)
    } else {
        Err(value_err(format!(
            "The 'cv' parameter of {owner} must be an int in the range \
             [2, inf), an object implementing 'split' and 'get_n_splits', an \
             iterable or None or a str among {{'prefit'}}. Got '{cv}' instead."
        )))
    }
}

// --------------------------------------------------------------------------- //
// `stack_method` — the classifier's response-method selection (STACK-CLF-01)
// --------------------------------------------------------------------------- //

/// The response methods a stacked classifier can draw meta features from, in
/// sklearn's `stack_method="auto"` PREFERENCE order.
///
/// The order is the semantics: `"auto"` walks this list and takes the first
/// method the estimator implements, so a classifier that exposes both
/// `predict_proba` and `decision_function` contributes probabilities.
pub const STACK_METHODS: [&str; 3] = ["predict_proba", "decision_function", "predict"];

/// One response method of a base estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackMethod {
    /// `predict_proba` — `(n, n_classes)`, and the only method whose block is
    /// column-dropped in the binary case (see [`classifier_meta_slices`]).
    PredictProba,
    /// `decision_function` — `(n,)` when binary, `(n, n_classes)` otherwise.
    DecisionFunction,
    /// `predict` — always one column, of ENCODED labels.
    Predict,
}

impl StackMethod {
    /// The Python method name, which is also what `stack_method_` reports.
    pub fn as_str(self) -> &'static str {
        match self {
            StackMethod::PredictProba => STACK_METHODS[0],
            StackMethod::DecisionFunction => STACK_METHODS[1],
            StackMethod::Predict => STACK_METHODS[2],
        }
    }

    /// Position in [`STACK_METHODS`] — the index into the `implements` flags
    /// [`resolve_stack_method`] takes.
    pub fn index(self) -> usize {
        match self {
            StackMethod::PredictProba => 0,
            StackMethod::DecisionFunction => 1,
            StackMethod::Predict => 2,
        }
    }

    /// Parse a method name, `None` when it is not one of the three.
    pub fn parse(name: &str) -> Option<StackMethod> {
        match name {
            "predict_proba" => Some(StackMethod::PredictProba),
            "decision_function" => Some(StackMethod::DecisionFunction),
            "predict" => Some(StackMethod::Predict),
            _ => None,
        }
    }
}

/// What the caller's `stack_method` string asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackMethodRequest {
    /// `"auto"` — take the first of [`STACK_METHODS`] the estimator has.
    Auto,
    /// A named method, which the estimator is then REQUIRED to implement.
    Fixed(StackMethod),
}

/// Validate the `stack_method` constructor argument.
///
/// sklearn declares it as
/// `StrOptions({"auto", "predict_proba", "decision_function", "predict"})`, so
/// an unrecognized value is an `InvalidParameterError` — the shim re-raises
/// this `ValueError` under that class, exactly as it does for `cv`.
///
/// **The option ORDER in the message is not part of parity.** sklearn renders
/// the constraint by iterating a Python `set`, whose iteration order for these
/// strings changes with `PYTHONHASHSEED` — the same call produces
/// `{'decision_function', 'auto', …}` in one process and `{'predict_proba', …}`
/// in the next. This function emits the declaration order; the oracle test
/// compares the two messages with the option set parsed out rather than as raw
/// text, because comparing them literally would be a coin flip.
pub fn stack_method_request(value: &str) -> Result<StackMethodRequest> {
    if value == "auto" {
        return Ok(StackMethodRequest::Auto);
    }
    match StackMethod::parse(value) {
        Some(method) => Ok(StackMethodRequest::Fixed(method)),
        None => Err(value_err(format!(
            "The 'stack_method' parameter of StackingClassifier must be a str \
             among {{'auto', 'predict_proba', 'decision_function', 'predict'}}. \
             Got '{value}' instead."
        ))),
    }
}

/// sklearn `_BaseStacking._method_name`: which method this estimator will be
/// asked for.
///
/// `implements[i]` answers "does the estimator have `STACK_METHODS[i]`?" — the
/// `hasattr` has to happen in Python (an `available_if` descriptor decides it at
/// access time, e.g. `SVC.predict_proba` behind `probability=True`), the
/// CONSEQUENCE does not.
///
/// [`StackMethodRequest::Auto`] takes the first available method and only fails
/// when the estimator has none of the three. A [`StackMethodRequest::Fixed`]
/// request fails when that one method is missing. Both produce sklearn's single
/// message, whose `{method}` renders as a Python list repr on the `"auto"`
/// branch because that is the value sklearn's f-string interpolates there.
pub fn resolve_stack_method(
    name: &str,
    request: StackMethodRequest,
    implements: [bool; 3],
) -> Result<StackMethod> {
    match request {
        StackMethodRequest::Auto => {
            for (i, &has) in implements.iter().enumerate() {
                if has {
                    return Ok(StackMethod::parse(STACK_METHODS[i]).expect("static name"));
                }
            }
            Err(value_err(format!(
                "Underlying estimator {name} does not implement the method \
                 ['predict_proba', 'decision_function', 'predict']."
            )))
        }
        StackMethodRequest::Fixed(method) => {
            if implements[method.index()] {
                Ok(method)
            } else {
                Err(value_err(format!(
                    "Underlying estimator {name} does not implement the method {}.",
                    method.as_str()
                )))
            }
        }
    }
}

// --------------------------------------------------------------------------- //
// classifier meta blocks — the column-dropping rule (STACK-CLF-01)
// --------------------------------------------------------------------------- //

/// The shape of one kept estimator's RAW response, before any column is
/// dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredShape {
    /// A 1-D `(n,)` response — `predict`, or a binary `decision_function`.
    Column,
    /// A 2-D `(n, cols)` response.
    Matrix(usize),
    /// A LIST of `(n, cols_j)` responses, one per target: what `predict_proba`
    /// returns for a multilabel `y` from estimators that fit one classifier per
    /// column (`RandomForestClassifier`, `MultiOutputClassifier`).
    MultiOutput(Vec<usize>),
}

/// Which columns of which raw response block become one meta block.
///
/// The shim slices `pred[:, start_col : start_col + n_cols]` — for a
/// [`PredShape::MultiOutput`] response, `sub` selects the list element first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetaSlice {
    /// Index into the KEPT estimator list (`stack_method_` order).
    pub block: usize,
    /// Index within a [`PredShape::MultiOutput`] list; `0` otherwise.
    pub sub: usize,
    /// First source column taken — `1` exactly where a probability column is
    /// dropped, `0` everywhere else.
    pub start_col: usize,
    /// How many source columns are taken; this is the meta block's width.
    pub n_cols: usize,
}

/// sklearn `_BaseStacking._concatenate_predictions`, minus the copy: the meta
/// blocks each kept estimator contributes.
///
/// Three rules, in sklearn's own order:
///
/// 1. A [`PredShape::MultiOutput`] response contributes one block per target,
///    each with its FIRST column dropped — unconditionally, whatever the
///    method, because those columns are the per-target probabilities and sum
///    to one.
/// 2. A 1-D response becomes a single column.
/// 3. A 2-D `predict_proba` response drops its first column when
///    `n_classes == 2`: `p(y=0) = 1 - p(y=1)`, so keeping both would hand the
///    final estimator two perfectly collinear features. Every other 2-D
///    response is taken whole.
///
/// `n_classes` is `len(classes_)`, passed in as sklearn uses it. Note that for
/// a multilabel `y` sklearn's `classes_` is a LIST OF PER-TARGET ARRAYS, so
/// `len(classes_)` counts TARGETS there, not classes — rule 3 therefore reads
/// "two targets" in that case. That is sklearn's behaviour, quirk included, and
/// rule 1 makes it unobservable for the estimators that actually return a list.
///
/// A zero-column meta block (a `predict_proba` that returned a single column,
/// say) is left for [`meta_layout`] to reject, so both classes report that the
/// same way.
pub fn classifier_meta_slices(
    methods: &[StackMethod],
    shapes: &[PredShape],
    n_classes: usize,
) -> Result<Vec<MetaSlice>> {
    if methods.len() != shapes.len() {
        return Err(value_err(format!(
            "have {} resolved stack methods but {} prediction shapes",
            methods.len(),
            shapes.len()
        )));
    }
    let mut slices = Vec::with_capacity(shapes.len());
    for (block, shape) in shapes.iter().enumerate() {
        match shape {
            PredShape::MultiOutput(cols) => {
                if cols.is_empty() {
                    return Err(value_err(format!(
                        "estimator at position {block} returned an empty list of \
                         prediction blocks"
                    )));
                }
                for (sub, &c) in cols.iter().enumerate() {
                    slices.push(MetaSlice {
                        block,
                        sub,
                        start_col: 1,
                        n_cols: c.saturating_sub(1),
                    });
                }
            }
            PredShape::Column => slices.push(MetaSlice {
                block,
                sub: 0,
                start_col: 0,
                n_cols: 1,
            }),
            PredShape::Matrix(c) => {
                let drop_first =
                    methods[block] == StackMethod::PredictProba && n_classes == 2;
                slices.push(MetaSlice {
                    block,
                    sub: 0,
                    start_col: usize::from(drop_first),
                    n_cols: if drop_first { c.saturating_sub(1) } else { *c },
                });
            }
        }
    }
    Ok(slices)
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
///
/// The two lists are ZIPPED, as sklearn's `zip(non_dropped_estimators,
/// self._n_feature_outs)` is — but only in the ONE direction where a mismatch
/// can legitimately occur:
///
/// * **more counts than names** is the multilabel case: a `predict_proba` that
///   returns one array per target contributes one meta block PER TARGET (see
///   [`classifier_meta_slices`]), so `n_feature_outs` outruns the estimator
///   names and sklearn silently names only the first block of each. Parity
///   means truncating too; raising would reject a fit sklearn completes.
/// * **more names than counts** cannot happen on either class — it would mean a
///   kept estimator contributed no block at all. Truncating there would drop
///   that estimator's columns from the names while `transform` still emits
///   them, so every name after the gap would silently describe the wrong
///   column. That is the same silent shift [`meta_layout`]'s zero-column
///   rejection exists to prevent, so it is rejected here rather than papered
///   over.
pub fn meta_feature_names(
    class_name: &str,
    kept_names: &[String],
    n_feature_outs: &[usize],
    input_features: Option<&[String]>,
) -> Result<Vec<String>> {
    if kept_names.len() > n_feature_outs.len() {
        return Err(value_err(format!(
            "have {} kept estimator names but only {} feature-out counts; a \
             kept estimator produced no prediction block",
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
