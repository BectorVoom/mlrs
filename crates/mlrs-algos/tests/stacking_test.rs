//! STACK-01 — `mlrs_algos::ensemble::stacking` structural core.
//!
//! Every assertion here is a rule a caller can observe through
//! `mlrs.StackingRegressor`: an exception message, a `get_feature_names_out()`
//! string, or the column order of the matrix `final_estimator_` is fitted on.
//! The message texts are the ones scikit-learn 1.9 emits verbatim — captured
//! from a live sklearn run and re-checked end-to-end by
//! `crates/mlrs-py/python/tests/test_oracle_stacking.py`, which asserts mlrs and
//! sklearn raise the SAME text. Changing a string here without changing it
//! there turns a parity guarantee into a silent divergence.
//!
//! No device work and no fixtures: this module is host bookkeeping, so the
//! suite runs identically on every backend.

use mlrs_algos::ensemble::stacking::{
    classifier_meta_slices, concatenate_predictions, cv_route_from_str, kept_indices,
    meta_feature_names, meta_layout, resolve_stack_method, stack_method_request, CvRoute,
    MetaLayout, MetaSlice, PredShape, StackMethod, StackMethodRequest, StackingError, DROP, PREFIT,
    STACK_METHODS,
};

fn s(items: &[&str]) -> Vec<String> {
    items.iter().map(|x| x.to_string()).collect()
}

fn msg(err: StackingError) -> String {
    let StackingError::Value(m) = err;
    m
}

// --------------------------------------------------------------------------- //
// validate_names — sklearn `_BaseComposition._validate_names`
// --------------------------------------------------------------------------- //

const CTOR_PARAMS: [&str; 6] = [
    "cv",
    "estimators",
    "final_estimator",
    "n_jobs",
    "passthrough",
    "verbose",
];

fn ctor() -> Vec<String> {
    s(&CTOR_PARAMS)
}

#[test]
fn validate_names_accepts_distinct_non_colliding_names() {
    assert!(
        mlrs_algos::ensemble::stacking::validate_names(&s(&["lr", "rf", "svr"]), &ctor()).is_ok()
    );
}

#[test]
fn validate_names_rejects_duplicates_with_sklearn_message() {
    let err = mlrs_algos::ensemble::stacking::validate_names(&s(&["a", "a"]), &ctor()).unwrap_err();
    assert_eq!(msg(err), "Names provided are not unique: ['a', 'a']");
}

#[test]
fn validate_names_rejects_ctor_argument_collision_with_sklearn_message() {
    let err = mlrs_algos::ensemble::stacking::validate_names(&s(&["cv"]), &ctor()).unwrap_err();
    assert_eq!(
        msg(err),
        "Estimator names conflict with constructor arguments: ['cv']"
    );
}

#[test]
fn validate_names_sorts_the_colliding_names() {
    // sklearn reports `sorted(invalid_names)`, NOT list order — a set is
    // unordered there, so the sort is what makes the message deterministic.
    let err = mlrs_algos::ensemble::stacking::validate_names(&s(&["verbose", "cv"]), &ctor())
        .unwrap_err();
    assert_eq!(
        msg(err),
        "Estimator names conflict with constructor arguments: ['cv', 'verbose']"
    );
}

#[test]
fn validate_names_rejects_double_underscore_with_sklearn_message() {
    let err = mlrs_algos::ensemble::stacking::validate_names(&s(&["a__b"]), &ctor()).unwrap_err();
    assert_eq!(
        msg(err),
        "Estimator names must not contain __: got ['a__b']"
    );
}

#[test]
fn validate_names_reports_duplication_before_collision() {
    // A list that trips BOTH rules reports the duplicate: sklearn runs the
    // uniqueness check first, and callers pattern-match on the message.
    let err =
        mlrs_algos::ensemble::stacking::validate_names(&s(&["cv", "cv"]), &ctor()).unwrap_err();
    assert_eq!(msg(err), "Names provided are not unique: ['cv', 'cv']");
}

#[test]
fn validate_names_escapes_quotes_like_python_repr() {
    let err =
        mlrs_algos::ensemble::stacking::validate_names(&s(&["it's", "it's"]), &ctor()).unwrap_err();
    assert_eq!(
        msg(err),
        r"Names provided are not unique: ['it\'s', 'it\'s']"
    );
}

// --------------------------------------------------------------------------- //
// kept_indices — the `'drop'` sentinel
// --------------------------------------------------------------------------- //

#[test]
fn kept_indices_keeps_list_order_and_skips_drops() {
    assert_eq!(
        kept_indices(&[false, true, false, false]).unwrap(),
        vec![0, 2, 3]
    );
}

#[test]
fn kept_indices_rejects_empty_estimators() {
    let err = kept_indices(&[]).unwrap_err();
    assert_eq!(
        msg(err),
        "Invalid 'estimators' attribute, 'estimators' should be a non-empty list of \
         (string, estimator) tuples."
    );
}

#[test]
fn kept_indices_rejects_all_dropped() {
    let err = kept_indices(&[true, true]).unwrap_err();
    assert_eq!(
        msg(err),
        "All estimators are dropped. At least one is required to be an estimator."
    );
}

#[test]
fn drop_sentinel_is_the_sklearn_literal() {
    assert_eq!(DROP, "drop");
}

// --------------------------------------------------------------------------- //
// cv_route_from_str — the `"prefit"` string parameter
// --------------------------------------------------------------------------- //

#[test]
fn cv_prefit_selects_the_prefit_route() {
    assert_eq!(
        cv_route_from_str(PREFIT, "StackingRegressor").unwrap(),
        CvRoute::Prefit
    );
    assert_eq!(PREFIT, "prefit");
}

#[test]
fn cv_rejects_any_other_string_with_sklearn_message() {
    let err = cv_route_from_str("bogus", "StackingRegressor").unwrap_err();
    assert_eq!(
        msg(err),
        "The 'cv' parameter of StackingRegressor must be an int in the range [2, inf), \
         an object implementing 'split' and 'get_n_splits', an iterable or None or a str \
         among {'prefit'}. Got 'bogus' instead."
    );
}

#[test]
fn cv_message_names_the_calling_class() {
    // The rule is shared; the message is not. A `StackingClassifier` user must
    // not be told their `StackingRegressor` is at fault.
    let err = cv_route_from_str("bogus", "StackingClassifier").unwrap_err();
    assert!(msg(err).starts_with("The 'cv' parameter of StackingClassifier must be an int"));
}

#[test]
fn cv_string_match_is_case_sensitive() {
    // sklearn's StrOptions is an exact set membership test; "Prefit" is not in it.
    assert!(cv_route_from_str("Prefit", "StackingRegressor").is_err());
}

// --------------------------------------------------------------------------- //
// meta_layout — column order and `_n_feature_outs`
// --------------------------------------------------------------------------- //

#[test]
fn meta_layout_of_scalar_regressors_is_one_column_each() {
    let l = meta_layout(&[1, 1, 1], 7, false).unwrap();
    assert_eq!(
        l,
        MetaLayout {
            n_feature_outs: vec![1, 1, 1],
            offsets: vec![0, 1, 2],
            n_meta: 3,
            width: 3,
        }
    );
}

#[test]
fn meta_layout_appends_passthrough_columns_last() {
    let l = meta_layout(&[1, 1], 4, true).unwrap();
    assert_eq!(l.n_meta, 2);
    assert_eq!(l.width, 6);
    // The estimator blocks keep offsets 0..n_meta; X starts exactly at n_meta.
    assert_eq!(l.offsets, vec![0, 1]);
}

#[test]
fn meta_layout_handles_multi_output_blocks() {
    let l = meta_layout(&[3, 1, 2], 0, false).unwrap();
    assert_eq!(l.offsets, vec![0, 3, 4]);
    assert_eq!(l.width, 6);
}

#[test]
fn meta_layout_rejects_a_zero_column_block() {
    let err = meta_layout(&[1, 0], 0, false).unwrap_err();
    assert_eq!(
        msg(err),
        "estimator at position 1 produced a prediction block with 0 columns"
    );
}

#[test]
fn meta_layout_rejects_no_blocks() {
    assert!(meta_layout(&[], 3, true).is_err());
}

// --------------------------------------------------------------------------- //
// meta_feature_names — sklearn `get_feature_names_out`
// --------------------------------------------------------------------------- //

#[test]
fn feature_names_single_column_blocks_have_no_index_suffix() {
    let names = meta_feature_names("stackingregressor", &s(&["a", "b"]), &[1, 1], None).unwrap();
    assert_eq!(names, s(&["stackingregressor_a", "stackingregressor_b"]));
}

#[test]
fn feature_names_multi_column_blocks_are_indexed_without_a_separator() {
    // sklearn writes `..._lr0`, not `..._lr_0` — the index is concatenated bare.
    let names = meta_feature_names("stackingclassifier", &s(&["lr"]), &[3], None).unwrap();
    assert_eq!(
        names,
        s(&[
            "stackingclassifier_lr0",
            "stackingclassifier_lr1",
            "stackingclassifier_lr2"
        ])
    );
}

#[test]
fn feature_names_append_input_features_under_passthrough() {
    let inputs = s(&["x0", "x1", "x2", "x3"]);
    let names =
        meta_feature_names("stackingregressor", &s(&["a", "b"]), &[1, 1], Some(&inputs)).unwrap();
    assert_eq!(
        names,
        s(&[
            "stackingregressor_a",
            "stackingregressor_b",
            "x0",
            "x1",
            "x2",
            "x3"
        ])
    );
}

#[test]
fn extra_feature_out_counts_truncate_like_sklearns_zip() {
    // sklearn zips `non_dropped_estimators` with `_n_feature_outs`, so a
    // multilabel `predict_proba` — which contributes one meta block PER TARGET
    // and therefore outruns the names — emits the SHORT list rather than
    // raising. Verified against sklearn 1.9: a one-estimator, three-target
    // stack reports exactly `['stackingclassifier_rf']`.
    let names = meta_feature_names("stackingclassifier", &s(&["rf"]), &[1, 1, 1], None).unwrap();
    assert_eq!(names, s(&["stackingclassifier_rf"]));
}

#[test]
fn extra_names_are_rejected_because_they_would_shift_every_later_name() {
    // The mirror case is NOT symmetric. More names than counts means a kept
    // estimator contributed no block, and truncating would leave `transform`
    // emitting columns that no name describes — the silent shift
    // `meta_layout`'s zero-column rejection also exists to prevent. sklearn
    // cannot reach this state, so rejecting it costs nothing in parity.
    let err = meta_feature_names("stackingregressor", &s(&["a", "b"]), &[1], None).unwrap_err();
    assert_eq!(
        msg(err),
        "have 2 kept estimator names but only 1 feature-out counts; a kept \
         estimator produced no prediction block"
    );
}

// --------------------------------------------------------------------------- //
// concatenate_predictions — the layout, executed
// --------------------------------------------------------------------------- //

#[test]
fn concatenate_places_blocks_in_kept_order() {
    let a = [1.0, 2.0, 3.0]; // 3 rows x 1 col
    let b = [10.0, 20.0, 30.0];
    let layout = meta_layout(&[1, 1], 0, false).unwrap();
    let out = concatenate_predictions(&layout, &[(&a, 1), (&b, 1)], 3, None).unwrap();
    assert_eq!(out, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
}

#[test]
fn concatenate_appends_x_after_every_prediction_block() {
    let a = [1.0, 2.0];
    let x = [7.0, 8.0, 9.0, 70.0, 80.0, 90.0]; // 2 rows x 3 cols
    let layout = meta_layout(&[1], 3, true).unwrap();
    let out = concatenate_predictions(&layout, &[(&a, 1)], 2, Some((&x, 3))).unwrap();
    assert_eq!(out, vec![1.0, 7.0, 8.0, 9.0, 2.0, 70.0, 80.0, 90.0]);
}

#[test]
fn concatenate_interleaves_multi_column_blocks_correctly() {
    let a = [1.0, 2.0, 3.0, 4.0]; // 2 rows x 2 cols
    let b = [5.0, 6.0]; // 2 rows x 1 col
    let layout = meta_layout(&[2, 1], 0, false).unwrap();
    let out = concatenate_predictions(&layout, &[(&a, 2), (&b, 1)], 2, None).unwrap();
    assert_eq!(out, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
}

#[test]
fn concatenate_rejects_a_missing_passthrough_x() {
    let a = [1.0, 2.0];
    let layout = meta_layout(&[1], 3, true).unwrap();
    let err = concatenate_predictions(&layout, &[(&a, 1)], 2, None).unwrap_err();
    assert_eq!(
        msg(err),
        "passthrough layout requires X, but none was given"
    );
}

#[test]
fn concatenate_rejects_a_mis_shaped_block() {
    let a = [1.0, 2.0, 3.0];
    let layout = meta_layout(&[1], 0, false).unwrap();
    let err = concatenate_predictions(&layout, &[(&a, 1)], 2, None).unwrap_err();
    assert_eq!(
        msg(err),
        "block 0 has 3 elements, expected n_rows * n_cols = 2"
    );
}

#[test]
fn concatenate_rejects_a_block_count_mismatch() {
    let a = [1.0, 2.0];
    let layout = meta_layout(&[1, 1], 0, false).unwrap();
    let err = concatenate_predictions(&layout, &[(&a, 1)], 2, None).unwrap_err();
    assert_eq!(msg(err), "layout describes 2 blocks but 1 were given");
}

// --------------------------------------------------------------------------- //
// stack_method — the classifier's response-method selection (STACK-CLF-01)
// --------------------------------------------------------------------------- //

const ALL: [bool; 3] = [true, true, true];
const NONE: [bool; 3] = [false, false, false];
/// A `LinearSVC`-shaped estimator: `decision_function` + `predict`, no proba.
const SVC_LIKE: [bool; 3] = [false, true, true];
/// A `GaussianNB`-shaped estimator: `predict_proba` + `predict`.
const NB_LIKE: [bool; 3] = [true, false, true];
/// A regressor: `predict` only. Legal here — sklearn allows regressors as base
/// estimators of a `StackingClassifier` (ordinal regression).
const PREDICT_ONLY: [bool; 3] = [false, false, true];

fn auto() -> StackMethodRequest {
    StackMethodRequest::Auto
}

fn fixed(name: &str) -> StackMethodRequest {
    StackMethodRequest::Fixed(StackMethod::parse(name).unwrap())
}

#[test]
fn stack_method_accepts_the_four_sklearn_options() {
    assert_eq!(
        stack_method_request("auto").unwrap(),
        StackMethodRequest::Auto
    );
    for name in STACK_METHODS {
        assert_eq!(
            stack_method_request(name).unwrap(),
            StackMethodRequest::Fixed(StackMethod::parse(name).unwrap())
        );
    }
}

#[test]
fn stack_method_rejects_anything_else_with_sklearns_message() {
    let err = stack_method_request("proba").unwrap_err();
    // The OPTION ORDER inside the braces is sklearn's set-iteration order and
    // changes with PYTHONHASHSEED, so the oracle test compares option SETS;
    // everything outside the braces is fixed text and is asserted here.
    let text = msg(err);
    assert!(text.starts_with(
        "The 'stack_method' parameter of StackingClassifier must be a str among {"
    ));
    assert!(text.ends_with("}. Got 'proba' instead."));
    for name in ["auto", "predict_proba", "decision_function", "predict"] {
        assert!(text.contains(&format!("'{name}'")), "missing option {name}");
    }
}

#[test]
fn stack_method_is_case_sensitive() {
    assert!(stack_method_request("Auto").is_err());
    assert!(stack_method_request("Predict").is_err());
}

#[test]
fn auto_prefers_proba_then_decision_then_predict() {
    assert_eq!(
        resolve_stack_method("lr", auto(), ALL).unwrap(),
        StackMethod::PredictProba
    );
    assert_eq!(
        resolve_stack_method("svc", auto(), SVC_LIKE).unwrap(),
        StackMethod::DecisionFunction
    );
    assert_eq!(
        resolve_stack_method("ridge", auto(), PREDICT_ONLY).unwrap(),
        StackMethod::Predict
    );
}

#[test]
fn auto_rejects_an_estimator_with_no_response_method() {
    let err = resolve_stack_method("weird", auto(), NONE).unwrap_err();
    assert_eq!(
        msg(err),
        "Underlying estimator weird does not implement the method \
         ['predict_proba', 'decision_function', 'predict']."
    );
}

#[test]
fn a_named_method_is_taken_verbatim_when_available() {
    assert_eq!(
        resolve_stack_method("nb", fixed("predict"), NB_LIKE).unwrap(),
        StackMethod::Predict
    );
    assert_eq!(
        resolve_stack_method("lr", fixed("decision_function"), ALL).unwrap(),
        StackMethod::DecisionFunction
    );
}

#[test]
fn a_named_method_the_estimator_lacks_is_sklearns_value_error() {
    let err = resolve_stack_method("svc", fixed("predict_proba"), SVC_LIKE).unwrap_err();
    assert_eq!(
        msg(err),
        "Underlying estimator svc does not implement the method predict_proba."
    );
    let err = resolve_stack_method("nb", fixed("decision_function"), NB_LIKE).unwrap_err();
    assert_eq!(
        msg(err),
        "Underlying estimator nb does not implement the method decision_function."
    );
}

// --------------------------------------------------------------------------- //
// classifier_meta_slices — the column-dropping rule
// --------------------------------------------------------------------------- //

fn slice(block: usize, sub: usize, start_col: usize, n_cols: usize) -> MetaSlice {
    MetaSlice {
        block,
        sub,
        start_col,
        n_cols,
    }
}

#[test]
fn binary_proba_drops_the_first_column() {
    // p(y=0) = 1 - p(y=1): both columns are perfectly collinear, so sklearn
    // hands the final estimator only the second.
    let slices =
        classifier_meta_slices(&[StackMethod::PredictProba], &[PredShape::Matrix(2)], 2).unwrap();
    assert_eq!(slices, vec![slice(0, 0, 1, 1)]);
}

#[test]
fn multiclass_proba_keeps_every_column() {
    let slices =
        classifier_meta_slices(&[StackMethod::PredictProba], &[PredShape::Matrix(3)], 3).unwrap();
    assert_eq!(slices, vec![slice(0, 0, 0, 3)]);
}

#[test]
fn a_binary_decision_function_is_one_column_and_is_not_dropped() {
    // `decision_function` on a binary problem returns `(n,)`, and the drop rule
    // is `predict_proba`-only — a signed margin has no collinear twin.
    let slices =
        classifier_meta_slices(&[StackMethod::DecisionFunction], &[PredShape::Column], 2).unwrap();
    assert_eq!(slices, vec![slice(0, 0, 0, 1)]);

    // A multiclass one is `(n, K)` and stays whole.
    let slices =
        classifier_meta_slices(&[StackMethod::DecisionFunction], &[PredShape::Matrix(3)], 3)
            .unwrap();
    assert_eq!(slices, vec![slice(0, 0, 0, 3)]);
}

#[test]
fn predict_is_always_a_single_column() {
    for n_classes in [2usize, 5] {
        let slices =
            classifier_meta_slices(&[StackMethod::Predict], &[PredShape::Column], n_classes)
                .unwrap();
        assert_eq!(slices, vec![slice(0, 0, 0, 1)]);
    }
}

#[test]
fn a_multi_output_response_contributes_one_dropped_block_per_target() {
    // The multilabel `predict_proba` shape: a list of per-target `(n, 2)`
    // blocks, each reduced to its second column.
    let slices = classifier_meta_slices(
        &[StackMethod::PredictProba],
        &[PredShape::MultiOutput(vec![2, 2, 2])],
        3,
    )
    .unwrap();
    assert_eq!(
        slices,
        vec![slice(0, 0, 1, 1), slice(0, 1, 1, 1), slice(0, 2, 1, 1)]
    );
}

#[test]
fn mixed_members_keep_estimator_order() {
    let slices = classifier_meta_slices(
        &[
            StackMethod::PredictProba,
            StackMethod::DecisionFunction,
            StackMethod::Predict,
        ],
        &[PredShape::Matrix(4), PredShape::Matrix(4), PredShape::Column],
        4,
    )
    .unwrap();
    assert_eq!(
        slices,
        vec![slice(0, 0, 0, 4), slice(1, 0, 0, 4), slice(2, 0, 0, 1)]
    );

    // …and the widths those slices imply are what the layout is built from.
    let widths: Vec<usize> = slices.iter().map(|s| s.n_cols).collect();
    let layout = meta_layout(&widths, 0, false).unwrap();
    assert_eq!(layout.offsets, vec![0, 4, 8]);
    assert_eq!(layout.width, 9);
}

#[test]
fn slices_reject_a_methods_shapes_length_mismatch() {
    let err = classifier_meta_slices(
        &[StackMethod::Predict],
        &[PredShape::Column, PredShape::Column],
        2,
    )
    .unwrap_err();
    assert_eq!(
        msg(err),
        "have 1 resolved stack methods but 2 prediction shapes"
    );
}

#[test]
fn slices_reject_an_empty_multi_output_list() {
    let err = classifier_meta_slices(
        &[StackMethod::PredictProba],
        &[PredShape::MultiOutput(vec![])],
        2,
    )
    .unwrap_err();
    assert_eq!(
        msg(err),
        "estimator at position 0 returned an empty list of prediction blocks"
    );
}

#[test]
fn a_degenerate_single_column_proba_is_rejected_by_the_layout() {
    // A one-column `predict_proba` on a binary problem slices down to zero
    // columns. `classifier_meta_slices` reports it as a width, and
    // `meta_layout` is the single place that rejects a zero-width block — so
    // both estimators report it identically.
    let slices =
        classifier_meta_slices(&[StackMethod::PredictProba], &[PredShape::Matrix(1)], 2).unwrap();
    assert_eq!(slices, vec![slice(0, 0, 1, 0)]);
    let widths: Vec<usize> = slices.iter().map(|s| s.n_cols).collect();
    let err = meta_layout(&widths, 0, false).unwrap_err();
    assert_eq!(
        msg(err),
        "estimator at position 0 produced a prediction block with 0 columns"
    );
}
