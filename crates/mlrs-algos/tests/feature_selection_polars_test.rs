//! A polars `DataFrame` driving a real feature selector, end to end (FSEL-01).
//!
//! `mlrs-core`'s `frame_test.rs` unit-tests the CONVERSION; this tests the
//! INTEGRATION, and the two are not the same claim. What can still be wrong after
//! the conversion is verified is the CONTRACT between them: whether the
//! `(values, rows, cols)` triple the conversion produces is the triple the
//! estimator actually consumes, or a plausible permutation of it. A transposed
//! buffer passes every conversion test that only checks its own output and then
//! silently scores the wrong columns here.
//!
//! It lives in `mlrs-algos` rather than `mlrs-core` because it needs both crates
//! and `mlrs-core` cannot depend on `mlrs-algos` (that is the dependency edge, and
//! it points the other way).
//!
//! Only compiled with `--features polars`:
//! `cargo test -p mlrs-algos --features cpu,polars --test feature_selection_polars_test`.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

#![cfg(feature = "polars")]

use mlrs_algos::feature_selection::{
    f_classif, univariate, variance_threshold::variances_and_support, GenericParam, ScoreFunc,
};
use mlrs_core::frame::{column_names, dataframe_to_rowmajor, series_to_vec, take_columns};
use polars::prelude::*;

/// A frame with one column that a selector must DROP (constant) and one that it
/// must KEEP, plus a class label as a separate `Series`.
///
/// Deliberately mixed-dtype: the informative columns are `f64`, the constant one
/// is `i32`, and a `Boolean` column carries real signal. A frame that is uniformly
/// `f64` cannot tell a correct per-column cast from a byte reinterpretation.
fn labelled_frame() -> (DataFrame, Series) {
    let df = df![
        "informative" => [0.1f64, 0.2, 0.15, 9.1, 9.4, 9.2],
        "constant" => [7i32, 7, 7, 7, 7, 7],
        "noise" => [0.5f64, 9.0, 0.4, 8.5, 0.6, 9.5],
        "flag" => [false, false, false, true, true, true],
    ]
    .expect("build frame");
    let y = Series::new("target".into(), [0i32, 0, 0, 1, 1, 1]);
    (df, y)
}

#[test]
fn variance_threshold_drops_the_constant_column_of_a_frame() {
    let (df, _) = labelled_frame();
    let (values, rows, cols, names) = dataframe_to_rowmajor::<f64>(&df).expect("convert");
    assert_eq!(names, vec!["informative", "constant", "noise", "flag"]);

    let (variances, mask) =
        variances_and_support(&values, rows, cols, 0.0).expect("variance_threshold");
    assert_eq!(
        mask,
        vec![true, false, true, true],
        "only the constant column has zero variance"
    );
    assert_eq!(
        variances[1], 0.0,
        "the constant column's variance is exactly 0"
    );

    // The selection is applied in the FRAME domain, so names and per-column
    // dtypes survive — which is the whole reason `take_columns` exists alongside
    // `rowmajor_to_dataframe`.
    let reduced = take_columns(&df, &mask).expect("apply the mask");
    assert_eq!(column_names(&reduced), vec!["informative", "noise", "flag"]);
    assert_eq!(reduced.height(), 6);
    assert_eq!(
        reduced.column("flag").unwrap().dtype(),
        &DataType::Boolean,
        "the Boolean column must not be flattened to a float by the round trip"
    );
}

#[test]
fn select_k_best_ranks_a_frames_columns() {
    let (df, y) = labelled_frame();
    let (values, rows, cols, names) = dataframe_to_rowmajor::<f64>(&df).expect("convert");
    let target: Vec<f64> = series_to_vec(&y).expect("convert target");

    let (scores, pvalues, mask) = univariate::fit_host(
        &values,
        &target,
        rows,
        cols,
        "k_best",
        GenericParam::Value(2.0),
        ScoreFunc::FClassif,
    )
    .expect("SelectKBest(k=2)");

    assert_eq!(mask.iter().filter(|&&k| k).count(), 2);
    // `informative` and `flag` separate the two classes perfectly, `noise` does
    // not, and `constant` scores `NaN` — so the two kept columns are the
    // perfect separators. Asserted by NAME, which is the point of carrying the
    // names through the conversion.
    let kept: Vec<&String> = names
        .iter()
        .zip(&mask)
        .filter(|(_, &k)| k)
        .map(|(n, _)| n)
        .collect();
    assert!(
        kept.contains(&&"informative".to_string()) && kept.contains(&&"flag".to_string()),
        "expected the two perfect separators, got {kept:?}"
    );
    assert!(
        scores[1].is_nan(),
        "the constant column's F must be NaN, got {}",
        scores[1]
    );
    assert!(pvalues.is_some(), "f_classif yields p-values");
}

#[test]
fn frame_ingress_agrees_with_a_hand_built_row_major_slice() {
    // The contract check: converting the frame must produce EXACTLY the buffer a
    // caller would have written by hand, so a selector cannot tell the two apart.
    // This is what a transposed conversion fails and a self-consistent one passes.
    let (df, y) = labelled_frame();
    let (values, rows, cols, _) = dataframe_to_rowmajor::<f64>(&df).expect("convert");
    let target: Vec<f64> = series_to_vec(&y).expect("convert target");

    #[rustfmt::skip]
    let by_hand: Vec<f64> = vec![
        0.10, 7.0, 0.5, 0.0,
        0.20, 7.0, 9.0, 0.0,
        0.15, 7.0, 0.4, 0.0,
        9.10, 7.0, 8.5, 1.0,
        9.40, 7.0, 0.6, 1.0,
        9.20, 7.0, 9.5, 1.0,
    ];
    assert_eq!(
        values, by_hand,
        "frame ingress must match the row-major layout"
    );

    let from_frame = f_classif(&values, &target, rows, cols).expect("f_classif on the frame");
    let from_slice = f_classif(&by_hand, &target, 6, 4).expect("f_classif on the slice");
    assert_eq!(
        format!("{:?}", from_frame.scores),
        format!("{:?}", from_slice.scores),
        "the two ingress routes must score identically"
    );
}
