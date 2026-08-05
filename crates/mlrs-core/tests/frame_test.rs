//! `mlrs_core::frame` — polars ingress/egress (FSEL-01).
//!
//! Only compiled with the `polars` feature; `cargo test -p mlrs-core --features
//! polars` runs it. The whole file is behind one `#![cfg]` gate so a default
//! `cargo test --workspace` neither builds polars nor reports these as skipped
//! (there is nothing to skip — the code does not exist in that configuration).
//!
//! What is verified: the ROW-MAJOR interleave (the one thing a column-major
//! source can get wrong), mixed-dtype and boolean casting, the round trip, the
//! mask-based column take that a selector's `transform` is for a polars caller,
//! and each rejection — a non-numeric column, a null, an empty frame — since
//! those are the module's contract with the caller and not merely error paths.
//!
//! Tests live in `crates/mlrs-core/tests/` (AGENTS.md §2).

#![cfg(feature = "polars")]

use mlrs_core::frame::{
    column_names, dataframe_to_rowmajor, rowmajor_to_dataframe, series_to_vec, take_columns,
    FrameError,
};
use polars::prelude::*;

/// A 3-row frame mixing `f64`, `i32` and `Boolean` columns.
///
/// The dtype mix is the point: a frame whose columns are all `f64` cannot
/// distinguish a correct per-column cast from one that reinterprets bytes.
fn mixed_frame() -> DataFrame {
    df![
        "a" => [1.5f64, 2.5, 3.5],
        "b" => [10i32, 20, 30],
        "c" => [true, false, true],
    ]
    .expect("build test frame")
}

#[test]
fn dataframe_converts_to_row_major() {
    let df = mixed_frame();
    let (values, rows, cols, names) =
        dataframe_to_rowmajor::<f64>(&df).expect("convert mixed frame");
    assert_eq!((rows, cols), (3, 3));
    assert_eq!(names, vec!["a", "b", "c"]);
    // ROW-major: row 0 first, then row 1. A column-major result would be
    // `[1.5, 2.5, 3.5, 10.0, ...]`, which is the mistake this asserts against.
    assert_eq!(
        values,
        vec![1.5, 10.0, 1.0, 2.5, 20.0, 0.0, 3.5, 30.0, 1.0],
        "booleans must cast to 1.0/0.0 and rows must interleave"
    );
}

#[test]
fn dataframe_converts_to_f32_too() {
    let df = mixed_frame();
    let (values, _, _, _) = dataframe_to_rowmajor::<f32>(&df).expect("convert as f32");
    assert_eq!(
        values,
        vec![1.5f32, 10.0, 1.0, 2.5, 20.0, 0.0, 3.5, 30.0, 1.0]
    );
}

#[test]
fn series_converts_to_a_flat_vec() {
    let s = Series::new("y".into(), [0i64, 1, 1, 0]);
    assert_eq!(
        series_to_vec::<f64>(&s).expect("convert y"),
        vec![0.0, 1.0, 1.0, 0.0]
    );
}

#[test]
fn row_major_round_trips_through_a_frame() {
    let df = mixed_frame();
    let (values, rows, _, names) = dataframe_to_rowmajor::<f64>(&df).expect("convert");
    let back = rowmajor_to_dataframe(&values, rows, &names).expect("rebuild");
    assert_eq!(column_names(&back), names);
    let (again, _, _, _) = dataframe_to_rowmajor::<f64>(&back).expect("reconvert");
    assert_eq!(again, values, "the round trip must be value-exact");
    // Every column comes back Float64 — the buffer carries one float type, so
    // the original `i32`/`Boolean` dtypes are NOT restored. Asserted rather than
    // left implicit, because it is exactly why `take_columns` exists.
    for col in back.columns() {
        assert_eq!(col.dtype(), &DataType::Float64);
    }
}

#[test]
fn take_columns_keeps_dtypes_and_names() {
    let df = mixed_frame();
    let out = take_columns(&df, &[true, false, true]).expect("take columns");
    assert_eq!(column_names(&out), vec!["a", "c"]);
    assert_eq!(out.height(), 3);
    // The ORIGINAL dtypes survive — this is the whole reason a selector's polars
    // `transform` stays in the frame domain instead of round-tripping through the
    // flat `f64` buffer the scores were computed on.
    assert_eq!(out.column("a").unwrap().dtype(), &DataType::Float64);
    assert_eq!(out.column("c").unwrap().dtype(), &DataType::Boolean);
}

#[test]
fn take_columns_accepts_an_all_false_mask() {
    // sklearn WARNS and returns an `n x 0` result rather than raising, so a
    // selector that selects nothing must still produce a frame.
    let out = take_columns(&mixed_frame(), &[false, false, false]).expect("empty selection");
    assert_eq!(out.width(), 0);
}

#[test]
fn take_columns_rejects_a_wrong_length_mask() {
    let err = take_columns(&mixed_frame(), &[true, false]).expect_err("mask length must match");
    assert!(matches!(
        err,
        FrameError::ShapeMismatch {
            cols: 3,
            len: 2,
            ..
        }
    ));
}

#[test]
fn non_numeric_columns_are_rejected_by_name() {
    let df = df!["a" => [1.0f64, 2.0], "label" => ["x", "y"]].expect("build");
    let err = dataframe_to_rowmajor::<f64>(&df).expect_err("a String column is not numeric");
    match err {
        FrameError::NonNumericColumn { column, dtype } => {
            assert_eq!(column, "label");
            // The dtype is in the message because "column 'label' is a String" is
            // actionable and "conversion failed" is not.
            assert!(
                dtype.contains("str") || dtype.contains("String"),
                "got {dtype}"
            );
        }
        other => panic!("expected NonNumericColumn, got {other:?}"),
    }
}

#[test]
fn nulls_are_rejected_but_nans_pass_through() {
    // A polars NULL is not a number, and substituting one would produce a
    // confidently wrong score.
    let with_null = df!["a" => [Some(1.0f64), None, Some(3.0)]].expect("build");
    let err = dataframe_to_rowmajor::<f64>(&with_null).expect_err("nulls must be rejected");
    assert!(matches!(err, FrameError::NullValues { count: 1, .. }));

    // A float NaN is a different thing and IS allowed: `VarianceThreshold` is
    // documented to accept NaN input (sklearn validates it with
    // `ensure_all_finite="allow-nan"`), so the conversion must not conflate the
    // two.
    let with_nan = df!["a" => [1.0f64, f64::NAN, 3.0]].expect("build");
    let (values, _, _, _) = dataframe_to_rowmajor::<f64>(&with_nan).expect("NaN is not a null");
    assert!(values[1].is_nan());
}

#[test]
fn empty_frames_are_rejected() {
    let no_rows = df!["a" => Vec::<f64>::new()].expect("build");
    assert!(matches!(
        dataframe_to_rowmajor::<f64>(&no_rows),
        Err(FrameError::Empty { rows: 0, .. })
    ));
    let no_cols = DataFrame::default();
    assert!(matches!(
        dataframe_to_rowmajor::<f64>(&no_cols),
        Err(FrameError::Empty { .. })
    ));
}

#[test]
fn rowmajor_to_dataframe_rejects_a_geometry_mismatch() {
    let names = vec!["a".to_string(), "b".to_string()];
    let err = rowmajor_to_dataframe(&[1.0, 2.0, 3.0], 2, &names)
        .expect_err("3 values cannot fill a 2x2 frame");
    assert!(matches!(
        err,
        FrameError::ShapeMismatch {
            rows: 2,
            cols: 2,
            len: 3
        }
    ));
}
