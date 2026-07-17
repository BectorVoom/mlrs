//! TASK-01 (METR-INFRA-01) — shared metrics-module bookkeeping.
//!
//! Exercises the shared label/weight bookkeeping substrate every
//! classification metric (TASK-03..11) builds on: sorted unique-class
//! discovery (or an explicit `labels` order, including absent classes) and
//! per-class weighted TP/FP/FN accumulation. No metric VALUE logic (accuracy,
//! confusion, etc.) is exercised here — that lives in
//! `metrics_classification_test.rs` / `metrics_regression_test.rs` starting
//! TASK-03/TASK-12.
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an
//! in-source `#[cfg(test)] mod tests`.

use mlrs_algos::metrics::{class_bookkeeping, MetricError};

/// Hand-built `y_true = [0,1,0,2]`, `y_pred = [0,1,1,2]`, no `sample_weight`,
/// no explicit `labels`: sorted unique classes `[0,1,2]`; per-class weighted
/// TP/FP/FN equal to their unweighted integer counts.
#[test]
fn unique_classes_and_weighted_counts_from_labels() {
    let y_true = [0i32, 1, 0, 2];
    let y_pred = [0i32, 1, 1, 2];

    let bk = class_bookkeeping(&y_true, &y_pred, None, None).expect("bookkeeping succeeds");

    assert_eq!(bk.classes, vec![0, 1, 2], "sorted unique classes");

    // class 0: true positions {0,2}, pred positions {0}.
    //   TP (true==0 & pred==0): row0 -> yes. row2 -> pred=1, no. TP=1.
    //   FP (true!=0 & pred==0): none other predicts 0. FP=0.
    //   FN (true==0 & pred!=0): row2 (true=0, pred=1). FN=1.
    // class 1: true positions {1}, pred positions {1,2}.
    //   TP: row1 (true=1,pred=1) -> yes. TP=1.
    //   FP: row2 (true=0,pred=1) -> yes. FP=1.
    //   FN: none (row1 correctly predicted). FN=0.
    // class 2: true positions {3}, pred positions {3}.
    //   TP=1, FP=0, FN=0.
    let expected = [
        (0i32, 1.0f64, 0.0f64, 1.0f64),
        (1, 1.0, 1.0, 0.0),
        (2, 1.0, 0.0, 0.0),
    ];
    for (i, &(class, tp, fp, fnv)) in expected.iter().enumerate() {
        assert_eq!(bk.classes[i], class, "class order at index {i}");
        assert!((bk.tp[i] - tp).abs() < 1e-12, "class {class} tp");
        assert!((bk.fp[i] - fp).abs() < 1e-12, "class {class} fp");
        assert!((bk.fnn[i] - fnv).abs() < 1e-12, "class {class} fn");
    }
}

/// A length mismatch or a negative/NaN `sample_weight` entry returns a typed
/// `MetricError` — no panic (SPEC §5 explicit no-panic requirement).
#[test]
fn length_mismatch_and_bad_weight_return_typed_errors() {
    let y_true = [0i32, 1, 2];
    let y_pred = [0i32, 1];
    assert!(matches!(
        class_bookkeeping(&y_true, &y_pred, None, None),
        Err(MetricError::LengthMismatch)
    ));

    let y_true2 = [0i32, 1, 0];
    let y_pred2 = [0i32, 1, 1];
    let bad_weight_negative = [1.0, 1.0, -1.0];
    assert!(matches!(
        class_bookkeeping(&y_true2, &y_pred2, Some(&bad_weight_negative), None),
        Err(MetricError::InvalidWeight)
    ));

    let bad_weight_nan = [1.0, f64::NAN, 1.0];
    assert!(matches!(
        class_bookkeeping(&y_true2, &y_pred2, Some(&bad_weight_nan), None),
        Err(MetricError::InvalidWeight)
    ));
}

/// An explicit `labels=[0,1,2]` where class `2` never appears in `y_true`/
/// `y_pred` still reports class `2` with `(tp,fp,fn)=(0,0,0)`.
#[test]
fn explicit_labels_include_absent_class_with_zero_counts() {
    let y_true = [0i32, 1, 0, 1];
    let y_pred = [0i32, 1, 1, 0];
    let labels = [0i32, 1, 2];

    let bk =
        class_bookkeeping(&y_true, &y_pred, None, Some(&labels)).expect("bookkeeping succeeds");
    assert_eq!(bk.classes, vec![0, 1, 2]);
    let idx2 = bk
        .classes
        .iter()
        .position(|&c| c == 2)
        .expect("class 2 present");
    assert_eq!(bk.tp[idx2], 0.0);
    assert_eq!(bk.fp[idx2], 0.0);
    assert_eq!(bk.fnn[idx2], 0.0);
}
