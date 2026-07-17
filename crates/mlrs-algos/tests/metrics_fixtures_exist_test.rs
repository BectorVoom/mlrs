//! TASK-02 (METR-ORACLE-01) — oracle fixture presence + array-name
//! completeness smoke test.
//!
//! Asserts the seven committed `metrics_*.npz` fixtures exist and every named
//! array in PLAN.md's fixture-naming-scheme table decodes as f64 via
//! `OracleCase::f64` (a `None` return means an un-cast integer array slipped
//! through the float-cast rule, Plan-Check Issue 5). This is the hard
//! prerequisite for every later Rust/Python oracle test's Red step to fail
//! for the RIGHT reason (missing implementation, not a missing fixture).
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an
//! in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_core::{load_npz, OracleCase};

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Every named array in `case` decodes as an `f64` slice (a `None` means an
/// un-cast integer array poisoned the whole fixture load per Plan-Check
/// Issue 5).
fn assert_all_present(case: &OracleCase, names: &[&str]) {
    for &name in names {
        assert!(
            case.f64(name).is_some(),
            "fixture '{}' is missing array '{name}' (or it failed to decode as float — \
             check the float-cast rule)",
            case.path()
        );
    }
}

#[test]
fn metrics_cls_binary_fixture_has_expected_arrays() {
    for dtype in ["f32", "f64"] {
        let path = fixture(&format!("metrics_cls_binary_{dtype}_seed42.npz"));
        let case = load_npz(&path)
            .unwrap_or_else(|e| panic!("load metrics_cls_binary_{dtype}_seed42.npz: {e}"));
        assert_all_present(
            &case,
            &[
                "y_true",
                "y_pred",
                "y_score",
                "sample_weight",
                "ref_accuracy",
                "ref_accuracy_sw",
                "ref_confusion",
                "ref_confusion_sw",
                "ref_precision_binary",
                "ref_recall_binary",
                "ref_f1_binary",
                "ref_precision_binary_sw",
                "ref_recall_binary_sw",
                "ref_f1_binary_sw",
                "ref_roc_auc",
                "ref_roc_auc_sw",
                "ref_pr_precision",
                "ref_pr_recall",
                "ref_pr_thresholds",
                "ref_pr_precision_sw",
                "ref_pr_recall_sw",
                "ref_pr_thresholds_sw",
                "ref_log_loss_binary",
                "y_prob_binary",
            ],
        );
    }
}

#[test]
fn metrics_cls_multiclass_fixture_has_expected_arrays() {
    for dtype in ["f32", "f64"] {
        let path = fixture(&format!("metrics_cls_multiclass_{dtype}_seed42.npz"));
        let case = load_npz(&path)
            .unwrap_or_else(|e| panic!("load metrics_cls_multiclass_{dtype}_seed42.npz: {e}"));
        assert_all_present(
            &case,
            &[
                "y_true",
                "y_pred",
                "y_proba",
                "sample_weight",
                "ref_accuracy",
                "ref_accuracy_sw",
                "ref_confusion",
                "ref_precision_macro",
                "ref_precision_micro",
                "ref_precision_weighted",
                "ref_precision_none",
                "ref_recall_macro",
                "ref_recall_micro",
                "ref_recall_weighted",
                "ref_recall_none",
                "ref_f1_macro",
                "ref_f1_micro",
                "ref_f1_weighted",
                "ref_f1_none",
                "ref_precision_macro_sw",
                "ref_recall_macro_sw",
                "ref_f1_macro_sw",
                "ref_log_loss",
                "ref_log_loss_sw",
                "ref_roc_auc_ovr_macro",
                "ref_roc_auc_ovr_weighted",
                "ref_roc_auc_ovo_macro",
                "ref_roc_auc_ovo_weighted",
                "ref_roc_auc_ovr_macro_sw",
                "ref_roc_auc_ovr_weighted_sw",
                "y_true_labelreorder",
                "y_pred_labelreorder",
                "labels_reorder",
                "ref_precision_labelreorder",
                "ref_recall_labelreorder",
                "ref_f1_labelreorder",
            ],
        );
        // ref_roc_auc_ovo_{macro,weighted}_sw are probe-gated (Q10 Branch A/B)
        // — not asserted here unconditionally; TASK-10/21 read the generator
        // docstring to know which branch applies.
    }
}

#[test]
fn metrics_cls_degenerate_fixture_has_expected_arrays() {
    let path = fixture("metrics_cls_degenerate_seed42.npz");
    let case = load_npz(&path).expect("load metrics_cls_degenerate_seed42.npz");
    assert_all_present(
        &case,
        &[
            "y_true_empty",
            "y_pred_empty",
            "labels_empty",
            "ref_confusion_empty",
            "y_true_one",
            "y_pred_one",
            "ref_confusion_one",
            "y_true_zp",
            "y_pred_zp",
            "ref_precision_zerodiv",
            "y_true_zr",
            "y_pred_zr",
            "ref_recall_zerodiv",
            "y_true_zf",
            "y_pred_zf",
            "ref_f1_zerodiv",
            "y_true_single_match",
            "y_pred_single_match",
            "ref_acc_single_match",
            "y_true_single_mismatch",
            "y_pred_single_mismatch",
            "ref_acc_single_mismatch",
            "y_true_singleclass",
            "y_score_singleclass",
            "y_true_clip",
            "y_prob_clip",
            "ref_log_loss_clip",
            "y_true_logloss_labelreorder",
            "y_prob_logloss_labelreorder",
            "labels_logloss_reorder",
            "ref_log_loss_labelreorder",
        ],
    );
}

#[test]
fn metrics_reg_fixture_has_expected_arrays() {
    for dtype in ["f32", "f64"] {
        let path = fixture(&format!("metrics_reg_{dtype}_seed42.npz"));
        let case =
            load_npz(&path).unwrap_or_else(|e| panic!("load metrics_reg_{dtype}_seed42.npz: {e}"));
        assert_all_present(
            &case,
            &[
                "y_true",
                "y_pred",
                "sample_weight",
                "ref_r2",
                "ref_r2_sw",
                "ref_mse",
                "ref_mse_sw",
                "ref_mae",
                "ref_mae_sw",
                "y_true_const",
                "y_pred_const",
                "ref_r2_const",
                "y_perfect",
                "ref_r2_perfect",
                "ref_mse_perfect",
                "ref_mae_perfect",
            ],
        );
    }
}
