//! Classification metrics sklearn-oracle tests (TASK-03..11, METR-CLS-01..09).
//!
//! Replays the committed `metrics_cls_{binary,multiclass}_{f32,f64}_seed42.npz`
//! and `metrics_cls_degenerate_seed42.npz` fixtures (TASK-02) against
//! `mlrs_algos::metrics::classification::*`. Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_algos::metrics::classification::{
    accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score,
    recall_score, roc_auc_score_binary, roc_auc_score_multiclass,
};
use mlrs_algos::metrics::{Average, MetricError, MultiClass, PrfOut, ZeroDivision};
use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// Weighted/general-value tolerance (SPEC §6 tier ≤1e-5).
const TOL: f64 = 1e-5;
/// Exact-tier tolerance for unweighted rational-in-integers comparisons.
const EXACT_TOL: f64 = 1e-9;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn load(name: &str) -> OracleCase {
    load_npz(fixture(name)).unwrap_or_else(|e| panic!("load {name}: {e}"))
}

/// Cast a fixture's float-stored (but integer-valued) label array to `i32`.
fn labels_i32(case: &OracleCase, name: &str) -> Vec<i32> {
    case.expect_f64(name)
        .iter()
        .map(|&v| v.round() as i32)
        .collect()
}

fn f64_vec(case: &OracleCase, name: &str) -> Vec<f64> {
    case.expect_f64(name).to_vec()
}

fn scalar(case: &OracleCase, name: &str) -> f64 {
    case.expect_f64(name)[0]
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} (diff {})",
        (got - want).abs()
    );
}

// ==================== TASK-03 — METR-CLS-01: accuracy_score ====================

#[test]
fn accuracy_score_matches_sklearn_oracle_binary_f64() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let got = accuracy_score(&y_true, &y_pred, None, true);
    assert_close(
        got,
        scalar(&case, "ref_accuracy"),
        EXACT_TOL,
        "accuracy binary f64",
    );
}

#[test]
fn accuracy_score_matches_sklearn_oracle_binary_f32() {
    let case = load("metrics_cls_binary_f32_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let got = accuracy_score(&y_true, &y_pred, None, true);
    assert_close(
        got,
        scalar(&case, "ref_accuracy"),
        1e-4,
        "accuracy binary f32",
    );
}

#[test]
fn accuracy_score_matches_sklearn_oracle_binary_f64_gated() {
    if capability::skip_f64_with_log() {
        return;
    }
    accuracy_score_matches_sklearn_oracle_binary_f64();
}

#[test]
fn accuracy_score_weighted_matches_sklearn_oracle() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    let got = accuracy_score(&y_true, &y_pred, Some(&sw), true);
    assert_close(
        got,
        scalar(&case, "ref_accuracy_sw"),
        TOL,
        "accuracy weighted",
    );
}

#[test]
fn accuracy_score_single_sample_degenerate() {
    let case = load("metrics_cls_degenerate_seed42.npz");
    let y_true_m = labels_i32(&case, "y_true_single_match");
    let y_pred_m = labels_i32(&case, "y_pred_single_match");
    assert_close(
        accuracy_score(&y_true_m, &y_pred_m, None, true),
        scalar(&case, "ref_acc_single_match"),
        EXACT_TOL,
        "single match",
    );
    let y_true_mm = labels_i32(&case, "y_true_single_mismatch");
    let y_pred_mm = labels_i32(&case, "y_pred_single_mismatch");
    assert_close(
        accuracy_score(&y_true_mm, &y_pred_mm, None, true),
        scalar(&case, "ref_acc_single_mismatch"),
        EXACT_TOL,
        "single mismatch",
    );
}

#[test]
fn accuracy_score_matches_sklearn_oracle_multiclass_f64() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let got = accuracy_score(&y_true, &y_pred, None, true);
    assert_close(
        got,
        scalar(&case, "ref_accuracy"),
        EXACT_TOL,
        "accuracy multiclass f64",
    );
}

// ==================== TASK-04 — METR-CLS-02: confusion_matrix ====================

fn flatten(m: &[Vec<f64>]) -> Vec<f64> {
    m.iter().flatten().copied().collect()
}

fn assert_matrix_close(got: &[Vec<f64>], want_flat: &[f64], n: usize, tol: f64, what: &str) {
    let got_flat = flatten(got);
    assert_eq!(got_flat.len(), want_flat.len(), "{what}: shape mismatch");
    for i in 0..n * n {
        assert!(
            (got_flat[i] - want_flat[i]).abs() <= tol,
            "{what}[{i}]: got {}, want {}",
            got_flat[i],
            want_flat[i]
        );
    }
}

#[test]
fn confusion_matrix_empty_class_via_explicit_labels() {
    let case = load("metrics_cls_degenerate_seed42.npz");
    let y_true = labels_i32(&case, "y_true_empty");
    let y_pred = labels_i32(&case, "y_pred_empty");
    let labels = labels_i32(&case, "labels_empty");
    let got = confusion_matrix(&y_true, &y_pred, Some(&labels), None);
    assert_matrix_close(
        &got,
        &f64_vec(&case, "ref_confusion_empty"),
        3,
        EXACT_TOL,
        "confusion_empty",
    );
}

#[test]
fn confusion_matrix_all_one_class() {
    let case = load("metrics_cls_degenerate_seed42.npz");
    let y_true = labels_i32(&case, "y_true_one");
    let y_pred = labels_i32(&case, "y_pred_one");
    let got = confusion_matrix(&y_true, &y_pred, None, None);
    assert_matrix_close(
        &got,
        &f64_vec(&case, "ref_confusion_one"),
        1,
        EXACT_TOL,
        "confusion_one",
    );
}

#[test]
fn confusion_matrix_matches_sklearn_oracle_binary() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let got = confusion_matrix(&y_true, &y_pred, None, None);
    assert_matrix_close(
        &got,
        &f64_vec(&case, "ref_confusion"),
        2,
        EXACT_TOL,
        "confusion binary",
    );
}

#[test]
fn confusion_matrix_weighted_matches_sklearn_oracle_binary() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    let got = confusion_matrix(&y_true, &y_pred, None, Some(&sw));
    assert_matrix_close(
        &got,
        &f64_vec(&case, "ref_confusion_sw"),
        2,
        TOL,
        "confusion binary weighted",
    );
}

#[test]
fn confusion_matrix_matches_sklearn_oracle_multiclass() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let got = confusion_matrix(&y_true, &y_pred, None, None);
    assert_matrix_close(
        &got,
        &f64_vec(&case, "ref_confusion"),
        3,
        EXACT_TOL,
        "confusion multiclass",
    );
}

// ==================== TASK-05/06/07 — precision/recall/f1 ====================

fn prf_scalar(out: PrfOut) -> f64 {
    match out {
        PrfOut::Scalar(v) => v,
        PrfOut::PerClass(_) => panic!("expected PrfOut::Scalar"),
    }
}

fn prf_per_class(out: PrfOut) -> Vec<f64> {
    match out {
        PrfOut::PerClass(v) => v,
        PrfOut::Scalar(_) => panic!("expected PrfOut::PerClass"),
    }
}

#[test]
fn precision_score_zero_division_no_predicted_positives() {
    let case = load("metrics_cls_degenerate_seed42.npz");
    let y_true = labels_i32(&case, "y_true_zp");
    let y_pred = labels_i32(&case, "y_pred_zp");
    let got = precision_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Binary,
        None,
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_precision_zerodiv"),
        EXACT_TOL,
        "precision zerodiv",
    );
}

#[test]
fn precision_score_binary_matches_sklearn_oracle() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let got = precision_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Binary,
        None,
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_precision_binary"),
        TOL,
        "precision binary",
    );
}

#[test]
fn precision_score_binary_weighted_matches_sklearn_oracle() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    let got = precision_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Binary,
        Some(&sw),
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_precision_binary_sw"),
        TOL,
        "precision binary sw",
    );
}

fn multiclass_avg(name: &str) -> Average {
    match name {
        "macro" => Average::Macro,
        "micro" => Average::Micro,
        "weighted" => Average::Weighted,
        "none" => Average::None_,
        other => panic!("unknown average {other}"),
    }
}

#[test]
fn precision_score_averages_matches_sklearn_oracle_multiclass() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    for avg in ["macro", "micro", "weighted"] {
        let got = precision_score(
            &y_true,
            &y_pred,
            None,
            1,
            multiclass_avg(avg),
            None,
            ZeroDivision::Zero,
        );
        assert_close(
            prf_scalar(got),
            scalar(&case, &format!("ref_precision_{avg}")),
            TOL,
            &format!("precision {avg}"),
        );
    }
    let got_none = precision_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::None_,
        None,
        ZeroDivision::Zero,
    );
    let want = f64_vec(&case, "ref_precision_none");
    let got_vec = prf_per_class(got_none);
    assert_eq!(got_vec.len(), want.len(), "precision none length");
    for i in 0..want.len() {
        assert_close(got_vec[i], want[i], TOL, &format!("precision none[{i}]"));
    }
}

#[test]
fn precision_score_macro_weighted_matches_sklearn_oracle() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    let got = precision_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Macro,
        Some(&sw),
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_precision_macro_sw"),
        TOL,
        "precision macro sw",
    );
}

#[test]
fn precision_score_labels_reorder_matches_sklearn_oracle() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true_labelreorder");
    let y_pred = labels_i32(&case, "y_pred_labelreorder");
    let labels = labels_i32(&case, "labels_reorder");
    let got = precision_score(
        &y_true,
        &y_pred,
        Some(&labels),
        1,
        Average::Macro,
        None,
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_precision_labelreorder"),
        TOL,
        "precision labelreorder",
    );
}

#[test]
fn recall_score_zero_division_no_true_positives() {
    let case = load("metrics_cls_degenerate_seed42.npz");
    let y_true = labels_i32(&case, "y_true_zr");
    let y_pred = labels_i32(&case, "y_pred_zr");
    let got = recall_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Binary,
        None,
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_recall_zerodiv"),
        EXACT_TOL,
        "recall zerodiv",
    );
}

#[test]
fn recall_score_binary_matches_sklearn_oracle() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let got = recall_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Binary,
        None,
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_recall_binary"),
        TOL,
        "recall binary",
    );
}

#[test]
fn recall_score_binary_weighted_matches_sklearn_oracle() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    let got = recall_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Binary,
        Some(&sw),
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_recall_binary_sw"),
        TOL,
        "recall binary sw",
    );
}

#[test]
fn recall_score_averages_matches_sklearn_oracle_multiclass() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    for avg in ["macro", "micro", "weighted"] {
        let got = recall_score(
            &y_true,
            &y_pred,
            None,
            1,
            multiclass_avg(avg),
            None,
            ZeroDivision::Zero,
        );
        assert_close(
            prf_scalar(got),
            scalar(&case, &format!("ref_recall_{avg}")),
            TOL,
            &format!("recall {avg}"),
        );
    }
    let got_none = recall_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::None_,
        None,
        ZeroDivision::Zero,
    );
    let want = f64_vec(&case, "ref_recall_none");
    let got_vec = prf_per_class(got_none);
    for i in 0..want.len() {
        assert_close(got_vec[i], want[i], TOL, &format!("recall none[{i}]"));
    }
}

#[test]
fn recall_score_macro_weighted_matches_sklearn_oracle() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    let got = recall_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Macro,
        Some(&sw),
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_recall_macro_sw"),
        TOL,
        "recall macro sw",
    );
}

#[test]
fn recall_score_labels_reorder_matches_sklearn_oracle() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true_labelreorder");
    let y_pred = labels_i32(&case, "y_pred_labelreorder");
    let labels = labels_i32(&case, "labels_reorder");
    let got = recall_score(
        &y_true,
        &y_pred,
        Some(&labels),
        1,
        Average::Macro,
        None,
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_recall_labelreorder"),
        TOL,
        "recall labelreorder",
    );
}

#[test]
fn f1_score_computed_from_tp_fp_fn_not_precision_times_recall() {
    // tp=1,fp=2,fn=0: direct formula 2*1/(2*1+2+0) = 2/4 = 0.5 exactly.
    // A naive P*R-derived value would independently round P=1/3, R=1.0 and
    // compute 2*P*R/(P+R) — still 0.5 in exact real arithmetic, but at f32
    // precision independently-rounded floats can diverge from the direct
    // formula's f64 accumulation. We assert the EXACT direct-formula value
    // at f64 tightness (1e-7) to lock in the "computed once, not composed"
    // contract (SPEC §5 CLS-05 note).
    let y_true = [1i32, 0, 0];
    let y_pred = [1i32, 1, 1];
    // class 1: tp=1 (row0), fp=2 (rows1,2 predict 1 but true 0), fn=0.
    let got = f1_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Binary,
        None,
        ZeroDivision::Zero,
    );
    assert_close(prf_scalar(got), 0.5, 1e-7, "f1 direct-formula");
}

#[test]
fn f1_score_zero_division_degenerate() {
    let case = load("metrics_cls_degenerate_seed42.npz");
    let y_true = labels_i32(&case, "y_true_zf");
    let y_pred = labels_i32(&case, "y_pred_zf");
    let got = f1_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Binary,
        None,
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_f1_zerodiv"),
        EXACT_TOL,
        "f1 zerodiv",
    );
}

#[test]
fn f1_score_binary_matches_sklearn_oracle() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let got = f1_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Binary,
        None,
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_f1_binary"),
        TOL,
        "f1 binary",
    );
}

#[test]
fn f1_score_binary_weighted_matches_sklearn_oracle() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    let got = f1_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Binary,
        Some(&sw),
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_f1_binary_sw"),
        TOL,
        "f1 binary sw",
    );
}

#[test]
fn f1_score_averages_matches_sklearn_oracle_multiclass() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    for avg in ["macro", "micro", "weighted"] {
        let got = f1_score(
            &y_true,
            &y_pred,
            None,
            1,
            multiclass_avg(avg),
            None,
            ZeroDivision::Zero,
        );
        assert_close(
            prf_scalar(got),
            scalar(&case, &format!("ref_f1_{avg}")),
            TOL,
            &format!("f1 {avg}"),
        );
    }
    let got_none = f1_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::None_,
        None,
        ZeroDivision::Zero,
    );
    let want = f64_vec(&case, "ref_f1_none");
    let got_vec = prf_per_class(got_none);
    for i in 0..want.len() {
        assert_close(got_vec[i], want[i], TOL, &format!("f1 none[{i}]"));
    }
}

#[test]
fn f1_score_macro_weighted_matches_sklearn_oracle() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    let got = f1_score(
        &y_true,
        &y_pred,
        None,
        1,
        Average::Macro,
        Some(&sw),
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_f1_macro_sw"),
        TOL,
        "f1 macro sw",
    );
}

#[test]
fn f1_score_labels_reorder_matches_sklearn_oracle() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true_labelreorder");
    let y_pred = labels_i32(&case, "y_pred_labelreorder");
    let labels = labels_i32(&case, "labels_reorder");
    let got = f1_score(
        &y_true,
        &y_pred,
        Some(&labels),
        1,
        Average::Macro,
        None,
        ZeroDivision::Zero,
    );
    assert_close(
        prf_scalar(got),
        scalar(&case, "ref_f1_labelreorder"),
        TOL,
        "f1 labelreorder",
    );
}

// ==================== TASK-08 — METR-CLS-06: log_loss ====================

/// TASK-08 empirical finding (documented deviation from SPEC Q5's assumed
/// `1e-15`): `scikit-learn==1.9.0`'s `log_loss` clips using the MACHINE
/// EPSILON of `y_proba`'s dtype (`np.finfo(np.float64).eps ≈
/// 2.220446049250313e-16`, i.e. Rust's `f64::EPSILON` exactly), not a fixed
/// `1e-15` — confirmed by reading the installed sklearn's `_log_loss` source
/// (`eps = xp.finfo(y_proba.dtype).eps`) and by this test failing at
/// `1e-15` and passing at `f64::EPSILON`. The Rust `log_loss` function
/// itself still takes a general `eps: f64` parameter (SPEC §4 contract
/// unchanged); only the DEFAULT value callers should pass to match sklearn
/// changes (TASK-15/16/19's `eps='auto'` PyO3/shim mapping must use
/// `f64::EPSILON`, not `1e-15`).
#[test]
fn log_loss_clips_zero_and_one_probabilities_to_finite_value() {
    let case = load("metrics_cls_degenerate_seed42.npz");
    let y_true = labels_i32(&case, "y_true_clip");
    let y_prob = f64_vec(&case, "y_prob_clip");
    let got = log_loss(&y_true, &y_prob, 2, None, None, f64::EPSILON, true);
    assert!(got.is_finite(), "log_loss clip must be finite, got {got}");
    assert_close(
        got,
        scalar(&case, "ref_log_loss_clip"),
        TOL,
        "log_loss clip",
    );
}

#[test]
fn log_loss_matches_sklearn_oracle_multiclass() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_prob = f64_vec(&case, "y_proba");
    let got = log_loss(&y_true, &y_prob, 3, None, None, f64::EPSILON, true);
    assert_close(
        got,
        scalar(&case, "ref_log_loss"),
        TOL,
        "log_loss multiclass",
    );
}

#[test]
fn log_loss_weighted_matches_sklearn_oracle() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_prob = f64_vec(&case, "y_proba");
    let sw = f64_vec(&case, "sample_weight");
    let got = log_loss(&y_true, &y_prob, 3, None, Some(&sw), f64::EPSILON, true);
    assert_close(
        got,
        scalar(&case, "ref_log_loss_sw"),
        TOL,
        "log_loss weighted",
    );
}

#[test]
fn log_loss_matches_sklearn_oracle_binary() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_prob = f64_vec(&case, "y_prob_binary");
    let got = log_loss(&y_true, &y_prob, 2, None, None, f64::EPSILON, true);
    assert_close(
        got,
        scalar(&case, "ref_log_loss_binary"),
        TOL,
        "log_loss binary",
    );
}

#[test]
fn log_loss_labels_reorder_matches_sklearn_oracle() {
    let case = load("metrics_cls_degenerate_seed42.npz");
    let y_true = labels_i32(&case, "y_true_logloss_labelreorder");
    let y_prob = f64_vec(&case, "y_prob_logloss_labelreorder");
    let labels = labels_i32(&case, "labels_logloss_reorder");
    let got = log_loss(&y_true, &y_prob, 2, Some(&labels), None, f64::EPSILON, true);
    assert_close(
        got,
        scalar(&case, "ref_log_loss_labelreorder"),
        TOL,
        "log_loss labelreorder",
    );
}

// ==================== TASK-09 — METR-CLS-07: roc_auc_score (binary) ====================

#[test]
fn roc_auc_score_binary_single_class_returns_error() {
    let case = load("metrics_cls_degenerate_seed42.npz");
    let y_true = labels_i32(&case, "y_true_singleclass");
    let y_score = f64_vec(&case, "y_score_singleclass");
    let got = roc_auc_score_binary(&y_true, &y_score, 1, None);
    assert!(
        matches!(got, Err(MetricError::SingleClassRocAuc)),
        "expected SingleClassRocAuc, got {got:?}"
    );
}

#[test]
fn roc_auc_score_binary_matches_sklearn_oracle_tie_heavy() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_score = f64_vec(&case, "y_score");
    let got = roc_auc_score_binary(&y_true, &y_score, 1, None).expect("roc_auc_score_binary");
    assert_close(got, scalar(&case, "ref_roc_auc"), TOL, "roc_auc binary");
}

#[test]
fn roc_auc_score_binary_weighted_matches_sklearn_oracle() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_score = f64_vec(&case, "y_score");
    let sw = f64_vec(&case, "sample_weight");
    let got = roc_auc_score_binary(&y_true, &y_score, 1, Some(&sw))
        .expect("roc_auc_score_binary weighted");
    assert_close(
        got,
        scalar(&case, "ref_roc_auc_sw"),
        TOL,
        "roc_auc binary weighted",
    );
}

// ==================== TASK-10 — METR-CLS-08: roc_auc_score (multiclass) ====================

#[test]
fn roc_auc_score_multiclass_ovr_macro_matches_sklearn_oracle() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_proba = f64_vec(&case, "y_proba");
    let got = roc_auc_score_multiclass(&y_true, &y_proba, 3, MultiClass::Ovr, Average::Macro, None)
        .expect("ovr macro");
    assert_close(
        got,
        scalar(&case, "ref_roc_auc_ovr_macro"),
        TOL,
        "roc_auc ovr macro",
    );
}

#[test]
fn roc_auc_score_multiclass_ovr_weighted_matches_sklearn_oracle() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_proba = f64_vec(&case, "y_proba");
    let got = roc_auc_score_multiclass(
        &y_true,
        &y_proba,
        3,
        MultiClass::Ovr,
        Average::Weighted,
        None,
    )
    .expect("ovr weighted");
    assert_close(
        got,
        scalar(&case, "ref_roc_auc_ovr_weighted"),
        TOL,
        "roc_auc ovr weighted",
    );
}

#[test]
fn roc_auc_score_multiclass_ovo_macro_matches_sklearn_oracle() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_proba = f64_vec(&case, "y_proba");
    let got = roc_auc_score_multiclass(&y_true, &y_proba, 3, MultiClass::Ovo, Average::Macro, None)
        .expect("ovo macro");
    assert_close(
        got,
        scalar(&case, "ref_roc_auc_ovo_macro"),
        TOL,
        "roc_auc ovo macro",
    );
}

#[test]
fn roc_auc_score_multiclass_ovo_weighted_matches_sklearn_oracle() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_proba = f64_vec(&case, "y_proba");
    let got = roc_auc_score_multiclass(
        &y_true,
        &y_proba,
        3,
        MultiClass::Ovo,
        Average::Weighted,
        None,
    )
    .expect("ovo weighted");
    assert_close(
        got,
        scalar(&case, "ref_roc_auc_ovo_weighted"),
        TOL,
        "roc_auc ovo weighted",
    );
}

#[test]
fn roc_auc_score_multiclass_ovr_weighted_sample_weight_matches_sklearn_oracle() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_proba = f64_vec(&case, "y_proba");
    let sw = f64_vec(&case, "sample_weight");
    let got_macro = roc_auc_score_multiclass(
        &y_true,
        &y_proba,
        3,
        MultiClass::Ovr,
        Average::Macro,
        Some(&sw),
    )
    .expect("ovr macro sw");
    assert_close(
        got_macro,
        scalar(&case, "ref_roc_auc_ovr_macro_sw"),
        TOL,
        "roc_auc ovr macro sw",
    );
    let got_weighted = roc_auc_score_multiclass(
        &y_true,
        &y_proba,
        3,
        MultiClass::Ovr,
        Average::Weighted,
        Some(&sw),
    )
    .expect("ovr weighted sw");
    assert_close(
        got_weighted,
        scalar(&case, "ref_roc_auc_ovr_weighted_sw"),
        TOL,
        "roc_auc ovr weighted sw",
    );
}

/// OvO + `sample_weight` carve-out (Plan-Check Issue 2 / Q10): TASK-02's probe
/// found scikit-learn==1.9.0 RAISES on `roc_auc_score(multi_class='ovo',
/// sample_weight=...)` — Branch A. No `ref_roc_auc_ovo_*_sw` fixture value
/// exists; assert the `Err(MetricError::WeightedOvoUnsupported)` gate.
#[test]
fn roc_auc_score_multiclass_ovo_weighted_sample_weight_gate() {
    let case = load("metrics_cls_multiclass_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_proba = f64_vec(&case, "y_proba");
    let sw = f64_vec(&case, "sample_weight");
    let got = roc_auc_score_multiclass(
        &y_true,
        &y_proba,
        3,
        MultiClass::Ovo,
        Average::Macro,
        Some(&sw),
    );
    assert!(
        matches!(got, Err(MetricError::WeightedOvoUnsupported)),
        "expected WeightedOvoUnsupported, got {got:?}"
    );
}

// ==================== TASK-11 — METR-CLS-09: precision_recall_curve ====================

#[test]
fn precision_recall_curve_sentinel_and_length_invariants() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_score = f64_vec(&case, "y_score");
    let (precision, recall, thresholds) = precision_recall_curve(&y_true, &y_score, 1, None);
    assert_eq!(precision.len(), thresholds.len() + 1, "precision length");
    assert_eq!(recall.len(), thresholds.len() + 1, "recall length");
    assert_eq!(precision.last(), Some(&1.0), "precision sentinel");
    assert_eq!(recall.last(), Some(&0.0), "recall sentinel");
    for w in thresholds.windows(2) {
        assert!(w[0] <= w[1], "thresholds must be non-decreasing: {w:?}");
    }
}

#[test]
fn precision_recall_curve_matches_sklearn_oracle_tie_heavy() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_score = f64_vec(&case, "y_score");
    let (precision, recall, thresholds) = precision_recall_curve(&y_true, &y_score, 1, None);
    let want_p = f64_vec(&case, "ref_pr_precision");
    let want_r = f64_vec(&case, "ref_pr_recall");
    let want_t = f64_vec(&case, "ref_pr_thresholds");
    assert_eq!(precision.len(), want_p.len(), "precision length vs oracle");
    assert_eq!(recall.len(), want_r.len(), "recall length vs oracle");
    assert_eq!(
        thresholds.len(),
        want_t.len(),
        "thresholds length vs oracle"
    );
    for i in 0..want_p.len() {
        assert_close(precision[i], want_p[i], TOL, &format!("pr_precision[{i}]"));
        assert_close(recall[i], want_r[i], TOL, &format!("pr_recall[{i}]"));
    }
    for i in 0..want_t.len() {
        assert_close(
            thresholds[i],
            want_t[i],
            TOL,
            &format!("pr_thresholds[{i}]"),
        );
    }
}

#[test]
fn precision_recall_curve_weighted_matches_sklearn_oracle() {
    let case = load("metrics_cls_binary_f64_seed42.npz");
    let y_true = labels_i32(&case, "y_true");
    let y_score = f64_vec(&case, "y_score");
    let sw = f64_vec(&case, "sample_weight");
    let (precision, recall, thresholds) = precision_recall_curve(&y_true, &y_score, 1, Some(&sw));
    let want_p = f64_vec(&case, "ref_pr_precision_sw");
    let want_r = f64_vec(&case, "ref_pr_recall_sw");
    let want_t = f64_vec(&case, "ref_pr_thresholds_sw");
    assert_eq!(
        precision.len(),
        want_p.len(),
        "precision_sw length vs oracle"
    );
    assert_eq!(recall.len(), want_r.len(), "recall_sw length vs oracle");
    assert_eq!(
        thresholds.len(),
        want_t.len(),
        "thresholds_sw length vs oracle"
    );
    for i in 0..want_p.len() {
        assert_close(
            precision[i],
            want_p[i],
            TOL,
            &format!("pr_precision_sw[{i}]"),
        );
        assert_close(recall[i], want_r[i], TOL, &format!("pr_recall_sw[{i}]"));
    }
    for i in 0..want_t.len() {
        assert_close(
            thresholds[i],
            want_t[i],
            TOL,
            &format!("pr_thresholds_sw[{i}]"),
        );
    }
}
