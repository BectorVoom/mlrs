//! Full-parameter-surface metrics oracle tests (METR-PARAM-01).
//!
//! Replays the committed `metrics_params_{f32,f64}_seed42.npz` fixture — whose
//! every reference value was read off the pinned `scikit-learn==1.9.0` by
//! `scripts/gen_oracle.py::gen_metrics_params` — against the parameters this
//! plan added to `mlrs_algos::metrics`:
//!
//! * `confusion_matrix(normalize=)` — `'true'` / `'pred'` / `'all'`
//! * `precision/recall/f1(average=)` — `'binary'`/`'micro'`/`'macro'`/
//!   `'weighted'`/`None`, and `zero_division` in all four forms
//! * `roc_auc_score(multi_class=, average=, max_fpr=, labels=)`
//! * `precision_recall_curve(drop_intermediate=)`
//! * `r2/mse/mae(multioutput=)` and `r2_score(force_finite=)`
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an
//! in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_algos::metrics::classification::{
    confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score_binary, roc_auc_score_multiclass,
};
use mlrs_algos::metrics::regression as reg;
use mlrs_algos::metrics::{
    Average, MetricError, MetricOut, MultiClass, MultiOutput, Normalize, PrfOut, PrfResult,
    ZeroDivision,
};
use mlrs_core::{load_npz, OracleCase};

/// Weighted/general-value tolerance (SPEC §6 tier ≤1e-5).
const TOL: f64 = 1e-5;
/// The f32 fixture stores f32-rounded inputs AND f32-computed sklearn
/// references, so the f64-accumulating replay lands within this looser band.
const ATOL_F32: f64 = 1e-4;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn load_f64() -> OracleCase {
    load_npz(fixture("metrics_params_f64_seed42.npz")).expect("load metrics_params_f64")
}

fn load_f32() -> OracleCase {
    load_npz(fixture("metrics_params_f32_seed42.npz")).expect("load metrics_params_f32")
}

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

/// Compare against a reference that may be non-finite: `-inf`/`NaN` must match
/// EXACTLY (they are the whole point of `force_finite=false`), finite values
/// within `tol`.
fn assert_close_or_nonfinite(got: f64, want: f64, tol: f64, what: &str) {
    if want.is_nan() {
        assert!(got.is_nan(), "{what}: got {got}, want NaN");
    } else if want.is_infinite() {
        assert_eq!(got, want, "{what}");
    } else {
        assert_close(got, want, tol, what);
    }
}

fn assert_vec_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} vs {}",
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_close_or_nonfinite(g, w, tol, &format!("{what}[{i}]"));
    }
}

fn prf_scalar(res: PrfResult) -> f64 {
    match res.out {
        PrfOut::Scalar(v) => v,
        PrfOut::PerClass(v) => panic!("expected a scalar, got {v:?}"),
    }
}

fn prf_per_class(res: PrfResult) -> Vec<f64> {
    match res.out {
        PrfOut::PerClass(v) => v,
        PrfOut::Scalar(v) => panic!("expected a per-class vector, got {v}"),
    }
}

fn auc_scalar(out: PrfOut) -> f64 {
    match out {
        PrfOut::Scalar(v) => v,
        PrfOut::PerClass(v) => panic!("expected a scalar, got {v:?}"),
    }
}

fn reg_scalar(out: MetricOut) -> f64 {
    match out {
        MetricOut::Scalar(v) => v,
        MetricOut::Raw(v) => panic!("expected a reduced scalar, got {v:?}"),
    }
}

fn reg_raw(out: MetricOut) -> Vec<f64> {
    match out {
        MetricOut::Raw(v) => v,
        MetricOut::Scalar(v) => panic!("expected a per-output vector, got {v}"),
    }
}

const CLASSES4: [i32; 4] = [0, 1, 2, 3];

// ==================== confusion_matrix(normalize=) ====================

fn assert_confusion_normalized(tag: &str, normalize: Normalize, weighted: bool) {
    let case = load_f64();
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    let got = confusion_matrix(
        &y_true,
        &y_pred,
        None,
        if weighted { Some(&sw) } else { None },
        Some(normalize),
    )
    .expect("confusion_matrix");
    let key = if weighted {
        format!("ref_cm_{tag}_sw")
    } else {
        format!("ref_cm_{tag}")
    };
    let want = f64_vec(&case, &key);
    let flat: Vec<f64> = got.iter().flat_map(|row| row.iter().copied()).collect();
    assert_vec_close(&flat, &want, TOL, &key);
}

#[test]
fn confusion_matrix_normalize_true_matches_sklearn_oracle() {
    assert_confusion_normalized("true", Normalize::True_, false);
    assert_confusion_normalized("true", Normalize::True_, true);
}

#[test]
fn confusion_matrix_normalize_pred_matches_sklearn_oracle() {
    assert_confusion_normalized("pred", Normalize::Pred, false);
    assert_confusion_normalized("pred", Normalize::Pred, true);
}

#[test]
fn confusion_matrix_normalize_all_matches_sklearn_oracle() {
    assert_confusion_normalized("all", Normalize::All, false);
    assert_confusion_normalized("all", Normalize::All, true);
}

#[test]
fn confusion_matrix_normalize_zero_divisor_is_zero_not_nan() {
    // Class 2 appears in neither y_true nor y_pred, so its row (and column)
    // sums to zero. sklearn `nan_to_num`s the resulting 0/0 to 0.0.
    let y_true = [0i32, 0, 1];
    let y_pred = [0i32, 1, 1];
    for normalize in [Normalize::True_, Normalize::Pred, Normalize::All] {
        let got = confusion_matrix(&y_true, &y_pred, Some(&[0, 1, 2]), None, Some(normalize))
            .expect("confusion_matrix");
        for (i, row) in got.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "cell ({i},{j}) is {v}, expected a finite value"
                );
            }
        }
        assert_eq!(
            got[2],
            vec![0.0, 0.0, 0.0],
            "empty class row must be all zero"
        );
    }
}

// ==================== precision/recall/f1(average=) ====================

fn prf_of(
    name: &str,
    y_true: &[i32],
    y_pred: &[i32],
    average: Average,
    sample_weight: Option<&[f64]>,
) -> PrfResult {
    let f = match name {
        "precision" => precision_score,
        "recall" => recall_score,
        "f1" => f1_score,
        other => panic!("unknown metric {other}"),
    };
    f(
        y_true,
        y_pred,
        None,
        1,
        average,
        sample_weight,
        ZeroDivision::Zero,
    )
    .expect("prf")
}

#[test]
fn prf_every_average_matches_sklearn_oracle() {
    let case = load_f64();
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    for name in ["precision", "recall", "f1"] {
        for (tag, average) in [
            ("micro", Average::Micro),
            ("macro", Average::Macro),
            ("weighted", Average::Weighted),
        ] {
            let got = prf_scalar(prf_of(name, &y_true, &y_pred, average, None));
            assert_close(got, scalar(&case, &format!("ref_{name}_{tag}")), TOL, tag);
            let got_sw = prf_scalar(prf_of(name, &y_true, &y_pred, average, Some(&sw)));
            assert_close(
                got_sw,
                scalar(&case, &format!("ref_{name}_{tag}_sw")),
                TOL,
                tag,
            );
        }
        let got_none = prf_per_class(prf_of(name, &y_true, &y_pred, Average::None_, None));
        assert_vec_close(
            &got_none,
            &f64_vec(&case, &format!("ref_{name}_none")),
            TOL,
            &format!("{name} average=None"),
        );
        let got_none_sw = prf_per_class(prf_of(name, &y_true, &y_pred, Average::None_, Some(&sw)));
        assert_vec_close(
            &got_none_sw,
            &f64_vec(&case, &format!("ref_{name}_none_sw")),
            TOL,
            &format!("{name} average=None weighted"),
        );
    }
}

#[test]
fn prf_average_binary_matches_sklearn_oracle() {
    let case = load_f64();
    let y_true = labels_i32(&case, "y_true_bin");
    let y_pred = labels_i32(&case, "y_pred_bin");
    for name in ["precision", "recall", "f1"] {
        let got = prf_scalar(prf_of(name, &y_true, &y_pred, Average::Binary, None));
        assert_close(
            got,
            scalar(&case, &format!("ref_{name}_binary")),
            TOL,
            &format!("{name} binary"),
        );
    }
}

#[test]
fn zero_division_policies_match_sklearn_oracle_and_report_the_hit() {
    let case = load_f64();
    let y_true = labels_i32(&case, "zd_true");
    let y_pred = labels_i32(&case, "zd_pred");
    for (tag, policy) in [
        // sklearn's `'warn'` string is the SAME value as 0 — the difference is
        // only the UndefinedMetricWarning the Python shim raises off
        // `zero_division_hit`.
        ("warn", ZeroDivision::Zero),
        ("zero", ZeroDivision::Zero),
        ("one", ZeroDivision::One),
        ("nan", ZeroDivision::Nan),
    ] {
        let res = precision_score(&y_true, &y_pred, None, 1, Average::Binary, None, policy)
            .expect("precision_score");
        assert!(
            res.zero_division_hit,
            "{tag}: a no-predicted-positives case must report the zero-division hit"
        );
        assert_close_or_nonfinite(
            prf_scalar(res),
            scalar(&case, &format!("ref_zd_{tag}")),
            TOL,
            tag,
        );
    }
}

#[test]
fn zero_division_hit_is_false_on_a_well_defined_problem() {
    let case = load_f64();
    let y_true = labels_i32(&case, "y_true");
    let y_pred = labels_i32(&case, "y_pred");
    for average in [
        Average::Micro,
        Average::Macro,
        Average::Weighted,
        Average::None_,
    ] {
        let res = prf_of("precision", &y_true, &y_pred, average, None);
        assert!(
            !res.zero_division_hit,
            "every class is both predicted and present; no zero division is possible"
        );
    }
}

// ==================== roc_auc_score(multi_class=, average=) ====================

#[test]
fn roc_auc_multiclass_every_average_matches_sklearn_oracle() {
    let case = load_f64();
    let y_true = labels_i32(&case, "y_true");
    let y_proba = f64_vec(&case, "y_proba");
    let sw = f64_vec(&case, "sample_weight");

    for (tag, average) in [
        ("macro", Average::Macro),
        ("weighted", Average::Weighted),
        ("micro", Average::Micro),
    ] {
        let got = auc_scalar(
            roc_auc_score_multiclass(&y_true, &y_proba, &CLASSES4, MultiClass::Ovr, average, None)
                .expect("ovr"),
        );
        assert_close(got, scalar(&case, &format!("ref_auc_ovr_{tag}")), TOL, tag);
        let got_sw = auc_scalar(
            roc_auc_score_multiclass(
                &y_true,
                &y_proba,
                &CLASSES4,
                MultiClass::Ovr,
                average,
                Some(&sw),
            )
            .expect("ovr weighted"),
        );
        assert_close(
            got_sw,
            scalar(&case, &format!("ref_auc_ovr_{tag}_sw")),
            TOL,
            tag,
        );
    }

    for (tag, average) in [("macro", Average::Macro), ("weighted", Average::Weighted)] {
        let got = auc_scalar(
            roc_auc_score_multiclass(&y_true, &y_proba, &CLASSES4, MultiClass::Ovo, average, None)
                .expect("ovo"),
        );
        assert_close(got, scalar(&case, &format!("ref_auc_ovo_{tag}")), TOL, tag);
    }
}

#[test]
fn roc_auc_multiclass_average_none_matches_sklearn_oracle() {
    let case = load_f64();
    let y_true = labels_i32(&case, "y_true");
    let y_proba = f64_vec(&case, "y_proba");
    let sw = f64_vec(&case, "sample_weight");
    let got = roc_auc_score_multiclass(
        &y_true,
        &y_proba,
        &CLASSES4,
        MultiClass::Ovr,
        Average::None_,
        None,
    )
    .expect("ovr none");
    match got {
        PrfOut::PerClass(v) => {
            assert_vec_close(&v, &f64_vec(&case, "ref_auc_ovr_none"), TOL, "ovr none")
        }
        PrfOut::Scalar(v) => panic!("expected a per-class vector, got {v}"),
    }
    let got_sw = roc_auc_score_multiclass(
        &y_true,
        &y_proba,
        &CLASSES4,
        MultiClass::Ovr,
        Average::None_,
        Some(&sw),
    )
    .expect("ovr none weighted");
    match got_sw {
        PrfOut::PerClass(v) => assert_vec_close(
            &v,
            &f64_vec(&case, "ref_auc_ovr_none_sw"),
            TOL,
            "ovr none sw",
        ),
        PrfOut::Scalar(v) => panic!("expected a per-class vector, got {v}"),
    }
}

#[test]
fn roc_auc_multiclass_non_zero_based_labels_match_sklearn_oracle() {
    // The same problem with labels 10..13: the class-index ENCODING must make
    // this identical to the 0..3 value (a hard-coded `t == c as i32` cannot).
    let case = load_f64();
    let y_true_shift = labels_i32(&case, "y_true_shift");
    let labels_shift = labels_i32(&case, "labels_shift");
    let y_proba = f64_vec(&case, "y_proba");
    let got = auc_scalar(
        roc_auc_score_multiclass(
            &y_true_shift,
            &y_proba,
            &labels_shift,
            MultiClass::Ovr,
            Average::Macro,
            None,
        )
        .expect("ovr shifted labels"),
    );
    assert_close(
        got,
        scalar(&case, "ref_auc_ovr_labels_shift"),
        TOL,
        "ovr shifted labels",
    );
    assert_close(
        got,
        scalar(&case, "ref_auc_ovr_macro"),
        TOL,
        "shifted labels must not change the value",
    );
}

#[test]
fn roc_auc_multiclass_rejects_unsupported_average_combinations() {
    let case = load_f64();
    let y_true = labels_i32(&case, "y_true");
    let y_proba = f64_vec(&case, "y_proba");
    // sklearn: `average must be one of ('macro', 'weighted', None)` for OvO.
    assert!(matches!(
        roc_auc_score_multiclass(
            &y_true,
            &y_proba,
            &CLASSES4,
            MultiClass::Ovo,
            Average::Micro,
            None
        ),
        Err(MetricError::UnsupportedAverage)
    ));
    // sklearn: `average=None is not implemented for multi_class='ovo'`.
    assert!(matches!(
        roc_auc_score_multiclass(
            &y_true,
            &y_proba,
            &CLASSES4,
            MultiClass::Ovo,
            Average::None_,
            None
        ),
        Err(MetricError::UnsupportedAverage)
    ));
    // `average='binary'` is meaningless for a multiclass problem.
    assert!(matches!(
        roc_auc_score_multiclass(
            &y_true,
            &y_proba,
            &CLASSES4,
            MultiClass::Ovr,
            Average::Binary,
            None
        ),
        Err(MetricError::UnsupportedAverage)
    ));
}

#[test]
fn roc_auc_multiclass_label_outside_classes_is_an_error_not_a_panic() {
    let y_true = [0i32, 1, 5];
    let y_score = [0.5, 0.5, 0.4, 0.6, 0.3, 0.7];
    assert!(matches!(
        roc_auc_score_multiclass(
            &y_true,
            &y_score,
            &[0, 1],
            MultiClass::Ovr,
            Average::Macro,
            None
        ),
        Err(MetricError::LabelNotInLabels)
    ));
}

// ==================== roc_auc_score(max_fpr=) ====================

#[test]
fn roc_auc_binary_max_fpr_matches_sklearn_oracle() {
    let case = load_f64();
    let y_true = labels_i32(&case, "y_true_bin");
    let y_score = f64_vec(&case, "y_score_bin");
    let sw = f64_vec(&case, "sw_bin");
    let max_fprs = f64_vec(&case, "max_fprs");
    let want = f64_vec(&case, "ref_auc_maxfpr");
    let want_sw = f64_vec(&case, "ref_auc_maxfpr_sw");
    for (i, &m) in max_fprs.iter().enumerate() {
        let got = roc_auc_score_binary(&y_true, &y_score, 1, None, Some(m)).expect("max_fpr");
        assert_close(got, want[i], TOL, &format!("max_fpr={m}"));
        let got_sw =
            roc_auc_score_binary(&y_true, &y_score, 1, Some(&sw), Some(m)).expect("max_fpr sw");
        assert_close(got_sw, want_sw[i], TOL, &format!("max_fpr={m} weighted"));
    }
}

#[test]
fn roc_auc_binary_max_fpr_one_equals_the_full_auc() {
    let case = load_f64();
    let y_true = labels_i32(&case, "y_true_bin");
    let y_score = f64_vec(&case, "y_score_bin");
    let full = roc_auc_score_binary(&y_true, &y_score, 1, None, None).expect("full");
    let at_one = roc_auc_score_binary(&y_true, &y_score, 1, None, Some(1.0)).expect("max_fpr=1");
    assert_eq!(full, at_one, "max_fpr=1 must short-circuit to the full AUC");
}

#[test]
fn roc_auc_binary_rejects_out_of_range_max_fpr() {
    let y_true = [0i32, 1, 0, 1];
    let y_score = [0.1, 0.9, 0.2, 0.8];
    for bad in [0.0, -0.5, 1.5, f64::NAN] {
        assert!(
            matches!(
                roc_auc_score_binary(&y_true, &y_score, 1, None, Some(bad)),
                Err(MetricError::InvalidMaxFpr)
            ),
            "max_fpr={bad} must be rejected"
        );
    }
}

// ==================== precision_recall_curve(drop_intermediate=) ====================

fn assert_pr_curve(case: &OracleCase, score_key: &str, tag: &str, drop: bool, weighted: bool) {
    let y_true = labels_i32(case, "y_true_bin");
    let scores = f64_vec(case, score_key);
    let sw = f64_vec(case, "sw_bin");
    let (p, r, t) = precision_recall_curve(
        &y_true,
        &scores,
        1,
        if weighted { Some(&sw) } else { None },
        drop,
    )
    .expect("precision_recall_curve");
    let suffix = if weighted { "_sw" } else { "" };
    assert_vec_close(
        &p,
        &f64_vec(case, &format!("ref_prc_p_{tag}{suffix}")),
        TOL,
        "precision",
    );
    assert_vec_close(
        &r,
        &f64_vec(case, &format!("ref_prc_r_{tag}{suffix}")),
        TOL,
        "recall",
    );
    assert_vec_close(
        &t,
        &f64_vec(case, &format!("ref_prc_t_{tag}{suffix}")),
        TOL,
        "thresholds",
    );
}

#[test]
fn precision_recall_curve_drop_intermediate_matches_sklearn_oracle() {
    let case = load_f64();
    // Tie-heavy scores: `drop_intermediate` is nearly a no-op.
    assert_pr_curve(&case, "y_score_bin", "nodrop", false, false);
    assert_pr_curve(&case, "y_score_bin", "drop", true, false);
    assert_pr_curve(&case, "y_score_bin", "nodrop", false, true);
    assert_pr_curve(&case, "y_score_bin", "drop", true, true);
    // Continuous scores: it removes a large fraction of the points, and the
    // two curves must NOT be the same length (a no-op implementation would
    // still pass the tie-heavy case above).
    assert_pr_curve(&case, "y_score_cont", "cont_nodrop", false, false);
    assert_pr_curve(&case, "y_score_cont", "cont_drop", true, false);
    assert!(
        f64_vec(&case, "ref_prc_p_cont_drop").len() < f64_vec(&case, "ref_prc_p_cont_nodrop").len(),
        "the continuous-score fixture must actually exercise the drop"
    );
}

#[test]
fn precision_recall_curve_all_negative_target_sets_recall_to_one() {
    // sklearn: "No positive class found in y_true, recall is set to one for
    // all thresholds."
    let y_true = [0i32, 0, 0, 0];
    let scores = [0.1, 0.4, 0.35, 0.8];
    let (precision, recall, _) =
        precision_recall_curve(&y_true, &scores, 1, None, false).expect("pr curve");
    assert!(
        recall[..recall.len() - 1].iter().all(|&v| v == 1.0),
        "recall must be 1.0 at every threshold, got {recall:?}"
    );
    assert!(
        precision[..precision.len() - 1].iter().all(|&v| v == 0.0),
        "precision must be 0.0 at every threshold, got {precision:?}"
    );
}

// ==================== r2/mse/mae(multioutput=) ====================

#[test]
fn regression_multioutput_matches_sklearn_oracle_f64() {
    let case = load_f64();
    let y_true = f64_vec(&case, "Y_true");
    let y_pred = f64_vec(&case, "Y_pred");
    let sw = f64_vec(&case, "sw_reg");
    let mo_weights = f64_vec(&case, "mo_weights");

    // raw_values / uniform_average, weighted and not, for all three metrics.
    for (name, raw) in [
        (
            "r2",
            reg_raw(
                reg::r2_score::<f64>(&y_true, &y_pred, 3, None, MultiOutput::RawValues, true)
                    .unwrap(),
            ),
        ),
        (
            "mse",
            reg_raw(
                reg::mean_squared_error::<f64>(&y_true, &y_pred, 3, None, MultiOutput::RawValues)
                    .unwrap(),
            ),
        ),
        (
            "mae",
            reg_raw(
                reg::mean_absolute_error::<f64>(&y_true, &y_pred, 3, None, MultiOutput::RawValues)
                    .unwrap(),
            ),
        ),
    ] {
        assert_vec_close(
            &raw,
            &f64_vec(&case, &format!("ref_{name}_raw_values")),
            TOL,
            &format!("{name} raw_values"),
        );
    }

    for (name, got) in [
        (
            "r2",
            reg_scalar(
                reg::r2_score::<f64>(
                    &y_true,
                    &y_pred,
                    3,
                    Some(&sw),
                    MultiOutput::UniformAverage,
                    true,
                )
                .unwrap(),
            ),
        ),
        (
            "mse",
            reg_scalar(
                reg::mean_squared_error::<f64>(
                    &y_true,
                    &y_pred,
                    3,
                    Some(&sw),
                    MultiOutput::UniformAverage,
                )
                .unwrap(),
            ),
        ),
        (
            "mae",
            reg_scalar(
                reg::mean_absolute_error::<f64>(
                    &y_true,
                    &y_pred,
                    3,
                    Some(&sw),
                    MultiOutput::UniformAverage,
                )
                .unwrap(),
            ),
        ),
    ] {
        assert_close(
            got,
            scalar(&case, &format!("ref_{name}_uniform_average_sw")),
            TOL,
            &format!("{name} uniform_average weighted"),
        );
    }

    // Explicit per-output weights.
    for (name, got) in [
        (
            "r2",
            reg_scalar(
                reg::r2_score::<f64>(
                    &y_true,
                    &y_pred,
                    3,
                    None,
                    MultiOutput::Weights(&mo_weights),
                    true,
                )
                .unwrap(),
            ),
        ),
        (
            "mse",
            reg_scalar(
                reg::mean_squared_error::<f64>(
                    &y_true,
                    &y_pred,
                    3,
                    None,
                    MultiOutput::Weights(&mo_weights),
                )
                .unwrap(),
            ),
        ),
        (
            "mae",
            reg_scalar(
                reg::mean_absolute_error::<f64>(
                    &y_true,
                    &y_pred,
                    3,
                    None,
                    MultiOutput::Weights(&mo_weights),
                )
                .unwrap(),
            ),
        ),
    ] {
        assert_close(
            got,
            scalar(&case, &format!("ref_{name}_weights")),
            TOL,
            &format!("{name} array-like multioutput"),
        );
    }

    // variance_weighted — r2 only.
    let varw = reg_scalar(
        reg::r2_score::<f64>(
            &y_true,
            &y_pred,
            3,
            None,
            MultiOutput::VarianceWeighted,
            true,
        )
        .unwrap(),
    );
    assert_close(
        varw,
        scalar(&case, "ref_r2_variance_weighted"),
        TOL,
        "r2 variance_weighted",
    );
    let varw_sw = reg_scalar(
        reg::r2_score::<f64>(
            &y_true,
            &y_pred,
            3,
            Some(&sw),
            MultiOutput::VarianceWeighted,
            true,
        )
        .unwrap(),
    );
    assert_close(
        varw_sw,
        scalar(&case, "ref_r2_variance_weighted_sw"),
        TOL,
        "r2 variance_weighted weighted",
    );
}

#[test]
fn regression_multioutput_matches_sklearn_oracle_f32() {
    let case = load_f32();
    let y_true: Vec<f32> = case.expect_f32("Y_true").to_vec();
    let y_pred: Vec<f32> = case.expect_f32("Y_pred").to_vec();
    let want: Vec<f64> = case
        .expect_f32("ref_r2_raw_values")
        .iter()
        .map(|&v| v as f64)
        .collect();
    let got = reg_raw(
        reg::r2_score::<f32>(&y_true, &y_pred, 3, None, MultiOutput::RawValues, true).unwrap(),
    );
    assert_vec_close(&got, &want, ATOL_F32, "r2 raw_values f32");
}

#[test]
fn error_metrics_reject_variance_weighted() {
    // sklearn 1.9.0's `mean_squared_error`/`mean_absolute_error` reject the
    // string outright; so does the algos layer.
    let y_true = [1.0f64, 2.0, 3.0, 4.0];
    let y_pred = [1.1f64, 2.1, 2.9, 4.2];
    assert!(matches!(
        reg::mean_squared_error::<f64>(&y_true, &y_pred, 2, None, MultiOutput::VarianceWeighted),
        Err(MetricError::UnsupportedMultiOutput)
    ));
    assert!(matches!(
        reg::mean_absolute_error::<f64>(&y_true, &y_pred, 2, None, MultiOutput::VarianceWeighted),
        Err(MetricError::UnsupportedMultiOutput)
    ));
}

#[test]
fn multioutput_weight_length_mismatch_is_an_error() {
    let y_true = [1.0f64, 2.0, 3.0, 4.0];
    let y_pred = [1.1f64, 2.1, 2.9, 4.2];
    assert!(matches!(
        reg::mean_squared_error::<f64>(&y_true, &y_pred, 2, None, MultiOutput::Weights(&[1.0])),
        Err(MetricError::BadMultiOutputWeights)
    ));
}

// ==================== r2_score(force_finite=) ====================

#[test]
fn r2_force_finite_matches_sklearn_oracle() {
    let case = load_f64();
    let y_true = f64_vec(&case, "Y_true_const");
    let y_pred = f64_vec(&case, "Y_pred_const");

    let forced = reg_raw(
        reg::r2_score::<f64>(&y_true, &y_pred, 3, None, MultiOutput::RawValues, true).unwrap(),
    );
    assert_vec_close(
        &forced,
        &f64_vec(&case, "ref_r2_ff_true_raw"),
        TOL,
        "force_finite=true raw",
    );

    // The whole point: the constant output becomes -inf rather than 0.0.
    let unforced = reg_raw(
        reg::r2_score::<f64>(&y_true, &y_pred, 3, None, MultiOutput::RawValues, false).unwrap(),
    );
    assert_vec_close(
        &unforced,
        &f64_vec(&case, "ref_r2_ff_false_raw"),
        TOL,
        "force_finite=false raw",
    );
    assert_eq!(unforced[0], f64::NEG_INFINITY);

    let unforced_uniform = reg_scalar(
        reg::r2_score::<f64>(
            &y_true,
            &y_pred,
            3,
            None,
            MultiOutput::UniformAverage,
            false,
        )
        .unwrap(),
    );
    assert_close_or_nonfinite(
        unforced_uniform,
        scalar(&case, "ref_r2_ff_false_uniform"),
        TOL,
        "force_finite=false uniform",
    );

    for (key, multioutput) in [
        ("ref_r2_ff_true_uniform", MultiOutput::UniformAverage),
        ("ref_r2_ff_true_varw", MultiOutput::VarianceWeighted),
    ] {
        let got =
            reg_scalar(reg::r2_score::<f64>(&y_true, &y_pred, 3, None, multioutput, true).unwrap());
        assert_close(got, scalar(&case, key), TOL, key);
    }
}

#[test]
fn r2_perfect_constant_prediction_is_one_under_force_finite_and_nan_without() {
    // ss_res == 0 AND ss_tot == 0: sklearn returns 1.0 when forced, `1 - 0/0 =
    // NaN` when not.
    let y = [2.0f64, 2.0, 2.0, 2.0];
    let forced = reg_scalar(
        reg::r2_score::<f64>(&y, &y, 1, None, MultiOutput::UniformAverage, true).unwrap(),
    );
    assert_eq!(forced, 1.0);
    let unforced = reg_scalar(
        reg::r2_score::<f64>(&y, &y, 1, None, MultiOutput::UniformAverage, false).unwrap(),
    );
    assert!(unforced.is_nan(), "expected NaN, got {unforced}");
}

#[test]
fn r2_with_fewer_than_two_samples_is_nan() {
    // sklearn returns a bare NaN (plus an UndefinedMetricWarning the shim
    // re-emits) BEFORE any multioutput reduction.
    let got = reg_scalar(
        reg::r2_score::<f64>(&[1.0], &[2.0], 1, None, MultiOutput::RawValues, true).unwrap(),
    );
    assert!(got.is_nan(), "expected NaN, got {got}");
}
