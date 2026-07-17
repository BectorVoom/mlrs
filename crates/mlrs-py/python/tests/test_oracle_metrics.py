"""``mlrs.metrics`` sklearn-oracle replay through the FULL Python binding path
(TASK-17..23, METR-CLS-01..09 / METR-REG-01..03).

A SECOND consumer of the `metrics_*.npz` fixtures (TASK-02) already gated at
the Rust layer (`metrics_classification_test.rs` / `metrics_regression_test.rs`,
TASK-03..14) — this file replays them through
`numpy -> mlrs.metrics -> _mlrs -> Rust`, following the
`test_oracle_neighbors.py` template (`_atol(fixture)` dtype-branch,
`@requires_f64`, `conftest.dtype_of`/`fixture_path`).
"""

import numpy as np
import pytest

import mlrs
import mlrs.metrics as mm
from conftest import dtype_of, fixture_path, requires_f64

BINARY_FIXTURES = ["metrics_cls_binary_f32_seed42", "metrics_cls_binary_f64_seed42"]
MULTICLASS_FIXTURES = ["metrics_cls_multiclass_f32_seed42", "metrics_cls_multiclass_f64_seed42"]
REG_FIXTURES = ["metrics_reg_f32_seed42", "metrics_reg_f64_seed42"]


def _atol(fixture):
    return 1e-5 if dtype_of(fixture) == np.float64 else 1e-4


def _load(fixture):
    return np.load(fixture_path(fixture))


def _is_f64(fixture):
    return dtype_of(fixture) == np.float64


def _maybe_skip_f64(fixture):
    if _is_f64(fixture) and not mlrs.backend_supports_f64():
        pytest.skip("backend does not support f64 (mlrs.backend_supports_f64() is False)")


# ==================== TASK-17 — accuracy_score + confusion_matrix ====================


@pytest.mark.parametrize("fixture", BINARY_FIXTURES)
@requires_f64
def test_accuracy_score_oracle(fixture):
    _maybe_skip_f64(fixture)
    d = _load(fixture)
    got = mm.accuracy_score(d["y_true"], d["y_pred"], sample_weight=d["sample_weight"])
    assert abs(got - d["ref_accuracy_sw"][0]) <= _atol(fixture)


def test_accuracy_score_single_sample_degenerate():
    d = _load("metrics_cls_degenerate_seed42")
    got_match = mm.accuracy_score(d["y_true_single_match"], d["y_pred_single_match"])
    assert abs(got_match - d["ref_acc_single_match"][0]) <= 1e-9
    got_mismatch = mm.accuracy_score(d["y_true_single_mismatch"], d["y_pred_single_mismatch"])
    assert abs(got_mismatch - d["ref_acc_single_mismatch"][0]) <= 1e-9


@pytest.mark.parametrize("fixture", BINARY_FIXTURES + MULTICLASS_FIXTURES)
@requires_f64
def test_confusion_matrix_oracle(fixture):
    _maybe_skip_f64(fixture)
    d = _load(fixture)
    got = mm.confusion_matrix(d["y_true"], d["y_pred"])
    assert np.allclose(got.astype(np.float64), d["ref_confusion"], atol=_atol(fixture))
    if "ref_confusion_sw" in d.files:
        got_sw = mm.confusion_matrix(d["y_true"], d["y_pred"], sample_weight=d["sample_weight"])
        assert np.allclose(got_sw, d["ref_confusion_sw"], atol=_atol(fixture))


def test_confusion_matrix_empty_class_via_labels():
    d = _load("metrics_cls_degenerate_seed42")
    got = mm.confusion_matrix(d["y_true_empty"], d["y_pred_empty"], labels=d["labels_empty"])
    assert np.array_equal(got.astype(np.float64), d["ref_confusion_empty"])


def test_confusion_matrix_all_one_class():
    d = _load("metrics_cls_degenerate_seed42")
    got = mm.confusion_matrix(d["y_true_one"], d["y_pred_one"])
    assert np.array_equal(got.astype(np.float64), d["ref_confusion_one"])


# ==================== TASK-18 — precision/recall/f1 ====================


def _prf_averages_oracle(fn, ref_prefix, fixture):
    _maybe_skip_f64(fixture)
    d = _load(fixture)
    for avg in ["macro", "micro", "weighted"]:
        got = fn(d["y_true"], d["y_pred"], average=avg)
        assert abs(got - d[f"ref_{ref_prefix}_{avg}"][0]) <= _atol(fixture)


@pytest.mark.parametrize("fixture", MULTICLASS_FIXTURES)
@requires_f64
def test_precision_score_averages_oracle(fixture):
    _prf_averages_oracle(mm.precision_score, "precision", fixture)


@pytest.mark.parametrize("fixture", MULTICLASS_FIXTURES)
@requires_f64
def test_recall_score_averages_oracle(fixture):
    _prf_averages_oracle(mm.recall_score, "recall", fixture)


@pytest.mark.parametrize("fixture", MULTICLASS_FIXTURES)
@requires_f64
def test_f1_score_averages_oracle(fixture):
    _prf_averages_oracle(mm.f1_score, "f1", fixture)


def test_precision_score_average_none_returns_per_class_array():
    d = _load("metrics_cls_multiclass_f64_seed42")
    got = mm.precision_score(d["y_true"], d["y_pred"], average=None)
    assert isinstance(got, np.ndarray)
    assert np.allclose(got, d["ref_precision_none"], atol=1e-5)


def test_recall_score_average_none_returns_per_class_array():
    d = _load("metrics_cls_multiclass_f64_seed42")
    got = mm.recall_score(d["y_true"], d["y_pred"], average=None)
    assert isinstance(got, np.ndarray)
    assert np.allclose(got, d["ref_recall_none"], atol=1e-5)


def test_f1_score_average_none_returns_per_class_array():
    d = _load("metrics_cls_multiclass_f64_seed42")
    got = mm.f1_score(d["y_true"], d["y_pred"], average=None)
    assert isinstance(got, np.ndarray)
    assert np.allclose(got, d["ref_f1_none"], atol=1e-5)


def test_precision_score_zero_division_degenerate():
    d = _load("metrics_cls_degenerate_seed42")
    got = mm.precision_score(d["y_true_zp"], d["y_pred_zp"], zero_division=0)
    assert abs(got - d["ref_precision_zerodiv"][0]) <= 1e-9


def test_recall_score_zero_division_degenerate():
    d = _load("metrics_cls_degenerate_seed42")
    got = mm.recall_score(d["y_true_zr"], d["y_pred_zr"], zero_division=0)
    assert abs(got - d["ref_recall_zerodiv"][0]) <= 1e-9


def test_f1_score_zero_division_degenerate():
    d = _load("metrics_cls_degenerate_seed42")
    got = mm.f1_score(d["y_true_zf"], d["y_pred_zf"], zero_division=0)
    assert abs(got - d["ref_f1_zerodiv"][0]) <= 1e-9


def test_precision_score_labels_reorder_oracle():
    d = _load("metrics_cls_multiclass_f64_seed42")
    got = mm.precision_score(
        d["y_true_labelreorder"], d["y_pred_labelreorder"], labels=d["labels_reorder"], average="macro"
    )
    assert abs(got - d["ref_precision_labelreorder"][0]) <= 1e-5


def test_recall_score_labels_reorder_oracle():
    d = _load("metrics_cls_multiclass_f64_seed42")
    got = mm.recall_score(
        d["y_true_labelreorder"], d["y_pred_labelreorder"], labels=d["labels_reorder"], average="macro"
    )
    assert abs(got - d["ref_recall_labelreorder"][0]) <= 1e-5


def test_f1_score_labels_reorder_oracle():
    d = _load("metrics_cls_multiclass_f64_seed42")
    got = mm.f1_score(
        d["y_true_labelreorder"], d["y_pred_labelreorder"], labels=d["labels_reorder"], average="macro"
    )
    assert abs(got - d["ref_f1_labelreorder"][0]) <= 1e-5


# ==================== TASK-19 — log_loss ====================


def test_log_loss_eps_auto_maps_to_fixed_epsilon():
    d = _load("metrics_cls_degenerate_seed42")
    got = mm.log_loss(d["y_true_clip"], d["y_prob_clip"], eps="auto")
    assert abs(got - d["ref_log_loss_clip"][0]) <= 1e-5


@pytest.mark.parametrize("fixture", MULTICLASS_FIXTURES)
@requires_f64
def test_log_loss_matches_sklearn_oracle_multiclass(fixture):
    _maybe_skip_f64(fixture)
    d = _load(fixture)
    got = mm.log_loss(d["y_true"], d["y_proba"])
    assert abs(got - d["ref_log_loss"][0]) <= _atol(fixture)
    got_sw = mm.log_loss(d["y_true"], d["y_proba"], sample_weight=d["sample_weight"])
    assert abs(got_sw - d["ref_log_loss_sw"][0]) <= _atol(fixture)


@pytest.mark.parametrize("fixture", BINARY_FIXTURES)
@requires_f64
def test_log_loss_matches_sklearn_oracle_binary(fixture):
    _maybe_skip_f64(fixture)
    d = _load(fixture)
    got = mm.log_loss(d["y_true"], d["y_prob_binary"])
    assert abs(got - d["ref_log_loss_binary"][0]) <= _atol(fixture)


def test_log_loss_labels_reorder_matches_sklearn_oracle():
    d = _load("metrics_cls_degenerate_seed42")
    got = mm.log_loss(
        d["y_true_logloss_labelreorder"], d["y_prob_logloss_labelreorder"], labels=d["labels_logloss_reorder"]
    )
    assert abs(got - d["ref_log_loss_labelreorder"][0]) <= 1e-5


# ==================== TASK-20 — roc_auc_score (binary) ====================


def test_roc_auc_score_binary_single_class_raises_value_error():
    d = _load("metrics_cls_degenerate_seed42")
    with pytest.raises(ValueError):
        mm.roc_auc_score(d["y_true_singleclass"], d["y_score_singleclass"])


@pytest.mark.parametrize("fixture", BINARY_FIXTURES)
@requires_f64
def test_roc_auc_score_binary_matches_sklearn_oracle(fixture):
    _maybe_skip_f64(fixture)
    d = _load(fixture)
    got = mm.roc_auc_score(d["y_true"], d["y_score"])
    assert abs(got - d["ref_roc_auc"][0]) <= _atol(fixture)


@pytest.mark.parametrize("fixture", BINARY_FIXTURES)
@requires_f64
def test_roc_auc_score_binary_weighted_matches_sklearn_oracle(fixture):
    _maybe_skip_f64(fixture)
    d = _load(fixture)
    got = mm.roc_auc_score(d["y_true"], d["y_score"], sample_weight=d["sample_weight"])
    assert abs(got - d["ref_roc_auc_sw"][0]) <= _atol(fixture)


# ==================== TASK-21 — roc_auc_score (multiclass OvR/OvO) ====================


@pytest.mark.parametrize(
    "multi_class,average", [("ovr", "macro"), ("ovr", "weighted"), ("ovo", "macro"), ("ovo", "weighted")]
)
def test_roc_auc_score_multiclass_oracle(multi_class, average):
    d = _load("metrics_cls_multiclass_f64_seed42")
    got = mm.roc_auc_score(d["y_true"], d["y_proba"], multi_class=multi_class, average=average)
    want = d[f"ref_roc_auc_{multi_class}_{average}"][0]
    assert abs(got - want) <= 1e-5


@pytest.mark.parametrize("average", ["macro", "weighted"])
def test_roc_auc_score_multiclass_ovr_sample_weight_oracle(average):
    d = _load("metrics_cls_multiclass_f64_seed42")
    got = mm.roc_auc_score(
        d["y_true"], d["y_proba"], multi_class="ovr", average=average, sample_weight=d["sample_weight"]
    )
    want = d[f"ref_roc_auc_ovr_{average}_sw"][0]
    assert abs(got - want) <= 1e-5


def test_roc_auc_score_multiclass_ovo_sample_weight_gate():
    # Branch A (TASK-02 probe against the pinned scikit-learn==1.9.0): OvO +
    # sample_weight RAISES. Mirrors TASK-10's Rust-layer
    # Err(MetricError::WeightedOvoUnsupported) exactly (same branch, not
    # independently re-decided).
    d = _load("metrics_cls_multiclass_f64_seed42")
    assert "ref_roc_auc_ovo_macro_sw" not in d.files, "Branch B fixture unexpectedly present"
    with pytest.raises(ValueError):
        mm.roc_auc_score(
            d["y_true"], d["y_proba"], multi_class="ovo", average="macro", sample_weight=d["sample_weight"]
        )


# ==================== TASK-22 — precision_recall_curve ====================


def test_precision_recall_curve_returns_three_arrays_with_sentinel():
    d = _load("metrics_cls_binary_f64_seed42")
    precision, recall, thresholds = mm.precision_recall_curve(d["y_true"], d["y_score"])
    assert len(precision) == len(thresholds) + 1
    assert len(recall) == len(thresholds) + 1
    assert precision[-1] == 1.0
    assert recall[-1] == 0.0


def test_precision_recall_curve_matches_sklearn_oracle():
    d = _load("metrics_cls_binary_f64_seed42")
    precision, recall, thresholds = mm.precision_recall_curve(d["y_true"], d["y_score"])
    assert np.allclose(precision, d["ref_pr_precision"], atol=1e-5)
    assert np.allclose(recall, d["ref_pr_recall"], atol=1e-5)
    assert np.allclose(thresholds, d["ref_pr_thresholds"], atol=1e-5)


def test_precision_recall_curve_weighted_matches_sklearn_oracle():
    d = _load("metrics_cls_binary_f64_seed42")
    precision, recall, thresholds = mm.precision_recall_curve(
        d["y_true"], d["y_score"], sample_weight=d["sample_weight"]
    )
    assert np.allclose(precision, d["ref_pr_precision_sw"], atol=1e-5)
    assert np.allclose(recall, d["ref_pr_recall_sw"], atol=1e-5)
    assert np.allclose(thresholds, d["ref_pr_thresholds_sw"], atol=1e-5)


# ==================== TASK-23 — r2_score / mean_squared_error / mean_absolute_error ====================


def test_r2_score_constant_target_oracle():
    d = _load("metrics_reg_f64_seed42")
    got = mm.r2_score(d["y_true_const"], d["y_pred_const"])
    assert abs(got - d["ref_r2_const"][0]) <= 1e-5


def test_r2_score_perfect_prediction_oracle():
    d = _load("metrics_reg_f64_seed42")
    got = mm.r2_score(d["y_perfect"], d["y_perfect"])
    assert abs(got - d["ref_r2_perfect"][0]) <= 1e-5


@pytest.mark.parametrize("fixture", REG_FIXTURES)
@requires_f64
def test_mean_squared_error_oracle(fixture):
    _maybe_skip_f64(fixture)
    d = _load(fixture)
    got = mm.mean_squared_error(d["y_true"], d["y_pred"])
    assert abs(got - d["ref_mse"][0]) <= _atol(fixture)
    got_sw = mm.mean_squared_error(d["y_true"], d["y_pred"], sample_weight=d["sample_weight"])
    assert abs(got_sw - d["ref_mse_sw"][0]) <= _atol(fixture)
    got_perfect = mm.mean_squared_error(d["y_perfect"], d["y_perfect"])
    assert abs(got_perfect - d["ref_mse_perfect"][0]) <= _atol(fixture)


@pytest.mark.parametrize("fixture", REG_FIXTURES)
@requires_f64
def test_mean_absolute_error_oracle(fixture):
    _maybe_skip_f64(fixture)
    d = _load(fixture)
    got = mm.mean_absolute_error(d["y_true"], d["y_pred"])
    assert abs(got - d["ref_mae"][0]) <= _atol(fixture)
    got_sw = mm.mean_absolute_error(d["y_true"], d["y_pred"], sample_weight=d["sample_weight"])
    assert abs(got_sw - d["ref_mae_sw"][0]) <= _atol(fixture)
    got_perfect = mm.mean_absolute_error(d["y_perfect"], d["y_perfect"])
    assert abs(got_perfect - d["ref_mae_perfect"][0]) <= _atol(fixture)


def test_r2_score_2d_input_raises_not_implemented_error():
    with pytest.raises(NotImplementedError):
        mm.r2_score(np.zeros((3, 2)), np.zeros((3, 2)))


def test_mean_squared_error_non_default_multioutput_raises_not_implemented_error():
    with pytest.raises(NotImplementedError):
        mm.mean_squared_error(np.array([1.0, 2.0]), np.array([1.0, 2.0]), multioutput="raw_values")
