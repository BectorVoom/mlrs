"""TASK-15 (METR-BIND-01) — `_mlrs` metrics PyO3 binding-surface smoke test.

Proves the binding surface is WIRED and error-mapped end to end: every one
of the 11 metrics (14 registrations including the `average=None` `_per_class`
split for precision/recall/f1) is callable from the low-level `mlrs._mlrs`
extension, and `MetricError` maps to a Python `ValueError` via the new
`metric_err_to_py`. This is a SMOKE test, NOT a numerical oracle — the ≤1e-5
tolerance is already gated by the Rust oracle tests
(`metrics_classification_test.rs` / `metrics_regression_test.rs`, TASK-03..14)
and the Python oracle replay (`test_oracle_metrics.py`, TASK-17..23).

Import-guarded (``pytest.importorskip``) so it skips cleanly if the extension
is unavailable, mirroring `test_naive_bayes.py`'s convention. Run via the
shipped maturin-develop py-test flow: build the `mlrs` extension, then
`pytest` this file.
"""

import pytest

_mlrs = pytest.importorskip("mlrs._mlrs")


def test_accuracy_score_length_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        _mlrs.accuracy_score([1, 0, 1], [1, 0], None, True)


def test_accuracy_score_callable():
    got = _mlrs.accuracy_score([1, 0, 1], [1, 0, 0], None, True)
    assert abs(got - 2.0 / 3.0) < 1e-9


def test_confusion_matrix_callable():
    got = _mlrs.confusion_matrix([0, 1, 0, 1], [0, 0, 1, 1], None, None)
    assert len(got) == 2 and len(got[0]) == 2


def test_precision_recall_f1_callable_and_per_class_variant():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    p = _mlrs.precision_score(y_true, y_pred, None, 1, "macro", None, 0.0)
    r = _mlrs.recall_score(y_true, y_pred, None, 1, "macro", None, 0.0)
    f = _mlrs.f1_score(y_true, y_pred, None, 1, "macro", None, 0.0)
    assert isinstance(p, float) and isinstance(r, float) and isinstance(f, float)

    p_pc = _mlrs.precision_score_per_class(y_true, y_pred, None, None, 0.0)
    r_pc = _mlrs.recall_score_per_class(y_true, y_pred, None, None, 0.0)
    f_pc = _mlrs.f1_score_per_class(y_true, y_pred, None, None, 0.0)
    assert len(p_pc) == 3 and len(r_pc) == 3 and len(f_pc) == 3

    # average='none' is rejected on the scalar variant (use *_per_class instead).
    with pytest.raises(ValueError):
        _mlrs.precision_score(y_true, y_pred, None, 1, "none", None, 0.0)


def test_log_loss_callable():
    y_true = [0, 1, 0, 1]
    y_prob = [0.9, 0.1, 0.1, 0.9, 0.8, 0.2, 0.2, 0.8]
    got = _mlrs.log_loss(y_true, y_prob, 2, None, None, 1e-15, True)
    assert got > 0.0


def test_roc_auc_binary_single_class_raises_value_error():
    with pytest.raises(ValueError):
        _mlrs.roc_auc_score_binary([1, 1, 1, 1], [0.1, 0.4, 0.6, 0.9], 1, None)


def test_roc_auc_multiclass_callable():
    y_true = [0, 1, 2, 0, 1, 2]
    y_score = [0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8, 0.7, 0.2, 0.1, 0.2, 0.7, 0.1, 0.1, 0.2, 0.7]
    got = _mlrs.roc_auc_score_multiclass(y_true, y_score, 3, "ovr", "macro", None)
    assert 0.0 <= got <= 1.0
    got_ovo = _mlrs.roc_auc_score_multiclass(y_true, y_score, 3, "ovo", "macro", None)
    assert 0.0 <= got_ovo <= 1.0
    # OvO + sample_weight: rejected (Branch A, TASK-02 probe against the
    # pinned scikit-learn==1.9.0).
    with pytest.raises(ValueError):
        _mlrs.roc_auc_score_multiclass(
            y_true, y_score, 3, "ovo", "macro", [1.0] * 6
        )


def test_precision_recall_curve_callable():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.4, 0.6]
    precision, recall, thresholds = _mlrs.precision_recall_curve(y_true, y_score, 1, None)
    assert len(precision) == len(thresholds) + 1
    assert len(recall) == len(thresholds) + 1
    assert precision[-1] == 1.0
    assert recall[-1] == 0.0


def test_r2_mse_mae_callable():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 1.9, 3.2]
    assert isinstance(_mlrs.r2_score(y_true, y_pred, None), float)
    assert isinstance(_mlrs.mean_squared_error(y_true, y_pred, None), float)
    assert isinstance(_mlrs.mean_absolute_error(y_true, y_pred, None), float)
    assert _mlrs.mean_squared_error(y_true, y_true, None) == 0.0
    assert _mlrs.mean_absolute_error(y_true, y_true, None) == 0.0


# ==================== code-review fix: bad sample_weight raises ValueError, no panic ====================
#
# A mismatched-length `sample_weight` previously indexed out of bounds inside
# Rust and crashed the process with an unhandled `pyo3_runtime.PanicException`
# instead of a catchable `ValueError`; precision/recall/f1 additionally
# discarded a valid `class_bookkeeping` `Err` via `.expect(...)`. These lock
# in the fixed, catchable-`ValueError` contract at the actual PyO3 boundary
# users hit.


def test_accuracy_score_bad_sample_weight_raises_value_error_not_panic():
    with pytest.raises(ValueError):
        _mlrs.accuracy_score([1, 0, 1], [1, 0, 0], [1.0], True)


def test_confusion_matrix_bad_sample_weight_raises_value_error_not_panic():
    with pytest.raises(ValueError):
        _mlrs.confusion_matrix([1, 0, 1], [1, 0, 0], None, [1.0])


def test_prf_bad_sample_weight_raises_value_error_not_panic():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    too_short = [1.0]
    with pytest.raises(ValueError):
        _mlrs.precision_score(y_true, y_pred, None, 1, "macro", too_short, 0.0)
    with pytest.raises(ValueError):
        _mlrs.recall_score(y_true, y_pred, None, 1, "macro", too_short, 0.0)
    with pytest.raises(ValueError):
        _mlrs.f1_score(y_true, y_pred, None, 1, "macro", too_short, 0.0)
    with pytest.raises(ValueError):
        _mlrs.precision_score_per_class(y_true, y_pred, None, too_short, 0.0)


def test_log_loss_bad_sample_weight_raises_value_error_not_panic():
    y_true = [0, 1, 0, 1]
    y_prob = [0.9, 0.1, 0.1, 0.9, 0.8, 0.2, 0.2, 0.8]
    with pytest.raises(ValueError):
        _mlrs.log_loss(y_true, y_prob, 2, None, [1.0], 1e-15, True)


def test_precision_recall_curve_bad_sample_weight_raises_value_error_not_panic():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.4, 0.6]
    with pytest.raises(ValueError):
        _mlrs.precision_recall_curve(y_true, y_score, 1, [1.0])


def test_regression_metrics_bad_sample_weight_raises_value_error_not_panic():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 1.9, 3.2]
    too_short = [1.0]
    with pytest.raises(ValueError):
        _mlrs.r2_score(y_true, y_pred, too_short)
    with pytest.raises(ValueError):
        _mlrs.mean_squared_error(y_true, y_pred, too_short)
    with pytest.raises(ValueError):
        _mlrs.mean_absolute_error(y_true, y_pred, too_short)
