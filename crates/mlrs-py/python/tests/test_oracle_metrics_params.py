"""``mlrs.metrics`` FULL-PARAMETER sklearn-oracle replay (METR-PARAM-01).

Replays the committed ``metrics_params_{f32,f64}_seed42.npz`` fixture — every
reference value read off the pinned ``scikit-learn==1.9.0`` by
``scripts/gen_oracle.py::gen_metrics_params`` — through the whole
``numpy -> mlrs.metrics -> _mlrs -> Rust`` path, once per STRING value of every
string-valued parameter:

* ``confusion_matrix(normalize=)``           — ``'true'``/``'pred'``/``'all'``/``None``
* ``precision/recall/f1(average=)``          — ``'binary'``/``'micro'``/``'macro'``/
  ``'weighted'``/``None``, and ``zero_division='warn'``/``0``/``1``/``nan``
* ``roc_auc_score(multi_class=, average=)``  — ``'ovr'``/``'ovo'`` x
  ``'micro'``/``'macro'``/``'weighted'``/``None`` (+ ``max_fpr``, ``labels``)
* ``precision_recall_curve(drop_intermediate=)`` (+ ``pos_label=None``)
* ``r2/mse/mae(multioutput=)``               — ``'raw_values'``/
  ``'uniform_average'``/``'variance_weighted'``/array-like (+ ``force_finite``)

The value assertions have a Rust-level twin in
``crates/mlrs-algos/tests/metrics_params_test.rs``; what only this file can
cover is the SHIM behavior — sklearn's validation messages, the
``UndefinedMetricWarning``s, the output dtypes, and the scalar-vs-array return
shapes.
"""

import numpy as np
import pytest

import mlrs.metrics as mm
from conftest import dtype_of, fixture_path

PARAM_FIXTURES = ["metrics_params_f32_seed42", "metrics_params_f64_seed42"]


def _atol(fixture):
    return 1e-5 if dtype_of(fixture) == np.float64 else 1e-4


def _load(fixture):
    return np.load(fixture_path(fixture))


def _close(got, want, atol):
    """`allclose` that also accepts matching non-finite entries (the
    `force_finite=False` references are `-inf`/`NaN`)."""
    got = np.atleast_1d(np.asarray(got, dtype=np.float64))
    want = np.atleast_1d(np.asarray(want, dtype=np.float64))
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    return np.allclose(got, want, atol=atol, equal_nan=True) and np.array_equal(
        np.isinf(got) * np.sign(got), np.isinf(want) * np.sign(want)
    )


# ==================== confusion_matrix(normalize=) ====================


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("normalize", ["true", "pred", "all"])
def test_confusion_matrix_normalize_oracle(fixture, normalize):
    d = _load(fixture)
    got = mm.confusion_matrix(d["y_true"], d["y_pred"], normalize=normalize)
    assert got.dtype == np.float64
    assert _close(got, d[f"ref_cm_{normalize}"], _atol(fixture))
    got_sw = mm.confusion_matrix(
        d["y_true"], d["y_pred"], sample_weight=d["sample_weight"], normalize=normalize
    )
    assert _close(got_sw, d[f"ref_cm_{normalize}_sw"], _atol(fixture))


def test_confusion_matrix_unnormalized_stays_int64():
    d = _load("metrics_params_f64_seed42")
    assert mm.confusion_matrix(d["y_true"], d["y_pred"]).dtype == np.int64


def test_confusion_matrix_rejects_an_unknown_normalize():
    d = _load("metrics_params_f64_seed42")
    with pytest.raises(ValueError, match="normalize"):
        mm.confusion_matrix(d["y_true"], d["y_pred"], normalize="row")


def test_confusion_matrix_rejects_labels_disjoint_from_y_true():
    d = _load("metrics_params_f64_seed42")
    with pytest.raises(ValueError, match="At least one label"):
        mm.confusion_matrix(d["y_true"], d["y_pred"], labels=[97, 98])


# ==================== precision/recall/f1(average=, zero_division=) ====================

_PRF = [("precision", "precision_score"), ("recall", "recall_score"), ("f1", "f1_score")]


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("name,fn_name", _PRF)
@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
def test_prf_average_oracle(fixture, name, fn_name, average):
    d = _load(fixture)
    fn = getattr(mm, fn_name)
    got = fn(d["y_true"], d["y_pred"], average=average, zero_division=0)
    assert isinstance(got, float)
    assert abs(got - d[f"ref_{name}_{average}"][0]) <= _atol(fixture)
    got_sw = fn(
        d["y_true"],
        d["y_pred"],
        average=average,
        sample_weight=d["sample_weight"],
        zero_division=0,
    )
    assert abs(got_sw - d[f"ref_{name}_{average}_sw"][0]) <= _atol(fixture)


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("name,fn_name", _PRF)
def test_prf_average_none_oracle(fixture, name, fn_name):
    d = _load(fixture)
    fn = getattr(mm, fn_name)
    got = fn(d["y_true"], d["y_pred"], average=None, zero_division=0)
    assert isinstance(got, np.ndarray)
    assert _close(got, d[f"ref_{name}_none"], _atol(fixture))
    got_sw = fn(
        d["y_true"], d["y_pred"], average=None, sample_weight=d["sample_weight"], zero_division=0
    )
    assert _close(got_sw, d[f"ref_{name}_none_sw"], _atol(fixture))


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("name,fn_name", _PRF)
def test_prf_average_binary_oracle(fixture, name, fn_name):
    d = _load(fixture)
    fn = getattr(mm, fn_name)
    got = fn(d["y_true_bin"], d["y_pred_bin"], average="binary", zero_division=0)
    assert abs(got - d[f"ref_{name}_binary"][0]) <= _atol(fixture)


def test_prf_rejects_unknown_average_and_samplewise():
    d = _load("metrics_params_f64_seed42")
    with pytest.raises(ValueError, match="average"):
        mm.f1_score(d["y_true"], d["y_pred"], average="bogus")
    with pytest.raises(ValueError, match="Samplewise"):
        mm.f1_score(d["y_true"], d["y_pred"], average="samples")


@pytest.mark.parametrize(
    "policy,ref_key",
    [("warn", "ref_zd_warn"), (0, "ref_zd_zero"), (1, "ref_zd_one"), (np.nan, "ref_zd_nan")],
)
def test_zero_division_policies_oracle(policy, ref_key):
    d = _load("metrics_params_f64_seed42")
    want = d[ref_key][0]
    with pytest.warns(UserWarning) if policy == "warn" else _no_warning_ctx():
        got = mm.precision_score(d["zd_true"], d["zd_pred"], zero_division=policy)
    if np.isnan(want):
        assert np.isnan(got)
    else:
        assert abs(got - want) <= 1e-12


class _no_warning_ctx:
    """A ``pytest.warns``-shaped no-op for the non-``'warn'`` policies (they
    return the same value with NO warning at all)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_zero_division_warn_is_silent_on_a_well_defined_problem():
    d = _load("metrics_params_f64_seed42")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Every class is both present and predicted, so the default
        # zero_division='warn' must not fire.
        mm.precision_score(d["y_true"], d["y_pred"], average="macro")


def test_zero_division_rejects_an_unknown_value():
    d = _load("metrics_params_f64_seed42")
    with pytest.raises(ValueError, match="zero_division"):
        mm.f1_score(d["y_true"], d["y_pred"], average="macro", zero_division="bogus")


# ==================== roc_auc_score(multi_class=, average=, max_fpr=, labels=) ====================


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("average", ["macro", "weighted", "micro"])
def test_roc_auc_ovr_average_oracle(fixture, average):
    d = _load(fixture)
    got = mm.roc_auc_score(d["y_true"], d["y_proba"], multi_class="ovr", average=average)
    assert abs(got - d[f"ref_auc_ovr_{average}"][0]) <= _atol(fixture)
    got_sw = mm.roc_auc_score(
        d["y_true"],
        d["y_proba"],
        multi_class="ovr",
        average=average,
        sample_weight=d["sample_weight"],
    )
    assert abs(got_sw - d[f"ref_auc_ovr_{average}_sw"][0]) <= _atol(fixture)


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("average", ["macro", "weighted"])
def test_roc_auc_ovo_average_oracle(fixture, average):
    d = _load(fixture)
    got = mm.roc_auc_score(d["y_true"], d["y_proba"], multi_class="ovo", average=average)
    assert abs(got - d[f"ref_auc_ovo_{average}"][0]) <= _atol(fixture)


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
def test_roc_auc_ovr_average_none_oracle(fixture):
    d = _load(fixture)
    got = mm.roc_auc_score(d["y_true"], d["y_proba"], multi_class="ovr", average=None)
    assert isinstance(got, np.ndarray)
    assert _close(got, d["ref_auc_ovr_none"], _atol(fixture))
    got_sw = mm.roc_auc_score(
        d["y_true"],
        d["y_proba"],
        multi_class="ovr",
        average=None,
        sample_weight=d["sample_weight"],
    )
    assert _close(got_sw, d["ref_auc_ovr_none_sw"], _atol(fixture))


def test_roc_auc_labels_shifted_class_ids_oracle():
    d = _load("metrics_params_f64_seed42")
    got = mm.roc_auc_score(
        d["y_true_shift"],
        d["y_proba"],
        multi_class="ovr",
        average="macro",
        labels=d["labels_shift"].astype(np.int64),
    )
    assert abs(got - d["ref_auc_ovr_labels_shift"][0]) <= 1e-5
    assert abs(got - d["ref_auc_ovr_macro"][0]) <= 1e-5


def test_roc_auc_multiclass_parameter_errors():
    d = _load("metrics_params_f64_seed42")
    y_true, y_proba = d["y_true"], d["y_proba"]
    with pytest.raises(ValueError, match="multi_class must be"):
        mm.roc_auc_score(y_true, y_proba)
    with pytest.raises(ValueError, match="average must be one of"):
        mm.roc_auc_score(y_true, y_proba, multi_class="ovo", average="micro")
    with pytest.raises(NotImplementedError):
        mm.roc_auc_score(y_true, y_proba, multi_class="ovo", average=None)
    with pytest.raises(ValueError, match="Partial AUC"):
        mm.roc_auc_score(y_true, y_proba, multi_class="ovr", max_fpr=0.5)
    with pytest.raises(ValueError, match="must be ordered"):
        mm.roc_auc_score(y_true, y_proba, multi_class="ovr", labels=[3, 2, 1, 0])
    with pytest.raises(ValueError, match="must be unique"):
        mm.roc_auc_score(y_true, y_proba, multi_class="ovr", labels=[0, 1, 2, 2])
    with pytest.raises(ValueError, match="sum up to 1"):
        mm.roc_auc_score(y_true, y_proba * 2.0, multi_class="ovr")
    with pytest.raises(ValueError, match="not supported"):
        mm.roc_auc_score(
            y_true, y_proba, multi_class="ovo", sample_weight=d["sample_weight"]
        )


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
def test_roc_auc_max_fpr_oracle(fixture):
    d = _load(fixture)
    for i, max_fpr in enumerate(d["max_fprs"]):
        got = mm.roc_auc_score(d["y_true_bin"], d["y_score_bin"], max_fpr=float(max_fpr))
        assert abs(got - d["ref_auc_maxfpr"][i]) <= _atol(fixture)
        got_sw = mm.roc_auc_score(
            d["y_true_bin"],
            d["y_score_bin"],
            max_fpr=float(max_fpr),
            sample_weight=d["sw_bin"],
        )
        assert abs(got_sw - d["ref_auc_maxfpr_sw"][i]) <= _atol(fixture)


def test_roc_auc_rejects_out_of_range_max_fpr():
    d = _load("metrics_params_f64_seed42")
    for bad in (0.0, -1.0, 1.5):
        with pytest.raises(ValueError, match="max_fpr"):
            mm.roc_auc_score(d["y_true_bin"], d["y_score_bin"], max_fpr=bad)


def test_roc_auc_binary_positive_class_is_the_larger_label():
    """sklearn binarizes against ``np.unique(y_true)``, so a ``{1, 2}`` target's
    POSITIVE class is 2 — not the hard-coded 1 an earlier shim assumed."""
    d = _load("metrics_params_f64_seed42")
    base = mm.roc_auc_score(d["y_true_bin"], d["y_score_bin"])
    shifted = mm.roc_auc_score(d["y_true_bin"] + 1, d["y_score_bin"])
    assert abs(base - shifted) <= 1e-12


# ==================== precision_recall_curve(drop_intermediate=, pos_label=) ====================


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize(
    "score_key,tag,drop",
    [
        ("y_score_bin", "nodrop", False),
        ("y_score_bin", "drop", True),
        ("y_score_cont", "cont_nodrop", False),
        ("y_score_cont", "cont_drop", True),
    ],
)
def test_precision_recall_curve_drop_intermediate_oracle(fixture, score_key, tag, drop):
    d = _load(fixture)
    p, r, t = mm.precision_recall_curve(
        d["y_true_bin"], d[score_key], drop_intermediate=drop
    )
    atol = _atol(fixture)
    assert _close(p, d[f"ref_prc_p_{tag}"], atol)
    assert _close(r, d[f"ref_prc_r_{tag}"], atol)
    assert _close(t, d[f"ref_prc_t_{tag}"], atol)


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("tag,drop", [("nodrop", False), ("drop", True)])
def test_precision_recall_curve_weighted_drop_intermediate_oracle(fixture, tag, drop):
    d = _load(fixture)
    p, r, t = mm.precision_recall_curve(
        d["y_true_bin"], d["y_score_bin"], sample_weight=d["sw_bin"], drop_intermediate=drop
    )
    atol = _atol(fixture)
    assert _close(p, d[f"ref_prc_p_{tag}_sw"], atol)
    assert _close(r, d[f"ref_prc_r_{tag}_sw"], atol)
    assert _close(t, d[f"ref_prc_t_{tag}_sw"], atol)


def test_precision_recall_curve_drop_intermediate_actually_shortens_the_curve():
    d = _load("metrics_params_f64_seed42")
    full = mm.precision_recall_curve(d["y_true_bin"], d["y_score_cont"])[0]
    dropped = mm.precision_recall_curve(
        d["y_true_bin"], d["y_score_cont"], drop_intermediate=True
    )[0]
    assert len(dropped) < len(full)


def test_precision_recall_curve_pos_label_none_on_plus_minus_one_target():
    d = _load("metrics_params_f64_seed42")
    p, r, t = mm.precision_recall_curve(d["y_true_pm1"], d["y_score_bin"])
    assert _close(p, d["ref_prc_p_pm1"], 1e-5)
    assert _close(r, d["ref_prc_r_pm1"], 1e-5)
    assert _close(t, d["ref_prc_t_pm1"], 1e-5)


def test_precision_recall_curve_pos_label_none_is_rejected_for_other_targets():
    with pytest.raises(ValueError, match="pos_label is not specified"):
        mm.precision_recall_curve([1, 2, 1, 2], [0.1, 0.2, 0.3, 0.4])


def test_precision_recall_curve_warns_when_no_positive_class():
    with pytest.warns(UserWarning, match="No positive class"):
        _, recall, _ = mm.precision_recall_curve([0, 0, 0], [0.1, 0.2, 0.3])
    assert np.array_equal(recall[:-1], np.ones(3))


# ==================== r2/mse/mae(multioutput=, force_finite=) ====================

_REG = [("r2", "r2_score"), ("mse", "mean_squared_error"), ("mae", "mean_absolute_error")]


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("name,fn_name", _REG)
def test_regression_multioutput_oracle(fixture, name, fn_name):
    d = _load(fixture)
    fn = getattr(mm, fn_name)
    atol = _atol(fixture)

    raw = fn(d["Y_true"], d["Y_pred"], multioutput="raw_values")
    assert isinstance(raw, np.ndarray) and raw.shape == (3,)
    assert _close(raw, d[f"ref_{name}_raw_values"], atol)

    uniform = fn(d["Y_true"], d["Y_pred"])
    assert isinstance(uniform, float)
    assert abs(uniform - d[f"ref_{name}_uniform_average"][0]) <= atol

    uniform_sw = fn(d["Y_true"], d["Y_pred"], sample_weight=d["sw_reg"])
    assert abs(uniform_sw - d[f"ref_{name}_uniform_average_sw"][0]) <= atol

    raw_sw = fn(d["Y_true"], d["Y_pred"], multioutput="raw_values", sample_weight=d["sw_reg"])
    assert _close(raw_sw, d[f"ref_{name}_raw_values_sw"], atol)

    weighted = fn(d["Y_true"], d["Y_pred"], multioutput=d["mo_weights"])
    assert abs(weighted - d[f"ref_{name}_weights"][0]) <= atol


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
def test_r2_variance_weighted_oracle(fixture):
    d = _load(fixture)
    got = mm.r2_score(d["Y_true"], d["Y_pred"], multioutput="variance_weighted")
    assert abs(got - d["ref_r2_variance_weighted"][0]) <= _atol(fixture)
    got_sw = mm.r2_score(
        d["Y_true"], d["Y_pred"], multioutput="variance_weighted", sample_weight=d["sw_reg"]
    )
    assert abs(got_sw - d["ref_r2_variance_weighted_sw"][0]) <= _atol(fixture)


def test_error_metrics_reject_variance_weighted():
    d = _load("metrics_params_f64_seed42")
    for fn in (mm.mean_squared_error, mm.mean_absolute_error):
        with pytest.raises(ValueError, match="multioutput"):
            fn(d["Y_true"], d["Y_pred"], multioutput="variance_weighted")


def test_regression_multioutput_validation_messages():
    d = _load("metrics_params_f64_seed42")
    with pytest.raises(ValueError, match="multioutput"):
        mm.r2_score(d["Y_true"], d["Y_pred"], multioutput="bogus")
    with pytest.raises(ValueError, match="equally many custom weights"):
        mm.r2_score(d["Y_true"], d["Y_pred"], multioutput=[1.0, 2.0])
    with pytest.raises(ValueError, match="only in multi-output"):
        mm.r2_score(np.arange(5.0), np.arange(5.0) + 1, multioutput=[1.0])
    with pytest.raises(ValueError, match="different shapes"):
        mm.r2_score(d["Y_true"], d["Y_pred"][:, :2])


def test_single_output_raw_values_returns_a_length_one_array():
    got = mm.mean_squared_error(np.arange(5.0), np.arange(5.0) + 1, multioutput="raw_values")
    assert isinstance(got, np.ndarray) and got.shape == (1,)
    assert abs(got[0] - 1.0) <= 1e-12


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
def test_r2_force_finite_oracle(fixture):
    d = _load(fixture)
    atol = _atol(fixture)
    forced = mm.r2_score(d["Y_true_const"], d["Y_pred_const"], multioutput="raw_values")
    assert _close(forced, d["ref_r2_ff_true_raw"], atol)
    assert forced[0] == 0.0

    unforced = mm.r2_score(
        d["Y_true_const"], d["Y_pred_const"], multioutput="raw_values", force_finite=False
    )
    assert unforced[0] == -np.inf
    assert _close(unforced[1:], d["ref_r2_ff_false_raw"][1:], atol)

    assert (
        abs(
            mm.r2_score(d["Y_true_const"], d["Y_pred_const"])
            - d["ref_r2_ff_true_uniform"][0]
        )
        <= atol
    )
    assert (
        abs(
            mm.r2_score(
                d["Y_true_const"], d["Y_pred_const"], multioutput="variance_weighted"
            )
            - d["ref_r2_ff_true_varw"][0]
        )
        <= atol
    )
    assert mm.r2_score(d["Y_true_const"], d["Y_pred_const"], force_finite=False) == -np.inf


def test_r2_fewer_than_two_samples_warns_and_returns_nan():
    with pytest.warns(UserWarning, match="not well-defined"):
        got = mm.r2_score([1.0], [2.0])
    assert np.isnan(got)
