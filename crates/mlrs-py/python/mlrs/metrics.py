"""``mlrs.metrics`` — sklearn-signature-faithful free functions (METR-SHIM-01).

Mirrors ``sklearn.metrics`` for the classification + regression metrics in
roadmap Phase 24's metrics criterion (accuracy/confusion/precision/recall/f1/
log_loss/roc_auc/precision_recall_curve/r2/mse/mae). Each function
normalizes its inputs (``np.asarray(...).ravel()``, the right dtype),
delegates to the corresponding ``_mlrs.<fn>`` PyO3 free function
(``crates/mlrs-py/src/metrics.rs``, TASK-15), and wraps the return in the
sklearn-faithful shape (scalar -> ``float``; confusion -> ``np.ndarray``;
PR-curve -> a 3-tuple of ``np.ndarray``).

These are plain free functions, NOT :class:`~mlrs.base.MlrsBase` subclasses —
no ``output_type``/``_normalize``/``_to_output`` estimator machinery applies
(they take already-materialized 1-D label/target vectors, never a device
array). Labels are cast to ``np.int32`` before crossing into ``_mlrs`` so
PyO3's ``Vec<i32>`` extraction succeeds (SPEC §5 METR-SHIM-01).

**Multioutput regression is a NON-GOAL** (SPEC §2, Plan-Check Issue 3):
``r2_score``/``mean_squared_error``/``mean_absolute_error`` raise
``NotImplementedError`` for a 2-D ``y_true``/``y_pred`` or a non-default
``multioutput=`` kwarg — fail-closed, BEFORE any ``.ravel()`` call, rather
than silently ``ravel()``-ing a 2-D array into a mathematically wrong 1-D
result.

The ``average=None`` (per-class vector) case for
``precision_score``/``recall_score``/``f1_score`` dispatches internally to
the ``_mlrs.<fn>_per_class`` sibling PyO3 function (TASK-15's resolved
API-asymmetry decision) — this module's own signatures stay
sklearn-faithful; the ``_per_class`` split is hidden here.
"""

import numpy as np


def _ext():
    """Lazily resolve the compiled ``_mlrs`` extension.

    Imported LOCALLY (not at module level) to avoid a circular import: this
    module is itself imported at `mlrs/__init__.py` load time (`from . import
    metrics`, a submodule import), before `_load_ext` is defined further down
    that same file — mirrors `random_projection.py`'s
    `johnson_lindenstrauss_min_dim` lazy-import convention exactly.
    """
    from . import _load_ext

    return _load_ext()


def _labels_i32(y):
    """Cast a label vector to a `PyO3 Vec<i32>`-compatible numpy dtype
    (SPEC §5 METR-SHIM-01)."""
    return np.asarray(y).ravel().astype(np.int32)


def _f64(y):
    return np.asarray(y).ravel().astype(np.float64)


def _sw(sample_weight):
    if sample_weight is None:
        return None
    return np.asarray(sample_weight).ravel().astype(np.float64)


# ==================== accuracy_score (METR-CLS-01) ====================


def accuracy_score(y_true, y_pred, *, sample_weight=None, normalize=True):
    """Fraction (or count, if ``normalize=False``) of exact matches.

    Matches ``sklearn.metrics.accuracy_score``. Returns a plain python
    ``float``.
    """
    ext = _ext()
    got = ext.accuracy_score(
        _labels_i32(y_true), _labels_i32(y_pred), _sw(sample_weight), bool(normalize)
    )
    return float(got)


# ==================== confusion_matrix (METR-CLS-02) ====================


def confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None):
    """The ``C×C`` confusion matrix. Matches
    ``sklearn.metrics.confusion_matrix``.

    Returns an ``int64`` array when unweighted (``sample_weight=None``), a
    ``float64`` array otherwise (weighted counts are not generally integral).
    """
    ext = _ext()
    labels_arr = None if labels is None else _labels_i32(labels)
    got = ext.confusion_matrix(
        _labels_i32(y_true), _labels_i32(y_pred), labels_arr, _sw(sample_weight)
    )
    dtype = np.int64 if sample_weight is None else np.float64
    return np.asarray(got, dtype=dtype)


# ==================== precision/recall/f1 (METR-CLS-03/04/05) ====================


def _zero_division_to_f64(zero_division):
    """Map sklearn's ``zero_division ∈ {0, 1, 'warn', np.nan}`` to the ``f64``
    sentinel the ``_mlrs`` layer expects (``NaN`` represents the ``'nan'``
    policy; ``'warn'`` maps to ``0`` at this boundary, SPEC §4)."""
    if isinstance(zero_division, str):
        if zero_division == "warn":
            return 0.0
        raise ValueError(f"unsupported zero_division string {zero_division!r}")
    if zero_division is np.nan or (isinstance(zero_division, float) and np.isnan(zero_division)):
        return float("nan")
    return float(zero_division)


def _prf(
    ext_scalar_fn,
    ext_per_class_fn,
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division=0,
):
    labels_arr = None if labels is None else _labels_i32(labels)
    zd = _zero_division_to_f64(zero_division)
    if average is None:
        got = ext_per_class_fn(_labels_i32(y_true), _labels_i32(y_pred), labels_arr, _sw(sample_weight), zd)
        return np.asarray(got, dtype=np.float64)
    got = ext_scalar_fn(
        _labels_i32(y_true),
        _labels_i32(y_pred),
        labels_arr,
        int(pos_label),
        str(average),
        _sw(sample_weight),
        zd,
    )
    return float(got)


def precision_score(
    y_true, y_pred, *, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division=0
):
    """Matches ``sklearn.metrics.precision_score``. ``average=None`` returns
    a per-class ``np.ndarray`` in the resolved class order."""
    ext = _ext()
    return _prf(
        ext.precision_score,
        ext.precision_score_per_class,
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def recall_score(
    y_true, y_pred, *, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division=0
):
    """Matches ``sklearn.metrics.recall_score``. ``average=None`` returns a
    per-class ``np.ndarray`` in the resolved class order."""
    ext = _ext()
    return _prf(
        ext.recall_score,
        ext.recall_score_per_class,
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def f1_score(
    y_true, y_pred, *, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division=0
):
    """Matches ``sklearn.metrics.f1_score``. ``average=None`` returns a
    per-class ``np.ndarray`` in the resolved class order."""
    ext = _ext()
    return _prf(
        ext.f1_score,
        ext.f1_score_per_class,
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


# ==================== log_loss (METR-CLS-06) ====================

# The machine epsilon of float64 — the ACTUAL default clipping epsilon the
# pinned scikit-learn==1.9.0 uses (`xp.finfo(y_proba.dtype).eps`), empirically
# confirmed at TASK-08 Rust Green time; NOT sklearn's older/deprecated fixed
# `1e-15` default some prior versions used. `eps='auto'` maps here (SPEC §4
# Q5, corrected).
_LOG_LOSS_AUTO_EPS = float(np.finfo(np.float64).eps)


def log_loss(y_true, y_pred, *, normalize=True, sample_weight=None, labels=None, eps="auto"):
    """Matches ``sklearn.metrics.log_loss``. ``y_pred`` is the row-major
    ``n_samples × n_classes`` probability matrix (or a 1-D positive-class
    probability vector for the binary case, which is expanded to
    ``[1-p, p]`` columns here, mirroring sklearn's own ``y_proba.ndim==1``
    handling).

    ``eps='auto'`` (the default) maps to the float64 machine epsilon (see
    the module-level `_LOG_LOSS_AUTO_EPS` note); a numeric `eps` is used
    verbatim.
    """
    ext = _ext()
    y_true_i32 = _labels_i32(y_true)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_pred_arr.ndim == 1:
        y_pred_arr = np.column_stack([1.0 - y_pred_arr, y_pred_arr])
    n_classes = y_pred_arr.shape[1]
    y_prob_flat = np.ascontiguousarray(y_pred_arr).ravel()

    resolved_eps = _LOG_LOSS_AUTO_EPS if eps == "auto" else float(eps)
    labels_arr = None if labels is None else _labels_i32(labels)

    got = ext.log_loss(
        y_true_i32, y_prob_flat, int(n_classes), labels_arr, _sw(sample_weight), resolved_eps, bool(normalize)
    )
    return float(got)


# ==================== roc_auc_score (METR-CLS-07/08) ====================


def roc_auc_score(
    y_true, y_score, *, average="macro", sample_weight=None, multi_class="raise", labels=None
):
    """Matches ``sklearn.metrics.roc_auc_score``. Dispatches on ``y_score``'s
    shape: a 1-D array is the BINARY path (``pos_label`` fixed to ``1``,
    mirroring the fixture convention); a 2-D ``(n_samples, n_classes)`` array
    is the MULTICLASS OvR/OvO path, requiring an explicit
    ``multi_class ∈ {'ovr', 'ovo'}`` (sklearn's own default ``'raise'``
    raises a ``ValueError`` for multiclass input with no explicit choice).
    """
    ext = _ext()
    y_true_i32 = _labels_i32(y_true)
    y_score_arr = np.asarray(y_score, dtype=np.float64)

    if y_score_arr.ndim == 1:
        got = ext.roc_auc_score_binary(y_true_i32, y_score_arr.ravel(), 1, _sw(sample_weight))
        return float(got)

    if multi_class == "raise":
        raise ValueError(
            "multi_class must be 'ovr' or 'ovo' for a multiclass y_score "
            "(sklearn.metrics.roc_auc_score's own 'raise' default behavior)"
        )
    n_classes = y_score_arr.shape[1]
    y_score_flat = np.ascontiguousarray(y_score_arr).ravel()
    got = ext.roc_auc_score_multiclass(
        y_true_i32, y_score_flat, int(n_classes), str(multi_class), str(average), _sw(sample_weight)
    )
    return float(got)


# ==================== precision_recall_curve (METR-CLS-09) ====================


def precision_recall_curve(y_true, probas_pred, *, pos_label=1, sample_weight=None):
    """Matches ``sklearn.metrics.precision_recall_curve``. Returns
    ``(precision, recall, thresholds)`` as a 3-tuple of ``np.ndarray``."""
    ext = _ext()
    precision, recall, thresholds = ext.precision_recall_curve(
        _labels_i32(y_true), _f64(probas_pred), int(pos_label), _sw(sample_weight)
    )
    return (
        np.asarray(precision, dtype=np.float64),
        np.asarray(recall, dtype=np.float64),
        np.asarray(thresholds, dtype=np.float64),
    )


# ==================== r2_score / mean_squared_error / mean_absolute_error ====================
# (METR-REG-01/02/03). Multioutput regression is a NON-GOAL (SPEC §2,
# Plan-Check Issue 3): fail CLOSED with NotImplementedError on a 2-D input or
# a non-default `multioutput=`, BEFORE any `.ravel()` call — never silently
# `ravel()` a 2-D array into a mathematically wrong 1-D value.


def _reject_multioutput(y_true, y_pred, multioutput):
    if multioutput != "uniform_average":
        raise NotImplementedError(
            "mlrs.metrics: multioutput is not supported; only the default "
            "multioutput='uniform_average' (single-output) is implemented"
        )
    if np.asarray(y_true).ndim > 1 or np.asarray(y_pred).ndim > 1:
        raise NotImplementedError(
            "mlrs.metrics: multioutput (2-D y_true/y_pred) is not supported; "
            "pass 1-D y_true/y_pred"
        )


def r2_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Matches ``sklearn.metrics.r2_score`` (1-D, single-output only —
    multioutput is a non-goal, SPEC §2)."""
    _reject_multioutput(y_true, y_pred, multioutput)
    ext = _ext()
    got = ext.r2_score(_f64(y_true), _f64(y_pred), _sw(sample_weight))
    return float(got)


def mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Matches ``sklearn.metrics.mean_squared_error`` (MSE only — no
    ``squared=`` parameter, SPEC §2 non-goal; 1-D single-output only)."""
    _reject_multioutput(y_true, y_pred, multioutput)
    ext = _ext()
    got = ext.mean_squared_error(_f64(y_true), _f64(y_pred), _sw(sample_weight))
    return float(got)


def mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Matches ``sklearn.metrics.mean_absolute_error`` (1-D single-output
    only)."""
    _reject_multioutput(y_true, y_pred, multioutput)
    ext = _ext()
    got = ext.mean_absolute_error(_f64(y_true), _f64(y_pred), _sw(sample_weight))
    return float(got)
