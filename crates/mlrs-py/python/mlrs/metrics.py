"""``mlrs.metrics`` — sklearn-signature-faithful free functions (METR-SHIM-01,
extended to sklearn's FULL parameter surface by METR-PARAM-01).

Mirrors ``sklearn.metrics`` for the classification + regression metrics in
roadmap Phase 24's metrics criterion (accuracy/confusion/precision/recall/f1/
log_loss/roc_auc/precision_recall_curve/r2/mse/mae). Each function
normalizes its inputs (``np.asarray(...)``, the right dtype), delegates to the
corresponding ``_mlrs.<fn>`` PyO3 free function
(``crates/mlrs-py/src/metrics.rs``), and wraps the return in the
sklearn-faithful shape (scalar -> ``float``; confusion -> ``np.ndarray``;
PR-curve -> a 3-tuple of ``np.ndarray``).

These are plain free functions, NOT :class:`~mlrs.base.MlrsBase` subclasses —
no ``output_type``/``_normalize``/``_to_output`` estimator machinery applies
(they take already-materialized label/target vectors, never a device array).
Integer (or boolean) class labels only; string labels are a separate, unrelated
non-goal.

Every O(n) argument crosses into ``_mlrs`` as a **zero-copy pyarrow
``float64`` array** (:func:`_pa`), not as a Python sequence: PyO3's ``Vec<T>``
extraction walks the sequence protocol element by element (~44 ns/element,
i.e. 44 ms of pure binding cost on a one-million-sample call), which used to
dominate every metric on this surface. Labels ride the same float64 capsule and
are rounded back to ``i32`` in Rust — the bridge's integer ingress is
``uint32``-only and metric labels can be negative. The O(K) parameters
(``labels``, ``multioutput`` weights) still cross as plain lists, where the
per-element cost is irrelevant.

Parameter surface (METR-PARAM-01). Every parameter of the eleven pinned
``scikit-learn==1.9.0`` signatures is implemented, and the VALUE logic for each
lives in Rust (``crates/mlrs-algos/src/metrics/``) — this module only parses
strings, validates domains with sklearn's own messages, and reshapes:

===========================  ====================================================
function                     parameters
===========================  ====================================================
``accuracy_score``           ``normalize``, ``sample_weight``
``confusion_matrix``         ``labels``, ``sample_weight``, ``normalize``
``precision/recall/f1``      ``labels``, ``pos_label``, ``average``,
                             ``sample_weight``, ``zero_division``
``log_loss``                 ``normalize``, ``sample_weight``, ``labels``, ``eps``
``roc_auc_score``            ``average``, ``sample_weight``, ``max_fpr``,
                             ``multi_class``, ``labels``
``precision_recall_curve``   ``pos_label``, ``sample_weight``,
                             ``drop_intermediate``
``r2_score``                 ``sample_weight``, ``multioutput``, ``force_finite``
``mean_squared_error``       ``sample_weight``, ``multioutput``
``mean_absolute_error``      ``sample_weight``, ``multioutput``
===========================  ====================================================

Two DELIBERATE divergences from sklearn survive the parameter work, both
inherited from METR-SHIM-01 and both in the "gate the error, not a value"
direction:

* ``roc_auc_score`` with a single class present in ``y_true`` raises
  ``ValueError`` where sklearn 1.9.0 returns ``NaN`` with an
  ``UndefinedMetricWarning``.
* ``roc_auc_score(multi_class='ovo', sample_weight=...)`` raises — as sklearn
  itself does — rather than silently ignoring the weights.

The ``average=None`` (per-class vector) case for
``precision_score``/``recall_score``/``f1_score``/``roc_auc_score`` dispatches
internally to the ``_mlrs.<fn>_per_class`` sibling PyO3 function; this module's
own signatures stay sklearn-faithful and the split is hidden here.
"""

import warnings

import numpy as np

try:  # pragma: no cover - sklearn is already a hard dependency of `mlrs`
    from sklearn.exceptions import UndefinedMetricWarning
except ImportError:  # pragma: no cover
    class UndefinedMetricWarning(UserWarning):
        """Fallback used only if sklearn is unavailable at import time."""


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
    """Cast a label vector to an integer numpy dtype for the SHIM's own logic
    (``np.unique``, ``np.isin``, ``pos_label`` resolution). What crosses the
    FFI is the :func:`_pa` form of the same values — see the module note."""
    return np.asarray(y).ravel().astype(np.int32)


def _f64(y):
    return np.ascontiguousarray(np.asarray(y).ravel(), dtype=np.float64)


def _pa(y):
    """Wrap a 1-D vector as the ``float64`` pyarrow array the ``_mlrs`` layer
    ingests ZERO-COPY (``crates/mlrs-py/src/metrics.rs``).

    ``pa.array`` of a C-contiguous ``float64`` numpy array is ~1 µs at ANY
    length — it adopts the buffer. The alternative (PyO3's ``Vec<f64>``
    extraction over the sequence protocol) measured ~44 ns/element, which made
    a one-million-sample metric spend 44 ms in the binding and ~1 ms in the
    reduction. Labels ride the same float64 path and are rounded back to
    ``i32`` on the Rust side: the bridge's integer ingress is ``uint32``-only
    and metric labels can be negative.
    """
    import pyarrow as pa

    return pa.array(_f64(y))


def _pa_labels(y):
    """:func:`_pa` for an integral label vector (values preserved exactly)."""
    return _pa(np.asarray(y).ravel())


def _param_labels(labels):
    """The O(K) ``labels``/``classes`` PARAMETER, as a plain list of ints —
    small enough that per-element extraction is free, and the Rust side wants a
    ``Vec<i32>``."""
    if labels is None:
        return None
    return [int(v) for v in np.asarray(labels).ravel()]


def _sw(sample_weight):
    if sample_weight is None:
        return None
    return _pa(sample_weight)


# ==================== accuracy_score (METR-CLS-01) ====================


def accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    """Fraction (or count, if ``normalize=False``) of exact matches.

    Matches ``sklearn.metrics.accuracy_score``. Returns a plain python
    ``float``.
    """
    ext = _ext()
    got = ext.accuracy_score(
        _pa_labels(y_true), _pa_labels(y_pred), _sw(sample_weight), bool(normalize)
    )
    return float(got)


# ==================== confusion_matrix (METR-CLS-02) ====================

_CM_NORMALIZE = ("true", "pred", "all")


def confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None):
    """The ``C×C`` confusion matrix. Matches
    ``sklearn.metrics.confusion_matrix``.

    ``normalize ∈ {'true', 'pred', 'all', None}`` (METR-PARAM-01) divides by
    the row sum, the column sum, or the grand total; a zero divisor produces a
    zero row/column/matrix rather than NaN (sklearn ``nan_to_num``s its own
    division).

    Returns an ``int64`` array for raw unweighted counts and a ``float64``
    array whenever the counts are weighted or normalized (neither is generally
    integral).
    """
    if normalize is not None and normalize not in _CM_NORMALIZE:
        raise ValueError(
            "The 'normalize' parameter of confusion_matrix must be a str among "
            "{'pred', 'true', 'all'} or None. Got %r instead." % (normalize,)
        )
    ext = _ext()
    y_true_i32 = _labels_i32(y_true)
    labels_arr = _param_labels(labels)
    if labels_arr is not None and not np.isin(labels_arr, y_true_i32).any():
        # sklearn's own guard — an all-disjoint `labels` would otherwise return
        # a silently all-zero matrix.
        raise ValueError("At least one label specified must be in y_true")
    got = ext.confusion_matrix(
        _pa_labels(y_true_i32),
        _pa_labels(y_pred),
        labels_arr,
        _sw(sample_weight),
        normalize,
    )
    dtype = np.int64 if (sample_weight is None and normalize is None) else np.float64
    return np.asarray(got, dtype=dtype)


# ==================== precision/recall/f1 (METR-CLS-03/04/05) ====================

_PRF_AVERAGE = ("binary", "micro", "macro", "weighted", "samples")


def _zero_division_to_f64(zero_division):
    """Map sklearn's ``zero_division ∈ {0, 1, 'warn', np.nan}`` to the ``f64``
    sentinel the ``_mlrs`` layer expects (``NaN`` represents the ``'nan'``
    policy; ``'warn'`` maps to ``0`` at this boundary, SPEC §4)."""
    if isinstance(zero_division, str):
        if zero_division == "warn":
            return 0.0
        raise ValueError(
            "The 'zero_division' parameter must be a float among {0.0, 1.0}, "
            "numpy.nan or a str among {'warn'}. Got %r instead." % (zero_division,)
        )
    if zero_division is np.nan or (isinstance(zero_division, float) and np.isnan(zero_division)):
        return float("nan")
    if float(zero_division) not in (0.0, 1.0):
        raise ValueError(
            "The 'zero_division' parameter must be a float among {0.0, 1.0}, "
            "numpy.nan or a str among {'warn'}. Got %r instead." % (zero_division,)
        )
    return float(zero_division)


def _check_average(average, fn_name):
    """Validate ``average`` against sklearn's option set, and reject the
    multilabel-only ``'samples'`` the way sklearn does for 1-D targets."""
    if average is None:
        return
    if average not in _PRF_AVERAGE:
        raise ValueError(
            "The 'average' parameter of %s must be a str among "
            "{'binary', 'micro', 'macro', 'weighted', 'samples'} or None. "
            "Got %r instead." % (fn_name, average)
        )
    if average == "samples":
        raise ValueError(
            "Samplewise metrics are not available outside of multilabel classification."
        )


def _prf(
    ext_scalar_fn,
    ext_per_class_fn,
    warn_message,
    fn_name,
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
):
    _check_average(average, fn_name)
    labels_arr = _param_labels(labels)
    zd = _zero_division_to_f64(zero_division)
    if average is None:
        values, zero_division_hit, classes = ext_per_class_fn(
            _pa_labels(y_true), _pa_labels(y_pred), labels_arr, _sw(sample_weight), zd
        )
        out = np.asarray(values, dtype=np.float64)
    else:
        value, zero_division_hit, classes = ext_scalar_fn(
            _pa_labels(y_true),
            _pa_labels(y_pred),
            labels_arr,
            int(pos_label),
            str(average),
            _sw(sample_weight),
            zd,
        )
        out = float(value)
    if average == "binary" and labels is None:
        # sklearn's two binary-only guards. `classes` is the resolved class
        # order the Rust layer already had to build (the sorted unique of
        # ``y_true ∪ y_pred`` when ``labels`` is None), so neither guard costs
        # an extra pass over the targets.
        #
        # They are skipped when the caller passed an explicit ``labels``:
        # sklearn derives its ``present_labels`` from the DATA there, which
        # `classes` no longer reports, and applying the guards to the
        # caller's list would reject calls sklearn accepts (e.g.
        # ``labels=[0, 1, 2]`` over a binary target). The VALUE is unaffected —
        # ``average='binary'`` reads the ``pos_label`` entry either way.
        if len(classes) > 2:
            raise ValueError(
                "Target is multiclass but average='binary'. Please choose another "
                "average setting, one of [None, 'micro', 'macro', 'weighted']."
            )
        # A pos_label absent from a SINGLE-class target is a zero-division
        # case, not an error — sklearn only rejects it once two labels are
        # actually present.
        if len(classes) >= 2 and int(pos_label) not in classes:
            # `%s` of a numpy array, not `%r` — sklearn's message renders the
            # label set as `[0 1]`, not `array([0, 1])`.
            raise ValueError(
                "pos_label=%r is not a valid label. It should be one of %s"
                % (pos_label, np.asarray(classes))
            )
    if zero_division_hit and isinstance(zero_division, str):
        # sklearn's `zero_division="warn"` default: same VALUE as 0, plus an
        # UndefinedMetricWarning. The Rust layer reports whether the reported
        # number actually consulted the policy, so no second O(n) pass is
        # needed to find out.
        warnings.warn(
            warn_message + " Use `zero_division` parameter to control this behavior.",
            UndefinedMetricWarning,
            stacklevel=3,
        )
    return out


def precision_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
):
    """Matches ``sklearn.metrics.precision_score``. ``average=None`` returns
    a per-class ``np.ndarray`` in the resolved class order."""
    ext = _ext()
    return _prf(
        ext.precision_score,
        ext.precision_score_per_class,
        "Precision is ill-defined and being set to 0.0 due to no predicted samples.",
        "precision_score",
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def recall_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
):
    """Matches ``sklearn.metrics.recall_score``. ``average=None`` returns a
    per-class ``np.ndarray`` in the resolved class order."""
    ext = _ext()
    return _prf(
        ext.recall_score,
        ext.recall_score_per_class,
        "Recall is ill-defined and being set to 0.0 due to no true samples.",
        "recall_score",
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def f1_score(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
):
    """Matches ``sklearn.metrics.f1_score``. ``average=None`` returns a
    per-class ``np.ndarray`` in the resolved class order."""
    ext = _ext()
    return _prf(
        ext.f1_score,
        ext.f1_score_per_class,
        "F-score is ill-defined and being set to 0.0 due to no true nor predicted samples.",
        "f1_score",
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


def log_loss(y_true, y_proba=None, *, normalize=True, sample_weight=None, labels=None, eps="auto"):
    """Matches ``sklearn.metrics.log_loss``. ``y_proba`` is the row-major
    ``n_samples × n_classes`` probability matrix (or a 1-D positive-class
    probability vector for the binary case, which is expanded to
    ``[1-p, p]`` columns here, mirroring sklearn's own ``y_proba.ndim==1``
    handling).

    ``eps='auto'`` (the default) maps to the float64 machine epsilon (see
    the module-level `_LOG_LOSS_AUTO_EPS` note); a numeric `eps` is used
    verbatim. sklearn 1.9.0 has no ``eps`` parameter at all (it was removed
    after 1.5); mlrs keeps it as an explicit knob whose default reproduces
    sklearn's hard-coded behavior.
    """
    ext = _ext()
    y_true_pa = _pa_labels(y_true)
    y_proba_arr = np.asarray(y_proba, dtype=np.float64)
    if y_proba_arr.ndim == 1:
        y_proba_arr = np.column_stack([1.0 - y_proba_arr, y_proba_arr])
    n_classes = y_proba_arr.shape[1]
    y_prob_flat = np.ascontiguousarray(y_proba_arr).ravel()

    resolved_eps = _LOG_LOSS_AUTO_EPS if eps == "auto" else float(eps)
    labels_arr = _param_labels(labels)

    got = ext.log_loss(
        y_true_pa,
        _pa(y_prob_flat),
        int(n_classes),
        labels_arr,
        _sw(sample_weight),
        resolved_eps,
        bool(normalize),
    )
    return float(got)


# ==================== roc_auc_score (METR-CLS-07/08) ====================

_ROC_AUC_AVERAGE = ("micro", "macro", "samples", "weighted")


def roc_auc_score(
    y_true,
    y_score,
    *,
    average="macro",
    sample_weight=None,
    max_fpr=None,
    multi_class="raise",
    labels=None,
):
    """Matches ``sklearn.metrics.roc_auc_score``.

    Dispatches on the target type exactly as sklearn does: more than two
    classes in ``y_true`` (or a ``y_score`` with more than two columns) takes
    the MULTICLASS OvR/OvO path and requires an explicit
    ``multi_class ∈ {'ovr', 'ovo'}``; otherwise the BINARY path integrates a
    single ROC curve whose positive class is the LARGER of the two labels
    present (sklearn's ``label_binarize(y_true, classes=np.unique(y_true))``
    convention — not a hard-coded ``1``).

    ``max_fpr ∈ (0, 1]`` (METR-PARAM-01) computes the McClish-standardized
    partial AUC and is binary-only, as in sklearn. ``labels`` fixes the
    multiclass column order (it must be sorted, unique, and cover ``y_true``).
    ``average=None`` returns the per-class OvR vector.
    """
    if average is not None and average not in _ROC_AUC_AVERAGE:
        raise ValueError(
            "The 'average' parameter of roc_auc_score must be a str among "
            "{'micro', 'macro', 'samples', 'weighted'} or None. Got %r instead." % (average,)
        )
    if max_fpr is not None and not (0.0 < float(max_fpr) <= 1.0):
        raise ValueError(
            "The 'max_fpr' parameter of roc_auc_score must be a float in the range "
            "(0.0, 1.0] or None. Got %r instead." % (max_fpr,)
        )

    ext = _ext()
    y_true_i32 = _labels_i32(y_true)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    present = np.unique(y_true_i32)
    is_multiclass = len(present) > 2 or (y_score_arr.ndim == 2 and y_score_arr.shape[1] > 2)

    if not is_multiclass:
        if y_score_arr.ndim != 1:
            raise ValueError(
                "y should be a 1d array, got an array of shape %r instead."
                % (y_score_arr.shape,)
            )
        # sklearn binarizes against `np.unique(y_true)`, making the SECOND
        # (larger) label the positive one; `labels` is ignored on this path.
        pos_label = int(present[-1]) if len(present) else 1
        got = ext.roc_auc_score_binary(
            _pa_labels(y_true_i32),
            _pa(y_score_arr),
            pos_label,
            _sw(sample_weight),
            None if max_fpr is None else float(max_fpr),
        )
        return float(got)

    # ---- multiclass ----
    if max_fpr is not None and float(max_fpr) != 1.0:
        raise ValueError(
            "Partial AUC computation not available in multiclass setting, 'max_fpr' "
            "must be set to `None`, received `max_fpr=%r` instead" % (max_fpr,)
        )
    if multi_class == "raise":
        raise ValueError("multi_class must be in ('ovo', 'ovr')")
    if multi_class not in ("ovo", "ovr"):
        raise ValueError(
            "multi_class='%s' is not supported for multiclass ROC AUC, multi_class "
            "must be in ('ovo', 'ovr')" % (multi_class,)
        )
    average_options = ("macro", "weighted", None)
    if multi_class == "ovr":
        average_options = ("micro",) + average_options
    if average not in average_options:
        raise ValueError(
            "average must be one of {0} for multiclass problems".format(average_options)
        )
    if average is None and multi_class == "ovo":
        raise NotImplementedError("average=None is not implemented for multi_class='ovo'.")
    if y_score_arr.ndim != 2:
        raise ValueError(
            "`y_score` needs to be of shape `(n_samples, n_classes)`, since `y_true` "
            "contains multiple classes. Got `y_score.shape=%r`." % (y_score_arr.shape,)
        )
    if not np.allclose(1, y_score_arr.sum(axis=1)):
        raise ValueError(
            "Target scores need to be probabilities for multiclass roc_auc, i.e. they "
            "should sum up to 1.0 over classes"
        )

    if labels is None:
        classes = present
    else:
        classes = _labels_i32(labels)
        uniq = np.unique(classes)
        if len(uniq) != len(classes):
            raise ValueError("Parameter 'labels' must be unique")
        if not np.array_equal(uniq, classes):
            raise ValueError("Parameter 'labels' must be ordered")
        if len(np.setdiff1d(y_true_i32, classes)):
            raise ValueError("'y_true' contains labels not in parameter 'labels'")
    if len(classes) != y_score_arr.shape[1]:
        if labels is None:
            raise ValueError(
                "Number of classes in y_true not equal to the number of columns in "
                "'y_score'"
            )
        raise ValueError(
            "Number of given labels, %d, not equal to the number of columns in "
            "'y_score', %d" % (len(classes), y_score_arr.shape[1])
        )

    y_score_flat = _pa(np.ascontiguousarray(y_score_arr).ravel())
    class_list = _param_labels(classes)
    if average is None:
        got = ext.roc_auc_score_multiclass_per_class(
            _pa_labels(y_true_i32), y_score_flat, class_list, str(multi_class), _sw(sample_weight)
        )
        return np.asarray(got, dtype=np.float64)
    got = ext.roc_auc_score_multiclass(
        _pa_labels(y_true_i32),
        y_score_flat,
        class_list,
        str(multi_class),
        str(average),
        _sw(sample_weight),
    )
    return float(got)


# ==================== precision_recall_curve (METR-CLS-09) ====================


def _resolve_pos_label(pos_label, y_true_i32):
    """sklearn's ``_check_pos_label_consistency``: ``pos_label=None`` is only
    unambiguous for ``{0, 1}``/``{-1, 1}``-valued targets."""
    if pos_label is not None:
        return int(pos_label)
    classes = np.unique(y_true_i32)
    if not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError(
            "y_true takes value in {%s} and pos_label is not specified: either make "
            "y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly."
            % (", ".join(str(int(c)) for c in classes),)
        )
    return 1


def precision_recall_curve(
    y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=False
):
    """Matches ``sklearn.metrics.precision_recall_curve``. Returns
    ``(precision, recall, thresholds)`` as a 3-tuple of ``np.ndarray``.

    ``pos_label=None`` (the sklearn default) resolves to ``1`` for
    ``{0, 1}``/``{-1, 1}`` targets and raises otherwise;
    ``drop_intermediate=True`` (METR-PARAM-01) removes the thresholds that add
    no recall — the interior points of a vertical run on the PR plot.
    """
    ext = _ext()
    y_true_i32 = _labels_i32(y_true)
    resolved_pos_label = _resolve_pos_label(pos_label, y_true_i32)
    if not (y_true_i32 == resolved_pos_label).any():
        warnings.warn(
            "No positive class found in y_true, recall is set to one for all thresholds."
        )
    # The three columns come back as pyarrow arrays (they are O(n) long in the
    # worst case); `np.asarray` views them in place.
    precision, recall, thresholds = ext.precision_recall_curve(
        _pa_labels(y_true_i32),
        _pa(y_score),
        resolved_pos_label,
        _sw(sample_weight),
        bool(drop_intermediate),
    )
    return (
        np.asarray(precision, dtype=np.float64),
        np.asarray(recall, dtype=np.float64),
        np.asarray(thresholds, dtype=np.float64),
    )


# ==================== r2_score / mean_squared_error / mean_absolute_error ====================
# (METR-REG-01/02/03 + the `multioutput`/`force_finite` parameters of
# METR-PARAM-01). 2-D `y_true`/`y_pred` are supported: the pair is flattened
# ROW-MAJOR and the Rust layer accumulates per output column in one pass.


def _check_reg_targets(y_true, y_pred, multioutput, allowed_strings, fn_name):
    """Shared regression input/parameter check.

    Returns ``(y_true_flat, y_pred_flat, n_samples, n_outputs, mo_string,
    mo_weights)`` where ``mo_string`` is either one of ``allowed_strings`` or
    the ``"weights"`` sentinel the PyO3 layer pairs with ``mo_weights``.
    Error messages mirror sklearn's ``_check_reg_targets``.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if y_true_arr.ndim > 2 or y_pred_arr.ndim > 2:
        raise ValueError("y_true and y_pred must be 1-D or 2-D arrays")
    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred have different shapes (%r != %r)"
            % (y_true_arr.shape, y_pred_arr.shape)
        )
    n_samples, n_outputs = y_true_arr.shape
    if n_samples == 0:
        raise ValueError(
            "Found array with 0 sample(s) (shape=(0, %d)) while a minimum of 1 is "
            "required." % (n_outputs,)
        )

    mo_weights = None
    if isinstance(multioutput, str):
        if multioutput not in allowed_strings:
            raise ValueError(
                "The 'multioutput' parameter of %s must be a str among {%s} or an "
                "array-like. Got %r instead."
                % (fn_name, ", ".join(repr(s) for s in allowed_strings), multioutput)
            )
        mo_string = multioutput
    else:
        weights = np.asarray(multioutput, dtype=np.float64).ravel()
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in multi-output cases.")
        if len(weights) != n_outputs:
            raise ValueError(
                "There must be equally many custom weights (%d) as outputs (%d)."
                % (len(weights), n_outputs)
            )
        mo_string = "weights"
        mo_weights = [float(v) for v in weights]

    return (
        np.ascontiguousarray(y_true_arr).ravel(),
        np.ascontiguousarray(y_pred_arr).ravel(),
        n_samples,
        n_outputs,
        mo_string,
        mo_weights,
    )


def _reg_out(values, mo_string):
    """``raw_values`` keeps the per-output vector; every other reduction is a
    scalar (the PyO3 layer returns a 1-element vector for it)."""
    if mo_string == "raw_values":
        return np.asarray(values, dtype=np.float64)
    return float(values[0])


def r2_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", force_finite=True):
    """Matches ``sklearn.metrics.r2_score``, including the 2-D multioutput
    forms (``'raw_values'``, ``'uniform_average'``, ``'variance_weighted'`` or
    an explicit per-output weight vector) and ``force_finite``
    (METR-PARAM-01).

    With ``force_finite=False`` a constant ``y_true`` yields ``-inf`` (or
    ``NaN`` for an exactly-matching prediction) instead of the clamped
    ``0.0``/``1.0``.
    """
    y_true_flat, y_pred_flat, n_samples, n_outputs, mo_string, mo_weights = _check_reg_targets(
        y_true,
        y_pred,
        multioutput,
        ("raw_values", "uniform_average", "variance_weighted"),
        "r2_score",
    )
    if n_samples < 2:
        warnings.warn(
            "R^2 score is not well-defined with less than two samples.",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return float("nan")
    ext = _ext()
    got = ext.r2_score(
        _pa(y_true_flat),
        _pa(y_pred_flat),
        n_outputs,
        _sw(sample_weight),
        mo_string,
        mo_weights,
        bool(force_finite),
    )
    return _reg_out(got, mo_string)


def mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Matches ``sklearn.metrics.mean_squared_error`` (MSE only — no
    ``squared=`` parameter, removed from sklearn itself in 1.4), including the
    2-D ``multioutput`` forms (METR-PARAM-01)."""
    y_true_flat, y_pred_flat, _, n_outputs, mo_string, mo_weights = _check_reg_targets(
        y_true, y_pred, multioutput, ("raw_values", "uniform_average"), "mean_squared_error"
    )
    ext = _ext()
    got = ext.mean_squared_error(
        _pa(y_true_flat), _pa(y_pred_flat), n_outputs, _sw(sample_weight), mo_string, mo_weights
    )
    return _reg_out(got, mo_string)


def mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Matches ``sklearn.metrics.mean_absolute_error``, including the 2-D
    ``multioutput`` forms (METR-PARAM-01)."""
    y_true_flat, y_pred_flat, _, n_outputs, mo_string, mo_weights = _check_reg_targets(
        y_true, y_pred, multioutput, ("raw_values", "uniform_average"), "mean_absolute_error"
    )
    ext = _ext()
    got = ext.mean_absolute_error(
        _pa(y_true_flat), _pa(y_pred_flat), n_outputs, _sw(sample_weight), mo_string, mo_weights
    )
    return _reg_out(got, mo_string)
