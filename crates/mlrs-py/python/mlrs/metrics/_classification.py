"""The rest of ``sklearn.metrics``' classification surface.

Every VALUE is computed in Rust (``mlrs_algos::metrics::{prfs,classification}``,
bound in ``crates/mlrs-py/src/metrics_ext.rs``); this module only validates
inputs with sklearn's own messages, emits sklearn's warnings, resolves label
conventions (``pos_label=None``, ``LabelBinarizer``'s ``{-1, +1}`` rule) and
shapes the outputs.

``precision_score`` / ``recall_score`` / ``f1_score`` are re-defined here as
views of :func:`precision_recall_fscore_support` — exactly sklearn's own
structure — which is what gives them multilabel-indicator input and
``average='samples'``. Their 1-D behavior is unchanged from the original shim
apart from one fix: with ``zero_division=np.nan`` a macro/weighted average now
skips the undefined classes, as sklearn's ``_nanaverage`` does.
"""

import warnings
from numbers import Real

import numpy as np

from ._base import (
    UndefinedMetricWarning,
    _ext,
    _owned,
    _pa,
    _pa_labels,
    _param_labels,
    _sw,
    _zero_division_to_f64,
)
from ._base import accuracy_score as _accuracy_score_1d
from ._base import confusion_matrix

__all__ = [
    "accuracy_score",
    "balanced_accuracy_score",
    "brier_score_loss",
    "class_likelihood_ratios",
    "classification_report",
    "cohen_kappa_score",
    "d2_brier_score",
    "d2_log_loss_score",
    "f1_score",
    "fbeta_score",
    "hamming_loss",
    "hinge_loss",
    "jaccard_score",
    "matthews_corrcoef",
    "multilabel_confusion_matrix",
    "precision_recall_fscore_support",
    "precision_score",
    "recall_score",
    "top_k_accuracy_score",
    "zero_one_loss",
]


# --------------------------------------------------------------------------- #
# target handling
# --------------------------------------------------------------------------- #


class _Targets:
    """A validated ``(y_true, y_pred)`` pair: 1-D labels, or a multilabel
    indicator pair flattened row-major with its ``n_labels``."""

    def __init__(self, kind, y_true, y_pred, n_labels=0):
        self.kind = kind  # "1d" or "multilabel"
        self.y_true = y_true
        self.y_pred = y_pred
        self.n_labels = n_labels

    @property
    def n_samples(self):
        if self.kind == "multilabel":
            return len(self.y_true) // self.n_labels
        return len(self.y_true)


def _is_2d_multi(a):
    return a.ndim == 2 and a.shape[1] > 1


def _check_targets(y_true, y_pred):
    """sklearn's ``_check_targets`` for the integer-label surface: a 1-D pair
    (binary/multiclass) or a multilabel-indicator pair. Raises sklearn's
    "Classification metrics can't handle a mix of ..." for a mixed pair."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if _is_2d_multi(yt) or _is_2d_multi(yp):
        if not (_is_2d_multi(yt) and _is_2d_multi(yp)):
            raise ValueError(
                "Classification metrics can't handle a mix of multilabel-indicator "
                "and multiclass targets"
            )
        if yt.shape != yp.shape:
            raise ValueError(
                "Found input variables with inconsistent numbers of samples: %r"
                % ([yt.shape[0], yp.shape[0]],)
            )
        for a in (yt, yp):
            if a.size and (a.min() < 0 or a.max() > 1):
                raise ValueError("multiclass-multioutput is not supported")
        n_labels = yt.shape[1]
        return _Targets(
            "multilabel",
            np.ascontiguousarray(yt, dtype=np.float64).ravel(),
            np.ascontiguousarray(yp, dtype=np.float64).ravel(),
            n_labels,
        )
    yt = yt.ravel()
    yp = yp.ravel()
    if len(yt) != len(yp):
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % ([len(yt), len(yp)],)
        )
    return _Targets("1d", yt, yp)


def _weights_or_none(sample_weight, n):
    if sample_weight is None:
        return None
    sw = np.asarray(sample_weight, dtype=np.float64).ravel()
    if len(sw) != n:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r" % ([n, len(sw)],)
        )
    return sw


def _label_cols(labels, n_labels):
    """``labels`` for a multilabel target: column indices in ``[0, n_labels)``,
    with sklearn's two range messages."""
    if labels is None:
        return None
    cols = np.asarray(labels).ravel().astype(np.int64)
    if len(cols) and not np.array_equal(cols, np.arange(n_labels)):
        if cols.max() > n_labels - 1:
            raise ValueError(
                "All labels must be in [0, n labels) for multilabel targets. Got %d > %d"
                % (cols.max(), n_labels - 1)
            )
        if cols.min() < 0:
            raise ValueError(
                "All labels must be in [0, n labels) for multilabel targets. Got %d < 0"
                % cols.min()
            )
    return [int(c) for c in cols]


def _warn_prf(average, modifier, msg_start, result_size):
    """sklearn's ``_warn_prf``, verbatim."""
    axis0, axis1 = "sample", "label"
    if average == "samples":
        axis0, axis1 = axis1, axis0
    msg = (
        "{0} ill-defined and being set to 0.0 {{0}} no {1} {2}s. Use `zero_division` "
        "parameter to control this behavior.".format(msg_start, modifier, axis0)
    )
    if result_size == 1:
        msg = msg.format("due to")
    else:
        msg = msg.format("in {0}s with".format(axis1))
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=4)


_SETWISE_AVERAGES = (None, "micro", "macro", "weighted", "samples")


def _check_set_wise(targets, average, labels, pos_label):
    """sklearn's ``_check_set_wise_labels`` (the part that needs no data
    pass): the ``average`` domain, the multilabel/binary incompatibility and
    the ``pos_label``-ignored warning. The two binary guards that need the
    resolved classes run after the Rust call (:func:`_binary_guards`)."""
    if average not in _SETWISE_AVERAGES and average != "binary":
        raise ValueError("average has to be one of " + str(_SETWISE_AVERAGES))
    if average == "binary":
        if targets.kind == "multilabel":
            raise ValueError(
                "Target is multilabel-indicator but average='binary'. Please choose "
                "another average setting, one of %r." % (list(_SETWISE_AVERAGES),)
            )
    elif pos_label not in (None, 1):
        warnings.warn(
            "Note that pos_label (set to %r) is ignored when average != 'binary' (got %r). "
            "You may use labels=[pos_label] to specify a single positive class."
            % (pos_label, average),
            UserWarning,
            stacklevel=3,
        )
    if average == "samples" and targets.kind != "multilabel":
        raise ValueError(
            "Samplewise metrics are not available outside of multilabel classification."
        )


def _binary_guards(classes, pos_label):
    if len(classes) > 2:
        raise ValueError(
            "Target is multiclass but average='binary'. Please choose another average "
            "setting, one of [None, 'micro', 'macro', 'weighted']."
        )
    if len(classes) >= 2 and int(pos_label) not in classes:
        raise ValueError(
            "pos_label=%r is not a valid label. It should be one of %s"
            % (pos_label, np.asarray(classes))
        )


def _result_size(average, n_out):
    return 1 if average in ("binary", "micro") else n_out


# --------------------------------------------------------------------------- #
# precision / recall / F-beta
# --------------------------------------------------------------------------- #


def precision_recall_fscore_support(
    y_true,
    y_pred,
    *,
    beta=1.0,
    labels=None,
    pos_label=1,
    average=None,
    warn_for=("precision", "recall", "f-score"),
    sample_weight=None,
    zero_division="warn",
):
    """Matches ``sklearn.metrics.precision_recall_fscore_support``, for 1-D
    integer targets and multilabel-indicator targets (``average='samples'``
    included). Returns ``(precision, recall, fbeta_score, support)``; the
    first three are floats and ``support`` is ``None`` for an averaged call,
    and all four are per-label arrays for ``average=None``."""
    if not isinstance(beta, Real) or beta < 0:
        raise ValueError(
            "The 'beta' parameter of precision_recall_fscore_support must be a float in "
            "the range [0.0, inf). Got %r instead." % (beta,)
        )
    zd = _zero_division_to_f64(zero_division)
    targets = _check_targets(y_true, y_pred)
    _check_set_wise(targets, average, labels, pos_label)
    sw = _weights_or_none(sample_weight, targets.n_samples)
    ext = _ext()
    if targets.kind == "1d":
        p, r, f, support, (pzd, rzd, fzd), classes = ext.prfs(
            _pa_labels(targets.y_true),
            _pa_labels(targets.y_pred),
            0,
            float(beta),
            _param_labels(labels),
            None,
            int(pos_label) if pos_label is not None else 1,
            "none" if average is None else str(average),
            _sw(sw),
            zd,
        )
        if average == "binary":
            _binary_guards(classes, pos_label)
    else:
        p, r, f, support, (pzd, rzd, fzd), _ = ext.prfs(
            _pa(targets.y_true),
            _pa(targets.y_pred),
            targets.n_labels,
            float(beta),
            None,
            _label_cols(labels, targets.n_labels),
            1,
            "none" if average is None else str(average),
            _sw(sw),
            zd,
        )
    if isinstance(zero_division, str):
        size = _result_size(average, len(p))
        if average == "samples":
            size = targets.n_samples
        if pzd and "precision" in warn_for:
            _warn_prf(average, "predicted", "Precision is", size)
        if rzd and "recall" in warn_for:
            _warn_prf(average, "true", "Recall is", size)
        if fzd and "f-score" in warn_for and not (np.isposinf(beta) or beta == 0):
            _warn_prf(average, "true nor predicted", "F-score is", size)
    if average is not None:
        return float(p[0]), float(r[0]), float(f[0]), None
    sup = np.asarray(support, dtype=np.float64)
    if sample_weight is None:
        sup = sup.astype(np.int64)
    return (
        np.asarray(p, dtype=np.float64),
        np.asarray(r, dtype=np.float64),
        np.asarray(f, dtype=np.float64),
        sup,
    )


def _single(index, warn_for, y_true, y_pred, beta, labels, pos_label, average, sample_weight, zero_division):
    out = precision_recall_fscore_support(
        y_true,
        y_pred,
        beta=beta,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=warn_for,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    return out[index]


def precision_score(
    y_true, y_pred, *, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn"
):
    """Matches ``sklearn.metrics.precision_score`` (1-D and multilabel)."""
    return _single(0, ("precision",), y_true, y_pred, 1.0, labels, pos_label, average, sample_weight, zero_division)


def recall_score(
    y_true, y_pred, *, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn"
):
    """Matches ``sklearn.metrics.recall_score`` (1-D and multilabel)."""
    return _single(1, ("recall",), y_true, y_pred, 1.0, labels, pos_label, average, sample_weight, zero_division)


def fbeta_score(
    y_true,
    y_pred,
    *,
    beta,
    labels=None,
    pos_label=1,
    average="binary",
    sample_weight=None,
    zero_division="warn",
):
    """Matches ``sklearn.metrics.fbeta_score``: ``(1+β²)·tp / ((1+β²)·tp + β²·fn
    + fp)``, computed from counts."""
    return _single(2, ("f-score",), y_true, y_pred, beta, labels, pos_label, average, sample_weight, zero_division)


def f1_score(
    y_true, y_pred, *, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn"
):
    """Matches ``sklearn.metrics.f1_score`` (1-D and multilabel)."""
    return _single(2, ("f-score",), y_true, y_pred, 1.0, labels, pos_label, average, sample_weight, zero_division)


def jaccard_score(
    y_true, y_pred, *, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn"
):
    """Matches ``sklearn.metrics.jaccard_score``: ``tp / (tp + fp + fn)`` per
    label, with sklearn's ``_average``. Unlike the precision/recall family,
    sklearn's ``jaccard_score`` does not accept ``zero_division=np.nan``."""
    if not isinstance(zero_division, str) and np.isnan(float(zero_division)):
        raise ValueError(
            "The 'zero_division' parameter of jaccard_score must be a float among {0, 1} or "
            "a str among {'warn'}. Got nan instead."
        )
    zd = _zero_division_to_f64(zero_division)
    targets = _check_targets(y_true, y_pred)
    _check_set_wise(targets, average, labels, pos_label)
    sw = _weights_or_none(sample_weight, targets.n_samples)
    ext = _ext()
    avg = "none" if average is None else str(average)
    if targets.kind == "1d":
        values, hit, classes = ext.jaccard_score(
            _pa_labels(targets.y_true),
            _pa_labels(targets.y_pred),
            0,
            _param_labels(labels),
            None,
            int(pos_label) if pos_label is not None else 1,
            avg,
            _sw(sw),
            zd,
        )
        if average == "binary":
            _binary_guards(classes, pos_label)
    else:
        values, hit, _ = ext.jaccard_score(
            _pa(targets.y_true),
            _pa(targets.y_pred),
            targets.n_labels,
            None,
            _label_cols(labels, targets.n_labels),
            1,
            avg,
            _sw(sw),
            zd,
        )
    if hit and isinstance(zero_division, str):
        size = targets.n_samples if average == "samples" else _result_size(average, len(values))
        _warn_prf(average, "true or predicted", "Jaccard is", size)
    if average is None:
        return np.asarray(values, dtype=np.float64)
    return float(values[0])


def multilabel_confusion_matrix(y_true, y_pred, *, sample_weight=None, labels=None, samplewise=False):
    """Matches ``sklearn.metrics.multilabel_confusion_matrix``: an
    ``(n_outputs, 2, 2)`` array of ``[[tn, fp], [fn, tp]]``; ``int64`` for
    unweighted counts, ``float64`` with ``sample_weight``."""
    targets = _check_targets(y_true, y_pred)
    sw = _weights_or_none(sample_weight, targets.n_samples)
    ext = _ext()
    if targets.kind == "1d":
        if samplewise:
            raise ValueError(
                "Samplewise metrics are not available outside of multilabel classification."
            )
        flat = ext.multilabel_confusion_matrix(
            _pa_labels(targets.y_true), _pa_labels(targets.y_pred), 0, _sw(sw), _param_labels(labels), None, False
        )
    else:
        flat = ext.multilabel_confusion_matrix(
            _pa(targets.y_true),
            _pa(targets.y_pred),
            targets.n_labels,
            _sw(sw),
            None,
            _label_cols(labels, targets.n_labels),
            bool(samplewise),
        )
    out = _owned(flat, shape=(-1, 2, 2))
    return out.astype(np.int64) if sample_weight is None else out


def classification_report(
    y_true,
    y_pred,
    *,
    labels=None,
    target_names=None,
    sample_weight=None,
    digits=2,
    output_dict=False,
    zero_division="warn",
):
    """Matches ``sklearn.metrics.classification_report`` (text or
    ``output_dict=True``). Every number in the report comes from
    :func:`precision_recall_fscore_support`; this function only formats."""
    targets = _check_targets(y_true, y_pred)
    if labels is None:
        if targets.kind == "multilabel":
            labels = np.arange(targets.n_labels)
        else:
            labels = np.unique(np.concatenate([targets.y_true, targets.y_pred]))
        labels_given = False
    else:
        labels = np.asarray(labels)
        labels_given = True
    micro_is_accuracy = targets.kind == "1d" and (
        not labels_given
        or set(labels.tolist()) >= set(np.unique(np.concatenate([targets.y_true, targets.y_pred])).tolist())
    )
    if target_names is not None and len(labels) != len(target_names):
        if labels_given:
            warnings.warn(
                "labels size, {0}, does not match size of target_names, {1}".format(
                    len(labels), len(target_names)
                )
            )
        else:
            raise ValueError(
                "Number of classes, {0}, does not match size of target_names, {1}. Try "
                "specifying the labels parameter".format(len(labels), len(target_names))
            )
    if target_names is None:
        target_names = ["%s" % lab for lab in labels]
    headers = ["precision", "recall", "f1-score", "support"]
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, sample_weight=sample_weight, zero_division=zero_division
    )
    rows = list(zip(target_names, p, r, f1, s))
    if targets.kind == "multilabel":
        average_options = ("micro", "macro", "weighted", "samples")
    else:
        average_options = ("micro", "macro", "weighted")
    if output_dict:
        report_dict = {row[0]: dict(zip(headers, [float(i) for i in row[1:]])) for row in rows}
    else:
        longest_last_line_heading = "weighted avg"
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"
        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += "\n"
    for average in average_options:
        if average.startswith("micro") and micro_is_accuracy:
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        avg = [avg_p, avg_r, avg_f1, np.sum(s)]
        if output_dict:
            report_dict[line_heading] = dict(zip(headers, [float(i) for i in avg]))
        elif line_heading == "accuracy":
            row_fmt_accuracy = (
                "{:>{width}s} " + " {:>9.{digits}}" * 2 + " {:>9.{digits}f}" + " {:>9}\n"
            )
            report += row_fmt_accuracy.format(line_heading, "", "", *avg[2:], width=width, digits=digits)
        else:
            report += row_fmt.format(line_heading, *avg, width=width, digits=digits)
    if output_dict:
        if "accuracy" in report_dict:
            report_dict["accuracy"] = report_dict["accuracy"]["precision"]
        return report_dict
    return report


# --------------------------------------------------------------------------- #
# label-based scores
# --------------------------------------------------------------------------- #


def accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    """Matches ``sklearn.metrics.accuracy_score``, now including multilabel
    indicator input (subset accuracy: every label of a sample must match)."""
    targets = _check_targets(y_true, y_pred)
    if targets.kind == "1d":
        return _accuracy_score_1d(targets.y_true, targets.y_pred, normalize=normalize, sample_weight=sample_weight)
    sw = _weights_or_none(sample_weight, targets.n_samples)
    return float(
        _ext().accuracy_score_indicator(
            _pa(targets.y_true), _pa(targets.y_pred), targets.n_labels, _sw(sw), bool(normalize)
        )
    )


def _require_1d(targets, fn):
    if targets.kind != "1d":
        raise ValueError("multilabel-indicator is not supported")


def balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False):
    """Matches ``sklearn.metrics.balanced_accuracy_score``."""
    targets = _check_targets(y_true, y_pred)
    _require_1d(targets, "balanced_accuracy_score")
    sw = _weights_or_none(sample_weight, targets.n_samples)
    score, dropped = _ext().balanced_accuracy_score(
        _pa_labels(targets.y_true), _pa_labels(targets.y_pred), _sw(sw), bool(adjusted)
    )
    if dropped:
        warnings.warn("y_pred contains classes not in y_true")
    return float(score)


def matthews_corrcoef(y_true, y_pred, *, sample_weight=None):
    """Matches ``sklearn.metrics.matthews_corrcoef`` (binary and multiclass)."""
    targets = _check_targets(y_true, y_pred)
    _require_1d(targets, "matthews_corrcoef")
    sw = _weights_or_none(sample_weight, targets.n_samples)
    return float(_ext().matthews_corrcoef(_pa_labels(targets.y_true), _pa_labels(targets.y_pred), _sw(sw)))


def cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None, replace_undefined_by=np.nan):
    """Matches ``sklearn.metrics.cohen_kappa_score``, including the two
    undefined cases that return ``replace_undefined_by`` with an
    ``UndefinedMetricWarning``."""
    if weights not in (None, "linear", "quadratic"):
        raise ValueError(
            "The 'weights' parameter of cohen_kappa_score must be a str among "
            "{'linear', 'quadratic'} or None. Got %r instead." % (weights,)
        )
    a = np.asarray(y1).ravel()
    b = np.asarray(y2).ravel()
    labels_arr = _param_labels(labels)
    if labels_arr is not None and not np.isin(labels_arr, a.astype(np.int64)).any():
        raise ValueError(
            "At least one label in `labels` must be present in `y1` (even though "
            "`cohen_kappa_score` is otherwise agnostic to the order of `y1` and `y2`)."
        )
    sw = _weights_or_none(sample_weight, len(a))
    value, status = _ext().cohen_kappa_score(_pa_labels(a), _pa_labels(b), labels_arr, weights, _sw(sw))
    if status == 1:
        warnings.warn(
            "`y2` contains no labels that are present in both `y1` and `labels`."
            "`cohen_kappa_score` is undefined and set to the value defined by the "
            "`replace_undefined_by` param, which is set to %s." % (replace_undefined_by,),
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return replace_undefined_by
    if status == 2:
        warnings.warn(
            "`y1`, `y2` and `labels` have only one label in common. `cohen_kappa_score` "
            "is undefined and set to the value defined by the the `replace_undefined_by` "
            "param, which is set to %s." % (replace_undefined_by,),
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return replace_undefined_by
    return float(value)


def hamming_loss(y_true, y_pred, *, sample_weight=None):
    """Matches ``sklearn.metrics.hamming_loss`` (1-D and multilabel)."""
    targets = _check_targets(y_true, y_pred)
    sw = _weights_or_none(sample_weight, targets.n_samples)
    ext = _ext()
    if targets.kind == "1d":
        return float(ext.hamming_loss(_pa_labels(targets.y_true), _pa_labels(targets.y_pred), 0, _sw(sw)))
    return float(ext.hamming_loss(_pa(targets.y_true), _pa(targets.y_pred), targets.n_labels, _sw(sw)))


def zero_one_loss(y_true, y_pred, *, normalize=True, sample_weight=None):
    """Matches ``sklearn.metrics.zero_one_loss`` (1-D and multilabel subset
    loss)."""
    targets = _check_targets(y_true, y_pred)
    sw = _weights_or_none(sample_weight, targets.n_samples)
    ext = _ext()
    if targets.kind == "1d":
        return float(
            ext.zero_one_loss(_pa_labels(targets.y_true), _pa_labels(targets.y_pred), 0, _sw(sw), bool(normalize))
        )
    return float(
        ext.zero_one_loss(_pa(targets.y_true), _pa(targets.y_pred), targets.n_labels, _sw(sw), bool(normalize))
    )


def hinge_loss(y_true, pred_decision, *, labels=None, sample_weight=None):
    """Matches ``sklearn.metrics.hinge_loss`` (binary, and Crammer–Singer
    multiclass)."""
    y_true = np.asarray(y_true).ravel()
    pred = np.asarray(pred_decision, dtype=np.float64)
    sw = _weights_or_none(sample_weight, len(y_true))
    if pred.shape[0] != len(y_true):
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % ([len(y_true), pred.shape[0]],)
        )
    y_true_unique = np.unique(labels if labels is not None else y_true)
    ext = _ext()
    if y_true_unique.size > 2:
        if pred.ndim <= 1:
            raise ValueError(
                "The shape of pred_decision cannot be 1d arraywith a multiclass target. "
                "pred_decision shape must be (n_samples, n_classes), that is (%d, %d). Got: %r"
                % (y_true.shape[0], y_true_unique.size, pred.shape)
            )
        if y_true_unique.size != pred.shape[1]:
            if labels is None:
                raise ValueError(
                    "Please include all labels in y_true or pass labels as third argument"
                )
            raise ValueError(
                "The shape of pred_decision is not consistent with the number of classes. "
                "With a multiclass target, pred_decision shape must be (n_samples, n_classes), "
                "that is (%d, %d). Got: %r" % (y_true.shape[0], y_true_unique.size, pred.shape)
            )
        classes = _param_labels(y_true_unique)
        return float(
            ext.hinge_loss_multiclass(
                _pa_labels(y_true), _pa(np.ascontiguousarray(pred).ravel()), classes, _sw(sw)
            )
        )
    if pred.ndim == 2 and pred.shape[1] != 1:
        raise ValueError(
            "y should be a 1d array, got an array of shape %r instead." % (pred.shape,)
        )
    # LabelBinarizer(neg_label=-1): the larger class is +1; a single class
    # binarizes to all -1.
    classes = np.unique(y_true)
    if len(classes) == 2:
        y_signed = np.where(y_true == classes[1], 1.0, -1.0)
    else:
        y_signed = np.full(len(y_true), -1.0)
    return float(ext.hinge_loss_binary(_pa(y_signed), _pa(pred.ravel()), _sw(sw)))


# --------------------------------------------------------------------------- #
# probabilistic scores
# --------------------------------------------------------------------------- #


def _resolve_binary_pos_label(pos_label, y_true):
    """The ``pos_label`` rule of sklearn's ``_validate_binary_probabilistic_prediction``:
    an explicit label wins; ``None`` is ``1`` for ``{0,1}``/``{-1,1}``-valued
    targets and otherwise the LARGEST class."""
    if pos_label is not None:
        return pos_label
    classes = np.unique(y_true)
    consistent = (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    )
    return 1 if consistent else classes[-1]


def _validate_binary_proba(y_true, y_proba, pos_label):
    y_true = np.asarray(y_true).ravel()
    p = np.asarray(y_proba, dtype=np.float64).ravel()
    if not np.all(np.isfinite(p)):
        raise ValueError("Input contains NaN, infinity or a value too large for dtype('float64').")
    if len(p) != len(y_true):
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r" % ([len(p), len(y_true)],)
        )
    if len(np.unique(y_true)) > 2:
        raise ValueError(
            "The type of the target inferred from y_true is multiclass but should be binary "
            "according to the shape of y_prob."
        )
    if p.size and p.max() > 1:
        raise ValueError("y_prob contains values greater than 1: %r" % (p.max(),))
    if p.size and p.min() < 0:
        raise ValueError("y_prob contains values less than 0: %r" % (p.min(),))
    pos = _resolve_binary_pos_label(pos_label, y_true)
    y01 = (y_true == pos).astype(np.float64)
    prob = np.column_stack([1.0 - p, p])
    return y01, prob, [0, 1]


def _validate_multiclass_proba(y_true, y_proba, labels):
    y_true = np.asarray(y_true).ravel()
    p = np.asarray(y_proba, dtype=np.float64)
    if p.size and p.max() > 1:
        raise ValueError("y_prob contains values greater than 1: %r" % (p.max(),))
    if p.size and p.min() < 0:
        raise ValueError("y_prob contains values lower than 0: %r" % (p.min(),))
    if p.shape[0] != len(y_true):
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r" % ([p.shape[0], len(y_true)],)
        )
    if labels is not None:
        lab = np.asarray(labels).ravel()
        classes = np.unique(lab)
        if not np.array_equal(classes, lab):
            warnings.warn(
                "Labels passed were %s. But this function assumes labels are ordered "
                "lexicographically. Pass the ordered labels=%s and ensure that the columns of "
                "y_prob correspond to this ordering." % (lab, classes.tolist()),
                UserWarning,
            )
        if not np.all(np.isin(y_true, lab)):
            raise ValueError(
                "y_true contains values %s not belonging to the passed labels %s."
                % (set(y_true.tolist()) - set(lab.tolist()), lab)
            )
    else:
        classes = np.unique(y_true)
    if len(classes) == 1:
        if labels is None:
            raise ValueError(
                "y_true contains only one label (%r). Please provide the list of all expected "
                "class labels explicitly through the labels argument." % (classes[0],)
            )
        raise ValueError("The labels array needs to contain at least two labels, got %s." % (classes,))
    if p.ndim == 1:
        p = p[:, None]
    if p.shape[1] == 1:
        p = np.concatenate([1 - p, p], axis=1)
    eps = np.finfo(np.asarray(y_proba).dtype if np.asarray(y_proba).dtype.kind == "f" else np.float64).eps
    if not np.all(np.isclose(p.sum(axis=1), 1, rtol=np.sqrt(eps))):
        warnings.warn("The y_prob values do not sum to one. Make sure to pass probabilities.", UserWarning)
    if len(classes) != p.shape[1]:
        if labels is None:
            raise ValueError(
                "y_true and y_prob contain different number of classes: %d vs %d. Please provide "
                "the true labels explicitly through the labels argument. Classes found in y_true: %s"
                % (len(classes), p.shape[1], classes)
            )
        raise ValueError(
            "The number of classes in labels is different from that in y_prob. Classes found in "
            "labels: %s" % (classes,)
        )
    return y_true, p, _param_labels(classes)


def _proba_dtype_eps(y_proba):
    dt = np.asarray(y_proba).dtype
    return float(np.finfo(dt if dt.kind == "f" else np.float64).eps)


def brier_score_loss(y_true, y_proba, *, sample_weight=None, pos_label=None, labels=None, scale_by_half="auto"):
    """Matches ``sklearn.metrics.brier_score_loss`` (binary 1-D probabilities
    and the multiclass ``(n_samples, n_classes)`` form)."""
    arr = np.asarray(y_proba)
    sw = _weights_or_none(sample_weight, arr.shape[0])
    if arr.ndim == 1 or arr.shape[1] == 1:
        y, prob, classes = _validate_binary_proba(y_true, arr, pos_label)
    else:
        y, prob, classes = _validate_multiclass_proba(y_true, arr, labels)
    if scale_by_half == "auto":
        scale_by_half = arr.ndim == 1 or arr.shape[1] < 3
    return float(
        _ext().probabilistic_score(
            0, _pa_labels(y), _pa(np.ascontiguousarray(prob).ravel()), classes, _sw(sw), bool(scale_by_half)
        )
    )


def _too_few(n):
    if n < 2:
        warnings.warn("D^2 score is not well-defined with less than two samples.", UndefinedMetricWarning)
        return True
    return False


def d2_brier_score(y_true, y_proba, *, sample_weight=None, pos_label=None, labels=None):
    """Matches ``sklearn.metrics.d2_brier_score``."""
    arr = np.asarray(y_proba)
    if _too_few(arr.shape[0]):
        return float("nan")
    sw = _weights_or_none(sample_weight, arr.shape[0])
    if arr.ndim == 1 or arr.shape[1] == 1:
        y, prob, classes = _validate_binary_proba(y_true, arr, pos_label)
    else:
        y, prob, classes = _validate_multiclass_proba(y_true, arr, labels)
    return float(
        _ext().probabilistic_score(1, _pa_labels(y), _pa(np.ascontiguousarray(prob).ravel()), classes, _sw(sw))
    )


def d2_log_loss_score(y_true, y_proba=None, *, sample_weight=None, labels=None, y_pred="deprecated"):
    """Matches ``sklearn.metrics.d2_log_loss_score`` (including the deprecated
    ``y_pred`` alias)."""
    if not (isinstance(y_pred, str) and y_pred == "deprecated"):
        if y_proba is not None:
            raise ValueError(
                "Cannot use both `y_pred` and `y_proba`. `y_pred` is deprecated, use `y_proba` instead."
            )
        warnings.warn(
            "`y_pred` was renamed to `y_proba` in version 1.9 and will be removed in 1.11. "
            "Use `y_proba` instead.",
            FutureWarning,
        )
        y_proba = y_pred
    arr = np.asarray(y_proba)
    if _too_few(arr.shape[0]):
        return float("nan")
    sw = _weights_or_none(sample_weight, arr.shape[0])
    y, prob, classes = _validate_multiclass_proba(y_true, arr, labels)
    return float(
        _ext().probabilistic_score(
            2,
            _pa_labels(y),
            _pa(np.ascontiguousarray(prob).ravel()),
            classes,
            _sw(sw),
            False,
            _proba_dtype_eps(y_proba),
        )
    )


def class_likelihood_ratios(y_true, y_pred, *, labels=None, sample_weight=None, replace_undefined_by=np.nan):
    """Matches ``sklearn.metrics.class_likelihood_ratios``: ``(LR+, LR-)``,
    with ``replace_undefined_by`` (``np.nan``, ``1.0`` or an ``{'LR+', 'LR-'}``
    dict) substituted for the undefined ratios."""
    targets = _check_targets(y_true, y_pred)
    if targets.kind != "1d" or len(np.unique(np.concatenate([targets.y_true, targets.y_pred]))) > 2:
        kind = "multilabel-indicator" if targets.kind != "1d" else "multiclass"
        raise ValueError(
            "class_likelihood_ratios only supports binary classification problems, got "
            "targets of type: %s" % kind
        )
    desired_lr_pos = desired_lr_neg = None
    if isinstance(replace_undefined_by, Real) and replace_undefined_by == 1.0:
        replace_undefined_by = {"LR+": 1.0, "LR-": 1.0}
    if isinstance(replace_undefined_by, dict):
        msg = (
            "The dictionary passed as `replace_undefined_by` needs to be in the form "
            "`{'LR+': `value_1`, 'LR-': `value_2`}` where the value for `LR+` ranges from "
            "`1.0` to `np.inf` or is `np.nan` and the value for `LR-` ranges from `0.0` to "
            "`1.0` or is `np.nan`; got `%s`." % (replace_undefined_by,)
        )
        if "LR+" not in replace_undefined_by or "LR-" not in replace_undefined_by:
            raise ValueError(msg)
        desired_lr_pos = replace_undefined_by["LR+"]
        desired_lr_neg = replace_undefined_by["LR-"]
        ok_pos = isinstance(desired_lr_pos, Real) and (np.isnan(desired_lr_pos) or desired_lr_pos >= 1.0)
        ok_neg = isinstance(desired_lr_neg, Real) and (
            np.isnan(desired_lr_neg) or 0.0 <= desired_lr_neg <= 1.0
        )
        if not (ok_pos and ok_neg):
            raise ValueError(msg)
    sw = _weights_or_none(sample_weight, targets.n_samples)
    lr_pos, lr_neg, no_positive, pos_undef, no_pred_pos, neg_undef = _ext().class_likelihood_ratios(
        _pa_labels(targets.y_true), _pa_labels(targets.y_pred), _param_labels(labels), _sw(sw)
    )
    is_nan_policy = isinstance(replace_undefined_by, float) and np.isnan(replace_undefined_by)
    if no_positive:
        warnings.warn(
            "No samples of the positive class are present in `y_true`. "
            "`positive_likelihood_ratio` and `negative_likelihood_ratio` are both set to "
            "`np.nan`. Use the `replace_undefined_by` param to control this behavior. To "
            "suppress this warning or turn it into an error, see Python's `warnings` module "
            "and `warnings.catch_warnings()`.",
            UndefinedMetricWarning,
            stacklevel=2,
        )
    if pos_undef:
        start = (
            "No samples were predicted for the positive class and `positive_likelihood_ratio` is "
            if no_pred_pos
            else "`positive_likelihood_ratio` is ill-defined and "
        )
        warnings.warn(
            start + "set to `np.nan`. Use the `replace_undefined_by` param to ",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        lr_pos = replace_undefined_by if is_nan_policy else desired_lr_pos
    if neg_undef:
        warnings.warn(
            "`negative_likelihood_ratio` is ill-defined and set to `np.nan`. Use the "
            "`replace_undefined_by` param to control this behavior. To suppress this warning "
            "or turn it into an error, see Python's `warnings` module and "
            "`warnings.catch_warnings()`.",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        lr_neg = replace_undefined_by if is_nan_policy else desired_lr_neg
    return float(lr_pos), float(lr_neg)


def top_k_accuracy_score(y_true, y_score, *, k=2, normalize=True, sample_weight=None, labels=None):
    """Matches ``sklearn.metrics.top_k_accuracy_score``, including its
    stable-sort tie rule (among equal scores the higher class index ranks
    first)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=np.float64)
    present = np.unique(y_true)
    y_type = "binary" if len(present) <= 2 else "multiclass"
    if y_type == "binary" and labels is not None and len(labels) > 2:
        y_type = "multiclass"
    if y_type == "binary":
        if y_score.ndim == 2 and y_score.shape[1] != 1:
            raise ValueError(
                "`y_true` is binary while y_score is 2d with %d classes. If `y_true` does not "
                "contain all the labels, `labels` must be provided." % y_score.shape[1]
            )
        y_score = y_score.ravel()
    elif y_score.ndim != 2:
        raise ValueError(
            "`y_score` needs to be of shape `(n_samples, n_classes)`, since `y_true` contains "
            "multiple classes. Got `y_score.shape=%r`." % (y_score.shape,)
        )
    if y_score.shape[0] != len(y_true):
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % ([len(y_true), y_score.shape[0]],)
        )
    sw = _weights_or_none(sample_weight, len(y_true))
    y_score_n_classes = y_score.shape[1] if y_score.ndim == 2 else 2
    if labels is None:
        classes = present
        n_classes = len(classes)
        if n_classes != y_score_n_classes:
            raise ValueError(
                "Number of classes in 'y_true' (%d) not equal to the number of classes in "
                "'y_score' (%d).You can provide a list of all known classes by assigning it to "
                "the `labels` parameter." % (n_classes, y_score_n_classes)
            )
    else:
        labels = np.asarray(labels).ravel()
        classes = np.unique(labels)
        n_classes = len(classes)
        if n_classes != len(labels):
            raise ValueError("Parameter 'labels' must be unique.")
        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered.")
        if n_classes != y_score_n_classes:
            raise ValueError(
                "Number of given labels (%d) not equal to the number of classes in 'y_score' (%d)."
                % (n_classes, y_score_n_classes)
            )
        if len(np.setdiff1d(y_true, classes)):
            raise ValueError("'y_true' contains labels not in parameter 'labels'.")
    if k >= n_classes:
        warnings.warn(
            "'k' (%d) greater than or equal to 'n_classes' (%d) will result in a perfect score "
            "and is therefore meaningless." % (k, n_classes),
            UndefinedMetricWarning,
        )
    ext = _ext()
    if y_type == "binary":
        y_enc = np.searchsorted(classes, y_true).astype(np.float64)
        return float(ext.top_k_accuracy_score(_pa(y_enc), _pa(y_score), int(k), None, bool(normalize), _sw(sw)))
    return float(
        ext.top_k_accuracy_score(
            _pa_labels(y_true),
            _pa(np.ascontiguousarray(y_score).ravel()),
            int(k),
            _param_labels(classes),
            bool(normalize),
            _sw(sw),
        )
    )
