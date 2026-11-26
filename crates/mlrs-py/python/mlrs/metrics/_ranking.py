"""``sklearn.metrics``' ranking and threshold metrics.

The curves (``roc_curve``, ``det_curve``, ``confusion_matrix_at_thresholds``)
come back from Rust as zero-copy pyarrow columns — they are O(distinct
scores) long. Every value is computed in ``mlrs_algos::metrics::ranking``;
``metric_at_thresholds`` is the one exception, since its whole point is to
call a USER metric once per threshold (the thresholds themselves still come
from the Rust curve).
"""

import warnings

import numpy as np

from ._base import UndefinedMetricWarning, _ext, _owned, _pa, _pa_labels, _resolve_pos_label, _sw

__all__ = [
    "auc",
    "average_precision_score",
    "confusion_matrix_at_thresholds",
    "coverage_error",
    "dcg_score",
    "det_curve",
    "label_ranking_average_precision_score",
    "label_ranking_loss",
    "metric_at_thresholds",
    "ndcg_score",
    "roc_curve",
]


def _np(a):
    return np.asarray(a, dtype=np.float64)


def _binary_inputs(y_true, y_score, sample_weight, pos_label):
    """sklearn's ``confusion_matrix_at_thresholds`` input contract: a binary
    target (or a multiclass one with an explicit ``pos_label``), finite scores,
    consistent lengths, and ``pos_label=None`` resolution."""
    y_true = np.asarray(y_true).ravel()
    y_score = _np(y_score).ravel()
    if len(y_true) != len(y_score):
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r" % ([len(y_true), len(y_score)],)
        )
    if not np.all(np.isfinite(y_score)):
        raise ValueError("Input contains NaN.")
    classes = np.unique(y_true)
    if len(classes) > 2 and pos_label is None:
        raise ValueError("multiclass format is not supported")
    pos = _resolve_pos_label(pos_label, y_true.astype(np.int64))
    sw = None
    if sample_weight is not None:
        sw = _np(sample_weight).ravel()
        if len(sw) != len(y_true):
            raise ValueError(
                "Found input variables with inconsistent numbers of samples: %r" % ([len(y_true), len(sw)],)
            )
    return y_true, y_score, sw, pos


def confusion_matrix_at_thresholds(y_true, y_score, *, pos_label=None, sample_weight=None):
    """Matches ``sklearn.metrics.confusion_matrix_at_thresholds``: ``(tns, fps,
    fns, tps, thresholds)``, one entry per distinct score, thresholds
    decreasing."""
    y_true, y_score, sw, pos = _binary_inputs(y_true, y_score, sample_weight, pos_label)
    cols = _ext().confusion_matrix_at_thresholds(_pa_labels(y_true), _pa(y_score), pos, _sw(sw))
    return tuple(_owned(c) for c in cols)


def roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True):
    """Matches ``sklearn.metrics.roc_curve``: ``(fpr, tpr, thresholds)`` with the
    leading ``(0, 0, inf)`` point."""
    y_true, y_score, sw, pos = _binary_inputs(y_true, y_score, sample_weight, pos_label)
    (fpr, tpr, thr), no_neg, no_pos = _ext().roc_curve(
        _pa_labels(y_true), _pa(y_score), pos, _sw(sw), bool(drop_intermediate)
    )
    if no_neg:
        warnings.warn(
            "No negative samples in y_true, false positive value should be meaningless",
            UndefinedMetricWarning,
        )
    if no_pos:
        warnings.warn(
            "No positive samples in y_true, true positive value should be meaningless",
            UndefinedMetricWarning,
        )
    return (_owned(fpr), _owned(tpr), _owned(thr))


def det_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=False):
    """Matches ``sklearn.metrics.det_curve``: ``(fpr, fnr, thresholds)``,
    thresholds increasing."""
    y_true, y_score, sw, pos = _binary_inputs(y_true, y_score, sample_weight, pos_label)
    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Only one class is present in y_true. Detection error tradeoff curve is not "
            "defined in that case."
        )
    fpr, fnr, thr = _ext().det_curve(_pa_labels(y_true), _pa(y_score), pos, _sw(sw), bool(drop_intermediate))
    return (_owned(fpr), _owned(fnr), _owned(thr))


def auc(x, y):
    """Matches ``sklearn.metrics.auc`` (trapezoidal rule, ``x`` monotonic in
    either direction)."""
    x = _np(x).ravel()
    y = _np(y).ravel()
    if len(x) != len(y):
        raise ValueError("Found input variables with inconsistent numbers of samples: %r" % ([len(x), len(y)],))
    if x.shape[0] < 2:
        raise ValueError(
            "At least 2 points are needed to compute area under curve, but x.shape = %s" % x.shape[0]
        )
    dx = np.diff(x)
    if np.any(dx < 0) and not np.all(dx <= 0):
        raise ValueError("x is neither increasing nor decreasing : {}.".format(x))
    return float(_ext().auc(_pa(x), _pa(y)))


_AVG = (None, "micro", "macro", "weighted", "samples")


def average_precision_score(y_true, y_score, *, average="macro", pos_label=1, sample_weight=None):
    """Matches ``sklearn.metrics.average_precision_score`` for binary, multiclass
    (one-vs-rest over ``np.unique(y_true)``) and multilabel-indicator targets."""
    if average not in _AVG:
        raise ValueError("average has to be one of {0}".format(_AVG))
    yt = np.asarray(y_true)
    ys = _np(y_score)
    sw = None if sample_weight is None else _np(sample_weight).ravel()
    ext = _ext()
    is_multilabel = yt.ndim == 2 and yt.shape[1] > 1
    if not is_multilabel:
        yt = yt.ravel()
        present = np.unique(yt)
        if len(present) <= 2:
            if len(present) == 2 and pos_label not in present:
                raise ValueError(
                    "pos_label=%r is not a valid label. It should be one of %s" % (pos_label, present)
                )
            got = ext.average_precision_score(_pa_labels(yt), _pa(ys.ravel()), 0, int(pos_label), "macro", _sw(sw))
            return float(got[0])
        if pos_label != 1:
            raise ValueError(
                "Parameter pos_label is fixed to 1 for multiclass y_true. Do not set pos_label or "
                "set pos_label to 1."
            )
        indicator = (yt[:, None] == present[None, :]).astype(np.float64)
        if ys.shape != indicator.shape:
            raise ValueError(
                "`y_score` needs to be of shape `(n_samples, n_classes)`, since `y_true` contains "
                "multiple classes. Got `y_score.shape=%r`." % (ys.shape,)
            )
    else:
        if pos_label != 1:
            raise ValueError(
                "Parameter pos_label is fixed to 1 for multilabel-indicator y_true. Do not set "
                "pos_label or set pos_label to 1."
            )
        if ys.shape != yt.shape:
            raise ValueError("y_true and y_score have different shape")
        indicator = yt.astype(np.float64)
    n_labels = indicator.shape[1]
    got = ext.average_precision_score(
        _pa(np.ascontiguousarray(indicator).ravel()),
        _pa(np.ascontiguousarray(ys).ravel()),
        n_labels,
        1,
        "none" if average is None else average,
        _sw(sw),
    )
    if average is None:
        return np.asarray(got, dtype=np.float64)
    return float(got[0])


def metric_at_thresholds(y_true, y_score, metric_func, *, sample_weight=None, metric_params=None):
    """Matches ``sklearn.metrics.metric_at_thresholds``: ``metric_func(y_true,
    y_pred)`` at every distinct score threshold (``y_pred = y_score >=
    threshold``), in DECREASING threshold order, plus the thresholds.

    Zero-weight samples are dropped first and the remaining inputs sorted by
    decreasing score, exactly as sklearn passes them to ``metric_func``.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = _np(y_score).ravel()
    if len(y_true) != len(y_score):
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r" % ([len(y_true), len(y_score)],)
        )
    if not np.all(np.isfinite(y_score)):
        raise ValueError("Input contains NaN.")
    sw = None
    if sample_weight is not None:
        sw = _np(sample_weight).ravel()
        keep = sw != 0
        y_true, y_score, sw = y_true[keep], y_score[keep], sw[keep]
    # A stable sort on the NEGATED score: descending, ties in input order
    # (sklearn's `argsort(stable=True, descending=True)`).
    order = np.argsort(-y_score, kind="stable")
    y_true, y_score = y_true[order], y_score[order]
    if sw is not None:
        sw = sw[order]
    # The distinct thresholds are exactly the Rust curve's (one per run, at the
    # run's last index); reuse it rather than re-deriving them here.
    if len(y_score):
        thresholds = np.asarray(
            _ext().confusion_matrix_at_thresholds(_pa(np.zeros(len(y_score))), _pa(y_score), 1, None)[4],
            dtype=np.float64,
        )
    else:
        thresholds = np.empty(0)
    params = dict(metric_params or {})
    if sw is not None:
        params["sample_weight"] = sw
    values = [metric_func(y_true, (y_score >= t).astype(np.int32), **params) for t in thresholds]
    return np.asarray(values), thresholds


def _multilabel_inputs(y_true, y_score, sample_weight, fmt_ok):
    yt = np.asarray(y_true)
    ys = _np(y_score)
    if yt.ndim != 2 or not fmt_ok(yt):
        raise ValueError("{0} format is not supported".format("multiclass" if yt.ndim == 1 else "multiclass-multioutput"))
    if yt.shape != ys.shape:
        raise ValueError("y_true and y_score have different shape")
    sw = None if sample_weight is None else _np(sample_weight).ravel()
    if sw is not None and len(sw) != yt.shape[0]:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r" % ([yt.shape[0], len(sw)],)
        )
    return (
        np.ascontiguousarray(yt, dtype=np.float64).ravel(),
        np.ascontiguousarray(ys).ravel(),
        yt.shape[1],
        sw,
    )


def _is_indicator(a):
    return a.size == 0 or (a.min() >= 0 and a.max() <= 1 and np.all(a == np.round(a)))


def coverage_error(y_true, y_score, *, sample_weight=None):
    """Matches ``sklearn.metrics.coverage_error``."""
    yt, ys, n_labels, sw = _multilabel_inputs(y_true, y_score, sample_weight, _is_indicator)
    return float(_ext().label_ranking_metric("coverage_error", _pa(yt), _pa(ys), n_labels, _sw(sw)))


def label_ranking_average_precision_score(y_true, y_score, *, sample_weight=None):
    """Matches ``sklearn.metrics.label_ranking_average_precision_score``."""
    yt, ys, n_labels, sw = _multilabel_inputs(y_true, y_score, sample_weight, _is_indicator)
    return float(_ext().label_ranking_metric("lrap", _pa(yt), _pa(ys), n_labels, _sw(sw)))


def label_ranking_loss(y_true, y_score, *, sample_weight=None):
    """Matches ``sklearn.metrics.label_ranking_loss``."""
    yt, ys, n_labels, sw = _multilabel_inputs(y_true, y_score, sample_weight, _is_indicator)
    return float(_ext().label_ranking_metric("label_ranking_loss", _pa(yt), _pa(ys), n_labels, _sw(sw)))


def _dcg_inputs(y_true, y_score, sample_weight):
    yt = _np(y_true)
    ys = _np(y_score)
    if yt.ndim != 2:
        raise ValueError(
            "Only ('multilabel-indicator', 'continuous-multioutput', 'multiclass-multioutput') "
            "formats are supported. Got %s instead" % ("binary" if yt.ndim == 1 else "unknown")
        )
    if yt.shape != ys.shape:
        raise ValueError("Found input variables with inconsistent numbers of samples.")
    sw = None if sample_weight is None else _np(sample_weight).ravel()
    return np.ascontiguousarray(yt).ravel(), np.ascontiguousarray(ys).ravel(), yt.shape[1], sw


def _check_k(k):
    if k is not None and (not isinstance(k, (int, np.integer)) or k < 1):
        raise ValueError("The 'k' parameter must be an int in the range [1, inf) or None. Got %r instead." % (k,))


def dcg_score(y_true, y_score, *, k=None, log_base=2, sample_weight=None, ignore_ties=False):
    """Matches ``sklearn.metrics.dcg_score`` (tie-averaged unless
    ``ignore_ties=True``)."""
    _check_k(k)
    yt, ys, n_labels, sw = _dcg_inputs(y_true, y_score, sample_weight)
    return float(
        _ext().dcg(_pa(yt), _pa(ys), n_labels, None if k is None else int(k), float(log_base), _sw(sw), bool(ignore_ties), False)
    )


def ndcg_score(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False):
    """Matches ``sklearn.metrics.ndcg_score``."""
    _check_k(k)
    yt_arr = _np(y_true)
    if yt_arr.size and yt_arr.min() < 0:
        raise ValueError("ndcg_score should not be used on negative y_true values.")
    if yt_arr.ndim > 1 and yt_arr.shape[1] <= 1:
        raise ValueError(
            "Computing NDCG is only meaningful when there is more than 1 document. Got %d instead."
            % yt_arr.shape[1]
        )
    yt, ys, n_labels, sw = _dcg_inputs(y_true, y_score, sample_weight)
    return float(
        _ext().dcg(_pa(yt), _pa(ys), n_labels, None if k is None else int(k), 2.0, _sw(sw), bool(ignore_ties), True)
    )
