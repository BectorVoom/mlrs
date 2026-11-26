"""The rest of ``sklearn.metrics``' regression surface.

Same contract as the three original regression metrics in ``_base``: 1-D or
2-D targets are flattened ROW-MAJOR, every value (including the
``multioutput`` reduction) is computed in Rust
(``mlrs_algos::metrics::regression``), and this module owns only sklearn's
parameter/domain messages, its warnings and the scalar-vs-array return.
"""

import warnings
from numbers import Real

import numpy as np

from ._base import UndefinedMetricWarning, _check_reg_targets, _ext, _pa, _reg_out, _sw

__all__ = [
    "d2_absolute_error_score",
    "d2_pinball_score",
    "d2_tweedie_score",
    "explained_variance_score",
    "max_error",
    "mean_absolute_percentage_error",
    "mean_gamma_deviance",
    "mean_pinball_loss",
    "mean_poisson_deviance",
    "mean_squared_log_error",
    "mean_tweedie_deviance",
    "median_absolute_error",
    "root_mean_squared_error",
    "root_mean_squared_log_error",
]

_MO = ("raw_values", "uniform_average")


def _call(name, fn_name, y_true, y_pred, sample_weight, multioutput, allowed=_MO, alpha=0.5, force_finite=True):
    yt, yp, n_samples, n_outputs, mo, mo_w = _check_reg_targets(y_true, y_pred, multioutput, allowed, fn_name)
    got = _ext().regression_metric(
        name, _pa(yt), _pa(yp), n_outputs, _sw(sample_weight), mo, mo_w, float(alpha), bool(force_finite)
    )
    return _reg_out(got, mo)


def _check_alpha(alpha, fn_name):
    if not isinstance(alpha, Real) or not (0.0 <= alpha <= 1.0):
        raise ValueError(
            "The 'alpha' parameter of %s must be a float in the range [0.0, 1.0]. Got %r instead."
            % (fn_name, alpha)
        )


def _check_log1p(y_true, y_pred, what):
    if np.any(np.asarray(y_true) <= -1) or np.any(np.asarray(y_pred) <= -1):
        raise ValueError(
            "%s cannot be used when targets contain values less than or equal to -1." % what
        )


def explained_variance_score(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", force_finite=True
):
    """Matches ``sklearn.metrics.explained_variance_score``."""
    return _call(
        "explained_variance",
        "explained_variance_score",
        y_true,
        y_pred,
        sample_weight,
        multioutput,
        allowed=_MO + ("variance_weighted",),
        force_finite=force_finite,
    )


def max_error(y_true, y_pred):
    """Matches ``sklearn.metrics.max_error`` (single-output)."""
    yt, yp, _, n_outputs, _, _ = _check_reg_targets(y_true, y_pred, "uniform_average", _MO, "max_error")
    if n_outputs != 1:
        raise ValueError("Multioutput not supported in max_error")
    return float(_ext().max_error(_pa(yt), _pa(yp), 1))


def mean_absolute_percentage_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Matches ``sklearn.metrics.mean_absolute_percentage_error``."""
    return _call("mape", "mean_absolute_percentage_error", y_true, y_pred, sample_weight, multioutput)


def root_mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Matches ``sklearn.metrics.root_mean_squared_error`` (per-output root,
    then the ``multioutput`` reduction)."""
    return _call("rmse", "root_mean_squared_error", y_true, y_pred, sample_weight, multioutput)


def mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Matches ``sklearn.metrics.mean_squared_log_error``."""
    _check_log1p(y_true, y_pred, "Mean Squared Logarithmic Error")
    return _call("msle", "mean_squared_log_error", y_true, y_pred, sample_weight, multioutput)


def root_mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Matches ``sklearn.metrics.root_mean_squared_log_error``."""
    _check_log1p(y_true, y_pred, "Root Mean Squared Logarithmic Error")
    return _call("rmsle", "root_mean_squared_log_error", y_true, y_pred, sample_weight, multioutput)


def median_absolute_error(y_true, y_pred, *, multioutput="uniform_average", sample_weight=None):
    """Matches ``sklearn.metrics.median_absolute_error`` (numpy's median, or
    sklearn's averaged weighted percentile with ``sample_weight``)."""
    return _call("median_ae", "median_absolute_error", y_true, y_pred, sample_weight, multioutput)


def mean_pinball_loss(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average"):
    """Matches ``sklearn.metrics.mean_pinball_loss``."""
    _check_alpha(alpha, "mean_pinball_loss")
    return _call("pinball", "mean_pinball_loss", y_true, y_pred, sample_weight, multioutput, alpha=alpha)


def _n_samples(y):
    return np.asarray(y).shape[0]


def d2_pinball_score(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average"):
    """Matches ``sklearn.metrics.d2_pinball_score``."""
    _check_alpha(alpha, "d2_pinball_score")
    yt, yp, n_samples, n_outputs, mo, mo_w = _check_reg_targets(
        y_true, y_pred, multioutput, _MO, "d2_pinball_score"
    )
    if n_samples < 2:
        warnings.warn("D^2 score is not well-defined with less than two samples.", UndefinedMetricWarning)
        return float("nan")
    got = _ext().regression_metric(
        "d2_pinball", _pa(yt), _pa(yp), n_outputs, _sw(sample_weight), mo, mo_w, float(alpha), True
    )
    return _reg_out(got, mo)


def d2_absolute_error_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Matches ``sklearn.metrics.d2_absolute_error_score`` (``d2_pinball_score``
    at ``alpha=0.5``)."""
    return d2_pinball_score(y_true, y_pred, sample_weight=sample_weight, alpha=0.5, multioutput=multioutput)


def _tweedie_check(y_true, y_pred, power, fn_name):
    yt, yp, n_samples, n_outputs, _, _ = _check_reg_targets(y_true, y_pred, "uniform_average", _MO, fn_name)
    if n_outputs != 1:
        raise ValueError("Multioutput not supported in %s" % fn_name)
    message = "Mean Tweedie deviance error with power=%s can only be used on " % (power,)
    if power < 0:
        if np.any(yp <= 0):
            raise ValueError(message + "strictly positive y_pred.")
    elif power == 0:
        pass
    elif 1 <= power < 2:
        if np.any(yt < 0) or np.any(yp <= 0):
            raise ValueError(message + "non-negative y and strictly positive y_pred.")
    elif power >= 2:
        if np.any(yt <= 0) or np.any(yp <= 0):
            raise ValueError(message + "strictly positive y and y_pred.")
    else:
        raise ValueError(
            "The 'power' parameter of %s must be a float in the range (-inf, 0.0] or a float "
            "in the range [1.0, inf). Got %r instead." % (fn_name, power)
        )
    return yt, yp, n_samples


def mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0):
    """Matches ``sklearn.metrics.mean_tweedie_deviance``."""
    yt, yp, _ = _tweedie_check(y_true, y_pred, power, "mean_tweedie_deviance")
    return float(_ext().tweedie(_pa(yt), _pa(yp), 1, _sw(sample_weight), float(power), False))


def mean_poisson_deviance(y_true, y_pred, *, sample_weight=None):
    """Matches ``sklearn.metrics.mean_poisson_deviance`` (Tweedie, power 1)."""
    return mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=1)


def mean_gamma_deviance(y_true, y_pred, *, sample_weight=None):
    """Matches ``sklearn.metrics.mean_gamma_deviance`` (Tweedie, power 2)."""
    return mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=2)


def d2_tweedie_score(y_true, y_pred, *, sample_weight=None, power=0):
    """Matches ``sklearn.metrics.d2_tweedie_score``."""
    yt, yp, n_samples = _tweedie_check(y_true, y_pred, power, "d2_tweedie_score")
    if n_samples < 2:
        warnings.warn("D^2 score is not well-defined with less than two samples.", UndefinedMetricWarning)
        return float("nan")
    return float(_ext().tweedie(_pa(yt), _pa(yp), 1, _sw(sample_weight), float(power), True))
