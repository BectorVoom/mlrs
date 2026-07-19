"""Time-series estimator shims (TSA-01, Phase 22) delegating to ``_mlrs``.

``ARIMA``/``AutoARIMA`` are NOT ``MlrsBase``/sklearn-mixin estimators — they
take a single 1-D series (``endog``), not an ``(n_samples, n_features)`` `X`
+ optional `y`, mirroring cuML's own ``tsa.arima.ARIMA`` shape. See the Rust
``mlrs_algos::timeseries::arima`` module docs for the full scope statement:
zero-mean only (no trend/constant), no seasonal (SARIMAX) component.
"""

import numpy as np

from . import _io


def _normalize_series(y):
    y = np.ascontiguousarray(np.asarray(y))
    if y.ndim != 1:
        raise ValueError("ARIMA: y must be a 1-D series")
    if y.dtype not in (np.float32, np.float64):
        y = y.astype(np.float64)
    xa, rows, cols = _io.normalize_X(y.reshape(-1, 1))
    return xa, rows


class ARIMA:
    """``ARIMA(order=(p, d, q))`` — zero-mean, no seasonal component (see
    module docs). Not fit/predict sklearn-shaped: ``fit(y)`` takes the
    series directly; ``forecast(n_periods)`` returns point forecasts.
    """

    def __init__(self, order=(0, 0, 0)):
        self.order = order
        self._mlrs_obj = None

    def fit(self, y):
        from . import _load_ext

        xa, n_obs = _normalize_series(y)
        obj = _load_ext().ARIMA(tuple(int(v) for v in self.order))
        obj.fit(xa, n_obs)
        self._mlrs_obj = obj
        return self

    def _check_fitted(self):
        if self._mlrs_obj is None or not self._mlrs_obj.is_fitted():
            raise ValueError("ARIMA: this instance is not fitted yet — call fit() first")

    def forecast(self, n_periods):
        self._check_fitted()
        return np.asarray(self._mlrs_obj.forecast(int(n_periods)))

    @property
    def ar_(self):
        self._check_fitted()
        return np.asarray(self._mlrs_obj.ar())

    @property
    def ma_(self):
        self._check_fitted()
        return np.asarray(self._mlrs_obj.ma())

    @property
    def sigma2_(self):
        self._check_fitted()
        return self._mlrs_obj.sigma2()

    @property
    def llf(self):
        self._check_fitted()
        return self._mlrs_obj.loglik()

    @property
    def aic(self):
        self._check_fitted()
        return self._mlrs_obj.aic()

    @property
    def aicc(self):
        self._check_fitted()
        return self._mlrs_obj.aicc()

    @property
    def bic(self):
        self._check_fitted()
        return self._mlrs_obj.bic()

    @property
    def converged(self):
        self._check_fitted()
        return self._mlrs_obj.converged()


class AutoARIMA:
    """Bounded ``(p, q)`` grid search over AICc at a FIXED ``d`` (mlrs does
    not auto-select ``d`` — a documented scope reduction from pmdarima's
    KPSS-driven ``d`` search; the exhaustive grid replaces pmdarima's
    stepwise Hyndman-Khandakar heuristic — see the Rust ``AutoArima`` docs).
    """

    def __init__(self, d=0, max_p=5, max_q=5):
        self.d = d
        self.max_p = max_p
        self.max_q = max_q
        self._mlrs_obj = None

    def fit(self, y):
        from . import _load_ext

        xa, n_obs = _normalize_series(y)
        obj = _load_ext().AutoARIMA()
        obj.fit(xa, n_obs, int(self.d), int(self.max_p), int(self.max_q))
        self._mlrs_obj = obj
        return self

    def _check_fitted(self):
        if self._mlrs_obj is None or not self._mlrs_obj.is_fitted():
            raise ValueError("AutoARIMA: this instance is not fitted yet — call fit() first")

    def forecast(self, n_periods):
        self._check_fitted()
        return np.asarray(self._mlrs_obj.forecast(int(n_periods)))

    @property
    def order_(self):
        self._check_fitted()
        return self._mlrs_obj.order()

    @property
    def ar_(self):
        self._check_fitted()
        return np.asarray(self._mlrs_obj.ar())

    @property
    def ma_(self):
        self._check_fitted()
        return np.asarray(self._mlrs_obj.ma())

    @property
    def aicc(self):
        self._check_fitted()
        return self._mlrs_obj.aicc()

    @property
    def llf(self):
        self._check_fitted()
        return self._mlrs_obj.loglik()
