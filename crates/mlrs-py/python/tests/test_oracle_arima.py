"""ARIMA oracle harness (TSA-01, Phase 22): the full binding-path replay of
the committed ARIMA fixture (see the Rust ``arima_test.rs`` / the Rust
``timeseries::arima`` module docs for the gate-tier rationale and the
documented scope: zero-mean only, no seasonal component).
"""

import numpy as np
import pytest

import mlrs
from conftest import fixture_path, requires_f64


@requires_f64
def test_arima_fit_band():
    d = np.load(fixture_path("arima_seed42.npz"))
    y = d["y"]
    sm_loglik = float(d["sm_loglik"][0])
    sm_forecast = d["sm_forecast"]

    est = mlrs.ARIMA(order=(2, 0, 1)).fit(y)
    assert est.llf >= sm_loglik - 1.0, (
        f"mlrs MLE loglik {est.llf} should be within 1.0 of statsmodels' {sm_loglik}"
    )
    fc = est.forecast(5)
    assert fc.shape == (5,)
    assert np.allclose(fc, sm_forecast, atol=1.5)


def test_arima_forecast_shape_and_finiteness():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(size=80))  # a random walk — d=1 is appropriate
    est = mlrs.ARIMA(order=(1, 1, 0)).fit(y)
    fc = est.forecast(10)
    assert fc.shape == (10,)
    assert np.all(np.isfinite(fc))
    assert est.ar_.shape == (1,)
    assert est.ma_.shape == (0,)


def test_arima_rejects_before_fit():
    est = mlrs.ARIMA(order=(1, 0, 0))
    with pytest.raises(ValueError):
        est.forecast(3)
    with pytest.raises(ValueError):
        _ = est.ar_


def test_arima_rejects_over_bound_order():
    with pytest.raises(ValueError):
        mlrs.ARIMA(order=(50, 0, 0)).fit(np.random.default_rng(0).normal(size=100))


def test_auto_arima_selects_a_competitive_order():
    d = np.load(fixture_path("arima_seed42.npz"))
    y = d["y"]
    auto = mlrs.AutoARIMA(d=0, max_p=3, max_q=3).fit(y)
    true_order_fit = mlrs.ARIMA(order=(2, 0, 1)).fit(y)
    assert auto.aicc <= true_order_fit.aicc + 1.0
    fc = auto.forecast(3)
    assert fc.shape == (3,)
