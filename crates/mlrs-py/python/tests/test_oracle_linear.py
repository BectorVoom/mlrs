"""Linear-model oracle harness (PY-01: full numpy->pyarrow->FFI->device path).

Re-validates the 1e-5 contract for the five linear estimators
(LinearRegression / Ridge / Lasso / ElasticNet / LogisticRegression) through the
FULL Python binding path — ``numpy -> pyarrow -> __arrow_c_array__ -> Rust FFI
-> validate -> device -> host -> numpy`` — by replaying the committed
``tests/fixtures/*.npz`` sklearn-reference blobs (a SECOND consumer; no fixture
regeneration). The .npz key names are written by ``scripts/gen_oracle.py``.

Comparison rules:
  - LinearRegression / Ridge / Lasso / ElasticNet: direct ``coef_`` / ``intercept_``.
  - Ridge fixtures sweep three ``alpha`` values (one coef row per alpha); each
    alpha is a separate parametrize case fit with its own ``alpha``.
  - LogisticRegression: the gauge-fixed ``predict_proba`` is the PRIMARY gate
    (Phase-5 D-12), NOT raw ``coef_`` (which is only defined up to the softmax
    gauge). Predicted labels are also asserted to match exactly. The fixture was
    fit at a TIGHT tolerance (gen_oracle ``tol=1e-10``), so the shim is fit at a
    matching tight tolerance for f64; f32 cannot resolve a multinomial softmax to
    1e-5, so its proba tolerance is the f32-achievable ``1e-4`` while the exact
    label match stays the hard gate.

f64 fixtures are skipped-with-reason on an f64-incapable backend (rocm) via the
``conftest.requires_f64`` marker (mirrors ``capability.rs::skip_f64_with_log``).
"""

import numpy as np
import pytest

import mlrs
from conftest import dtype_of, fixture_path, proba_allclose, requires_f64


def _atol(fixture):
    """abs tolerance: strict 1e-5 for f64; f32 accumulates ~1e-6 epsilon, so the
    direct-coef cases use 1e-4 (still far below any algorithmic drift)."""
    return 1e-5 if dtype_of(fixture) == np.float64 else 1e-4


# --- direct coef_/intercept_ estimators ------------------------------------

DIRECT_CASES = [
    ("linear_regression_f32_seed42", lambda d: mlrs.LinearRegression(fit_intercept=True)),
    ("linear_regression_f64_seed42", lambda d: mlrs.LinearRegression(fit_intercept=True)),
    ("lasso_f32_seed42", lambda d: mlrs.Lasso(alpha=float(d["alpha"][0]), fit_intercept=True, max_iter=1000, tol=1e-4)),
    ("lasso_f64_seed42", lambda d: mlrs.Lasso(alpha=float(d["alpha"][0]), fit_intercept=True, max_iter=1000, tol=1e-4)),
    ("elastic_net_f32_seed42", lambda d: mlrs.ElasticNet(alpha=float(d["alpha"][0]), l1_ratio=float(d["l1_ratio"][0]), fit_intercept=True, max_iter=1000, tol=1e-4)),
    ("elastic_net_f64_seed42", lambda d: mlrs.ElasticNet(alpha=float(d["alpha"][0]), l1_ratio=float(d["l1_ratio"][0]), fit_intercept=True, max_iter=1000, tol=1e-4)),
]


@pytest.mark.parametrize("fixture,builder", DIRECT_CASES, ids=[c[0] for c in DIRECT_CASES])
@requires_f64
def test_linear_coef_oracle(fixture, builder):
    """PY-01: LinearRegression/Lasso/ElasticNet match sklearn coef_/intercept_."""
    d = np.load(fixture_path(fixture))
    est = builder(d).fit(d["X"], d["y"])
    atol = _atol(fixture)
    assert np.allclose(np.ravel(np.asarray(est.coef_)), np.ravel(d["coef"]), atol=atol, rtol=0.0)
    assert np.allclose(np.ravel(np.asarray(est.intercept_)), np.ravel(d["intercept"]), atol=atol, rtol=0.0)


# Ridge: each fixture stores coef rows for three alphas — one case per alpha.
RIDGE_CASES = [
    (name, i)
    for name in ("ridge_f32_seed42", "ridge_f64_seed42")
    for i in range(3)
]


@pytest.mark.parametrize("fixture,alpha_idx", RIDGE_CASES, ids=[f"{n}-a{i}" for n, i in RIDGE_CASES])
@requires_f64
def test_ridge_oracle(fixture, alpha_idx):
    """PY-01: Ridge matches sklearn coef_/intercept_ across the alpha sweep."""
    d = np.load(fixture_path(fixture))
    alpha = float(d["alpha"][alpha_idx])
    est = mlrs.Ridge(alpha=alpha, fit_intercept=True).fit(d["X"], d["y"])
    atol = _atol(fixture)
    assert np.allclose(np.ravel(np.asarray(est.coef_)), np.ravel(d["coef"][alpha_idx]), atol=atol, rtol=0.0)
    assert np.allclose(np.ravel(np.asarray(est.intercept_)), np.ravel(d["intercept"][alpha_idx]), atol=atol, rtol=0.0)


# LogisticRegression: gauge-fixed predict_proba is the primary gate (D-12).
# (fixture, fit_tol, proba_atol). f64 -> tight tol + 1e-5 proba; f32 -> the
# f32-achievable tol + 1e-4 proba (the exact label match is the hard gate).
LOGISTIC_CASES = [
    ("logistic_binary_f32_seed42", 1e-6, 1e-4),
    ("logistic_binary_f64_seed42", 1e-10, 1e-5),
    ("logistic_multi_f32_seed42", 1e-4, 1e-4),
    ("logistic_multi_f64_seed42", 1e-10, 1e-5),
]


@pytest.mark.parametrize("fixture,fit_tol,proba_atol", LOGISTIC_CASES, ids=[c[0] for c in LOGISTIC_CASES])
@requires_f64
def test_logistic_proba_oracle(fixture, fit_tol, proba_atol):
    """PY-01/D-12: LogisticRegression matches the gauge-fixed predict_proba.

    Compares ``predict_proba`` (the gauge-invariant gate), NOT raw ``coef_``,
    and asserts the predicted labels match the reference exactly.
    """
    d = np.load(fixture_path(fixture))
    est = mlrs.LogisticRegression(
        C=float(d["C"][0]), fit_intercept=True, max_iter=20000, tol=fit_tol
    ).fit(d["X"], d["y"])
    proba = est.predict_proba(d["Xq"])
    assert proba_allclose(proba, d["predict_proba"], atol=proba_atol)
    pred = np.asarray(est.predict(d["Xq"])).astype(np.int64).ravel()
    assert np.array_equal(pred, d["predict"].astype(np.int64).ravel())
