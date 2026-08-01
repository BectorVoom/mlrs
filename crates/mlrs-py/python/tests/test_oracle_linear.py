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
from conftest import dtype_of, fixture_path, proba_allclose, requires_f64  # noqa: F401


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


# Ridge FULL parameter surface: every sklearn `solver`, `fit_intercept`,
# `positive`, and `sample_weight` case in the `ridge_params_*` fixture, driven
# through the Python shim (the Rust twin is
# `crates/mlrs-algos/tests/ridge_params_test.rs`). Each entry is
# (case name, ctor kwargs, uses sample_weight, expected `solver_`).
RIDGE_PARAM_CASES = [
    ("auto", {"solver": "auto"}, False, "cholesky"),
    ("cholesky", {"solver": "cholesky"}, False, "cholesky"),
    ("svd", {"solver": "svd"}, False, "svd"),
    ("lsqr", {"solver": "lsqr"}, False, "lsqr"),
    ("sparse_cg", {"solver": "sparse_cg"}, False, "sparse_cg"),
    ("sag", {"solver": "sag"}, False, "sag"),
    ("saga", {"solver": "saga"}, False, "saga"),
    ("lbfgs_pos", {"solver": "lbfgs", "positive": True}, False, "lbfgs"),
    ("auto_pos", {"solver": "auto", "positive": True}, False, "lbfgs"),
    ("cholesky_noint", {"solver": "cholesky", "fit_intercept": False}, False, "cholesky"),
    ("svd_noint", {"solver": "svd", "fit_intercept": False}, False, "svd"),
    ("lsqr_noint", {"solver": "lsqr", "fit_intercept": False}, False, "lsqr"),
    ("sag_noint", {"solver": "sag", "fit_intercept": False}, False, "sag"),
    ("lbfgs_pos_noint", {"solver": "lbfgs", "positive": True, "fit_intercept": False}, False, "lbfgs"),
    ("cholesky_sw", {"solver": "cholesky"}, True, "cholesky"),
    ("svd_sw", {"solver": "svd"}, True, "svd"),
    ("lsqr_sw", {"solver": "lsqr"}, True, "lsqr"),
    ("sparse_cg_sw", {"solver": "sparse_cg"}, True, "sparse_cg"),
    ("sag_sw", {"solver": "sag"}, True, "sag"),
    ("saga_sw", {"solver": "saga"}, True, "saga"),
    ("lbfgs_pos_sw", {"solver": "lbfgs", "positive": True}, True, "lbfgs"),
    ("cholesky_noint_sw", {"solver": "cholesky", "fit_intercept": False}, True, "cholesky"),
]

# sklearn populates `n_iter_` only for `lsqr` and the SAG family
# (`_ridge_regression` leaves it None for every other solver).
_N_ITER_SOLVERS = {"lsqr", "sag", "saga"}

RIDGE_PARAM_IDS = [
    f"{fx}-{case}"
    for fx in ("ridge_params_f32_seed42", "ridge_params_f64_seed42")
    for case, _, _, _ in RIDGE_PARAM_CASES
]


@pytest.mark.parametrize(
    "fixture,case,kwargs,use_sw,expect_solver",
    [
        (fx, case, kwargs, use_sw, want)
        for fx in ("ridge_params_f32_seed42", "ridge_params_f64_seed42")
        for case, kwargs, use_sw, want in RIDGE_PARAM_CASES
    ],
    ids=RIDGE_PARAM_IDS,
)
def test_ridge_params_oracle(fixture, case, kwargs, use_sw, expect_solver):
    """PY-01: every sklearn Ridge parameter matches sklearn through the shim.

    Both sides are fitted at the fixture's tight ``tol``/``max_iter`` so the
    iterative solvers are compared at their CONVERGED optimum (see
    ``gen_oracle.py::gen_ridge_params``).

    Skips PER FIXTURE DTYPE rather than wearing the blanket ``requires_f64``
    marker the older cases use: that marker skips the whole function, so on an
    f64-incapable backend (wgpu / rocm) the f32 half — which is exactly the half
    those backends CAN run, and the one the GPU gate cares about — would be
    thrown away with it.
    """
    if dtype_of(fixture) == np.float64 and not mlrs.backend_supports_f64():
        pytest.skip("backend does not support f64")
    d = np.load(fixture_path(fixture))
    est = mlrs.Ridge(
        alpha=float(d["alpha"][0]),
        tol=float(d["tol"][0]),
        max_iter=int(d["max_iter"][0]),
        random_state=0,
        **kwargs,
    )
    est.fit(d["X"], d["y"], sample_weight=d["sample_weight"] if use_sw else None)

    atol = _atol(fixture)
    assert np.allclose(
        np.ravel(np.asarray(est.coef_)), np.ravel(d[f"coef_{case}"]), atol=atol, rtol=0.0
    )
    assert np.allclose(
        np.ravel(np.asarray(est.intercept_)),
        np.ravel(d[f"intercept_{case}"]),
        atol=atol,
        rtol=0.0,
    )
    # `solver_` — the resolved solver, including auto -> cholesky / lbfgs.
    assert est.solver_ == expect_solver
    # `n_iter_` — Some exactly where sklearn populates it.
    assert (est.n_iter_ is not None) == (expect_solver in _N_ITER_SOLVERS)
    if kwargs.get("positive"):
        assert (np.asarray(est.coef_) >= -atol).all()


def test_ridge_rejects_bad_params():
    """The sklearn ``ValueError``s Ridge raises for invalid parameter combos.

    Deliberately driven from the f32 fixture: on an f64-incapable backend an f64
    ``X`` raises ``ValueError`` from the capability guard, which would make every
    ``pytest.raises(ValueError)`` below pass for the WRONG reason.
    """
    d = np.load(fixture_path("ridge_params_f32_seed42"))
    X, y = d["X"], d["y"]
    with pytest.raises(ValueError):
        mlrs.Ridge(alpha=-1.0).fit(X, y)
    with pytest.raises(ValueError):
        mlrs.Ridge(tol=-1.0).fit(X, y)
    with pytest.raises(ValueError):
        mlrs.Ridge(max_iter=0).fit(X, y)
    with pytest.raises(ValueError):
        mlrs.Ridge(solver="lbfgs").fit(X, y)  # lbfgs requires positive=True
    with pytest.raises(ValueError):
        mlrs.Ridge(solver="cholesky", positive=True).fit(X, y)
    with pytest.raises(ValueError):
        mlrs.Ridge(solver="newton-cholesky").fit(X, y)


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
