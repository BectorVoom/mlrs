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


# --- BayesianRidge (LINEAR-06) --------------------------------------------- #
#
# (case name, ctor kwargs, use_sample_weight, wide?) — mirrors
# `gen_oracle.py::gen_bayesian_ridge`'s `cases` list one-for-one.
BAYES_CASES = [
    ("default", {}, False, False),
    ("noint", {"fit_intercept": False}, False, False),
    ("maxiter1", {"max_iter": 1}, False, False),
    ("maxiter5", {"max_iter": 5}, False, False),
    ("tol_tight", {"tol": 1e-8, "max_iter": 1000}, False, False),
    ("tol_loose", {"tol": 1e-1}, False, False),
    ("priors", {"alpha_1": 1.0, "alpha_2": 5.0, "lambda_1": 50.0, "lambda_2": 1.0}, False, False),
    ("priors_zero", {"alpha_1": 0.0, "alpha_2": 0.0, "lambda_1": 0.0, "lambda_2": 0.0}, False, False),
    ("init", {"alpha_init": 2.5, "lambda_init": 0.1}, False, False),
    ("init_alpha_only", {"alpha_init": 10.0}, False, False),
    ("score", {"compute_score": True}, False, False),
    ("score_maxiter3", {"compute_score": True, "max_iter": 3}, False, False),
    ("score_noint", {"compute_score": True, "fit_intercept": False}, False, False),
    ("sw", {}, True, False),
    ("sw_noint", {"fit_intercept": False}, True, False),
    ("sw_score", {"compute_score": True}, True, False),
    ("wide", {}, False, True),
    ("wide_noint", {"fit_intercept": False}, False, True),
    ("wide_score", {"compute_score": True}, False, True),
]

BAYES_IDS = [
    f"{fx}-{case}"
    for fx in ("bayesian_ridge_f32_seed42", "bayesian_ridge_f64_seed42")
    for case, _, _, _ in BAYES_CASES
]


@pytest.mark.parametrize(
    "fixture,case,kwargs,use_sw,wide",
    [
        (fx, case, kwargs, use_sw, wide)
        for fx in ("bayesian_ridge_f32_seed42", "bayesian_ridge_f64_seed42")
        for case, kwargs, use_sw, wide in BAYES_CASES
    ],
    ids=BAYES_IDS,
)
def test_bayesian_ridge_oracle(fixture, case, kwargs, use_sw, wide):
    """PY-01: every sklearn BayesianRidge parameter matches through the shim.

    Gates SIX fitted attributes per case, not just ``coef_``: a wrong evidence
    update that lands on a similar penalty would still reproduce ``coef_`` to a
    few digits while missing ``alpha_``, ``lambda_`` and ``n_iter_``.

    ``alpha_`` / ``lambda_`` / ``sigma_`` / ``scores_`` are compared at the
    DESIGN's tolerance even though the fixture stores them as f64: both engines
    accumulate them in f64, so the storage width is not what limits agreement —
    the f32 design's input bytes are.

    Skips per fixture dtype (not via the blanket ``requires_f64`` marker) for the
    reason ``test_ridge_params_oracle`` documents.
    """
    if dtype_of(fixture) == np.float64 and not mlrs.backend_supports_f64():
        pytest.skip("backend does not support f64")
    d = np.load(fixture_path(fixture))
    x, y = (d["X_wide"], d["y_wide"]) if wide else (d["X"], d["y"])

    est = mlrs.BayesianRidge(**kwargs)
    est.fit(x, y, sample_weight=d["sample_weight"] if use_sw else None)

    # abs-OR-rel (`|got - exp| <= atol + rtol*|exp|`), the same rule
    # `bayesian_ridge_test.rs::assert_close` applies and the one the project
    # contract states. The other cases in this file can use a pure ABSOLUTE
    # tolerance because `coef_`/`intercept_` are O(1); `alpha_` is not — in the
    # interpolating `wide` regime it is ~5e5, where a 1e-5 absolute bound is
    # three orders tighter than f64 can represent a difference at all.
    atol = _atol(fixture)
    close = lambda got, want: np.allclose(  # noqa: E731
        np.ravel(np.asarray(got, dtype=np.float64)),
        np.ravel(np.asarray(want, dtype=np.float64)),
        atol=atol,
        rtol=atol,
    )
    assert close(est.coef_, d[f"coef_{case}"]), "coef_"
    assert close(est.intercept_, d[f"intercept_{case}"]), "intercept_"
    assert close(est.alpha_, d[f"alpha_{case}"]), "alpha_"
    assert close(est.lambda_, d[f"lambda_{case}"]), "lambda_"
    assert close(est.sigma_, d[f"sigma_{case}"]), "sigma_"
    assert est.n_iter_ == int(d[f"n_iter_{case}"][0]), "n_iter_"

    # `sigma_` is square and `X_scale_` is all ones whatever the fit.
    n_features = x.shape[1]
    assert np.asarray(est.sigma_).shape == (n_features, n_features)
    assert np.allclose(np.asarray(est.X_scale_), 1.0)

    if kwargs.get("compute_score"):
        # sklearn appends one score per iteration PLUS a final post-loop one.
        assert len(est.scores_) == est.n_iter_ + 1
        assert close(est.scores_, d[f"scores_{case}"]), "scores_"
    else:
        # sklearn leaves the attribute unset without `compute_score`; the shim
        # spells that as None.
        assert est.scores_ is None


@pytest.mark.parametrize(
    "fixture", ("bayesian_ridge_f32_seed42", "bayesian_ridge_f64_seed42")
)
def test_bayesian_ridge_predict_std_oracle(fixture):
    """``predict(X, return_std=True)`` on HELD-OUT rows.

    The only place ``sigma_`` becomes an observable rather than a stored
    attribute — and the only place sklearn centers ``X`` by ``X_offset_``, which
    a naive transcription of the formula gets wrong.
    """
    if dtype_of(fixture) == np.float64 and not mlrs.backend_supports_f64():
        pytest.skip("backend does not support f64")
    d = np.load(fixture_path(fixture))
    est = mlrs.BayesianRidge().fit(d["X"], d["y"])
    mean, std = est.predict(d["X_test"], return_std=True)

    atol = _atol(fixture)
    assert np.allclose(np.ravel(mean), np.ravel(d["pred_default"]), atol=atol, rtol=0.0)
    assert np.allclose(
        np.ravel(np.asarray(std, dtype=np.float64)),
        np.ravel(d["predstd_default"]),
        atol=atol,
        rtol=0.0,
    )
    # `return_std=False` (the default) returns the mean ALONE, not a 1-tuple.
    assert np.allclose(np.ravel(est.predict(d["X_test"])), np.ravel(mean), atol=atol, rtol=0.0)


def test_bayesian_ridge_rejects_bad_params():
    """sklearn's `_parameter_constraints` rejections, through the shim."""
    import pytest as _pytest

    for kwargs in (
        {"max_iter": 0},
        {"tol": 0.0},          # closed="neither" — unlike Ridge, 0 is rejected
        {"tol": -1e-3},
        {"alpha_1": -1.0},
        {"alpha_2": -1.0},
        {"lambda_1": -1.0},
        {"lambda_2": -1.0},
        {"alpha_init": -1.0},
        {"lambda_init": -1.0},
    ):
        with _pytest.raises(Exception):
            mlrs.BayesianRidge(**kwargs).fit(np.eye(4, 3), np.arange(4.0))

    # The boundary values sklearn ACCEPTS (`closed="left"` on the hyperpriors).
    mlrs.BayesianRidge(
        alpha_1=0.0, alpha_2=0.0, lambda_1=0.0, lambda_2=0.0
    ).fit(np.eye(4, 3), np.arange(4.0))


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
