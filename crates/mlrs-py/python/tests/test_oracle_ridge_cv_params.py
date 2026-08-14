"""``RidgeCV`` full-parameter surface through the Python shim, vs LIVE sklearn.

Every ``sklearn.linear_model.RidgeCV`` parameter is exercised here against a
live ``sklearn.linear_model.RidgeCV`` rather than a committed fixture, for the
reason the KMeans/KNN param suites give: what has to be checked is exactly the
thing a fixture would freeze away — that mlrs's resolution rules still agree
with sklearn's as sklearn evolves them. ``RidgeCV`` is the sharpest case of
that in this repo: sklearn 1.9 *rewrote* ``_check_gcv_mode`` (``'eigen'`` and
``'auto'`` became the same route, and the dense ``n > d`` default moved from an
SVD of the design to an eigendecomposition of ``XᵀX``), so a fixture generated
against 1.4 would still pass while agreeing with nothing anyone runs.

The two STRING-valued parameters get the most attention, as the campaign
requires:

* ``gcv_mode`` — ``None`` / ``'auto'`` / ``'svd'`` / ``'eigen'``, each compared
  against sklearn run in the SAME mode, at BOTH shape regimes (``n > d``, which
  sklearn routes to ``"cov"``, and ``n <= d``, which it routes to ``"gram"``),
  plus the rejection of an unknown string.
* ``scoring`` — every regression scorer name sklearn ships that is defined for
  this design, each compared against sklearn under the same name, plus ``None``
  (the ``−mean(looe²)`` arm the Rust engine scores itself) and a user callable.

``alpha_`` is asserted EXACTLY. It is a choice from a discrete grid, so an
approximate match would hide the only failure mode that matters here: picking a
different penalty. The designs below are built so the LOO curve has one clear
minimum well inside the grid, which is checked by ``test_alpha_is_interior``
rather than assumed.

f64 designs are skipped-with-reason on an f64-incapable backend via
``conftest.default_float_dtype`` / ``live_atol``.
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge as SkRidge
from sklearn.linear_model import RidgeCV as SkRidgeCV
from sklearn.metrics import get_scorer_names, mean_squared_error
from sklearn.model_selection import KFold

import mlrs
from conftest import default_float_dtype, live_atol

# The alpha grid every case uses unless it says otherwise. Log-spaced and wide
# enough that the winner is interior for these designs (asserted below), so a
# one-rung disagreement is a real defect and not a boundary artifact.
ALPHAS = np.logspace(-2, 3, 11)


def regression(n=400, d=12, n_targets=0, seed=0, noise=0.4, dtype=None):
    """A well-conditioned random regression design.

    Deliberately NOT collinear: `RidgeCV`'s GCV engine reaches its spectrum
    through `XᵀX` (see `ridge_cv.rs`'s module docs on why, and on the
    condition-number cost that buys), so a design engineered to be
    near-singular would be testing the emulation's known limit rather than the
    estimator. The near-singular case has its own test at the bottom, with the
    tolerance that regime honestly supports.
    """
    dtype = dtype or default_float_dtype()
    rs = np.random.RandomState(seed)
    x = rs.normal(size=(n, d))
    k = n_targets or 1
    coef = rs.normal(size=(d, k))
    y = x @ coef + 3.0 + noise * rs.normal(size=(n, k))
    if n_targets == 0:
        y = y[:, 0]
    return (
        np.ascontiguousarray(x, dtype=dtype),
        np.ascontiguousarray(y, dtype=dtype),
    )


def assert_matches(est, sk, what, atol=None, check_alpha=True):
    """mlrs's fitted `RidgeCV` equals sklearn's."""
    atol = live_atol() if atol is None else atol
    if check_alpha:
        assert np.allclose(
            np.asarray(est.alpha_, dtype=np.float64),
            np.asarray(sk.alpha_, dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        ), f"{what}: alpha_ {est.alpha_!r} != sklearn {sk.alpha_!r}"

    coef = np.asarray(est.coef_, dtype=np.float64)
    ref = np.asarray(sk.coef_, dtype=np.float64)
    assert coef.shape == ref.shape, (
        f"{what}: coef_ shape {coef.shape} != sklearn {ref.shape}"
    )
    assert np.allclose(coef, ref, atol=atol, rtol=0.0), (
        f"{what}: coef_ differs (max {np.abs(coef - ref).max():.3e})"
    )

    icp = np.asarray(est.intercept_, dtype=np.float64)
    icp_ref = np.asarray(sk.intercept_, dtype=np.float64)
    assert icp.shape == icp_ref.shape, (
        f"{what}: intercept_ shape {icp.shape} != sklearn {icp_ref.shape}"
    )
    assert np.allclose(icp, icp_ref, atol=atol, rtol=0.0), (
        f"{what}: intercept_ differs (max {np.abs(icp - icp_ref).max():.3e})"
    )

    best = np.asarray(est.best_score_, dtype=np.float64)
    best_ref = np.asarray(sk.best_score_, dtype=np.float64)
    assert best.shape == best_ref.shape, (
        f"{what}: best_score_ shape {best.shape} != sklearn {best_ref.shape}"
    )
    assert np.allclose(
        best, best_ref, atol=atol, rtol=1e-6
    ), f"{what}: best_score_ {best!r} != sklearn {best_ref!r}"


def assert_predict_matches(est, sk, X, what, atol=None):
    atol = live_atol() if atol is None else atol
    p = np.asarray(est.predict(X), dtype=np.float64)
    ref = np.asarray(sk.predict(X), dtype=np.float64)
    assert p.shape == ref.shape, (
        f"{what}: predict shape {p.shape} != sklearn {ref.shape}"
    )
    assert np.allclose(p, ref, atol=atol, rtol=0.0), (
        f"{what}: predict differs (max {np.abs(p - ref).max():.3e})"
    )


# ---------------------------------------------------------------------------
# The grid the winner is chosen from must actually have an interior winner --
# otherwise every "alpha_ matches" assertion below is vacuous.
# ---------------------------------------------------------------------------


def test_alpha_is_interior():
    X, y = regression()
    sk = SkRidgeCV(alphas=ALPHAS).fit(X, y)
    idx = int(np.argmin(np.abs(ALPHAS - sk.alpha_)))
    assert 0 < idx < len(ALPHAS) - 1, (
        f"sklearn picked a grid ENDPOINT (alpha_={sk.alpha_}); the design or "
        "the grid needs adjusting or the alpha_ assertions prove nothing"
    )


# ---------------------------------------------------------------------------
# gcv_mode -- the first string parameter, at both shape regimes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gcv_mode", [None, "auto", "svd", "eigen"])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((400, 12), id="tall(cov)"),
        pytest.param((30, 60), id="wide(gram)"),
    ],
)
def test_gcv_mode(gcv_mode, shape):
    n, d = shape
    X, y = regression(n=n, d=d)
    est = mlrs.RidgeCV(alphas=ALPHAS, gcv_mode=gcv_mode).fit(X, y)
    sk = SkRidgeCV(alphas=ALPHAS, gcv_mode=gcv_mode).fit(X, y)
    assert_matches(est, sk, f"gcv_mode={gcv_mode!r} shape={shape}")
    assert_predict_matches(est, sk, X, f"gcv_mode={gcv_mode!r} shape={shape}")


def test_gcv_mode_values_agree_with_each_other():
    """mlrs derives all three modes from ONE eigendecomposition, so they must be
    IDENTICAL here -- not merely close.

    This is the assertion that keeps the module doc honest. If a future change
    gives one mode its own arm, this test fails and the docs (and the perf
    claim that the three cost the same) have to be revisited.
    """
    X, y = regression()
    fits = {
        m: mlrs.RidgeCV(alphas=ALPHAS, gcv_mode=m).fit(X, y)
        for m in (None, "auto", "svd", "eigen")
    }
    base = np.asarray(fits["auto"].coef_, dtype=np.float64)
    for m, f in fits.items():
        assert np.array_equal(np.asarray(f.coef_, dtype=np.float64), base), (
            f"gcv_mode={m!r} is not bit-identical to 'auto'"
        )


def test_gcv_mode_rejects_unknown_string():
    X, y = regression(n=60, d=4)
    with pytest.raises(ValueError, match="gcv_mode"):
        mlrs.RidgeCV(alphas=ALPHAS, gcv_mode="lanczos").fit(X, y)


# ---------------------------------------------------------------------------
# scoring -- the second string parameter
# ---------------------------------------------------------------------------

# Every regression scorer sklearn ships, filtered to the ones defined for a
# dense single-target design. Read from `get_scorer_names()` rather than
# hard-coded so a scorer added by a future sklearn is covered the day it lands
# (and one removed stops being a spurious failure).
REGRESSION_SCORERS = sorted(
    name
    for name in get_scorer_names()
    if name == "r2"
    or name == "explained_variance"
    or name == "max_error"
    or (name.startswith("neg_") and "mean" in name)
    or name in ("neg_median_absolute_error", "d2_absolute_error_score")
)


@pytest.mark.parametrize("scoring", REGRESSION_SCORERS)
def test_scoring_string(scoring):
    X, y = regression()
    # `neg_mean_squared_log_error` / `neg_mean_poisson_deviance` and friends are
    # undefined for a signed target; skip the cell rather than reshape the
    # design out from under every other scorer.
    try:
        sk = SkRidgeCV(alphas=ALPHAS, scoring=scoring).fit(X, y)
    except ValueError as exc:
        pytest.skip(f"sklearn cannot score this design with {scoring!r}: {exc}")
    est = mlrs.RidgeCV(alphas=ALPHAS, scoring=scoring).fit(X, y)
    assert_matches(est, sk, f"scoring={scoring!r}")
    assert_predict_matches(est, sk, X, f"scoring={scoring!r}")


def test_scoring_none_is_the_negative_mean_squared_loo_error():
    """`scoring=None` is NOT `'neg_mean_squared_error'` on the same predictions
    -- it is the LOO squared error of the PREPROCESSED target, which differs
    whenever `sample_weight` rescales the rows. Both are checked against
    sklearn, which is the only way to catch a shim that quietly substitutes one
    for the other."""
    X, y = regression()
    a = mlrs.RidgeCV(alphas=ALPHAS, scoring=None).fit(X, y)
    b = SkRidgeCV(alphas=ALPHAS, scoring=None).fit(X, y)
    assert_matches(a, b, "scoring=None")
    # ... and with weights, where the two arms genuinely diverge.
    rs = np.random.RandomState(3)
    w = rs.uniform(0.2, 3.0, size=X.shape[0])
    a = mlrs.RidgeCV(alphas=ALPHAS, scoring=None).fit(X, y, sample_weight=w)
    b = SkRidgeCV(alphas=ALPHAS, scoring=None).fit(X, y, sample_weight=w)
    assert_matches(a, b, "scoring=None + sample_weight")


def test_scoring_callable():
    X, y = regression()

    def scorer(estimator, X_, y_):
        return -mean_squared_error(y_, estimator.predict(X_))

    est = mlrs.RidgeCV(alphas=ALPHAS, scoring=scorer).fit(X, y)
    sk = SkRidgeCV(alphas=ALPHAS, scoring=scorer).fit(X, y)
    assert_matches(est, sk, "scoring=<callable>")


# ---------------------------------------------------------------------------
# cv -- the GridSearchCV arm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cv",
    [
        pytest.param(3, id="int"),
        pytest.param(5, id="int5"),
        pytest.param(KFold(n_splits=4), id="KFold"),
        pytest.param(KFold(n_splits=4, shuffle=True, random_state=7), id="shuffled"),
    ],
)
def test_cv(cv):
    X, y = regression()
    # Splitters are stateless across `split()` calls (`KFold` is not clonable by
    # `sklearn.base.clone` at all), so the same object is handed to both.
    est = mlrs.RidgeCV(alphas=ALPHAS, cv=cv)
    sk = SkRidgeCV(alphas=ALPHAS, cv=cv)
    est.fit(X, y)
    sk.fit(X, y)
    assert_matches(est, sk, f"cv={cv!r}")
    assert_predict_matches(est, sk, X, f"cv={cv!r}")


def test_cv_iterable_of_splits():
    X, y = regression(n=120, d=6)
    n = X.shape[0]
    idx = np.arange(n)
    splits = [(idx[: n // 2], idx[n // 2 :]), (idx[n // 2 :], idx[: n // 2])]
    est = mlrs.RidgeCV(alphas=ALPHAS, cv=list(splits)).fit(X, y)
    sk = SkRidgeCV(alphas=ALPHAS, cv=list(splits)).fit(X, y)
    assert_matches(est, sk, "cv=<iterable>")


@pytest.mark.parametrize("scoring", ["r2", "neg_mean_absolute_error"])
def test_cv_with_scoring(scoring):
    X, y = regression()
    est = mlrs.RidgeCV(alphas=ALPHAS, cv=4, scoring=scoring).fit(X, y)
    sk = SkRidgeCV(alphas=ALPHAS, cv=4, scoring=scoring).fit(X, y)
    assert_matches(est, sk, f"cv=4 scoring={scoring!r}")


@pytest.mark.parametrize("scoring", [None, "r2", "neg_mean_absolute_error"])
def test_cv_with_sample_weight(scoring):
    """The weights reach BOTH the per-fold fit and the held-out scorer.

    sklearn's ``GridSearchCV`` forwards ``sample_weight`` to a scorer that
    accepts it, so the CV score is a WEIGHTED R² by default. Scoring the folds
    unweighted still agrees to three decimals on most designs — which is exactly
    why it needs a test rather than an eyeball.
    """
    X, y = regression()
    rs = np.random.RandomState(11)
    w = rs.uniform(0.3, 2.0, size=X.shape[0])
    est = mlrs.RidgeCV(alphas=ALPHAS, cv=4, scoring=scoring).fit(
        X, y, sample_weight=w
    )
    sk = SkRidgeCV(alphas=ALPHAS, cv=4, scoring=scoring).fit(X, y, sample_weight=w)
    assert_matches(est, sk, f"cv=4 + sample_weight scoring={scoring!r}")


@pytest.mark.parametrize("scoring", ["r2", "neg_mean_absolute_error"])
def test_gcv_scoring_with_sample_weight(scoring):
    """The GCV arm forwards the weights to the scorer too — and, unlike the
    grid arm, without an accepts-check (sklearn's own asymmetry)."""
    X, y = regression()
    rs = np.random.RandomState(13)
    w = rs.uniform(0.4, 2.5, size=X.shape[0])
    est = mlrs.RidgeCV(alphas=ALPHAS, scoring=scoring).fit(X, y, sample_weight=w)
    sk = SkRidgeCV(alphas=ALPHAS, scoring=scoring).fit(X, y, sample_weight=w)
    assert_matches(est, sk, f"gcv scoring={scoring!r} + sample_weight")


@pytest.mark.parametrize("scoring", [None, "r2"])
def test_cv_multi_target(scoring):
    """The grid arm with a 2-D `y`: the fold score is R² averaged UNIFORMLY over
    targets, and the refit at the winner is a multi-output `Ridge`."""
    X, y = regression(n=300, d=8, n_targets=3)
    est = mlrs.RidgeCV(alphas=ALPHAS, cv=4, scoring=scoring).fit(X, y)
    sk = SkRidgeCV(alphas=ALPHAS, cv=4, scoring=scoring).fit(X, y)
    assert_matches(est, sk, f"cv=4 multi-target scoring={scoring!r}")
    assert_predict_matches(est, sk, X, f"cv=4 multi-target scoring={scoring!r}")


def test_cv_accepts_alpha_zero():
    """sklearn's boundary rule: `alpha = 0` is rejected for the GCV arm (the LOO
    identity divides by it) and ACCEPTED for the explicit-cv arm."""
    X, y = regression(n=120, d=6)
    alphas = [0.0, 1.0, 10.0]
    est = mlrs.RidgeCV(alphas=alphas, cv=3).fit(X, y)
    sk = SkRidgeCV(alphas=alphas, cv=3).fit(X, y)
    assert_matches(est, sk, "cv=3 alphas=[0, ...]")
    with pytest.raises(ValueError):
        mlrs.RidgeCV(alphas=alphas).fit(X, y)


# ---------------------------------------------------------------------------
# fit_intercept / sample_weight / alphas spellings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_fit_intercept_and_sample_weight(fit_intercept, weighted):
    X, y = regression()
    kw = {}
    if weighted:
        rs = np.random.RandomState(5)
        kw["sample_weight"] = rs.uniform(0.25, 4.0, size=X.shape[0])
    est = mlrs.RidgeCV(alphas=ALPHAS, fit_intercept=fit_intercept).fit(X, y, **kw)
    sk = SkRidgeCV(alphas=ALPHAS, fit_intercept=fit_intercept).fit(X, y, **kw)
    what = f"fit_intercept={fit_intercept} weighted={weighted}"
    assert_matches(est, sk, what)
    assert_predict_matches(est, sk, X, what)


@pytest.mark.parametrize(
    "alphas",
    [
        pytest.param((0.1, 1.0, 10.0), id="tuple-default"),
        pytest.param([0.5, 5.0], id="list"),
        pytest.param(np.array([0.01, 0.1, 1.0, 10.0]), id="ndarray"),
        pytest.param(2.5, id="scalar"),
    ],
)
def test_alphas_spellings(alphas):
    X, y = regression()
    est = mlrs.RidgeCV(alphas=alphas).fit(X, y)
    sk = SkRidgeCV(alphas=alphas).fit(X, y)
    assert_matches(est, sk, f"alphas={alphas!r}")


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_gcv_rejects_non_positive_alpha(bad):
    X, y = regression(n=60, d=4)
    with pytest.raises(ValueError):
        mlrs.RidgeCV(alphas=[bad, 1.0]).fit(X, y)


# ---------------------------------------------------------------------------
# store_cv_results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scoring", [None, "r2"])
@pytest.mark.parametrize("n_targets", [0, 3])
def test_store_cv_results(scoring, n_targets):
    X, y = regression(n_targets=n_targets)
    est = mlrs.RidgeCV(alphas=ALPHAS, scoring=scoring, store_cv_results=True).fit(X, y)
    sk = SkRidgeCV(alphas=ALPHAS, scoring=scoring, store_cv_results=True).fit(X, y)
    what = f"store_cv_results scoring={scoring!r} n_targets={n_targets}"
    assert_matches(est, sk, what)
    got = np.asarray(est.cv_results_, dtype=np.float64)
    ref = np.asarray(sk.cv_results_, dtype=np.float64)
    assert got.shape == ref.shape, (
        f"{what}: cv_results_ shape {got.shape} != sklearn {ref.shape}"
    )
    # The LOO residuals themselves, not just their argmin -- a shim that got the
    # winner right off wrong per-sample values would pass every other assertion.
    scale = max(1.0, float(np.abs(ref).max()))
    assert np.allclose(got, ref, atol=live_atol() * scale, rtol=1e-6), (
        f"{what}: cv_results_ differs (max {np.abs(got - ref).max():.3e})"
    )


def test_store_cv_results_is_off_by_default():
    X, y = regression(n=80, d=5)
    est = mlrs.RidgeCV(alphas=ALPHAS).fit(X, y)
    assert not hasattr(est, "cv_results_")


def test_store_cv_results_with_cv_is_rejected():
    X, y = regression(n=80, d=5)
    with pytest.raises(ValueError, match="store_cv_results"):
        mlrs.RidgeCV(alphas=ALPHAS, cv=3, store_cv_results=True).fit(X, y)


# ---------------------------------------------------------------------------
# alpha_per_target + multi-target y
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha_per_target", [False, True])
@pytest.mark.parametrize("scoring", [None, "r2"])
def test_multi_target(alpha_per_target, scoring):
    # Targets with very different noise levels, so the per-target optimum is
    # genuinely a different alpha for each -- otherwise `alpha_per_target=True`
    # and `False` agree by accident and the parameter is untested.
    rs = np.random.RandomState(2)
    dtype = default_float_dtype()
    X = np.ascontiguousarray(rs.normal(size=(300, 10)), dtype=dtype)
    coef = rs.normal(size=(10, 3))
    y = X @ coef + 1.5
    y = y + rs.normal(size=y.shape) * np.array([0.05, 1.0, 6.0])
    y = np.ascontiguousarray(y, dtype=dtype)

    est = mlrs.RidgeCV(
        alphas=ALPHAS, alpha_per_target=alpha_per_target, scoring=scoring
    ).fit(X, y)
    sk = SkRidgeCV(
        alphas=ALPHAS, alpha_per_target=alpha_per_target, scoring=scoring
    ).fit(X, y)
    what = f"multi-target alpha_per_target={alpha_per_target} scoring={scoring!r}"
    assert_matches(est, sk, what)
    assert_predict_matches(est, sk, X, what)
    if alpha_per_target:
        assert np.asarray(est.alpha_).shape == (3,)
        assert len(set(np.asarray(sk.alpha_).tolist())) > 1, (
            "the design no longer separates the per-target optima -- "
            "alpha_per_target is being tested vacuously"
        )
    else:
        assert np.isscalar(est.alpha_) or np.asarray(est.alpha_).ndim == 0


def test_alpha_per_target_with_cv_is_rejected():
    X, y = regression(n=80, d=5, n_targets=2)
    with pytest.raises(ValueError, match="alpha_per_target"):
        mlrs.RidgeCV(alphas=ALPHAS, cv=3, alpha_per_target=True).fit(X, y)


def test_column_vector_y_keeps_sklearns_shapes():
    """A `(n, 1)` `y` is 2-D to sklearn but single-target: `coef_` ravels and
    `intercept_` stays an ARRAY. Getting one of the two wrong is invisible until
    a caller stacks the results."""
    X, y = regression()
    y2 = y.reshape(-1, 1)
    est = mlrs.RidgeCV(alphas=ALPHAS).fit(X, y2)
    sk = SkRidgeCV(alphas=ALPHAS).fit(X, y2)
    assert_matches(est, sk, "y.shape=(n, 1)")
    assert np.asarray(est.coef_).shape == np.asarray(sk.coef_).shape
    assert np.asarray(est.intercept_).shape == np.asarray(sk.intercept_).shape
    assert_predict_matches(est, sk, X, "y.shape=(n, 1)")


# ---------------------------------------------------------------------------
# sklearn-contract surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cv", [None, 3])
def test_pyarrow_input_round_trips(cv):
    """A pyarrow input must survive BOTH engines.

    The grid arm hands `X` to the splitter, and an earlier draft passed it
    through `np.asarray` first — which is fine for numpy and pandas and breaks a
    pyarrow Table. The splitter only ever needs the row count, which
    `_num_samples` reads natively, so `X` goes through unconverted.
    """
    pa = pytest.importorskip("pyarrow")
    X, y = regression(n=150, d=6)
    tbl = pa.table({f"c{j}": X[:, j] for j in range(X.shape[1])})
    est = mlrs.RidgeCV(alphas=ALPHAS, cv=cv).fit(tbl, y)
    sk = SkRidgeCV(alphas=ALPHAS, cv=cv).fit(X, y)
    assert_matches(est, sk, f"pyarrow input cv={cv!r}")


def test_get_params_round_trips():
    from sklearn.base import clone

    est = mlrs.RidgeCV(
        alphas=[0.1, 1.0],
        fit_intercept=False,
        scoring="r2",
        cv=3,
        gcv_mode="svd",
        store_cv_results=False,
        alpha_per_target=False,
    )
    params = est.get_params()
    for k, v in {
        "alphas": [0.1, 1.0],
        "fit_intercept": False,
        "scoring": "r2",
        "cv": 3,
        "gcv_mode": "svd",
        "store_cv_results": False,
        "alpha_per_target": False,
    }.items():
        assert params[k] == v, f"get_params()[{k!r}] == {params[k]!r}"
    assert clone(est).get_params() == params


def test_not_fitted_raises():
    from sklearn.exceptions import NotFittedError

    est = mlrs.RidgeCV()
    with pytest.raises(NotFittedError):
        est.predict(np.zeros((2, 3)))
    with pytest.raises(NotFittedError):
        _ = est.coef_


def test_score_matches_sklearn():
    X, y = regression()
    est = mlrs.RidgeCV(alphas=ALPHAS).fit(X, y)
    sk = SkRidgeCV(alphas=ALPHAS).fit(X, y)
    assert abs(est.score(X, y) - sk.score(X, y)) <= live_atol()


def test_matches_a_plain_ridge_at_the_chosen_alpha():
    """The winner's coefficients must be the ones a plain `Ridge` at that alpha
    produces. This is an INTERNAL consistency gate that does not go through
    sklearn's `RidgeCV` at all, so it still fires if both libraries were to pick
    the same wrong alpha."""
    X, y = regression()
    est = mlrs.RidgeCV(alphas=ALPHAS).fit(X, y)
    ref = SkRidge(alpha=est.alpha_).fit(X, y)
    assert np.allclose(
        np.asarray(est.coef_, dtype=np.float64),
        np.asarray(ref.coef_, dtype=np.float64),
        atol=live_atol(),
        rtol=0.0,
    )


def test_near_singular_design_is_close_but_looser():
    """The documented cost of reaching the spectrum through `XᵀX`.

    A rank-deficient design squares an already-huge condition number, so the
    gate here is looser than the 1e-5 the rest of this file holds. It is a
    test rather than a caveat in a comment because the point is that the answer
    stays USABLE, not that it stays exact: assert it tracks sklearn to a
    relative 1e-4 and that nothing goes NaN.
    """
    rs = np.random.RandomState(9)
    dtype = default_float_dtype()
    base = rs.normal(size=(200, 5))
    # Two exactly-duplicated columns -> rank 5 in a 7-column design.
    X = np.ascontiguousarray(
        np.hstack([base, base[:, :2]]) + 1e-9 * rs.normal(size=(200, 7)),
        dtype=dtype,
    )
    y = np.ascontiguousarray(base @ rs.normal(size=5) + 0.5, dtype=dtype)
    est = mlrs.RidgeCV(alphas=ALPHAS).fit(X, y)
    sk = SkRidgeCV(alphas=ALPHAS).fit(X, y)
    assert np.all(np.isfinite(np.asarray(est.coef_, dtype=np.float64)))
    p = np.asarray(est.predict(X), dtype=np.float64)
    ref = np.asarray(sk.predict(X), dtype=np.float64)
    scale = max(1.0, float(np.abs(ref).max()))
    assert np.allclose(p, ref, atol=1e-4 * scale, rtol=0.0), (
        f"near-singular predict differs (max {np.abs(p - ref).max():.3e})"
    )


# ---------------------------------------------------------------------------
# device -- the string parameter added by RIDGECV-02
#
# `device` is VALUE-NEUTRAL, so there is nothing about it for an oracle to
# compare against sklearn (sklearn has no such parameter). What the oracle CAN
# do, and what these cases do, is re-run the two value-bearing string
# parameters on the DEVICE arm and demand the same agreement with sklearn the
# host arm gives -- because "the arms agree with each other" (the Rust gate in
# `ridge_cv_device_test.rs`) and "the arms agree with sklearn" are different
# claims, and only the second one is what a user gets.
#
# Every case here SKIPS with a reason when the backend has no device arm, and
# it establishes that by fitting and reading `device_` rather than by guessing
# from the backend name -- a `device='gpu'` that silently fell back would
# otherwise re-run the host suite and report a pass.
# ---------------------------------------------------------------------------


def _device_arm_or_skip():
    """Fit a `device='gpu'` probe on a small tall design, or skip the case."""
    X, y = regression(n=300, d=8)
    probe = mlrs.RidgeCV(alphas=ALPHAS, device="gpu").fit(X, y)
    if probe.device_ != "gpu":
        pytest.skip(
            "no RidgeCV device arm on this backend "
            f"(device='gpu' reported device_={probe.device_!r})"
        )


@pytest.mark.parametrize("gcv_mode", [None, "auto", "svd", "eigen"])
def test_device_arm_gcv_mode(gcv_mode):
    _device_arm_or_skip()
    X, y = regression()
    est = mlrs.RidgeCV(alphas=ALPHAS, gcv_mode=gcv_mode, device="gpu").fit(X, y)
    assert est.device_ == "gpu"
    sk = SkRidgeCV(alphas=ALPHAS, gcv_mode=gcv_mode).fit(X, y)
    assert_matches(est, sk, f"device='gpu' gcv_mode={gcv_mode!r}")
    assert_predict_matches(est, sk, X, f"device='gpu' gcv_mode={gcv_mode!r}")


@pytest.mark.parametrize("scoring", REGRESSION_SCORERS)
def test_device_arm_scoring_string(scoring):
    _device_arm_or_skip()
    X, y = regression()
    try:
        sk = SkRidgeCV(alphas=ALPHAS, scoring=scoring).fit(X, y)
    except ValueError as exc:
        pytest.skip(f"sklearn cannot score this design with {scoring!r}: {exc}")
    est = mlrs.RidgeCV(alphas=ALPHAS, scoring=scoring, device="gpu").fit(X, y)
    assert est.device_ == "gpu"
    assert_matches(est, sk, f"device='gpu' scoring={scoring!r}")


def test_device_rejects_an_unknown_string():
    """`device` is validated in the shim with the same `StrOptions` shape every
    other string hyperparameter uses, so a typo is a `ValueError` naming the
    estimator rather than a silent fallback. Backend-independent: the rejection
    happens before any arm is chosen."""
    X, y = regression(n=120, d=5)
    with pytest.raises(ValueError, match="device"):
        mlrs.RidgeCV(alphas=ALPHAS, device="cuda").fit(X, y)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"fit_intercept": False},
        {"store_cv_results": True},
        {"alpha_per_target": True},
    ],
    ids=["default", "no-intercept", "store-cv-results", "alpha-per-target"],
)
def test_device_arm_matches_sklearn_across_the_surface(kwargs):
    """The rest of the parameter surface on the device arm.

    `alpha_per_target` needs a 2-D `y`, so the design follows the parameter.
    """
    _device_arm_or_skip()
    multi = kwargs.get("alpha_per_target", False)
    X, y = regression(n_targets=3) if multi else regression()
    est = mlrs.RidgeCV(alphas=ALPHAS, device="gpu", **kwargs).fit(X, y)
    assert est.device_ == "gpu"
    sk = SkRidgeCV(alphas=ALPHAS, **kwargs).fit(X, y)
    assert_matches(est, sk, f"device='gpu' {kwargs}")
    assert_predict_matches(est, sk, X, f"device='gpu' {kwargs}")


def test_device_arm_with_sample_weight():
    _device_arm_or_skip()
    X, y = regression()
    rs = np.random.RandomState(11)
    w = np.abs(rs.normal(size=X.shape[0])) + 0.05
    est = mlrs.RidgeCV(alphas=ALPHAS, device="gpu").fit(X, y, sample_weight=w)
    assert est.device_ == "gpu"
    sk = SkRidgeCV(alphas=ALPHAS).fit(X, y, sample_weight=w)
    assert_matches(est, sk, "device='gpu' + sample_weight")


def test_device_arm_store_cv_results_values_match_sklearn():
    """`cv_results_` is the one output the sweep writes per ROW, so it is the
    one place a device indexing bug would show up without moving `alpha_`."""
    _device_arm_or_skip()
    X, y = regression()
    est = mlrs.RidgeCV(alphas=ALPHAS, store_cv_results=True, device="gpu").fit(X, y)
    sk = SkRidgeCV(alphas=ALPHAS, store_cv_results=True).fit(X, y)
    got = np.asarray(est.cv_results_, dtype=np.float64)
    ref = np.asarray(sk.cv_results_, dtype=np.float64)
    assert got.shape == ref.shape, f"cv_results_ shape {got.shape} != {ref.shape}"
    scale = max(1.0, float(np.abs(ref).max()))
    assert np.allclose(got, ref, atol=live_atol() * scale, rtol=0.0), (
        f"device cv_results_ differs (max {np.abs(got - ref).max():.3e})"
    )
