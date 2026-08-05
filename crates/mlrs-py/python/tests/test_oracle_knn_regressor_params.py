"""KNeighborsRegressor full-parameter surface through the Python shim.

Two kinds of check live here, and the split is deliberate:

* **Oracle replay** — the committed ``knn_reg_params_*.npz`` fixtures
  (``scripts/gen_oracle.py::gen_knn_regressor_params``) are replayed through the
  full binding path for every ``weights`` x ``metric`` combination the DEVICE
  serves, plus multi-output targets and ``kneighbors`` under a non-default
  metric. No regeneration — this is a second consumer of the same blobs the
  Rust tests read.

* **Live-sklearn comparison** — the parameters whose implementation is host-side
  Python (``weights=<callable>``, ``metric=<callable>``, ``kneighbors(X=None)``,
  ``kneighbors_graph``, ``effective_metric_``/``effective_metric_params_``) are
  compared against a live ``sklearn`` instance instead. A committed fixture
  would pin numpy against numpy for those; comparing against sklearn is the
  check that can actually fail, and it is what keeps this shim honest as sklearn
  evolves its own resolution rules.

f64 fixtures are skipped-with-reason on an f64-incapable backend (rocm) via the
``conftest.requires_f64`` marker.
"""

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor as SkKNR

import mlrs
from conftest import dtype_of, fixture_path, requires_f64

PARAM_FIXTURES = ["knn_reg_params_f32_seed42", "knn_reg_params_f64_seed42"]

# (metric kwargs for both libraries, fixture-key metric name)
METRICS = [
    ({"metric": "euclidean"}, "euclidean"),
    ({"metric": "manhattan"}, "manhattan"),
    ({"metric": "chebyshev"}, "chebyshev"),
    ({"metric": "minkowski", "p": 3.0}, "minkowski"),
    ({"metric": "cosine"}, "cosine"),
]

WEIGHTS = ["uniform", "distance"]

# Every STRING the `metric` parameter accepts. Five distance FUNCTIONS, nine
# spellings — the fixture stores each one separately (`alias_<metric>_<weights>`,
# generated under `algorithm='auto'`), so `metric='l1'` is gated against
# sklearn-under-`'l1'` rather than against the assumption that mlrs folds it onto
# `manhattan` correctly.
METRIC_STRINGS = [
    "minkowski",
    "euclidean",
    "l2",
    "manhattan",
    "l1",
    "cityblock",
    "chebyshev",
    "infinity",
    "cosine",
]

# Every STRING the `algorithm` parameter accepts, and the metric-set restriction
# each one carries. mlrs runs brute force for all four; sklearn genuinely builds
# a tree for two, and the fixture's `alg_<algorithm>_<weights>` arrays are those
# tree answers.
ALGORITHM_STRINGS = ["auto", "brute", "kd_tree", "ball_tree"]


def _atol(fixture):
    return 1e-5 if dtype_of(fixture) == np.float64 else 1e-4


def _live_data(seed=7, n_train=45, n_query=10, d=3):
    """A live (non-fixture) design for the sklearn-comparison tests.

    Shifted off the origin so the cosine cases are well conditioned, and with
    one query row copied from a training row so the ``weights='distance'``
    ``1/0`` branch is reachable here too — a callable-weights path that never
    sees a zero distance would not exercise the same code sklearn's does.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_train, d)) * 2.0 + 5.0
    xq = rng.standard_normal((n_query, d)) * 2.0 + 5.0
    xq[1] = x[4]
    y = x @ rng.standard_normal(d) + 0.5
    y_multi = np.column_stack([y, x @ rng.standard_normal(d) - 1.0])
    return x, xq, y, y_multi


# ---------------------------------------------------------------------------
# Oracle replay: weights x metric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("metric_kwargs,metric_name", METRICS)
@requires_f64
def test_weights_metric_matrix_oracle(fixture, weights, metric_kwargs, metric_name):
    """Every device-served ``weights`` x ``metric`` pair matches sklearn."""
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    reg = mlrs.KNeighborsRegressor(
        n_neighbors=k, weights=weights, **metric_kwargs
    ).fit(d["X"], d["y"])
    pred = np.asarray(reg.predict(d["Xq"]), dtype=np.float64).ravel()
    assert np.allclose(
        pred,
        np.asarray(d[f"predict_{metric_name}_{weights}"], dtype=np.float64).ravel(),
        atol=_atol(fixture),
        rtol=0.0,
    )


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("weights", WEIGHTS)
@requires_f64
def test_multi_output_oracle(fixture, weights):
    """A 2-D target predicts a 2-D result, column for column."""
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    reg = mlrs.KNeighborsRegressor(n_neighbors=k, weights=weights).fit(
        d["X"], d["y_multi"]
    )
    pred = np.asarray(reg.predict(d["Xq"]), dtype=np.float64)
    expected = np.asarray(d[f"predict_multi_{weights}"], dtype=np.float64)
    # SHAPE first: a flattened (n_query * n_outputs,) result would still
    # `allclose` against a raveled reference, so the shape is the assert that
    # catches a lost second dimension.
    assert pred.shape == expected.shape
    assert np.allclose(pred, expected, atol=_atol(fixture), rtol=0.0)


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("metric", METRIC_STRINGS)
@requires_f64
def test_every_metric_string_oracle(fixture, weights, metric):
    """Every STRING ``metric`` accepts matches sklearn under THAT SAME string.

    The `weights` x `metric` matrix above covers the five distance functions;
    this covers the nine spellings, which is a different claim. An alias is
    resolved by ``_resolve_metric`` before anything numeric happens, so a wrong
    fold (``'l1'`` -> Euclidean, say) would produce a perfectly self-consistent
    wrong answer that only a per-STRING oracle catches.

    Left at the default ``algorithm='auto'`` deliberately: it is the only value
    that accepts all nine, and it is what the fixture was generated under.
    """
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    reg = mlrs.KNeighborsRegressor(
        n_neighbors=k, weights=weights, metric=metric
    ).fit(d["X"], d["y"])
    pred = np.asarray(reg.predict(d["Xq"]), dtype=np.float64).ravel()
    assert np.allclose(
        pred,
        np.asarray(d[f"alias_{metric}_{weights}"], dtype=np.float64).ravel(),
        atol=_atol(fixture),
        rtol=0.0,
    )


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("algorithm", ALGORITHM_STRINGS)
@requires_f64
def test_every_algorithm_string_oracle(fixture, weights, algorithm):
    """mlrs's brute-force answer matches sklearn's under EVERY ``algorithm``.

    ``alg_kd_tree_*`` and ``alg_ball_tree_*`` are sklearn's TREE predictions, so
    this is the check that mlrs resolving every strategy to brute force is a
    genuine equivalence and not just an internally consistent shortcut.

    The design contains one duplicated training pair, which two search
    strategies may order differently — but both copies carry the same target
    (the fixture derives ``y`` from ``X`` after duplicating the row), so the tie
    cannot move a prediction either way.
    """
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    reg = mlrs.KNeighborsRegressor(
        n_neighbors=k, weights=weights, algorithm=algorithm
    ).fit(d["X"], d["y"])
    pred = np.asarray(reg.predict(d["Xq"]), dtype=np.float64).ravel()
    assert np.allclose(
        pred,
        np.asarray(d[f"alg_{algorithm}_{weights}"], dtype=np.float64).ravel(),
        atol=_atol(fixture),
        rtol=0.0,
    )


@pytest.mark.parametrize("metric", METRIC_STRINGS)
@pytest.mark.parametrize("algorithm", ALGORITHM_STRINGS)
def test_metric_algorithm_validity_matches_sklearn(metric, algorithm):
    """mlrs accepts a ``(metric, algorithm)`` pair EXACTLY when sklearn does.

    The two exclusions are not symmetric and neither is about what can be
    computed — mlrs runs brute force for every algorithm and could evaluate all
    nine metrics in every case:

      * ``cosine`` is brute-only (no tree can index it);
      * ``infinity`` is tree-only (it is a tree spelling of Chebyshev that
        sklearn's ``pairwise_distances`` has never known).

    Accepting a pair sklearn rejects would let a script succeed here and fail
    there, which is worse than the restriction.
    """
    x, _, y, _ = _live_data()

    def sk_raises():
        try:
            SkKNR(n_neighbors=3, metric=metric, algorithm=algorithm).fit(x, y)
            return False
        except ValueError:
            return True

    def mlrs_raises():
        try:
            mlrs.KNeighborsRegressor(
                n_neighbors=3, metric=metric, algorithm=algorithm
            ).fit(x, y)
            return False
        except ValueError:
            return True

    assert mlrs_raises() == sk_raises()


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@requires_f64
def test_kneighbors_non_default_metric_oracle(fixture):
    """``kneighbors`` reports the CONFIGURED metric's neighbours.

    Checked in its own test rather than being inferred from ``predict``: a
    wrong-metric neighbour set can still average to a nearly-right prediction on
    a smooth target, so the distances and indices are what actually pin it.

    Indices are compared as a per-row SET, not position by position. The fixture
    contains a duplicated training row (deliberately — it is what gives a
    coincident query two zero-distance neighbours), so those two indices are an
    exact distance TIE and their relative order is not determined by the
    problem. mlrs's ``top_k`` resolves ties to the lowest index; sklearn's brute
    path makes no such guarantee and here returns the other order. Asserting
    positional equality would be asserting an implementation detail of sklearn's
    sort. The distances, which ARE fully determined, are still compared
    elementwise above.
    """
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    reg = mlrs.KNeighborsRegressor(n_neighbors=k, metric="manhattan").fit(
        d["X"], d["y"]
    )
    dist, idx = reg.kneighbors(d["Xq"])
    assert np.allclose(
        np.asarray(dist, dtype=np.float64),
        np.asarray(d["distances_manhattan"], dtype=np.float64),
        atol=_atol(fixture),
        rtol=0.0,
    )
    got = np.sort(np.asarray(idx).astype(np.int64), axis=1)
    want = np.sort(d["indices_manhattan"].astype(np.int64), axis=1)
    assert np.array_equal(got, want)


# ---------------------------------------------------------------------------
# Live sklearn comparison: the host-side parameters
# ---------------------------------------------------------------------------


def test_callable_weights_matches_sklearn():
    """``weights=<callable>`` is applied to the same neighbour set sklearn uses.

    Both single- and multi-output, because the callable path builds the weighted
    mean itself and the two shapes broadcast differently.
    """
    x, xq, y, y_multi = _live_data()

    def w(dist):
        return 1.0 / (1.0 + dist)

    for target in (y, y_multi):
        got = np.asarray(
            mlrs.KNeighborsRegressor(weights=w).fit(x, target).predict(xq)
        )
        want = SkKNR(algorithm="brute", weights=w).fit(x, target).predict(xq)
        assert got.shape == want.shape
        assert np.allclose(got, want, atol=1e-5, rtol=0.0)


def test_callable_metric_matches_sklearn():
    """``metric=<callable>`` runs the whole pairwise pass host-side."""
    x, xq, y, _ = _live_data()

    def m(a, b):
        return float(np.sum(np.abs(a - b) ** 1.5))

    got = np.asarray(mlrs.KNeighborsRegressor(metric=m).fit(x, y).predict(xq))
    want = SkKNR(algorithm="brute", metric=m).fit(x, y).predict(xq)
    assert np.allclose(got, want, atol=1e-5, rtol=0.0)

    # `effective_metric_` is the callable itself, as sklearn reports it.
    assert mlrs.KNeighborsRegressor(metric=m).fit(x, y).effective_metric_ is m


def test_callable_weights_agrees_with_builtin_formula():
    """A callable spelling of ``'distance'`` must reproduce the device answer.

    This is the cross-check between the two implementations of the SAME rule:
    the device kernel's ``1/d`` weighting and the host ``_get_weights`` copy.
    The zero-distance query in ``_live_data`` makes both take their degenerate
    branch, which is the half that can silently produce NaN.
    """
    x, xq, y, _ = _live_data()

    def as_distance(dist):
        with np.errstate(divide="ignore"):
            w = 1.0 / dist
        inf_mask = np.isinf(w)
        inf_row = np.any(inf_mask, axis=1)
        w[inf_row] = inf_mask[inf_row]
        return w

    device = np.asarray(
        mlrs.KNeighborsRegressor(weights="distance").fit(x, y).predict(xq)
    )
    host = np.asarray(
        mlrs.KNeighborsRegressor(weights=as_distance).fit(x, y).predict(xq)
    )
    assert np.isfinite(device).all()
    assert np.allclose(device, host, atol=1e-5, rtol=0.0)


def test_kneighbors_self_query_matches_sklearn():
    """``kneighbors(X=None)`` drops each point's own self-match, as sklearn does."""
    x, _, y, _ = _live_data()
    dist, idx = mlrs.KNeighborsRegressor().fit(x, y).kneighbors()
    want_d, want_i = SkKNR(algorithm="brute").fit(x, y).kneighbors()
    assert np.allclose(np.asarray(dist, dtype=np.float64), want_d, atol=1e-5, rtol=0.0)
    assert np.array_equal(np.asarray(idx).astype(np.int64), want_i.astype(np.int64))
    # No row may keep itself as a neighbour.
    assert not np.any(np.asarray(idx) == np.arange(x.shape[0])[:, None])


@pytest.mark.parametrize("mode", ["connectivity", "distance"])
@pytest.mark.parametrize("self_query", [False, True])
def test_kneighbors_graph_matches_sklearn(mode, self_query):
    """The sparse neighbour graph matches sklearn in both modes."""
    x, xq, y, _ = _live_data()
    query = None if self_query else xq
    got = mlrs.KNeighborsRegressor().fit(x, y).kneighbors_graph(query, mode=mode)
    want = SkKNR(algorithm="brute").fit(x, y).kneighbors_graph(query, mode=mode)
    assert got.shape == want.shape
    assert np.allclose(got.toarray(), want.toarray(), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"p": 1},
        {"p": float("inf")},
        {"p": 3},
        {"metric": "cosine"},
        {"metric": "manhattan"},
        {"metric": "l2"},
        {"metric_params": {"p": 3}},
    ],
)
def test_effective_metric_resolution_matches_sklearn(kwargs):
    """``effective_metric_`` / ``effective_metric_params_`` match sklearn exactly.

    Including the awkward parts: the ``minkowski`` -> ``euclidean`` /
    ``manhattan`` / ``chebyshev`` collapse REMOVES ``p`` from the params, while
    the non-collapsed case carries BOTH ``p`` and ``w``. A dict holding only
    ``p`` is not the dict sklearn produces.
    """
    x, _, y, _ = _live_data()
    with warnings.catch_warnings():
        # The `metric_params={'p': ...}` case warns in BOTH libraries; the
        # warning itself is asserted separately below.
        warnings.simplefilter("ignore", SyntaxWarning)
        got = mlrs.KNeighborsRegressor(**kwargs).fit(x, y)
        want = SkKNR(algorithm="brute", **kwargs).fit(x, y)
    assert got.effective_metric_ == want.effective_metric_
    assert got.effective_metric_params_ == want.effective_metric_params_
    assert got.n_samples_fit_ == want.n_samples_fit_
    assert got.n_features_in_ == want.n_features_in_


def test_metric_params_p_overrides_init_p_with_warning():
    """A ``p`` in ``metric_params`` wins over the ``__init__`` one, and warns."""
    x, xq, y, _ = _live_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est = mlrs.KNeighborsRegressor(p=2, metric_params={"p": 1}).fit(x, y)
    assert any(issubclass(w.category, SyntaxWarning) for w in caught)
    # p=1 wins -> manhattan, NOT the p=2 euclidean from __init__.
    assert est.effective_metric_ == "manhattan"
    want = SkKNR(algorithm="brute", metric="manhattan").fit(x, y).predict(xq)
    assert np.allclose(np.asarray(est.predict(xq)), want, atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# Pass-through parameters and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["auto", "brute", "kd_tree", "ball_tree"])
def test_every_algorithm_accepted_and_equivalent(algorithm):
    """All four ``algorithm`` values are accepted and give the SAME answer.

    ``algorithm`` selects how the neighbours are found, never which ones they
    are, so a grid search over it must complete with identical predictions —
    which is exactly what sklearn does.
    """
    x, xq, y, _ = _live_data()
    got = np.asarray(
        mlrs.KNeighborsRegressor(algorithm=algorithm).fit(x, y).predict(xq)
    )
    want = SkKNR(algorithm=algorithm).fit(x, y).predict(xq)
    assert np.allclose(got, want, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("leaf_size", [1, 30, 200])
@pytest.mark.parametrize("n_jobs", [None, 1, -1])
def test_leaf_size_and_n_jobs_are_accepted_and_inert(leaf_size, n_jobs):
    """``leaf_size`` / ``n_jobs`` round-trip and cannot change the result.

    Both tune machinery mlrs does not have (a tree; a host thread pool). They
    are accepted so sklearn code and grid searches keep working, and this pins
    that accepting them stayed inert rather than quietly perturbing anything.
    """
    x, xq, y, _ = _live_data()
    est = mlrs.KNeighborsRegressor(leaf_size=leaf_size, n_jobs=n_jobs)
    assert est.get_params()["leaf_size"] == leaf_size
    assert est.get_params()["n_jobs"] == n_jobs
    got = np.asarray(est.fit(x, y).predict(xq))
    want = np.asarray(mlrs.KNeighborsRegressor().fit(x, y).predict(xq))
    assert np.allclose(got, want, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"algorithm": "invalid"}, "Algorithm is not supported"),
        ({"algorithm": "kd_tree", "metric": "cosine"}, "not valid"),
        ({"algorithm": "ball_tree", "metric": "cosine"}, "not valid"),
        ({"algorithm": "brute", "metric": "infinity"}, "not valid"),
        ({"metric": "nope"}, "Metric is not supported"),
        ({"weights": "nope"}, "weights not recognized"),
        ({"p": 0.5}, "p must be greater or equal to one"),
        ({"leaf_size": 0}, "leaf_size == 0"),
        ({"n_neighbors": 0}, "n_neighbors == 0"),
        ({"metric_params": {"w": [1, 1, 1]}}, "weighted minkowski"),
    ],
)
def test_invalid_parameters_rejected_at_fit(kwargs, message):
    """Bad values raise ``ValueError`` at ``fit``, never at ``__init__``.

    Deferring to ``fit`` is the sklearn contract (``__init__`` stores arguments
    verbatim so ``get_params``/``clone`` round-trip), so constructing must
    SUCCEED for every case here — that half is asserted too.
    """
    x, _, y, _ = _live_data()
    est = mlrs.KNeighborsRegressor(**kwargs)  # must not raise
    with pytest.raises(ValueError, match=message):
        est.fit(x, y)


# ---------------------------------------------------------------------------
# Deferred upload (KNN-REG-FIT): fit must stay observationally identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_in", ["X", "y"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_fit_still_rejects_non_finite_input(bad_in, bad_value):
    """The NaN/inf rejection fires at ``fit``, not at the first ``predict``.

    ``fit`` defers the device upload, and the finite scan moved from numpy's
    ``check_array`` into the Rust call along with it. If that scan had drifted
    to the query path with the upload, a NaN training set would fit "fine" and
    only blow up later — sklearn raises at ``fit``, so mlrs must too, with the
    same ``ValueError``.
    """
    x, _, y, _ = _live_data()
    x, y = x.copy(), y.copy()
    if bad_in == "X":
        x[3, 1] = bad_value
    else:
        y[3] = bad_value
    with pytest.raises(ValueError, match="infinity|NaN|too large"):
        mlrs.KNeighborsRegressor().fit(x, y)


def test_fitted_attributes_available_without_any_query():
    """A fitted-but-never-queried estimator answers every fitted attribute.

    These are the attributes sklearn code reads between ``fit`` and ``predict``
    (and that ``check_is_fitted`` scans for). They are shape facts, so the
    deferred upload must not be needed to answer them — if any of them forced
    materialization, deferring would buy nothing on the path that matters.
    """
    x, _, y, y_multi = _live_data()
    est = mlrs.KNeighborsRegressor().fit(x, y)
    assert est.n_samples_fit_ == x.shape[0]
    assert est.n_features_in_ == x.shape[1]
    assert est.effective_metric_ == "euclidean"
    assert est.effective_metric_params_ == {}
    assert est._fit_X.shape == x.shape
    assert est._y.shape == y.shape
    assert np.allclose(est._fit_X, x, atol=1e-6, rtol=0.0)

    multi = mlrs.KNeighborsRegressor().fit(x, y_multi)
    assert multi._y.shape == y_multi.shape


def test_repeated_queries_and_refit_are_stable():
    """Materialization is idempotent, and a refit re-materializes.

    Two consecutive ``predict`` calls must agree (the first one uploads, the
    second must reuse rather than re-upload or read stale state), and fitting
    the SAME object on different data must not keep serving the old training
    set — which is what would happen if the deferred data were consumed but the
    already-materialized arm left in place.
    """
    x, xq, y, _ = _live_data()
    est = mlrs.KNeighborsRegressor()
    est.fit(x, y)
    first = np.asarray(est.predict(xq))
    second = np.asarray(est.predict(xq))
    assert np.array_equal(first, second)

    # Refit on a DIFFERENT target; the predictions must follow the new data.
    est.fit(x, y * 3.0 + 1.0)
    refit = np.asarray(est.predict(xq))
    want = SkKNR(algorithm="brute").fit(x, y * 3.0 + 1.0).predict(xq)
    assert np.allclose(refit, want, atol=1e-5, rtol=0.0)
    assert not np.allclose(refit, first, atol=1e-5, rtol=0.0)


def test_predict_before_fit_still_raises_not_fitted():
    """An unfitted estimator raises ``NotFittedError``, not an upload error."""
    from sklearn.exceptions import NotFittedError

    _, xq, _, _ = _live_data()
    with pytest.raises(NotFittedError):
        mlrs.KNeighborsRegressor().predict(xq)


def test_clone_round_trips_the_full_parameter_set():
    """``clone`` preserves every parameter, and the clone refits to the same answer.

    The refit half matters on its own: the wrapper rebuilds the core estimator
    from parameters held on the Python object, and an implementation that read
    them back out of the fitted Rust handle instead would silently fall back to
    the defaults on a second fit — which is what every ``cross_val_score`` /
    ``GridSearchCV`` does.
    """
    x, xq, y, _ = _live_data()
    est = mlrs.KNeighborsRegressor(
        n_neighbors=3,
        weights="distance",
        metric="manhattan",
        leaf_size=7,
        n_jobs=2,
    )
    assert clone(est).get_params() == est.get_params()

    first = np.asarray(est.fit(x, y).predict(xq))
    refit = np.asarray(est.fit(x, y).predict(xq))
    cloned = np.asarray(clone(est).fit(x, y).predict(xq))
    want = SkKNR(
        algorithm="brute", n_neighbors=3, weights="distance", metric="manhattan"
    ).fit(x, y).predict(xq)
    assert np.allclose(first, want, atol=1e-5, rtol=0.0)
    assert np.allclose(refit, want, atol=1e-5, rtol=0.0)
    assert np.allclose(cloned, want, atol=1e-5, rtol=0.0)
