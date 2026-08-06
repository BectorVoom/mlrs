"""NearestNeighbors full-parameter surface through the Python shim (NEIGH-PARAMS).

Mirrors ``test_oracle_knn_regressor_params.py`` with the ``weights`` axis
dropped (``NearestNeighbors`` has no vote/mean to weight) and
``radius_neighbors``/``radius_neighbors_graph`` coverage added. Three kinds of
check live here:

* **Oracle replay** — the committed ``nn_params_*.npz`` / ``nn_radius_*.npz``
  fixtures (``scripts/gen_oracle.py::gen_nearest_neighbors_params`` /
  ``gen_nearest_neighbors_radius``) are replayed through the full binding path
  for every ``metric`` the DEVICE serves, every STRING spelling of ``metric``
  and ``algorithm``, and ``radius_neighbors`` under every metric. No
  regeneration — a second consumer of the same blobs the Rust tests read.

* **Live-sklearn comparison** — the parameters whose implementation is
  host-side Python (``metric=<callable>``, ``kneighbors``/``radius_neighbors``
  with ``X=None``, the ``_graph`` siblings,
  ``effective_metric_``/``effective_metric_params_``) are compared against a
  live ``sklearn`` instance instead.

* **Pass-through / validation** — ``leaf_size``/``n_jobs``/``radius`` accept
  and validate correctly, and every invalid combination raises at ``fit``.

f64 fixtures are skipped-with-reason on an f64-incapable backend (rocm) via the
``conftest.requires_f64`` marker.
"""

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors as SkNN

import mlrs
from conftest import dtype_of, fixture_path, requires_f64

PARAM_FIXTURES = ["nn_params_f32_seed42", "nn_params_f64_seed42"]
RADIUS_FIXTURES = ["nn_radius_f32_seed42", "nn_radius_f64_seed42"]

# (metric kwargs for both libraries, fixture-key metric name)
METRICS = [
    ({"metric": "euclidean"}, "euclidean"),
    ({"metric": "manhattan"}, "manhattan"),
    ({"metric": "chebyshev"}, "chebyshev"),
    ({"metric": "minkowski", "p": 3.0}, "minkowski"),
    ({"metric": "cosine"}, "cosine"),
]

# Every STRING the `metric` parameter accepts (see
# `test_oracle_knn_regressor_params.py::METRIC_STRINGS` for the identical
# rationale: five distance FUNCTIONS, nine spellings).
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

# Every STRING the `algorithm` parameter accepts.
ALGORITHM_STRINGS = ["auto", "brute", "kd_tree", "ball_tree"]


def _atol(fixture):
    return 1e-5 if dtype_of(fixture) == np.float64 else 1e-4


def _live_data(seed=7, n_train=45, n_query=10, d=3):
    """A live (non-fixture) design for the sklearn-comparison tests.

    Shifted off the origin so the cosine cases are well conditioned, and with
    one query row copied from a training row so ``kneighbors``/
    ``radius_neighbors`` self-match / zero-distance edges are reachable here
    too.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_train, d)) * 2.0 + 5.0
    xq = rng.standard_normal((n_query, d)) * 2.0 + 5.0
    xq[1] = x[4]
    return x, xq


# ---------------------------------------------------------------------------
# Oracle replay: kneighbors metric matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("metric_kwargs,metric_name", METRICS)
@requires_f64
def test_metric_matrix_oracle(fixture, metric_kwargs, metric_name):
    """Every device-served ``metric`` matches sklearn's ``kneighbors``."""
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    nn = mlrs.NearestNeighbors(n_neighbors=k, **metric_kwargs).fit(d["X"])
    dist, idx = nn.kneighbors(d["Xq"])
    assert np.allclose(
        np.asarray(dist, dtype=np.float64),
        np.asarray(d[f"distances_{metric_name}"], dtype=np.float64),
        atol=_atol(fixture),
        rtol=0.0,
    )
    assert np.array_equal(
        np.asarray(idx).astype(np.int64),
        d[f"indices_{metric_name}"].astype(np.int64),
    )


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("metric", METRIC_STRINGS)
@requires_f64
def test_every_metric_string_oracle(fixture, metric):
    """Every STRING ``metric`` accepts matches sklearn under THAT SAME string.

    Left at the default ``algorithm='auto'`` deliberately — the only value
    that accepts all nine, and what the fixture was generated under.
    """
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    nn = mlrs.NearestNeighbors(n_neighbors=k, metric=metric).fit(d["X"])
    dist, idx = nn.kneighbors(d["Xq"])
    assert np.allclose(
        np.asarray(dist, dtype=np.float64),
        np.asarray(d[f"alias_distances_{metric}"], dtype=np.float64),
        atol=_atol(fixture),
        rtol=0.0,
    )
    assert np.array_equal(
        np.asarray(idx).astype(np.int64),
        d[f"alias_indices_{metric}"].astype(np.int64),
    )


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("algorithm", ALGORITHM_STRINGS)
@requires_f64
def test_every_algorithm_string_oracle(fixture, algorithm):
    """mlrs's brute-force answer matches sklearn's under EVERY ``algorithm``.

    ``alg_kd_tree_*`` / ``alg_ball_tree_*`` are sklearn's genuine TREE
    answers — mlrs always runs brute force, so this is the check that
    resolving every strategy to brute force is a genuine equivalence.
    """
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    nn = mlrs.NearestNeighbors(n_neighbors=k, algorithm=algorithm).fit(d["X"])
    dist, idx = nn.kneighbors(d["Xq"])
    assert np.allclose(
        np.asarray(dist, dtype=np.float64),
        np.asarray(d[f"alg_distances_{algorithm}"], dtype=np.float64),
        atol=_atol(fixture),
        rtol=0.0,
    )
    assert np.array_equal(
        np.asarray(idx).astype(np.int64),
        d[f"alg_indices_{algorithm}"].astype(np.int64),
    )


@pytest.mark.parametrize("metric", METRIC_STRINGS)
@pytest.mark.parametrize("algorithm", ALGORITHM_STRINGS)
def test_metric_algorithm_validity_matches_sklearn(metric, algorithm):
    """mlrs accepts a ``(metric, algorithm)`` pair EXACTLY when sklearn does."""
    x, _ = _live_data()

    def sk_raises():
        try:
            SkNN(n_neighbors=3, metric=metric, algorithm=algorithm).fit(x)
            return False
        except ValueError:
            return True

    def mlrs_raises():
        try:
            mlrs.NearestNeighbors(
                n_neighbors=3, metric=metric, algorithm=algorithm
            ).fit(x)
            return False
        except ValueError:
            return True

    assert mlrs_raises() == sk_raises()


# ---------------------------------------------------------------------------
# Oracle replay: radius_neighbors metric matrix (NEIGH-RADIUS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", RADIUS_FIXTURES)
@pytest.mark.parametrize("metric_kwargs,metric_name", METRICS)
@requires_f64
def test_radius_neighbors_metric_matrix_oracle(fixture, metric_kwargs, metric_name):
    """Every device-served ``metric`` matches sklearn's ``radius_neighbors``.

    Compared row by row against the FLAT, ascending-train-index-order fixture
    (``gen_nearest_neighbors_radius``): both mlrs and sklearn's brute-force
    path scan candidates in that same order and keep the ones within
    ``radius``, so — unlike ``kneighbors``' top-k tie-break — there is no
    ordering ambiguity, and the per-row slices compare EXACTLY.
    """
    d = np.load(fixture_path(fixture))
    radius = float(d[f"radius_{metric_name}"][0])
    counts = d[f"radius_counts_{metric_name}"].astype(np.int64)
    flat_dist = np.asarray(d[f"radius_distances_{metric_name}"], dtype=np.float64)
    flat_idx = d[f"radius_indices_{metric_name}"].astype(np.int64)

    nn = mlrs.NearestNeighbors(**metric_kwargs).fit(d["X"])
    dist, idx = nn.radius_neighbors(d["Xq"], radius=radius)

    offset = 0
    for q in range(len(dist)):
        want_n = int(counts[q])
        assert len(idx[q]) == want_n, f"row {q}: match count mismatch"
        assert np.array_equal(
            np.asarray(idx[q]).astype(np.int64), flat_idx[offset : offset + want_n]
        )
        assert np.allclose(
            np.asarray(dist[q], dtype=np.float64),
            flat_dist[offset : offset + want_n],
            atol=_atol(fixture),
            rtol=0.0,
        )
        offset += want_n


@pytest.mark.parametrize("fixture", RADIUS_FIXTURES)
@requires_f64
def test_radius_neighbors_graph_oracle(fixture):
    """``radius_neighbors_graph(mode='distance')`` matches the flat oracle."""
    d = np.load(fixture_path(fixture))
    radius = float(d["radius_euclidean"][0])
    nn = mlrs.NearestNeighbors(metric="euclidean").fit(d["X"])
    got = nn.radius_neighbors_graph(d["Xq"], radius=radius, mode="distance")

    counts = d["radius_counts_euclidean"].astype(np.int64)
    flat_dist = np.asarray(d["radius_distances_euclidean"], dtype=np.float64)
    flat_idx = d["radius_indices_euclidean"].astype(np.int64)
    dense = np.zeros(got.shape, dtype=np.float64)
    offset = 0
    for q, n in enumerate(counts):
        dense[q, flat_idx[offset : offset + n]] = flat_dist[offset : offset + n]
        offset += int(n)
    assert np.allclose(got.toarray(), dense, atol=_atol(fixture), rtol=0.0)


# ---------------------------------------------------------------------------
# Live sklearn comparison: host-side parameters and radius_neighbors semantics
# ---------------------------------------------------------------------------


def test_callable_metric_matches_sklearn():
    """``metric=<callable>`` runs the whole pairwise pass host-side, for both
    ``kneighbors`` and ``radius_neighbors``."""
    x, xq = _live_data()

    def m(a, b):
        return float(np.sum(np.abs(a - b) ** 1.5))

    got_d, got_i = mlrs.NearestNeighbors(metric=m).fit(x).kneighbors(xq)
    want_d, want_i = SkNN(algorithm="brute", metric=m).fit(x).kneighbors(xq)
    assert np.allclose(np.asarray(got_d, dtype=np.float64), want_d, atol=1e-5, rtol=0.0)
    assert np.array_equal(np.asarray(got_i).astype(np.int64), want_i.astype(np.int64))

    radius = float(np.median(want_d))
    got_rd, got_ri = mlrs.NearestNeighbors(metric=m).fit(x).radius_neighbors(xq, radius=radius)
    want_rd, want_ri = SkNN(algorithm="brute", metric=m).fit(x).radius_neighbors(xq, radius=radius)
    for q in range(len(got_ri)):
        assert set(np.asarray(got_ri[q]).tolist()) == set(want_ri[q].tolist())

    assert mlrs.NearestNeighbors(metric=m).fit(x).effective_metric_ is m


def test_kneighbors_self_query_matches_sklearn():
    """``kneighbors(X=None)`` drops each point's own self-match, as sklearn does."""
    x, _ = _live_data()
    dist, idx = mlrs.NearestNeighbors().fit(x).kneighbors()
    want_d, want_i = SkNN(algorithm="brute").fit(x).kneighbors()
    assert np.allclose(np.asarray(dist, dtype=np.float64), want_d, atol=1e-5, rtol=0.0)
    assert np.array_equal(np.asarray(idx).astype(np.int64), want_i.astype(np.int64))
    assert not np.any(np.asarray(idx) == np.arange(x.shape[0])[:, None])


def test_radius_neighbors_self_query_matches_sklearn():
    """``radius_neighbors(X=None)`` drops each point's own self-match."""
    x, _ = _live_data()
    radius = 3.0
    dist, idx = mlrs.NearestNeighbors().fit(x).radius_neighbors(radius=radius)
    want_d, want_i = SkNN(algorithm="brute").fit(x).radius_neighbors(radius=radius)
    for q in range(x.shape[0]):
        assert set(np.asarray(idx[q]).tolist()) == set(want_i[q].tolist())
        assert q not in np.asarray(idx[q]).tolist()


@pytest.mark.parametrize("mode", ["connectivity", "distance"])
@pytest.mark.parametrize("self_query", [False, True])
def test_kneighbors_graph_matches_sklearn(mode, self_query):
    x, xq = _live_data()
    query = None if self_query else xq
    got = mlrs.NearestNeighbors().fit(x).kneighbors_graph(query, mode=mode)
    want = SkNN(algorithm="brute").fit(x).kneighbors_graph(query, mode=mode)
    assert got.shape == want.shape
    assert np.allclose(got.toarray(), want.toarray(), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("mode", ["connectivity", "distance"])
@pytest.mark.parametrize("self_query", [False, True])
def test_radius_neighbors_graph_matches_sklearn_live(mode, self_query):
    x, xq = _live_data()
    query = None if self_query else xq
    radius = 3.0
    got = mlrs.NearestNeighbors().fit(x).radius_neighbors_graph(query, radius=radius, mode=mode)
    want = SkNN(algorithm="brute").fit(x).radius_neighbors_graph(query, radius=radius, mode=mode)
    assert got.shape == want.shape
    assert np.allclose(got.toarray(), want.toarray(), atol=1e-5, rtol=0.0)


def test_radius_neighbors_sort_results():
    x, xq = _live_data()
    radius = 3.0
    dist, idx = (
        mlrs.NearestNeighbors().fit(x).radius_neighbors(xq, radius=radius, sort_results=True)
    )
    for row in dist:
        row = np.asarray(row, dtype=np.float64)
        assert np.all(np.diff(row) >= 0.0)


def test_radius_neighbors_sort_results_requires_return_distance():
    x, _ = _live_data()
    nn = mlrs.NearestNeighbors().fit(x)
    with pytest.raises(ValueError, match="return_distance"):
        nn.radius_neighbors(radius=1.0, return_distance=False, sort_results=True)


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
    x, _ = _live_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        got = mlrs.NearestNeighbors(**kwargs).fit(x)
        want = SkNN(algorithm="brute", **kwargs).fit(x)
    assert got.effective_metric_ == want.effective_metric_
    assert got.effective_metric_params_ == want.effective_metric_params_
    assert got.n_samples_fit_ == want.n_samples_fit_
    assert got.n_features_in_ == want.n_features_in_


def test_metric_params_p_overrides_init_p_with_warning():
    x, xq = _live_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est = mlrs.NearestNeighbors(p=2, metric_params={"p": 1}).fit(x)
    assert any(issubclass(w.category, SyntaxWarning) for w in caught)
    assert est.effective_metric_ == "manhattan"
    want_d, _ = SkNN(algorithm="brute", metric="manhattan").fit(x).kneighbors(xq)
    got_d, _ = est.kneighbors(xq)
    assert np.allclose(np.asarray(got_d, dtype=np.float64), want_d, atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# Pass-through parameters and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["auto", "brute", "kd_tree", "ball_tree"])
def test_every_algorithm_accepted_and_equivalent(algorithm):
    x, xq = _live_data()
    got_d, got_i = mlrs.NearestNeighbors(algorithm=algorithm).fit(x).kneighbors(xq)
    want_d, want_i = SkNN(algorithm=algorithm).fit(x).kneighbors(xq)
    assert np.allclose(np.asarray(got_d, dtype=np.float64), want_d, atol=1e-5, rtol=0.0)
    assert np.array_equal(np.asarray(got_i).astype(np.int64), want_i.astype(np.int64))


@pytest.mark.parametrize("leaf_size", [1, 30, 200])
@pytest.mark.parametrize("n_jobs", [None, 1, -1])
def test_leaf_size_and_n_jobs_are_accepted_and_inert(leaf_size, n_jobs):
    x, xq = _live_data()
    est = mlrs.NearestNeighbors(leaf_size=leaf_size, n_jobs=n_jobs)
    assert est.get_params()["leaf_size"] == leaf_size
    assert est.get_params()["n_jobs"] == n_jobs
    got_d, got_i = est.fit(x).kneighbors(xq)
    want_d, want_i = mlrs.NearestNeighbors().fit(x).kneighbors(xq)
    assert np.array_equal(np.asarray(got_d), np.asarray(want_d))
    assert np.array_equal(np.asarray(got_i), np.asarray(want_i))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"algorithm": "invalid"}, "Algorithm is not supported"),
        ({"algorithm": "kd_tree", "metric": "cosine"}, "not valid"),
        ({"algorithm": "ball_tree", "metric": "cosine"}, "not valid"),
        ({"algorithm": "brute", "metric": "infinity"}, "not valid"),
        ({"metric": "nope"}, "Metric is not supported"),
        ({"p": 0.5}, "p must be greater or equal to one"),
        ({"leaf_size": 0}, "leaf_size == 0"),
        ({"n_neighbors": 0}, "n_neighbors == 0"),
        ({"radius": -1.0}, "radius must be non-negative"),
        ({"metric_params": {"w": [1, 1, 1]}}, "weighted minkowski"),
    ],
)
def test_invalid_parameters_rejected_at_fit(kwargs, message):
    x, _ = _live_data()
    est = mlrs.NearestNeighbors(**kwargs)  # must not raise
    with pytest.raises(ValueError, match=message):
        est.fit(x)


# ---------------------------------------------------------------------------
# Deferred upload: fit must stay observationally identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_fit_still_rejects_non_finite_input(bad_value):
    x, _ = _live_data()
    x = x.copy()
    x[3, 1] = bad_value
    with pytest.raises(ValueError, match="infinity|NaN|too large"):
        mlrs.NearestNeighbors().fit(x)


def test_fitted_attributes_available_without_any_query():
    x, _ = _live_data()
    est = mlrs.NearestNeighbors().fit(x)
    assert est.n_samples_fit_ == x.shape[0]
    assert est.n_features_in_ == x.shape[1]
    assert est.effective_metric_ == "euclidean"
    assert est.effective_metric_params_ == {}
    assert est._fit_X.shape == x.shape
    assert np.allclose(est._fit_X, x, atol=1e-6, rtol=0.0)


def test_repeated_queries_and_refit_are_stable():
    x, xq = _live_data()
    est = mlrs.NearestNeighbors()
    est.fit(x)
    first = np.asarray(est.kneighbors(xq)[1])
    second = np.asarray(est.kneighbors(xq)[1])
    assert np.array_equal(first, second)

    est.fit(x[::-1].copy())
    assert not np.array_equal(first, np.asarray(est.kneighbors(xq)[1]))


def test_predict_before_fit_still_raises_not_fitted():
    from sklearn.exceptions import NotFittedError

    _, xq = _live_data()
    with pytest.raises(NotFittedError):
        mlrs.NearestNeighbors().kneighbors(xq)
    with pytest.raises(NotFittedError):
        mlrs.NearestNeighbors().radius_neighbors(xq, radius=1.0)


def test_clone_round_trips_the_full_parameter_set():
    x, xq = _live_data()
    est = mlrs.NearestNeighbors(
        n_neighbors=3,
        radius=2.5,
        metric="manhattan",
        leaf_size=7,
        n_jobs=2,
    )
    assert clone(est).get_params() == est.get_params()

    first_d, first_i = est.fit(x).kneighbors(xq)
    refit_d, refit_i = est.fit(x).kneighbors(xq)
    cloned_d, cloned_i = clone(est).fit(x).kneighbors(xq)
    want_d, want_i = SkNN(algorithm="brute", n_neighbors=3, metric="manhattan").fit(x).kneighbors(xq)
    for got_d, got_i in [(first_d, first_i), (refit_d, refit_i), (cloned_d, cloned_i)]:
        assert np.allclose(np.asarray(got_d, dtype=np.float64), want_d, atol=1e-5, rtol=0.0)
        assert np.array_equal(np.asarray(got_i).astype(np.int64), want_i.astype(np.int64))
