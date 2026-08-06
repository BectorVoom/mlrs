"""KNeighborsClassifier full-parameter surface through the Python shim.

Two kinds of check live here, and the split is deliberate:

* **Oracle replay** — the committed ``knn_clf_params_*.npz`` fixtures
  (``scripts/gen_oracle.py::gen_knn_classifier_params``) are replayed through the
  full binding path for every ``weights`` x ``metric`` combination the DEVICE
  serves, every STRING spelling of ``metric`` (nine) and ``algorithm`` (four),
  multi-output targets, and ``kneighbors`` under a non-default metric. No
  regeneration — this is a second consumer of the same blobs the Rust tests read.

* **Live-sklearn comparison** — the parameters whose implementation is host-side
  Python (``weights=<callable>``, ``metric=<callable>``, the label ENCODING that
  makes string/boolean targets work, ``kneighbors(X=None)``,
  ``kneighbors_graph``, ``effective_metric_``/``effective_metric_params_``) are
  compared against a live ``sklearn`` instance instead. A committed fixture would
  pin numpy against numpy for those; comparing against sklearn is the check that
  can actually fail, and it is what keeps this shim honest as sklearn evolves its
  own resolution rules.

``predict`` and ``predict_proba`` are BOTH asserted everywhere, because
``predict`` is an argmax: a proba matrix that is mis-normalized, scaled by a
constant, or column-swapped can still argmax to the right label on all twelve
fixture queries. Labels are compared EXACTLY (they are identifiers, so a
tolerance would pass a prediction naming a different class); probabilities carry
the 1e-5 bar.

f64 fixtures are skipped-with-reason on an f64-incapable backend (rocm) via the
``conftest.requires_f64`` marker.
"""

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning
from sklearn.neighbors import KNeighborsClassifier as SkKNC

import mlrs
from conftest import dtype_of, fixture_path, requires_f64

PARAM_FIXTURES = ["knn_clf_params_f32_seed42", "knn_clf_params_f64_seed42"]

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


def _labels(x):
    return np.asarray(x).astype(np.int64).ravel()


def _live_data(seed=7, n_train=45, n_query=10, d=3, n_classes=3):
    """A live (non-fixture) design for the sklearn-comparison tests.

    Shifted off the origin so the cosine cases are well conditioned, and with
    one query row copied from a training row so the ``weights='distance'``
    ``1/0`` branch is reachable here too — a callable-weights path that never
    sees a zero distance would not exercise the same code sklearn's does.

    The labels are NON-CONTIGUOUS (`{0, 2, 7}`-style, via `* 2 + 1`) for the same
    reason the fixture's are: with `{0, 1, 2}` the dense column index and the
    class id coincide, and a `predict` that skipped the `classes_` lookup would
    look correct.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_train, d)) * 2.0 + 5.0
    xq = rng.standard_normal((n_query, d)) * 2.0 + 5.0
    xq[1] = x[4]
    lin = x @ rng.standard_normal(d)
    cuts = np.quantile(lin, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.searchsorted(cuts, lin) * 2 + 1
    y_multi = np.column_stack([y, (lin > np.median(lin)).astype(np.int64) * 3])
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
    clf = mlrs.KNeighborsClassifier(
        n_neighbors=k, weights=weights, **metric_kwargs
    ).fit(d["X"], d["y"])

    got = _labels(clf.predict(d["Xq"]))
    assert np.array_equal(got, _labels(d[f"predict_{metric_name}_{weights}"]))

    proba = np.asarray(clf.predict_proba(d["Xq"]), dtype=np.float64)
    want = np.asarray(d[f"proba_{metric_name}_{weights}"], dtype=np.float64)
    assert proba.shape == want.shape
    assert np.allclose(proba, want, atol=_atol(fixture), rtol=0.0)


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
    clf = mlrs.KNeighborsClassifier(
        n_neighbors=k, weights=weights, metric=metric
    ).fit(d["X"], d["y"])

    assert np.array_equal(
        _labels(clf.predict(d["Xq"])), _labels(d[f"alias_{metric}_{weights}"])
    )
    assert np.allclose(
        np.asarray(clf.predict_proba(d["Xq"]), dtype=np.float64),
        np.asarray(d[f"alias_proba_{metric}_{weights}"], dtype=np.float64),
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
    strategies may order differently — but both copies carry the same label (the
    fixture buckets ``y`` from ``X`` after duplicating the row), so the tie
    cannot move a vote either way.
    """
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    clf = mlrs.KNeighborsClassifier(
        n_neighbors=k, weights=weights, algorithm=algorithm
    ).fit(d["X"], d["y"])

    assert np.array_equal(
        _labels(clf.predict(d["Xq"])), _labels(d[f"alg_{algorithm}_{weights}"])
    )
    assert np.allclose(
        np.asarray(clf.predict_proba(d["Xq"]), dtype=np.float64),
        np.asarray(d[f"alg_proba_{algorithm}_{weights}"], dtype=np.float64),
        atol=_atol(fixture),
        rtol=0.0,
    )


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@pytest.mark.parametrize("weights", WEIGHTS)
@requires_f64
def test_multi_output_oracle(fixture, weights):
    """A 2-D target predicts a 2-D result, column for column.

    ``predict_proba`` becomes a LIST of per-column matrices whose widths DIFFER
    here (three classes then two), which is what catches an implementation that
    reused column 0's ``classes_`` for column 1 — that would produce two
    three-wide matrices and fail on shape before any value is compared.
    """
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    clf = mlrs.KNeighborsClassifier(n_neighbors=k, weights=weights).fit(
        d["X"], d["y_multi"]
    )

    pred = np.asarray(clf.predict(d["Xq"])).astype(np.int64)
    want = np.asarray(d[f"predict_multi_{weights}"]).astype(np.int64)
    # SHAPE first: a flattened result would still compare equal against a
    # raveled reference, so the shape is the assert that catches a lost second
    # dimension.
    assert pred.shape == want.shape
    assert np.array_equal(pred, want)

    probabilities = clf.predict_proba(d["Xq"])
    assert isinstance(probabilities, list) and len(probabilities) == 2
    for col, proba in enumerate(probabilities):
        ref = np.asarray(d[f"proba_multi_{weights}_{col}"], dtype=np.float64)
        got = np.asarray(proba, dtype=np.float64)
        assert got.shape == ref.shape
        assert np.allclose(got, ref, atol=_atol(fixture), rtol=0.0)

    # `classes_` is a LIST of per-column arrays, and the two label sets are
    # disjoint in this fixture.
    assert isinstance(clf.classes_, list) and len(clf.classes_) == 2
    assert np.array_equal(clf.classes_[0].astype(np.int64), _labels(d["classes"]))
    assert np.array_equal(clf.classes_[1].astype(np.int64), _labels(d["classes_b"]))


@pytest.mark.parametrize("fixture", PARAM_FIXTURES)
@requires_f64
def test_kneighbors_non_default_metric_oracle(fixture):
    """``kneighbors`` reports the CONFIGURED metric's neighbours.

    Checked in its own test rather than being inferred from ``predict``: a
    wrong-metric neighbour set can still vote for the right class on a
    well-separated design, so the distances are what actually pin it.

    Indices are compared as a per-row SET. The fixture contains a duplicated
    training row (deliberately — it is what gives a coincident query two
    zero-distance neighbours), so those two indices are an exact distance TIE
    whose relative order is not determined by the problem: mlrs's ``top_k``
    resolves it to the lowest index, sklearn's brute path makes no such
    guarantee. The distances, which ARE fully determined, are compared
    elementwise.
    """
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    clf = mlrs.KNeighborsClassifier(n_neighbors=k, metric="manhattan").fit(
        d["X"], d["y"]
    )
    dist, idx = clf.kneighbors(d["Xq"])
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

    Both single- and multi-output, because the callable path builds the vote
    itself and the two shapes broadcast differently.
    """
    x, xq, y, y_multi = _live_data()

    def w(dist):
        return 1.0 / (1.0 + dist)

    for target in (y, y_multi):
        est = mlrs.KNeighborsClassifier(weights=w).fit(x, target)
        want = SkKNC(algorithm="brute", weights=w).fit(x, target)
        got = np.asarray(est.predict(xq))
        assert got.shape == want.predict(xq).shape
        assert np.array_equal(got.astype(np.int64), want.predict(xq).astype(np.int64))

        got_p, want_p = est.predict_proba(xq), want.predict_proba(xq)
        if isinstance(want_p, list):
            assert len(got_p) == len(want_p)
            for a, b in zip(got_p, want_p):
                assert np.allclose(np.asarray(a), b, atol=1e-5, rtol=0.0)
        else:
            assert np.allclose(np.asarray(got_p), want_p, atol=1e-5, rtol=0.0)


def test_callable_metric_matches_sklearn():
    """``metric=<callable>`` runs the whole pairwise pass host-side."""
    x, xq, y, _ = _live_data()

    def m(a, b):
        return float(np.sum(np.abs(a - b) ** 1.5))

    got = mlrs.KNeighborsClassifier(metric=m).fit(x, y)
    want = SkKNC(algorithm="brute", metric=m).fit(x, y)
    assert np.array_equal(
        np.asarray(got.predict(xq)).astype(np.int64),
        want.predict(xq).astype(np.int64),
    )
    assert np.allclose(
        np.asarray(got.predict_proba(xq)), want.predict_proba(xq), atol=1e-5, rtol=0.0
    )
    # `effective_metric_` is the callable itself, as sklearn reports it.
    assert got.effective_metric_ is m


def test_callable_weights_agrees_with_builtin_formula():
    """A callable spelling of ``'distance'`` must reproduce the device answer.

    This is the cross-check between the two implementations of the SAME rule:
    the Rust vote's ``1/d`` weighting and the host ``_get_weights`` copy. The
    zero-distance query in ``_live_data`` makes both take their degenerate
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

    device = mlrs.KNeighborsClassifier(weights="distance").fit(x, y)
    host = mlrs.KNeighborsClassifier(weights=as_distance).fit(x, y)
    dp = np.asarray(device.predict_proba(xq))
    hp = np.asarray(host.predict_proba(xq))
    assert np.isfinite(dp).all()
    assert np.allclose(dp, hp, atol=1e-5, rtol=0.0)
    assert np.array_equal(
        np.asarray(device.predict(xq)).astype(np.int64),
        np.asarray(host.predict(xq)).astype(np.int64),
    )


# ---------------------------------------------------------------------------
# The label space: encoding, dtypes, shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "encode",
    [
        pytest.param(lambda y: np.array(["a", "bb", "ccc"])[y // 2], id="str"),
        pytest.param(lambda y: (y > 1).astype(bool), id="bool"),
        pytest.param(lambda y: (y.astype(np.float64) - 3.0), id="float"),
        pytest.param(lambda y: y.astype(np.int64) * 1000, id="wide_int"),
    ],
)
def test_non_integer_label_spaces_match_sklearn(encode):
    """``classes_`` keeps ``y``'s own dtype, and ``predict`` returns those labels.

    The Rust core votes over DENSE class indices and its label ingress is the
    shared float path, so it can only carry integer-valued targets; the shim does
    sklearn's ``np.unique(..., return_inverse=True)`` encoding itself and maps
    back on the way out. That is exactly what makes a STRING or BOOLEAN target
    work at all — without it, ``fit`` would raise on the float cast — and it is
    checked against a live sklearn rather than a fixture because the encoding is
    a host-side reimplementation of sklearn's own host code.
    """
    x, xq, y, _ = _live_data()
    target = encode(y)

    got = mlrs.KNeighborsClassifier(n_neighbors=3).fit(x, target)
    want = SkKNC(algorithm="brute", n_neighbors=3).fit(x, target)

    assert np.array_equal(got.classes_, want.classes_)
    assert got.classes_.dtype == want.classes_.dtype
    pred = np.asarray(got.predict(xq))
    assert np.array_equal(pred, want.predict(xq))
    assert pred.dtype == want.predict(xq).dtype
    assert np.allclose(
        np.asarray(got.predict_proba(xq)), want.predict_proba(xq), atol=1e-5, rtol=0.0
    )


def test_column_vector_y_warns_and_stays_single_output():
    """A ``(n, 1)`` ``y`` is treated as 1-D with sklearn's ``DataConversionWarning``.

    Without this branch it would silently become a ONE-output multi-output
    problem: ``predict`` would return ``(n, 1)`` and ``predict_proba`` a
    one-element list, neither of which is what sklearn returns.
    """
    x, xq, y, _ = _live_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est = mlrs.KNeighborsClassifier().fit(x, y.reshape(-1, 1))
    assert any(issubclass(w.category, DataConversionWarning) for w in caught)
    assert est.outputs_2d_ is False
    assert np.asarray(est.predict(xq)).ndim == 1
    assert np.asarray(est.predict_proba(xq)).ndim == 2


def test_score_matches_sklearn():
    """``ClassifierMixin.score`` works end to end (accuracy on the label space)."""
    x, xq, y, _ = _live_data()
    yq = SkKNC(algorithm="brute").fit(x, y).predict(xq)
    got = mlrs.KNeighborsClassifier().fit(x, y).score(xq, yq)
    want = SkKNC(algorithm="brute").fit(x, y).score(xq, yq)
    assert got == pytest.approx(want)


# ---------------------------------------------------------------------------
# Neighbour-query surface
# ---------------------------------------------------------------------------


def test_kneighbors_self_query_matches_sklearn():
    """``kneighbors(X=None)`` drops each point's own self-match, as sklearn does."""
    x, _, y, _ = _live_data()
    dist, idx = mlrs.KNeighborsClassifier().fit(x, y).kneighbors()
    want_d, want_i = SkKNC(algorithm="brute").fit(x, y).kneighbors()
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
    got = mlrs.KNeighborsClassifier().fit(x, y).kneighbors_graph(query, mode=mode)
    want = SkKNC(algorithm="brute").fit(x, y).kneighbors_graph(query, mode=mode)
    assert got.shape == want.shape
    assert np.allclose(got.toarray(), want.toarray(), atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# Metric resolution and pass-through parameters
# ---------------------------------------------------------------------------


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
        got = mlrs.KNeighborsClassifier(**kwargs).fit(x, y)
        want = SkKNC(algorithm="brute", **kwargs).fit(x, y)
    assert got.effective_metric_ == want.effective_metric_
    assert got.effective_metric_params_ == want.effective_metric_params_
    assert got.n_samples_fit_ == want.n_samples_fit_
    assert got.n_features_in_ == want.n_features_in_


def test_metric_params_p_overrides_init_p_with_warning():
    """A ``p`` in ``metric_params`` wins over the ``__init__`` one, and warns."""
    x, xq, y, _ = _live_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est = mlrs.KNeighborsClassifier(p=2, metric_params={"p": 1}).fit(x, y)
    assert any(issubclass(w.category, SyntaxWarning) for w in caught)
    # p=1 wins -> manhattan, NOT the p=2 euclidean from __init__.
    assert est.effective_metric_ == "manhattan"
    want = SkKNC(algorithm="brute", metric="manhattan").fit(x, y).predict(xq)
    assert np.array_equal(
        np.asarray(est.predict(xq)).astype(np.int64), want.astype(np.int64)
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

    def raises(build):
        try:
            build().fit(x, y)
            return False
        except ValueError:
            return True

    assert raises(
        lambda: mlrs.KNeighborsClassifier(
            n_neighbors=3, metric=metric, algorithm=algorithm
        )
    ) == raises(lambda: SkKNC(n_neighbors=3, metric=metric, algorithm=algorithm))


@pytest.mark.parametrize("algorithm", ALGORITHM_STRINGS)
def test_every_algorithm_accepted_and_equivalent(algorithm):
    """All four ``algorithm`` values are accepted and give the SAME answer.

    ``algorithm`` selects how the neighbours are found, never which ones they
    are, so a grid search over it must complete with identical predictions —
    which is exactly what sklearn does.
    """
    x, xq, y, _ = _live_data()
    got = mlrs.KNeighborsClassifier(algorithm=algorithm).fit(x, y).predict(xq)
    want = SkKNC(algorithm=algorithm).fit(x, y).predict(xq)
    assert np.array_equal(np.asarray(got).astype(np.int64), want.astype(np.int64))


@pytest.mark.parametrize("leaf_size", [1, 30, 200])
@pytest.mark.parametrize("n_jobs", [None, 1, -1])
def test_leaf_size_and_n_jobs_are_accepted_and_inert(leaf_size, n_jobs):
    """``leaf_size`` / ``n_jobs`` round-trip and cannot change the result.

    Both tune machinery mlrs does not have (a tree; a host thread pool). They
    are accepted so sklearn code and grid searches keep working, and this pins
    that accepting them stayed inert rather than quietly perturbing anything.
    """
    x, xq, y, _ = _live_data()
    est = mlrs.KNeighborsClassifier(leaf_size=leaf_size, n_jobs=n_jobs)
    assert est.get_params()["leaf_size"] == leaf_size
    assert est.get_params()["n_jobs"] == n_jobs
    got = np.asarray(est.fit(x, y).predict(xq))
    want = np.asarray(mlrs.KNeighborsClassifier().fit(x, y).predict(xq))
    assert np.array_equal(got, want)


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
    est = mlrs.KNeighborsClassifier(**kwargs)  # must not raise
    with pytest.raises(ValueError, match=message):
        est.fit(x, y)


# ---------------------------------------------------------------------------
# Deferred upload (KNN-REG-FIT): fit must stay observationally identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_fit_still_rejects_non_finite_X(bad_value):
    """The NaN/inf rejection fires at ``fit``, not at the first ``predict``.

    ``fit`` defers the device upload, and the finite scan moved from numpy's
    ``check_array`` into the Rust call along with it. If that scan had drifted
    to the query path with the upload, a NaN training set would fit "fine" and
    only blow up later — sklearn raises at ``fit``, so mlrs must too.
    """
    x, _, y, _ = _live_data()
    x = x.copy()
    x[3, 1] = bad_value
    with pytest.raises(ValueError, match="infinity|NaN|too large"):
        mlrs.KNeighborsClassifier().fit(x, y)


def test_fitted_attributes_available_without_any_query():
    """A fitted-but-never-queried estimator answers every fitted attribute.

    These are the attributes sklearn code reads between ``fit`` and ``predict``
    (and that ``check_is_fitted`` scans for). They are shape facts, so the
    deferred upload must not be needed to answer them — if any of them forced
    materialization, deferring would buy nothing on the path that matters.
    """
    x, _, y, y_multi = _live_data()
    est = mlrs.KNeighborsClassifier().fit(x, y)
    assert est.n_samples_fit_ == x.shape[0]
    assert est.n_features_in_ == x.shape[1]
    assert est.effective_metric_ == "euclidean"
    assert est.effective_metric_params_ == {}
    assert np.array_equal(est.classes_, np.unique(y))
    assert est._fit_X.shape == x.shape
    assert est._y.shape == y.shape
    assert np.allclose(est._fit_X, x, atol=1e-6, rtol=0.0)

    multi = mlrs.KNeighborsClassifier().fit(x, y_multi)
    assert multi._y.shape == y_multi.shape
    assert len(multi.classes_) == 2


def test_repeated_queries_and_refit_are_stable():
    """Materialization is idempotent, and a refit re-materializes.

    Two consecutive ``predict`` calls must agree (the first one uploads, the
    second must reuse rather than re-upload or read stale state), and fitting
    the SAME object on different data must not keep serving the old training
    set — which is what would happen if the deferred data were consumed but the
    already-materialized arm left in place.
    """
    x, xq, y, _ = _live_data()
    est = mlrs.KNeighborsClassifier()
    est.fit(x, y)
    first = np.asarray(est.predict(xq))
    assert np.array_equal(first, np.asarray(est.predict(xq)))

    # Refit with the label space SHIFTED; the predictions must follow.
    est.fit(x, y + 100)
    refit = np.asarray(est.predict(xq)).astype(np.int64)
    want = SkKNC(algorithm="brute").fit(x, y + 100).predict(xq)
    assert np.array_equal(refit, want.astype(np.int64))
    assert not np.array_equal(refit, first.astype(np.int64))


def test_predict_before_fit_still_raises_not_fitted():
    """An unfitted estimator raises ``NotFittedError``, not an upload error."""
    from sklearn.exceptions import NotFittedError

    _, xq, _, _ = _live_data()
    with pytest.raises(NotFittedError):
        mlrs.KNeighborsClassifier().predict(xq)
    with pytest.raises(NotFittedError):
        mlrs.KNeighborsClassifier().predict_proba(xq)


def test_clone_round_trips_the_full_parameter_set():
    """``clone`` preserves every parameter, and the clone refits to the same answer.

    The refit half matters on its own: the wrapper rebuilds the core estimator
    from parameters held on the Python object, and an implementation that read
    them back out of the fitted Rust handle instead would silently fall back to
    the defaults on a second fit — which is what every ``cross_val_score`` /
    ``GridSearchCV`` does.
    """
    x, xq, y, _ = _live_data()
    est = mlrs.KNeighborsClassifier(
        n_neighbors=3,
        weights="distance",
        metric="manhattan",
        leaf_size=7,
        n_jobs=2,
    )
    assert clone(est).get_params() == est.get_params()

    want = (
        SkKNC(
            algorithm="brute",
            n_neighbors=3,
            weights="distance",
            metric="manhattan",
        )
        .fit(x, y)
        .predict(xq)
        .astype(np.int64)
    )
    for label, obj in (("first", est), ("refit", est), ("clone", clone(est))):
        got = np.asarray(obj.fit(x, y).predict(xq)).astype(np.int64)
        assert np.array_equal(got, want), label
