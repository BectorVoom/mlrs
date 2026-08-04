"""Neighbors oracle harness (PY-01: full binding path).

Re-validates the 1e-5 contract for the three neighbors estimators through the
FULL Python binding path by replaying the committed k-NN ``.npz`` fixtures (a
SECOND consumer; no regeneration):
  - NearestNeighbors.kneighbors -> ``distances`` (1e-5) + ``indices`` (exact).
  - KNeighborsClassifier.predict -> ``predict_class`` (exact labels) and
    predict_proba -> ``predict_proba`` (1e-5).
  - KNeighborsRegressor.predict -> ``predict_reg`` (1e-5).

Each fixture carries a single ``X``/``Xq``/``k`` plus the per-task reference
keys. f64 fixtures are skipped-with-reason on an f64-incapable backend (rocm)
via the ``conftest.requires_f64`` marker.
"""

import numpy as np
import pytest

import mlrs
from conftest import dtype_of, fixture_path, requires_f64


def _atol(fixture):
    return 1e-5 if dtype_of(fixture) == np.float64 else 1e-4


NEIGHBORS_FIXTURES = ["knn_f32_seed42", "knn_f64_seed42"]


@pytest.mark.parametrize("fixture", NEIGHBORS_FIXTURES)
@requires_f64
def test_nearest_neighbors_oracle(fixture):
    """PY-01: NearestNeighbors.kneighbors distances (1e-5) + indices (exact)."""
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    nn = mlrs.NearestNeighbors(n_neighbors=k).fit(d["X"])
    dist, idx = nn.kneighbors(d["Xq"])
    assert np.allclose(
        np.asarray(dist, dtype=np.float64),
        np.asarray(d["distances"], dtype=np.float64),
        atol=_atol(fixture),
        rtol=0.0,
    )
    assert np.array_equal(
        np.asarray(idx).astype(np.int64),
        d["indices"].astype(np.int64),
    )


@pytest.mark.parametrize("fixture", NEIGHBORS_FIXTURES)
@requires_f64
def test_kneighbors_classifier_oracle(fixture):
    """PY-01: KNeighborsClassifier predict (exact) + predict_proba (1e-5)."""
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    clf = mlrs.KNeighborsClassifier(n_neighbors=k).fit(d["X"], d["y_class"])
    pred = np.asarray(clf.predict(d["Xq"])).astype(np.int64).ravel()
    assert np.array_equal(pred, d["predict_class"].astype(np.int64).ravel())
    assert np.allclose(
        np.asarray(clf.predict_proba(d["Xq"]), dtype=np.float64),
        np.asarray(d["predict_proba"], dtype=np.float64),
        atol=_atol(fixture),
        rtol=0.0,
    )


@pytest.mark.parametrize("fixture", NEIGHBORS_FIXTURES)
@requires_f64
def test_kneighbors_regressor_oracle(fixture):
    """PY-01: KNeighborsRegressor predict matches the oracle within 1e-5."""
    d = np.load(fixture_path(fixture))
    k = int(d["k"][0])
    reg = mlrs.KNeighborsRegressor(n_neighbors=k).fit(d["X"], d["y_reg"])
    pred = np.asarray(reg.predict(d["Xq"]), dtype=np.float64).ravel()
    assert np.allclose(
        pred,
        np.asarray(d["predict_reg"], dtype=np.float64).ravel(),
        atol=_atol(fixture),
        rtol=0.0,
    )


# ---------------------------------------------------------------------------
# Deferred upload (KNN-REG-FIT): all three shims must stay observationally
# identical after moving the device upload out of `fit`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["NearestNeighbors", "KNeighborsClassifier"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_fit_still_rejects_non_finite_X(estimator, bad_value):
    """The NaN/inf rejection fires at ``fit``, not at the first query.

    ``fit`` defers the device upload, and the finite scan moved from numpy's
    ``check_array`` into the Rust call along with it. If that scan had drifted
    to the query path, a NaN training set would fit "fine" and only blow up
    later — sklearn raises at ``fit``, so mlrs must too.
    """
    X = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32)
    X[4, 1] = bad_value
    y = np.arange(20, dtype=np.int32) % 2
    est = getattr(mlrs, estimator)()
    with pytest.raises(ValueError, match="infinity|NaN|too large"):
        est.fit(X) if estimator == "NearestNeighbors" else est.fit(X, y)


def test_classifier_rejects_non_finite_labels_at_fit():
    """A non-finite LABEL is rejected at ``fit`` too.

    ``y``'s scan is not a separate pass: the Rust label preparation validates
    integer-ness and finiteness together, because a NaN label would otherwise
    become class ``0`` under the saturating cast with no error at all.
    """
    X = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32)
    y = np.arange(20, dtype=np.float32)
    y[7] = np.nan
    with pytest.raises(ValueError):
        mlrs.KNeighborsClassifier().fit(X, y)


def test_classes_available_without_any_query():
    """``classes_`` is readable the instant ``fit`` returns.

    The shim reads it on the line after ``fit``, so the labels are prepared
    eagerly even though the matrix upload is deferred. A non-contiguous target
    also has to round-trip (WR-01), which is why the labels here are ``{0, 2,
    7}`` rather than ``range(k)``.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3)).astype(np.float32)
    y = rng.choice([0, 2, 7], size=30).astype(np.int32)
    clf = mlrs.KNeighborsClassifier(n_neighbors=3).fit(X, y)
    assert list(clf.classes_) == [0, 2, 7]
    assert clf.n_features_in_ == 3
    # And predict still only ever returns labels drawn from that set.
    assert set(np.asarray(clf.predict(X)).tolist()) <= {0, 2, 7}


@pytest.mark.parametrize("estimator", ["NearestNeighbors", "KNeighborsClassifier"])
def test_repeated_queries_and_refit_are_stable(estimator):
    """Materialization is idempotent, and a refit re-materializes.

    Two consecutive queries must agree (the first uploads, the second must
    reuse), and refitting the SAME object on different data must not keep
    serving the old training set — which is what would happen if the deferred
    data were consumed but the materialized arm left in place.
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    Q = rng.standard_normal((6, 3)).astype(np.float32)
    y = (rng.standard_normal(40) > 0).astype(np.int32)

    if estimator == "NearestNeighbors":
        est = mlrs.NearestNeighbors(n_neighbors=3)
        est.fit(X)
        first = np.asarray(est.kneighbors(Q)[1])
        assert np.array_equal(first, np.asarray(est.kneighbors(Q)[1]))
        est.fit(X[::-1].copy())
        assert not np.array_equal(first, np.asarray(est.kneighbors(Q)[1]))
    else:
        est = mlrs.KNeighborsClassifier(n_neighbors=3)
        est.fit(X, y)
        first = np.asarray(est.predict(Q))
        assert np.array_equal(first, np.asarray(est.predict(Q)))
        est.fit(X, 1 - y)
        assert np.array_equal(np.asarray(est.predict(Q)), 1 - first)


@pytest.mark.parametrize("estimator", ["NearestNeighbors", "KNeighborsClassifier"])
def test_n_neighbors_survives_a_refit(estimator):
    """The hyperparameter is re-read from the wrapper on every ``fit``.

    It used to be recovered from the core enum's unfit payload, which is gone
    once the estimator has been fitted — so a second ``fit`` on the same object
    silently fell back to ``n_neighbors=5``. Every ``clone``-and-refit path
    (``cross_val_score``, ``GridSearchCV``) does exactly that.
    """
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 3)).astype(np.float32)
    Q = rng.standard_normal((4, 3)).astype(np.float32)
    y = (rng.standard_normal(30) > 0).astype(np.int32)
    if estimator == "NearestNeighbors":
        est = mlrs.NearestNeighbors(n_neighbors=7)
        est.fit(X)
        est.fit(X)  # the refit that used to reset k to 5
        assert np.asarray(est.kneighbors(Q)[1]).shape[1] == 7
    else:
        est = mlrs.KNeighborsClassifier(n_neighbors=7)
        est.fit(X, y)
        est.fit(X, y)
        # A 7-neighbour vote over 2 classes can never tie at 3.5, so a silent
        # fallback to k=5 would be invisible here — compare against sklearn
        # instead, which is the actual contract.
        from sklearn.neighbors import KNeighborsClassifier as SkKNC

        want = SkKNC(n_neighbors=7, algorithm="brute").fit(X, y).predict(Q)
        assert np.array_equal(np.asarray(est.predict(Q)).ravel(), want.ravel())


def test_invalid_algorithm():
    """Verify that NearestNeighbors, KNeighborsClassifier, and KNeighborsRegressor raise ValueError on invalid algorithm.

    float32 operands deliberately: the ``algorithm`` guard is dtype-independent
    and lives in the Python shim, so keeping the fitted arrays off f64 lets this
    run on an f64-incapable backend (rocm) instead of being skipped by
    ``requires_f64`` like the fixture-replay tests above. With float64 the
    accepted-algorithm ``fit`` calls would reach ``guard_f64`` and ERROR rather
    than skip.
    """
    X = np.random.randn(10, 4).astype(np.float32)
    y_class = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    y_reg = np.random.randn(10).astype(np.float32)

    # NearestNeighbors
    nn = mlrs.NearestNeighbors(algorithm="invalid")
    with pytest.raises(ValueError, match="Algorithm is not supported"):
        nn.fit(X)
    # auto and brute are accepted
    mlrs.NearestNeighbors(algorithm="auto").fit(X)
    mlrs.NearestNeighbors(algorithm="brute").fit(X)

    # KNeighborsClassifier
    clf = mlrs.KNeighborsClassifier(algorithm="invalid")
    with pytest.raises(ValueError, match="Algorithm is not supported"):
        clf.fit(X, y_class)
    # auto and brute are accepted
    mlrs.KNeighborsClassifier(algorithm="auto").fit(X, y_class)
    mlrs.KNeighborsClassifier(algorithm="brute").fit(X, y_class)

    # KNeighborsRegressor
    reg = mlrs.KNeighborsRegressor(algorithm="invalid")
    with pytest.raises(ValueError, match="Algorithm is not supported"):
        reg.fit(X, y_reg)
    # auto and brute are accepted
    mlrs.KNeighborsRegressor(algorithm="auto").fit(X, y_reg)
    mlrs.KNeighborsRegressor(algorithm="brute").fit(X, y_reg)
