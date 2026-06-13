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
