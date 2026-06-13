"""Neighbors oracle harness (PY-01: full binding path).

Wave-0 COLLECTING stub: parametrizes over the committed k-NN fixtures and
``importorskip("mlrs")`` so it collects green pre-wrapper. Plans 04-06 wire
assertions against the fixture ``distances`` / ``indices`` / ``predict_class``
/ ``predict_proba`` / ``predict_reg`` keys for NearestNeighbors /
KNeighborsClassifier / KNeighborsRegressor.
"""

import pytest

from conftest import load_fixture  # noqa: F401  (Plan 04 uses)

# Req: PY-01 (1e-5 oracle parity through the Python binding path).
NEIGHBORS_FIXTURES = [
    "knn_f32_seed42",
    "knn_f64_seed42",
]


@pytest.mark.parametrize("fixture", NEIGHBORS_FIXTURES)
def test_neighbors_oracle(fixture):
    """PY-01: NearestNeighbors/KNN match the sklearn oracle within 1e-5."""
    mlrs = pytest.importorskip("mlrs")  # noqa: F841  (wired in Plan 04)
    pytest.xfail("neighbors wrappers land in Plan 03/04")
