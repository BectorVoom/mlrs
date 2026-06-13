"""Clustering oracle harness (PY-01: full binding path, label-perm compare).

Wave-0 COLLECTING stub: parametrizes over the committed KMeans/DBSCAN fixtures
and ``importorskip("mlrs")`` so it collects green pre-wrapper. Plans 04-06 wire
``label_perm_allclose`` against the fixture ``labels`` / ``centers`` /
``core_sample_indices`` keys (KMeans/DBSCAN cluster ids are permutation-free).
"""

import pytest

from conftest import (  # noqa: F401  (Plan 04 uses these)
    label_perm_allclose,
    load_fixture,
)

# Req: PY-01 (1e-5 oracle parity through the Python binding path).
CLUSTER_FIXTURES = [
    "kmeans_f32_seed42",
    "kmeans_f64_seed42",
    "dbscan_f32_seed42",
    "dbscan_f64_seed42",
]


@pytest.mark.parametrize("fixture", CLUSTER_FIXTURES)
def test_cluster_oracle(fixture):
    """PY-01: KMeans/DBSCAN match the sklearn oracle (label-permutation)."""
    mlrs = pytest.importorskip("mlrs")  # noqa: F841  (wired in Plan 04)
    pytest.xfail("cluster wrappers land in Plan 03/04")
