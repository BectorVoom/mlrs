"""Clustering oracle harness (PY-01: full binding path, label-perm compare).

Re-validates the 1e-5 contract for KMeans and DBSCAN through the FULL Python
binding path by replaying the committed KMeans/DBSCAN ``.npz`` fixtures (a SECOND
consumer; no regeneration). Cluster ids are arbitrary, so ``labels_`` is compared
up to a label permutation (``conftest.label_perm_allclose``, the analog of
``crates/mlrs-algos/tests/kmeans_test.rs``); KMeans ``cluster_centers_`` are then
aligned through the recovered bijection (``label_perm_remap``) before a numeric
``allclose``, and ``inertia_`` (permutation-invariant) is compared directly.

f64 fixtures are skipped-with-reason on an f64-incapable backend (rocm) via the
``conftest.requires_f64`` marker.
"""

import numpy as np
import pytest

import mlrs
from conftest import (
    dtype_of,
    fixture_path,
    label_perm_allclose,
    label_perm_remap,
    requires_f64,
)


def _atol(fixture):
    return 1e-5 if dtype_of(fixture) == np.float64 else 1e-4


KMEANS_FIXTURES = ["kmeans_f32_seed42", "kmeans_f64_seed42"]
DBSCAN_FIXTURES = ["dbscan_f32_seed42", "dbscan_f64_seed42"]


@pytest.mark.parametrize("fixture", KMEANS_FIXTURES)
@requires_f64
def test_kmeans_oracle(fixture):
    """PY-01: KMeans labels_ (label-perm), cluster_centers_ (remapped), inertia_."""
    d = np.load(fixture_path(fixture))
    n_clusters = int(d["centers"].shape[0])
    est = mlrs.KMeans(
        n_clusters=n_clusters, max_iter=300, tol=1e-4, random_state=42
    ).fit(d["X"])

    labels = np.asarray(est.labels_).astype(np.int64).ravel()
    ref_labels = d["labels"].astype(np.int64).ravel()
    assert label_perm_allclose(labels, ref_labels)

    # Align our cluster ids to the reference's, then compare the per-cluster
    # centers numerically (a label permutation reorders the center rows).
    mapping = label_perm_remap(labels, ref_labels)
    assert mapping is not None
    centers = np.asarray(est.cluster_centers_, dtype=np.float64)
    ref_centers = np.asarray(d["centers"], dtype=np.float64)
    aligned = np.empty_like(ref_centers)
    for our_id, ref_id in mapping.items():
        aligned[ref_id] = centers[our_id]
    assert np.allclose(aligned, ref_centers, atol=_atol(fixture), rtol=0.0)

    # inertia is permutation-invariant — direct compare (relative for scale).
    inertia = float(est.inertia_)
    ref_inertia = float(d["inertia"][0])
    assert abs(inertia - ref_inertia) <= _atol(fixture) * (1.0 + abs(ref_inertia))


@pytest.mark.parametrize("fixture", DBSCAN_FIXTURES)
@requires_f64
def test_dbscan_oracle(fixture):
    """PY-01: DBSCAN labels_ match the sklearn oracle up to a label permutation."""
    d = np.load(fixture_path(fixture))
    est = mlrs.DBSCAN(
        eps=float(d["eps"][0]), min_samples=int(d["min_samples"][0])
    ).fit(d["X"])
    labels = np.asarray(est.labels_).astype(np.int64).ravel()
    ref_labels = d["labels"].astype(np.int64).ravel()
    assert label_perm_allclose(labels, ref_labels)
    # core_sample_indices_ is an exact set (sample ids, no permutation).
    core = np.sort(np.asarray(est.core_sample_indices_).astype(np.int64).ravel())
    ref_core = np.sort(d["core_sample_indices"].astype(np.int64).ravel())
    assert np.array_equal(core, ref_core)
