"""pytest fixtures + oracle-compare helpers for the mlrs Python harness.

This is the Wave-0 scaffold (the "Nyquist" test tree): it COLLECTS green now
and converts to real assertions in Plans 04-06 once the wrappers exist. It is a
SECOND consumer of the committed ``tests/fixtures/*.npz`` oracle blobs (the same
sklearn-reference fixtures produced by ``scripts/gen_oracle.py`` and consumed by
the Rust ``crates/mlrs-algos/tests/*`` tests) — re-validating the 1e-5 contract
through the full ``numpy -> pyarrow -> __arrow_c_array__ -> Rust FFI -> device ->
host -> numpy`` path (RESEARCH 06 §pytest Oracle Harness). No new fixtures.

Helpers mirror ``mlrs_core::oracle``:
  - ``sign_flip_allclose`` — PCA/SVD ``components_`` column/row-sign invariance
    (the analog of ``pca_test.rs`` / ``truncated_svd_test.rs`` sign-flip compare).
  - ``label_perm_allclose`` — KMeans/DBSCAN ``labels_`` label-permutation
    invariance (the analog of ``kmeans_test.rs`` label-perm compare).
  - ``requires_f64`` — a skip marker keyed on the surfaced
    ``mlrs.backend_supports_f64()`` capability flag (mirrors
    ``capability.rs::skip_f64_with_log``); the whole f64 case is skipped on an
    f64-incapable backend (rocm), with a logged reason.

The fixture loader, helpers, and marker are import-guarded so the suite collects
even before the ``mlrs`` extension is built (Wave 0).
"""

import os

import numpy as np
import pytest

# Repo-root `tests/fixtures/` holds the committed .npz oracle blobs. This file
# lives at crates/mlrs-py/python/tests/conftest.py → four parents up is the root.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
FIXTURE_DIR = os.path.join(_REPO_ROOT, "tests", "fixtures")


def load_fixture(name):
    """Load a committed ``.npz`` oracle blob by basename (with or without ext).

    Returns the ``numpy.lib.npyio.NpzFile`` mapping (e.g. ``["X"]``, ``["coef"]``,
    ``["labels"]``, ``["components_"]`` — the keys written by ``gen_oracle.py``).
    """
    if not name.endswith(".npz"):
        name = name + ".npz"
    path = os.path.join(FIXTURE_DIR, name)
    return np.load(path)


def fixture_path(name):
    """Absolute path to a committed fixture (for ``os.path.exists`` skips)."""
    if not name.endswith(".npz"):
        name = name + ".npz"
    return os.path.join(FIXTURE_DIR, name)


def sign_flip_allclose(a, b, atol=1e-5):
    """Compare two ``components_``-style matrices up to per-row sign flip.

    PCA/TruncatedSVD components are only defined up to a sign per component, so
    a row of ``a`` matches the corresponding row of ``b`` if either the row or
    its negation is within ``atol`` (mirrors the Rust sign-flip oracle compare).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    a2 = a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(1, -1)
    b2 = b.reshape(b.shape[0], -1) if b.ndim > 1 else b.reshape(1, -1)
    for ra, rb in zip(a2, b2):
        if not (
            np.allclose(ra, rb, atol=atol) or np.allclose(ra, -rb, atol=atol)
        ):
            return False
    return True


def label_perm_allclose(a, b):
    """Compare two integer label vectors up to a label permutation.

    KMeans/DBSCAN cluster ids are arbitrary; two labelings match if there is a
    bijection between their label sets that makes them identical (and noise
    label ``-1`` maps to ``-1``). Mirrors the Rust label-permutation compare.
    """
    a = np.asarray(a).astype(np.int64).ravel()
    b = np.asarray(b).astype(np.int64).ravel()
    if a.shape != b.shape:
        return False
    mapping = {}
    seen_targets = set()
    for av, bv in zip(a, b):
        if av == -1 or bv == -1:
            if av != bv:
                return False
            continue
        if av in mapping:
            if mapping[av] != bv:
                return False
        else:
            if bv in seen_targets:
                return False
            mapping[av] = bv
            seen_targets.add(bv)
    return True


def _backend_supports_f64():
    """Query the surfaced capability flag; ``True`` if mlrs is not importable
    yet (so collection is not blocked at Wave 0)."""
    try:
        import mlrs

        return bool(mlrs.backend_supports_f64())
    except Exception:
        return True


# f64-capability skip marker (mirrors capability.rs::skip_f64_with_log): on an
# f64-incapable backend wheel (rocm), f64 oracle cases skip with a clear reason.
requires_f64 = pytest.mark.skipif(
    not _backend_supports_f64(),
    reason="backend does not support f64 (mlrs.backend_supports_f64() is False)",
)
