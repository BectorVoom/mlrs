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


def label_perm_remap(pred, ref):
    """Return ``pred`` relabeled so its cluster ids match ``ref``'s ids.

    Companion to ``label_perm_allclose`` for the cases (KMeans
    ``cluster_centers_``) where, after establishing the labelings agree up to a
    permutation, we need the bijection itself to align the *per-cluster*
    quantities (centers) before a numeric ``allclose``. Returns ``None`` when no
    consistent bijection exists (the caller then fails the test).
    """
    pred = np.asarray(pred).astype(np.int64).ravel()
    ref = np.asarray(ref).astype(np.int64).ravel()
    if pred.shape != ref.shape:
        return None
    mapping = {}
    for pv, rv in zip(pred, ref):
        if pv == -1 or rv == -1:
            if pv != rv:
                return None
            continue
        if pv in mapping:
            if mapping[pv] != rv:
                return None
        else:
            mapping[pv] = rv
    return mapping


def proba_allclose(a, b, atol=1e-5):
    """Compare two ``predict_proba`` matrices row-normalized (gauge-fixed).

    LogisticRegression probabilities are the gauge-invariant gate (Phase-5
    D-12): raw ``coef_`` is only defined up to the softmax gauge, but the
    per-sample class probabilities are unique. Each row of a probability matrix
    sums to 1, so a direct ``np.allclose`` already compares the gauge-fixed
    quantity; this helper additionally re-normalizes each row (guarding against
    a backend that returns un-normalized scores) before the ``atol`` check.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    a = a / np.clip(a.sum(axis=1, keepdims=True), 1e-300, None)
    b = b / np.clip(b.sum(axis=1, keepdims=True), 1e-300, None)
    return bool(np.allclose(a, b, atol=atol, rtol=0.0))


def dtype_of(fixture_name):
    """``np.float32`` / ``np.float64`` parsed from a fixture basename suffix."""
    return np.float32 if "_f32_" in fixture_name else np.float64


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
