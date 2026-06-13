"""Decomposition oracle harness (PY-01: full binding path, sign-flip compare).

Re-validates the 1e-5 contract for PCA and TruncatedSVD through the FULL Python
binding path by replaying the committed PCA/TruncatedSVD ``.npz`` fixtures (a
SECOND consumer; no regeneration). Each component is defined only up to a sign,
so ``components_`` is compared per-row up to a sign flip
(``conftest.sign_flip_allclose``, the analog of
``crates/mlrs-algos/tests/pca_test.rs`` / ``truncated_svd_test.rs``). The SAME
per-component sign is recovered from ``components_`` and applied to the
``transform`` output columns (a consistent gauge), while the sign-invariant
quantities — ``mean_`` / ``explained_variance_`` / ``explained_variance_ratio_``
/ ``singular_values_`` — are compared directly.

f64 fixtures are skipped-with-reason on an f64-incapable backend (rocm) via the
``conftest.requires_f64`` marker.
"""

import numpy as np
import pytest

import mlrs
from conftest import (
    dtype_of,
    fixture_path,
    requires_f64,
    sign_flip_allclose,
)


def _atol(fixture):
    return 1e-5 if dtype_of(fixture) == np.float64 else 1e-4


# (fixture, n_components, is_pca)
PCA_FIXTURES = [
    ("pca_f32_seed42", 3, True),
    ("pca_f64_seed42", 3, True),
    ("pca_tall_f32_seed42", 3, True),
    ("pca_tall_f64_seed42", 3, True),
    ("pca_wide_f32_seed42", 2, True),
    ("pca_wide_f64_seed42", 2, True),
    ("truncated_svd_f32_seed42", 3, False),
    ("truncated_svd_f64_seed42", 3, False),
]


def _row_signs(comp, ref):
    """Per-component sign (+1/-1) aligning each ``comp`` row to ``ref``."""
    comp = np.asarray(comp, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return np.array(
        [1.0 if np.dot(c, r) >= 0 else -1.0 for c, r in zip(comp, ref)]
    )


@pytest.mark.parametrize(
    "fixture,n_components,is_pca",
    PCA_FIXTURES,
    ids=[c[0] for c in PCA_FIXTURES],
)
@requires_f64
def test_decomposition_oracle(fixture, n_components, is_pca):
    """PY-01: PCA/TruncatedSVD match the sklearn oracle (sign-flip components_)."""
    d = np.load(fixture_path(fixture))
    atol = _atol(fixture)
    est = (
        mlrs.PCA(n_components=n_components)
        if is_pca
        else mlrs.TruncatedSVD(n_components=n_components)
    ).fit(d["X"])

    comp = np.asarray(est.components_, dtype=np.float64)
    ref_comp = np.asarray(d["components_"], dtype=np.float64)
    assert sign_flip_allclose(comp, ref_comp, atol=atol)

    # Recover the per-component sign and apply it to the transform columns.
    signs = _row_signs(comp, ref_comp)
    tr = np.asarray(est.transform(d["X"]), dtype=np.float64)
    ref_tr = np.asarray(d["transform"], dtype=np.float64)
    assert np.allclose(tr * signs[None, :], ref_tr, atol=atol, rtol=0.0)

    # Sign-invariant scalars compared directly. The PCA fixtures carry
    # explained_variance_ + explained_variance_ratio_ + mean_; the TruncatedSVD
    # fixtures carry explained_variance_ + singular_values_ (no ratio/mean key).
    if "explained_variance_ratio_" in d.files:
        assert np.allclose(
            np.ravel(np.asarray(est.explained_variance_ratio_)),
            np.ravel(d["explained_variance_ratio_"]),
            atol=atol,
            rtol=0.0,
        )
    if is_pca:
        assert np.allclose(
            np.ravel(np.asarray(est.mean_)),
            np.ravel(d["mean_"]),
            atol=atol,
            rtol=0.0,
        )
        assert np.allclose(
            np.ravel(np.asarray(est.explained_variance_)),
            np.ravel(d["explained_variance_"]),
            atol=atol,
            rtol=0.0,
        )
    else:
        assert np.allclose(
            np.ravel(np.asarray(est.singular_values_)),
            np.ravel(d["singular_values_"]),
            atol=atol,
            rtol=0.0,
        )
