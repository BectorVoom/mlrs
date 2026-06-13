"""Decomposition oracle harness (PY-01: full binding path, sign-flip compare).

Wave-0 COLLECTING stub: parametrizes over the committed PCA/TruncatedSVD
fixtures and ``importorskip("mlrs")`` so it collects green pre-wrapper. Plans
04-06 wire ``sign_flip_allclose`` against the fixture ``components_`` /
``explained_variance_`` / ``singular_values_`` / ``transform`` keys (components
are defined only up to a per-row sign).
"""

import pytest

from conftest import (  # noqa: F401  (Plan 04 uses these)
    load_fixture,
    sign_flip_allclose,
)

# Req: PY-01 (1e-5 oracle parity through the Python binding path).
DECOMPOSITION_FIXTURES = [
    "pca_f32_seed42",
    "pca_f64_seed42",
    "pca_tall_f32_seed42",
    "pca_tall_f64_seed42",
    "pca_wide_f32_seed42",
    "pca_wide_f64_seed42",
    "truncated_svd_f32_seed42",
    "truncated_svd_f64_seed42",
]


@pytest.mark.parametrize("fixture", DECOMPOSITION_FIXTURES)
def test_decomposition_oracle(fixture):
    """PY-01: PCA/TruncatedSVD match the sklearn oracle (sign-flip)."""
    mlrs = pytest.importorskip("mlrs")  # noqa: F841  (wired in Plan 04)
    pytest.xfail("decomposition wrappers land in Plan 03/04")
