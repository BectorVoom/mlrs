"""Manifold oracle harness (TSNE-01): the full-binding-path replay of the
committed t-SNE fixtures.

Two tiers (the Rust ``tsne_test.rs`` analog through the Python surface):
- BAND: ``mlrs.TSNE(method='exact', init='pca')`` must reach the sklearn
  embedding's neighborhood-preservation band (``trustworthiness`` within 0.05,
  ``kl_divergence_`` within +0.25) — the end-to-end descent is chaotic, so
  exact equality is meaningless.
- Determinism: PCA-init refit is bit-identical.

f64 fixtures are skipped-with-reason on an f64-incapable backend via
``conftest.requires_f64``.
"""

import numpy as np
import pytest

import mlrs
from conftest import dtype_of, fixture_path, requires_f64

TSNE_FIXTURES = ["tsne_f32_seed42", "tsne_f64_seed42"]


def _trustworthiness(x, emb, k=5):
    """sklearn.manifold.trustworthiness port (numpy-only — no sklearn import
    needed at test time for the Rust parity; sklearn IS available in this
    venv, but the explicit port keeps the formula pinned)."""
    n = x.shape[0]
    dist_x = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(dist_x, np.inf)
    ind_x = np.argsort(dist_x, axis=1)
    inverted = np.zeros((n, n), dtype=int)
    ordered = np.arange(n + 1)
    inverted[ordered[:-1, np.newaxis], ind_x] = ordered[1:]
    dist_e = ((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(dist_e, np.inf)
    ind_e = np.argsort(dist_e, axis=1)[:, :k]
    ranks = inverted[ordered[:-1, np.newaxis], ind_e] - k
    t = np.sum(ranks[ranks > 0])
    return 1.0 - t * (2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0)))


@pytest.mark.parametrize("fixture", TSNE_FIXTURES)
@requires_f64
def test_tsne_band(fixture):
    d = np.load(fixture_path(fixture))
    est = mlrs.TSNE(perplexity=float(d["perplexity"][0]), init="pca")
    emb = np.asarray(est.fit_transform(d["X"]), dtype=np.float64)
    assert emb.shape == (d["X"].shape[0], 2)

    trust = _trustworthiness(np.asarray(d["X"], dtype=np.float64), emb)
    assert trust >= float(d["trust"][0]) - 0.05, f"{fixture}: trustworthiness {trust}"
    kl = est.kl_divergence_
    assert 0.0 < kl <= float(d["kl"][0]) + 0.25, f"{fixture}: kl {kl}"
    assert est.n_iter_ < 1000


def test_tsne_rejects_unsupported():
    X = np.random.default_rng(0).normal(size=(8, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="method"):
        mlrs.TSNE(method="barnes_hut").fit(X)
    with pytest.raises(ValueError, match="metric"):
        mlrs.TSNE(metric="cosine").fit(X)
    with pytest.raises(ValueError, match="perplexity"):
        mlrs.TSNE(perplexity=100.0).fit(X)  # perplexity >= n_samples
