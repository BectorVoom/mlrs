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
from sklearn.manifold import TSNE as SkTSNE

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
        mlrs.TSNE(method="fft_interpolation").fit(X)
    with pytest.raises(ValueError, match="metric"):
        mlrs.TSNE(metric="not_a_metric").fit(X)
    with pytest.raises(ValueError, match="perplexity"):
        mlrs.TSNE(perplexity=100.0).fit(X)  # perplexity >= n_samples


# --------------------------------------------------------------------------- #
# TSNE-PARAMS: the string-valued surface, through the FULL binding path.
#
# The Rust-side gates in `crates/mlrs-algos/tests/tsne_params_test.rs` already
# assert the distance and joint-probability matrices against sklearn per metric.
# What those cannot reach is the PyO3 boundary itself: whether `metric_params`
# is unpacked into the right scipy keyword, whether an ndarray `init` survives
# the flatten, and whether a callable `metric` is realized correctly. These
# exercise exactly that, against LIVE sklearn rather than a committed fixture.
# --------------------------------------------------------------------------- #

# Every `metric=` string sklearn's TSNE accepts and can evaluate on a generic
# float design. `haversine` (2 features only), `nan_euclidean`, `precomputed`
# and `wminkowski` (removed from scipy) are handled separately below.
_TSNE_METRICS = [
    "euclidean", "l2", "sqeuclidean", "l1", "manhattan", "cityblock",
    "chebyshev", "minkowski", "cosine", "correlation", "canberra",
    "braycurtis", "seuclidean", "mahalanobis", "hamming", "matching",
    "jaccard", "dice", "rogerstanimoto", "russellrao", "sokalsneath", "yule",
]


def _metric_design(seed=0):
    """Blobs with genuine zeros — see `gen_oracle.py::_tsne_metric_design` for
    why the zeros are what make the bool-cast metrics informative."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0] * 5, [6.0, 6.0, -6.0, 3.0, -3.0], [-7.0, 4.0, 5.0, -5.0, 2.5]])
    x = np.vstack([centers[b] + 0.7 * rng.standard_normal((12, 5)) for b in range(3)])
    x = np.where(rng.random(x.shape) < 0.35, 0.0, x)
    for i in range(x.shape[0]):
        if not np.any(x[i]):
            x[i, i % x.shape[1]] = 1.0 + 0.1 * i
    return x


@pytest.mark.parametrize("metric", _TSNE_METRICS)
def test_tsne_every_metric_string_matches_sklearn(metric):
    """Each metric string reproduces sklearn's input-space geometry EXACTLY,
    through the full binding path.

    ## Why the embedding is frozen instead of optimized
    Running the descent and comparing the result cannot gate a metric. t-SNE's
    dynamics are chaotic: from an identical `P` and an identical init, an
    `f64` rounding difference in the last bit amplifies over hundreds of
    iterations into a different local optimum. Measured on this design, mlrs
    and sklearn land on visibly different optima for several metrics — mlrs
    better for `cosine`, `dice`, `jaccard`, sklearn better for `canberra`,
    `braycurtis` — with neither systematically ahead. Any gate tight enough to
    be informative would fail on that noise, and any gate loose enough to pass
    would not detect a genuinely wrong metric.

    So the descent is switched off: `learning_rate=1e-30` leaves the embedding
    at the injected init, which makes the reported `kl_divergence_` a pure
    deterministic function of `P` — and therefore of the metric. That value
    agrees with sklearn's to ~1e-15 relative for all 22 strings, which is a far
    stronger statement than any band over an optimized run, and it still
    discriminates: the frozen KL ranges from 0.90 (`yule`) to 1.40
    (`rogerstanimoto`) across the metric set.

    ## Why `max_iter=300` and not 250
    sklearn's `kl_divergence_` is `np.finfo(float).max` whenever
    `max_iter == 250` exactly: `_tsne` runs phase 2 with `it = 250` and
    `max_iter = 250`, so `_gradient_descent`'s `for i in range(250, 250)` never
    executes and its `error = np.finfo(float).max` sentinel is returned
    verbatim. mlrs does not have that hole — it re-evaluates the KL against the
    un-exaggerated `P` after the schedule ends, in every branch — but the
    ORACLE does, so the fixture steps off the boundary.
    """
    X = _metric_design()
    common = dict(
        perplexity=8.0, metric=metric, method="exact",
        max_iter=300, learning_rate=1e-30,
    )
    init = 1e-4 * np.random.default_rng(31).standard_normal((X.shape[0], 2))

    est = mlrs.TSNE(init=init.copy(), **common)
    emb = np.asarray(est.fit_transform(X), dtype=np.float64)
    assert emb.shape == (X.shape[0], 2)
    assert np.isfinite(emb).all(), f"metric={metric}: embedding must be finite"

    sk = SkTSNE(init=init.copy(), **common).fit(X)
    assert sk.kl_divergence_ < 1e300, "the oracle hit sklearn's max_iter=250 sentinel"
    np.testing.assert_allclose(
        est.kl_divergence_, sk.kl_divergence_, rtol=1e-10, atol=0.0,
        err_msg=f"metric={metric}: the joint probabilities differ from sklearn's",
    )


@pytest.mark.parametrize("metric", _TSNE_METRICS)
def test_tsne_every_metric_string_tracks_sklearn_over_a_short_horizon(metric):
    """Second tier: the whole DESCENT — gradient, gains, momentum, both phases
    — tracks sklearn per metric, not just the `P` the frozen gate pins.

    The trick is the step size. t-SNE's divergence from an identical start is
    driven by how far the trajectory travels, not by how many iterations it
    runs, so shrinking `learning_rate` while keeping the full 300-iteration
    schedule keeps the two implementations in the regime where they agree.
    Measured on this design, the worst relative disagreement across all 22
    metrics is 8e-10 at `learning_rate=1e-3` and 1e-2 at `1e-1` — so `1e-3` is
    the last horizon that still gates, and it exercises every line of the
    update rule while doing it.

    A gate on the FULLY optimized run is deliberately not attempted: there,
    mlrs and sklearn land on different local optima with neither systematically
    ahead (mlrs reaches a lower KL for `cosine`/`dice`/`jaccard`, sklearn for
    `canberra`/`braycurtis`), and for the near-degenerate `russellrao`/`yule`
    geometries BOTH can finish above the KL they started from — sklearn's
    `yule` optimum is 1.156 against an initial 0.902. Nothing tight can be
    asserted about that, and asserting something loose would gate nothing.
    """
    X = _metric_design()
    common = dict(
        perplexity=8.0, metric=metric, method="exact",
        max_iter=300, learning_rate=1e-3,
    )
    init = 1e-4 * np.random.default_rng(31).standard_normal((X.shape[0], 2))

    est = mlrs.TSNE(init=init.copy(), **common)
    emb = np.asarray(est.fit_transform(X), dtype=np.float64)
    sk = SkTSNE(init=init.copy(), **common).fit(X)

    assert np.isfinite(emb).all()
    # The descent must actually have moved, or this would pass vacuously.
    assert not np.allclose(emb, init), f"metric={metric}: the embedding never moved"
    np.testing.assert_allclose(
        est.kl_divergence_, sk.kl_divergence_, rtol=1e-6, atol=0.0,
        err_msg=f"metric={metric}: the descent diverged from sklearn's",
    )


@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
def test_tsne_method_matches_sklearn_band_from_shared_init(method):
    """Both `method` values, started from the SAME injected embedding so the
    only remaining difference is arithmetic."""
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(c, 0.7, (40, 5)) for c in [0.0, 6.0, -7.0]])
    init = 1e-4 * rng.standard_normal((X.shape[0], 2))

    est = mlrs.TSNE(perplexity=10.0, method=method, init=init.copy(), random_state=0)
    emb = np.asarray(est.fit_transform(X), dtype=np.float64)
    sk = SkTSNE(perplexity=10.0, method=method, init=init.copy(), random_state=0).fit(X)

    t_mlrs = _trustworthiness(X, emb)
    t_sk = _trustworthiness(X, sk.embedding_)
    assert t_mlrs >= t_sk - 0.05, f"method={method}: trust {t_mlrs} vs sklearn {t_sk}"
    assert 0.0 < est.kl_divergence_ <= sk.kl_divergence_ + 0.25


@pytest.mark.parametrize("init", ["pca", "random", "array"])
def test_tsne_every_init_string_reaches_sklearns_band(init):
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(c, 0.7, (40, 5)) for c in [0.0, 6.0, -7.0]])
    arg = 1e-4 * rng.standard_normal((X.shape[0], 2)) if init == "array" else init

    est = mlrs.TSNE(perplexity=10.0, method="exact", init=arg, random_state=0)
    emb = np.asarray(est.fit_transform(X), dtype=np.float64)
    sk_init = arg.copy() if init == "array" else arg
    sk = SkTSNE(perplexity=10.0, method="exact", init=sk_init, random_state=0).fit(X)

    t_mlrs = _trustworthiness(X, emb)
    t_sk = _trustworthiness(X, sk.embedding_)
    assert t_mlrs >= t_sk - 0.05, f"init={init}: trust {t_mlrs} vs sklearn {t_sk}"


def test_tsne_learning_rate_auto_equals_the_resolved_constant():
    """`'auto'` is `max(n / early_exaggeration / 4, 50)` — assert EXACT equality
    with that number rather than a band, which could not tell it from a nearby
    constant."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 4))
    resolved = max(X.shape[0] / 12.0 / 4.0, 50.0)

    a = np.asarray(
        mlrs.TSNE(perplexity=10.0, learning_rate="auto", init="pca", max_iter=300)
        .fit_transform(X),
        dtype=np.float64,
    )
    b = np.asarray(
        mlrs.TSNE(perplexity=10.0, learning_rate=resolved, init="pca", max_iter=300)
        .fit_transform(X),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(a, b)


def test_tsne_metric_params_reach_the_pair_loop():
    """`metric_params={'p': ...}` must change the geometry; `p=2` must
    reproduce plain euclidean exactly."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(50, 4))
    common = dict(perplexity=8.0, init="pca", method="exact", max_iter=300)

    eu = np.asarray(mlrs.TSNE(metric="euclidean", **common).fit_transform(X), dtype=np.float64)
    mk2 = np.asarray(
        mlrs.TSNE(metric="minkowski", metric_params={"p": 2.0}, **common).fit_transform(X),
        dtype=np.float64,
    )
    mk3 = np.asarray(
        mlrs.TSNE(metric="minkowski", metric_params={"p": 3.0}, **common).fit_transform(X),
        dtype=np.float64,
    )
    np.testing.assert_allclose(mk2, eu, rtol=1e-9, atol=1e-9)
    assert not np.allclose(mk3, eu), "p=3 must not reproduce the euclidean embedding"

    with pytest.raises(ValueError, match="metric_params"):
        mlrs.TSNE(metric="minkowski", metric_params={"bogus": 1.0}, **common).fit(X)


def test_tsne_callable_metric_matches_the_precomputed_route():
    """A callable `metric` is realized by evaluating it into a dense distance
    matrix; that must agree with passing the same matrix as
    `metric='precomputed'`."""
    rng = np.random.default_rng(13)
    X = rng.normal(size=(40, 3))

    def cityblock(u, v):
        return float(np.abs(u - v).sum())

    common = dict(perplexity=8.0, init="random", method="exact", max_iter=300, random_state=0)
    via_callable = np.asarray(
        mlrs.TSNE(metric=cityblock, **common).fit_transform(X), dtype=np.float64
    )
    dist = np.abs(X[:, None, :] - X[None, :, :]).sum(-1)
    via_precomputed = np.asarray(
        mlrs.TSNE(metric="precomputed", **common).fit_transform(dist), dtype=np.float64
    )
    np.testing.assert_allclose(via_callable, via_precomputed, rtol=1e-9, atol=1e-9)


def test_tsne_callable_metric_works_with_the_default_pca_init():
    """A callable `metric` is realized as a precomputed matrix internally, and
    `metric='precomputed'` forbids `init='pca'` — but a CALLABLE metric must
    not inherit that, because sklearn keeps the original design and PCA-
    initializes from it, and `'pca'` is the default init. Without the carve-out
    `TSNE(metric=my_callable)` would fail out of the box while sklearn
    succeeds."""
    rng = np.random.default_rng(37)
    X = rng.normal(size=(40, 4))

    def cityblock(u, v):
        return float(np.abs(u - v).sum())

    # The default init must simply work.
    est = mlrs.TSNE(perplexity=8.0, metric=cityblock, max_iter=300)
    emb = np.asarray(est.fit_transform(X), dtype=np.float64)
    assert emb.shape == (40, 2)
    assert np.isfinite(emb).all()

    # ...and must agree with sklearn's own callable + PCA-init run, which is
    # deterministic once the step size keeps the trajectory short.
    common = dict(perplexity=8.0, max_iter=300, learning_rate=1e-3, init="pca")
    a = mlrs.TSNE(metric=cityblock, method="exact", **common)
    a.fit(X)
    b = SkTSNE(metric=cityblock, method="exact", **common).fit(X)
    np.testing.assert_allclose(a.kl_divergence_, b.kl_divergence_, rtol=1e-6)

    # The explicit `precomputed` spelling still refuses `init='pca'`.
    dist = np.abs(X[:, None, :] - X[None, :, :]).sum(-1)
    with pytest.raises(ValueError, match="pca"):
        mlrs.TSNE(perplexity=8.0, metric="precomputed", init="pca").fit(dist)


def test_tsne_haversine_and_precomputed_geometry_rules():
    rng = np.random.default_rng(17)
    X2 = rng.uniform(-1.0, 1.0, size=(40, 2))
    # haversine is defined on exactly 2 features and must fit there...
    est = mlrs.TSNE(perplexity=8.0, metric="haversine", init="random", max_iter=250)
    assert np.isfinite(np.asarray(est.fit_transform(X2), dtype=np.float64)).all()
    # ...and be rejected anywhere else.
    with pytest.raises(ValueError):
        mlrs.TSNE(perplexity=8.0, metric="haversine", init="random").fit(
            rng.normal(size=(40, 5))
        )
    # `precomputed` needs a square X, and rules out init='pca'.
    with pytest.raises(ValueError):
        mlrs.TSNE(perplexity=8.0, metric="precomputed", init="random").fit(
            rng.normal(size=(40, 5))
        )
    dist = np.abs(X2[:, None, :] - X2[None, :, :]).sum(-1)
    with pytest.raises(ValueError, match="pca"):
        mlrs.TSNE(perplexity=8.0, metric="precomputed", init="pca").fit(dist)


def test_tsne_wminkowski_is_rejected_like_sklearn():
    """sklearn accepts the string at validation and then fails, because scipy
    removed the metric. mlrs fails too rather than silently substituting one."""
    X = np.random.default_rng(19).normal(size=(30, 4))
    with pytest.raises(ValueError):
        mlrs.TSNE(perplexity=8.0, metric="wminkowski", init="random").fit(X)


def test_tsne_n_jobs_and_verbose_are_value_neutral():
    """Both only move the wall clock, so anything short of bit-identical output
    is a bug — the parallel reductions run in point order by construction."""
    rng = np.random.default_rng(23)
    X = rng.normal(size=(80, 5))
    common = dict(perplexity=10.0, init="pca", max_iter=300)
    base = np.asarray(mlrs.TSNE(n_jobs=1, **common).fit_transform(X), dtype=np.float64)
    for n_jobs in (2, 4, -1, None):
        got = np.asarray(mlrs.TSNE(n_jobs=n_jobs, **common).fit_transform(X), dtype=np.float64)
        np.testing.assert_array_equal(got, base, err_msg=f"n_jobs={n_jobs}")
    for verbose in (1, 2):
        got = np.asarray(mlrs.TSNE(verbose=verbose, **common).fit_transform(X), dtype=np.float64)
        np.testing.assert_array_equal(got, base, err_msg=f"verbose={verbose}")
    with pytest.raises(ValueError):
        mlrs.TSNE(n_jobs=0, **common).fit(X)


def test_tsne_barnes_hut_component_cap_matches_sklearn():
    X = np.random.default_rng(29).normal(size=(40, 5))
    with pytest.raises(ValueError):
        mlrs.TSNE(perplexity=8.0, method="barnes_hut", n_components=4, init="random").fit(X)
    # The exact method has no such cap.
    est = mlrs.TSNE(
        perplexity=8.0, method="exact", n_components=4, init="random", max_iter=250
    )
    assert np.asarray(est.fit_transform(X)).shape == (40, 4)
