"""Manifold oracle harness (TSNE-01): the full-binding-path replay of the
committed t-SNE fixtures.

Two tiers (the Rust ``tsne_test.rs`` analog through the Python surface):
- BAND: ``mlrs.TSNE(method='exact', init='pca')`` must reach the sklearn
  embedding's neighborhood-preservation band (``trustworthiness`` within 0.05,
  ``kl_divergence_`` within +0.25) — the end-to-end descent is chaotic, so
  exact equality is meaningless.
- Determinism: PCA-init refit is bit-identical.

f64 fixtures are skipped-with-reason on an f64-incapable backend via
``conftest.requires_f64``. The string-parameter gates (``metric`` / ``init``)
do not load a fixture — they synthesize their design — so they instead run at
both dtypes via ``DESIGN_DTYPES`` and skip only the f64 arm, which keeps every
metric and init string covered on an f64-incapable backend.
"""

import numpy as np
import pytest
from sklearn.manifold import TSNE as SkTSNE

import mlrs
from conftest import dtype_of, fixture_path, requires_f64

TSNE_FIXTURES = ["tsne_f32_seed42", "tsne_f64_seed42"]

# The two design dtypes the string-parameter gates below run at. The f64 arm
# skips on an f64-incapable backend; the f32 arm runs EVERYWHERE, so those gates
# keep covering every metric/init string on rocm rather than vanishing with the
# f64 arm (they previously built a float64 design unconditionally and died in
# the shim's dtype guard rather than skipping).
DESIGN_DTYPES = [np.float32, np.float64]

# The widest float dtype the ACTIVE backend can run. Every gate in this file
# synthesizes its own design instead of loading a dtype-tagged fixture, and
# numpy hands back float64 by default — so without this they all built an f64
# design unconditionally and died in the shim's dtype guard on an f64-incapable
# backend (rocm) rather than adapting or skipping. The gates that are
# mlrs-vs-mlrs (exact-equality, value-neutrality, two-routes-agree) or that
# assert a validation ERROR are dtype-insensitive, so they simply build at this
# dtype and keep running everywhere; only the gates that compare VALUES against
# a float64 sklearn oracle parametrize over `DESIGN_DTYPES` and skip the f64 arm.
DESIGN_DTYPE = np.float64 if mlrs.backend_supports_f64() else np.float32


def _skip_unsupported_dtype(dtype):
    if dtype == np.float64 and not mlrs.backend_supports_f64():
        pytest.skip("backend does not support f64")


def _route_rtol():
    """Tolerance for an mlrs-vs-mlrs "two routes agree" comparison.

    Both sides run at ``DESIGN_DTYPE`` through different code paths that are
    mathematically identical (minkowski p=2 vs euclidean; a callable metric vs
    the precomputed matrix it realizes to), so this bounds path divergence, not
    dtype error — but the divergence is still resolved at the design's
    precision, so it scales with it.
    """
    return 1e-9 if DESIGN_DTYPE == np.float64 else 1e-5


def _kl_rtol(dtype):
    """Relative tolerance on ``kl_divergence_`` for a design of this dtype.

    Both KL gates below compare mlrs at `dtype` against sklearn at float64 — the
    f32 arm is held to the TRUE answer, not to a float32 re-derivation of it, so
    it gates the dtype path rather than just its self-consistency.

    ``1e-10`` is what the f64 arm reaches. The f32 arm is measured (rocm,
    gfx1151) at a worst relative disagreement of 2.7e-7 across all 22 metrics on
    BOTH gates, which is far tighter than the f32 epsilon would suggest because
    the perplexity search sklearn and mlrs share already rounds to float32. The
    band is set an order above that measurement, and still discriminates
    enormously: the frozen KL spans 0.90 (`yule`) to 1.40 (`rogerstanimoto`)
    across the metric set, four orders above this tolerance.
    """
    return 1e-10 if dtype == np.float64 else 5e-6


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


@pytest.mark.parametrize("dtype", DESIGN_DTYPES)
@pytest.mark.parametrize("metric", _TSNE_METRICS)
def test_tsne_every_metric_string_matches_sklearn(metric, dtype):
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
    _skip_unsupported_dtype(dtype)
    X = _metric_design()
    common = dict(
        perplexity=8.0, metric=metric, method="exact",
        max_iter=300, learning_rate=1e-30,
    )
    init = 1e-4 * np.random.default_rng(31).standard_normal((X.shape[0], 2))

    est = mlrs.TSNE(init=init.astype(dtype).copy(), **common)
    emb = np.asarray(est.fit_transform(X.astype(dtype)), dtype=np.float64)
    assert emb.shape == (X.shape[0], 2)
    assert np.isfinite(emb).all(), f"metric={metric}: embedding must be finite"

    # The oracle always runs at float64 — the f32 arm is held to the TRUE KL.
    sk = SkTSNE(init=init.copy(), **common).fit(X)
    assert sk.kl_divergence_ < 1e300, "the oracle hit sklearn's max_iter=250 sentinel"
    np.testing.assert_allclose(
        est.kl_divergence_, sk.kl_divergence_, rtol=_kl_rtol(dtype), atol=0.0,
        err_msg=f"metric={metric}: the joint probabilities differ from sklearn's",
    )


@pytest.mark.parametrize("dtype", DESIGN_DTYPES)
@pytest.mark.parametrize("metric", _TSNE_METRICS)
def test_tsne_every_metric_string_tracks_sklearn_over_a_short_horizon(metric, dtype):
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
    _skip_unsupported_dtype(dtype)
    X = _metric_design()
    common = dict(
        perplexity=8.0, metric=metric, method="exact",
        max_iter=300, learning_rate=1e-3,
    )
    init = 1e-4 * np.random.default_rng(31).standard_normal((X.shape[0], 2))

    est = mlrs.TSNE(init=init.astype(dtype).copy(), **common)
    emb = np.asarray(est.fit_transform(X.astype(dtype)), dtype=np.float64)
    # The oracle always runs at float64 — the f32 arm is held to the TRUE KL.
    sk = SkTSNE(init=init.copy(), **common).fit(X)

    assert np.isfinite(emb).all()
    # The descent must actually have moved, or this would pass vacuously.
    assert not np.allclose(emb, init), f"metric={metric}: the embedding never moved"
    np.testing.assert_allclose(
        est.kl_divergence_, sk.kl_divergence_, rtol=max(_kl_rtol(dtype), 1e-6),
        atol=0.0,
        err_msg=f"metric={metric}: the descent diverged from sklearn's",
    )


@pytest.mark.parametrize("dtype", DESIGN_DTYPES)
@pytest.mark.parametrize("method", ["barnes_hut", "exact"])
def test_tsne_method_matches_sklearn_band_from_shared_init(method, dtype):
    """Both `method` values, started from the SAME injected embedding so the
    only remaining difference is arithmetic.

    Both assertions are BANDS (neighbourhood preservation within 0.05, KL within
    +0.25), so they are dtype-insensitive and the f32 arm is held to exactly the
    same slack — the oracle stays at float64 either way.
    """
    _skip_unsupported_dtype(dtype)
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(c, 0.7, (40, 5)) for c in [0.0, 6.0, -7.0]])
    init = 1e-4 * rng.standard_normal((X.shape[0], 2))

    est = mlrs.TSNE(
        perplexity=10.0, method=method, init=init.astype(dtype).copy(), random_state=0
    )
    emb = np.asarray(est.fit_transform(X.astype(dtype)), dtype=np.float64)
    sk = SkTSNE(perplexity=10.0, method=method, init=init.copy(), random_state=0).fit(X)

    t_mlrs = _trustworthiness(X, emb)
    t_sk = _trustworthiness(X, sk.embedding_)
    assert t_mlrs >= t_sk - 0.05, f"method={method}: trust {t_mlrs} vs sklearn {t_sk}"
    assert 0.0 < est.kl_divergence_ <= sk.kl_divergence_ + 0.25


@pytest.mark.parametrize("dtype", DESIGN_DTYPES)
@pytest.mark.parametrize("init", ["pca", "random", "array"])
def test_tsne_every_init_string_reaches_sklearns_band(init, dtype):
    _skip_unsupported_dtype(dtype)
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(c, 0.7, (40, 5)) for c in [0.0, 6.0, -7.0]])
    arg = 1e-4 * rng.standard_normal((X.shape[0], 2)) if init == "array" else init

    est_init = arg.astype(dtype).copy() if init == "array" else arg
    est = mlrs.TSNE(perplexity=10.0, method="exact", init=est_init, random_state=0)
    emb = np.asarray(est.fit_transform(X.astype(dtype)), dtype=np.float64)
    sk_init = arg.copy() if init == "array" else arg
    sk = SkTSNE(perplexity=10.0, method="exact", init=sk_init, random_state=0).fit(X)

    # A neighbourhood-preservation band, not a value gate — it is the dtype-
    # insensitive tier, so both arms share the 0.05 slack (measured f32 margin
    # on rocm is ~0.048-0.052, i.e. the band is doing real work at both dtypes).
    t_mlrs = _trustworthiness(X, emb)
    t_sk = _trustworthiness(X, sk.embedding_)
    assert t_mlrs >= t_sk - 0.05, f"init={init}: trust {t_mlrs} vs sklearn {t_sk}"


def test_tsne_learning_rate_auto_equals_the_resolved_constant():
    """`'auto'` is `max(n / early_exaggeration / 4, 50)` — assert EXACT equality
    with that number rather than a band, which could not tell it from a nearby
    constant."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 4)).astype(DESIGN_DTYPE)
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
    reproduce plain euclidean exactly.

    ## Why the horizon is SHORT
    This is a claim about the METRIC — that `minkowski(p=2)` and `euclidean`
    compute the same distances — so it has to be read before t-SNE's chaos
    buries it. The two routes reach those distances by different formulas
    (`sqrt(Σd²)` vs `(Σ|d|^p)^(1/p)`), which agree bit-for-bit in float64 but
    differ in the last ulp or two in float32. Over the full 300-iteration
    descent that ulp amplifies exponentially — measured on this design at
    float32: 4.7e-10 apart at 1 iteration, 1.1e-4 at 10, and fully diverged
    (relative difference ~1.9) by 50. A long horizon therefore tests the
    chaos, not the metric, and can only be made to pass by pinning the test to
    float64. Two iterations isolate the metric and gate it at BOTH dtypes.
    """
    rng = np.random.default_rng(11)
    X = rng.normal(size=(50, 4)).astype(DESIGN_DTYPE)
    common = dict(perplexity=8.0, init="pca", method="exact", max_iter=2)

    eu = np.asarray(mlrs.TSNE(metric="euclidean", **common).fit_transform(X), dtype=np.float64)
    mk2 = np.asarray(
        mlrs.TSNE(metric="minkowski", metric_params={"p": 2.0}, **common).fit_transform(X),
        dtype=np.float64,
    )
    mk3 = np.asarray(
        mlrs.TSNE(metric="minkowski", metric_params={"p": 3.0}, **common).fit_transform(X),
        dtype=np.float64,
    )
    tol = _route_rtol()
    np.testing.assert_allclose(mk2, eu, rtol=tol, atol=tol)
    assert not np.allclose(mk3, eu), "p=3 must not reproduce the euclidean embedding"

    with pytest.raises(ValueError, match="metric_params"):
        mlrs.TSNE(metric="minkowski", metric_params={"bogus": 1.0}, **common).fit(X)


def test_tsne_callable_metric_matches_the_precomputed_route():
    """A callable `metric` is realized by evaluating it into a dense distance
    matrix; that must agree with passing the same matrix as
    `metric='precomputed'`.

    Short horizon for the same reason as
    :func:`test_tsne_metric_params_reach_the_pair_loop` — this gates the
    REALIZATION of the callable, and the two routes build the matrix by
    different arithmetic, so a long chaotic descent would turn a last-ulp
    float32 difference into a total divergence and force the gate to float64.
    """
    rng = np.random.default_rng(13)
    X = rng.normal(size=(40, 3)).astype(DESIGN_DTYPE)

    def cityblock(u, v):
        return float(np.abs(u - v).sum())

    common = dict(perplexity=8.0, init="random", method="exact", max_iter=2, random_state=0)
    via_callable = np.asarray(
        mlrs.TSNE(metric=cityblock, **common).fit_transform(X), dtype=np.float64
    )
    dist = np.abs(X[:, None, :] - X[None, :, :]).sum(-1).astype(DESIGN_DTYPE)
    via_precomputed = np.asarray(
        mlrs.TSNE(metric="precomputed", **common).fit_transform(dist), dtype=np.float64
    )
    tol = _route_rtol()
    np.testing.assert_allclose(via_callable, via_precomputed, rtol=tol, atol=tol)


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
    emb = np.asarray(est.fit_transform(X.astype(DESIGN_DTYPE)), dtype=np.float64)
    assert emb.shape == (40, 2)
    assert np.isfinite(emb).all()

    # ...and must agree with sklearn's own callable + PCA-init run, which is
    # deterministic once the step size keeps the trajectory short. The oracle
    # stays at float64 whatever the design dtype is, so the tolerance follows
    # the design (`_kl_rtol`) rather than being pinned to the f64 figure.
    common = dict(perplexity=8.0, max_iter=300, learning_rate=1e-3, init="pca")
    a = mlrs.TSNE(metric=cityblock, method="exact", **common)
    a.fit(X.astype(DESIGN_DTYPE))
    b = SkTSNE(metric=cityblock, method="exact", **common).fit(X)
    np.testing.assert_allclose(
        a.kl_divergence_, b.kl_divergence_, rtol=max(_kl_rtol(DESIGN_DTYPE), 1e-6)
    )

    # The explicit `precomputed` spelling still refuses `init='pca'`.
    dist = np.abs(X[:, None, :] - X[None, :, :]).sum(-1).astype(DESIGN_DTYPE)
    with pytest.raises(ValueError, match="pca"):
        mlrs.TSNE(perplexity=8.0, metric="precomputed", init="pca").fit(dist)


def test_tsne_haversine_and_precomputed_geometry_rules():
    rng = np.random.default_rng(17)
    X2 = rng.uniform(-1.0, 1.0, size=(40, 2)).astype(DESIGN_DTYPE)
    # haversine is defined on exactly 2 features and must fit there...
    est = mlrs.TSNE(perplexity=8.0, metric="haversine", init="random", max_iter=250)
    assert np.isfinite(np.asarray(est.fit_transform(X2), dtype=np.float64)).all()
    # ...and be rejected anywhere else.
    with pytest.raises(ValueError):
        mlrs.TSNE(perplexity=8.0, metric="haversine", init="random").fit(
            rng.normal(size=(40, 5)).astype(DESIGN_DTYPE)
        )
    # `precomputed` needs a square X, and rules out init='pca'.
    with pytest.raises(ValueError):
        mlrs.TSNE(perplexity=8.0, metric="precomputed", init="random").fit(
            rng.normal(size=(40, 5)).astype(DESIGN_DTYPE)
        )
    dist = np.abs(X2[:, None, :] - X2[None, :, :]).sum(-1).astype(DESIGN_DTYPE)
    with pytest.raises(ValueError, match="pca"):
        mlrs.TSNE(perplexity=8.0, metric="precomputed", init="pca").fit(dist)


def test_tsne_wminkowski_is_rejected_like_sklearn():
    """sklearn accepts the string at validation and then fails, because scipy
    removed the metric. mlrs fails too rather than silently substituting one."""
    X = np.random.default_rng(19).normal(size=(30, 4)).astype(DESIGN_DTYPE)
    with pytest.raises(ValueError):
        mlrs.TSNE(perplexity=8.0, metric="wminkowski", init="random").fit(X)


def test_tsne_n_jobs_and_verbose_are_value_neutral():
    """Both only move the wall clock, so anything short of bit-identical output
    is a bug — the parallel reductions run in point order by construction."""
    rng = np.random.default_rng(23)
    X = rng.normal(size=(80, 5)).astype(DESIGN_DTYPE)
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
    X = np.random.default_rng(29).normal(size=(40, 5)).astype(DESIGN_DTYPE)
    with pytest.raises(ValueError):
        mlrs.TSNE(perplexity=8.0, method="barnes_hut", n_components=4, init="random").fit(X)
    # The exact method has no such cap.
    est = mlrs.TSNE(
        perplexity=8.0, method="exact", n_components=4, init="random", max_iter=250
    )
    assert np.asarray(est.fit_transform(X)).shape == (40, 4)
