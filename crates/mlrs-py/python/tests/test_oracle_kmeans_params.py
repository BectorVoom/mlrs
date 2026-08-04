"""KMeans full-parameter surface through the Python shim, vs LIVE sklearn.

Every string-valued ``sklearn.cluster.KMeans`` parameter is exercised here
against a live ``sklearn`` instance rather than a committed fixture, because
what has to be checked is exactly the thing a fixture would freeze away: that
mlrs's resolution rules still agree with sklearn's as sklearn evolves them.

**Why live sklearn and not a value fixture for ``init``.** mlrs's k-means++ is
the D^2-weighted host sampler seeded by SplitMix64; sklearn's draws from a numpy
``RandomState`` and additionally runs ``2 + log(k)`` greedy local trials per
center. Same distribution, different stream -- so the two libraries start from
DIFFERENT centers and, on a hard design, would land in different local optima.
The oracle is therefore run on WELL-SEPARATED blobs, where the k-means
objective has a single basin every reasonable init falls into: there the fitted
values must agree to 1e-5, and any disagreement is a real defect rather than an
RNG artifact. ``algorithm`` needs no such care -- Elkan is an exact
acceleration, so it is compared against sklearn under an EXPLICIT init where
both libraries provably run the same iteration.

f64 paths are skipped-with-reason on an f64-incapable backend via
``conftest.requires_f64``.
"""

import numpy as np
import pytest
from sklearn.cluster import KMeans as SkKMeans

import mlrs
from conftest import label_perm_allclose, label_perm_remap, requires_f64

ATOL = 1e-5


def blobs(n=240, d=4, k=5, seed=0, spread=0.4, scale=100.0):
    """Well-separated Gaussian blobs: `k` centers spread over `[0, scale)^d`
    with noise an order of magnitude below the inter-center distance, so the
    k-means objective has one basin and every init reaches it."""
    rs = np.random.RandomState(seed)
    centers = rs.uniform(0.0, scale, size=(k, d))
    labels = np.arange(n) % k
    return (
        np.ascontiguousarray(
            centers[labels] + rs.normal(0.0, spread, size=(n, d)),
            dtype=np.float64,
        ),
        labels,
    )


def assert_matches_sklearn(est, sk, X, what):
    """mlrs's fit equals sklearn's up to a cluster-id permutation."""
    labels = np.asarray(est.labels_).astype(np.int64).ravel()
    ref = np.asarray(sk.labels_).astype(np.int64).ravel()
    assert label_perm_allclose(labels, ref), f"{what}: labels_ are not a permutation"

    mapping = label_perm_remap(labels, ref)
    assert mapping is not None, f"{what}: no label bijection"
    centers = np.asarray(est.cluster_centers_, dtype=np.float64)
    ref_centers = np.asarray(sk.cluster_centers_, dtype=np.float64)
    aligned = np.empty_like(ref_centers)
    for ours, theirs in mapping.items():
        aligned[theirs] = centers[ours]
    assert np.allclose(aligned, ref_centers, atol=ATOL, rtol=0.0), (
        f"{what}: cluster_centers_ differ (max "
        f"{np.abs(aligned - ref_centers).max():.3e})"
    )

    inertia = float(est.inertia_)
    ref_inertia = float(sk.inertia_)
    assert abs(inertia - ref_inertia) <= ATOL * (1.0 + abs(ref_inertia)), (
        f"{what}: inertia_ {inertia!r} != sklearn {ref_inertia!r}"
    )


# ---------------------------------------------------------------------------
# init -- the two string strategies
# ---------------------------------------------------------------------------


# How many restarts each init needs before "both libraries reach the global
# optimum" is a property of the DESIGN rather than a coin flip.
#
# k-means++ is single-basin here by construction: its D^2 weighting makes a
# second center in an already-covered blob vanishingly unlikely, so one restart
# already lands on the optimum (measured: 20/20 seeds, every n_init).
#
# `random` is NOT. It draws k rows uniformly, so with k blobs the chance of one
# center per blob is only ~k!/k^k per restart; empty-cluster relocation rescues
# many of the rest, but at n_init=10 the two libraries still diverged on 2 of 20
# seeds -- not a defect (mlrs's draw is verified uniform in
# `mlrs-backend/tests/kmeanspp_test.rs`), just an RNG lottery that would make a
# VALUE comparison flaky. 30 restarts makes it a real oracle: 20/20 seeds, both
# libraries, at k=3 and k=5.
INIT_RESTARTS = {"k-means++": 1, "random": 30}


@pytest.mark.parametrize("init", ["k-means++", "random"])
@requires_f64
def test_init_string_matches_sklearn(init):
    """Both ``init`` strings reach sklearn's fit once the design is genuinely
    single-basin for that init (see ``INIT_RESTARTS``)."""
    X, truth = blobs(seed=1)
    n_init = INIT_RESTARTS[init]
    kw = dict(n_clusters=5, max_iter=300, tol=1e-4, random_state=42)
    est = mlrs.KMeans(init=init, n_init=n_init, **kw).fit(X)
    sk = SkKMeans(init=init, n_init=n_init, **kw).fit(X)

    assert_matches_sklearn(est, sk, X, f"init={init}")
    # Both must also have recovered the GENERATIVE partition -- the check that
    # the agreement above is agreement on the right answer, not on a shared
    # failure mode.
    assert label_perm_allclose(
        np.asarray(est.labels_).astype(np.int64).ravel(), truth
    ), f"init={init}: did not recover the true blobs"
    assert label_perm_allclose(
        np.asarray(sk.labels_).astype(np.int64).ravel(), truth
    ), f"init={init}: sklearn did not recover the true blobs either — the "
    "design is no longer single-basin for this init"


@requires_f64
def test_init_array_matches_sklearn():
    """An explicit ``(k, n_features)`` init array runs the SAME iteration in
    both libraries, so this is an exact value oracle -- no basin argument
    needed."""
    X, _ = blobs(seed=2)
    init = np.ascontiguousarray(X[[3, 17, 40, 61, 92]], dtype=np.float64)
    kw = dict(n_clusters=5, max_iter=300, tol=1e-4, random_state=0)

    est = mlrs.KMeans(init=init, **kw).fit(X)
    sk = SkKMeans(init=init, n_init=1, **kw).fit(X)
    assert_matches_sklearn(est, sk, X, "init=<array>")

    # sklearn forces one run for an explicit init regardless of n_init.
    assert est._n_init == 1
    assert mlrs.KMeans(init=init, n_init=7, **kw).fit(X)._n_init == 1


@requires_f64
def test_init_callable_matches_sklearn():
    """A callable ``init(X, k, random_state=...)`` is evaluated by the shim and
    passed down as an explicit array -- exactly what sklearn's
    ``_init_centroids`` does with it.

    Two sklearn quirks the callable below is written around, both real and both
    documented in ``mlrs.cluster.KMeans._resolve_init``:

    * sklearn passes its callable the MEAN-CENTERED X (an internal numerical
      detail that leaks into the user's callback); mlrs passes the caller's own
      X. The callable here selects rows by INDEX, so it is invariant to that
      difference and the comparison stays well-posed.
    * sklearn runs ``check_array(centers, copy=False)`` on the result and then
      iterates INTO that buffer, so a callable returning a VIEW of X has
      sklearn overwrite its own input mid-fit (it then burns all 300 iterations
      and reports a worse inertia than the one at its own cluster means).
      ``.copy()`` below is load-bearing for sklearn, not for mlrs.
    """
    X, _ = blobs(seed=3)

    def pick_first(X_, k, random_state=None):
        return np.ascontiguousarray(X_[:k], dtype=np.float64).copy()

    kw = dict(n_clusters=5, max_iter=300, tol=1e-4, random_state=0)
    est = mlrs.KMeans(init=pick_first, **kw).fit(X)
    sk = SkKMeans(init=pick_first, n_init=1, **kw).fit(X)
    assert_matches_sklearn(est, sk, X, "init=<callable>")


def test_init_rejects_unknown_string():
    """Unrecognised ``init`` strings are rejected, as sklearn's ``StrOptions``
    does -- including ``'kmeans++'``, the spelling users actually mistype."""
    X, _ = blobs(n=60, k=3, seed=4)
    for bad in ["kmeans++", "K-Means++", "auto", ""]:
        with pytest.raises(Exception, match="(?i)init"):
            mlrs.KMeans(n_clusters=3, init=bad).fit(X)


def test_init_array_wrong_shape_is_rejected():
    X, _ = blobs(n=60, d=4, k=3, seed=5)
    with pytest.raises(ValueError, match="init has shape"):
        mlrs.KMeans(n_clusters=3, init=np.zeros((2, 4))).fit(X)


# ---------------------------------------------------------------------------
# n_init
# ---------------------------------------------------------------------------


@requires_f64
def test_n_init_auto_resolution_matches_sklearn():
    """``n_init='auto'`` resolves to sklearn's ``_n_init`` for every ``init``
    form: 1 for k-means++, 10 for random, 1 for an explicit array."""
    X, _ = blobs(n=120, k=4, seed=6)
    init_arr = np.ascontiguousarray(X[[0, 1, 2, 3]], dtype=np.float64)

    for init, expected in [
        ("k-means++", 1),
        ("random", 10),
        (init_arr, 1),
    ]:
        est = mlrs.KMeans(n_clusters=4, init=init, n_init="auto").fit(X)
        sk = SkKMeans(n_clusters=4, init=init, n_init="auto").fit(X)
        name = init if isinstance(init, str) else "<array>"
        assert est._n_init == expected, f"init={name}: mlrs _n_init"
        assert sk._n_init == expected, f"init={name}: sklearn _n_init moved"


@requires_f64
def test_n_init_explicit_count_matches_sklearn():
    """An explicit ``n_init`` reaches the same fit as sklearn's.

    Run under ``k-means++`` deliberately: the point being tested is that an
    explicit count is HONOURED and the best-of-N selection agrees, not that two
    different RNG streams happen to draw the same lucky restart (see
    ``INIT_RESTARTS``).
    """
    X, _ = blobs(seed=7)
    kw = dict(
        n_clusters=5, init="k-means++", max_iter=300, tol=1e-4, random_state=42
    )
    est = mlrs.KMeans(n_init=4, **kw).fit(X)
    sk = SkKMeans(n_init=4, **kw).fit(X)
    assert est._n_init == 4
    assert sk._n_init == 4
    assert_matches_sklearn(est, sk, X, "n_init=4")


@requires_f64
def test_n_init_more_restarts_never_worse():
    """The parameter's whole contract: on a design where a single random init
    routinely lands in a local optimum, more restarts cannot increase inertia."""
    rs = np.random.RandomState(3)
    # Unevenly spaced elongated clusters -- init genuinely matters here.
    X = np.ascontiguousarray(
        np.column_stack(
            [
                (np.arange(300) % 6) ** 2 * 1.5 + rs.uniform(0, 3, 300),
                rs.uniform(0, 12, 300),
            ]
        ),
        dtype=np.float64,
    )
    kw = dict(n_clusters=6, init="random", random_state=3)
    one = float(mlrs.KMeans(n_init=1, **kw).fit(X).inertia_)
    ten = float(mlrs.KMeans(n_init=10, **kw).fit(X).inertia_)
    assert ten <= one * (1.0 + 1e-12), f"n_init=10 inertia {ten} > n_init=1 {one}"


def test_n_init_rejects_bad_values():
    X, _ = blobs(n=60, k=3, seed=8)
    with pytest.raises(ValueError, match="n_init"):
        mlrs.KMeans(n_clusters=3, n_init="ten").fit(X)
    with pytest.raises(Exception, match="n_init"):
        mlrs.KMeans(n_clusters=3, n_init=0).fit(X)


# ---------------------------------------------------------------------------
# algorithm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
@requires_f64
def test_algorithm_matches_sklearn_under_explicit_init(algorithm):
    """Both ``algorithm`` values reproduce sklearn's fit. Under an explicit
    init this is an EXACT oracle: the two libraries run the same iteration from
    the same centers, and Elkan only prunes distances that cannot win."""
    X, _ = blobs(seed=9)
    init = np.ascontiguousarray(X[[5, 23, 44, 70, 101]], dtype=np.float64)
    kw = dict(n_clusters=5, init=init, max_iter=300, tol=1e-4, random_state=0)

    est = mlrs.KMeans(algorithm=algorithm, **kw).fit(X)
    sk = SkKMeans(algorithm=algorithm, n_init=1, **kw).fit(X)
    assert_matches_sklearn(est, sk, X, f"algorithm={algorithm}")
    assert est._algorithm == algorithm


@requires_f64
def test_algorithm_arms_agree_exactly():
    """Elkan and Lloyd must return the IDENTICAL labeling, iteration count and
    inertia -- a pruning bug that happened to land on an equally good optimum
    still fails here."""
    X, _ = blobs(seed=10)
    init = np.ascontiguousarray(X[[1, 30, 55, 88, 120]], dtype=np.float64)
    kw = dict(n_clusters=5, init=init, max_iter=300, tol=1e-4, random_state=0)

    lloyd = mlrs.KMeans(algorithm="lloyd", **kw).fit(X)
    elkan = mlrs.KMeans(algorithm="elkan", **kw).fit(X)

    assert np.array_equal(
        np.asarray(lloyd.labels_).ravel(), np.asarray(elkan.labels_).ravel()
    ), "elkan and lloyd labels_ differ"
    assert lloyd.n_iter_ == elkan.n_iter_, "elkan and lloyd iteration counts differ"
    assert np.allclose(
        np.asarray(lloyd.cluster_centers_, dtype=np.float64),
        np.asarray(elkan.cluster_centers_, dtype=np.float64),
        atol=ATOL,
        rtol=0.0,
    )
    assert abs(float(lloyd.inertia_) - float(elkan.inertia_)) <= ATOL * (
        1.0 + abs(float(lloyd.inertia_))
    )


@requires_f64
def test_algorithm_elkan_degrades_to_lloyd_at_k1():
    """sklearn silently rewrites ``algorithm='elkan'`` to ``'lloyd'`` when
    ``n_clusters == 1`` (no other center to bound against); mlrs matches, and
    the single center is still the mean of X."""
    X, _ = blobs(n=80, d=3, k=1, seed=11)
    est = mlrs.KMeans(n_clusters=1, algorithm="elkan", random_state=0).fit(X)
    assert est._algorithm == "lloyd"
    assert np.allclose(
        np.asarray(est.cluster_centers_, dtype=np.float64).ravel(),
        X.mean(axis=0),
        atol=1e-5,
        rtol=0.0,
    )


def test_algorithm_rejects_unknown_string():
    X, _ = blobs(n=60, k=3, seed=12)
    for bad in ["full", "auto", "Elkan", ""]:
        with pytest.raises(Exception, match="(?i)algorithm"):
            mlrs.KMeans(n_clusters=3, algorithm=bad).fit(X)


# ---------------------------------------------------------------------------
# verbose / copy_x / random_state / n_iter_ / get_params
# ---------------------------------------------------------------------------


@requires_f64
def test_verbose_and_copy_x_are_inert():
    """Both are accepted for signature compatibility and change nothing: mlrs
    never prints from the library and never writes into the caller's X."""
    X, _ = blobs(n=120, k=4, seed=13)
    before = X.copy()
    kw = dict(n_clusters=4, init="k-means++", random_state=0)

    base = mlrs.KMeans(**kw).fit(X)
    for verbose, copy_x in [(1, True), (0, False), (2, False)]:
        got = mlrs.KMeans(verbose=verbose, copy_x=copy_x, **kw).fit(X)
        assert np.array_equal(
            np.asarray(got.labels_).ravel(), np.asarray(base.labels_).ravel()
        ), f"verbose={verbose} copy_x={copy_x} changed labels_"
        assert float(got.inertia_) == float(base.inertia_)
    assert np.array_equal(X, before), "fit must never write into the caller's X"


@requires_f64
def test_random_state_is_reproducible():
    X, _ = blobs(seed=14)
    kw = dict(n_clusters=5, init="random", n_init=3)
    a = mlrs.KMeans(random_state=1234, **kw).fit(X)
    b = mlrs.KMeans(random_state=1234, **kw).fit(X)
    assert np.array_equal(
        np.asarray(a.labels_).ravel(), np.asarray(b.labels_).ravel()
    )
    assert float(a.inertia_) == float(b.inertia_)


@requires_f64
def test_n_iter_is_the_winning_run_not_the_sum():
    """``n_iter_`` reports the winning restart's iteration count, so a
    10-restart fit must not report roughly ten times the 1-restart count."""
    X, _ = blobs(seed=15)
    kw = dict(n_clusters=5, init="random", random_state=9)
    one = mlrs.KMeans(n_init=1, **kw).fit(X)
    ten = mlrs.KMeans(n_init=10, **kw).fit(X)
    assert one.n_iter_ >= 1
    assert ten.n_iter_ < 300
    # sklearn's n_iter_ for the same design is single-run sized too.
    sk = SkKMeans(n_init=10, **kw).fit(X)
    assert ten.n_iter_ <= 5 * max(int(sk.n_iter_), 1), (
        f"n_iter_ {ten.n_iter_} looks like a cross-restart sum "
        f"(sklearn reports {sk.n_iter_})"
    )


def test_get_params_covers_sklearns_ctor_surface():
    """Every ``sklearn.cluster.KMeans`` ctor parameter is present in the shim's
    ``get_params()`` with sklearn's default -- the check that catches a
    parameter added upstream and never mirrored here."""
    ours = mlrs.KMeans().get_params()
    theirs = SkKMeans().get_params()
    missing = sorted(set(theirs) - set(ours))
    assert not missing, f"KMeans is missing sklearn parameters: {missing}"
    for name, default in theirs.items():
        assert ours[name] == default, (
            f"KMeans.{name} default is {ours[name]!r}, sklearn's is {default!r}"
        )
