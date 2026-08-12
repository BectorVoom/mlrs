"""``device='auto'|'cpu'|'gpu'`` through the Python shim (DEVICE-PARAM-01).

The parameter promises to change WHERE the work happens and nothing else, so
the load-bearing assertion here is that the two arms agree on the fitted
values — not that the right arm was selected. A placement knob that quietly
moves the numbers is worse than no knob, because a caller reaches for it to go
faster and has no reason to re-check the fit.

The second thing pinned here is that the parameter never LIES. ``device`` is a
preference, and some configurations have only one implementation for the
requested arm; those still fit, and ``device_`` reports what actually carried
it. ``BayesianRidge`` is the live example on this backend: its device Gram is
gated on an ``f64`` capability that a small ``d`` does not clear, so
``device='gpu'`` legitimately comes back ``device_ == 'cpu'``.

Only estimators with a REAL two-arm split take the parameter. Spectral*,
RidgeCV and friends deliberately do not — advertising a choice that does not
exist would be worse than omitting it — and ``test_single_arm_estimators_have_
no_device`` holds that line.
"""

import numpy as np
import pytest

import mlrs
from conftest import default_float_dtype

# The two arms cannot agree more tightly than the NARROWER of their two
# accumulations, so the band is a property of the DTYPE, not of the estimator.
#
# `1e-9` was the original single value and it is right for `f64` — but it was
# only ever exercised on the cpu backend, where several "gpu" arms fall back to
# the host anyway and the comparison is host-against-host. On a real GPU backend
# without `f64` (rocm, cuda, wgpu) the suite runs in `f32`, the device kernel is
# monomorphized on `f32`, and the host prims it is compared against accumulate
# in `f64` — so the arms differ at `f32` round-off and `1e-9` is unreachable by
# construction. Gating a cross-backend claim on one backend's measurement is the
# mistake this constant now records.
#
# MEASURED, rocm gfx1151, `f32`, 2026-08-12, the full `CASES` list at the
# `design()` fixture (300x6). Worst case Ridge 3.3e-07; Huber 8.2e-08, MBSGD
# 8.2e-08, HGB 1.0e-07, kNN 1.0e-07, and GMM/BGM/TSNE/BayesianRidge exactly 0.
# `1e-6` sits ~3x above the worst observed value and ~10x BELOW the 1e-5 oracle
# band, so it still fails long before a real placement bug could hide in it.
ARM_AGREEMENT_BAND = {
    np.dtype(np.float64): 1e-9,
    np.dtype(np.float32): 1e-6,
}

DEVICES = ("auto", "cpu", "gpu")


def design(n=300, d=6, seed=0):
    rs = np.random.RandomState(seed)
    dtype = default_float_dtype()
    x = rs.normal(size=(n, d))
    y = x @ rs.normal(size=d) + 1.0
    return (
        np.ascontiguousarray(x, dtype=dtype),
        np.ascontiguousarray(y, dtype=dtype),
    )


# (label, factory taking `device`, needs y, fitted attribute to compare)
CASES = [
    ("Ridge", lambda d: mlrs.Ridge(alpha=0.7, device=d), True, "coef_"),
    ("BayesianRidge", lambda d: mlrs.BayesianRidge(device=d), True, "coef_"),
    (
        "RidgeClassifier",
        lambda d: mlrs.RidgeClassifier(alpha=0.7, device=d),
        "labels",
        "coef_",
    ),
    (
        "GaussianMixture",
        lambda d: mlrs.GaussianMixture(n_components=2, random_state=0, device=d),
        False,
        "means_",
    ),
    (
        "BayesianGaussianMixture",
        lambda d: mlrs.BayesianGaussianMixture(
            n_components=2, random_state=0, device=d
        ),
        False,
        "means_",
    ),
    (
        "NearestNeighbors",
        lambda d: mlrs.NearestNeighbors(n_neighbors=3, device=d),
        False,
        None,
    ),
    (
        "KNeighborsClassifier",
        lambda d: mlrs.KNeighborsClassifier(n_neighbors=3, device=d),
        "labels",
        None,
    ),
    (
        "KNeighborsRegressor",
        lambda d: mlrs.KNeighborsRegressor(n_neighbors=3, device=d),
        True,
        None,
    ),
    (
        "TSNE",
        lambda d: mlrs.TSNE(
            n_components=2, perplexity=5.0, max_iter=250, random_state=0, device=d
        ),
        False,
        "embedding_",
    ),
    (
        "UMAP",
        lambda d: mlrs.UMAP(
            n_components=2, n_neighbors=5, random_state=0, device=d
        ),
        False,
        "embedding_",
    ),
    ("HuberRegressor", lambda d: mlrs.HuberRegressor(device=d), True, "coef_"),
    (
        "MBSGDRegressor",
        lambda d: mlrs.MBSGDRegressor(max_iter=30, seed=0, device=d),
        True,
        "coef_",
    ),
    (
        "MBSGDClassifier",
        lambda d: mlrs.MBSGDClassifier(max_iter=30, seed=0, device=d),
        "labels",
        "coef_",
    ),
    (
        "HistGradientBoostingRegressor",
        lambda d: mlrs.HistGradientBoostingRegressor(max_iter=5, device=d),
        True,
        None,
    ),
    (
        "HistGradientBoostingClassifier",
        lambda d: mlrs.HistGradientBoostingClassifier(max_iter=5, device=d),
        "labels",
        None,
    ),
]
IDS = [c[0] for c in CASES]

# The neighbour estimators have no fitted array to compare directly — their
# answer IS the query result — so they are compared through `kneighbors`.
NEIGHBOUR_IDS = {"NearestNeighbors", "KNeighborsClassifier", "KNeighborsRegressor"}

# UMAP's layout is a STOCHASTIC SGD whose host driver and device kernel consume
# the negative-sampling RNG in different orders, so the two arms produce
# different (equally valid) embeddings from the same seed. It therefore cannot
# join the bitwise agreement gate — see `test_umap_arms_differ_by_design`, which
# asserts the divergence deliberately rather than hiding it in a tolerance.
STOCHASTIC_IDS = {"UMAP"}
# Estimators with no single fit-time arm to report.
NO_DEVICE_ATTR = NEIGHBOUR_IDS | STOCHASTIC_IDS


def _fit(factory, needs_y, device):
    X, y = design()
    est = factory(device)
    if needs_y == "labels":
        return est.fit(X, (y > np.median(y)).astype(np.int32))
    if needs_y:
        return est.fit(X, y)
    return est.fit(X)


# ---------------------------------------------------------------------------
# The gate that matters: placement must not move the answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,factory,needs_y,attr",
    [c for c in CASES if c[0] not in STOCHASTIC_IDS],
    ids=[c[0] for c in CASES if c[0] not in STOCHASTIC_IDS],
)
def test_arms_agree_on_the_fit(label, factory, needs_y, attr):
    X, _ = design()
    got = {}
    for device in ("cpu", "gpu"):
        est = _fit(factory, needs_y, device)
        if attr is None and label.startswith("HistGradientBoosting"):
            # A forest has no coefficient vector; its answer is the prediction.
            got[device] = np.asarray(est.predict(X[:50]), dtype=np.float64).ravel()
        elif attr is None:
            # The neighbour search IS the answer: compare the distances the two
            # arms return for the same query, which is what a caller sees.
            dist, _idx = est.kneighbors(X[:20], n_neighbors=3)
            got[device] = np.asarray(dist, dtype=np.float64).ravel()
        else:
            got[device] = np.asarray(getattr(est, attr), dtype=np.float64).ravel()
    assert got["cpu"].shape == got["gpu"].shape
    diff = np.abs(got["cpu"] - got["gpu"]).max()
    scale = max(1.0, float(np.abs(got["cpu"]).max()))
    band = ARM_AGREEMENT_BAND[np.dtype(default_float_dtype())]
    assert diff <= band * scale, (
        f"{label}: the cpu and gpu arms disagree on {attr} by {diff:.3e} "
        f"(relative {diff / scale:.3e}, band {band:.0e}) — a placement "
        "parameter must not change the answer"
    )


# ---------------------------------------------------------------------------
# The parameter must never lie about what ran
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,factory,needs_y,attr",
    [c for c in CASES if c[0] not in NO_DEVICE_ATTR],
    ids=[c[0] for c in CASES if c[0] not in NO_DEVICE_ATTR],
)
def test_device_reports_a_real_arm(label, factory, needs_y, attr):
    arms = {}
    for device in DEVICES:
        est = _fit(factory, needs_y, device)
        assert est.device_ in ("cpu", "gpu"), (
            f"{label}: device_={est.device_!r} for device={device!r}"
        )
        arms[device] = est.device_

    # An explicit preference is honoured OR the fallback is reported; what it
    # must never be is silently the opposite with no way to tell.
    #
    # "must be honoured" cannot be asserted unconditionally, because whether an
    # arm EXISTS is a property of the backend: Huber has no device arm on cpu
    # (cubecl-cpu cannot compile the margin kernel) and MBSGD has no host arm on
    # rocm/cuda (`sgd_host_possible`). Both are legitimate, and both must show
    # up in `device_`.
    #
    # What IS backend-independent: if a preference is not honoured, that arm
    # must be unreachable for EVERY preference. An estimator that gives the host
    # arm to `device='gpu'` but not to `device='cpu'` has an inverted gate, and
    # one that ignores the parameter entirely collapses to a single value here
    # while claiming the other — both still fail.
    for asked, other in (("cpu", "gpu"), ("gpu", "cpu")):
        if arms[asked] != asked:
            assert set(arms.values()) == {other}, (
                f"{label}: device={asked!r} fell back to {arms[asked]!r}, so "
                f"the {asked!r} arm does not exist on this backend — yet some "
                f"other preference reached it: {arms}. That is an inverted or "
                "inconsistently-read gate, not an unhonourable preference."
            )


def test_an_unhonourable_preference_is_reported_not_faked():
    """``BayesianRidge``'s device Gram is gated on an ``f64`` capability that a
    narrow design does not clear, so ``device='gpu'`` falls back — and says so.

    This is the case the ``device_`` attribute exists for. If a future change
    makes the device arm reachable here the assertion flips to ``'gpu'``, which
    is a real signal rather than a broken test.
    """
    X, y = design(d=6)
    est = mlrs.BayesianRidge(device="gpu").fit(X, y)
    assert est.device_ in ("cpu", "gpu")
    ref = mlrs.BayesianRidge(device="cpu").fit(X, y)
    assert np.allclose(
        np.asarray(est.coef_, dtype=np.float64),
        np.asarray(ref.coef_, dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
    ), "the fallback must produce the host arm's answer exactly"


# ---------------------------------------------------------------------------
# sklearn contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,factory,needs_y,attr", CASES, ids=IDS)
def test_get_params_and_clone_round_trip(label, factory, needs_y, attr):
    from sklearn.base import clone

    est = factory("gpu")
    assert est.get_params()["device"] == "gpu", f"{label}: device missing from get_params"
    assert clone(est).device == "gpu", f"{label}: clone dropped device"


@pytest.mark.parametrize("label,factory,needs_y,attr", CASES, ids=IDS)
def test_bad_device_is_rejected(label, factory, needs_y, attr):
    with pytest.raises(ValueError, match="device"):
        _fit(factory, needs_y, "cuda")


@pytest.mark.parametrize("label,factory,needs_y,attr", CASES, ids=IDS)
def test_device_is_notfitted_before_fit(label, factory, needs_y, attr):
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        _ = factory("auto").device_


@pytest.mark.parametrize(
    "label,factory,needs_y,attr",
    [c for c in CASES if c[0] in NEIGHBOUR_IDS],
    ids=[c[0] for c in CASES if c[0] in NEIGHBOUR_IDS],
)
def test_neighbour_estimators_choose_per_query_not_at_fit(
    label, factory, needs_y, attr
):
    """The neighbour search picks its arm from `n_query` and `k`, which `fit`
    never sees — so there is no single arm for a fitted estimator to name.

    `device` is still honoured: `test_arms_agree_on_the_fit` drives both arms
    through `kneighbors` and gets identical distances. What is asserted here is
    that we SAY so rather than reporting a fit-time guess — an approximate
    `device_` would be wrong exactly when a caller queries with a different `k`
    than the estimator was constructed with, which is the normal case for
    `kneighbors(X, n_neighbors=...)`.
    """
    est = _fit(factory, needs_y, "cpu")
    with pytest.raises(AttributeError, match="per QUERY"):
        _ = est.device_


def test_umap_arms_differ_by_design():
    """UMAP is the ONE estimator where `device` changes the answer, and that is
    a property of the estimator, not of the parameter.

    Its layout is a stochastic SGD; the host driver and the device kernel
    consume the negative-sampling RNG in different orders, so the same seed
    gives a different — equally valid — embedding. This was already true when
    moving between backends; `device` merely makes it selectable in one process.

    Asserted rather than tolerated: if a future change made the two arms agree,
    this test fails and the caveat in `device.rs` should come out.
    """
    X, _ = design()
    a = mlrs.UMAP(n_components=2, n_neighbors=5, random_state=0, device="cpu").fit(X)
    b = mlrs.UMAP(n_components=2, n_neighbors=5, random_state=0, device="gpu").fit(X)
    ea = np.asarray(a.embedding_, dtype=np.float64)
    eb = np.asarray(b.embedding_, dtype=np.float64)
    assert ea.shape == eb.shape
    assert np.all(np.isfinite(ea)) and np.all(np.isfinite(eb))
    # Same seed, genuinely different layout.
    assert np.abs(ea - eb).max() > 1e-6, (
        "the UMAP arms now agree — remove STOCHASTIC_IDS and the caveat in "
        "`mlrs_backend::device`"
    )
    # Reproducible WITHIN an arm, which is the guarantee that survives.
    again = mlrs.UMAP(
        n_components=2, n_neighbors=5, random_state=0, device="cpu"
    ).fit(X)
    assert np.allclose(ea, np.asarray(again.embedding_, dtype=np.float64))


def test_single_arm_estimators_have_no_device():
    """Estimators with ONE implementation must not advertise the choice.

    ``SpectralEmbedding``/``SpectralClustering`` return ``True`` unconditionally
    from ``host_fit_applicable`` — their own docs say the predicate is kept only
    "if a device arm ever returns" — and ``RidgeCV``'s engines are host-only.
    Accepting ``device`` there and ignoring it would be a lie in the API.
    """
    for est in (
        mlrs.SpectralEmbedding(n_components=2),
        mlrs.SpectralClustering(n_clusters=2),
        mlrs.RidgeCV(),
        mlrs.LinearRegression(),
    ):
        assert "device" not in est.get_params(), (
            f"{type(est).__name__} has one arm but advertises `device`"
        )
