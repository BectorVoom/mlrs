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


AGGLOMERATIVE_FIXTURES = [
    "agglomerative_euclidean_f32_seed42",
    "agglomerative_euclidean_f64_seed42",
    "agglomerative_manhattan_f32_seed42",
    "agglomerative_manhattan_f64_seed42",
    "agglomerative_cosine_f32_seed42",
    "agglomerative_cosine_f64_seed42",
]


@pytest.mark.parametrize("fixture", AGGLOMERATIVE_FIXTURES)
@requires_f64
def test_agglomerative_oracle(fixture):
    """AGGLO-01: labels_ and children_ EXACTLY match sklearn (no permutation —
    mlrs ports the unstructured single-linkage pipeline line-for-line)."""
    d = np.load(fixture_path(fixture))
    metric = fixture.split("_")[1]
    for k, key in ((2, "labels_k2"), (3, "labels_k3"), (5, "labels_k5")):
        est = mlrs.AgglomerativeClustering(n_clusters=k, metric=metric).fit(d["X"])
        labels = np.asarray(est.labels_).astype(np.int64).ravel()
        ref = d[key].astype(np.int64).ravel()
        assert np.array_equal(labels, ref), f"{fixture} k={k}: labels mismatch"
        children = np.asarray(est.children_).astype(np.int64)
        ref_children = d["children"].astype(np.int64)
        assert np.array_equal(children, ref_children), f"{fixture} k={k}: children"
        assert est.n_leaves_ == d["X"].shape[0]
        assert est.n_connected_components_ == 1
        assert est.n_clusters_ == k


def test_agglomerative_rejects_unsupported():
    """Unsupported linkage / metric raise loudly (never silently degrade)."""
    X = np.random.default_rng(0).normal(size=(8, 3))
    with pytest.raises(ValueError, match="linkage"):
        mlrs.AgglomerativeClustering(linkage="ward").fit(X)
    with pytest.raises(ValueError, match="metric"):
        mlrs.AgglomerativeClustering(metric="chebyshev").fit(X)


# ---------------------------------------------------------------------------
# HDBSCAN string-valued-parameter surface (HDBS-PARAMS).
#
# THE point of gating these here rather than in Rust: the eleven `metric`
# strings collapse onto six `Metric` enum values at the shim boundary (`l2` IS
# `euclidean`, `cityblock`/`l1` ARE `manhattan`, `infinity` IS `chebyshev`, `p`
# IS `minkowski`). An alias wired to the WRONG enum is invisible to every
# Rust-side test — it can only be caught by driving the string through the full
# Python path and comparing against sklearn under that same string. Same for
# `algorithm` and `store_centers`, whose strings never reach Rust as strings.
#
# The fixture (`scripts/gen_oracle.py::gen_hdbscan_params`) is n = 600 > the
# KD-tree route's 512-row floor, so `algorithm='auto'` genuinely builds and
# calibrates a tree and the four-way agreement below is not vacuous.
# ---------------------------------------------------------------------------

HDBSCAN_PARAM_FIXTURES = ["hdbscan_params_f32_seed42", "hdbscan_params_f64_seed42"]

HDBSCAN_METRIC_STRINGS = [
    "euclidean", "l2",
    "manhattan", "cityblock", "l1",
    "chebyshev", "infinity",
    "minkowski", "p",
    "cosine",
    "precomputed",
]
HDBSCAN_ALGORITHM_STRINGS = ["auto", "brute", "kd_tree", "ball_tree"]
HDBSCAN_CSM_STRINGS = ["eom", "leaf"]


def _hdbscan_param_est(d, **over):
    """Build + fit an mlrs HDBSCAN on the params fixture with `over` applied."""
    mcs = int(d["min_cluster_size"][0])
    x = d["X_precomputed"] if over.get("metric") == "precomputed" else d["X"]
    return mlrs.HDBSCAN(min_cluster_size=mcs, **over).fit(x)


@pytest.mark.parametrize("fixture", HDBSCAN_PARAM_FIXTURES)
@pytest.mark.parametrize("metric", HDBSCAN_METRIC_STRINGS)
@requires_f64
def test_hdbscan_metric_string_oracle(fixture, metric):
    """Every accepted `metric=` string reproduces sklearn's partition.

    Compared up to a label permutation with noise PINNED (`label_perm_allclose`)
    — cluster ids are arbitrary but the noise/cluster split is not.
    """
    d = np.load(fixture_path(fixture))
    over = {"metric": metric}
    if metric in ("minkowski", "p"):
        over["metric_params"] = {"p": float(d["minkowski_p"][0])}
    est = _hdbscan_param_est(d, **over)
    labels = np.asarray(est.labels_).astype(np.int64).ravel()
    ref = d[f"labels_metric_{metric}"].astype(np.int64).ravel()
    assert label_perm_allclose(labels, ref), f"{fixture} metric={metric}"


@pytest.mark.parametrize("fixture", HDBSCAN_PARAM_FIXTURES)
@pytest.mark.parametrize("algorithm", HDBSCAN_ALGORITHM_STRINGS)
@requires_f64
def test_hdbscan_algorithm_string_oracle(fixture, algorithm):
    """Every `algorithm=` string reproduces sklearn's partition."""
    d = np.load(fixture_path(fixture))
    est = _hdbscan_param_est(d, algorithm=algorithm)
    labels = np.asarray(est.labels_).astype(np.int64).ravel()
    ref = d[f"labels_algorithm_{algorithm}"].astype(np.int64).ravel()
    assert label_perm_allclose(labels, ref), f"{fixture} algorithm={algorithm}"


@pytest.mark.parametrize("fixture", HDBSCAN_PARAM_FIXTURES)
@requires_f64
def test_hdbscan_algorithm_is_exactly_value_neutral(fixture):
    """All four `algorithm=` values give BIT-IDENTICAL labels and probabilities.

    Stronger than the sklearn gate above, and deliberately so: in mlrs the
    algorithm only decides which candidate pairs a core-distance query is
    allowed to SKIP, never how an evaluated pair is computed, so the four routes
    are the same arithmetic on the same numbers. Equality — not a tolerance — is
    therefore the honest assertion, and any future route that computes distances
    a second way will fail here instead of hiding inside a 1e-5 band.
    """
    d = np.load(fixture_path(fixture))
    base = _hdbscan_param_est(d, algorithm="auto")
    base_labels = np.asarray(base.labels_).astype(np.int64).ravel()
    base_probs = np.asarray(base.probabilities_, dtype=np.float64).ravel()
    for algorithm in HDBSCAN_ALGORITHM_STRINGS[1:]:
        est = _hdbscan_param_est(d, algorithm=algorithm)
        got = np.asarray(est.labels_).astype(np.int64).ravel()
        assert np.array_equal(got, base_labels), (
            f"{fixture} algorithm={algorithm}: labels differ from 'auto'"
        )
        probs = np.asarray(est.probabilities_, dtype=np.float64).ravel()
        assert np.array_equal(probs, base_probs), (
            f"{fixture} algorithm={algorithm}: probabilities differ from 'auto'"
        )


@pytest.mark.parametrize("fixture", HDBSCAN_PARAM_FIXTURES)
@pytest.mark.parametrize("leaf_size", [1, 8, 40, 256])
@requires_f64
def test_hdbscan_leaf_size_is_exactly_value_neutral(fixture, leaf_size):
    """`leaf_size=` moves no value — it only sets how many points a leaf scan
    covers between box tests. Forced onto the tree route so the knob is live."""
    d = np.load(fixture_path(fixture))
    base = _hdbscan_param_est(d, algorithm="kd_tree")
    est = _hdbscan_param_est(d, algorithm="kd_tree", leaf_size=leaf_size)
    assert np.array_equal(
        np.asarray(est.labels_).astype(np.int64),
        np.asarray(base.labels_).astype(np.int64),
    ), f"{fixture} leaf_size={leaf_size}: labels differ from the default"


@pytest.mark.parametrize("fixture", HDBSCAN_PARAM_FIXTURES)
@pytest.mark.parametrize("n_jobs", [1, 2, 4, -1, -2])
@requires_f64
def test_hdbscan_n_jobs_is_exactly_value_neutral(fixture, n_jobs):
    """`n_jobs=` only decides how the row range is cut across workers; each block
    computes the same rows it would serially, so no value can move a label."""
    d = np.load(fixture_path(fixture))
    base = _hdbscan_param_est(d)
    est = _hdbscan_param_est(d, n_jobs=n_jobs)
    assert np.array_equal(
        np.asarray(est.labels_).astype(np.int64),
        np.asarray(base.labels_).astype(np.int64),
    ), f"{fixture} n_jobs={n_jobs}: labels differ from the default"


@pytest.mark.parametrize("fixture", HDBSCAN_PARAM_FIXTURES)
@pytest.mark.parametrize("csm", HDBSCAN_CSM_STRINGS)
@requires_f64
def test_hdbscan_cluster_selection_method_string_oracle(fixture, csm):
    """Every `cluster_selection_method=` string reproduces sklearn's partition."""
    d = np.load(fixture_path(fixture))
    est = _hdbscan_param_est(d, cluster_selection_method=csm)
    labels = np.asarray(est.labels_).astype(np.int64).ravel()
    ref = d[f"labels_csm_{csm}"].astype(np.int64).ravel()
    assert label_perm_allclose(labels, ref), f"{fixture} csm={csm}"


@pytest.mark.parametrize("fixture", HDBSCAN_PARAM_FIXTURES)
@pytest.mark.parametrize("store_centers", [None, "centroid", "medoid", "both"])
@requires_f64
def test_hdbscan_store_centers_string_oracle(fixture, store_centers):
    """Every `store_centers=` string matches sklearn in BOTH senses.

    * PRESENCE — sklearn leaves the attribute absent unless this exact value
      asked for it, and `hasattr` is the documented way to test that, so the
      shim must be absent-not-None too.
    * VALUE — the centre rows match sklearn to 1e-5 once both sides' rows are
      put in a common order (cluster ids are arbitrary, so row ORDER is too;
      lexsort is a permutation-invariant canonical form).
    """
    d = np.load(fixture_path(fixture))
    atol = _atol(fixture)
    est = _hdbscan_param_est(d, store_centers=store_centers)
    for name, asked in (
        ("centroids", store_centers in ("centroid", "both")),
        ("medoids", store_centers in ("medoid", "both")),
    ):
        if not asked:
            assert not hasattr(est, f"{name}_"), (
                f"{fixture} store_centers={store_centers!r}: {name}_ must be "
                f"absent (sklearn parity), not None"
            )
            continue
        got = np.asarray(getattr(est, f"{name}_"), dtype=np.float64)
        ref = d[f"{name}_store"].astype(np.float64)
        assert got.shape == ref.shape, (
            f"{fixture} store_centers={store_centers!r}: {name}_ shape "
            f"{got.shape} != sklearn {ref.shape}"
        )
        got_sorted = got[np.lexsort(got.T[::-1])]
        ref_sorted = ref[np.lexsort(ref.T[::-1])]
        assert np.allclose(got_sorted, ref_sorted, atol=atol, rtol=0.0), (
            f"{fixture} store_centers={store_centers!r}: {name}_ values"
        )


@pytest.mark.parametrize(
    "algorithm,metric",
    [
        # sklearn: "<metric> is not a valid metric for a KDTree/BallTree-based
        # algorithm". A box bound is only a valid lower bound for a distance that
        # aggregates monotonely over the feature axes; cosine is normalized and
        # precomputed has no feature axes, so neither can be traversed. This is a
        # real restriction of the algorithm, and mlrs reproduces it.
        ("kd_tree", "cosine"),
        ("kd_tree", "precomputed"),
        ("ball_tree", "cosine"),
        ("ball_tree", "precomputed"),
    ],
)
def test_hdbscan_rejects_impossible_algorithm_metric_pairs(algorithm, metric):
    """Combinations the tree route genuinely cannot serve are refused.

    Rejected at build, before any data is touched — the pair is knowable without
    it (sklearn raises at ``fit``).
    """
    X = np.random.default_rng(0).normal(size=(40, 3))
    arr = X @ X.T if metric == "precomputed" else X
    with pytest.raises(ValueError):
        mlrs.HDBSCAN(algorithm=algorithm, metric=metric).fit(arr)


@pytest.mark.parametrize(
    "alias,canonical",
    [("infinity", "chebyshev"), ("p", "minkowski")],
)
def test_hdbscan_brute_accepts_tree_only_metric_aliases(alias, canonical):
    """``algorithm='brute'`` accepts ``infinity``/``p`` — a DELIBERATE divergence.

    Recorded as SK-001 in ``docs/upstream-sklearn-issues.md``.

    sklearn 1.9.0 raises here, and that is a gap in its validation rather than a
    property of the algorithm: its ``metric`` constraint is the union of the tree
    and pairwise metric sets, but its brute path goes through
    ``pairwise_distances``, whose table lacks these two tree-only aliases. So a
    value sklearn's own validation accepts is rejected mid-``fit`` by a helper,
    with an error naming ``pairwise_distances``' parameter. Its tree paths check
    properly; only ``brute`` is missing the check. Reported upstream.

    mlrs has no such asymmetry — the aliases resolve to the same ``Metric`` on
    every route, and ``algorithm`` is value-neutral by construction — so
    refusing the pair would mean inventing a restriction the engine does not
    have. This test pins BOTH halves of that: the call succeeds, and it produces
    exactly what the canonical spelling produces, which is the reason the
    rejection was never justified.
    """
    d = np.load(fixture_path("hdbscan_params_f64_seed42"))
    mcs = int(d["min_cluster_size"][0])
    over = {"metric_params": {"p": float(d["minkowski_p"][0])}} if alias == "p" else {}
    est_alias = mlrs.HDBSCAN(
        min_cluster_size=mcs, algorithm="brute", metric=alias, **over
    ).fit(d["X"])
    est_canon = mlrs.HDBSCAN(
        min_cluster_size=mcs, algorithm="brute", metric=canonical, **over
    ).fit(d["X"])
    assert np.array_equal(
        np.asarray(est_alias.labels_).astype(np.int64),
        np.asarray(est_canon.labels_).astype(np.int64),
    ), f"metric={alias!r} must be identical to metric={canonical!r} under brute"


def test_hdbscan_rejects_unsupported_strings():
    """Unknown strings raise loudly, naming the accepted set (never degrade)."""
    X = np.random.default_rng(0).normal(size=(40, 3))
    with pytest.raises(ValueError, match="metric"):
        mlrs.HDBSCAN(metric="braycurtis").fit(X)
    with pytest.raises(ValueError, match="algorithm"):
        mlrs.HDBSCAN(algorithm="kdtree").fit(X)
    with pytest.raises(ValueError, match="cluster_selection_method"):
        mlrs.HDBSCAN(cluster_selection_method="excess_of_mass").fit(X)
    with pytest.raises(ValueError, match="store_centers"):
        mlrs.HDBSCAN(store_centers="centroids").fit(X)
    # metric_params is sklearn-shaped but mlrs reads only 'p'; a typo'd key is
    # rejected rather than silently clustering under the default exponent.
    with pytest.raises(ValueError, match="metric_params"):
        mlrs.HDBSCAN(metric="minkowski", metric_params={"P": 3.0}).fit(X)


@pytest.mark.parametrize("store_centers", ["centroid", "medoid", "both"])
def test_hdbscan_store_centers_all_noise_is_empty_not_absent(store_centers):
    """A fit that finds NO cluster still exposes the requested centre block, as
    an empty ``(0, n_features)`` array.

    The distinction matters and is easy to get backwards: "you did not ask for
    this block" is an absent attribute, but "you asked and there was nothing to
    put in it" is an EMPTY one. sklearn draws the line in exactly that place, so
    caller code doing ``len(est.centroids_)`` keeps working on a degenerate fit
    instead of raising ``AttributeError``.
    """
    X = np.random.default_rng(1).uniform(-20, 20, size=(40, 3))
    est = mlrs.HDBSCAN(min_cluster_size=25, store_centers=store_centers).fit(X)
    assert (np.asarray(est.labels_) == -1).all(), "fixture must be all-noise"
    for name, asked in (
        ("centroids_", store_centers in ("centroid", "both")),
        ("medoids_", store_centers in ("medoid", "both")),
    ):
        if not asked:
            assert not hasattr(est, name)
            continue
        got = np.asarray(getattr(est, name))
        assert got.shape == (0, 3), f"{name}: expected (0, 3), got {got.shape}"
