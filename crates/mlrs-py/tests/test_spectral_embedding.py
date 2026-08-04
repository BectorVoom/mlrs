"""SPECTRAL-PERF-CPU — `mlrs.cluster.SpectralEmbedding` vs scikit-learn, live FFI.

Unlike the other Python suites in this directory this one IS a numerical oracle,
because the thing being asserted is a property the Rust oracle tests cannot
reach: that the SHIM forwards sklearn's full eight-parameter surface, resolves
the defaults sklearn's own way, and hands back the same fitted attributes. A
parameter that is silently dropped at the binding boundary produces a perfectly
self-consistent — and wrong — embedding, which no Rust-side fixture would catch.

## The one thing that makes or breaks a test in this file

**A disconnected affinity graph makes an elementwise comparison meaningless.**
The normalized Laplacian has one zero eigenvalue per connected component, so on
a graph with `c` components the smallest `c` eigenvectors span a fully degenerate
null space and are defined only up to an arbitrary rotation within it. sklearn
and mlrs will then return different — both correct — bases. Every test here
therefore asserts `n_graph_components == 1` before comparing values, and the
data is chosen to guarantee it.

That constraint runs OPPOSITE to the usual fixture intuition: `make_blobs` with
well-separated clusters is exactly the case that DISCONNECTS a kNN graph (each
blob becomes its own component). Uniform-random data, which has no cluster
structure at all, is what keeps the graph connected. Where a test does want
separated structure it uses `rbf`, whose affinity is strictly positive
everywhere and so is always connected.

Run via the shipped cpu-extension flow (`scripts/build_cpu_ext.sh`, then
`pytest` this file). The module is import-guarded so it skips cleanly rather
than erroring at collection when the extension is not built.
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("mlrs._mlrs")

from mlrs.cluster import SpectralEmbedding as MlrsSE  # noqa: E402
from sklearn.manifold import SpectralEmbedding as SkSE  # noqa: E402
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel  # noqa: E402

# The strict project contract (CLAUDE.md: abs/rel error <= 1e-5 vs sklearn).
# Observed on these cases is ~1e-15; the band is left at the contract value
# rather than tightened to the observation, so a real regression is what trips
# it rather than ordinary FP jitter.
TOL = 1e-5


def uniform(n, d, seed=0):
    """Uniform rows — no cluster structure, so the kNN graph stays connected."""
    return np.ascontiguousarray(np.random.default_rng(seed).random((n, d)))


def fit_both(X, **kw):
    """Fit both engines on `X`, returning `(sklearn_est, mlrs_est)`.

    sklearn's "Graph is not fully connected" warning is advisory and changes
    nothing in its own code path, so it is silenced here — the connectivity
    assertion in :func:`assert_matches` is what actually guards the comparison.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SkSE(**kw).fit(X), MlrsSE(**kw).fit(X)


def assert_matches(sk, ml, tol=TOL):
    """Assert the two embeddings agree, after guarding against degeneracy.

    Each column is defined only up to a global sign. Both engines apply
    sklearn's deterministic sign flip, but that flip keys on the argmax entry, so
    a near-tie in magnitude can legitimately land the two on opposite signs —
    aligning here keeps the assertion about the subspace rather than about that
    tie-break.
    """
    ncc = ml._mlrs_obj.n_graph_components()
    assert ncc == 1, (
        f"the affinity graph has {ncc} components, so the kept eigenvectors span "
        f"a degenerate null space and an elementwise comparison is vacuous — fix "
        f"the test data, do not loosen the tolerance"
    )
    a = np.asarray(sk.embedding_, dtype=np.float64)
    b = np.asarray(ml.embedding_, dtype=np.float64)
    assert a.shape == b.shape, f"shape {b.shape} != sklearn {a.shape}"
    worst = 0.0
    for c in range(a.shape[1]):
        u, v = a[:, c], b[:, c]
        if float(np.dot(u, v)) < 0.0:
            v = -v
        worst = max(worst, float(np.max(np.abs(u - v))))
    assert worst <= tol, f"max|Δembedding| = {worst:.3e} exceeds {tol:e}"
    return worst


# --------------------------------------------------------------------------- #
# affinities
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n,d,k", [(120, 5, 2), (400, 8, 3), (900, 6, 2)])
def test_nearest_neighbors_matches_sklearn(n, d, k):
    """The default affinity, across the dense/Lanczos routing threshold.

    `n = 900` is above `spectral_host::DENSE_N`, so this parametrization is what
    makes the iterative solver — not just the dense one — an FFI-tested path.
    """
    X = uniform(n, d)
    sk, ml = fit_both(X, n_components=k, random_state=0)
    assert_matches(sk, ml)


def test_rbf_matches_sklearn():
    X = uniform(300, 4)
    sk, ml = fit_both(X, n_components=3, affinity="rbf", random_state=0)
    assert_matches(sk, ml)
    assert ml.gamma_ == pytest.approx(sk.gamma_)


def test_precomputed_matches_sklearn():
    """`affinity="precomputed"` — `X` IS the affinity, passed through verbatim."""
    A = rbf_kernel(uniform(250, 6), gamma=0.5)
    sk, ml = fit_both(A, n_components=3, affinity="precomputed", random_state=0)
    assert_matches(sk, ml)
    # sklearn reports n_features_in_ = n_samples for a pairwise affinity.
    assert ml.n_features_in_ == sk.n_features_in_ == A.shape[0]


def test_precomputed_nearest_neighbors_matches_sklearn():
    """`X` is a precomputed DISTANCE matrix, not an affinity."""
    D = euclidean_distances(uniform(250, 6))
    sk, ml = fit_both(
        D,
        n_components=3,
        affinity="precomputed_nearest_neighbors",
        n_neighbors=25,
        random_state=0,
    )
    assert_matches(sk, ml)


# --------------------------------------------------------------------------- #
# parameter resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n,expected", [(60, 6), (100, 10), (255, 25), (999, 99)])
def test_n_neighbors_none_resolves_to_n_over_10(n, expected):
    """sklearn resolves `n_neighbors=None` to `max(int(n_samples / 10), 1)`.

    Truncating division, floored at 1. The pre-rewrite shim hard-coded 10, which
    silently built a different graph from sklearn's for every `n != 100` — this
    test is the regression guard for that specific bug.
    """
    _, ml = fit_both(uniform(n, 4), n_components=2, random_state=0)
    assert ml.n_neighbors_ == expected


def test_explicit_n_neighbors_is_honored():
    X = uniform(300, 5)
    sk, ml = fit_both(X, n_components=2, n_neighbors=17, random_state=0)
    assert ml.n_neighbors_ == sk.n_neighbors_ == 17
    assert_matches(sk, ml)


def test_gamma_none_resolves_to_inverse_n_features():
    _, ml = fit_both(uniform(200, 8), n_components=2, affinity="rbf", random_state=0)
    assert ml.gamma_ == pytest.approx(1.0 / 8)


def test_explicit_gamma_is_honored():
    X = uniform(200, 5)
    sk, ml = fit_both(X, n_components=2, affinity="rbf", gamma=2.5, random_state=0)
    assert ml.gamma_ == pytest.approx(2.5)
    assert_matches(sk, ml)


@pytest.mark.parametrize("solver", [None, "arpack", "lobpcg", "amg"])
def test_eigen_solver_values_are_accepted(solver):
    """All four sklearn `eigen_solver` values name a route to the SAME invariant
    subspace. mlrs has one solver and routes every value to it, so the parameter
    selects nothing — but it must still be accepted, and the result must not
    depend on which was named.

    Only mlrs is fitted here. `eigen_solver="amg"` is a DELIBERATE divergence:
    sklearn raises `"The eigen_solver was set to 'amg', but pyamg is not
    available."` because that route is an optional third-party dependency for it,
    whereas mlrs has one built-in solver and nothing to be missing. Refusing the
    value to imitate a missing dependency mlrs does not have would be parity
    theatre; accepting it is the honest behavior.
    """
    X = uniform(200, 5)
    ml = MlrsSE(n_components=2, eigen_solver=solver, random_state=0).fit(X)
    base = MlrsSE(n_components=2, random_state=0).fit(X)
    assert np.allclose(ml.embedding_, base.embedding_)


def test_unknown_eigen_solver_is_rejected():
    with pytest.raises(ValueError):
        MlrsSE(n_components=2, eigen_solver="definitely-not-a-solver").fit(
            uniform(80, 4)
        )


def test_unknown_affinity_is_rejected():
    with pytest.raises(ValueError):
        MlrsSE(n_components=2, affinity="not-an-affinity").fit(uniform(80, 4))


def test_negative_gamma_is_rejected():
    with pytest.raises(ValueError):
        MlrsSE(n_components=2, affinity="rbf", gamma=-1.0).fit(uniform(80, 4))


def test_eigen_tol_auto_and_explicit_agree():
    """`eigen_tol="auto"` is sklearn's default and means "run to machine
    precision" for arpack. An explicit tight tolerance must not move the answer.
    """
    X = uniform(300, 5)
    _, a = fit_both(X, n_components=2, random_state=0)
    _, b = fit_both(X, n_components=2, eigen_tol=1e-12, random_state=0)
    assert np.allclose(a.embedding_, b.embedding_)


# --------------------------------------------------------------------------- #
# fitted attributes and estimator protocol
# --------------------------------------------------------------------------- #


def test_affinity_matrix_is_sparse_for_knn_and_matches_sklearn():
    sparse = pytest.importorskip("scipy.sparse")
    X = uniform(200, 5)
    sk, ml = fit_both(X, n_components=2, random_state=0)
    assert sparse.issparse(ml.affinity_matrix_), (
        "the kNN affinity must stay SPARSE — densifying it is what made the old "
        "implementation unable to fit at realistic sample counts"
    )
    assert sparse.issparse(sk.affinity_matrix_)
    delta = np.max(np.abs(ml.affinity_matrix_.toarray() - sk.affinity_matrix_.toarray()))
    assert delta == 0.0, f"affinity graph differs from sklearn's by {delta}"


def test_affinity_matrix_is_dense_for_rbf_and_matches_sklearn():
    X = uniform(150, 4)
    sk, ml = fit_both(X, n_components=2, affinity="rbf", random_state=0)
    assert isinstance(ml.affinity_matrix_, np.ndarray)
    assert np.allclose(ml.affinity_matrix_, sk.affinity_matrix_, atol=1e-12)


def test_attributes_absent_on_the_wrong_branch():
    """sklearn sets `n_neighbors_` only on the kNN branch and `gamma_` only on a
    kernel branch. Reproduce the absence, not just the presence."""
    _, knn = fit_both(uniform(120, 4), n_components=2, random_state=0)
    knn.n_neighbors_  # present
    with pytest.raises(AttributeError):
        knn.gamma_
    _, rbf = fit_both(uniform(120, 4), n_components=2, affinity="rbf", random_state=0)
    rbf.gamma_  # present
    with pytest.raises(AttributeError):
        rbf.n_neighbors_


def test_disconnected_graph_warns_like_sklearn():
    """Four well-separated blobs disconnect the kNN graph; sklearn warns and
    changes nothing, and so does mlrs."""
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=200, n_features=4, centers=4, cluster_std=0.3,
                      random_state=0)
    X = np.ascontiguousarray(X, dtype=np.float64)
    with pytest.warns(UserWarning, match="not fully connected"):
        est = MlrsSE(n_components=2, n_neighbors=5, random_state=0).fit(X)
    assert est._mlrs_obj.n_graph_components() > 1


def test_refit_honors_constructor_params():
    """A second `fit` must use the SAME hyperparameters, not revert to defaults
    (the params are persisted on the wrapper, not only in its unfit arm)."""
    est = MlrsSE(n_components=3, n_neighbors=13, random_state=0)
    est.fit(uniform(150, 4, seed=1))
    est.fit(uniform(200, 4, seed=2))
    assert est.n_neighbors_ == 13
    assert est.embedding_.shape == (200, 3)


def test_fit_transform_equals_embedding():
    X = uniform(150, 4)
    est = MlrsSE(n_components=2, random_state=0)
    out = est.fit_transform(X)
    assert np.array_equal(out, est.embedding_)


def test_float32_input_matches_sklearn_within_band():
    """f32 ingress. mlrs computes the graph and spectrum in f64 regardless of the
    input dtype and casts the result back to f32, so it returns f32 where sklearn
    widens to f64 — the VALUES still agree well inside the contract band."""
    X = np.ascontiguousarray(uniform(300, 5), dtype=np.float32)
    sk, ml = fit_both(X, n_components=2, random_state=0)
    assert ml.embedding_.dtype == np.float32
    assert_matches(sk, ml, tol=1e-5)


def test_no_sample_cap():
    """The former implementation rejected `n_samples > 64` outright (the dense
    device eigensolver staged `MAX_DIM x MAX_DIM` shared memory). Nothing above
    64 could be fitted at all; this is the regression guard."""
    X = uniform(2000, 8)
    est = MlrsSE(n_components=2, random_state=0).fit(X)
    assert est.embedding_.shape == (2000, 2)
    assert np.all(np.isfinite(est.embedding_))
