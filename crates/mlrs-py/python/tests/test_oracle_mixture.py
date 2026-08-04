"""GaussianMixture oracle harness (MIX-01: full numpy->pyarrow->FFI->host path).

Re-validates the 1e-5 contract through the FULL Python binding path by replaying
the committed ``tests/fixtures/gaussian_mixture_{f32,f64}_seed42.npz`` blobs
(``scripts/gen_oracle.py::gen_gaussian_mixture``). The Rust suite
(``crates/mlrs-algos/tests/gaussian_mixture_test.rs``) already gates the
arithmetic; what this file adds is everything BETWEEN numpy and the estimator:
the Arrow ingress, the ``covariance_type``-dependent RESHAPE of
``covariances_`` / ``precisions_`` (four different shapes from one flat buffer —
the most likely place for a silent egress bug), the ``output_type`` routing, and
the sklearn-named ctor spelling.

Both string-valued hyperparameters are covered here at full width, per the
project's "oracle tests for every string-valued parameter" rule:
``covariance_type`` in {full, tied, diag, spherical} x ``init_params`` in
{kmeans, k-means++, random, random_from_data}.

f64 fixtures are skipped-with-reason PER FIXTURE via ``_skip_unsupported_dtype``
rather than with the module-level ``requires_f64`` marker, which would throw
away the f32 half too (same reasoning as ``test_oracle_preprocessing.py``).
"""

import numpy as np
import pytest

import mlrs
from conftest import dtype_of, fixture_path

COV_TYPES = ["full", "tied", "diag", "spherical"]
INIT_PARAMS = ["kmeans", "k-means++", "random", "random_from_data"]
FIXTURES = ["gaussian_mixture_f32_seed42", "gaussian_mixture_f64_seed42"]
K = 3
D = 4


def _skip_unsupported_dtype(fixture):
    if dtype_of(fixture) == np.float64 and not mlrs.backend_supports_f64():
        pytest.skip("backend does not support f64")


def _atol(fixture):
    return 1e-5 if dtype_of(fixture) == np.float64 else 1e-4


def _case_tag(init):
    """The fixture's case-name spelling of an ``init_params`` value.

    The generator strips ``-`` from the name so ``k-means++`` is stored as
    ``kmeans++`` — distinct from the plain ``kmeans`` case.
    """
    return init.replace("-", "")


def _match_components(ours, reference):
    """Align our components with the reference's by nearest mean.

    The fitted component ORDER depends on the initialization, which depends on
    an RNG mlrs does not share with numpy (D-09). The fixture's blobs are
    separated by ~5 sigma, so nearest-mean matching is unambiguous.
    """
    perm, taken = [], set()
    for r in range(K):
        best, bd = None, np.inf
        for o in range(K):
            if o in taken:
                continue
            dist = np.sum((ours[o] - reference[r]) ** 2)
            if dist < bd:
                bd, best = dist, o
        taken.add(best)
        perm.append(best)
    return perm


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("cov", COV_TYPES)
@pytest.mark.parametrize("init", INIT_PARAMS)
def test_covariance_type_x_init_params_oracle(fixture, cov, init):
    """Every ``covariance_type`` x ``init_params`` reaches sklearn's optimum.

    Compared up to a component permutation, and fitted to ``tol=1e-12`` so both
    engines sit on the same stationary point rather than anywhere inside a
    default ``1e-3`` band.
    """
    _skip_unsupported_dtype(fixture)
    d = np.load(fixture_path(fixture))
    x = d["X"]
    name = f"{cov}_{_case_tag(init)}"

    est = mlrs.GaussianMixture(
        n_components=K,
        covariance_type=cov,
        init_params=init,
        tol=1e-12,
        max_iter=2000,
        random_state=0,
    ).fit(x)

    atol = _atol(fixture)
    ref_means = np.asarray(d[f"means_{name}"], dtype=np.float64)
    got_means = np.asarray(est.means_, dtype=np.float64)
    perm = _match_components(got_means, ref_means)

    np.testing.assert_allclose(got_means[perm], ref_means, atol=atol, rtol=atol)
    np.testing.assert_allclose(
        np.asarray(est.weights_, dtype=np.float64)[perm],
        np.asarray(d[f"weights_{name}"], dtype=np.float64),
        atol=atol,
        rtol=atol,
    )
    # `tied` has ONE shared covariance, so there is nothing to permute.
    got_cov = np.asarray(est.covariances_, dtype=np.float64)
    ref_cov = np.asarray(d[f"cov_{name}"], dtype=np.float64).reshape(got_cov.shape)
    if cov != "tied":
        got_cov = got_cov[perm]
    np.testing.assert_allclose(got_cov, ref_cov, atol=atol, rtol=atol)
    np.testing.assert_allclose(
        est.lower_bound_, float(d[f"lower_bound_{name}"][0]), atol=atol, rtol=atol
    )


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("cov", COV_TYPES)
def test_injected_init_oracle(fixture, cov):
    """Fully-injected init: no RNG anywhere, so EVERYTHING is compared exactly.

    Including ``n_iter_`` / ``converged_`` (integer agreements, not tolerances)
    and the whole scoring surface on the disjoint query block.
    """
    _skip_unsupported_dtype(fixture)
    d = np.load(fixture_path(fixture))
    x, xq = d["X"], d["Xq"]
    name = f"inj_{cov}"
    atol = _atol(fixture)

    shape = {
        "full": (K, D, D),
        "tied": (D, D),
        "diag": (K, D),
        "spherical": (K,),
    }[cov]
    est = mlrs.GaussianMixture(
        n_components=K,
        covariance_type=cov,
        tol=1e-8,
        max_iter=200,
        weights_init=d[f"winit_{cov}"],
        means_init=np.asarray(d[f"minit_{cov}"]).reshape(K, D),
        precisions_init=np.asarray(d[f"pinit_{cov}"]).reshape(shape),
    ).fit(x)

    np.testing.assert_allclose(est.weights_, d[f"weights_{name}"], atol=atol, rtol=atol)
    np.testing.assert_allclose(est.means_, d[f"means_{name}"].reshape(K, D), atol=atol, rtol=atol)

    # The reshape is the point of this assertion: one flat Rust buffer has to
    # come back as four DIFFERENT sklearn shapes.
    assert est.covariances_.shape == shape
    assert est.precisions_.shape == shape
    assert est.precisions_cholesky_.shape == shape
    np.testing.assert_allclose(
        np.ravel(est.covariances_), d[f"cov_{name}"], atol=atol, rtol=atol
    )
    np.testing.assert_allclose(
        np.ravel(est.precisions_cholesky_), d[f"prec_chol_{name}"], atol=atol, rtol=atol
    )

    assert est.n_iter_ == int(d[f"n_iter_{name}"][0])
    # `lower_bounds_` is the per-iteration ascent, not just its endpoint.
    assert est.lower_bounds_.shape == (est.n_iter_,)
    np.testing.assert_allclose(
        est.lower_bounds_, d[f"lower_bounds_{name}"], atol=atol, rtol=atol
    )
    assert est.converged_ == bool(d[f"converged_{name}"][0])
    np.testing.assert_allclose(
        est.lower_bound_, float(d[f"lower_bound_{name}"][0]), atol=atol, rtol=atol
    )

    # --- the scoring surface ------------------------------------------------ #
    np.testing.assert_array_equal(est.predict(xq), d[f"predict_{name}"].astype(np.int32))
    np.testing.assert_allclose(
        np.ravel(est.predict_proba(xq)), d[f"proba_{name}"], atol=atol, rtol=atol
    )
    np.testing.assert_allclose(
        est.score_samples(xq), d[f"score_samples_{name}"], atol=atol, rtol=atol
    )
    assert est._n_parameters() == int(d[f"n_parameters_{name}"][0])
    np.testing.assert_allclose(est.bic(xq), float(d[f"bic_{name}"][0]), atol=atol, rtol=atol)
    np.testing.assert_allclose(est.aic(xq), float(d[f"aic_{name}"][0]), atol=atol, rtol=atol)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_fit_predict_matches_predict(fixture):
    """``fit_predict`` returns the terminal E-step's labels, and re-``predict``ing
    the training design reproduces them."""
    _skip_unsupported_dtype(fixture)
    d = np.load(fixture_path(fixture))
    x = d["X"]
    est = mlrs.GaussianMixture(n_components=K, random_state=0, tol=1e-10, max_iter=500)
    labels = est.fit_predict(x)
    assert labels.shape == (x.shape[0],)
    np.testing.assert_array_equal(labels, est.predict(x))


@pytest.mark.parametrize("fixture", FIXTURES)
def test_predict_proba_is_a_distribution(fixture):
    _skip_unsupported_dtype(fixture)
    d = np.load(fixture_path(fixture))
    x = d["X"]
    est = mlrs.GaussianMixture(n_components=K, random_state=0).fit(x)
    proba = est.predict_proba(d["Xq"])
    assert proba.shape == (d["Xq"].shape[0], K)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        np.exp(np.asarray(est.predict_log_proba(d["Xq"]), dtype=np.float64)),
        np.asarray(proba, dtype=np.float64),
        atol=1e-6,
    )


def test_unknown_string_hyperparameters_raise():
    """Both string-valued hyperparameters reject an unknown value at ``fit``.

    mlrs validates at the first ``fit`` rather than at construction (the Unfit
    arm stores the raw strings until then, D-09), so the error surfaces here.
    """
    x = np.load(fixture_path("gaussian_mixture_f64_seed42"))["X"]
    if not mlrs.backend_supports_f64():
        x = x.astype(np.float32)
    with pytest.raises(ValueError, match="covariance_type"):
        mlrs.GaussianMixture(n_components=K, covariance_type="diagonal").fit(x)
    with pytest.raises(ValueError, match="init"):
        mlrs.GaussianMixture(n_components=K, init_params="kmeans++").fit(x)


def test_sample_returns_the_right_shapes():
    x = np.load(fixture_path("gaussian_mixture_f64_seed42"))["X"]
    if not mlrs.backend_supports_f64():
        x = x.astype(np.float32)
    est = mlrs.GaussianMixture(n_components=K, random_state=0).fit(x)
    drawn, y = est.sample(500, seed=1)
    assert drawn.shape == (500, D)
    assert y.shape == (500,)
    assert set(np.unique(y)).issubset(set(range(K)))
