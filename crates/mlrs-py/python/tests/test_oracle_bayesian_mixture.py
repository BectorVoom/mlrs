"""BayesianGaussianMixture oracle harness (MIX-02: numpy->pyarrow->FFI->host).

Re-validates the 1e-5 contract through the FULL Python binding path by replaying
the committed ``tests/fixtures/bayesian_mixture_{f32,f64}_seed42.npz`` blobs
(``scripts/gen_oracle.py::gen_bayesian_mixture``). The Rust suite
(``crates/mlrs-algos/tests/bayesian_mixture_test.rs``) already gates the
arithmetic; what this file adds is everything BETWEEN numpy and the estimator,
and this estimator has more of it than any other mixture wrapper:

* ``weight_concentration_`` comes back as a 2-TUPLE under
  ``dirichlet_process`` and a single array under ``dirichlet_distribution`` —
  one Rust accessor, two Python shapes, branching on an empty second element.
* ``degrees_of_freedom_`` is a scalar under ``covariance_type='tied'`` and an
  array otherwise.
* ``covariance_prior_`` has FOUR shapes from one flat buffer: ``(d, d)`` /
  ``(d, d)`` / ``(d,)`` / scalar.
* ``n_components`` is keyword-ONLY here (sklearn's signature), unlike
  ``GaussianMixture``.

All three string-valued hyperparameters are covered at full width, per the
project's "oracle tests for every string-valued parameter" rule:
``covariance_type`` in {full, tied, diag, spherical} x ``init_params`` in
{kmeans, k-means++, random, random_from_data} x
``weight_concentration_prior_type`` in {dirichlet_process,
dirichlet_distribution}.

f64 fixtures are skipped-with-reason PER FIXTURE via ``_skip_unsupported_dtype``
rather than with the module-level ``requires_f64`` marker, which would throw
away the f32 half too (same reasoning as ``test_oracle_mixture.py``).
"""

import numpy as np
import pytest

import mlrs
from conftest import dtype_of, fixture_path

COV_TYPES = ["full", "tied", "diag", "spherical"]
INIT_PARAMS = ["kmeans", "k-means++", "random", "random_from_data"]
PRIOR_TYPES = {"dirichlet_process": "dp", "dirichlet_distribution": "dd"}
FIXTURES = ["bayesian_mixture_f32_seed42", "bayesian_mixture_f64_seed42"]
K = 3
D = 4
# Family 1 runs this many restarts on BOTH sides — see the Rust test's comment
# for why one restart is not enough when the initialization is sparse.
N_INIT = 5


def _skip_unsupported_dtype(fixture):
    if dtype_of(fixture) == np.float64 and not mlrs.backend_supports_f64():
        pytest.skip("backend does not support f64")


def _atol(fixture):
    return 1e-5 if dtype_of(fixture) == np.float64 else 1e-4


def _case_tag(init):
    """The fixture's case-name spelling of an ``init_params`` value."""
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
@pytest.mark.parametrize("ptype", list(PRIOR_TYPES))
def test_string_parameter_cross_oracle(fixture, cov, init, ptype):
    """Every ``covariance_type`` x ``init_params`` x prior type is pinned.

    Compared up to a component permutation, at ``tol=1e-12`` so both engines sit
    on the same stationary point. Two of the thirty-two cases are marked
    unstable by the generator (``tied`` + ``dirichlet_process`` + a sparse
    init): there the variational objective has several attracting basins, so
    the values are not comparable across two RNGs and only the qualitative
    outcome — that the Dirichlet process pruned a component — is asserted.
    """
    _skip_unsupported_dtype(fixture)
    d = np.load(fixture_path(fixture))
    x = d["X"]
    name = f"{cov}_{_case_tag(init)}_{PRIOR_TYPES[ptype]}"

    est = mlrs.BayesianGaussianMixture(
        n_components=K,
        covariance_type=cov,
        init_params=init,
        weight_concentration_prior_type=ptype,
        tol=1e-12,
        max_iter=2000,
        n_init=N_INIT,
        random_state=0,
    ).fit(x)

    if float(d[f"stable_{name}"][0]) < 0.5:
        w = np.asarray(est.weights_, dtype=np.float64)
        assert np.isfinite(w).all()
        # The sum is checked at the ACCESSOR's precision, not the engine's:
        # `weights_` is narrowed to the design dtype on egress, so the f32 arm
        # can only round-trip a normalized vector to ~1e-7.
        assert abs(w.sum() - 1.0) < _atol(fixture)
        assert w.min() < 1e-2, (
            f"{name} is a documented dirichlet_process collapse, but no "
            f"component was pruned (weights_={w})"
        )
        return

    atol = _atol(fixture)
    ref_means = np.asarray(d[f"means_{name}"], dtype=np.float64)
    got_means = np.asarray(est.means_, dtype=np.float64)
    perm = _match_components(got_means, ref_means)
    np.testing.assert_allclose(got_means[perm], ref_means, atol=atol, rtol=atol)

    # `mean_precision_` is `beta0 + nk`, so this pins the component COUNTS.
    np.testing.assert_allclose(
        np.asarray(est.mean_precision_)[perm],
        np.asarray(d[f"beta_{name}"], dtype=np.float64),
        atol=atol,
        rtol=atol,
    )
    # `tied` has ONE shared covariance and ONE shared Wishart, so neither is
    # permuted.
    got_cov = np.asarray(est.covariances_, dtype=np.float64)
    ref_cov = np.asarray(d[f"cov_{name}"], dtype=np.float64).reshape(got_cov.shape)
    got_dof = np.atleast_1d(np.asarray(est.degrees_of_freedom_, dtype=np.float64))
    if cov != "tied":
        got_cov = got_cov[perm]
        got_dof = got_dof[perm]
    np.testing.assert_allclose(got_cov, ref_cov, atol=atol, rtol=atol)
    np.testing.assert_allclose(
        got_dof,
        np.asarray(d[f"dof_{name}"], dtype=np.float64),
        atol=atol,
        rtol=atol,
    )

    if ptype == "dirichlet_distribution":
        # Exchangeable, so `weights_` and the bound survive a permutation.
        np.testing.assert_allclose(
            np.asarray(est.weights_, dtype=np.float64)[perm],
            np.asarray(d[f"weights_{name}"], dtype=np.float64),
            atol=atol,
            rtol=atol,
        )
        np.testing.assert_allclose(
            est.lower_bound_, float(d[f"lower_bound_{name}"][0]), atol=atol, rtol=atol
        )
    else:
        # Stick-breaking is order-dependent, so sklearn's weights are values
        # for ITS component order. What is checked here is the SHAPE contract
        # the shim owns; the recursion itself is pinned by the Rust suite's
        # fixed-`nk` family.
        a, b = est.weight_concentration_
        assert a.shape == (K,) and b.shape == (K,)
        assert abs(float(np.asarray(est.weights_).sum()) - 1.0) < _atol(fixture)


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("cov", COV_TYPES)
@pytest.mark.parametrize("ptype", list(PRIOR_TYPES))
def test_rng_free_oracle(fixture, cov, ptype):
    """``n_components=1`` with ``init_params='random'``: no RNG, so EVERYTHING
    is compared exactly — every posterior, every resolved prior, ``n_iter_`` /
    ``converged_`` as integer agreements, and the whole scoring surface on the
    disjoint query block.

    At ``k=1`` the ``'random'`` route draws an ``n x 1`` responsibility matrix
    and row-normalizes it, which is ``1.0`` everywhere in both engines whatever
    the stream produced.
    """
    _skip_unsupported_dtype(fixture)
    d = np.load(fixture_path(fixture))
    x, xq = d["X"], d["Xq"]
    tag = PRIOR_TYPES[ptype]
    name = f"k1{cov}_{tag}"
    atol = _atol(fixture)

    est = mlrs.BayesianGaussianMixture(
        n_components=1,
        covariance_type=cov,
        init_params="random",
        weight_concentration_prior_type=ptype,
        tol=1e-8,
        max_iter=200,
        random_state=0,
    ).fit(x)

    def close(got, key, what):
        np.testing.assert_allclose(
            np.ravel(np.asarray(got, dtype=np.float64)),
            np.ravel(np.asarray(d[key], dtype=np.float64)),
            atol=atol,
            rtol=atol,
            err_msg=f"{name}: {what}",
        )

    # -- the resolved priors, each in sklearn's own SHAPE ------------------- #
    close(est.weight_concentration_prior_, f"pwc_{name}", "weight_concentration_prior_")
    close(est.mean_precision_prior_, f"pbeta_{name}", "mean_precision_prior_")
    close(est.mean_prior_, f"pmean_{name}", "mean_prior_")
    close(est.degrees_of_freedom_prior_, f"pdof_{name}", "degrees_of_freedom_prior_")
    close(est.covariance_prior_, f"pcov_{name}", "covariance_prior_")
    expected_prior_shape = {
        "full": (D, D),
        "tied": (D, D),
        "diag": (D,),
        "spherical": (),
    }[cov]
    assert np.shape(est.covariance_prior_) == expected_prior_shape

    # -- the variational posteriors, and their two shapes ------------------- #
    wc = est.weight_concentration_
    if ptype == "dirichlet_process":
        assert isinstance(wc, tuple) and len(wc) == 2
        close(wc[0], f"wca_{name}", "weight_concentration_[0]")
        close(wc[1], f"wcb_{name}", "weight_concentration_[1]")
    else:
        assert isinstance(wc, np.ndarray)
        close(wc, f"wca_{name}", "weight_concentration_")
    close(est.weights_, f"weights_{name}", "weights_")
    close(est.mean_precision_, f"beta_{name}", "mean_precision_")
    close(est.degrees_of_freedom_, f"dof_{name}", "degrees_of_freedom_")
    # sklearn's scalar-vs-array asymmetry under `tied`.
    assert np.ndim(est.degrees_of_freedom_) == (0 if cov == "tied" else 1)
    close(est.means_, f"means_{name}", "means_")
    close(est.covariances_, f"cov_{name}", "covariances_")
    close(est.precisions_cholesky_, f"prec_chol_{name}", "precisions_cholesky_")

    # -- the convergence record --------------------------------------------- #
    close(est.lower_bound_, f"lower_bound_{name}", "lower_bound_")
    close(est.lower_bounds_, f"lower_bounds_{name}", "lower_bounds_")
    assert est.n_iter_ == int(d[f"n_iter_{name}"][0])
    assert est.converged_ == bool(d[f"converged_{name}"][0])

    # -- the scoring surface ------------------------------------------------- #
    np.testing.assert_array_equal(est.predict(xq), d[f"predict_{name}"].astype(np.int32))
    close(est.predict_proba(xq), f"proba_{name}", "predict_proba")
    close(est.score_samples(xq), f"score_samples_{name}", "score_samples")
    close(est.score(xq), f"score_{name}", "score")
    # sklearn's mixtures define no `predict_log_proba`; mlrs's is pinned against
    # `ln(predict_proba)`, which is the definition.
    np.testing.assert_allclose(
        np.exp(np.asarray(est.predict_log_proba(xq), dtype=np.float64)).ravel(),
        np.asarray(d[f"proba_{name}"], dtype=np.float64).ravel(),
        atol=atol,
        rtol=atol,
    )


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("cov", COV_TYPES)
def test_prior_sweep_oracle(fixture, cov):
    """Each of the five priors moved off its default, at ``k=1`` so the
    comparison stays exact.

    These are the parameters with no analogue in ``GaussianMixture``, so nothing
    else in the suite would notice one being dropped on the way across the FFI
    boundary — which is the specific failure this file exists to catch, since
    they cross as scalars and flat lists rather than as arrays.
    """
    _skip_unsupported_dtype(fixture)
    d = np.load(fixture_path(fixture))
    x = d["X"]
    atol = _atol(fixture)
    sweeps = [
        {"weight_concentration_prior": 0.01},
        {"mean_precision_prior": 5.0},
        {"degrees_of_freedom_prior": float(D) + 3.5},
        {"mean_prior": np.array([1.0, -2.0, 0.5, 3.0])},
    ]
    for ptype, tag in PRIOR_TYPES.items():
        for i, kwargs in enumerate(sweeps + [None]):
            if kwargs is None:
                # `covariance_prior`'s shape depends on `covariance_type`, so
                # the value is read back from the fixture rather than restated.
                cp = np.asarray(d[f"cpin_{cov}"], dtype=np.float64)
                kwargs = {"covariance_prior": cp[0] if cov == "spherical" else cp}
                if cov in ("full", "tied"):
                    kwargs["covariance_prior"] = cp.reshape(D, D)
            name = f"pr{i}{cov}_{tag}"
            est = mlrs.BayesianGaussianMixture(
                n_components=1,
                covariance_type=cov,
                init_params="random",
                weight_concentration_prior_type=ptype,
                tol=0.0,
                max_iter=1,
                random_state=0,
                **kwargs,
            ).fit(x)
            for attr, key in (
                ("means_", "means"),
                ("covariances_", "cov"),
                ("mean_precision_", "beta"),
                ("degrees_of_freedom_", "dof"),
                ("weights_", "weights"),
                ("lower_bound_", "lower_bound"),
            ):
                np.testing.assert_allclose(
                    np.ravel(np.asarray(getattr(est, attr), dtype=np.float64)),
                    np.ravel(np.asarray(d[f"{key}_{name}"], dtype=np.float64)),
                    atol=atol,
                    rtol=atol,
                    err_msg=f"{name}: {attr}",
                )


def test_fit_predict_matches_predict():
    """``fit_predict`` returns the terminal E-step's labels, and ``predict`` on
    the training design agrees with them."""
    d = np.load(fixture_path(FIXTURES[0]))
    x = d["X"]
    est = mlrs.BayesianGaussianMixture(n_components=K, random_state=0, max_iter=50)
    labels = est.fit_predict(x)
    assert labels.shape == (x.shape[0],)
    np.testing.assert_array_equal(labels, est.predict(x))


def test_predict_proba_is_a_distribution():
    d = np.load(fixture_path(FIXTURES[0]))
    x, xq = d["X"], d["Xq"]
    est = mlrs.BayesianGaussianMixture(n_components=K, random_state=0).fit(x)
    p = np.asarray(est.predict_proba(xq), dtype=np.float64)
    assert p.shape == (xq.shape[0], K)
    assert (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(p.argmax(axis=1), est.predict(xq))


def test_unknown_string_hyperparameters_raise():
    """All THREE string hyperparameters reject an unknown value at ``fit``."""
    d = np.load(fixture_path(FIXTURES[0]))
    x = d["X"]
    for kwargs in (
        {"covariance_type": "blockdiag"},
        {"init_params": "kmeans++"},
        {"weight_concentration_prior_type": "dirichlet"},
    ):
        with pytest.raises(Exception):
            mlrs.BayesianGaussianMixture(n_components=K, **kwargs).fit(x)


def test_n_components_is_keyword_only():
    """sklearn makes ``n_components`` keyword-only on this estimator (and NOT on
    ``GaussianMixture``); the shim reproduces that, so a positional call that
    would silently work on the wrong estimator fails here."""
    with pytest.raises(TypeError):
        mlrs.BayesianGaussianMixture(3)


def test_sample_returns_the_right_shapes():
    d = np.load(fixture_path(FIXTURES[0]))
    x = d["X"]
    est = mlrs.BayesianGaussianMixture(n_components=K, random_state=0).fit(x)
    xs, ys = est.sample(25)
    assert xs.shape == (25, D)
    assert ys.shape == (25,)
    assert set(np.unique(ys)).issubset(set(range(K)))
