"""RANSACRegressor full-parameter surface through the Python shim (RANSAC-01).

Four kinds of check live here, and the split is deliberate:

* **Oracle replay** — the committed ``ransac_{f32,f64}_seed42.npz`` fixtures
  (``scripts/gen_oracle.py::gen_ransac``) are replayed through the full binding
  path for every case. No regeneration: this is a second consumer of the same
  blobs the Rust tests read, so it verifies the SHIM rather than the engine.

* **The string-valued parameter** — ``loss`` is the only one, and
  ``test_only_loss_is_string_valued`` asserts that against sklearn's own
  ``_parameter_constraints`` so a future option cannot go silently uncovered.
  Both of its values get an oracle case, and
  ``test_unknown_loss_string_is_rejected`` pins the rejection message.

* **The two-path routing** — this shim has a native Rust path and a Python
  fallback for base estimators Rust cannot host (see ``mlrs/linear.py``).
  ``test_fallback_matches_rust_path`` drives the SAME configuration down both
  and asserts they agree, which is what makes the fallback a fallback rather
  than a second, differently-wrong implementation.

* **Live-sklearn comparison** — the behaviours that live in the shim rather than
  in Rust (the ``ConvergenceWarning``, the ``estimator_`` materialization, the
  borrowed ``RandomState`` advancing exactly as sklearn's would, a raising
  callback surfacing its OWN exception) are compared against a live sklearn
  instance, because a committed fixture would only pin the shim against itself.

f64 fixtures are skipped-with-reason on an f64-incapable backend (rocm) via the
``conftest.requires_f64`` marker.
"""

import inspect
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import RANSACRegressor as SkRANSAC
from sklearn.utils._param_validation import StrOptions

import mlrs
from conftest import dtype_of, fixture_path, requires_f64

FIXTURES = ["ransac_f32_seed42", "ransac_f64_seed42"]

N_SAMPLES, N_FEATURES, N_TEST = 300, 5, 11
SEED = 42

# Every case in the fixture, as `(key, ctor kwargs, uses sample_weight)`.
# Mirrors `ransac_test.rs::CASES` exactly.
CASES = [
    ("default", {}, False),
    ("loss_abs", {"loss": "absolute_error"}, False),
    ("loss_sq", {"loss": "squared_error"}, False),
    ("ms_int", {"min_samples": 20}, False),
    ("ms_frac", {"min_samples": 0.3}, False),
    ("ms_one", {"min_samples": 1}, False),
    ("rt_tight", {"residual_threshold": 0.4}, False),
    ("rt_loose", {"residual_threshold": 8.0}, False),
    ("trials_3", {"max_trials": 3}, False),
    ("trials_500", {"max_trials": 500}, False),
    ("stopprob_low", {"stop_probability": 0.1}, False),
    ("stopprob_high", {"stop_probability": 0.999999}, False),
    ("stopprob_one", {"stop_probability": 1.0}, False),
    ("stop_inliers", {"stop_n_inliers": 200}, False),
    ("stop_score", {"stop_score": 0.5}, False),
    ("max_skips", {"residual_threshold": 0.05, "max_skips": 3}, False),
    (
        "noint",
        {
            "estimator": SkLinearRegression(fit_intercept=False),
            "min_samples": N_FEATURES + 1,
        },
        False,
    ),
    ("data_valid", {"is_data_valid": lambda X, y: float(np.abs(X).max()) < 2.0}, False),
    (
        "model_valid",
        {"is_model_valid": lambda m, X, y: float(np.abs(m.coef_).max()) < 3.0},
        False,
    ),
    ("sw", {}, True),
    ("sw_ms20", {"min_samples": 20}, True),
    ("sw_sq", {"loss": "squared_error"}, True),
    (
        "sw_noint",
        {
            "estimator": SkLinearRegression(fit_intercept=False),
            "min_samples": N_FEATURES + 1,
        },
        True,
    ),
]

# Comparison bands. Both libraries run the SAME algorithm on the SAME
# sub-samples (mlrs reproduces numpy's MT19937), and both solve the consensus
# least-squares exactly, so the f64 gap is pure summation order. The f32 band is
# wider for the reason `ransac_test.rs` documents: scipy solves the sub-sample
# at the design's own precision while mlrs solves it in f64, so the agreement is
# bounded by f32's precision rather than by either solver's.
BAND = {np.float64: 1e-9, np.float32: 2e-5}
# Relative distance from `residual_threshold` the fixture measured for the
# nearest row. Below this the exact `inlier_mask_` comparison is not well-posed
# and only the consensus SIZE is checked (`ransac_test.rs` module docs).
MASK_EXACT_MARGIN = 1e-4


def _load(fixture):
    return np.load(fixture_path(fixture + ".npz"))


def _design(z, dt):
    x = np.asarray(z["X"], dtype=dt).reshape(N_SAMPLES, N_FEATURES)
    y = np.asarray(z["y"], dtype=dt).reshape(N_SAMPLES)
    xt = np.asarray(z["X_test"], dtype=dt).reshape(N_TEST, N_FEATURES)
    sw = np.asarray(z["sample_weight"], dtype=np.float64).reshape(N_SAMPLES)
    return x, y, xt, sw


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("case,kwargs,use_sw", CASES, ids=[c[0] for c in CASES])
@requires_f64
def test_oracle_case(fixture, case, kwargs, use_sw):
    """Every fitted attribute the fixture ships, through the binding path."""
    z = _load(fixture)
    dt = dtype_of(fixture)
    x, y, xt, sw = _design(z, dt)

    est = mlrs.RANSACRegressor(random_state=SEED, **kwargs)
    with warnings.catch_warnings():
        # The `max_skips` case is DESIGNED to trip the ConvergenceWarning;
        # `test_convergence_warning_matches_sklearn` is what pins it.
        warnings.simplefilter("ignore", ConvergenceWarning)
        est.fit(x, y, sample_weight=sw if use_sw else None)

    band = BAND[dt]
    # The TRAJECTORY first: an implementation that reached a similar consensus
    # by a different route would pass a coefficient check and fail this one.
    counters = z[f"counters_{case}"]
    assert (
        est.n_trials_,
        est.n_skips_no_inliers_,
        est.n_skips_invalid_data_,
        est.n_skips_invalid_model_,
    ) == tuple(int(v) for v in counters)

    exp_mask = z[f"inlier_mask_{case}"].astype(bool)
    got_mask = np.asarray(est.inlier_mask_, dtype=bool)
    if float(z[f"margin_{case}"][0]) > MASK_EXACT_MARGIN:
        np.testing.assert_array_equal(got_mask, exp_mask)
    else:
        assert int(got_mask.sum()) == int(exp_mask.sum())

    np.testing.assert_allclose(
        np.ravel(est.estimator_.coef_), z[f"coef_{case}"], rtol=band, atol=band
    )
    np.testing.assert_allclose(
        np.ravel(np.atleast_1d(est.estimator_.intercept_)),
        z[f"intercept_{case}"],
        rtol=band,
        atol=band,
    )
    np.testing.assert_allclose(
        np.asarray(est.predict(xt), dtype=np.float64),
        np.asarray(z[f"pred_{case}"], dtype=np.float64),
        rtol=band,
        atol=band,
    )


# --------------------------------------------------------------------------- #
# The parameter surface itself
# --------------------------------------------------------------------------- #


def test_ctor_signature_matches_sklearn():
    """Same parameter NAMES and DEFAULTS as sklearn, plus mlrs' `output_type`."""
    ours = inspect.signature(mlrs.RANSACRegressor.__init__).parameters
    theirs = inspect.signature(SkRANSAC.__init__).parameters
    extra = set(ours) - set(theirs)
    assert extra == {"output_type"}, f"unexpected extra parameters: {extra}"
    missing = set(theirs) - set(ours)
    assert not missing, f"missing sklearn parameters: {missing}"
    for name, p in theirs.items():
        if name == "self":
            continue
        assert ours[name].default == p.default or (
            np.isinf(ours[name].default) and np.isinf(p.default)
        ), f"{name}: default {ours[name].default!r} != sklearn's {p.default!r}"


def test_only_loss_is_string_valued():
    """`loss` is the ONE string-valued parameter, and this is what keeps it so.

    Read off sklearn's own ``_parameter_constraints`` rather than asserted from
    memory, so a future sklearn that adds a second ``StrOptions`` parameter
    fails HERE (and in ``gen_ransac``'s matching premise) instead of letting it
    go untested.
    """
    str_params = {
        name: sorted(next(c.options for c in cs if isinstance(c, StrOptions)))
        for name, cs in SkRANSAC._parameter_constraints.items()
        if any(isinstance(c, StrOptions) for c in cs)
    }
    assert str_params == {"loss": ["absolute_error", "squared_error"]}
    # And both options have an oracle case above.
    covered = {kw.get("loss") for _n, kw, _s in CASES if "loss" in kw}
    assert covered == {"absolute_error", "squared_error"}


def test_unknown_loss_string_is_rejected():
    """An option sklearn does not offer raises, with sklearn's own wording."""
    x = np.zeros((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError, match="loss"):
        mlrs.RANSACRegressor(loss="huber").fit(x, y)
    with pytest.raises(ValueError, match="loss"):
        SkRANSAC(loss="huber").fit(x, y)


@requires_f64
def test_callable_loss_takes_the_python_path_and_still_matches():
    """A callable ``loss`` is the one RANSAC-own parameter the Rust engine
    cannot take (it needs a materialized ``y_pred``), so it routes to the shim's
    Python loop — which must still agree with sklearn index for index."""
    z = _load("ransac_f64_seed42")
    x, y, xt, _sw = _design(z, np.float64)

    def loss(y_true, y_pred):
        return np.abs(y_true - y_pred)

    ours = mlrs.RANSACRegressor(loss=loss, random_state=SEED).fit(x, y)
    theirs = SkRANSAC(loss=loss, random_state=SEED).fit(x, y)
    assert ours._from_rust is False
    assert ours.n_trials_ == theirs.n_trials_
    np.testing.assert_array_equal(
        np.asarray(ours.inlier_mask_), theirs.inlier_mask_
    )
    np.testing.assert_allclose(
        np.ravel(ours.estimator_.coef_), np.ravel(theirs.estimator_.coef_), atol=1e-9
    )


@requires_f64
def test_fallback_matches_rust_path():
    """The Python fallback and the Rust engine agree on the same configuration.

    Driven by ``estimator=Ridge(alpha=0)`` — a base the Rust engine declines, so
    the shim takes its Python loop — against the default ``LinearRegression``
    base, which it hosts. A ridge with no penalty IS ordinary least squares, so
    the two must land on the same consensus and (to solver precision) the same
    coefficients. That is what makes the fallback a fallback and not a second
    implementation free to disagree.
    """
    from sklearn.linear_model import Ridge

    z = _load("ransac_f64_seed42")
    x, y, _xt, _sw = _design(z, np.float64)

    rust = mlrs.RANSACRegressor(random_state=SEED).fit(x, y)
    fallback = mlrs.RANSACRegressor(
        estimator=Ridge(alpha=0.0, solver="lsqr"),
        min_samples=N_FEATURES + 1,
        random_state=SEED,
    ).fit(x, y)
    assert rust._from_rust is True
    assert fallback._from_rust is False
    assert rust.n_trials_ == fallback.n_trials_
    np.testing.assert_array_equal(
        np.asarray(rust.inlier_mask_), np.asarray(fallback.inlier_mask_)
    )
    np.testing.assert_allclose(
        np.ravel(rust.estimator_.coef_),
        np.ravel(fallback.estimator_.coef_),
        rtol=1e-6,
        atol=1e-6,
    )


# --------------------------------------------------------------------------- #
# Live-sklearn comparison: what lives in the shim, not in Rust
# --------------------------------------------------------------------------- #


@requires_f64
def test_convergence_warning_matches_sklearn():
    """A consensus found DESPITE blowing ``max_skips`` warns; it does not raise."""
    z = _load("ransac_f64_seed42")
    x, y, _xt, _sw = _design(z, np.float64)
    kw = dict(residual_threshold=0.05, max_skips=3, random_state=SEED)
    with pytest.warns(ConvergenceWarning, match="max_skips"):
        mlrs.RANSACRegressor(**kw).fit(x, y)
    with pytest.warns(ConvergenceWarning, match="max_skips"):
        SkRANSAC(**kw).fit(x, y)


@requires_f64
def test_no_consensus_set_raises_sklearns_message():
    """No consensus at all is a ``ValueError``, with sklearn's exact wording."""
    z = _load("ransac_f64_seed42")
    x, y, _xt, _sw = _design(z, np.float64)
    never = lambda X, yy: False  # noqa: E731
    for cls in (mlrs.RANSACRegressor, SkRANSAC):
        with pytest.raises(ValueError, match="could not find a valid consensus set"):
            cls(is_data_valid=never, random_state=SEED).fit(x, y)
        with pytest.raises(ValueError, match="skipped more iterations"):
            cls(is_data_valid=never, max_skips=3, random_state=SEED).fit(x, y)


@requires_f64
def test_min_samples_larger_than_n_samples_raises_sklearns_message():
    z = _load("ransac_f64_seed42")
    x, y, _xt, _sw = _design(z, np.float64)
    for cls in (mlrs.RANSACRegressor, SkRANSAC):
        with pytest.raises(ValueError, match="may not be larger than number of"):
            cls(min_samples=N_SAMPLES + 1, random_state=SEED).fit(x, y)


@requires_f64
def test_a_raising_callback_surfaces_its_own_exception():
    """The bridge stashes the real ``PyErr`` and re-raises it after the unwind,
    so a user's ``KeyError`` reaches them as their ``KeyError`` — not as the
    core's ``CallbackAborted`` placeholder."""
    z = _load("ransac_f64_seed42")
    x, y, _xt, _sw = _design(z, np.float64)

    def boom(X, yy):
        raise KeyError("bridge boom")

    for cls in (mlrs.RANSACRegressor, SkRANSAC):
        with pytest.raises(KeyError, match="bridge boom"):
            cls(is_data_valid=boom, random_state=SEED).fit(x, y)


@requires_f64
def test_random_state_object_is_advanced_like_sklearns():
    """A caller who passes a live ``RandomState`` observes it advance to exactly
    where sklearn would have left it — the borrow contract
    ``mlrs.model_selection._rust_rng`` implements."""
    z = _load("ransac_f64_seed42")
    x, y, _xt, _sw = _design(z, np.float64)
    draws = []
    for cls in (mlrs.RANSACRegressor, SkRANSAC):
        rs = np.random.RandomState(SEED)
        cls(random_state=rs).fit(x, y)
        draws.append(int(rs.randint(0, 2**31 - 1)))
    assert draws[0] == draws[1]


@requires_f64
def test_multi_output_matches_sklearn():
    """A 2-D ``y`` sums the loss over the target axis, and ``coef_`` keeps
    sklearn's ``(n_targets, n_features)`` shape."""
    rng = np.random.default_rng(11)
    n, d, t = 250, 4, 3
    x = rng.standard_normal((n, d))
    yy = x @ rng.standard_normal((d, t)) + 1.0 + 0.1 * rng.standard_normal((n, t))
    yy[rng.choice(n, 50, replace=False)] += 30.0

    ours = mlrs.RANSACRegressor(random_state=SEED).fit(x, yy)
    theirs = SkRANSAC(random_state=SEED).fit(x, yy)
    assert ours.n_trials_ == theirs.n_trials_
    np.testing.assert_array_equal(np.asarray(ours.inlier_mask_), theirs.inlier_mask_)
    assert np.asarray(ours.estimator_.coef_).shape == (t, d)
    np.testing.assert_allclose(
        np.asarray(ours.predict(x[:20])), theirs.predict(x[:20]), atol=1e-9
    )


@requires_f64
def test_score_delegates_to_the_consensus_model():
    z = _load("ransac_f64_seed42")
    x, y, _xt, _sw = _design(z, np.float64)
    ours = mlrs.RANSACRegressor(random_state=SEED).fit(x, y)
    theirs = SkRANSAC(random_state=SEED).fit(x, y)
    assert ours.score(x, y) == pytest.approx(theirs.score(x, y), abs=1e-9)


def test_unfitted_attribute_access_raises_not_fitted():
    from sklearn.exceptions import NotFittedError

    est = mlrs.RANSACRegressor()
    for attr in ("estimator_", "inlier_mask_", "n_trials_", "n_skips_no_inliers_"):
        with pytest.raises(NotFittedError):
            getattr(est, attr)
