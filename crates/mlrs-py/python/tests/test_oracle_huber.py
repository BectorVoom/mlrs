"""HuberRegressor full-parameter surface through the Python shim (HUBER-01).

Three kinds of check live here, and the split is deliberate:

* **Oracle replay** — the committed ``huber_{f32,f64}_seed42.npz`` fixtures
  (``scripts/gen_oracle.py::gen_huber``) are replayed through the full binding
  path for every converged case. No regeneration: this is a second consumer of
  the same blobs the Rust tests read, so it verifies the SHIM rather than the
  solver.

* **The parameter surface itself** — ``test_ctor_signature_matches_sklearn``
  compares mlrs' ``__init__`` parameter names and defaults against sklearn's,
  and ``test_no_string_valued_parameters`` asserts that NONE of them is a
  string. That second one is the direct answer to "run oracle tests for all
  string-valued parameters": this estimator has none — ``epsilon``,
  ``max_iter``, ``alpha``, ``warm_start``, ``fit_intercept`` and ``tol`` are
  float/int/bool — and the assertion is what keeps that true. If a future
  sklearn adds a ``solver=``-style option, this test and ``gen_huber``'s
  ``StrOptions`` premise both fail rather than the new parameter silently going
  uncovered.

* **Live-sklearn comparison** — the behaviours that live in the shim rather
  than in Rust (``warm_start`` reusing the previous wrapper, the
  ``ConvergenceWarning`` on a hit cap, ``sample_weight`` plumbing) are compared
  against a live sklearn instance, because a committed fixture would only pin
  the shim against itself.

f64 fixtures are skipped-with-reason on an f64-incapable backend (rocm) via the
``conftest.requires_f64`` marker.
"""

import inspect
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor as SkHuber

import mlrs
from conftest import dtype_of, fixture_path, requires_f64

FIXTURES = ["huber_f32_seed42", "huber_f64_seed42"]

N_SAMPLES, N_FEATURES = 240, 6

# The CONVERGED cases, as `(fixture key, ctor kwargs, uses sample_weight)`.
# Mirrors `huber_test.rs::VALUE_CASES` exactly — the truncated `max_iter` cases
# are deliberately absent: they stop mid-trajectory, and mlrs' strong-Wolfe line
# search is not scipy's Moré-Thuente, so their iterates are not comparable. The
# Rust suite gates what IS well-posed about them.
VALUE_CASES = [
    ("default", {}, False),
    ("noint", {"fit_intercept": False}, False),
    ("eps105", {"epsilon": 1.05}, False),
    ("eps2", {"epsilon": 2.5}, False),
    ("eps10", {"epsilon": 10.0}, False),
    ("eps105_noint", {"epsilon": 1.05, "fit_intercept": False}, False),
    ("alpha0", {"alpha": 0.0}, False),
    ("alpha1", {"alpha": 1.0}, False),
    ("alpha100", {"alpha": 100.0}, False),
    ("tol_tight", {"tol": 1e-12}, False),
    ("sw", {}, True),
    ("sw_noint", {"fit_intercept": False}, True),
    ("sw_eps105", {"epsilon": 1.05}, True),
    ("sw_alpha1", {"alpha": 1.0}, True),
]

# Multiple of sklearn's OWN measured distance from the minimizer that the band
# allows, and the floor under it. Same derivation as `huber_test.rs`: sklearn
# leaves scipy's `factr` at `1e7` and so stops on the relative-f criterion
# before its gradient test can fire, which puts a floor on how closely ANY
# implementation can agree with it. The fixture ships that residual per case.
RESIDUAL_SLACK = 4.0
BAND_FLOOR = {np.float64: 1e-7, np.float32: 2e-3}


def _load(fixture):
    return np.load(fixture_path(fixture + ".npz"))


def _band(blob, name, dtype):
    return max(RESIDUAL_SLACK * float(blob[f"residual_{name}"][0]), BAND_FLOOR[dtype])


@pytest.mark.parametrize("fixture", FIXTURES)
@requires_f64
def test_value_cases_match_fixture(fixture):
    """Every converged case: coef_, intercept_, scale_, outliers_, predict."""
    blob = _load(fixture)
    dtype = dtype_of(fixture)
    x = np.ascontiguousarray(blob["X"].astype(dtype))
    y = np.ascontiguousarray(blob["y"].astype(dtype))
    sw = np.ascontiguousarray(blob["sample_weight"].astype(dtype))
    x_test = np.ascontiguousarray(blob["X_test"].astype(dtype))

    for name, kwargs, use_sw in VALUE_CASES:
        band = _band(blob, name, dtype)
        est = mlrs.HuberRegressor(max_iter=1000, **kwargs)
        est.fit(x, y, sample_weight=sw if use_sw else None)

        np.testing.assert_allclose(
            np.asarray(est.coef_, dtype=np.float64),
            blob[f"coef_{name}"].astype(np.float64),
            atol=band,
            rtol=band,
            err_msg=f"{fixture}/{name}: coef_",
        )
        np.testing.assert_allclose(
            float(est.intercept_),
            float(blob[f"intercept_{name}"][0]),
            atol=band,
            rtol=band,
            err_msg=f"{fixture}/{name}: intercept_",
        )
        np.testing.assert_allclose(
            float(est.scale_),
            float(blob[f"scale_{name}"][0]),
            atol=band,
            rtol=band,
            err_msg=f"{fixture}/{name}: scale_",
        )
        np.testing.assert_allclose(
            np.asarray(est.predict(x_test), dtype=np.float64),
            blob[f"pred_{name}"].astype(np.float64),
            atol=band,
            rtol=band,
            err_msg=f"{fixture}/{name}: predict",
        )

        # `outliers_` is a boolean mask, so it is compared EXACTLY where the
        # fixture proved no sample sits within reach of the two solvers' gap,
        # and by count otherwise (f32 designs always by count: a mask flip
        # there is float round-off, not the estimator).
        expected_mask = blob[f"outliers_{name}"].astype(bool)
        got_mask = np.asarray(est.outliers_, dtype=bool)
        assert got_mask.shape == (N_SAMPLES,), f"{fixture}/{name}: outliers_ shape"
        stable = bool(blob[f"outliers_stable_{name}"][0])
        if stable and dtype is np.float64:
            np.testing.assert_array_equal(
                got_mask, expected_mask, err_msg=f"{fixture}/{name}: outliers_"
            )
        else:
            assert abs(int(got_mask.sum()) - int(expected_mask.sum())) <= 2, (
                f"{fixture}/{name}: outlier count "
                f"{int(got_mask.sum())} vs {int(expected_mask.sum())}"
            )

        assert 0 < est.n_iter_ <= 1000, f"{fixture}/{name}: n_iter_ = {est.n_iter_}"


def test_ctor_signature_matches_sklearn():
    """mlrs' ctor carries sklearn's parameters, with sklearn's defaults.

    `output_type` and `device` are mlrs-only — the first selects the array
    library results come back in, the second where the objective runs
    (DEVICE-PARAM-01), and NEITHER changes the fitted model. Both are excluded;
    everything else must match name-for-name and default-for-default, because a
    shim whose defaults have drifted silently changes the model a user gets from
    `HuberRegressor()`.
    """
    sk = inspect.signature(SkHuber.__init__).parameters
    ours = inspect.signature(mlrs.HuberRegressor.__init__).parameters

    sk_names = {n for n in sk if n != "self"}
    our_names = {n for n in ours if n not in ("self", "output_type", "device")}
    assert our_names == sk_names, (
        f"parameter surface drift: mlrs-only={our_names - sk_names}, "
        f"sklearn-only={sk_names - our_names}"
    )
    for n in sorted(sk_names):
        assert ours[n].default == sk[n].default, (
            f"default drift on `{n}`: mlrs={ours[n].default!r} "
            f"sklearn={sk[n].default!r}"
        )


def test_no_string_valued_parameters():
    """HuberRegressor has NO string-valued parameter, and this pins it.

    Every parameter that reaches the MODEL is a float, an int or a bool —
    there is no `solver=`, no `loss=`, nothing with a `StrOptions` constraint —
    so there is no string-valued parameter for an oracle case to cover.
    `gen_huber` asserts the same fact against sklearn's own
    `_parameter_constraints`; this asserts it against the defaults both classes
    actually expose. If a future sklearn adds one, both fail rather than the new
    parameter going untested.

    mlrs' own `output_type` and `device` are strings but are excluded: they
    select the egress array library and the execution arm, and a value-neutral
    knob has nothing for an ORACLE to compare against sklearn. `device` is
    covered instead by `test_device_param.py`, which asserts the two arms agree.
    """
    for cls in (SkHuber, mlrs.HuberRegressor):
        params = inspect.signature(cls.__init__).parameters
        strings = {
            n: p.default
            for n, p in params.items()
            if n not in ("self", "output_type", "device")
            and isinstance(p.default, str)
        }
        assert not strings, (
            f"{cls.__module__}.{cls.__name__} grew string-valued parameter(s) "
            f"{strings} — add oracle coverage for them and update this test"
        )


@requires_f64
def test_warm_start_continues_the_previous_fit():
    """A second `fit` with `warm_start=True` seeds from the first's parameters.

    Checked against a live sklearn instance rather than a fixture: the seeding
    lives in the SHIM (it has to reuse the previous wrapper, because the Rust
    `fit` consumes the estimator), so what matters is that the behaviour matches
    sklearn's, not that it reproduces a stored number.
    """
    rng = np.random.default_rng(3)
    n, d = 300, 5
    x = rng.standard_normal((n, d))
    y = x @ rng.standard_normal(d) + 1.5 + 0.4 * rng.standard_normal(n)
    idx = rng.choice(n, 24, replace=False)
    y[idx] += 25.0 * rng.standard_normal(24) + 15.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ours = mlrs.HuberRegressor(warm_start=True, max_iter=5)
        ours.fit(x, y)
        first = np.asarray(ours.coef_, dtype=np.float64).copy()
        ours.fit(x, y)
        second = np.asarray(ours.coef_, dtype=np.float64)

        theirs = SkHuber(warm_start=True, max_iter=5)
        theirs.fit(x, y)
        sk_first = theirs.coef_.copy()
        theirs.fit(x, y)
        sk_second = theirs.coef_

    assert not np.allclose(first, second), "mlrs: the warm-started refit did not move"
    assert not np.allclose(sk_first, sk_second), "sklearn: the refit did not move"
    # Both continued rather than restarted: the second fit is closer to the
    # converged answer than the first.
    converged = SkHuber(max_iter=1000).fit(x, y).coef_
    assert np.abs(second - converged).max() < np.abs(first - converged).max()

    # Without `warm_start` the same cap restarts cold and lands where the FIRST
    # warm fit did.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        cold = mlrs.HuberRegressor(max_iter=5)
        cold.fit(x, y)
        cold.fit(x, y)
    np.testing.assert_allclose(
        np.asarray(cold.coef_, dtype=np.float64), first, atol=1e-12
    )


@requires_f64
def test_hit_cap_warns_like_sklearn():
    """A `max_iter`-truncated fit raises `ConvergenceWarning`; a converged one does not.

    The second half is the regression gate on a real bug: mlrs stops on the
    `ftol` plateau for essentially every well-conditioned fit (deliberately —
    that is a tighter place to stop than scipy's `factr`), and treating that as
    non-convergence made EVERY default fit warn while sitting closer to the
    optimum than sklearn.
    """
    rng = np.random.default_rng(5)
    n, d = 300, 5
    x = rng.standard_normal((n, d))
    y = x @ rng.standard_normal(d) + 1.5 + 0.4 * rng.standard_normal(n)
    y[rng.choice(n, 24, replace=False)] += 30.0

    with pytest.warns(ConvergenceWarning):
        mlrs.HuberRegressor(max_iter=2).fit(x, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        mlrs.HuberRegressor(max_iter=1000).fit(x, y)


@requires_f64
def test_sample_weight_changes_the_fit_and_matches_sklearn():
    """`sample_weight` reaches the solver and lands where sklearn does."""
    rng = np.random.default_rng(9)
    n, d = 400, 5
    x = rng.standard_normal((n, d))
    y = x @ rng.standard_normal(d) + 1.5 + 0.4 * rng.standard_normal(n)
    y[rng.choice(n, 32, replace=False)] += 30.0
    sw = np.abs(rng.standard_normal(n)) + 0.25

    ours = mlrs.HuberRegressor(max_iter=1000).fit(x, y, sample_weight=sw)
    plain = mlrs.HuberRegressor(max_iter=1000).fit(x, y)
    theirs = SkHuber(max_iter=1000).fit(x, y, sample_weight=sw)

    assert not np.allclose(
        np.asarray(ours.coef_), np.asarray(plain.coef_)
    ), "sample_weight did not change the fit"
    np.testing.assert_allclose(
        np.asarray(ours.coef_, dtype=np.float64), theirs.coef_, atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(float(ours.scale_), float(theirs.scale_), rtol=1e-4)


@requires_f64
def test_weighted_fit_equals_repeated_fit():
    """Integer `sample_weight` == repeating rows, to MACHINE PRECISION.

    Every term of the Huber objective is linear in the weights, so this is an
    exact identity rather than an approximation, and it is the real gate on the
    weighted row loop (which is a separate monomorphization from the unweighted
    one — see `huber_objective.rs`'s `WEIGHTED` const generic).

    sklearn's `check_sample_weight_equivalence_on_dense_data` tests the same
    identity but at `rtol=1e-7` on an ill-conditioned 30-row fixture, which is
    three orders below any iterative solver's stopping accuracy there —
    scikit-learn's own HuberRegressor fails it too, so mlrs xfails it in
    `test_estimator_checks.py` and gates the identity HERE instead, on a design
    where both solves actually converge.
    """
    rng = np.random.default_rng(0)
    n, d = 40, 4
    x = rng.standard_normal((n, d))
    y = x @ rng.standard_normal(d) + 1.0 + 0.3 * rng.standard_normal(n)
    y[rng.choice(n, 4, replace=False)] += 20.0
    sw = rng.integers(0, 5, size=n).astype(float)

    x_rep = np.repeat(x, sw.astype(int), axis=0)
    y_rep = np.repeat(y, sw.astype(int))

    weighted = mlrs.HuberRegressor(max_iter=1000).fit(x, y, sample_weight=sw)
    repeated = mlrs.HuberRegressor(max_iter=1000).fit(x_rep, y_rep)

    np.testing.assert_allclose(
        np.asarray(weighted.coef_, dtype=np.float64),
        np.asarray(repeated.coef_, dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        float(weighted.scale_), float(repeated.scale_), atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        float(weighted.intercept_), float(repeated.intercept_), atol=1e-12, rtol=1e-12
    )


@requires_f64
def test_builder_rejects_what_sklearn_rejects():
    """`epsilon < 1`, `alpha < 0` and `tol < 0` are ValueErrors, as in sklearn.

    And the permissive boundaries sklearn ALLOWS are allowed here too:
    `epsilon == 1`, `max_iter == 0`, `alpha == 0`, `tol == 0`.
    """
    rng = np.random.default_rng(11)
    x = rng.standard_normal((60, 4))
    y = x @ rng.standard_normal(4) + 1.0

    for kwargs in ({"epsilon": 0.9}, {"alpha": -1.0}, {"tol": -1e-9}):
        with pytest.raises(ValueError):
            mlrs.HuberRegressor(**kwargs).fit(x, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for kwargs in (
            {"epsilon": 1.0},
            {"max_iter": 0},
            {"alpha": 0.0},
            {"tol": 0.0},
        ):
            mlrs.HuberRegressor(**kwargs).fit(x, y)
