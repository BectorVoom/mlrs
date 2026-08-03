"""sklearn oracle for the NB family's ``fit(X, y, sample_weight=...)`` parameter.

The five ``mlrs.naive_bayes`` shims mirror sklearn's fit signature, and
``sample_weight`` is the only parameter it carries beyond ``X`` and ``y``. Its
semantics are sklearn's:

  * the discrete four scale each row's contribution to the per-class counts —
    ``Y *= sample_weight.T`` before ``feature_count_ += Y.T @ X`` — so
    ``class_count_`` becomes ``Σ w`` per class;
  * ``GaussianNB`` takes weighted per-class means and variances, but its
    ``epsilon_`` variance floor stays the **unweighted** ``var_smoothing ·
    max_j Var(X[:,j])`` (sklearn computes it before it looks at the weights).

This file is the ORACLE for that: every case fits both engines on the same
input and compares ``predict_log_proba``, which reads the fitted tables through
the decision function undamped (a divergence in any count, prior or variance
lands here). The Rust side additionally gates the weighted-equals-repeated
invariant bitwise-ish in ``crates/mlrs-algos/tests/*_nb_test.rs``; this proves
the values themselves agree with sklearn across the FFI.

Run requires the compiled ``_mlrs`` (``maturin develop`` with a backend
pyproject); collected-and-skipped otherwise so the suite stays green pre-build.
"""

import numpy as np
import pytest

pytest.importorskip("mlrs")

import sklearn.naive_bayes as sk_nb  # noqa: E402
from mlrs import naive_bayes as mlrs_nb  # noqa: E402

# max |Δ log P(c|x)| tolerated against sklearn. The joint log-likelihood is a
# host f64 sum of table lookups in both engines, so the gap is round-off; this
# is the project-wide 1e-5 oracle gate with room to spare.
BAND = 1e-9

N, D, C = 120, 6, 3


def _data(kind, seed=0):
    """The input each estimator is for, plus a weight vector with zeros in it.

    Zero weights are the load-bearing case: they must DROP the row, which is
    what separates a real weighted fit from one that quietly ignores the
    parameter.
    """
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        x = rng.standard_normal((N, D))
    elif kind == "categorical":
        x = rng.integers(0, 4, size=(N, D)).astype(np.float64)
    elif kind == "bernoulli":
        x = (rng.random((N, D)) < 0.4).astype(np.float64)
    else:
        x = rng.poisson(2.0, size=(N, D)).astype(np.float64)
    y = rng.integers(0, C, size=N)
    # Integer weights incl. zeros, and a fractional set — sklearn supports both
    # and only the fractional one rules out an "expand to repeated rows"
    # implementation.
    w_int = (np.arange(N) % 4).astype(np.float64)
    w_frac = rng.random(N) * 2.0 + 0.25
    return x, y, w_int, w_frac


ESTIMATORS = ["gaussian", "multinomial", "bernoulli", "complement", "categorical"]
NAMES = {
    "gaussian": "GaussianNB",
    "multinomial": "MultinomialNB",
    "bernoulli": "BernoulliNB",
    "complement": "ComplementNB",
    "categorical": "CategoricalNB",
}


def _pair(kind):
    name = NAMES[kind]
    return getattr(mlrs_nb, name), getattr(sk_nb, name)


def _assert_agrees(kind, x, y, w):
    """Fit both engines with `w` and compare log-probabilities on `x`."""
    Mlrs, Sk = _pair(kind)
    got = np.asarray(
        Mlrs().fit(x, y, sample_weight=w).predict_log_proba(x), dtype=np.float64
    )
    want = np.asarray(
        Sk().fit(x, y, sample_weight=w).predict_log_proba(x), dtype=np.float64
    )
    assert got.shape == want.shape
    dev = float(np.max(np.abs(got - want)))
    assert dev <= BAND, f"{kind}: max|Δlog P(c|x)| = {dev:.3e} > {BAND:.1e}"


@pytest.mark.parametrize("kind", ESTIMATORS)
def test_integer_sample_weight_matches_sklearn(kind):
    """Integer weights (including ZEROS, which must drop the row)."""
    x, y, w_int, _ = _data(kind)
    _assert_agrees(kind, x, y, w_int)


@pytest.mark.parametrize("kind", ESTIMATORS)
def test_fractional_sample_weight_matches_sklearn(kind):
    """Fractional weights — no repeated-row implementation can fake these."""
    x, y, _, w_frac = _data(kind)
    _assert_agrees(kind, x, y, w_frac)


@pytest.mark.parametrize("kind", ESTIMATORS)
def test_all_ones_weight_equals_unweighted(kind):
    """``sample_weight=1`` everywhere is the unweighted fit, in BOTH engines."""
    x, y, _, _ = _data(kind)
    Mlrs, _sk = _pair(kind)
    ones = np.ones(N)
    weighted = np.asarray(
        Mlrs().fit(x, y, sample_weight=ones).predict_log_proba(x), dtype=np.float64
    )
    plain = np.asarray(Mlrs().fit(x, y).predict_log_proba(x), dtype=np.float64)
    assert np.max(np.abs(weighted - plain)) <= BAND


@pytest.mark.parametrize("kind", ESTIMATORS)
def test_integer_weight_equals_repeated_rows(kind):
    """The sklearn contract `check_sample_weight_equivalence_on_dense_data` states:
    an integer weight is indistinguishable from repeating that row."""
    x, y, w_int, _ = _data(kind)
    Mlrs, _sk = _pair(kind)
    xr = np.repeat(x, w_int.astype(int), axis=0)
    yr = np.repeat(y, w_int.astype(int))
    weighted = np.asarray(
        Mlrs().fit(x, y, sample_weight=w_int).predict_log_proba(x), dtype=np.float64
    )
    repeated = np.asarray(Mlrs().fit(xr, yr).predict_log_proba(x), dtype=np.float64)
    if kind == "gaussian":
        # GaussianNB's `epsilon_` floor is the UNWEIGHTED max column variance,
        # and repeating rows changes it — so the two fits differ by that floor
        # alone. It is `var_smoothing = 1e-9` times a variance, i.e. far below
        # the class-conditional terms; a loose band still catches a real
        # divergence in the means or variances.
        assert np.max(np.abs(weighted - repeated)) <= 1e-6
    else:
        assert np.max(np.abs(weighted - repeated)) <= BAND


@pytest.mark.parametrize("kind", ESTIMATORS)
def test_bad_sample_weight_raises_value_error(kind):
    """Length mismatch, non-finite, negative, and all-zero are all ValueError.

    sklearn raises for the first, second and fourth; mlrs additionally rejects a
    NEGATIVE weight, which sklearn's NB passes straight through into a
    ``log`` of a negative count (a silent NaN model). That divergence is
    deliberate — see ``linear/ridge.rs::validate_sample_weight``.
    """
    x, y, _, _ = _data(kind)
    Mlrs, _sk = _pair(kind)
    for label, w in [
        ("too short", np.ones(N - 1)),
        ("NaN", np.r_[np.nan, np.ones(N - 1)]),
        ("negative", np.r_[-1.0, np.ones(N - 1)]),
        ("all zero", np.zeros(N)),
    ]:
        with pytest.raises(ValueError):
            Mlrs().fit(x, y, sample_weight=w)


@pytest.mark.parametrize("kind", ESTIMATORS)
def test_all_zero_sample_weight_message(kind):
    """``check_all_zero_sample_weights_error`` greps the message for both
    "weight" and "zero" — lock the wording so a reworded error cannot silently
    fail that estimator check."""
    import re

    x, y, _, _ = _data(kind)
    Mlrs, _sk = _pair(kind)
    with pytest.raises(ValueError) as exc:
        Mlrs().fit(x, y, sample_weight=np.zeros(N))
    assert re.search(r"(.*weight.*zero.*)|(.*zero.*weight.*)", str(exc.value))
