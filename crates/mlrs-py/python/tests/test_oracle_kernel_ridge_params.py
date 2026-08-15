"""``KernelRidge`` full-parameter surface through the Python shim, vs LIVE sklearn.

Every ``sklearn.kernel_ridge.KernelRidge`` parameter is exercised here against a
live ``sklearn.kernel_ridge.KernelRidge`` rather than a committed fixture, for
the reason the KMeans / KNN / RidgeCV param suites give: what has to be checked
is exactly the thing a fixture would freeze away — that mlrs's resolution rules
still agree with sklearn's as sklearn evolves them. Two of those rules are
sklearn's rather than ours and are not written down anywhere a fixture could
capture:

* ``gamma=None`` resolves to ``1/n_features`` for FOUR of the five γ-taking
  kernels and RAISES for the fifth (``chi2``), because
  ``KernelRidge._get_kernel`` forwards ``self.gamma`` unconditionally into
  ``chi2_kernel``'s ``K *= gamma``.
* an indefinite Gram — which ``additive_chi2`` produces at every ``alpha`` —
  makes the Cholesky inapplicable, and sklearn silently re-solves in the
  least-squares sense with a warning rather than failing.

The ONE string-valued parameter is ``kernel``, and it gets the attention the
campaign requires: all nine names sklearn's ``StrOptions`` admits
(``additive_chi2``, ``chi2``, ``cosine``, ``laplacian``, ``linear``, ``poly``,
``polynomial``, ``precomputed``, ``rbf``, ``sigmoid``), each compared against
sklearn under the SAME name, at both the default and an explicit ``gamma``,
single- and multi-target, plus the rejection of an unknown string and the
callable that is the parameter's non-string half.

Designs are NON-NEGATIVE throughout: the chi² pair requires it (sklearn's
``check_non_negative``), and using one design for all nine kernels is what makes
a cross-kernel disagreement a kernel bug rather than a fixture difference.

f64 designs are skipped-with-reason on an f64-incapable backend via
``conftest.default_float_dtype`` / ``live_atol``.
"""

import warnings

import numpy as np
import pytest
from sklearn.kernel_ridge import KernelRidge as SkKernelRidge
from sklearn.metrics.pairwise import pairwise_kernels

import mlrs
from conftest import default_float_dtype, live_atol, requires_f64

# Every kernel NAME sklearn accepts, with the `gamma` each needs to be
# well-defined. `chi2` is the one that has no default (see the module docstring);
# `None` everywhere else means "exercise sklearn's own resolution rule", which is
# the half of this parameter a hardcoded gamma would stop testing.
KERNEL_NAMES = [
    ("linear", None),
    ("rbf", None),
    ("poly", None),
    ("polynomial", None),
    ("sigmoid", None),
    ("laplacian", None),
    ("cosine", None),
    ("chi2", 0.7),
    ("additive_chi2", None),
]

# The subset that reads `gamma`, for the explicit-gamma sweep.
GAMMA_KERNELS = ["rbf", "poly", "polynomial", "sigmoid", "laplacian", "chi2"]

def design(n=48, d=5, n_targets=0, seed=0, dtype=None):
    """A non-negative random design with a smooth target.

    Non-negative because the chi² kernels require it and every kernel here is
    compared on the SAME data — a per-kernel design would let a disagreement
    hide behind "different inputs". Scaled into ``[0.1, 1.1]`` rather than
    ``[0, 1]`` so no feature pair sums to exactly zero: that is the one input
    where sklearn's chi² term guard (``if nom != 0``) is observable, and it gets
    its own test below rather than being mixed into every case.
    """
    dtype = dtype or default_float_dtype()
    rs = np.random.RandomState(seed)
    x = rs.rand(n, d) + 0.1
    k = n_targets or 1
    coef = rs.rand(d, k) + 0.5
    y = x @ coef + 0.05 * rs.randn(n, k)
    if n_targets == 0:
        y = y[:, 0]
    return (
        np.ascontiguousarray(x, dtype=dtype),
        np.ascontiguousarray(y, dtype=dtype),
    )


def fit_both(kernel, X, y, X_test, sample_weight=None, **params):
    """Fit mlrs and sklearn identically and return ``(mlrs_pred, sk_pred, pair)``.

    The warning both implementations raise on an indefinite Gram is captured,
    not suppressed — ``test_indefinite_gram_warns`` asserts on it, and letting it
    through here would turn every ``additive_chi2`` case into a warning-filter
    question.
    """
    est = mlrs.KernelRidge(kernel=kernel, **params)
    sk = SkKernelRidge(kernel=kernel, **params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X, y, sample_weight=sample_weight)
        sk.fit(X, y, sample_weight=sample_weight)
    return est.predict(X_test), sk.predict(X_test), (est, sk)


def assert_matches(got, expected, what, atol=None):
    atol = live_atol() if atol is None else atol
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert got.shape == expected.shape, (
        f"{what}: shape {got.shape} != sklearn's {expected.shape}"
    )
    err = np.max(np.abs(got - expected)) if got.size else 0.0
    assert err <= atol, f"{what}: max abs error {err:.3e} > {atol:.1e}"


# --------------------------------------------------------------------------- #
# kernel — the string-valued parameter                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kernel,gamma", KERNEL_NAMES)
def test_every_kernel_name_matches_sklearn(kernel, gamma):
    """Each of the nine names, on one shared design, vs sklearn under the same
    name. `gamma=None` where sklearn has a default exercises the
    `1/n_features` resolution rather than pinning a value past it."""
    X, y = design()
    X_test, _ = design(n=11, seed=3)
    got, expected, _ = fit_both(kernel, X, y, X_test, gamma=gamma)
    assert_matches(got, expected, f"predict(kernel={kernel!r})")


@pytest.mark.parametrize("kernel,gamma", KERNEL_NAMES)
def test_every_kernel_name_matches_on_dual_coef(kernel, gamma):
    """`dual_coef_`, not just `predict`.

    `predict` contracts the duals against the cross-kernel, and a compensating
    pair of errors in both could survive that contraction. The duals are the
    fitted state itself.
    """
    X, y = design()
    est = mlrs.KernelRidge(kernel=kernel, gamma=gamma)
    sk = SkKernelRidge(kernel=kernel, gamma=gamma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X, y)
        sk.fit(X, y)
    assert_matches(
        est.dual_coef_, sk.dual_coef_, f"dual_coef_(kernel={kernel!r})"
    )


@pytest.mark.parametrize("kernel,gamma", KERNEL_NAMES)
def test_every_kernel_name_matches_multi_target(kernel, gamma):
    """The same nine names with a 2-column `y` — the multi-RHS solve path."""
    X, y = design(n_targets=2)
    X_test, _ = design(n=7, seed=5)
    got, expected, _ = fit_both(kernel, X, y, X_test, gamma=gamma)
    assert got.shape == (X_test.shape[0], 2)
    assert_matches(got, expected, f"multi-target predict(kernel={kernel!r})")


@pytest.mark.parametrize("kernel", GAMMA_KERNELS)
@pytest.mark.parametrize("gamma", [0.05, 0.5, 2.0])
def test_explicit_gamma_matches_sklearn(kernel, gamma):
    """An explicit `gamma` is used verbatim by both, across three magnitudes."""
    X, y = design()
    X_test, _ = design(n=9, seed=7)
    got, expected, _ = fit_both(kernel, X, y, X_test, gamma=gamma)
    assert_matches(got, expected, f"predict(kernel={kernel!r}, gamma={gamma})")


def test_poly_and_polynomial_are_the_same_kernel():
    """sklearn admits both spellings for one kernel; so must mlrs, and they must
    agree bit-for-bit with each other rather than merely both being close to
    sklearn."""
    X, y = design()
    X_test, _ = design(n=6, seed=9)
    a = mlrs.KernelRidge(kernel="poly").fit(X, y).predict(X_test)
    b = mlrs.KernelRidge(kernel="polynomial").fit(X, y).predict(X_test)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_unknown_kernel_name_is_rejected():
    """An unrecognised string is a `ValueError` naming the parameter, at `fit`
    (mlrs validates at fit; sklearn at fit too, via `_validate_params`)."""
    X, y = design(n=12)
    with pytest.raises(ValueError, match="kernel"):
        mlrs.KernelRidge(kernel="gaussian").fit(X, y)


def test_non_string_non_callable_kernel_is_rejected():
    X, y = design(n=12)
    with pytest.raises(ValueError, match="kernel"):
        mlrs.KernelRidge(kernel=3).fit(X, y)


# --------------------------------------------------------------------------- #
# kernel — precomputed and callable, the two non-evaluating halves             #
# --------------------------------------------------------------------------- #


def test_precomputed_matches_sklearn():
    """`kernel='precomputed'` takes K itself at fit and the cross-kernel at
    predict. Fed an rbf Gram so the answer is checkable against the rbf run."""
    X, y = design()
    X_test, _ = design(n=10, seed=11)
    k = pairwise_kernels(X, metric="rbf", gamma=0.4)
    k_test = pairwise_kernels(X_test, X, metric="rbf", gamma=0.4)
    got, expected, _ = fit_both("precomputed", k, y, k_test)
    assert_matches(got, expected, "predict(kernel='precomputed')")


def test_precomputed_agrees_with_the_kernel_it_precomputes():
    """The precomputed route and the named route must agree, or one of them is
    computing a different kernel than it claims."""
    X, y = design()
    X_test, _ = design(n=10, seed=11)
    named = mlrs.KernelRidge(kernel="rbf", gamma=0.4).fit(X, y).predict(X_test)
    k = pairwise_kernels(X, metric="rbf", gamma=0.4)
    k_test = pairwise_kernels(X_test, X, metric="rbf", gamma=0.4)
    pre = mlrs.KernelRidge(kernel="precomputed").fit(k, y).predict(k_test)
    assert_matches(pre, named, "precomputed vs named rbf")


def test_precomputed_rejects_a_non_square_fit_matrix():
    X, y = design()
    with pytest.raises(ValueError, match="precomputed|square"):
        mlrs.KernelRidge(kernel="precomputed").fit(X, y)


def test_precomputed_sets_the_pairwise_tag():
    """Without the tag, sklearn's splitters subset a precomputed Gram by rows
    only and hand the estimator a non-square training matrix."""
    assert mlrs.KernelRidge(kernel="precomputed").__sklearn_tags__().input_tags.pairwise
    assert not mlrs.KernelRidge(kernel="rbf").__sklearn_tags__().input_tags.pairwise


def test_callable_kernel_matches_sklearn():
    """A callable is the non-string half of the `kernel` parameter. sklearn
    applies it pairwise to ROWS; the shim routes the result through the
    precomputed path, so this also checks that route end to end."""

    def k(a, b):
        return float(np.dot(a, b)) ** 2 + 1.0

    X, y = design(n=24)
    X_test, _ = design(n=6, d=5, seed=13)
    got, expected, _ = fit_both(k, X, y, X_test)
    assert_matches(got, expected, "predict(kernel=callable)")


def test_kernel_params_reach_a_callable_kernel():
    def k(a, b, scale=1.0):
        return float(np.dot(a, b)) * scale

    X, y = design(n=24)
    X_test, _ = design(n=6, d=5, seed=15)
    got, expected, _ = fit_both(
        k, X, y, X_test, kernel_params={"scale": 4.0}
    )
    assert_matches(got, expected, "predict(kernel=callable, kernel_params=…)")


def test_kernel_params_is_ignored_for_a_string_kernel():
    """sklearn reads `kernel_params` only on the callable branch — a string
    kernel takes its coefficients from gamma/degree/coef0 and ignores it. Being
    stricter here would reject calls sklearn accepts."""
    X, y = design()
    X_test, _ = design(n=6, seed=17)
    with_params = (
        mlrs.KernelRidge(kernel="rbf", kernel_params={"gamma": 99.0})
        .fit(X, y)
        .predict(X_test)
    )
    without = mlrs.KernelRidge(kernel="rbf").fit(X, y).predict(X_test)
    assert np.array_equal(np.asarray(with_params), np.asarray(without))


# --------------------------------------------------------------------------- #
# gamma — including the kernel that has no default                            #
# --------------------------------------------------------------------------- #


def test_gamma_none_resolves_to_one_over_n_features():
    """The resolution rule, checked by CONSTRUCTION rather than by trusting the
    default: an explicit `1/n_features` must give exactly what `None` gives."""
    X, y = design(d=5)
    X_test, _ = design(n=6, d=5, seed=19)
    a = mlrs.KernelRidge(kernel="rbf").fit(X, y).predict(X_test)
    b = mlrs.KernelRidge(kernel="rbf", gamma=1.0 / 5).fit(X, y).predict(X_test)
    assert_matches(a, b, "gamma=None vs gamma=1/n_features", atol=0.0)


def test_chi2_has_no_gamma_default():
    """`chi2` is the exception to the rule above, in sklearn and here.

    sklearn fails with a numpy dtype error out of `K *= gamma`; mlrs raises a
    `ValueError` that names the parameter. What is asserted is that BOTH refuse
    — resolving `1/n_features` here would return a number where sklearn returns
    an exception.
    """
    X, y = design()
    with pytest.raises(ValueError, match="gamma"):
        mlrs.KernelRidge(kernel="chi2").fit(X, y)
    with pytest.raises(Exception):
        SkKernelRidge(kernel="chi2").fit(X, y)


def test_negative_gamma_is_rejected():
    """sklearn's interval is `[0, inf)`."""
    X, y = design()
    with pytest.raises(ValueError, match="gamma"):
        mlrs.KernelRidge(kernel="rbf", gamma=-1.0).fit(X, y)


@pytest.mark.parametrize("kernel", ["linear", "cosine", "additive_chi2"])
def test_gamma_is_inert_for_the_kernels_that_do_not_read_it(kernel):
    """A `gamma` passed to a kernel that takes none must be ignored, not
    rejected — sklearn's `filter_params=True` drops it silently."""
    X, y = design()
    X_test, _ = design(n=6, seed=21)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = mlrs.KernelRidge(kernel=kernel).fit(X, y).predict(X_test)
        b = mlrs.KernelRidge(kernel=kernel, gamma=17.0).fit(X, y).predict(X_test)
    assert np.array_equal(np.asarray(a), np.asarray(b))


# --------------------------------------------------------------------------- #
# alpha — scalar and per-target                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [1e-3, 1.0, 100.0])
def test_scalar_alpha_matches_sklearn(alpha):
    X, y = design()
    X_test, _ = design(n=8, seed=23)
    got, expected, _ = fit_both("rbf", X, y, X_test, alpha=alpha)
    assert_matches(got, expected, f"predict(alpha={alpha})")


@requires_f64
def test_zero_alpha_matches_sklearn():
    """`alpha=0` is a legal sklearn value (the interval is `[0, inf)`) and gets
    its own f64-gated cell rather than a rung on the sweep above.

    An rbf Gram over 48 nearby points has a spectrum that decays past f32's
    resolution, so at `alpha=0` there is nothing left to regularise it and the
    system is numerically singular THERE and not in f64. sklearn is always f64,
    so on an f32 backend the two would be solving genuinely different problems
    — and the disagreement would be about the precision, not about `alpha`.
    """
    X, y = design()
    X_test, _ = design(n=8, seed=23)
    got, expected, _ = fit_both("rbf", X, y, X_test, alpha=0.0)
    assert_matches(got, expected, "predict(alpha=0.0)")


def test_per_target_alpha_matches_sklearn():
    """An array-like `alpha` penalises each target column separately — the path
    that gives up the shared Cholesky factorisation."""
    X, y = design(n_targets=3)
    X_test, _ = design(n=8, seed=25)
    got, expected, _ = fit_both(
        "rbf", X, y, X_test, alpha=[0.01, 1.0, 50.0]
    )
    assert_matches(got, expected, "predict(alpha=[…])")


def test_per_target_alpha_actually_differs_per_target():
    """Guards the test above: if the per-target alphas were silently collapsed
    to the first, the comparison against sklearn would still pass only if
    sklearn collapsed them too. This pins that the columns genuinely differ from
    the all-same-alpha fit."""
    X, y = design(n_targets=2)
    X_test, _ = design(n=8, seed=27)
    varied = (
        mlrs.KernelRidge(kernel="rbf", alpha=[0.01, 50.0])
        .fit(X, y)
        .predict(X_test)
    )
    flat = (
        mlrs.KernelRidge(kernel="rbf", alpha=0.01).fit(X, y).predict(X_test)
    )
    assert np.allclose(varied[:, 0], flat[:, 0], atol=live_atol())
    assert not np.allclose(varied[:, 1], flat[:, 1], atol=1e-3)


def test_uniform_alpha_vector_equals_the_scalar():
    """A per-target vector whose entries are all equal must take the shared
    factorisation and land on the scalar answer EXACTLY — the split is an
    optimisation, not a different computation."""
    X, y = design(n_targets=2)
    X_test, _ = design(n=8, seed=29)
    vec = (
        mlrs.KernelRidge(kernel="rbf", alpha=[2.0, 2.0])
        .fit(X, y)
        .predict(X_test)
    )
    scalar = mlrs.KernelRidge(kernel="rbf", alpha=2.0).fit(X, y).predict(X_test)
    assert np.array_equal(np.asarray(vec), np.asarray(scalar))


def test_alpha_length_must_match_the_target_count():
    X, y = design(n_targets=2)
    with pytest.raises(ValueError, match="alpha"):
        mlrs.KernelRidge(alpha=[1.0, 2.0, 3.0]).fit(X, y)


def test_negative_alpha_is_rejected():
    X, y = design()
    with pytest.raises(ValueError, match="alpha"):
        mlrs.KernelRidge(alpha=-1.0).fit(X, y)


# --------------------------------------------------------------------------- #
# degree / coef0                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("degree", [0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
def test_degree_matches_sklearn(degree):
    """sklearn's interval is `[0, inf)` over the REALS — `degree=0.5` is a legal
    configuration there, and the poly map's `powf` is why."""
    X, y = design()
    X_test, _ = design(n=8, seed=31)
    got, expected, _ = fit_both("poly", X, y, X_test, degree=degree)
    assert_matches(got, expected, f"predict(degree={degree})")


def test_negative_degree_is_rejected():
    X, y = design()
    with pytest.raises(ValueError, match="degree"):
        mlrs.KernelRidge(kernel="poly", degree=-1.0).fit(X, y)


@pytest.mark.parametrize("coef0", [-2.0, 0.0, 1.0, 5.0])
@pytest.mark.parametrize("kernel", ["poly", "sigmoid"])
def test_coef0_matches_sklearn(kernel, coef0):
    X, y = design()
    X_test, _ = design(n=8, seed=33)
    got, expected, _ = fit_both(kernel, X, y, X_test, coef0=coef0, gamma=0.3)
    assert_matches(got, expected, f"predict(kernel={kernel!r}, coef0={coef0})")


def test_degree_is_inert_for_the_kernels_that_do_not_read_it():
    X, y = design()
    X_test, _ = design(n=6, seed=35)
    a = mlrs.KernelRidge(kernel="rbf").fit(X, y).predict(X_test)
    b = mlrs.KernelRidge(kernel="rbf", degree=9.0).fit(X, y).predict(X_test)
    assert np.array_equal(np.asarray(a), np.asarray(b))


# --------------------------------------------------------------------------- #
# sample_weight                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kernel,gamma", KERNEL_NAMES)
def test_sample_weight_matches_sklearn(kernel, gamma):
    """The `S·K·S` similarity transform, across every kernel — the weighting is
    kernel-independent, so a per-kernel failure here would mean the transform
    landed in the wrong place in one of the paths."""
    X, y = design()
    X_test, _ = design(n=8, seed=37)
    rs = np.random.RandomState(41)
    sw = rs.rand(X.shape[0]) * 3.0 + 0.05
    got, expected, _ = fit_both(
        kernel, X, y, X_test, sample_weight=sw, gamma=gamma
    )
    assert_matches(got, expected, f"weighted predict(kernel={kernel!r})")


def test_sample_weight_with_per_target_alpha_matches_sklearn():
    """Weighting and per-target alphas together — the two paths that both
    rewrite the solve, exercised at once."""
    X, y = design(n_targets=2)
    X_test, _ = design(n=8, seed=39)
    rs = np.random.RandomState(43)
    sw = rs.rand(X.shape[0]) + 0.1
    got, expected, _ = fit_both(
        "rbf", X, y, X_test, sample_weight=sw, alpha=[0.05, 5.0]
    )
    assert_matches(got, expected, "weighted per-target predict")


def test_a_uniform_sample_weight_rescales_alpha():
    """A constant weight is NOT a no-op, and the exact way it is not is worth
    pinning.

    `α` goes on the diagonal AFTER the `S·K·S` scaling, so with `S = sI` the
    system is `(s²K + αI)c̃ = s·y`, which unwinds to `(K + (α/s²)I)c = y`. A
    constant weight `w` therefore divides the EFFECTIVE penalty by `w`. The
    obvious guess — that a constant weight cancels — is what this test exists to
    rule out, and it is what sklearn does too.
    """
    X, y = design()
    X_test, _ = design(n=8, seed=45)
    w = 2.5
    sw = np.full(X.shape[0], w)
    weighted = (
        mlrs.KernelRidge(kernel="rbf", alpha=1.0)
        .fit(X, y, sample_weight=sw)
        .predict(X_test)
    )
    rescaled = (
        mlrs.KernelRidge(kernel="rbf", alpha=1.0 / w).fit(X, y).predict(X_test)
    )
    assert_matches(weighted, rescaled, "uniform weight w vs alpha/w")

    unweighted = (
        mlrs.KernelRidge(kernel="rbf", alpha=1.0).fit(X, y).predict(X_test)
    )
    assert not np.allclose(
        np.asarray(weighted, dtype=np.float64),
        np.asarray(unweighted, dtype=np.float64),
        atol=1e-3,
    ), "a constant weight must move the fit — it rescales the penalty"


def test_zero_sample_weight_drops_the_sample():
    """A zero weight must remove the row's influence, which is checked against
    sklearn rather than against a refit on the subset — the dual coefficients
    keep a (zero) entry for the dropped row, so the two are not the same object
    to compare."""
    X, y = design()
    X_test, _ = design(n=8, seed=47)
    sw = np.ones(X.shape[0])
    sw[:5] = 0.0
    got, expected, _ = fit_both("rbf", X, y, X_test, sample_weight=sw)
    assert_matches(got, expected, "predict with zeroed weights")


def test_all_zero_sample_weight_is_rejected():
    X, y = design()
    with pytest.raises(ValueError, match="(?i)weight.*zero|zero.*weight"):
        mlrs.KernelRidge().fit(X, y, sample_weight=np.zeros(X.shape[0]))


def test_negative_sample_weight_is_rejected():
    X, y = design()
    sw = np.ones(X.shape[0])
    sw[3] = -1.0
    with pytest.raises(ValueError, match="(?i)sample_weight"):
        mlrs.KernelRidge().fit(X, y, sample_weight=sw)


# --------------------------------------------------------------------------- #
# the indefinite-Gram fallback                                                 #
# --------------------------------------------------------------------------- #


def test_indefinite_gram_warns_like_sklearn():
    """`additive_chi2` gives an indefinite `(K + αI)` at every alpha. sklearn
    warns and re-solves in the least-squares sense; so must mlrs, or the kernel
    is not usable at all."""
    X, y = design()
    with pytest.warns(UserWarning, match="(?i)singular matrix"):
        mlrs.KernelRidge(kernel="additive_chi2").fit(X, y)
    with pytest.warns(UserWarning, match="(?i)singular matrix"):
        SkKernelRidge(kernel="additive_chi2").fit(X, y)


@requires_f64
def test_a_rank_deficient_gram_takes_the_minimum_norm_solution():
    """The OTHER way the Cholesky can refuse: a genuinely singular `(K + αI)`.

    f64-only, and not merely for tolerance: "rank-deficient" is a statement
    about which singular values are ZERO, and f32 rounding lifts this design's
    null space to ~1e-7 — large enough that the elimination fast path succeeds
    and returns a perfectly valid non-minimum-norm solution, while sklearn (which
    is always f64) still takes its fallback. The two would then disagree about
    which of infinitely many solutions to return, which is a property of the
    precision rather than of either implementation.

    `alpha=0` with a linear kernel and `n > d` gives `K = XXᵀ` of rank `d < n` —
    exactly singular, not merely indefinite. There the least-squares solution is
    not unique and `lstsq` picks the MINIMUM-NORM one; any other solution of the
    same system would satisfy `K·c = y` just as well while being a different
    model. This is the case that needs the pseudo-inverse arm rather than the
    elimination one, so it is the case that proves the arm is reachable.
    """
    dtype = default_float_dtype()
    rs = np.random.RandomState(2)
    X = np.ascontiguousarray(rs.rand(14, 3) + 0.1, dtype=dtype)
    y = np.ascontiguousarray(X @ np.array([1.0, 2.0, 3.0]), dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = mlrs.KernelRidge(kernel="linear", alpha=0.0).fit(X, y).dual_coef_
        expected = (
            SkKernelRidge(kernel="linear", alpha=0.0).fit(X, y).dual_coef_
        )
    assert_matches(got, expected, "rank-deficient dual_coef_")

    # And it really is the minimum-norm one, not just some solution: an
    # arbitrary solution of a rank-3 system in 14 unknowns would generally have
    # a larger norm.
    k = np.asarray(X, dtype=np.float64) @ np.asarray(X, dtype=np.float64).T
    min_norm = np.linalg.pinv(k) @ np.asarray(y, dtype=np.float64)
    assert_matches(got, min_norm, "rank-deficient minimum-norm solution")


def test_a_positive_definite_kernel_does_not_warn():
    """The guard on the test above: if the fallback fired unconditionally, the
    warning assertion would pass while meaning nothing."""
    X, y = design()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mlrs.KernelRidge(kernel="rbf").fit(X, y)


def test_chi2_term_guard_matches_sklearn_on_a_shared_zero_feature():
    """A feature that is zero in BOTH rows makes the chi² term `0/0`, and
    sklearn's `_chi2_kernel_fast` SKIPS it. This is the one input where that
    guard is observable — most designs never hit it, which is why the shared
    design above is deliberately kept away from zero."""
    dtype = default_float_dtype()
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 3.0, 0.5],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 1.5],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=dtype,
    )
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype)
    got, expected, _ = fit_both("chi2", X, y, X, gamma=0.5)
    assert np.all(np.isfinite(np.asarray(got, dtype=np.float64)))
    assert_matches(got, expected, "chi2 with shared zero features")


@pytest.mark.parametrize("kernel", ["chi2", "additive_chi2"])
def test_chi2_kernels_reject_negative_input(kernel):
    """sklearn's `check_non_negative` refuses; so does mlrs, naming the kernel."""
    X, y = design()
    X = X.copy()
    X[2, 1] = -0.5
    with pytest.raises(ValueError, match="(?i)negative"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlrs.KernelRidge(kernel=kernel, gamma=0.5).fit(X, y)


# --------------------------------------------------------------------------- #
# cross-parameter: the surface still round-trips                               #
# --------------------------------------------------------------------------- #

# Model-file persistence for the new kernels is covered in Rust
# (`crates/mlrs-algos/tests/kernel_persist_test.rs`), where the save/load surface
# actually lives — the Python shim does not expose it.


def test_get_params_round_trips_the_full_surface():
    """`clone` has to reproduce the estimator, which means every ctor argument
    comes back out of `get_params` verbatim (the `__init__` purity rule)."""
    from sklearn.base import clone

    est = mlrs.KernelRidge(
        alpha=[1.0, 2.0],
        kernel="laplacian",
        gamma=0.25,
        degree=2.5,
        coef0=-1.0,
        kernel_params={"unused": 1},
    )
    params = est.get_params()
    assert params["alpha"] == [1.0, 2.0]
    assert params["kernel"] == "laplacian"
    assert params["gamma"] == 0.25
    assert params["degree"] == 2.5
    assert params["coef0"] == -1.0
    assert params["kernel_params"] == {"unused": 1}
    assert clone(est).get_params() == params
