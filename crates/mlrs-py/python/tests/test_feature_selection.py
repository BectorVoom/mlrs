"""``mlrs.feature_selection`` parity tests (FSEL-01).

Compared against a LIVE scikit-learn rather than a stored fixture, the same choice
``test_model_selection.py`` makes: the Rust oracle tests
(``feature_selection_score_test.rs`` / ``feature_selection_selector_test.rs``)
already pin the numerics against committed sklearn blobs, so what is left to
verify here is the BINDING and the container plumbing — that the shim reaches the
right Rust entry point with the right arguments, and hands back the right kind of
container. A live comparison catches a wiring error that a fixture cannot
distinguish from a numerics change.

## Container coverage
Every selector is exercised over numpy / pandas / polars / pyarrow / python-list
input. pandas and polars are parametrised with ``importorskip`` so the suite runs
on a machine without them, but they are both in this project's dev environment,
so they are normally exercised.

The container assertions are the point of the module: mlrs's ``transform`` returns
the SAME kind of container it was given (D-03), which sklearn does not do — so
these are the tests that would catch the divergence going the wrong way.
"""

import warnings

import numpy as np
import pytest
import sklearn.feature_selection as skfs

from mlrs import feature_selection as mfs

N_SAMPLES = 60
N_FEATURES = 6
N_CLASSES = 3


@pytest.fixture(scope="module")
def design():
    """``(X, y_class, y_reg)`` with the degenerate columns that decide masks.

    Column 2 is CONSTANT (``f_classif`` scores it ``NaN``) and column 4 duplicates
    column 0 (an exact score TIE). Both are what separate a correct tie-break and
    NaN-ranking from a plausible one — the same reasoning the Rust fixture's
    design docstring records. ``X`` is shifted non-negative so ``chi2`` accepts it.
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y_class = np.repeat(np.arange(N_CLASSES), N_SAMPLES // N_CLASSES).astype(np.int64)
    y_reg = 1.5 * x[:, 0] - x[:, 1] + 0.2 * rng.standard_normal(N_SAMPLES)
    x[:, 2] = 0.75
    x[:, 4] = x[:, 0]
    x = x - x.min(axis=0) + 0.5
    return x, y_class, y_reg


def _pandas(x):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])


def _polars(x):
    pl = pytest.importorskip("polars")
    return pl.DataFrame({f"f{i}": x[:, i] for i in range(x.shape[1])})


def _arrow(x):
    pa = pytest.importorskip("pyarrow")
    return pa.table({f"f{i}": x[:, i] for i in range(x.shape[1])})


def _allclose(got, want, what, atol=1e-8):
    """NaN-aware allclose — a ``NaN`` score is a positive claim, not a gap."""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape, f"{what}: shape {got.shape} != {want.shape}"
    both_nan = np.isnan(got) & np.isnan(want)
    assert np.array_equal(np.isnan(got), np.isnan(want)), f"{what}: NaN pattern differs"
    finite = ~both_nan & np.isfinite(want)
    assert np.allclose(
        got[finite], want[finite], rtol=1e-6, atol=atol
    ), f"{what}: got={got} want={want}"


# ------------------------------------------------------------------------- #
# Score functions
# ------------------------------------------------------------------------- #


def test_f_classif_matches_sklearn(design):
    x, y, _ = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want_f, want_p = skfs.f_classif(x, y)
        got_f, got_p = mfs.f_classif(x, y)
    _allclose(got_f, want_f, "f_classif scores")
    _allclose(got_p, want_p, "f_classif pvalues")
    # The constant column must be NaN on BOTH sides — the branch, not just the
    # number.
    assert np.isnan(got_f[2]) and np.isnan(want_f[2])


def test_chi2_matches_sklearn(design):
    x, y, _ = design
    want_c, want_p = skfs.chi2(x, y)
    got_c, got_p = mfs.chi2(x, y)
    _allclose(got_c, want_c, "chi2 scores")
    _allclose(got_p, want_p, "chi2 pvalues")


def test_chi2_rejects_negative_input(design):
    x, y, _ = design
    bad = x.copy()
    bad[0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        mfs.chi2(bad, y)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("force_finite", [True, False])
def test_regression_scores_match_sklearn(design, center, force_finite):
    x, _, y = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want_r = skfs.r_regression(x, y, center=center, force_finite=force_finite)
        got_r = mfs.r_regression(x, y, center=center, force_finite=force_finite)
        want_f, want_p = skfs.f_regression(
            x, y, center=center, force_finite=force_finite
        )
        got_f, got_p = mfs.f_regression(x, y, center=center, force_finite=force_finite)
    # Column 2 is constant, so its `r` is a residue/0 division whose value (and
    # sign) come from the summation order — sklearn's from BLAS. Compared for the
    # same REGIME only, exactly as the Rust oracle test does; see its
    # `NOISE_DRIVEN_COLUMNS` docs.
    keep = [c for c in range(N_FEATURES) if c != 2]
    _allclose(got_r[keep], want_r[keep], "r_regression")
    _allclose(got_f[keep], want_f[keep], "f_regression scores")
    _allclose(got_p[keep], want_p[keep], "f_regression pvalues")

    # The constant column's `r` is `residue / residue'`, both pure cancellation
    # noise, so its value is an artifact of which summation happened to cancel
    # exactly. Observed across the two implementations and the four flag
    # combinations: `NaN` (both residues zero), `±inf` (denominator zero only),
    # `0.0` (the `NaN` mapped by `force_finite`), and a small finite number
    # (neither exactly zero — mlrs reaches this one, sklearn's einsum-based norm
    # does not). None of the four is more correct than the others and none is
    # reproducible across BLAS builds.
    #
    # What IS true is the exact-arithmetic answer: a constant column has zero
    # correlation with anything. So the assertion is that each side produces
    # either a value NEAR zero or an outright divergence — which still fails if
    # mlrs claimed a real correlation like 0.7, and is the strongest statement the
    # arithmetic supports.
    def _degenerate(v):
        return abs(v) < 0.05 or not np.isfinite(v)

    assert _degenerate(got_r[2]), f"constant column r should be ~0 or non-finite, got {got_r[2]}"
    assert _degenerate(want_r[2]), f"sklearn's own is {want_r[2]}"


def test_f_oneway_matches_sklearn(design):
    x, y, _ = design
    groups = [x[y == k] for k in range(N_CLASSES)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want_f, want_p = skfs.f_oneway(*groups)
        got_f, got_p = mfs.f_oneway(*groups)
    _allclose(got_f, want_f, "f_oneway scores")
    _allclose(got_p, want_p, "f_oneway pvalues")


def test_f_oneway_rejects_a_single_group(design):
    x, _, _ = design
    with pytest.raises(ValueError):
        mfs.f_oneway(x)


@pytest.mark.parametrize("n_neighbors", [2, 3, 5])
def test_mutual_info_classif_matches_sklearn(design, n_neighbors):
    """`random_state` is explicit on both sides: it is what makes the noise
    streams comparable at all, and without it sklearn draws from numpy's
    process-global stream while mlrs seeds 0 (a documented divergence)."""
    x, y, _ = design
    want = skfs.mutual_info_classif(
        x, y, n_neighbors=n_neighbors, random_state=0, discrete_features=False
    )
    got = mfs.mutual_info_classif(
        x, y, n_neighbors=n_neighbors, random_state=0, discrete_features=False
    )
    _allclose(got, want, f"mutual_info_classif k={n_neighbors}", atol=1e-6)


def test_mutual_info_regression_matches_sklearn(design):
    x, _, y = design
    want = skfs.mutual_info_regression(
        x, y, random_state=0, discrete_features=False
    )
    got = mfs.mutual_info_regression(x, y, random_state=0, discrete_features=False)
    # The ordinary contract. This was 2e-3 while `_compute_mi_cd` evaluated its
    # k-th-neighbour distance as `|a - b|` on every label group, where sklearn
    # switches to a brute-force GEMM identity for small ones; see
    # `MI_REGRESSION_BAND` in the Rust suite.
    _allclose(got, want, "mutual_info_regression", atol=1e-6)


def test_mutual_info_discrete_features_forms_agree(design):
    """sklearn accepts a bool, a MASK and an INDEX ARRAY for
    ``discrete_features``; the mask and index forms naming the same columns must
    give the same answer, which is what the shim's resolution has to get right."""
    x, y, _ = design
    mask = np.zeros(N_FEATURES, dtype=bool)
    mask[2] = True
    by_mask = mfs.mutual_info_classif(x, y, discrete_features=mask, random_state=1)
    by_index = mfs.mutual_info_classif(
        x, y, discrete_features=np.array([2]), random_state=1
    )
    _allclose(by_mask, by_index, "discrete_features mask vs index form", atol=0)


def test_mutual_info_rejects_bad_discrete_features_string(design):
    x, y, _ = design
    with pytest.raises(ValueError, match="Invalid string value"):
        mfs.mutual_info_classif(x, y, discrete_features="nope")


# ------------------------------------------------------------------------- #
# Selectors: masks and fitted attributes against sklearn
# ------------------------------------------------------------------------- #

#: ``(mlrs factory, sklearn factory, label)`` for every selector mlrs fits itself.
#: The parameter values sit on branch boundaries rather than in their middles —
#: ``k=0``/``"all"``/``k > n_features``, ``percentile`` at both endpoints, an
#: ``alpha`` strict enough to select nothing.
_UNIVARIATE_CASES = [
    (lambda: mfs.SelectKBest(mfs.f_classif, k=0), lambda: skfs.SelectKBest(skfs.f_classif, k=0), "kbest0"),
    (lambda: mfs.SelectKBest(mfs.f_classif, k=2), lambda: skfs.SelectKBest(skfs.f_classif, k=2), "kbest2"),
    (lambda: mfs.SelectKBest(mfs.f_classif, k="all"), lambda: skfs.SelectKBest(skfs.f_classif, k="all"), "kbestall"),
    (lambda: mfs.SelectKBest(mfs.f_classif, k=99), lambda: skfs.SelectKBest(skfs.f_classif, k=99), "kbest99"),
    (lambda: mfs.SelectKBest(mfs.chi2, k=3), lambda: skfs.SelectKBest(skfs.chi2, k=3), "kbest_chi2"),
    (lambda: mfs.SelectPercentile(mfs.f_classif, percentile=0), lambda: skfs.SelectPercentile(skfs.f_classif, percentile=0), "pct0"),
    (lambda: mfs.SelectPercentile(mfs.f_classif, percentile=33), lambda: skfs.SelectPercentile(skfs.f_classif, percentile=33), "pct33"),
    (lambda: mfs.SelectPercentile(mfs.f_classif, percentile=100), lambda: skfs.SelectPercentile(skfs.f_classif, percentile=100), "pct100"),
    (lambda: mfs.SelectFpr(mfs.f_classif, alpha=0.05), lambda: skfs.SelectFpr(skfs.f_classif, alpha=0.05), "fpr"),
    (lambda: mfs.SelectFpr(mfs.f_classif, alpha=1e-12), lambda: skfs.SelectFpr(skfs.f_classif, alpha=1e-12), "fpr_none"),
    (lambda: mfs.SelectFdr(mfs.f_classif, alpha=0.05), lambda: skfs.SelectFdr(skfs.f_classif, alpha=0.05), "fdr"),
    (lambda: mfs.SelectFdr(mfs.f_classif, alpha=1e-12), lambda: skfs.SelectFdr(skfs.f_classif, alpha=1e-12), "fdr_none"),
    (lambda: mfs.SelectFwe(mfs.f_classif, alpha=0.05), lambda: skfs.SelectFwe(skfs.f_classif, alpha=0.05), "fwe"),
    (lambda: mfs.GenericUnivariateSelect(mfs.f_classif, mode="k_best", param=3), lambda: skfs.GenericUnivariateSelect(skfs.f_classif, mode="k_best", param=3), "gen_kbest"),
    (lambda: mfs.GenericUnivariateSelect(mfs.f_classif, mode="percentile", param=50), lambda: skfs.GenericUnivariateSelect(skfs.f_classif, mode="percentile", param=50), "gen_pct"),
    (lambda: mfs.GenericUnivariateSelect(mfs.f_classif, mode="fdr", param=0.05), lambda: skfs.GenericUnivariateSelect(skfs.f_classif, mode="fdr", param=0.05), "gen_fdr"),
]


@pytest.mark.parametrize(
    "make_mlrs,make_sk,label",
    _UNIVARIATE_CASES,
    ids=[c[2] for c in _UNIVARIATE_CASES],
)
def test_univariate_selector_matches_sklearn(design, make_mlrs, make_sk, label):
    x, y, _ = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = make_mlrs().fit(x, y)
        want = make_sk().fit(x, y)
    # The support mask is an EXACT contract: a tolerance here would let an
    # off-by-one selection pass, which is the only failure that matters.
    assert np.array_equal(got.get_support(), want.get_support()), f"{label}: mask"
    assert np.array_equal(
        got.get_support(indices=True), want.get_support(indices=True)
    ), f"{label}: indices"
    _allclose(got.scores_, want.scores_, f"{label}: scores_")
    if want.pvalues_ is not None:
        _allclose(got.pvalues_, want.pvalues_, f"{label}: pvalues_")
    assert got.n_features_in_ == want.n_features_in_


@pytest.mark.parametrize("threshold", [0.0, 0.2, 1.0])
def test_variance_threshold_matches_sklearn(design, threshold):
    """Including the thresholds that drop EVERYTHING.

    `threshold=1.0` is above every column's variance in this design, and sklearn
    RAISES there rather than returning an empty selector — so the comparison is
    "both raise, or both agree", not "both agree". Written that way rather than by
    tuning the threshold to keep something, because the raise is part of the
    contract and hard-coding which thresholds trigger it would silently stop
    testing it if the design ever changed.
    """
    x, _, _ = design
    try:
        want = skfs.VarianceThreshold(threshold=threshold).fit(x)
    except ValueError:
        with pytest.raises(ValueError, match="No feature in X meets the variance"):
            mfs.VarianceThreshold(threshold=threshold).fit(x)
        return
    got = mfs.VarianceThreshold(threshold=threshold).fit(x)
    _allclose(got.variances_, want.variances_, "variances_")
    assert np.array_equal(got.get_support(), want.get_support())


def test_variance_threshold_accepts_nan_like_sklearn(design):
    """The one selector that tolerates NaN input — and the shim must not reject
    it at the ``check_array`` boundary before Rust sees it."""
    x, _, _ = design
    nan_x = x.copy()
    nan_x[0:4, 0] = np.nan
    got = mfs.VarianceThreshold(threshold=0.0).fit(nan_x)
    want = skfs.VarianceThreshold(threshold=0.0).fit(nan_x)
    _allclose(got.variances_, want.variances_, "nan variances_")
    assert np.array_equal(got.get_support(), want.get_support())


def test_variance_threshold_rejects_all_dropped():
    with pytest.raises(ValueError, match="No feature in X meets the variance"):
        mfs.VarianceThreshold().fit(np.ones((5, 3)))


def test_custom_score_func_uses_the_rust_selection_rule(design):
    """A caller-supplied ``score_func`` still goes through the SAME mask rule.

    The shim evaluates the callable itself (Rust must not call back into Python
    mid-fit) and hands the scores to ``univariate_select_from_scores``. Handing
    sklearn's own ``f_classif`` in as an opaque lambda must therefore reproduce
    the built-in path exactly — which is what proves the two routes share a rule
    rather than each having one.
    """
    x, y, _ = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opaque = mfs.SelectKBest(lambda a, b: skfs.f_classif(a, b), k=3).fit(x, y)
        builtin = mfs.SelectKBest(mfs.f_classif, k=3).fit(x, y)
    assert np.array_equal(opaque.get_support(), builtin.get_support())
    _allclose(opaque.scores_, builtin.scores_, "custom vs builtin scores_")


def test_p_value_mode_rejects_a_scores_only_score_func(design):
    x, _, y = design
    with pytest.raises(ValueError, match="requires p-values"):
        mfs.SelectFdr(mfs.r_regression, alpha=0.05).fit(x, y)


def test_kbest_warns_when_k_exceeds_n_features(design):
    x, y, _ = design
    with pytest.warns(UserWarning, match="greater than n_features"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mfs.SelectKBest(mfs.f_classif, k=99).fit(x, y)


def test_empty_selection_warns_on_transform(design):
    x, y, _ = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = mfs.SelectKBest(mfs.f_classif, k=0).fit(x, y)
    with pytest.warns(UserWarning, match="No features were selected"):
        out = est.transform(x)
    assert np.asarray(out).shape == (N_SAMPLES, 0)


# ------------------------------------------------------------------------- #
# Containers: numpy / pandas / polars / pyarrow / list
# ------------------------------------------------------------------------- #


@pytest.mark.parametrize("wrap", [np.asarray, _pandas, _polars, _arrow, lambda x: x.tolist()])
def test_fit_is_container_agnostic(design, wrap):
    """Every container gives the SAME mask as numpy — the ingress must not depend
    on how the values arrived."""
    x, y, _ = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = mfs.SelectKBest(mfs.f_classif, k=3).fit(x, y)
        other = mfs.SelectKBest(mfs.f_classif, k=3).fit(wrap(x), y)
    assert np.array_equal(base.get_support(), other.get_support())
    _allclose(other.scores_, base.scores_, "container scores_")


def test_transform_mirrors_a_polars_frame(design):
    """polars in, polars out — with only the kept columns, under their own names.

    This is mlrs's ``output_type="input"`` contract and a deliberate divergence
    from sklearn, whose ``transform`` returns numpy here.
    """
    pl = pytest.importorskip("polars")
    x, y, _ = design
    df = _polars(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = mfs.SelectKBest(mfs.f_classif, k=3).fit(df, y)
    out = est.transform(df)
    assert isinstance(out, pl.DataFrame)
    kept = [f"f{i}" for i in np.nonzero(est.get_support())[0]]
    assert out.columns == kept
    assert out.height == N_SAMPLES
    # The VALUES are the original columns, untouched by the float64 scoring view.
    for name in kept:
        _allclose(out[name].to_numpy(), df[name].to_numpy(), f"polars col {name}", atol=0)


def test_transform_mirrors_a_pandas_frame(design):
    pd = pytest.importorskip("pandas")
    x, y, _ = design
    df = _pandas(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = mfs.SelectKBest(mfs.f_classif, k=3).fit(df, y)
    out = est.transform(df)
    assert isinstance(out, pd.DataFrame)
    kept = [f"f{i}" for i in np.nonzero(est.get_support())[0]]
    assert list(out.columns) == kept
    # The pandas INDEX survives, which a numpy round-trip would drop.
    assert out.index.equals(df.index)


def test_transform_mirrors_a_pyarrow_table(design):
    pa = pytest.importorskip("pyarrow")
    x, y, _ = design
    tbl = _arrow(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = mfs.SelectKBest(mfs.f_classif, k=3).fit(tbl, y)
    out = est.transform(tbl)
    assert isinstance(out, pa.Table)
    assert out.column_names == [f"f{i}" for i in np.nonzero(est.get_support())[0]]


def test_numpy_in_numpy_out(design):
    x, y, _ = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = mfs.SelectKBest(mfs.f_classif, k=3).fit(x, y)
    out = est.transform(x)
    assert isinstance(out, np.ndarray)
    # And it equals sklearn's, which is the numpy-path parity claim.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = skfs.SelectKBest(skfs.f_classif, k=3).fit(x, y).transform(x)
    _allclose(out, want, "numpy transform", atol=0)


def test_output_type_numpy_overrides_the_container_mirror(design):
    """``output_type="numpy"`` restores sklearn's exact egress for a caller who
    wants it, even from a polars frame."""
    pytest.importorskip("polars")
    x, y, _ = design
    df = _polars(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = mfs.SelectKBest(mfs.f_classif, k=3, output_type="numpy").fit(df, y)
    out = est.transform(df)
    assert isinstance(out, np.ndarray)
    assert out.shape == (N_SAMPLES, 3)


def test_feature_names_in_and_out(design):
    pytest.importorskip("polars")
    x, y, _ = design
    df = _polars(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = mfs.SelectKBest(mfs.f_classif, k=3).fit(df, y)
    assert list(est.feature_names_in_) == [f"f{i}" for i in range(N_FEATURES)]
    kept = [f"f{i}" for i in np.nonzero(est.get_support())[0]]
    assert list(est.get_feature_names_out()) == kept


def test_feature_names_out_generates_positional_names_for_numpy(design):
    x, y, _ = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = mfs.SelectKBest(mfs.f_classif, k=2).fit(x, y)
    assert not hasattr(est, "feature_names_in_")
    kept = [f"x{i}" for i in np.nonzero(est.get_support())[0]]
    assert list(est.get_feature_names_out()) == kept


def test_inverse_transform_zero_fills_dropped_columns(design):
    x, y, _ = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = mfs.SelectKBest(mfs.f_classif, k=3).fit(x, y)
        want_est = skfs.SelectKBest(skfs.f_classif, k=3).fit(x, y)
    z = est.transform(x)
    got = est.inverse_transform(z)
    want = want_est.inverse_transform(want_est.transform(x))
    _allclose(got, want, "inverse_transform", atol=0)
    dropped = np.nonzero(~est.get_support())[0]
    assert np.all(np.asarray(got)[:, dropped] == 0.0)


def test_fit_transform_round_trips(design):
    """``TransformerMixin.fit_transform`` comes for free from the mixin, and must
    equal ``fit`` then ``transform``."""
    x, y, _ = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = mfs.SelectKBest(mfs.f_classif, k=2).fit_transform(x, y)
        b = mfs.SelectKBest(mfs.f_classif, k=2).fit(x, y).transform(x)
    _allclose(a, b, "fit_transform", atol=0)


# ------------------------------------------------------------------------- #
# The four meta-selectors
# ------------------------------------------------------------------------- #


def _ridge():
    from sklearn.linear_model import Ridge

    return Ridge(alpha=1.0, fit_intercept=False)


@pytest.mark.parametrize(
    "make_mlrs,make_sk,label",
    [
        (
            lambda: mfs.SelectFromModel(_ridge(), threshold="mean"),
            lambda: skfs.SelectFromModel(_ridge(), threshold="mean"),
            "sfm_mean",
        ),
        (
            lambda: mfs.SelectFromModel(_ridge(), max_features=3, threshold=-np.inf),
            lambda: skfs.SelectFromModel(_ridge(), max_features=3, threshold=-np.inf),
            "sfm_maxf",
        ),
        (
            lambda: mfs.RFE(_ridge(), n_features_to_select=3),
            lambda: skfs.RFE(_ridge(), n_features_to_select=3),
            "rfe3",
        ),
        (
            lambda: mfs.RFE(_ridge(), n_features_to_select=2, step=2),
            lambda: skfs.RFE(_ridge(), n_features_to_select=2, step=2),
            "rfe_step2",
        ),
        (
            lambda: mfs.RFECV(_ridge(), cv=3),
            lambda: skfs.RFECV(_ridge(), cv=3),
            "rfecv",
        ),
        (
            lambda: mfs.SequentialFeatureSelector(_ridge(), n_features_to_select=3, cv=3),
            lambda: skfs.SequentialFeatureSelector(_ridge(), n_features_to_select=3, cv=3),
            "sfs_fwd",
        ),
        (
            lambda: mfs.SequentialFeatureSelector(
                _ridge(), n_features_to_select=3, cv=3, direction="backward"
            ),
            lambda: skfs.SequentialFeatureSelector(
                _ridge(), n_features_to_select=3, cv=3, direction="backward"
            ),
            "sfs_bwd",
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_meta_selector_matches_sklearn(design, make_mlrs, make_sk, label):
    """The meta-selectors reuse sklearn's fit, so the mask must be IDENTICAL.

    That is a real assertion rather than a tautology: the mlrs subclasses add an
    ``output_type`` parameter and override three methods, and getting the MRO or
    the ``__init__`` forwarding wrong would silently change the fitted mask or
    break ``clone``.
    """
    x, _, y = design
    got = make_mlrs().fit(x, y)
    want = make_sk().fit(x, y)
    assert np.array_equal(got.get_support(), want.get_support()), f"{label}: mask"
    if hasattr(want, "ranking_"):
        assert np.array_equal(got.ranking_, want.ranking_), f"{label}: ranking_"


def test_meta_selector_transform_mirrors_polars(design):
    pl = pytest.importorskip("polars")
    x, _, y = design
    df = _polars(x)
    est = mfs.RFE(_ridge(), n_features_to_select=3).fit(df, y)
    out = est.transform(df)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == [f"f{i}" for i in np.nonzero(est.get_support())[0]]


def test_meta_selectors_round_trip_get_params(design):
    """``output_type`` must be a real, cloneable constructor parameter.

    sklearn's ``clone`` reconstructs from ``get_params()``, so an ``__init__`` that
    forwards to the sklearn base without storing its own parameter verbatim breaks
    every pipeline and grid search the estimator appears in.
    """
    from sklearn.base import clone

    for est in (
        mfs.SelectFromModel(_ridge(), output_type="numpy"),
        mfs.RFE(_ridge(), n_features_to_select=2, output_type="numpy"),
        mfs.RFECV(_ridge(), cv=3, output_type="numpy"),
        mfs.SequentialFeatureSelector(_ridge(), cv=3, output_type="numpy"),
    ):
        params = est.get_params()
        assert params["output_type"] == "numpy"
        assert clone(est).get_params()["output_type"] == "numpy"


def test_selectors_round_trip_get_params():
    """Same ``clone`` contract for the selectors mlrs fits itself."""
    from sklearn.base import clone

    for est in (
        mfs.VarianceThreshold(threshold=0.3),
        mfs.SelectKBest(mfs.chi2, k=4),
        mfs.SelectPercentile(mfs.f_classif, percentile=20),
        mfs.SelectFpr(mfs.f_classif, alpha=0.01),
        mfs.SelectFdr(mfs.f_classif, alpha=0.01),
        mfs.SelectFwe(mfs.f_classif, alpha=0.01),
        mfs.GenericUnivariateSelect(mfs.chi2, mode="fdr", param=0.02),
    ):
        assert clone(est).get_params() == est.get_params()


def test_pipeline_composition(design):
    """A selector must work inside a sklearn ``Pipeline`` — the integration that
    matters most in practice, and the one that exercises ``fit_transform``,
    ``get_params`` and the tags together."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    x, _, y = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = make_pipeline(
            mfs.SelectKBest(mfs.f_regression, k=3), Ridge(alpha=1.0)
        ).fit(x, y)
        preds = pipe.predict(x)
    assert preds.shape == (N_SAMPLES,)


def test_unfitted_selector_raises_not_fitted(design):
    from sklearn.exceptions import NotFittedError

    x, _, _ = design
    with pytest.raises(NotFittedError):
        mfs.SelectKBest(mfs.f_classif, k=2).transform(x)
    with pytest.raises(NotFittedError):
        mfs.SelectKBest(mfs.f_classif, k=2).get_support()


def test_transform_rejects_a_wrong_feature_count(design):
    x, y, _ = design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = mfs.SelectKBest(mfs.f_classif, k=2).fit(x, y)
    with pytest.raises(ValueError, match="features, but"):
        est.transform(x[:, :3])
