#!/usr/bin/env python3
"""Seeded sklearn oracle-fixture generator for `feature_selection` (FSEL-01).

A SEPARATE module from ``gen_oracle.py`` rather than three more thousand-line
functions inside it. ``gen_oracle.py`` is 7 300 lines covering 30-odd
estimators; the feature-selection surface is 18 public names whose fixtures
share one design matrix and one recording helper, so keeping them together in
their own file is what makes the parameter cross readable. ``gen_oracle.py``
calls :func:`main` from its own ``main()`` so a single
``python3 scripts/gen_oracle.py`` still regenerates everything.

Same contract as ``gen_oracle.py``: ``numpy.random.default_rng(seed)`` is the
authoritative RNG, the blobs are committed, and CI never runs this.
Regen in a /tmp venv (PEP 668)::

    python3 -m venv /tmp/oracle-venv
    /tmp/oracle-venv/bin/pip install numpy scipy scikit-learn
    /tmp/oracle-venv/bin/python scripts/gen_feature_selection_oracle.py

## The design matrix, and why every one of its odd columns is there

``_design`` pins the DEGENERATE cases the selectors' masks actually turn on,
because a matrix of clean random normals exercises none of the branches that
break:

* **column 2 is constant** — `f_classif`'s `msw == 0` divide, so `scores_` is
  `NaN` and `_clean_nans`' `f64::MIN` mapping decides its rank. Without this
  column a `SelectKBest` test cannot distinguish a correct implementation from
  one that treats `NaN` as `+inf`.
* **column 3 is all-zero** — `chi2`'s `expected == 0` divide (`NaN` again, by a
  different route) and `VarianceThreshold`'s exact-zero variance.
* **column 5 is a duplicate of column 0** — two features with IDENTICAL scores,
  which is the only way to test that `SelectKBest`'s stable sort breaks a tie
  toward the higher index and that `SelectPercentile`'s tie-refill runs at all.
* **column 6 is an exact affine function of y** — `r == 1`, so
  `f_regression`'s `r²/(1−r²)` is `+inf` and the `force_finite` branch writes
  `f64::MAX`. This is the branch whose sentinel value a caller compares against.
* **column 7 is heavily TIED (rounded to 1 decimal)** — the case
  `mutual_info_*`'s noise exists for, and therefore the only column whose
  mutual information depends on matching numpy's MT19937 stream.

`X` is non-negative overall (shifted) so the SAME design serves `chi2`, which
rejects negatives.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXTURE_DIR = os.path.join(_REPO_ROOT, "tests", "fixtures")

SEED = 42
N_SAMPLES = 90
N_FEATURES = 8
N_CLASSES = 3

# Ridge penalty for the meta-selector inner model. `fit_intercept=False` and a
# non-zero alpha make `coef_ = (XᵀX + αI)⁻¹Xᵀy` a closed form with no centering
# and no iteration, so the Rust test's own `ImportanceEstimator` can reproduce it
# EXACTLY rather than approximately — which is what lets the meta-selectors'
# masks be compared for equality instead of for closeness.
META_ALPHA = 1.0


def _design(seed: int = SEED):
    """The shared `(X, y_class, y_reg)` design — see the module docstring."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y_class = np.repeat(np.arange(N_CLASSES), N_SAMPLES // N_CLASSES).astype(float)
    # A continuous target correlated with columns 0 and 1, so `f_regression`
    # ranks are meaningful rather than noise.
    y_reg = 2.0 * x[:, 0] - 1.5 * x[:, 1] + 0.3 * rng.standard_normal(N_SAMPLES)

    x[:, 2] = 1.75                      # constant  -> f_classif NaN
    x[:, 5] = x[:, 0]                   # duplicate -> exact score tie
    x[:, 6] = 3.0 * y_reg + 0.5         # r == 1    -> f_regression +inf branch
    x[:, 7] = np.round(x[:, 7], 1)      # heavy ties -> mutual_info noise matters
    x = x - x.min(axis=0) + 0.25        # shift non-negative for chi2 ...
    x[:, 3] = 0.0                       # ... then zero column 3 (chi2 0/0)
    return x, y_class, y_reg


def _c(arr, dtype):
    """Cast to ``dtype`` and force C (ROW-MAJOR) contiguity.

    The ``ascontiguousarray`` is load-bearing, not hygiene. ``mlrs_core::oracle``
    reads a committed ``.npz`` through ``npyz`` into a FLAT ``Vec<f64>`` and does
    not consult the ``.npy`` header's ``fortran_order`` flag, so the Rust side
    always interprets the buffer as row-major — while numpy happily stores an
    F-ordered array and records the flag.

    ``est.transform(X)`` is exactly such an array: column fancy-indexing
    (``X[:, mask]``, which is what ``SelectorMixin.transform`` does) returns a
    FORTRAN-ordered copy, so recording it directly wrote a transposed buffer and
    made a CORRECT device gather look wrong at the second element. Every 2-D
    array recorded here goes through this helper for that reason.
    """
    return np.ascontiguousarray(arr, dtype=dtype)


def _dtype_tag(dtype):
    return "f32" if dtype == np.float32 else "f64"


def _mask(m):
    """A boolean support mask as the float64 array the fixture reader accepts.

    `mlrs_core::oracle::load_npz` decodes only 4- and 8-byte FLOAT dtypes — it
    raises "is not a 4- or 8-byte float" for anything else — so a bool or int8
    mask cannot be read back at all. Masks therefore travel as `float64` 0.0/1.0,
    which is exact, and the Rust side compares them with `== 1.0` EXACT equality
    rather than a tolerance: a support mask is a discrete contract, and comparing
    it approximately would let an off-by-one selection pass.
    """
    return np.ascontiguousarray(np.asarray(m, dtype=bool), dtype=np.float64)


def _ints(v):
    """An integer array (rankings, feature counts) as float64, same reason as
    :func:`_mask` — and equally exact, since every value here is far inside
    float64's integer range. Compared with exact equality on the Rust side."""
    return np.ascontiguousarray(v, dtype=np.float64)


def gen_feature_selection_scores(seed: int = SEED, dtype=np.float32) -> str:
    """The five closed-form score functions across their parameter surface.

    One archive holding, for the shared design: `f_classif`, `chi2`, and
    `f_regression` / `r_regression` under all four `(center, force_finite)`
    combinations — the cross that matters because `force_finite` is what decides
    the `f64::MAX` / `0.0` sentinels and `center` changes the degrees of freedom.
    Plus a three-group `f_oneway` recorded directly, since it is public in
    sklearn and is not merely `f_classif`'s helper.
    """
    from sklearn.feature_selection import (
        chi2,
        f_classif,
        f_oneway,
        f_regression,
        r_regression,
    )

    x, y_class, y_reg = _design(seed)
    out = {
        "X": _c(x, dtype),
        "y_class": _c(y_class, dtype),
        "y_reg": _c(y_reg, dtype),
    }

    # `f_classif` on a constant column emits "Features [2] are constant."; the
    # NaN score that warning accompanies is exactly what the fixture pins.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f, p = f_classif(x, y_class)
        out["f_classif_scores"] = np.ascontiguousarray(f, dtype=np.float64)
        out["f_classif_pvalues"] = np.ascontiguousarray(p, dtype=np.float64)

        c, cp = chi2(x, y_class)
        out["chi2_scores"] = np.ascontiguousarray(c, dtype=np.float64)
        out["chi2_pvalues"] = np.ascontiguousarray(cp, dtype=np.float64)

        for center in (True, False):
            for force in (True, False):
                tag = f"c{int(center)}_ff{int(force)}"
                fr, pr = f_regression(x, y_reg, center=center, force_finite=force)
                out[f"f_regression_scores_{tag}"] = np.ascontiguousarray(fr, dtype=np.float64)
                out[f"f_regression_pvalues_{tag}"] = np.ascontiguousarray(pr, dtype=np.float64)
                rr = r_regression(x, y_reg, center=center, force_finite=force)
                out[f"r_regression_{tag}"] = np.ascontiguousarray(rr, dtype=np.float64)

        # `f_oneway` called directly on the three class groups.
        groups = [x[y_class == k] for k in range(N_CLASSES)]
        fo, po = f_oneway(*groups)
        out["f_oneway_scores"] = np.ascontiguousarray(fo, dtype=np.float64)
        out["f_oneway_pvalues"] = np.ascontiguousarray(po, dtype=np.float64)
        out["f_oneway_group_sizes"] = np.ascontiguousarray(
            [g.shape[0] for g in groups], dtype=np.float64
        )

    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    path = os.path.join(
        _FIXTURE_DIR, f"fsel_scores_{_dtype_tag(dtype)}_seed{seed}.npz"
    )
    np.savez(path, **out)
    return path


def gen_feature_selection_mutual_info(seed: int = SEED, dtype=np.float32) -> str:
    """Both `mutual_info_*` estimators, with an EXPLICIT `random_state`.

    `random_state=0` and `=42` are both recorded: the noise stream is what the
    Rust `numpy_rng` replica has to match, and two seeds catch a replica that
    happens to agree on one. `n_neighbors` is swept over `{2, 3, 5}` because a
    small class in `_compute_mi_cd` clamps `k` per group, and `discrete_features`
    is swept over `{False, True, mask}` because each dispatches to a different
    one of the three estimators.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    x, y_class, y_reg = _design(seed)
    out = {"X": _c(x, dtype), "y_class": _c(y_class, dtype), "y_reg": _c(y_reg, dtype)}
    # Column 7 is the heavily-tied one; marking it discrete exercises the
    # discrete-feature/continuous-target and discrete/discrete branches.
    mask = np.zeros(N_FEATURES, dtype=bool)
    mask[7] = True
    out["discrete_mask"] = _mask(mask)

    for rs in (0, 42):
        for k in (2, 3, 5):
            out[f"mi_classif_rs{rs}_k{k}"] = np.ascontiguousarray(
                mutual_info_classif(
                    x, y_class, n_neighbors=k, random_state=rs, discrete_features=False
                ),
                dtype=np.float64,
            )
            out[f"mi_regression_rs{rs}_k{k}"] = np.ascontiguousarray(
                mutual_info_regression(
                    x, y_reg, n_neighbors=k, random_state=rs, discrete_features=False
                ),
                dtype=np.float64,
            )
    # `discrete_features=True`: every column discrete. Against a DISCRETE target
    # this is the contingency-table estimator for every column; against a
    # continuous one it is `_compute_mi_cd` with the roles swapped.
    #
    # Recorded on a BINNED copy of the design, not on `X` itself, because
    # sklearn CRASHES on the latter: `_compute_mi_cd` drops every point whose
    # "label" is unique, a continuous column's values are all unique, so the
    # surviving sample is EMPTY and `KDTree(c)` raises "Found array with 0
    # sample(s)". That is a real sklearn limitation rather than a behaviour worth
    # pinning — `discrete_features=True` means count/categorical data, and
    # binning is what a caller reaching for it actually has. (mlrs returns `0.0`
    # for the degenerate all-unique case instead of raising, which is documented
    # in `mutual_info.rs`; a fixture cannot compare against an exception.)
    x_disc = np.round(x).astype(float)
    out["X_disc"] = _c(x_disc, dtype)
    out["mi_classif_all_discrete"] = np.ascontiguousarray(
        mutual_info_classif(x_disc, y_class, discrete_features=True, random_state=0),
        dtype=np.float64,
    )
    out["mi_regression_all_discrete"] = np.ascontiguousarray(
        mutual_info_regression(x_disc, y_reg, discrete_features=True, random_state=0),
        dtype=np.float64,
    )
    # The MASK form: one discrete column among continuous ones, which is the
    # only case where both branches run inside a single call and the RNG stream
    # is consumed for a SUBSET of the columns (`n_cont = 7`, not 8).
    out["mi_classif_mask"] = np.ascontiguousarray(
        mutual_info_classif(x, y_class, discrete_features=mask, random_state=0),
        dtype=np.float64,
    )
    out["mi_regression_mask"] = np.ascontiguousarray(
        mutual_info_regression(x, y_reg, discrete_features=mask, random_state=0),
        dtype=np.float64,
    )

    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    path = os.path.join(
        _FIXTURE_DIR, f"fsel_mutual_info_{_dtype_tag(dtype)}_seed{seed}.npz"
    )
    np.savez(path, **out)
    return path


def gen_feature_selection_univariate(seed: int = SEED, dtype=np.float32) -> str:
    """The six univariate filters across their full parameter surface.

    Records the support MASK (an exact-equality contract) plus `scores_` and
    `pvalues_` for each configuration. The parameter values are chosen to sit on
    the branch boundaries rather than in their middles:

    * `k` includes `0` (select nothing), `"all"`, and `12 > n_features` (the
      warn-and-keep-everything path);
    * `percentile` includes `0` and `100` (the pre-`_clean_nans` short-circuits)
      and `37.5`, which lands the threshold BETWEEN two scores so the tie-refill
      budget is exercised;
    * the three `alpha`s span "selects most", "selects some", "selects none",
      because an all-false mask is its own branch in `SelectFdr`.
    """
    from sklearn.feature_selection import (
        GenericUnivariateSelect,
        SelectFdr,
        SelectFpr,
        SelectFwe,
        SelectKBest,
        SelectPercentile,
        chi2,
        f_classif,
    )

    x, y_class, _ = _design(seed)
    out = {"X": _c(x, dtype), "y_class": _c(y_class, dtype)}

    def record(name, est):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(x, y_class)
            out[f"{name}_support"] = _mask(est.get_support())
            out[f"{name}_scores"] = np.ascontiguousarray(est.scores_, dtype=np.float64)
            if getattr(est, "pvalues_", None) is not None:
                out[f"{name}_pvalues"] = np.ascontiguousarray(est.pvalues_, dtype=np.float64)

    for k in (0, 1, 3, 12, "all"):
        record(f"kbest_{k}", SelectKBest(f_classif, k=k))
    record("kbest_chi2_3", SelectKBest(chi2, k=3))
    for p in (0, 25, 37.5, 50, 100):
        record(f"percentile_{p}", SelectPercentile(f_classif, percentile=p))
    for a in (1e-8, 0.05, 0.5):
        record(f"fpr_{a}", SelectFpr(f_classif, alpha=a))
        record(f"fdr_{a}", SelectFdr(f_classif, alpha=a))
        record(f"fwe_{a}", SelectFwe(f_classif, alpha=a))
    for mode, param in (
        ("percentile", 30),
        ("k_best", 4),
        ("k_best", "all"),
        ("fpr", 0.05),
        ("fdr", 0.05),
        ("fwe", 0.05),
    ):
        record(
            f"generic_{mode}_{param}",
            GenericUnivariateSelect(f_classif, mode=mode, param=param),
        )

    # `transform` / `inverse_transform` on ONE configuration, so the column
    # gather/scatter kernels have a value oracle rather than only a mask one.
    est = SelectKBest(f_classif, k=3).fit(x, y_class)
    out["kbest_3_transform"] = _c(est.transform(x), dtype)
    out["kbest_3_inverse"] = _c(est.inverse_transform(est.transform(x)), dtype)

    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    path = os.path.join(
        _FIXTURE_DIR, f"fsel_univariate_{_dtype_tag(dtype)}_seed{seed}.npz"
    )
    np.savez(path, **out)
    return path


def gen_feature_selection_variance(seed: int = SEED, dtype=np.float32) -> str:
    """`VarianceThreshold` across its three distinct behaviours.

    `threshold=0` (the peak-to-peak substitution), `threshold>0` (the plain
    variance comparison), and a NaN-containing design (the `nanvar` path, which
    only this selector reaches).
    """
    from sklearn.feature_selection import VarianceThreshold

    x, _, _ = _design(seed)
    out = {"X": _c(x, dtype)}
    for t in (0.0, 0.25, 1.0):
        est = VarianceThreshold(threshold=t).fit(x)
        tag = str(t).replace(".", "p")
        out[f"vt_{tag}_variances"] = np.ascontiguousarray(est.variances_, dtype=np.float64)
        out[f"vt_{tag}_support"] = _mask(est.get_support())

    # The NaN design: scatter NaNs into columns 0 and 4 so `nanvar` differs from
    # `var` and the per-column non-NaN COUNT differs between columns.
    x_nan = x.copy()
    x_nan[0:5, 0] = np.nan
    x_nan[3:9, 4] = np.nan
    out["X_nan"] = _c(x_nan, dtype)
    est = VarianceThreshold(threshold=0.0).fit(x_nan)
    out["vt_nan_variances"] = np.ascontiguousarray(est.variances_, dtype=np.float64)
    out["vt_nan_support"] = _mask(est.get_support())

    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    path = os.path.join(
        _FIXTURE_DIR, f"fsel_variance_{_dtype_tag(dtype)}_seed{seed}.npz"
    )
    np.savez(path, **out)
    return path


def _meta_design(seed: int = SEED):
    """`_design` with the DUPLICATE column made independent, for the meta-selectors.

    `_design`'s column 5 is a copy of column 0, which is exactly what
    `SelectKBest`'s documented tie-break needs: two features with bit-identical
    scores. For the meta-selectors it is poison. `Ridge` splits a coefficient
    equally between two identical columns, so their squared importances TIE, and
    `RFE` then eliminates whichever of the pair its solver's last bits happened to
    rank lower. mlrs solves the normal equations by Gaussian elimination and
    sklearn by Cholesky, so the two disagree about a tie that is exact in real
    arithmetic — producing a different, equally correct mask at
    `n_features_to_select=3`, where only one of the pair survives.

    Breaking the duplicate removes the coin flip and lets every meta-selector mask
    be compared for EXACT equality, which is the whole point of choosing a
    closed-form inner model. The other degenerate columns (constant, all-zero,
    perfectly correlated) are KEPT: both sides handle those deterministically.
    """
    x, y_class, y_reg = _design(seed)
    rng = np.random.default_rng(seed + 1)
    x[:, 5] = rng.standard_normal(N_SAMPLES) + 3.0
    return x, y_class, y_reg


def gen_feature_selection_meta(seed: int = SEED, dtype=np.float32) -> str:
    """The four meta-selectors, driven by `Ridge(alpha=1, fit_intercept=False)`.

    The inner model is chosen so the Rust test can reproduce it EXACTLY:
    `coef_ = (XᵀX + αI)⁻¹Xᵀy` is a closed form with no centering and no
    iteration, so the masks compare for EQUALITY rather than closeness. A
    tree-based inner model would have made the fixture a comparison of two RNG
    streams instead of a comparison of two selectors.

    `RFECV` and `SequentialFeatureSelector` additionally need a CV split and a
    score; `cv=3` unshuffled `KFold` and the default `r2` scorer are used, both
    of which are deterministic — the Rust `Cv::Folds{stratified:false}` layout
    and an R² fold scorer reproduce them.
    """
    from sklearn.feature_selection import (
        RFE,
        RFECV,
        SelectFromModel,
        SequentialFeatureSelector,
    )
    from sklearn.linear_model import Ridge

    # The DE-DUPLICATED design (see `_meta_design`): a duplicate column makes
    # `RFE`'s elimination order a coin flip between two tied importances.
    x, _, y_reg = _meta_design(seed)
    out = {"X": _c(x, dtype), "y_reg": _c(y_reg, dtype)}

    def ridge():
        return Ridge(alpha=META_ALPHA, fit_intercept=False)

    # The reference coefficients on the FULL design, so a Rust-side inner-model
    # mismatch is diagnosed directly instead of showing up as a wrong mask.
    out["ridge_coef_full"] = np.ascontiguousarray(
        ridge().fit(x, y_reg).coef_, dtype=np.float64
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # -- SelectFromModel: threshold forms x max_features x norm_order --
        for tag, kwargs in (
            ("none", {}),
            ("mean", {"threshold": "mean"}),
            ("median", {"threshold": "median"}),
            ("scaled", {"threshold": "1.25*mean"}),
            ("num", {"threshold": 0.05}),
            ("maxf3", {"max_features": 3}),
            ("maxf3_num", {"max_features": 3, "threshold": 0.5}),
        ):
            est = SelectFromModel(ridge(), **kwargs).fit(x, y_reg)
            out[f"sfm_{tag}_support"] = _mask(est.get_support())
            out[f"sfm_{tag}_threshold"] = np.ascontiguousarray(
                [est.threshold_], dtype=np.float64
            )

        # -- RFE: n_features_to_select x step --
        for tag, kwargs in (
            ("default", {}),
            ("n3", {"n_features_to_select": 3}),
            ("n3_step2", {"n_features_to_select": 3, "step": 2}),
            ("frac", {"n_features_to_select": 0.5}),
            ("stepfrac", {"n_features_to_select": 2, "step": 0.3}),
        ):
            est = RFE(ridge(), **kwargs).fit(x, y_reg)
            out[f"rfe_{tag}_support"] = _mask(est.get_support())
            out[f"rfe_{tag}_ranking"] = np.ascontiguousarray(est.ranking_, dtype=np.float64)

        # -- RFECV --
        for tag, kwargs in (
            ("cv3", {"cv": 3}),
            ("cv3_min3", {"cv": 3, "min_features_to_select": 3}),
            ("cv3_step2", {"cv": 3, "step": 2}),
        ):
            est = RFECV(ridge(), **kwargs).fit(x, y_reg)
            out[f"rfecv_{tag}_support"] = _mask(est.get_support())
            out[f"rfecv_{tag}_ranking"] = np.ascontiguousarray(est.ranking_, dtype=np.float64)
            out[f"rfecv_{tag}_n_features"] = np.ascontiguousarray(
                est.cv_results_["n_features"], dtype=np.float64
            )
            out[f"rfecv_{tag}_mean_test_score"] = np.ascontiguousarray(
                est.cv_results_["mean_test_score"], dtype=np.float64
            )
            out[f"rfecv_{tag}_std_test_score"] = np.ascontiguousarray(
                est.cv_results_["std_test_score"], dtype=np.float64
            )

        # -- SequentialFeatureSelector --
        for tag, kwargs in (
            ("fwd3", {"n_features_to_select": 3, "cv": 3}),
            ("bwd3", {"n_features_to_select": 3, "cv": 3, "direction": "backward"}),
            ("auto", {"cv": 3}),
            ("tol", {"cv": 3, "tol": 0.01}),
        ):
            est = SequentialFeatureSelector(ridge(), **kwargs).fit(x, y_reg)
            out[f"sfs_{tag}_support"] = _mask(est.get_support())
            out[f"sfs_{tag}_n_selected"] = np.ascontiguousarray(
                [est.n_features_to_select_], dtype=np.float64
            )

    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    path = os.path.join(_FIXTURE_DIR, f"fsel_meta_{_dtype_tag(dtype)}_seed{seed}.npz")
    np.savez(path, **out)
    return path


def main() -> None:
    """Regenerate every feature-selection fixture, f32 and f64."""
    for dtype in (np.float32, np.float64):
        print(f"wrote {gen_feature_selection_scores(dtype=dtype)}")
        print(f"wrote {gen_feature_selection_mutual_info(dtype=dtype)}")
        print(f"wrote {gen_feature_selection_univariate(dtype=dtype)}")
        print(f"wrote {gen_feature_selection_variance(dtype=dtype)}")
        print(f"wrote {gen_feature_selection_meta(dtype=dtype)}")


if __name__ == "__main__":
    main()
