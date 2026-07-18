"""sklearn ``estimator_checks`` triage for the 12 mlrs estimators (criterion 1).

This is the empirical Wave-0 triage the phase gate calls for: run the *relevant*
subset of ``sklearn.utils.estimator_checks`` against every estimator and let the
RELEVANT checks pass, while the by-design-unsupported checks are declared as
``expected_failed_checks`` (sklearn's native xfail mechanism, >=1.6) carrying a
documented reason. Criterion 1 says "relevant", NOT "all checks pass" — so a
check that fails ONLY because mlrs intentionally does not support it (sparse /
object-dtype / pickling / sklearn-faithful predict-less surface / v1 contiguous
int labels) is xfailed-with-reason here and recorded in ``checks_triage.md``.

Any check that failed for a REAL bug was fixed upstream in the Plan-04 shim
(this plan added ``n_features_in_`` via ``MlrsBase._post_fit`` so the default
``check_is_fitted`` scan, ``check_n_features_in`` and the unfitted-raises checks
pass) — it is NOT masked here. The mapping below is the discovered set; growing
it to hide a genuine regression would violate the triage rule.

Run requires the compiled ``_mlrs`` (``maturin develop`` with a backend
pyproject); collected-and-skipped otherwise so the suite stays green pre-build.
"""

import pytest

pytest.importorskip("mlrs")

import mlrs  # noqa: E402  (after importorskip)
from sklearn.utils.estimator_checks import (  # noqa: E402
    parametrize_with_checks,
)


def _estimators():
    """The full v1 estimator set, constructed with valid v1 hyperparameters.

    ``n_clusters`` / ``n_components`` are given explicitly where the v1 ctor has
    no sklearn-compatible default that the check harness's tiny fixtures admit.
    Plan 16-11 extends this from the original 12 to the full shim set (the 15
    newly-added classes are appended below); the fit-free subset
    (``check_no_attributes_set_in_init`` / ``check_parameters_default_constructible``
    / ``check_get_params_invariance``) runs GREEN for every class pre-build, and
    is NEVER xfailed (A7 / D-07 step 4).
    """
    return [
        mlrs.LinearRegression(),
        mlrs.Ridge(),
        mlrs.Lasso(),
        mlrs.ElasticNet(),
        mlrs.LogisticRegression(),
        mlrs.KMeans(n_clusters=3),
        mlrs.DBSCAN(),
        mlrs.PCA(n_components=2),
        mlrs.TruncatedSVD(n_components=2),
        mlrs.NearestNeighbors(),
        mlrs.KNeighborsClassifier(),
        mlrs.KNeighborsRegressor(),
        # --- Plan 16-11: the 15 newly-added shim classes. ----------------- #
        mlrs.LinearSVC(),
        mlrs.LinearSVR(),
        mlrs.MBSGDClassifier(),
        mlrs.MBSGDRegressor(),
        mlrs.GaussianNB(),
        mlrs.MultinomialNB(),
        mlrs.BernoulliNB(),
        mlrs.ComplementNB(),
        mlrs.CategoricalNB(),
        mlrs.KernelRidge(),
        mlrs.KernelDensity(),
        mlrs.SpectralClustering(n_clusters=3),
        mlrs.SpectralEmbedding(n_components=2),
        mlrs.UMAP(n_components=2),
        mlrs.HDBSCAN(),
        # --- TASK-16 (PY-ENS-05, RF): small, cheap hyperparameters for the
        # check-sweep's tiny fixtures (not the class defaults). ------------ #
        mlrs.RandomForestClassifier(n_estimators=5, max_depth=3),
        mlrs.RandomForestRegressor(n_estimators=5, max_depth=3),
        # --- TASK-25 (PY-ENS-05, HGB): small max_iter for the check-sweep's
        # tiny fixtures (not the class default). --------------------------- #
        mlrs.HistGradientBoostingClassifier(max_iter=10),
        mlrs.HistGradientBoostingRegressor(max_iter=10),
    ]


# --- by-design-unsupported checks, xfailed WITH A DOCUMENTED REASON --------- #
#
# These are NOT bugs. Each entry says WHY mlrs intentionally does not satisfy
# the check. The same triage is recorded human-readably in checks_triage.md.

# Checks every mlrs estimator skips by design (dense-float-only, non-picklable
# device handle, object-dtype message divergence).
_COMMON = {
    "check_estimator_sparse_tag": (
        "mlrs ingests dense Arrow only; the sparse input tag is off by design "
        "(__sklearn_tags__.input_tags.sparse = False)."
    ),
    "check_estimator_sparse_array": (
        "sparse input unsupported by design — dense Arrow ingress only."
    ),
    "check_estimator_sparse_matrix": (
        "sparse input unsupported by design — dense Arrow ingress only."
    ),
    "check_estimators_pickle": (
        "the fitted state is an opaque Rust `_mlrs` #[pyclass] device handle "
        "that is not picklable in v1 (model serialization is out of v1 scope)."
    ),
    "check_dtype_object": (
        "object/string-dtype X is rejected, but via numpy's float-cast error "
        "whose message does not match sklearn's expected substring; mlrs is "
        "dense-float-only by design (the rejection still happens)."
    ),
}

# Supervised-target message/shape conventions mlrs does not replicate in v1.
#
# NOTE: check_supervised_y_no_nan is intentionally NOT in this map. As of the
# WR-01 fix, _io.normalize_y runs sklearn check_array(ensure_all_finite=True)
# on y, so a NaN/Inf target is rejected with sklearn's own ValueError message
# for EVERY supervised estimator. The check therefore PASSES (verified
# empirically) and must not be xfailed — xfailing it would xpass and break the
# suite, and would falsely advertise an unsupported behavior we now support.
_SUPERVISED = {
    "check_supervised_y_2d": (
        "mlrs does not emit sklearn's DataConversionWarning when a column-vector "
        "y is passed; 1-D y is the v1 contract (no silent 2-D->1-D reshape warn)."
    ),
    "check_requires_y_none": (
        "the 'y is required' error message does not match sklearn's expected "
        "pattern verbatim (a regressor/classifier still raises on y=None)."
    ),
}

# Classifier-only: v1 uses contiguous int labels 0..n-1; string / continuous
# targets are out of the v1 label contract.
_CLASSIFIER = {
    "check_classifiers_classes": (
        "v1 classifiers require contiguous int32 labels 0..n_classes-1; string "
        "class labels are out of the v1 label contract."
    ),
    "check_classifiers_regression_target": (
        "continuous-target rejection is not emitted with sklearn's exact "
        "message; v1 expects pre-encoded discrete labels."
    ),
}

# n_iter_ convergence attribute is not surfaced by the iterative solvers in v1.
_N_ITER = {
    "check_non_transformer_estimators_n_iter": (
        "the iterative solvers (coordinate descent / L-BFGS / Lloyd) do not "
        "surface an `n_iter_` attribute in v1."
    ),
}

# Small-fixture numerical edge cases sklearn probes with 1-sample / degenerate
# inputs that the v1 solvers do not special-case with sklearn's message.
_FIT2D_1SAMPLE = {
    "check_fit2d_1sample": (
        "a 1-sample fit is not special-cased with sklearn's exact '1 sample' "
        "message; the solver instead raises/produces a degenerate result."
    ),
}


def _merge(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out


# Per-estimator-CLASS expected-failure maps. The callable below dispatches on
# type(est).__name__ so a single mapping covers the parametrized instances.
_EXPECTED = {
    "LinearRegression": _merge(_COMMON, _SUPERVISED),
    "Ridge": _merge(_COMMON, _SUPERVISED),
    "Lasso": _merge(_COMMON, _SUPERVISED, _N_ITER),
    "ElasticNet": _merge(_COMMON, _SUPERVISED, _N_ITER),
    # LogisticRegression's y (like every supervised estimator's) now goes
    # through check_array in _io.normalize_y (WR-01), so check_supervised_y_no_nan
    # PASSES; it is not in _SUPERVISED anymore so no per-estimator carve-out is
    # needed here.
    "LogisticRegression": _merge(
        _COMMON,
        _SUPERVISED,
        _CLASSIFIER,
        _N_ITER,
        _FIT2D_1SAMPLE,
    ),
    "KMeans": _merge(_COMMON, _N_ITER),
    "DBSCAN": _merge(_COMMON),
    "PCA": _merge(_COMMON, _FIT2D_1SAMPLE),
    "TruncatedSVD": _merge(_COMMON, _FIT2D_1SAMPLE),
    "NearestNeighbors": _merge(_COMMON),
    "KNeighborsClassifier": _merge(_COMMON, _SUPERVISED, _CLASSIFIER),
    "KNeighborsRegressor": _merge(_COMMON, _SUPERVISED),
    # --- Plan 16-11: the 15 newly-added shim classes. --------------------- #
    # Linear SVM / MBSGD: iterative solvers (no n_iter_); SVR/MBSGDRegressor are
    # supervised regressors, SVC/MBSGDClassifier supervised classifiers.
    "LinearSVR": _merge(_COMMON, _SUPERVISED, _N_ITER),
    "MBSGDRegressor": _merge(_COMMON, _SUPERVISED, _N_ITER),
    "LinearSVC": _merge(_COMMON, _SUPERVISED, _CLASSIFIER, _N_ITER),
    "MBSGDClassifier": _merge(_COMMON, _SUPERVISED, _CLASSIFIER, _N_ITER),
    # Naive-Bayes: supervised classifiers with v1 contiguous-int labels.
    "GaussianNB": _merge(_COMMON, _SUPERVISED, _CLASSIFIER),
    "MultinomialNB": _merge(_COMMON, _SUPERVISED, _CLASSIFIER),
    "BernoulliNB": _merge(_COMMON, _SUPERVISED, _CLASSIFIER),
    "ComplementNB": _merge(_COMMON, _SUPERVISED, _CLASSIFIER),
    "CategoricalNB": _merge(_COMMON, _SUPERVISED, _CLASSIFIER),
    # KernelRidge: supervised regressor.
    "KernelRidge": _merge(_COMMON, _SUPERVISED),
    # KernelDensity: unsupervised density estimator (dense-float-only commons).
    "KernelDensity": _merge(_COMMON),
    # Cluster / manifold: unsupervised, dense-float-only commons.
    "SpectralClustering": _merge(_COMMON),
    "HDBSCAN": _merge(_COMMON),
    "SpectralEmbedding": _merge(_COMMON),
    "UMAP": _merge(_COMMON),
    # --- TASK-16 (PY-ENS-05, RF): empirically triaged against a real sweep
    # run (Green-time, not assumed) — see this file's own module docstring
    # ("Criterion 1 says 'relevant', NOT 'all checks pass'"). RandomForest*
    # is a supervised, dense-float-only, non-picklable-fitted-state estimator
    # like every other supervised shim above, so the same _COMMON/_SUPERVISED
    # carve-outs apply; the classifier additionally needs _CLASSIFIER (v1
    # contiguous-int-label contract) and _FIT2D_1SAMPLE (a 1-sample fit is not
    # special-cased with sklearn's exact '1 sample' message — same failure
    # LogisticRegression/PCA/TruncatedSVD already carry). Neither needs
    # _N_ITER: RF's tree-growth loop has no iterative-solver `n_iter_`
    # convergence concept, and check_non_transformer_estimators_n_iter did
    # NOT fail in the Green-time sweep for either estimator.
    "RandomForestClassifier": _merge(
        _COMMON, _SUPERVISED, _CLASSIFIER, _FIT2D_1SAMPLE
    ),
    "RandomForestRegressor": _merge(_COMMON, _SUPERVISED),
    # --- TASK-25 (PY-ENS-05, HGB): empirically triaged against a real sweep
    # run (Green-time, not assumed). HistGradientBoosting* is a supervised,
    # dense-float-only, non-picklable-fitted-state estimator like the other
    # supervised shims, so the same _COMMON/_SUPERVISED carve-outs apply; the
    # classifier additionally needs _CLASSIFIER (v1 contiguous-int-label
    # contract) and _FIT2D_1SAMPLE (mirrors RandomForestClassifier/
    # LogisticRegression/PCA/TruncatedSVD's own "1-sample fit is not
    # special-cased with sklearn's exact message" failure). UNLIKE RandomForest
    # (whose tree-growth loop has no iterative-solver convergence concept),
    # BOTH HGB estimators DO fail check_non_transformer_estimators_n_iter in
    # the Green-time sweep (the boosting-round loop has no surfaced `n_iter_`
    # attribute in v1) — so _N_ITER is included for both here, unlike RF.
    "HistGradientBoostingClassifier": _merge(
        _COMMON, _SUPERVISED, _CLASSIFIER, _N_ITER, _FIT2D_1SAMPLE
    ),
    "HistGradientBoostingRegressor": _merge(_COMMON, _SUPERVISED, _N_ITER),
}


def _expected_failed_checks(estimator):
    """sklearn>=1.6 hook: ``{check_name: reason}`` for by-design xfails.

    Returns the documented unsupported-check map for ``estimator``'s class. The
    reasons are the empirical triage recorded in ``checks_triage.md``. The
    RELEVANT checks (not in this map) must pass.
    """
    return dict(_EXPECTED.get(type(estimator).__name__, {}))


# The three fit-free checks that MUST run green for every estimator (they need
# no compiled `_mlrs`): a faithful pure __init__ + zero-arg constructibility +
# get_params invariance. They must NEVER appear in any per-class xfail map (A7 /
# D-07 step 4) — masking one would hide a real SHIM-01 regression.
_FIT_FREE_CHECKS = (
    "check_no_attributes_set_in_init",
    "check_parameters_default_constructible",
    "check_get_params_invariance",
)


def test_fit_free_checks_never_xfailed():
    """No estimator xfails the fit-free subset (they must run green)."""
    for est in _estimators():
        xfails = _expected_failed_checks(est)
        leaked = [c for c in _FIT_FREE_CHECKS if c in xfails]
        assert not leaked, (
            f"{type(est).__name__} xfails fit-free check(s) {leaked} — these "
            f"must run green (SHIM-01); remove them from the xfail map"
        )


@parametrize_with_checks(
    _estimators(),
    expected_failed_checks=_expected_failed_checks,
)
def test_estimator_checks(estimator, check):
    """Run the relevant sklearn check; by-design ones xfail with a reason."""
    check(estimator)
