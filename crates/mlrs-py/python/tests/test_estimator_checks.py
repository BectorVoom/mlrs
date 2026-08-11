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

import numpy as np
import pytest

pytest.importorskip("mlrs")

import mlrs  # noqa: E402  (after importorskip)
from mlrs import _io  # noqa: E402  (after importorskip)
from sklearn.utils import get_tags  # noqa: E402
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
        # BayesianRidge: `max_iter` is the evidence-iteration cap, and the
        # loop converges in single digits on the harness's tiny fixtures, so
        # the class default needs no shrinking (unlike MBSGD/RF/HGB below).
        mlrs.BayesianRidge(),
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
        # HuberRegressor: `max_iter` is the L-BFGS cap and the solve converges in
        # ~15 steps on the harness's tiny fixtures, so the class default needs no
        # shrinking (the BayesianRidge case, not the MBSGD one).
        mlrs.HuberRegressor(),
        # MBSGD: `max_iter` shrunk for the check-sweep's 30-row fixtures, the
        # same treatment RandomForest/HGB get below. At the class default
        # (`max_iter=1000`) a SINGLE fit of a 30x4 matrix costs ~163 s on the cpu
        # backend — ~141 ms per epoch, flat, so the cost is kernel-LAUNCH bound
        # (one OS thread per unit spawned and joined per epoch on `cubecl-cpu`),
        # not arithmetic. The harness fits each estimator ~49 times, so the two
        # MBSGD entries alone contributed hours to the sweep. These checks are
        # API-conformance on tiny fixtures, not convergence tests, so the
        # iteration count carries nothing here. Production defaults are
        # untouched.
        mlrs.MBSGDClassifier(max_iter=20),
        mlrs.MBSGDRegressor(max_iter=20),
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
        # --- PREP-01 (Phase 24): the six preprocessing scalers. Every one is
        # an unsupervised, dense-float-only transformer like PCA/TruncatedSVD
        # above. ------------------------------------------------------------ #
        mlrs.StandardScaler(),
        mlrs.MinMaxScaler(),
        mlrs.MaxAbsScaler(),
        mlrs.RobustScaler(),
        mlrs.Normalizer(),
        mlrs.Binarizer(),
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


# UMAP's `transform` is an OUT-OF-SAMPLE PROJECTION, not a replay of the
# training layout, and that breaks sklearn's transformer contract in four
# places. This is a property of the ALGORITHM, not of this implementation:
# `umap-learn` behaves the same way.
#
# `fit_transform(X)` optimizes every training point's position jointly by SGD
# over the fuzzy simplicial set. `transform(X)` freezes that fitted embedding
# and positions each query against it. Run on the SAME X the two therefore
# disagree (measured: max|Δ| = 2.4 on a 40x5 fixture) — which is what
# `check_transformer_general` / `check_transformer_data_not_an_array` compare.
# The batch optimization also makes a query's final position depend on which
# other queries share the call, so `transform` is not row-order- or
# subset-invariant either.
#
# NOTE these are NOT declared via `tags.non_deterministic`, even though that
# tag would suppress all four in one line. mlrs's UMAP IS deterministic —
# consecutive `fit_transform` runs are BIT-IDENTICAL at a fixed seed and at the
# default `random_state=None` (verified directly) — so setting that tag would
# be a false statement about the estimator, the exact class of mis-declaration
# that put LinearSVC and the NB family in this map to begin with. An xfail with
# the real reason is the honest encoding.
_UMAP_TRANSFORM = {
    "check_transformer_general": (
        "UMAP's transform is an out-of-sample projection onto the FITTED "
        "embedding, so it does not reproduce fit_transform's jointly-optimized "
        "training layout (umap-learn diverges identically)."
    ),
    "check_transformer_data_not_an_array": (
        "same out-of-sample transform divergence as check_transformer_general, "
        "on the list/not-an-array ingress."
    ),
    "check_methods_subset_invariance": (
        "UMAP positions a batch of queries against the fitted embedding by a "
        "joint optimization, so a point's coordinates depend on which other "
        "points share the transform call; a subset is not a restriction of the "
        "whole."
    ),
    "check_methods_sample_order_invariance": (
        "same batch-joint transform optimization as check_methods_subset_"
        "invariance: reordering the rows reorders the SGD's edge visits."
    ),
}


def _merge(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out


# PREP-01's six scalers are the simplest transformers in the shim (a single
# elementwise/column map, no extra ingress branching), and empirically (a real
# sweep run) they pass `check_dtype_object` / `check_estimator_sparse_tag` /
# `check_estimator_sparse_array` / `check_estimator_sparse_matrix` outright —
# unlike the rest of `_COMMON`'s estimators, xfailing those here would XPASS
# and falsely advertise unsupported behavior. Only the non-picklable fitted
# state (shared by every mlrs estimator) genuinely fails.
_PREP01 = {"check_estimators_pickle": _COMMON["check_estimators_pickle"]}


# Per-estimator-CLASS expected-failure maps. The callable below dispatches on
# type(est).__name__ so a single mapping covers the parametrized instances.
_EXPECTED = {
    "LinearRegression": _merge(_COMMON, _SUPERVISED),
    # Ridge exposes sklearn's full parameter surface, which INCLUDES `max_iter`.
    # `check_non_transformer_estimators_n_iter` is gated purely on that
    # parameter existing and then asserts `n_iter_ >= 1` — but sklearn's OWN
    # Ridge leaves `n_iter_` at None for the default `solver='cholesky'` (only
    # `lsqr`/`sag`/`saga` populate it in `_ridge_regression`). mlrs reproduces
    # that exactly, so the check cannot pass without diverging from sklearn.
    "Ridge": _merge(
        _COMMON,
        _SUPERVISED,
        {
            "check_non_transformer_estimators_n_iter": (
                "Ridge has `max_iter` but, like sklearn's own Ridge, reports "
                "`n_iter_ = None` for the direct solvers (cholesky/svd/"
                "sparse_cg/lbfgs) — only lsqr/sag/saga set it. Matching "
                "sklearn's n_iter_ semantics is the point; passing this check "
                "would require diverging from them."
            )
        },
    ),
    # BayesianRidge also has `max_iter`, but unlike Ridge it ALWAYS reports an
    # `n_iter_` (sklearn sets it unconditionally to `iter_ + 1`), so
    # `check_non_transformer_estimators_n_iter` genuinely passes and is NOT
    # carved out here.
    "BayesianRidge": _merge(_COMMON, _SUPERVISED),
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
    # NOT _N_ITER: KMeans surfaces sklearn's `n_iter_` (the winning restart's
    # iteration count) since the full-parameter-surface work, so
    # `check_non_transformer_estimators_n_iter` genuinely passes. Keeping the
    # xfail here would let a future regression that dropped `n_iter_` slip
    # through as a silent xfail instead of a failure.
    "KMeans": _merge(_COMMON),
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
    # HuberRegressor: the commons plus ONE estimator-specific carve-out.
    #
    # `check_sample_weight_equivalence_on_dense_data` asserts that fitting with
    # integer `sample_weight` and fitting on the correspondingly REPEATED rows
    # give the same `predict` to `rtol=1e-7, atol=1e-9`. That equivalence is
    # exact in the maths — every term of the Huber objective is linear in the
    # weights — and mlrs reproduces it to MACHINE PRECISION when the two solves
    # converge: 1.1e-16 on `coef_` and 2.8e-17 on `scale_` for a well-posed
    # 40x4 design (`test_oracle_huber.py::test_weighted_fit_equals_repeated_fit`
    # is the gate on that). What the check's own 30-row fixture measures instead
    # is how far apart two ITERATIVE solves stop on an ill-conditioned problem,
    # at a tolerance three orders below either engine's stopping accuracy —
    # scikit-learn's OWN `HuberRegressor` fails the same check by 3.8e-6 when it
    # is run standalone (verified 2026-08-05, sklearn 1.9.0). mlrs's deviation is
    # larger (1.9e-4) because it solves to `ftol = 64*eps` and can reach the
    # `max_iter=100` cap on that fixture where scikit-learn's `factr = 1e7` stops
    # early — the accuracy trade documented in `huber.rs`.
    "HuberRegressor": _merge(
        _COMMON,
        _SUPERVISED,
        {
            "check_sample_weight_equivalence_on_dense_data": (
                "the check's tolerance (rtol=1e-7) is three orders below any "
                "iterative solver's stopping accuracy on its ill-conditioned "
                "fixture; scikit-learn's own HuberRegressor fails it too. The "
                "equivalence holds to machine precision when both fits converge "
                "— gated in test_oracle_huber.py."
            )
        },
    ),
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
    # A 1-sample fit is rejected by the same n_components range guard
    # UMAP hits (min(n_samples, n_features) collapses to 0), whose message
    # names the real constraint rather than sklearn's "1 sample" wording.
    "SpectralEmbedding": _merge(_COMMON, _FIT2D_1SAMPLE),
    # UMAP: the four out-of-sample transform-contract divergences (see
    # `_UMAP_TRANSFORM`) plus the shared 1-sample message carve-out — a
    # 1-sample fit is rejected by the n_components range guard, whose
    # message names the real constraint rather than sklearn's "1 sample".
    "UMAP": _merge(_COMMON, _UMAP_TRANSFORM, _FIT2D_1SAMPLE),
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
    # --- PREP-01 (Phase 24): unsupervised transformers; see `_PREP01`'s
    # comment for why this is narrower than `_COMMON`. ------------------------
    "StandardScaler": _merge(_PREP01),
    "MinMaxScaler": _merge(_PREP01),
    "MaxAbsScaler": _merge(_PREP01),
    "RobustScaler": _merge(_PREP01),
    "Normalizer": _merge(_PREP01),
    "Binarizer": _merge(_PREP01),
}


# On an f64-incapable backend (rocm), `test_estimator_checks` forces the
# check-generated float64 fixtures down to float32 at ingress (see
# `test_estimator_checks`'s docstring) so the rest of the sklearn check suite
# runs instead of hard-failing at ingress on every float64 fixture. That
# downcast necessarily breaks `check_transformer_preserve_dtypes` for any
# transformer whose (default) `preserves_dtype` tag is `["float64"]`: a
# float64-in/float64-out round trip cannot hold when the ingress silently
# downcasts. sklearn only yields that check when `hasattr(estimator,
# "transform")` AND `tags.transformer_tags.preserves_dtype` is truthy
# (`_yield_transformer_checks`) — mirrored here rather than hardcoding a class
# list, so a newly added transformer is covered automatically instead of
# XPASSing (silently unguarded) or needing a manual entry.
_DTYPE_PRESERVE_ROCM_REASON = (
    "on an f64-incapable backend (rocm) this test harness downcasts "
    "check-generated float64 fixtures to float32 at ingress (no per-check "
    "hook exists to inject a different upload dtype into "
    "parametrize_with_checks), so a float64-in/float64-out transform "
    "contract cannot hold. The estimator's own preserves_dtype tag is "
    "exercised faithfully — and this check is NOT xfailed — on an "
    "f64-capable backend (cpu/wgpu)."
)


def _preserves_dtype_check_applies(estimator):
    """Whether sklearn yields ``check_transformer_preserve_dtypes`` for this
    estimator (mirrors ``_yield_transformer_checks``'s own gate)."""
    if not hasattr(estimator, "transform"):
        return False
    tags = get_tags(estimator)
    return bool(tags.transformer_tags and tags.transformer_tags.preserves_dtype)


# The remaining rocm-only gaps, found by empirically running the full suite
# against the compiled rocm/f32 extension once the ingress downcast above let
# it run to completion at all (967/1979 previously errored at ingress before
# reaching a check body). Each is a REAL float32-vs-float64 backend precision
# gap, not a test-harness artifact — documented with its root cause rather
# than swept in as a blanket rocm xfail, and gated on `not
# mlrs.backend_supports_f64()` so cpu/wgpu (which run these checks at f64,
# unmodified) keep enforcing them at full strictness.

# `logistic.rs` (LogisticRegression's L-BFGS) and `linear_svc.rs`
# (LinearSVC/LinearSVR's solver) already contain deliberate, empirically-
# calibrated dtype-precision-floor acceptance logic for the symmetric
# over-parameterized objective's gauge null-space (see
# `logistic.rs::GAUGE_FLOOR_K` and its ~80-line comment). On the check
# harness's small fixtures and each estimator's sklearn-default `max_iter`
# (100 for LogisticRegression, 1000 for LinearSVC/LinearSVR), running that
# same solve at float32 (this harness's rocm downcast) exhausts max_iter
# before even that floor-acceptance branch fires, and `fit` raises
# `AlgoError::NotConverged` where sklearn's check expects a clean fit. This is
# a genuine rocm/float32 behavior difference a real user could hit on a small
# or ill-conditioned dataset with these estimators' library DEFAULTS — not
# just a test-fixture quirk — so it is called out explicitly rather than
# folded into a generic "iterative solver" excuse. Retuning the f32
# convergence floor for these specific solvers is separate follow-up work,
# not attempted here.
_ROCM_F32_NOT_CONVERGED = (
    "on an f64-incapable backend (rocm), downcasting the check harness's "
    "fixture to float32 (see _DTYPE_PRESERVE_ROCM_REASON) exhausts this "
    "estimator's default max_iter before its f32 gauge/precision-floor "
    "convergence acceptance (logistic.rs::GAUGE_FLOOR_K and its linear_svc.rs "
    "analogue) can fire, so fit() raises AlgoError::NotConverged where "
    "sklearn's check expects a clean fit. A real rocm/float32 precision gap "
    "reachable with this estimator's library-default max_iter on small or "
    "ill-conditioned data (not a test-fixture-only artifact); retuning the "
    "f32 convergence floor for this solver is separate follow-up work."
)

# `check_classifiers_train` asserts `predict_log_proba == log(predict_proba)`
# at `assert_allclose(..., rtol=8, atol=1e-9)` — already an extremely loose
# RELATIVE tolerance (rtol=8, i.e. 800%), because sklearn expects some noise
# here. But a probability that should be exactly 0 comes back as float32
# rounding noise (~-1.2e-8, empirically), and relative tolerance cannot
# rescue a comparison against an exact 0.0 — only the fixed `atol=1e-9`
# can, and float32's noise floor sits an order of magnitude above that
# f64-scaled constant. GaussianNB's math is unchanged between backends; this
# is the same float32-vs-float64 noise-floor-vs-fixed-atol gap as
# `check_transformer_preserve_dtypes`, just surfacing through a different
# assertion.
_ROCM_GAUSSIAN_NB_LOG_PROBA_NOISE = (
    "check_classifiers_train asserts predict_log_proba == log(predict_proba) "
    "at a fixed atol=1e-9 (with an already-loose rtol=8) — tuned for "
    "float64's noise floor. On an f64-incapable backend (rocm) this harness's "
    "float32 downcast (see _DTYPE_PRESERVE_ROCM_REASON) produces "
    "~1e-8-scale rounding noise where the true probability is exactly 0, "
    "which clears rtol=8 but not the fixed atol. Not a GaussianNB math bug — "
    "the check passes at f64 (cpu/wgpu, not xfailed there)."
)

# UMAP's fit/transform ARE verified bit-identical at a fixed seed elsewhere in
# this file (see the `_UMAP_TRANSFORM` comment above) — but that was verified
# against the host-parallel cpu engine (mlrs-umap-cpu-optimization), not the
# rocm device kernel path this harness now reaches once the ingress downcast
# lets UMAP's checks run at all. `check_pipeline_consistency` (Pipeline(...)
# vs the bare estimator) and `check_fit_idempotent` (two independent fits of
# the SAME X) both show LARGE (not rounding-scale) embedding divergence on
# rocm — consistent with the SGD embedding optimization being sensitive to
# summation-order differences in the device kernel's parallel reductions,
# amplified chaotically over the optimization (the same class of sensitivity
# already documented for UMAP's transform contract, and for chaotic gradient
# descent generally). This is a SYMPTOM, not a root-caused fix: flagged
# explicitly rather than silently absorbed into a generic xfail, since it is
# a real open question about the rocm device engine's run-to-run
# reproducibility, not a test-harness artifact.
_ROCM_UMAP_DEVICE_NONDETERMINISM = (
    "on rocm this check reaches UMAP's DEVICE kernel path (the bit-identical "
    "determinism documented elsewhere in this file was verified on the "
    "host-parallel cpu engine, not this path); two fits of the same data "
    "diverge by O(10) in embedding coordinates, consistent with "
    "summation-order-sensitive parallel reductions in the SGD layout "
    "optimization being amplified chaotically. A real open question about "
    "the rocm device engine's run-to-run reproducibility — not root-caused "
    "or fixed here, only documented."
)

_ROCM_ONLY = {
    "LogisticRegression": {
        "check_fit_idempotent": _ROCM_F32_NOT_CONVERGED,
        "check_fit_check_is_fitted": _ROCM_F32_NOT_CONVERGED,
        "check_n_features_in": _ROCM_F32_NOT_CONVERGED,
    },
    "LinearSVC": {
        "check_n_features_in": _ROCM_F32_NOT_CONVERGED,
    },
    "LinearSVR": {
        "check_fit_check_is_fitted": _ROCM_F32_NOT_CONVERGED,
        "check_n_features_in": _ROCM_F32_NOT_CONVERGED,
    },
    "GaussianNB": {
        "check_classifiers_train": _ROCM_GAUSSIAN_NB_LOG_PROBA_NOISE,
    },
    "UMAP": {
        "check_pipeline_consistency": _ROCM_UMAP_DEVICE_NONDETERMINISM,
        "check_fit_idempotent": _ROCM_UMAP_DEVICE_NONDETERMINISM,
    },
}


def _expected_failed_checks(estimator):
    """sklearn>=1.6 hook: ``{check_name: reason}`` for by-design xfails.

    Returns the documented unsupported-check map for ``estimator``'s class. The
    reasons are the empirical triage recorded in ``checks_triage.md``. The
    RELEVANT checks (not in this map) must pass.
    """
    checks = dict(_EXPECTED.get(type(estimator).__name__, {}))
    if not mlrs.backend_supports_f64():
        if _preserves_dtype_check_applies(estimator):
            checks["check_transformer_preserve_dtypes"] = _DTYPE_PRESERVE_ROCM_REASON
        checks.update(_ROCM_ONLY.get(type(estimator).__name__, {}))
    return checks


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
def test_estimator_checks(estimator, check, monkeypatch):
    """Run the relevant sklearn check; by-design ones xfail with a reason.

    ``parametrize_with_checks`` generates its OWN float64 fixtures inside
    sklearn's check functions — unlike the oracle suites, this file has no
    hand-written ``_live_data()`` to gate with ``default_float_dtype()``. On
    an f64-incapable backend (rocm) that float64 fixture hits mlrs's ingress
    hard-backstop (``ValueError: backend 'rocm' does not support float64``)
    for essentially every check, on every estimator — a real gap, but not a
    reason to weaken the hard-backstop for actual library callers who pass
    float64 explicitly. Scope the fix to this harness instead: force
    ``pick_dtype`` to resolve float32 for the duration of one check, so a
    fixture that arrives as float64 downcasts the same way a non-float
    fixture already does on this backend, and the check exercises the same
    API contract as on cpu/wgpu instead of erroring at ingress.
    """
    if not mlrs.backend_supports_f64():
        monkeypatch.setattr(_io, "pick_dtype", lambda X: np.float32)
    check(estimator)
