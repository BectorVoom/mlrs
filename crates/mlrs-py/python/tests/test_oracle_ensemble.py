"""Random-forest oracle harness (PY-ENS-01, RF-IMP-02, RF-OOB-02: full binding
path — RandomForestClassifier).

Replays the committed ``rf_cls_{f32,f64}_seed42.npz`` fixtures (produced by
``scripts/gen_oracle.py``; the SAME fixtures already gated at the Rust layer
by ``crates/mlrs-algos/tests/random_forest_classifier_test.rs``, TASK-01..07)
through the FULL ``mlrs.RandomForestClassifier`` Python binding path — a
SECOND consumer, no regeneration (mirrors ``test_oracle_neighbors.py``'s own
harness shape, RESEARCH.md §5.4):

  - Deterministic tier (``bootstrap=False, max_features=1.0`` [see Deviation
    note below], ``n_estimators=2, max_depth=12`` — mirrors
    ``random_forest_classifier_test.rs``'s own ``RF_DET_MAX_DEPTH``/2-tree
    construction): train-set ``predict`` matches ``det_pred_train`` exactly;
    ``predict_proba`` matches ``det_proba_train`` within ``1e-5``/``1e-4``.
  - Statistical tier (``n_estimators=64, max_depth=8`` — mirrors
    ``RF_STAT_N_ESTIMATORS``/``RF_STAT_MAX_DEPTH``, bootstrap/sqrt
    defaults): held-out accuracy within ``ACC_MARGIN=0.05`` of
    ``stat_acc_test``.
  - ``feature_importances_`` (RF-IMP-02): ``atol=0.05`` on the deterministic
    tier — **NOT** an exact match (SPEC.md ``spec_revision: 2`` / TASK-02's
    Objective: sklearn's own splitter breaks near-tied candidate splits using
    internal state independent of the public seed, so sklearn's own two
    deterministic-tier trees are not bit-identical to each other even though
    mlrs's are; the qualitative dominant-feature-ranking test in the Rust
    suite remains the PRIMARY correctness signal for RF-IMP-01, this is a
    secondary end-to-end replay) + sums to 1.
  - ``oob_score_`` (RF-OOB-02): statistical-tier band (``OOB_MARGIN=0.10``,
    duplicated from
    ``crates/mlrs-algos/tests/random_forest_classifier_test.rs::OOB_MARGIN``/
    ``OOB_MARGIN_F32`` — TASK-06; tune both together, see that file's own
    Green-time-observed-divergence doc-comment before ever widening either)
    + ``AttributeError`` (sklearn parity, NOT a silent ``None``) when
    ``oob_score=False`` (the default).
  - ``ValueError`` on a bogus ``max_features`` string and on
    ``oob_score=True, bootstrap=False``.

DEVIATION from the plan's literal Red-test prose (``max_features=None``):
confirmed via ``crates/mlrs-py/src/estimators/ensemble.rs::PyRandomForestClassifier::new``
(TASK-08's own documented, PyO3-boundary-forced deviation, restated in
``ensemble.py``'s own module docstring) that ``max_features=None`` — whether
omitted or passed explicitly — collapses to the CLASSIFIER's sklearn default
``"sqrt"``, NOT sklearn's "all features" encoding (PyO3's
``Option<&Bound<PyAny>>`` extraction cannot distinguish "argument omitted"
from "argument explicitly None"). The Rust-layer deterministic-tier oracle
test (``random_forest_classifier_test.rs``, TASK-01/02) builds with
``MaxFeatures::All`` explicitly, which is required for ``sample_features`` to
short-circuit to its zero-RNG identity pass (``mf == d``, a plain numeric
comparison — ``crates/mlrs-backend/src/prims/random_forest.rs``) and
reproduce sklearn's bit-identical-tree assumption. This module therefore
passes ``max_features=1.0`` (the documented, unambiguous "all features"
encoding per ``ensemble.py``'s own module docstring), which resolves to
``MaxFeaturesArg::Frac(1.0)`` -> ``MaxFeatures::Value(ceil(1.0 * n_features))
== Value(n_features)`` — numerically identical to ``MaxFeatures::All`` for
this fixture's ``n_features=5`` (``MaxFeatures::Value(v).resolve(..) == v``,
``crates/mlrs-algos/src/ensemble/mod.rs``), so ``mf == d`` still holds and the
same zero-RNG identity path fires. Passing ``max_features=None`` here would
silently exercise ``MaxFeatures::Sqrt`` instead and NOT reproduce the
deterministic-tier exact-match precondition.
"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

import mlrs
from conftest import dtype_of, fixture_path, requires_f64


def _atol(fixture):
    return 1e-5 if dtype_of(fixture) == np.float64 else 1e-4


RF_CLS_FIXTURES = ["rf_cls_f32_seed42", "rf_cls_f64_seed42"]

# Statistical-tier held-out accuracy margin — mirrors
# crates/mlrs-algos/tests/random_forest_classifier_test.rs::ACC_MARGIN.
ACC_MARGIN = 0.05

# Deterministic-tier feature_importances_ tolerance — mirrors
# random_forest_classifier_test.rs::IMPORTANCE_TOL / IMPORTANCE_TOL_F32
# (SPEC.md spec_revision 2, TASK-02 Green-time resolution).
IMPORTANCE_ATOL = 0.05

# Statistical-tier oob_score_ margin — mirrors
# random_forest_classifier_test.rs::OOB_MARGIN / OOB_MARGIN_F32 (TASK-06).
OOB_MARGIN = 0.10

RF_STAT_N_ESTIMATORS = 64
RF_STAT_MAX_DEPTH = 8
RF_DET_N_ESTIMATORS = 2
RF_DET_MAX_DEPTH = 12


def _det_classifier(**kw):
    """Deterministic-tier construction, mirroring
    random_forest_classifier_test.rs::check_deterministic_tier's
    `.n_estimators(2).bootstrap(false).max_features(MaxFeatures::All)
    .max_depth(RF_DET_MAX_DEPTH)` builder chain."""
    return mlrs.RandomForestClassifier(
        n_estimators=RF_DET_N_ESTIMATORS,
        max_depth=RF_DET_MAX_DEPTH,
        bootstrap=False,
        max_features=1.0,
        seed=42,
        **kw,
    )


def _stat_classifier(**kw):
    """Statistical-tier construction, mirroring
    random_forest_classifier_test.rs::check_statistical_tier's
    `.n_estimators(RF_STAT_N_ESTIMATORS).max_depth(RF_STAT_MAX_DEPTH)`
    (bootstrap/max_features left at their sklearn-parity defaults)."""
    return mlrs.RandomForestClassifier(
        n_estimators=RF_STAT_N_ESTIMATORS,
        max_depth=RF_STAT_MAX_DEPTH,
        seed=42,
        **kw,
    )


@pytest.mark.parametrize("fixture", RF_CLS_FIXTURES)
@requires_f64
def test_random_forest_classifier_deterministic(fixture):
    """PY-ENS-01: deterministic-tier .fit(X,y).predict(X) matches
    det_pred_train exactly; .predict_proba(X) matches det_proba_train within
    1e-5 (f64) / 1e-4 (f32)."""
    d = np.load(fixture_path(fixture))
    clf = _det_classifier().fit(d["X"], d["y"])

    pred = np.asarray(clf.predict(d["X"])).astype(np.int64).ravel()
    assert np.array_equal(pred, d["det_pred_train"].astype(np.int64).ravel())

    proba = np.asarray(clf.predict_proba(d["X"]), dtype=np.float64)
    assert np.allclose(
        proba,
        d["det_proba_train"].astype(np.float64),
        atol=_atol(fixture),
        rtol=0.0,
    )


@pytest.mark.parametrize("fixture", RF_CLS_FIXTURES)
@requires_f64
def test_random_forest_classifier_statistical(fixture):
    """PY-ENS-01: statistical-tier held-out accuracy within ACC_MARGIN of
    stat_acc_test."""
    d = np.load(fixture_path(fixture))
    clf = _stat_classifier().fit(d["X"], d["y"])

    pred = np.asarray(clf.predict(d["Xq"])).astype(np.int64).ravel()
    yq = d["yq"].astype(np.int64).ravel()
    acc = float(np.mean(pred == yq))
    sk_acc = float(d["stat_acc_test"][0])
    assert acc >= sk_acc - ACC_MARGIN, (
        f"held-out accuracy {acc} below sklearn {sk_acc} - {ACC_MARGIN}"
    )


def test_random_forest_classifier_max_features_invalid_raises():
    """PY-ENS-01: an unrecognized max_features string raises ValueError.

    The shim's ``__init__`` only stores ctor args verbatim (sklearn purity
    rule — matches every other mlrs estimator's ``__init__``), so the actual
    parse/raise happens inside ``fit()``, where the low-level
    ``_mlrs.RandomForestClassifier`` constructor eagerly parses
    ``max_features`` (``parse_max_features``,
    ``crates/mlrs-py/src/estimators/ensemble.rs``)."""
    X = np.zeros((4, 2), dtype=np.float32)
    y = np.zeros((4,), dtype=np.float32)
    clf = mlrs.RandomForestClassifier(max_features="bogus")
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_random_forest_classifier_max_features_none_is_all_features():
    """PY-ENS-01 sklearn parity (code-review fix): at the user-facing shim
    layer, ``max_features=None`` means "use all features" (the shim forwards an
    explicit None as the ``"all"`` sentinel), distinct from the OMITTED default
    ``"sqrt"``. None, ``"all"``, and ``1.0`` must yield the identical
    all-features forest; ``"sqrt"`` (1 of 2 features here) is a different one.
    ``get_params()`` still reports the caller's original None so sklearn
    ``clone()`` round-trips faithfully."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(40, 2)).astype(np.float32)
    y = (X[:, 0] > 0.0).astype(np.int64)

    def proba(mf):
        clf = mlrs.RandomForestClassifier(
            n_estimators=8, max_depth=4, max_features=mf, bootstrap=False, seed=42
        )
        return np.asarray(clf.fit(X, y).predict_proba(X), dtype=np.float64)

    p_none, p_all, p_one, p_sqrt = proba(None), proba("all"), proba(1.0), proba("sqrt")
    np.testing.assert_allclose(p_none, p_all, atol=1e-6)
    np.testing.assert_allclose(p_none, p_one, atol=1e-6)
    assert not np.allclose(p_none, p_sqrt, atol=1e-6), (
        "shim max_features=None must behave as all-features, distinct from sqrt"
    )
    # get_params() reports the caller's original None (not the internal "all"),
    # so a sklearn clone() reconstructs the estimator faithfully.
    params = mlrs.RandomForestClassifier(max_features=None).get_params()
    assert params["max_features"] is None


def test_random_forest_classifier_not_fitted_raises():
    """PY-ENS-01/RF-IMP-02/RF-OOB-02: predict / predict_proba /
    feature_importances_ / oob_score_ before fit all raise the project's
    standard NotFittedError (mirrors naive_bayes.py's pattern)."""
    clf = mlrs.RandomForestClassifier()
    X = np.zeros((1, 5), dtype=np.float32)
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.predict_proba(X)
    with pytest.raises(NotFittedError):
        _ = clf.feature_importances_
    with pytest.raises(NotFittedError):
        _ = clf.oob_score_


@pytest.mark.parametrize("fixture", RF_CLS_FIXTURES)
@requires_f64
def test_random_forest_classifier_feature_importances_close(fixture):
    """RF-IMP-02: feature_importances_ within IMPORTANCE_ATOL of sklearn's
    ref_feature_importances on the deterministic tier — NOT an exact match,
    see module docstring / SPEC.md spec_revision 2."""
    d = np.load(fixture_path(fixture))
    clf = _det_classifier().fit(d["X"], d["y"])

    got = np.asarray(clf.feature_importances_, dtype=np.float64)
    ref = d["ref_feature_importances"].astype(np.float64)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=IMPORTANCE_ATOL, rtol=0.0)


@pytest.mark.parametrize("fixture", RF_CLS_FIXTURES)
@requires_f64
def test_random_forest_classifier_feature_importances_sums_to_one(fixture):
    """RF-IMP-02: feature_importances_ is a length-n_features array of
    non-negative values summing to 1."""
    d = np.load(fixture_path(fixture))
    clf = _det_classifier().fit(d["X"], d["y"])

    got = np.asarray(clf.feature_importances_, dtype=np.float64)
    assert got.shape == (d["X"].shape[1],)
    assert (got >= 0.0).all()
    assert np.isclose(got.sum(), 1.0, atol=1e-4)


@pytest.mark.parametrize("fixture", RF_CLS_FIXTURES)
@requires_f64
def test_random_forest_classifier_oob_score_statistical_band(fixture):
    """RF-OOB-02: oob_score_ (oob_score=True, bootstrap=True) is within
    OOB_MARGIN of sklearn's ref_oob_score on the statistical tier."""
    d = np.load(fixture_path(fixture))
    clf = _stat_classifier(oob_score=True, bootstrap=True).fit(d["X"], d["y"])

    got = float(clf.oob_score_)
    ref = float(d["ref_oob_score"][0])
    assert abs(got - ref) <= OOB_MARGIN, (
        f"oob_score_: got {got}, sklearn {ref}, margin {OOB_MARGIN}"
    )


def test_random_forest_classifier_oob_score_false_raises_attribute_error():
    """RF-OOB-02: reading oob_score_ raises AttributeError (sklearn parity,
    NOT a silent None) when the estimator was fitted with oob_score=False
    (the default)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(20, 3)).astype(np.float32)
    y = (X[:, 0] > 0.0).astype(np.float32)
    clf = mlrs.RandomForestClassifier(n_estimators=4, max_depth=3, seed=42)
    clf.fit(X, y)
    with pytest.raises(AttributeError):
        _ = clf.oob_score_


def test_random_forest_classifier_oob_true_bootstrap_false_raises_value_error():
    """RF-OOB-01/02: oob_score=True, bootstrap=False raises ValueError at
    fit() (BuildError::OobRequiresBootstrap mapped through build_err_to_py)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(20, 3)).astype(np.float32)
    y = (X[:, 0] > 0.0).astype(np.float32)
    clf = mlrs.RandomForestClassifier(
        oob_score=True, bootstrap=False, n_estimators=4, max_depth=3
    )
    with pytest.raises(ValueError):
        clf.fit(X, y)


# ---------------------------------------------------------------------------
# TASK-15 — PY-ENS-02, RF-IMP-02, RF-OOB-02: RandomForestRegressor
#
# Replays the committed ``rf_reg_{f32,f64}_seed42.npz`` fixtures (produced by
# ``scripts/gen_oracle.py``; the SAME fixtures already gated at the Rust
# layer by ``crates/mlrs-algos/tests/random_forest_regressor_test.rs``,
# TASK-01/03/04/05/07) through the FULL ``mlrs.RandomForestRegressor``
# Python binding path — mirrors TASK-14's classifier section exactly, minus
# ``predict_proba``/``classes_`` (not applicable to a regressor), plus
# R²/PRED_TOL in place of exact-label/accuracy comparisons.
#
# Unlike the classifier, no ``max_features=None`` deviation is needed here:
# ``RandomForestRegressor``'s ``ensemble.py`` default IS already ``1.0``
# (sklearn's own regressor default is "all features"), so the deterministic
# tier's zero-RNG ``mf == d`` precondition (``sample_features``,
# ``crates/mlrs-backend/src/prims/random_forest.rs``) is reproduced by
# passing ``max_features=1.0`` explicitly, same as the classifier section.
# ---------------------------------------------------------------------------

RF_REG_FIXTURES = ["rf_reg_f32_seed42", "rf_reg_f64_seed42"]

# Deterministic-tier train-prediction tolerance — mirrors
# random_forest_regressor_test.rs::PRED_TOL (single f32/f64 constant there;
# this module keeps the existing dtype-branching _atol() helper for
# consistency with the classifier section's predict_proba comparison).
PRED_TOL = 1e-5

# Statistical-tier held-out R² margin — mirrors
# random_forest_regressor_test.rs::R2_MARGIN.
R2_MARGIN = 0.05

# Deterministic-tier feature_importances_ tolerance — mirrors
# random_forest_regressor_test.rs::IMPORTANCE_TOL / IMPORTANCE_TOL_F32
# (SPEC.md spec_revision 2, TASK-03 Green-time resolution — same root cause
# and same numeric value as the classifier's IMPORTANCE_ATOL above, kept as
# an independently-named constant per this file's own established
# per-estimator-independence convention, TASK-14's Risk note).
REG_IMPORTANCE_ATOL = 0.05

# Statistical-tier oob_score_ margin — mirrors
# random_forest_regressor_test.rs::OOB_MARGIN / OOB_MARGIN_F32 (TASK-07).
# Independently named/tuned from the classifier's OOB_MARGIN above (both
# happen to be 0.10, same as the Rust regressor test's own independently-
# tuned constant pair) — tune both together with
# random_forest_regressor_test.rs, see that file's own Green-time-observed-
# divergence doc-comment before ever widening either.
REG_OOB_MARGIN = 0.10


def _det_regressor(**kw):
    """Deterministic-tier construction, mirroring
    random_forest_regressor_test.rs::check_deterministic_tier's
    `.n_estimators(2).bootstrap(false).max_features(MaxFeatures::All)
    .max_depth(RF_DET_MAX_DEPTH)` builder chain."""
    return mlrs.RandomForestRegressor(
        n_estimators=RF_DET_N_ESTIMATORS,
        max_depth=RF_DET_MAX_DEPTH,
        bootstrap=False,
        max_features=1.0,
        seed=42,
        **kw,
    )


def _stat_regressor(**kw):
    """Statistical-tier construction, mirroring
    random_forest_regressor_test.rs::check_statistical_tier's
    `.n_estimators(RF_STAT_N_ESTIMATORS).max_depth(RF_STAT_MAX_DEPTH)`
    (bootstrap/max_features left at their sklearn-parity defaults)."""
    return mlrs.RandomForestRegressor(
        n_estimators=RF_STAT_N_ESTIMATORS,
        max_depth=RF_STAT_MAX_DEPTH,
        seed=42,
        **kw,
    )


@pytest.mark.parametrize("fixture", RF_REG_FIXTURES)
@requires_f64
def test_random_forest_regressor_deterministic(fixture):
    """PY-ENS-02: deterministic-tier .fit(X,y).predict(X) matches
    det_pred_train within PRED_TOL (1e-5 f64 / 1e-4 f32, via the existing
    _atol() dtype branch)."""
    d = np.load(fixture_path(fixture))
    reg = _det_regressor().fit(d["X"], d["y"])

    pred = np.asarray(reg.predict(d["X"]), dtype=np.float64)
    ref = d["det_pred_train"].astype(np.float64)
    assert np.allclose(pred, ref, atol=_atol(fixture), rtol=0.0)


@pytest.mark.parametrize("fixture", RF_REG_FIXTURES)
@requires_f64
def test_random_forest_regressor_statistical(fixture):
    """PY-ENS-02: statistical-tier held-out R² within R2_MARGIN of
    stat_r2_test."""
    d = np.load(fixture_path(fixture))
    reg = _stat_regressor().fit(d["X"], d["y"])

    pred = np.asarray(reg.predict(d["Xq"]), dtype=np.float64)
    yq = d["yq"].astype(np.float64)
    ss_res = float(np.sum((yq - pred) ** 2))
    ss_tot = float(np.sum((yq - yq.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    sk_r2 = float(d["stat_r2_test"][0])
    assert r2 >= sk_r2 - R2_MARGIN, (
        f"held-out R2 {r2} below sklearn {sk_r2} - {R2_MARGIN}"
    )


def test_random_forest_regressor_max_features_invalid_raises():
    """PY-ENS-02: an unrecognized max_features string raises ValueError (same
    parse/raise shape as the classifier — see
    test_random_forest_classifier_max_features_invalid_raises)."""
    X = np.zeros((4, 2), dtype=np.float32)
    y = np.zeros((4,), dtype=np.float32)
    reg = mlrs.RandomForestRegressor(max_features="bogus")
    with pytest.raises(ValueError):
        reg.fit(X, y)


def test_random_forest_regressor_not_fitted_raises():
    """PY-ENS-02/RF-IMP-02/RF-OOB-02: predict / feature_importances_ /
    oob_score_ before fit all raise the project's standard NotFittedError
    (mirrors the classifier's own not-fitted test, minus predict_proba/
    classes_, which do not exist on the regressor)."""
    reg = mlrs.RandomForestRegressor()
    X = np.zeros((1, 5), dtype=np.float32)
    with pytest.raises(NotFittedError):
        reg.predict(X)
    with pytest.raises(NotFittedError):
        _ = reg.feature_importances_
    with pytest.raises(NotFittedError):
        _ = reg.oob_score_


@pytest.mark.parametrize("fixture", RF_REG_FIXTURES)
@requires_f64
def test_random_forest_regressor_feature_importances_close(fixture):
    """RF-IMP-02: feature_importances_ within REG_IMPORTANCE_ATOL of
    sklearn's ref_feature_importances on the deterministic tier — NOT an
    exact match, see module docstring / SPEC.md spec_revision 2."""
    d = np.load(fixture_path(fixture))
    reg = _det_regressor().fit(d["X"], d["y"])

    got = np.asarray(reg.feature_importances_, dtype=np.float64)
    ref = d["ref_feature_importances"].astype(np.float64)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=REG_IMPORTANCE_ATOL, rtol=0.0)


@pytest.mark.parametrize("fixture", RF_REG_FIXTURES)
@requires_f64
def test_random_forest_regressor_feature_importances_sums_to_one(fixture):
    """RF-IMP-02: feature_importances_ is a length-n_features array of
    non-negative values summing to 1."""
    d = np.load(fixture_path(fixture))
    reg = _det_regressor().fit(d["X"], d["y"])

    got = np.asarray(reg.feature_importances_, dtype=np.float64)
    assert got.shape == (d["X"].shape[1],)
    assert (got >= 0.0).all()
    assert np.isclose(got.sum(), 1.0, atol=1e-4)


@pytest.mark.parametrize("fixture", RF_REG_FIXTURES)
@requires_f64
def test_random_forest_regressor_oob_score_statistical_band(fixture):
    """RF-OOB-02: oob_score_ (oob_score=True, bootstrap=True) is within
    REG_OOB_MARGIN of sklearn's ref_oob_score on the statistical tier."""
    d = np.load(fixture_path(fixture))
    reg = _stat_regressor(oob_score=True, bootstrap=True).fit(d["X"], d["y"])

    got = float(reg.oob_score_)
    ref = float(d["ref_oob_score"][0])
    assert abs(got - ref) <= REG_OOB_MARGIN, (
        f"oob_score_: got {got}, sklearn {ref}, margin {REG_OOB_MARGIN}"
    )


def test_random_forest_regressor_oob_score_false_raises_attribute_error():
    """RF-OOB-02: reading oob_score_ raises AttributeError (sklearn parity,
    NOT a silent None) when the estimator was fitted with oob_score=False
    (the default)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(20, 3)).astype(np.float32)
    y = X[:, 0].astype(np.float32)
    reg = mlrs.RandomForestRegressor(n_estimators=4, max_depth=3, seed=42)
    reg.fit(X, y)
    with pytest.raises(AttributeError):
        _ = reg.oob_score_


def test_random_forest_regressor_oob_true_bootstrap_false_raises_value_error():
    """RF-OOB-01/02: oob_score=True, bootstrap=False raises ValueError at
    fit() (BuildError::OobRequiresBootstrap mapped through build_err_to_py)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(20, 3)).astype(np.float32)
    y = X[:, 0].astype(np.float32)
    reg = mlrs.RandomForestRegressor(
        oob_score=True, bootstrap=False, n_estimators=4, max_depth=3
    )
    with pytest.raises(ValueError):
        reg.fit(X, y)


# ---------------------------------------------------------------------------
# TASK-24 — PY-ENS-03/04: HistGradientBoostingClassifier/Regressor oracle
# replay, GATED on a clean `git status` for the in-flight HGB algos churn
# (`crates/mlrs-backend/src/prims/hist_gradient_boosting.rs`,
# `crates/mlrs-kernels/src/gbt.rs`, `scripts/gen_oracle.py`, and all four
# `tests/fixtures/hgb_{cls,reg}_{f32,f64}_seed42.npz`).
#
# Step 0 (mandatory per PLAN.md TASK-24/TASK-17) — `git status --short`
# re-run FRESH at this task's own execution time (2026-07-18, NOT reusing
# TASK-17's 2026-07-17 snapshot):
#
#     git status --short -- \
#       crates/mlrs-backend/src/prims/hist_gradient_boosting.rs \
#       crates/mlrs-kernels/src/gbt.rs scripts/gen_oracle.py \
#       tests/fixtures/hgb_cls_f32_seed42.npz tests/fixtures/hgb_cls_f64_seed42.npz \
#       tests/fixtures/hgb_reg_f32_seed42.npz tests/fixtures/hgb_reg_f64_seed42.npz
#
#     ->  M crates/mlrs-backend/src/prims/hist_gradient_boosting.rs
#         M crates/mlrs-kernels/src/gbt.rs
#         M scripts/gen_oracle.py
#         M tests/fixtures/hgb_cls_f32_seed42.npz
#         M tests/fixtures/hgb_cls_f64_seed42.npz
#         M tests/fixtures/hgb_reg_f32_seed42.npz
#         M tests/fixtures/hgb_reg_f64_seed42.npz
#
# STILL DIRTY (all 7 paths `M`) — identical to TASK-17's finding, PLAN-CHECK's
# 3-pass re-confirmation, and research.md's original 2026-07-17 discovery.
# Per TASK-17's documented mechanism (PLAN.md "Resolved planning decisions" +
# TASK-17's own TDD Sequence), TASK-24 therefore takes the DIRTY branch: the
# FULL test suite below is written structurally (every assertion present,
# correctly shaped, mirroring TASK-14/15's RF oracle-replay rigor one-for-
# one), but ONLY the deterministic-tier EXACT-MATCH assertions
# (`test_hgb_classifier_deterministic_multiclass`,
# `test_hgb_classifier_deterministic_binary`,
# `test_hgb_regressor_deterministic`) are `@pytest.mark.xfail(strict=False)`
# -marked. The statistical-tier band assertions and every structural
# assertion (not-fitted, invalid `n_bins`) are NOT xfailed — they do not
# depend on HGB fixture freshness, per TASK-17/TASK-24's own explicit
# instruction not to blanket-xfail the whole file.
#
# NOTE (observed at this task's own Green time, recorded per the "note XPASS
# prominently" instruction): a manual, throwaway numeric probe (not part of
# this committed suite) found the CURRENT dirty-state deterministic-tier
# fixtures already replay within the existing Rust-test tolerances
# (`PROBA_TOL_F64=1e-5`/`PROBA_TOL_F32=1e-4`, `PRED_TOL_F64=1e-5`/
# `PRED_TOL_F32=1e-4`) through this Python binding path — i.e. the
# `xfail`-marked tests below are LIKELY to report `XPASS`, not `XFAIL`, when
# actually run. This is NOT treated as grounds to un-xfail here: per
# TASK-17/TASK-24's own explicit guardrail, an `XPASS` is a *signal* the
# churn may have settled, not proof — un-xfailing requires a human/orchestrator
# re-check of `git status` at commit time, not an executor's own numeric
# probe overriding the dirty `git status` gate. `strict=False` ensures an
# `XPASS` here reports as a non-fatal `XPASS` in the suite summary rather
# than a hard failure, exactly as designed.
# ---------------------------------------------------------------------------

HGB_CLS_FIXTURES = ["hgb_cls_f32_seed42", "hgb_cls_f64_seed42"]
HGB_REG_FIXTURES = ["hgb_reg_f32_seed42", "hgb_reg_f64_seed42"]

# Deterministic-tier construction — mirrors
# hist_gradient_boosting_{classifier,regressor}_test.rs::det_builder /
# check_deterministic_tier's builder chain (`n_bins=255`, NOT the class
# default 64 — SPEC §5 PY-ENS-03/04's explicit note, PLAN.md Risk 2).
HGB_DET_KW = dict(
    max_iter=20,
    learning_rate=0.1,
    max_depth=6,
    n_bins=255,
    min_samples_leaf=5,
    l2_regularization=0.0,
)

# Statistical-tier held-out margins — mirror
# hist_gradient_boosting_classifier_test.rs::ACC_MARGIN /
# hist_gradient_boosting_regressor_test.rs::R2_MARGIN.
HGB_ACC_MARGIN = 0.05
HGB_R2_MARGIN = 0.05

_HGB_XFAIL_REASON = (
    "HGB algos churn in flight -- see .planning/plans/py-ensemble/PLAN.md "
    "TASK-17/TASK-24; crates/mlrs-backend/src/prims/hist_gradient_boosting.rs, "
    "crates/mlrs-kernels/src/gbt.rs, scripts/gen_oracle.py, and all four "
    "tests/fixtures/hgb_{cls,reg}_{f32,f64}_seed42.npz are uncommitted "
    "(git status --short re-run fresh 2026-07-18 at TASK-24's own Green "
    "time: all 7 paths still 'M', identical to TASK-17's finding)."
)


def _hgb_classifier(**kw):
    return mlrs.HistGradientBoostingClassifier(**kw)


def _hgb_regressor(**kw):
    return mlrs.HistGradientBoostingRegressor(**kw)


@pytest.mark.xfail(reason=_HGB_XFAIL_REASON, strict=False)
@pytest.mark.parametrize("fixture", HGB_CLS_FIXTURES)
@requires_f64
def test_hgb_classifier_deterministic_multiclass(fixture):
    """PY-ENS-03: deterministic-tier (n_bins=255 override) 3-class
    .fit(X,y).predict(X) matches det_pred_train exactly; .predict_proba(X)
    matches det_proba_train within 1e-5 (f64) / 1e-4 (f32).

    XFAIL (strict=False): HGB algos churn dirty per TASK-17/TASK-24's own
    git-status gate re-check — see module docstring above."""
    d = np.load(fixture_path(fixture))
    clf = _hgb_classifier(**HGB_DET_KW).fit(d["X"], d["y"])

    pred = np.asarray(clf.predict(d["X"])).astype(np.int64).ravel()
    ref = d["det_pred_train"].astype(np.int64).ravel()
    assert np.array_equal(pred, ref)

    proba = np.asarray(clf.predict_proba(d["X"]), dtype=np.float64)
    refp = d["det_proba_train"].astype(np.float64)
    assert np.allclose(proba, refp, atol=_atol(fixture), rtol=0.0)


@pytest.mark.xfail(reason=_HGB_XFAIL_REASON, strict=False)
@pytest.mark.parametrize("fixture", HGB_CLS_FIXTURES)
@requires_f64
def test_hgb_classifier_deterministic_binary(fixture):
    """PY-ENS-03: deterministic-tier (n_bins=255 override) binary
    (y_bin/sigmoid) .fit(X,y_bin).predict(X) matches det_pred_bin_train
    exactly; .predict_proba(X) matches det_proba_bin_train within tolerance.

    XFAIL (strict=False): HGB algos churn dirty — see module docstring."""
    d = np.load(fixture_path(fixture))
    clf = _hgb_classifier(**HGB_DET_KW).fit(d["X"], d["y_bin"])

    pred = np.asarray(clf.predict(d["X"])).astype(np.int64).ravel()
    ref = d["det_pred_bin_train"].astype(np.int64).ravel()
    assert np.array_equal(pred, ref)

    proba = np.asarray(clf.predict_proba(d["X"]), dtype=np.float64)
    refp = d["det_proba_bin_train"].astype(np.float64)
    assert np.allclose(proba, refp, atol=_atol(fixture), rtol=0.0)


@pytest.mark.parametrize("fixture", HGB_CLS_FIXTURES)
@requires_f64
def test_hgb_classifier_statistical(fixture):
    """PY-ENS-03: statistical-tier (class defaults, noisy labels) held-out
    accuracy within HGB_ACC_MARGIN of stat_acc_test. NOT xfailed — a
    statistical band, not an exact-match assertion; does not depend on the
    HGB fixture-freshness gate."""
    d = np.load(fixture_path(fixture))
    clf = _hgb_classifier().fit(d["X"], d["y_noisy"])

    pred = np.asarray(clf.predict(d["Xq"])).astype(np.int64).ravel()
    yq = d["yq"].astype(np.int64).ravel()
    acc = float(np.mean(pred == yq))
    sk_acc = float(d["stat_acc_test"][0])
    assert acc >= sk_acc - HGB_ACC_MARGIN, (
        f"held-out accuracy {acc} below sklearn {sk_acc} - {HGB_ACC_MARGIN}"
    )


def test_hgb_classifier_not_fitted_raises():
    """PY-ENS-03: predict / predict_proba before fit raise the project's
    standard NotFittedError. Structural — not xfailed."""
    clf = mlrs.HistGradientBoostingClassifier()
    X = np.zeros((1, 5), dtype=np.float32)
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.predict_proba(X)


def test_hgb_classifier_invalid_n_bins_raises():
    """PY-ENS-03: n_bins outside 2..=256 raises ValueError at fit()
    (BuildError mapped through build_err_to_py). Structural — not xfailed."""
    X = np.zeros((4, 2), dtype=np.float32)
    y = np.zeros((4,), dtype=np.float32)
    clf = mlrs.HistGradientBoostingClassifier(n_bins=257)
    with pytest.raises(ValueError):
        clf.fit(X, y)


@pytest.mark.xfail(reason=_HGB_XFAIL_REASON, strict=False)
@pytest.mark.parametrize("fixture", HGB_REG_FIXTURES)
@requires_f64
def test_hgb_regressor_deterministic(fixture):
    """PY-ENS-04: deterministic-tier (n_bins=255 override)
    .fit(X,y).predict(X) matches det_pred_train within tolerance (1e-5 f64 /
    1e-4 f32).

    XFAIL (strict=False): HGB algos churn dirty — see module docstring."""
    d = np.load(fixture_path(fixture))
    reg = _hgb_regressor(**HGB_DET_KW).fit(d["X"], d["y"])

    pred = np.asarray(reg.predict(d["X"]), dtype=np.float64)
    ref = d["det_pred_train"].astype(np.float64)
    assert np.allclose(pred, ref, atol=_atol(fixture), rtol=0.0)


@pytest.mark.parametrize("fixture", HGB_REG_FIXTURES)
@requires_f64
def test_hgb_regressor_statistical(fixture):
    """PY-ENS-04: statistical-tier (class defaults) held-out R2 within
    HGB_R2_MARGIN of stat_r2_test. NOT xfailed — a statistical band, not an
    exact-match assertion; does not depend on the HGB fixture-freshness
    gate."""
    d = np.load(fixture_path(fixture))
    reg = _hgb_regressor().fit(d["X"], d["y"])

    pred = np.asarray(reg.predict(d["Xq"]), dtype=np.float64)
    yq = d["yq"].astype(np.float64)
    ss_res = float(np.sum((yq - pred) ** 2))
    ss_tot = float(np.sum((yq - yq.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    sk_r2 = float(d["stat_r2_test"][0])
    assert r2 >= sk_r2 - HGB_R2_MARGIN, (
        f"held-out R2 {r2} below sklearn {sk_r2} - {HGB_R2_MARGIN}"
    )


def test_hgb_regressor_not_fitted_raises():
    """PY-ENS-04: predict before fit raises the project's standard
    NotFittedError. Structural — not xfailed."""
    reg = mlrs.HistGradientBoostingRegressor()
    X = np.zeros((1, 5), dtype=np.float32)
    with pytest.raises(NotFittedError):
        reg.predict(X)


def test_hgb_regressor_invalid_n_bins_raises():
    """PY-ENS-04: n_bins outside 2..=256 raises ValueError at fit()
    (BuildError mapped through build_err_to_py). Structural — not xfailed."""
    X = np.zeros((4, 2), dtype=np.float32)
    y = np.zeros((4,), dtype=np.float32)
    reg = mlrs.HistGradientBoostingRegressor(n_bins=257)
    with pytest.raises(ValueError):
        reg.fit(X, y)
