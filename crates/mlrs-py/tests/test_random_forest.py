"""PY-ENS-01/RF-IMP-02/RF-OOB-02 — RandomForestClassifier Python live-FFI
smoke (TASK-08).

Proves the FFI path + dtype dispatch + `max_features` string/int/float/None
parsing + the `oob_score`/`bootstrap` builder cross-check + the not-fitted
guard, through the REAL binding surface this task delivers: the low-level
``mlrs._mlrs.RandomForestClassifier`` extension class, mirroring
``test_naive_bayes.py``'s structure (PY-06 precedent).

This is a SMOKE test, NOT a numerical oracle: the sklearn-parity numeric
contract (``atol=0.05`` deterministic-tier / ``ACC_MARGIN``/``OOB_MARGIN``
statistical-tier bands) is already gated by the Rust oracle tests in
``crates/mlrs-algos/tests/random_forest_classifier_test.rs`` (TASK-01..07).
Here we assert, ACROSS THE FFI BOUNDARY: construction with sklearn-named
kwargs; ``fit(X, y)`` then ``predict_labels``/``predict_proba`` shape/sum-to-1;
``feature_importances_`` (RF-IMP-02) is a length-``n_features`` array summing
to 1; ``oob_score_`` (RF-OOB-02) is ``None`` when ``oob_score=False`` and a
float when ``oob_score=True``; a bogus ``max_features`` string and
``oob_score=True, bootstrap=False`` both raise a Python ``ValueError``; and
calling ``predict_labels``/``predict_proba``/accessors before ``fit`` raises
(the ``not_fitted``-mapped ``PyValueError``, sklearn's ``NotFittedError``
shape at the Rust boundary — the Python shim, TASK-11, re-raises the concrete
``sklearn.exceptions.NotFittedError``).

NOTE (TASK-08 execution evidence): ``mlrs._mlrs.RandomForestClassifier`` is
NOT YET REGISTERED on the compiled extension as of this task — registration
is TASK-10's scope (Wave 4a, `crates/mlrs-py/src/lib.rs`). This file is
created now (matching this task's own Files list) so it is ready to exercise
the binding surface TASK-08 just landed; it becomes fully green once TASK-10
registers the class and a wheel is built (``maturin develop``). Every test
below is import-guarded via ``pytest.importorskip`` (the existing
``test_naive_bayes.py`` precedent) so it skips cleanly rather than erroring at
collection when the extension/pyarrow is unavailable, and each test's own
``hasattr`` guard below additionally skips-with-reason if the class is not
yet registered (rather than failing) — this is NOT a numeric-tolerance
weakening, it is the same "skip cleanly if the environment/build state isn't
ready yet" convention `test_naive_bayes.py`/`pyclass_smoke_test.rs` already
established (10-05/08-05 precedent).

Run via the shipped maturin-develop py-test flow (build the ``mlrs``
extension, then ``pytest`` this file).
"""

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
_mlrs = pytest.importorskip("mlrs._mlrs")

_HAS_RF = hasattr(_mlrs, "RandomForestClassifier")
requires_rf = pytest.mark.skipif(
    not _HAS_RF,
    reason="mlrs._mlrs.RandomForestClassifier not yet registered (TASK-10, Wave 4a)",
)

_F64_OK = bool(_mlrs.backend_supports_f64()) if hasattr(_mlrs, "backend_supports_f64") else False
requires_f64 = pytest.mark.skipif(
    not _F64_OK,
    reason="backend does not support f64 (mlrs._mlrs.backend_supports_f64() is False)",
)

_DTYPES = [
    np.float32,
    pytest.param(np.float64, marks=requires_f64),
]
_DTYPE_IDS = ["f32", "f64"]


def _arrow(a, dtype):
    """Fresh-contiguous row-major 1-D pyarrow float array (offset 0, no parent
    aliasing — the Rust bridge HARD-REJECTS sliced/offset arrays)."""
    flat = np.ascontiguousarray(a, dtype=dtype).ravel(order="C")
    at = pa.float32() if dtype == np.float32 else pa.float64()
    return pa.array(flat, type=at)


def _toy_forest_data(dtype):
    """40-row x 2-feature dataset: feature 0 perfectly separates 2 classes via
    a threshold, feature 1 is uniform noise (mirrors the Rust-layer oracle
    tests' dominant-feature synthetic geometry, TASK-01/02)."""
    rng = np.random.default_rng(42)
    n = 40
    f0 = rng.uniform(-1.0, 1.0, size=n)
    f1 = rng.uniform(-1.0, 1.0, size=n)
    y = (f0 > 0.0).astype(dtype)
    X = np.stack([f0, f1], axis=1).astype(dtype)
    return X, y


def _proba(est, X, dtype, rows, cols):
    if dtype == np.float32:
        flat = np.asarray(est.predict_proba_f32(_arrow(X, dtype), rows, cols))
    else:
        flat = np.asarray(est.predict_proba_f64(_arrow(X, dtype), rows, cols))
    return flat.reshape(rows, -1)


@requires_rf
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_random_forest_classifier_fit_predict(dtype):
    """PY-ENS-01: fit -> predict_labels/predict_proba, RF-IMP-02
    feature_importances_ sums to 1, RF-OOB-02 oob_score_ is None by default."""
    X, y = _toy_forest_data(dtype)
    rows, cols = X.shape
    est = _mlrs.RandomForestClassifier(n_estimators=8, max_depth=4, seed=42)
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    assert est.is_fitted()
    assert est.dtype() == ("f32" if dtype == np.float32 else "f64")

    labels = np.asarray(est.predict_labels(_arrow(X, dtype), rows, cols))
    assert labels.shape == (rows,)

    proba = _proba(est, X, dtype, rows, cols)
    assert proba.shape[0] == rows
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    classes = np.asarray(est.classes_())
    assert len(classes) == 2

    imp_fn = est.feature_importances_f32 if dtype == np.float32 else est.feature_importances_f64
    importances = np.asarray(imp_fn())
    assert importances.shape == (cols,)
    assert np.isclose(importances.sum(), 1.0, atol=1e-6)

    oob_fn = est.oob_score_f32 if dtype == np.float32 else est.oob_score_f64
    assert oob_fn() is None, "oob_score_ must be None when oob_score=False (default)"


@requires_rf
def test_random_forest_classifier_oob_score_true():
    """RF-OOB-02: oob_score=True, bootstrap=True (default) -> oob_score_f32()
    returns a float in [0, 1]."""
    dtype = np.float32
    X, y = _toy_forest_data(dtype)
    rows, cols = X.shape
    est = _mlrs.RandomForestClassifier(n_estimators=8, max_depth=4, oob_score=True, seed=42)
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    score = est.oob_score_f32()
    assert score is not None
    assert 0.0 <= score <= 1.0


@requires_rf
def test_predict_before_fit_raises():
    """PY-ENS-01: predict_labels before fit raises (the not_fitted-mapped
    PyValueError, sklearn's NotFittedError shape at the Rust boundary)."""
    dtype = np.float32
    X, _ = _toy_forest_data(dtype)
    rows, cols = X.shape
    est = _mlrs.RandomForestClassifier()
    with pytest.raises(ValueError):
        est.predict_labels(_arrow(X, dtype), rows, cols)


@requires_rf
def test_feature_importances_before_fit_raises():
    """RF-IMP-02: feature_importances_f32 before fit raises NotFittedError."""
    est = _mlrs.RandomForestClassifier()
    with pytest.raises(ValueError):
        est.feature_importances_f32()


@requires_rf
def test_max_features_bogus_string_raises_value_error():
    """A bogus max_features string surfaces as a Python ValueError at fit
    (the Unfit arm stores the parsed MaxFeaturesArg; a bad string is rejected
    inside the #[new] constructor itself — either way, ValueError, not a
    panic or a silent fallback)."""
    with pytest.raises(ValueError):
        _mlrs.RandomForestClassifier(max_features="bogus")


@requires_rf
def test_oob_score_true_without_bootstrap_raises_value_error():
    """RF-OOB-01/02: oob_score=True, bootstrap=False raises ValueError at fit
    (BuildError::OobRequiresBootstrap mapped through build_err_to_py)."""
    dtype = np.float32
    X, y = _toy_forest_data(dtype)
    rows, cols = X.shape
    est = _mlrs.RandomForestClassifier(oob_score=True, bootstrap=False)
    with pytest.raises(ValueError):
        est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)


@requires_rf
def test_max_features_int_and_float_accepted():
    """max_features accepts a positive int and a (0, 1] float without error
    (the Value/Frac MaxFeaturesArg paths)."""
    dtype = np.float32
    X, y = _toy_forest_data(dtype)
    rows, cols = X.shape
    for mf in (1, 2, 0.5, 1.0):
        est = _mlrs.RandomForestClassifier(n_estimators=4, max_depth=3, max_features=mf)
        est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
        assert est.is_fitted()


@requires_rf
def test_max_features_all_sentinel_and_ffi_none_contract():
    """FFI-level (``_mlrs``) max_features contract (code-review fix): the
    ``"all"`` sentinel string, the float ``1.0``, and the explicit count ``2``
    (== n_features) all mean all-features and produce the identical forest,
    genuinely distinct from ``"sqrt"`` (which floors to 1 of 2 features). At the
    ``_mlrs`` boundary an explicit ``None`` cannot be told apart from an omitted
    argument (PyO3 ``Option``), so it resolves to the classifier's ``"sqrt"``
    default — sklearn's ``None``-means-all parity is provided by the ``mlrs.*``
    SHIM layer and covered by the shim test suite, not here."""
    dtype = np.float32
    X, y = _toy_forest_data(dtype)
    rows, cols = X.shape
    assert cols == 2

    def proba(mf):
        est = _mlrs.RandomForestClassifier(
            n_estimators=8, max_depth=4, max_features=mf, bootstrap=False, seed=42
        )
        est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
        return _proba(est, X, dtype, rows, cols)

    p_all, p_one, p_two, p_sqrt, p_none = (
        proba("all"), proba(1.0), proba(2), proba("sqrt"), proba(None)
    )
    np.testing.assert_allclose(p_all, p_one, atol=1e-6)  # "all" == 1.0
    np.testing.assert_allclose(p_all, p_two, atol=1e-6)  # "all" == count 2 (all features)
    np.testing.assert_allclose(p_none, p_sqrt, atol=1e-6)  # FFI None == "sqrt" default
    assert not np.allclose(p_all, p_sqrt, atol=1e-6), (
        "all-features must differ from sqrt (=1 of 2 features) on this data"
    )


@requires_rf
def test_max_features_fraction_uses_floor_not_ceil():
    """sklearn parity (code-review fix): a float max_features resolves via
    truncation `int(frac * n_features)` (sklearn's own rule), NOT `ceil`. With
    2 features, 0.6 -> floor(1.2)=1 (subsample 1 feature), matching
    max_features=1 and DIFFERING from all-features; the old ceil rule gave 2
    (== all-features)."""
    dtype = np.float32
    X, y = _toy_forest_data(dtype)
    rows, cols = X.shape
    assert cols == 2

    def proba(mf):
        est = _mlrs.RandomForestClassifier(
            n_estimators=8, max_depth=4, max_features=mf, bootstrap=False, seed=42
        )
        est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
        return _proba(est, X, dtype, rows, cols)

    p_frac, p_one, p_all = proba(0.6), proba(1), proba("all")
    np.testing.assert_allclose(p_frac, p_one, atol=1e-6)  # floor(0.6*2) == 1
    assert not np.allclose(p_frac, p_all, atol=1e-6), (
        "0.6 must floor to 1 feature (distinct from all-features); the old "
        "ceil rule gave 2 (== all-features)"
    )


# ===========================================================================
# TASK-09 — RandomForestRegressor (PY-ENS-02, RF-IMP-02, RF-OOB-02).
# ===========================================================================

_HAS_RF_REG = hasattr(_mlrs, "RandomForestRegressor")
requires_rf_reg = pytest.mark.skipif(
    not _HAS_RF_REG,
    reason="mlrs._mlrs.RandomForestRegressor not yet registered (TASK-10, Wave 4a)",
)


def _toy_forest_reg_data(dtype):
    """40-row x 2-feature dataset: feature 0 is strongly (linearly) correlated
    with a continuous target, feature 1 is uniform noise (regressor mirror of
    `_toy_forest_data`, TASK-01/03's synthetic geometry)."""
    rng = np.random.default_rng(42)
    n = 40
    f0 = rng.uniform(-1.0, 1.0, size=n)
    f1 = rng.uniform(-1.0, 1.0, size=n)
    y = (2.0 * f0).astype(dtype)
    X = np.stack([f0, f1], axis=1).astype(dtype)
    return X, y


@requires_rf_reg
def test_regressor_predict_before_fit_raises():
    """PY-ENS-02: predict_f32 before fit raises (the not_fitted-mapped
    PyValueError, sklearn's NotFittedError shape at the Rust boundary)."""
    dtype = np.float32
    X, _ = _toy_forest_reg_data(dtype)
    rows, cols = X.shape
    est = _mlrs.RandomForestRegressor()
    with pytest.raises(ValueError):
        est.predict_f32(_arrow(X, dtype), rows, cols)


@requires_rf_reg
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_random_forest_regressor_fit_predict(dtype):
    """PY-ENS-02: fit -> predict, RF-IMP-02 feature_importances_ sums to 1,
    RF-OOB-02 oob_score_ is None by default."""
    X, y = _toy_forest_reg_data(dtype)
    rows, cols = X.shape
    est = _mlrs.RandomForestRegressor(n_estimators=8, max_depth=4, seed=42)
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    assert est.is_fitted()
    assert est.dtype() == ("f32" if dtype == np.float32 else "f64")

    predict_fn = est.predict_f32 if dtype == np.float32 else est.predict_f64
    preds = np.asarray(predict_fn(_arrow(X, dtype), rows, cols))
    assert preds.shape == (rows,)

    imp_fn = est.feature_importances_f32 if dtype == np.float32 else est.feature_importances_f64
    importances = np.asarray(imp_fn())
    assert importances.shape == (cols,)
    assert np.isclose(importances.sum(), 1.0, atol=1e-6)

    oob_fn = est.oob_score_f32 if dtype == np.float32 else est.oob_score_f64
    assert oob_fn() is None, "oob_score_ must be None when oob_score=False (default)"


@requires_rf_reg
def test_random_forest_regressor_oob_score_true():
    """RF-OOB-02: oob_score=True, bootstrap=True (default) -> oob_score_f32()
    returns a finite float."""
    dtype = np.float32
    X, y = _toy_forest_reg_data(dtype)
    rows, cols = X.shape
    est = _mlrs.RandomForestRegressor(n_estimators=8, max_depth=4, oob_score=True, seed=42)
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    score = est.oob_score_f32()
    assert score is not None
    assert np.isfinite(score)


@requires_rf_reg
def test_regressor_max_features_default_is_all_not_sqrt():
    """PY-ENS-02: the regressor's max_features default is sklearn's "all"
    (mlrs MaxFeatures::All), NOT the classifier's "sqrt" — an indirect check
    (MaxFeatures itself is not Python-visible): construct with no args, fit,
    confirm no ValueError is raised (a bogus/rejected value would raise)."""
    dtype = np.float32
    X, y = _toy_forest_reg_data(dtype)
    rows, cols = X.shape
    est = _mlrs.RandomForestRegressor(n_estimators=4, max_depth=3, seed=42)
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    assert est.is_fitted()


@requires_rf_reg
def test_regressor_feature_importances_before_fit_raises():
    """RF-IMP-02: feature_importances_f32 before fit raises NotFittedError."""
    est = _mlrs.RandomForestRegressor()
    with pytest.raises(ValueError):
        est.feature_importances_f32()


@requires_rf_reg
def test_regressor_max_features_bogus_string_raises_value_error():
    """A bogus max_features string surfaces as a Python ValueError at
    construction time (mirrors the classifier's own test)."""
    with pytest.raises(ValueError):
        _mlrs.RandomForestRegressor(max_features="bogus")


@requires_rf_reg
def test_regressor_oob_score_true_without_bootstrap_raises_value_error():
    """RF-OOB-01/02: oob_score=True, bootstrap=False raises ValueError at fit
    (BuildError::OobRequiresBootstrap mapped through build_err_to_py)."""
    dtype = np.float32
    X, y = _toy_forest_reg_data(dtype)
    rows, cols = X.shape
    est = _mlrs.RandomForestRegressor(oob_score=True, bootstrap=False)
    with pytest.raises(ValueError):
        est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)


# ===========================================================================
# TASK-18 — HistGradientBoostingClassifier (PY-ENS-03, structural).
#
# This is a STRUCTURAL smoke pass, not a numerical oracle: the sklearn-parity
# numeric contract (deterministic-tier n_bins=255 exact-match / statistical-
# tier band) is TASK-24's scope, explicitly GATED on a clean git status for
# hist_gradient_boosting.rs/gbt.rs/gen_oracle.py/hgb_*.npz (TASK-17 found
# these dirty as of this task's own execution — re-verified below). Here we
# assert, across the FFI boundary, only what TASK-18 itself delivers:
# construction with sklearn-named kwargs; fit(X, y) -> predict_labels/
# predict_proba shape/sum-to-1; classes_; the not-fitted guard; and — the
# explicit SPEC §2 non-goal this task's own Completion Criteria calls out —
# the ABSENCE of feature_importances_f32/_f64 and oob_score_f32/_f64 (not
# applicable to HGB, unlike RandomForest).
# ===========================================================================

_HAS_HGB_CLF = hasattr(_mlrs, "HistGradientBoostingClassifier")
requires_hgb_clf = pytest.mark.skipif(
    not _HAS_HGB_CLF,
    reason="mlrs._mlrs.HistGradientBoostingClassifier not yet registered (TASK-20, Wave 10a)",
)


def _toy_hgb_clf_data(dtype):
    """40-row x 2-feature dataset: feature 0 perfectly separates 2 classes via
    a threshold, feature 1 is uniform noise (mirrors `_toy_forest_data`'s own
    synthetic geometry, TASK-08's precedent)."""
    rng = np.random.default_rng(42)
    n = 40
    f0 = rng.uniform(-1.0, 1.0, size=n)
    f1 = rng.uniform(-1.0, 1.0, size=n)
    y = (f0 > 0.0).astype(dtype)
    X = np.stack([f0, f1], axis=1).astype(dtype)
    return X, y


@requires_hgb_clf
def test_hgb_classifier_predict_before_fit_raises():
    """PY-ENS-03: predict_labels before fit raises (the not_fitted-mapped
    PyValueError, sklearn's NotFittedError shape at the Rust boundary) — the
    plan's own named Red test for this task."""
    dtype = np.float32
    X, _ = _toy_hgb_clf_data(dtype)
    rows, cols = X.shape
    est = _mlrs.HistGradientBoostingClassifier()
    with pytest.raises(ValueError):
        est.predict_labels(_arrow(X, dtype), rows, cols)


@requires_hgb_clf
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_hist_gradient_boosting_classifier_fit_predict(dtype):
    """PY-ENS-03 (structural): fit -> predict_labels/predict_proba shape and
    sum-to-1 — no numeric-tolerance assertion against sklearn here (TASK-24's
    gated scope); this proves only that the FFI path/dtype dispatch works."""
    X, y = _toy_hgb_clf_data(dtype)
    rows, cols = X.shape
    est = _mlrs.HistGradientBoostingClassifier(max_iter=10, max_depth=3)
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    assert est.is_fitted()
    assert est.dtype() == ("f32" if dtype == np.float32 else "f64")

    labels = est.predict_labels(_arrow(X, dtype), rows, cols)
    assert len(labels) == rows

    proba = _proba(est, X, dtype, rows, cols)
    assert proba.shape == (rows, len(est.classes_()))
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


@requires_hgb_clf
def test_hgb_classifier_max_iter_and_learning_rate_defaults():
    """PY-ENS-03: constructor defaults (max_iter=100, learning_rate=0.1,
    max_depth=6, n_bins=64 (NOT 255 — the deterministic-tier n_bins=255
    override is a TEST-TIME construction argument, TASK-24's scope, not a
    changed default), l2_regularization=0.0, min_samples_leaf=20) accept a
    no-arg construction without error (indirect check, mirrors
    `test_regressor_max_features_default_is_all_not_sqrt`'s pattern —
    individual hyperparameters are not Python-visible read-back attributes)."""
    est = _mlrs.HistGradientBoostingClassifier()
    assert not est.is_fitted()
    assert est.dtype() is None


@requires_hgb_clf
def test_hgb_classifier_has_no_feature_importances_or_oob_score():
    """SPEC §2 explicit non-goal: sklearn's own HistGradientBoostingClassifier
    exposes neither feature_importances_ nor oob_score_ (boosting is not a
    bagging/OOB scheme) — this task's own Completion Criteria requires
    verifying their ABSENCE, not merely omitting them by accident."""
    est = _mlrs.HistGradientBoostingClassifier()
    assert not hasattr(est, "feature_importances_f32")
    assert not hasattr(est, "feature_importances_f64")
    assert not hasattr(est, "oob_score_f32")
    assert not hasattr(est, "oob_score_f64")


# ===========================================================================
# TASK-19 — HistGradientBoostingRegressor (PY-ENS-04, structural). Mirrors
# TASK-18's HGB classifier section exactly, minus classes_/predict_labels/
# predict_proba, plus a float predict_f32/_f64 (mirrors the RF regressor's
# own TASK-09 predict shape). Same structural-only scope note as TASK-18: the
# sklearn-parity numeric contract is TASK-24's gated scope, not this task's.
# ===========================================================================

_HAS_HGB_REG = hasattr(_mlrs, "HistGradientBoostingRegressor")
requires_hgb_reg = pytest.mark.skipif(
    not _HAS_HGB_REG,
    reason="mlrs._mlrs.HistGradientBoostingRegressor not yet registered (TASK-20, Wave 10a)",
)


def _toy_hgb_reg_data(dtype):
    """40-row x 2-feature dataset: feature 0 is strongly (linearly) correlated
    with a continuous target, feature 1 is uniform noise (mirrors
    `_toy_forest_reg_data`'s own synthetic geometry, TASK-09's precedent)."""
    rng = np.random.default_rng(42)
    n = 40
    f0 = rng.uniform(-1.0, 1.0, size=n)
    f1 = rng.uniform(-1.0, 1.0, size=n)
    y = (2.0 * f0).astype(dtype)
    X = np.stack([f0, f1], axis=1).astype(dtype)
    return X, y


@requires_hgb_reg
def test_hgb_regressor_predict_before_fit_raises():
    """PY-ENS-04: predict_f32 before fit raises (the not_fitted-mapped
    PyValueError, sklearn's NotFittedError shape at the Rust boundary) — the
    plan's own named Red test for this task."""
    dtype = np.float32
    X, _ = _toy_hgb_reg_data(dtype)
    rows, cols = X.shape
    est = _mlrs.HistGradientBoostingRegressor()
    with pytest.raises(ValueError):
        est.predict_f32(_arrow(X, dtype), rows, cols)


@requires_hgb_reg
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_hist_gradient_boosting_regressor_fit_predict(dtype):
    """PY-ENS-04 (structural): fit -> predict — no numeric-tolerance assertion
    against sklearn here (TASK-24's gated scope); this proves only that the
    FFI path/dtype dispatch works."""
    X, y = _toy_hgb_reg_data(dtype)
    rows, cols = X.shape
    est = _mlrs.HistGradientBoostingRegressor(max_iter=10, max_depth=3)
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    assert est.is_fitted()
    assert est.dtype() == ("f32" if dtype == np.float32 else "f64")

    predict_fn = est.predict_f32 if dtype == np.float32 else est.predict_f64
    preds = np.asarray(predict_fn(_arrow(X, dtype), rows, cols))
    assert preds.shape == (rows,)
    assert np.all(np.isfinite(preds))


@requires_hgb_reg
def test_hgb_regressor_max_iter_and_learning_rate_defaults():
    """PY-ENS-04: constructor defaults (max_iter=100, learning_rate=0.1,
    max_depth=6, n_bins=64 (NOT 255 — TASK-24's gated scope), l2_regularization=0.0,
    min_samples_leaf=20) accept a no-arg construction without error (indirect
    check, mirrors `test_hgb_classifier_max_iter_and_learning_rate_defaults`'s
    pattern — individual hyperparameters are not Python-visible read-back
    attributes)."""
    est = _mlrs.HistGradientBoostingRegressor()
    assert not est.is_fitted()
    assert est.dtype() is None


@requires_hgb_reg
def test_hgb_regressor_has_no_feature_importances_or_oob_score():
    """SPEC §2 explicit non-goal: sklearn's own HistGradientBoostingRegressor
    exposes neither feature_importances_ nor oob_score_ (boosting is not a
    bagging/OOB scheme) — this task's own Completion Criteria requires
    verifying their ABSENCE, not merely omitting them by accident (mirrors
    TASK-18's classifier check)."""
    est = _mlrs.HistGradientBoostingRegressor()
    assert not hasattr(est, "feature_importances_f32")
    assert not hasattr(est, "feature_importances_f64")
    assert not hasattr(est, "oob_score_f32")
    assert not hasattr(est, "oob_score_f64")


@requires_hgb_reg
def test_hgb_regressor_has_no_classes_or_predict_proba():
    """PY-ENS-04: the regressor has no classifier-only surface (classes_,
    predict_labels, predict_proba_f32/_f64) — mirrors the RF regressor's own
    absence-of-classifier-surface shape (not explicitly asserted for RF, but
    explicit here since TASK-18's HGB classifier section establishes the
    ABSENCE-check convention for this file's HGB tests)."""
    est = _mlrs.HistGradientBoostingRegressor()
    assert not hasattr(est, "classes_")
    assert not hasattr(est, "predict_labels")
    assert not hasattr(est, "predict_proba_f32")
    assert not hasattr(est, "predict_proba_f64")
