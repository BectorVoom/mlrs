"""ForestInference oracle harness (FIL-01, Phase 20): import a REAL fitted
sklearn RandomForest and compare mlrs device inference against sklearn.

Leaf routing is exact by construction (the ``next_up`` threshold bump maps
sklearn's ``<=`` onto the device ``<``), so predictions differ only by the
reduction association: ``predict_proba``/regression means gate at ≤1e-5
(f64) / ≤1e-4 (f32); classifier labels must be EXACTLY equal.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import mlrs
from mlrs import _io
from conftest import requires_f64


def _data(seed=0, n=120, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y_cls = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 0.5).astype(int)
    y_reg = X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    Xq = rng.normal(size=(40, d))
    return X, y_cls, y_reg, Xq


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@requires_f64
def test_fil_classifier_matches_sklearn(dtype):
    if dtype == np.float64 and not mlrs.backend_supports_f64:
        pytest.skip("f64 unsupported on this backend")
    X, y, _, Xq = _data()
    sk = RandomForestClassifier(
        n_estimators=12, max_depth=6, random_state=0
    ).fit(X, y)
    fil = mlrs.ForestInference.load_from_sklearn(sk, dtype=dtype)
    assert fil.n_trees == 12

    Xq_t = Xq.astype(dtype)
    atol = 1e-5 if dtype == np.float64 else 1e-4
    got = fil.predict_proba(Xq_t)
    ref = sk.predict_proba(Xq_t.astype(np.float64))
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=atol), (
        f"max abs diff {np.abs(got - ref).max()}"
    )
    labels = fil.predict(Xq_t)
    assert np.array_equal(labels, sk.predict(Xq_t.astype(np.float64)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@requires_f64
def test_fil_regressor_matches_sklearn(dtype):
    if dtype == np.float64 and not mlrs.backend_supports_f64:
        pytest.skip("f64 unsupported on this backend")
    X, _, y, Xq = _data(seed=3)
    sk = RandomForestRegressor(
        n_estimators=10, max_depth=6, random_state=0
    ).fit(X, y)
    fil = mlrs.ForestInference.load_from_sklearn(sk, dtype=dtype)

    Xq_t = Xq.astype(dtype)
    atol = 1e-5 if dtype == np.float64 else 1e-4
    got = fil.predict(Xq_t)
    ref = sk.predict(Xq_t.astype(np.float64))
    assert np.allclose(got, ref, atol=atol), (
        f"max abs diff {np.abs(got - ref).max()}"
    )


def test_fil_rejects_unfitted_and_deep():
    with pytest.raises(ValueError, match="estimators_"):
        mlrs.ForestInference.load_from_sklearn(RandomForestClassifier())
    # A depth > 16 source tree raises a clear error.
    X, y, _, _ = _data(n=400)
    deep = RandomForestClassifier(
        n_estimators=1, max_depth=None, min_samples_leaf=1, random_state=0
    ).fit(np.random.default_rng(1).normal(size=(4000, 3)), 
          np.arange(4000) % 97)
    depths = [e.tree_.max_depth for e in deep.estimators_]
    if max(depths) > 16:
        with pytest.raises(ValueError):
            mlrs.ForestInference.load_from_sklearn(deep)
    else:
        pytest.skip("source forest not deep enough to exercise the cap")


def test_fil_shap_values_matches_shap_treeexplainer():
    """SHAP-01, Python path: ForestInference.shap_values() replays
    shap.TreeExplainer on the SAME sklearn model, ≤1e-5, plus additive
    efficiency vs predict_proba."""
    shap = pytest.importorskip("shap")
    X, y, _, Xq = _data(seed=5, n=80, d=4)
    sk = RandomForestClassifier(n_estimators=6, max_depth=4, random_state=0).fit(X, y)
    fil = mlrs.ForestInference.load_from_sklearn(sk, dtype=np.float32)

    expl = shap.TreeExplainer(sk)
    ref_sv = np.asarray(expl.shap_values(Xq))
    ref_ev = np.asarray(expl.expected_value)

    phi, ev = fil.shap_values(Xq.astype(np.float32))
    assert np.allclose(ev, ref_ev, atol=1e-5)
    assert np.allclose(phi, ref_sv, atol=1e-4)

    proba = fil.predict_proba(Xq.astype(np.float32))
    assert np.allclose(phi.sum(axis=1) + ev, proba, atol=1e-4)


def test_fil_shap_values_requires_cover():
    """A raw-array import without node_sample_weight raises on shap_values."""
    trees = mlrs._mlrs.ForestInference.load_from_arrays(
        [1, -1, -1], [2, -1, -1], [0, -2, -2], [0.0, 0.0, 0.0],
        [0.0, -1.0, 1.0], [], [3], 1, "regressor", 1, "f32",
    )
    xa, rows, cols = _io.normalize_X(np.array([[0.5]], dtype=np.float32))
    with pytest.raises(ValueError, match="cover"):
        trees.shap_values(xa, rows, cols)


def test_native_rf_shap_values_additive_efficiency():
    """RandomForestClassifier.shap_values() (self-consistency gate — no
    external oracle for mlrs's own split policy)."""
    X, y, _, Xq = _data(seed=7, n=60, d=3)
    est = mlrs.RandomForestClassifier(n_estimators=5, max_depth=4, seed=1).fit(X, y)
    phi, ev = est.shap_values(X, Xq)
    proba = est.predict_proba(Xq)
    assert np.allclose(phi.sum(axis=1) + ev, proba, atol=1e-4)


def test_fil_rejects_multioutput_regressor():
    """A multi-output RandomForestRegressor must be rejected, not silently
    imported keeping only output 0 (review finding)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    Y = np.column_stack([X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]])  # 2 targets
    m = RandomForestRegressor(n_estimators=3, max_depth=4, random_state=0).fit(X, Y)
    assert m.n_outputs_ == 2
    with pytest.raises(ValueError, match="multi-output"):
        mlrs.ForestInference.load_from_sklearn(m)


def test_native_rf_shap_values_cross_dtype_query():
    """RandomForest.shap_values must coerce a query whose dtype differs from
    the fit dtype (float32 fit, float64 query) instead of raising an opaque
    'unsupported dtype' from the Rust downcast (review finding)."""
    X, y, _, Xq = _data(seed=11, n=50, d=3)
    est = mlrs.RandomForestClassifier(n_estimators=4, max_depth=3, seed=1).fit(
        X.astype(np.float32), y
    )
    assert est._mlrs_obj.dtype() == "f32"
    # Query + reference given as float64 — must not raise, and additive
    # efficiency must still hold.
    phi, ev = est.shap_values(X.astype(np.float64), Xq.astype(np.float64))
    proba = est.predict_proba(Xq.astype(np.float64))
    assert np.all(np.isfinite(phi))
    assert np.allclose(phi.sum(axis=1) + ev, proba, atol=1e-4)
