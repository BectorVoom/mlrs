"""Phase-10 SGD / linear-SVM Python smoke test (SGDSVM-01..04 — PY-06 share).

Proves the FFI path + dtype dispatch + the construction-time ValueError behavior
(D-05/D-09) end to end for the four Phase-10 estimators, through the REAL binding
surface this plan delivers: the low-level ``mlrs._mlrs`` extension classes
``MBSGDClassifier``, ``MBSGDRegressor``, ``LinearSVC`` and ``LinearSVR`` (the
pure-Python sklearn-shim wrappers are Phase-11 PY-06 scope, not this incremental
wrapper share).

This is a SMOKE test, NOT a numerical oracle: the ≤1e-5 / documented-tolerance
numerical contract is already gated by the Rust oracle tests in Plans 10-03
(``mbsgd_*_test.rs``) and 10-04 (``linear_sv*_test.rs``). Here we assert:

  * each estimator's ``fit(X, y).predict(...)`` returns the right SHAPE and a
    sane prediction (classifiers split the two clusters; the classifier
    ``predict_proba`` rows are in [0,1] and sum to 1); and
  * a bad enum string (``loss='bogus'``) raises a Python ``ValueError`` at the
    first ``fit`` (the D-05/D-09 construction-time behavior — mlrs surfaces the
    sklearn construction error at ``fit`` since the ``Unfit`` arm stores the raw
    strings until then),

exercising BOTH the f32 and f64 dtype-dispatch arms. The f64 case is gated behind
``mlrs._mlrs.backend_supports_f64()`` so it skips-with-reason on an f64-incapable
backend (rocm), mirroring the ``capability.rs::skip_f64_with_log`` precedent.

Run via the shipped maturin-develop py-test flow (build the ``mlrs`` extension,
then ``pytest`` this file). The whole module is import-guarded so it skips
cleanly (never errors at collection) if the extension or pyarrow is unavailable.
"""

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
_mlrs = pytest.importorskip("mlrs._mlrs")


_F64_OK = bool(_mlrs.backend_supports_f64())
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


def _toy_binary(dtype):
    """Two well-separated clusters at ∓2 on feature 0 → a clean ±1 split."""
    X = np.array(
        [
            [-2.0, 0.1],
            [-1.9, -0.2],
            [-2.1, 0.0],
            [-1.8, 0.2],
            [2.0, -0.1],
            [1.9, 0.2],
            [2.1, 0.0],
            [1.8, -0.2],
        ],
        dtype=dtype,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=dtype)
    return X, y


def _toy_regression(dtype):
    """A smooth linear target y = 3*x0 - x1 over the same well-spread design."""
    X, _ = _toy_binary(dtype)
    y = (3.0 * X[:, 0] - X[:, 1]).astype(dtype)
    return X, y


# --- MBSGDClassifier: fit -> predict_labels + predict_proba -----------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_mbsgd_classifier_predict(dtype):
    """SGDSVM-01: PyMBSGDClassifier fit->predict_labels + predict_proba (FFI)."""
    X, y = _toy_binary(dtype)
    rows, cols = X.shape

    est = _mlrs.MBSGDClassifier(
        loss="log_loss", learning_rate="constant", eta0=0.1, max_iter=50, seed=0
    )
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    assert est.is_fitted()
    assert est.dtype() == ("f32" if dtype == np.float32 else "f64")

    labels = np.asarray(est.predict_labels(_arrow(X, dtype), rows, cols))
    assert labels.shape == (rows,)
    assert set(np.unique(labels)).issubset({0, 1})
    # Clean split: the two clusters land in different classes.
    assert labels[0] == labels[1]
    assert labels[0] != labels[4]

    if dtype == np.float32:
        proba = np.asarray(est.predict_proba_f32(_arrow(X, dtype), rows, cols))
    else:
        proba = np.asarray(est.predict_proba_f64(_arrow(X, dtype), rows, cols))
    proba = proba.reshape(rows, 2)
    assert proba.shape == (rows, 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


# --- MBSGDRegressor: fit -> predict -----------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_mbsgd_regressor_predict(dtype):
    """SGDSVM-02: PyMBSGDRegressor fit->predict (FFI), both dtypes."""
    X, y = _toy_regression(dtype)
    rows, cols = X.shape

    est = _mlrs.MBSGDRegressor(
        learning_rate="constant", eta0=0.01, max_iter=100, seed=0
    )
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    assert est.is_fitted()

    if dtype == np.float32:
        pred = np.asarray(est.predict_f32(_arrow(X, dtype), rows, cols))
    else:
        pred = np.asarray(est.predict_f64(_arrow(X, dtype), rows, cols))
    assert pred.shape == (rows,)
    assert np.all(np.isfinite(pred))
    # The −2 cluster predicts below the +2 cluster (the linear target separates).
    assert pred[0] < pred[4]


# --- LinearSVC: fit -> predict_labels ---------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_linear_svc_predict(dtype):
    """SGDSVM-03: PyLinearSVC fit->predict_labels (FFI), both dtypes."""
    X, y = _toy_binary(dtype)
    rows, cols = X.shape

    est = _mlrs.LinearSVC(C=1.0)
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    assert est.is_fitted()

    labels = np.asarray(est.predict_labels(_arrow(X, dtype), rows, cols))
    assert labels.shape == (rows,)
    assert set(np.unique(labels)).issubset({0, 1})
    assert labels[0] == labels[1]
    assert labels[0] != labels[4]


# --- LinearSVR: fit -> predict ----------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_linear_svr_predict(dtype):
    """SGDSVM-04: PyLinearSVR fit->predict (FFI), both dtypes."""
    X, y = _toy_regression(dtype)
    rows, cols = X.shape

    est = _mlrs.LinearSVR(C=1.0, epsilon=0.0)
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    assert est.is_fitted()

    if dtype == np.float32:
        pred = np.asarray(est.predict_f32(_arrow(X, dtype), rows, cols))
    else:
        pred = np.asarray(est.predict_f64(_arrow(X, dtype), rows, cols))
    assert pred.shape == (rows,)
    assert np.all(np.isfinite(pred))
    assert pred[0] < pred[4]


# --- Construction-time ValueError (D-05/D-09) -------------------------------


@pytest.mark.parametrize(
    "ctor, kwargs",
    [
        (_mlrs.MBSGDClassifier, {"loss": "bogus"}),
        (_mlrs.MBSGDRegressor, {"loss": "bogus"}),
        (_mlrs.LinearSVC, {"penalty": "bogus"}),
        (_mlrs.LinearSVR, {"loss": "bogus"}),
    ],
    ids=["classifier", "regressor", "svc", "svr"],
)
def test_bad_enum_string_raises_value_error(ctor, kwargs):
    """A bogus enum string surfaces as a Python ValueError at the first fit
    (D-05/D-09 — the Unfit arm stores the raw string until fit, then the
    TryFrom/build path maps it through build_err_to_py to PyValueError)."""
    X, y = _toy_binary(np.float32)
    rows, cols = X.shape
    est = ctor(**kwargs)
    with pytest.raises(ValueError):
        est.fit(_arrow(X, np.float32), _arrow(y, np.float32), rows, cols)
