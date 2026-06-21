"""Phase-8 kernel-family Python smoke test (KERNEL-01 / KERNEL-02 — PY-06 share).

Proves the FFI path + dtype dispatch + the new ``score_samples`` exposure end to
end for the two Phase-8 estimators, through the REAL binding surface this plan
delivers: the low-level ``mlrs._mlrs`` extension classes ``KernelRidge`` and
``KernelDensity`` (the pure-Python sklearn-shim wrappers for the kernel family
are Plan-04 / Phase-11 scope, not this incremental wrapper share).

This is a SMOKE test, NOT a numerical oracle: the ≤1e-5 / documented-tolerance
numerical contract is already gated by the Rust oracle tests in Plans 08-03
(``kernel_ridge_test.rs``) and 08-04 (``kernel_density_test.rs``). Here we assert:

  * ``KernelRidge.fit(X, y).predict(Xq)`` returns the right SHAPE and tracks a
    small sklearn ``KernelRidge`` reference within a loose smoke band; and
  * ``KernelDensity.fit(X).score_samples(Q)`` returns a length-``rows(Q)`` vector
    of finite log-densities tracking sklearn ``KernelDensity`` within a loose
    band,

exercising BOTH the f32 and f64 dtype-dispatch arms. The f64 case is gated behind
the backend-capability flag (``mlrs._mlrs.backend_supports_f64()``) so it
skips-with-reason on an f64-incapable backend (rocm), mirroring the v1
``capability.rs::skip_f64_with_log`` precedent (STATE.md [06-05]).

Run via the shipped maturin-develop py-test flow (build the ``mlrs`` extension,
then ``pytest`` this file). The whole module is import-guarded so it skips
cleanly (never errors at collection) if the extension or pyarrow is unavailable.
"""

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
_mlrs = pytest.importorskip("mlrs._mlrs")
sklearn_kr = pytest.importorskip("sklearn.kernel_ridge")
sklearn_kd = pytest.importorskip("sklearn.neighbors")


_F64_OK = bool(_mlrs.backend_supports_f64())
requires_f64 = pytest.mark.skipif(
    not _F64_OK,
    reason="backend does not support f64 (mlrs._mlrs.backend_supports_f64() is False)",
)


def _arrow(a, dtype):
    """Fresh-contiguous row-major 1-D pyarrow float array (offset 0, no parent
    aliasing — the Rust bridge HARD-REJECTS sliced/offset arrays, mirrors
    ``mlrs._io.normalize_X``)."""
    flat = np.ascontiguousarray(a, dtype=dtype).ravel(order="C")
    at = pa.float32() if dtype == np.float32 else pa.float64()
    return pa.array(flat, type=at)


def _toy_regression(dtype):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((12, 3)).astype(dtype)
    # A smooth target so the rbf kernel ridge fit is well-conditioned.
    y = (np.sin(X[:, 0]) + 0.5 * X[:, 1] - X[:, 2]).astype(dtype)
    Xq = rng.standard_normal((5, 3)).astype(dtype)
    return X, y, Xq


def _toy_density(dtype):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((20, 2)).astype(dtype)
    Q = rng.standard_normal((6, 2)).astype(dtype)
    return X, Q


# --- KernelRidge: fit/predict shape + smoke-band match ----------------------

@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        pytest.param(np.float64, marks=requires_f64),
    ],
    ids=["f32", "f64"],
)
def test_kernel_ridge_predict(dtype):
    """KERNEL-01: PyKernelRidge fit->predict through the FFI path, both dtypes."""
    X, y, Xq = _toy_regression(dtype)
    rows, cols = X.shape
    nq = Xq.shape[0]

    est = _mlrs.KernelRidge(kernel="rbf", alpha=1.0, gamma=0.5)
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols, 1)
    assert est.is_fitted()
    assert est.dtype() == ("f32" if dtype == np.float32 else "f64")

    if dtype == np.float32:
        pred = np.asarray(est.predict_f32(_arrow(Xq, dtype), nq, cols))
    else:
        pred = np.asarray(est.predict_f64(_arrow(Xq, dtype), nq, cols))

    # Shape contract: single-target -> length n_query.
    assert pred.shape == (nq,)
    assert np.all(np.isfinite(pred))

    # Smoke band vs sklearn (loose — the strict oracle is the Rust test).
    ref = sklearn_kr.KernelRidge(kernel="rbf", alpha=1.0, gamma=0.5)
    ref.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    expected = ref.predict(np.asarray(Xq, dtype=np.float64))
    atol = 1e-4 if dtype == np.float64 else 1e-2
    assert np.allclose(pred.astype(np.float64), expected, atol=atol, rtol=1e-3)


# --- KernelDensity: fit/score_samples shape + finiteness + smoke band -------

@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        pytest.param(np.float64, marks=requires_f64),
    ],
    ids=["f32", "f64"],
)
def test_kernel_density_score_samples(dtype):
    """KERNEL-02: PyKernelDensity fit->score_samples (the one new method)."""
    X, Q = _toy_density(dtype)
    rows, cols = X.shape
    nq = Q.shape[0]

    est = _mlrs.KernelDensity(kernel="gaussian", bandwidth=1.0)
    est.fit(_arrow(X, dtype), rows, cols)
    assert est.is_fitted()
    assert est.bandwidth_() == pytest.approx(1.0)

    if dtype == np.float32:
        logd = np.asarray(est.score_samples_f32(_arrow(Q, dtype), nq, cols))
    else:
        logd = np.asarray(est.score_samples_f64(_arrow(Q, dtype), nq, cols))

    # Shape contract (D-12): length n_query of finite log-densities.
    assert logd.shape == (nq,)
    assert np.all(np.isfinite(logd))

    # Smoke band vs sklearn KernelDensity log-densities.
    ref = sklearn_kd.KernelDensity(kernel="gaussian", bandwidth=1.0)
    ref.fit(np.asarray(X, dtype=np.float64))
    expected = ref.score_samples(np.asarray(Q, dtype=np.float64))
    atol = 1e-4 if dtype == np.float64 else 1e-2
    assert np.allclose(logd.astype(np.float64), expected, atol=atol, rtol=1e-3)
