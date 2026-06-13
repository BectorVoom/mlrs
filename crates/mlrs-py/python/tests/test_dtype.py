"""dtype dispatch + f64-capability gating + GIL release (PY-05/PY-03/D-04/D-05).

Real assertions through the compiled ``_mlrs`` extension:

  1. **dtype preservation (D-05/PY-05).** ``float32`` input -> ``float32`` fitted
     attributes / predictions; ``float64`` input -> ``float64`` — each dtype
     reaches its own monomorphization and the egress mirrors it.
  2. **integer-input default dtype (D-05 / RESEARCH Pitfall 5).** Non-float input
     defaults to ``float64`` on an f64-capable backend and ``float32`` on an
     incapable one, decided via ``mlrs.backend_supports_f64()``.
  3. **f64-on-incapable-backend raise (D-04 / T-06-15).**
     ``test_f64_on_incapable_backend_raises`` runs ONLY where f64 is unsupported
     (e.g. the rocm wheel) and asserts a clear ``ValueError`` naming the backend
     and float64 — never a silent downcast.
  4. **GIL release (PY-03 / RESEARCH Pitfall 6).** ``py.detach`` must release the
     GIL during device compute so other Python threads progress. The smoke runs a
     compute-heavy ``.fit`` in a worker thread while the main thread advances a
     pure-Python counter; a held GIL would freeze the counter at ~0.

Device-threading note (cpu backend): the CubeCL CPU runtime's memory handles are
thread-affine — a device buffer allocated under the thread that first initialized
the global client cannot be read back from a different thread. mlrs's documented
v1 model is a SINGLE process-global device client (true cross-thread parallelism
is out of v1 scope; ``crates/mlrs-py/src/lib.rs`` concurrency note). The GIL-release
smoke therefore runs in a FRESH subprocess so the worker thread is the first to
touch the device (binding the client to that worker thread); the main thread then
proves GIL release by its concurrent pure-Python progress. Running it in-process
after the dtype tests (which init the client on the MAIN thread) would hit the
cpu cross-thread read-back limitation, not a GIL-release failure.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import mlrs
from conftest import fixture_path, requires_f64

_F64 = bool(mlrs.backend_supports_f64())


@pytest.mark.parametrize(
    "np_dtype",
    [
        np.float32,
        pytest.param(np.float64, marks=requires_f64),
    ],
)
def test_dtype_preserved(np_dtype):
    """PY-05/D-05: f32-in -> f32-out, f64-in -> f64-out through the binding path."""
    d = np.load(fixture_path("ridge_f64_seed42"))
    X = np.ascontiguousarray(d["X"], dtype=np_dtype)
    y = np.ascontiguousarray(d["y"], dtype=np_dtype)
    est = mlrs.Ridge(alpha=1.0, fit_intercept=True).fit(X, y)
    assert np.asarray(est.coef_).dtype == np_dtype
    assert np.asarray(est.predict(X)).dtype == np_dtype


def test_integer_input_default_dtype():
    """D-05 / Pitfall 5: integer input defaults to f64 where supported, else f32.

    The default is decided by ``mlrs.backend_supports_f64()`` — never an
    arbitrary downcast.
    """
    d = np.load(fixture_path("ridge_f64_seed42"))
    X_int = (d["X"] * 100).astype(np.int64)
    y_int = (d["y"] * 100).astype(np.int64)
    est = mlrs.Ridge(alpha=1.0, fit_intercept=True).fit(X_int, y_int)
    expected = np.float64 if _F64 else np.float32
    assert np.asarray(est.coef_).dtype == expected


@pytest.mark.skipif(
    _F64,
    reason="backend supports f64 — the D-04 raise only fires on an f64-incapable "
    "backend (e.g. the rocm wheel)",
)
def test_f64_on_incapable_backend_raises():
    """D-04 / T-06-15: explicit f64 on an f64-incapable backend raises, no downcast.

    Runs only on a wheel where ``backend_supports_f64()`` is False (rocm). Asserts
    a clear ``ValueError`` mentioning the backend and float64 rather than a silent
    downcast to f32.
    """
    d = np.load(fixture_path("ridge_f64_seed42"))
    X = np.ascontiguousarray(d["X"], dtype=np.float64)
    y = np.ascontiguousarray(d["y"], dtype=np.float64)
    with pytest.raises(ValueError) as excinfo:
        mlrs.Ridge(alpha=1.0).fit(X, y)
    msg = str(excinfo.value).lower()
    assert "float64" in msg or "f64" in msg


# The GIL-release smoke runs in this fresh child interpreter so the WORKER thread
# is the first to touch the device (the cpu client is thread-affine — see module
# docstring). The child computes a heavy ``.fit`` in a worker thread while the main
# thread spins a pure-Python counter; a released GIL lets the counter advance into
# the millions, a held GIL freezes it near zero. Exit codes: 0 = GIL released
# (counter high), 4 = GIL held (counter low), 2 = deadlock, 3 = worker error.
_GIL_CHILD = textwrap.dedent(
    """
    import sys, threading, time
    import numpy as np
    import mlrs

    f64 = bool(mlrs.backend_supports_f64())
    rng = np.random.default_rng(0)
    dt = np.float64 if f64 else np.float32
    X = rng.standard_normal((4000, 200)).astype(dt)
    y = rng.standard_normal(4000).astype(dt)

    done = threading.Event()
    err = {}

    def worker():
        try:
            mlrs.Lasso(alpha=0.01, max_iter=2000, tol=1e-9).fit(X, y)
        except BaseException as exc:  # PanicException is not an Exception
            err["exc"] = repr(exc)
        finally:
            done.set()

    counter = 0
    t = threading.Thread(target=worker)
    t.start()
    deadline = time.perf_counter() + 60.0
    while not done.is_set() and time.perf_counter() < deadline:
        counter += 1
    t.join(timeout=60.0)

    if t.is_alive():
        print("DEADLOCK", flush=True)
        sys.exit(2)
    if "exc" in err:
        print("WORKER_ERROR", err["exc"], flush=True)
        sys.exit(3)
    print("COUNTER", counter, flush=True)
    sys.exit(0 if counter > 10_000 else 4)
    """
)


def test_gil_released_during_compute():
    """PY-03 / Pitfall 6: ``py.detach`` releases the GIL during device compute.

    Spawns a fresh child interpreter (so the worker thread owns the cpu client)
    that runs a compute-heavy ``.fit`` in a worker thread while its main thread
    spins a pure-Python counter. A released GIL drives the counter into the
    millions (child exit 0); a held GIL would freeze it near zero (exit 4).
    """
    env = dict(os.environ)
    # Propagate the in-tree editable build's preload (mimalloc static-TLS
    # workaround) to the child if the parent test run set it.
    proc = subprocess.run(
        [sys.executable, "-c", _GIL_CHILD],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    detail = f"stdout={proc.stdout!r} stderr={proc.stderr[-500:]!r}"
    assert proc.returncode != 2, f"worker .fit deadlocked — {detail}"
    assert proc.returncode != 3, f"worker .fit raised — {detail}"
    assert proc.returncode != 4, (
        "main thread barely advanced during the worker compute — the GIL "
        f"appears NOT to have been released by py.detach. {detail}"
    )
    assert proc.returncode == 0, f"GIL-release child failed — {detail}"
