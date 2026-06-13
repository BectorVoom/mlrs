"""D-08 driver-absent import probe — fail closed with ImportError, never abort.

Mitigates T-06-16 (DoS): a wheel whose backend runtime/driver is absent must
raise a clean Python `ImportError` at `import mlrs`, NOT segfault/abort the host
process. The Rust `#[pymodule] _mlrs` init wraps the cubecl driver handshake in
`std::panic::catch_unwind` and translates a caught panic into a `PyImportError`
(the wheel keeps `panic = "unwind"`); the pure-Python `_load_ext` converts a
genuinely-unimportable `.so` into the same clear `ImportError` (recursion-safe,
Plan 06-05). This module asserts both halves at the Python boundary:

  1. driver PRESENT  -> `import mlrs` succeeds and the capability surface works;
  2. driver ABSENT   -> the loader path raises a clear `ImportError` whose
     message points the user at the backend wheels (asserted by simulating an
     unimportable `_mlrs`; the LIVE driver-absent path on a real foreign-backend
     wheel is the Task-3 human-verify item, since a driver cannot be removed from
     this host).

`grep ImportError` here is the key-link the plan asserts.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


def test_import_succeeds_with_driver_present():
    """Driver present (cpu gate): `import mlrs` works and capability is live."""
    mlrs = pytest.importorskip("mlrs")
    # The lazily-resolved capability surface forces the `_mlrs` import (and thus
    # the D-08 probe). On the cpu gate the driver is present, so this returns a
    # boolean rather than raising.
    assert isinstance(mlrs.backend_supports_f64(), bool)


def test_loader_raises_clear_importerror_when_ext_unimportable():
    """`_load_ext` raises a CLEAR ImportError when `_mlrs` cannot be imported.

    Simulates the driver-absent / missing-`.so` path WITHOUT removing the real
    driver: monkeypatch `importlib.import_module` so resolving `mlrs._mlrs`
    raises `ImportError`, then assert `_load_ext` re-raises a clear message that
    names the backend wheels (and does NOT recurse — Plan 06-05 fix).
    """
    pytest.importorskip("mlrs")
    import mlrs

    real_import_module = importlib.import_module

    def _fake_import_module(name, *args, **kwargs):
        if name == "mlrs._mlrs":
            raise ImportError("simulated: backend driver/runtime absent")
        return real_import_module(name, *args, **kwargs)

    orig = importlib.import_module
    importlib.import_module = _fake_import_module
    try:
        with pytest.raises(ImportError) as exc:
            mlrs._load_ext()
    finally:
        importlib.import_module = orig

    msg = str(exc.value)
    # The clear, actionable message (not a bare ModuleNotFoundError / RecursionError).
    assert "backend" in msg.lower()
    assert "mlrs-cpu" in msg or "mlrs-cuda" in msg or "backend wheel" in msg.lower()


def test_driver_absent_import_is_importerror_not_abort_in_subprocess():
    """A failed `_mlrs` import surfaces as ImportError in a CHILD interpreter.

    Runs in a fresh subprocess so a hypothetical abort would show as a non-zero
    signal exit, not an `ImportError` traceback. We force the failure by shadowing
    `mlrs._mlrs` with a module that raises on import, proving the package
    `__getattr__`/`_load_ext` path fails CLOSED (ImportError) rather than killing
    the interpreter. This is the in-environment backstop for T-06-16; the LIVE
    foreign-driver path is the Task-3 human-verify.
    """
    pytest.importorskip("mlrs")
    code = (
        "import importlib\n"
        "import mlrs\n"
        "real = importlib.import_module\n"
        "def fake(name, *a, **k):\n"
        "    if name == 'mlrs._mlrs':\n"
        "        raise ImportError('simulated absent driver')\n"
        "    return real(name, *a, **k)\n"
        "importlib.import_module = fake\n"
        "try:\n"
        "    mlrs._load_ext()\n"
        "except ImportError as e:\n"
        "    print('IMPORTERROR_OK')\n"
        "else:\n"
        "    print('NO_ERROR')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    # The child must exit 0 (clean ImportError handled) — a crash/abort would
    # produce a negative returncode (killed by signal) instead.
    assert out.returncode == 0, (
        f"child interpreter aborted instead of raising ImportError "
        f"(rc={out.returncode}); stderr:\n{out.stderr}"
    )
    assert "IMPORTERROR_OK" in out.stdout, out.stdout
