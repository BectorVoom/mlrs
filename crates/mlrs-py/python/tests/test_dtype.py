"""dtype dispatch + f64-capability gating (PY-03/PY-04: f32/f64 + backend caps).

Wave-0 COLLECTING stub: ``importorskip("mlrs")`` + the ``requires_f64`` marker
so it collects green pre-wrapper and auto-skips f64 cases on an f64-incapable
backend (rocm). Plan 04-06 wire real assertions that:
  - f32 and f64 inputs each dispatch to the matching monomorphization (D-06),
  - integer/list inputs default to f64 where supported and to f32 on an
    f64-incapable backend (D-05, via mlrs.backend_supports_f64()), and
  - an explicit f64 request on a rocm wheel raises the D-04 error rather than
    silently downcasting.
"""

import pytest

from conftest import requires_f64

# Req: PY-03 (f32/f64 dtype dispatch) + PY-04 (backend capability gating).


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dtype_dispatch(dtype):
    """PY-03: f32/f64 inputs each reach the matching code path."""
    mlrs = pytest.importorskip("mlrs")  # noqa: F841  (wired in Plan 04)
    pytest.xfail("dtype dispatch lands in Plan 04")


@requires_f64
def test_f64_supported_path():
    """PY-04: f64 path runs only where the backend advertises f64 support."""
    mlrs = pytest.importorskip("mlrs")  # noqa: F841  (wired in Plan 04)
    pytest.xfail("f64 capability path lands in Plan 04")
