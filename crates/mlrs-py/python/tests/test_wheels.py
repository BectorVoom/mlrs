"""Per-backend wheel build + distribution-name/abi3 assertions (PY-04/D-07/D-09).

Closes criterion 4's wheel half: each of the four backends builds a wheel under
its DISTINCT distribution name (`mlrs_cpu` / `mlrs_wgpu` / `mlrs_cuda` /
`mlrs_rocm`) carrying the `cp312-abi3` tag (one wheel covers Python >=3.12,
D-09), and every wheel exposes the SAME `import mlrs` namespace because
`module-name = "mlrs._mlrs"` is constant across the templates (D-07).

Build invocation (standardizes Plan 06-05 item 2 — `maturin build -m <pyproject>`
does NOT work on maturin 1.14, which wants a Cargo manifest): the helper
`_wheel_build.build_wheel(backend)` copies the per-backend pyproject to a
temporary repo-root `pyproject.toml`, runs `maturin build --release` from the
repo root, and removes the temp file. See `_wheel_build.py`.

Environment honesty (CLAUDE.md backend gates):
  * cpu  — built + a FRESH-subprocess `import mlrs` is asserted (primary gate).
  * wgpu — built; import smoke runs only if a wgpu adapter is present.
  * rocm — built; import smoke runs only on a ROCm/HIP host (f32 there).
  * cuda — built (COMPILE-ONLY here, no CUDA driver); its import smoke is the
           Task-3 human-verify item, NOT asserted in this environment.

These builds are slow (release codegen across the backend runtime). They are
opt-in via `MLRS_BUILD_WHEELS=1` so the default `pytest` run stays fast; CI / the
phase gate sets the env var. Each backend can be limited via
`MLRS_WHEEL_BACKENDS=cpu,wgpu`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile

import pytest

import _wheel_build as wb

pytestmark = pytest.mark.skipif(
    os.environ.get("MLRS_BUILD_WHEELS") != "1",
    reason="wheel builds are slow; set MLRS_BUILD_WHEELS=1 to run "
    "(the phase gate / CI does).",
)


def _selected_backends():
    sel = os.environ.get("MLRS_WHEEL_BACKENDS")
    if sel:
        return tuple(b.strip() for b in sel.split(",") if b.strip())
    return wb.BACKENDS


def _build_timeout():
    return int(os.environ.get("MLRS_WHEEL_TIMEOUT", "1800"))


def _abi3_tagged(wheel_name: str) -> bool:
    """True iff the wheel filename carries the abi3 / cp312-abi3 tag (D-09)."""
    return "abi3" in wheel_name


def _wheel_contains_ext(wheel_path) -> bool:
    """The built wheel ships the `mlrs/_mlrs*.so` extension (module-name D-07)."""
    with zipfile.ZipFile(wheel_path) as zf:
        return any(
            n.startswith("mlrs/_mlrs") and n.endswith((".so", ".pyd"))
            for n in zf.namelist()
        )


@pytest.mark.parametrize("backend", _selected_backends())
def test_wheel_builds_with_dist_name_and_abi3(backend):
    """Each backend wheel builds as `mlrs_<backend>-*` with an abi3 tag (PY-04).

    Asserts T-06-17: the filename matches the backend distribution name AND
    carries the abi3/cp312-abi3 tag, so a wheel can never ship to the wrong
    backend under the wrong name or as a non-abi3 (per-minor) build.
    """
    proc = wb.build_wheel(backend, timeout=_build_timeout())
    assert proc.returncode == 0, (
        f"maturin build for {backend} failed (rc={proc.returncode})\n"
        f"STDOUT:\n{proc.stdout[-2000:]}\n\nSTDERR:\n{proc.stderr[-2000:]}"
    )

    wheel = wb.latest_wheel(backend)
    assert wheel is not None, (
        f"no mlrs_{backend}-*.whl in {wb.WHEELS_DIR} after a successful build"
    )
    # Distinct distribution name (T-06-17).
    assert wheel.name.startswith(f"mlrs_{backend}-"), wheel.name
    # abi3 / cp312-abi3 single-3.12+ wheel tag (D-09).
    assert _abi3_tagged(wheel.name), (
        f"{wheel.name} is not abi3-tagged (expected a cp312-abi3 wheel)"
    )
    assert "cp312" in wheel.name, f"{wheel.name} is not a cp312 (>=3.12) wheel"
    # The constant `module-name = mlrs._mlrs` ships the extension inside `mlrs/`
    # so every backend wheel exposes `import mlrs` (D-07).
    assert _wheel_contains_ext(wheel), (
        f"{wheel.name} does not contain mlrs/_mlrs*.so (module-name D-07)"
    )


def test_cpu_wheel_imports_as_mlrs(tmp_path):
    """The cpu wheel installs into a fresh venv and `import mlrs` succeeds.

    Primary gate (CLAUDE.md: cpu is testable). Proves the constant
    `import mlrs` namespace AND that the local_dynamic_tls allocator fix lets
    the wheel `dlopen` with NO `LD_PRELOAD` (Plan 06-05 item 1, closed here).
    """
    if "cpu" not in _selected_backends():
        pytest.skip("cpu not in MLRS_WHEEL_BACKENDS")

    # Ensure a fresh cpu wheel exists.
    if wb.latest_wheel("cpu") is None:
        proc = wb.build_wheel("cpu", timeout=_build_timeout())
        assert proc.returncode == 0, proc.stderr[-2000:]
    wheel = wb.latest_wheel("cpu")
    assert wheel is not None

    # Fresh, isolated venv so the editable install does not mask the wheel.
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv)], check=True, timeout=120
    )
    pip = venv / "bin" / "pip"
    py = venv / "bin" / "python"
    subprocess.run(
        [str(pip), "install", "-q", "--disable-pip-version-check", str(wheel)],
        check=True,
        timeout=600,
    )
    # Import in the FRESH interpreter, with NO LD_PRELOAD set.
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    out = subprocess.run(
        [
            str(py),
            "-c",
            "import mlrs; print(mlrs.__name__); "
            "print(bool(mlrs.backend_supports_f64()))",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, (
        f"`import mlrs` from the installed cpu wheel failed:\n{out.stderr}"
    )
    assert out.stdout.splitlines()[0] == "mlrs", out.stdout
    # cpu supports f64.
    assert out.stdout.splitlines()[1] == "True", out.stdout


def test_rocm_wheel_imports_as_mlrs_when_runtime_present(tmp_path):
    """The rocm wheel imports as `mlrs` on a ROCm/HIP host (f32 gate).

    rocm is a RUNNABLE GPU gate (per project notes: gfx1100 / ROCm 7.1.1 runs
    f32; f64 is unsupported there). Skipped unless a rocm wheel exists AND the
    host can load it. Asserts the constant `import mlrs` namespace from the rocm
    distribution and that `backend_supports_f64()` is FALSE (no silent f64 on a
    backend that cannot do it — backstops T-06-15 at the wheel layer). cuda is
    NOT importable here (no driver) — its import smoke is the Task-3 human-verify.
    """
    if "rocm" not in _selected_backends():
        pytest.skip("rocm not in MLRS_WHEEL_BACKENDS")
    wheel = wb.latest_wheel("rocm")
    if wheel is None:
        pytest.skip("no rocm wheel built (build it with MLRS_BUILD_WHEELS=1)")

    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv)], check=True, timeout=120
    )
    pip = venv / "bin" / "pip"
    py = venv / "bin" / "python"
    inst = subprocess.run(
        [str(pip), "install", "-q", "--disable-pip-version-check", str(wheel)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if inst.returncode != 0:
        pytest.skip(f"rocm wheel not installable on this host:\n{inst.stderr}")

    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    out = subprocess.run(
        [
            str(py),
            "-c",
            "import mlrs; print(mlrs.__name__); "
            "print(bool(mlrs.backend_supports_f64()))",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if out.returncode != 0:
        # No ROCm runtime on this host -> the D-08 probe raises ImportError; that
        # is the driver-absent path (covered by test_import_probe), not a wheel
        # defect. Skip the positive import smoke here.
        pytest.skip(
            "rocm runtime not available on this host (import raised); "
            f"stderr:\n{out.stderr}"
        )
    assert out.stdout.splitlines()[0] == "mlrs", out.stdout
    # rocm does NOT support f64 (gfx1100 / cubecl-cpp 0.10): must report False.
    assert out.stdout.splitlines()[1] == "False", out.stdout
