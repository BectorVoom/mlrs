"""Standardized per-backend ``maturin build`` driver (PY-04 / Plan 06-05 item 2).

Plan 06-05 flagged that ``maturin build -m <backend>.pyproject.toml`` does NOT
work on maturin 1.14: ``-m / --manifest-path`` expects a *Cargo* manifest, not a
pyproject. maturin reads ``[tool.maturin]`` from the pyproject in the CURRENT
working directory instead. The four per-backend templates under
``crates/mlrs-py/pyproject/`` carry repo-root-relative paths
(``manifest-path = "crates/mlrs-py/Cargo.toml"``,
``python-source = "crates/mlrs-py/python"``), so the portable invocation is:

  1. copy ``pyproject/<backend>.pyproject.toml`` to the repo-root ``pyproject.toml``
  2. run ``maturin build --release`` from the repo root
  3. remove the temporary root ``pyproject.toml``

This module centralizes that dance so ``test_wheels.py`` (and a human running it
by hand) use ONE command form. It refuses to clobber a pre-existing root
``pyproject.toml``.

``patchelf`` requirement: backends that link an EXTERNAL GPU runtime (rocm links
``libamdhip64`` / ``libhsa-runtime64`` / ``libdrm`` etc.) require ``patchelf`` so
maturin can repath/bundle those shared libraries into the wheel. cpu / wgpu /
cuda do not need it here (self-contained or stub-linked). Install with
``pip install patchelf`` (or ``maturin[patchelf]``); a failed rocm build whose
error names ``patchelf`` is THIS missing tool, not a code defect.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

# tests/ -> python/ -> mlrs-py/ -> crates/ -> repo root.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
PYPROJECT_DIR = REPO_ROOT / "crates" / "mlrs-py" / "pyproject"
WHEELS_DIR = REPO_ROOT / "target" / "wheels"

BACKENDS = ("cpu", "wgpu", "cuda", "rocm")


def backend_pyproject(backend: str) -> pathlib.Path:
    """Path to the per-backend maturin template (``cpu`` / ``wgpu`` / ...)."""
    return PYPROJECT_DIR / f"{backend}.pyproject.toml"


def build_wheel(
    backend: str,
    *,
    maturin: str | None = None,
    timeout: int = 1200,
) -> subprocess.CompletedProcess:
    """Build the ``mlrs-<backend>`` wheel via the temp-root-pyproject dance.

    Returns the completed ``maturin build --release`` process (caller asserts on
    ``returncode`` and ``stdout``). Raises ``FileExistsError`` if a root
    ``pyproject.toml`` already exists (never clobbers).
    """
    src = backend_pyproject(backend)
    if not src.is_file():
        raise FileNotFoundError(src)

    root_pyproject = REPO_ROOT / "pyproject.toml"
    if root_pyproject.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing {root_pyproject}; "
            "remove it or build from a clean tree."
        )

    maturin_bin = maturin or _default_maturin()
    shutil.copyfile(src, root_pyproject)
    try:
        proc = subprocess.run(
            [maturin_bin, "build", "--release"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        # Always remove the temp root pyproject, even on build failure/timeout.
        try:
            root_pyproject.unlink()
        except FileNotFoundError:
            pass
    return proc


def _default_maturin() -> str:
    """The ``maturin`` next to the running interpreter, else bare ``maturin``."""
    candidate = pathlib.Path(sys.executable).with_name("maturin")
    return str(candidate) if candidate.exists() else "maturin"


def latest_wheel(backend: str) -> pathlib.Path | None:
    """The newest ``mlrs_<backend>-*.whl`` in ``target/wheels`` (or ``None``)."""
    dist = f"mlrs_{backend}-"
    if not WHEELS_DIR.is_dir():
        return None
    # A real abi3 wheel is megabytes; skip any truncated stub a failed/aborted
    # build may have left behind (e.g. a 22-byte placeholder) so callers do not
    # mistake it for a successful build.
    min_bytes = 1024
    cands = sorted(
        (
            p
            for p in WHEELS_DIR.glob("*.whl")
            if p.name.startswith(dist) and p.stat().st_size >= min_bytes
        ),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None
