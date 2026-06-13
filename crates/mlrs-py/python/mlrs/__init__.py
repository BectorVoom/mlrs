"""mlrs — cuML in Rust, sklearn-compatible Python surface.

One pure-Python ``mlrs`` package over a per-backend compiled extension
(``mlrs._mlrs``). Each backend wheel (``mlrs-cpu`` / ``mlrs-wgpu`` /
``mlrs-cuda`` / ``mlrs-rocm``) ships the SAME ``import mlrs`` namespace with a
different compiled ``_mlrs`` inside (D-07).

This file:
  1. Imports the compiled ``_mlrs`` extension, guarded so a not-yet-built tree
     (Wave 0, before ``maturin develop``) raises a CLEAR message instead of a
     bare ``ModuleNotFoundError``. Importing ``_mlrs`` is also what triggers the
     D-08 driver probe → ``ImportError`` when a backend's runtime is absent
     (wired in Plan 02).
  2. Re-exports the 12 sklearn-compatible estimators from the family modules
     (D-01) — the pure-Python shims are importable WITHOUT the extension; only
     ``_mlrs`` itself is guarded, so the package structure parses pre-wrappers.
"""

try:
    from . import _mlrs as _mlrs  # noqa: F401  (compiled per backend wheel)
except ImportError as _exc:  # pragma: no cover - exercised post-build
    raise ImportError(
        "mlrs: the compiled '_mlrs' backend extension is not available. "
        "Install a backend wheel (mlrs-cpu / mlrs-wgpu / mlrs-cuda / "
        "mlrs-rocm), or build it in-tree with "
        "`maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml`."
    ) from _exc

from .cluster import DBSCAN, KMeans
from .decomposition import PCA, TruncatedSVD
from .linear import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from .neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)

__all__ = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "LogisticRegression",
    "KMeans",
    "DBSCAN",
    "PCA",
    "TruncatedSVD",
    "NearestNeighbors",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
]
