"""mlrs — cuML in Rust, sklearn-compatible Python surface.

One pure-Python ``mlrs`` package over a per-backend compiled extension
(``mlrs._mlrs``). Each backend wheel (``mlrs-cpu`` / ``mlrs-wgpu`` /
``mlrs-cuda`` / ``mlrs-rocm``) ships the SAME ``import mlrs`` namespace with a
different compiled ``_mlrs`` inside (D-07).

This file:
  1. Re-exports the 12 sklearn-compatible estimators from the family modules
     (D-01). The pure-Python shims are importable WITHOUT the extension — only
     the *delegate calls* (``fit`` / fitted-attr access) reach ``_mlrs``, so the
     package structure (and ``get_params`` / ``set_params`` / ``clone``) works
     pre-build for the estimator-checks and param round-trip tests.
  2. Lazily exposes the compiled ``_mlrs`` extension (and the surfaced
     ``backend_supports_f64`` flag) via module ``__getattr__`` — accessing
     ``mlrs._mlrs`` / ``mlrs.backend_supports_f64`` (or calling ``fit``) on a
     not-yet-built tree raises a CLEAR ``ImportError`` instead of a bare
     ``ModuleNotFoundError``. Importing ``_mlrs`` is also what triggers the D-08
     driver probe -> ``ImportError`` when a backend's runtime is absent.
"""

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
    "backend_supports_f64",
]

_EXT_MISSING_MSG = (
    "mlrs: the compiled '_mlrs' backend extension is not available. "
    "Install a backend wheel (mlrs-cpu / mlrs-wgpu / mlrs-cuda / mlrs-rocm), "
    "or build it in-tree with "
    "`maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml`."
)


def _load_ext():
    """Import the compiled ``_mlrs`` extension or raise a clear ImportError.

    Importing ``_mlrs`` triggers the D-08 driver probe (wired in Plan 02), so a
    backend whose runtime is absent surfaces as an ``ImportError`` here too.
    """
    try:
        from . import _mlrs as ext
    except ImportError as exc:  # pragma: no cover - exercised post-build
        raise ImportError(_EXT_MISSING_MSG) from exc
    return ext


def __getattr__(name):
    """Lazily resolve ``_mlrs`` and the surfaced capability flag (PEP 562).

    Keeps ``import mlrs`` + estimator construction working on a not-yet-built
    tree while still giving a clear error the moment the extension is actually
    needed.
    """
    if name == "_mlrs":
        return _load_ext()
    if name == "backend_supports_f64":
        return _load_ext().backend_supports_f64
    raise AttributeError(f"module 'mlrs' has no attribute {name!r}")
