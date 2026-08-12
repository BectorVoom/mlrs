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

from .cluster import (
    DBSCAN,
    HDBSCAN,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
    SpectralEmbedding,
)
from .covariance import EmpiricalCovariance, LedoitWolf
from .decomposition import PCA, IncrementalPCA, TruncatedSVD
from .density import KernelDensity
from .ensemble import (
    ForestInference,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingRegressor,
)
from .kernel_ridge import KernelRidge
from .linear import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LinearSVC,
    LinearSVR,
    LogisticRegression,
    MBSGDClassifier,
    MBSGDRegressor,
    BayesianRidge,
    HuberRegressor,
    RANSACRegressor,
    Ridge,
    RidgeClassifier,
)
from .manifold import TSNE, UMAP
from .mixture import BayesianGaussianMixture, GaussianMixture
from .naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from .neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
from .preprocessing import (
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)
from .random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)
from .timeseries import ARIMA, AutoARIMA

# The host-only sklearn metrics surface (METR-SHIM-01): a SUBMODULE import,
# NOT top-level `__all__` names (SPEC §5 explicit instruction — avoids e.g.
# `mlrs.accuracy_score` colliding with the estimator namespace). Access via
# `mlrs.metrics.accuracy_score(...)` etc.
from . import metrics  # noqa: F401

# The sklearn model-selection surface (MODSEL-01/02 + MODSEL-RS-01..08): same
# SUBMODULE convention as `metrics` above — access via
# `mlrs.model_selection.train_test_split(...)`. The splitters, the
# ParameterGrid/Sampler combinatorics, the search + successive-halving
# schedules, the CV aggregation and the decision-threshold tuning all run in
# Rust (`mlrs_algos::model_selection`, via `_mlrs`); the module itself owns the
# sklearn-compatible classes and the container handling. `_mlrs` is imported
# LAZILY there, so this import still succeeds on a tree where the extension was
# never built.
from . import model_selection  # noqa: F401

# The sklearn feature-selection surface (FSEL-01): same SUBMODULE convention as
# `metrics` and `model_selection` above — access via
# `mlrs.feature_selection.SelectKBest(...)`. Its score functions and selector
# fits delegate to `_mlrs`; its container-native column gather (numpy / pandas /
# polars / pyarrow / list in, the same kind out) is pure host bookkeeping in
# `mlrs._frame`.
from . import feature_selection  # noqa: F401

__all__ = [
    "LinearRegression",
    "Ridge",
    "RidgeClassifier",
    "BayesianRidge",
    "HuberRegressor",
    "RANSACRegressor",
    "Lasso",
    "ElasticNet",
    "LogisticRegression",
    "LinearSVC",
    "LinearSVR",
    "MBSGDClassifier",
    "MBSGDRegressor",
    "GaussianNB",
    "MultinomialNB",
    "BernoulliNB",
    "ComplementNB",
    "CategoricalNB",
    "KernelRidge",
    "KernelDensity",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "ForestInference",
    "StackingRegressor",
    "KMeans",
    "DBSCAN",
    "HDBSCAN",
    "AgglomerativeClustering",
    "SpectralClustering",
    "BayesianGaussianMixture",
    "GaussianMixture",
    "TSNE",
    "SpectralEmbedding",
    "UMAP",
    "PCA",
    "TruncatedSVD",
    "IncrementalPCA",
    "NearestNeighbors",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "EmpiricalCovariance",
    "LedoitWolf",
    "GaussianRandomProjection",
    "SparseRandomProjection",
    "johnson_lindenstrauss_min_dim",
    "ARIMA",
    "AutoARIMA",
    "StandardScaler",
    "MinMaxScaler",
    "MaxAbsScaler",
    "RobustScaler",
    "Normalizer",
    "Binarizer",
    "backend_supports_f64",
    "backend_supports_f64_transcendental",
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

    Uses ``importlib.import_module("mlrs._mlrs")`` rather than ``from . import
    _mlrs``: the latter routes a *failed* submodule import back through this
    module's ``__getattr__`` (``_handle_fromlist`` -> ``getattr(mlrs, "_mlrs")``
    -> ``__getattr__("_mlrs")`` -> ``_load_ext()``), which recurses infinitely
    when the ``.so`` is genuinely unimportable. ``import_module`` resolves the
    submodule through the import system directly, so a real failure raises the
    intended clear ``ImportError`` instead of a ``RecursionError`` that masks it.
    """
    import importlib

    try:
        ext = importlib.import_module("mlrs._mlrs")
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
    if name == "backend_supports_f64_transcendental":
        return _load_ext().backend_supports_f64_transcendental
    raise AttributeError(f"module 'mlrs' has no attribute {name!r}")
