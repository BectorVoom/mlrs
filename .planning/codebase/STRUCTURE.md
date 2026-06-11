# Codebase Structure

**Analysis Date:** 2026-06-11

## Directory Layout

```
cuml-main/
├── build.sh                    # Main build entry point for all targets
├── pyproject.toml              # Python tooling config (ruff, codespell, cython-lint)
├── dependencies.yaml           # RAPIDS dependency spec
├── VERSION                     # Package version
├── cpp/                        # libcuml++ C++/CUDA library
│   ├── CMakeLists.txt          # Main C++ build (875 lines)
│   ├── include/cuml/           # Public C++ API headers (installed with library)
│   │   ├── cluster/            # Clustering algorithm headers
│   │   ├── common/             # Handle, logger, export macros
│   │   ├── datasets/           # Dataset generation headers
│   │   ├── decomposition/      # PCA, TSVD headers
│   │   ├── ensemble/           # Random forest headers
│   │   ├── explainer/          # SHAP headers
│   │   ├── genetic/            # Symbolic regression headers
│   │   ├── linear_model/       # GLM, OLS, Ridge, QN headers
│   │   ├── manifold/           # UMAP, t-SNE headers
│   │   ├── matrix/             # Matrix utility headers
│   │   ├── metrics/            # Scoring function headers
│   │   ├── neighbors/          # KNN headers
│   │   ├── prims/              # Primitive headers (opg/)
│   │   ├── solvers/            # Solver headers
│   │   ├── svm/                # SVM headers
│   │   ├── tree/               # Decision tree headers
│   │   └── tsa/                # Time series headers
│   ├── src/                    # Algorithm implementations (.cu, .cuh)
│   │   ├── arima/              # ARIMA / AutoARIMA
│   │   ├── common/             # nvtx.hpp shared utility
│   │   ├── datasets/           # Dataset generation
│   │   ├── dbscan/             # DBSCAN clustering
│   │   ├── decisiontree/       # Decision tree impl
│   │   ├── explainer/          # SHAP explainers
│   │   ├── genetic/            # Symbolic regression (GPEA)
│   │   ├── glm/                # OLS, Ridge, QN (Quasi-Newton)
│   │   ├── hdbscan/            # HDBSCAN
│   │   ├── hierarchy/          # Agglomerative clustering
│   │   ├── holtwinters/        # Holt-Winters exponential smoothing
│   │   ├── kde/                # Kernel Density Estimation
│   │   ├── kmeans/             # K-Means
│   │   ├── knn/                # K-Nearest Neighbors
│   │   ├── matrix/             # Matrix ops
│   │   ├── metrics/            # Scoring metrics
│   │   ├── pca/                # PCA
│   │   ├── randomforest/       # Random forest
│   │   ├── solver/             # CD, SGD solvers
│   │   ├── spectral/           # Spectral clustering / embedding
│   │   ├── svm/                # SVM (classification + regression)
│   │   ├── tsa/                # Time series analysis helpers
│   │   ├── tsne/               # t-SNE
│   │   ├── tsvd/               # Truncated SVD
│   │   └── umap/               # UMAP
│   ├── src_prims/              # Header-only shared math primitives
│   │   ├── common/             # Shared utilities
│   │   ├── functions/          # Activation, loss functions
│   │   ├── linalg/             # Linear algebra primitives
│   │   ├── matrix/             # Matrix manipulation primitives
│   │   ├── opg/                # Multi-GPU (OPG) primitives
│   │   ├── random/             # RNG utilities
│   │   ├── selection/          # Selection/sort primitives
│   │   ├── sparse/             # Sparse matrix primitives
│   │   └── timeSeries/         # Time series math primitives
│   ├── tests/                  # C++ unit/integration tests
│   │   ├── sg/                 # Single-GPU tests (.cu per algorithm)
│   │   └── mg/                 # Multi-GPU tests
│   ├── bench/                  # C++ benchmarks
│   ├── examples/               # C++ usage examples (e.g., symreg/)
│   └── cmake/                  # Build helper CMake modules
│       └── modules/            # ConfigureAlgorithms.cmake, ConfigureCUDA.cmake
├── python/
│   ├── libcuml/                # Build stub: finds/builds libcuml++ as dependency
│   │   ├── CMakeLists.txt
│   │   └── libcuml/__init__.py
│   └── cuml/                   # The `cuml` Python package source tree
│       ├── cuml/               # Actual package directory
│       │   ├── __init__.py     # Package entry point; re-exports estimators
│       │   ├── internals/      # Core infrastructure (Base, CumlArray, I/O, handle)
│       │   ├── accel/          # cuml.accel transparent acceleration layer
│       │   ├── dask/           # Multi-GPU Dask wrappers
│       │   ├── common/         # Shared Python utilities (array descriptors, doc utils)
│       │   ├── cluster/        # KMeans, DBSCAN, HDBSCAN, Agglomerative, Spectral
│       │   ├── decomposition/  # PCA, TruncatedSVD, IncrementalPCA
│       │   ├── ensemble/       # RandomForest (classifier + regressor)
│       │   ├── explainer/      # SHAP (kernel, permutation, tree)
│       │   ├── fil/            # Forest Inference Library
│       │   ├── linear_model/   # OLS, Ridge, Lasso, ElasticNet, Logistic, MBSGD
│       │   ├── manifold/       # UMAP, t-SNE
│       │   ├── metrics/        # Scoring metrics, cluster metrics
│       │   ├── neighbors/      # KNN, KNeighborsClassifier/Regressor, KDE
│       │   ├── preprocessing/  # Scalers, encoders, imputers
│       │   ├── solvers/        # SGD, CD (Coordinate Descent), QN
│       │   ├── svm/            # SVC, SVR
│       │   ├── tsa/            # ARIMA, AutoARIMA, Holt-Winters
│       │   ├── naive_bayes/    # Naive Bayes (BernoulliNB, MultinomialNB, etc.)
│       │   ├── pipeline/       # Pipeline
│       │   ├── compose/        # ColumnTransformer
│       │   ├── covariance/     # EmpiricalCovariance, MinCovDet
│       │   ├── feature_extraction/ # TF-IDF, HashingVectorizer
│       │   ├── kernel_ridge/   # KernelRidge
│       │   ├── multiclass/     # OneVsRestClassifier, OneVsOneClassifier
│       │   ├── model_selection/# train_test_split, cross_val_score
│       │   ├── random_projection/ # Random Projection
│       │   ├── datasets/       # Dataset generators (make_blobs, etc.)
│       │   ├── experimental/   # LARS (experimental algorithms)
│       │   ├── prims/          # Python-exposed primitives
│       │   ├── comm/           # Communicator abstraction (multi-GPU)
│       │   ├── benchmark/      # Python benchmarking utilities
│       │   ├── health_checks/  # GPU health check utilities
│       │   ├── testing/        # Test utilities and fixtures
│       │   ├── thirdparty_adapters/ # Adapters for 3rd-party library compatibility
│       │   └── _thirdparty/    # Vendored third-party Python code
│       ├── cuml_accel_tests/   # Tests specific to cuml.accel
│       │   ├── integration/    # Integration tests for cuml.accel
│       │   └── upstream/       # Upstream sklearn/hdbscan/umap test suites run under accel
│       └── tests/              # Main Python test suite (139 .py files)
│           ├── conftest.py
│           ├── dask/           # Dask multi-GPU tests
│           ├── explainer/      # SHAP explainer tests
│           ├── stemmer_tests/
│           └── ts_datasets/    # Time series test data
├── cmake/                      # Top-level CMake utilities
│   ├── RAPIDS.cmake
│   ├── rapids_config.cmake
│   └── modules/                # Reusable CMake find/configure modules
├── conda/
│   ├── environments/           # Per-CUDA/arch conda env YAML files
│   └── recipes/                # conda-build recipes (cuml/, libcuml/)
├── ci/                         # CI/CD scripts
│   ├── checks/                 # Lint and style checks
│   ├── notebooks/              # Notebook CI runners
│   ├── release/                # Release automation
│   └── utils/                  # Shared CI utilities
├── docs/
│   └── source/                 # Sphinx documentation source
│       ├── api/                # Auto-generated API docs
│       ├── cuml-accel/         # cuml.accel user guide
│       └── conf.py             # Sphinx configuration
├── notebooks/                  # Example Jupyter notebooks
│   ├── data/                   # Notebook data files
│   └── tools/                  # Notebook tooling
├── wiki/                       # Developer wiki
│   ├── cpp/                    # C++ contributor notes
│   ├── mnmg/                   # Multi-node multi-GPU notes
│   └── python/                 # Python contributor notes
└── thirdparty/
    └── LICENSES/               # Third-party license files
```

## Directory Purposes

**`cpp/include/cuml/`:**
- Purpose: Installed public C++ API — the only headers Cython (and external C++ consumers) should include
- Contains: One `.hpp` per algorithm category; use `cuml/common/export.hpp` for `CUML_EXPORT` macro
- Key files: `cpp/include/cuml/linear_model/glm.hpp`, `cpp/include/cuml/cluster/kmeans.hpp`, `cpp/include/cuml/common/export.hpp`

**`cpp/src/`:**
- Purpose: Private algorithm implementations — not installed, not included externally
- Contains: `.cu` (CUDA device + host code), `.cuh` (device-callable headers, not for Cython)
- Key files: `cpp/src/glm/glm.cu`, `cpp/src/kmeans/kmeans_fit.cu`, `cpp/src/umap/`

**`cpp/src_prims/`:**
- Purpose: Header-only math primitive library shared across algorithms; also has standalone test suite
- Contains: Template headers for linalg, matrix, sparse, random, selection operations

**`python/cuml/cuml/internals/`:**
- Purpose: Core infrastructure — the foundation all estimators build on
- Key files:
  - `base.py` — `Base` class, `get_handle()`
  - `array.py` — `CumlArray` unified GPU array
  - `input_utils.py` — `input_to_cuml_array()` conversion entry point
  - `outputs.py` — `output_type` management, `@reflect` decorator
  - `interop.py` — `InteropMixin`, `to_gpu()`, `to_cpu()`
  - `validation.py` — `check_inputs()` input validation
  - `mixins.py` — Tag mixins (`RegressorMixin`, `ClassifierMixin`, etc.)
  - `logger.pyx` — Cython wrapper for spdlog

**`python/cuml/cuml/accel/`:**
- Purpose: The `cuml.accel` transparent acceleration subsystem
- Key files:
  - `accelerator.py` — `Accelerator`, `AccelModule` (import hook machinery)
  - `core.py` — `install()`, `enabled()`, module registration, `_OVERRIDES`/`_PATCHES` lists
  - `estimator_proxy.py` — `ProxyBase`, `ProxyBaseMeta`, `is_proxy()`
  - `__main__.py` — CLI entry point (`python -m cuml.accel`)
  - `_overrides/` — Per-package GPU override namespaces (e.g., `_overrides/sklearn/linear_model/`)
  - `_patches/` — sklearn compatibility patches (pipeline, compose, utils)

**`python/cuml/cuml/dask/`:**
- Purpose: Dask-based multi-GPU distributed wrappers; mirrors single-GPU module structure
- Key pattern: Each `dask/{module}/*.py` wraps the corresponding `cuml/{module}` estimator via Dask futures

**`python/cuml/cuml/common/`:**
- Purpose: Shared Python utilities not part of the internals infrastructure
- Key files: `array_descriptor.py` (`CumlArrayDescriptor`), `doc_utils.py`, `sparse_utils.py`

## Key File Locations

**Build Entry Points:**
- `build.sh` — Shell build script; all targets (libcuml, cuml, prims, bench, docs)
- `cpp/CMakeLists.txt` — C++ library CMake (875 lines); controls algorithm inclusion
- `python/libcuml/CMakeLists.txt` — Python package CMake; locates libcuml++
- `cmake/rapids_config.cmake` — RAPIDS version configuration

**Core Algorithm Headers (C++):**
- `cpp/include/cuml/linear_model/glm.hpp` — OLS, Ridge, QN
- `cpp/include/cuml/cluster/kmeans.hpp` — KMeans fit/predict/transform
- `cpp/include/cuml/cluster/dbscan.hpp` — DBSCAN
- `cpp/include/cuml/neighbors/knn.hpp` — KNN
- `cpp/include/cuml/manifold/umap.hpp` — UMAP
- `cpp/include/cuml/ensemble/randomforest.hpp` — Random forest
- `cpp/include/cuml/svm/svm_model.h` — SVM

**Python Infrastructure:**
- `python/cuml/cuml/internals/base.py` — `Base`, `get_handle()`
- `python/cuml/cuml/internals/array.py` — `CumlArray`
- `python/cuml/cuml/internals/input_utils.py` — `input_to_cuml_array()`
- `python/cuml/cuml/internals/outputs.py` — `reflect`, `using_output_type`
- `python/cuml/cuml/internals/interop.py` — `InteropMixin`

**Representative Cython Bindings:**
- `python/cuml/cuml/linear_model/linear_regression.pyx` — OLS (canonical example of Cython binding pattern)
- `python/cuml/cuml/cluster/kmeans.pyx` — KMeans
- `python/cuml/cuml/neighbors/nearest_neighbors.pyx` — KNN
- `python/cuml/cuml/manifold/umap/umap.pyx` — UMAP

**Multi-GPU Cython Bindings (`_mg.pyx`):**
- `python/cuml/cuml/linear_model/linear_regression_mg.pyx`
- `python/cuml/cuml/neighbors/nearest_neighbors_mg.pyx`
- `python/cuml/cuml/decomposition/pca_mg.pyx`

**Tests:**
- `python/cuml/tests/` — Main Python test suite (139 files)
- `python/cuml/cuml_accel_tests/` — cuml.accel-specific tests
- `cpp/tests/sg/` — Single-GPU C++ tests
- `cpp/tests/mg/` — Multi-GPU C++ tests

**Configuration:**
- `pyproject.toml` — ruff (line-length=79), codespell, cython-lint (max-line-length=95)
- `conda/environments/all_cuda-*.yaml` — Per-CUDA-version dependency environments
- `dependencies.yaml` — RAPIDS ecosystem dependency declarations

## Naming Conventions

**C++ Files:**
- Algorithm CUDA implementation: `{algorithm}.cu` or `{algorithm}_fit.cu` / `{algorithm}_predict.cu` (e.g., `kmeans_fit.cu`)
- Device-callable headers: `{algorithm}.cuh`
- Multi-GPU variants: `{algorithm}_mg.cu` (e.g., `ols_mg.cu`)
- Public API headers: `{algorithm_category}.hpp` under `cpp/include/cuml/{category}/`

**Python Files:**
- Cython algorithm binding: `{algorithm_name}.pyx` (e.g., `linear_regression.pyx`, `kmeans.pyx`)
- Multi-GPU Cython: `{algorithm_name}_mg.pyx`
- Pure Python estimator wrapper: `{algorithm_name}.py`
- Package init: `__init__.py` re-exports public API

**Python Classes:**
- Estimators: `PascalCase` matching sklearn (e.g., `LinearRegression`, `KMeans`, `RandomForestClassifier`)
- Base/mixin classes: `Base`, `InteropMixin`, `RegressorMixin`, `TagsMixin`
- Proxy classes (accel): `ProxyBase`, `ProxyBaseMeta`

**Directories:**
- C++ algorithm dirs: `snake_case` matching algorithm name (e.g., `decisiontree/`, `randomforest/`)
- Python module dirs: `snake_case` matching sklearn API (e.g., `linear_model/`, `model_selection/`)

## Where to Add New Code

**New C++ Algorithm:**
1. Public API header: `cpp/include/cuml/{category}/{algorithm}.hpp` — declare function signatures with `raft::handle_t&` and raw device pointers
2. Implementation: `cpp/src/{algorithm}/{algorithm}.cu` + `{algorithm}.cuh` for device internals
3. Register in CMake: `cpp/CMakeLists.txt` under `ConfigureAlgorithms.cmake` or directly
4. C++ test: `cpp/tests/sg/{algorithm}_test.cu`

**New Python Estimator:**
1. Cython binding: `python/cuml/cuml/{module}/{algorithm}.pyx` — `cdef extern from` the public header, inherit from `Base` + appropriate mixins
2. If pure Python wrapping Cython: `python/cuml/cuml/{module}/{algorithm}.py`
3. Add `InteropMixin` and define `_cpu_class_path`, `_params_from_cpu`, `_params_to_cpu` for cuml.accel compatibility
4. Register in `python/cuml/cuml/{module}/__init__.py`
5. Add accel override: `python/cuml/cuml/accel/_overrides/sklearn/{module}/` (or appropriate package)
6. Python test: `python/cuml/tests/test_{algorithm}.py`

**New cuml.accel Override:**
1. Add module name to `_OVERRIDES` set in `python/cuml/cuml/accel/core.py:85`
2. Create `python/cuml/cuml/accel/_overrides/{package}/{module}.py` returning a dict mapping class names to cuml GPU classes
3. Ensure the cuml class implements `InteropMixin`

**New Dask Multi-GPU Estimator:**
1. Add `python/cuml/cuml/dask/{module}/{algorithm}.py`
2. Add corresponding `{algorithm}_mg.pyx` Cython binding if new MG C++ code is needed
3. Add MG C++ impl at `cpp/src/{algorithm}/{algorithm}_mg.cu`

**Utilities and Shared Helpers:**
- Python math/array helpers: `python/cuml/cuml/internals/` (if core) or `python/cuml/cuml/common/` (if algorithm-agnostic utility)
- C++ math primitives: `cpp/src_prims/{category}/` (header-only)

## Special Directories

**`cpp/src_prims/`:**
- Purpose: Header-only CUDA/C++ primitives used by multiple algorithms
- Generated: No
- Committed: Yes
- Note: Has its own test suite built via `build.sh prims`

**`thirdparty/LICENSES/`:**
- Purpose: Licenses for vendored third-party code
- Generated: No
- Committed: Yes

**`python/cuml/cuml/_thirdparty/`:**
- Purpose: Vendored Python third-party code (e.g., stemmer, stop words)
- Generated: No
- Committed: Yes

**`conda/environments/`:**
- Purpose: Per-CUDA-version (12.9, 13.2) per-arch (x86_64, aarch64) environment YAML files
- Generated: No (maintained manually + via `dependencies.yaml`)
- Committed: Yes

**`cpp/build/` (when present after build):**
- Purpose: CMake build artifacts, compiled `.so` objects
- Generated: Yes
- Committed: No (`.gitignore`d)

---

*Structure analysis: 2026-06-11*
