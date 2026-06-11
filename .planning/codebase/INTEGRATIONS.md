# External Integrations

**Analysis Date:** 2026-06-11
**Subject:** RAPIDS cuML v26.08.00 (`cuml-main/`)
**Note:** This is reference material for a future Rust port of cuML algorithms.

---

## GPU / CUDA Runtime

**CUDA Toolkit:**
- Versions: 12.9 and 13.2 (primary); 12.2–13.x for consumers
- Required CUDA libraries (linked at build time):
  - `cuBLAS` — dense linear algebra on GPU
  - `cuFFT` — Fast Fourier Transform on GPU (linked when `LINK_CUFFT=ON`)
  - `cuRAND` — random number generation on GPU
  - `cuSolver` — sparse/dense solvers on GPU
  - `cuSparse` — sparse matrix operations on GPU
  - `cuda-profiler-api` — NVTX / profiling hooks (when `NVTX=ON`)
- Runtime wheel dependency: `nvidia-nvjitlink` (>=12.9 / >=13.0 matched to CTK minor)
- Python bindings: `cuda-python` >=12.9.2 / >=13.0.1 (official NVIDIA Python CUDA bindings)
- CUDA init: `rapids_cuda_init_runtime(USE_STATIC ON)` in `cuml-main/cpp/CMakeLists.txt`

**Numba CUDA:**
- `numba-cuda` >=0.22.2,<0.29.0 — JIT-compiles Python/NumPy functions to CUDA kernels
- Provides GPU acceleration for some Python-layer operations without writing Cython

---

## RAPIDS Ecosystem Dependencies

All pinned to the same `26.8.*` release train as cuML itself. Version management centralized in `cuml-main/dependencies.yaml`.

**RMM (RAPIDS Memory Manager):**
- C++ library: `librmm` 26.8.* — custom GPU allocator, pool allocator, logging
- Python bindings: `rmm` 26.8.*
- CMake fetch: `cuml-main/cpp/cmake/thirdparty/get_rmm.cmake` (via `rapids_cpm_rmm`)
- Used by: all GPU memory allocations within cuML C++ layer
- Logging level configurable: `RMM_LOGGING_LEVEL` CMake variable (default: INFO)

**RAFT (Reusable Accelerated Functions and Tools):**
- C++ headers-only + compiled: `libraft` 26.8.* (headers variant: `libraft-headers`)
- Python bindings: `pylibraft` 26.8.*
- Distributed variant: `raft-dask` 26.8.* (multi-GPU Dask integration)
- CMake fetch: `cuml-main/cpp/cmake/thirdparty/get_raft.cmake` (via `rapids_cpm_find`)
  - `GIT_REPOSITORY https://github.com/rapidsai/raft.git`
  - `SOURCE_SUBDIR cpp`
  - Optional `distributed` component for multi-GPU tests
- Provides: distance computation, nearest-neighbor primitives, linear algebra, random projections, sparse primitives
- Used by: nearly all algorithm implementations in `cuml-main/cpp/src/`

**cuVS (CUDA Vector Search):**
- C++ library: `libcuvs` 26.8.*
- Python package: `cuvs` 26.8.* (test/conda environment only)
- CMake fetch: `cuml-main/cpp/cmake/thirdparty/get_cuvs.cmake` (via `rapids_cpm_find`)
  - `GIT_REPOSITORY https://github.com/rapidsai/cuvs.git`
  - `SOURCE_SUBDIR cpp`
  - Options: `BUILD_CAGRA_HNSWLIB OFF`, `BUILD_CUVS_BENCH OFF`
  - Multi-GPU support: `BUILD_MG_ALGOS ON` (OFF when `SINGLEGPU`)
- Provides: approximate nearest neighbor (ANN) search — CAGRA, IVF-Flat, IVF-PQ, brute-force KNN
- Linked when: `LINK_CUVS=ON` (set for full builds via `ConfigureAlgorithms.cmake`)
- Static linking option: `CUML_USE_CUVS_STATIC`

**cuDF (CUDA DataFrame):**
- Python package: `cudf` 26.8.*
- Used in Python layer for GPU DataFrame input/output in estimators
- Distributed variant: `dask-cudf` 26.8.* (multi-GPU DataFrames)
- No direct C++ dependency; accessed entirely through Python API

**nvForest (NVIDIA Forest / Tree Models):**
- C++ library: `libnvforest` 26.8.*
- Python bindings: `nvforest` 26.8.*
- CMake fetch: `cuml-main/cpp/cmake/thirdparty/get_nvforest.cmake`
- Linked as: `nvforest::nvforest++` when `LINK_NVFOREST=ON`
- Provides: GPU-accelerated random forest and gradient boosting inference
- Static exclusion option: `CUML_EXCLUDE_NVFOREST_FROM_ALL`

**RAPIDS Logger:**
- C++ / Python: `rapids-logger` 0.2.*
- Used for structured logging throughout libcuml
- Macros created via `create_logger_macros(CUML "ML::default_logger()" include/cuml/common)` in `cuml-main/cpp/CMakeLists.txt`

**Dask / Multi-GPU Cluster:**
- `rapids-dask-dependency` 26.8.* — pins Dask version for RAPIDS compatibility
- `dask-cuda` 26.8.* — GPU-aware Dask workers (cluster launch, UCX transport)
- Multi-GPU algorithm implementations: `cuml-main/python/cuml/cuml/dask/`
- Multi-node testing: `BUILD_CUML_MPI_COMMS` option (MPI+NCCL communicator)

---

## Third-Party C++ Libraries (CPM-managed)

**CCCL (CUDA C++ Core Libraries):**
- Includes: Thrust, CUB, libcu++
- Fetch: `cuml-main/cpp/cmake/thirdparty/get_cccl.cmake` (via `rapids_cpm_cccl`)
- Must be fetched before RMM and RAFT (ordering enforced in `cuml-main/cpp/CMakeLists.txt`)
- Provides: GPU parallel algorithms (sort, reduce, scan, etc.), device STL

**Treelite:**
- Version: 4.7.0 (pinned tag `74b25ecedb964ccac37d034860cc5c1224e73e91`)
- Source: `https://github.com/dmlc/treelite.git`
- CMake fetch: `cuml-main/cpp/cmake/thirdparty/get_treelite.cmake`
- Python package: `treelite` >=4.7.0,<5.0.0
- Linked when: `LINK_TREELITE=ON` (set for full builds)
- Static option: `CUML_USE_TREELITE_STATIC`
- Provides: serialization and inference for decision tree ensembles (Random Forest, XGBoost export)
- Also wrapped in Cython: `cuml-main/python/cuml/cuml/internals/treelite.pyx`

**GPUTreeShap:**
- Source: `https://github.com/rapidsai/gputreeshap.git`
- Pinned commit: `93292317b23ef733f881c881865f5d5728dc2fea`
- CMake fetch: `cuml-main/cpp/cmake/thirdparty/get_gputreeshap.cmake`
- Header-only library; provides GPU-accelerated SHAP value computation for tree models
- Linked when: `all_algo` or `treeshap_algo` in `ConfigureAlgorithms.cmake`
- Used by SHAP explainer: `cuml-main/cpp/src/explainer/`, `cuml-main/python/cuml/cuml/explainer/tree_shap.pyx`

**Google Test (GTest):**
- Fetched via `rapids_cpm_gtest(BUILD_STATIC)` when `BUILD_CUML_TESTS` or `BUILD_PRIMS_TESTS`
- Used in `cuml-main/cpp/test/`

**Google Benchmark (GBench):**
- Fetched via `rapids_cpm_gbench(BUILD_STATIC)` when `BUILD_CUML_BENCH`
- Used in `cuml-main/cpp/bench/`

---

## Thirdparty Code (Vendored)

Licenses tracked in `cuml-main/thirdparty/LICENSES/`:

| Source | License | What's vendored |
|--------|---------|-----------------|
| scikit-learn | BSD-3-Clause | Preprocessing utilities, sklearn compat layer |
| UMAP | BSD-3-Clause | UMAP algorithm reference |
| faiss | MIT | Nearest-neighbor search concepts |
| H2O4GPU | Apache-2.0 | Historical GPU ML reference |

Python-side vendored sklearn code lives in:
- `cuml-main/python/cuml/cuml/_thirdparty/sklearn/` — preprocessing, utils
- `cuml-main/python/cuml/cuml/_thirdparty/_sklearn_compat.py`

---

## scikit-learn API Compatibility

**Design intent:** cuML estimators implement the scikit-learn estimator API so they can be used as drop-in replacements.

**Key mechanisms:**
- Python estimators inherit from sklearn's `BaseEstimator` and mixins (fit/predict/transform pattern)
- `cuml.accel` module — transparent drop-in accelerator:
  - `cuml-main/python/cuml/cuml/accel/` — accelerator entry points
  - Activated via `python -m cuml.accel` or `import cuml.accel; cuml.accel.install()`
  - Patches sklearn imports to redirect to GPU-backed cuML equivalents at runtime
  - `cuml-main/python/cuml/cuml/accel/accelerator.py`, `estimator_proxy.py`, `_overrides/`, `_patches/`
- sklearn compatibility test suite: `cuml-main/python/cuml/cuml_accel_tests/upstream/scikit-learn/`
- CI job: `conda-python-scikit-learn-accel-tests` in `cuml-main/.github/workflows/pr.yaml`

**sklearn versions tested:**
- Oldest: 1.6.0
- Intermediate: 1.7.2
- Latest: 1.8.0
(Exact pins in `dependencies.yaml` `test_python_accel_sklearn` section)

**ONNX export:**
- `skl2onnx` + `onnxruntime` — export cuML/sklearn models to ONNX for inference
- Available in test and docs environments; not required for runtime

---

## XGBoost Integration

- `rapids-xgboost` 26.8.* (conda) / `xgboost-cu12` or `xgboost-cu13` >=2.1.0 (wheel)
- Used for test integration and FIL (Forest Inference Library) model format compatibility
- Separated into `test_python_xgboost` dependency group (excluded from devcontainer builds due to `librmm` conflict)

---

## CI / CD

**Platform:** GitHub Actions
- Workflow files: `cuml-main/.github/workflows/pr.yaml`, `build.yaml`, `test.yaml`
- Reuses: `rapidsai/shared-workflows` (centralized RAPIDS CI infrastructure)
- GPU runners: NVIDIA GHA runners (`nv-gha-runners/get-pr-info`)
- Telemetry: OpenTelemetry (`OTEL_SERVICE_NAME: "pr-cuml"`)

**CI job matrix (from `pr.yaml`):**
- `conda-cpp-build` / `conda-cpp-tests` — C++ library and tests
- `conda-python-build` / `conda-python-tests-singlegpu` — Python single-GPU
- `conda-python-tests-dask` — Multi-GPU Dask tests
- `conda-python-tests-cudf-pandas-integration` — cuDF pandas integration
- `conda-python-scikit-learn-accel-tests` — sklearn upstream test suite via `cuml.accel`
- `conda-python-cuml-accel-upstream-tests` — cuML accel upstream (UMAP, HDBSCAN)
- `conda-notebook-tests` — Jupyter notebook regression tests
- `wheel-build-libcuml` / `wheel-build-cuml` — PyPI wheel builds
- `wheel-tests-cuml` / `wheel-tests-cuml-dask` — Wheel smoke tests
- `clang-tidy` — static analysis
- `docs-build` — documentation build
- `devcontainer` — devcontainer image validation

**Nightly CI check:** PRs verified against passing nightly builds (max 14 days allowed since last success).

**Code coverage:** `codecov.yml` present at `cuml-main/codecov.yml`

**Shell script CI helpers (`cuml-main/ci/`):**
- `build_cpp.sh`, `build_python.sh`, `build_wheel.sh`, `build_wheel_cuml.sh`, `build_wheel_libcuml.sh`
- `test_cpp.sh`, `test_python_singlegpu.sh`, `test_python_dask.sh`, `test_python_scikit_learn_tests.sh`
- `run_clang_tidy.sh`, `check_style.sh`

---

## conda / Package Distribution

**Conda recipes:**
- `cuml-main/conda/recipes/cuml/recipe.yaml` — Python cuml conda package
- `cuml-main/conda/recipes/libcuml/recipe.yaml` — C++ libcuml conda package
- Build configs: `conda_build_config.yaml` in each recipe directory

**Dependency file generation:**
- Tool: `rapids-dependency-file-generator` v1.20.0 (run as pre-commit hook)
- Source of truth: `cuml-main/dependencies.yaml`
- Outputs: conda env YAMLs, pyproject.toml dependency sections, requirements files
- Hook config in `cuml-main/.pre-commit-config.yaml`

**Conda channels (in priority order):**
1. `rapidsai-nightly`
2. `rapidsai`
3. `conda-forge`

---

## Devcontainer

- `cuml-main/.devcontainer/Dockerfile` — Development container definition
- `cuml-main/.devcontainer/README.md` — Setup instructions
- Validated in CI via `devcontainer` job

---

## Monitoring / Observability

**Error tracking:** Not detected (no Sentry or similar SDK)

**Logging:**
- C++ layer: `rapids-logger` structured logging; macros `CUML_LOG_*` generated from `ML::default_logger()`; headers at `include/cuml/common/`
- Log levels: TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, OFF (configurable as `LIBCUML_LOGGING_LEVEL` CMake variable, default: DEBUG for build; INFO for RMM)
- Python layer: `rich` for terminal output; `cuml-main/python/cuml/cuml/internals/logger.pyx` for bridging to C++ logger

**Profiling:**
- NVTX markers: optional, enabled via `NVTX=ON` CMake option; CUDA Profiler API included as build dep
- `nvidia-ml-py` >=12 (`pynvml`) — GPU monitoring in Python tests

---

*Integration audit: 2026-06-11*
