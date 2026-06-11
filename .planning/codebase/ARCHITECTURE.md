<!-- refreshed: 2026-06-11 -->
# Architecture

**Analysis Date:** 2026-06-11

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│                       User / Application Layer                        │
│   sklearn-compatible API  │  cuml.accel (transparent acceleration)   │
└────────────┬──────────────┴──────────────────┬────────────────────────┘
             │                                  │
             ▼                                  ▼
┌──────────────────────────┐    ┌──────────────────────────────────────┐
│  Python API (pure .py)   │    │  cuml.accel Accelerator Layer        │
│  `python/cuml/cuml/`     │    │  `python/cuml/cuml/accel/`           │
│  Base, estimators,       │    │  AccelModule wraps sklearn/hdbscan/  │
│  Dask multi-GPU wrappers │    │  umap modules via sys.modules swap   │
└────────────┬─────────────┘    └──────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   Cython Binding Layer (.pyx)                         │
│  `python/cuml/cuml/{module}/*.pyx`                                   │
│  cdef extern from "cuml/…/header.hpp" → calls C++ functions          │
│  CumlArray wraps GPU pointers, handle_t passed via pylibraft          │
└────────────┬─────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────────┐
│              libcuml++ C++/CUDA Library                               │
│  Public API:  `cpp/include/cuml/`  (algorithm headers per category)  │
│  Impl:        `cpp/src/`           (per-algorithm .cu / .cuh files)  │
│  Primitives:  `cpp/src_prims/`     (shared math/linalg/matrix prims) │
│  Handle:      `raft::handle_t`     (CUDA stream + resource manager)  │
└────────────┬─────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────────┐
│             RAPIDS Ecosystem & GPU Runtime                            │
│  raft (handle, linalg), rmm (memory), cuDF (DataFrames),             │
│  cuBLAS/cuSolver/cuSPARSE, CUDA kernels                              │
└──────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | Key Paths |
|-----------|----------------|-----------|
| libcuml++ public headers | C function/class declarations consumed by Cython | `cpp/include/cuml/` |
| libcuml++ algorithm impls | CUDA kernels + C++ algorithm bodies | `cpp/src/` |
| src_prims | Shared low-level math/linear algebra/matrix primitives (header-only) | `cpp/src_prims/` |
| Cython layer | Bridges Python to C++; manages raw GPU pointer extraction | `python/cuml/cuml/**/*.pyx` |
| CumlArray | Unified GPU array abstraction over cupy/cuDF/numba/numpy | `python/cuml/cuml/internals/array.py` |
| Base estimator | scikit-learn-compatible base class; owns handle, output_type, verbose | `python/cuml/cuml/internals/base.py` |
| InteropMixin | Bidirectional CPU↔GPU model conversion (`as_sklearn`/`from_sklearn`) | `python/cuml/cuml/internals/interop.py` |
| cuml.accel | Transparent acceleration of sklearn/hdbscan/umap via module proxying | `python/cuml/cuml/accel/` |
| Dask layer | Multi-GPU distributed wrappers using Dask+UCX | `python/cuml/cuml/dask/` |
| libcuml Python pkg | CMake build stub that finds/builds libcuml++ | `python/libcuml/` |

## Pattern Overview

**Overall:** C++/CUDA kernel library with a thin Cython binding layer, wrapped by a scikit-learn-compatible Python API, with an optional transparent acceleration overlay.

**Key Characteristics:**
- All compute happens in `raft::handle_t`-scoped CUDA operations; the handle carries the CUDA stream, stream pool, and device resources.
- Python estimators hold no GPU buffers themselves — they own `CumlArray` descriptors that point into device memory managed by RMM (RAPIDS Memory Manager).
- `cuml.accel` performs Python-level module swapping at import time: `sklearn.linear_model.LinearRegression` → `cuml` proxy, with CPU fallback on unsupported configurations.
- Multi-GPU scaling uses Dask (cuml.dask.*) with separate `*_mg.pyx` Cython files and MPI/UCX communicators.

## Layers

**C++ Public API Layer:**
- Purpose: Declares all algorithm entry points callable from Cython
- Location: `cpp/include/cuml/`
- Contains: One `.hpp` per algorithm category (e.g., `cuml/linear_model/glm.hpp`, `cuml/cluster/kmeans.hpp`)
- Depends on: `raft::handle_t`, RMM device pointers
- Used by: Cython `.pyx` files via `cdef extern from`

**C++ Implementation Layer:**
- Purpose: CUDA kernel implementations, algorithm logic
- Location: `cpp/src/`
- Contains: Per-algorithm subdirectory with `.cu`, `.cuh`, `.cpp` files (79 `.cu`, 101 `.cuh`, 2 `.cpp`)
- Depends on: `cpp/src_prims/`, raft, RMM, cuBLAS/cuSolver

**Primitives Layer:**
- Purpose: Shared header-only math primitives used across algorithms
- Location: `cpp/src_prims/`
- Contains: `linalg/`, `matrix/`, `random/`, `selection/`, `sparse/`, `timeSeries/`, `functions/`, `common/`
- Depends on: raft, CUDA runtime
- Used by: `cpp/src/` algorithm implementations

**Cython Binding Layer:**
- Purpose: Exposes C++ functions to Python; converts Python objects to raw GPU pointers
- Location: `python/cuml/cuml/**/*.pyx` (57 total `.pyx` files)
- Pattern: `cdef extern from "cuml/…"` → extracts `uintptr_t` from `CumlArray` → passes to C++ function
- Depends on: `pylibraft.common.handle_t`, `CumlArray`, `libcuml++`

**Python Estimator Layer:**
- Purpose: sklearn-compatible estimators; manage IO, output type, verbosity
- Location: `python/cuml/cuml/{algorithm_module}/` (pure `.py` + `.pyx`)
- Depends on: `cuml.internals.base.Base`, `CumlArray`, Cython bindings
- Used by: End users, cuml.accel proxies, Dask wrappers

**cuml.accel Acceleration Layer:**
- Purpose: Transparently replace sklearn/hdbscan/umap estimators with cuml GPU equivalents
- Location: `python/cuml/cuml/accel/`
- Contains: `accelerator.py` (module proxy machinery), `core.py` (install/enable), `estimator_proxy.py` (ProxyBase), `_overrides/` (per-package GPU replacements), `_patches/` (sklearn pipeline compatibility)
- Used by: `python -m cuml.accel myscript.py`, `cuml.accel.install()`, IPython magic

**Dask Multi-GPU Layer:**
- Purpose: Distributed multi-GPU training and inference
- Location: `python/cuml/cuml/dask/`
- Contains: Mirrors single-GPU structure (`cluster/`, `linear_model/`, `decomposition/`, etc.)
- Depends on: Single-GPU cuml estimators, Dask, `*_mg.pyx` Cython MG bindings, UCX

## Data Flow

### Primary Single-GPU Training Path

1. User calls `estimator.fit(X, y)` where `X` is numpy/pandas/cuDF/cupy array
2. `Base.fit` decorated with `@reflect(reset=True)` captures the input type for output mirroring
3. `input_utils.input_to_cuml_array()` converts input → `CumlArray` (device memory via RMM) (`python/cuml/cuml/internals/input_utils.py`)
4. Cython `.pyx` extracts raw `uintptr_t` device pointer from `CumlArray.ptr`
5. `get_handle()` returns thread-local `pylibraft.common.handle.Handle` (wraps `raft::handle_t`) (`python/cuml/cuml/internals/base.py:22`)
6. C++ function called with `handle_t&`, raw float*/double* pointers, and shape metadata (`cpp/src/{algorithm}/`)
7. CUDA kernels execute on the default CUDA stream owned by the handle
8. Result pointers written into pre-allocated `CumlArray` output buffers
9. `@reflect` decorator coerces output `CumlArray` → requested `output_type` (numpy/cuDF/pandas/cupy) via `CumlArray.to_output()` (`python/cuml/cuml/internals/outputs.py`)

### cuml.accel Transparent Acceleration Path

1. `cuml.accel.install()` called (or `python -m cuml.accel`) (`python/cuml/cuml/accel/core.py:164`)
2. `Accelerator.install()` replaces entries in `sys.modules` for listed packages with `AccelModule` proxies (`python/cuml/cuml/accel/accelerator.py`)
3. User code does `from sklearn.linear_model import LinearRegression` → gets cuml `ProxyBase` subclass
4. `ProxyBase.__fit__` tries GPU path first; if `UnsupportedOnGPU` raised, falls back to CPU sklearn
5. `InteropMixin.as_sklearn()` / `from_sklearn()` enable CPU↔GPU state transfer at any point (`python/cuml/cuml/internals/interop.py`)

### Multi-GPU Dask Path

1. User calls `cuml.dask.linear_model.LinearRegression(...).fit(dask_cudf_frame)`
2. Dask MG estimator partitions data and dispatches to workers via Dask scheduler
3. Each worker calls `*_mg.pyx` Cython binding with a `DeviceResourcesSNMG` handle (`python/cuml/cuml/internals/base.py:44`)
4. MG C++ functions (`cpp/src/{algo}/*_mg.cu`) communicate via NCCL/UCX

**State Management:**
- Estimator fitted parameters stored as `CumlArrayDescriptor` class-level descriptors that lazily convert between device/host on access (`python/cuml/cuml/common/array_descriptor.py`)
- Thread-local `_THREAD_STATE.handle` prevents CUDA stream conflicts across threads (`python/cuml/cuml/internals/base.py:19`)

## Key Abstractions

**`raft::handle_t` (C++) / `pylibraft.common.handle.Handle` (Python):**
- Purpose: Owns the CUDA stream, stream pool, and all device resource allocators. Every algorithm call receives a `const raft::handle_t&`.
- Python access: `cuml.internals.base.get_handle()` returns thread-local or explicit-stream handle
- Files: `python/cuml/cuml/internals/base.py`, `cpp/include/cuml/cluster/kmeans.hpp` (all public headers)

**`CumlArray`:**
- Purpose: Unified array abstraction — wraps cupy/cuDF/numba/numpy arrays, exposes `.ptr` (device pointer) and `.to_output(type)` for type conversion
- Files: `python/cuml/cuml/internals/array.py`

**`Base` estimator:**
- Purpose: scikit-learn-compatible base; owns `handle`, `output_type`, `verbose`; coordinates `@reflect` input/output type inference
- Files: `python/cuml/cuml/internals/base.py`

**`InteropMixin`:**
- Purpose: Declares `_cpu_class_path`, `_params_from_cpu`, `_params_to_cpu`, `_attrs_from_cpu`, `_attrs_to_cpu` for bidirectional CPU↔GPU model serialization
- Files: `python/cuml/cuml/internals/interop.py`

**`Accelerator` / `AccelModule`:**
- Purpose: Python import hook that replaces `sys.modules` entries; `AccelModule.__getattr__` checks caller module exclusion list before returning GPU override
- Files: `python/cuml/cuml/accel/accelerator.py`

**`ProxyBase`:**
- Purpose: A dynamically-generated `sklearn.BaseEstimator` subclass that delegates to cuml GPU estimator, with CPU fallback; created per wrapped class
- Files: `python/cuml/cuml/accel/estimator_proxy.py`

## Entry Points

**Build:**
- Location: `build.sh`
- Targets: `libcuml` (C++ shared lib), `cuml` (Python package), `prims`, `bench`, `cppdocs`, `pydocs`

**Python Package:**
- Location: `python/cuml/cuml/__init__.py`
- Imports algorithm modules and re-exports estimators

**cuml.accel CLI:**
- Location: `python/cuml/cuml/accel/__main__.py`
- Invocation: `python -m cuml.accel [--disable-uvm] [-m module | script.py]`
- Calls `cuml.accel.core.install()` then runs target

**cuml.accel programmatic:**
- Location: `python/cuml/cuml/accel/core.py:164`
- `cuml.accel.install()` — call before importing sklearn

**C++ Tests:**
- Location: `cpp/tests/sg/` (single-GPU), `cpp/tests/mg/` (multi-GPU)
- Built via `build.sh prims` or `build.sh libcuml`

## Architectural Constraints

- **CUDA stream ownership:** All algorithms must use the `handle_t`-owned stream. Algorithms must not create independent CUDA streams without registering them with the handle.
- **Memory management:** Device allocations go through RMM; raw `cudaMalloc` is not used in algorithm implementations.
- **Float32/float64 symmetry:** All C++ public API functions are overloaded for both `float` and `double`. Cython bindings call both via Python-level dtype dispatch.
- **Output type mirroring:** Methods decorated with `@reflect` must not return raw device arrays — they must go through `CumlArray.to_output()` to respect the configured `output_type`.
- **Global state:** Thread-local handle in `_THREAD_STATE` (`python/cuml/cuml/internals/base.py:19`); global output type in `cuml.internals.global_settings` (`python/cuml/cuml/internals/global_settings.py`).
- **Circular imports:** `cuml.internals.interop` uses local imports inside functions to avoid circular dependencies with `cuml.internals.base`.

## Anti-Patterns

### Direct sklearn import inside cuml.accel scope

**What happens:** A cuml internal module does `import sklearn.linear_model` after `cuml.accel` is installed.
**Why it's wrong:** The `_exclude_from_acceleration` function in `core.py:111` excludes `cuml.*` modules from acceleration, but importing sklearn from within cuml internals bypasses the proxy and may create inconsistent state.
**Do this instead:** cuml internals use `cuml.internals.interop.to_gpu/to_cpu` for conversion; sklearn is accessed only via `_cpu_class_path` lazy import in `InteropMixin._get_cpu_class()` (`python/cuml/cuml/internals/interop.py:90`).

### Raw device pointer without CumlArray

**What happens:** Passing a raw cupy pointer directly to a Cython binding without wrapping in `CumlArray`.
**Why it's wrong:** Memory order (C/F contiguous), dtype, and ownership are not tracked; strides may be incorrect.
**Do this instead:** Use `input_to_cuml_array()` (`python/cuml/cuml/internals/input_utils.py`) to produce a `CumlArray`, then extract `.ptr`.

## Error Handling

**Strategy:** C++ exceptions propagated through Cython's `except +` clause on all `cdef extern` declarations. Python layer catches and re-raises as standard Python exceptions.

**Patterns:**
- `UnsupportedOnGPU` / `UnsupportedOnCPU` (`python/cuml/cuml/internals/interop.py:44`) signal unsupported configurations in `cuml.accel` — triggers CPU fallback in `ProxyBase`
- Logging via `cuml.internals.logger` (wraps spdlog) for C++-level verbosity; separate `cuml.accel.core.Logger` for accel-layer messages

## Cross-Cutting Concerns

**Logging:** `cuml.internals.logger` (`python/cuml/cuml/internals/logger.pyx`) wraps spdlog; level controlled by `Base(verbose=…)` or `cuml.set_global_output_type`. `cuml.accel` has its own `Logger` class at `python/cuml/cuml/accel/core.py:17`.
**Validation:** `cuml.internals.validation.check_inputs()` (`python/cuml/cuml/internals/validation.py`) normalizes and validates input arrays at the Python boundary before Cython dispatch.
**NVTX Profiling:** `cuml.internals.nvtx` decorators (`python/cuml/cuml/internals/nvtx.py`) wrap key methods for Nsight Systems visibility.
**Output type configuration:** Global default set via `cuml.set_global_output_type()`; per-estimator via `Base(output_type=…)`; per-call via `using_output_type()` context manager (`python/cuml/cuml/internals/outputs.py`).

---

*Architecture analysis: 2026-06-11*
