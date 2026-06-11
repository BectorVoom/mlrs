# Coding Conventions

**Analysis Date:** 2026-06-11

---

## License Headers

Every source file (`.cpp`, `.cu`, `.cuh`, `.hpp`, `.h`, `.py`, `.pyx`, `.pxd`, `.sh`, `.cmake`,
`CMakeLists.txt`, `pyproject.toml`) must begin with an SPDX header. The
`verify-copyright` pre-commit hook enforces this automatically.

```cpp
// C++ / CUDA
/*
 * SPDX-FileCopyrightText: Copyright (c) <year(s)>, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
```

```python
# Python / Cython / shell
# SPDX-FileCopyrightText: Copyright (c) <year(s)>, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
```

Third-party vendored code in `python/cuml/cuml/_thirdparty/` uses `BSD-3-Clause`
or a dual `Apache-2.0 AND BSD-3-Clause` identifier — enforced by dedicated
`verify-copyright` hook entries in `.pre-commit-config.yaml`.

---

## C++ / CUDA Conventions

### Tooling

| Tool | Config | Invocation |
|---|---|---|
| clang-format v20 | `cpp/.clang-format` | pre-commit `clang-format` hook |
| clang-tidy | `cpp/.clang-tidy` | `ci/run_clang_tidy.sh` |
| cmake-format / cmake-lint | via `cpp/scripts/run-cmake-format.sh` | pre-commit hooks |
| include-check | `cpp/scripts/include_checker.py` | pre-commit hook |

### Formatting

Config file: `cpp/.clang-format` (based on Google style, customised).

| Setting | Value |
|---|---|
| Standard | C++20 |
| Column limit | 100 |
| Indent width | 2 spaces |
| Tab | Never (spaces only) |
| Pointer alignment | Left (`T* ptr`) |
| Brace style | WebKit (`BraceWrapping`) |
| Max empty lines | 1 |
| Template declarations | Always break before `<` |
| Constructor initializers | One per line or all on one line |

Include ordering (enforced by `IncludeBlocks: Regroup`, descending priority):
1. Quoted includes (`"..."`) — priority 1
2. Benchmark/test local headers (`<common/…>`, `<benchmarks/…>`, `<tests/…>`) — priority 2
3. `<cuml/…>` — priority 3
4. Other RAPIDS (`<cudf/…>`, `<raft/…>`, `<kvikio/…>`) — priority 4
5. RMM (`<rmm/…>`) — priority 5
6. CCCL / CUDA (`<thrust/…>`, `<cub/…>`, `<cuda/…>`) — priority 6
7. System includes with a `.` in the name — priority 7
8. STL includes (no `.`) — priority 8

### Naming (from `cpp/.clang-tidy`)

| Entity | Convention | Example |
|---|---|---|
| Classes, structs | `CamelCase` | `DbscanTest`, `PcaInputs` |
| Typedefs, type aliases | `CamelCase` | `IdxT`, `HandleType` |
| Enums | `CamelCase` | `EpsNnMethod` |
| Enum constants | `CamelCase` with `k` prefix | `kBruteForce` |
| Functions | `CamelCase` | `basicTest`, `runBenchmark` |
| Namespaces | `lower_case` | `ML`, `Bench` |
| Member variables (public) | `lower_case` | `n_row`, `cluster_std` |
| Member variables (private/protected) | `lower_case` with `_` suffix | `handle_`, `stream_` |
| `constexpr` / `static const` variables | `CamelCase` with `k` prefix | `kMaxBatchSize` |
| Template type parameters | `CamelCase` | `T`, `IdxT` |

Note: `readability-identifier-naming` is present in `.clang-tidy` but is
disabled in the active `Checks` line (`-readability-identifier-naming`). Naming
patterns above reflect the actual codebase conventions visible in source files.

### clang-tidy Checks

Active check groups: `clang-diagnostic-*`, `clang-analyzer-*`.
Disabled: `modernize-*`, `readability-identifier-naming`,
`clang-diagnostic-#pragma-messages`, `clang-diagnostic-switch`.
`WarningsAsErrors: '*'` — all active checks are errors.
`.cu` files and `_deps/` are excluded via `[tool.run-clang-tidy]` in
`pyproject.toml`.

### Error Handling

| Macro | Purpose |
|---|---|
| `RAFT_EXPECTS(cond, msg)` | Precondition check — throws `raft::exception` |
| `RAFT_FAIL(msg)` | Unconditional failure |
| `RAFT_CUDA_TRY(call)` | Checks CUDA API return code |

Usage pattern (from `cpp/src/knn/knn.cu`, `cpp/src/glm/qn_mg.cu`):
```cpp
RAFT_EXPECTS(input_data.size() == 1, "Expected single partition");
RAFT_FAIL("Unrecognized index type.");
RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));
```

`DeprecationWarning` in C++ user-facing headers is expressed as compile-time
`[[deprecated("message")]]` attributes — not RAFT macros.

### Logging

Use `cuml::internals::logger` (Cython) or `CUML_LOG_*` macros in C++ with
log-level constants (`level_enum`). Do not use `printf` or `std::cout` in
library code.

---

## Python / Cython Conventions

### Tooling

| Tool | Config | Invocation |
|---|---|---|
| ruff (lint + format) | `pyproject.toml [tool.ruff]` | pre-commit `ruff-check` / `ruff-format` |
| isort | `python/cuml/pyproject.toml [tool.isort]` | pre-commit `isort` hook |
| black | `python/cuml/pyproject.toml [tool.black]` | legacy config; ruff-format is the active formatter |
| cython-lint | `pyproject.toml [tool.cython-lint]` | pre-commit `cython-lint` hook |
| codespell | `pyproject.toml [tool.codespell]` | pre-commit `codespell` hook |
| shellcheck | — | pre-commit `shellcheck` hook (`--severity=warning`) |

Run all checks:
```bash
pre-commit run --all-files
```

### Formatting

| Setting | Value |
|---|---|
| Line length (Python) | 79 characters (`[tool.ruff] line-length = 79`) |
| Line length (Cython `.pyx`/`.pxd`) | 95 characters (`[tool.cython-lint]`) |
| Target Python | 3.11+ (`requires-python = ">=3.11"`) |
| isort profile | `black` |

Ruff extends lint rules to `*.pyx` and `*.pxd` files, but skips
`E999`, `E225`, `E226`, `E227` for Cython syntax compatibility.
`F401` (unused import) is suppressed in `__init__.py` files.
`_thirdparty/` directories are excluded from ruff, isort, and copyright checks.
`_stop_words.py` is excluded from ruff-format.

### Import Organisation

Order enforced by isort (`profile = "black"`):
1. Standard library imports
2. Third-party imports (`numpy`, `cupy`, `cudf`, `sklearn`, …)
3. Local / cuML imports (`from cuml.internals…`, `from cuml.common…`)

Example (from `python/cuml/cuml/cluster/dbscan.pyx`):
```python
import cupy as cp

from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals import logger, reflect
from cuml.internals.array import CumlArray
from cuml.internals.base import Base, get_handle
```

### Docstrings

Use **NumPy docstring format** for all public API. The `generate_docstring`
decorator from `cuml.common.doc_utils` auto-generates common parameter blocks.
`numpydoc<1.9` is pinned as a test dependency (doc validation).

Pattern (from `python/cuml/cuml/cluster/dbscan.pyx`):
```python
@generate_docstring(skip_parameters_heading=True)
def fit(self, X, y=None, sample_weight=None, out_dtype="int32"):
    """
    Perform DBSCAN clustering.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        ...
    """
```

### Type Hints

Python 3.11+ type hints are used for new internal utilities and public methods.
Cython `.pyx` files use Cython-static typing (`cdef`, `cpdef`) for performance-
critical paths; pure-Python type annotations are not added to `.pyx` files.

### Warning Policy

**Never use `DeprecationWarning`** in new code.
Use `FutureWarning` for deprecations of public API (enforced by pre-commit
`no-deprecationwarning` hook — pattern match on `DeprecationWarning[,)]` in
`.py` and `.pyx` files outside `tests/`).

Files currently using `FutureWarning`:
- `python/cuml/cuml/fil/compat.py`
- `python/cuml/cuml/internals/validation.py`

---

## Estimator / Mixin Pattern

All cuML Python estimators follow the scikit-learn estimator protocol via
cuML's internal base classes.

**Primary base class:** `cuml.internals.base.Base`
(`python/cuml/cuml/internals/base.py`)

`Base` inherits from `TagsMixin` and defines:
- Thread-local `raft.handle_t` management via `get_handle()`
- `output_type` routing (numpy / cudf / cupy / numba, etc.)
- `verbose` / logging-level forwarding
- `_get_param_names()` for sklearn parameter discovery

**Mixins** (`python/cuml/cuml/internals/mixins.py`):

| Mixin | Purpose |
|---|---|
| `TagsMixin` | `_get_tags()` — collects static tags via MRO traversal |
| `RegressorMixin` | Adds `score(X, y)` with R² metric |
| `ClassifierMixin` | Adds `score(X, y)` with accuracy metric |
| `ClusterMixin` | Adds `fit_predict(X)` |
| `FMajorInputTagMixin` | Tags preferred input as row-major (C order) |
| `CMajorInputTagMixin` | Tags preferred input as column-major (F order) |
| `SparseInputTagMixin` | Tags estimator as accepting sparse input |
| `StringInputTagMixin` | Tags estimator as accepting string input |
| `AllowNaNTagMixin` | Tags estimator as tolerating NaN input |
| `StatelessTagMixin` | Tags estimator as stateless |
| `InteropMixin` | CPU ↔ GPU interop (`to_cpu()` / `to_gpu()`) via `python/cuml/cuml/internals/interop.py` |

**Static tags** are contributed by `_more_static_tags()` class methods;
dynamic tags by `_get_tags()` instance methods (same name, different dispatch
via `_tags_class_and_instance` descriptor).

**Typical estimator declaration:**
```python
class DBSCAN(Base, ClusterMixin, CMajorInputTagMixin, InteropMixin):
    ...
    @staticmethod
    def _more_static_tags():
        return {"preferred_input_order": "C", "allow_nan": False}
```

---

## Cython Conventions

- `.pyx` files contain the Python-facing class and call C++ via `cdef extern`
  blocks.
- `.pxd` files contain pure declarations; **do not use `import` in `.pxd`
  files** — only `cimport` (enforced by `no-import-in-pxd` pre-commit hook).
- `cdef extern from "..."` blocks replicate C++ function signatures with
  Cython-compatible types (`handle_t&`, `uintptr_t`, `int64_t`, …).
- C++ float/double overloads are both declared in `cdef extern` blocks and
  dispatched at runtime from Python based on input dtype.
- Avoid `except +` on internal-only `cdef` functions; use `except +` on
  functions that can throw C++ exceptions that must propagate to Python.

---

## CMake Conventions

- cmake-format and cmake-lint run via `cpp/scripts/run-cmake-format.sh`
  (pre-commit hooks).
- Excludes `thirdparty/` paths.
- Format config pulled from `rapids-cmake` branch at CI time.

---

*Convention analysis: 2026-06-11*
