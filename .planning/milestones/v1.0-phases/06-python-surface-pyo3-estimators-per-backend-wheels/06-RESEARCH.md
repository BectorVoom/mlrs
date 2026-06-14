# Phase 6: Python Surface — PyO3 Estimators & Per-Backend Wheels - Research

**Researched:** 2026-06-13
**Domain:** PyO3 stable-ABI extension modules, maturin multi-distribution wheel packaging, Arrow PyCapsule FFI ingest, sklearn-compatible Python shim, sklearn ≥1.6 estimator-checks
**Confidence:** HIGH on the version/compat pins and the maturin/PyO3/arrow patterns (verified against current crates.io + official docs + the local cubecl source tree); MEDIUM on the exact estimator_checks subset (sklearn behavior verified from docs, but per-estimator pass/fail is empirical and must be discovered at Wave 0).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01: Thin Python shim over a compiled core.** Compiled PyO3 extension (working name `_mlrs`) exposes low-level entry points; the importable `mlrs` package is **pure Python**, each estimator subclassing `sklearn.base.BaseEstimator` + appropriate mixins, delegating compute to `_mlrs`. `get_params`/`set_params`/`clone`/`__repr__` come from sklearn; numpy↔Arrow glue + `output_type` routing live in Python. The wheel ships Python source alongside the abi3 extension.
- **D-02: Hybrid ingress — cuML-style API surface, Arrow PyCapsule boundary.** The shim accepts the cuML-style range of inputs (numpy / pyarrow / lists), normalizes to a **contiguous 1-D pyarrow float array** (row-major flatten of `X`), crosses via `__arrow_c_array__` PyCapsule + explicit `(rows, cols)` tuple. Rust imports the capsule via arrow-rs FFI (release-callback ownership — no bare `&[u8]` borrow) and **reuses the existing `validate_f32`/`validate_f64` bridge unchanged**. Adds `pyarrow` as a runtime dependency (accepted).
- **D-03: Egress — adopt cuML's `output_type` routing.** Configurable `output_type` constructor param + a global override, default `"input"` = mirror the container the data arrived in, via a `to_output(output_type)` equivalent. v1 supported set is **numpy + pyarrow only**. `labels_`/neighbor indices materialize as **int32**. Rust returns host buffers (`Vec<F>`/`Vec<i32>`) + shape; the shim wraps to the resolved container.
- **D-04: f64 on an f64-incapable backend → capability-query + clear error.** Extension exposes a capability flag (built on `capability.rs`). Passing float64 to a backend that cannot run it (notably `mlrs-rocm`) raises a clear Python exception. Never silently downcast.
- **D-05: Preserve input float dtype; non-float defaults to f64 where supported.** f32-in→f32-out; f64-in→f64-out. Integer/list/other inputs default to float64 on f64-capable backends, float32 on f64-incapable backends (rocm).
- **D-06: Internal dtype dispatch via an enum on the Arrow array dtype.** A `#[pyclass]` cannot be generic over `F`; inspect the incoming pyarrow float type and dispatch `Estimator<f32>` vs `Estimator<f64>` via an internal enum. The shim does not expose `fit_f32`/`fit_f64`. (Exact enum/wrapper shape = Claude's discretion.)
- **D-07: Constant `import mlrs`, distinct distribution names.** Every backend wheel exposes the same top-level `import mlrs`. Distribution names differ: `mlrs-cpu`/`mlrs-wgpu`/`mlrs-cuda`/`mlrs-rocm`. A user installs **exactly one** backend wheel (two would overwrite each other); document + guard against double-install where feasible.
- **D-08: Import-time driver probe + clear error.** On `import mlrs`, probe for the backend driver / attempt cubecl client init; if absent, raise **`ImportError`** with a clear, actionable message. Fails fast before `fit()`.
- **D-09: `abi3-py312` stable ABI** (locked by criterion 4) — one wheel per backend covers Python ≥ 3.12.

### Carried forward (reaffirmed, not re-decided)
- Wrapped surface is the `mlrs-algos` trait set: `Fit` (returns `&mut self`), `Predict` (regressors), `Transform`/`inverse_transform` (PCA/TruncatedSVD), `PredictLabels` (clustering/classifier, i32 labels), `KNeighbors` (distances + i32 indices), `PredictProba` (per-class fractions). i32 everywhere for labels/indices → numpy `int32` at egress.
- Rust estimators take an explicit `&mut BufferPool<ActiveRuntime>` and `(rows, cols)` per call; fitted state device-resident, host-materialized at accessors. The PyO3 layer owns the pool + client and releases the GIL around device compute.
- `mlrs-py` is the single cdylib + global allocator site (mimalloc, FOUND-09, already wired). Source/test separation per AGENTS.md (tests in `crates/mlrs-py/tests/` + Python `pytest`). `thiserror` in libs / `anyhow` at the binding boundary. Deps track latest.
- `ActiveRuntime` is feature-selected (exactly one of cpu/wgpu/cuda/rocm) — one source builds N wheels, one backend per wheel.
- Gate = cpu(f64) + rocm(f32); **f64-on-rocm skips-with-log**. cuda compiles only; wgpu opportunistic.

### Claude's Discretion
- Exact PyO3 wrapper/enum shape for dtype dispatch (D-06); module/file layout of the `mlrs` package and `_mlrs` extension; which sklearn mixins each estimator composes.
- BufferPool + cubecl client ownership/lifecycle across the boundary (process-global vs per-estimator) + thread-safety under `Python::detach`/joblib.
- The subset of `sklearn.utils.estimator_checks` treated as "relevant" per estimator family.
- Exact `get_params`/`set_params` hyperparameter names per estimator (must match sklearn).
- The maturin multi-distribution mechanism.
- `score()` metric per family (R² regressors, accuracy classifiers) — inherit from sklearn mixins.

### Deferred Ideas (OUT OF SCOPE)
- cupy/cuDF/numba output_type targets; full cuML array-interface/DLPack ingress; multiple backends installable side-by-side (distinct import names); `cuml.accel`-style transparent acceleration; multi-GPU / Dask Python surface.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PY-01 | All v1 estimators exposed as PyO3 `#[pyclass]` with sklearn-compatible `fit`/`predict`/`transform`/`score`, `fit` returns `self` | D-01 shim + per-family mixin map (§Architecture); `score` inherited from `RegressorMixin`/`ClassifierMixin` (§sklearn estimator_checks). The `#[pyclass]` is `_mlrs`-internal; the *public* `fit returns self` is the pure-Python shim returning `self`. |
| PY-02 | `get_params`/`set_params` + sklearn-named constructor hyperparameters | Per-estimator hyperparameter name map (§Hyperparameter Mapping); `get_params`/`set_params` come free from `BaseEstimator` given a sklearn-faithful `__init__` (§Pitfall: `__init__` purity). |
| PY-03 | NumPy/Arrow inputs cross via Arrow PyCapsule with correct ownership/lifetime (no bare `&[u8]` borrows); GIL released during compute | `pyo3-arrow` `PyArray` owned-arg ingest = release-callback-correct FFI (§Arrow PyCapsule); `Python::detach` releases GIL around `&mut BufferPool` compute (§PyO3 GIL). Reuses `validate_f32`/`validate_f64` (§bridge composition). |
| PY-04 | Per-backend wheels via maturin under distinct dist names; absent driver → clear error | maturin `[project].name` per-backend + `module-name="mlrs._mlrs"` (§Maturin Multi-Distribution); import-time `catch_unwind` probe → `ImportError` (§Import-Time Probe). |
| PY-05 | f32 + f64 inputs (runtime dtype dispatch), Python ≥ 3.12 | `AnyEstimator` enum dispatch on the pyarrow float type (D-06, §dtype Dispatch); `abi3-py312` (§PyO3 abi3). f64→rocm guarded by capability (D-04). |
</phase_requirements>

## Summary

Phase 6 is a **wrap-only** binding + packaging phase: the 11 `mlrs-algos` estimators already exist and pass their oracle gates; this phase adds (1) a PyO3 `abi3-py312` extension `_mlrs` inside the existing `mlrs-py` cdylib, (2) a pure-Python `mlrs` package of sklearn-compatible shim estimators, and (3) maturin per-backend wheels. There is **no algorithm work and no new numerics**.

The single highest-impact finding is a **version-compatibility lock**: `arrow` 59's `pyarrow` feature transitively pins **PyO3 0.28.3**, and `pyo3-arrow` 0.18 also targets PyO3 0.28. Because exactly one PyO3 version may link into a cdylib, the entire binding stack **must use PyO3 0.28**, *not* the latest 0.29 — even though the workspace policy is "track latest." This is the one place that policy is overridden by a hard ABI constraint; it is called out as a locked-by-compat decision in `## Project Constraints`. Verify before any other Phase-6 code lands.

The two ROADMAP-flagged unknowns are both resolved with concrete, documented patterns. **Maturin multi-distribution**: set the distribution name from PEP-621 `[project].name` (which maturin prefers over the cargo package name) and the import module from `[tool.maturin] module-name = "mlrs._mlrs"`; generate one `pyproject.toml` per backend (templated, distinct `name`) over the single `mlrs-py` crate built with `--features <backend>`. **Arrow PyCapsule import**: take an owned `pyo3_arrow::PyArray` argument in the `#[pyfunction]`/`#[pymethods]` — it consumes the `__arrow_c_array__` capsule with correct release-callback ownership via arrow-rs FFI, yielding an `ArrayRef` that downcasts to `Float32Array`/`Float64Array` and feeds the existing `validate_f32`/`validate_f64` bridge unchanged. A third decisive finding: cubecl's `Runtime::client()` returns `ComputeClient` **directly (not `Result`) and `.unwrap()`s internally**, so a missing driver **panics**; the D-08 import probe must wrap `active_client()` in `std::panic::catch_unwind` and translate a caught panic into a Python `ImportError`.

**Primary recommendation:** Pin **PyO3 0.28 + arrow 59 (pyarrow feature) + pyo3-arrow 0.18 + maturin 1.14 + abi3-py312** as the binding stack; structure as `python-source` pure-Python `mlrs/` package + `_mlrs` extension via `module-name="mlrs._mlrs"`; dispatch f32/f64 through a per-estimator `AnyEstimator` enum; release the GIL with `Python::detach`; probe the driver at import via `catch_unwind`; and use **one process-global `BufferPool`+client behind a `Mutex`** owned by the module.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| sklearn `BaseEstimator` semantics (`get_params`/`set_params`/`clone`/`__repr__`/tags) | Pure-Python shim (`mlrs/`) | — | D-01: sklearn gives these free from a faithful `__init__`; re-implementing in Rust is brittle. |
| numpy/list → pyarrow normalization (ingress surface, D-02) | Pure-Python shim | — | cuML-style accept-anything lives in Python; only a flat pyarrow float array crosses the boundary. |
| `output_type` mirror routing (egress, D-03) | Pure-Python shim | — | Host `Vec<F>`/`Vec<i32>`+shape returned by Rust; Python wraps to numpy/pyarrow. |
| Arrow PyCapsule FFI import + validation | `_mlrs` (Rust/PyO3) | `mlrs-backend::bridge` | PY-03: capsule consumed Rust-side via pyo3-arrow; reuses `validate_f32/f64`. |
| dtype dispatch f32 vs f64 (D-06) | `_mlrs` (Rust) | — | `#[pyclass]` can't be generic over `F`; an internal enum dispatches. |
| Device compute (`fit`/`predict`/…) + GIL release | `_mlrs` (Rust) | `mlrs-algos` traits | PY-03: `Python::detach` around the trait calls. |
| Capability query (f64-on-rocm guard, D-04) | `_mlrs` (Rust) | `mlrs-backend::capability` | Surfaces `supports_type(F64)` as a flag; raises a Python exception. |
| Import-time driver probe (D-08) | `_mlrs` module init | `mlrs-backend::runtime` | `catch_unwind(active_client)` → `ImportError`. |
| BufferPool + cubecl client lifecycle | `_mlrs` module-global state | `mlrs-backend::pool` | Process-global behind a `Mutex` (recommendation below). |
| Per-backend wheel build + naming | maturin / pyproject (build) | Cargo features | PY-04: `[project].name` per backend, `--features <backend>`. |
| `#[global_allocator]` (mimalloc) | `mlrs-py` cdylib | — | FOUND-09, already wired; stays the single allocator site. |

## Project Constraints (from CLAUDE.md / AGENTS.md)

- **Source/test separation (AGENTS.md §2):** No in-source `#[cfg(test)] mod tests`. Rust tests live in `crates/mlrs-py/tests/`; Python tests in a `pytest` tree (e.g. `python/tests/` or `crates/mlrs-py/python/tests/`). This is a HARD rule.
- **CubeCL generics-over-float (AGENTS.md §3):** No new kernels in this phase, but any Rust touching the estimator generics stays `<F: Float + CubeElement + Pod>`.
- **Error handling (CLAUDE.md / memory):** `thiserror` in library crates, `anyhow` at the binding boundary. PyO3 errors map a boundary `anyhow`/typed error → `PyErr` via `From` (§Error Mapping).
- **Deps track latest — OVERRIDDEN here by a hard ABI pin:** the workspace policy is "track latest" (Cargo.toml comment), but PyO3 **must** be 0.28 (not 0.29) to match arrow 59's pyarrow feature and pyo3-arrow 0.18. Document this exception explicitly in `crates/mlrs-py/Cargo.toml`.
- **CubeCL build-error protocol (AGENTS.md §4):** if a cubecl build error appears, read `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md` before any fix.
- **Workspace single-source pins (Cargo.toml `[workspace.dependencies]`):** add `pyo3`, `pyo3-arrow`, and `arrow`'s pyarrow feature wiring there; `mlrs-py` references them with `workspace = true`.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `pyo3` | **0.28** | Rust↔Python bindings, `#[pyclass]`/`#[pymethods]`/`#[pymodule]`, `abi3-py312`, `Python::detach` GIL release | The canonical PyO3 binding crate; 0.28 is the version arrow 59 + pyo3-arrow 0.18 are built against (ABI pin). `[VERIFIED: crates.io + arrow 59 cargo-tree]` |
| `arrow` | **59** (feature `pyarrow`) | Arrow types (`Float32Array`/`Float64Array`) + the `pyarrow` FFI bridge; already a workspace dep | Already pinned in `[workspace.dependencies]` (arrow = "59"); the `pyarrow` feature adds the PyO3 0.28 bridge. `[VERIFIED: workspace Cargo.toml + cargo tree]` |
| `pyo3-arrow` | **0.18** | Owned-arg `PyArray` ingest of `__arrow_c_array__` capsules with correct release-callback ownership; buffer-protocol numpy fallback | Purpose-built zero-copy PyCapsule consumer; targets PyO3 0.28, matching the pin. `[VERIFIED: crates.io 0.18.0 + docs.rs]` |
| `maturin` | **1.14** | Build backend: compiles `mlrs-py` cdylib → abi3 wheel; per-backend dist naming | The standard PyO3 build/publish tool; 1.14 is current and supports `module-name` + PEP-621 `[project].name` precedence. `[VERIFIED: pip index versions]` |
| `mimalloc` | 0.1 (already wired) | `#[global_allocator]` in the cdylib | FOUND-09 — unchanged, single allocator site. `[VERIFIED: existing source]` |
| `anyhow` | 1 (already wired) | Boundary error handling → `PyErr` | D-10 / CLAUDE.md; already in `mlrs-py`. `[VERIFIED: existing Cargo.toml]` |

### Supporting (Python runtime / build)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `scikit-learn` | ≥1.6 (test+runtime) | `BaseEstimator`+mixins the shim subclasses; `estimator_checks` | Runtime dep of the `mlrs` package (D-01 shim inherits from it) + test dep. `[CITED: scikit-learn.org dev/develop]` |
| `pyarrow` | latest | Ingress normalization target (D-02) + arrow-out egress (D-03) | Accepted runtime dep of `mlrs`. `[ASSUMED — version unpinned; confirm a floor at planning]` |
| `numpy` | ≥1.26 (runtime) | Default in/out container; the buffer-protocol path | Runtime dep; pyo3-arrow consumes numpy via buffer protocol zero-copy. `[ASSUMED]` |
| `pytest` | latest (test) | Python oracle + estimator-checks harness | Test-only. |

> **Do NOT add the `numpy` *Rust* crate** (`numpy = "0.29"`): it is unneeded. Ingress crosses as Arrow (D-02), and egress returns host `Vec<F>` that the Python shim wraps. Adding it would also drag a *second* PyO3 version (0.29) into the link — the exact ABI conflict to avoid.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `pyo3-arrow` `PyArray` | `arrow::pyarrow::FromPyArrow` directly (`ArrayData::from_pyarrow_bound`) | arrow's own `pyarrow` module also consumes the capsule correctly and needs **no extra crate**. Tradeoff: pyo3-arrow adds the numpy-buffer-protocol fallback and a cleaner owned-arg ergonomic; arrow-native requires the input already be a pyarrow object. **Recommendation:** since the shim *normalizes to a pyarrow array anyway (D-02)*, `arrow::pyarrow` alone is sufficient and drops a dependency — prefer it unless the buffer-protocol fallback proves valuable. Pick one at planning; both are PyO3-0.28-correct. |
| PyO3 0.29 (latest) | — | Rejected: incompatible with arrow 59 pyarrow + pyo3-arrow 0.18 (would link two PyO3 versions). The "track latest" policy yields to the ABI constraint. |
| `setuptools-rust` | maturin | maturin is the standard for abi3 + simple wheel layout; setuptools-rust is heavier and unneeded. |

**Installation (Cargo side — `crates/mlrs-py/Cargo.toml`, deps via workspace):**
```toml
# crates/mlrs-py/Cargo.toml
[dependencies]
pyo3 = { workspace = true, features = ["abi3-py312", "extension-module"] }
arrow = { workspace = true, features = ["pyarrow"] }   # or pyo3-arrow = { workspace = true }
anyhow = { workspace = true }
mimalloc = { workspace = true }
mlrs-algos = { path = "../mlrs-algos" }
mlrs-backend = { path = "../mlrs-backend" }   # for capability/runtime/bridge/pool

[features]
cpu  = ["mlrs-backend/cpu",  "mlrs-algos/cpu"]
wgpu = ["mlrs-backend/wgpu", "mlrs-algos/wgpu"]
cuda = ["mlrs-backend/cuda", "mlrs-algos/cuda"]
rocm = ["mlrs-backend/rocm", "mlrs-algos/rocm"]
```
```toml
# [workspace.dependencies] additions (single-source pins)
pyo3 = { version = "0.28", default-features = false }   # 0.28 PINNED for arrow/pyo3-arrow ABI compat
# pyo3-arrow = "0.18"   # if chosen over arrow::pyarrow
```

**Version verification:** maturin 1.14.0 (pip index versions, 2026-06-13). pyo3 latest crates.io = 0.29.0 but **arrow 59 `pyarrow` feature pulls pyo3 0.28.3** (verified via `cargo tree -p arrow --features pyarrow`). pyo3-arrow 0.18.0 (cargo search). pyo3 0.28 exposes `abi3-py312` (verified via `cargo add pyo3@0.28 --features abi3-py312 --dry-run`).

## Package Legitimacy Audit

> The `gsd-tools query package-legitimacy` seam is not on PATH in this environment (consistent with the project-memory note that gsd query verbs are stubbed in this build). Verdicts below are from manual verification against authoritative sources (crates.io, the PyO3/Apache GitHub orgs, and the local cubecl source tree). The planner should treat the two `[ASSUMED]` Python deps as requiring a `checkpoint:human-verify` floor-version pin.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `pyo3` 0.28 | crates.io | 7+ yrs | very high | github.com/PyO3/pyo3 | OK | Approved |
| `arrow` 59 (pyarrow) | crates.io | 5+ yrs | very high | github.com/apache/arrow-rs | OK | Approved (already a workspace dep) |
| `pyo3-arrow` 0.18 | crates.io | ~2 yrs | moderate | github.com/kylebarron/arro3 (geoarrow ecosystem) | OK | Approved (optional — see Alternatives) |
| `maturin` 1.14 | PyPI | 6+ yrs | very high | github.com/PyO3/maturin | OK | Approved |
| `scikit-learn` ≥1.6 | PyPI | established | very high | github.com/scikit-learn/scikit-learn | OK | Approved |
| `pyarrow` | PyPI | established | very high | github.com/apache/arrow | OK | Approved (floor version `[ASSUMED]` — pin at planning) |
| `numpy` | PyPI | established | very high | github.com/numpy/numpy | OK | Approved (floor `[ASSUMED]`) |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```text
  ┌─────────────────────────── Python process ───────────────────────────┐
  │                                                                       │
  │  user code:  import mlrs;  m = mlrs.KMeans(n_clusters=3); m.fit(X)     │
  │                              │                                         │
  │            ┌─────────────────▼──────────────────┐  PURE-PYTHON SHIM    │
  │            │  mlrs/  (sklearn BaseEstimator +    │  (D-01)              │
  │            │  RegressorMixin/ClassifierMixin/…)  │                      │
  │            │  • accept numpy/pyarrow/list (D-02 surface)               │
  │            │  • normalize X → contiguous 1-D pyarrow float array       │
  │            │  • resolve output_type (D-03 mirror)                      │
  │            │  • get_params/set_params/clone (free from sklearn)        │
  │            └─────────────────┬──────────────────┘                      │
  │           pyarrow float array (__arrow_c_array__ capsule) + (rows,cols)│
  │                              │  ┌─ raises: ImportError@import (D-08),   │
  │                              ▼  │  ValueError(f64-on-rocm) (D-04)       │
  │            ┌─────────────────────────────────────┐  _mlrs EXTENSION    │
  │            │  _mlrs  (PyO3 abi3-py312 #[pymodule])│  (Rust, in mlrs-py) │
  │            │  • PyArray owned arg → arrow-rs FFI  │  release-cb owned   │
  │            │    (PY-03 no bare &[u8] borrow)      │                     │
  │            │  • downcast Float32Array/Float64Array│                     │
  │            │  • validate_f32 / validate_f64 ◄─────┼── REUSED bridge.rs  │
  │            │  • enum AnyKMeans{F32(..),F64(..)}   │  dtype dispatch D-06 │
  │            │  • Python::detach { ... device ... } │  GIL released PY-03  │
  │            └─────────────────┬──────────────────┘                      │
  │                              │ DeviceArray::from_host + (rows,cols)     │
  │   module-global  Mutex<BufferPool<ActiveRuntime>> + client (owns GPU)   │
  │                              ▼                                          │
  │            ┌─────────────────────────────────────┐                     │
  │            │  mlrs-algos traits (Fit/Predict/…)   │  UNCHANGED          │
  │            │  → mlrs-backend prims → cubecl kernels│  device compute     │
  │            └─────────────────┬──────────────────┘                      │
  │     fitted state device-resident; .coef(pool)/labels_ → Vec<F>/Vec<i32> │
  │                              │ host buffers + shape                     │
  │            back up to the shim → to_output(numpy | pyarrow) (D-03)      │
  └───────────────────────────────────────────────────────────────────────┘

  BUILD:  one mlrs-py crate ─ maturin build --features <backend> ─┬─► mlrs-cpu  (abi3 wheel, import mlrs)
          per-backend pyproject.toml ([project].name differs)     ├─► mlrs-wgpu
          module-name = "mlrs._mlrs"                              ├─► mlrs-cuda
                                                                  └─► mlrs-rocm
```

### Recommended Project Structure
```
crates/mlrs-py/
├── Cargo.toml                # pyo3 0.28 + arrow pyarrow + features cpu/wgpu/cuda/rocm
├── src/
│   ├── lib.rs                # #[pymodule] _mlrs; module-init driver probe (D-08); global pool
│   ├── allocator.rs          # mimalloc (FOUND-09, unchanged)
│   ├── ingress.rs            # PyArray/capsule → Float32/64Array → validate → DeviceArray
│   ├── egress.rs             # Vec<F>/Vec<i32> + shape → returned to Python (numpy/arrow wrap is shim-side)
│   ├── capability.rs         # capability flag + f64-on-rocm guard → PyErr (D-04)
│   ├── dispatch.rs           # AnyEstimator enum macro (D-06)
│   └── estimators/           # one #[pyclass] wrapper per estimator (or macro-generated)
│       ├── kmeans.rs  ...     # 11 wrappers
├── python/                   # python-source for maturin (pure-Python shim, D-01)
│   └── mlrs/
│       ├── __init__.py       # re-export estimators; triggers _mlrs import → probe
│       ├── _io.py            # numpy/list→pyarrow normalize (D-02), output_type (D-03)
│       ├── base.py           # MlrsBase(BaseEstimator) — output_type, _to_output
│       ├── linear.py         # LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
│       ├── cluster.py        # KMeans, DBSCAN
│       ├── decomposition.py  # PCA, TruncatedSVD
│       └── neighbors.py      # NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
│   └── tests/                # pytest: oracle + estimator_checks (AGENTS.md §2 separation)
├── pyproject/                # per-backend templated pyproject.toml (cpu/wgpu/cuda/rocm)
└── tests/                    # Rust integration tests (allocator_test.rs lives here already)
```

### Pattern 1: Maturin Multi-Distribution (one crate → N dist names, one `import mlrs`)
**What:** maturin derives the **distribution name** from PEP-621 `[project].name` (it *prefers `pyproject.toml [project].name` over `Cargo.toml [package].name`*), and the **import module name** from `[tool.maturin] module-name`. Set `module-name = "mlrs._mlrs"` so the compiled extension is a submodule of the pure-Python `mlrs` package, regardless of the distribution name.
**When to use:** every per-backend wheel build.
**Example:**
```toml
# pyproject/cpu.pyproject.toml  (one per backend; only [project].name + the feature differ)
[build-system]
requires = ["maturin>=1.14,<2"]
build-backend = "maturin"

[project]
name = "mlrs-cpu"                 # ← DISTRIBUTION name (mlrs-wgpu / mlrs-cuda / mlrs-rocm in the others)
requires-python = ">=3.12"
dynamic = ["version"]            # pull version from Cargo.toml
dependencies = ["numpy>=1.26", "pyarrow>=14", "scikit-learn>=1.6"]

[tool.maturin]
manifest-path = "crates/mlrs-py/Cargo.toml"
module-name = "mlrs._mlrs"        # ← IMPORT module: always `mlrs`, extension at mlrs._mlrs
python-source = "crates/mlrs-py/python"   # pure-Python `mlrs/` package
features = ["cpu"]                # ← the ONLY backend-specific Cargo feature (wgpu/cuda/rocm in the others)
```
Build: `maturin build -m pyproject/cpu.pyproject.toml --release` (or copy the chosen file to `pyproject.toml` and `maturin build --features cpu` — same result). All four wheels expose `import mlrs`; their distribution names differ; installing two overwrites the shared `mlrs/` namespace (D-07 accepted constraint).
**Source:** maturin project_layout + metadata docs (`module-name`, `python-source`, `[project].name` precedence).

### Pattern 2: Arrow PyCapsule Ingest with Release-Callback Ownership (PY-03 / D-02)
**What:** accept an **owned** arrow-capsule type as the `#[pymethods]` argument; arrow-rs FFI takes ownership of the C `ArrowArray` (including its release callback) and produces an owned `ArrayRef`/`ArrayData` — never a borrow into a Python-owned buffer.
**When to use:** every `fit`/`predict`/`transform`/`kneighbors` entry that receives `X` (and `y`).
**Example (arrow-native path — drops the pyo3-arrow dep):**
```rust
// Source: arrow::pyarrow::FromPyArrow (arrow 59, feature = "pyarrow"); composes with bridge.rs
use arrow::array::{ArrayData, Float32Array, Float64Array, make_array};
use arrow::pyarrow::FromPyArrow;
use pyo3::prelude::*;
use mlrs_backend::bridge::{validate_f32, validate_f64};

#[pymethods]
impl PyKMeans {
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, rows: usize, cols: usize) -> PyResult<()> {
        // 1. Consume the __arrow_c_array__ capsule: arrow-rs owns the FFI array (release cb).
        let data = ArrayData::from_pyarrow_bound(x)?;     // owned ArrayData, no &[u8] borrow
        let array = make_array(data);                     // ArrayRef
        // 2. Reuse the EXISTING validated bridge unchanged (D-02). Dispatch on dtype (D-06).
        let result = py.detach(|| {                        // 3. release the GIL around device compute
            let mut pool = GLOBAL_POOL.lock().unwrap();
            match array.data_type() {
                arrow::datatypes::DataType::Float32 => {
                    let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
                    let validated = validate_f32(arr)?;    // offset/null/align hard-reject
                    self.inner.fit_f32(&mut pool, validated, (rows, cols))
                }
                arrow::datatypes::DataType::Float64 => {
                    // D-04: guard f64 against an f64-incapable backend BEFORE compute
                    let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
                    let validated = validate_f64(arr)?;
                    self.inner.fit_f64(&mut pool, validated, (rows, cols))
                }
                dt => return Err(unsupported_dtype(dt)),
            }
        });
        result.map_err(into_pyerr)
    }
}
```
> NOTE the existing `bridge::validate_f32/f64` *reject sliced/offset arrays* (it requires the values view to cover the whole backing buffer). The Python shim must therefore hand a **freshly-allocated contiguous** pyarrow array (the row-major flatten of `X`, D-02) — not a zero-copy slice of a larger numpy buffer. This is consistent with D-02's "normalize to a contiguous 1-D pyarrow float array."

### Pattern 3: dtype Dispatch via per-estimator enum (D-06)
**What:** a `#[pyclass]` can't be generic over `F`. Wrap both monomorphizations in an internal enum; pick the arm by the incoming pyarrow float type. Use a macro to avoid 11× boilerplate.
**Example:**
```rust
// Source: D-06 pattern; macro keeps it low-boilerplate across all 11 estimators
enum AnyKMeans { Unfit { n_clusters: usize, seed: u64 },
                 F32(mlrs_algos::cluster::KMeans<f32>),
                 F64(mlrs_algos::cluster::KMeans<f64>) }
// fit_f32 constructs the F32 arm from the stored hyperparameters; fit_f64 the F64 arm.
// predict_labels matches the fitted arm. A `macro_rules! any_estimator!` generates this
// for each (PyClass, AlgoType, traits) triple.
```
**When to use:** all 11 wrappers. **Recommendation:** drive it with a `macro_rules!` that takes the estimator type, its constructor params, and which traits it implements (Fit + one of Predict/Transform/PredictLabels/KNeighbors/PredictProba), emitting the enum + the `#[pymethods]`.

### Pattern 4: Import-Time Driver Probe → ImportError (D-08) — must catch a PANIC
**What:** `mlrs_backend::runtime::active_client()` calls `ActiveRuntime::client(&device)`, whose `Runtime::client` returns `ComputeClient` **directly (no `Result`) and `.unwrap()`s internally** (verified in `cubecl-hip-0.10.0/src/runtime.rs`). A missing/incompatible driver therefore **panics**, it does not return an error. The probe must `catch_unwind`.
**Example:**
```rust
// Source: cubecl-hip 0.10 runtime.rs (client() unwraps); D-08
#[pymodule]
fn _mlrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Probe ONCE at import; translate a missing driver into a clean ImportError.
    let probe = std::panic::catch_unwind(|| {
        let client = mlrs_backend::runtime::active_client();   // may panic if driver absent
        let _ = client.properties();                            // touch it to force init
    });
    if probe.is_err() {
        return Err(pyo3::exceptions::PyImportError::new_err(format!(
            "mlrs-{0} requires the {0} runtime/driver; none was detected. \
             Install the {0} driver or use a different mlrs backend wheel.",
            mlrs_backend::capability::active_backend_name()
        )));
    }
    // ... register the estimator #[pyclass]es, build the global pool, etc.
    Ok(())
}
```
> Cost/safety: a single client construction + `properties()` query at import is cheap (microseconds-to-low-ms on cpu; one device enumeration on rocm/cuda). It is a one-time import side-effect, acceptable per D-08. Wrap it in `catch_unwind` so a panic in the HIP/CUDA `.unwrap()` becomes `ImportError`, not a process abort. (`catch_unwind` requires the panic to unwind, not abort — ensure the cdylib profile does **not** set `panic = "abort"`; default is `unwind`, so this works.)

### Pattern 5: GIL Release with `Python::detach` (PY-03)
**What:** PyO3 renamed `allow_threads`→`detach` (and `with_gil`→`attach`) in 0.26 for free-threading terminology; 0.28 uses `Python::detach` (the old `allow_threads` remains as a deprecated alias). Wrap the device-compute closure so other Python threads run during the kernel.
**Note:** the closure passed to `detach` must be `Send` and must not touch Python objects. The `&mut BufferPool` and `DeviceArray` are plain Rust — fine. The module-global pool behind a `Mutex` (below) satisfies `Send`.

### Anti-Patterns to Avoid
- **Re-implementing `BaseEstimator`/`clone()`/`get_params` in Rust.** D-01 explicitly rejected this. Keep the `#[pyclass]` low-level; let the pure-Python shim inherit sklearn semantics.
- **Borrowing `&[u8]` out of a Python buffer for the lifetime of compute.** PY-03 forbids it. Take an *owned* arrow array; arrow-rs FFI owns the release callback.
- **Silent f64→f32 downcast on rocm.** D-04 forbids it — raise a clear exception (the 1e-5 contract trumps convenience).
- **Pulling the `numpy` Rust crate or PyO3 0.29.** Either drags a second PyO3 version into the cdylib link → ABI conflict.
- **A non-sklearn-faithful `__init__`** (storing transformed params, validating in `__init__`). Breaks `get_params`/`clone`/`set_params` and fails `estimator_checks`. Store every constructor arg verbatim as a same-named attribute; validate only in `fit`.
- **`panic = "abort"` in the wheel profile.** Would turn the D-08 probe panic into a hard crash instead of an `ImportError`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Consume `__arrow_c_array__` capsule | Manual `PyCapsule_GetPointer` + `FFI_ArrowArray::from_raw` + lifetime juggling | `arrow::pyarrow::FromPyArrow` (`ArrayData::from_pyarrow_bound`) or `pyo3_arrow::PyArray` owned arg | Release-callback ownership is subtle; the libraries get it right and arrow-rs is already a workspace dep. |
| sklearn `get_params`/`set_params`/`clone`/`__repr__`/tags | Hand-rolled introspection | subclass `sklearn.base.BaseEstimator` (+ mixins) in the pure-Python shim | D-01; sklearn gives these from a faithful `__init__`. |
| `score()` (R²/accuracy) | Hand-written metric | `RegressorMixin`/`ClassifierMixin` `.score()` | Inherited; matches sklearn semantics exactly. |
| Per-backend wheel naming | Custom build script renaming wheels | maturin `[project].name` + `module-name` | maturin's documented mechanism; renaming wheels by hand breaks metadata. |
| Multi-version Python ABI | Build one wheel per minor Python | `abi3-py312` | One wheel covers Python ≥3.12 (D-09). |
| Stable PyErr conversion | Manual error strings | `pyo3::exceptions::Py*::new_err` + `From<MyErr> for PyErr` | Maps boundary errors to the right Python exception type. |

**Key insight:** every "hard" part of this phase (capsule ownership, sklearn semantics, wheel naming, ABI) has a first-party library mechanism. The phase's real work is *gluing them in the right order with the right version pins*, not building primitives.

## Hyperparameter Mapping (PY-02 — sklearn-named constructor params per estimator)

> Verified against the Rust constructors in `crates/mlrs-algos/src/**`. These are the names the pure-Python `__init__` must expose (and store verbatim). Defaults match sklearn unless noted. `[VERIFIED: mlrs-algos source]` for the Rust side; `[CITED: scikit-learn]` for the sklearn default.

| Estimator | sklearn `__init__` params (mlrs v1 subset) | Rust constructor | Notes |
|-----------|---------------------------------------------|------------------|-------|
| `LinearRegression` | `fit_intercept=True` | `new(fit_intercept)` | |
| `Ridge` | `alpha=1.0`, `fit_intercept=True` | `new(alpha, fit_intercept)` | alpha is `F`. |
| `Lasso` | `alpha=1.0`, `fit_intercept=True`, `max_iter=1000`, `tol=1e-4` | `new(alpha, fit_intercept)` / `with_opts(.., max_iter, tol)` | |
| `ElasticNet` | `alpha=1.0`, `l1_ratio=0.5`, `fit_intercept=True`, `max_iter=1000`, `tol=1e-4` | `new(alpha, l1_ratio, fit_intercept)` / `with_opts` | |
| `LogisticRegression` | `C=1.0`, `fit_intercept=True`, `max_iter=100`, `tol=1e-4` | `new(c, fit_intercept)` / `with_opts(c, fit_intercept, max_iter, tol)` | Rust field `c`; expose as **`C`** (sklearn name). Internal `LOG_DEFAULT_MAX_ITER=300`/`tol=1e-5` are solver headroom — the *sklearn-named* defaults the shim advertises are 100/1e-4. |
| `PCA` | `n_components` | `new(n_components)` | v1 requires explicit int `n_components` (no `None`/`'mle'`). |
| `TruncatedSVD` | `n_components=2` | `new(n_components)` | |
| `KMeans` | `n_clusters=8`, `max_iter=300`, `tol=1e-4`, `random_state` (→ seed) | `new(n_clusters, seed)` / `with_init` (oracle) | Map `random_state`→`seed`. `init='k-means++'` is the only supported value. |
| `DBSCAN` | `eps=0.5`, `min_samples=5` | `new(eps, min_samples)` | |
| `NearestNeighbors` | `n_neighbors=5` | `new(n_neighbors)` | |
| `KNeighborsClassifier` | `n_neighbors=5` | `new(n_neighbors)` | |
| `KNeighborsRegressor` | `n_neighbors=5` | `new(n_neighbors)` | |

**Mixin composition (D-01, Claude's discretion — recommended):**
- `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `KNeighborsRegressor` → `RegressorMixin` (gives `.score` = R²).
- `LogisticRegression`, `KNeighborsClassifier` → `ClassifierMixin` (gives `.score` = accuracy).
- `KMeans`, `DBSCAN` → `ClusterMixin` (gives `fit_predict`).
- `PCA`, `TruncatedSVD` → `TransformerMixin` (gives `fit_transform`).
- `NearestNeighbors` → no scoring mixin (it exposes `kneighbors`, not `predict`).

## sklearn estimator_checks (criterion 1 — D-01)

**sklearn ≥1.6 tag API (the change to handle):** v1.6 made estimator tags a **public** `__sklearn_tags__()` method returning a `Tags` dataclass; the old `_get_tags`/`_more_tags` are deprecated, and `_estimator_type` as a class attr is deprecated in favor of `Tags.estimator_type`. `check_estimator` and `parametrize_with_checks` read these tags to decide *which* checks to run. `[CITED: scikit-learn.org/dev/whats_new/v1.6 + developers/develop]`

**Recommendation:** subclass `BaseEstimator` (+ the family mixin), which already implements `__sklearn_tags__`. Override `__sklearn_tags__` only to set the non-default tags mlrs needs:
- `tags.input_tags.sparse = False` (mlrs ingests dense Arrow only).
- `tags.array_api_support = False`.
- `tags.input_tags.allow_nan = False` (the bridge hard-rejects nulls; NaN handling is not a v1 contract).
- For the f64-on-rocm case, do **not** advertise a dtype tag that promises f64 — the check harness will feed float64 by default; on a rocm wheel that must raise the D-04 error. (See "Which checks need shim adjustment.")

**"Relevant" check subset per family (recommendation — scope at Wave 0 by running `check_estimator` and triaging):**
| Family | Run these check groups | Skip / xfail (with reason) |
|--------|------------------------|----------------------------|
| All | `check_estimator_cloneable`, `check_get_params_invariance`, `check_set_params`, `check_estimators_unfitted` (raises `NotFittedError`), `check_fit_returns_self`, `check_no_attributes_set_in_init`, `check_parameters_default_constructible` | array-api / sparse / NaN checks (unsupported by design) |
| Regressors (LinReg/Ridge/Lasso/ElasticNet/KNNReg) | `check_regressors_train`, `check_regressor_data_not_an_array`, `check_supervised_y_2d` (if 1-D y enforced), `check_regressors_int` | sample_weight checks if unsupported |
| Classifiers (LogReg/KNNClf) | `check_classifiers_train`, `check_classifiers_classes`, `check_classifier_data_not_an_array`, `check_predict_proba` (KNNClf/LogReg both have it) | multilabel / multioutput |
| Clusterers (KMeans/DBSCAN) | `check_clustering`, `check_clusterer_compute_labels_predict` (KMeans only — DBSCAN has no `predict`), `check_fit_predict` | DBSCAN: skip predict-based checks (no standalone predict, per algos D-08) |
| Transformers (PCA/TruncatedSVD) | `check_transformer_general`, `check_transformer_data_not_an_array`, `check_fit_transform` | PCA `inverse_transform` round-trip if exact-reconstruction not promised |
| Neighbors (NearestNeighbors) | `check_estimators_fit_returns_self`, fit/`kneighbors` shape checks | predict-based checks (no `predict`) |

**Which checks need shim adjustment (anticipate these):**
1. **`__init__` purity:** `check_no_attributes_set_in_init` + `check_parameters_default_constructible` require `__init__` to store every arg verbatim under the same name and set nothing else. (e.g. `self.C = C`, not `self.c`.)
2. **`fit` returns `self`:** the shim's `fit` must `return self` (the Rust `_mlrs` call is a side-effect; the *Python* method returns `self`). PY-01 literal requirement.
3. **Fitted attributes end with `_`:** sklearn checks expect `coef_`, `labels_`, `cluster_centers_`, `components_`, etc. The shim exposes these as properties materializing the host buffer from `_mlrs` lazily (D-03 egress).
4. **`NotFittedError` before fit:** accessing fitted attrs / calling `predict` before `fit` must raise sklearn's `NotFittedError` (use `sklearn.utils.validation.check_is_fitted`).
5. **Input validation:** use `BaseEstimator._validate_data` / `check_array` in the shim *before* normalizing to pyarrow, so the dimension/shape/dtype error messages match what `estimator_checks` expects. NaN/inf inputs: `check_array(..., force_all_finite=True)` raises the sklearn-standard error rather than tripping the Rust null/align bridge with a less-recognizable message.
6. **Tags:** override `__sklearn_tags__` as above so the harness doesn't run sparse/array-api/NaN checks that mlrs intentionally doesn't support.

**Confidence:** MEDIUM — the *mechanism* is HIGH-confidence (sklearn docs), but exactly which checks pass per estimator is empirical. Plan a **Wave-0 triage task**: run `check_estimator(est)` for each of the 11, record pass/fail, and either fix the shim or mark a check skipped-with-reason. Do not promise "all checks pass" — criterion 1 says "*relevant* checks."

## Common Pitfalls

### Pitfall 1: PyO3 version skew links two ABIs into one cdylib
**What goes wrong:** using PyO3 0.29 (latest) with arrow 59's pyarrow feature (pyo3 0.28) makes Cargo link two PyO3 versions; the build fails or, worse, produces a wheel that crashes at import.
**Why it happens:** "track latest" policy vs. arrow's transitive pin.
**How to avoid:** pin `pyo3 = "0.28"` in `[workspace.dependencies]`; run `cargo tree -i pyo3` and assert a single version before building any wheel.
**Warning signs:** `cargo tree` shows `pyo3 v0.28.x` *and* `v0.29.x`; linker symbol conflicts on `PyInit`.

### Pitfall 2: D-08 probe aborts instead of raising ImportError
**What goes wrong:** cubecl's `client()` `.unwrap()`s; a missing driver panics; without `catch_unwind` (or with `panic=abort`) the Python process dies instead of raising `ImportError`.
**Why it happens:** `Runtime::client` returns `ComputeClient`, not `Result` — the failure surfaces as a panic, not an `Err`.
**How to avoid:** wrap the probe in `std::panic::catch_unwind`; keep `panic = "unwind"` (default) in the wheel build profile.
**Warning signs:** `import mlrs` segfaults / aborts on a machine without the driver instead of a clean traceback.

### Pitfall 3: sliced pyarrow array rejected by the bridge
**What goes wrong:** a zero-copy slice of a larger numpy buffer becomes a pyarrow array whose values view does not cover the whole backing buffer; `validate_f32/f64` rejects it as an offset/slice (it requires `byte_offset==0 && inner.len()==values.len()*elem`).
**Why it happens:** the bridge is a hard-reject security boundary (FOUND-06); it treats slices as untrusted aliased data.
**How to avoid:** the shim must produce a **freshly contiguous** pyarrow array (the D-02 row-major flatten), e.g. `pa.array(np.ascontiguousarray(X).ravel())`, not a view/slice.
**Warning signs:** `BridgeError::Offset` from `fit` on otherwise-valid data.

### Pitfall 4: non-faithful `__init__` breaks `clone`/`get_params`
**What goes wrong:** validating or transforming params in `__init__` (e.g. casting, deriving fields) makes `clone()` produce a different estimator and fails `check_parameters_default_constructible`.
**How to avoid:** `__init__` stores each arg verbatim; all validation happens in `fit`.

### Pitfall 5: f64 default silently fails on rocm wheel
**What goes wrong:** D-05 says non-float inputs default to f64 *where supported*; on a rocm wheel f64 is unsupported, so a naive default-to-f64 path hits the D-04 error for integer/list inputs.
**How to avoid:** the dtype-resolution logic must query the capability flag (`supports_type(F64)`) and default to **f32 on f64-incapable backends** (D-05 explicitly). Surface the capability flag from `_mlrs` so the Python shim can pick the default dtype without a failed round-trip.
**Warning signs:** integer/list inputs raise the f64 error on rocm.

### Pitfall 6: GIL-held during compute (missed `detach`)
**What goes wrong:** forgetting `py.detach(...)` holds the GIL through the kernel; joblib/threaded callers serialize, violating PY-03's "GIL released during compute."
**How to avoid:** every device-compute method wraps the trait call in `py.detach`. The closure must be `Send` (the `Mutex<BufferPool>` global satisfies this) and touch no Python objects.

## Runtime State Inventory

> Phase 6 is greenfield Python-surface work over an existing Rust core — it adds files, it does not rename/migrate stored state. No runtime-state migration applies. The one cross-cutting *new* runtime state is the module-global pool/client (a design choice, below), not a migration of existing data.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore keys/IDs renamed; estimators are in-memory. | None |
| Live service config | None — no external service config. | None |
| OS-registered state | None. | None |
| Secrets/env vars | None. (Build may read a `MLRS_BACKEND`-style env to pick which pyproject to build, but that's build-time, not runtime state.) | None |
| Build artifacts | The cdylib name and wheel tags change once PyO3/maturin land; `cargo clean -p mlrs-py` after the Cargo.toml dep change avoids a stale rlib. New: `target/wheels/*.whl` per backend. | `cargo clean -p mlrs-py` once after adding pyo3; rebuild. |

## BufferPool + Client Lifecycle (Claude's-discretion recommendation)

**Recommendation: ONE process-global `BufferPool<ActiveRuntime>` + client, behind a `std::sync::Mutex` (or `OnceLock<Mutex<…>>`), owned by the `_mlrs` module.**

Rationale:
- **Client init is expensive and the cubecl client is internally `Arc`-shared** — constructing one per estimator wastes the device-context handshake. cubecl's `ComputeClient` is `Clone` (cheap, ref-counted), so a single owned client is the natural unit.
- **The pool is `!Sync`-by-use** (it holds a `HashMap` free-list mutated on every `acquire`/`release`); a `Mutex` makes module-global access sound and gives the `Send` closure body `Python::detach` needs.
- **Concurrency model:** under `py.detach`, two Python threads can both try to compute. The `Mutex` serializes device access — correct, and matches the reality that a single device is one compute queue. This means mlrs **does not** give intra-process GPU parallelism across estimators in v1 (joblib `n_jobs>1` over mlrs estimators will serialize on the device mutex). Document this as the accepted v1 semantics (it matches "single-device first" from PROJECT scope). True parallelism would need per-thread clients/streams — out of v1 scope (Deferred: multi-GPU/concurrency).
- **Per-estimator pool alternative** (each `#[pyclass]` owns its own `BufferPool`): rejected — it fragments buffer reuse (the whole point of FOUND-05), multiplies device memory, and still needs per-pool locking for thread safety. A shared pool maximizes reuse (the memory-gate invariant the project has guarded since Phase 2).
- **`#[pyclass]` thread-safety:** PyO3 requires `#[pyclass]` to be `Send`. The wrappers hold only the `AnyEstimator` enum (device-resident handles + plain hyperparameters) — `Send` as long as `DeviceArray`/`Handle` are `Send` (cubecl handles are `Send`). The shared pool is *not* stored in the pyclass; it's the module global. Mark estimator state behind the global mutex; do not store a `Rc`/`RefCell`.

**Open sub-question for the planner:** whether to make the pyclass `#[pyclass(frozen)]` + interior `Mutex` on the per-estimator fitted state (cleaner for free-threaded 3.13+/3.14) or rely on the GIL for per-estimator mutation while only the *pool* is mutex-guarded. v1 targets abi3-py312 (GIL builds), so GIL-guarded per-estimator mutation + mutex-guarded shared pool is sufficient and simplest. Revisit if free-threaded wheels become a target.

## pytest Oracle Harness for Python (criterion 1)

**Goal:** re-validate the 1e-5 contract through the FULL `numpy → pyarrow → __arrow_c_array__ → Rust FFI → validate → device → compute → host → numpy` path, not just the Rust layer.

**Approach (extends the existing committed-fixture pattern):**
- The `.npz` oracle fixtures already committed under `tests/fixtures/` (generated by `scripts/gen_oracle.py`, sklearn reference outputs) are **reusable as-is** — they are dtype-tagged sklearn reference blobs (`coef_`, `labels_`, `components_`, …). The Python test loads the same `.npz` with `numpy.load`, feeds the input array through the *Python estimator* (`mlrs.KMeans(...).fit(X)`), and asserts the fitted attribute matches the fixture's reference within 1e-5 (reusing the Phase-1 sign-flip / label-permutation comparison logic, re-expressed in Python — or call `numpy`/`scipy` equivalents).
- **No new fixture generation is strictly required** for the happy path — the Python oracle is a *second consumer* of the existing blobs, proving the binding path preserves the numerics. (gen_oracle.py needs numpy/scipy/sklearn via a /tmp venv per PEP 668; fixtures are committed blobs — CI never regenerates. From project memory.)
- **What IS new in Python:** a small `conftest.py` fixture loader + the sign-flip (PCA/SVD `components_`) and label-permutation (KMeans/DBSCAN `labels_`) comparison helpers, mirroring `mlrs_core::oracle`. For LogisticRegression, the Python oracle must gauge-fix `predict_proba` (the PRIMARY gate, per Phase-5 D-12) — compare probabilities, not raw `coef_`.
- **dtype × backend gating:** the Python harness must skip f64 cases on a rocm wheel with a logged reason (mirror `skip_f64_with_log`). Surface the capability flag from `_mlrs` (e.g. `mlrs.backend_supports_f64()`) so pytest can `@pytest.mark.skipif(not mlrs.backend_supports_f64())`.
- **estimator_checks** run alongside the oracle tests in the same pytest tree (Wave-0 triage task above).

**Wave-0 test scaffolding (gaps to fill before the wrappers exist):**
- `python/tests/conftest.py` — fixture loader + sign-flip/label-perm helpers + a `capability` skip marker.
- `python/tests/test_oracle_<family>.py` — one per family, `@pytest.mark.parametrize` over the committed fixtures.
- `python/tests/test_estimator_checks.py` — `parametrize_with_checks([...11 estimators...])` triage.
- A `maturin develop --features cpu` (or rocm) step in the test workflow so `import mlrs` resolves the just-built extension.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `Python::allow_threads` / `with_gil` / `prepare_freethreaded_python` | `Python::detach` / `attach` / `initialize` | PyO3 0.26 | Use `detach` for GIL release; old names are deprecated aliases (still compile on 0.28). |
| sklearn `_get_tags`/`_more_tags` (private) | `__sklearn_tags__()` → `Tags` (public), `_estimator_type` → `Tags.estimator_type` | sklearn 1.6 | Override `__sklearn_tags__`; inherit the rest from `BaseEstimator`. |
| `package.metadata.maturin.name` | `[tool.maturin] module-name` + PEP-621 `[project].name` | maturin ≥1.x | Use `module-name` for the import name; `[project].name` for the distribution name. |
| Passing Arrow C pointers as integers | `__arrow_c_array__` PyCapsule protocol (capsule carries the release callback) | Arrow ~14 / formalized since | Robust ownership; the capsule frees data on error. Consumed via `arrow::pyarrow` / `pyo3-arrow`. |

**Deprecated/outdated:**
- PyO3 `allow_threads`/`with_gil` — prefer `detach`/`attach` (aliases still work).
- sklearn `_get_tags` — deprecated; `__sklearn_tags__` is the public API.
- Manual `FFI_ArrowArray::from_raw` plumbing — prefer the library `FromPyArrow`/`PyArray`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `pyarrow`/`numpy` Python floor versions (>=14 / >=1.26) | Standard Stack / Maturin pattern | Low — pin a conservative floor at planning; pyarrow ≥14 has the capsule protocol. |
| A2 | The exact `relevant` estimator_checks subset per family is as tabulated | sklearn estimator_checks | MEDIUM — empirical; Wave-0 triage will correct it. Do not promise "all checks pass." |
| A3 | `arrow::pyarrow::ArrayData::from_pyarrow_bound` is the 0.28-era method name | Pattern 2 | Low — the trait is `FromPyArrow`; method name may be `from_pyarrow_bound`/`from_pyarrow`. Verify against arrow 59 docs.rs at planning (docs.rs page 404'd this session; method exists, name to confirm). |
| A4 | cubecl client init cost at import is "cheap enough" on rocm/cuda | Import-Time Probe | Low-MEDIUM — one device enumeration; acceptable as a one-time import side-effect, but confirm rocm init latency on gfx1100 isn't seconds. |
| A5 | A shared `Mutex<BufferPool>` is the right concurrency unit | BufferPool Lifecycle | Low — it is the safe default; the only cost is no intra-process device parallelism (accepted v1 scope). |
| A6 | `pyo3-arrow` 0.18 targets PyO3 0.28 (so it composes with arrow 59) | Standard Stack | Low — docs.rs reports 0.18.x→PyO3 0.28; if it lags, use `arrow::pyarrow` directly (the dep-free alternative). |

## Open Questions (RESOLVED)

1. **arrow::pyarrow vs pyo3-arrow — pick one. — RESOLVED: use `arrow::pyarrow`.**
   - What we know: both consume the capsule with correct ownership against PyO3 0.28. `arrow::pyarrow` adds no dependency (arrow is already pinned); `pyo3-arrow` adds the numpy buffer-protocol fallback + ergonomic owned `PyArray` arg.
   - What's unclear: whether the shim's numpy→pyarrow normalization (D-02) makes the buffer-protocol fallback redundant (it likely does).
   - **RESOLUTION:** Use `arrow::pyarrow` (one fewer dep; the shim already normalizes to a freshly-contiguous pyarrow array, so the buffer-protocol fallback is redundant). Encoded in Plan 06-01 Task 1 (deps) + Plan 06-02 Task 1 (owned ingress). `pyo3-arrow` is NOT added.

2. **Exact arrow 59 `FromPyArrow` method name** (`from_pyarrow_bound` vs `from_pyarrow`) — docs.rs 404'd this session. — **RESOLVED in-execution.**
   - **RESOLUTION:** Plan 06-01 Task 5 produces an `arrow_symbol_probe` deliverable confirming the exact method name locally via `cargo doc -p arrow --features pyarrow` BEFORE Plan 06-02 (which `depends_on` 06-01) consumes it. Sequenced correctly; trivial to resolve, no design impact.

3. **DBSCAN/NearestNeighbors `predict`-less surface vs estimator_checks. — RESOLVED: skip predict-based checks for these two, documented.**
   - What we know: DBSCAN has no standalone `predict` (algos D-08); NearestNeighbors exposes `kneighbors`, not `predict`.
   - **RESOLUTION:** Skip predict-based estimator_checks for DBSCAN and NearestNeighbors with a documented reason (sklearn-faithful — sklearn's own DBSCAN has no `predict`). Encoded in Plan 06-04 (predict-less shim surface) + Plan 06-06 `checks_triage.md` (documented skips).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | the whole phase | ✓ | 3.12.3 | — (abi3-py312 needs ≥3.12) |
| Rust/Cargo | build | ✓ | (workspace toolchain) | — |
| maturin | wheel build | ✗ (not yet installed) | target 1.14 | `pip install maturin` in a venv (PEP 668 → /tmp venv, per project memory pattern) |
| pyarrow (Python) | D-02 ingress + tests | ✗ (not yet installed) | floor ≥14 | install in the test venv |
| scikit-learn (Python) | D-01 shim + checks | ✗ (not yet installed) | ≥1.6 | install in the test venv |
| numpy (Python) | in/out container | likely ✓ (oracle venv) | ≥1.26 | /tmp venv |
| ROCm/HIP runtime | rocm wheel test gate | ✓ | rocminfo at /opt/rocm | — (gfx1100 confirmed runnable, f32) |
| cpu backend | cpu wheel gate (f64) | ✓ | cubecl-cpu MLIR | — |

**Missing dependencies with no fallback:** none — all Python tooling installs into a venv (PEP 668 requires `/tmp/oracle-venv`-style venvs per project memory).
**Missing dependencies with fallback:** maturin, pyarrow, scikit-learn — `pip install` into a venv before building/testing.

## Validation Architecture

> `.planning/config.json` `workflow.nyquist_validation` was not inspected as `false`; treated as enabled.

### Test Framework
| Property | Value |
|----------|-------|
| Framework (Rust) | `cargo test` (existing `crates/mlrs-py/tests/`) |
| Framework (Python) | `pytest` (NEW — Wave 0) |
| Config file | none yet for pytest — add `python/tests/conftest.py` (Wave 0) |
| Quick run command | `cargo test -p mlrs-py --features cpu` (Rust); `pytest python/tests -x -k <name>` (Python, after `maturin develop --features cpu`) |
| Full suite command | `maturin develop --features cpu && pytest python/tests` then repeat `--features rocm` (f32 only) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PY-01 | 12 estimators `fit`/`predict`/`transform`/`score`, `fit` returns self; oracle + relevant checks | integration | `pytest python/tests/test_oracle_*.py python/tests/test_estimator_checks.py` | ❌ Wave 0 |
| PY-02 | `get_params`/`set_params` + sklearn names | unit | `pytest python/tests/test_params.py` (or via `check_get_params_invariance`) | ❌ Wave 0 |
| PY-03 | Arrow PyCapsule ingest, ownership, GIL released | integration + Rust | `cargo test -p mlrs-py --features cpu` (ingress ownership) + a threaded GIL-release pytest | ❌ Wave 0 |
| PY-04 | per-backend wheel names + import-fail on absent driver | build/smoke | `maturin build -m pyproject/cpu.pyproject.toml`; assert wheel name `mlrs_cpu-*`; a probe-failure test | ❌ Wave 0 |
| PY-05 | f32 + f64 dispatch; f64-on-rocm raises | unit | `pytest python/tests/test_dtype.py` (f32+f64 on cpu; f64 raises on rocm) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p mlrs-py --features cpu` + targeted `pytest -k`.
- **Per wave merge:** `maturin develop --features cpu && pytest python/tests` (cpu f64); `--features rocm` for the f32 subset.
- **Phase gate:** full Python oracle + relevant estimator_checks green on cpu(f64); f32 subset green on rocm; all four wheels build with the right names; absent-driver import test passes.

### Wave 0 Gaps
- [ ] `crates/mlrs-py/Cargo.toml` — add pyo3 0.28 (abi3-py312, extension-module) + arrow pyarrow + mlrs-algos/mlrs-backend + backend features.
- [ ] `[workspace.dependencies]` — add `pyo3 = "0.28"` (with the ABI-pin comment).
- [ ] `crates/mlrs-py/python/mlrs/` — pure-Python shim package skeleton (base + per-family modules).
- [ ] `pyproject/{cpu,wgpu,cuda,rocm}.pyproject.toml` — per-backend templated configs (`[project].name`, `features`, `module-name="mlrs._mlrs"`, `python-source`).
- [ ] `python/tests/conftest.py` — fixture loader + sign-flip/label-perm helpers + capability skip marker.
- [ ] `python/tests/test_oracle_*.py`, `test_estimator_checks.py`, `test_dtype.py`, `test_params.py` — stubs.
- [ ] Framework install: `pip install maturin pyarrow scikit-learn numpy pytest` into a venv (PEP 668).

## Security Domain

> `security_enforcement` treated as enabled (absent = enabled).

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a (library, no auth surface) |
| V3 Session Management | no | n/a |
| V4 Access Control | no | n/a |
| V5 Input Validation | **yes** | The Arrow bridge (`validate_f32/f64`) hard-rejects offset/null/misaligned buffers BEFORE any `unsafe` transmute (FOUND-06) — reused unchanged at the PyCapsule boundary. Hyperparameters validated in `fit` (`InvalidK`/`InvalidAlpha`/`InvalidEps`/`InvalidMinSamples`, already enforced in algos). The shim adds sklearn `check_array` for shape/finite validation. |
| V6 Cryptography | no | n/a |

### Known Threat Patterns for the PyO3↔Arrow boundary
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Use-after-free / double-free of the Arrow C array across the FFI boundary | Tampering | Owned `FromPyArrow`/`PyArray` ingest — arrow-rs owns the release callback; never borrow `&[u8]` into a Python-owned buffer (PY-03). |
| Uploading aliased parent-buffer data from a sliced array | Tampering / Info-disclosure | `validate_no_offset` hard-rejects slices (bridge.rs); shim hands a freshly contiguous array. |
| Meaningless null-slot values silently computed on | Tampering | `validate_no_nulls` hard-rejects nullable arrays with set null bits. |
| Misaligned transmute (UB) | Tampering | `cast_validated` (`bytemuck::try_cast_slice`) returns a recoverable error, never panics. |
| Untrusted hyperparameter → out-of-bounds device gather | Tampering | Algos validate `k`/`alpha`/`eps`/`min_samples`/`n_components` BEFORE any launch (existing). |
| Driver-absent panic crashing the host process | DoS | `catch_unwind` at import → clean `ImportError` (D-08). |

## Sources

### Primary (HIGH confidence)
- Context7 `/pyo3/pyo3` — `Python::detach`/`attach` rename (0.26), abi3 features, `create_exception!`, custom exceptions, `#[pyclass]` Send.
- Context7 `/pyo3/maturin` — `abi3-py312` feature config, `module-name`, `python-source`, conditional `features`, `[project].dynamic` metadata.
- Local crate source: `cubecl-hip-0.10.0/src/runtime.rs` (`Runtime::client` returns `ComputeClient`, `.unwrap()`s internally → panic on absent driver) — verified `cubecl-runtime-0.10.0/src/runtime.rs` `fn client(..) -> ComputeClient<Self>`.
- `cargo tree -p arrow --features pyarrow` → **pyo3 v0.28.3** (the ABI pin).
- `cargo add pyo3@0.28 --features abi3-py312 --dry-run` → abi3-py312 present.
- `pip index versions maturin` → 1.14.0; `cargo search pyo3 / pyo3-arrow` → 0.29 / 0.18.
- Existing mlrs source: `bridge.rs`, `capability.rs`, `runtime.rs`, `pool.rs`, `device_array.rs`, `traits.rs`, all 11 estimator constructors.

### Secondary (MEDIUM confidence)
- maturin user guide (project_layout, distribution, config) — `[project].name` precedence over cargo name, `module-name` mechanism.
- arrow.apache.org PyCapsule Interface spec; docs.rs `pyo3-arrow` — owned-`PyArray` capsule consumption + release callback ownership.
- scikit-learn.org v1.6 whats_new + developers/develop — `__sklearn_tags__` public tag API, `parametrize_with_checks`, `_estimator_type` deprecation.

### Tertiary (LOW confidence)
- WebSearch summaries for arrow-rs `FromPyArrow` exact method name (docs.rs 404 this session — confirm locally).

## Metadata

**Confidence breakdown:**
- Standard stack / version pins: HIGH — verified via cargo-tree, cargo-add, pip index, and the local cubecl source.
- Maturin multi-distribution: HIGH — mechanism confirmed (`module-name` + `[project].name` precedence) across maturin docs + Context7.
- Arrow PyCapsule import: MEDIUM-HIGH — pattern confirmed; exact arrow 59 method name to verify locally.
- Import probe (panic→ImportError): HIGH — cubecl `client()` panic behavior verified in source.
- estimator_checks subset: MEDIUM — mechanism HIGH, per-estimator pass/fail empirical (Wave-0 triage).
- BufferPool concurrency: MEDIUM-HIGH — shared-Mutex is the safe default; the no-intra-process-parallelism tradeoff is the only open consequence.

**Research date:** 2026-06-13
**Valid until:** ~2026-07-13 (30 days) — PyO3/arrow/maturin are stable on these majors; re-verify the PyO3↔arrow version pin if arrow or pyo3 bumps a major.

## RESEARCH COMPLETE
