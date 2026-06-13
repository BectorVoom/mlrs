---
phase: 06-python-surface-pyo3-estimators-per-backend-wheels
plan: 02
subsystem: bindings
tags: [pyo3, arrow, pycapsule, ffi, capability, dispatch, gil, buffer-pool, abi3]

# Dependency graph
requires:
  - phase: 06-python-surface-pyo3-estimators-per-backend-wheels
    plan: 01
    provides: pyo3 0.28 pin, mlrs-py crate wiring, arrow FromPyArrow symbol, allocator site
  - phase: 03-backend-runtime-bridge-pool
    provides: bridge::validate_f32/f64, BufferPool, DeviceArray, capability::feature_enabled, runtime::active_client
  - phase: 04-closed-form-estimators
    provides: Estimator<F> monomorphizations the dispatch enum wraps
  - phase: 05-iterative-estimators-clustering-neighbors
    provides: Lasso/ElasticNet/LogisticRegression/KMeans/DBSCAN/KNN generics
provides:
  - "Owned Arrow PyCapsule ingress (from_pyarrow_bound, no &[u8] borrow) reusing bridge::validate_f32/f64 UNCHANGED (PY-03/D-02)"
  - "f64-on-incapable-backend guard -> clear PyValueError + backend_supports_f64() flag (D-04/D-05)"
  - "boundary BridgeError/AlgoError/anyhow -> PyErr mapping (Value/Type/Runtime)"
  - "#[pymodule] _mlrs with catch_unwind D-08 driver probe -> PyImportError (T-06-05)"
  - "single process-global Mutex<BufferPool<ActiveRuntime>> + global_pool() accessor"
  - "any_estimator! dtype-dispatch macro skeleton + documented py.detach/guard_f64 contracts (D-06)"
  - "device->host egress helpers (Vec<F>/Vec<i32> + shape; numpy/arrow wrap is shim-side, D-03)"
  - "pyo3 extension-module gated behind a crate feature so cargo test links libpython (auto-initialize)"
affects: [06-03-pyclass-wrappers, 06-04-python-shim-logic, 06-05-wheel-build-tests, 06-06-estimator-checks]

# Tech tracking
tech-stack:
  added: [pyo3 macros feature, pyo3 auto-initialize (dev), arrow pyarrow FromPyArrow ingress, mlrs-core + bytemuck direct deps]
  patterns: [owned-capsule FFI ingress, catch_unwind import probe, process-global mutex pool, dtype-dispatch macro skeleton, extension-module-as-feature for testable pyo3 cdylib]

key-files:
  created:
    - crates/mlrs-py/src/errors.rs
    - crates/mlrs-py/src/ingress.rs
    - crates/mlrs-py/src/capability.rs
    - crates/mlrs-py/src/egress.rs
    - crates/mlrs-py/src/dispatch.rs
    - crates/mlrs-py/tests/ingress_test.rs
    - crates/mlrs-py/tests/probe_test.rs
  modified:
    - crates/mlrs-py/src/lib.rs
    - crates/mlrs-py/Cargo.toml
    - crates/mlrs-py/pyproject/cpu.pyproject.toml
    - crates/mlrs-py/pyproject/wgpu.pyproject.toml
    - crates/mlrs-py/pyproject/cuda.pyproject.toml
    - crates/mlrs-py/pyproject/rocm.pyproject.toml
    - Cargo.lock

key-decisions:
  - "extension-module is a crate FEATURE (default-off), enabled by the four maturin templates; cargo test links libpython via a pyo3 auto-initialize dev-dependency"
  - "pyo3 'macros' feature added at the mlrs-py crate (the workspace pin is default-features=false to keep extension-module out of the workspace default)"
  - "Rust integration tests assert ingress control-flow + typed BridgeError; the concrete PyValueError/PyTypeError class is asserted by the Python pytest oracle (Plan 05), because an extension-module-mode test binary cannot link the interpreter"
  - "arrow 59 slice(0,n) REBASES the ScalarBuffer to a fresh non-aliasing buffer, so a from-start slice is contiguous and ACCEPTED; only a non-zero-offset slice aliases and is hard-rejected"

requirements-completed: [PY-03, PY-04, PY-05]

# Metrics
duration: 16min
completed: 2026-06-13
---

# Phase 6 Plan 02: Core Binding Plumbing (ingress / egress / dispatch / capability / probe) Summary

**Owned Arrow PyCapsule ingress reusing the `bridge::validate_f32/f64` hard-reject validator unchanged, the `#[pymodule] _mlrs` with a `catch_unwind` driver probe (clean `PyImportError`, never a process abort), a single process-global `Mutex<BufferPool>`, the f64-on-incapable-backend guard + `backend_supports_f64()` flag, and the `any_estimator!` dtype-dispatch macro skeleton — every shared primitive Plan 03's 12 `#[pyclass]` wrappers consume, delivered interface-first.**

## Performance

- **Duration:** ~16 min
- **Completed:** 2026-06-13
- **Tasks:** 2 executed
- **Files:** 14 (7 created, 7 modified)
- **Tests:** 14 Rust tests pass (3 allocator pre-existing + 7 ingress + 4 probe)

## Accomplishments

- **Task 1 — ingress + capability guard + boundary errors (`61e25d9`):**
  - `errors.rs`: maps `BridgeError::{Offset,HasNulls,Misaligned}` → `PyValueError`, `BridgeError::DataTypeMismatch` + non-float dtype → `PyTypeError`, `AlgoError` → `PyValueError`, opaque `anyhow` → `PyRuntimeError`. The single auditable place the cdylib chooses a Python exception class; the typed `Display` text is preserved verbatim.
  - `ingress.rs`: `capsule_to_array` consumes `__arrow_c_array__` via `arrow::pyarrow::FromPyArrow` (`ArrayData::from_pyarrow_bound` — the Plan-01 resolved symbol) into an **owned** `ArrayRef` (no `&[u8]` borrow into the Python buffer — PY-03/T-06-03). `validated_f32`/`validated_f64` feed `mlrs_backend::bridge::validate_f32`/`validate_f64` **unchanged**, then `DeviceArray::from_host`. `float_dtype` + `as_f32`/`as_f64` implement the D-06 dispatch key with a `PyTypeError` on a non-float dtype.
  - `capability.rs`: `supports_f64()` wraps `feature_enabled(F64)`; `guard_f64()` returns a clear `PyValueError` ("backend '…' does not support float64 — pass float32 or install mlrs-cpu") on an f64-incapable backend (D-04 — never a silent downcast).
- **Task 2 — `_mlrs` pymodule + global pool + dispatch macro (`b1627de`):**
  - Grew `lib.rs` into `#[pymodule] _mlrs`: a `std::panic::catch_unwind(active_client + properties())` probe at import → `PyImportError` on a missing driver (D-08/T-06-05), so `import mlrs` raises a clean Python error instead of aborting CPython. `mod allocator;` stays the single allocator site (FOUND-09).
  - `static GLOBAL_POOL: OnceLock<Mutex<BufferPool<ActiveRuntime>>>` + `global_pool()` — one shared pool/client behind a mutex (the documented single-device-mutex v1 concurrency model; joblib `n_jobs>1` serializes on the device).
  - `backend_supports_f64()` `#[pyfunction]` registered on the module (D-05).
  - `egress.rs`: `vec_f_to_py`/`vec_i32_to_py`/`labels_to_py` return `(Vec, shape)` via the metered read path (D-10), numpy/arrow wrap deferred to the shim (D-03).
  - `dispatch.rs`: `macro_rules! any_estimator!` emitting the `{ Unfit{..}, F32(Estimator<f32>), F64(Estimator<f64>) }` enum; the module doc fixes the two load-bearing contracts Plan 03's `#[pymethods] fit` bodies extend — `py.detach(|| global_pool().lock()…)` GIL release (PY-03) and `guard_f64()?` before the F64 arm (D-04).

## Task Commits

1. **Task 1: ingress + capability guard + boundary errors** — `61e25d9` (feat)
2. **Task 2: _mlrs pymodule — catch_unwind probe, global pool, dispatch macro** — `b1627de` (feat)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `extension-module` made a crate feature so Rust tests link**
- **Found during:** Task 1 (running `ingress_test`).
- **Issue:** PyO3's `extension-module` feature (hard-wired by Plan 01) tells the linker CPython symbols come from the host interpreter at import, so the wheel must NOT link libpython. A Rust integration-test binary then fails to link (`undefined symbol: PyExc_ValueError`, `Py_InitializeEx`, …) — the test verb in the plan (`cargo test … --test ingress_test`) could not run at all.
- **Fix:** Removed `extension-module` from the always-on pyo3 features; added it as a crate feature `extension-module = ["pyo3/extension-module"]` enabled by all four maturin templates (`features = ["<backend>", "extension-module"]`); added a `[dev-dependencies] pyo3 = { …, "auto-initialize" }` so `cargo test`/`cargo build` link + initialize a real libpython (present on this machine: `libpython3.12.so`). This is the standard testable-pyo3-cdylib pattern and is required for the plan's own verification command.
- **Files:** `crates/mlrs-py/Cargo.toml`, `crates/mlrs-py/pyproject/{cpu,wgpu,cuda,rocm}.pyproject.toml`.
- **Commit:** `61e25d9`.

**2. [Rule 3 - Blocking] Added `mlrs-core` + `bytemuck` direct deps; enabled pyo3 `macros`**
- **Found during:** Task 1 (errors.rs/egress.rs compile) and Task 2 (`#[pymodule]`).
- **Issue:** `errors.rs` matches on `mlrs_core::error::BridgeError` directly (the type lives in `mlrs-core`, not re-exported by `mlrs-backend`); `egress.rs` needs the `bytemuck::Pod` bound. `#[pymodule]`/`#[pyfunction]`/`wrap_pyfunction!` need pyo3's `macros` feature, which the workspace `default-features = false` pin omits (Plan 01 used no macros).
- **Fix:** Added `mlrs-core` (path) and `bytemuck` (workspace) as direct `mlrs-py` deps — both are lightweight workspace deps and add NO second pyo3 ABI. Added `macros` to the `mlrs-py` pyo3 feature list (crate-local, not workspace-wide).
- **Files:** `crates/mlrs-py/Cargo.toml`.
- **Commit:** `61e25d9` (deps), `b1627de` (macros).

### Plan-text vs. reality notes (not deviations)

- **Rust test assertions assert control-flow + typed errors, not the Python exception class.** Because an `extension-module`-mode wheel must not link the interpreter, a Rust test cannot call `PyErr::is_instance_of`/`Python::attach` against a live interpreter in the wheel link mode. The tests therefore assert `Ok`/`Err` control-flow and the underlying typed `BridgeError::Offset` (pure Rust); the concrete `PyValueError`/`PyTypeError` class is asserted end-to-end by the Python pytest oracle (Plan 05/06), where a live interpreter exists. The error→class mapping itself is unit-visible in `errors.rs` and grep-verified by the acceptance criteria.
- **`from_start_slice` is accepted, not rejected.** The plan's mental model expected a from-the-start slice to be rejected; arrow 59's `slice(0, n)` actually REBASES the `ScalarBuffer` to a fresh, non-aliasing buffer (`inner.len() == values.len()*elem`, `ptr_offset() == 0`), so it is genuinely contiguous and correctly ACCEPTED. Only a non-zero-offset slice (`slice(1, 3)`) aliases parent data and is hard-rejected (`BridgeError::Offset`). Both behaviors are pinned by tests; this documents the true bridge boundary (T-06-04 is mitigated for the aliasing case, which is the threat).

## Threat-Model Outcomes

| Threat ID | Disposition | Evidence |
|-----------|-------------|----------|
| T-06-03 (use-after-free of Arrow C array) | mitigated | `capsule_to_array` returns an OWNED `ArrayRef` via `from_pyarrow_bound`; no `&[u8]` borrow (grep: no `&[u8]` in ingress.rs code). |
| T-06-04 (aliased parent-buffer slice) | mitigated | `validate_no_offset` (reused unchanged) rejects the non-zero-offset slice → `BridgeError::Offset` → `PyValueError`; `sliced_f32_is_hard_rejected_before_upload` asserts it. |
| T-06-05 (driver-absent panic → process abort) | mitigated | `catch_unwind(active_client + properties)` → `PyImportError`; profile keeps `panic = "unwind"`; `probe_is_panic_safe` asserts catch. |
| T-06-06 (silent f64→f32 downcast) | mitigated | `guard_f64()` raises `PyValueError` before compute; `f64_guard_matches_backend_capability` asserts the verdict tracks capability. |
| T-06-07 (misaligned transmute UB) | mitigated | `bridge::cast_validated` (bytemuck `try_cast_slice`) reused unchanged. |

## Known Stubs

- `dispatch.rs::any_estimator!` is a documented **skeleton** (emits the dispatch enum; the `#[pymethods] fit` bodies are added by Plan 03) — intentional, interface-first per this plan's objective. The `GLOBAL_POOL`/`global_pool()` carry `#[allow(dead_code)]` with a note that Plan 03 is the consumer. No data-flow stub reaches a UI; nothing blocks the plan goal (which is to deliver the primitives Plan 03 consumes).

## Verification

- `cargo build -p mlrs-py --features cpu` compiles the `_mlrs` `#[pymodule]`.
- `cargo build -p mlrs-py --features cpu,extension-module` compiles the wheel link mode.
- `cargo test -p mlrs-py --features cpu --test ingress_test` → 7 pass; `--test probe_test` → 4 pass.
- `cargo check --workspace --features cpu` clean (no regression).
- `cargo tree -p mlrs-py --features cpu -i pyo3` → exactly one `pyo3 v0.28.3` (single ABI, Pitfall 1 guarded).
- Acceptance greps all match: `validate_f32|validate_f64` in `ingress.rs`; `does not support float64` in `capability.rs`; `catch_unwind` + `PyImportError` + `OnceLock<Mutex<BufferPool>>` + `backend_supports_f64` in `lib.rs`; `macro_rules! any_estimator` + `detach` in `dispatch.rs`.
- clippy on the new `mlrs-py` files is clean (remaining workspace clippy warnings are all pre-existing in mlrs-algos/mlrs-backend/mlrs-kernels, out of scope).

## Next Phase Readiness

- **Plan 03 (`#[pyclass]` wrappers)** has every primitive in-hand: `ingress::{capsule_to_array, float_dtype, as_f32/f64, validated_f32/f64}`, `capability::guard_f64`, `egress::{vec_f_to_py, vec_i32_to_py}`, `errors::*` mappers, `global_pool()`, and the `any_estimator!` macro skeleton with the `py.detach`/`guard_f64` contracts documented. It registers its 12 `#[pyclass]`es on `m` in `_mlrs`.
- No blockers.

## Self-Check: PASSED

- Files: all 7 created files FOUND; `lib.rs` + 4 pyproject templates + Cargo.{toml,lock} modified.
- Commits: `61e25d9` FOUND, `b1627de` FOUND.

---
*Phase: 06-python-surface-pyo3-estimators-per-backend-wheels*
*Completed: 2026-06-13*
