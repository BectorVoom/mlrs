---
phase: 06-python-surface-pyo3-estimators-per-backend-wheels
plan: 01
subsystem: infra
tags: [pyo3, maturin, arrow, pyarrow, sklearn, python, wheels, abi3, pytest]

# Dependency graph
requires:
  - phase: 03-backend-runtime-bridge-pool
    provides: mlrs-backend capability flags + Arrow->CubeCL bridge (validate_f32/f64)
  - phase: 04-closed-form-estimators
    provides: LinearRegression/Ridge/PCA/TruncatedSVD in mlrs-algos
  - phase: 05-iterative-estimators-clustering-neighbors
    provides: Lasso/ElasticNet/LogisticRegression/KMeans/DBSCAN/KNN in mlrs-algos
provides:
  - "pyo3 0.28 single-ABI workspace pin with an ABI-pin rationale comment (PY-05)"
  - "mlrs-py crate wired with pyo3 (abi3-py312, extension-module) + arrow pyarrow + cpu/wgpu/cuda/rocm features"
  - "Four per-backend maturin pyproject templates (mlrs-cpu/-wgpu/-cuda/-rocm) with constant module-name mlrs._mlrs"
  - "Pure-Python mlrs package skeleton exposing the 12 sklearn-compatible estimator shells (D-01)"
  - "pytest Nyquist scaffold: conftest helpers + 7 collecting stubs (one per PY req family)"
  - "RESOLVED arrow-59 FromPyArrow symbol (ArrayData::from_pyarrow_bound) recorded for Plan 02"
affects: [06-02-ingress-egress, 06-03-pyclass-wrappers, 06-04-python-shim-logic, 06-05-wheel-build-tests, 06-06-estimator-checks]

# Tech tracking
tech-stack:
  added: [pyo3 0.28, arrow pyarrow feature, maturin (build-time), sklearn BaseEstimator/mixins (Python)]
  patterns: [maturin multi-distribution one-crate-N-dists, abi3-py312 single wheel, sklearn-faithful __init__ purity, Nyquist collecting-stub test scaffold]

key-files:
  created:
    - crates/mlrs-py/pyproject/cpu.pyproject.toml
    - crates/mlrs-py/pyproject/wgpu.pyproject.toml
    - crates/mlrs-py/pyproject/cuda.pyproject.toml
    - crates/mlrs-py/pyproject/rocm.pyproject.toml
    - crates/mlrs-py/python/mlrs/__init__.py
    - crates/mlrs-py/python/mlrs/base.py
    - crates/mlrs-py/python/mlrs/_io.py
    - crates/mlrs-py/python/mlrs/linear.py
    - crates/mlrs-py/python/mlrs/cluster.py
    - crates/mlrs-py/python/mlrs/decomposition.py
    - crates/mlrs-py/python/mlrs/neighbors.py
    - crates/mlrs-py/python/tests/conftest.py
    - crates/mlrs-py/python/tests/test_oracle_linear.py
    - crates/mlrs-py/python/tests/test_oracle_cluster.py
    - crates/mlrs-py/python/tests/test_oracle_decomposition.py
    - crates/mlrs-py/python/tests/test_oracle_neighbors.py
    - crates/mlrs-py/python/tests/test_estimator_checks.py
    - crates/mlrs-py/python/tests/test_dtype.py
    - crates/mlrs-py/python/tests/test_params.py
    - crates/mlrs-py/src/arrow_symbol_probe.rs
  modified:
    - Cargo.toml
    - crates/mlrs-py/Cargo.toml

key-decisions:
  - "Python runtime-dep floors locked (Task 1 human gate): numpy>2.0.0 (numpy 2.x only, strictly > 2.0.0), pyarrow>=14, scikit-learn>=1.6 — written verbatim into all four templates"
  - "arrow-59 FromPyArrow exposes exactly ONE ingress method, ArrayData::from_pyarrow_bound(&Bound<PyAny>); no non-_bound variant exists (RESOLVED Open Question Q2 / A3)"
  - "pyo3 pinned 0.28 (not latest 0.29) as the single linked ABI — the only workspace dep that overrides track-latest"

patterns-established:
  - "Maturin multi-distribution: one crate -> N dist names, constant module-name=mlrs._mlrs so every backend wheel is import mlrs"
  - "sklearn-faithful __init__ purity: every ctor arg stored verbatim under the same name (self.C=C), validation deferred to fit"
  - "Nyquist test scaffold: collecting stubs (importorskip/xfail) that collect green pre-wrapper and convert to assertions in later plans"

requirements-completed: [PY-01, PY-02, PY-03, PY-04, PY-05]

# Metrics
duration: 6min
completed: 2026-06-13
---

# Phase 6 Plan 01: Python-Surface Wave-0 Scaffold Summary

**pyo3 0.28 single-ABI pin + four maturin per-backend pyproject templates + 12-estimator pure-Python mlrs skeleton + a green-collecting pytest scaffold, with the arrow-59 `ArrayData::from_pyarrow_bound` ingress symbol resolved for Plan 02.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-06-13T21:24:00+09:00 (approx, first task commit prep)
- **Completed:** 2026-06-13T12:30:55Z
- **Tasks:** 4 executed (Task 1 was the resolved human decision gate; Tasks 2-5 implemented)
- **Files modified:** 21 (19 created, 2 modified)

## Accomplishments
- Pinned `pyo3 = { version = "0.28", default-features = false }` in the workspace with an explicit ABI-pin comment; `cargo tree -p mlrs-py --features cpu -i pyo3` resolves exactly one `pyo3 v0.28.3` (no 0.29). `cargo build -p mlrs-py --features cpu` compiles.
- Wired `mlrs-py` with pyo3 (`abi3-py312`, `extension-module`), arrow (`pyarrow`), the `mlrs-algos`/`mlrs-backend` path deps, and a `[features]` block forwarding cpu/wgpu/cuda/rocm to the sub-crates.
- Created the four per-backend maturin templates, identical except `[project].name` and `[tool.maturin].features`, all with constant `module-name = "mlrs._mlrs"` and `python-source` — every wheel is `import mlrs`.
- Stood up the pure-Python `mlrs` package: `base.py`/`_io.py` shells + 12 sklearn-compatible estimator shells with faithful `__init__`s (`fit` raises `NotImplementedError` until Plan 04); `__init__.py` guards the not-yet-built `_mlrs` extension import and re-exports all 12 classes.
- Built the pytest Nyquist scaffold: `conftest.py` (fixture loader, `sign_flip_allclose`, `label_perm_allclose`, `requires_f64` marker) + 7 collecting stubs, each tagged with its PY-0x requirement.
- RESOLVED the open arrow-59 unknown by reading the vendored `arrow-pyarrow-59.0.0` source: the `FromPyArrow` trait has exactly one method, `from_pyarrow_bound(&Bound<PyAny>)`, reachable as `arrow::pyarrow::FromPyArrow`; recorded in `arrow_symbol_probe.rs` for Plan 02.

## Task Commits

Each task was committed atomically:

1. **Task 2: Pin pyo3 0.28 + wire backend features** - `2780514` (feat)
2. **Task 3: Four per-backend maturin pyproject templates** - `df3b6e6` (feat)
3. **Task 4: Pure-Python mlrs package skeleton** - `55de960` (feat)
4. **Task 5: pytest scaffold + arrow FromPyArrow symbol** - `a1f6296` (test)

_Task 1 was the resolved blocking-human decision gate (Python dep floors); no code/commit._

## Files Created/Modified
- `Cargo.toml` - Added the `pyo3 = "0.28"` workspace pin with ABI-pin rationale comment.
- `crates/mlrs-py/Cargo.toml` - pyo3/arrow deps + path deps + cpu/wgpu/cuda/rocm `[features]`.
- `crates/mlrs-py/pyproject/{cpu,wgpu,cuda,rocm}.pyproject.toml` - per-backend maturin templates (dep floors numpy>2.0.0, pyarrow>=14, scikit-learn>=1.6).
- `crates/mlrs-py/python/mlrs/{__init__,base,_io,linear,cluster,decomposition,neighbors}.py` - pure-Python package + 12 estimator shells.
- `crates/mlrs-py/python/tests/{conftest,test_oracle_linear,test_oracle_cluster,test_oracle_decomposition,test_oracle_neighbors,test_estimator_checks,test_dtype,test_params}.py` - pytest Nyquist scaffold.
- `crates/mlrs-py/src/arrow_symbol_probe.rs` - non-compiled doc recording the resolved arrow-59 ingress symbol.

## Decisions Made
- **Python dep floors (Task 1 human gate):** `numpy>2.0.0` (intentional and exact — numpy 2.x only, strictly greater than 2.0.0, NOT normalized to `>=2.0.0`/`>=1.26`), `pyarrow>=14`, `scikit-learn>=1.6`. Written verbatim into all four templates.
- **arrow-59 ingress symbol:** `ArrayData::from_pyarrow_bound` is the only `FromPyArrow` method in arrow-59 (no non-`_bound` variant); the trait is re-exported via `pub use arrow_pyarrow as pyarrow`. Plan 02 uses `arrow::pyarrow::FromPyArrow::from_pyarrow_bound`.
- **pyo3 0.28 pin** is the single linked ABI — the only dep overriding the workspace track-latest policy (arrow-59's pyarrow feature transitively pins 0.28; mixing 0.29 links two PyInit ABIs and crashes the wheel at import).

## Deviations from Plan
None - plan executed exactly as written. The approved dep floor `numpy>2.0.0` (from the Task 1 decision) was used verbatim, which differs from the RESEARCH default `numpy>=1.26` quoted in the plan's option text — this is the intended outcome of the human decision gate, not a deviation.

## Issues Encountered
None. The arrow-59 `FromPyArrow` method name (the one open research unknown) was resolved by reading the vendored crate source at `~/.cargo/registry/src/.../arrow-pyarrow-59.0.0/src/lib.rs` (trait at L95-99, `impl FromPyArrow for ArrayData` at L260) — confirming `from_pyarrow_bound` and the absence of any non-`_bound` variant.

## User Setup Required
The plan's `user_setup` notes that the Python build/test toolchain (maturin/pyarrow/scikit-learn/numpy/pytest) is not installed and PEP 668 blocks system pip. A `/tmp/mlrs-venv` was NOT required for this plan — all Python verification used AST parsing (no third-party imports). The venv is needed starting Plan 05 (`maturin develop` + `pytest --collect-only`). To create it:

```
python3 -m venv /tmp/mlrs-venv
/tmp/mlrs-venv/bin/pip install 'maturin>=1.14,<2' 'pyarrow>=14' 'scikit-learn>=1.6' 'numpy>2.0.0' pytest
```

## Next Phase Readiness
- **Plan 02 (ingress/egress)** is unblocked: the arrow FromPyArrow symbol is pinned, pyo3 0.28 + arrow pyarrow compile, and the crate has its backend features.
- **Plan 03 (#[pyclass] wrappers)** has the 12 target class names + the dispatch enum plan documented.
- **Plan 04 (shim logic)** has the importable pure-Python shells with faithful `__init__`s to fill in.
- **Plan 05/06 (wheels + tests)** have the four templates and the collecting test scaffold to flesh out; the `/tmp/mlrs-venv` must exist before `maturin develop`/`pytest`.
- No blockers.

## Self-Check: PASSED

---
*Phase: 06-python-surface-pyo3-estimators-per-backend-wheels*
*Completed: 2026-06-13*
