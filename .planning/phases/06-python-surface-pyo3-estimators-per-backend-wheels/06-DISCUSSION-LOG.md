# Phase 6: Python Surface — PyO3 Estimators & Per-Backend Wheels - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-13
**Phase:** 6-Python Surface — PyO3 Estimators & Per-Backend Wheels
**Areas discussed:** Package topology, NumPy/Arrow I/O (ingress + egress), dtype × backend, Wheel naming & UX

> Note: the command was invoked as `/gsd-discuss-phase 7`, but Phase 7 does not
> exist — the roadmap ends at Phase 6. The user confirmed they meant Phase 6.

---

## Package topology

| Option | Description | Selected |
|--------|-------------|----------|
| Thin Python shim over compiled core | Compiled `_mlrs` extension exposes low-level entry points; importable `mlrs` is pure Python subclassing sklearn.base + mixins, delegating to `_mlrs`. get_params/repr/tags/clone from sklearn; numpy↔Arrow glue + output_type in Python. Matches cuML. | ✓ |
| Pure compiled #[pyclass], no Python layer | Each estimator is a pyclass and IS the API; sklearn BaseEstimator semantics hand-implemented in Rust. Zero Python in wheel. | |
| Pyclass + minimal Python __init__.py only | Estimators are pyclasses; tiny Python `__init__.py` adds only the hardest sklearn glue. | |

**User's choice:** Thin Python shim over compiled core.
**Notes:** Cleanest path to passing `sklearn.utils.estimator_checks`; mirrors cuML's Cython-core + Python-Base architecture.

---

## NumPy/Arrow I/O — Ingress

User directive: "I want to select same method of cuML." Verified from the
reference source that cuML's ingress uses `__array_interface__` /
`__cuda_array_interface__` / DLPack (via `input_to_cuml_array`), which conflicts
with the locked PY-03 Arrow-PyCapsule mandate. Reconciliation options presented:

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid: cuML-style API, Arrow PyCapsule boundary | Shim accepts cuML-style inputs (numpy/pyarrow/lists), normalizes to 1-D pyarrow, crosses via `__arrow_c_array__` PyCapsule + (rows,cols), imported via arrow-rs FFI, reuses existing validate_f32/f64 bridge. PY-03 intact, no requirement change. | ✓ |
| Full cuML ingress — amend PY-03 | Adopt array-interface/DLPack end-to-end; requires editing locked PY-03 + ROADMAP criterion 3. | |
| Keep strict Arrow PyCapsule ingress only | Arrow-native (or pyarrow-convertible) only; no cuML accept-anything surface. | |

**User's choice:** Hybrid — cuML-style API surface, Arrow PyCapsule boundary.
**Notes:** Gives cuML-like input flexibility while honoring PY-03 literally; adds pyarrow as a runtime dependency. (An earlier pre-interrupt selection of "normalize to pyarrow, cross via PyCapsule" for a numpy-only framing was superseded by this cuML-aligned reframing after the user reselected Area 2.)

## NumPy/Arrow I/O — Egress

| Option | Description | Selected |
|--------|-------------|----------|
| cuML output_type routing | Configurable `output_type` param + global override, default 'input' = mirror input container; v1 set = numpy + pyarrow; labels/indices int32. Matches cuML Base.output_type / to_output. | ✓ |
| Keep always-numpy | Always return numpy ndarray. (Initially selected, then revised to the cuML method.) | |
| Symmetric Arrow PyCapsule both directions | Outputs cross back as PyCapsule. | |
| Mirror input container (output_type routing) | (Folded into the cuML option above.) | |

**User's choice:** cuML output_type routing.
**Notes:** PY-03 doesn't constrain egress, so this is fully compatible; numpy-in→numpy-out under mirror keeps estimator_checks passing. v1 supported output set narrower than cuML (no cupy/cuDF/numba).

---

## dtype × backend

**Q1 — f64 on an f64-incapable backend (notably mlrs-rocm):**

| Option | Description | Selected |
|--------|-------------|----------|
| Capability-query + clear error | Capability flag (on capability.rs); f64 on f64-incapable backend raises a clear Python exception. Preserve dtype where supported; never silently lose precision. | ✓ |
| cuML-style warn + downcast to f32 | UserWarning + compute f32, like cuML convert_dtype. | |
| Downcast silently, no warning | Compute f32 with no message. | |

**Q2 — default float dtype + preservation:**

| Option | Description | Selected |
|--------|-------------|----------|
| Preserve input float dtype; non-float → f64 | f32→f32, f64→f64; int/list default f64 (f64-capable) / f32 (incapable). Extension dispatches on Arrow dtype via internal enum. | ✓ |
| cuML-style default float32 | Default inputs to float32. | |
| Always compute f64 where available | Upcast everything to f64 on f64-capable backends. | |

**User's choice:** Capability-query + clear error; preserve input float dtype (non-float → f64 where supported).
**Notes:** Departs from cuML's silent dtype conversion deliberately — rocm f64-incapability is a hard limit and silent downcast would break the 1e-5 contract.

---

## Wheel naming & UX

**Q1 — import name vs distribution name:**

| Option | Description | Selected |
|--------|-------------|----------|
| Constant `import mlrs`, distinct dist names | All wheels expose `import mlrs`; dist names differ (mlrs-cpu/-wgpu/-cuda/-rocm). Code portable; install exactly one. | ✓ |
| Distinct import names per backend | `import mlrs_cpu` etc.; allows side-by-side, breaks portability. | |

**Q2 — missing-driver failure:**

| Option | Description | Selected |
|--------|-------------|----------|
| Import-time probe + clear error | Probe driver / cubecl client init on `import mlrs`; raise ImportError with actionable message if absent. Matches criterion 4 wording. | ✓ |
| Lazy probe on first compute | Import succeeds; error on first estimator construction / fit(). | |

**User's choice:** Constant `import mlrs` + distinct dist names; import-time driver probe with clear error.
**Notes:** Shared `mlrs` namespace means install exactly one backend wheel — accepted constraint. abi3-py312 locked by criterion 4.

---

## Claude's Discretion

- PyO3 wrapper/enum shape for dtype dispatch; module/file layout of the `mlrs`
  Python package and `_mlrs` extension; sklearn mixin composition per estimator.
- BufferPool + cubecl client ownership/lifecycle across the boundary and
  thread-safety under `Python::allow_threads` / joblib (flagged for research).
- The "relevant" `sklearn.utils.estimator_checks` subset per estimator family.
- Exact sklearn-matching `get_params`/`set_params` hyperparameter names per estimator.
- The maturin multi-distribution naming mechanism (research flag).
- `score()` metric per family (inherited from sklearn mixins).

## Deferred Ideas

- cupy/cuDF/numba output_type targets (v1 = numpy + pyarrow only).
- Full cuML array-interface/DLPack ingress (would amend PY-03).
- Multiple backends installable side-by-side (distinct import names).
- `cuml.accel`-style transparent sklearn acceleration (V2-07, out of v1 scope).
- Multi-GPU / Dask Python surface (V2-06, out of v1 scope).
