---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 00
subsystem: trait-surface
tags: [typestate, traits, shim, ast-purity, wave-0]
requires: []
provides:
  - "typestate::PredictLabels<F> accessor trait"
  - "typestate::KNeighbors<F> accessor trait"
  - "typestate::ScoreSamples<F> accessor trait"
  - "typestate::PredictProba<F> accessor trait"
  - "typestate::PredictLogProba<F> accessor trait"
  - "typestate::Transform::inverse_transform default method"
  - "test_init_purity_ast static SHIM-01 gate"
affects:
  - "Plans 16-01..16-08 (estimator sweep — accessor traits now exist as migration targets)"
  - "Plan 16-10 (shim coverage — AST purity gate grows with the matrix)"
  - "Plan 16-11 (traits.rs deletion — typestate is now the full 9-trait surface)"
tech-stack:
  added: []
  patterns:
    - "&self accessor traits impl'd ONLY on Fitted-tagged estimator (no associated type Fitted)"
    - "f64 builder-setter convention; build::<F>() narrows (A5)"
    - "AST-based static __init__ purity check (no FFI, no compiled extension)"
key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/typestate.rs
    - crates/mlrs-algos/tests/typestate_test.rs
    - crates/mlrs-py/python/tests/test_params.py
decisions:
  - "5 accessor traits ported verbatim from traits.rs signatures (no validation added/dropped at the trait boundary) — D-01/T-16-V5"
  - "Transform::inverse_transform default returns AlgoError::Unsupported, matching traits.rs verbatim"
  - "f64 builder-setter convention locked + documented in typestate.rs module doc (A5)"
  - "AST purity gate parametrized over the shared ALL_12 list so Plan 10 matrix expansion grows it automatically"
metrics:
  duration: ~12m
  completed: 2026-06-24
  tasks: 3
  files: 3
status: complete
---

# Phase 16 Plan 00: typestate trait-surface expansion + AST shim-purity gate — Summary

Grew `mlrs_algos::typestate` from 4 of 9 traits to the full 9-trait surface (5 `&self` accessor traits ported verbatim from `traits.rs` + a `Transform::inverse_transform` default), proved coherence with a kernel-free `typestate_test.rs` gate, locked the `f64` builder-setter convention in the module doc, and added a static AST-based `__init__`-purity test that enforces SHIM-01 without importing the compiled `_mlrs` extension.

This is the Wave-0 BLOCKING prerequisite: the 5 accessor traits are the migration targets that classifiers/neighbors/density/NB estimators (Plans 01-08) and the shim work (Plan 10) consume — no estimator could migrate to typestate until these existed.

## What Was Built

### Task 1 — 5 accessor traits + inverse_transform default (`crates/mlrs-algos/src/typestate.rs`, commit `1246d29`)

Added 5 new `&self` accessor traits, each `<F> where F: Float + CubeElement + Pod`, signatures ported VERBATIM from the corresponding `traits.rs` traits (same method name, same host-geometry args, same `Result<_, AlgoError>` return):

| Trait | Method | Returns |
|-------|--------|---------|
| `PredictLabels<F>` | `predict_labels(&self, pool, x, shape)` | `DeviceArray<ActiveRuntime, i32>` |
| `KNeighbors<F>` | `kneighbors(&self, pool, x, shape, k: usize)` | `(DeviceArray<F>, DeviceArray<i32>)` |
| `ScoreSamples<F>` | `score_samples(&self, pool, x, shape)` | `DeviceArray<ActiveRuntime, F>` |
| `PredictProba<F>` | `predict_proba(&self, pool, x, shape)` | `DeviceArray<ActiveRuntime, F>` |
| `PredictLogProba<F>` | `predict_log_proba(&self, pool, x, shape)` | `DeviceArray<ActiveRuntime, F>` |

Common host-geometry arg shape for all 5: `(pool: &mut BufferPool<ActiveRuntime>, x: &DeviceArray<ActiveRuntime, F>, shape: (usize, usize))` (`KNeighbors` additionally takes `k: usize` as its last arg). None carry an associated `type Fitted` — they read fitted state, so they are impl'd ONLY on the `Fitted`-tagged estimator (mirroring `Transform`).

Added the `Transform::inverse_transform` default method (PCA reconstruction path), ported verbatim from `traits.rs:145-155`:
```rust
fn inverse_transform(&self, _pool, _z, _shape) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
    Err(AlgoError::Unsupported { estimator: "transform", operation: "inverse_transform" })
}
```

Module-doc updates: noted `traits.rs` is being hard-deleted this phase (D-01) and typestate is now the single surface (superseding the old "30 estimators continue to compile against traits.rs ... FROZEN" note); documented the **f64 builder-setter convention** (A5) — setters take `f64`, `build::<F>()` narrows via cast. The existing Fit/Predict/Transform/PartialFit traits and the sealed `State` machinery were NOT altered.

### Task 2 — typestate_test.rs coherence proof (`crates/mlrs-algos/tests/typestate_test.rs`, commit `7bffa5a`)

Extended the existing test file with a kernel-free type/trait-surface proof:
- A test-only `Marker<F, S = Unfit>` estimator carrying `PhantomData<S>`.
- `impl PredictLabels` + `impl ScoreSamples` on `Marker<F, Fitted>` — references **2 of the 5** new traits by name (proving they are importable from `mlrs_algos::typestate` and impl'able only on `Fitted`).
- `impl Transform for Marker<F, Fitted>` that leaves `inverse_transform` defaulted; the test `transform_inverse_transform_default_returns_unsupported` asserts the default surfaces `AlgoError::Unsupported { operation: "inverse_transform", .. }`.
- `new_accessor_traits_resolve_on_fitted_marker` exercises both accessor methods on a host-built `DeviceArray` (no CubeCL kernel launched).

All 6 tests in the file pass under `cargo test --features cpu --test typestate_test` (4 pre-existing + 2 new).

### Task 3 — AST __init__-purity gate (`crates/mlrs-py/python/tests/test_params.py`, commit `f268af1`)

Added `test_init_purity_ast`, parametrized over the shared `ALL_12` list (so Plan 10's matrix expansion grows it automatically). It parses `inspect.getsource(cls.__init__)` with the `ast` module and asserts every body statement is a bare `self.<name> = <name>` assignment:
- statement is `ast.Assign` (rejects `if`/`for`/`raise`/`assert`/expression-call bodies),
- single target of shape `self.<attr>` (`ast.Attribute` over `ast.Name` `id == "self"`),
- value is a bare `ast.Name` (rejects `ast.Call`/`ast.BinOp`/`ast.Compare` — any validation/computation),
- `tgt.attr == stmt.value.id` (stored under the SAME identifier — `self.x = x`, never `self.x = y`).

This is the strongest SHIM-01 guarantee achievable without FFI (D-07 step 3). Added `import ast` + `import inspect` at module top. The existing runtime purity test (`test_init_purity_stores_kwargs_verbatim`) was left untouched — the AST test is additive and stricter.

## Deviations from Plan

None — plan executed exactly as written. The data-independent verification adjustment below was an implementation detail, not a plan deviation:
- In `typestate_test.rs` the `inverse_transform`-default assertion uses an explicit `match` instead of `Result::expect_err`, because `DeviceArray` does not implement `Debug` (which `expect_err` requires on the `Ok` type). The assertion semantics are unchanged.

## Authentication Gates

None.

## Verification Environment Note (Python)

`crates/mlrs-py/python/mlrs/__init__.py` imports `_io`, which imports `pyarrow` unconditionally — so importing the `mlrs` package (even for a pure-Python AST test) requires `pyarrow`, which is not in the base interpreter here (consistent with the project memory note "Python wheel untestable in env"). The AST test itself needs NO compiled `_mlrs` extension; it only needs the importable Python package. It was verified green in a throwaway venv with `pyarrow scikit-learn numpy pytest` installed:
```
$VENV/bin/python -m pytest tests/test_params.py -k init_purity -q
24 passed, 26 deselected
```
The 24 = 12 new AST cases + 12 pre-existing runtime-purity cases (both match `-k init_purity`). CI/wheel environments that ship `pyarrow` run this unchanged.

## Acceptance Evidence

- `cargo build -p mlrs-algos --features cpu` — Finished (clean).
- `grep -cE 'pub trait (PredictLabels|KNeighbors|ScoreSamples|PredictProba|PredictLogProba)' typestate.rs` → **5**.
- `grep -c 'fn inverse_transform' typestate.rs` → **1**.
- `cargo test --features cpu --test typestate_test` → **6 passed**.
- `grep -c 'import ast' test_params.py` → **1**.
- `pytest test_params.py -k init_purity` → **24 passed** (no compiled extension).
- AST test FAILS on an injected `self.alpha = float(alpha)` impure body (verified, then reverted clean — `git diff linear.py` empty).
- `traits.rs` still present (its deletion is Plan 11; the existing 4 typestate traits + sealed State machinery unchanged).

## For Downstream Plans

- The 5 accessor traits live in `crates/mlrs-algos/src/typestate.rs`; import as `use mlrs_algos::typestate::{PredictLabels, KNeighbors, ScoreSamples, PredictProba, PredictLogProba};`. Each is `&self`, `Fitted`-only, no associated type.
- `Transform` now carries the `inverse_transform` default (PCA overrides it; all other transformers leave the `Unsupported` default).
- Builder-setter convention (A5): setters take `f64`; `build::<F>()` narrows — documented in the typestate.rs module header.
- The AST purity gate is at `crates/mlrs-py/python/tests/test_params.py::test_init_purity_ast`, parametrized over `ALL_12`. Plan 10 expands `EXPECTED_PARAMS`/`ALL_12`; the AST test grows with it automatically (no Plan-10 edit to the test body needed).

## Self-Check: PASSED

- `crates/mlrs-algos/src/typestate.rs` — FOUND, modified, builds.
- `crates/mlrs-algos/tests/typestate_test.rs` — FOUND, 6 tests pass.
- `crates/mlrs-py/python/tests/test_params.py` — FOUND, 24 purity cases pass.
- Commit `1246d29` — FOUND.
- Commit `7bffa5a` — FOUND.
- Commit `f268af1` — FOUND.
