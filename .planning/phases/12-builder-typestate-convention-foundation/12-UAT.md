---
status: testing
phase: 12-builder-typestate-convention-foundation
source: [12-VERIFICATION.md]
started: 2026-06-23T02:13:16Z
updated: 2026-06-23T02:13:16Z
---

## Current Test

number: 1
name: Live PyO3 estimator integration for PyUMAP and PyHDBSCAN
expected: |
  In an environment with maturin + pyarrow installed, build the wheel
  (`maturin develop --features cpu`) and verify via the real PyO3 capsule boundary:
  - `UMAP().fit(X, rows, cols)` stores AnyUmap::F32/F64; `embedding_f32()`/`embedding_f64()`
    returns a Vec of length rows*2 (zeros shell).
  - `HDBSCAN().fit(X, rows, cols)` stores AnyHdbscan::F32/F64; `labels_()` returns Vec<i32>
    of length rows, all -1.
  - Both raise PyValueError (NotFittedError analog) when an accessor is called before fit.
  - `build_err_to_py` surfaces `BuildError::InvalidMinDist` / `InvalidMinClusterSize`
    as Python ValueError.
awaiting: user response

## Tests

### 1. Live PyO3 estimator integration for PyUMAP and PyHDBSCAN
expected: |
  Build the wheel with maturin (cpu feature) and, through the real Python interpreter
  + pyarrow capsule FFI path:
  - UMAP().fit(X, rows, cols) stores AnyUmap::F32/F64; embedding_f32()/embedding_f64()
    returns rows*2 zeros.
  - HDBSCAN().fit(X, rows, cols) stores AnyHdbscan::F32/F64; labels_() returns rows × i32, all -1.
  - Accessing an estimator before fit raises PyValueError (NotFittedError analog).
  - BuildError::InvalidMinDist / InvalidMinClusterSize surface as Python ValueError.
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
