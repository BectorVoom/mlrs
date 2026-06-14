---
status: testing
phase: 06-python-surface-pyo3-estimators-per-backend-wheels
source: [06-VERIFICATION.md]
started: 2026-06-14T00:00:00Z
updated: 2026-06-14T00:00:00Z
---

## Current Test

number: 1
name: cuda wheel live import on a CUDA host
expected: |
  On a host WITH a CUDA driver: `pip install target/wheels/mlrs_cuda-*.whl`
  then `python -c "import mlrs; print(mlrs.__name__)"` prints `mlrs` with no error.
awaiting: user response

## Tests

### 1. cuda wheel live import on a CUDA host
expected: |
  On a host WITH a CUDA driver, installing the cuda wheel and `import mlrs`
  succeeds (prints `mlrs`). This is the one backend whose live import could not
  be exercised in the build environment (cuda = compile-only per project
  constraints). User-approved deferral from plan 06-06 Task 3.
result: [pending]

### 2. foreign-driver-absent ImportError on real hardware
expected: |
  On a host WITHOUT the matching driver (e.g. the cuda wheel on a non-CUDA box),
  `python -c "import mlrs"` raises a clear `ImportError` naming the backend —
  NOT a segfault/abort. Verifies the catch_unwind import probe + `panic = "unwind"`
  on real hardware. User-approved deferral from plan 06-06 Task 3.
result: [pending]

### 3. two-backend-wheel namespace overwrite in one env
expected: |
  Installing two backend wheels (e.g. mlrs_cpu and mlrs_wgpu) into a single
  environment overwrites the shared `mlrs` namespace; the last-installed backend
  wins (D-07, accepted-by-design). Confirm behavior is the documented overwrite,
  not a partial/broken hybrid install.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
