---
status: passed
phase: 08-kernel-family
source: [08-VERIFICATION.md]
started: 2026-06-21T00:00:00Z
updated: 2026-06-21T00:00:00Z
---

## Current Test

number: 1
name: Python FFI smoke test (PyKernelRidge / PyKernelDensity)
expected: |
  `maturin develop` (cpu) builds the extension, then
  `pytest crates/mlrs-py/tests/test_kernel.py -v` reports 4 passed:
  test_kernel_ridge_predict[f32], [f64], test_kernel_density_score_samples[f32], [f64].
  Shapes correct, log-densities finite, f32/f64 dtype dispatch works (f64 arms RUN on cpu, not skipped).
awaiting: none — user approved (prior 4/4 from 08-05 execution accepted)

## Tests

### 1. Python FFI smoke test (PyKernelRidge / PyKernelDensity)
expected: maturin builds cpu extension; pytest test_kernel.py → 4 passed (f32+f64 × predict + score_samples); shapes correct; log-densities finite; dtype dispatch works.
result: passed (user approved 2026-06-21 — prior 4/4 from 08-05 execution accepted; not re-run due to disk/maturin-rebuild cost)

note: |
  This was already run GREEN 4/4 during plan 08-05 execution
  (`maturin develop --release` with the cpu.pyproject.toml temp-root dance into a
  /tmp venv holding numpy/pyarrow/scikit-learn/pytest; PEP 668). It is re-listed here
  only because the verifier's sandbox could not launch maturin. The Rust FFI surface
  (kernel.rs pyclass wiring, _mlrs registration, dtype-suffixed accessors) is fully
  verified. All Rust oracle/memory gates pass. This UAT item is a re-confirmation, not
  a discovered gap.

  Repro:
    # one-time venv (PEP 668):
    python3 -m venv /tmp/mlrs-py-venv && . /tmp/mlrs-py-venv/bin/activate
    pip install maturin numpy pyarrow scikit-learn pytest
    # build cpu extension (see 08-05-SUMMARY for the cpu.pyproject.toml temp-root dance):
    maturin develop --release   # with the cpu feature pyproject
    pytest crates/mlrs-py/tests/test_kernel.py -v

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
