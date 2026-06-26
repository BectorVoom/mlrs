---
status: complete
phase: 16-builder-retrofit-sweep-shim-coverage
source: [16-VERIFICATION.md]
started: 2026-06-25T05:05:00Z
updated: 2026-06-26T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. UMAP full oracle suite (numeric property + transform sub-gate)
expected: |
  All umap_test tests pass after the Phase-16 typestate convergence. During phase
  verification, 22/27 gates ran GREEN with zero failures — including every structural
  gate, smooth_knn (5 metrics), fuzzy_union (5 metrics), ab_fit, defaults_equal,
  fit_rejects_n_components_ge_n, all 5 spectral_init property gates, and the layout
  property + reproducibility gates. The 5 `transform_property_*` gates (out-of-sample
  transform) were still running clean (no failure/panic) when in-session polling stopped
  after ~30 min — the documented "backend test suite slow" landmine under the CPU-MLIR
  backend. This UAT item confirms the transform sub-gate completes green on a host that
  can run the full suite to completion (CI, or a longer local run).
result: pass
evidence: |
  Full suite run to completion on 2026-06-26:
  `test result: ok. 35 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished
  in 14237.04s` (exit 0). All 5 `transform_property_{euclidean,manhattan,cosine,chebyshev,
  minkowski}` out-of-sample gates passed, plus all 5 `layout_property_*`, all 5
  `spectral_init_*`, `reproducible_f64`, `fit_roundtrip`, and the `fit_no_leak` PoolStats
  memory gate. Zero failures — the typestate convergence did not perturb any UMAP numeric
  path. The single behavior-unverified item from 16-VERIFICATION.md is now settled GREEN.

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none]
