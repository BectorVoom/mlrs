---
status: testing
phase: 16-builder-retrofit-sweep-shim-coverage
source: [16-VERIFICATION.md]
started: 2026-06-25T05:05:00Z
updated: 2026-06-25T05:05:00Z
---

## Current Test

number: 1
name: UMAP full oracle suite — fit/transform/fit_transform numeric property gates after typestate convergence
expected: |
  `cargo test -p mlrs-algos --features cpu --test umap_test` passes ALL tests, in
  particular the transform sub-gate (`transform_property_{euclidean,manhattan,cosine,
  chebyshev,minkowski}`) on out-of-sample points, plus trustworthiness / kNN-overlap /
  same-random_state reproducibility property gates.
awaiting: user response

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
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
