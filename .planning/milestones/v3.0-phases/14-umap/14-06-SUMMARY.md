---
phase: 14-umap
plan: 06
subsystem: manifold
tags: [umap, validation, input-guard, security, cr-02]
status: complete
requires:
  - "Umap::fit pipeline (Plan 04/05)"
  - "AlgoError::InvalidNComponents (error.rs:41)"
  - "SpectralEmbedding::fit guard (spectral_embedding.rs:155) as source-of-truth pattern"
provides:
  - "n_components < n guard in Umap::fit (typed-error rejection before any device launch)"
  - "fit_rejects_n_components_ge_n regression test"
affects:
  - crates/mlrs-algos/src/manifold/umap.rs
  - crates/mlrs-algos/tests/umap_test.rs
tech-stack:
  added: []
  patterns:
    - "Data-dependent fit-entry guard returning AlgoError::InvalidNComponents (mirrors SpectralEmbedding::fit)"
key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/manifold/umap.rs
    - crates/mlrs-algos/tests/umap_test.rs
decisions:
  - "Guard condition uses `self.n_components >= n` (equivalently n_components + 1 > n) — the spectral drop_first path needs n_components+1 eigenvectors, so n_components must be strictly < n."
  - "Fix rejects bad input at fit entry rather than touching spectral.rs::recover — matches the sibling SpectralEmbedding contract and stops the underflow before recover is ever reached."
  - "Lower bound (n_components >= 1) left to build() per CONTEXT; the fit-path guard is the data-DEPENDENT upper bound only."
metrics:
  duration_min: 3
  completed: "2026-06-24"
  tasks: 2
  files: 2
---

# Phase 14 Plan 06: UMAP n_components Guard (CR-02) Summary

Closed GAP 2 (CR-02 BLOCKER): `Umap::fit` now rejects `n_components >= n` with the typed `AlgoError::InvalidNComponents { estimator: "umap", .. }` before any device launch, preventing the `spectral::recover` `n - 1 - r` usize underflow (panic in debug / OOB device read in release) on adversarial-but-valid-typed input.

## What Was Built

**Task 1 — Guard (`umap.rs`, commit `8ed8fdb`)**
Added a data-dependent guard in `Umap::fit` immediately after `validate_geometry(x, shape)?` and before `run_umap_layout`. When `self.n_components >= n` it returns `AlgoError::InvalidNComponents { estimator: "umap", requested: self.n_components, max: n.saturating_sub(1) }`. The guard mirrors `SpectralEmbedding::fit` (spectral_embedding.rs:155) in structure and error type. No kernel / `spectral.rs` changes — the bad input is rejected upstream so `recover` is never reached.

**Task 2 — Test (`umap_test.rs`, commit `0244de2`)**
Added `fit_rejects_n_components_ge_n` (`#[test]`): builds a tiny `n=2`, `d=3` inline host buffer with `n_components=2` (so `n_components >= n`), calls `.fit(...)`, and asserts via `matches!` that the result is `Err(AlgoError::InvalidNComponents { estimator: "umap", requested: 2, .. })` — proving a typed error, not a panic/OOB. Gated through `gate_f64` (skip-with-log on f64-incapable backends). Added `AlgoError` to the error imports.

## Verification

| Check | Result |
|-------|--------|
| `cargo build -p mlrs-algos --features cpu` | PASS (Task 1) |
| `cargo test ... fit_rejects_n_components_ge_n` | PASS (1 passed) |
| `grep InvalidNComponents umap.rs` (fit body) | line 460 — present |
| Regression: `build_rejects_bad_min_dist` | PASS |
| Regression: `fit_roundtrip` | PASS |
| Regression: `defaults_equal` | PASS |
| Regression: `metrics_table_covers` | PASS |

## Threat Mitigation

| Threat ID | Disposition | Outcome |
|-----------|-------------|---------|
| T-14-06-01 (Tampering/DoS — recover underflow) | mitigate | Closed — guard rejects `n_components >= n` at fit entry, before launch. |
| T-14-06-02 (Info disclosure — release OOB read) | mitigate | Closed — same guard eliminates the OOB index path entirely. |

## Deviations from Plan

None — plan executed exactly as written. Both acceptance criteria sets met; no auto-fixes (Rules 1–3) or architectural decisions (Rule 4) required. No authentication gates.

## Known Stubs

None.

## Self-Check: PASSED

- FOUND: crates/mlrs-algos/src/manifold/umap.rs
- FOUND: crates/mlrs-algos/tests/umap_test.rs
- FOUND: commit 8ed8fdb (guard)
- FOUND: commit 0244de2 (test)
