---
phase: 15
plan: 07
subsystem: cluster/hdbscan
tags: [hdbscan, clustering, rust-api, typestate, fit_predict, gap-closure]
requires:
  - "Hdbscan Fit::fit pipeline (15-03..15-06)"
  - "Hdbscan<F, Fitted>::labels accessor"
provides:
  - "Hdbscan::fit_predict convenience method on impl<F> Hdbscan<F, Unfit>"
affects:
  - "Phase 16 PyO3 surface (wraps fit_predict)"
tech-stack:
  added: []
  patterns:
    - "Consuming-self fit_predict wrapper (typestate Unfit -> Fitted -> dropped)"
key-files:
  created: []
  modified:
    - "crates/mlrs-algos/src/cluster/hdbscan.rs"
    - "crates/mlrs-algos/tests/hdbscan_test.rs"
decisions:
  - "fit_predict consumes self (NOT &mut self) because Hdbscan::Fit::fit moves self and returns a new Hdbscan<F, Fitted>; this diverges from DBSCAN's &mut self fit_predict."
  - "Behavioral-equivalence test only (no oracle re-gate): underlying fit/labels_ path is already exhaustively gated by the existing 39-test suite."
metrics:
  duration_min: 5
  completed: 2026-06-24
  tasks: 2
  files: 2
status: complete
requirements: [HDBS-01]
---

# Phase 15 Plan 07: Hdbscan::fit_predict Gap-Closure Summary

Added the missing `Hdbscan::fit_predict` convenience method — the single Phase-15
verification gap (HDBS-01, BLOCKED) — closing ROADMAP Phase 15 SC #1's
`fit`/`fit_predict` requirement at the Rust level, plus a behavioral-equivalence test.

## What Was Built

- **`Hdbscan::fit_predict`** (`crates/mlrs-algos/src/cluster/hdbscan.rs`): a `pub fn`
  on the existing `impl<F> Hdbscan<F, Unfit>` block (alongside `new`/`builder`/
  `into_builder`). Signature:
  `pub fn fit_predict(self, pool: &mut BufferPool<ActiveRuntime>, x: &DeviceArray<ActiveRuntime, F>, shape: (usize, usize)) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError>`.
  It consumes `self`, calls `Fit::fit(self, pool, x, None, shape)?`, reads
  `labels()` off the returned `Hdbscan<F, Fitted>`, and returns a fresh
  device-resident `i32` buffer via `DeviceArray::from_host(pool, &labels)`.
  Noise stays pinned at `-1` by construction in `labels_`.

- **`fit_predict_matches_fit_then_labels`** test (`crates/mlrs-algos/tests/hdbscan_test.rs`):
  fits two identical `Hdbscan::<f64>::new()` estimators on a tiny self-contained
  two-cluster Euclidean blob (16 rows, 2 features, no fixture). One goes through
  `Fit::fit` + `labels()`; the other through `fit_predict` (read back via `to_host`).
  Asserts element-for-element label equality. Honors the `skip_f64` capability gate;
  runs and passes on the cpu f64 gate.

## Critical Typestate Correctness (the plan's trap)

The verifier's `missing:` proposal copied DBSCAN's `&mut self` signature, which would
NOT compile for Hdbscan. Confirmed from the code (`hdbscan.rs:463-469`) that
`Fit::fit` CONSUMES `self` and returns a new `Hdbscan<F, Fitted>`. The wrapper
therefore consumes `self` too (receiver `self` by value), and the test constructs
two separate estimators since both `fit` and `fit_predict` move `self`.

## Verification

- `grep -c 'fn fit_predict' crates/mlrs-algos/src/cluster/hdbscan.rs` → `1` (was `0`).
- `cargo build -p mlrs-algos --features cpu` → finished cleanly.
- `cargo test -p mlrs-algos --test hdbscan_test --features cpu fit_predict_matches_fit_then_labels -- --exact`
  → `1 passed; 0 failed`.
- Full oracle suite intentionally NOT re-run (plan directive; wrapped path already
  gated by the existing 39-test suite).

## Deviations from Plan

None - plan executed exactly as written. Derived the consuming-`self` signature
from the source as instructed; both tasks landed purely additively with no changes
to `Fit::fit`, the `labels` accessor, fields, or any other method.

## Commits

- `cff2458` feat(15-07): add Hdbscan::fit_predict convenience method (HDBS-01)
- `2394efd` test(15-07): fit_predict == fit-then-labels() equivalence (HDBS-01)

## Self-Check: PASSED
- `crates/mlrs-algos/src/cluster/hdbscan.rs` — FOUND (fit_predict present)
- `crates/mlrs-algos/tests/hdbscan_test.rs` — FOUND (test present, passing)
- Commit `cff2458` — FOUND
- Commit `2394efd` — FOUND
