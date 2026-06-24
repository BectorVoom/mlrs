---
phase: quick-260625-8ri
plan: 01
subsystem: error-handling
tags: [error-handling, primerror, in-04, pyo3-boundary, panic-free]
requires:
  - "AlgoError::Prim(#[from] PrimError) (mlrs-algos error.rs)"
provides:
  - "PrimError::InternalNone variant (internal-invariant / unexpected-None)"
  - "Panic-free Shared-path column_reduce call sites in 6 mlrs-algos estimators"
affects:
  - "crates/mlrs-core/src/error.rs"
  - "crates/mlrs-algos/src/**"
tech-stack:
  added: []
  patterns:
    - "Typed-error propagation (.ok_or(...)? ) replacing documented-unreachable .expect() panics across the PyO3 boundary"
key-files:
  created: []
  modified:
    - crates/mlrs-core/src/error.rs
    - crates/mlrs-algos/src/decomposition/pca.rs
    - crates/mlrs-algos/src/linear/linear_regression.rs
    - crates/mlrs-algos/src/linear/ridge.rs
    - crates/mlrs-algos/src/covariance/ledoit_wolf.rs
    - crates/mlrs-algos/src/covariance/empirical_covariance.rs
    - crates/mlrs-algos/src/density/kernel_density.rs
decisions:
  - "InternalNone carries allocation-free &'static str operand + context fields, matching the operand-label style of every existing PrimError variant (no owned String)."
  - "Used the bare AlgoError::Prim(PrimError::InternalNone{..}) value with .ok_or(...)? (no #[from] needed at the call site) so the propagated value is already an AlgoError."
metrics:
  duration: ~6m
  completed: 2026-06-25
status: complete
---

# Quick Task 260625-8ri: Implement IN-04 — add a PrimError variant Summary

Added a typed `PrimError::InternalNone` variant in `mlrs-core` for the "primitive documented to return `Some` on a path nonetheless returned `None`" internal-invariant condition, then replaced all 6 `.expect("shared path is never plane-gated to None")` panics at Shared-path `column_reduce` call sites in `mlrs-algos` with `.ok_or(AlgoError::Prim(PrimError::InternalNone{..}))?` typed-error propagation — keeping the PyO3 boundary panic-free if the reduce prim's plane-gating contract ever drifts.

## What Was Built

### Task 1 — `PrimError::InternalNone` variant (mlrs-core)
- New enum variant appended after `Overflow` in `crates/mlrs-core/src/error.rs`.
- Fields: `operand: &'static str`, `context: &'static str` (allocation-free, sibling-style).
- thiserror message: `primitive '{operand}' returned an unexpected None ({context}); an internal invariant was violated`.
- Doc comment cites IN-04 and the canonical `column_reduce(.., ReducePath::Shared, ..)` case.
- No match-arm wiring needed: `AlgoError::Prim` uses `#[from]`, and the PyO3 boundary maps via `Display` (`to_string()`).
- Commit: `1423239`

### Task 2 — convert 6 expect() unwraps (mlrs-algos)
- Replaced the `.expect(...)` after each Shared-path `column_reduce(..)?` with `.ok_or(AlgoError::Prim(PrimError::InternalNone { operand: "column_reduce", context: "ReducePath::Shared" }))?` at:
  - `decomposition/pca.rs`
  - `linear/linear_regression.rs`
  - `linear/ridge.rs`
  - `covariance/ledoit_wolf.rs`
  - `covariance/empirical_covariance.rs`
  - `density/kernel_density.rs`
- Added `PrimError` to the `mlrs_core::{...}` import group in `empirical_covariance.rs` and `ledoit_wolf.rs` (the other 4 already imported it).
- No new buffer-release / cleanup introduced — the early `?` on the documented-unreachable `None` branch matches each function's existing `?`-site error-path convention.
- Backend/src and `*_test.rs` occurrences left untouched (out of scope for IN-04).
- Commit: `2a0fcd4`

## Verification

- `cargo check -p mlrs-core` — compiles cleanly (NOTE: mlrs-core has no `cpu` Cargo feature; the `--features cpu` in the plan applies to the consumer crates. mlrs-core checks plainly).
- `cargo check -p mlrs-algos -p mlrs-py --features cpu --lib` — compiles cleanly (only pre-existing dead-code warnings in `mlrs-py/src/estimators/spectral.rs`).
- `cargo check -p mlrs-backend --features cpu --lib` — compiles cleanly (confirms the new variant broke no exhaustive match in the consuming backend crate).
- `grep -rn "shared path is never plane-gated to None" crates/mlrs-algos/src/` — zero matches.
- Backend/src literal occurrences unchanged (5 files, deliberately out of scope).

## Deviations from Plan

**1. [Rule 3 - Blocking] `--features cpu` not valid for `mlrs-core`**
- **Found during:** Task 1 verification.
- **Issue:** The plan's verify command `cargo check -p mlrs-core --features cpu` errors — `mlrs-core` does not declare a `cpu` feature (only `mlrs-backend`, `mlrs-algos`, `mlrs-py` do).
- **Fix:** Ran `cargo check -p mlrs-core` (no feature flag), which is the correct check for that crate. The plan's intent — "mlrs-core compiles cleanly with the new variant" — is satisfied. No source change required.
- **Files modified:** none.

## Known Stubs

None.

## Threat Flags

None — change is internal error-handling refinement; no new network/auth/file/schema surface introduced. The change strictly reduces panic surface across the PyO3 trust boundary.

## Self-Check: PASSED

- Files exist:
  - FOUND: crates/mlrs-core/src/error.rs (InternalNone variant present)
  - FOUND: all 6 mlrs-algos files modified
- Commits exist:
  - FOUND: 1423239 (Task 1)
  - FOUND: 2a0fcd4 (Task 2)
