---
phase: 15-hdbscan
plan: 01
subsystem: mlrs-core
status: complete
tags: [label-perm, hdbscan, HDBS-02, oracle-gate, test-utility]
requires: []
provides:
  - "mlrs_core::label_perm::best_match_accuracy_pinned_noise"
  - "mlrs_core::best_match_accuracy_pinned_noise (re-export)"
affects:
  - "crates/mlrs-core/src/label_perm.rs"
  - "crates/mlrs-core/src/lib.rs"
tech-stack:
  added: []
  patterns:
    - "Greedy confusion-matrix best_mapping over -1-filtered vocabularies + force-pinned (-1, -1)"
key-files:
  created: []
  modified:
    - "crates/mlrs-core/src/label_perm.rs"
    - "crates/mlrs-core/src/lib.rs"
    - "crates/mlrs-core/tests/helpers_test.rs"
decisions:
  - "Tests placed in crates/mlrs-core/tests/helpers_test.rs (existing convention), NOT a src/ mod tests — AGENTS.md §2 strictly prohibits in-src test modules, overriding the plan's 'in-crate test module' wording (CLAUDE.md precedence)."
metrics:
  duration: "1m36s"
  completed: "2026-06-24T04:43:02Z"
  tasks: 1
  files_modified: 3
requirements: [HDBS-02]
---

# Phase 15 Plan 01: -1-Pinned Label-Permutation Matcher Summary

Added `best_match_accuracy_pinned_noise` to `mlrs_core::label_perm`: a label-permutation matcher that treats the HDBSCAN noise sentinel `-1` as a fixed point (maps only to `-1`, and only from `-1`), while non-noise cluster ids match up to permutation via the existing greedy `best_mapping`. This is the exact-label compare primitive for the HDBSCAN HDBS-02 gate, where a sklearn-noise / mlrs-cluster confusion must count as a genuine failure rather than be permuted away.

## What Was Built

- **`best_match_accuracy_pinned_noise(pred: &[i64], reference: &[i64]) -> f64`** in `crates/mlrs-core/src/label_perm.rs`. It filters `-1` out of both label vocabularies before building the greedy mapping (so noise never enters the confusion matrix or the assignment), then force-inserts `(-1, -1)`. Each `pred[i]` is remapped through the resulting map (a `-1` stays `-1`); accuracy is the fraction of indices where the remapped value equals `reference[i]`. Empty input returns `1.0` (vacuous match), matching the `best_match_accuracy` convention. Reuses the existing `best_mapping` / `confusion` / `sorted_unique` helpers verbatim.
- **Re-export** `mlrs_core::best_match_accuracy_pinned_noise` added to the existing `pub use label_perm::{...}` line in `crates/mlrs-core/src/lib.rs`.
- **Five unit tests** in `crates/mlrs-core/tests/helpers_test.rs` covering all five plan behaviors: cluster permutation with `-1` fixed (1.0), noise/cluster confusion as mismatch (0.8 < 1.0), all-noise == all-noise (1.0), pure permutation with no noise (1.0), and empty vacuous match (1.0).

## How It Was Verified

- `cargo test -p mlrs-core --test helpers_test pinned_noise` → 5 passed.
- `cargo test -p mlrs-core --test helpers_test` → 16 passed, 0 failed (no regression on the pre-existing sign-flip / label-perm / oracle tests).
- `cargo clippy -p mlrs-core --tests` → clean.
- `cargo fmt -p mlrs-core -- --check` → clean.
- Source assertion confirmed (acceptance criterion): `-1` is filtered from both vocabularies (`label_perm.rs:150-151`) before `best_mapping` and force-inserted as `(-1, -1)` (`label_perm.rs:155`); no code path maps `-1` to a non-`-1` id.

## TDD Gate Compliance

Plan-level `tdd="true"` task; RED → GREEN cycle satisfied:
- **RED** (`18d4ed7`, `test(15-01)`): failing tests added; `cargo test` failed to compile with `unresolved import best_match_accuracy_pinned_noise` (the function did not yet exist) — a genuine RED, not a spuriously-passing test.
- **GREEN** (`c41a150`, `feat(15-01)`): implementation added; all 5 tests pass.
- **REFACTOR**: skipped — implementation is minimal (~30 lines reusing existing helpers), clippy-clean, no duplication to extract.

## Deviations from Plan

### [Rule 3 / CLAUDE.md precedence] Test placement in tests/ directory, not a src/ module

- **Found during:** Task 1 (RED phase).
- **Issue:** The plan's `<action>` instructed adding the test to "the existing in-crate test module of `label_perm.rs`" and to "follow the existing `label_perm.rs` test placement convention already in that file." No `#[cfg(test)] mod tests` exists in `label_perm.rs`, and **AGENTS.md §2 strictly prohibits** unit tests inside production source files (`embedding mod tests at the bottom of a source file is strictly prohibited`). The genuine project convention is the separate `crates/mlrs-core/tests/helpers_test.rs`, whose own header states "Per AGENTS.md these live in `tests/`, never as a `#[cfg(test)] mod tests` inside `src/`."
- **Fix:** Placed the five tests in `crates/mlrs-core/tests/helpers_test.rs` alongside the existing `best_match_accuracy` tests. CLAUDE.md states AGENTS.md test-separation is a hard project constraint that takes precedence over plan instructions.
- **Files modified:** `crates/mlrs-core/tests/helpers_test.rs`.
- **Commits:** `18d4ed7` (test), `c41a150` (import fmt).

### [Incidental] rustfmt normalized adjacent pre-existing code

- **Found during:** Task 1 (GREEN phase), running `cargo fmt -p mlrs-core`.
- **Issue:** rustfmt reformatted the over-width assert in my new test and, incidentally, a few pre-existing lines it touched in `confusion()` (the `pred_idx`/`ref_idx` collect chains) and one sign-flip test assert.
- **Fix:** Accepted the rustfmt output — these are formatter-mandated normalizations, no behavior change. Verified by re-running the full suite (16 passed) after fmt.
- **Files modified:** `crates/mlrs-core/src/label_perm.rs`, `crates/mlrs-core/tests/helpers_test.rs`.
- **Commit:** `c41a150`.

## Known Stubs

None — the function is fully implemented and exercised by passing tests.

## Self-Check: PASSED

- FOUND: `crates/mlrs-core/src/label_perm.rs` contains `fn best_match_accuracy_pinned_noise`.
- FOUND: `crates/mlrs-core/src/lib.rs` re-exports `best_match_accuracy_pinned_noise`.
- FOUND: commit `18d4ed7` (RED), `c41a150` (GREEN) in git history.
- All 5 new tests pass; full `mlrs-core` helpers suite green.
