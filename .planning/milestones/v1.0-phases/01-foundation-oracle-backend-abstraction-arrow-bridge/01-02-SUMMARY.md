---
phase: 01-foundation-oracle-backend-abstraction-arrow-bridge
plan: 02
subsystem: testing
tags: [oracle, tolerance, assert_close, npyz, npz, thiserror, sign-flip, label-permutation, sklearn-parity]

# Dependency graph
requires:
  - phase: 01-foundation-oracle-backend-abstraction-arrow-bridge (Plan 01)
    provides: 5-crate workspace, SPIKE-FINDINGS.md (npyz by_name + writer API, A4)
provides:
  - "mlrs_core::compare::{is_close, assert_close, assert_slice_close} — abs AND rel + near-zero guard (D-09)"
  - "mlrs_core::tolerance::{Tolerance, F32_TOL, F64_TOL, Tolerance::for_family} — growable policy (D-08)"
  - "mlrs_core::error::BridgeError — thiserror enum consumed by Plan 03's Arrow bridge (D-07/FOUND-06)"
  - "mlrs_core::sign_flip — svd_flip-style sign canonicalization (FOUND-08)"
  - "mlrs_core::label_perm — best-permutation cluster-label matching (FOUND-08)"
  - "mlrs_core::oracle::{load_npz, OracleCase} — named-.npz fixture loader for f32/f64 (D-01/02/03)"
  - "docs/tolerance-policy.md — documented per-family f32 tolerance policy"
affects: [01-03-bridge, 01-05-oracle-integration, phase-03, phase-04, phase-05]

# Tech tracking
tech-stack:
  added: []  # all deps already declared in Plan 01 workspace; this plan only consumes npyz + thiserror
  patterns:
    - "abs AND rel comparison with near-zero abs-only fallback (D-09)"
    - "growth-point function (Tolerance::for_family) instead of a populated table (D-08)"
    - "dtype-byte-width branch when decoding npyz arrays (strict typed reader)"
    - "throwaway .npz fixture generated via npyz writer (no numpy at test time)"

key-files:
  created:
    - crates/mlrs-core/tests/compare_test.rs
    - crates/mlrs-core/tests/helpers_test.rs
    - crates/mlrs-core/tests/fixtures/oracle_case.npz
    - crates/mlrs-core/examples/gen_fixture.rs
    - docs/tolerance-policy.md
  modified:
    - crates/mlrs-core/src/tolerance.rs
    - crates/mlrs-core/src/compare.rs
    - crates/mlrs-core/src/error.rs
    - crates/mlrs-core/src/sign_flip.rs
    - crates/mlrs-core/src/label_perm.rs
    - crates/mlrs-core/src/oracle.rs
    - crates/mlrs-core/src/lib.rs

key-decisions:
  - "NEAR_ZERO_FLOOR = 1e-8 (three orders below the 1e-5 abs tolerance, so the guard never loosens the abs check)"
  - "Tolerance::for_family() is the D-08 growth point; returns global default for all families in Phase 1"
  - "BridgeError lives in mlrs-core (not mlrs-backend) so Plan 03 can consume it without a reverse dependency"
  - "oracle::load_npz branches on stored dtype byte-width (npyz typed reader is strict) and exposes both f32 and f64 views per array"
  - "sign_flip uses sklearn svd_flip convention: make the largest-magnitude element non-negative"
  - "label_perm uses greedy confusion-matrix assignment (exact for small label sets; Hungarian can replace the core later)"

patterns-established:
  - "All tests in crates/mlrs-core/tests/*.rs — never `mod tests` in src (AGENTS.md)"
  - "Mixed-dtype .npz fixtures committed as binary; generated via npyz writer, no Python"

requirements-completed: [FOUND-06, FOUND-07, FOUND-08]

# Metrics
duration: ~25min
completed: 2026-06-11
---

# Phase 01 Plan 02: Oracle Comparison Harness Summary

**Host-side sklearn-oracle spine for mlrs-core: `assert_close` (abs AND rel + 1e-8 near-zero guard), a growable tolerance policy, sklearn-style sign-flip and greedy label-permutation helpers, a strict-dtype npyz `.npz` loader, and the typed `BridgeError` enum Plan 03's Arrow bridge consumes.**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-06-11
- **Tasks:** 2 (Task 0 legitimacy gate was pre-approved separately)
- **Files modified/created:** 12 files (974 insertions)
- **Tests:** 25 passing (14 compare + 11 helpers), 0 failing

## Accomplishments
- `is_close`/`assert_close`/`assert_slice_close` enforce **both** absolute AND relative error with a documented `NEAR_ZERO_FLOOR = 1e-8` abs-only fallback (D-09); informative panic reports got/expected/abs/rel.
- `Tolerance` + `F32_TOL`/`F64_TOL` (both `{abs:1e-5, rel:1e-5}`) as a growable policy with a `for_family()` extension point that returns the global default today (D-08).
- `BridgeError` thiserror enum (`Offset`, `HasNulls`, `Misaligned`, `DataTypeMismatch`) exported from `mlrs-core` ready for Plan 03 (D-07/FOUND-06).
- `sign_flip` canonicalizes SVD/PCA component signs (largest-magnitude element made non-negative); sign-flipped-but-equal vectors pass `assert_close`, genuinely different ones still fail.
- `label_perm` maps cluster labels via greedy confusion-matrix best-permutation matching; `[1,1,0,0]` vs `[0,0,1,1]` matches, `[0,1,0,1]` vs `[0,0,1,1]` does not.
- `oracle::load_npz` reads named arrays by name (npyz `by_name`) for both f32 and f64 from a committed throwaway `.npz` fixture — no Python at test time.
- `docs/tolerance-policy.md` documents the global default, near-zero floor rationale, and per-family growth path (FOUND-08).

## Task Commits

Each task was committed atomically:

1. **Task 1: Tolerance policy + assert_close + BridgeError** - `f3ae9cf` (feat)
2. **Task 2: Sign-flip + label-permutation + named-.npz loader** - `988edd6` (feat)

**Plan metadata:** see final tracking commit (docs: complete plan)

_Task 0 (package-legitimacy blocking-human gate) was approved before this executor ran: npyz 0.9.1 as the npz reader, log 0.4 / env_logger 0.11 as the logger._

## Files Created/Modified
- `crates/mlrs-core/src/tolerance.rs` - Tolerance struct, F32_TOL/F64_TOL, for_family() growth point
- `crates/mlrs-core/src/compare.rs` - is_close/assert_close/assert_slice_close + NEAR_ZERO_FLOOR
- `crates/mlrs-core/src/error.rs` - BridgeError thiserror enum (4 Arrow-violation variants)
- `crates/mlrs-core/src/sign_flip.rs` - canonical_sign/align_sign/align_rows (svd_flip convention)
- `crates/mlrs-core/src/label_perm.rs` - best_mapping/remap/best_match_accuracy/is_perfect_match
- `crates/mlrs-core/src/oracle.rs` - load_npz/load_npz_reader/OracleCase (npyz by_name, dtype-branched)
- `crates/mlrs-core/src/lib.rs` - re-exports of the most-used symbols
- `crates/mlrs-core/tests/compare_test.rs` - 14 tests (D-08/D-09 + BridgeError)
- `crates/mlrs-core/tests/helpers_test.rs` - 11 tests (sign-flip, label-perm, npz load)
- `crates/mlrs-core/tests/fixtures/oracle_case.npz` - committed mixed-dtype fixture (a/x/y/expected)
- `crates/mlrs-core/examples/gen_fixture.rs` - throwaway npyz-writer fixture generator
- `docs/tolerance-policy.md` - documented tolerance policy (FOUND-08)

## Decisions Made
- **NEAR_ZERO_FLOOR = 1e-8** — chosen three orders of magnitude below `tol.abs = 1e-5` so the guard only suppresses spurious near-zero relative-error failures and can never loosen the absolute check (covered by an explicit test per threat T-02-02).
- **`BridgeError` in `mlrs-core`** — placed in the dependency-free foundation crate so Plan 03's bridge in `mlrs-backend` consumes it without a reverse dependency.
- **dtype-byte-width branch in `load_npz`** — npyz's typed reader is strict (`into_vec::<f64>()` only on 8-byte floats, `into_vec::<f32>()` only on 4-byte), so the loader inspects `DType::num_bytes()`, decodes at native precision, and derives the other view.
- **Greedy (not Hungarian) label permutation** — exact for the small label cardinalities of oracle fixtures; the API isolates the core so a Hungarian solver can drop in later.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `--features cpu` is invalid for `mlrs-core`**
- **Found during:** Task 1 (running the plan's `<verify>` command)
- **Issue:** The plan's verify command `cargo test -p mlrs-core --features cpu compare` fails — `mlrs-core` is intentionally backend-feature-free (Plan 01 / Criterion 1), so it has no `cpu` feature.
- **Fix:** Ran the equivalent without the feature flag: `cargo test -p mlrs-core --test compare_test` / `--test helpers_test`. Behavior and coverage are identical; only the (inapplicable) feature flag was dropped.
- **Files modified:** none (test-invocation only)
- **Verification:** 25/25 tests pass; full `cargo test -p mlrs-core` green.
- **Committed in:** n/a (no file change)

**2. [Rule 1 - Lint] clippy `assertions_on_constants` in compare_test.rs**
- **Found during:** Task 2 (clippy pass)
- **Issue:** `assert!(NEAR_ZERO_FLOOR < F32_TOL.abs)` is a constant-valued assertion; clippy warns.
- **Fix:** Converted to a `const { assert!(..) }` block (compile-time check, no warning).
- **Files modified:** crates/mlrs-core/tests/compare_test.rs
- **Verification:** `cargo clippy -p mlrs-core --all-targets` clean.
- **Committed in:** `988edd6` (Task 2 commit)

---

**Total deviations:** 2 (1 blocking test-invocation adjustment, 1 lint fix)
**Impact on plan:** No scope creep. The verify-command flag mismatch reflects the deliberate feature-free design of `mlrs-core`; results are equivalent.

## Issues Encountered
- **npyz strict typed reader:** confirmed via crate source that `into_vec::<T>()` requires an exact dtype byte-width match, so the loader cannot blindly decode every array as f64. Resolved by branching on `DType::num_bytes()` and keeping both precision views.
- **No `NpzWriter::finish()`:** the npz central directory is finalized on drop (matching the Plan 01 spike); the generator drops the writer explicitly.

## Threat Flags
None — no new security surface beyond the plan's threat model. The committed `.npz` is first-party (T-02-01 accept); the near-zero guard is documented + tested (T-02-02 mitigate); package legitimacy was gated in Task 0 (T-02-SC mitigate).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `BridgeError` is exported and ready for Plan 03's Arrow→CubeCL bridge.
- `assert_close` / sign-flip / label-perm / `load_npz` form the verification spine every downstream oracle test consumes (Plan 05, Phases 3/4/5).
- Tolerance policy is a growable structure; per-family rows are added in later phases only when a family needs them.

## Self-Check: PASSED

All 10 created/modified key files exist on disk; both task commits (`f3ae9cf`, `988edd6`) are in git history; `grep -rn "mod tests" crates/mlrs-core/src/` is empty; `cargo test -p mlrs-core` is 25/25 green; `cargo clippy -p mlrs-core --all-targets` clean.

---
*Phase: 01-foundation-oracle-backend-abstraction-arrow-bridge*
*Completed: 2026-06-11*
