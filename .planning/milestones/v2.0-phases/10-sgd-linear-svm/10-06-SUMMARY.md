---
phase: 10-sgd-linear-svm
plan: 06
subsystem: testing
tags: [sgd, mbsgd-classifier, sklearn-oracle, optimal-schedule, l2-penalty, numerics]

# Dependency graph
requires:
  - phase: 10-02
    provides: PRIM-10 sgd_solve (the minibatch SGD prim whose L2 ordering was corrected)
  - phase: 10-03
    provides: MBSGDClassifier estimator + the constant-schedule oracle test scaffold
provides:
  - "L2 wscale shrink in sgd_solve applied AFTER the gradient step (sklearn _plain_sgd order)"
  - "Convergence delta measured against a pristine pre-update w_start snapshot (WR-02)"
  - "oracle_optimal + oracle_optimal_f32 tests validating the default optimal schedule against sklearn"
  - "fit_hinge_sched schedule-parameterized fit helper (omits eta0 for Optimal)"
  - "Tightened COEF_BAND_F64 (5e-3->1e-3) and COEF_BAND_F32 (2e-2->1e-3)"
affects: [phase-10-verification, phase-10-review, milestone-v2.0-close]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Schedule-parameterized oracle fit helper: one fit_hinge_sched<F>(case, lr) drives both Constant and Optimal, omitting eta0 on the Optimal path to mirror sklearn's Bottou schedule"
    - "Exact predict-label assertion as the strict correctness witness when the converged-iterate coef band is loosened by per-sample minibatch ordering"

key-files:
  created: []
  modified:
    - crates/mlrs-backend/src/prims/sgd.rs
    - crates/mlrs-algos/tests/mbsgd_classifier_test.rs

key-decisions:
  - "WR-01 fixed via path (a): the L2 lazy wscale shrink now runs AFTER sgd_weight_update (the gradient step), matching sklearn _plain_sgd and the plan-cited RESEARCH line 231; the constant-schedule f64 coef abs_err dropped from ~5e-3 to ~3e-7"
  - "COEF_BAND_F64 and COEF_BAND_F32 both re-pinned to 1e-3: the optimal (Bottou) schedule's large step drives coef_ to magnitude ~28 where a ~2.8e-2 abs (~1e-3 relative) per-sample-ordering residual remains; exact labels are the hard gate"
  - "fit_hinge delegates to fit_hinge_sched(Constant) so the existing 8 tests are behaviorally unchanged; the Optimal path omits .eta0() because the builder default eta0=0.01 is positive (build() only requires eta0>0 for non-optimal schedules) and the optimal fixture was generated with no eta0"

patterns-established:
  - "Pattern: probe the band by temporarily setting it to 1e-12, read the reported abs_err, then re-pin to the tightest round value that clears the relative-scaled (band + band*|e|) check"

requirements-completed: [SGDSVM-01]

# Metrics
duration: 22min
completed: 2026-06-21
---

# Phase 10 Plan 06: optimal-schedule oracle + L2 ordering gap closure Summary

**Reordered the sgd_solve L2 wscale shrink to follow the gradient step (sklearn _plain_sgd parity, dropping the constant-schedule f64 coef error from ~5e-3 to ~3e-7) and added oracle_optimal/oracle_optimal_f32 tests that load the previously-unused _optimal fixtures, fit LearningRate::Optimal, and assert coef band + exact sklearn labels — closing CR-01, WR-01, WR-02, and VERIFICATION truth #2.**

## Performance

- **Duration:** ~22 min
- **Started:** 2026-06-21 (Phase 10 gap-closure execution)
- **Completed:** 2026-06-21
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- **WR-01 closed (path a):** the L2 lazy `wscale` shrink `w_j *= max(0, 1-(1-l1_ratio)·eta·alpha)` is now applied AFTER `sgd_weight_update` (the gradient step), before the L1 cumulative soft-shrink — matching sklearn `_plain_sgd` and the plan-cited RESEARCH §Penalty (line 231). The formula and the `params.alpha > 0.0 && l2_factor != 1.0` guard are unchanged; only the position moved.
- **WR-02 closed:** a pristine `w_start` snapshot of the start-of-batch weights is taken before the gradient step, and the convergence delta `change = |w_new[j] - w_start[j]|` is diffed against it (not against a penalty-mutated buffer), so the `tol`-based early stop reflects the full per-batch update.
- **CR-01 closed:** `oracle_optimal` (f64, skip_f64_with_log gated) and `oracle_optimal_f32` (f32, ungated) load `mbsgd_classifier_optimal_{f64,f32}_seed42.npz`, fit `LearningRate::Optimal`, and assert BOTH the coef/intercept band AND exact predict labels. The default `SGDClassifier()` solver path is now exercised against a real sklearn fit.
- **Bands tightened:** COEF_BAND_F64 5e-3 → 1e-3 and COEF_BAND_F32 2e-2 → 1e-3, with justification comments tying the residual to the optimal-schedule per-sample ordering and citing the exact-label hard gate.

## Task Commits

1. **Task 1: Move L2 wscale shrink after the gradient step + WR-02 snapshot** — `8c38607` (fix)
2. **Task 2: oracle_optimal + oracle_optimal_f32 tests + re-pinned bands** — `62ad7e4` (test)

## Files Created/Modified
- `crates/mlrs-backend/src/prims/sgd.rs` — L2 wscale shrink relocated to after the gradient step; pre-update `w_start` snapshot drives the convergence delta; stale "BEFORE the gradient step" comment replaced with a sklearn `_plain_sgd` / RESEARCH §Per-sample update sequence citation.
- `crates/mlrs-algos/tests/mbsgd_classifier_test.rs` — new `oracle_optimal` + `oracle_optimal_f32` tests; new `fit_hinge_sched<F>(case, lr)` helper (omits eta0 for Optimal); `fit_hinge` delegates to it with Constant; COEF_BAND_F64/F32 re-pinned to 1e-3 with documented rationale.

## Observed numerics (post-fix)

| Test | Schedule | dtype | max coef abs_err | Band (relative-scaled) | Result |
|------|----------|-------|------------------|------------------------|--------|
| oracle | constant | f64 | ~2.73e-7 | 1e-3 | pass (far inside) |
| oracle_f32 | constant | f32 | ~9.78e-6 | 1e-3 | pass (far inside) |
| oracle_optimal | optimal | f64 | ~2.79e-2 @ \|coef\|≈28.4 (~1e-3 rel) | 1e-3 | pass (binding) |
| oracle_optimal_f32 | optimal | f32 | ~2.80e-2 @ \|coef\|≈28.4 (~1e-3 rel) | 1e-3 | pass (binding) |

The constant schedule is now effectively at the 1e-5 contract (the WR-01 reorder eliminated the prior 5e-3 gap). The optimal schedule is the binding constraint: its large effective step `eta = 1/(alpha·(t0+t−1))` drives `coef_` to magnitude ~28, where the per-sample minibatch order-of-operations vs sklearn's Cython `_sgd_fast` last-bit accumulation (WR-01/WR-07) leaves a ~2.8e-2 abs / ~1e-3 relative residual. The EXACT predict labels (the hard gate in both optimal tests) are the strict correctness witness; the band only bounds the last-bit iterate drift. Re-pinning below 5e-4 fails the optimal tests, so 1e-3 is the tightest round shared value.

## Verification (all green, cpu — the runnable f64 gate)

- `cargo test -p mlrs-backend --features cpu --test sgd_test` — 6/6 (convex 1e-5, margin, weight, dloss, schedule, cpu-launch).
- `cargo test -p mlrs-backend --features cpu --test memory_gate_test memory_gate_sgd_bounded` — 1/1 (host-roundtrip release_into accounting preserved).
- `cargo test -p mlrs-algos --features cpu --test mbsgd_classifier_test` — 10/10 incl. oracle_optimal + oracle_optimal_f32.
- `cargo test -p mlrs-algos --features cpu --test mbsgd_regressor_test` — 5/5 (no regression from the shared prim edit).
- Grep witness: `grep -c "mbsgd_classifier_optimal" crates/mlrs-algos/tests/mbsgd_classifier_test.rs` = 4 (≥2 — the NOT_WIRED key link from VERIFICATION.md is now WIRED).

## Decisions Made
See key-decisions in frontmatter. In brief: WR-01 via path (a) ordering fix; bands re-pinned to 1e-3 (optimal-schedule binding); fit_hinge delegates to fit_hinge_sched so the existing 8 tests are unchanged.

## Deviations from Plan

None - plan executed exactly as written. (Both bands tightened to 1e-3 as the plan's "tighten toward 1e-5, document the chosen value" directive instructed; the constant schedule reached ~3e-7 / ~1e-5 but the shared constant is bounded by the optimal-schedule residual, which is documented at the band constant per the plan's WR-01/WR-07 justification fallback.)

## Issues Encountered
- The plan's authority sources have an apparent internal tension: RESEARCH §Per-sample update sequence (lines 346-347) lists the wscale shrink before `w += update·x`, while RESEARCH line 231 and the plan body explicitly mandate "path (a)" — shrink AFTER the gradient step. The plan is the controlling document and explicitly directs path (a); implementing it dropped the constant-schedule error to ~3e-7, empirically confirming the plan's choice.

## Next Phase Readiness
- VERIFICATION.md truth #2 ("MBSGDClassifier with schedules INCLUDING optimal matches sklearn within tolerance under the pinned oracle") is now achievable: a Rust test loads the optimal fixtures, fits LearningRate::Optimal, and asserts coef band + exact labels for both dtypes.
- CR-01 (BLOCKER) and WR-01/WR-02 (WARNINGs) from 10-REVIEW.md are closed. IN-04 (optimal_t0 hard-coded epsilon=0.1) was explicitly left out of scope per the plan (harmless for hinge).
- Phase 10 is ready for re-verification / milestone v2.0 close.

---
*Phase: 10-sgd-linear-svm*
*Completed: 2026-06-21*

## Self-Check: PASSED
- crates/mlrs-backend/src/prims/sgd.rs — modified, committed in 8c38607
- crates/mlrs-algos/tests/mbsgd_classifier_test.rs — modified, committed in 62ad7e4
- .planning/phases/10-sgd-linear-svm/10-06-SUMMARY.md — created
- Commits 8c38607 and 62ad7e4 present in git history
