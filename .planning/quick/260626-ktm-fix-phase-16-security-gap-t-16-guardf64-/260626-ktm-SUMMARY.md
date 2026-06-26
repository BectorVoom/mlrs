---
phase: quick-260626-ktm
plan: 01
subsystem: mlrs-py (Python binding layer)
status: complete
tags: [security, poison-recovery, T-16-GUARDF64, covariance, lock_pool]
requires:
  - crate::lock_pool() poison-recovering pool guard (WR-02/WR-04, pre-existing in lib.rs)
provides:
  - covariance.rs PyEmpiricalCovariance/PyLedoitWolf bindings locking exclusively via lock_pool()
  - lib.rs lock_pool doc comment reflecting zero legacy panicking holders
affects:
  - crates/mlrs-py/src/estimators/covariance.rs
  - crates/mlrs-py/src/lib.rs
tech-stack:
  added: []
  patterns:
    - "All estimator pool locks route through crate::lock_pool() (no global_pool().lock().expect)"
key-files:
  created: []
  modified:
    - crates/mlrs-py/src/estimators/covariance.rs
    - crates/mlrs-py/src/lib.rs
decisions:
  - "Used a single mechanical global string replacement (sed) of the exact legacy expression — every live occurrence in covariance.rs was one of the 12 target sites; behavior-preserving."
metrics:
  duration: ~2m
  completed: 2026-06-26
  tasks: 2
  files: 2
---

# Phase quick-260626-ktm Plan 01: Close T-16-GUARDF64 lock_pool half (covariance.rs migration) Summary

Migrated the sole remaining estimator module on the legacy panicking pool lock — swapped all 12 `crate::global_pool().lock().expect("pool mutex")` sites in `covariance.rs` to the poison-recovering `crate::lock_pool()`, and refreshed the now-stale `lock_pool` doc comment in `lib.rs` so it accurately states zero legacy holders remain.

## What Was Built

- **Task 1 — covariance.rs lock-path swap (commit `d3bcd72`):** Replaced all 12 legacy `crate::global_pool().lock().expect("pool mutex")` expressions with `crate::lock_pool()`. The two `let mut pool = ...` fit-body sites (lines 85, 222) and the ten `let pool = ...` accessor sites are otherwise unchanged; `py.detach` closures, indentation, and control flow untouched. This was a behavior-preserving mechanical swap — both expressions return `MutexGuard<'static, BufferPool<ActiveRuntime>>`.
- **Task 2 — lib.rs doc-comment refresh (commit `e391eed`):** Updated the "## This is the SANCTIONED lock path (WR-04)" block. The stale parenthetical listing `linear`/`cluster`/`decomposition`/`covariance`/`neighbors`/`projection` wrappers as still carrying the legacy panicking form was removed; the doc now states every estimator wrapper uses `lock_pool` exclusively, covariance was the last legacy holder migrated under T-16-GUARDF64, and new estimators MUST continue to use `lock_pool`. The WR-02 rationale (which keeps the `global_pool().lock().expect("pool mutex")` reference explaining WHY a surviving legacy site re-bricks the interpreter) and the WR-06 accounting re-baseline sections are unchanged.

## Why

A single surviving legacy `.lock().expect("pool mutex")` re-panics on a poisoned mutex and re-bricks the Python interpreter, defeating the WR-02 poison recovery for the whole binding layer once the pool is poisoned (one recoverable device fault/OOM → permanent process-wide brick, a DoS-class regression). `covariance.rs` was the last holdout; converting it makes brick-prevention complete across the binding layer.

## Verification

All plan gates pass:

- `grep -rnE 'global_pool\(\)\.lock\(\)\.expect' crates/mlrs-py/src/estimators/` → ONLY `naive_bayes.rs:19` (the intentional doc-comment counter-example).
- `grep -c 'crate::lock_pool()' crates/mlrs-py/src/estimators/covariance.rs` → `12`.
- lib.rs: `neighbors`/`projection` stale list → 0 matches; "still carry the legacy panicking form" → 0 matches.
- `cargo build -p mlrs-py --features cpu` → clean (finished in 30.74s; 2 pre-existing warnings from the `any_estimator_typestate` macro, unrelated to this change — no errors).

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None — this change closes existing threat T-16-GUARDF64 and introduces no new security surface.

## Self-Check: PASSED

- FOUND: crates/mlrs-py/src/estimators/covariance.rs (12 lock_pool() sites, 0 legacy)
- FOUND: crates/mlrs-py/src/lib.rs (refreshed doc comment, 0 stale module-list matches)
- FOUND commit: d3bcd72 (Task 1)
- FOUND commit: e391eed (Task 2)
