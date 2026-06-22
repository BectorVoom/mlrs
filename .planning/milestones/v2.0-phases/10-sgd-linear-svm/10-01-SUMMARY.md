---
phase: 10-sgd-linear-svm
plan: 01
subsystem: api
tags: [sgd, linear-svm, builder-pattern, cubecl, pyo3, oracle-fixtures, thiserror]

requires:
  - phase: 05-iterative-linear
    provides: coordinate-descent solver (cd_solve), logistic classes_ remap, elastic_net struct/predict precedent
  - phase: 09-spectral-family
    provides: Wave-0 scaffold pattern (front-load shared-file edits + #[ignore] Nyquist scaffolds + isolated oracle regen)
provides:
  - Typed Loss/Penalty/LearningRate enums + single-source TryFrom<&str> (D-04/D-05)
  - SgdConfig shared lowering target (D-06)
  - BuildError enum + build_err_to_py PyValueError mapper (D-08/D-09)
  - Four builder-fronted estimator homes (MBSGDClassifier/Regressor, LinearSVC/SVR) with sklearn defaults (D-01/D-03)
  - sgd_margin + sgd_weight_update SharedMemory-free #[cube(launch)] kernels (PRIM-10)
  - sgd_solve host signature (geometry guard real, todo!() compute body)
  - Four any_estimator! Unfit dispatch enums (sklearn strings + scalars)
  - 12 committed pinned-deterministic .npz oracle fixtures
  - Six #[ignore] Nyquist test scaffolds (fixture-load + shape)
affects: [10-02-sgd-prim-compute, 10-03-mbsgd-estimators, 10-04-linear-svm, 10-05-pyo3-wrap]

tech-stack:
  added: []
  patterns:
    - "Builder pattern + build() -> Result<Estimator<F>, BuildError> (D-01, Phase-10 INTRODUCES it via the four new estimators only)"
    - "Split validation (D-08): data-INDEPENDENT params validated at build() (BuildError); data-DEPENDENT at fit() (AlgoError)"
    - "Single PyO3 mapper build_err_to_py covers builder + enum-parse failures (D-09)"

key-files:
  created:
    - crates/mlrs-algos/src/linear/sgd_config.rs
    - crates/mlrs-algos/src/linear/mbsgd_classifier.rs
    - crates/mlrs-algos/src/linear/mbsgd_regressor.rs
    - crates/mlrs-algos/src/linear/linear_svc.rs
    - crates/mlrs-algos/src/linear/linear_svr.rs
    - crates/mlrs-kernels/src/sgd.rs
    - crates/mlrs-backend/src/prims/sgd.rs
    - crates/mlrs-backend/tests/sgd_test.rs
    - crates/mlrs-algos/tests/sgd_config_test.rs
    - crates/mlrs-algos/tests/mbsgd_classifier_test.rs
    - crates/mlrs-algos/tests/mbsgd_regressor_test.rs
    - crates/mlrs-algos/tests/linear_svc_test.rs
    - crates/mlrs-algos/tests/linear_svr_test.rs
    - crates/mlrs-py/tests/sgd_smoke_test.rs
  modified:
    - crates/mlrs-algos/src/error.rs
    - crates/mlrs-algos/src/linear/mod.rs
    - crates/mlrs-kernels/src/lib.rs
    - crates/mlrs-backend/src/prims/mod.rs
    - crates/mlrs-py/src/errors.rs
    - crates/mlrs-py/src/estimators/linear.rs
    - crates/mlrs-backend/tests/memory_gate_test.rs
    - scripts/gen_oracle.py

key-decisions:
  - "sgd_solve takes flat SgdParams (prim-local enums), NOT the algos SgdConfig — mlrs-backend does not depend on mlrs-algos (Rule 3 deviation from the plan's literal config: &SgdConfig signature; circular dependency)"
  - "Oracle fixtures regenerated in the /tmp venv (numpy 2.4.6/scipy 1.18.0/sklearn 1.9.0) in isolation; ConvergenceWarning is EXPECTED (tol=0 + fixed max_iter runs all epochs deterministically)"
  - "MBSGDClassifier hinge fixture emits constant + optimal schedule variants (A1/Pitfall 3 t0 isolation) + a log-loss variant for predict_proba"

patterns-established:
  - "Wave-0 builder scaffold: build() SIGNATURE final now, validation predicates land Wave-1; default field initializers encode sklearn defaults"
  - "Two-pass GATHER SGD kernels (sgd_margin per-sample, sgd_weight_update per-coordinate) — single-owner, SharedMemory/INFINITY-free (cpu-MLIR safe)"

requirements-completed: [PRIM-10, SGDSVM-01, SGDSVM-02, SGDSVM-03, SGDSVM-04]

duration: 50min
completed: 2026-06-21
---

# Phase 10 Plan 01: SGD / Linear-SVM Wave-0 Scaffold Summary

**Front-loaded the entire Phase-10 construction surface — typed enums + four builder-fronted estimators, BuildError + build_err_to_py, the SharedMemory-free SGD kernels + geometry-guarded sgd_solve stub, four Unfit PyO3 dispatch enums, and 12 pinned-deterministic sklearn oracle fixtures — into one compiling, file-disjoint wave so Waves 1/2/3 are parallel-safe.**

## What Was Built

**Task 1 (commit 14f2a7c):** `sgd_config.rs` with the three typed enums (`Loss`/`Penalty`/`LearningRate`), each with `name()` + a single-source `TryFrom<&str>` accepting sklearn spellings and the legacy aliases (`log`/`log_loss`, `squared_error`/`squared_loss`), the 14-field `SgdConfig` lowering target, plus a sibling `BuildError` enum (8 variants) in `error.rs`. Four estimator modules each carry their struct + `*Builder` whose default field initializers encode the sklearn defaults (classifier hinge/optimal; regressor squared_error/invscaling/power_t=0.25; svc squared_hinge/C=1.0; svr squared_epsilon_insensitive). The `build() -> Result<Estimator<F>, BuildError>` signature is final; validation predicates are Wave-1. Registered the five modules in `linear/mod.rs`. The D-04 `affinity: String` anti-pattern grep returns 0.

**Task 2 (commit ddb581f):** `sgd_margin` (pass-1 per-sample margin) + `sgd_weight_update` (pass-2 per-coordinate GATHER) `#[cube(launch)]` kernels, SharedMemory/INFINITY-free by construction (the shipped `coordinate.rs` GATHER idiom). `sgd_solve` host signature with a REAL geometry guard (rejects `x.len() != n*d`, `y.len() != n`, empty) before a `todo!()` compute body, plus `SGD_DEFAULT_MAX_ITER`/`SGD_DEFAULT_TOL`. `build_err_to_py(BuildError) -> PyValueError` (D-09) and four `any_estimator!` Unfit dispatch enums storing sklearn strings + scalars (the four `#[pyclass]` registrations are Wave-3).

**Task 3 (commit 98b88fe):** Four oracle generators added to `gen_oracle.py`, run in the isolated /tmp venv, emitting 12 committed pinned-deterministic `.npz` fixtures. Six `#[ignore]` Nyquist test scaffolds (assert fixture-load + shape only) plus the live `sgd_config_test` (3 passing TryFrom/default tests + 1 ignored Wave-1 validation test). Updated the 10-VALIDATION.md Per-Task Verification Map.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] sgd_solve takes flat SgdParams, not algos SgdConfig**
- **Found during:** Task 2
- **Issue:** The plan's literal signature `sgd_solve<F>(..., config: &SgdConfig)` is impossible — `SgdConfig` lives in `mlrs-algos`, and `mlrs-backend` (where the prim lives) does NOT depend on `mlrs-algos` (the dependency runs the other way). Importing `SgdConfig` would create a circular crate dependency.
- **Fix:** Followed the shipped `cd_solve` precedent (flat scalar args). Added prim-local `SgdLoss`/`SgdSchedule` enums + a flat `SgdParams` struct; the algos estimator lowers its `SgdConfig` into these at the call site (the Wave-1 plan's job). Geometry guard + signature are final; compute is `todo!()`.
- **Files modified:** crates/mlrs-backend/src/prims/sgd.rs
- **Commit:** ddb581f

### Notes (not deviations)

- **ConvergenceWarning during oracle gen is EXPECTED, not an error:** the fixtures pin `tol=0` + fixed `max_iter` so sklearn runs all epochs without early-stopping (Pitfall 2/7 determinism). sklearn warns; the fixtures are correct.
- **Builder call ergonomics:** `MBSGDClassifier::<f32>::builder().build::<f32>()` carries F twice (the non-generic builder + the generic `build`). Functionally correct; the Wave-1/3 callers operate in a generic `F` context so the turbofish friction does not appear there.

## Authentication Gates

None.

## Known Stubs

All Wave-0 scaffold stubs are intentional and documented; each is resolved by a named later wave:

| Stub | File | Reason / Resolver |
|------|------|-------------------|
| `build()` returns Ok without validating | mbsgd_classifier/regressor/svc/svr.rs | SIGNATURE final (D-01); validation predicates land Wave-1/2 (10-02/10-04) |
| `sgd_solve` `todo!()` compute body | prims/sgd.rs | Geometry guard real; epoch loop fills Wave-1 (10-02) |
| Four `#[pyclass]` not registered on `_mlrs` | estimators/linear.rs | Only the Unfit dispatch enums land here; Wave-3 (10-05) wires the pyclasses |
| Six `#[ignore]` test scaffolds | tests/*.rs | Assert fixture-load + shape only; Waves 1/2 un-ignore as compute/fit land |

No stub blocks the plan's goal (a compiling, parallel-safe scaffold). The estimator fitted-state fields carry `#[allow(dead_code)]` until their Wave fills the fit body — matching the shipped scaffold convention.

## Verification Evidence

- `cargo build -p mlrs-algos --features cpu` — exit 0
- `cargo build -p mlrs-kernels` (bare, no feature) — exit 0
- `cargo build -p mlrs-backend --features cpu` — exit 0
- `cargo test -p mlrs-algos -p mlrs-backend --features cpu --no-run` — exit 0 (all six scaffolds compile)
- `cargo test -p mlrs-py --features cpu --test sgd_smoke_test --no-run` — exit 0
- `sgd_config_test` live run: 3 passed, 1 ignored (Wave-1 validation)
- Grep gates: `SharedMemory`==0, `INFINITY`==0 (comment-filtered) on sgd.rs; `affinity: String`==0 on sgd_config.rs
- 12 committed `.npz` fixtures present; no other phase blobs modified (isolated regen)
- Every f64 oracle scaffold references `skip_f64_with_log`

## Self-Check: PASSED

All created files verified present on disk; all three task commits (14f2a7c, ddb581f, 98b88fe) verified in git history.
