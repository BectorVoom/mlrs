---
phase: 10-sgd-linear-svm
plan: 03
subsystem: api
tags: [sgd, mbsgd-classifier, mbsgd-regressor, builder-pattern, oracle-fixtures, sgd-solve, predict-proba]

requires:
  - phase: 10-sgd-linear-svm
    provides: "PRIM-10 validated sgd_solve epoch loop + dloss/schedule helpers + SgdParams/SgdLoss/SgdSchedule flat contract (10-02); SgdConfig + BuildError + the four builder-fronted estimator homes (10-01)"
  - phase: 05-iterative-linear
    provides: "logistic.rs classes_ distinct-sorted ±1 remap precedent + PredictLabels/PredictProba; elastic_net::predict_linear shared X·coef+intercept GEMM-then-broadcast"
provides:
  - "MBSGDClassifier (SGDSVM-01): build() D-08 validation + Fit (±1 remap) + PredictLabels (margin sign) + PredictProba (log-loss sigmoid) on the validated sgd_solve"
  - "MBSGDRegressor (SGDSVM-02): build() D-08 validation + Fit + Predict (reuses predict_linear) on the validated sgd_solve"
  - "lower_config(SgdConfig) -> SgdParams: shared estimator→prim lowering (pub(crate) in mbsgd_classifier.rs, consumed by the regressor)"
  - "Un-ignored sgd_config_test build-validation cases (alpha/eta0/l1_ratio/loss-for-estimator)"
affects: [10-05-pyo3-wrap]

tech-stack:
  added: []
  patterns:
    - "Estimator→prim lowering: build()-validated SgdConfig lowered into the prim-local flat SgdParams at the fit call site (the cd_solve flat-scalar precedent; the prim cannot depend on algos)"
    - "Shared decision-margin host helper feeds both PredictLabels (sign) and PredictProba (sigmoid) from one on-device matvec GEMM"

key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/linear/mbsgd_classifier.rs
    - crates/mlrs-algos/src/linear/mbsgd_regressor.rs
    - crates/mlrs-algos/tests/mbsgd_classifier_test.rs
    - crates/mlrs-algos/tests/mbsgd_regressor_test.rs
    - crates/mlrs-algos/tests/sgd_config_test.rs

key-decisions:
  - "lower_config lives pub(crate) in mbsgd_classifier.rs (consumed by the regressor) rather than a new file — honors the plan's file-disjoint boundary (no linear/mod.rs edit, the 10-04 svm_lbfgs_fit-in-linear_svc.rs precedent)"
  - "Pure-L1 penalty lowers to l1_ratio=1.0 with apply_l1=true; L2-only lowers to l1_ratio=0.0/apply_l1=false (the prim runs the elastic-net shrink math, so L1 is the l1_ratio=1 case)"
  - "predict_proba returns the log-loss sigmoid σ(margin) over a SHARED decision-margin helper (one matvec GEMM feeds both labels-sign and proba-sigmoid); n_query×2 row-major [P(class0),P(class1)]"
  - "coef/intercept oracle bands documented (f64 5e-3, f32 2e-2) — the host-driven minibatch order-of-operations differs from sklearn's Cython _sgd_fast last-bit accumulation; the EXACT predict labels (the hard gate) are the strict correctness witness"

patterns-established:
  - "MBSGD estimator = build() data-independent validation (D-08) → Fit (classes_/target prep + lower_config + sgd_solve) → predict via shared margin/predict_linear; the prim owns the epoch loop, the estimator owns the encoding + validation split"

requirements-completed: [SGDSVM-01, SGDSVM-02]

duration: 22min
completed: 2026-06-21
---

# Phase 10 Plan 03: MBSGDClassifier / MBSGDRegressor Summary

**Wired the two minibatch-SGD estimators onto the validated PRIM-10 `sgd_solve` — `MBSGDClassifier` (±1 remap → margin-sign labels + log-loss-sigmoid proba) and `MBSGDRegressor` (raw target → `predict_linear` reuse), each gated by the pinned-deterministic sklearn oracle: the classifier reproduces sklearn's predict labels EXACTLY (the hard gate, f32 + f64) with coef/proba within band, the D-03 default litmus holds for both, and the D-08 split validation (data-independent at `build()`, data-dependent at `fit()`) is proven.**

## Performance

- **Duration:** ~22 min
- **Started:** 2026-06-21T08:35:00Z
- **Completed:** 2026-06-21T08:57:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- `MBSGDClassifier::build()` fills the D-08 data-independent validation (alpha>=0, l1_ratio∈[0,1] for ElasticNet, eta0>0 unless Optimal, loss-for-classifier {Hinge,Log,SquaredHinge}); `Fit` does the distinct-sorted `classes_` + binary ±1 remap (Pitfall 4), lowers `SgdConfig` into `SgdParams` (`lower_config`), and delegates to the validated `sgd_solve`. `PredictLabels` is the decision-margin sign → `classes_` roundtrip (i32); `PredictProba` is the log-loss sigmoid `1/(1+exp(-margin))` over a shared decision-margin helper.
- `MBSGDRegressor::build()` fills its D-08 validation (adds the `epsilon>=0` guard + loss-for-regressor {SquaredLoss,EpsilonInsensitive,SquaredEpsilonInsensitive}); `Fit` lowers via the SAME `lower_config` and delegates to `sgd_solve`; `Predict` reuses `elastic_net::predict_linear` (no duplicated GEMM path).
- HARD classifier gate green: `exact_labels` (f32 + f64) match sklearn's predicted labels EXACTLY (integers, no band) on the pinned constant-schedule hinge fixture.
- Oracle bands green: classifier coef/intercept + log-loss `predict_proba`, regressor coef/intercept/predict (squared_error + invscaling) + the epsilon-insensitive tube path.
- D-03 default litmus green for both estimators (bare `builder().build()` equals sklearn's `SGDClassifier`/`SGDRegressor` defaults).
- Un-ignored the `sgd_config_test` build-validation cases (alpha/eta0/l1_ratio/loss-for-estimator) now that `build()` validates.

## Task Commits

Each task was committed atomically:

1. **Task 1: MBSGDClassifier — build() validation + Fit + PredictLabels + PredictProba + oracle** - `d9ec94f` (feat)
2. **Task 2: MBSGDRegressor — build() validation + Fit + Predict + oracle** - `ca119a4` (feat)

_Note: the RED test scaffolds (the `#[ignore]` Nyquist fixtures) were committed at Wave-0 (10-01); the prim they consume was validated standalone at Wave-1 (10-02). This Wave-2 plan lands the GREEN gate — the estimator impl + the un-ignored value-comparison tests together — for each task._

## Files Created/Modified

- `crates/mlrs-algos/src/linear/mbsgd_classifier.rs` — `build()` D-08 validation body; `Fit`/`PredictLabels`/`PredictProba` impls; shared `decision_margin` helper; `pub(crate) lower_config`; host_to_f64/f64_to_host trio
- `crates/mlrs-algos/src/linear/mbsgd_regressor.rs` — `build()` D-08 validation body; `Fit` (delegates to `sgd_solve`); `Predict` (reuses `predict_linear`); coef/intercept accessors
- `crates/mlrs-algos/tests/mbsgd_classifier_test.rs` — `exact_labels`(f32+f64 HARD gate), `oracle`(f32+f64 band), `proba`(f32+f64 log-loss sigmoid), `default_matches_sklearn`, `build_rejects_bad_alpha`
- `crates/mlrs-algos/tests/mbsgd_regressor_test.rs` — `oracle`(f32+f64), `oracle_epsilon_f32`(tube path), `default_matches_sklearn`, `build_rejects_bad_hyperparams`
- `crates/mlrs-algos/tests/sgd_config_test.rs` — un-ignored `build_rejects_bad_alpha`; added `build_rejects_bad_hyperparams` (eta0/l1_ratio/loss-for-estimator)

## Decisions Made

- **`lower_config` lives `pub(crate)` in `mbsgd_classifier.rs`** (consumed by the regressor) rather than a new file — keeps the plan's file-disjoint boundary intact (no `linear/mod.rs` edit), mirroring the 10-04 `svm_lbfgs_fit`-in-`linear_svc.rs` precedent.
- **Penalty lowering:** pure-L1 → `l1_ratio=1.0` + `apply_l1=true`; L2-only → `l1_ratio=0.0` + `apply_l1=false`; ElasticNet passes its `l1_ratio` through. The prim runs the elastic-net lazy-L2 / cumulative-L1 shrink, so L1 is the `l1_ratio=1` case.
- **`predict_proba` shares the decision-margin helper** with `predict_labels` (one on-device matvec GEMM feeds both the labels sign and the proba sigmoid), returning `n_query×2` `[P(class0), P(class1)]` with `P(class1) = σ(margin)`.
- **Oracle coef/intercept bands** (classifier/regressor f64 5e-3, f32 2e-2; proba f64 1e-2 / f32 3e-2) — the host-driven minibatch order-of-operations differs from sklearn's Cython `_sgd_fast` last-bit accumulation, so the converged iterate agrees to a documented band; the EXACT predict labels (the classifier hard gate) are the strict correctness witness.

## Deviations from Plan

None — plan executed exactly as written. Both estimators wired onto `sgd_solve` as specified; `build()` validation split (D-08), the classifier ±1 remap (Pitfall 4), `predict_proba` sigmoid (D-05 log-loss), the regressor `predict_linear` reuse, the D-03 default litmus, and the un-ignored `sgd_config` validation cases all landed per the plan's `<action>` blocks.

## Issues Encountered

- One trivial compile fix during Task 1 (a moved `String` borrowed in an `assert!` format arg — added `ref value` to the `matches!` pattern); test-local, fixed before the commit. Not a plan deviation.

## Threat Flags

None — no new network/auth/filesystem surface. The estimator funnels untrusted hyperparameters into typed `BuildError` at `build()` (T-10-03-01) and untrusted shapes/labels into typed `AlgoError` at `fit()` BEFORE the device launch (T-10-03-02); a device-launch failure across the estimator boundary is wrapped into `AlgoError::Prim` via `?` (T-10-03-03, never an unwinding panic). The proba/label math is pure host arithmetic over validated device results.

## Known Stubs

None introduced. Both estimators are fully wired (fit + predict + proba where applicable); no hardcoded empty values or placeholder text.

## Self-Check: PASSED

All five modified source/test files verified present on disk; both task commits (`d9ec94f`, `ca119a4`) verified in git history. Full plan suite green: `mbsgd_classifier_test` 8/8, `mbsgd_regressor_test` 5/5, `sgd_config_test` 5/5 (cpu f32+f64).

## Next Phase Readiness

- SGDSVM-01 (MBSGDClassifier) + SGDSVM-02 (MBSGDRegressor) are met; with 10-04's LinearSVC/LinearSVR (SGDSVM-03/04) already complete, all four Phase-10 estimators now have validated algos `fit`/`predict` surfaces.
- Wave-3 (10-05 PyO3 wrap) can now register the four `#[pyclass]` estimators on `_mlrs`, lowering the Unfit dispatch enums (10-01) into the now-complete `build()`/`fit` paths.

---
*Phase: 10-sgd-linear-svm*
*Completed: 2026-06-21*
