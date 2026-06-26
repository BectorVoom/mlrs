---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 03
subsystem: linear-estimators
tags: [typestate, builder-retrofit, classifiers, svm, logistic, linear-svc, linear-svr, mbsgd-classifier, accessor-traits, wave-3]
requires:
  - "typestate::{Fit, Predict, validate_geometry, Unfit, Fitted} (Plan 16-00)"
  - "typestate::{PredictLabels, PredictProba} accessor traits (Plan 16-00 additions)"
  - "any_estimator_typestate! macro (dispatch.rs)"
  - "Shape-A with_opts-fold recipe proven in lasso.rs/elastic_net.rs (Plan 16-02)"
  - "Shape-B trait-swap recipe proven in mbsgd_regressor.rs (Plan 16-01)"
provides:
  - "LogisticRegression<F, S=Unfit> on the typestate surface (Shape-A with_opts-fold; Fit + PredictLabels + PredictProba on Fitted — first consumer of the Plan-00 accessor traits)"
  - "LinearSVC<F, S=Unfit> on the typestate surface (Shape-B trait-swap; Fit + PredictLabels)"
  - "LinearSVR<F, S=Unfit> on the typestate surface (Shape-B trait-swap; Fit + Predict)"
  - "MBSGDClassifier<F, S=Unfit> on the typestate surface (Shape-B trait-swap; Fit + PredictLabels + PredictProba)"
  - "LogisticRegressionBuilder (.c/.fit_intercept/.max_iter/.tol; build::<F>())"
  - "PyLogisticRegression / PyLinearSVC / PyLinearSVR / PyMBSGDClassifier on any_estimator_typestate! (Fitted arms)"
  - "linear.rs (PyO3) fully off mlrs_algos::traits — the legacy glob is removed"
affects:
  - "Plans 16-04..16-08 (bulk sweep continues; the accessor-trait composition on Fitted is now proven end-to-end)"
  - "Plan 16-11 (traits.rs deletion — LogisticRegression/LinearSVC/LinearSVR/MBSGDClassifier no longer reference crate::traits; linear/ module fully migrated)"
tech-stack:
  added: []
  patterns:
    - "Accessor-trait composition on Fitted: PredictLabels/PredictProba (the Plan-00 additions) impl ONLY on T<F, Fitted> exactly like Predict — first end-to-end validation that the new accessor traits compose with the consuming-self Fit transition"
    - "PyO3 multi-accessor UFCS: a classifier wrap aliases Fit/Predict/PredictLabels/PredictProba as Typestate* and calls each via UFCS at the migrated arm; resolves the four-way fit/predict/predict_labels/predict_proba method-name collision the typestate module-doc warns about"
    - "Shape-A C>0 relocation: the data-INDEPENDENT C>0 check moves from the fit-body AlgoError::InvalidC to LogisticRegressionBuilder::build()->BuildError::InvalidC (the construction-time sibling already existed in error.rs); the fit body keeps c64 only for the l2_reg = 1/(C·n) compute"
    - "Shape-B config-getter split: config()/c()/intercept_scaling() (read pre-fit) stay on T<F, Unfit>; config() is ALSO kept on Fitted (parity); classes()/coef()/intercept() move to Fitted only (they read fitted state)"
key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/linear/logistic.rs
    - crates/mlrs-algos/tests/logistic_test.rs
    - crates/mlrs-algos/src/linear/linear_svc.rs
    - crates/mlrs-algos/tests/linear_svc_test.rs
    - crates/mlrs-algos/src/linear/linear_svr.rs
    - crates/mlrs-algos/tests/linear_svr_test.rs
    - crates/mlrs-algos/src/linear/mbsgd_classifier.rs
    - crates/mlrs-algos/tests/mbsgd_classifier_test.rs
    - crates/mlrs-py/src/estimators/linear.rs
decisions:
  - "LogisticRegression C>0 check relocated from the fit-body AlgoError::InvalidC (logistic.rs old :202-207) to LogisticRegressionBuilder::build()->BuildError::InvalidC (T-05-10-01 / Pitfall 7); the fit body still computes c64 = host_to_f64(self.c) because l2_reg = 1/(C·n) needs it — the relocation drops only the data-INDEPENDENT VALIDATION, not the value's compute use. validate_geometry replaces the inline data-DEPENDENT shape guard"
  - "Shape-B config()/c()/intercept_scaling() getters kept on the Unfit impl (the default_matches_sklearn / D-03 litmus tests read them off the BUILT Unfit value); config() additionally exposed on Fitted (parity with the pre-retrofit single impl); classes()/coef()/intercept() moved to Fitted only"
  - "PyO3 linear.rs: after all four arms migrated, the legacy `use mlrs_algos::traits::{...}` glob was the last reference in the file and is DELETED — linear.rs is now 100% on the typestate surface (the file-level path-collision is resolved by the Typestate* aliases the whole file already uses)"
  - "MBSGDClassifier is NOT a PartialFit consumer — confirmed by reading; no PartialFit impl added (only IncrementalPCA consumes PartialFit per the typestate module doc)"
metrics:
  duration: ~16m
  completed: 2026-06-24
  tasks: 3
  files: 9
status: complete
---

# Phase 16 Plan 03: Linear sweep part 2 — LogisticRegression + LinearSVC/SVR + MBSGDClassifier typestate retrofit — Summary

Migrated the four classifier/SVM estimators that exercise the NEW accessor traits added in Plan 00 (`PredictLabels`, `PredictProba`) onto the `mlrs_algos::typestate` surface, each under its own commit gated by its sklearn oracle suite AND `cargo build -p mlrs-py --features cpu`. **LogisticRegression** is shape A (with_opts-fold into a builder, plus the accessor traits); **LinearSVC / LinearSVR / MBSGDClassifier** are shape B (builders pre-existing → trait-swap only). This is the FIRST plan that depends on the Plan-00 accessor-trait additions actually existing, and it validates end-to-end that `PredictLabels`/`PredictProba` compose on a `Fitted`-tagged estimator. All four fit-body compute paths are byte-identical to pre-retrofit (D-03), all four oracle suites stay green (28 tests), and `linear.rs` (PyO3) is now fully off the legacy `mlrs_algos::traits` glob — the linear module is completely migrated.

## What Was Built

### Task 1 — LogisticRegression (Shape A, with_opts-fold; Fit + PredictLabels + PredictProba), commit `449dc35`

`crates/mlrs-algos/src/linear/logistic.rs`:
- `struct LogisticRegression<F, S = Unfit>` — added `_state: PhantomData<S>` as the only new field; `c`/`fit_intercept`/`max_iter`/`tol`/`n_classes`/`classes_`/`n_features`/`coef_`/`intercept_` UNCHANGED (D-03).
- Replaced the arg-taking `new(c, fit_intercept)` + `with_opts(c, fit_intercept, max_iter, tol)` with **zero-arg `new()`** on `impl<F> LogisticRegression<F, Unfit>` setting sklearn-equivalent defaults (`c = 1.0`, `fit_intercept = true`, `max_iter = LOG_DEFAULT_MAX_ITER = 300`, `tol = LOG_DEFAULT_TOL = 1e-5` — read from the existing constants, not invented). Added `builder()`, `into_builder()`, `hyperparams_eq()`, `impl Default`.
- New `LogisticRegressionBuilder { c, fit_intercept, max_iter, tol }` **subsumes BOTH `new` AND `with_opts`** with `.c(f64)/.fit_intercept(bool)/.max_iter(usize)/.tol(f64)` setters; `Default` = `LogisticRegression::<f64, Unfit>::new().into_builder()` (Pitfall 1 — single source); `build<F>()` relocates the data-INDEPENDENT `C > 0` check (from the old fit-body `AlgoError::InvalidC`) to `BuildError::InvalidC`, casting `c`/`tol` to `F` via `f64_to_host`.
- Imports: dropped `use crate::traits::{Fit, PredictLabels, PredictProba}`; added `use crate::error::{AlgoError, BuildError}` and `use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, PredictProba, Unfit}`.
- `impl Fit for LogisticRegression<F, Unfit>` (`type Fitted = LogisticRegression<F, Fitted>`): consuming `fit(self) -> Result<LogisticRegression<F, Fitted>, AlgoError>`; the inline shape guard → `validate_geometry`; the C>0 fit-body check removed (relocated to build); **every L-BFGS compute line** (`l2_reg = 1/(C·n)`, the `softmax_loss_grad` closure, `lbfgs_minimize`, the gauge-floor-accept convergence logic, the final `result.x` → `coef_dev`/`intercept_dev`) byte-identical; reconstructs field-by-field into the `Fitted` value.
- `coef`/`intercept`/`n_classes` accessors + `impl PredictProba` + `impl PredictLabels` moved onto `impl<F> LogisticRegression<F, Fitted>`; accessors now return `Vec<F>` (not `Result`) via `.expect(...)`; `predict_proba`'s two `ok_or(NotFitted)` guards → `.expect(...)`.

`crates/mlrs-algos/tests/logistic_test.rs`: trait import → typestate; the `fit_and_predict` call site → builder + consuming-self chain; the `with_opts(c, true, 1, 1e-5)` cap-hit test → `builder().c(c).fit_intercept(true).max_iter(1).tol(1e-5).build()`; accessors un-`.expect()`'d (now infallible). Added `defaults_equal` (BLDR-01: `LogisticRegression::new().hyperparams_eq(&LogisticRegression::builder().build()?)`).

`crates/mlrs-py/src/estimators/linear.rs` (PyLogisticRegression arm): `AnyLogisticRegression` → `any_estimator_typestate!`; fit builds via `LogisticRegression::<f*>::builder().c(c).fit_intercept(..).max_iter(..).tol(tol).build::<f*>().map_err(build_err_to_py)?` then `TypestateFit::fit(est, ...)` storing the `Fitted` value; **dropped the `c as f32` / `tol as f32` casts** (builder setters are f64). `predict_labels` → `TypestatePredictLabels::predict_labels(est, ...)`; `predict_proba` → `TypestatePredictProba::predict_proba(est, ...)`; coef/intercept accessors un-`.map_err`'d. The sklearn-named capital `C` → Rust `c` mapping at the PyO3 boundary (`#[new]` signature) is UNCHANGED; `guard_f64()`/`lock_pool()`/`validated_f32/f64` UNCHANGED.

**Gate:** `cargo test --features cpu --test logistic_test` → **7 passed** (binary/multi predict_proba 1e-5 + predict exact f64+f32, cap-hit NotConverged, fixture_loads, defaults_equal). `cargo build -p mlrs-py --features cpu` → clean.

### Task 2a — LinearSVC trait-swap (Shape B; Fit + PredictLabels), commit `1cea776`

`crates/mlrs-algos/src/linear/linear_svc.rs` (builder body / validation `:213-267` UNTOUCHED — the squared-hinge L-BFGS `svm_lbfgs_fit` helper unchanged):
- `struct LinearSVC<F, S = Unfit>` + `_state: PhantomData<S>`.
- The ONLY builder edit: `build<F>() -> Result<LinearSVC<F, Unfit>, BuildError>` (was `LinearSVC<F>`); returned literal gains `_state: PhantomData`.
- Import swap → `use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, Unfit}`.
- `impl Fit for LinearSVC<F, Unfit>` (`type Fitted = LinearSVC<F, Fitted>`): consuming `fit(self) -> Result<LinearSVC<F, Fitted>, AlgoError>`; the inline shape guard → `validate_geometry`; the `classes_` ±1 remap + `svm_lbfgs_fit` drive byte-identical; reconstructs into the `Fitted` value.
- `config()`/`c()`/`intercept_scaling()` getters split: `config`/`c`/`intercept_scaling` on `LinearSVC<F, Unfit>` (the `default_matches_sklearn` D-03 litmus reads them off the built Unfit value); `config`/`classes`/`coef`/`intercept` on `LinearSVC<F, Fitted>` (infallible `.expect`). `impl PredictLabels for LinearSVC<F, Fitted>` (body unchanged; the two `ok_or(NotFitted)` → `.expect`).

`crates/mlrs-algos/tests/linear_svc_test.rs`: trait → typestate; `fit_svc` call site → consuming-self chain; accessors un-`.expect()`'d. `default_matches_sklearn` / `build_rejects_bad_hyperparams` unchanged (operate on the built Unfit value).

`crates/mlrs-py/src/estimators/linear.rs` (PyLinearSVC arm): `AnyLinearSVC` → `any_estimator_typestate!`; fit `builder()...build::<f*>()? + TypestateFit::fit(est, ...)` storing `Fitted`; `predict_labels` → `TypestatePredictLabels::predict_labels(est, ...)`; accessors un-`.map_err`'d.

**Gate:** `cargo test --features cpu --test linear_svc_test` → **6 passed** (exact labels f32+f64, coef/intercept band f32+f64, default_matches_sklearn, build_rejects_bad_hyperparams). `cargo build -p mlrs-py --features cpu` → clean.

### Task 2b — LinearSVR trait-swap (Shape B; Fit + Predict), commit `0d3033e`

`crates/mlrs-algos/src/linear/linear_svr.rs` (builder body / validation UNTOUCHED; shared `svm_lbfgs_fit` + `predict_linear` unchanged):
- `struct LinearSVR<F, S = Unfit>` + `_state: PhantomData<S>`; `build<F>() -> Result<LinearSVR<F, Unfit>, BuildError>` (+ `_state` in the literal).
- Import swap → `use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit}`.
- Consuming-self `Fit::fit`; the inline shape guard → `validate_geometry`; the squared-eps-insensitive `svm_lbfgs_fit` drive byte-identical; reconstructs into `Fitted`. `config`/`c`/`intercept_scaling` stay on `Unfit`; `config`/`coef`/`intercept` on `Fitted` (infallible). `impl Predict for LinearSVR<F, Fitted>` (body `predict_linear`, unchanged).

`crates/mlrs-algos/tests/linear_svr_test.rs`: trait → typestate; `fit_svr` → consuming-self chain; accessors un-`.expect()`'d. `fixture_loads` / `default_matches_sklearn` / `build_rejects_bad_hyperparams` unchanged.

`crates/mlrs-py/src/estimators/linear.rs` (PyLinearSVR arm): `AnyLinearSVR` → `any_estimator_typestate!`; fit + UFCS `TypestateFit::fit`; predict → `TypestatePredict::predict`; accessors un-`.map_err`'d. **Dropped the now-unused legacy `Predict` from the file's `mlrs_algos::traits` glob** (LinearSVR was its last `Predict` consumer).

**Gate:** `cargo test --features cpu --test linear_svr_test` → **5 passed** (coef/intercept/predict band f32+f64, fixture_loads, default_matches_sklearn). `cargo build -p mlrs-py --features cpu` → clean.

### Task 3 — MBSGDClassifier trait-swap (Shape B; Fit + PredictLabels + PredictProba), commit `44e62cb`

`crates/mlrs-algos/src/linear/mbsgd_classifier.rs` (builder body / validation `:201-287` UNTOUCHED; the shared `lower_config` free function — also called by `mbsgd_regressor.rs` — UNCHANGED):
- `struct MBSGDClassifier<F, S = Unfit>` + `_state: PhantomData<S>`; `build<F>() -> Result<MBSGDClassifier<F, Unfit>, BuildError>` (+ `_state`).
- Import swap → `use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, PredictProba, Unfit}`.
- Consuming-self `Fit::fit`; the inline shape guard → `validate_geometry`; the `classes_` ±1 remap + `lower_config` + `sgd_solve` drive byte-identical; reconstructs into `Fitted`. `builder`/`config` on `Unfit`; `config`/`classes`/`coef`/`intercept` on `Fitted` (infallible). `impl PredictLabels` + `impl PredictProba` + the shared `decision_margin` inherent helper all moved onto `MBSGDClassifier<F, Fitted>` (the two `ok_or(NotFitted)` in `decision_margin` → `.expect`). **Confirmed (read): no `PartialFit` impl** — MBSGDClassifier is not a PartialFit consumer (only IncrementalPCA is).

`crates/mlrs-algos/tests/mbsgd_classifier_test.rs`: trait → typestate; both `fit_hinge_sched` and `fit_log_proba` fit chains → consuming-self; accessors un-`.expect()`'d. The constant/optimal-schedule oracle + proba + `default_matches_sklearn` + `build_rejects_bad_alpha` tests operate unchanged (the litmus reads `config()` off the built Unfit value).

`crates/mlrs-py/src/estimators/linear.rs` (PyMBSGDClassifier arm): `AnyMBSGDClassifier` → `any_estimator_typestate!`; fit + UFCS `TypestateFit::fit`; `predict_labels` / `predict_proba` → `TypestatePredictLabels` / `TypestatePredictProba` UFCS; accessors un-`.map_err`'d. **Removed the now-fully-unused legacy `use mlrs_algos::traits::{Fit, PredictLabels, PredictProba};` glob** — linear.rs is now 100% on the typestate surface (updated the file-level comment accordingly).

**Gate:** `cargo test --features cpu --test mbsgd_classifier_test` → **10 passed** (exact labels f32+f64, coef/intercept band constant + optimal schedule f32+f64, predict_proba f32+f64, default_matches_sklearn, build_rejects_bad_alpha). `cargo build -p mlrs-py --features cpu` → clean.

## The Accessor-Trait Composition (validated for the remaining sweep plans)

This plan is the first end-to-end proof that the Plan-00 accessor traits compose on a `Fitted` estimator. The pattern the NB / KNN plans (16-04..16-08) copy:

**Estimator side** — the accessor trait impls move onto the `Fitted` sibling exactly like `Predict`:
```rust
impl<F> PredictLabels<F> for Estimator<F, Fitted> { fn predict_labels(&self, ...) {...} }
impl<F> PredictProba<F>  for Estimator<F, Fitted> { fn predict_proba(&self, ...)  {...} }
```
They carry NO `type Fitted` (they read fitted state, they do not transition), so they need no signature change beyond the `<F, Fitted>` receiver.

**PyO3 side** — alias every consumed trait and call via UFCS at the migrated arm (the four-way method-name collision `fit`/`predict`/`predict_labels`/`predict_proba` is why UFCS is mandatory):
```rust
use mlrs_algos::typestate::{
    Fit as TypestateFit, Predict as TypestatePredict,
    PredictLabels as TypestatePredictLabels, PredictProba as TypestatePredictProba,
};
// at the F32/F64 (Fitted) arm:
TypestatePredictLabels::predict_labels(est, &mut pool, &xd, (rows, cols))
TypestatePredictProba::predict_proba(est, &mut pool, &xd, (rows, cols))
```
Switch the `Any*` enum to `any_estimator_typestate!`. Drop any `as f32`/`as f64` casts (builder setters are f64). Once a file's LAST legacy-trait estimator migrates, DELETE the `mlrs_algos::traits` glob.

## Deviations from Plan

None. All four estimators followed the documented recipes exactly — LogisticRegression the Plan-02 with_opts-fold (its `C>0` check relocates to `build()` like the Lasso/ElasticNet `alpha>=0`), and LinearSVC/LinearSVR/MBSGDClassifier the Plan-01 Shape-B trait-swap (builder bodies untouched). The four fit bodies are byte-identical (verified: `git diff` on each shows no change to any `softmax_loss_grad`/`lbfgs_minimize`/`svm_lbfgs_fit`/`sgd_solve`/`l2_reg`/`gemm` compute line — only signature, return, guard-call, and struct-reconstruction edits). The legacy `Predict`/glob removals in linear.rs are the natural consequence of migrating the file's last legacy consumers (the plan's "switch the Any* enum + UFCS at migrated arms" instruction taken to its terminal state), not a scope deviation.

## Authentication Gates

None.

## Threat Mitigations (from the plan's threat_model)

- **T-16-V5** (classifier/SVM fit geometry guard + relocated validation): `validate_geometry(x, shape)?` is at the TOP of all four ported `fit`s, before any device launch. The data-INDEPENDENT `C > 0` check (LogisticRegression) is relocated to `LogisticRegressionBuilder::build()` → `BuildError::InvalidC`, NOT dropped; the shape-B builders' `C > 0`/loss/penalty validation is UNTOUCHED. ✓
- **T-16-GUARDF64** (F64 guard): `crate::capability::guard_f64()?` preserved verbatim before every F64 upload in the four migrated PyO3 fits; `lock_pool()` (poison-recovering) kept. ✓
- **T-16-ARM** (Fitted arm type): `AnyLogisticRegression`/`AnyLinearSVC`/`AnyLinearSVR`/`AnyMBSGDClassifier` switched to `any_estimator_typestate!` so each fitted value is typed `T<f*, Fitted>` — no `Unfit` value stored in a fitted arm. ✓

## Known Stubs

None.

## Threat Flags

None — no new network/auth/file/schema surface introduced; a trait-surface retrofit with byte-identical compute.

## Verification Environment Note

`cargo build -p mlrs-py --features cpu` is the cross-crate PyO3 gate (the live Python pytest of the wheel is untestable in this environment per the project memory note "Python wheel untestable in env" — no maturin/pyarrow). The four oracle suites + the mlrs-py build are the compensating Rust gates; the Python-boundary behavior (Unfit-arm accessor → `not_fitted` → PyValueError) is unchanged from the pre-retrofit shells.

## Acceptance Evidence

- `cargo test --features cpu --test logistic_test` → **7 passed** (predict_proba 1e-5 / predict exact, cap-hit NotConverged, defaults_equal).
- `cargo test --features cpu --test linear_svc_test` → **6 passed** (exact labels HARD gate + coef/intercept band).
- `cargo test --features cpu --test linear_svr_test` → **5 passed** (coef/intercept/predict band).
- `cargo test --features cpu --test mbsgd_classifier_test` → **10 passed** (exact labels + constant/optimal schedule band + predict_proba).
- `cargo build -p mlrs-py --features cpu` → Finished (2 pre-existing spectral.rs dead-code warnings only, out of scope).
- `! grep -q 'crate::traits'` on logistic.rs / linear_svc.rs / linear_svr.rs / mbsgd_classifier.rs → all clean.
- `grep -cE 'typestate::(PredictLabels|PredictProba)...Fitted' logistic.rs` → 2 (accessor traits impl'd on Fitted).
- No real `crate::any_estimator!` invocation and no `mlrs_algos::traits` import remain in linear.rs (only stale doc-comment mentions; the active glob is deleted).
- No `PartialFit` impl in mbsgd_classifier.rs (verified by `grep`).
- D-03: per-file `git diff` shows ZERO compute-line changes (signature/return/guard-call/reconstruction only).

## Self-Check: PASSED

- `crates/mlrs-algos/src/linear/logistic.rs` — FOUND, builds, 7 tests pass.
- `crates/mlrs-algos/src/linear/linear_svc.rs` — FOUND, builds, 6 tests pass.
- `crates/mlrs-algos/src/linear/linear_svr.rs` — FOUND, builds, 5 tests pass.
- `crates/mlrs-algos/src/linear/mbsgd_classifier.rs` — FOUND, builds, 10 tests pass.
- `crates/mlrs-py/src/estimators/linear.rs` — FOUND, mlrs-py builds (legacy traits glob removed).
- Commit `449dc35` (LogisticRegression) — FOUND.
- Commit `1cea776` (LinearSVC) — FOUND.
- Commit `0d3033e` (LinearSVR) — FOUND.
- Commit `44e62cb` (MBSGDClassifier) — FOUND.
