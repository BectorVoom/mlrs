---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 01
subsystem: linear-estimators
tags: [typestate, builder-retrofit, pilots, ridge, mbsgd-regressor, wave-1]
requires:
  - "typestate::{Fit, Predict, validate_geometry, Unfit, Fitted} (Plan 16-00)"
  - "any_estimator_typestate! macro (Plan 04 / dispatch.rs)"
provides:
  - "Ridge<F, S=Unfit> on the typestate surface (Shape-A full build-out recipe, proven)"
  - "MBSGDRegressor<F, S=Unfit> on the typestate surface (Shape-B trait-swap recipe, proven)"
  - "RidgeBuilder (f64 setters: .alpha(f64)/.fit_intercept(bool); build::<F>())"
  - "PyRidge / PyMBSGDRegressor on any_estimator_typestate! (Fitted arms)"
affects:
  - "Plans 16-02..16-08 (bulk estimator sweep — both retrofit shapes now have an in-tree, green-gated recipe to copy)"
  - "Plan 16-11 (traits.rs deletion — Ridge + MBSGDRegressor no longer reference crate::traits)"
tech-stack:
  added: []
  patterns:
    - "Shape-A: struct gains <F, S=Unfit> + PhantomData<S>; zero-arg new() = single source of sklearn defaults; builder Default re-derived via new().into_builder(); data-independent check relocates to build()->BuildError; fit consumes self -> T<F, Fitted> (compute byte-identical); accessors/Predict move onto T<F, Fitted> dropping NotFitted -> .expect()"
    - "Shape-B: builder UNTOUCHED; only struct param + build::<F>() return type (-> T<F, Unfit>) + trait swap + consuming-self fit"
    - "PyO3 cross-surface mid-migration: alias typestate as `Fit as TypestateFit` / `Predict as TypestatePredict`, call via UFCS at the migrated arms only (legacy traits glob stays for the file's other estimators)"
key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/linear/ridge.rs
    - crates/mlrs-algos/tests/ridge_test.rs
    - crates/mlrs-algos/src/linear/mbsgd_regressor.rs
    - crates/mlrs-algos/tests/mbsgd_regressor_test.rs
    - crates/mlrs-py/src/estimators/linear.rs
decisions:
  - "Ridge alpha>=0 check relocated from fit-body (AlgoError::InvalidAlpha) to build() (BuildError::InvalidAlpha); geometry stays in fit via validate_geometry (T-16-V5 / Pitfall 7)"
  - "linear.rs keeps the legacy `mlrs_algos::traits` glob (7 other estimators still on it) and imports typestate Fit/Predict under TypestateFit/TypestatePredict aliases, called via UFCS at the Ridge/MBSGD arms only — avoids the two-surface path collision the typestate module doc warns about"
  - "MBSGDRegressor (shape B, builder-only) has no zero-arg new(), so no hyperparams_eq; the pre-existing default_matches_sklearn test is its BLDR-01 defaults-equality gate. config() exposed on BOTH Unfit and Fitted (default_matches_sklearn reads it off the built Unfit value)"
metrics:
  duration: ~10m
  completed: 2026-06-24
  tasks: 2
  files: 5
status: complete
---

# Phase 16 Plan 01: Pilots — Ridge + MBSGDRegressor typestate retrofit — Summary

Migrated the two structurally-distinct pilot estimators onto the `mlrs_algos::typestate` surface, each under its own commit gated by its sklearn oracle suite AND `cargo build -p mlrs-py --features cpu`. **Ridge** (shape A — no builder, arg-taking `new`) got the full build-out; **MBSGDRegressor** (shape B — builder already shipped) got the minimal trait-swap. Both fit-body compute lines are byte-identical to pre-retrofit (D-03 invariant verified via `git diff`), and both oracle suites stay green at their documented tolerances. The mechanical recipe for the broad sweep (Plans 16-02..16-08) is now proven in-tree.

## What Was Built

### Task 1 — Pilot A: Ridge full build-out (Shape A), commit `c42bfbb`

`crates/mlrs-algos/src/linear/ridge.rs`:
- `struct Ridge<F, S = Unfit>` — added `_state: PhantomData<S>` as the ONLY new field; `alpha`/`fit_intercept`/`coef_`/`intercept_` UNCHANGED (D-03).
- Replaced the arg-taking `new(alpha, fit_intercept)` with **zero-arg `new()`** on `impl<F> Ridge<F, Unfit>` setting sklearn defaults (`alpha = F::from_int(1)` = 1.0, `fit_intercept = true`) — the single source of defaults. Added `builder()`, `into_builder()`, `hyperparams_eq()`, and `impl Default for Ridge<F, Unfit>`.
- New `RidgeBuilder { alpha: f64, fit_intercept: bool }` with **f64 setters** `.alpha(v: f64)` / `.fit_intercept(v: bool)` (A5 convention), `Default` = `Ridge::<f64, Unfit>::new().into_builder()` (Pitfall 1 — single source, no re-listed literals), and `build<F>() -> Result<Ridge<F, Unfit>, BuildError>` performing the data-INDEPENDENT `alpha >= 0` check → `BuildError::InvalidAlpha` (relocated from the old fit-body `AlgoError::InvalidAlpha` at ridge.rs:140-146 — Pitfall 7), casting the f64 `alpha` to `F` via `f64_to_host::<F>`.
- Imports: dropped `use crate::traits::{Fit, Predict}`; added `use crate::error::{AlgoError, BuildError}` and `use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit}`.
- `impl Fit for Ridge<F, Unfit>` (`type Fitted = Ridge<F, Fitted>`): converted `fit(&mut self) -> &mut Self` to **consuming `fit(self) -> Result<Ridge<F, Fitted>, AlgoError>`**. Replaced the inline geometry guard with `validate_geometry(x, shape)?`; **every compute line** (centering, raw Gram via `gemm(transa)`, diagonal-α injection, Xᵀy, `cholesky_solve`, intercept recovery) is byte-identical. Reconstructs field-by-field into the `Fitted` value with `_state: PhantomData`.
- `coef`/`intercept` accessors + `impl Predict` moved onto `impl<F> Ridge<F, Fitted>`; accessors now return `Vec<F>`/`F` (not `Result`) via `.expect("Some by construction on Ridge<F, Fitted>")`; predict drops its two `ok_or(NotFitted)` for `.expect(...)`.

`crates/mlrs-algos/tests/ridge_test.rs`: trait import → typestate; 2 call sites migrated (`Ridge::<F>::new(args)` → `Ridge::<F>::builder().alpha(..).fit_intercept(..).build::<F>()?.fit(..)?`); accessors un-`.expect()`'d (now infallible). Added `defaults_equal` (BLDR-01: `Ridge::new().hyperparams_eq(&Ridge::builder().build()?)`).

`crates/mlrs-py/src/estimators/linear.rs` (PyRidge arm): `AnyRidge` → `any_estimator_typestate!`; fit builds via `Ridge::<f*>::builder().alpha(alpha).fit_intercept(fit_intercept).build::<f*>().map_err(build_err_to_py)?` then `TypestateFit::fit(est, ...)` (UFCS, aliased), storing the `Fitted` value; **dropped the `as f32` cast** (builder setter is f64). predict → `TypestatePredict::predict(est, ...)`; coef/intercept accessors un-`.map_err`'d. `guard_f64()`, `lock_pool()`, `validated_f32/f64` UNCHANGED.

**Gate:** `cargo test --features cpu --test ridge_test` → **6 passed** (5 pre-existing oracle/consistency + 1 new `defaults_equal`). `cargo build -p mlrs-py --features cpu` → clean.

### Task 2 — Pilot B: MBSGDRegressor trait-swap (Shape B), commit `5d6d01a`

`crates/mlrs-algos/src/linear/mbsgd_regressor.rs` (builder body UNTOUCHED — validation :217-265 + SgdConfig lowering :266-281 unchanged):
- `struct MBSGDRegressor<F, S = Unfit>` + `_state: PhantomData<S>` (config/coef_/intercept_ unchanged).
- The ONLY builder edit: `build<F>() -> Result<MBSGDRegressor<F, Unfit>, BuildError>` (was `MBSGDRegressor<F>`); returned literal gains `_state: PhantomData`.
- Import swap → `use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit}`.
- `impl Fit for MBSGDRegressor<F, Unfit>` (`type Fitted = MBSGDRegressor<F, Fitted>`): consuming `fit(self) -> Result<MBSGDRegressor<F, Fitted>, AlgoError>`; the `lower_config` + `sgd_solve` drive is byte-identical; geometry guard → `validate_geometry`; reconstructs into the `Fitted` value.
- `coef`/`intercept` moved onto `impl<F> MBSGDRegressor<F, Fitted>` (infallible `.expect`); `config()` exposed on BOTH `Unfit` and `Fitted` (the `default_matches_sklearn` test reads `config()` off the built `Unfit` value). `impl Predict for MBSGDRegressor<F, Fitted>` (body unchanged — `predict_linear`).

`crates/mlrs-algos/tests/mbsgd_regressor_test.rs`: trait import → typestate; 2 fit call sites → consuming-self chain; accessors un-`.expect()`'d. `default_matches_sklearn` / `build_rejects_bad_hyperparams` unchanged (they operate on the built `Unfit` value).

`crates/mlrs-py/src/estimators/linear.rs` (PyMBSGDRegressor arm): `AnyMBSGDRegressor` → `any_estimator_typestate!`; fit `build::<f*>()? + TypestateFit::fit(est, ...)` storing `Fitted`; predict → `TypestatePredict::predict`; accessors un-`.map_err`'d.

**Gate:** `cargo test --features cpu --test mbsgd_regressor_test` → **5 passed** (`oracle`, `oracle_f32`, `oracle_epsilon_f32`, `default_matches_sklearn`, `build_rejects_bad_hyperparams`). `cargo build -p mlrs-py --features cpu` → clean.

## The Proven Recipe (for the bulk-sweep plans to copy)

**RidgeBuilder setter signatures (Shape-A reference):**
```rust
pub fn alpha(mut self, v: f64) -> Self { self.alpha = v; self }
pub fn fit_intercept(mut self, v: bool) -> Self { self.fit_intercept = v; self }
pub fn build<F>(self) -> Result<Ridge<F, Unfit>, BuildError> where F: Float + CubeElement + Pod { /* data-indep check -> BuildError; f64_to_host::<F>(self.alpha) */ }
impl Default for RidgeBuilder { fn default() -> Self { Ridge::<f64, Unfit>::new().into_builder() } }
```

**Consuming-self fit (both shapes):**
```rust
impl<F> Fit<F> for Estimator<F, Unfit> { type Fitted = Estimator<F, Fitted>;
  fn fit(self, pool, x, y, shape) -> Result<Estimator<F, Fitted>, AlgoError> {
    validate_geometry(x, shape)?;            // data-DEPENDENT guard, BEFORE any launch
    /* …compute body byte-identical… */
    Ok(Estimator { /* hyperparams from self */, coef_: Some(..), _state: PhantomData })
  }}
```

**PyO3 mid-migration aliasing (the file-level path-collision fix):** when a PyO3 file still has legacy-trait estimators, keep the `use mlrs_algos::traits::{...}` glob and add
```rust
use mlrs_algos::typestate::{Fit as TypestateFit, Predict as TypestatePredict};
```
then call `TypestateFit::fit(est, …)` / `TypestatePredict::predict(est, …)` (UFCS) at the migrated arms ONLY. Switch the `Any*` enum to `any_estimator_typestate!`. Drop any `alpha as f32`-style cast (builder setter is already f64). Keep `guard_f64()` / `lock_pool()` / `validated_f32/f64` / `build_err_to_py` / `algo_err_to_py` verbatim.

## Deviations from Plan

**[Plan-intent adjustment, not a code deviation] MBSGDRegressor `hyperparams_eq`.** The plan's Task-2 action says "Add the `hyperparams_eq` defaults-equality assertion." MBSGDRegressor is shape B (builder-only — no zero-arg `new()`), so there is no `new()` vs `builder().build()` pair to compare and no `hyperparams_eq` method to add (its hyperparameters live in the lowered `SgdConfig`, not as struct fields). The pre-existing `default_matches_sklearn` test (`builder().build()` reproduces every sklearn `SGDRegressor` default) IS the BLDR-01 defaults-equality gate for this estimator and was preserved green. No `hyperparams_eq` was invented. This matches the §2 Shape-B delta which is explicitly "typestate-param + trait-swap ONLY" and does not add a `new()`.

**[Rule 2 - correctness] `config()` exposed on the Fitted arm too.** The plan moves accessors onto `MBSGDRegressor<F, Fitted>`. `config()` was on the single pre-retrofit impl; it is kept on `Unfit` (the `default_matches_sklearn` test needs it on the built value) and also added to the `Fitted` impl so a fitted estimator can still read its lowered config (parity with the pre-retrofit single-impl surface). No behavior change.

No other deviations — both fit bodies are byte-identical (verified: `git diff c42bfbb~1 c42bfbb -- ridge.rs` shows zero changes to any `gemm`/`cholesky_solve`/`column_reduce`/centering/diagonal-α compute line).

## Authentication Gates

None.

## Threat Mitigations (from the plan's threat_model)

- **T-16-V5** (geometry guard / relocated alpha check): `validate_geometry(x, shape)?` is at the TOP of both ported `fit`s, before any device launch; the `alpha >= 0` check is relocated to `RidgeBuilder::build()` → `BuildError::InvalidAlpha`, NOT dropped. ✓
- **T-16-GUARDF64** (F64 guard): `crate::capability::guard_f64()?` preserved verbatim before both F64 uploads; `lock_pool()` (poison-recovering) kept. ✓
- **T-16-ARM** (Fitted arm type): `AnyRidge`/`AnyMBSGDRegressor` switched to `any_estimator_typestate!` so a fitted value is typed `T<f*, Fitted>` — no `Unfit` value stored in a fitted arm. ✓

## Known Stubs

None.

## Threat Flags

None — no new network/auth/file/schema surface introduced; this is a trait-surface retrofit with byte-identical compute.

## Verification Environment Note

`cargo build -p mlrs-py --features cpu` is the cross-crate PyO3 gate (the live Python pytest of the wheel is untestable in this environment per the project memory note "Python wheel untestable in env" — no maturin/pyarrow). Both oracle suites + the mlrs-py build are the compensating Rust gates; the Python-boundary behavior (Unfit-arm accessor → `not_fitted` → PyValueError) is unchanged from the pre-retrofit shells.

## Acceptance Evidence

- `cargo test --features cpu --test ridge_test` → **6 passed** (1e-5 oracle f64+f32, defaults_equal).
- `cargo test --features cpu --test mbsgd_regressor_test` → **5 passed** (documented-band oracle f64+f32, default_matches_sklearn).
- `cargo build -p mlrs-py --features cpu` → Finished (2 pre-existing spectral.rs dead-code warnings only, out of scope).
- `grep -c 'fn new(' ridge.rs` → **1** (zero-arg only); `grep -c 'new(alpha' ridge.rs` → **0**.
- `! grep -q 'crate::traits' ridge.rs` and `! grep -q 'crate::traits' mbsgd_regressor.rs` → both clean.
- `any_estimator_typestate!` at linear.rs:196 (AnyRidge) and :859 (AnyMBSGDRegressor).
- D-03: `git diff c42bfbb~1 c42bfbb -- ridge.rs` shows ZERO compute-line changes (signature/return/guard-call only).

## Self-Check: PASSED

- `crates/mlrs-algos/src/linear/ridge.rs` — FOUND, builds, 6 tests pass.
- `crates/mlrs-algos/src/linear/mbsgd_regressor.rs` — FOUND, builds, 5 tests pass.
- `crates/mlrs-py/src/estimators/linear.rs` — FOUND, mlrs-py builds.
- Commit `c42bfbb` (Ridge) — FOUND.
- Commit `5d6d01a` (MBSGDRegressor) — FOUND.
