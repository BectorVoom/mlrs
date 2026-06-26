---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 02
subsystem: linear-estimators
tags: [typestate, builder-retrofit, with-opts-fold, linear, wave-2]
requires:
  - "typestate::{Fit, Predict, validate_geometry, Unfit, Fitted} (Plan 16-00)"
  - "any_estimator_typestate! macro (dispatch.rs)"
  - "Shape-A recipe proven in ridge.rs (Plan 16-01)"
provides:
  - "LinearRegression<F, S=Unfit> on the typestate surface (Shape-A, single-flag builder)"
  - "Lasso<F, S=Unfit> + LassoBuilder (subsumes new + with_opts: alpha/fit_intercept/max_iter/tol)"
  - "ElasticNet<F, S=Unfit> + ElasticNetBuilder (alpha/l1_ratio/fit_intercept/max_iter/tol)"
  - "PyLinearRegression / PyLasso / PyElasticNet on any_estimator_typestate! (Fitted arms)"
affects:
  - "Plans 16-03..16-08 (bulk sweep continues; with_opts-fold pattern now proven)"
  - "Plan 16-11 (traits.rs deletion — these three no longer reference crate::traits)"
tech-stack:
  added: []
  patterns:
    - "with_opts fold: a multi-arg secondary constructor (max_iter, tol) collapses into builder setters; zero-arg new() carries all defaults; the data-independent validation that lived in the shared fit helper (cd_fit alpha>=0 / l1_ratio in [0,1]) relocates to build()->BuildError"
    - "OLS infallible-but-typed build(): LinearRegression has no data-independent hyperparam, so build() returns Result<_, BuildError> that never errs — keeps the build_err_to_py PyO3 mapper shape-identical across the linear family"
key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/linear/linear_regression.rs
    - crates/mlrs-algos/tests/linear_regression_test.rs
    - crates/mlrs-algos/src/linear/lasso.rs
    - crates/mlrs-algos/tests/lasso_test.rs
    - crates/mlrs-algos/src/linear/elastic_net.rs
    - crates/mlrs-algos/tests/elastic_net_test.rs
    - crates/mlrs-py/src/estimators/linear.rs
decisions:
  - "LinearRegression build() is infallible-but-typed (Result<_, BuildError>): OLS has no data-INDEPENDENT hyperparameter to validate, but the Result is kept for uniformity with the penalized builders so the PyO3 build_err_to_py mapper is shape-identical across the family (plan Task-1 action)"
  - "Lasso/ElasticNet data-independent checks (alpha>=0, l1_ratio in [0,1]) relocated from the SHARED cd_fit helper to each builder's build()->BuildError; cd_fit itself is UNCHANGED (other call paths keep their own guard) so the fit-body math stays byte-identical (D-03 / Pitfall 7)"
  - "Builder defaults are single-sourced from new() via into_builder() (Pitfall 1); Lasso/ElasticNet new() read CD_DEFAULT_MAX_ITER/CD_DEFAULT_TOL — the same constants the old new()/with_opts used"
metrics:
  duration: ~9m
  completed: 2026-06-24
  tasks: 3
  files: 7
status: complete
---

# Phase 16 Plan 02: Linear sweep part 1 — LinearRegression + Lasso + ElasticNet typestate retrofit — Summary

Migrated the three shape-A linear estimators onto the `mlrs_algos::typestate` surface, each under its own commit gated by its sklearn oracle suite AND `cargo build -p mlrs-py --features cpu`. **LinearRegression** is the simple single-flag case; **Lasso** and **ElasticNet** are the first `with_opts(...)` cases — their multi-arg secondary constructor (`max_iter`, `tol`) folds entirely into builder setters, and the data-independent hyperparameter validation that lived in the shared `cd_fit` helper relocates to each builder's `build()`. All three fit-body compute paths are byte-identical to pre-retrofit (D-03), and all three oracle suites stay green at 1e-5 (f64 + f32).

## What Was Built

### Task 1 — LinearRegression (Shape A, single-arg new), commit `b8854a1`

`crates/mlrs-algos/src/linear/linear_regression.rs`:
- `struct LinearRegression<F, S = Unfit>` — added `_state: PhantomData<S>` as the only new field; `fit_intercept`/`coef_`/`intercept_` unchanged (D-03). `RCOND`/`NEAR_ZERO_FLOOR` consts unchanged.
- Replaced `new(fit_intercept)` with **zero-arg `new()`** (`fit_intercept = true`, sklearn default) on `impl<F> LinearRegression<F, Unfit>`; added `builder()`, `into_builder()`, `hyperparams_eq()`, `impl Default`.
- New `LinearRegressionBuilder { fit_intercept: bool }` with `.fit_intercept(bool)`, `Default` = `LinearRegression::<f64, Unfit>::new().into_builder()`, and `build<F>() -> Result<_, BuildError>` that is **infallible-but-typed** (OLS has no data-INDEPENDENT hyperparam; the `Result` is kept for family uniformity so `build_err_to_py` is shape-identical).
- Imports: dropped `use crate::traits::{Fit, Predict}`; added `use crate::error::{AlgoError, BuildError}` and the typestate surface.
- `impl Fit for LinearRegression<F, Unfit>` (consuming `self -> Result<LinearRegression<F, Fitted>, AlgoError>`): inline geometry guard → `validate_geometry`; **every SVD compute line** (centering, `column_reduce`, thin `svd`, σ⁺ cutoff, `gemm` compositions, intercept recovery) byte-identical; reconstructs into the `Fitted` value.
- `coef`/`intercept` accessors + `impl Predict` moved onto `impl<F> LinearRegression<F, Fitted>`; accessors return `Vec<F>`/`F` via `.expect(...)`; predict drops its two `ok_or(NotFitted)`.

`crates/mlrs-algos/tests/linear_regression_test.rs`: trait import → typestate; 2 fit call sites → builder + consuming-self; accessors un-`.expect()`'d; added `defaults_equal` (BLDR-01).

`crates/mlrs-py/src/estimators/linear.rs` (PyLinearRegression arm): `AnyLinearRegression` → `any_estimator_typestate!`; fit builds via `LinearRegression::<f*>::builder().fit_intercept(..).build::<f*>().map_err(build_err_to_py)?` then `TypestateFit::fit(...)`; predict → `TypestatePredict::predict(...)`; coef/intercept arms un-`.map_err`'d. `guard_f64()` / `lock_pool()` / `validated_f32/f64` unchanged.

**Gate:** `cargo test --features cpu --test linear_regression_test` → **7 passed** (6 pre-existing oracle/collinear + `defaults_equal`). `cargo build -p mlrs-py --features cpu` → clean.

### Task 2 — Lasso (Shape A, new + with_opts → builder), commit `9b2a9ed`

`crates/mlrs-algos/src/linear/lasso.rs`:
- `struct Lasso<F, S = Unfit>` + `_state: PhantomData<S>`; `alpha`/`fit_intercept`/`max_iter`/`tol`/`coef_`/`intercept_` unchanged.
- Zero-arg `new()` sets all four defaults (`alpha = 1.0`, `fit_intercept = true`, `max_iter = CD_DEFAULT_MAX_ITER`, `tol = CD_DEFAULT_TOL` — read from the existing constants, not invented). `builder()`/`into_builder()`/`hyperparams_eq()`/`Default`.
- `LassoBuilder { alpha: f64, fit_intercept: bool, max_iter: usize, tol: f64 }` **subsumes both `new` AND `with_opts`** with `.alpha`/`.fit_intercept`/`.max_iter`/`.tol` setters; `build<F>()` relocates the data-INDEPENDENT `alpha >= 0` check (from the `cd_fit` helper) to `BuildError::InvalidAlpha`. `fn with_opts` REMOVED.
- Consuming-self `Fit::fit`: `validate_geometry` guard added at the top; the `cd_fit(...)` CD-solver delegation (penalty map, centering, `cd_solve`, intercept recovery) is **byte-identical**. Predict/accessors moved onto `Lasso<F, Fitted>`.

`crates/mlrs-algos/tests/lasso_test.rs`: trait import → typestate; fit call site → builder (`new`-with-defaults form); accessors un-wrapped; added `defaults_equal`.

`crates/mlrs-py/src/estimators/linear.rs` (PyLasso arm): `AnyLasso` → `any_estimator_typestate!`; fit `builder().alpha(..).fit_intercept(..).max_iter(..).tol(..).build::<f*>()? + TypestateFit::fit`; **dropped the `alpha as f32` cast** (builder setter is f64); predict → UFCS; accessors un-`.map_err`'d.

**Gate:** `cargo test --features cpu --test lasso_test` → **4 passed** (`fixture_loads`, `lasso_sparse_coef_match_sklearn_f32`, `lasso_coef_intercept_match_sklearn_f64`, `defaults_equal`). `cargo build -p mlrs-py --features cpu` → clean.

### Task 3 — ElasticNet (Shape A, new + with_opts → builder), commit `ea286a6`

`crates/mlrs-algos/src/linear/elastic_net.rs` (Lasso's sibling — same CD-solver family):
- `struct ElasticNet<F, S = Unfit>` + `_state`; hyperparam fields unchanged. The shared `predict_linear` helper (used by both Lasso and ElasticNet) is UNTOUCHED.
- Zero-arg `new()` sets all five defaults (`alpha = 1.0`, `l1_ratio = 0.5`, `fit_intercept = true`, `max_iter = CD_DEFAULT_MAX_ITER`, `tol = CD_DEFAULT_TOL`). `builder()`/`into_builder()`/`hyperparams_eq()`/`Default`.
- `ElasticNetBuilder` exposes all five setters `.alpha`/`.l1_ratio`/`.fit_intercept`/`.max_iter`/`.tol`; `build<F>()` relocates BOTH data-INDEPENDENT checks — `alpha >= 0` (`BuildError::InvalidAlpha`) and `0 <= l1_ratio <= 1` (`BuildError::InvalidL1Ratio`) — from the `cd_fit` helper. `fn with_opts` REMOVED.
- Consuming-self `Fit::fit`: `validate_geometry` guard; `cd_fit(...)` delegation byte-identical. Predict/accessors onto `ElasticNet<F, Fitted>`.

`crates/mlrs-algos/tests/elastic_net_test.rs`: trait → typestate; fit call site → builder; accessors un-wrapped; `defaults_equal` added.

`crates/mlrs-py/src/estimators/linear.rs` (PyElasticNet arm): `AnyElasticNet` → `any_estimator_typestate!`; fit builder + UFCS; dropped `alpha as f32` / `l1_ratio as f32` casts; predict UFCS; accessors un-`.map_err`'d.

**Gate:** `cargo test --features cpu --test elastic_net_test` → **4 passed** (`fixture_loads`, `elastic_net_coef_match_sklearn_f32`, `elastic_net_coef_intercept_match_sklearn_f64`, `defaults_equal`). `cargo build -p mlrs-py --features cpu` → clean.

## The with_opts-Fold Recipe (for the remaining sweep plans with secondary constructors)

1. zero-arg `new()` sets EVERY hyperparameter from the existing `new()`/`with_opts()` defaults — read the literals/constants, do NOT re-invent.
2. The builder exposes ALL of them as `f64`/`usize`/`bool` setters; `with_opts` is DELETED (its arguments are now setters).
3. Any data-INDEPENDENT validation that lived in a SHARED fit helper (here `cd_fit`'s `alpha>=0` / `l1_ratio∈[0,1]`) relocates to `build()->BuildError`. The shared helper is left UNCHANGED (other estimators may still call it) so the fit-body math stays byte-identical — defense-in-depth, not a removed check.
4. PyO3 arm: switch `any_estimator!` → `any_estimator_typestate!`, set every former `with_opts` arg via a builder setter, drop the `as f32` casts, call via `TypestateFit::fit` / `TypestatePredict::predict` (UFCS aliases).

## Deviations from Plan

None — all three estimators followed the Pilot-A recipe (with the documented with_opts-fold for Lasso/ElasticNet) exactly as the plan's Task actions specified. The LinearRegression "infallible-but-typed build()" is the plan's own Task-1 instruction ("no data-independent hyperparam to validate beyond construction — keep build infallible-but-typed for uniformity"), not a deviation. The three fit bodies are byte-identical (verified: the per-file `git diff` shows no change to any `svd`/`gemm`/`column_reduce`/`cd_fit`/`cutoff` compute line — only signature, return, guard-call, and struct-reconstruction edits).

## Authentication Gates

None.

## Threat Mitigations (from the plan's threat_model)

- **T-16-V5** (geometry guard / relocated validation): `validate_geometry(x, shape)?` is at the TOP of all three ported `fit`s, before any device launch (2 sites per file — fit + predict). The `alpha >= 0` (Lasso/ElasticNet) and `l1_ratio ∈ [0,1]` (ElasticNet) checks are relocated to each builder's `build()` → `BuildError`, NOT dropped. ✓
- **T-16-GUARDF64** (F64 guard): `crate::capability::guard_f64()?` preserved verbatim before every F64 upload (9 `guard_f64` sites across the migrated + unmigrated arms in linear.rs); `lock_pool()` (poison-recovering) kept. ✓
- **T-16-ARM** (Fitted arm type): `AnyLinearRegression`/`AnyLasso`/`AnyElasticNet` switched to `any_estimator_typestate!` so each fitted value is typed `T<f*, Fitted>`. ✓

## Known Stubs

None.

## Threat Flags

None — no new network/auth/file/schema surface; a trait-surface retrofit with byte-identical compute.

## Verification Environment Note

`cargo build -p mlrs-py --features cpu` is the cross-crate PyO3 gate (live Python pytest of the wheel is untestable here per the project memory "Python wheel untestable in env"). The three oracle suites + the mlrs-py build are the compensating Rust gates; the Python-boundary behavior (Unfit-arm accessor → `not_fitted` → PyValueError) is unchanged from the pre-retrofit shells.

## Acceptance Evidence

- `cargo test --features cpu --test linear_regression_test` → **7 passed** (1e-5 oracle f64+f32, collinear cutoff, defaults_equal).
- `cargo test --features cpu --test lasso_test` → **4 passed** (sparse-coef f32, coef/intercept f64, fixture_loads, defaults_equal).
- `cargo test --features cpu --test elastic_net_test` → **4 passed** (coef f32, coef/intercept f64, fixture_loads, defaults_equal).
- `cargo build -p mlrs-py --features cpu` → Finished (2 pre-existing spectral.rs dead-code warnings only, out of scope).
- `! grep -q 'crate::traits'` on linear_regression.rs / lasso.rs / elastic_net.rs → all clean.
- `! grep -q 'fn with_opts'` on lasso.rs / elastic_net.rs → both clean (folded into builders).
- `any_estimator_typestate!` for AnyLinearRegression / AnyLasso / AnyElasticNet (linear.rs).
- D-03: per-file `git diff` shows ZERO compute-line changes (signature/return/guard-call/reconstruction only).

## Self-Check: PASSED

- `crates/mlrs-algos/src/linear/linear_regression.rs` — FOUND, builds, 7 tests pass.
- `crates/mlrs-algos/src/linear/lasso.rs` — FOUND, builds, 4 tests pass.
- `crates/mlrs-algos/src/linear/elastic_net.rs` — FOUND, builds, 4 tests pass.
- `crates/mlrs-py/src/estimators/linear.rs` — FOUND, mlrs-py builds.
- Commit `b8854a1` (LinearRegression) — FOUND.
- Commit `9b2a9ed` (Lasso) — FOUND.
- Commit `ea286a6` (ElasticNet) — FOUND.
