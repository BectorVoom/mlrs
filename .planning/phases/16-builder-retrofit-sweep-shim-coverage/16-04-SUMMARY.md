---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 04
subsystem: decomposition-estimators
tags: [typestate, builder-retrofit, decomposition, transformer, partial-fit, inverse-transform, wave-4]
requires:
  - "typestate::{Fit, Transform (incl. inverse_transform default), PartialFit, Fitted, Unfit} (Plan 16-00)"
  - "any_estimator_typestate! macro (dispatch.rs)"
  - "Shape-A recipe proven in ridge.rs (16-01) + linear sweep (16-02/16-03)"
provides:
  - "Pca<F, S=Unfit> + PcaBuilder (.n_components) + Fitted Transform incl. inverse_transform override"
  - "TruncatedSvd<F, S=Unfit> + TruncatedSvdBuilder (.n_components) + Fitted Transform (inverse_transform = Unsupported default)"
  - "IncrementalPCA<F, S=Unfit> + IncrementalPcaBuilder (.n_components/.whiten/.batch_size) + PartialFit on BOTH Unfit and Fitted + Fitted Transform"
  - "PyPCA / PyTruncatedSVD / PyIncrementalPCA on any_estimator_typestate! (Fitted arms)"
affects:
  - "Plans 16-05..16-08 (bulk sweep continues; PartialFit multi-transition + inverse_transform override patterns now proven)"
  - "Plan 16-11 (traits.rs deletion — the decomposition module + its PyO3 file no longer reference crate::traits)"
tech-stack:
  added: []
  patterns:
    - "Data-DEPENDENT n_components bound (1..=min(n_samples,n_features)) stays in fit -> AlgoError::InvalidNComponents; the builder build() is infallible-but-typed (Result<_, BuildError> that never errs) for family uniformity, same shape as LinearRegression's OLS build()"
    - "Transform::inverse_transform: PCA OVERRIDES the typestate default (reconstruction body ported verbatim onto Fitted); TruncatedSVD/IncrementalPCA-no-recon leave the default -> AlgoError::Unsupported (PCA is the first transformer to exercise the Plan-00 default override)"
    - "PartialFit multi-transition (Pitfall 5): impl on BOTH <F, Unfit> (type Fitted = <F, Fitted>, first batch) AND <F, Fitted> (type Fitted = Self, subsequent batches) via a shared consuming merge_batch helper; Fit::fit threads the stream through the transition (first batch consumes Unfit, the rest consume Fitted)"
    - "Consuming-self PyO3 partial_fit: mem::replace the arm out behind an Unfit placeholder, run the consuming partial_fit, store the next state; on a dtype-mismatch / ingress error the unconsumed prior arm is returned alongside the PyErr (Result<Any, (PyErr, Any)>) and restored, preserving the pre-retrofit 'mismatched batch leaves the fitted estimator intact' semantics"
key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/decomposition/pca.rs
    - crates/mlrs-algos/tests/pca_test.rs
    - crates/mlrs-algos/src/decomposition/truncated_svd.rs
    - crates/mlrs-algos/tests/truncated_svd_test.rs
    - crates/mlrs-algos/src/decomposition/incremental_pca.rs
    - crates/mlrs-algos/tests/incremental_pca_test.rs
    - crates/mlrs-py/src/estimators/decomposition.rs
decisions:
  - "PCA/TruncatedSvd/IncrementalPCA n_components bound is data-DEPENDENT (min(n_samples,n_features)), so it CANNOT move to build(); it stays in fit (AlgoError::InvalidNComponents). The builders' build() are therefore infallible-but-typed (kept Result<_, BuildError> for PyO3 build_err_to_py shape-uniformity across the family — same call as LinearRegression's OLS build())"
  - "Zero-arg new() default n_components = 2 (PCA/TruncatedSvd/IncrementalPCA): the pre-retrofit new(n_components) had NO natural default; 2 mirrors PyTruncatedSVD's #[new] default and PyPCA's smoke-test default. The default is only used when no .n_components() setter is called; every oracle test sets it explicitly, so fit math is unaffected (D-03)"
  - "IncrementalPCA carries TWO PhantomData fields: the pre-existing _marker: PhantomData<F> (the estimator is generic over upload precision F while running stats stay f64) PLUS the new _state: PhantomData<S> (the typestate marker). The state getters n_components/whiten/batch_size/n_samples_seen are state-generic (impl<F,S>) because the PyO3 re-fit path reads them off a Fitted arm"
  - "PyIncrementalPCA partial_fit moves the dispatch enum arm out via std::mem::replace behind an Unfit placeholder (the consuming typestate partial_fit takes the estimator by value). The dispatch returns Result<Any, (PyErr, Any)> so a dtype-mismatch or ingress error restores the prior arm intact — the merge-error path (est already consumed) leaves the Unfit placeholder, matching that the partial result is unrecoverable"
metrics:
  duration: ~17m
  completed: 2026-06-24
  tasks: 3
  files: 7
status: complete
---

# Phase 16 Plan 04: decomposition sweep — PCA + TruncatedSVD + IncrementalPCA typestate retrofit — Summary

Migrated the three `decomposition/` estimators onto the `mlrs_algos::typestate` surface, each under its own commit gated by its sklearn oracle suite AND `cargo build -p mlrs-py --features cpu`. **PCA** is the first transformer to exercise the Plan-00 `Transform::inverse_transform` default override (its reconstruction body is ported verbatim onto `Fitted`); **TruncatedSVD** is the plain shape-A transformer (leaves the `Unsupported` default); **IncrementalPCA** is the ONLY `PartialFit` consumer — the multi-transition typestate case (Pitfall 5), with `PartialFit` impl'd on BOTH `Unfit` (first batch) and `Fitted` (subsequent batches) for `Fitted → Fitted` streaming. All three fit/merge compute paths are byte-identical to pre-retrofit (D-03), and all three oracle suites stay green (33 tests: 11 + 7 + 15) at 1e-5 (f64) / the pinned f32 bands. The decomposition module is complete; its PyO3 file (`mlrs-py/src/estimators/decomposition.rs`) no longer references `mlrs_algos::traits`.

## What Was Built

### Task 1 — PCA (Shape A, transformer; inverse_transform override), commit `1b79580`

`crates/mlrs-algos/src/decomposition/pca.rs`:
- `struct Pca<F, S = Unfit>` — added `_state: PhantomData<S>`; all six fitted/hyperparam fields unchanged (D-03).
- Zero-arg `new()` (`n_components = 2`) on `impl<F> Pca<F, Unfit>`; `builder()`/`into_builder()`/`hyperparams_eq()`/`impl Default`.
- `PcaBuilder { n_components: usize }` with `.n_components(usize)`; `Default = Pca::<f64, Unfit>::new().into_builder()`; `build<F>() -> Result<_, BuildError>` **infallible-but-typed** (PCA's `n_components` bound is `1..=min(n_samples,n_features)` — data-DEPENDENT — so it stays in `fit`; the `Result` is kept for family uniformity).
- Imports: dropped `use crate::traits::{Fit, Transform}`; added `crate::error::{AlgoError, BuildError}` + typestate surface.
- `impl Fit for Pca<F, Unfit>` (consuming `self -> Pca<F, Fitted>`): every compute line (`column_reduce` mean, host centering, thin `svd`, `S²/(n−1)` / total-var / ratio, `align_rows` svd_flip, truncation) byte-identical; reconstructs into the `Fitted` value.
- Accessors + `impl Transform` moved onto `Pca<F, Fitted>` — accessors drop `ok_or(NotFitted)` → `.expect(...)` and return bare `Vec<F>`. **`inverse_transform` is OVERRIDDEN** with PCA's reconstruction body ported verbatim (the typestate default returns `Unsupported`; PCA is the first transformer to override it — Plan-00's whole reason).

`pca_test.rs`: trait import → typestate; fit call site → `builder().n_components(nc).build::<F>()?.fit(...)`; accessors un-`.expect()`'d; added `pca_defaults_equal` (BLDR-01).

`mlrs-py/src/estimators/decomposition.rs` (PyPCA arm): `AnyPca` → `any_estimator_typestate!`; fit via `builder().n_components(..).build::<f*>().map_err(build_err_to_py)?` then `TypestateFit::fit(...)`; `transform`/`inverse_transform` via `TypestateTransform::{transform,inverse_transform}` UFCS; accessors un-`.map_err`'d; `lock_pool()`. The TruncatedSVD/IncrementalPCA arms kept the legacy `traits` glob at this commit (mid-migration, per the sequential-execution recipe).

**Gate:** `cargo test --features cpu --test pca_test` → **11 passed** (tall + wide oracle f32/f64, transform, inverse_transform, `defaults_equal`). `cargo build -p mlrs-py --features cpu` → clean.

### Task 2 — TruncatedSVD (Shape A, transformer), commit `6a17d52`

`crates/mlrs-algos/src/decomposition/truncated_svd.rs`: the same shape-A build-out as PCA (struct `<F, S = Unfit>` + `PhantomData`, zero-arg `new()` (`n_components = 2`), `TruncatedSvdBuilder.n_components(usize)`, infallible-but-typed `build()`). Consuming-self `Fit::fit` (BYTE-IDENTICAL uncentered-SVD math: thin `svd`, `align_rows`, the var-of-transform-columns `explained_variance_` Pitfall-2 difference, the original-X total-variance ratio denominator). `Transform` + accessors moved onto `Fitted`. **NO `inverse_transform` override** — TruncatedSVD has no reconstruction, so it leaves the typestate `Unsupported` default (only PCA reconstructs in v1, D-01).

`truncated_svd_test.rs`: trait → typestate; builder + consuming-self fit; accessors un-`.expect()`'d; `truncated_svd_defaults_equal` added.

PyTruncatedSVD arm: `AnyTruncatedSvd` → `any_estimator_typestate!`; fit via builder + `TypestateFit::fit`; transform via `TypestateTransform::transform`; accessors un-`.map_err`'d; `lock_pool()`. The IncrementalPCA arm kept the legacy `traits` glob at this commit.

**Gate:** `cargo test --features cpu --test truncated_svd_test` → **7 passed** (components/singular_values f32/f64, explained_variance, transform, `defaults_equal`). `cargo build -p mlrs-py --features cpu` → clean.

### Task 3 — IncrementalPCA (Shape A; PartialFit multi-transition — Pitfall 5), commit `9f88ef9`

`crates/mlrs-algos/src/decomposition/incremental_pca.rs`:
- `struct IncrementalPCA<F, S = Unfit>` — added `_state: PhantomData<S>` ALONGSIDE the pre-existing `_marker: PhantomData<F>` (the estimator is generic over upload precision `F` while the running stats stay `f64`).
- Zero-arg `new()` (`n_components = 2`, `whiten = false`, `batch_size = None`); `IncrementalPcaBuilder` with `.n_components(usize).whiten(bool).batch_size(Option<usize>)`; infallible-but-typed `build()` (the `n_components` bound is data-DEPENDENT and the `batch_size >= 1` check is on the `Some(bs)` form — both stay in `fit`).
- **`PartialFit` impl'd on BOTH states (Pitfall 5):** `impl PartialFit for IncrementalPCA<F, Unfit>` (`type Fitted = IncrementalPCA<F, Fitted>`, first batch) AND `impl PartialFit for IncrementalPCA<F, Fitted>` (`type Fitted = Self`, subsequent batches). Both delegate to a shared consuming `merge_batch` helper whose ONLY compute delta vs pre-retrofit is `self.state.take()` → `self.state` (the owned move; `merge::<F>` receives the same `Option<IncrementalSvdState>` — mathematically identical).
- Consuming-self `Fit::fit` threads the stream through the transition: the first `gen_batches` slice consumes the `Unfit` self (`Unfit → Fitted`), every subsequent slice consumes the `Fitted` value (`Fitted → Fitted`). The pre-retrofit `self.state = None; self.n_features = 0` reset is now a no-op (a freshly-built `Unfit` carries no state), so it is dropped. `gen_batches` / the per-batch upload / the validation lines are byte-identical.
- `Transform` (transform + inverse_transform with whitening) + the fitted accessors + `whiten_scales` moved onto `Fitted` (accessors drop `NotFitted` → `.expect` via `fitted_state()`). `n_components`/`whiten`/`batch_size`/`n_samples_seen` getters and `validate_batch` are **state-generic** (`impl<F, S>`) because the PyO3 re-fit path reads the getters off a `Fitted` arm.

`incremental_pca_test.rs`: trait import → typestate; the `partial_fit` stream and the `n_samples_seen` accumulation test rebind across the consuming transition (first batch `Unfit → Fitted`, then a `Fitted → Fitted` loop); `fit_via_fit` uses builder + consuming-self fit; `collect_fit` takes `&IncrementalPCA<F, Fitted>`; accessors un-`.expect()`'d; added `incremental_pca_defaults_equal`.

PyIncrementalPCA arm: `AnyIncrementalPCA` → `any_estimator_typestate!`; `fit` + the first-batch `partial_fit` via builder + `TypestateFit::fit` / `TypestatePartialFit::partial_fit`. The streaming `partial_fit` **moves the dispatch arm out via `std::mem::replace`** behind an `Unfit` placeholder (the consuming typestate `partial_fit` takes the estimator by value), then threads the consuming transition; the dispatch returns `Result<AnyIncrementalPCA, (PyErr, AnyIncrementalPCA)>` so a dtype-mismatch / ingress / guard error **restores the prior arm intact** (preserving the pre-retrofit "mismatched batch leaves the fitted estimator untouched" semantics). transform/inverse/accessors → UFCS + un-`.map_err`'d; **the legacy `traits` glob was removed** (decomposition module complete).

**Gate:** `cargo test --features cpu --test incremental_pca_test` → **15 passed** (explicit partial_fit stream + one-shot fit, whiten on/off, multi-batch `n_samples_seen` accumulation, explained_variance_ratio, transform/inverse_transform, `defaults_equal`). `cargo build -p mlrs-py --features cpu` → clean.

## Deviations from Plan

None — all three estimators followed the Shape-A recipe (with the documented inverse_transform-override for PCA and the PartialFit multi-transition for IncrementalPCA) exactly as the plan's Task actions specified. Two plan-anticipated implementation details, both called out in the plan/decisions, not deviations:

1. **Infallible-but-typed `build()` for all three** — the plan's Task-1 action ("if n_components has no natural default ... do not invent") + the data-DEPENDENT `n_components` bound mean the bound CANNOT move to `build()`; it stays in `fit` (`AlgoError::InvalidNComponents`), exactly like LinearRegression's OLS `build()` in 16-02. The `Result<_, BuildError>` is kept for `build_err_to_py` family-uniformity.
2. **Zero-arg `new()` default `n_components = 2`** — the plan instructed "default per current new(); if no natural default, mirror sklearn's None/all-components behavior the current code uses; do not invent." The pre-retrofit `new(n_components)` had NO None/all-components code path (it required and validated a concrete `usize`), so inventing one would CHANGE behavior. `2` (matching the existing PyTruncatedSVD `#[new]` default and PyPCA smoke-test default) is the minimal zero-arg default; it is only used when no `.n_components()` setter is called, which never happens in any oracle test — fit math is unaffected (D-03).

## Authentication Gates

None.

## Threat Mitigations (from the plan's threat_model)

- **T-16-V5** (geometry guard / relocated validation): every ported `fit`/`partial_fit` keeps the data-DEPENDENT geometry guard at the TOP, before any device launch. PCA/TruncatedSvd retain their inline `(n_samples ≤ 1)` / geometry / `n_components`-range guards verbatim; IncrementalPCA keeps `validate_batch` (geometry + n_features-agreement + per-batch `n_components ≤ min(b,p)`) on EVERY transition (Unfit and Fitted, via the shared `merge_batch`) and the `fit`-level geometry + `n_components` + `batch_size >= 1` + `n_components > batch_size` guards. No validation was dropped in the move; the `n_components` bound stayed in `fit` (it is data-DEPENDENT). ✓
- **T-16-GUARDF64** (F64 guard): `crate::capability::guard_f64()?` preserved verbatim before every F64 upload in the migrated PyPCA / PyTruncatedSVD / PyIncrementalPCA fit/partial_fit arms (including both the first-batch and subsequent-batch F64 partial_fit paths). ✓
- **T-16-ARM** (Fitted arm type + IncrementalPCA Fitted→Fitted): `AnyPca`/`AnyTruncatedSvd`/`AnyIncrementalPCA` switched to `any_estimator_typestate!` so each fitted value is typed `T<f*, Fitted>`. IncrementalPCA's `partial_fit` on `Fitted` returns `Self` (not a fresh `Unfit`), so a stream of `partial_fit` calls stays in `Fitted` and accumulates running state. ✓

## Known Stubs

None.

## Threat Flags

None — no new network/auth/file/schema surface; a trait-surface retrofit with byte-identical compute.

## Verification Environment Note

`cargo build -p mlrs-py --features cpu` is the cross-crate PyO3 gate (live Python pytest of the wheel is untestable here per the project memory "Python wheel untestable in env"). The three oracle suites + the mlrs-py build are the compensating Rust gates; the Python-boundary streaming behavior (partial_fit state threading, dtype-mismatch restoration) is exercised at the Rust trait level by `incremental_pca_test.rs`'s explicit `partial_fit` stream + `n_samples_seen` accumulation tests.

## Acceptance Evidence

- `cargo test --features cpu --test pca_test` → **11 passed** (tall/wide oracle f32+f64, transform, inverse_transform, defaults_equal).
- `cargo test --features cpu --test truncated_svd_test` → **7 passed** (components/singular_values f32+f64, explained_variance, transform, defaults_equal).
- `cargo test --features cpu --test incremental_pca_test` → **15 passed** (partial_fit stream + one-shot fit, whiten on/off, n_samples_seen accumulation, ev_ratio, transform/inverse, defaults_equal).
- `cargo build -p mlrs-py --features cpu` → Finished (only 2 pre-existing spectral.rs dead-code warnings, out of scope).
- `! grep -q 'crate::traits'` on pca.rs / truncated_svd.rs / incremental_pca.rs → all clean.
- `grep -c 'fn inverse_transform' pca.rs` → 1 (override present on Fitted).
- `grep -cE 'PartialFit.*IncrementalPCA<F, (Unfit|Fitted)>'` on incremental_pca.rs → **2** (impl'd on BOTH states — Pitfall 5).
- `! grep -q 'mlrs_algos::traits'` on the PyO3 decomposition.rs → clean (decomposition module complete; the legacy glob is gone).
- D-03: per-file `git diff` shows ZERO compute-line changes for PCA/TruncatedSvd; the ONLY IncrementalPCA delta is `self.state.take()` → `self.state` (the consuming-self move — same `merge::<F>` argument).

## For Downstream Plans

- **PartialFit multi-transition pattern (for any future streaming estimator, e.g. MBSGD* if it adopts streaming):** impl `PartialFit` on BOTH `<F, Unfit>` (`type Fitted = <F, Fitted>`) and `<F, Fitted>` (`type Fitted = Self`) via a shared consuming `merge_batch`-style helper; thread the one-shot `Fit::fit` through the transition (first batch from `Unfit`, rest from `Fitted`).
- **Consuming-self PyO3 streaming arm:** `std::mem::replace` the dispatch enum arm out behind an `Unfit` placeholder, run the consuming `partial_fit`, store the next state; return `Result<Any, (PyErr, Any)>` so error paths can restore the prior arm.
- **inverse_transform override:** a transformer WITH a reconstruction overrides `Transform::inverse_transform` on its `Fitted` impl (body ported verbatim); a transformer WITHOUT one leaves the typestate `Unsupported` default — no override needed.
- **Data-DEPENDENT bounds stay in fit:** any hyperparameter bound that depends on the data shape (here `n_components ≤ min(n_samples, n_features)`) MUST stay in `fit` → `AlgoError`; the builder `build()` is infallible-but-typed (`Result<_, BuildError>`) for PyO3 mapper uniformity.

## Self-Check: PASSED

- `crates/mlrs-algos/src/decomposition/pca.rs` — FOUND, modified, builds, 11 tests pass.
- `crates/mlrs-algos/src/decomposition/truncated_svd.rs` — FOUND, modified, builds, 7 tests pass.
- `crates/mlrs-algos/src/decomposition/incremental_pca.rs` — FOUND, modified, builds, 15 tests pass.
- `crates/mlrs-py/src/estimators/decomposition.rs` — FOUND, modified, mlrs-py builds.
- Commit `1b79580` (PCA) — FOUND.
- Commit `6a17d52` (TruncatedSVD) — FOUND.
- Commit `9f88ef9` (IncrementalPCA) — FOUND.
