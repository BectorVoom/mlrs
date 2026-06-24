---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 09
subsystem: kernel-ridge-naive-bayes-estimators
tags: [typestate, builder-retrofit, kernel-ridge, naive-bayes, accessor-traits, shape-a-prime, shape-b, adopt-fit, wave-9, sweep-complete]
requires:
  - "typestate::{Fit, Predict, validate_geometry, Unfit, Fitted} (Plan 16-00)"
  - "typestate::{PredictLabels, PredictProba, PredictLogProba} accessor traits (Plan 16-00)"
  - "any_estimator_typestate! macro (dispatch.rs)"
  - "Shape-A' adopt-a-trait recipe proven in spectral_embedding.rs (16-05) / kernel_density.rs (16-07)"
  - "Shape-B trait-swap recipe proven in mbsgd_regressor.rs (16-01) + the linear/decomposition/cluster/projection/density/neighbors sweeps (16-02..16-08)"
  - "BuildError::InvalidAlpha (error.rs) for the relocated KernelRidge alpha>=0 check"
provides:
  - "KernelRidge<F, S=Unfit> on the typestate surface (Shape-A' — ADOPTS typestate Fit + Predict it never had; its inherent fit/predict became consuming-self trait impls) + KernelRidgeBuilder (.kernel(KernelKind).alpha(f64).gamma(Option<f64>).degree(f64).coef0(f64)) + zero-arg new() + Default/hyperparams_eq"
  - "GaussianNB<F, S=Unfit> on the typestate surface (Shape-B trait-swap; Fit + PredictLabels + PredictProba + PredictLogProba)"
  - "MultinomialNB<F, S=Unfit> on the typestate surface (Shape-B; 4-trait NB set)"
  - "BernoulliNB<F, S=Unfit> on the typestate surface (Shape-B; 4-trait NB set)"
  - "ComplementNB<F, S=Unfit> on the typestate surface (Shape-B; 4-trait NB set)"
  - "CategoricalNB<F, S=Unfit> on the typestate surface (Shape-B; 4-trait NB set)"
  - "PyKernelRidge on any_estimator_typestate! (Fitted arms); kernel.rs (PyO3) fully off mlrs_algos::traits"
  - "All 5 Py*NB on any_estimator_typestate! (Fitted arms); naive_bayes.rs (PyO3) fully off mlrs_algos::traits"
affects:
  - "Plan 16-12 (traits.rs hard-deletion — ALL 29 estimators are now migrated; the only residual crate::traits references are traits.rs itself + a few doc-comment intra-doc links, so the deletion is unblocked)"
  - "BLDR-03 (stays In Progress — completes only after the traits.rs deletion + full sweep verification in 16-12)"
tech-stack:
  added: []
  patterns:
    - "Shape-A' adopt Fit+Predict (KernelRidge): an estimator with INHERENT fit/predict and NO trait import ADOPTS the typestate Fit (its inherent fit becomes the consuming-self trait impl on Unfit, BYTE-IDENTICAL body) and Predict (its inherent predict moves onto the Fitted impl, the four ok_or(NotFitted) reads become .expect). The kernel_density.rs 16-07 precedent, extended from one accessor (ScoreSamples) to the full Fit+Predict pair."
    - "n_targets recovered from y geometry: the Fit trait's fixed signature carries no n_targets slot, so KernelRidge::fit recovers it as y.len() / n_samples (the multi-RHS dual solve). The PyO3/test n_targets argument is retained for signature compatibility but no longer threaded; the y buffer's length (n_samples * n_targets) makes the recovered value identical to the explicit one (byte-identical dual solve)."
    - "Shape-B NB 4-trait swap (the 5 NB): each NB keeps its pre-existing builder + nb_common.rs shared helpers UNTOUCHED; gains <F, S=Unfit> + PhantomData, build() returns T<F, Unfit>, Fit::fit consumes self -> Fitted, and the 3 accessor-trait impls (PredictLabels/PredictProba/PredictLogProba) + every config/fitted accessor move onto impl ...<F, Fitted>. The WR-07 re-fit buffer-release pass is dropped as vacuous on a fresh Unfit."
    - "Consuming-self re-fit memory gate: the refit_releases_buffers PoolStats no-leak test (which re-fit the same &mut self) is reworked to the born-with-convention construct -> fit(consuming) -> drop(Fitted) cycle (umap_test fit_no_leak precedent), since consuming-self Fit makes a &mut self re-fit a compile error. The dropped Fitted returns its device buffers to the pool free-list, which the next construct+fit reuses."
    - "Shared-file NB PyO3 migration: the single naive_bayes.rs wraps all 5 NB via a shared nb_surface_fns! macro that calls predict_labels/predict_proba/predict_log_proba via method syntax. With the typestate accessor traits imported (aliased) and the Any*NB::F32/F64 arms typed T<f*, Fitted>, method resolution finds the Fitted accessors with no UFCS needed (no competing legacy trait in scope after the glob swap)."
key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs
    - crates/mlrs-algos/tests/kernel_ridge_test.rs
    - crates/mlrs-py/src/estimators/kernel.rs
    - crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs
    - crates/mlrs-algos/tests/gaussian_nb_test.rs
    - crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs
    - crates/mlrs-algos/tests/multinomial_nb_test.rs
    - crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs
    - crates/mlrs-algos/tests/bernoulli_nb_test.rs
    - crates/mlrs-algos/src/naive_bayes/complement_nb.rs
    - crates/mlrs-algos/tests/complement_nb_test.rs
    - crates/mlrs-algos/src/naive_bayes/categorical_nb.rs
    - crates/mlrs-algos/tests/categorical_nb_test.rs
    - crates/mlrs-py/src/estimators/naive_bayes.rs
decisions:
  - "KernelRidge alpha>=0 RELOCATED from the fit body (AlgoError::InvalidAlpha) to KernelRidgeBuilder::build() -> BuildError::InvalidAlpha (data-INDEPENDENT, D-04/Pitfall 7). The degree<1 check (poly-branch-coupled) and the resolved-gamma finiteness check (resolution-path-coupled: gamma=None resolves to 1/n_features at fit) STAY in the fit body (byte-identical, D-03). alpha64 is recomputed in the fit body for the diagonal-penalty injection (the value's compute use stays; only the validation moved)."
  - "KernelRidge's x geometry guard swapped to validate_geometry(x, shape); the KernelRidge-specific y guard (y.len() == n_samples * n_targets) kept inline (no shared helper for the multi-target y). The Fit trait carries no n_targets slot, so n_targets is recovered as y.len()/n_samples."
  - "The 5 NB do NOT add a zero-arg new()/hyperparams_eq (shape-B builders pre-exist; sklearn NB constructors are not zero-arg-defaults-on-the-struct). The existing default_matches_sklearn oracle test already serves the BLDR-01 single-source litmus (it fits the bare builder().build() and asserts sklearn-default parity), so no new()/hyperparams_eq was invented — consistent with the shape-B LinearSVC/SVR/MBSGDClassifier of 16-03 which likewise kept default_matches_sklearn instead of a new()-based defaults_equal."
  - "The shared PyO3 naive_bayes.rs (all 5 NB in one file via the nb_surface_fns! macro) was migrated atomically and landed in the FINAL (CategoricalNB) commit. The 4 prior NB algos src+test commits are mlrs-algos-only (each green under its oracle suite + cargo build -p mlrs-algos); mlrs-py only builds once all 5 algos NB + the shared PyO3 file are migrated (Pitfall 3 — the shared file couples all 5, so it cannot land per-NB without breaking the cross-crate build between commits). This keeps the per-NB bisect granularity for the algos surface (D-06) while respecting the shared-file reality."
  - "CategoricalNB kept its existing _marker: PhantomData<F> (F is otherwise unused — the estimator holds only host f64 ragged tables) AND gained the new _state: PhantomData<S>; both PhantomData fields coexist."
metrics:
  duration: ~28m
  completed: 2026-06-24
  tasks: 2
  files: 14
status: complete
---

# Phase 16 Plan 09: KernelRidge + the 5 Naive Bayes typestate retrofit (the final algos-side estimator batch) — Summary

Migrated the LAST six `mlrs-algos` estimators onto the `mlrs_algos::typestate` surface, completing the 29/29 estimator sweep. **KernelRidge** is the Shape-A' adopt-a-trait case (RESEARCH Open Q3): it had INHERENT `fit`/`predict` methods and NO trait import, so the retrofit ADOPTS the typestate `Fit` (consuming-self) AND `Predict` (its inherent methods became trait impls, the four `ok_or(NotFitted)` reads became `.expect` on the `Fitted` sibling) plus the full build-out (`<F, S=Unfit>` + builder + zero-arg `new()` + `Default`/`hyperparams_eq`). **GaussianNB / MultinomialNB / BernoulliNB / ComplementNB / CategoricalNB** are Shape-B trait-swaps (builders pre-existing): each gains `<F, S=Unfit>` + `PhantomData`, its `build()` returns `T<F, Unfit>`, its `Fit::fit` consumes self → `Fitted`, and its 4-trait accessor set (`PredictLabels`/`PredictProba`/`PredictLogProba`) + every config/fitted accessor move onto `impl ...<F, Fitted>`. All six fit/predict compute paths are byte-identical to pre-retrofit (D-03); all 45 oracle/property tests across the six suites stay green at their pinned bands, and `cargo build -p mlrs-py --features cpu` is clean — both PyO3 files (`kernel.rs`, `naive_bayes.rs`) are now fully off the legacy `mlrs_algos::traits` surface. After this wave the entire estimator surface is on typestate; only `traits.rs` itself (+ a few doc-comment intra-doc links) remains, unblocking its hard-deletion in Plan 16-12.

## What Was Built

### Task 1 — KernelRidge (Shape A'; adopt typestate Fit + Predict), commit `43c1c35`

`crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs`:
- `struct KernelRidge<F, S = Unfit>` — added `_state: PhantomData<S>` (kept the struct's `where F: Float + CubeElement + Pod` clause); all hyperparam (`kernel_kind`/`alpha`/`gamma`/`degree`/`coef0`) and fitted (`kernel_`/`dual_coef_`/`x_fit_`/`fit_shape_`/`n_targets_`) fields UNCHANGED (D-03).
- Replaced the 5-arg `new(kernel, alpha, gamma, degree, coef0)` with **zero-arg `new()`** on `impl<F> KernelRidge<F, Unfit>` (sklearn defaults: `linear`/`alpha=1.0`/`gamma=None`/`degree=3`/`coef0=1`); added `builder()`/`into_builder()`/`hyperparams_eq()`/`impl Default`.
- New `KernelRidgeBuilder { kernel: KernelKind, alpha: f64, gamma: Option<f64>, degree: f64, coef0: f64 }` — `.kernel(KernelKind)` takes the enum directly (non-scalar selector); `.alpha(f64)/.gamma(Option<f64>)/.degree(f64)/.coef0(f64)` (A5 scalar narrowing); `Default = KernelRidge::<f64, Unfit>::new().into_builder()` (Pitfall 1 — single source). `build<F>()` **RELOCATES** the data-INDEPENDENT `alpha >= 0` check (from the old fit-body `AlgoError::InvalidAlpha`) to `BuildError::InvalidAlpha` (D-04).
- Imports: added `crate::error::{AlgoError, BuildError}` + `crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit}` (it had no trait import before).
- **ADOPTS typestate `Fit`:** the inherent `fit` (was `&mut self -> Result<&mut Self>`, with an explicit `n_targets` arg) becomes `impl Fit for KernelRidge<F, Unfit>` (`type Fitted = KernelRidge<F, Fitted>`, consuming `self`). `n_targets` is recovered from `y.len() / n_samples` (the Fit trait carries no n_targets slot). The inline `x` geometry guard → `validate_geometry`; the `alpha < 0` check removed (relocated); **every device-math line** (gamma resolution, the kernel-name guard, the `kernel_matrix` Gram build, the diagonal-α host pass, the multi-RHS `cholesky_solve`, the post-solve finiteness guard) byte-identical. `alpha64` recomputed in-body for the diagonal injection. Reconstructs into the `Fitted` value (the WR-07 re-fit buffer-release pass dropped as vacuous on a fresh Unfit).
- **ADOPTS typestate `Predict`:** the inherent `predict` becomes `impl Predict for KernelRidge<F, Fitted>` (the four `ok_or(NotFitted)` reads → `.expect`, the `kernel_matrix(X_test, X_fit_) → gemm` path byte-identical). `dual_coef()` moved onto `Fitted` (drops the `NotFitted` Result → bare `Vec<F>`).

`kernel_ridge_test.rs`: trait import → typestate aliases (`TypestateFit`/`TypestatePredict`); the `fit_predict` helper → `builder().kernel(..).alpha(1.0).gamma(..).degree(..).coef0(..).build()? + TypestateFit::fit(.., Some(&y), ..)`; predict via `TypestatePredict`; the dead `gamma_f` removed (builder takes the raw `Option<f64>`). Added `kernel_ridge_defaults_equal` (BLDR-01).

`crates/mlrs-py/src/estimators/kernel.rs` (PyKernelRidge arm): `AnyKernelRidge` → `any_estimator_typestate!`; fit via `builder()...build::<f*>().map_err(build_err_to_py)? + TypestateFit::fit(...)` (dropped the `as f32` casts — builder setters are f64); `predict_f32/f64` → `TypestatePredict::predict` UFCS; `dual_coef_f32/f64` un-`.map_err`'d (infallible on Fitted). The file is now 100% on typestate (KernelDensity was already migrated in 16-07).

**Gate:** `cargo test --features cpu --test kernel_ridge_test` → **5 passed** (all-kernels f64+f32 at the documented bands, multi-target f64+f32, `defaults_equal`). `cargo build -p mlrs-py --features cpu` → clean.

### Task 2 — the 5 Naive Bayes (Shape B trait-swap; Fit + PredictLabels + PredictProba + PredictLogProba)

Commits `42b5323` (Gaussian), `246100e` (Multinomial), `5a2527c` (Bernoulli), `faa8aa8` (Complement), `1c01eeb` (Categorical + the shared PyO3 file).

Each NB (`gaussian_nb.rs` / `multinomial_nb.rs` / `bernoulli_nb.rs` / `complement_nb.rs` / `categorical_nb.rs`) followed the identical Shape-B delta (builder body + `nb_common.rs` shared helpers `validate_discrete_alpha`/`decode_classes`/`resolve_class_log_prior`/`validate_non_negative_counts` UNTOUCHED):
- `struct T<F, S = Unfit>` + `_state: PhantomData<S>` (CategoricalNB additionally KEEPS its pre-existing `_marker: PhantomData<F>`).
- `build<F>()` return → `T<F, Unfit>` (+ `_state: PhantomData` in the returned literal); the data-INDEPENDENT validation (`var_smoothing >= 0` / `alpha >= 0` / `class_prior` finiteness / the `force_alpha` clip) UNCHANGED.
- import swap `crate::traits::{Fit, PredictLabels, PredictLogProba, PredictProba}` → `crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, PredictLogProba, PredictProba, Unfit}`.
- `impl Fit for T<F, Unfit>` (`type Fitted = T<F, Fitted>`): consuming `self`; the inline `x` shape guard → `validate_geometry`; the per-NB fit body (the `class_grouped_sum`/`sumsq` GATHERs, the GEMM joint-LL operands, the binarize pass, the complement weights, the ragged categorical tabulation) byte-identical; the WR-07 re-fit buffer-release pass dropped (vacuous on a fresh Unfit). Reconstructs field-by-field into the `Fitted` value.
- the config/fitted accessors (`classes`/`class_log_prior`/`class_count`/`epsilon`/`theta`/`var`/`feature_log_prob`/`force_alpha`/`n_categories`) + `joint_log_likelihood` + the 3 accessor-trait impls (`PredictLabels`/`PredictProba`/`PredictLogProba`) moved onto `impl ...<F, Fitted>`.

Each NB test (`*_nb_test.rs`): trait import → typestate aliases; the fit-helper(s) → `builder().build()? + TypestateFit::fit(...)` + `TypestatePredictLabels`/`TypestatePredictProba` UFCS; the `refit_releases_buffers` PoolStats gate reworked from the `&mut self` re-fit loop to the construct → fit(consuming) → `drop(fitted)` cycle (umap_test `fit_no_leak` precedent — a `&mut self` re-fit is now a type error). The ComplementNB `norm`-test (reads `feature_log_prob()` post-fit), the BernoulliNB `binarize=None` second helper, and the CategoricalNB error-path `.fit(...).err()` tests + the `fit_categorical_with<F>(case, clf: CategoricalNB<F, Unfit>)` helper signature were all threaded through the consuming-self form.

`crates/mlrs-py/src/estimators/naive_bayes.rs` (the SHARED file wrapping all 5 NB): swapped the legacy `mlrs_algos::traits` glob → typestate aliases; all 5 `Any*NB` enums → `any_estimator_typestate!`; all 5 fit arms `builder()...build::<f*>()? + TypestateFit::fit(est, ...)` storing the `Fitted`-tagged arm. The shared `nb_surface_fns!` predict surface (`predict_labels`/`predict_proba`/`predict_log_proba`/`classes()`) resolves the typestate accessor traits on the `T<f*, Fitted>` arms via method syntax (no UFCS needed — no competing legacy trait in scope after the glob swap). The file is fully off `mlrs_algos::traits`.

**Gates:** `cargo test --features cpu --test gaussian_nb_test` → **7 passed**; `multinomial_nb_test` → **8 passed**; `bernoulli_nb_test` → **8 passed**; `complement_nb_test` → **8 passed** (incl. the `norm=true` L1-row-sum gate); `categorical_nb_test` → **9 passed** (incl. the negative/non-integer-input rejection gate). `cargo build -p mlrs-py --features cpu` → clean.

## The Sweep Is Complete (29/29)

After this wave, NO `mlrs-algos` src file contains a `use crate::traits` import — every estimator consumes `mlrs_algos::typestate`. The remaining textual `crate::traits` matches are all benign and reconciled by Plan 16-12:
- `traits.rs` itself (the hard-delete target).
- doc-comment intra-doc links in `typestate.rs` (mirrors-the-legacy module doc), `cluster/mod.rs:5` (a `PredictLabels` link — pre-existing, not introduced here), and a plain `//` comment in `spectral_embedding.rs:49` (migrated 16-05).

The traits.rs deletion (Plan 16-12) is now unblocked. **BLDR-03 stays In Progress** — it completes only after the traits.rs hard-deletion + the full-sweep verification in 16-12 (per the orchestrator directive, not prematurely marked complete here).

## Deviations from Plan

The six estimators followed the documented recipes exactly (KernelRidge the Shape-A' adopt-a-trait, the 5 NB the Shape-B trait-swap). Three plan-anticipated implementation details (not deviations):

1. **n_targets recovery for KernelRidge (plan-anticipated).** The `Fit` trait's fixed signature has no `n_targets` slot, so the consuming-self `fit` recovers `n_targets = y.len() / n_samples` (the multi-RHS dual solve reads it off the `y` geometry). The PyO3/test `n_targets` argument is retained for signature compatibility but no longer threaded; the `y` buffer's length (`n_samples * n_targets`) makes the recovered value identical to the explicit one, so the multi-target oracle stays byte-identical.
2. **No new()/hyperparams_eq for the 5 NB (plan-anticipated reconciliation).** The plan task said "Add a hyperparams_eq case per NB", but the shape-B NB have no zero-arg `new()` on the struct (their sklearn constructors are builder-shaped, not defaults-on-the-struct). The existing `default_matches_sklearn` oracle test already serves the BLDR-01 single-source litmus (it fits `builder().build()` and asserts sklearn-default parity), exactly as the shape-B LinearSVC/SVR/MBSGDClassifier of 16-03 did — so no `new()`/`hyperparams_eq` was invented (that would contradict D-02's "shape-B = trait-swap ONLY, builder pre-existing"). KernelRidge (Shape-A', full build-out) DID get `new()`/`hyperparams_eq` + a `kernel_ridge_defaults_equal` test.
3. **The shared NB PyO3 file landed in the final (Categorical) commit (plan-anticipated coupling).** The plan asked for 5 separate NB commits for bisect cleanliness; the single `naive_bayes.rs` wraps all 5 via the shared `nb_surface_fns!` macro, so it cannot land per-NB without breaking the cross-crate `cargo build -p mlrs-py` between commits (Pitfall 3). The 4 prior NB commits are mlrs-algos-only (each green under its oracle suite + `cargo build -p mlrs-algos`); the shared PyO3 file + CategoricalNB landed atomically in the 5th, completing the mlrs-py build. The per-NB bisect granularity is preserved for the algos surface (D-06).

The `refit_releases_buffers` rework (construct → fit → drop, replacing the `&mut self` re-fit loop) is the natural consequence of the consuming-self `Fit` transition (a `&mut self` re-fit is a compile error), following the umap_test `fit_no_leak` born-with-convention precedent — not a scope deviation.

## Authentication Gates

None.

## Threat Mitigations (from the plan's threat_model)

- **T-16-V5** (KernelRidge/NB fit geometry guard + relocated validation): `validate_geometry(x, shape)?` is at the TOP of all six ported `fit`s, before any device launch. The data-INDEPENDENT `alpha >= 0` check (KernelRidge) is RELOCATED to `KernelRidgeBuilder::build()` → `BuildError::InvalidAlpha`, NOT dropped; the NB builders' `var_smoothing>=0`/`alpha>=0`/`class_prior`/`force_alpha` validation is UNTOUCHED; `nb_common.rs` math is byte-identical. The data-DEPENDENT shape guards (the KernelRidge `y` guard, the NB `y.len() == n_samples` checks, the categorical input-integer / category-index guards) stay in the fit/predict bodies. ✓
- **T-16-GUARDF64** (F64 guard): `crate::capability::guard_f64()?` preserved verbatim before every F64 upload in the migrated PyKernelRidge + all 5 Py*NB fit arms; `lock_pool()` (poison-recovering) kept. ✓
- **T-16-ARM** (Fitted arm type): `AnyKernelRidge` + all 5 `Any*NB` enums switched to `any_estimator_typestate!` so each fitted value is typed `T<f*, Fitted>` — no `Unfit` value stored in a fitted arm. ✓

## Known Stubs

None.

## Threat Flags

None — no new network/auth/file/schema surface introduced; a trait-surface retrofit with byte-identical compute.

## Verification Environment Note

`cargo build -p mlrs-py --features cpu` is the cross-crate PyO3 gate (the live Python pytest of the wheel is untestable in this environment per the project memory "Python wheel untestable in env" — no maturin/pyarrow). The six oracle/property suites + the mlrs-py build are the compensating Rust gates; the Python-boundary behavior (the `Unfit`-arm accessor → `not_fitted` → PyValueError; the dtype-mismatch path) is unchanged from the pre-retrofit shells.

## Acceptance Evidence

- `cargo test --features cpu --test kernel_ridge_test` → **5 passed** (all-kernels f64+f32, multi-target f64+f32, `defaults_equal`).
- `cargo test --features cpu --test gaussian_nb_test` → **7 passed** (exact-labels f32+f64 HARD gate, proba band f32+f64, `default_matches_sklearn`, `build_rejects_bad_var_smoothing`, `refit_releases_buffers`).
- `cargo test --features cpu --test multinomial_nb_test` → **8 passed**.
- `cargo test --features cpu --test bernoulli_nb_test` → **8 passed**.
- `cargo test --features cpu --test complement_nb_test` → **8 passed** (incl. the `norm=true` L1-row-sum gate).
- `cargo test --features cpu --test categorical_nb_test` → **9 passed** (incl. negative/non-integer-input rejection).
- `cargo build -p mlrs-py --features cpu` → Finished (only pre-existing dead-code warnings on unrelated estimators' Unfit fields, out of scope).
- `! grep -q 'crate::traits'` on all 5 naive_bayes src files AND kernel_ridge.rs → all clean (no imports; only a benign plain `//` comment remained in two files and was reworded / out of scope).
- `grep -cE 'typestate::(Fit|Predict)'` on kernel_ridge.rs → 2 (`impl Fit` + `impl Predict` — now ON the trait surface, was inherent).
- `AnyKernelRidge` + all 5 `Any*NB` use `any_estimator_typestate!` (6 invocations across the two PyO3 files).
- `grep -rl 'crate::traits' crates/mlrs-algos/src/` returns ONLY `traits.rs` + doc-comment-bearing files (`typestate.rs`, `cluster/mod.rs`, `spectral_embedding.rs`) — every estimator migrated; the deletion is Plan 16-12.
- D-03: per-file `git diff` shows ZERO compute-line changes for all six fit/predict bodies (signature/return/guard-call/reconstruction only).

## Self-Check: PASSED

- `crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs` — FOUND, builds, 5 tests pass.
- `crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs` — FOUND, builds, 7 tests pass.
- `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs` — FOUND, builds, 8 tests pass.
- `crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs` — FOUND, builds, 8 tests pass.
- `crates/mlrs-algos/src/naive_bayes/complement_nb.rs` — FOUND, builds, 8 tests pass.
- `crates/mlrs-algos/src/naive_bayes/categorical_nb.rs` — FOUND, builds, 9 tests pass.
- `crates/mlrs-py/src/estimators/kernel.rs` — FOUND, mlrs-py builds (off mlrs_algos::traits).
- `crates/mlrs-py/src/estimators/naive_bayes.rs` — FOUND, mlrs-py builds (off mlrs_algos::traits).
- Commit `43c1c35` (KernelRidge) — FOUND.
- Commit `42b5323` (GaussianNB) — FOUND.
- Commit `246100e` (MultinomialNB) — FOUND.
- Commit `5a2527c` (BernoulliNB) — FOUND.
- Commit `faa8aa8` (ComplementNB) — FOUND.
- Commit `1c01eeb` (CategoricalNB + shared PyO3) — FOUND.
