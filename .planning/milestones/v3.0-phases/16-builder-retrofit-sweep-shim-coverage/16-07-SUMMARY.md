---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 07
subsystem: projection-density-estimators
tags: [typestate, builder-retrofit, projection, density, transformer, score-samples, enum-setter, shape-a-prime, wave-7]
requires:
  - "typestate::{Fit, Transform, ScoreSamples, validate_geometry, Unfit, Fitted} (Plan 16-00; ScoreSamples added in 16-00)"
  - "any_estimator_typestate! macro (dispatch.rs)"
  - "Shape-A recipe proven in ridge.rs (16-01) + linear/decomposition/cluster/covariance sweeps (16-02..16-06)"
  - "Shape-A' adopt-a-trait recipe proven in spectral_embedding.rs (16-05)"
provides:
  - "GaussianRandomProjection<F, S=Unfit> + GaussianRandomProjectionBuilder (.n_components(NComponents).seed(u64).eps(f64)) + Fitted Transform + components/n_components_ accessors"
  - "SparseRandomProjection<F, S=Unfit> + SparseRandomProjectionBuilder (+ .density(Option<f64>)) + Fitted Transform + accessors"
  - "KernelDensity<F, S=Unfit> + KernelDensityBuilder (.kernel(KdKernel).bandwidth(BandwidthSpec)) + ADOPTED typestate Fit + Fitted-gated ScoreSamples + bandwidth() accessor"
  - "PyGaussianRandomProjection / PySparseRandomProjection / PyKernelDensity on any_estimator_typestate! (Fitted arms)"
affects:
  - "Plans 16-08..16-11 (bulk sweep continues; the enum-typed builder setter + Shape-A' ScoreSamples-only adoption patterns now proven)"
  - "Plan 16-12 (traits.rs deletion — the projection + density modules and their PyO3 files no longer reference crate::traits/mlrs_algos::traits)"
tech-stack:
  added: []
  patterns:
    - "Enum-typed builder setter: a non-scalar hyperparameter selector (NComponents 'auto'/fixed; KdKernel; BandwidthSpec) is taken by the builder setter DIRECTLY (no f64 narrowing — A5 covers only scalar narrowing). The setter signature is `fn n_components(self, v: NComponents)`, not `f64`."
    - "Infallible-but-typed build() when ALL hyperparameter validation is resolution-path-coupled or data-DEPENDENT: projection eps∈(0,1)/density∈(0,1] are coupled to the Auto/Fixed/None fit-resolution paths, and n_components<1 is data-DEPENDENT (vs n_features), so NONE move to build(); KernelDensity's bandwidth>0 is coupled to scott/silverman fit-resolution. All stay in fit (AlgoError); build() returns Result<_, BuildError> that never errs, for build_err_to_py family uniformity (same as SpectralEmbedding 16-05 / LinearRegression OLS 16-02)."
    - "Shape-A' ScoreSamples-only adoption (KernelDensity): an estimator with an INHERENT fit + an OLD legacy-traits accessor impl (no Fit trait) ADOPTS typestate Fit (its inherent fit becomes the consuming-self trait impl on Unfit) and MOVES its accessor onto the Fitted impl using the typestate accessor trait. The no-op re-fit buffer release (self.x_fit_.take()) is dropped — a freshly-built Unfit carries no fitted state (mirrors IncrementalPCA's dropped reset, 16-04)."
key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/projection/gaussian.rs
    - crates/mlrs-algos/src/projection/sparse.rs
    - crates/mlrs-algos/src/density/kernel_density.rs
    - crates/mlrs-algos/src/density/mod.rs
    - crates/mlrs-algos/tests/random_projection_test.rs
    - crates/mlrs-algos/tests/kernel_density_test.rs
    - crates/mlrs-py/src/estimators/projection.rs
    - crates/mlrs-py/src/estimators/kernel.rs
decisions:
  - "GaussianRandomProjection/SparseRandomProjection/KernelDensity build() are all infallible-but-typed (Result<_, BuildError> that never errs). Every data-INDEPENDENT-looking guard is actually resolution-path-coupled (eps/density/bandwidth resolve at fit against n_samples/n_features via Auto/None/scott/silverman) or data-DEPENDENT (n_components<1 vs n_features), so nothing can move to build() without changing behaviour. The Result is kept for build_err_to_py PyO3 mapper shape-uniformity across the Phase-16 family (the SpectralEmbedding 16-05 precedent)."
  - "The builder n_components setter takes the NComponents enum DIRECTLY (not f64). A5's f64-setter convention covers scalar narrowing only; NComponents (Auto/Fixed selector), KdKernel, and BandwidthSpec are non-scalar selectors taken verbatim. The PyO3 wrapper keeps its Option<usize> sentinel and maps it to NComponents via the existing resolve_n_components helper, then feeds the builder."
  - "Zero-arg new() defaults: Gaussian/Sparse = (NComponents::Auto, seed=0, eps=0.1[, density=None]) matching the pre-retrofit PyO3 unfit_default(); KernelDensity = (KdKernel::Gaussian, BandwidthSpec::Numeric(1.0)) matching sklearn's KernelDensity(kernel='gaussian', bandwidth=1.0) default. Every oracle/property test sets its hyperparameters explicitly, so the new() default is exercised only by the BLDR-01 defaults_equal test — fit math is unaffected (D-03)."
  - "KernelDensity is Shape-A' (RESEARCH Open Q3/A3): it had an INHERENT fit + a legacy-traits ScoreSamples impl and NO Fit trait. The retrofit ADOPTS typestate Fit (the inherent fit's body is ported byte-identical onto impl Fit for KernelDensity<F, Unfit>) and moves ScoreSamples onto impl ScoreSamples for KernelDensity<F, Fitted> using crate::typestate::ScoreSamples. It now sits fully on the single trait surface (it gained a Fit it never had + the typestate ScoreSamples)."
  - "The KernelDensity inline geometry guard (n_samples==0 || n_features==0 || x.len()!=n*p → ShapeMismatch) was swapped for the shared validate_geometry (identical semantics) — both because it is the typestate-family convention (SpectralEmbedding 16-05) and to consume the imported symbol. This is a guard relocation, not a compute change (D-03)."
metrics:
  duration: ~12m
  completed: 2026-06-24
  tasks: 3
  files: 8
status: complete
---

# Phase 16 Plan 07: projection + density sweep — GaussianRandomProjection + SparseRandomProjection + KernelDensity typestate retrofit — Summary

Migrated the two `projection/` transformers and the `density/` KernelDensity estimator onto the `mlrs_algos::typestate` surface, each under its own commit gated by its sklearn oracle/property suite AND `cargo build -p mlrs-py --features cpu`. **GaussianRandomProjection** and **SparseRandomProjection** are Shape-A transformers whose builder `n_components` setter takes the `NComponents` enum DIRECTLY (the first enum-typed builder setter in the sweep — A5's f64 convention covers scalar narrowing only). **KernelDensity** is the Shape-A' adopt-a-trait case (RESEARCH Open Q3/A3): it had an INHERENT `fit` plus an OLD legacy-`traits` `ScoreSamples` impl and NO `Fit` trait, so the retrofit ADOPTS the typestate `Fit` (its inherent fit becomes the consuming-self trait impl on `Unfit`) and moves `ScoreSamples` onto the `Fitted` impl gated on the typestate accessor trait. All three fit/score compute paths are byte-identical to pre-retrofit (D-03); all 16 oracle/property tests stay green (10 projection + 6 KernelDensity) at their pinned bands. The projection and density modules are complete; neither their estimator files nor their PyO3 files reference `crate::traits`/`mlrs_algos::traits`.

## What Was Built

### Task 1 — GaussianRandomProjection (Shape A, transformer; NComponents enum setter), commit `1efd541`

`crates/mlrs-algos/src/projection/gaussian.rs`:
- `struct GaussianRandomProjection<F, S = Unfit>` — added `_state: PhantomData<S>`; all hyperparam (`n_components: NComponents`, `seed`, `eps`) and fitted (`components_`, `n_components_`, `n_features`) fields unchanged (D-03).
- Zero-arg `new()` (`NComponents::Auto`, `seed = 0`, `eps = 0.1` — matching the pre-retrofit PyO3 `unfit_default()`) on `impl<F> GaussianRandomProjection<F, Unfit>`; `builder()`/`into_builder()`/`hyperparams_eq()`/`impl Default`.
- `GaussianRandomProjectionBuilder { n_components: NComponents, seed: u64, eps: f64 }` — **`.n_components(NComponents)` takes the enum directly**, `.seed(u64)`, `.eps(f64)`; `build<F>()` **infallible-but-typed** (eps∈(0,1) is resolution-path-coupled — the `Auto` path resolves it via `johnson_lindenstrauss_min_dim` against the fit `n_samples`, the `Fixed` path validates inline — and `n_components<1` is data-DEPENDENT vs `n_features`, so both stay in `fit`).
- Imports: dropped `use crate::traits::{Fit, Transform}`; added `crate::error::{AlgoError, BuildError}` + the typestate surface (`validate_geometry, Fit, Fitted, Transform, Unfit`).
- `impl Fit for GaussianRandomProjection<F, Unfit>` (consuming `self -> GaussianRandomProjection<F, Fitted>`): inline geometry guard → `validate_geometry`; the JL n_components resolution, the `eps` Fixed-path guard, the `nc < 1` guard, and the `rng::gaussian_matrix` draw are byte-identical; reconstructs into the `Fitted` value.
- `components` accessor + `impl Transform` moved onto `GaussianRandomProjection<F, Fitted>` — `components` drops `ok_or(NotFitted)` → `.expect(...)` and returns bare `Vec<F>`.

`random_projection_test.rs`: trait import → typestate (kept the legacy glob for the still-unmigrated Sparse at this commit, per the sequential recipe); all ~5 Gaussian fit sites → `builder().n_components(NComponents::Fixed(..)).seed(..).eps(..).build::<F>()? + TypestateFit::fit(...)`; `components()`/`transform()` un-`.expect()`'d / via `TypestateTransform`; added `gaussian_random_projection_defaults_equal` (BLDR-01).

`mlrs-py/src/estimators/projection.rs` (PyGaussianRandomProjection arm): `AnyGaussianRandomProjection` → `any_estimator_typestate!`; fit via `builder().n_components(nc).seed(seed).eps(eps).build::<f*>().map_err(build_err_to_py)?` then `TypestateFit::fit(...)`; transform via `TypestateTransform::transform` UFCS; `components_*` un-`.map_err`'d; the `global_pool().lock().expect("pool mutex")` form swapped to the poison-recovering `crate::lock_pool()` (WR-04). The Sparse arm kept the legacy `traits` glob at this commit.

**Gate:** `cargo test --features cpu --test random_projection_test` → **9 passed** (JL value oracle + Gaussian/Sparse moment/distortion/self-consistency/seed-reproducibility property gates + `gaussian_..._defaults_equal`; Sparse still on the legacy surface). `cargo build -p mlrs-py --features cpu` → clean.

### Task 2 — SparseRandomProjection (Shape A, transformer; multi-arg builder), commit `bccc3b7`

`crates/mlrs-algos/src/projection/sparse.rs`: the same Shape-A build-out as Gaussian, with the extra `.density(Option<f64>)` setter (the only multi-arg-`new` delta). `struct SparseRandomProjection<F, S = Unfit>` + `_state`; zero-arg `new()` (`Auto`/`seed=0`/`eps=0.1`/`density=None`); `SparseRandomProjectionBuilder.n_components(NComponents).seed(u64).eps(f64).density(Option<f64>)`; infallible-but-typed `build()` (eps + density∈(0,1] are resolution-path-coupled — `density=None` resolves to `1/sqrt(n_features)` at fit). Consuming-self `Fit::fit` (BYTE-IDENTICAL: density resolution, eps guard, `nc<1` guard, `rng::sparse_achlioptas_matrix` draw); `Transform` + accessors onto `Fitted`. Removed the now-unused `mlrs_core::PrimError` import (the inline ShapeMismatch guard was replaced by `validate_geometry`).

`random_projection_test.rs`: **dropped the legacy `traits` glob** (projection module complete); the remaining Sparse fit sites → builder + `TypestateFit::fit`; `components()`/`transform()` via typestate; added `sparse_random_projection_defaults_equal`.

PySparseRandomProjection arm: `AnySparseRandomProjection` → `any_estimator_typestate!`; fit via builder (+`.density(..)`) + `TypestateFit::fit`; transform via `TypestateTransform`; accessors un-`.map_err`'d; `lock_pool()`. **Dropped the legacy `traits` glob from the PyO3 file** (projection module fully on typestate).

**Gate:** `cargo test --features cpu --test random_projection_test` → **10 passed** (all 9 above + `sparse_..._defaults_equal`). `cargo build -p mlrs-py --features cpu` → clean.

### Task 3 — KernelDensity (Shape A'; adopt typestate Fit, gate ScoreSamples on Fitted), commit `1cd0b0e`

`crates/mlrs-algos/src/density/kernel_density.rs`:
- `struct KernelDensity<F, S = Unfit>` — added `_state: PhantomData<S>` (kept the existing `where F: Float + CubeElement + Pod` clause on the struct); the kernel/bandwidth-spec hyperparams and the `x_fit_`/`bandwidth_`/`fit_shape_` fitted fields unchanged (D-03).
- Zero-arg `new()` (`KdKernel::Gaussian`, `BandwidthSpec::Numeric(1.0)` — sklearn's `KernelDensity` default) on `impl<F> KernelDensity<F, Unfit>`; `builder()`/`into_builder()`/`hyperparams_eq()`/`impl Default`.
- `KernelDensityBuilder { kernel: KdKernel, bandwidth: BandwidthSpec }` — **both setters take the enum/spec directly** (non-scalar selectors); `build<F>()` infallible-but-typed (the `bandwidth>0` check is resolution-path-coupled — scott/silverman resolve the numeric bandwidth at fit; the kernel name is a closed enum).
- Imports: dropped `use crate::traits::ScoreSamples`; added `crate::error::{AlgoError, BuildError}` + `crate::typestate::{validate_geometry, Fit, Fitted, ScoreSamples, Unfit}`.
- **ADOPTS typestate `Fit`:** the INHERENT `fit` (formerly `&mut self -> Result<&mut Self>`) becomes `impl Fit for KernelDensity<F, Unfit>` (`type Fitted = KernelDensity<F, Fitted>`, consuming `self`). Every compute line — the `InvalidKernel` guard, the scott/silverman/numeric bandwidth resolution, the `InvalidBandwidth` finite-positive guard, the `X_fit_` device copy — is byte-identical; the inline geometry guard → `validate_geometry`; the no-op re-fit buffer release (`self.x_fit_.take()`) is **dropped** (a freshly-built `Unfit` carries no fitted state — the IncrementalPCA-reset precedent, 16-04). Reconstructs into the `Fitted` value.
- **Moves `ScoreSamples` onto `Fitted`:** `impl ScoreSamples for KernelDensity<F, Fitted>` using `crate::typestate::ScoreSamples`; the three `x_fit_`/`bandwidth_`/`fit_shape_` `ok_or(NotFitted)` reads become `.expect(...)` (Some by construction on `Fitted`); the q-geometry/`DimMismatch` guards and the entire `distance → kde_*_map → row_reduce → host logsumexp+log_norm−log(N)` assembly are byte-identical. The `bandwidth()` accessor moved onto `Fitted` and now returns bare `f64` (no `NotFitted` Result).
- Fixed the `density/mod.rs` module-doc link `crate::traits::ScoreSamples` → `crate::typestate::ScoreSamples` (Rule 3 — see Deviations).

`kernel_density_test.rs`: trait import → typestate; the `fit_score` + `run_bandwidth_rules` sites → `builder().kernel(..).bandwidth(..).build::<F>()? + TypestateFit::fit(.., None, ..)`; `score_samples` via `TypestateScoreSamples`; `bandwidth()` un-`.expect()`'d (now bare `f64`); added `kernel_density_defaults_equal` (BLDR-01).

PyKernelDensity arm: `AnyKernelDensity` → `any_estimator_typestate!`; fit via builder + `TypestateFit::fit`; `score_samples_*` via `TypestateScoreSamples::score_samples` UFCS; `bandwidth_()` un-`.map_err`'d (bare `f64`). The `PyKernelRidge` arm in the same file is UNCHANGED — KernelRidge uses INHERENT `fit`/`predict` methods (no trait glob), so the file references no other estimator-trait surface after this commit.

**Gate:** `cargo test --features cpu --test kernel_density_test` → **6 passed** (six-kernel f32/f64 oracle, scott/silverman bandwidth-rule f32/f64 + resolved-`bandwidth_` cross-check, `score_samples` length-n shape, `defaults_equal`). `cargo build -p mlrs-py --features cpu` → clean.

## Deviations from Plan

The three estimators followed the Shape-A / Shape-A' recipes exactly as the plan's Task actions specified. Two plan-anticipated implementation details (not deviations) and one auto-fix:

1. **Infallible-but-typed `build()` for all three** (plan-anticipated) — the plan's Task actions instructed relocating only data-INDEPENDENT validation to `build()`. For these three estimators EVERY guard is resolution-path-coupled (eps/density/bandwidth resolve at fit) or data-DEPENDENT (`n_components<1` vs `n_features`), so nothing can move to `build()` without changing behaviour. The `Result<_, BuildError>` is kept for `build_err_to_py` family-uniformity, exactly like SpectralEmbedding (16-05).
2. **Test-file naming** (plan-anticipated reconciliation) — the plan named `gaussian_projection_test.rs` / `sparse_projection_test.rs`; the actual in-tree file is the COMBINED `crates/mlrs-algos/tests/random_projection_test.rs` (it holds both estimators' property gates). Both estimators' sites were migrated in that one file (Task 1 migrated the Gaussian sites + kept the legacy glob for Sparse; Task 2 migrated the Sparse sites + dropped the glob).

**[Rule 3 - Blocking issue] Fixed a dangling doc link in `density/mod.rs`**
- **Found during:** Task 3 (the `! grep crate::traits` acceptance check passes for the estimator file, but a module-doc grep surfaced `density/mod.rs:6` still linking `crate::traits::ScoreSamples`).
- **Issue:** the module doc-comment intra-doc link `[`ScoreSamples`](crate::traits::ScoreSamples)` would become a broken `cargo doc` link the moment `traits.rs` is hard-deleted (Plan 16-12), since KernelDensity now lives on `typestate::ScoreSamples`.
- **Fix:** retargeted the doc link to `crate::typestate::ScoreSamples`.
- **Files modified:** `crates/mlrs-algos/src/density/mod.rs`.
- **Commit:** `1cd0b0e` (folded into the Task-3 commit).

## Authentication Gates

None.

## Threat Mitigations (from the plan's threat_model)

- **T-16-V5** (geometry guard / relocated validation): `validate_geometry(x, shape)?` is at the TOP of all three ported `fit`s, BEFORE any RNG / distance device launch (and `score_samples` keeps its own q-geometry + `n_features`-agreement guards). No validation was dropped in the move: the data-INDEPENDENT-looking eps/density/bandwidth checks are resolution-path-coupled and stay in `fit` (they CANNOT move to `build()` without changing behaviour), and `n_components<1` is data-DEPENDENT. ✓
- **T-16-GUARDF64** (F64 guard): `crate::capability::guard_f64()?` preserved verbatim before every F64 upload in the migrated PyGaussianRandomProjection / PySparseRandomProjection / PyKernelDensity fit arms. ✓
- **T-16-ARM** (Fitted arm type): `AnyGaussianRandomProjection` / `AnySparseRandomProjection` / `AnyKernelDensity` switched to `any_estimator_typestate!` so each fitted value is typed `T<f*, Fitted>`. ✓

## Known Stubs

None.

## Threat Flags

None — no new network/auth/file/schema surface; a trait-surface retrofit with byte-identical compute.

## Verification Environment Note

`cargo build -p mlrs-py --features cpu` is the cross-crate PyO3 gate (live Python pytest of the wheel is untestable here per the project memory "Python wheel untestable in env"). The three oracle/property suites + the mlrs-py build are the compensating Rust gates; the Python-boundary behavior (Unfit-arm accessor → `not_fitted` → PyValueError; the `NComponents` Option<usize> sentinel → builder enum setter) is unchanged from the pre-retrofit shells.

## Acceptance Evidence

- `cargo test --features cpu --test random_projection_test` → **10 passed** (JL value oracle, Gaussian/Sparse moment + JL-distortion + self-consistency + seed-reproducibility property gates, `gaussian_..._defaults_equal`, `sparse_..._defaults_equal`).
- `cargo test --features cpu --test kernel_density_test` → **6 passed** (six-kernel f32+f64, scott/silverman bandwidth-rule f32+f64 + resolved-`bandwidth_`, score_samples length-n, `defaults_equal`).
- `cargo build -p mlrs-py --features cpu` → Finished (only 2 pre-existing spectral.rs dead-code warnings, out of scope).
- `! grep -q 'crate::traits'` on gaussian.rs / sparse.rs / kernel_density.rs → all clean.
- `! grep -q 'mlrs_algos::traits'` on the PyO3 projection.rs / kernel.rs → both clean (projection + density modules complete).
- `grep -cE 'typestate::(Fit|ScoreSamples)'` on kernel_density.rs → **2** (adopts typestate Fit + the typestate ScoreSamples).
- The Gaussian/Sparse builder `n_components` setter signature is `fn n_components(self, v: NComponents)` (enum, not f64).
- D-03: per-file `git diff` shows ZERO compute-line changes for all three fit/score bodies (signature/return/guard-call/reconstruction only; the only KernelDensity behavioural delta is dropping the no-op re-fit buffer release, which a fresh `Unfit` makes vacuous).

## For Downstream Plans

- **Enum-typed builder setter:** when a hyperparameter is a non-scalar selector (an enum like `NComponents`/`KdKernel`, a spec like `BandwidthSpec`, or a `String`), the builder setter takes it BY VALUE directly — A5's f64-setter convention applies ONLY to scalar narrowing. The PyO3 wrapper keeps whatever sentinel it already uses (e.g. `Option<usize>` for `n_components`) and maps it to the enum before feeding the builder.
- **Shape-A' ScoreSamples-only adoption:** an estimator with an inherent fit + a single legacy accessor impl (no Fit trait) adopts typestate `Fit` (inherent fit → consuming-self trait impl on `Unfit`, body verbatim) AND moves its accessor onto the `Fitted` impl using the typestate accessor trait. Drop any no-op re-fit reset (a fresh `Unfit` carries no fitted state).
- **Infallible-but-typed build() is the norm for unsupervised transformers / density estimators** whose hyperparameter validity is resolution-path-coupled (resolved at fit against the data geometry): keep `build()` returning `Result<_, BuildError>` for PyO3 mapper uniformity even when it never errs.

## Self-Check: PASSED

- `crates/mlrs-algos/src/projection/gaussian.rs` — FOUND, modified, builds, tests pass.
- `crates/mlrs-algos/src/projection/sparse.rs` — FOUND, modified, builds, tests pass.
- `crates/mlrs-algos/src/density/kernel_density.rs` — FOUND, modified, builds, tests pass.
- `crates/mlrs-algos/src/density/mod.rs` — FOUND, modified (doc-link fix), builds.
- `crates/mlrs-py/src/estimators/projection.rs` — FOUND, modified, mlrs-py builds.
- `crates/mlrs-py/src/estimators/kernel.rs` — FOUND, modified, mlrs-py builds.
- Commit `1efd541` (GaussianRandomProjection) — FOUND.
- Commit `bccc3b7` (SparseRandomProjection) — FOUND.
- Commit `1cd0b0e` (KernelDensity) — FOUND.
