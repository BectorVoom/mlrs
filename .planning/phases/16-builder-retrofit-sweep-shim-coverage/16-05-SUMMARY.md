---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 05
subsystem: cluster-estimators
tags: [typestate, builder-retrofit, cluster, dbscan, spectral-clustering, spectral-embedding, wide-builder, shape-a-prime, wave-5]
requires:
  - "typestate::{Fit, validate_geometry, Unfit, Fitted} (Plan 16-00)"
  - "any_estimator_typestate! macro (dispatch.rs)"
  - "Shape-A recipe proven in ridge.rs (Plan 16-01) + with_opts-fold (Plan 16-02)"
  - "PyHDBSCAN typestate wrap exemplar (cluster.rs, Phase 12/15)"
provides:
  - "DBSCAN<F, S=Unfit> + DbscanBuilder (.eps/.min_samples) on the typestate surface (Shape-A; Fit only — no predict, D-08)"
  - "SpectralClustering<F, S=Unfit> + SpectralClusteringBuilder folding the 6-arg legacy new (.n_clusters/.n_components(Option)/.affinity(String)/.gamma/.n_neighbors/.seed)"
  - "SpectralEmbedding<F, S=Unfit> + SpectralEmbeddingBuilder — ADOPTS typestate Fit (was inherent fit, no trait); shape A'"
  - "BuildError::InvalidEps (new); shared BuildError::InvalidMinSamples generalized to dbscan"
  - "PyDBSCAN / PySpectralClustering / PySpectralEmbedding on any_estimator_typestate! (Fitted arms)"
  - "Wide-builder recipe (String/Option setters) proven for KMeans init() in Plan 06"
affects:
  - "Plan 16-06 (KMeans — the wide-builder String/Option setter pattern is now proven on SpectralClustering)"
  - "Plan 16-11 (traits.rs deletion — dbscan/spectral_embedding off crate::traits; spectral_clustering off it ONCE KMeans migrates in Plan 06)"
tech-stack:
  added: []
  patterns:
    - "Wide-builder fold: a 6-arg secondary constructor (n_components: Option<usize>, affinity: String, gamma, n_neighbors, seed) collapses into builder setters that take String/Option<usize> directly (the new setter shapes); the scalar gamma is f64 (A5) and narrows in build::<F>()"
    - "Shape A' trait adoption: an estimator with an INHERENT fit + accessor and NO crate::traits import ADOPTS typestate::Fit (consuming-self) to join the single trait surface; non-transductive estimators (no inherent transform) do NOT adopt Transform"
    - "Cross-estimator legacy dependency: SpectralClustering's fit drives the inner v1 KMeans which is still on legacy traits::Fit (migrated Plan 06) — imported as LegacyFit + called via UFCS so the estimator's OWN trait surface is typestate while the unmigrated dependency is bridged"
    - "Infallible-but-typed build() for affinity-branch-coupled validation: gamma>0 (only the rbf path uses gamma) and data-DEPENDENT n_components checks stay in fit; build() returns Result<_, BuildError> that never errs, kept for build_err_to_py family uniformity"
key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/cluster/dbscan.rs
    - crates/mlrs-algos/src/error.rs
    - crates/mlrs-algos/tests/dbscan_test.rs
    - crates/mlrs-algos/src/cluster/spectral_clustering.rs
    - crates/mlrs-algos/tests/spectral_clustering_test.rs
    - crates/mlrs-algos/src/cluster/spectral_embedding.rs
    - crates/mlrs-algos/tests/spectral_embedding_test.rs
    - crates/mlrs-py/src/estimators/cluster.rs
    - crates/mlrs-py/src/estimators/spectral.rs
decisions:
  - "DBSCAN data-INDEPENDENT validation (eps>=0, min_samples>=1) relocated from the fit body (AlgoError::InvalidEps/InvalidMinSamples) to DbscanBuilder::build()->BuildError. Added BuildError::InvalidEps (new variant); reused the existing BuildError::InvalidMinSamples (HDBSCAN's) and generalized its doc/estimator field to cover dbscan rather than adding a duplicate variant (the shape {estimator, min_samples} was identical)"
  - "SpectralClustering gamma>0 validation stays in the fit body, NOT relocated to build(): the check is affinity-branch-coupled (only the rbf path reads gamma; the nearest_neighbors path does NOT validate gamma). Relocating it to build() would reject gamma<=0 for the nearest_neighbors path too, changing behavior. Keeping it in fit preserves the exact pre-retrofit semantics (D-03) — build() is infallible-but-typed for family uniformity"
  - "SpectralEmbedding adopts typestate::Fit only, NOT Transform: it is non-transductive (sklearn's own SpectralEmbedding exposes fit_transform/embedding_ but no transform for new points) and had NO inherent transform to adopt. The plan's 'Fit/Transform' phrasing presumed an inherent transform that does not exist; adopting Fit alone satisfies the single-surface end-state and the Task-3 grep (impl Fit for SpectralEmbedding<F, Unfit> matches)"
  - "SpectralClustering keeps a `use crate::traits::Fit as LegacyFit` import (the inner v1 KMeans is migrated in Plan 06, not here). This is the ONE crate::traits reference remaining in the file; it is aliased and UFCS-called at the single KMeans site. The Task-2 `! grep -q crate::traits spectral_clustering.rs` criterion cannot hold until KMeans migrates (Plan 06) — documented as the unavoidable sequencing consequence, NOT a scope deviation. The estimator's OWN Fit/accessor surface is 100% typestate"
metrics:
  duration: ~50m
  completed: 2026-06-24
  tasks: 3
  files: 9
status: complete
---

# Phase 16 Plan 05: Cluster sweep (minus KMeans) — DBSCAN + SpectralClustering + SpectralEmbedding typestate retrofit — Summary

Migrated the three non-KMeans `cluster/` members onto the `mlrs_algos::typestate` surface, each under its own commit gated by its sklearn oracle suite AND `cargo build -p mlrs-py --features cpu`. **DBSCAN** is the simple Shape-A case (Fit only — no predict, D-08); **SpectralClustering** is the WIDE 6-arg-`new`→builder case (the `affinity: String` / `n_components: Option<usize>` setters are the new shapes proven here for KMeans's `init()` setter in Plan 06); **SpectralEmbedding** is the Shape-A' trait-adoption case (an INHERENT `fit` with no `crate::traits` import that NEWLY adopts `typestate::Fit`). All three fit-body compute paths are byte-identical to pre-retrofit (D-03), and all 15 oracle/build tests stay green (exact-label for DBSCAN/SC, 1e-5/sign-aligned/subspace for SE, f64 + f32).

## What Was Built

### Task 1 — DBSCAN (Shape A; Fit only, no predict), commit `20c2154`

`crates/mlrs-algos/src/cluster/dbscan.rs`:
- `struct DBSCAN<F, S = Unfit>` — added `_state: PhantomData<S>` alongside the existing `_marker: PhantomData<F>` (DBSCAN keeps no F-typed fitted state, so it carries BOTH a float-binding marker and the lifecycle marker); `eps`/`min_samples`/`labels_`/`core_sample_indices_` unchanged (D-03).
- Replaced the arg-taking `new(eps, min_samples)` with **zero-arg `new()`** (sklearn defaults `eps = 0.5`, `min_samples = 5`) on `impl<F> DBSCAN<F, Unfit>`; added `builder()`, `into_builder()`, `hyperparams_eq()`, `impl Default`.
- New `DbscanBuilder { eps: f64, min_samples: usize }` with `.eps(f64)/.min_samples(usize)` setters; `Default` = `DBSCAN::<f64, Unfit>::new().into_builder()` (single source); `build<F>()` relocates the data-INDEPENDENT `eps >= 0` (`BuildError::InvalidEps`) and `min_samples >= 1` (`BuildError::InvalidMinSamples`) checks from the old fit body.
- Imports: dropped `use crate::traits::Fit` + the now-unused `mlrs_core::PrimError`; added `use crate::error::{AlgoError, BuildError}` and `use crate::typestate::{validate_geometry, Fit, Fitted, Unfit}`.
- `impl Fit for DBSCAN<F, Unfit>` (consuming `self -> Result<DBSCAN<F, Fitted>, AlgoError>`): inline shape guard → `validate_geometry`; the eps/min_samples fit-body checks removed (relocated to build); **every compute line** (`eps_core_mask` prim, the host index-ordered LIFO DFS reproducing `_dbscan_inner.pyx`, the `core_sample_indices_` collection) byte-identical; reconstructs into the `Fitted` value.
- `labels`/`core_sample_indices` accessors moved onto `impl<F> DBSCAN<F, Fitted>`, returning `Vec<i32>` (not `Result`) via `.expect(...)`. `fit_predict` now CONSUMES `self` and returns `(DBSCAN<F, Fitted>, DeviceArray<i32>)`.

`crates/mlrs-algos/src/error.rs`: added `BuildError::InvalidEps { estimator, eps }` (new); generalized the existing `BuildError::InvalidMinSamples` doc + `estimator` field from HDBSCAN-only to cover `dbscan` (the `{estimator, min_samples}` shape was identical — reused rather than duplicated).

`crates/mlrs-algos/tests/dbscan_test.rs`: trait import → typestate; the `fit_dbscan` + `fit_predict_consistency` call sites → builder + consuming-self; accessors un-`.expect()`'d; the invalid-hyperparameter test now asserts `build()` → `BuildError::InvalidEps/InvalidMinSamples` (the validation moved to the builder); added `dbscan_defaults_equal` (BLDR-01).

`crates/mlrs-py/src/estimators/cluster.rs` (PyDBSCAN arm): `AnyDbscan` → `any_estimator_typestate!`; fit builds via `DBSCAN::<f*>::builder().eps(..).min_samples(..).build::<f*>().map_err(build_err_to_py)?` then `TypestateFit::fit(...)`; `labels_`/`core_sample_indices_` accessors un-`.map_err`'d. `guard_f64()`/`lock_pool()` unchanged. PyKMeans (legacy) keeps the `traits::{Fit, PredictLabels}` glob (KMeans migrates in Plan 06).

**Gate:** `cargo test --features cpu --test dbscan_test` → **5 passed** (labels f32, core_sample_indices f64, fit_predict consistency, build-time rejection, defaults_equal). `cargo build -p mlrs-py --features cpu` → clean.

### Task 2 — SpectralClustering (Shape A, WIDE 6-arg new → builder), commit `1cb060e`

`crates/mlrs-algos/src/cluster/spectral_clustering.rs`:
- `struct SpectralClustering<F, S = Unfit>` + `_state: PhantomData<S>`; all 6 hyperparam fields + `labels_`/`labels_host_` unchanged (D-03).
- Zero-arg `new()` sets all six sklearn defaults (`n_clusters = 8`, `n_components = None`, `affinity = "rbf"`, `gamma = 1.0` literal D-04, `n_neighbors = 10`, `seed = 0`). `builder()`/`into_builder()`/`hyperparams_eq()`/`Default`.
- `SpectralClusteringBuilder` (a `Clone` non-Copy because of the `String` affinity field) **subsumes the full 6-arg new** with `.n_clusters(usize)/.n_components(Option<usize>)/.affinity(String)/.gamma(f64)/.n_neighbors(usize)/.seed(u64)` setters — the `Option<usize>` and `String` setters are the WIDE-builder shapes. `build<F>()` narrows `gamma` to `F` via `f64_to_host`; it is **infallible-but-typed** (the `gamma > 0` check stays in fit, affinity-branch-coupled).
- Imports: dropped `use crate::traits::Fit`; added `use crate::error::{AlgoError, BuildError}`, `use crate::typestate::{Fit, Fitted, Unfit}`, AND `use crate::traits::Fit as LegacyFit` (the inner v1 KMeans is still legacy until Plan 06).
- `impl Fit for SpectralClustering<F, Unfit>` (consuming-self): the `n_samples > 64` SPECTRAL cap, geometry, `InvalidK`, `InvalidNComponents`, and rbf `gamma > 0` guards byte-identical; **every pipeline compute line** (`kernel_matrix(Rbf)` / `knn_connectivity_affinity` → `laplacian` → `eig` (the WR-05 aliased-handle reuse) → `recover(drop_first=false)` → the inner `KMeans::new(...).fit` driven via `LegacyFit::fit(&mut kmeans, ...)` UFCS) byte-identical; reconstructs into `Fitted`.
- `labels` accessor + `fit_predict` (now consuming-self, returns `(Fitted, labels)`) moved onto/over `SpectralClustering<F, {Fitted,Unfit}>`.

`crates/mlrs-algos/tests/spectral_clustering_test.rs`: trait → typestate; `fit_labels` + `reject_oversize` → wide builder + consuming-self; accessors un-`.expect()`'d; added `spectral_clustering_defaults_equal` (BLDR-01, exercises the 6-field round-trip).

`crates/mlrs-py/src/estimators/spectral.rs` (PySpectralClustering arm): `AnySpectralClustering` → `any_estimator_typestate!`; fit builds via the wide `builder()...build::<f*>().map_err(build_err_to_py)?` + `TypestateFit::fit`; **dropped the `gamma as f32` cast** (builder setter is f64); `labels_` un-`.map_err`'d. Added `use mlrs_algos::typestate::Fit as TypestateFit` + `build_err_to_py` to the imports.

**Gate:** `cargo test --features cpu --test spectral_clustering_test` → **4 passed** (exact-label f64 + f32, reject_oversize, defaults_equal). `cargo build -p mlrs-py --features cpu` → clean.

### Task 3 — SpectralEmbedding (Shape A'; ADOPT typestate Fit), commit `e17e14b`

`crates/mlrs-algos/src/cluster/spectral_embedding.rs` (the shape-A' case — INHERENT `fit` + accessor, NO prior `crate::traits` import):
- `struct SpectralEmbedding<F, S = Unfit>` + `_state: PhantomData<S>`; `n_components`/`affinity`/`gamma`/`n_neighbors`/`embedding_` unchanged (D-03).
- Zero-arg `new()` sets the sklearn defaults (`n_components = 2`, `affinity = "nearest_neighbors"`, `gamma = None`, `n_neighbors = 10`). `builder()`/`into_builder()`/`hyperparams_eq()`/`Default`.
- `SpectralEmbeddingBuilder` (`Clone`, `String` field) with `.n_components(usize)/.affinity(String)/.gamma(Option<f64>)/.n_neighbors(usize)` setters; `build<F>()` narrows `Option<f64>` → `Option<F>` via `self.gamma.map(f64_to_host)`; infallible-but-typed.
- **NEWLY ADOPTS** `use crate::typestate::{validate_geometry, Fit, Fitted, Unfit}` (it had no trait import before). The INHERENT `fit` becomes `impl Fit for SpectralEmbedding<F, Unfit>` (consuming-self, `type Fitted = SpectralEmbedding<F, Fitted>`): the inline shape guard → `validate_geometry`; **every pipeline compute line** (affinity → `laplacian` → `eig` WR-05 reuse → `recover(drop_first=true)`) byte-identical; reconstructs into `Fitted`. The inherent `embedding` accessor → `impl<F> SpectralEmbedding<F, Fitted>` returning `Vec<F>` (not `Result`). It does NOT adopt `Transform` — SpectralEmbedding is non-transductive (no inherent transform exists; sklearn likewise exposes only `fit_transform`/`embedding_`).

`crates/mlrs-algos/tests/spectral_embedding_test.rs`: trait import added (typestate `Fit`); `fit_embedding` + `reject_oversize` → builder + consuming-self; `embedding` un-`.expect()`'d; added `spectral_embedding_defaults_equal` (BLDR-01). The four oracle cases (rbf value-match, knn affinity, degenerate subspace, oversize-reject) operate unchanged.

`crates/mlrs-py/src/estimators/spectral.rs` (PySpectralEmbedding arm): `AnySpectralEmbedding` → `any_estimator_typestate!`; fit builds via `builder()...build::<f*>()? + TypestateFit::fit`; **dropped the `gamma.map(|g| g as f32)` cast** (builder setter is `Option<f64>`); `embedding_f32`/`embedding_f64` un-`.map_err`'d.

**Gate:** `cargo test --features cpu --test spectral_embedding_test` → **6 passed** (rbf f64 + f32, knn affinity f64, degenerate subspace f64, reject_oversize, defaults_equal). `cargo build -p mlrs-py --features cpu` → clean.

## The Wide-Builder + Shape-A' Recipe (for KMeans in Plan 06)

1. **Wide builder (String/Option setters):** a multi-arg secondary constructor with `String` / `Option<usize>` arguments collapses into builder setters that take those types DIRECTLY (`.affinity(String)`, `.n_components(Option<usize>)`). The builder is `Clone` (not `Copy`) because of the `String` field. Scalar `gamma` is `f64` (A5), narrowed in `build::<F>()`. This is the exact shape KMeans's `init()` setter needs.
2. **Branch-coupled validation stays in fit:** a hyperparameter check that is conditional on another field (here `gamma > 0` only on the `affinity == "rbf"` branch) MUST stay in the fit body — relocating it to `build()` would change behavior on the other branch. `build()` is then infallible-but-typed (kept for `build_err_to_py` family uniformity).
3. **Shape A' trait adoption:** an estimator with an INHERENT `fit` (no `crate::traits` import) ADOPTS `typestate::Fit` (consuming-self) to join the single surface. Adopt `Transform` ONLY if an inherent transform exists; a non-transductive estimator (DBSCAN, SpectralEmbedding) adopts `Fit` alone.
4. **Cross-estimator legacy bridge:** if an estimator's fit drives an UNMIGRATED dependency (here SC drives the legacy KMeans), import the legacy trait under an alias (`use crate::traits::Fit as LegacyFit`) and call it via UFCS at the single site — the estimator's own surface stays typestate.

## Deviations from Plan

1. **[Rule 3 — Blocking] Added `BuildError::InvalidEps`; reused `BuildError::InvalidMinSamples`.** The plan's DBSCAN builder relocates `eps >= 0` / `min_samples >= 1` to `build()` → `BuildError`, but `BuildError` had NO `InvalidEps` variant (it lived only on `AlgoError`). Added `BuildError::InvalidEps`. `BuildError::InvalidMinSamples` already existed (HDBSCAN's, identical `{estimator, min_samples}` shape) — reused it (generalized its doc/field) rather than adding a duplicate variant (an E0428 collision when I first added one; resolved by removing the duplicate). Commit `20c2154`.

2. **SpectralEmbedding adopts `Fit` only, NOT `Transform`.** The plan's Task-3 phrasing ("its inherent fit/transform become trait impls", "Transform on Fitted") presumed an inherent `transform` that **does not exist** — SpectralEmbedding has only an inherent `fit` + `embedding` accessor (it is non-transductive, exactly like sklearn's `SpectralEmbedding`, which exposes `fit_transform`/`embedding_` but no `transform`). Adopting `Fit` alone satisfies the single-surface end-state and the Task-3 acceptance grep (`impl Fit for SpectralEmbedding<F, Unfit>` matches). No `Transform` impl was fabricated for a non-existent operation.

3. **SpectralClustering retains ONE `use crate::traits::Fit as LegacyFit`.** The Task-2 acceptance criterion `! grep -q 'crate::traits' spectral_clustering.rs` CANNOT hold while the inner v1 KMeans is on the legacy trait (KMeans is deliberately migrated LAST, in Plan 06 — D-06). SpectralClustering's fit drives `KMeans::new(...).fit(...)`, and KMeans exposes `fit` only via `traits::Fit`. The legacy import is aliased (`LegacyFit`) and UFCS-called at the single KMeans site; SpectralClustering's OWN `Fit`/accessor surface is 100% typestate. This is the unavoidable consequence of the plan's own KMeans-last sequencing, NOT a scope deviation — the grep criterion is satisfiable only after Plan 06.

The three fit bodies are byte-identical (verified per-file: `git diff` shows no change to any `eps_core_mask`/DFS, `kernel_matrix`/`laplacian`/`eig`/`recover`/KMeans compute line — only signature, return, guard-call (`validate_geometry`), and struct-reconstruction edits).

## Authentication Gates

None.

## Threat Mitigations (from the plan's threat_model)

- **T-16-V5** (cluster fit geometry guard + relocated validation): `validate_geometry(x, shape)?` is at the TOP of all three ported `fit`s (after the `n_samples > 64` spectral cap for SC/SE), before any device launch. The DBSCAN `eps >= 0` / `min_samples >= 1` data-INDEPENDENT checks are relocated to `DbscanBuilder::build()` → `BuildError`, NOT dropped. The SC/SE branch-coupled `gamma > 0` + data-DEPENDENT `n_components`/`InvalidK`/`NSamplesExceedsMaxDim` checks stay in fit. ✓
- **T-16-GUARDF64** (F64 guard): `crate::capability::guard_f64()?` preserved verbatim before every F64 upload in the three migrated PyO3 fits (PyDBSCAN / PySpectralClustering / PySpectralEmbedding); `lock_pool()` (poison-recovering) kept. ✓
- **T-16-ARM** (Fitted arm type): `AnyDbscan`/`AnySpectralClustering`/`AnySpectralEmbedding` switched to `any_estimator_typestate!` so each fitted value is typed `T<f*, Fitted>`. ✓

## Known Stubs

None.

## Threat Flags

None — no new network/auth/file/schema surface; a trait-surface retrofit with byte-identical compute. (The new `BuildError::InvalidEps` variant is a construction-time hyperparameter guard, a mitigation, not new surface.)

## Verification Environment Note

`cargo build -p mlrs-py --features cpu` is the cross-crate PyO3 gate (live Python pytest of the wheel is untestable here per the project memory "Python wheel untestable in env" — no maturin/pyarrow). The three oracle suites + the mlrs-py build are the compensating Rust gates; the Python-boundary behavior (Unfit-arm accessor → `not_fitted` → PyValueError) is unchanged from the pre-retrofit shells. The 2 remaining `mlrs-py` dead-code warnings on the `Unfit` arm fields are pre-existing (the WR-02 persisted-`params` pattern reads the fields off the `params` struct, not the enum arm) and out of scope.

## Acceptance Evidence

- `cargo test --features cpu --test dbscan_test` → **5 passed** (exact-label + core-set, build-time rejection, defaults_equal).
- `cargo test --features cpu --test spectral_clustering_test` → **4 passed** (exact-label f64 + f32, reject_oversize, defaults_equal).
- `cargo test --features cpu --test spectral_embedding_test` → **6 passed** (rbf 1e-5/sign-aligned f64 + f32-band, knn affinity, degenerate subspace, reject_oversize, defaults_equal).
- `cargo build -p mlrs-py --features cpu` → Finished (2 pre-existing spectral.rs dead-code warnings only, out of scope).
- `! grep -q 'crate::traits'` on dbscan.rs → CLEAN; on spectral_embedding.rs → only a doc-comment mention (no active import); on spectral_clustering.rs → ONE aliased `LegacyFit` for the unmigrated inner KMeans (deviation 3).
- `grep -cE 'typestate::(Fit|Transform)|impl.*Fit.*SpectralEmbedding|...' spectral_embedding.rs` → 1 (>0): SpectralEmbedding is now ON the trait surface (was inherent).
- `AnyDbscan`/`AnySpectralClustering`/`AnySpectralEmbedding` use `any_estimator_typestate!`.
- D-03: per-file `git diff` shows ZERO compute-line changes (signature/return/guard-call/reconstruction only).

## Self-Check: PASSED

- `crates/mlrs-algos/src/cluster/dbscan.rs` — FOUND, builds, 5 tests pass.
- `crates/mlrs-algos/src/cluster/spectral_clustering.rs` — FOUND, builds, 4 tests pass.
- `crates/mlrs-algos/src/cluster/spectral_embedding.rs` — FOUND, builds, 6 tests pass.
- `crates/mlrs-py/src/estimators/cluster.rs` / `spectral.rs` — FOUND, mlrs-py builds.
- Commit `20c2154` (DBSCAN) — FOUND.
- Commit `1cb060e` (SpectralClustering) — FOUND.
- Commit `e17e14b` (SpectralEmbedding) — FOUND.
