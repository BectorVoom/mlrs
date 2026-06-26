---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 06
subsystem: cluster-covariance-estimators
tags: [typestate, builder-retrofit, kmeans, multi-constructor, wide-builder, init-option, covariance, empirical-covariance, ledoit-wolf, shape-a, wave-6]
requires:
  - "typestate::{Fit, PredictLabels, validate_geometry, Unfit, Fitted} (Plan 16-00)"
  - "any_estimator_typestate! macro (dispatch.rs)"
  - "Shape-A recipe proven in ridge.rs (Plan 16-01) + with_opts-fold (Plan 16-02)"
  - "Wide-builder (String/Option setters) proven on SpectralClustering (Plan 16-05)"
provides:
  - "KMeans<F, S=Unfit> + KMeansBuilder folding all THREE legacy constructors (new/with_init/with_opts) — .n_clusters/.seed/.max_iter/.tol + .init(Option<Vec<f64>>)"
  - "EmpiricalCovariance<F, S=Unfit> + EmpiricalCovarianceBuilder (.assume_centered/.store_precision) on the typestate surface"
  - "LedoitWolf<F, S=Unfit> + LedoitWolfBuilder (.assume_centered) on the typestate surface"
  - "PyKMeans / PyEmpiricalCovariance / PyLedoitWolf on any_estimator_typestate! (Fitted arms)"
  - "SpectralClustering 100% off crate::traits (the LegacyFit bridge removed once its inner KMeans adopted typestate Fit)"
  - "Wide-builder Option-of-DATA setter pattern proven (KMeans .init narrows Vec<f64> → Vec<F> in build::<F>())"
affects:
  - "Plan 16-11/16-12 (traits.rs deletion — cluster/ + covariance/ now fully off crate::traits; spectral_clustering's last LegacyFit reference is gone)"
tech-stack:
  added: []
  patterns:
    - "Multi-constructor fold: a THREE-constructor estimator (new/with_init/with_opts) collapses into ONE wide builder; the deterministic-oracle injected-init path (with_init) becomes an .init(Option<Vec<f64>>) setter"
    - "Wide-builder Option-of-DATA setter: a builder setter that carries F-typed array data (not a scalar) stays NON-generic by storing the data as Option<Vec<f64>> and narrowing it (Vec<f64> → Vec<F> via f64_to_host) once in build::<F>() — the same A5 'setters are f64, build narrows' rule extended from scalars to vectors"
    - "assign() shared helper on impl<F, S>: an associated function used by BOTH the Unfit fit (assignment loop) and the Fitted predict_labels lives on impl<F, S> KMeans<F, S> so it is reachable from both lifecycle states without duplication"
    - "Partial-accessor split on Fitted: covariance_/location_ are always-Some on Fitted → infallible Vec<F>; precision_ keeps Result<_, AlgoError> because NotFitted now means 'store_precision was false' (a runtime 'not stored' condition, distinct from the unfitted state the typestate rules out)"
    - "Infallible KMeans build() called from a composing estimator: SpectralClustering's fit builds the inner KMeans via the builder and .expect()s the infallible build() (KMeans has NO data-INDEPENDENT validation — k/init/geometry are all data-DEPENDENT and stay in fit), avoiding a new From<BuildError> for AlgoError impl"
key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/cluster/kmeans.rs
    - crates/mlrs-algos/src/cluster/spectral_clustering.rs
    - crates/mlrs-algos/tests/kmeans_test.rs
    - crates/mlrs-algos/src/covariance/empirical_covariance.rs
    - crates/mlrs-algos/tests/empirical_covariance_test.rs
    - crates/mlrs-algos/src/covariance/ledoit_wolf.rs
    - crates/mlrs-algos/tests/ledoit_wolf_test.rs
    - crates/mlrs-py/src/estimators/cluster.rs
    - crates/mlrs-py/src/estimators/covariance.rs
key-decisions:
  - "KMeans .init setter stores Option<Vec<f64>> (NOT Vec<F>) so the builder stays non-generic; the injected init narrows to Vec<F> in build::<F>() via f64_to_host, mirroring the A5 scalar convention. The kmeans_test injected-init fixture path uses .init(Some(init_host)) where init_host is the fixture's f64 init read verbatim (no per-F conversion in the test)."
  - "KMeans build() is infallible-but-typed: KMeans has NO data-INDEPENDENT hyperparameter. 1 <= n_clusters <= n_samples, the injected-init dimension (len == k*n_features), and the geometry are ALL data-DEPENDENT and stay in fit (D-03 byte-identical). The Result is kept for build_err_to_py family uniformity."
  - "SpectralClustering's inner KMeans now uses the typestate builder + consuming Fit::fit; the LegacyFit (use crate::traits::Fit as LegacyFit) import is REMOVED so spectral_clustering.rs is 100% off crate::traits. The infallible KMeans build() is .expect()'d in SC's fit rather than introducing a From<BuildError> for AlgoError impl (out of scope)."
  - "EmpiricalCovariance precision_ accessor keeps Result<_, AlgoError>: on Fitted, covariance_/location_ are always Some (→ infallible), but precision_ is Some ONLY when store_precision was true, so its NotFitted ('not stored') case is a legitimate runtime condition the typestate does not eliminate."
patterns-established:
  - "Multi-constructor fold (new/with_init/with_opts → one wide builder)"
  - "Wide-builder Option-of-DATA setter (Vec data narrowed Vec<f64> → Vec<F> in build::<F>())"
  - "Cross-estimator legacy-bridge REMOVAL: once a composed dependency (inner KMeans) adopts typestate, the composer (SpectralClustering) drops its aliased LegacyFit import"
requirements-completed: [BLDR-03]
duration: 12min
completed: 2026-06-24
status: complete
---

# Phase 16 Plan 06: KMeans (multi-constructor) + covariance/ typestate retrofit — Summary

**KMeans's three legacy constructors (`new`/`with_init`/`with_opts`) collapse into one wide `KMeansBuilder` with an `.init(Option<Vec<f64>>)` setter, and the covariance module (EmpiricalCovariance, LedoitWolf) joins the typestate surface — clearing the last multi-constructor stress case and finishing the cluster + covariance families, all with byte-identical fit compute.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-06-24T12:19:18Z
- **Completed:** 2026-06-24
- **Tasks:** 3
- **Files modified:** 9

## What Was Built

### Task 1 — KMeans (LATE multi-constructor; `new` + `with_init` + `with_opts` → one wide builder), commit `21170f1`

`crates/mlrs-algos/src/cluster/kmeans.rs`:
- `struct KMeans<F, S = Unfit>` + `_state: PhantomData<S>`; all hyperparam + fitted fields unchanged (D-03), including the `init: Option<Vec<F>>` injected-init field.
- Replaced the THREE arg-taking constructors with **zero-arg `new()`** (sklearn defaults `n_clusters = 8`, `max_iter = 300`, `tol = 1e-4`, `seed = 0`, `init = None`) on `impl<F> KMeans<F, Unfit>`; added `builder()`, `into_builder()` (promotes the injected `Vec<F>` init → `Vec<f64>`), `hyperparams_eq()` (compares the init vectors element-wise in f64), `impl Default`.
- New `KMeansBuilder { n_clusters, seed, max_iter, tol, init: Option<Vec<f64>> }` — the WIDE builder that **subsumes all three constructors** with `.n_clusters(usize)/.seed(u64)/.max_iter(usize)/.tol(f64)` PLUS the injected-init `.init(Option<Vec<f64>>)` setter (the `with_init` replacement). `build<F>()` narrows the scalars AND the init (`Vec<f64> → Vec<F>` via `f64_to_host`) to `F`; it is **infallible-but-typed** (KMeans has no data-INDEPENDENT validation — `1 ≤ k ≤ n_samples`, the init dimension, and the geometry are all data-DEPENDENT and stay in `fit`). `fn new(n_clusters, seed)`, `fn with_init`, `fn with_opts` all REMOVED.
- Imports: dropped `use crate::traits::{Fit, PredictLabels}`; added `use crate::error::{AlgoError, BuildError}`, `use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, Unfit}`, `std::marker::PhantomData`, and `f64_to_host`.
- `impl Fit for KMeans<F, Unfit>` (consuming `self -> Result<KMeans<F, Fitted>, AlgoError>`): the `InvalidK` (`1 ≤ k ≤ n_samples`) guard stays atop; the inline shape guard → `validate_geometry`; **every compute line** (the injected-vs-kmeans++ init branch incl. the init dimension check, the `tol_scaled = tol·mean_var` host pass, the Lloyd loop with the strict label-equality break BEFORE the tol check, the `inertia_rows_host`/`lloyd_update` empty-cluster relocation, the one-final-assignment pass, the `inertia` prim) byte-identical; reconstructs into the `Fitted` value.
- `PredictLabels` moved onto `impl<F> KMeans<F, Fitted>` (the `cluster_centers_` access drops `ok_or(NotFitted)` → `.expect(...)`). `cluster_centers`/`labels`/`inertia` accessors moved onto `KMeans<F, Fitted>` returning `Vec<F>`/`Vec<i32>`/`F` directly. `assign()` lives on `impl<F, S> KMeans<F, S>` so both the `Unfit` fit and the `Fitted` predict reach it. `release_into` (called by SpectralClustering) onto `KMeans<F, Fitted>`; `fit_predict` now consumes `Unfit` self and returns `(KMeans<F, Fitted>, DeviceArray<i32>)`.

`crates/mlrs-algos/src/cluster/spectral_clustering.rs`: the inner v1 KMeans now uses the typestate builder + consuming `Fit::fit` (build via `KMeans::<F>::builder().n_clusters(..).seed(..).build::<F>().expect(infallible)` then `kmeans.fit(...)?`, `labels(pool)` un-`?`'d, `release_into` on the `Fitted` value). The `use crate::traits::Fit as LegacyFit` import is **REMOVED** — `spectral_clustering.rs` is now 100% off `crate::traits` (the unavoidable Plan-05 sequencing consequence is resolved here, per the cross-plan note).

`crates/mlrs-algos/tests/kmeans_test.rs`: trait import → typestate; the three `with_init(KM_K, init_host)` call sites → `builder().n_clusters(KM_K).init(Some(init_host)).build()?.fit(...)?` (the injected init stays in f64 for the setter); accessors un-`.expect()`'d / un-`unwrap()`'d; the WR-03 constant-feature test → builder + consuming-self; added `defaults_equal` (BLDR-01).

`crates/mlrs-py/src/estimators/cluster.rs` (PyKMeans arm): `AnyKMeans` → `any_estimator_typestate!`; fit builds via `KMeans::<f*>::builder().n_clusters(..).seed(..).max_iter(..).tol(..).build::<f*>().map_err(build_err_to_py)?` then `TypestateFit::fit` (the `with_opts` args are now setters; the sklearn `random_state → seed` mapping preserved); `cluster_centers_`/`labels_`/`inertia_` accessors un-`.map_err`'d; `predict_labels` via the typestate `PredictLabels`. The legacy `mlrs_algos::traits` glob is GONE (PyKMeans was the file's last legacy consumer) — `Fit` aliased `TypestateFit`, `PredictLabels` aliased `TypestatePredictLabels`.

**Gate:** `cargo test --features cpu --test kmeans_test` → **7 passed** (centers/labels exact-permutation f32+f64 incl. injected init, inertia f32+f64, predict-consistency, WR-03 constant-feature, defaults_equal). `cargo test --features cpu --test spectral_clustering_test` → **4 passed** (the inner-KMeans regression). `cargo build -p mlrs-py --features cpu` → clean (2 pre-existing spectral.rs dead-code warnings only).

### Task 2 — EmpiricalCovariance (Shape A; Fit only), commit `316eab9`

`crates/mlrs-algos/src/covariance/empirical_covariance.rs`:
- `struct EmpiricalCovariance<F, S = Unfit>` + `_state`; the OnceLock memo caches + device-resident attrs unchanged (D-03).
- Zero-arg-style `new(assume_centered, store_precision)` kept on `impl<F> EmpiricalCovariance<F, Unfit>` (the two booleans ARE the only hyperparameters); added `builder()`/`into_builder()`/`hyperparams_eq()`/`impl Default` (= `new(false, true)`, sklearn defaults).
- `EmpiricalCovarianceBuilder { assume_centered, store_precision }` with both setters; `build<F>()` is **infallible-but-typed** (both knobs are booleans — no data-INDEPENDENT range to validate).
- Imports: dropped `use crate::traits::Fit` + the now-unused `PrimError`; added `BuildError` + the typestate surface + `PhantomData`.
- `impl Fit for EmpiricalCovariance<F, Unfit>` (consuming-self): the inline shape guard → `validate_geometry`; **every compute line** (the `assume_centered` location branch, the MLE Gram via the `covariance` prim / the `mle_gram_uncentered` host Gram, the eig-based `pinvh`) byte-identical; reconstructs into `Fitted` with fresh memo caches.
- `covariance_`/`location_` accessors moved onto `impl<F> EmpiricalCovariance<F, Fitted>` returning `Vec<F>` (always-Some on `Fitted`, memoized); `precision_` KEPT as `Result<_, AlgoError>` (its `NotFitted` now means `store_precision == false`, a runtime "not stored" condition — the `attr` helper retained for it).

`crates/mlrs-algos/tests/empirical_covariance_test.rs`: trait → typestate; the `new(.., true)` fit helper → builder + consuming-self; `covariance_`/`location_` un-`.expect()`'d (precision_ still `.expect()`'d on the Result); added `defaults_equal`.

`crates/mlrs-py/src/estimators/covariance.rs` (PyEmpiricalCovariance arm): `AnyEmpiricalCovariance` → `any_estimator_typestate!`; fit builds via the builder + `TypestateFit::fit` + `build_err_to_py`; `covariance_*`/`location_*` accessors un-`.map_err`'d (precision_* unchanged). `guard_f64()` preserved.

**Gate:** `cargo test --features cpu --test empirical_covariance_test` → **7 passed** (full-rank attrs f32+f64, rank-deficient pinvh f32+f64, assume_centered f32+f64, defaults_equal). `cargo build -p mlrs-py --features cpu` → clean.

### Task 3 — LedoitWolf (Shape A; Fit only), commit `56756bc`

`crates/mlrs-algos/src/covariance/ledoit_wolf.rs` (EmpiricalCovariance's sibling):
- `struct LedoitWolf<F, S = Unfit>` + `_state`; `assume_centered` + the device-resident `covariance_`/`location_` + the `shrinkage_: Option<f64>` scalar + memo caches unchanged.
- `new(assume_centered)` on `impl<F> LedoitWolf<F, Unfit>`; `builder()`/`into_builder()`/`hyperparams_eq()`/`impl Default` (= `new(false)`). `LedoitWolfBuilder { assume_centered }`, single `.assume_centered(bool)` setter, **infallible-but-typed** `build<F>()`.
- Imports: dropped `use crate::traits::Fit` + the unused `PrimError`; added `BuildError` + typestate + `PhantomData`.
- `impl Fit for LedoitWolf<F, Unfit>` (consuming-self): inline shape guard → `validate_geometry`; **every β/δ/μ shrinkage-estimate compute line** (the host-centered X, the f64 Gram, `emp_cov_trace`, `mu`, `beta_`, `delta_`, `beta`/`delta`, the `shrinkage = beta/delta` clip, the `(1−shrinkage)·emp_cov + shrinkage·μ·I` reassembly) byte-identical — the 1e-5 shrinkage math is unchanged. Reconstructs into `Fitted`.
- `covariance_`/`location_`/`shrinkage_` accessors moved onto `impl<F> LedoitWolf<F, Fitted>` returning `Vec<F>`/`Vec<F>`/`f64` directly (all always-Some on `Fitted`).

`crates/mlrs-algos/tests/ledoit_wolf_test.rs`: trait → typestate; the `new(false)` fit helper → builder + consuming-self; accessors un-Result'd; added `defaults_equal`.

`crates/mlrs-py/src/estimators/covariance.rs` (PyLedoitWolf arm): `AnyLedoitWolf` → `any_estimator_typestate!`; fit via builder + `TypestateFit::fit` + `build_err_to_py`; `covariance_*`/`location_*`/`shrinkage_` accessors un-`.map_err`'d. `guard_f64()` preserved. (This commit carries BOTH covariance PyO3 arms since they share the `Fit` import in one file; the EmpiricalCovariance arm body was already migrated in Task 2's algos work — committing both arms together keeps `mlrs-py` buildable.)

**Gate:** `cargo test --features cpu --test ledoit_wolf_test` → **5 passed** (n=12 f32+f64, n=40 f32+f64, defaults_equal). `cargo build -p mlrs-py --features cpu` → clean.

## The Multi-Constructor + Option-of-DATA Recipe (for any remaining multi-constructor estimators)

1. **Fold N constructors into one builder.** Every former constructor's distinguishing argument becomes a builder setter; `new()` is zero-arg and carries the sklearn defaults (single source, D-08). All N constructors are DELETED.
2. **Option-of-DATA setter (the KMeans `init` shape).** When a setter carries F-typed array DATA (not a scalar), keep the builder NON-generic: store it as `Option<Vec<f64>>` and narrow it to `Vec<F>` (via `f64_to_host` per element) ONCE inside `build::<F>()`. This extends the A5 "setters are f64, build narrows" convention from scalars to vectors — the builder never carries an `F` type parameter.
3. **Shared associated helper across states.** A helper used by BOTH the `Unfit` fit and a `Fitted` accessor (KMeans's `assign`) lives on `impl<F, S> T<F, S>` so it is reachable from either lifecycle state.
4. **Partial-accessor split.** On `Fitted`, attributes that are ALWAYS populated become infallible (`Vec<F>`); an attribute that may be absent for a NON-lifecycle reason (EmpiricalCovariance's `precision_` when `store_precision == false`) keeps its `Result` — the typestate eliminates the unfitted case, not every runtime absence.
5. **Composer drops its legacy bridge.** Once a composed dependency (inner KMeans) adopts typestate, the composer (SpectralClustering) removes its aliased `LegacyFit` import and drives the dependency through the typestate `Fit` — `.expect()`-ing the infallible inner `build()` rather than adding a `From<BuildError> for AlgoError` impl.

## Deviations from Plan

None — all three estimators followed the recipe exactly as the plan's Task actions specified.
- The KMeans `init(Option<Vec<f64>>)` setter (rather than `Option<Vec<F>>`) is the plan's own instruction ("follow whichever the spectral_clustering wide-builder established, keeping the injected-init fixture working") — spectral_clustering proved the non-generic `f64`-narrowing builder, and this extends it to vector data. The injected-init fixture at `kmeans_test.rs` uses it and passes.
- The infallible KMeans `build()` `.expect()` inside SpectralClustering's fit (vs a new `From<BuildError>` impl) is the minimal-scope choice: KMeans's `build()` genuinely cannot error (no data-INDEPENDENT validation), so the `expect` is unreachable, and adding a crate-wide `From` impl was out of scope.

The three fit bodies are byte-identical (verified per-file: the per-commit `git diff` shows ZERO change to any `lloyd_update`/`inertia`/`kmeanspp`/`argmin`/`distance`/`tol_scaled` (KMeans), `covariance`/`mle_gram_uncentered`/`pinvh`/`eig` (EmpiricalCovariance), or `beta_`/`delta_`/`shrinkage` (LedoitWolf) compute line — only signature, return, guard-call (`validate_geometry`), and struct-reconstruction edits).

## Authentication Gates

None.

## Threat Mitigations (from the plan's threat_model)

- **T-16-V5** (geometry guard + injected-init validation): `validate_geometry(x, shape)?` is at the TOP of all three ported `fit`s (after the KMeans `InvalidK` guard), before any device launch. The KMeans injected-init dimension check (`init.len() == k · n_features`) is PRESERVED in `fit` (data-DEPENDENT, not dropped). ✓
- **T-16-GUARDF64** (F64 guard): `crate::capability::guard_f64()?` preserved verbatim before every F64 upload in the three migrated PyO3 fits (PyKMeans / PyEmpiricalCovariance / PyLedoitWolf). ✓
- **T-16-ARM** (Fitted arm type): `AnyKMeans`/`AnyEmpiricalCovariance`/`AnyLedoitWolf` switched to `any_estimator_typestate!` so each fitted value is typed `T<f*, Fitted>`. ✓

## Known Stubs

None.

## Threat Flags

None — no new network/auth/file/schema surface; a trait-surface retrofit with byte-identical compute.

## Verification Environment Note

`cargo build -p mlrs-py --features cpu` is the cross-crate PyO3 gate (live Python pytest of the wheel is untestable here per the project memory "Python wheel untestable in env" — no maturin/pyarrow). The three oracle suites + the mlrs-py build are the compensating Rust gates; the Python-boundary behavior (Unfit-arm accessor → `not_fitted` → PyValueError) is unchanged from the pre-retrofit shells. The 2 remaining `mlrs-py` dead-code warnings are the pre-existing spectral.rs `Unfit`-arm field warnings (out of scope, documented in 16-05).

## Acceptance Evidence

- `cargo test --features cpu --test kmeans_test` → **7 passed** (exact-label permutation + 1e-5 centers/inertia f32+f64 incl. injected init, predict-consistency, WR-03 constant-feature, defaults_equal).
- `cargo test --features cpu --test empirical_covariance_test` → **7 passed** (full-rank + rank-deficient pinvh + assume_centered f32+f64, defaults_equal).
- `cargo test --features cpu --test ledoit_wolf_test` → **5 passed** (n=12 + n=40 f32+f64, defaults_equal).
- `cargo test --features cpu --test spectral_clustering_test` → **4 passed** (inner-KMeans regression after the typestate migration).
- `cargo build -p mlrs-py --features cpu` → Finished (2 pre-existing spectral.rs dead-code warnings only).
- `! grep -q 'crate::traits'` on kmeans.rs / spectral_clustering.rs / empirical_covariance.rs / ledoit_wolf.rs → ALL CLEAN.
- `! grep -q 'fn with_init\|fn with_opts'` on kmeans.rs → CLEAN (both folded into the builder).
- `any_estimator_typestate!` for AnyKMeans / AnyEmpiricalCovariance / AnyLedoitWolf (and now all three cluster.rs enums — no `any_estimator!` legacy remains in cluster.rs).
- D-03: per-commit `git diff` shows ZERO compute-line changes (signature/return/guard-call/reconstruction only).

## Self-Check: PASSED

- `crates/mlrs-algos/src/cluster/kmeans.rs` — FOUND, builds, 7 tests pass.
- `crates/mlrs-algos/src/cluster/spectral_clustering.rs` — FOUND, 100% off crate::traits, 4 tests pass.
- `crates/mlrs-algos/src/covariance/empirical_covariance.rs` — FOUND, builds, 7 tests pass.
- `crates/mlrs-algos/src/covariance/ledoit_wolf.rs` — FOUND, builds, 5 tests pass.
- `crates/mlrs-py/src/estimators/cluster.rs` / `covariance.rs` — FOUND, mlrs-py builds.
- Commit `21170f1` (KMeans) — FOUND.
- Commit `316eab9` (EmpiricalCovariance) — FOUND.
- Commit `56756bc` (LedoitWolf + covariance PyO3) — FOUND.
