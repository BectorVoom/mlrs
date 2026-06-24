---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 08
subsystem: neighbors-estimators
tags: [typestate, builder-retrofit, neighbors, knn, accessor-traits, kneighbors, wave-8]
requires:
  - "typestate::{Fit, Predict, validate_geometry, Unfit, Fitted} (Plan 16-00)"
  - "typestate::{KNeighbors, PredictLabels, PredictProba} accessor traits (Plan 16-00 additions)"
  - "any_estimator_typestate! macro (dispatch.rs)"
  - "Shape-A recipe proven in ridge.rs (Plan 16-01) + accessor-trait composition proven in logistic.rs (Plan 16-03)"
provides:
  - "NearestNeighbors<F, S=Unfit> on the typestate surface (Shape-A; Fit + KNeighbors on Fitted — first end-to-end consumer of the Plan-00 KNeighbors accessor trait)"
  - "KNeighborsClassifier<F, S=Unfit> on the typestate surface (Shape-A; Fit + PredictLabels + PredictProba on Fitted)"
  - "KNeighborsRegressor<F, S=Unfit> on the typestate surface (Shape-A; Fit + Predict on Fitted)"
  - "NearestNeighborsBuilder / KNeighborsClassifierBuilder / KNeighborsRegressorBuilder (.n_neighbors(usize); build::<F>() -> Result<_, BuildError>)"
  - "PyNearestNeighbors / PyKNeighborsClassifier / PyKNeighborsRegressor on any_estimator_typestate! (Fitted arms)"
  - "neighbors.rs (PyO3) fully off mlrs_algos::traits — the legacy glob is removed"
affects:
  - "Plan 16-09..16-12 (bulk sweep continues; the KNeighbors accessor trait is now proven end-to-end)"
  - "Plan 16-12 (traits.rs deletion — NearestNeighbors/KNeighborsClassifier/KNeighborsRegressor no longer reference crate::traits; neighbors/ module fully migrated)"
tech-stack:
  added: []
  patterns:
    - "KNeighbors accessor-trait composition on Fitted: the (distances F, indices i32) tuple-returning KNeighbors trait (the Plan-00 addition) impls ONLY on T<F, Fitted> exactly like Predict — first end-to-end validation that the new KNeighbors accessor trait composes with the consuming-self Fit transition"
    - "usize-typed builder setter: n_neighbors is a count, so the builder setter is `.n_neighbors(v: usize)` (not the f64 A5 convention used for scalar hyperparams) — mirroring umap.rs's usize n_neighbors/n_components setters; no `as f32` cast existed to drop"
    - "Shape-A n_neighbors>=1 relocation: the data-INDEPENDENT n_neighbors>=1 check is added at build()->BuildError::InvalidNComponents (umap.rs precedent); the data-DEPENDENT k vs n_train check stays VERBATIM in the shared neighbor_indices core (T-16-V5), keeping fit/kneighbors byte-identical (tie-break-sensitive)"
    - "Shared-core byte-identical contract: NearestNeighbors, KNeighborsClassifier, and KNeighborsRegressor all call the same pub(crate) neighbor_indices core — left UNTOUCHED so the lowest-index tie-break + distances stay identical across all three migrated estimators (Pitfall 8)"
key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/neighbors/nearest.rs
    - crates/mlrs-algos/tests/nearest_neighbors_test.rs
    - crates/mlrs-algos/src/neighbors/classifier.rs
    - crates/mlrs-algos/tests/knn_classifier_test.rs
    - crates/mlrs-algos/src/neighbors/regressor.rs
    - crates/mlrs-algos/tests/knn_regressor_test.rs
    - crates/mlrs-algos/src/neighbors/mod.rs
    - crates/mlrs-py/src/estimators/neighbors.rs
decisions:
  - "n_neighbors builder setters are usize (NOT the f64 A5 convention): n_neighbors is a discrete count, and the umap.rs sibling already uses `pub fn n_neighbors(mut self, v: usize)`. There were no `as f32`/`as f64` casts on n_neighbors at the PyO3 arms to drop (the value flows through as usize end-to-end), so the with_opts-fold's `drop the as-cast` step is N/A here."
  - "Data-INDEPENDENT n_neighbors>=1 validation added at build()->BuildError::InvalidNComponents (mirroring umap.rs:417-423, which reuses InvalidNComponents for n_neighbors). The data-DEPENDENT k<1 || k>n_train guard in the shared neighbor_indices core (nearest.rs) is LEFT VERBATIM — defense-in-depth, NOT a moved check — because (a) it is tie-break/oracle-sensitive (the plan flags fit/kneighbors as byte-identical) and (b) k is a per-CALL argument to kneighbors, not only the constructed n_neighbors, so the fit-time guard cannot be dropped (T-16-V5)."
  - "train_shape()/n_classes() relocated to the Fitted impl and made infallible (return the value directly via .expect on the Some-by-construction field), dropping their old Result<_, NotFitted> signatures — the compile-time typestate replaces the runtime guard (D-03). The PyO3 n_classes forwarder is correspondingly un-.map_err'd."
  - "PyO3 neighbors.rs uses crate::global_pool().lock().expect(\"pool mutex\") for pool locking (NOT crate::lock_pool()); this pre-existing pool-lock idiom was LEFT UNCHANGED (out of scope — the plan migrates the trait surface + builder, D-13 keeps the GIL/pool/guard scaffolding intact). The three arms were migrated in three separate commits so cargo build -p mlrs-py never broke (Pitfall 3)."
  - "After the regressor (the file's last legacy consumer) migrated, the `use mlrs_algos::traits::{...}` glob in neighbors.rs was DELETED — neighbors.rs (PyO3) is now 100% on the typestate surface, imported under Typestate* aliases + called via UFCS to resolve the fit/predict/predict_labels/predict_proba/kneighbors method-name collisions."
metrics:
  duration: ~10m
  completed: 2026-06-24
  tasks: 3
  files: 8
status: complete
---

# Phase 16 Plan 08: Neighbors sweep — NearestNeighbors + KNeighborsClassifier + KNeighborsRegressor typestate retrofit — Summary

Migrated the three brute-force k-NN estimators onto the `mlrs_algos::typestate` surface, each under its own commit gated by its sklearn oracle suite AND `cargo build -p mlrs-py --features cpu`. All three are **Shape-A** (no pre-existing builder — single-arg `new(n_neighbors)` collapses to zero-arg `new()` + a `.n_neighbors(usize)` builder). This plan is the first end-to-end proof that the Plan-00 **`KNeighbors`** accessor trait (the `(distances, indices)` tuple-returning surface) composes on a `Fitted`-tagged estimator alongside the consuming-self `Fit` transition. The fit/kneighbors compute — critically the shared `neighbor_indices` core that carries the **lowest-index tie-break** (project memory: tie-break-sensitive) — is byte-identical to pre-retrofit (D-03); all three oracle suites stay green (13 tests: exact-index tie-break / exact-label / 1e-5), and `neighbors.rs` (PyO3) is now fully off the legacy `mlrs_algos::traits` glob — the neighbors module is completely migrated.

## What Was Built

### Task 1 — NearestNeighbors (Shape A; Fit + KNeighbors), commit `9c1e402`

`crates/mlrs-algos/src/neighbors/nearest.rs`:
- `struct NearestNeighbors<F, S = Unfit>` — added `_state: PhantomData<S>` as the only new field; `n_neighbors`/`x_train_`/`train_shape_` UNCHANGED (D-03).
- Replaced the arg-taking `new(n_neighbors)` with **zero-arg `new()`** on `impl<F> NearestNeighbors<F, Unfit>` (sklearn default `n_neighbors = 5`, read from `NN_DEFAULT_N_NEIGHBORS`). Added `builder()`, `into_builder()`, `hyperparams_eq()`, `impl Default`, and `n_neighbors()` (pre-fit read) on Unfit.
- New `NearestNeighborsBuilder { n_neighbors: usize }` with `.n_neighbors(usize)` setter; `Default` = `NearestNeighbors::<f64, Unfit>::new().into_builder()` (Pitfall 1 — single source); `build<F>()` validates the data-INDEPENDENT `n_neighbors >= 1` → `BuildError::InvalidNComponents`.
- Imports: dropped `use crate::traits::{Fit, KNeighbors}`; added `use crate::error::{AlgoError, BuildError}` and `use crate::typestate::{validate_geometry, Fit, Fitted, KNeighbors, Unfit}`.
- `impl Fit for NearestNeighbors<F, Unfit>` (`type Fitted = NearestNeighbors<F, Fitted>`): consuming `fit(self) -> Result<NearestNeighbors<F, Fitted>, AlgoError>`; the inline shape guard → `validate_geometry`; the device-resident `from_host` training-matrix staging unchanged; reconstructs field-by-field into the `Fitted` value (no more `take()`/`Ok(self)`).
- `n_neighbors()` + the now-infallible `train_shape() -> (usize, usize)` accessor moved onto `impl<F> NearestNeighbors<F, Fitted>` (`.expect` on the Some-by-construction field). `impl KNeighbors for NearestNeighbors<F, Fitted>` — the `kneighbors` body and the shared `pub(crate) fn neighbor_indices` core (incl. the lowest-index tie-break logic) copied VERBATIM (verified byte-identical).

`crates/mlrs-algos/tests/nearest_neighbors_test.rs`: trait import → typestate; the two `new(KNN_K)` + `.fit` sites (the oracle body + the `rejects_bad_k` guard test) → `builder().n_neighbors(KNN_K).build()?.fit(...)` consuming-self chain. Added `defaults_equal` (BLDR-01).

`crates/mlrs-py/src/estimators/neighbors.rs` (PyNearestNeighbors arm): `AnyNearestNeighbors` → `any_estimator_typestate!`; fit builds via `builder().n_neighbors(n).build::<f*>().map_err(build_err_to_py)?` then `TypestateFit::fit(...)` storing the `Fitted` value; `kneighbors_f32`/`kneighbors_f64` → `TypestateKNeighbors::kneighbors(est, ...)`. The legacy `traits` glob kept (still `Fit`/`Predict`/`PredictLabels`/`PredictProba` for the not-yet-migrated classifier/regressor arms), but loses `KNeighbors` (NN was its only consumer).

**Gate:** `cargo test --features cpu --test nearest_neighbors_test` → **5 passed** (distances 1e-5 + EXACT indices f32+f64, fixture_loads, rejects_bad_k, defaults_equal). `cargo build -p mlrs-py --features cpu` → clean.

### Task 2 — KNeighborsClassifier (Shape A; Fit + PredictLabels + PredictProba), commit `6525105`

`crates/mlrs-algos/src/neighbors/classifier.rs`:
- `struct KNeighborsClassifier<F, S = Unfit>` + `_state`; `n_neighbors`/`x_train_`/`train_shape_`/`y_class_`/`classes_`/`n_classes_` UNCHANGED.
- Zero-arg `new()` (`n_neighbors = 5`, `KNN_CLF_DEFAULT_N_NEIGHBORS`); `builder()`/`into_builder()`/`hyperparams_eq()`/`Default`/`n_neighbors()` on Unfit. `KNeighborsClassifierBuilder` (`.n_neighbors(usize)`; `build<F>()` validates `n_neighbors >= 1` → `BuildError::InvalidNComponents`).
- Import swap → `use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, PredictProba, Unfit}` + `BuildError`.
- Consuming-self `Fit::fit`: `validate_geometry` guard added; the CR-03 dense-class-remap (`classes_` sort/dedup + `binary_search` to dense index) is byte-identical; reconstructs into `Fitted`. `n_neighbors()` + the now-infallible `n_classes() -> usize` moved onto `KNeighborsClassifier<F, Fitted>`. `impl PredictProba` + `impl PredictLabels` moved onto `Fitted` (the `y_class_.ok_or(NotFitted)` → `.expect`); the neighbor-vote fraction loop (`inv_k` accumulation) + `argmax_rows` + the `classes_[col]` inverse-map copied verbatim.

`crates/mlrs-algos/tests/knn_classifier_test.rs`: trait → typestate; the `new(KNN_K)` + `.fit` chain → builder consuming-self; `clf.n_classes().expect("fitted")` → `clf.n_classes()` (now infallible). Added `defaults_equal`.

`crates/mlrs-py/src/estimators/neighbors.rs` (PyKNeighborsClassifier arm): `AnyKNeighborsClassifier` → `any_estimator_typestate!`; fit + UFCS `TypestateFit::fit`; `predict_labels` → `TypestatePredictLabels::predict_labels`; `predict_proba_f32/f64` → `TypestatePredictProba::predict_proba`; `n_classes` forwarder un-`.map_err`'d (infallible on Fitted). Added `PredictLabels`/`PredictProba` typestate aliases; dropped them from the legacy glob (NN/classifier no longer need them there).

**Gate:** `cargo test --features cpu --test knn_classifier_test` → **4 passed** (EXACT majority-vote labels + predict_proba 1e-5 f32+f64, fixture_loads, defaults_equal). `cargo build -p mlrs-py --features cpu` → clean.

### Task 3 — KNeighborsRegressor (Shape A; Fit + Predict), commit `f4aa334`

`crates/mlrs-algos/src/neighbors/regressor.rs`:
- `struct KNeighborsRegressor<F, S = Unfit>` + `_state`; `n_neighbors`/`x_train_`/`train_shape_`/`y_reg_` UNCHANGED.
- Zero-arg `new()` (`n_neighbors = 5`, `KNN_REG_DEFAULT_N_NEIGHBORS`); `builder()`/`into_builder()`/`hyperparams_eq()`/`Default`/`n_neighbors()` on Unfit. `KNeighborsRegressorBuilder` (`.n_neighbors(usize)`; `build<F>()` validates `n_neighbors >= 1`).
- Import swap → `use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit}` + `BuildError`.
- Consuming-self `Fit::fit`: `validate_geometry` guard; the `y.len() != n_train` target-shape check kept (data-DEPENDENT, stays in fit); the `y.to_host` + `from_host` staging unchanged; reconstructs into `Fitted`. `n_neighbors()` on `Fitted`. `impl Predict for KNeighborsRegressor<F, Fitted>`: the neighbor-mean loop (`acc += host_to_f64(y_reg[train_idx])` → `pred[q] = f64_to_host(acc * inv_k)`) copied verbatim; the `y_reg_.ok_or(NotFitted)` → `.expect`.

`crates/mlrs-algos/tests/knn_regressor_test.rs`: trait → typestate; the `new(KNN_K)` + `.fit` chain → builder consuming-self. Added `defaults_equal`.

`crates/mlrs-py/src/estimators/neighbors.rs` (PyKNeighborsRegressor arm): `AnyKNeighborsRegressor` → `any_estimator_typestate!`; fit + UFCS `TypestateFit::fit`; `predict_f32/f64` → `TypestatePredict::predict`. **This was the file's last legacy consumer** — the `use mlrs_algos::traits::{Fit, Predict};` glob was DELETED; neighbors.rs (PyO3) is now 100% on the typestate surface (imports the five `Typestate*` aliases + UFCS). `crates/mlrs-algos/src/neighbors/mod.rs` module-doc trait links retargeted `crate::traits::*` → `crate::typestate::*`.

**Gate:** `cargo test --features cpu --test knn_regressor_test` → **4 passed** (neighbor-mean predict 1e-5 f32+f64, fixture_loads, defaults_equal). `cargo build -p mlrs-py --features cpu` → clean.

## The KNeighbors Accessor-Trait Composition (validated for the remaining sweep plans)

This plan is the first end-to-end proof that the Plan-00 `KNeighbors` accessor trait — the only accessor trait that returns a `(DeviceArray<F>, DeviceArray<i32>)` tuple and takes a per-call `k: usize` — composes on a `Fitted` estimator. The pattern:

**Estimator side** — the accessor impl moves onto the `Fitted` sibling exactly like `Predict`, carrying NO `type Fitted`:
```rust
impl<F> KNeighbors<F> for NearestNeighbors<F, Fitted> {
    fn kneighbors(&self, pool, x, shape, k) -> Result<(DeviceArray<F>, DeviceArray<i32>), _> { /* verbatim */ }
}
```

**PyO3 side** — alias every consumed lifecycle/accessor trait and call via UFCS at the migrated arm (the five-way `fit`/`predict`/`predict_labels`/`predict_proba`/`kneighbors` method-name collision is why UFCS is mandatory):
```rust
use mlrs_algos::typestate::{
    Fit as TypestateFit, KNeighbors as TypestateKNeighbors, Predict as TypestatePredict,
    PredictLabels as TypestatePredictLabels, PredictProba as TypestatePredictProba,
};
TypestateKNeighbors::kneighbors(est, &mut pool, &xd, (rows, cols), k)
```
Switch the `Any*` enum to `any_estimator_typestate!`. n_neighbors flows as `usize` end-to-end (no `as`-cast to drop — distinct from the f64-setter linear family). Once the file's LAST legacy-trait estimator migrates, DELETE the `mlrs_algos::traits` glob.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Corrected the test-file paths in the plan frontmatter**
- **Found during:** Tasks 1–3 (read_first resolution).
- **Issue:** The plan's `files_modified` / task `<files>` named `crates/mlrs-algos/tests/kneighbors_classifier_test.rs` and `crates/mlrs-algos/tests/kneighbors_regressor_test.rs`, which do not exist. The actual fixtures-driven oracle files are `knn_classifier_test.rs` and `knn_regressor_test.rs` (the NearestNeighbors one, `nearest_neighbors_test.rs`, matched). The `<verify>` `--test` targets in the plan use the WRONG names (`kneighbors_*`) and would have failed.
- **Fix:** Migrated and verified against the real files `knn_classifier_test.rs` / `knn_regressor_test.rs` / `nearest_neighbors_test.rs` (the `cargo test --test` invocations use the real crate test names, which are the file stems).
- **Files modified:** `crates/mlrs-algos/tests/knn_classifier_test.rs`, `crates/mlrs-algos/tests/knn_regressor_test.rs`.
- **Commits:** `6525105`, `f4aa334`.

No other deviations. All three estimators followed the Shape-A recipe (ridge.rs / umap.rs exemplar). The `n_neighbors >= 1` build() check is the documented data-INDEPENDENT relocation (umap.rs precedent for the same `n_neighbors` param); the data-DEPENDENT `k vs n_train` guard stays verbatim in the shared `neighbor_indices` core (T-16-V5). The three fit/kneighbors/predict compute paths are byte-identical (verified: the `neighbor_indices` core diffs IDENTICAL; the classifier vote/argmax and regressor mean loops show ZERO compute-line changes — only signature, return, guard-call, and struct-reconstruction edits).

## Authentication Gates

None.

## Threat Mitigations (from the plan's threat_model)

- **T-16-V5** (neighbors fit + kneighbors geometry guard + relocated validation): `validate_geometry(x, shape)?` is at the TOP of all three ported `fit`s, before any device staging. The shared `neighbor_indices` core's `k < 1 || k > n_train` → `AlgoError::InvalidK` guard is LEFT VERBATIM in the kneighbors path (validate-before-launch, ASVS V5) — NOT dropped. The data-INDEPENDENT `n_neighbors >= 1` half is ADDED at `build()` → `BuildError::InvalidNComponents` (defense in depth). The `y.len() != n_train` target-shape check stays in the classifier/regressor fit. ✓
- **T-16-GUARDF64** (F64 guard): `crate::capability::guard_f64()?` preserved verbatim before every F64 upload in the three migrated PyO3 fits. ✓
- **T-16-ARM** (Fitted arm type): `AnyNearestNeighbors`/`AnyKNeighborsClassifier`/`AnyKNeighborsRegressor` switched to `any_estimator_typestate!` so each fitted value is typed `T<f*, Fitted>` — no `Unfit` value stored in a fitted arm. ✓

## Known Stubs

None.

## Threat Flags

None — no new network/auth/file/schema surface introduced; a trait-surface retrofit with byte-identical compute.

## Verification Environment Note

`cargo build -p mlrs-py --features cpu` is the cross-crate PyO3 gate (live Python pytest of the wheel is untestable here per the project memory "Python wheel untestable in env" — no maturin/pyarrow). The three oracle suites + the mlrs-py build are the compensating Rust gates; the Python-boundary behavior (Unfit-arm accessor → `not_fitted` → PyValueError) is unchanged from the pre-retrofit shells.

## Acceptance Evidence

- `cargo test --features cpu --test nearest_neighbors_test` → **5 passed** (distances 1e-5 + EXACT lowest-index tie-break indices f32+f64, rejects_bad_k, defaults_equal).
- `cargo test --features cpu --test knn_classifier_test` → **4 passed** (EXACT majority-vote labels + predict_proba 1e-5).
- `cargo test --features cpu --test knn_regressor_test` → **4 passed** (neighbor-mean predict 1e-5).
- `cargo build -p mlrs-py --features cpu` → Finished (2 pre-existing spectral.rs dead-code warnings only, out of scope).
- `! grep -q 'crate::traits'` on nearest.rs / classifier.rs / regressor.rs → all clean; `grep -rln 'crate::traits' crates/mlrs-algos/src/neighbors/` → empty.
- `grep -c 'typestate::KNeighbors'` on nearest.rs > 0 (impl + import on the typestate surface).
- No `mlrs_algos::traits` import remains in PyO3 neighbors.rs (only a doc-comment mention; the active glob is deleted).
- D-03: the `neighbor_indices` shared core diffs BYTE-IDENTICAL; the classifier vote/argmax and regressor mean loops show ZERO compute-line changes.

## Self-Check: PASSED

- `crates/mlrs-algos/src/neighbors/nearest.rs` — FOUND, builds, 5 tests pass.
- `crates/mlrs-algos/src/neighbors/classifier.rs` — FOUND, builds, 4 tests pass.
- `crates/mlrs-algos/src/neighbors/regressor.rs` — FOUND, builds, 4 tests pass.
- `crates/mlrs-py/src/estimators/neighbors.rs` — FOUND, mlrs-py builds (legacy traits glob removed).
- Commit `9c1e402` (NearestNeighbors) — FOUND.
- Commit `6525105` (KNeighborsClassifier) — FOUND.
- Commit `f4aa334` (KNeighborsRegressor) — FOUND.
</content>
</invoke>
