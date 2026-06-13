---
phase: 05-distance-based-iterative-solver-estimators
plan: 07
subsystem: estimators
tags: [kmeans, dbscan, cluster, lloyd, kmeans-plus-plus, eps-core-mask, host-dfs, label-perm, predict-labels, injected-init, cpu-mlir, rocm-build]

# Dependency graph
requires:
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 01
    provides: "cluster/mod.rs + kmeans/dbscan_test.rs #[ignore] stubs; PredictLabels trait; InvalidK/InvalidEps/InvalidMinSamples AlgoError variants; kmeans/dbscan_{f32,f64}_seed42.npz fixtures (injected init D-09, noise=-1 D-06)"
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 03
    provides: "prims::kmeans::{lloyd_update (empty-cluster relocation), inertia, kmeanspp_sample} validated within 1e-5 vs sklearn"
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 04
    provides: "prims::dbscan::eps_core_mask → EpsCoreMask {is_core, counts, adjacency, neighbors(i)} (D-04 host readback)"
  - phase: 04-closed-form-estimators
    provides: "Ridge<F> Fit-shell precedent (validate-before-launch, device-resident fitted state, host accessors, host_to_f64 bitcast); ridge_test oracle precedent"
  - phase: 02-foundational-primitives
    provides: "prims::distance (squared Euclidean, sqrt=false), prims::reduce::argmin_rows (lowest-index tie-break)"
provides:
  - "mlrs_algos::cluster::kmeans::KMeans<F> — Fit (k-means++/injected init + Lloyd loop) + PredictLabels (i32 new-point assignment) + fit_predict; cluster_centers_/labels_(i32)/inertia_ device-resident"
  - "mlrs_algos::cluster::dbscan::DBSCAN<F> — Fit (device eps_core_mask + host index-ordered LIFO DFS per _dbscan_inner.pyx) + fit_predict; labels_(i32, noise=-1) + core_sample_indices_(i32); NO standalone predict (D-08)"
  - "kmeans_test.rs (5) + dbscan_test.rs (4) sklearn oracles GREEN on cpu f32+f64 (centers/inertia 1e-5 + labels up to permutation via best_match_accuracy==1.0; core_sample_indices_ exact)"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Estimator-owns-the-host-sequential-state idiom: DBSCAN's cluster expansion is the host index-ordered LIFO DFS (exactly _dbscan_inner.pyx) over the device-computed eps_core_mask adjacency — the device does the n² threshold/count (D-04), the estimator owns the inherently-sequential graph walk"
    - "Lloyd convergence reproduces sklearn's strict-OR-tol order (Pitfall 6): STRICT array_equal(labels,labels_old) break FIRST, THEN center_shift_tot <= tol·mean(var(X,axis=0)); one final assignment pass if not strict-converged so labels_ reflects the final centers"
    - "label_perm oracle (D-09): centers compared up to the SAME permutation as labels (best_mapping(fitted,sklearn) maps each fitted cluster id to its sklearn id before the centroid-row compare); inertia is permutation-invariant so compared directly"
    - "PredictLabels (i32), NOT Predict<F>, for KMeans.predict (D-08 — discrete cluster ids); DBSCAN implements neither (non-transductive, fit_predict only)"

key-files:
  created:
    - "crates/mlrs-algos/src/cluster/kmeans.rs (KMeans<F>: Fit + PredictLabels + fit_predict)"
    - "crates/mlrs-algos/src/cluster/dbscan.rs (DBSCAN<F>: Fit + fit_predict, host DFS)"
  modified:
    - "crates/mlrs-algos/src/cluster/mod.rs (added pub mod dbscan; pub mod kmeans;)"
    - "crates/mlrs-algos/tests/kmeans_test.rs (de-#[ignore]d: 5 real sklearn oracles from injected init)"
    - "crates/mlrs-algos/tests/dbscan_test.rs (de-#[ignore]d: 4 real sklearn oracles + invalid-hyperparam guards)"

key-decisions:
  - "KMeans labels stored i32 (D-06 u32→i32 widen) even though non-negative, so the PredictLabels trait surface is shared with DBSCAN's -1 noise; predict_labels assigns new points to fitted centers via the SAME distance+argmin_rows path (NOT Predict<F>, D-08)"
  - "Centers oracle compares up to the label permutation: best_mapping maps each fitted cluster id to its sklearn id, then the centroid ROWS are compared within 1e-5 — robust to KMeans' arbitrary cluster numbering (D-09). inertia_ is a permutation-invariant scalar so compared directly"
  - "DBSCAN re-validates eps>=0/min_samples>=1 at the ESTIMATOR boundary with the precise typed AlgoError (InvalidEps/InvalidMinSamples) even though the prim re-checks as ShapeMismatch — the host→estimator contract surfaces the specific hyperparameter error (ASVS V5, T-05-07-01)"
  - "DBSCAN carries F via PhantomData (no F-typed fitted state — labels/core indices are i32); the host DFS walks EpsCoreMask.neighbors(v) in ascending index order, expanding only from core points, label_num++ per seed — bit-for-bit _dbscan_inner.pyx (Pitfall 7 border determinism)"

patterns-established:
  - "Clustering-estimator oracle shape: load fixture (injected init for deterministic Lloyd), fit, compare permutation-invariant scalars (inertia) directly + permutation-variant arrays (centers/labels) via best_mapping/best_match_accuracy==1.0; f64 cpu-gated by skip_f64_with_log, f32 on rocm"

requirements-completed: [CLUSTER-01, CLUSTER-02]

# Metrics
duration: 6min
completed: 2026-06-13
---

# Phase 5 Plan 07: Clustering Estimators (KMeans + DBSCAN) Summary

**The two distance-based clustering estimators landed on the validated Wave-2 primitives: `KMeans<F>` (k-means++/injected init + the Lloyd loop reproducing sklearn's strict-OR-tol convergence, `cluster_centers_`/`labels_`(i32)/`inertia_`, `PredictLabels` new-point assignment + `fit_predict`) and `DBSCAN<F>` (device `eps_core_mask` + the host index-ordered LIFO DFS exactly per `_dbscan_inner.pyx`, `labels_`(noise=-1) + `core_sample_indices_`, no standalone predict) — both matching sklearn up to a label permutation within 1e-5 from the injected init, 5/5 + 4/4 oracles GREEN on cpu f32+f64 with the rocm(f32) test target building.**

## Performance

- **Duration:** ~6 min
- **Tasks:** 2 (both `type="auto"`)
- **Files modified:** 5 (2 created estimators + cluster/mod.rs + 2 de-ignored tests)

## Accomplishments
- Added `cluster/kmeans.rs` with `KMeans<F>` mirroring the `ridge.rs` shell: validate `1 <= n_clusters <= n_samples` → `AlgoError::InvalidK` BEFORE any launch (T-05-07-01); init centers from an INJECTED `k×d` array (`with_init`, D-09 oracle) or the `kmeanspp_sample` D²-weighted host sampler (`new`, D-09a, n_init=1 D-09b); the Lloyd loop assigns via `distance(sqrt=false)` + `argmin_rows`, updates via `lloyd_update` (empty-cluster relocation inside the prim), breaks on the STRICT `array_equal(labels,labels_old)` FIRST then on `center_shift_tot <= tol·mean(var(X,axis=0))` (Pitfall 6), runs one FINAL assignment pass if not strict-converged, and computes `inertia` via the prim. Stores `cluster_centers_`(F) + `labels_`(i32, D-06) + `inertia_` device-resident; implements `PredictLabels` (new-point nearest-centroid → i32 labels, NOT `Predict<F>` — D-08) + `fit_predict`.
- Added `cluster/dbscan.rs` with `DBSCAN<F>` mirroring the Fit shell: validate `eps >= 0` → `InvalidEps` and `min_samples >= 1` → `InvalidMinSamples` BEFORE launch (ASVS V5); call `eps_core_mask` (device n² distance + `<= eps²` threshold + self-inclusive count → host `is_core` + `n×n` adjacency, D-04); run the HOST index-ordered LIFO DFS EXACTLY per `_dbscan_inner.pyx` (labels init -1, seeds `0..n` in index order, skip labeled/non-core, push `i`, pop LIFO, label, expand only from core points via `neighbors(v)` ascending, `label_num++` per seed — Pitfall 7 deterministic border join); stores `labels_`(i32, noise=-1) + `core_sample_indices_`(i32 ascending) device-resident; `fit_predict`; NO standalone `predict` (D-08, documented).
- De-`#[ignore]`d `kmeans_test.rs` (5 tests): centers/labels up to permutation (`best_mapping` remaps centers, `best_match_accuracy == 1.0` for labels) + inertia within 1e-5 from the injected init, f32 + f64 (cpu-gated), plus a `predict_labels`-reproduces-fitted-labels consistency gate.
- De-`#[ignore]`d `dbscan_test.rs` (4 tests): `core_sample_indices_` EXACT integer-set match + `labels_` (noise=-1) up to permutation, f32 + f64 (cpu-gated), a `fit_predict` consistency + noise-present gate, and an invalid-hyperparameter guard (`InvalidEps`/`InvalidMinSamples`).
- Verified the gate: `cargo test --features cpu -p mlrs-algos --test kmeans_test --test dbscan_test` 9/9 GREEN (both f64 cases run on cpu); `cargo build -p mlrs-algos --features rocm --tests` GREEN.

## Task Commits

1. **Task 1: KMeans<F> k-means++ + Lloyd loop Fit/PredictLabels/fit_predict + kmeans oracle** — `01d8443` (feat)
2. **Task 2: DBSCAN<F> device core-mask + host index-ordered DFS Fit/fit_predict + dbscan oracle** — `450021d` (feat)

## Files Created/Modified
- `crates/mlrs-algos/src/cluster/kmeans.rs` — `KMeans<F>` (Fit + PredictLabels + fit_predict; `assign` helper = distance+argmin_rows; `with_init`/`new`; `cluster_centers`/`labels`/`inertia` accessors).
- `crates/mlrs-algos/src/cluster/dbscan.rs` — `DBSCAN<F>` (Fit + fit_predict; host LIFO DFS per `_dbscan_inner.pyx`; `labels`/`core_sample_indices` accessors; PhantomData<F>).
- `crates/mlrs-algos/src/cluster/mod.rs` — added `pub mod dbscan; pub mod kmeans;`.
- `crates/mlrs-algos/tests/kmeans_test.rs` — 5 real oracles (centers/labels permutation + inertia + predict consistency).
- `crates/mlrs-algos/tests/dbscan_test.rs` — 4 real oracles (core_sample_indices exact + labels permutation + fit_predict + invalid-hyperparam guard).

## Decisions Made
- **KMeans labels i32 + PredictLabels, not Predict<F> (D-08):** KMeans.predict returns discrete cluster ids, so the estimator implements `PredictLabels` (i32) sharing the trait surface with DBSCAN's -1 noise (D-06 widen). `predict_labels` reuses the exact `distance(sqrt=false)+argmin_rows` assignment path, so re-predicting the training X reproduces `labels_` bit-for-bit (a consistency gate).
- **Centers compared up to the label permutation (D-09):** KMeans cluster numbering is arbitrary, so the oracle uses `best_mapping(fitted_labels, sklearn_labels)` to map each fitted cluster id to its sklearn id, then compares the centroid ROWS within 1e-5. `inertia_` is a permutation-invariant scalar so it is compared directly. `best_match_accuracy == 1.0` is the strict label-permutation gate.
- **DBSCAN re-validates at the estimator boundary with the precise typed error:** the prim re-checks `eps`/`min_samples` as a `ShapeMismatch`, but the estimator surfaces `InvalidEps`/`InvalidMinSamples` so the host→estimator contract reports the specific hyperparameter violation before the prim is even called (ASVS V5).
- **DBSCAN host DFS is exactly `_dbscan_inner.pyx`:** seeds scanned in `0..n` index order, expansion only from core points, neighbors walked ascending, `label_num` incremented per seed — so a border point joins the FIRST core point reaching it in index order (Pitfall 7, deterministic). `F` is carried via `PhantomData` since DBSCAN keeps no F-typed fitted state.

## Deviations from Plan

None — plan executed exactly as written. (No deviation rules triggered. One trivial test-compile adjustment: `Result<&mut Self, _>::expect_err` requires the Ok type to be `Debug`, and `&mut DBSCAN<F>` is not, so the invalid-hyperparameter test maps the Ok arm to `()` before `expect_err` — a test-local idiom, not a source or behavior change. The estimator shells, validation, Lloyd convergence order, host DFS, label-perm oracle, and both feature gates all landed as specified.)

## Known Stubs

None. Both estimators are fully implemented and the oracle tests exercise real device output (centers/inertia from `lloyd_update`/`inertia`, labels from the real assignment + host DFS) — no hardcoded/empty values flow to the assertions.

## Issues Encountered

- One test-compile error: `expect_err` on `fit`'s `Result<&mut Self, AlgoError>` needs the Ok variant to be `Debug` (it is `&mut DBSCAN<F>`, which is not). Resolved by `.map(|_| ())` before `expect_err` — no source impact. The cubecl-cpu MLIR gate accepted the consumed prims (the estimators add only host orchestration over already-validated kernels), so no launch-time failure surfaced.

## Next Phase Readiness
- **CLUSTER-01 + CLUSTER-02 complete:** both clustering estimators match sklearn up to a label permutation within 1e-5 from the injected init, with `core_sample_indices_` exact. The Wave-2 primitive-first gate paid off — the estimators are thin host orchestration over the validated `lloyd_update`/`inertia`/`kmeanspp_sample`/`eps_core_mask` prims.
- **cluster/mod.rs now owns `pub mod kmeans; pub mod dbscan;`** (the 05-01 stub is filled). No `lib.rs`/`prims/mod.rs` edits, so sibling Wave-3 estimator plans (neighbors/linear) stay file-disjoint.
- No blockers. cpu(f64) full oracle + rocm(f32) test-target build both green.

## Threat Flags

None — no new network/auth/file surface. The two trust boundaries in the register are mitigated exactly as specified: T-05-07-01 (KMeans `n_clusters` / DBSCAN `eps`,`min_samples`) by validate-before-launch → typed `AlgoError::InvalidK`/`InvalidEps`/`InvalidMinSamples` BEFORE any prim launch; T-05-07-02 (KMeans empty cluster) handled inside `lloyd_update` (05-03, no NaN); T-05-07-03 (DBSCAN n²) handled inside `eps_core_mask` (05-04) — the estimator does not re-allocate. Zero new dependencies (T-05-07-SC accept).

## Self-Check: PASSED

- All created files verified present (`cluster/kmeans.rs`, `cluster/dbscan.rs`, both de-ignored tests, this SUMMARY).
- Both task commits verified in git history (`01d8443`, `450021d`).
- `cargo test --features cpu -p mlrs-algos --test kmeans_test --test dbscan_test` 9/9 green (KMeans 5/5 incl. both f64; DBSCAN 4/4 incl. f64 core-mask); `cargo build -p mlrs-algos --features rocm --tests` green; `cluster/mod.rs` carries both `pub mod` lines; `lib.rs`/`prims/mod.rs` untouched.

---
*Phase: 05-distance-based-iterative-solver-estimators*
*Completed: 2026-06-13*
