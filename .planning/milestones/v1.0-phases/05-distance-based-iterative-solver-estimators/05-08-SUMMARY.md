---
phase: 05-distance-based-iterative-solver-estimators
plan: 08
subsystem: algos
tags: [neighbors, knn, nearest-neighbors, classifier, regressor, brute-force, top-k, oracle, cpu, rocm, i32-indices]

# Dependency graph
requires:
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 01
    provides: "KNeighbors/PredictLabels/PredictProba traits + InvalidK AlgoError + neighbors/mod.rs stub + knn_{f32,f64}_seed42.npz fixtures + i32 DeviceArray (D-06) + 3 #[ignore] estimator test stubs"
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 02
    provides: "prims::topk::top_k — validated select-k (distances + u32 indices, lowest-index tie, sqrt at boundary)"
  - phase: 02-foundational-primitives
    provides: "prims::distance (squared-Euclidean), reduce::argmax_rows (lowest-index tie)"
provides:
  - "mlrs_algos::neighbors::nearest::NearestNeighbors<F> — Fit + KNeighbors (k sqrt-Euclidean distances + i32 indices, NEIGH-01)"
  - "mlrs_algos::neighbors::classifier::KNeighborsClassifier<F> — Fit + PredictLabels (majority vote) + PredictProba (per-class fraction, NEIGH-02)"
  - "mlrs_algos::neighbors::regressor::KNeighborsRegressor<F> — Fit + Predict<F> (neighbor mean, NEIGH-03)"
  - "neighbor_indices() shared kneighbors core (validate-before-launch + distance + top_k + u32→i32) reused by all three estimators"
  - "3 activated oracle tests (10 cases) GREEN on cpu(f64) within 1e-5 + index/predict-exact vs sklearn; rocm(f32) test-target build green"
affects: [05-09, 05-10, 05-11]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Estimator-family code reuse: NearestNeighbors exposes a pub(crate) neighbor_indices() core (validate k+geometry → distance(sqrt=false) → top_k(sqrt=true) → u32→i32 cast, returning the host u32 buffer too) so the classifier vote and regressor mean run on EXACTLY the same neighbor set / tie-break (Pitfall 8)"
    - "Vote/mean host-combine over the small n_query×k gather (uniform 1/k weights); the heavy distance+select stays on-device, materializing only the index buffer (D-03 boundary)"
    - "predict_labels = argmax_rows(predict_proba) over the contiguous [0,n_classes) label space — the column index IS the class id, lowest-class-index tie inherited from argmax_rows"

key-files:
  created:
    - "crates/mlrs-algos/src/neighbors/nearest.rs"
    - "crates/mlrs-algos/src/neighbors/classifier.rs"
    - "crates/mlrs-algos/src/neighbors/regressor.rs"
  modified:
    - "crates/mlrs-algos/src/neighbors/mod.rs (pub mod nearest/classifier/regressor)"
    - "crates/mlrs-algos/tests/nearest_neighbors_test.rs (de-#[ignore], real oracle + bad-k guard)"
    - "crates/mlrs-algos/tests/knn_classifier_test.rs (de-#[ignore], predict exact + proba 1e-5)"
    - "crates/mlrs-algos/tests/knn_regressor_test.rs (de-#[ignore], predict 1e-5)"

key-decisions:
  - "Shared neighbor_indices() core lives in nearest.rs (pub(crate)); classifier/regressor import it so all three estimators agree on neighbors, distances, and the lowest-index tie-break (no re-derivation, no drift)"
  - "n_classes inferred as max(y_class)+1 over the contiguous label space (sklearn classes_ are [0,n)); proba columns indexed directly by class id"
  - "predict (labels) is argmax_rows of the proba matrix rather than a separate vote count — guarantees the exact lowest-class-index tie convention already validated for argmax_rows, and proba is the gauge for the 1e-5 gate"
  - "fit re-stages X via from_host so the estimator owns its training buffer independent of the caller's input handle (mirrors device-residency ownership)"

patterns-established:
  - "KNN estimator family: one validated kneighbors core, three thin Fit + (KNeighbors | PredictLabels+PredictProba | Predict) wrappers"

requirements-completed: [NEIGH-01, NEIGH-02, NEIGH-03]

# Metrics
duration: 18min
completed: 2026-06-13
---

# Phase 5 Plan 08: KNN Estimator Family Summary

**The three brute-force neighbors estimators (NEIGH-01/02/03) landed on the validated top-k primitive: `NearestNeighbors.kneighbors` returns k sqrt-Euclidean distances + i32 indices within 1e-5 (lowest-index tie), and `KNeighborsClassifier` (majority vote + per-class fraction) / `KNeighborsRegressor` (neighbor mean) reuse the exact same neighbor set — all three GREEN vs sklearn on cpu(f64) and building on rocm(f32).**

## Performance
- **Duration:** ~18 min
- **Tasks:** 2
- **Files modified:** 8 (3 source created, 1 module index, 3 tests activated)

## Accomplishments
- `NearestNeighbors<F>`: `Fit` stores the device-resident training matrix (y ignored — unsupervised); `KNeighbors::kneighbors` validates `1 <= k <= n_train` → `InvalidK` + query geometry BEFORE launch, then `distance(xq, X, sqrt=false)` → `top_k(.., k, sqrt=true)`, returning the k true sqrt-Euclidean distances (F) + i32 neighbor indices (u32→i32 host cast, D-06).
- Factored the kneighbors path into a `pub(crate) neighbor_indices()` core that ALSO returns the host u32 index buffer, so the classifier vote and regressor mean gather neighbor targets without a second round-trip and on EXACTLY the same neighbor set / tie-break (Pitfall 8).
- `KNeighborsClassifier<F>`: `Fit` stores X + i32 class targets (n_classes = max+1); `PredictProba` forms the per-class neighbor FRACTION (uniform 1/k); `PredictLabels` is `argmax_rows(proba)` with the lowest-class-index tie-break.
- `KNeighborsRegressor<F>`: `Fit` stores X + F targets; `Predict<F>` returns the MEAN of the k neighbor targets (uniform).
- Activated all three oracle tests (10 cases) — `nearest_neighbors` (distances 1e-5 + indices exact + bad-k InvalidK guard), `knn_classifier` (predict exact + proba 1e-5), `knn_regressor` (predict 1e-5) — GREEN on cpu f32+f64; rocm `--tests` build green.

## Task Commits
1. **Task 1: NearestNeighbors<F> (Fit + KNeighbors) + oracle (NEIGH-01)** — `82dc351` (feat)
2. **Task 2: KNeighborsClassifier + KNeighborsRegressor + oracles (NEIGH-02/03)** — `cb3fd12` (feat)

## Files Created/Modified
- `crates/mlrs-algos/src/neighbors/nearest.rs` — `NearestNeighbors<F>` + the shared `neighbor_indices()` core.
- `crates/mlrs-algos/src/neighbors/classifier.rs` — `KNeighborsClassifier<F>` (PredictProba + PredictLabels).
- `crates/mlrs-algos/src/neighbors/regressor.rs` — `KNeighborsRegressor<F>` (Predict<F>).
- `crates/mlrs-algos/src/neighbors/mod.rs` — `pub mod nearest; pub mod classifier; pub mod regressor;` (lib.rs untouched — owned by the Wave-0 scaffold).
- `crates/mlrs-algos/tests/{nearest_neighbors,knn_classifier,knn_regressor}_test.rs` — de-`#[ignore]`d real sklearn oracles.

## Decisions Made
- **One validated kneighbors core, three thin wrappers:** `neighbor_indices()` (pub(crate), in nearest.rs) is the single site that validates k+geometry and composes distance+top_k; classifier/regressor import it so neighbors, distances, and the lowest-index tie-break never drift between the three estimators.
- **`predict_labels = argmax_rows(predict_proba)`:** rather than a parallel vote-count, the label is the argmax of the proba row — reusing the already-validated lowest-class-index tie convention of `argmax_rows`, with proba as the 1e-5 gate.
- **n_classes = max(y)+1 over the contiguous [0,n) label space** (sklearn `classes_`); proba columns indexed directly by class id.

## Deviations from Plan

None — plan executed exactly as written. No deviation rules triggered: the trait impls, the InvalidK validate-before-launch guard, the u32→i32 index cast, and all three oracle activations landed as specified, GREEN on cpu(f64) within 1e-5 and index/predict-exact, with the rocm(f32) test-target build green.

## Known Stubs

None. All three estimators are fully implemented; the oracle tests exercise real device output (distance+top_k+gather) against the committed sklearn fixtures — no hardcoded/empty values flow to the assertions.

## Threat Flags

None — no new network/auth/file surface. The only trust boundary (caller → kneighbors/predict `k` + query geometry, T-05-08-01) is mitigated exactly as the threat register specified: `neighbor_indices()` validates `1 <= k <= n_train` → `AlgoError::InvalidK` and `n_query*d == xq.len()` + feature-count match BEFORE any prim launch, and the `top_k` prim re-validates (05-02). Tie-break determinism (T-05-08-02) is the lowest-index from top_k + lowest-class-index from argmax_rows, pinned by the index-exact / predict-exact oracle assertions.

## Next Phase Readiness
- **NEIGH-01/02/03 complete** on the cpu(f64)+rocm(f32) gate; the neighbors family is the validated reference for any later PyO3 wrapping (Phase 6) of the Fit/KNeighbors/PredictLabels/PredictProba/Predict surface.
- `neighbors/mod.rs` now registers all three estimators; `lib.rs` untouched, so sibling Wave-3 estimator plans stay file-disjoint.
- No blockers.

## Self-Check: PASSED
- All created files verified present (nearest.rs, classifier.rs, regressor.rs, this SUMMARY).
- Both task commits verified in git history (`82dc351`, `cb3fd12`).
- `cargo test --features cpu -p mlrs-algos --test nearest_neighbors_test --test knn_classifier_test --test knn_regressor_test` 10/10 green (incl. f64 + bad-k guard); `cargo build -p mlrs-algos --features rocm --tests` green; lib.rs untouched.
