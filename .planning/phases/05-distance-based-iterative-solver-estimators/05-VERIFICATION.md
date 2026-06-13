---
phase: 05-distance-based-iterative-solver-estimators
verified: 2026-06-13T08:00:00Z
status: human_needed
score: 4/4
overrides_applied: 1
overrides:
  - must_have: "LogisticRegression binary predict_proba matching scikit-learn's lbfgs solver within 1e-5"
    reason: "D-12 user-approved tradeoff: estimator uses symmetric-multinomial for ALL K; sklearn K=2 uses binomial sigmoid (differs ~3.6e-3 under L2). Binary validates against scipy self-reference at strict 1e-5; multiclass IS sklearn-faithful at strict 1e-5. Documented in REQUIREMENTS.md LINEAR-05 and 05-10-SUMMARY."
    accepted_by: "BectorVoom"
    accepted_at: "2026-06-13T04:00:00Z"
human_verification:
  - test: "Run the full phase-5 estimator + prim test suite and confirm all tests pass"
    expected: "All 30+ oracle tests GREEN on cpu(f64); rocm(f32) build targets compile"
    why_human: "Context provides a post-fix targeted run was GREEN (REALEXIT=0), but the sandbox disk budget prevents re-running compilation here. The context is authoritative evidence but a human should confirm suite is still clean on HEAD."
---

# Phase 5: Distance-Based & Iterative-Solver Estimators — Verification Report

**Phase Goal:** A data scientist can fit the clustering, neighbors, and iterative-solver linear models matching scikit-learn within tolerance (up to label permutation where applicable), completing the v1 algorithm surface in Rust.
**Verified:** 2026-06-13T08:00:00Z
**Status:** human_needed (all automated truths VERIFIED; one human confirmation item for test-suite run)
**Re-verification:** No — initial verification.

---

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | KMeans exposes cluster_centers_/labels_/inertia_, predicts new points up to label permutation; DBSCAN exposes labels_ (noise=-1) and core_sample_indices_ up to label permutation | VERIFIED | `crates/mlrs-algos/src/cluster/kmeans.rs` — full Fit + PredictLabels impl with injected-init Lloyd loop, sklearn-exact convergence (Pitfall 6), device-resident fitted state. `crates/mlrs-algos/src/cluster/dbscan.rs` — Fit + fit_predict, DFS exactly matches `_dbscan_inner.pyx`. Oracle tests in `kmeans_test.rs` (5 tests) and `dbscan_test.rs` (4 tests) load committed `.npz` fixtures and assert within 1e-5 / label-permutation with no `#[ignore]`. |
| 2 | NearestNeighbors returns k nearest distances/indices within 1e-5; KNeighborsClassifier predict/predict_proba and KNeighborsRegressor predict match sklearn within tolerance | VERIFIED | `crates/mlrs-algos/src/neighbors/` — `nearest.rs` (Fit + KNeighbors), `classifier.rs` (Fit + PredictLabels + PredictProba), `regressor.rs` (Fit + Predict). Shared `neighbor_indices` core (brute-force on validated top_k prim). Oracle tests in `nearest_neighbors_test.rs` (4 tests), `knn_classifier_test.rs` (3 tests), `knn_regressor_test.rs` (3 tests) — all active (no `#[ignore]`), fixtures committed. |
| 3 | Lasso and ElasticNet share a coordinate-descent kernel (Lasso = l1_ratio==1) and produce coef_ matching sklearn's CD solver within tolerance including exact sparsity | VERIFIED | `crates/mlrs-algos/src/linear/lasso.rs` — thin wrapper over `cd_fit` with l1_ratio=1. `crates/mlrs-algos/src/linear/elastic_net.rs` — full Fit + Predict delegating to shared `cd_fit`. `crates/mlrs-algos/src/linear/coordinate_descent.rs` — shared host helper with penalty mapping (Pitfall 1), center-then-solve intercept (D-13). Oracle tests in `lasso_test.rs` (3 tests) and `elastic_net_test.rs` (3 tests) — active, fixtures committed. |
| 4 | LogisticRegression (L-BFGS, symmetric softmax) handles binary and multiclass with predict/predict_proba matching reference lbfgs solver within tolerance | PASSED (override) | `crates/mlrs-algos/src/linear/logistic.rs` — full Fit + PredictLabels + PredictProba via `lbfgs_minimize` over `softmax_loss_grad` kernel. Binary = symmetric-multinomial self-reference (D-12 approved override); multiclass = sklearn-faithful strict 1e-5 on cpu(f64). Oracle tests in `logistic_test.rs` (5 tests: fixture_loads + 4 predict_proba gates). Override: Binary validates at strict 1e-5 against scipy on the exact D-12 objective, NOT sklearn binomial (differs ~3.6e-3 under L2 — user-approved, documented in REQUIREMENTS.md LINEAR-05). |

**Score:** 4/4 truths verified (1 via override)

---

### Code-Review Blocker Fix Confirmation

Both BLOCKERs from 05-REVIEW.md are confirmed fixed in the codebase:

**CR-01 (Empty-cluster relocation):** Fixed. `prims/kmeans.rs::lloyd_update` signature now accepts `dist_to_assigned: &[f64]` (per-sample squared distance to assigned center). The relocation block implements sklearn's `_relocate_empty_clusters_dense` exactly: sort indices descending by `dist_to_assigned`, assign the n_empty farthest to empty clusters, mutate sums + counts for donor and recipient. The estimator (`kmeans.rs`) pre-computes `inertia_rows_host` before each `lloyd_update` call and passes it. The `lloyd_test.rs::lloyd_relocates_empty_cluster` test asserts the relocated center is EXACTLY the farthest-from-assigned-center sample row (T-05-03-02).

**CR-02 (k-means++ draw modulo bias and FP fall-through):** Fixed. `SplitMix64::next_below` uses rejection sampling (`zone = u64::MAX - (u64::MAX % bound); loop { if v < zone { return v % bound; } }`), eliminating modulo bias. The weighted draw now falls back to `min_d2.iter().rposition(|&w| w > 0.0)` when the picked index has non-positive weight, closing the FP-rounding fall-through.

**WR-01 through WR-03 fixes:**
- WR-03 (usize→u32 guard): `guard_u32` function added to `prims/kmeans.rs`, called for n, d, k before every kernel launch. Same pattern in other prim modules.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-algos/src/cluster/kmeans.rs` | KMeans Fit + PredictLabels + fit_predict | VERIFIED | Full impl, 436 lines, Fit and PredictLabels traits implemented |
| `crates/mlrs-algos/src/cluster/dbscan.rs` | DBSCAN Fit + fit_predict | VERIFIED | Full impl, 247 lines, LIFO DFS exact per sklearn |
| `crates/mlrs-algos/src/neighbors/nearest.rs` | NearestNeighbors Fit + KNeighbors | VERIFIED | Full impl with shared `neighbor_indices` core |
| `crates/mlrs-algos/src/neighbors/classifier.rs` | KNeighborsClassifier Fit + PredictLabels + PredictProba | VERIFIED | Full impl |
| `crates/mlrs-algos/src/neighbors/regressor.rs` | KNeighborsRegressor Fit + Predict | VERIFIED | Full impl |
| `crates/mlrs-algos/src/linear/lasso.rs` | Lasso Fit + Predict (l1_ratio=1 wrapper) | VERIFIED | Thin wrapper, delegates to cd_fit, no duplicate CD loop |
| `crates/mlrs-algos/src/linear/elastic_net.rs` | ElasticNet Fit + Predict | VERIFIED | Full impl delegating to shared cd_fit |
| `crates/mlrs-algos/src/linear/coordinate_descent.rs` | Shared cd_fit helper | VERIFIED | Penalty mapping, center-then-solve, cd_solve delegation |
| `crates/mlrs-algos/src/linear/logistic.rs` | LogisticRegression Fit + PredictLabels + PredictProba | VERIFIED | Full impl via lbfgs_minimize + softmax_loss_grad |
| `crates/mlrs-algos/src/traits.rs` | PredictLabels, KNeighbors, PredictProba traits | VERIFIED | All three traits defined and re-exported from lib.rs |
| `crates/mlrs-backend/src/prims/kmeans.rs` | lloyd_update (CR-01 fix), kmeanspp_sample (CR-02 fix), inertia, inertia_rows_host | VERIFIED | CR-01 relocation exact; CR-02 rejection sampling; guard_u32 WR-03 |
| `crates/mlrs-backend/src/prims/dbscan.rs` | eps_core_mask | VERIFIED | Exists and wired |
| `crates/mlrs-backend/src/prims/topk.rs` | top_k | VERIFIED | Exists and wired |
| `crates/mlrs-backend/src/prims/coordinate_descent.rs` | cd_solve | VERIFIED | Exists and wired |
| `crates/mlrs-backend/src/prims/lbfgs.rs` | lbfgs_minimize, softmax_loss_grad | VERIFIED | Exists and wired |
| `crates/mlrs-kernels/src/kmeans.rs` | centroid_sumcount, inertia_rows kernels | VERIFIED | Exists |
| `crates/mlrs-kernels/src/dbscan.rs` | eps_core_count kernel | VERIFIED | Exists |
| `crates/mlrs-kernels/src/topk.rs` | select_k kernel | VERIFIED | Exists |
| `crates/mlrs-kernels/src/coordinate.rs` | col_dot, residual_axpy, enet_gap kernels | VERIFIED | Exists |
| `crates/mlrs-kernels/src/lbfgs.rs` | softmax_loss_grad kernel | VERIFIED | Exists |
| `tests/fixtures/kmeans_{f32,f64}_seed42.npz` | Committed sklearn fixtures with injected init | VERIFIED | Both files present |
| `tests/fixtures/dbscan_{f32,f64}_seed42.npz` | Committed sklearn fixtures | VERIFIED | Both files present |
| `tests/fixtures/knn_{f32,f64}_seed42.npz` | Committed sklearn fixtures | VERIFIED | Both files present |
| `tests/fixtures/lasso_{f32,f64}_seed42.npz` | Committed sklearn fixtures | VERIFIED | Both files present |
| `tests/fixtures/elastic_net_{f32,f64}_seed42.npz` | Committed sklearn fixtures | VERIFIED | Both files present |
| `tests/fixtures/logistic_binary_{f32,f64}_seed42.npz` | Self-reference fixtures (D-12) | VERIFIED | Both files present |
| `tests/fixtures/logistic_multi_{f32,f64}_seed42.npz` | sklearn-faithful multiclass fixtures | VERIFIED | Both files present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `kmeans.rs` | `prims::kmeans` | `use mlrs_backend::prims::kmeans::{inertia, inertia_rows_host, kmeanspp_sample, lloyd_update}` | WIRED | Import present; all four functions called in fit() |
| `kmeans.rs` | `prims::distance`, `prims::reduce::argmin_rows` | `use mlrs_backend::prims::distance::distance` | WIRED | Used in `Self::assign()` which is called 2×/iter + final pass |
| `dbscan.rs` | `prims::dbscan::eps_core_mask` | `use mlrs_backend::prims::dbscan::eps_core_mask` | WIRED | Called in `fit()` |
| `nearest.rs` | `prims::distance`, `prims::topk::top_k` | imports + call in `kneighbors()` | WIRED | Both called in `neighbor_indices()` |
| `classifier.rs`, `regressor.rs` | `neighbors::nearest::neighbor_indices` | `use crate::neighbors::nearest::neighbor_indices` | WIRED | Called in predict paths |
| `lasso.rs` | `linear::coordinate_descent::cd_fit` | `use crate::linear::coordinate_descent::{cd_fit, ...}` | WIRED | l1_ratio=1 passed |
| `elastic_net.rs` | `linear::coordinate_descent::cd_fit` | same | WIRED | l1_ratio from struct |
| `coordinate_descent.rs` | `prims::coordinate_descent::cd_solve` | `use mlrs_backend::prims::coordinate_descent::cd_solve` | WIRED | Called after centering |
| `logistic.rs` | `prims::lbfgs::{lbfgs_minimize, softmax_loss_grad, LBFGS_FTOL, LBFGS_MAXLS}` | imports | WIRED | Used in fit() |
| memory_gate_test.rs | `prims::coordinate_descent::cd_solve`, `prims::lbfgs::lbfgs_minimize` | direct imports | WIRED | Phase-5 gates at lines 41, 47 |

### Data-Flow Trace (Level 4)

All six estimators store fitted state as device-resident `DeviceArray` (confirmed present in struct fields, `None` until `fit`, set in `fit()` return). Predict paths read from fitted device arrays via GEMM or gather. No hardcoded empty returns found.

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `kmeans.rs::cluster_centers_` | `centers: DeviceArray` | Lloyd loop over device distance + centroid-sumcount kernel | Yes — device kernel + host mean divide | FLOWING |
| `kmeans.rs::labels_` | `labels_i32: Vec<i32>` | argmin_rows over distance matrix | Yes | FLOWING |
| `kmeans.rs::inertia_` | `inertia_val: F` | `inertia()` prim | Yes | FLOWING |
| `dbscan.rs::labels_` | `labels: Vec<i32>` | host DFS over device adjacency | Yes | FLOWING |
| `logistic.rs::coef_` + `intercept_` | extracted from `theta` after `lbfgs_minimize` | L-BFGS over `softmax_loss_grad` kernel | Yes | FLOWING |
| `lasso.rs::coef_` + `intercept_` | from `cd_fit` return | `cd_solve` over centered design | Yes | FLOWING |

### Behavioral Spot-Checks

Step 7b: SKIPPED — sandbox disk budget prohibits recompilation. Context provides documented post-fix targeted run (all 16 phase-5 binaries REALEXIT=0 on cpu). This is accepted as the behavioral evidence.

### Probe Execution

No `probe-*.sh` files declared or present for this phase.

---

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|---------------|-------------|--------|----------|
| LINEAR-03 | 05-05, 05-09 | Lasso CD coef_ sparse within 1e-5 | SATISFIED | `lasso.rs` exists, delegates to `cd_fit(l1_ratio=1)`, oracle tests active with fixtures |
| LINEAR-04 | 05-05, 05-09 | ElasticNet CD coef_ within 1e-5 | SATISFIED | `elastic_net.rs` exists, shared `cd_fit`, oracle tests active |
| LINEAR-05 | 05-06, 05-10 | LogisticRegression L-BFGS binary+multiclass predict_proba | SATISFIED (with D-12 caveat) | Binary = scipy self-reference at 1e-5; multiclass = sklearn-faithful 1e-5; per REQUIREMENTS.md and user-approved override |
| CLUSTER-01 | 05-03, 05-07 | KMeans k-means++ init, cluster_centers_/labels_/inertia_, predict | SATISFIED | `kmeans.rs` full impl + prim layer + oracle tests green |
| CLUSTER-02 | 05-04, 05-07 | DBSCAN eps/min_samples, labels_ noise=-1, core_sample_indices_ | SATISFIED | `dbscan.rs` full impl + prim layer + oracle tests green |
| NEIGH-01 | 05-02, 05-08 | NearestNeighbors brute-force kneighbors within 1e-5 | SATISFIED | `nearest.rs` + top_k prim + oracle tests |
| NEIGH-02 | 05-08 | KNeighborsClassifier predict/predict_proba | SATISFIED | `classifier.rs` + oracle tests |
| NEIGH-03 | 05-08 | KNeighborsRegressor predict | SATISFIED | `regressor.rs` + oracle tests |

All 8 phase-5 requirement IDs are marked `[x] Complete` in REQUIREMENTS.md traceability table.

---

### Anti-Patterns Found

No TBD / FIXME / XXX debt markers found in any phase-5 source or test file. No stub return patterns (`return null`, `return {}`, placeholder text) found. All test functions are active (no `#[ignore]` in estimator test files). All traits have substantive implementations.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | — | — | — |

**INFO only (not blockers):**

| Finding | From Review | Impact |
|---------|-------------|--------|
| WR-01: `.expect()` in LogReg L-BFGS closure | 05-REVIEW WR-01 | Panic on PrimError across PyO3 boundary — v1 Rust-only, no Phase-6 yet; deferred |
| WR-02: debug_assert bounds in KNN | 05-REVIEW WR-02 | Index bounds checked in debug, unchecked in release; deferred |
| WR-04: Double-centering narrowing in CD | 05-REVIEW WR-04 | F32 precision risk in soft-threshold denominator; oracle passes, deferred |
| WR-05: LogReg non-contiguous label space | 05-REVIEW WR-05 | Silent mis-behavior on {0,2} label gaps; fixture contiguous, deferred |
| WR-06: DBSCAN eps message wording | 05-REVIEW WR-06 | Diagnostic clarity only, deferred |
| WR-07: Softmax kernel triple dot-product | 05-REVIEW WR-07 | Latent correctness fragility; oracle holds, deferred |
| IN-01..IN-05 | 05-REVIEW INFO | Stale docs, message wording, round-trip, magic constant, duplicated helper; deferred |
| ROADMAP header checkbox | ROADMAP.md line 20 | Phase-5 top-level `[ ]` not updated to `[x]`; progress table shows `7/11` (stale); code is complete — planning artifact only |
| CR-01 estimator-level oracle gap | 05-REVIEW CR-01 note | `lloyd_relocates_empty_cluster` validates prim-level exact relocation; the estimator-level kmeans oracle fixture uses balanced clusters that never empty during Lloyd, so the full `fit()` path through lifted relocation is not exercised by an end-to-end empty-cluster fixture. Recommend adding a follow-up estimator-level fixture with k > natural_clusters. |

---

### Human Verification Required

#### 1. Phase-5 Test Suite on HEAD

**Test:** From the workspace root, run the 16 phase-5 and adjacent test binaries:
```
cargo test -p mlrs-backend --test lloyd_test --features cpu
cargo test -p mlrs-backend --test kmeanspp_test --features cpu
cargo test -p mlrs-backend --test dbscan_mask_test --features cpu
cargo test -p mlrs-backend --test topk_test --features cpu
cargo test -p mlrs-backend --test memory_gate_test --features cpu
cargo test -p mlrs-algos --test kmeans_test --features cpu
cargo test -p mlrs-algos --test dbscan_test --features cpu
cargo test -p mlrs-algos --test nearest_neighbors_test --features cpu
cargo test -p mlrs-algos --test knn_classifier_test --features cpu
cargo test -p mlrs-algos --test knn_regressor_test --features cpu
cargo test -p mlrs-algos --test lasso_test --features cpu
cargo test -p mlrs-algos --test elastic_net_test --features cpu
cargo test -p mlrs-algos --test logistic_test --features cpu
```
**Expected:** All tests pass (REALEXIT=0). Memory gate suite shows 11/11 green on cpu(f32).
**Why human:** Sandbox disk budget prohibits compilation here. Context documents these were GREEN post-fix, but a human should confirm on current HEAD before closing the phase.

---

### Gaps Summary

No gaps block the phase goal. All 8 requirement IDs are satisfied in the codebase with substantive implementations and active oracle tests. Both code-review BLOCKERs (CR-01 empty-cluster relocation, CR-02 k-means++ draw) are fixed and confirmed in source. Warnings WR-01..WR-07 and INFO findings are acknowledged, deferred, and do not violate the core 1e-5 correctness contract.

The only item preventing `status: passed` is a single human confirmation that the test suite is still green on HEAD — a mechanical run, not a design question.

**Follow-up recommendation (not a gap):** Add an estimator-level KMeans oracle fixture that exercises the empty-cluster relocation path end-to-end through `KMeans::fit()` (a dataset where k > natural cluster count so Lloyd forces relocation during fitting). This closes the coverage gap noted in CR-01 context.

---

_Verified: 2026-06-13T08:00:00Z_
_Verifier: Claude (gsd-verifier)_
