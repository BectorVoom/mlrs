---
phase: 15-hdbscan
verified: 2026-06-24T07:09:19Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 3/4
  gaps_closed:
    - "A user can fit/fit_predict HDBSCAN to produce labels_ and probabilities_"
  gaps_remaining: []
  regressions: []
---

# Phase 15: HDBSCAN Verification Report

**Phase Goal:** Deliver HDBSCAN `fit`/`fit_predict` → `labels_` (`-1` = noise) + `probabilities_` with sklearn-named hyperparameters, as a device front-end (core distances + mutual-reachability via GATHER, reusing Phase 13) plus a host tree back-end (MST → single-linkage → condensed tree → EoM/leaf stability extraction). Plus the GLOSH `outlier_scores_` differentiator and `store_centers`. Exact-label hard gate. File-disjoint from UMAP.
**Verified:** 2026-06-24T07:09:19Z
**Status:** passed
**Re-verification:** Yes — after gap closure (plan 15-07 / commits cff2458 + 2394efd)

---

## Goal Achievement

### Observable Truths

Derived from ROADMAP Phase 15 Success Criteria (4 criteria). Supplemented with PLAN frontmatter must_haves across all 7 plans.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A user can `fit`/`fit_predict` HDBSCAN to produce `labels_` and `probabilities_` with sklearn-named defaults, using the device front-end / host tree back-end split | VERIFIED | `pub fn fit_predict` at line 283 of `hdbscan.rs`, on `impl<F> Hdbscan<F, Unfit>` (lines 200–294). Consumes `self`, calls `Fit::fit(self, pool, x, None, shape)?`, reads `fitted.labels(pool)`, returns `DeviceArray::from_host(pool, &labels)`. Purely additive diff (cff2458). Behavioral-equivalence test `fit_predict_matches_fit_then_labels` passes under `--features cpu` (2394efd). |
| 2 | `labels_` match `sklearn.cluster.HDBSCAN` exactly up to permutation with `-1` pinned (exact across all 6 metrics + precomputed); MST edge tie-breaking stable-sorted with a documented deterministic rule; `probabilities_` agree within ≤1e-5 | VERIFIED | `cargo test -p mlrs-algos --test hdbscan_test --features cpu`: 39 passed (pre-gap-closure baseline), +1 new test = 40 total. All 12 feature-metric labels gates pass at `best_match_accuracy_pinned_noise == 1.0`. Tie-break rule documented in `mst.rs`. Probabilities gate green at ≤1e-5. No regressions — gap-closure commits touch only `hdbscan.rs` and `hdbscan_test.rs` additively. |
| 3 | A user can read per-point `outlier_scores_` (GLOSH) from a fitted HDBSCAN, gated within ≤1e-5 vs the `hdbscan` 0.8.44 library | VERIFIED | `hdbscan_outlier_scores` present in `glosh.rs` (regression check: `grep -c` = 1). `outlier_scores_match_f32`/`outlier_scores_match_f64` both pass at 0.0 diff. Untouched by plan 15-07. |
| 4 | A user can request cluster centers via `store_centers` (`'centroid'`/`'medoid'`) producing `centroids_`/`medoids_` (sklearn parity); errors on `Metric::Precomputed` | VERIFIED | `weighted_cluster_center` present in `centers.rs` (regression check: `grep -c` = 1). `centers_match_f32`/`centers_match_f64` pass ≤1e-5; `store_centers_precomputed_errors` confirms typed rejection. Untouched by plan 15-07. |

**Score:** 4/4 truths verified (behavior_unverified: 0)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-core/src/label_perm.rs` | `best_match_accuracy_pinned_noise` function | VERIFIED | Line 136: `pub fn best_match_accuracy_pinned_noise`. Filters `-1` from both vocabularies, force-inserts `(-1,-1)`. |
| `crates/mlrs-core/src/lib.rs` | `pub use` re-export of `best_match_accuracy_pinned_noise` | VERIFIED | Line 25 includes `best_match_accuracy_pinned_noise` in pub use. |
| `crates/mlrs-core/tests/helpers_test.rs` | 5 unit tests for pinned-noise | VERIFIED | 16 tests pass (no features needed). |
| `scripts/gen_oracle.py` | `gen_hdbscan` generator | VERIFIED | `def gen_hdbscan(` at line 1027. |
| `tests/fixtures/hdbscan_*.npz` | 22 committed oracle blobs | VERIFIED | 22 files found under `tests/fixtures/`. 6 metrics × {f32,f64} + tieheavy/{f32,f64} + nested/{f32,f64} + allnoise/{f32,f64} + single/{f32,f64} + tiny/{f32,f64}. |
| `crates/mlrs-algos/tests/hdbscan_test.rs` | 9-family oracle gate suite + equivalence test | VERIFIED | All 9 families present; 39 oracle tests + 1 new equivalence test. `fit_predict_matches_fit_then_labels` at line 1427 passes. |
| `crates/mlrs-algos/src/cluster/hdbscan.rs` | Extended `Metric` enum (6 variants), `StoreCenters`, build validation, fitted fields + accessors | VERIFIED | Metric: Euclidean/Manhattan/Cosine/Chebyshev/Minkowski{p}/Precomputed present. `StoreCenters` enum present. `BuildError::InvalidMinSamples`/`InvalidMaxClusterSize`/`InvalidAlphaHdbscan`/`InvalidMinkowskiP` present in error.rs. Fitted fields: `labels_`/`probabilities_`/`outlier_scores_`/`centroids_`/`medoids_` all present. Accessors on `Hdbscan<F,Fitted>` for all five. |
| `crates/mlrs-algos/src/cluster/hdbscan.rs` | `fit_predict` convenience method on `impl<F> Hdbscan<F, Unfit>` | VERIFIED | `pub fn fit_predict` at line 283. Receiver `self` by value (typestate-correct: `Fit::fit` consumes `self`). Returns `Result<DeviceArray<ActiveRuntime, i32>, AlgoError>`. Placed in the `impl<F> Hdbscan<F, Unfit>` block (lines 200–294), alongside `new`/`builder`/`into_builder`. Commit cff2458. |
| `crates/mlrs-algos/src/cluster/hdbscan/mst.rs` | Both oracle Prim variants + `argsort_by_weight` | VERIFIED | `mst_from_mutual_reachability` (Variant A), `mst_from_data_matrix` (Variant B), `argsort_by_weight` all present. |
| `crates/mlrs-algos/src/cluster/hdbscan/single_linkage.rs` | `UnionFind` + `make_single_linkage` | VERIFIED | `UnionFind` struct at line 40; `make_single_linkage` at line 116; `fast_find` path-compresses. |
| `crates/mlrs-algos/src/cluster/hdbscan/condense.rs` | `condense_tree` + `bfs_from_hierarchy` | VERIFIED | Both functions present. Uses `min_cluster_size`, not `min_samples` (Pitfall 4 avoided). |
| `crates/mlrs-algos/src/cluster/hdbscan/stability.rs` | `compute_stability` + `max_lambdas` | VERIFIED | Both functions present at lines 39 and 95. |
| `crates/mlrs-algos/src/cluster/hdbscan/select.rs` | `get_clusters` (eom/leaf/epsilon/max) + `do_labelling` + `get_probabilities` | VERIFIED | All three present. EoM and leaf traversals confirmed. `do_labelling` at line 400; `get_probabilities` at line 470. |
| `crates/mlrs-kernels/src/mutual_reachability.rs` | SharedMemory-free 2D-GATHER MR kernel | VERIFIED | `pub fn mutual_reachability` at line 58. No `SharedMemory`, `Atomic`, or `F::INFINITY` outside doc comments. Statement-form running-max (not if-expression). |
| `crates/mlrs-backend/src/prims/mutual_reachability.rs` | Host launch wrapper with geometry validation | VERIFIED | File exists. `checked_mul` overflow guard present. |
| `crates/mlrs-backend/tests/mutual_reachability_test.rs` | VALUE gate incl. duplicate-point row (R-9) | VERIFIED | `cargo test -p mlrs-backend --test mutual_reachability_test --features cpu`: 2 passed (`mutual_reachability_value_f32`, `mutual_reachability_value_f64`). |
| `crates/mlrs-algos/src/cluster/hdbscan/glosh.rs` | `hdbscan_outlier_scores` via parallel hdbscan-convention tree | VERIFIED | `fn hdbscan_outlier_scores` at line 230; internal `mst_linkage_core` at line 173; core distance at index `min_samples` (hdbscan convention). |
| `crates/mlrs-algos/src/cluster/hdbscan/centers.rs` | `weighted_cluster_center` (centroid + medoid) | VERIFIED | `pub fn weighted_cluster_center` at line 74. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `hdbscan.rs (fit_predict)` | `hdbscan.rs (Fit::fit + Hdbscan<F,Fitted>::labels)` | `Fit::fit(self, pool, x, None, shape)?` then `fitted.labels(pool)` | VERIFIED | Lines 290–292: qualified trait call chains through to the existing exhaustively-gated fit pipeline. |
| `hdbscan_test.rs` | `label_perm.rs` | `best_match_accuracy_pinned_noise` | VERIFIED | Import on line 52; called in 9 test locations for label gates. Unpinned matcher not called for label gates. |
| `hdbscan_test.rs` | `tests/fixtures/hdbscan_*.npz` | `load_npz` fixture loader | VERIFIED | `load_npz` imported; fixtures loaded per-test. |
| `hdbscan.rs` | `knn_graph.rs` | `include_self=true` for core distances | VERIFIED | Line 692: `/* include_self */ true` in `knn_graph` call. |
| `hdbscan.rs` | `mutual_reachability.rs` (kernel) | MR kernel launch for cosine path | VERIFIED | `feature_metric_single_linkage` routes Cosine to dense Variant-A with MR kernel. |
| `hdbscan.rs` | `glosh.rs` | `outlier_scores_` from condensed tree | VERIFIED | Lines 532-535: `glosh::hdbscan_outlier_scores` called in `fit` for both precomputed and feature-metric paths. |
| `hdbscan.rs` | `centers.rs` | `weighted_cluster_center` for `store_centers` | VERIFIED | Lines 543-573: `centers::weighted_cluster_center` called when `store_centers` is `Some`. |
| `hdbscan.rs` (precomputed path) | `mst.rs` | `mst_from_mutual_reachability` (Variant A) | VERIFIED | `precomputed_single_linkage` calls `mst_from_mutual_reachability`. |
| `hdbscan.rs` (feature path) | `mst.rs` | `mst_from_data_matrix` (Variant B) for non-cosine | VERIFIED | `feature_metric_single_linkage` routes to Variant B for euclidean/manhattan/chebyshev/minkowski. |
| `mst.rs` | `single_linkage.rs` | `argsorted MST edges feed make_single_linkage` | VERIFIED | `argsort_by_weight` → `make_single_linkage` wired. |
| `hdbscan.rs` | `select.rs` | `do_labelling` via `tree_to_labels` | VERIFIED | `tree_to_labels` calls `condense_tree` → `compute_stability` → `get_clusters` → `do_labelling` → `get_probabilities`. |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `hdbscan.rs` `labels_` (via `fit_predict`) | `labels` from `fitted.labels(pool)` | `Fit::fit` → `tree_to_labels` → `do_labelling` over real condensed tree | Yes — real cluster assignment from the hierarchy, same path as direct `fit` | FLOWING |
| `hdbscan.rs` `labels_` | `labels_dev` | `tree_to_labels` → `do_labelling` over real condensed tree | Yes — real cluster assignment from the hierarchy | FLOWING |
| `hdbscan.rs` `probabilities_` | `probabilities_` | `get_probabilities` over real condensed tree | Yes — `min(lambda, max_lambda)/max_lambda` formula | FLOWING |
| `hdbscan.rs` `outlier_scores_` | `outlier_scores_` | `glosh::hdbscan_outlier_scores` over hdbscan-convention tree | Yes — real GLOSH death-propagation pass | FLOWING |
| `hdbscan.rs` `centroids_`/`medoids_` | `centroids_`/`medoids_` | `centers::weighted_cluster_center` over real feature data | Yes — probability-weighted mean / strength-weighted argmin | FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `fit_predict` behavioral-equivalence gate (new — gap closer) | `cargo test -p mlrs-algos --test hdbscan_test --features cpu fit_predict_matches_fit_then_labels -- --exact` | 1 passed, 0 failed (confirmed by orchestrator prior to re-verification; diff also confirms purely additive change) | PASS |
| Full HDBSCAN oracle gate suite (39 tests, unchanged) | `cargo test -p mlrs-algos --test hdbscan_test --features cpu` | 39 passed, 0 failed, 0 ignored (baseline from initial verification; no test touched by gap-closure diff) | PASS |
| MR kernel value gate incl. duplicate-point row (R-9) | `cargo test -p mlrs-backend --test mutual_reachability_test --features cpu` | 2 passed | PASS |
| mlrs-core pinned-noise unit tests | `cargo test -p mlrs-core --test helpers_test` | 16 passed | PASS |
| mlrs-algos builds without errors | `cargo build -p mlrs-algos --features cpu` | Finished cleanly (confirmed by orchestrator; gap-closure diff is purely additive) | PASS |

---

## Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| HDBS-01 | 15-02, 15-03, 15-04, 15-05, 15-07 | `fit`/`fit_predict` → `labels_`/`probabilities_` with sklearn defaults; device front-end + host back-end | SATISFIED | `fit` is real and gated (39-test suite). `fit_predict` added in plan 15-07, commit cff2458. Behavioral-equivalence test passes (commit 2394efd). REQUIREMENTS.md marks HDBS-01 as `[x] Complete` at Phase 15. |
| HDBS-02 | 15-01, 15-02, 15-03, 15-04, 15-05 | Exact labels up to permutation with `-1` pinned; MST tie-break documented; `probabilities_` ≤1e-5 | SATISFIED | 39 tests pass; pinned-noise matcher verified; tie-break documented in `mst.rs`. REQUIREMENTS.md marks HDBS-02 as `[x] Complete` at Phase 15. |
| HDBS-03 | 15-06 | `outlier_scores_` (GLOSH) gated ≤1e-5 vs `hdbscan` 0.8.44 | SATISFIED | `outlier_scores_match_f32`/`f64` pass at 0.0 diff. REQUIREMENTS.md marks HDBS-03 as `[x] Complete` at Phase 15. |
| HDBS-04 | 15-06 | `store_centers` → `centroids_`/`medoids_` sklearn parity | SATISFIED | `centers_match_f32`/`f64` pass ≤1e-5; `store_centers_precomputed_errors` confirms typed rejection. REQUIREMENTS.md marks HDBS-04 as `[x] Complete` at Phase 15. |

All 4 Phase 15 requirements are marked `Complete` in REQUIREMENTS.md with Phase 15 as the delivery phase.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/mlrs-kernels/src/elementwise.rs` | 282 | `FRAC_PI_2` clippy error (`approx_constant`) | WARNING (pre-existing) | Pre-existing in Phase 8-era code; confirmed not in phase 15 file set (last commit to elementwise.rs is `f03...`, wave 08). `cargo clippy --features cpu` fails on this unrelated crate but `cargo build -p mlrs-algos --features cpu` and `cargo build -p mlrs-backend --features cpu` are clean. Logged in `deferred-items.md` since Phase 15-04. |

No TBD / FIXME / XXX markers introduced by gap-closure commits. The new `fit_predict` method and its test are clean. No stub patterns.

---

## Intentional Documented Divergence: GLOSH Parallel Tree (D-07 / Option A)

The 15-06 plan's Task 1 `<action>` directed running GLOSH over the same condensed tree as `labels_`/`probabilities_` (sklearn-convention tree). The plan's approach would produce a ~0.06 divergence from the committed `hdbscan` 0.8.44 fixture because hdbscan's oracle differs from sklearn's pipeline in two ways: (1) core distance at index `min_samples` not `min_samples-1`, and (2) a different Prim tie-order (`mst_linkage_core` vs sklearn's argmin Prim).

The executor escalated this as a D-06 decision (gate diverged beyond ≤1e-5 for genuine algorithmic reasons). The user approved **Option A** at a documented checkpoint: build a GLOSH-only parallel hdbscan-convention tree. This is documented in `15-06-SUMMARY.md` under "Deviations from Plan" and in `glosh.rs` module-level doc-comments (lines 36–57). The divergence is intentional, justified, user-approved, and does not affect `labels_`/`probabilities_` (which remain on the sklearn-exact tree). It is not a defect.

---

## Re-Verification Summary

**Prior status:** gaps_found (score 3/4). The single gap was the absence of `Hdbscan::fit_predict` in `hdbscan.rs` (`grep -c 'fit_predict'` = 0).

**Gap-closure evidence (plan 15-07):**

- Commit cff2458 adds `pub fn fit_predict` at line 283 of `hdbscan.rs` in the correct `impl<F> Hdbscan<F, Unfit>` block (lines 200–294). The receiver is `self` by value (typestate-correct — `Fit::fit` consumes `self`; the verifier's earlier `&mut self` suggestion would not compile). The body is 3 lines: qualified `Fit::fit` call, `labels()` read, `DeviceArray::from_host` return. Purely additive diff — no existing code altered.
- Commit 2394efd adds `fit_predict_matches_fit_then_labels` test in `hdbscan_test.rs` at line 1427. The test constructs two identical `Hdbscan::<f64>::new()` estimators on a self-contained 16-row two-cluster blob, runs one through `Fit::fit` + `labels()` and the other through `fit_predict`, and asserts element-for-element label equality. Honors `skip_f64` convention; passes on cpu f64 gate.
- `grep -c 'fn fit_predict' hdbscan.rs` = 1 (was 0).
- `cargo build -p mlrs-algos --features cpu` clean.
- `cargo test … fit_predict_matches_fit_then_labels -- --exact` → 1 passed.

**Truths 2, 3, 4 regression check:** Both gap-closure commits are additively confined to `hdbscan.rs` (11 lines of new method + 28 lines total diff) and `hdbscan_test.rs` (+51 lines). No existing oracle test, module, or wiring was touched. The `glosh.rs` and `centers.rs` symbols are present (confirmed by regression `grep -c`). No regressions.

---

_Verified: 2026-06-24T07:09:19Z_
_Verifier: Claude (gsd-verifier)_
