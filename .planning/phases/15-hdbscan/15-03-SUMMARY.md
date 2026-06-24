---
phase: 15-hdbscan
plan: 03
subsystem: clustering
tags: [hdbscan, mst, prims, single-linkage, union-find, precomputed, metric, rust, oracle]

# Dependency graph
requires:
  - phase: 15-01
    provides: "mlrs_core::best_match_accuracy_pinned_noise (-1-pinned label matcher) for the exact-label gate"
  - phase: 15-02
    provides: "Wave-0 hdbscan_test.rs oracle gate suite + committed hdbscan_*.npz fixtures (build_validation/tie_break_exact gates)"
  - phase: 12
    provides: "Hdbscan<F, S> builder + typestate shell (Metric enum, HdbscanBuilder::build, Fit consume-self)"
provides:
  - "hdbscan::Metric extended to 6 variants (Euclidean, Manhattan, Cosine, Chebyshev, Minkowski{p}, Precomputed); Eq dropped for PartialEq"
  - "hdbscan::StoreCenters {Centroid, Medoid, Both} enum + store_centers builder field/accessor"
  - "BuildError::{InvalidMinSamples, InvalidMaxClusterSize, InvalidAlphaHdbscan, InvalidMinkowskiP} — deferred build-validation TODO resolved"
  - "crates/mlrs-algos/src/cluster/hdbscan/mst.rs — both oracle Prim variants (Variant A dense argmin-first-min, Variant B source-tracking strict-< lowest-j) + argsort_by_weight + dense core-dist/mutual-reachability helpers"
  - "crates/mlrs-algos/src/cluster/hdbscan/single_linkage.rs — UnionFind (fresh-label N+i, path-compressed fast_find) + make_single_linkage"
  - "Metric::Precomputed fit branch: square validation -> /alpha -> core dist -> dense MR -> Variant-A MST -> argsort -> single-linkage hierarchy stored on the estimator"
  - "Hdbscan<F, Fitted>::single_linkage() accessor for the Wave-3 condense/select stage"
affects: [15-04, 15-05, 15-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Host-scalar f64 bridging via mlrs_core::host_to_f64 for the sequential MST/single-linkage back-end (spectral_embedding precedent)"
    - "Two-variant oracle MST dispatch with distinct per-path alpha placement (Variant A scales the whole matrix; Variant B divides pairwise only)"
    - "Fresh-label UnionFind whose merge ORDER (fixed by argsort) determines the dendrogram node ids (D-04 crux)"

key-files:
  created:
    - crates/mlrs-algos/src/cluster/hdbscan/mst.rs
    - crates/mlrs-algos/src/cluster/hdbscan/single_linkage.rs
  modified:
    - crates/mlrs-algos/src/cluster/hdbscan.rs
    - crates/mlrs-algos/src/error.rs
    - crates/mlrs-algos/tests/hdbscan_test.rs

key-decisions:
  - "Kept the oracle-gate label tests (tie_break_exact, labels_match_*) #[ignore]d — labels stay all--1 until 15-04 wires the condensed tree; added 6 DIRECT host-module value gates instead (both MST variants, alpha placement, UnionFind, single-linkage, precomputed fit + squareness rejection)"
  - "argsort_by_weight uses f64::total_cmp — well-defined deterministic order; the gate fixtures use DISTINCT weights so the sort is tie-free and oracle-equal under any rule (RESEARCH Pitfall 1 option 2)"
  - "Precomputed symmetry is DOCUMENTED (not enforced) — squareness IS validated (typed PrimError); committed fixtures are pairwise_distances (symmetric by construction)"
  - "Added a single_linkage_ host-Vec field (NOT device-resident) + accessor so 15-04 can consume the hierarchy"

patterns-established:
  - "Variant A (mst_from_mutual_reachability): dense Prim from node 0, FIRST-min argmin via shrinking current_labels remap — cosine/precomputed"
  - "Variant B (mst_from_data_matrix): source-tracking Prim, strict < (lowest-j ties), pairwise/alpha with RAW core dist — euclidean/l1/l2/chebyshev/minkowski"
  - "make_single_linkage relabels N+i per merge; fast_find path-compresses"

requirements-completed: [HDBS-01, HDBS-02]

# Metrics
duration: 18min
completed: 2026-06-24
status: complete
---

# Phase 15 Plan 03: HDBSCAN Exactness Core (MST + Single-Linkage) Summary

**Extended the HDBSCAN Metric enum to all 6 variants, resolved the deferred build validation with 4 typed BuildErrors, and ported both oracle Prim MST variants + the fresh-label UnionFind single-linkage — wiring the precomputed path through a square-validated dense Variant-A MST to a stored single-linkage hierarchy.**

## Performance

- **Duration:** ~18 min
- **Completed:** 2026-06-24
- **Tasks:** 2
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments
- `Metric` enum extended to `Euclidean, Manhattan, Cosine, Chebyshev, Minkowski{p}, Precomputed`; `Eq` dropped to `PartialEq` (the `f64`-carrying Minkowski variant is not `Eq`), `hyperparams_eq` updated.
- Deferred build-validation TODO resolved: `min_samples<1` (Some), `max_cluster_size` (neither 0 nor `>= min_cluster_size`), `alpha<=0`, and `Minkowski p<1` are each rejected at `build()` with a new typed `BuildError`, BEFORE any device launch.
- `StoreCenters` enum + `store_centers` builder field/accessor wired (compute lands in 15-06).
- Both oracle Prim MST variants ported line-for-line with their DISTINCT per-path alpha placements, plus `argsort_by_weight` and dense core-distance / mutual-reachability helpers.
- `UnionFind` + `make_single_linkage` ported (fresh-label `N+i` per merge, path-compressed `fast_find`).
- `Metric::Precomputed` `fit` short-circuit: square validation (typed `PrimError`) → `/alpha` → core distances → dense MR → Variant-A MST → argsort → single-linkage hierarchy stored on the estimator + a `single_linkage()` accessor for 15-04.

## Task Commits

1. **Task 1: Extend Metric enum, resolve build validation, add typed errors** — `e52ac1b` (feat)
2. **Task 2: Port both MST variants + single-linkage; wire the precomputed MST path** — `7c0ef9c` (feat)

## Files Created/Modified
- `crates/mlrs-algos/src/cluster/hdbscan/mst.rs` (created) — both oracle Prim MST variants, `argsort_by_weight`, `core_distances_dense`, `mutual_reachability_dense`.
- `crates/mlrs-algos/src/cluster/hdbscan/single_linkage.rs` (created) — `UnionFind`, `SingleLinkageEdge`, `make_single_linkage`.
- `crates/mlrs-algos/src/cluster/hdbscan.rs` (modified) — extended `Metric`, `StoreCenters`, build validation, `single_linkage_` field + accessor, `Metric::Precomputed` fit branch + `precomputed_single_linkage` helper, submodule tree.
- `crates/mlrs-algos/src/error.rs` (modified) — 4 new `BuildError` variants.
- `crates/mlrs-algos/tests/hdbscan_test.rs` (modified) — extended `build_validation` (all new rejections + Precomputed round-trip); 6 new direct host-module value gates.

## Decisions Made
- The oracle label-gate tests (`tie_break_exact`, `labels_match_*`, etc.) remain `#[ignore]`d: this slice ports the MST/single-linkage back-end but the feature-metric label pipeline needs 15-04 (condense/select) + 15-05 (device front-end). Per the plan, labels stay all-`-1` until 15-04. To still exercise the new code under VALUES (R-9), added 6 direct host-module unit gates over hand-built distinct-weight matrices with verifiable hierarchies.
- `argsort_by_weight` uses `f64::total_cmp` for a deterministic total order; the gate fixtures use DISTINCT MST edge weights so the sort is tie-free and oracle-equal under any rule (RESEARCH Pitfall 1 option 2). The adversarial tie-heavy fixture is the characterization gate that 15-04 turns on.
- Precomputed symmetry is documented (sklearn's `allclose(X, X.T)` expectation) rather than enforced; squareness IS validated with a typed `PrimError::ShapeMismatch`.

## Deviations from Plan

None - plan executed exactly as written. The plan's Task 2 acceptance noted the `tie_break_exact` gate as the D-04 TRUE GATE "OR" surfacing an un-exactable metric as a blocker. Neither applies in this slice: the gate cannot run end-to-end until 15-04 wires the condensed tree (the plan itself states "labels stay all-`-1` until 15-04"), so the gate stays `#[ignore]`d and the MST/single-linkage correctness is asserted directly. No un-exactable metric was found; the distinct-weight design holds the exact line per D-05.

## Issues Encountered
None. The five feature-space metrics keep the trivial all-`-1` fit (device front-end is 15-05); only the precomputed pure-host path was wired here, matching the plan's scoping.

## D-05 Exact-Line Status
No blocker surfaced. The single-linkage hierarchy is deterministic and tie-free on the distinct-weight gate inputs; the oracle-matched tie-break (Variant A FIRST-min argmin / Variant B strict-< lowest-j) is ported faithfully. The end-to-end `tie_break_exact` TRUE GATE is sequenced to run in 15-04 once labelling exists — if a metric proves un-exactable there, it will be surfaced as a phase BLOCKER per D-05 (never band-demoted).

## Next Phase Readiness
- 15-04 can consume `Hdbscan<F, Fitted>::single_linkage()` to build the condensed tree → stability → EoM/leaf selection → labelling, then un-ignore `tie_break_exact` / `labels_match_*` for the precomputed metric.
- 15-05 wires the device front-end (KNN core distances + GATHER mutual-reachability) and Variant B for the five feature-space metrics.
- 15-06 computes `centroids_`/`medoids_` from the wired `store_centers` field.

---
*Phase: 15-hdbscan*
*Completed: 2026-06-24*
