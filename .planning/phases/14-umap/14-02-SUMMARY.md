---
phase: 14-umap
plan: 02
subsystem: manifold
tags: [umap, manifold, smooth-knn, fuzzy-simplicial-set, t-conorm, membership-strengths, host-f64, value-gate]

# Dependency graph
requires:
  - phase: 14-umap (plan 01)
    provides: "empty umap_internals.rs wired pub(crate) in mod.rs; 21 umap-learn 0.5.12 oracle fixtures; RED smooth_knn_* / fuzzy_union_* value-gate harness in umap_test.rs"
  - phase: 13-knn-graph-primitive-feasibility-keystone
    provides: "directed (n,k) KNN graph (self-dropped, ascending per row) — the input to the host stages"
provides:
  - "smooth_knn_dist(knn_dist, n, k, n_neighbors, local_connectivity) -> (sigmas, rhos): per-row ρ/σ binary search, host f64, ≤1e-5 vs umap-learn 0.5.12 × 5 metrics"
  - "compute_membership_strengths(knn_idx, knn_dist, rhos, sigmas, n, k) -> (rows, cols, vals): directed COO membership exp (umap-learn verified formula)"
  - "fuzzy_union(rows, cols, vals, n, set_op_mix_ratio) -> symmetric COO: t-conorm UMAP symmetrization (D-04), scipy CSR-canonical (row,col) order"
  - "smooth_knn_* and fuzzy_union_* value-gate families GREEN for all 5 metrics"
affects: [14-03 (a/b LM + spectral/random init consume the symmetric fuzzy graph), 14-04 (layout kernel + real fit over the validated graph foundation), 14-05 (transform)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure host f64 numeric stage fns (no device kernel, no DeviceArray) — value-gate to ≤1e-5 vs umap-learn's own numpy/numba f64 without device-reduction-order drift"
    - "HashMap/BTreeSet COO union → deterministic ascending (row,col) order matching scipy CSR canonical form (byte-stable value-gate + downstream consumers)"
    - "Float32-input fidelity at the value-gate boundary: round-trip f64→f32→f64 the dumped true-distance fixture to reconstruct umap's pynndescent float32 stage input (RESEARCH Pitfall 6)"

key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/manifold/umap_internals.rs
    - crates/mlrs-algos/tests/umap_test.rs

key-decisions:
  - "Membership/sigma value-gate feeds f32-cast knn_dist (umap's actual pynndescent float32 stage input), not the f64 true-distance fixture: f64 input matched umap COO to only ~1.0e-5 *relative* on the worst edge (exp amplifies the f32↔f64 distance gap past the bound); f32-cast input drives the whole pipeline to ≤1.6e-7 for all 5 metrics. Stage fns stay pure f64 — the f32 round reconstructs umap's numba input, the faithful per-stage gate."
  - "NPY_FLOATMAX = f32::MAX (not f64::MAX/INFINITY) for the σ binary-search upper bound + doubling branch — matches umap-learn's np.finfo(np.float32).max the fixtures were produced with."
  - "fuzzy_union prunes zeros both before (eliminate_zeros on A) and after the t-conorm (eliminate_zeros on G), matching umap's exact entry set so the edge count and COO order line up with the dumped graph."
  - "COO emitted in ascending (row,col) order via BTreeSet — scipy's CSR-canonical order — so the value-gate and downstream spectral-init/layout consumers see a byte-stable graph."

patterns-established:
  - "Host f64 UMAP deterministic-stage idiom (smooth-kNN → membership → t-conorm) mirrors the rng.rs host-glue shape: one input array in, Vecs out, no device launch."
  - "f32-input reconstruction at the test boundary for stages umap runs on pynndescent float32 (keeps the production fn pure f64 while faithfully matching umap's COO)."

requirements-completed: [UMAP-02]

# Metrics
duration: ~25m (continuation: finish + value-gate Task 2)
tasks-completed: 2
files-changed: 2
completed: 2026-06-23
---

# Phase 14 Plan 02: UMAP Deterministic Host Stages (smooth-kNN ρ/σ, membership, t-conorm union) Summary

Landed UMAP's deterministic fuzzy-simplicial-set core as three pure host f64 routines in `umap_internals.rs` — the smooth-kNN ρ/σ per-row binary search, the membership-strength exp, and the t-conorm fuzzy-set union (UMAP's symmetrization, D-04) — each value-gated ≤1e-5 (f64) against the committed umap-learn 0.5.12 fixtures for all 5 metrics.

## What was built

- **`smooth_knn_dist`** (Task 1, committed `0903733` by the prior executor): ρ-first local-connectivity interpolation over each row's non-zero distances, then the per-row binary search on `d − ρ` for σ s.t. `Σ exp(-(max(0, d−ρ))/σ) = log2(n_neighbors)`, with the verified umap constants (`SMOOTH_K_TOLERANCE=1e-5`, `MIN_K_DIST_SCALE=1e-3`, `n_iter=64`, `NPY_FLOATMAX=f32::MAX`) and the per-row + global σ floors. `smooth_knn_*` GREEN × 5 metrics.
- **`compute_membership_strengths`** (Task 2, committed `637574f`): directed `(n,k)` COO emit with the verified formula `val = if (d − ρ ≤ 0 || σ == 0) { 1.0 } else { exp(-(d − ρ)/σ) }`; `rows[i*k+j]=i`, `cols=knn_idx[i,j].round()`, self edges → 0 (inert — Phase-13 KNN is already self-dropped). Bounds come straight from the Phase-13 prim output (threat T-14-05).
- **`fuzzy_union`** (Task 2, committed `637574f`): builds directed `A` as a HashMap (zero-pruned), forms `G = mix*(A + Aᵀ − A∘Aᵀ) + (1−mix)*A∘Aᵀ` over the union of directed keys and their transposes, prunes trailing zeros, and emits COO in ascending `(row,col)` order (scipy CSR canonical). `fuzzy_union_*` GREEN × 5 metrics (≤1.6e-7 f64 vs umap COO).

## Verification

- `cargo test -p mlrs-algos --features cpu --test umap_test smooth_knn` → 5 passed (Task 1 regression-checked).
- `cargo test -p mlrs-algos --features cpu --test umap_test fuzzy_union` → 5 passed.
- `cargo build -p mlrs-algos --features cpu` → exit 0, 0 warnings.
- Source assertions: `fn smooth_knn_dist` (L50), `fn compute_membership_strengths` (L165), `fn fuzzy_union` (L220) present; `umap_internals.rs` = 269 lines (min 80).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] fuzzy_union value-gate failed for euclidean on one edge (rel 1.01e-5)**
- **Found during:** Task 2 verification — `fuzzy_union_euclidean` failed at edge 164 (r=9,c=50): produced `0.17315900` vs umap `0.17315726`, rel error `1.005e-5` (1% over the 1e-5 *relative* bound; abs error `1.7e-6` was within tolerance).
- **Root cause:** umap-learn feeds its stages the pynndescent KNN distances, which are **float32**; the Plan-01 fixture dumps the f64 "true distance" (RESEARCH Pitfall 6). Running the membership `exp` on the f64 distances amplifies the f32↔f64 distance gap (~1.35e-5 in `d`) past the relative bound on the few edges where `(d−ρ)/σ` is largest. Verified numerically: f64-dist pipeline → max rel `1.005e-5` (fails); f32-cast-dist pipeline → max rel `1.583e-7` (passes comfortably) across all 5 metrics.
- **Fix:** the `fuzzy_union_*` value-gate now round-trips the dumped `knn_dist` f64→f32→f64 before driving `smooth_knn_dist`/`compute_membership_strengths`, reconstructing umap's actual numba float32 stage input. The stage fns stay pure f64 (correct as-is); only the test boundary reconstructs umap's input precision. This is the faithful per-stage gate, consistent with Pitfall 6 ("dump umap-learn's actual per-stage arrays — NOT recompute").
- **Files modified:** `crates/mlrs-algos/tests/umap_test.rs`
- **Commit:** `637574f`

The `umap_internals.rs` membership + union implementation was complete and correct as left by the interrupted executor; no source-logic changes were needed — only the test wiring to feed umap's true float32 stage input.

## TDD Gate Compliance

Plan is `type: execute` with `tdd="true"` tasks. The RED gate (failing `smooth_knn_*` / `fuzzy_union_*` tests) was established in Plan 01 (`test(14-01)` commit `f437443`). Task 1 GREEN landed as `feat(14-02): 0903733`; Task 2 GREEN as `feat(14-02): 637574f`. No separate per-task RED commit in this plan — the RED harness pre-existed from the substrate plan, which the tdd cycle satisfies (RED then GREEN, gate sequence intact).

## Threat surface

No new security-relevant surface beyond the plan's `<threat_model>`. The mitigations are present: bounded `n_iter=64` (T-14-03), umap's typed-zero guards / σ floors / ρ≤0 fallback → no NaN (T-14-04), `cols` from the bounds-validated Phase-13 prim with host indexing that panics rather than OOB-reads (T-14-05).

## Known Stubs

None. `smooth_knn_dist`, `compute_membership_strengths`, and `fuzzy_union` are fully implemented and value-gated; no placeholder/empty-data paths remain in `umap_internals.rs` for this plan's scope. (`init_graph_transform` is Plan 05's, not stubbed here.)

## Self-Check: PASSED

- `crates/mlrs-algos/src/manifold/umap_internals.rs` — FOUND (269 lines, all 3 fns present)
- `crates/mlrs-algos/tests/umap_test.rs` — FOUND (fuzzy_union pipeline wired)
- Commit `0903733` (Task 1) — FOUND
- Commit `637574f` (Task 2) — FOUND
