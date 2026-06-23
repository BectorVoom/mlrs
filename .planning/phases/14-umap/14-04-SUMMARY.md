---
phase: 14-umap
plan: 04
subsystem: manifold
tags: [umap, cubecl, cpu-mlir, sgd-layout, gather-kernel, splitmix64, property-gate, trustworthiness, knn-overlap, reproducibility]

# Dependency graph
requires:
  - phase: 14-umap (Plan 02)
    provides: "smooth_knn_dist / compute_membership_strengths / fuzzy_union host f64 stages (the fit pipeline's deterministic graph foundation)"
  - phase: 14-umap (Plan 03)
    provides: "fit_ab LM curve fit + spectral_init/random_init/noisy_scale_coords (a/b + init the fit pipeline consumes)"
  - phase: 13-knn-graph-primitive-feasibility-keystone
    provides: "knn_graph<F> directed (n,k) prim (include_self=false UMAP path) + the cpu-MLIR launch-shape landmines (002-A/002-B)"
provides:
  - "umap_layout_step<F> — the ONE new device kernel: vertex-owner GATHER SGD step, cpu-MLIR-safe (launches f32+f64), frozen-subset-capable (move_other), host-drawn neg_idx GATHER"
  - "Umap::fit/fit_transform real bodies — full KNN→smooth-kNN→membership→t-conorm→a/b→init→SGD-layout pipeline for all 5 metrics"
  - "host_epoch_driver + make_epochs_per_sample — umap's per-edge sample schedule with order-deterministic SplitMix64 neg draws (D-05)"
  - "calibrated property-gate thresholds (PROPERTY_EPS=0.02, ARI_BAND=0.05) recorded in 14-VALIDATION.md + GREEN layout_property (5 metrics) and reproducible_f64"
affects: [14-05 (transform reuses umap_layout_step with owners=new points, move_other=false)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Vertex-owner GATHER SGD kernel (CUBE_POS_X/UNIT_POS_X==0 per-owner shape, self-contained-nested accumulate, finite-literal clip ±4, static F::powf) — the cpu-MLIR-safe layout-step idiom"
    - "Host epoch driver replicating umap's epoch_of_next_sample / epoch_of_next_negative schedule: per-epoch active-edge CSR + host-drawn neg CSR uploaded then launched (sgd_solve precedent generalized to a per-owner GATHER)"
    - "Per-(seed, epoch, edge) SplitMix64 substream for negative sampling — order-deterministic, byte-identical per (backend,dtype), no device RNG (D-05)"
    - "Relative-to-oracle property gate: trustworthiness/kNN-overlap/downstream-ARI ≥ umap−ε (D-04), thresholds calibrated from the first fixture run's measured margins, never absolute floors"

key-files:
  created:
    - crates/mlrs-kernels/src/umap_layout.rs
    - crates/mlrs-backend/tests/umap_layout_test.rs
  modified:
    - crates/mlrs-kernels/src/lib.rs
    - crates/mlrs-algos/src/manifold/umap.rs
    - crates/mlrs-algos/tests/umap_test.rs
    - .planning/phases/14-umap/14-VALIDATION.md

key-decisions:
  - "umap_layout_step buffer contract is CSR-per-owner (pos_offsets/pos_tail + neg_offsets/neg_idx) so one launch processes EVERY owner's attractive+repulsive contributions per epoch; the host builds the active-edge CSR each epoch from umap's sample schedule rather than streaming single edges (matches the per-owner topk launch shape and keeps the kernel SharedMemory-free)."
  - "The launch-smoke test lives in crates/mlrs-backend/tests/ (NOT mlrs-kernels) because mlrs-kernels carries ZERO backend runtime feature — the kernel can only be LAUNCHED against mlrs_backend::runtime::ActiveRuntime. The plan's `cargo test -p mlrs-kernels --features cpu` verify command was therefore re-routed to `-p mlrs-backend` (Rule 3 blocking — the named crate has no runtime/cpu feature)."
  - "Calibrated PROPERTY_EPS=0.02 (≈28× the worst measured trust margin of +0.0007 on euclidean) and ARI_BAND=0.05 (measured ARI gap 0.0000 on all 5 metrics). mlrs MATCHES or BEATS umap on overlap for every metric; the gate is tight and relative (D-04), not an absolute floor."
  - "downstream-ARI clusters BOTH embeddings with the same deterministic host Lloyd k-means (k=3 true classes, first-k-row init, 50 iters) vs the true labels — both recover the 3 clusters exactly (ARI 1.0). A host-only, sklearn-free, reproducible gate."
  - "fit_transform is an inherent method on Umap<F, Unfit> (no trait slot exists); it runs fit then returns the host embedding Vec, mirroring umap-learn's UMAP.fit_transform."

patterns-established:
  - "cpu-MLIR-safe SGD GATHER kernel: per-owner CUBE_POS_X launch + per-iteration self-contained dist²/grad accumulate + finite clip — reusable for any future per-owner stochastic update kernel."
  - "Order-deterministic host RNG plumbing for a stochastic device algorithm: draw on the host into a per-epoch device buffer keyed by (seed, counter), keeping the kernel RNG-free and the result byte-reproducible (D-05)."

requirements-completed: [UMAP-01, UMAP-03]

# Metrics
duration: ~75min (dominated by the slow cpu Jacobi spectral-eig in the property-gate runs: ~1717s for the 5-metric calibration sweep + ~1032s for the euclidean+reproducible confirmation)
completed: 2026-06-24
status: complete
---

# Phase 14 Plan 04: UMAP SGD Layout Kernel + Real fit Pipeline Summary

**Authored the phase's one new device kernel `umap_layout_step<F>` (vertex-owner GATHER SGD step, cpu-MLIR-safe and frozen-subset-capable), wired `Umap::fit`/`fit_transform` as a host epoch driver over the full KNN→fuzzy→union→a/b→init→SGD pipeline with order-deterministic host negative sampling, and calibrated the property-gate so all 5 metrics track umap-learn 0.5.12 to within +0.0007 trustworthiness while reproducing byte-identical embeddings per same seed.**

## Performance

- **Duration:** ~75 min wall (most of it the slow cpu Jacobi spectral-eig: the 5-metric `layout_property` calibration sweep took 1717s, the euclidean+reproducible confirmation 1032s)
- **Tasks:** 3
- **Files modified/created:** 6 (2 created, 4 modified)

## Accomplishments

- **Spike flag item 1 RESOLVED:** `umap_layout_step<F>` LAUNCHES under cpu-MLIR for both f32 and f64 (and moves coordinates — a value assertion, not a non-panic check), obeying every landmine: `CUBE_POS_X`/`UNIT_POS_X==0` per-owner shape, self-contained-nested accumulate (no 002-B cross-sibling miscompile), finite-literal `clip(±4)`, static `F::powf`, host-drawn `neg_idx` GATHER, frozen-subset `move_other` toggle — NO SharedMemory/Atomic/F::INFINITY/instance-powf.
- **Real `fit`/`fit_transform`:** the full deterministic-then-stochastic pipeline runs for all 5 metrics, producing a real `(n, n_components)` embedding via `knn_graph`(include_self=false) → `smooth_knn_dist`/`compute_membership_strengths`/`fuzzy_union` → `fit_ab` → spectral/random init + `noisy_scale_coords` → the host epoch driver launching `umap_layout_step` per epoch.
- **Spike flag item 2 RESOLVED:** calibrated `PROPERTY_EPS=0.02` / `ARI_BAND=0.05` from the first-run measured margins (recorded per metric in 14-VALIDATION.md); `layout_property_<metric>` GREEN for all 5 metrics (relative-to-umap, D-04) and `reproducible_f64` GREEN (byte-identical across two same-`random_state` fits, D-05).

## Task Commits

1. **Task 1: umap_layout_step kernel + cpu-MLIR launch-smoke** - `ba96b4c` (feat)
2. **Task 2: real fit/fit_transform host epoch driver + full pipeline** - `cf31482` (feat)
3. **Task 3: calibrate thresholds, layout_property + reproducible GREEN** - `6807dd9` (test)

## Files Created/Modified

- `crates/mlrs-kernels/src/umap_layout.rs` - NEW `umap_layout_step<F>` vertex-owner GATHER SGD kernel (CSR-per-owner positive + negative edges, squared-distance attractive/repulsive gradients, finite clip, frozen-subset)
- `crates/mlrs-kernels/src/lib.rs` - `pub mod umap_layout;` + `pub use umap_layout::umap_layout_step;` (the sgd/topk re-export idiom)
- `crates/mlrs-backend/tests/umap_layout_test.rs` - NEW launch-smoke gate: launches the kernel f32+f64 under cpu-MLIR and asserts coordinates MOVE (R-9 / 002-B guard)
- `crates/mlrs-algos/src/manifold/umap.rs` - real `fit` body, `fit_transform` method, `map_metric`, `make_epochs_per_sample`, `run_umap_layout` pipeline, `host_epoch_driver` (replaces the trivial-zeros shell)
- `crates/mlrs-algos/tests/umap_test.rs` - calibrated `PROPERTY_EPS`/`ARI_BAND`; finalized `layout_property_<metric>` (trust/overlap/ARI ≥ umap−ε) + downstream-ARI host k-means; CALIB witness prints
- `.planning/phases/14-umap/14-VALIDATION.md` - per-metric calibrated thresholds + measured scores recorded (Spike flag item 2)

## Decisions Made

See frontmatter `key-decisions`. The load-bearing ones: CSR-per-owner kernel contract (one launch covers all owners' attract+repel per epoch), the launch-smoke test routed to `mlrs-backend` (the only crate with a runtime), tight relative thresholds calibrated from real margins, and a deterministic host-k-means downstream-ARI.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Kernel launch-smoke test + verify command re-routed from `mlrs-kernels` to `mlrs-backend`**
- **Found during:** Task 1
- **Issue:** The plan's verify command `cargo test -p mlrs-kernels --features cpu umap_layout` cannot work — `mlrs-kernels` deliberately carries ZERO backend runtime feature (Criterion 1 / FOUND-02), so it has no `cpu` feature and no `ActiveRuntime` to launch against. A kernel can only be LAUNCHED (the Spike flag item 1 "verified-at-launch" gate) from `mlrs-backend`, which selects the runtime.
- **Fix:** Placed the launch-smoke test in `crates/mlrs-backend/tests/umap_layout_test.rs` (mirroring the existing `sgd_test.rs`/`topk_test.rs` precedent where kernel-launch tests live in the backend crate) and ran it via `cargo test -p mlrs-backend --features cpu --test umap_layout_test`. The kernel itself + re-export still live in `mlrs-kernels` exactly as the plan specifies.
- **Files modified:** `crates/mlrs-backend/tests/umap_layout_test.rs` (created)
- **Verification:** `cargo test -p mlrs-backend --features cpu --test umap_layout_test` → 3 passed (f32 move-both, f64 move-both, f64 owner-only), each asserting coordinates MOVE.
- **Committed in:** `ba96b4c` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking).
**Impact on plan:** The deviation is purely about WHERE the launch test lives and which crate the verify command targets — it does not change the kernel, the re-export, or any acceptance behaviour. No scope creep.

## Issues Encountered

- The cpu Jacobi spectral-eig is the long pole: with the fixture design `n=60 ≤ MAX_DIM=64`, `init='spectral'` runs the dense Jacobi `eig` (~5–6 min per metric), so the 5-metric `layout_property` calibration sweep took ~28 min and the euclidean+reproducible confirmation ~17 min. Test runs were scoped to the touched tests (never the full backend suite) per the plan note.
- Calibration outcome was better than the RESEARCH-flagged borderline risk: mlrs tracks umap to within +0.0007 trustworthiness and matches-or-beats it on kNN-overlap for every metric, with downstream-ARI exactly 1.0 everywhere — no metric needed a loosened gate.

## Known Stubs

None for this plan's scope. `umap_layout_step`, the host epoch driver, and the real `fit`/`fit_transform` are fully implemented and gated. The `transform` body remains the Phase-12 zeros shell — that is Plan 05's responsibility (it will reuse `umap_layout_step` with `owners = new points`, `move_other = 0`), not a stub introduced here.

## Threat surface

No new security-relevant surface beyond the plan's `<threat_model>`. The mitigations are present: host-validated launch geometry before every `ArrayArg::from_raw_parts` and in-kernel `row < n_owners` / `other < n_vertices` bounds-guards (T-14-10); `SplitMix64::next_below` rejection sampling, never `% n` (T-14-11); finite `n_epochs` (T-14-12); NO device RNG — all draws host-side keyed by (seed, epoch, edge) (T-14-13); no `F::INFINITY` + the `0.001 + dist²` repulsive fudge (T-14-14); no package installs (T-14-SC).

## User Setup Required

None - no external service configuration required. (The property-gate fixtures are pre-committed blobs; CI never runs the generator.)

## Next Phase Readiness

- Plan 05 (`transform`) can now reuse `umap_layout_step` directly: place the `m` new points contiguously after the `n` frozen training rows, launch with `n_owners = m` and `move_other = 0` (the frozen-subset path is proven by the `umap_layout_step_launches_f64_owner_only` smoke test).
- The host epoch driver, `make_epochs_per_sample`, and the per-(seed, epoch, edge) negative-sampling substream are reusable for the transform's reduced-epoch (n_epochs=100) frozen layout.
- f64 is the cpu gate; rocm f32 is the opportunistic GPU gate (f64-on-rocm skips-with-log) — the kernel is generic and feature-free, so the rocm f32 path compiles and launches by construction.

## Self-Check: PASSED

- `crates/mlrs-kernels/src/umap_layout.rs` — FOUND (contains `umap_layout_step`)
- `crates/mlrs-backend/tests/umap_layout_test.rs` — FOUND
- `crates/mlrs-algos/src/manifold/umap.rs` — FOUND (contains `knn_graph` + full pipeline)
- `crates/mlrs-algos/tests/umap_test.rs` — FOUND (calibrated thresholds)
- `.planning/phases/14-umap/14-VALIDATION.md` — FOUND (contains ε + per-metric thresholds)
- Commits `ba96b4c`, `cf31482`, `6807dd9` — all FOUND in git history

---
*Phase: 14-umap*
*Completed: 2026-06-24*
