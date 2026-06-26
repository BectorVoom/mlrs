---
phase: 15-hdbscan
plan: 05
subsystem: cluster/hdbscan
status: complete
tags: [hdbscan, kernel, cpu-mlir, knn-graph, mutual-reachability, device-front-end]
requirements: [HDBS-01, HDBS-02]
dependency_graph:
  requires:
    - "15-03: Metric enum, mst.rs (Variant A/B + core_distances_dense), single_linkage.rs"
    - "15-04: host back-end (condense/stability/select), tree_to_labels, precomputed path"
    - "Phase-13: knn_graph prim (include_self=true), Metric, distance kernels"
  provides:
    - "mlrs_kernels::mutual_reachability — SharedMemory-free 2D-GATHER MR kernel"
    - "mlrs_backend::prims::mutual_reachability::mutual_reachability_device — host launch wrapper"
    - "Hdbscan::fit feature-metric device front-end (all 5 metrics exact)"
  affects:
    - "15-06: store_centers/outlier_scores build on the now-complete feature-metric fit"
tech_stack:
  added: []
  patterns:
    - "per-element 2D GATHER kernel (chebyshev_dist running-max precedent)"
    - "device stage -> to_host -> host walk -> from_host (DBSCAN shape)"
    - "Variant-A (dense cosine + MR kernel) vs Variant-B (source-tracking, no n*n) MST routing"
key_files:
  created:
    - crates/mlrs-kernels/src/mutual_reachability.rs
    - crates/mlrs-backend/src/prims/mutual_reachability.rs
    - crates/mlrs-backend/tests/mutual_reachability_test.rs
  modified:
    - crates/mlrs-kernels/src/lib.rs
    - crates/mlrs-backend/src/prims/mod.rs
    - crates/mlrs-algos/src/cluster/hdbscan.rs
    - crates/mlrs-algos/tests/hdbscan_test.rs
decisions:
  - "MR kernel VALUE gate (R-9) lives in mlrs-backend, not mlrs-kernels (no runtime in the kernel crate)"
  - "Cosine = dense Variant-A via the MR kernel; the other 4 = Variant-B (no n*n resident)"
  - "Variant-A cosine: scale whole matrix /alpha for core, kernel does d/alpha on RAW dist (equivalent)"
metrics:
  tasks_completed: 2
  files_created: 3
  files_modified: 4
  commits: 2
  completed: 2026-06-24
---

# Phase 15 Plan 05: HDBSCAN feature-metric device front-end Summary

A SharedMemory-free 2D-GATHER `mutual_reachability` kernel plus the feature-metric `Hdbscan::fit` wiring (KNN-prim core distances + Variant-A/B MST routing → Wave-3 host back-end), extending the exact -1-pinned label gate from precomputed to all 5 feature metrics × {f32, f64} with a sub-quadratic PoolStats memory gate.

## What was built

### Task 1 — `mutual_reachability` GATHER kernel (commit `647a89a`)
- `mlrs_kernels::mutual_reachability` (`#[cube(launch)]`): per-element 2D GATHER computing `out[i*rows_y+j] = max(core_i, core_j, d_ij/alpha)` via the cpu-MLIR-safe STATEMENT-form three-way running max (the `chebyshev_dist` running-max precedent). SharedMemory-free, no loop, no cross-thread state. `ABSOLUTE_POS_X/Y`, `CubeDim{16,16}`, guarded `if i<rows_x { if j<rows_y {…} }`. `alpha` by value (cubecl 0.10).
- `mlrs_backend::prims::mutual_reachability::mutual_reachability_device`: validate-before-launch host wrapper (owns `ActiveRuntime`); `checked_mul` on `rows_x*rows_y` + `u32`-fit guards (T-15-05-V5/OVF) before the `unsafe` launch.
- VALUE gate (`tests/mutual_reachability_test.rs`): launches under cpu-MLIR f64 + f32 and asserts MR values against an in-test host reference incl. a **duplicate-point row** (R-9, the silent-miscompile catch), with an explicit `max(core1,core2)` assertion on the duplicate pair. Both `alpha=1` and `alpha=2` arms exercised.

### Task 2 — feature-metric `fit` device front-end (commit `9875dcc`)
- `Hdbscan::fit` now routes the 5 feature metrics through `feature_metric_single_linkage`:
  - **Core distances**: `knn_graph(include_self=true, k=min_samples)` → column `min_samples-1` of the ascending per-row distances (RAW core).
  - **Cosine → Variant A**: build the dense `n×n` cosine distance matrix (`cosine_distance_matrix`, matching sklearn `pairwise_distances('cosine')`, Pitfall 3 — all pairs, not the kNN set); scale the whole matrix by `alpha` for core distances; launch the MR **kernel** on the device (RAW dist + scaled core + alpha); dense Variant-A Prim.
  - **euclidean/manhattan/chebyshev/minkowski → Variant B**: `mst_from_data_matrix` with a host `host_pairwise` closure (`pair_distance /= alpha`, RAW core) — **no `n×n` device block** ever resident.
  - Hierarchy → the Wave-3 host back-end (`tree_to_labels`) → labels + probabilities, identical to the precomputed path.
- `knn_metric()` maps the estimator `Metric` onto the Phase-13 prim `Metric`.
- Un-ignored gates (now real fits): `labels_match_sklearn_{euclidean,manhattan,cosine,chebyshev,minkowski}_{f32,f64}` — all exact -1-pinned permutations of sklearn.
- Un-ignored `selection_knob_alpha_feature_path` (the 15-04 deferral): the feature Variant-B `alpha=0.5` path matches the feature-path `labels_alpha` oracle exactly.
- `memory_gate`: added the sub-quadratic `peak_bytes < n*n*8` assertion (Variant-B no-`n×n` guarantee) alongside the re-fit-no-`live_bytes`-growth idiom; `n=64`.

## Verification

| Gate | Command | Result |
|------|---------|--------|
| MR kernel VALUE (R-9) | `cargo test --features cpu -p mlrs-backend --test mutual_reachability_test` | 2 passed |
| Feature-metric labels | `cargo test --features cpu --test hdbscan_test labels_match_sklearn` | 12 passed (5 metrics × {f32,f64} + precomputed) |
| Memory gate | `cargo test --features cpu --test hdbscan_test memory_gate` | 1 passed (sub-quadratic peak) |
| Alpha feature path | `cargo test --features cpu --test hdbscan_test selection_knob_alpha_feature_path` | 1 passed |
| Full HDBSCAN suite | `cargo test --features cpu --test hdbscan_test` | 34 passed, 4 ignored (centers/outlier_scores → 15-06), 0 failed |

MR kernel is SharedMemory-free (`grep -Ec 'SharedMemory\|Atomic\|INFINITY'` == 0), runs on cpu-MLIR f64 without panic; f64-on-rocm skips-with-log via the capability gate.

## Deviations from Plan

### [Rule 3 — Blocking] Module/function name collision in the kernel re-export
- **Found during:** Task 1 (compiling `mlrs-kernels`).
- **Issue:** `pub mod mutual_reachability;` (module, type namespace) + `pub use mutual_reachability::mutual_reachability;` (function) collided — `E0255 the name 'mutual_reachability' is defined multiple times`.
- **Fix:** Re-export the kernel under an alias `pub use mutual_reachability::mutual_reachability as mutual_reachability_kernel;`; the backend wrapper calls `mutual_reachability_kernel::launch`. The kernel fn itself keeps the planned name `pub fn mutual_reachability` (the must_have `contains` is satisfied in the kernel file).
- **Files:** `crates/mlrs-kernels/src/lib.rs`, `crates/mlrs-backend/src/prims/mutual_reachability.rs`
- **Commit:** `647a89a`

### [Rule 3 — Blocking] Plan's Task-1 verify command cannot run as written
- **Found during:** Task 1.
- **Issue:** `cargo test --features cpu -p mlrs-kernels mutual_reachability` errors — `mlrs-kernels` carries NO backend feature (Criterion 1: runtime-free crate), and has no runtime to LAUNCH a kernel. The prior-wave context flagged this `--features cpu` propagation gap.
- **Fix:** The kernel VALUE gate (incl. R-9) lives in `mlrs-backend` (`tests/mutual_reachability_test.rs`), where `ActiveRuntime` exists — mirroring how `distance.rs` kernels are value-tested through the `knn_graph` prim, not in `mlrs-kernels`. The kernel still compiles under `cargo build -p mlrs-kernels`. Logged in `deferred-items.md`.
- **Files:** `crates/mlrs-backend/tests/mutual_reachability_test.rs`
- **Commit:** `647a89a`

## Deferred Issues (out of scope — not caused by this plan)
- `cargo clippy --features cpu` fails in `mlrs-kernels` (`elementwise.rs:282` `approx_constant`, pre-existing, documented in `deferred-items.md` since 15-04). New MR-kernel code is clean except the INTENTIONAL `collapsible_if` warning (the nested guard is the mandated cpu-MLIR shape, matching `distance.rs`). `cargo build --features cpu -p mlrs-algos`/`-p mlrs-backend` are warning-free on all new code.
- `centers_match` / `outlier_scores_match` gates stay `#[ignore]`d — they need the `store_centers` / GLOSH accessors landing in plan 15-06.

## Known Stubs
None — the feature-metric `fit` is fully wired; `labels_`, `probabilities_`, and `single_linkage_` are all populated for the 5 feature metrics (no all-`-1` trivial path remains).

## Self-Check: PASSED
- `crates/mlrs-kernels/src/mutual_reachability.rs` — FOUND
- `crates/mlrs-backend/src/prims/mutual_reachability.rs` — FOUND
- `crates/mlrs-backend/tests/mutual_reachability_test.rs` — FOUND
- commit `647a89a` — FOUND
- commit `9875dcc` — FOUND
