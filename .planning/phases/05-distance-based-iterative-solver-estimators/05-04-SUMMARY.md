---
phase: 05-distance-based-iterative-solver-estimators
plan: 04
subsystem: kernels
tags: [dbscan, eps-core-mask, cluster, gather, no-atomics, cpu-mlir, oracle, primitive-first, d-04-readback]

# Dependency graph
requires:
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 01
    provides: "dbscan.rs/prims/dbscan.rs/dbscan_mask_test.rs stubs + lib.rs/prims/mod.rs registrations; dbscan_{f32,f64}_seed42.npz fixtures; i32 DeviceArray (D-06)"
  - phase: 02-foundational-primitives
    provides: "prims::distance (GEMM-expansion squared-Euclidean, sqrt=false), dist_combine_clamp 2D (i,j) map shape"
provides:
  - "mlrs_kernels::dbscan::eps_core_count — feature-free #[cube] eps-threshold + per-row core-count kernel, GATHER per row (no atomics/no SharedMemory, cubecl-cpu safe)"
  - "mlrs_backend::prims::dbscan::eps_core_mask — validate-before-launch wrapper: n² device distance + threshold → D-04 host readback of (is_core mask, counts, n×n adjacency)"
  - "EpsCoreMask host struct (is_core/counts/adjacency + neighbors(i)) for the estimator's index-ordered DFS"
  - "dbscan_mask_test.rs standalone oracle GREEN on cpu(f64): is_core integer-exact vs sklearn core_sample_indices_ (f32+f64), self-inclusivity, bad-input guards"
affects: [05-07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GATHER-per-row #[cube] (one unit per point owns its whole row of the n² matrix) — every output slot single-owner, so NO cross-unit atomic (cubecl-cpu does not lower cross-unit atomics) and NO scatter; uses only F/u32 accumulators + if-guards, no SharedMemory/no mutable bool/no F::INFINITY"
    - "f64-precise host scalar into a #[cube] kernel via bytemuck reinterpret (covariance.rs:281 idiom) — NOT F::new(f32), which truncates the f64 threshold and breaks the integer-exact count oracle"
    - "DBSCAN bad-hyperparameter (eps<0, min_samples<1) surfaces as PrimError::ShapeMismatch on a synthetic operand (distance.rs all-geometry-as-ShapeMismatch convention; PrimError has no InvalidEps/InvalidMinSamples)"

key-files:
  created: []
  modified:
    - "crates/mlrs-kernels/src/dbscan.rs (filled the 05-01 stub: eps_core_count #[cube] GATHER kernel)"
    - "crates/mlrs-backend/src/prims/dbscan.rs (filled the 05-01 stub: eps_core_mask wrapper + EpsCoreMask struct + validate + f64_to_f)"
    - "crates/mlrs-backend/tests/dbscan_mask_test.rs (de-#[ignore]d: real sklearn core-mask oracle + self-inclusivity + bad-input guards)"

key-decisions:
  - "GATHER per ROW (one unit per point scans its own row, accumulates a private u32 count, writes its own adjacency row) — chosen over a 2D (i,j) map + scatter+atomic because the critical-constraint note says cubecl-cpu (the primary gate) does NOT lower cross-unit atomics. Every output slot (count[i], adj[i,*]) has a single owner, so no atomic is needed and the kernel launches cleanly on cpu(f64)."
  - "Core decision count >= min_samples applied HOST-SIDE after the D-04 readback (the device does only the n² threshold/count) — matches the plan's 'keep device work on the n² threshold/count' guidance and keeps the kernel branch-free of the min_samples scalar."
  - "eps2 passed as a bytemuck-reinterpreted F (full f64 precision), not F::new((eps*eps) as f32) — the f64 oracle is integer-exact on the count, so an f32-truncated threshold near a fixture boundary point could flip a core/non-core bit."
  - "EpsCoreMask returns is_core + counts + the full n×n bool adjacency (+ a neighbors(i) helper) — the estimator's plan-07 DFS walks the adjacency in ascending index order; this prim does NOT expand clusters (D-04: sequential graph traversal is the estimator's host job)."

patterns-established:
  - "cubecl-cpu-safe GATHER idiom: when a parallel reduction would need a cross-unit atomic (scatter+accumulate), re-shape it so each unit owns a complete output partition (here: row i) and accumulates privately — no atomic, no SharedMemory, launches on the cpu MLIR gate"

requirements-completed: [CLUSTER-02]

# Metrics
duration: 8min
completed: 2026-06-13
---

# Phase 5 Plan 04: DBSCAN eps-core-mask Primitive (D-04) Summary

**The genuinely-new DBSCAN device primitive: a feature-free `#[cube]` eps-threshold + per-row core-count kernel (`eps_core_count`, GATHER-per-row, no atomics/no SharedMemory) plus its `eps_core_mask` validate-before-launch wrapper, composing the Phase-2 pairwise-distance prim to compute the `n²` squared-distance matrix, threshold it self-inclusively at `eps²`, and read the core mask + `n × n` adjacency back to host as the single documented D-04 round-trip — standalone oracle GREEN on cpu(f64), `is_core` INTEGER-EXACT vs sklearn `core_sample_indices_` before the DBSCAN estimator (plan 07) consumes it (D-01 primitive-first).**

## Performance

- **Duration:** ~8 min
- **Tasks:** 2 (both TDD)
- **Files modified:** 3 (kernel + prim + test — all 05-01 stubs filled; zero shared-file edits)

## Accomplishments
- Filled `mlrs_kernels::dbscan::eps_core_count`: one UNIT per point `i` scans its own row of the `n × n` squared-distance matrix, writing each self-inclusive adjacency bit `adj[i*n+j] = (d2[i,j] <= eps²)` and accumulating the row's eps-neighbor count in a private `u32`. A GATHER — every output slot is single-owner, so NO cross-unit atomic and NO scatter (the cubecl-cpu MLIR lowering does not lower cross-unit atomics). Generic `<F: Float + CubeElement>`, scalar `eps2: F` / `n: u32` by value, no hardcoded plane width, `if i < n` bounds-check.
- Filled `mlrs_backend::prims::dbscan::eps_core_mask`: validates `n*d == x.len()` AND `eps >= 0` (finite) AND `min_samples >= 1` → `PrimError::ShapeMismatch` BEFORE any unsafe launch (T-05-04-01 / ASVS V5); composes the Phase-2 `distance(x, x, sqrt=false)` for the `n²` matrix; launches `eps_core_count` with a bytemuck-reinterpreted f64-precise `eps2`; reads the per-row count + `n × n` adjacency back to host (the cholesky tiny-readback idiom scaled to n²) — the D-04 documented single round-trip; derives `is_core[i] = count[i] >= min_samples` host-side; releases the n² scratch back to the pool. Returns an `EpsCoreMask { is_core, counts, adjacency }` (+ `neighbors(i)`) for the estimator's DFS.
- De-`#[ignore]`d `dbscan_mask_test.rs` with the real standalone oracle: `eps_core_mask(X, n, d, eps, min_samples)` vs the committed `dbscan_{f32,f64}_seed42.npz` sklearn reference — `is_core[i]` true iff `i ∈ core_sample_indices`, INTEGER-EXACT (no tolerance — it is a count threshold), f64 cpu-gated. Added a self-inclusivity case (every `count[i] >= 1`, diagonal adjacency bit set — Pitfall 7) and a bad-input guard (n*d mismatch / negative eps / min_samples=0 → typed `ShapeMismatch`).
- Verified the full gate: `cargo build -p mlrs-kernels` green; `cargo test --features cpu -p mlrs-backend --test dbscan_mask_test` 5/5 green (incl. f32 + f64 + self-inclusivity + guard); `cargo build -p mlrs-backend --features rocm --tests` green; lib.rs/prims/mod.rs untouched.

## Task Commits

1. **Task 1: dbscan eps-threshold + per-row core-count `#[cube]` kernel** — `9f52a5e` (feat)
2. **Task 2: eps_core_mask wrapper + standalone oracle** — `71a854b` (feat)

## Files Created/Modified
- `crates/mlrs-kernels/src/dbscan.rs` — `eps_core_count` `#[cube]` GATHER kernel; `pub use self::eps_core_count as dbscan_eps_core_count` inside the file. (lib.rs untouched.)
- `crates/mlrs-backend/src/prims/dbscan.rs` — `eps_core_mask` wrapper + `EpsCoreMask` struct + `validate` + `f64_to_f` + `launch_dims_1d`. (prims/mod.rs untouched.)
- `crates/mlrs-backend/tests/dbscan_mask_test.rs` — `check_dbscan_mask` oracle body + 5 tests (fixture_loads, sklearn f32/f64 core-mask, self-inclusivity f64, bad-input guard).

## Decisions Made
- **GATHER per row, not 2D-map + scatter+atomic:** the natural shape (mirroring `dist_combine_clamp`'s 2D `(i,j)` map) would have each `(i,j)` thread test `d2[i,j] <= eps²` and atomically increment `count[i]` — a SCATTER needing a cross-unit atomic. The critical-constraint note states cubecl-cpu (the primary correctness gate) does NOT lower cross-unit atomics. So the kernel was written as a GATHER: one unit per point `i` scans its OWN row sequentially, accumulating a private `u32` and writing its own adjacency row. Every output slot has a single owner → no atomic, no SharedMemory → it launches cleanly on cpu(f64). (See Deviations — this is a plan-permitted Claude-discretion shaping, not a rule trigger.)
- **Core decision host-side:** `is_core[i] = count[i] >= min_samples` is computed on the host after the readback (the plan grants Claude's discretion to "keep device work on the n² threshold/count"). The kernel never sees `min_samples`, so the device output is purely the threshold/count and the host owns the cluster-membership decision (consistent with the D-04 host-graph-walk split).
- **f64-precise `eps2` via bytemuck:** `F::new` only takes `f32`, which would truncate the f64 threshold. Since the oracle is integer-exact on the count, a boundary point at exactly `d² ≈ eps²` could flip core/non-core under f32 truncation. The threshold is constructed with the `covariance.rs:281` reinterpret idiom so f64 keeps full precision.
- **Bad eps/min_samples as `ShapeMismatch`:** `PrimError` has no `InvalidEps`/`InvalidMinSamples` variant (those live in `AlgoError`, the estimator layer); following `distance.rs`/`topk.rs` convention all prim-level operand violations surface as `ShapeMismatch` on a synthetic operand name (`"eps"` / `"min_samples"`).

## Deviations from Plan

### Auto-fixed Issues

None that triggered a deviation rule. The plan's `<action>` explicitly grants Claude's discretion on the kernel shape ("output a per-row `core_count` and/or the adjacency bitmask"; "the core decision `count >= min_samples` may be applied host-side after readback"). Two discretion calls were made within that latitude:

1. **GATHER-per-row kernel shape (in-scope design choice):** rather than the 2D `(i,j)` map + per-row atomic count, the kernel uses one unit per row (GATHER) so no cross-unit atomic is needed — required to launch on the cubecl-cpu MLIR gate (the critical-constraint note). The kernel's public signature (`d2`, `adj`, `count`, `eps2`, `n`) and the self-inclusive `<= eps²` count contract are exactly as the plan specifies; only the internal parallelization shape is the GATHER form. Committed in Task 1 (`9f52a5e`).
2. **f64-precise `eps2` (correctness):** used the bytemuck reinterpret rather than `F::new(f32)` so the f64 oracle's integer-exact count is not threatened by an f32-truncated threshold. Committed in Task 2 (`71a854b`).

Both are within the plan's stated discretion + the project's correctness mandate; neither is an architectural change (no new tables, no library switch, no public-API restructuring beyond the `EpsCoreMask` return type the plan's signature sketch already anticipated).

## Known Stubs

None. Both stub files were fully implemented and the oracle test exercises real device output (the `is_core` mask is derived from the kernel's actual `count` readback, not a hardcoded value).

## Issues Encountered
- None blocking. The GATHER re-shaping (vs the atomic scatter the 2D map would imply) was anticipated by the critical-constraint note and applied up front, so the kernel launched on cpu(f64) on the first run — no `failed to run pass` MLIR panic (the 05-02/05-03 failure mode was avoided by construction).

## Next Phase Readiness
- **Plan 05-07 (DBSCAN estimator) unblocked:** `eps_core_mask` returns `EpsCoreMask { is_core, counts, adjacency }` with a `neighbors(i)` ascending-index helper — exactly the host inputs the estimator's index-ordered DFS cluster expansion (Pitfall 7) needs. The D-04 split holds: this prim does the n² device threshold/count + the single readback; the estimator does the sequential host graph walk.
- No blockers. cpu(f64) full + rocm(f32) test-target build both green; lib.rs/prims/mod.rs untouched so the sibling Wave-2 prim plans stay file-disjoint.

## Threat Flags

None — no new network/auth/file surface. The two trust boundaries in the register are mitigated exactly as specified: T-05-04-01 (geometry + hyperparameters) by validate-before-launch → `PrimError::ShapeMismatch`; T-05-04-02 (n² allocation) by the bounded-and-reused n² distance matrix (released back to the pool after the kernel), documented in the wrapper as the D-04 accepted brute-force v1 cost (plan 11 gates the bound). Zero new dependencies (T-05-04-SC accept).

## Self-Check: PASSED

- All modified files verified present (dbscan.rs kernel, prims/dbscan.rs wrapper, dbscan_mask_test.rs, this SUMMARY).
- Both task commits verified in git history (`9f52a5e`, `71a854b`).
- `cargo test --features cpu -p mlrs-backend --test dbscan_mask_test` 5/5 green (incl. f32 + f64 + self-inclusivity + bad-input guard); `cargo build -p mlrs-kernels` + `-p mlrs-backend --features rocm --tests` green; lib.rs/prims/mod.rs untouched.

---
*Phase: 05-distance-based-iterative-solver-estimators*
*Completed: 2026-06-13*
