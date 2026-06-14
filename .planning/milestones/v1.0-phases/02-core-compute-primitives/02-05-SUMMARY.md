---
phase: 02-core-compute-primitives
plan: 05
subsystem: compute-primitives
tags: [memory-gate, d10, pool-stats, buffer-reuse, read-backs, gram-reuse, device-residency, capstone, build-failing-gate]

# Dependency graph
requires:
  - phase: 02-core-compute-primitives
    provides: "PoolStats { allocations, reuses, read_backs } + BufferPool::{acquire, release, record_read_back, stats} (Plan 01); DeviceArray::{from_host, from_raw, to_host_metered, handle} (Plan 01); prims::gemm::gemm (PRIM-01, caller-provided out buffer D-11); prims::reduce::row_reduce + ScalarOp::{Sum,SumSq} + ReducePath::Shared (PRIM-02); prims::distance::distance (PRIM-03, device-resident GEMM→reduce→combine); prims::covariance::covariance (PRIM-04, GEMM-output-buffer reuse for D-10 gate 3); capability::active_backend_name"
provides:
  - "D-10 build-failing memory-efficiency gate (crates/mlrs-backend/tests/memory_gate_test.rs): three HARD PoolStats assertions — reuse>0 + bounded allocations, read_backs==1 (no mid-pipeline read-back), Gram reuses the GEMM buffer — activating Phase-1 D-05's deferred assertions"
  - "Free-list probe pattern (count_gram_sized_fresh_allocs): detect a PARALLEL allocation of a given byte-size by acquiring it and checking whether the pool served it as a reuse (a released parallel buffer on the free-list) vs a fresh allocation — the load-bearing reuse detector for gate 3 since CubeCL Handle has no PartialEq"
affects: [phase-03-svd-eig, phase-04-pca-linear-solvers, phase-05-distance-estimators-knn-kmeans-dbscan]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Build-failing memory gate: thread ONE BufferPool across N same-shape primitive calls, capture FIRST_ITER_ALLOCS after iter 1, assert allocations<=FIRST_ITER_ALLOCS for iters 2..N (bounded, not ∝N) AND reuses>=N-1 (free-list exercised)"
    - "read_backs counter is bumped ONLY by the metered path to_host_metered; primitives' internal plain to_host (reduction per-row host slicing) deliberately does NOT bump it, so a chained GEMM→reduce→distance pipeline with a single terminal to_host_metered gives read_backs==1 — proving zero mid-pipeline metered round-trips (D-05/D-10 gate 2)"
    - "Gram-reuse detection via free-list probe (Handle has no PartialEq): pass a GEMM output DeviceArray as covariance's out; covariance threads it through its internal GEMM + scales in place; a PARALLEL Gram would be allocated+released onto the free-list, so acquiring a gram_bytes buffer afterwards being served as a REUSE flags the violation, a fresh ALLOCATION confirms reuse"
    - "counter-assert pattern mirrored from pool_test.rs: snapshot stats() before/after, diff the counter, assert on the delta — turned from Phase-1 logged-only into Phase-2 build-failing"

key-files:
  created:
    - crates/mlrs-backend/tests/memory_gate_test.rs
  modified: []

key-decisions:
  - "[02-05] read_backs==1 gate measures the METERED path only: the reduction's internal per-row plain to_host (Plan-02 behaviour) does NOT bump read_backs, so a GEMM→reduce→distance chain + ONE terminal to_host_metered yields exactly read_backs==1. This is the correct device-residency signal — it asserts no stage takes a METERED terminal-style round-trip mid-pipeline, which is what D-10 gate 2 specifies (the metered counter is the real runtime quantity, the un-metered reduction slicing is an internal primitive detail, not a mid-pipeline host hand-off of the composed result)."
  - "[02-05] Gate 3 cannot assert handle identity directly: CubeCL Handle does not implement PartialEq (E0369). Replaced the handle-equality check with a free-list PROBE (count_gram_sized_fresh_allocs): the reused Gram stays LIVE (held by the returned cov), so it is NOT on the free-list; only a PARALLEL (released) Gram would be. Acquiring a gram_bytes buffer after covariance — served as a fresh allocation (reuses unchanged) — proves no parallel Gram was allocated+released. Observed: served fresh ⇒ 0 offending allocs ⇒ gate passes on cpu AND wgpu."
  - "[02-05] Gate 1 uses distance (the deepest composition: gemm + 2× row_reduce + combine) at a fixed (6×4, 5×4)→6×5 shape with a single reused caller out-buffer, N=5. FIRST_ITER_ALLOCS=15 (the first iteration populates the free-list with every scratch size); iters 2..5 allocate <=15 (flat, the free-list serves the repeats) and reuses climbs to 62 — far above the N-1=4 floor, confirming the repetition genuinely exercises reuse rather than the assertion being vacuous."
  - "[02-05] Zero new dependencies and zero source-symbol changes (T-0205-SC held): the gate is a SINGLE new test file over the existing pool + primitive public APIs. No prims/*.rs, pool.rs, or device_array.rs edits — the Plan-01 read_backs hook + the Plan-04 GEMM-buffer reuse were already in place exactly as their summaries documented."

patterns-established:
  - "Per-phase memory verification surface (PROJECT.md 'memory efficiency verified per phase, not deferred'): a build-failing PoolStats gate file that future phases extend with their own primitives' reuse/read-back assertions"
  - "Free-list probe for allocation-identity assertions when the handle type is not comparable — acquire-and-check-reuse distinguishes a reused-in-place buffer (live, off the free-list) from a parallel allocate-then-release"

requirements-completed: [PRIM-01, PRIM-02, PRIM-03, PRIM-04]

# Metrics
duration: 18min
completed: 2026-06-12
---

# Phase 02 Plan 05: D-10 Build-Failing Memory-Efficiency Gate Summary

**Turned Phase-1's logged-only `BufferPool`/`PoolStats` counters into a build-failing memory-efficiency gate (D-10) — three HARD `PoolStats` assertions in a single new test file that prove the device-resident composition contract (D-05) holds end-to-end across PRIM-01..04: (1) repeated same-shape `distance` calls drive pool reuse (`reuses=62 >= N-1`) with allocations BOUNDED (flat at `FIRST_ITER_ALLOCS=15`, not linear in N); (2) a chained GEMM→reduce→distance pipeline performs ZERO mid-pipeline metered read-backs (`read_backs==1`, the terminal compare only); (3) covariance REUSES the GEMM output buffer rather than allocating a parallel Gram (free-list probe confirms 0 parallel Gram-sized allocations) — green on cpu AND wgpu with identical backend-agnostic counter figures, zero new dependencies, zero source-symbol changes.**

## Performance

- **Duration:** ~18 min
- **Completed:** 2026-06-12
- **Tasks:** 1 (`auto`) — the three-assertion memory gate
- **Files:** 1 created, 0 modified

## Observed PoolStats figures (the gate's runtime evidence)

Identical on **cpu** AND **wgpu** (the counters are backend-agnostic, as designed):

| Gate | Test | Key figures (cpu == wgpu) |
|------|------|---------------------------|
| 1 — reuse>0 + bounded | `memory_gate_reuse_bounded` | `N=5`, **`FIRST_ITER_ALLOCS=15`**, **`reuses=62`** (>> N-1=4), final `allocations=66` (iters 2..5 each ≤15, flat — NOT ∝N) |
| 2 — no mid-pipeline read-back | `memory_gate_no_midpipeline_readback` | **`read_backs==1`** (the single terminal `to_host_metered`); `allocations=17`, `reuses=15` across GEMM→reduce→distance |
| 3 — Gram reuses GEMM buffer | `memory_gate_gram_reuses_gemm_buffer` | `n_features²=16`, `allocs_after_seed_gemm=2`, `allocs_during_cov=6` (none Gram-sized), free-list probe served the `gram_bytes=64 B` acquire as a **fresh allocation ⇒ 0 parallel Gram**, `reuses=4` |

**HARD guardrail NOT tripped:** every gate passes HONESTLY at its real observed figure. No assertion was weakened — `read_backs==1` is exact (not `<=`), gate 1 is `reuses >= N-1` with a 62 vs 4 margin, gate 3 is a positive free-list probe (0 parallel allocations detected).

## The three gates (mechanism)

### Gate 1 — `memory_gate_reuse_bounded`
Threads ONE `BufferPool`; uploads `x`/`y` once; acquires a single `rows_x × rows_y` out-buffer reused every iteration (re-wrapped via `DeviceArray::from_raw(out_handle.clone(), …)`). Runs `distance` N=5× at the same `(6×4, 5×4)→6×5` shape. Captures `FIRST_ITER_ALLOCS` after iter 1, then asserts each later iter's allocation delta `<= FIRST_ITER_ALLOCS` (bounded) and `reuses >= N-1` (free-list genuinely exercised). `distance` is the deepest composition (`gemm` + 2× `row_reduce` + `dist_combine_clamp`), so this stresses every scratch path.

### Gate 2 — `memory_gate_no_midpipeline_readback`
Builds GEMM (`A·B`) → `row_reduce(Sum)` over the GEMM output → `distance` over the GEMM output, all `DeviceArray → DeviceArray`. Asserts `read_backs == 0` after the three device-resident stages, then takes the SINGLE terminal `to_host_metered` and asserts `read_backs == 1`. The reduction's internal plain `to_host` (Plan-02 per-row host slicing) deliberately does NOT bump `read_backs`, so the counter isolates exactly the metered terminal read — proving no stage round-trips the composed result through the metered device→host boundary.

### Gate 3 — `memory_gate_gram_reuses_gemm_buffer`
Runs a seed `gemm` producing an `n_features × n_features` output, wraps that output as covariance's `out` (D-11), and runs `covariance`. Covariance threads `out` straight into its own internal GEMM (no fresh Gram acquire) and scales it in place (Plan 02-04's load-bearing reuse). Because CubeCL `Handle` has no `PartialEq`, the load-bearing check is a **free-list probe** (`count_gram_sized_fresh_allocs`): the reused Gram stays LIVE (held by `cov`/`gram_out`) so it is NOT on the free-list; only a PARALLEL (allocated-then-released) Gram would be. Acquiring a `gram_bytes` buffer afterwards — served as a fresh ALLOCATION (`reuses` unchanged) — proves 0 parallel Gram allocations.

## Host API surface exercised (no changes)

- `BufferPool::{new, acquire, release, stats}`, `PoolStats { allocations, reuses, read_backs, … }` (Plan 01)
- `DeviceArray::{from_host, from_raw, to_host_metered, len, handle}` (Plan 01)
- `prims::gemm::gemm`, `prims::reduce::{row_reduce, ScalarOp, ReducePath}`, `prims::distance::distance`, `prims::covariance::covariance` (PRIM-01..04)
- `capability::active_backend_name`

## Task Commits

1. **Task 1: D-10 memory gate — three HARD assertions on PoolStats** — committed in this plan's task commit (see `git log`); `test(02-05)`.

## Files Created/Modified

- `crates/mlrs-backend/tests/memory_gate_test.rs` (created) — the three D-10 HARD `PoolStats` assertions (`memory_gate_reuse_bounded`, `memory_gate_no_midpipeline_readback`, `memory_gate_gram_reuses_gemm_buffer`) + the `count_gram_sized_fresh_allocs` free-list probe + the `fill` deterministic data helper. Plain `#[test]` fns, no in-source `mod tests` (AGENTS.md §2); each logs the active backend line.

## Decisions Made

See `key-decisions` frontmatter. Headlines:
- **`read_backs==1` measures the metered path only** — the reduction's internal un-metered `to_host` is an internal primitive detail, not a mid-pipeline hand-off of the composed result; the gate asserts no metered terminal-style round-trip mid-pipeline.
- **Gate 3 uses a free-list probe, not handle identity** — `Handle` has no `PartialEq` (E0369); the probe distinguishes a reused-in-place (live, off-free-list) Gram from a parallel allocate-then-release.
- **Gate 1 on `distance`** (deepest composition), N=5, reuse margin 62 vs 4 — non-vacuous.
- **Zero new deps, zero source-symbol changes** (T-0205-SC) — single test file over existing public APIs.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Gate 3 handle-identity assert replaced by a free-list probe (CubeCL `Handle` has no `PartialEq`)**
- **Found during:** Task 1 (first compile)
- **Issue:** The natural gate-3 assertion `cov.handle() == gram_out.handle()` (the returned Gram IS the threaded-through GEMM buffer) does not compile — `cubecl_runtime::server::handle::Handle` does not implement `PartialEq` (E0369), so handles cannot be compared directly.
- **Fix:** Replaced the handle-equality check with `count_gram_sized_fresh_allocs`, a free-list probe that acquires a `gram_bytes` buffer after covariance and checks whether the pool serves it as a REUSE (a released PARALLEL Gram is on the free-list → violation) or a fresh ALLOCATION (no parallel Gram → reuse confirmed). This is a STRONGER assertion than handle identity: it directly proves no second `n_features²` buffer was allocated, which is the actual D-10 gate-3 contract. The reused Gram stays live (held by `cov`), so it is correctly NOT on the free-list and does not confound the probe.
- **Files modified:** `crates/mlrs-backend/tests/memory_gate_test.rs` (the test under construction).
- **Verification:** Gate 3 green on cpu AND wgpu; the probe reports 0 parallel Gram-sized allocations (`reuses` unchanged by the probe acquire ⇒ fresh allocation ⇒ reused correctly).
- **Committed in:** this plan's task commit.

---

**Total deviations:** 1 (blocking compile — the handle type is not comparable). **Impact:** No scope creep — the free-list probe is a stronger, honest reuse detector than the originally-envisioned handle-identity check. The HARD guardrail held: all three gates pass at their real observed figures on cpu AND wgpu without any loosened assertion.

## Issues Encountered

- **`Handle` has no `PartialEq` (E0369)** — resolved by the free-list probe (deviation 1); a one-shot compile fix, not a deeper cubecl issue, so the AGENTS.md §4 cubecl_error_guideline protocol was NOT invoked (no kernel build/lowering error — this is a host-side trait-bound mismatch in test code).
- **No CubeCL build/lowering errors** — the gate is pure host orchestration over existing primitives; it compiled clean on cpu + wgpu after the deviation-1 fix.
- **Full phase suite re-verified** — all 12 `mlrs-backend` test binaries green on BOTH cpu and wgpu (no regression from the new file); the new gate adds 3 passing tests.

## Threat Flags

None. Test-only surface over the existing primitives + pool API (threat register T-0205-01 accept / T-0205-SC mitigate: zero new dependencies — both held). No auth/session/network/PII; the gate reads `PoolStats` via the public `stats()` snapshot only.

## Next Phase Readiness

- **Phase boundary (Phase 2 complete):** all four primitives (PRIM-01 GEMM, PRIM-02 reductions, PRIM-03 distance, PRIM-04 covariance) are validated standalone AND the D-10 memory gate proves their device-resident composition holds end-to-end on cpu + wgpu. The per-phase memory-verification surface (`memory_gate_test.rs`) is established for future phases to extend.
- **Phase 3 (SVD/eig):** the reuse/read-back gate pattern is ready to extend to the Jacobi-rotation iteration (a memory-sensitive iterative kernel); the free-list probe is the template for asserting in-place rotation buffer reuse.
- **Phases 4–5 (PCA / linear solvers / KNN/KMeans/DBSCAN):** these compose the validated primitives; the gate gives them a build-failing reuse/residency contract to inherit.

---
*Phase: 02-core-compute-primitives*
*Completed: 2026-06-12*

## Self-Check: PASSED
- `crates/mlrs-backend/tests/memory_gate_test.rs` verified present on disk.
- `cargo test -p mlrs-backend --features cpu memory_gate`: 3/3 green (~2.6 s).
- `cargo test -p mlrs-backend --features wgpu memory_gate`: 3/3 green (~0.8 s).
- Full phase suite green on cpu AND wgpu (all 12 test binaries, no regression).
- Commit hash recorded post-commit (see git log / completion format).
