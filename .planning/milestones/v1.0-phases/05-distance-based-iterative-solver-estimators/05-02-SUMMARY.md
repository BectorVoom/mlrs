---
phase: 05-distance-based-iterative-solver-estimators
plan: 02
subsystem: kernels
tags: [topk, select-k, knn, neighbors, cubecl, lowest-index-tie-break, oracle, primitive-first, cpu-mlir]

# Dependency graph
requires:
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 01
    provides: "topk.rs/prims/topk.rs/topk_test.rs stubs + lib.rs/prims/mod.rs registrations; knn_{f32,f64}_seed42.npz fixtures; i32 DeviceArray (D-06)"
  - phase: 02-foundational-primitives
    provides: "prims::distance (GEMM-expansion squared-Euclidean), reduce::argmin_shared (value+index pair carry, lowest-index tie), sqrt_elem boundary kernel"
provides:
  - "mlrs_kernels::topk::select_k — feature-free #[cube] partial-select-k kernel, lowest-index tie-break (D-02)"
  - "mlrs_backend::prims::topk::top_k — validate-before-launch wrapper returning device-resident (distances, u32 indices)"
  - "topk_test.rs standalone oracle GREEN on cpu(f64): distances within 1e-5 + indices exact vs sklearn kneighbors, constructed-tie + bad-geometry guards"
affects: [05-07, 05-08, 05-09]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Selection-by-rank #[cube] (no SharedMemory/no mutable bool/no F::INFINITY/no descending-shift) — the cubecl-cpu MLIR lowering rejects those; rank-order on the (value,index) PAIR reproduces the argmin_shared lowest-index tie k-fold"
    - "cubecl-cpu cross-loop u32 flags need an explicit type annotation (`let mut admit: u32 = 0u32;`) or the cube macro fails NativeExpand inference"
    - "top-k validates 1<=k<=cols + rows*cols==len as PrimError::ShapeMismatch (no dedicated InvalidK in PrimError) before any unsafe launch (distance.rs precedent)"

key-files:
  created: []
  modified:
    - "crates/mlrs-kernels/src/topk.rs (filled the 05-01 stub: select_k #[cube] selection-by-rank kernel)"
    - "crates/mlrs-backend/src/prims/topk.rs (filled the 05-01 stub: top_k launch wrapper, validate-before-launch, optional sqrt boundary, device-resident out)"
    - "crates/mlrs-backend/tests/topk_test.rs (de-#[ignore]d: real sklearn oracle + tie + bad-geometry assertions)"

key-decisions:
  - "select_k rewritten from a SharedMemory insertion-select to a SharedMemory-free selection-by-rank because the cubecl-cpu MLIR 'run pass' lowering rejected the mutable-bool / F::INFINITY / descending-shift constructs (cpu is the primary correctness gate) — the rank scan is provably identical to a k-fold argmin_shared over the distinct-index pairs"
  - "bad k (k<1 or k>cols) reported as PrimError::ShapeMismatch{operand:\"k\"} since PrimError has no InvalidK variant — matches distance.rs's all-geometry-as-ShapeMismatch convention"
  - "top-k selects on the SQUARED distance and sqrts ONLY the returned rows*k values at the boundary (Pitfall 8 / D-08) — monotone, so indices are unaffected"

patterns-established:
  - "cubecl-cpu-safe #[cube] selection idiom: order by the (value,index) pair, emit slot-by-slot as 'minimum pair strictly greater than the previous winner', seeded so slot 0 admits the global minimum — no SharedMemory, no bool, no infinity sentinel"

requirements-completed: [NEIGH-01, NEIGH-02, NEIGH-03]

# Metrics
duration: 23min
completed: 2026-06-13
---

# Phase 5 Plan 02: Top-k Select Primitive (D-02) Summary

**The genuinely-new neighbors device primitive: a feature-free `#[cube]` partial-select-k kernel (`select_k`, lowest-index tie-break) plus its `top_k` validate-before-launch wrapper, composing the Phase-2 pairwise-distance prim to return the k nearest (distances + i32-bound indices) per query row — standalone oracle GREEN on cpu(f64) within 1e-5 and index-exact vs sklearn `kneighbors` before any KNN estimator consumes it (D-01 primitive-first).**

## Performance

- **Duration:** ~23 min
- **Tasks:** 2 (both TDD)
- **Files modified:** 3 (kernel + prim + test — all 05-01 stubs filled; zero shared-file edits)

## Accomplishments
- Filled `mlrs_kernels::topk::select_k`: one cube per query row, unit 0 emits the k smallest by **selection-by-rank** over the `(value, index)` pair, applying the exact `argmin_shared` lowest-index tie rule (strictly-smaller value wins; on equal value the lower column index wins). Generic `<F: Float + CubeElement>`, scalar `rows/cols/k` by value, no hardcoded plane width.
- Filled `mlrs_backend::prims::topk::top_k`: validates `rows*cols == dist.len()` AND `1 <= k <= cols` → `PrimError::ShapeMismatch` BEFORE any unsafe launch (T-05-02-01 / ASVS V5); threads optional reused `out_val`/`out_idx` (D-11); launches `select_k::launch::<F, ActiveRuntime>` one cube per row; applies the optional Euclidean sqrt to ONLY the returned `rows×k` values (Pitfall 8 / D-08); returns device-resident `(distances, u32 indices)`.
- De-`#[ignore]`d `topk_test.rs` with the real standalone oracle: `distance(Xq, X, sqrt=false)` → `top_k(.., k, sqrt=true)` vs the committed `knn_{f32,f64}_seed42.npz` sklearn `kneighbors` reference — distances within 1e-5 AND indices EXACT, f64 cpu-gated. Added a constructed-tie case (two equal distances → lower index first) and a bad-geometry guard (k=0, k>cols, rows*cols mismatch → typed `ShapeMismatch`).
- Verified the full gate: `cargo build -p mlrs-kernels` green; `cargo test --features cpu -p mlrs-backend --test topk_test` 6/6 green (incl. f64 + tie + guard); `cargo build -p mlrs-backend --features rocm --tests` green.

## Task Commits

1. **Task 1: fill the top-k `#[cube]` kernel (partial-select-k, lowest-index tie-break)** — `16e2242` (feat)
2. **Task 2: fill the `top_k` launch wrapper + standalone oracle** — `5d0b958` (feat)

## Files Created/Modified
- `crates/mlrs-kernels/src/topk.rs` — `select_k` `#[cube]` selection-by-rank kernel; `pub use self::select_k as topk_select_k` inside the file. (lib.rs untouched.)
- `crates/mlrs-backend/src/prims/topk.rs` — `top_k` wrapper + `validate_geometry` + `launch_dims_rows`/`launch_dims_1d` helpers. (prims/mod.rs untouched.)
- `crates/mlrs-backend/tests/topk_test.rs` — `check_topk` oracle body + 6 tests (fixture_loads, sklearn f32/f64, tie-break, bad-geometry, the pre-existing i32 round-trip).

## Decisions Made
- **Selection-by-rank, not SharedMemory insertion:** the Task-1 kernel was first written as a `SharedMemory` k-length insertion-select (mirroring the `argmin_shared` shape). It compiled but the **cubecl-cpu MLIR `run pass` lowering panicked** at launch on the mutable-`bool` flags, the `F::INFINITY` associated const, and the descending-shift loop. Since cpu(f64) is the primary correctness gate, Task 2 rewrote `select_k` to use ONLY `F`/`u32` accumulators and `if`-guards: order candidates by the `(value, index)` pair and emit each slot as "the minimum pair strictly greater than the previous slot's winner". Over distinct row indices this is provably identical to a k-fold `argmin_shared`. (See Deviations.)
- **`PrimError::ShapeMismatch` for bad `k`:** `PrimError` has no `InvalidK` variant; following `distance.rs`'s convention all geometry violations (including `k<1`/`k>cols`) surface as `ShapeMismatch` (operand `"k"` / `"dist"`).
- **Squared-in, sqrt-the-k-out:** top-k selects on the squared distance (order-preserving) and sqrts only the returned `rows×k` values — never the whole `rows×cols` matrix (Pitfall 8 / D-08).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Blocking issue] `select_k` kernel rewritten for cubecl-cpu MLIR compatibility**
- **Found during:** Task 2 (running the oracle on cpu)
- **Issue:** The Task-1 `select_k` kernel (SharedMemory k-length insertion-select with mutable `bool` flags, an `F::INFINITY`/`F::new(f32::INFINITY)` seed, and a descending-shift `while q > pos` loop) compiled fine but the **cubecl-cpu backend panicked at launch** with `failed to run pass` (MLIR lowering in `cubecl_cpu::compiler::module::Module::run_pass`). The kernel never executed, so `out_val`/`out_idx` came back all-zero and the oracle failed. The `F::INFINITY` const additionally produced a `From<NativeExpand<F>>` compile error, and cross-loop `let mut <flag> = 0u32;` produced an `E0283` NativeExpand inference failure.
- **Fix:** Rewrote `select_k` as **selection-by-rank** using only `F`/`u32` accumulators + `if`-guards (no `SharedMemory`, no mutable `bool`, no `F::INFINITY`, no descending shift); seeded slot 0 by a direct min-scan and slots 1..k as "minimum pair strictly greater than the previous winner". Pinned the `u32` flag types explicitly (`let mut admit: u32 = 0u32;`) to satisfy the cube macro. The lowest-index tie semantics are preserved exactly (constructed-tie oracle case pins it).
- **Files modified:** `crates/mlrs-kernels/src/topk.rs`
- **Commit:** `5d0b958` (the kernel change rides with Task 2 since it was discovered while wiring the oracle)

This is within scope (directly caused by this plan's own kernel) and no architectural change — the kernel's public signature, output layout, and tie contract are unchanged; only the internal algorithm shape changed to land on the cpu gate.

## Known Stubs

None. Both stub files were fully implemented and the oracle test exercises real device output (no hardcoded/empty values flow to the assertions).

## Issues Encountered
- The cubecl-cpu MLIR lowering is stricter than the cube-macro typecheck: a kernel can compile yet panic at launch (`failed to run pass`). Avoid `SharedMemory` + mutable `bool` + float-infinity consts + descending-index shift loops in cpu-gated kernels; prefer `F`/`u32` accumulators with explicitly-typed `u32` flags. Captured in the new patterns-established entry for downstream Wave-2 kernel plans.

## Next Phase Readiness
- **Plans 05-07..09 (neighbors estimators) unblocked:** `top_k` returns `(distances: DeviceArray<F>, indices: DeviceArray<u32>)` device-resident; the KNN consumers re-upload `u32` → `i32` (D-06, already confirmed in 05-01). The standalone oracle is green within 1e-5 + index-exact, satisfying the D-01 primitive-first gate.
- No blockers. cpu(f64) full + rocm(f32) test-target build both green; lib.rs/prims/mod.rs untouched so the sibling Wave-2 prim plans stay file-disjoint.

## Threat Flags

None — no new network/auth/file surface; the only trust boundary is the validated `top_k(rows, cols, k)` geometry, mitigated exactly as the threat register specified (validate-before-launch → `PrimError::ShapeMismatch`).

## Self-Check: PASSED

- All modified files verified present (topk.rs kernel, prims/topk.rs wrapper, topk_test.rs, this SUMMARY).
- Both task commits verified in git history (`16e2242`, `5d0b958`).
- `cargo test --features cpu -p mlrs-backend --test topk_test` 6/6 green (incl. f64, tie, bad-geometry); `cargo build -p mlrs-kernels` + `-p mlrs-backend --features rocm --tests` green; lib.rs/prims/mod.rs untouched.
