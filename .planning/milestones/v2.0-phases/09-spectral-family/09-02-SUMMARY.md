---
phase: 09-spectral-family
plan: 02
subsystem: backend-prims
tags: [spectral, laplacian, prim, cubecl, degree-normalization, memory-gate]

# Dependency graph
requires:
  - phase: 09-spectral-family
    plan: 01
    provides: laplacian(pool, A, n) -> (L, dd) signature + geometry guard, laplacian_map kernel stub, 3 ignored laplacian tests
  - phase: 02-foundations
    provides: row_reduce(Sum) GATHER reduction, BufferPool/PoolStats, DeviceArray
  - phase: 08-kernel-family
    provides: base-op -> in-place-map idiom (kernel_matrix.rs), PoolStats one-gate-per-prim precedent, f32 band precedent
provides:
  - laplacian(pool, A, n) -> (L, dd) FILLED compute (PRIM-09 standalone-validated)
  - laplacian_map device kernel (off-diag -a/(dd_i*dd_j), diagonal 1-isolated)
  - degree_guard device kernel (dd = where(w==0, 1, sqrt(w)) typed-zero guard)
  - zero_diag_copy device kernel (fill_diagonal(m,0) non-in-place copy)
affects: [09-03-spectral-embedding, 09-04-spectral-clustering-pyo3]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "laplacian.rs is a thin host orchestration over row_reduce + 3 SharedMemory-free map kernels (the kernel_matrix.rs base-op->map idiom)"
    - "typed-zero guard (degree_guard: dd=1 for isolated) + diagonal forced to 0 replaces the would-be 1/sqrt(0) infinite value (T-9-LAP)"
    - "STATEMENT-form CubeCL guards (let mut val = default; if cond { val = .. }) per Cubecl_conditionals.md, never if-expressions"

key-files:
  created: []
  modified:
    - crates/mlrs-kernels/src/elementwise.rs
    - crates/mlrs-kernels/src/lib.rs
    - crates/mlrs-backend/src/prims/laplacian.rs
    - crates/mlrs-backend/tests/laplacian_test.rs

key-decisions:
  - "laplacian_map threads the degree vector w alongside dd so the diagonal uses the TRUE isolated test (w[i]==0), not a dd==1 heuristic (dd==1 can legitimately occur for degree==1)"
  - "zero_diag_copy is a NON-in-place copy into a fresh working buffer so the caller's affinity A is never mutated (the prim RECEIVES A and RETURNS a new L)"
  - "degree_guard runs as a device kernel (not a host map) to keep the whole pipeline device-resident; only row_reduce's internal per-row to_host is unmetered scratch"
  - "f32 band = 1e-4/1e-4 (Phase-8 precedent); measured f32 L max_abs 2.98e-8, f64 5.6e-17 — both far inside band"

patterns-established:
  - "Pattern: one device prim, three SharedMemory-free/atomics-free/infinity-free map kernels, validated against scipy _laplacian_dense before any estimator wiring (primitive-first gate)"

requirements-completed: [PRIM-09]

# Metrics
duration: 5min
completed: 2026-06-21
---

# Phase 9 Plan 02: Normalized Graph-Laplacian Primitive (PRIM-09) Summary

**Filled `laplacian(pool, A, n) -> (L, dd)` with a 4-step host orchestration (zero_diag_copy -> row_reduce(Sum) degree GATHER -> degree_guard typed-zero dd -> laplacian_map build) over three new SharedMemory-free/atomics-free/infinity-free device kernels, reproducing scipy `_laplacian_dense` to f64 max_abs 5.6e-17 / f32 2.98e-8 with the zero-degree node NaN/inf-free and the PoolStats memory gate green — the primitive-first gate is satisfied.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-06-21T02:57:52Z
- **Completed:** 2026-06-21T03:02:21Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- **laplacian_map kernel** (`elementwise.rs`): off-diagonal `out[i*n+j] = -a/(dd_i*dd_j)` (GATHER divisor by row/col index, no scatter/atomics), diagonal `1 - isolated` where the isolated flag is the TRUE `w[i]==0` test (the degree vector `w` is threaded in alongside the guarded `dd`). STATEMENT-form guards; SharedMemory-free, atomics-free, no infinity constant.
- **degree_guard kernel** (`elementwise.rs`): `dd[i] = if w[i]==0 { 1 } else { sqrt(w[i]) }` — the typed-zero guard (T-9-LAP) that replaces the would-be `1/sqrt(0)` infinite value with the typed `1`. STATEMENT-form.
- **zero_diag_copy kernel** (`elementwise.rs`): non-in-place `np.fill_diagonal(m, 0)` copy so the degree row-sum excludes the self edge (scipy order) without mutating the caller's `A`.
- **laplacian.rs host path**: 4-step orchestration acquiring a working buffer, computing the degree via `row_reduce(Sum, Shared)` (the single-owner GATHER), guarding it, and running the build map into a fresh `L`; the transient working buffer and degree vector are released (`live_bytes` conserves), and `dd` is returned alongside `L` for the downstream D-07 recovery.
- **3 un-ignored tests green** (4 test fns: value f64 + value f32 + zero_degree + memory_gate): f64 L max_abs `5.55e-17` (strict `F64_TOL`), f32 `2.98e-8` (band `1e-4`); the isolated node has `dd==1`, `L` row all-zero, `L` diagonal `0`, all entries finite; `read_backs==0`, `live_bytes` conserved, `peak_bytes` plateaus.

## Task Commits

1. **Task 1: laplacian_map + degree_guard SharedMemory-free kernels** - `c2ec4fb` (feat)
2. **Task 2: laplacian.rs host orchestration + green value/zero_degree/memory_gate** - `545d5b7` (feat)

## Files Created/Modified

- `crates/mlrs-kernels/src/elementwise.rs` - Replaced the Wave-0 `laplacian_map` stub body with the real off-diag/diagonal build (added a `w` degree arg); added `degree_guard` and `zero_diag_copy` kernels.
- `crates/mlrs-kernels/src/lib.rs` - Re-exported `degree_guard` and `zero_diag_copy` (`laplacian_map` was already exported).
- `crates/mlrs-backend/src/prims/laplacian.rs` - Replaced the `todo!()` compute with the 4-step path + a `launch_dims_1d` helper; removed the Wave-0 stub `#[allow(unused_imports)]` shims and stub docs.
- `crates/mlrs-backend/tests/laplacian_test.rs` - Un-ignored all scaffolds; wired the real `laplacian` call for value (f32+f64), zero_degree (no NaN/inf + isolated-node structure), and memory_gate (live/peak/read_backs).

## Decisions Made

- **Thread `w` (degree) into `laplacian_map`** so the diagonal uses the exact `w[i]==0` isolated test rather than inferring isolation from `dd==1` (which is ambiguous when a node legitimately has degree 1). This made the diagonal `1 - isolated` byte-exact against the scipy reference.
- **`zero_diag_copy` is non-in-place** — the prim contract RECEIVES `A` and RETURNS a new `L`, so the caller's affinity buffer is never mutated. (The committed fixtures already have a zeroed `A` diagonal, so the step is idempotent there, but it is required for the rbf-affinity diagonal `exp(0)=1` the estimators will pass in Waves 2/3.)
- **`degree_guard` is a device kernel**, keeping the pipeline device-resident; `row_reduce`'s internal per-row `to_host` is unmetered scratch, so the memory gate's `read_backs==0` holds.
- **Added a third kernel (`zero_diag_copy`) under Task 2's commit** rather than Task 1, since it is part of the host-orchestration step-1 the plan attributes to Task 2 ("an in-place map kernel or a reuse of an existing diag-zero idiom"). No existing diag-zero idiom was found in the codebase.

## Deviations from Plan

None - plan executed exactly as written. Two within-intent refinements:

- **[Refinement] `laplacian_map` gained a `w` (degree) argument** beyond the plan's `(a, dd, output, n)` arity. The plan's `<behavior>` left the isolated-detection form to discretion ("detected via dd[i]==1 ... OR by passing the isolated mask — choose the gather-clean form"); threading `w` is the gather-clean, unambiguous form and is byte-exact against the reference. This changed the Wave-0 stub signature, but the only consumer is `laplacian.rs` (updated in the same plan).
- **[Refinement] Added `zero_diag_copy` as a new kernel** for step 1 (no existing diag-zero idiom existed). Documented above.

## Issues Encountered

- `cargo build --features cpu -p mlrs-kernels` errors (`mlrs-kernels` is backend-feature-free per its crate design); the correct Task-1 build command is `cargo build -p mlrs-kernels` (no feature). The full pipeline is exercised via `cargo test --features cpu -p mlrs-backend --test laplacian_test`. No code impact.

## User Setup Required

None - no external service configuration. Fixtures are committed `.npz` blobs (no regeneration needed; the prim value-matches the already-committed scipy reference).

## Known Stubs

None - the `laplacian` compute is fully implemented; no placeholder values, no `todo!()`, no hardcoded empties. The Wave-0 stub bodies (kernel placeholder + `todo!()` host path) were both replaced.

## Next Phase Readiness

- PRIM-09 is standalone-validated BEFORE any estimator consumes it (the primitive-first gate). `laplacian` returns BOTH `L` and `dd`; `dd` is the length-n vector the D-07 recovery in 09-03 divides each recovered eigenvector by.
- The three new kernels are SharedMemory-free, atomics-free, and infinity-free (grep-clean + manual inspection) — the cpu-MLIR-safe profile inherited by any backend.
- Waves 2/3 (09-03 SpectralEmbedding, 09-04 SpectralClustering) wire `laplacian` after building the rbf / kNN-connectivity affinity; the rbf diagonal `exp(0)=1` is handled by `zero_diag_copy` step 1.

## Self-Check: PASSED

Both task commits (`c2ec4fb`, `545d5b7`) exist in git history; the 4 laplacian tests pass on cpu (f32+f64); the `INFINITY` and `SharedMemory` grep gates are clean on the new sources.

---
*Phase: 09-spectral-family*
*Completed: 2026-06-21*
