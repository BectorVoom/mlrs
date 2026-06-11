---
phase: 01-foundation-oracle-backend-abstraction-arrow-bridge
plan: 04
subsystem: infra
tags: [cubecl, buffer-pool, free-list, device-array, memory-efficiency, bytemuck]

# Dependency graph
requires:
  - phase: 01-01
    provides: "runtime::{Client, ActiveRuntime, active_client}; SPIKE-FINDINGS (Bytes/empty/read_one/Handle resolved); pool.rs + device_array.rs stubs"
provides:
  - "mlrs_backend::pool::{BufferPool<R>, PoolStats} — byte-size-keyed free-list with logged-only allocations/reuses/peak_bytes/live_bytes counters (D-04/D-05)"
  - "mlrs_backend::device_array::DeviceArray<R,F> — length+dtype wrapper over a CubeCL Handle, pool-metered allocation, host read-back round-trip"
affects: [01-05, phase-2-primitives, memory-efficiency-verification]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "mlrs-level HashMap<usize, Vec<Handle>> free-list on top of client.empty (NOT CubeCL MemoryConfiguration tuning — RESEARCH Open Question 4)"
    - "Logged-only counters (log::info! at log_stats()/Drop) — no hard reuse-rate phase gate in Phase 1 (D-05)"
    - "Length-carrying typed device buffer wrapper: read-back size derived from carried len, never caller geometry (T-04-01 mitigation)"

key-files:
  created: []
  modified:
    - crates/mlrs-backend/src/pool.rs
    - crates/mlrs-backend/src/device_array.rs
    - crates/mlrs-backend/tests/pool_test.rs

key-decisions:
  - "Pool counters are LOGGED ONLY in Phase 1 (D-05); tests assert the counters API increments correctly, not a reuse-rate threshold"
  - "DeviceArray::from_host meters the byte footprint through BufferPool::acquire then releases the metering handle, because cubecl 0.10 has no in-place host-write into an empty handle; the populated client.create handle is held"
  - "Reuse is keyed by EXACT byte size — distinct sizes never alias (verified)"
  - "to_host slices to exactly len elements to guard against any runtime trailing padding"

patterns-established:
  - "Pattern: mlrs-level buffer-reuse free-list over reclaimed CubeCL handles with logged counters"
  - "Pattern: typed DeviceArray<R,F> wrapper carrying len+dtype for safe read-back size derivation"

requirements-completed: [FOUND-05]

# Metrics
duration: 5min
completed: 2026-06-11
---

# Phase 01 Plan 04: Memory-Efficiency Layer (Buffer Pool + DeviceArray) Summary

**Byte-size-keyed buffer-reuse pool with logged-only allocations/reuses/peak/live counters, plus `DeviceArray<R,F>` wrapping CubeCL buffers with pool-metered allocation and a lossless host↔device round-trip on cpu and wgpu.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-06-11T12:07:02Z
- **Completed:** 2026-06-11T12:11:24Z
- **Tasks:** 2 completed (TDD)
- **Files modified:** 3

## Accomplishments

- `BufferPool<R>` implements a `HashMap<usize, Vec<Handle>>` free-list keyed by byte size: `acquire` reuses a released handle of matching size (`reuses += 1`) or allocates via `client.empty` (`allocations += 1`); `release` returns the handle for later reuse. `live_bytes`/`peak_bytes` track usage with `peak` as a high-water mark.
- `PoolStats { allocations, reuses, peak_bytes, live_bytes }` exposed and surfaced via `log::info!` at `log_stats()` and on `Drop` — logged-only, no hard reuse-rate assertion (D-05).
- `DeviceArray<R,F>` wraps a `cubecl::server::Handle` + `len` + dtype marker; `from_host` routes allocation accounting through the pool and uploads with a single host copy (A3 honest single-upload); `to_host` reads back via `client.read_one` + `bytemuck::cast_slice` into a `Vec<F>` of exactly `len` elements.
- Round-trip proven on both cpu and wgpu (5/5 tests each); full crate suite green (cpu).

## Task Commits

Each task was committed atomically (TDD: test → feat):

1. **RED (both tasks): failing pool/device_array tests** - `abf5ab5` (test)
2. **Task 1: BufferPool free-list + logged-only PoolStats** - `4993e59` (feat)
3. **Task 2: DeviceArray<R,F> pool-routed alloc + read-back** - `49e78d7` (feat, includes clippy `size_of_val` cleanup)

**Plan metadata:** (this SUMMARY + STATE/ROADMAP) committed separately.

_Note: pool.rs and device_array.rs share one RED commit because the single integration test file (`pool_test.rs`) exercises both; each implementation then landed as its own GREEN feat commit._

## Files Created/Modified

- `crates/mlrs-backend/src/pool.rs` - `BufferPool<R>` byte-size free-list + `PoolStats` (logged-only counters); `acquire`/`release`/`stats`/`log_stats`/`client`; `Drop` logs stats.
- `crates/mlrs-backend/src/device_array.rs` - `DeviceArray<R,F>` wrapper; `from_host` (pool-metered upload), `to_host` (read-back), `len`/`is_empty`/`handle`.
- `crates/mlrs-backend/tests/pool_test.rs` - 5 integration tests: reuse increments reuses-not-allocations, distinct sizes don't alias, counters/logging API, DeviceArray round-trip through pool, empty array.

## Verification

- `cargo test -p mlrs-backend --features cpu --test pool_test` → 5 passed.
- `cargo test -p mlrs-backend --features wgpu --test pool_test` → 5 passed (AMD Radeon RADV GFX1152 adapter).
- `cargo test -p mlrs-backend --features cpu` (full crate) → all green.
- `cargo clippy -p mlrs-backend --features cpu --tests` → no warnings in pool.rs / device_array.rs (one pre-existing `runtime.rs:40` `default_constructed_unit_structs` warning is out of scope, untouched).
- `grep -rn "mod tests" crates/mlrs-backend/src/` → empty (AGENTS.md §2 separation honored).

## Decisions Made

- **Logged-only counters (D-05):** Phase-1 tests assert the counters API increments correctly (`reuses == 1` after acquire→release→acquire of the same size, `allocations` unchanged), NOT a reuse-rate threshold. Hard memory/reuse assertions are deferred to Phase 2 where realistic primitive workloads exercise allocation.
- **Allocation metering vs. upload:** cubecl 0.10 has no public in-place write into an `empty(size)` handle. `from_host` therefore meters the byte footprint through `BufferPool::acquire` (counters + live/peak) and returns that handle to the free-list, then uploads the actual data via `client.create(Bytes::from_bytes_vec(...))`. This keeps every array's footprint visible in `PoolStats` while still performing exactly one upload copy.
- **No CubeCL `MemoryConfiguration` tuning** (RESEARCH Open Question 4): the simplest correct mlrs-level free-list is used; CubeCL allocator tuning is deferred unless profiling demands it.

## Deviations from Plan

None — plan executed as written. Tasks 1 and 2 share the single `pool_test.rs` integration target (the plan explicitly allows the combined test target), so the TDD RED commit covers both before the two GREEN feat commits.

A minor clippy cleanup (`len * size_of::<F>()` → `size_of_val(host)`, drop unused `size_of` import) was folded into the Task 2 commit before it was finalized — [Rule 1 - lint] correctness-neutral.

## Known Stubs

None. `pool.rs` and `device_array.rs` previously contained Wave-0 TODO stubs; both are now fully implemented. No empty/placeholder values flow to any consumer.

## Threat Surface

Threat register dispositions honored:
- **T-04-01 (read-back OOB):** mitigated — `DeviceArray` carries `len`; `to_host` derives the result length from `len` and slices to exactly `len` elements. No new `unsafe` introduced (read-back is fully safe via `bytemuck::cast_slice`).
- **T-04-02 (unbounded free-list):** accepted as planned — Phase-1 workloads are trivial; logged counters surface growth; bounded-pool policy is a Phase-2 concern.
- **T-04-SC (no new package installs):** honored — no dependencies added.

No new security-relevant surface beyond the planned threat model.

## Self-Check: PASSED

- Files: pool.rs, device_array.rs, pool_test.rs, 01-04-SUMMARY.md all present.
- Commits: abf5ab5 (test/RED), 4993e59 (feat/pool), 49e78d7 (feat/device_array) all in git history.
