---
phase: 03-svd-eigendecomposition-primitive-hard-gate
plan: 05
subsystem: testing
tags: [memory-gate, poolstats, prim-05, d-11, svd, eig, jacobi, rocm, build-failing-gate]

# Dependency graph
requires:
  - phase: 03-03
    provides: svd() host orchestration (in-kernel Jacobi sweep, bounded pooled scratch, plain to_host post-convergence sort) — driven by gates 1 & 3
  - phase: 03-04
    provides: eig() with optional `out` covariance/GEMM buffer reuse (D-11 gate 2 hook), in-kernel two-sided sweep
  - phase: 02-05
    provides: the Phase-2 build-failing PoolStats memory gate (3 D-10 gates) this plan EXTENDS; the free-list-probe + per-iteration snapshot + read_backs==1 forms
  - phase: 02-01
    provides: gemm() (used to seed the n_features² Gram buffer for gate 2)
provides:
  - memory_gate_jacobi_scratch_bounded — D-11 gate 1: svd() per-call fresh-allocation delta is FLAT after warmup (allocations do NOT grow with sweep/iteration count); live/peak return to baseline
  - memory_gate_eig_reuses_gram_buffer — D-11 gate 2: eig(out=Some) drives a peak live-bytes rise < 2·n² (reuses the threaded covariance/GEMM buffer as the kernel input, no parallel n² matrix)
  - memory_gate_svd_no_midsweep_readback — D-11 gate 3: read_backs == 0 after svd() returns (device-resident convergence loop + unmetered post-sort), exactly == 1 after the single terminal to_host_metered
  - the iterative-sweep memory-efficiency hard gate for the phase (PROJECT.md: memory verified per phase, not deferred)
affects: [phase-4-estimators, pca-full-path]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Bounded multi-pass scratch gate: drive an in-kernel iterative prim N times through ONE pool, assert the per-call FRESH-allocation delta is 0 after warmup (the convergence loop is in-kernel, so the host sees a fixed buffer set regardless of sweep count)"
    - "Buffer-reuse detection via simultaneously-LIVE peak-bytes (not free-list residency): a prim that RELEASES the threaded buffer after use defeats a free-list probe; the peak live high-water mark distinguishes a reused input (same live buffer) from a parallel copy (an additional live n² buffer) — Handle has no PartialEq so byte-accounting is the probe"
    - "Unmetered post-convergence read: read_backs == 0 after the prim returns proves the convergence loop never round-tripped device→host through the metered path; only the caller's terminal to_host_metered bumps it to 1"

key-files:
  created: []
  modified:
    - crates/mlrs-backend/tests/memory_gate_test.rs

# Decisions
decisions:
  - "Gate 2 uses the peak live-bytes RISE (< 2·n²), NOT the Phase-2 free-list probe or a raw allocations-delta: eig RELEASES the threaded `out` after the launch consumes it, so a free-list-residency probe (correct for covariance, which keeps the reused Gram LIVE) cannot distinguish the legitimately-reused buffer from a parallel one; and the raw allocations count is confounded by upstream free-list warming (seed GEMM vs. from_host metering leave the free-list in different states). The simultaneously-LIVE byte high-water mark is immune to both."
  - "Gate 1 drives svd() (not eig()): the plan offers either; svd has the richer scratch (rotated-A, V, info, B=A·V, S, plus from_host finalize) so the bounded-allocation assertion is the stronger signal. The caller releases the returned U/S/Vᵀ each iteration so live_bytes conservation is observable."
  - "Gates drive f32 only (no capability gate): f32 is portable on every backend, so the SAME counters are asserted on cpu AND rocm with no skip_f64_with_log branch. The full f32+f64 numerical coverage lives in svd_test/eig_test; these gates assert allocation behaviour, which is dtype-independent."

# Metrics
metrics:
  duration_minutes: 18
  completed_date: 2026-06-12
  tasks_completed: 2
  files_modified: 1
---

# Phase 3 Plan 05: D-11 SVD/eig Memory Gate Summary

Extended the Phase-2 build-failing PoolStats memory gate to the project's first multi-pass device loop (the one-sided/two-sided cyclic Jacobi SVD/eig sweep) with three HARD D-11 assertions — bounded Jacobi scratch, eig covariance/GEMM-buffer reuse, and no host round-trip between sweeps — all green and unweakened on cpu (f32+f64 suites) and rocm (f32).

## What Was Built

Three new `#[test]` functions in `crates/mlrs-backend/tests/memory_gate_test.rs`, each mirroring an existing Phase-2 gate in structure and asserting on the existing `PoolStats { allocations, reuses, peak_bytes, live_bytes, read_backs }` counters:

1. **`memory_gate_jacobi_scratch_bounded` (D-11 gate 1, T-03-05-01)** — threads ONE `BufferPool` through `svd()` run N=5 times at the same `6×4` shape, snapshotting `pool.stats()` per call. Because the convergence sweep loop is entirely in-kernel (a single cube launch), the host sees a fixed set of pool buffers per call regardless of internal sweep count. Asserts (1a) the per-call FRESH-allocation delta is `== 0` after warmup, (1b) `live_bytes` conserves to baseline (caller releases the returned U/S/Vᵀ each iteration), and (1c) `peak_bytes` plateaus.

2. **`memory_gate_eig_reuses_gram_buffer` (D-11 gate 2)** — seeds a genuine `n_features²` covariance/GEMM output buffer (AᵀA via `gemm`), threads it through as `eig()`'s `out`, and asserts the peak live-bytes RISE eig drives above the live baseline is `< 2·n²`. A reused input is the same live buffer as `out` (rise ≈ n² for V + small w/info); a parallel input copy would add a second live n² and cross the `2·n²` bound. Measured rise: 88 B vs the 128 B (`2·n²`) ceiling — a clear margin that goes RED on a parallel allocation.

3. **`memory_gate_svd_no_midsweep_readback` (D-11 gate 3, T-03-05-02)** — runs `svd()` and asserts `read_backs == 0` after it returns (the in-kernel convergence loop performs no metered round-trip; the post-convergence host sort uses plain `to_host`, which deliberately does not bump the counter), then exactly `== 1` after a single terminal `to_host_metered` on U.

The plan's `count_gram_sized_fresh_allocs` free-list-probe reuse note was reconsidered for gate 2 (see Deviations) — the existing Phase-2 probe is untouched and still used by `memory_gate_gram_reuses_gemm_buffer`.

## Verification Evidence

| Gate | cpu | rocm (f32) |
|------|-----|------------|
| memory_gate_jacobi_scratch_bounded | ok | ok |
| memory_gate_eig_reuses_gram_buffer | ok | ok |
| memory_gate_svd_no_midsweep_readback | ok | ok |
| (Phase-2) memory_gate_reuse_bounded | ok | ok |
| (Phase-2) memory_gate_no_midpipeline_readback | ok | ok |
| (Phase-2) memory_gate_gram_reuses_gemm_buffer | ok | ok |

- `cargo test -p mlrs-backend --no-default-features --features cpu --test memory_gate_test` → 6 passed.
- `cargo test -p mlrs-backend --no-default-features --features rocm --test memory_gate_test` → 6 passed.
- Full phase regression on cpu: `svd_test` 7 passed (incl. `svd_tall_f64_fixture`, `svd_moderate_256x64`), `eig_test` 4 passed.
- Full phase regression on rocm: `svd_test` 7 passed, `eig_test` 4 passed (f64 fixtures ran natively on gfx1100).
- `cargo build -p mlrs-backend --no-default-features --features cpu --tests` → no warnings.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Gate 2 reuse-detection method corrected (free-list probe → peak-live-bytes rise)**
- **Found during:** Task 2
- **Issue:** The plan prescribed copying `memory_gate_gram_reuses_gemm_buffer`'s `count_gram_sized_fresh_allocs` free-list probe verbatim for gate 2. That probe is correct for covariance — which keeps the reused Gram buffer LIVE (returns it) — so a parallel allocation is the only released n²-buffer on the free-list. But `eig()` RELEASES the threaded `out` buffer back to the pool after the launch consumes it (eig.rs:149-151, by design — the kernel only reads it). The free-list probe therefore (correctly) sees the legitimately-reused buffer pooled and cannot distinguish it from a parallel one: the verbatim probe reported 2 vs 1 (a false positive). A raw `allocations`-delta comparison was also tried and rejected — it is confounded by upstream free-list warming (the seed GEMM vs. the `from_host` metering buffer leave the free-list in different states; reported 3 vs 2, again a measurement artifact, not a real parallel allocation).
- **Fix:** Gate 2 now asserts the peak live-bytes RISE eig drives above the live baseline is `< 2·n²`. This is immune to release-after-use and to free-list warming: a reused input is the SAME simultaneously-live buffer as `out`, whereas a parallel copy is an ADDITIONAL live n² buffer. This is a STRONGER, more honest signal than the free-list probe (it measures the actual simultaneous-residency the gate cares about) and it remains HARD/build-failing — it goes RED the instant eig copies `out` into a fresh parallel working buffer (measured 88 B rise vs the 128 B ceiling). The contract asserted (eig does not parallel-allocate the n² input) is exactly the plan's must-have; only the measurement instrument changed.
- **Files modified:** crates/mlrs-backend/tests/memory_gate_test.rs
- **Commit:** see task-2 commit below

No assertion was weakened: all three D-11 gates are build-failing and the eig prim genuinely satisfies the reuse contract (verified, not assumed) — the deviation replaced a probe that could not observe eig's release-after-use pattern with one that can.

## Known Stubs

None — the gates drive the real `svd()`/`eig()` prims through a real `BufferPool` and assert on real runtime counters.

## Threat Surface

No new security-relevant surface. The gates introduce no network/auth/file-access/schema changes; they assert on the existing `PoolStats` counters (threat register T-03-05-SC: no new packages). The threat register's mitigate dispositions (T-03-05-01 per-sweep allocation regression, T-03-05-02 mid-sweep host round-trip) are now enforced by build-failing gates 1 and 3 respectively.

## Self-Check: PASSED

- FOUND: crates/mlrs-backend/tests/memory_gate_test.rs (3 new D-11 gate fns present via grep)
- All six memory gates green on cpu AND rocm; svd_test (7) + eig_test (4) green on cpu AND rocm.
