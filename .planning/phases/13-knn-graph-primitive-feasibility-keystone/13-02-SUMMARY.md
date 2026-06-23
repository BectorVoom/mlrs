---
phase: 13-knn-graph-primitive-feasibility-keystone
plan: 02
subsystem: infra
tags: [knn, cubecl, cpu-mlir, distance-kernels, minkowski, self-drop, prim-11]

# Dependency graph
requires:
  - phase: 13-01
    provides: "distance.rs registered cpu-MLIR authoring-contract scaffold; per-metric oracle fixtures; knn_graph_test.rs RED harness"
  - phase: spike-13
    provides: "spike-findings-mlrs skill — VALIDATED verbatim kernel shapes (001 direct distance, 002 self-drop) + cpu-MLIR landmines"
provides:
  - "manhattan_dist / chebyshev_dist / minkowski_dist direct pairwise feature-loop distance kernels (cpu-MLIR-safe, generic over F)"
  - "self_drop_gather per-row index-identity GATHER kernel (CUBE_POS_X shape, self-contained nested-count shift)"
  - "pub use distance::{chebyshev_dist, manhattan_dist, minkowski_dist, self_drop_gather} re-export from mlrs-kernels"
  - "self_drop_gather launch smoke test (cpu f32+f64 green, rocm f32 green / f64 skip-with-log)"
affects: [13-03-prim, umap-phase-14, hdbscan-phase-15]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Direct pairwise 2D feature-loop distance kernel (ABSOLUTE_POS_{X,Y}, while kk<cols, F/u32 accumulators, STATIC F::powf)"
    - "Per-row index-identity self-drop GATHER (CUBE_POS_X/UNIT_POS_X==0, per-slot nested-count shift, no cross-sibling accumulator)"
    - "Backend-feature-free kernel + launch proof in mlrs-backend/tests (kernels crate cannot select a runtime; mirrors spike_test.rs)"

key-files:
  created:
    - crates/mlrs-backend/tests/self_drop_gather_test.rs
  modified:
    - crates/mlrs-kernels/src/distance.rs
    - crates/mlrs-kernels/src/lib.rs

key-decisions:
  - "Launch smoke test lives in mlrs-backend/tests (not mlrs-kernels): the kernels crate is backend-feature-free and cannot select ActiveRuntime — same constraint spike_test.rs documents"
  - "Build verify runs `cargo build -p mlrs-kernels` bare, not `--features cpu` (crate has no cpu feature; inherited Rule-3 correction from 13-01 SUMMARY dev #2)"
  - "Test value comparisons use byte-cast host_to_f64/from_f64 helpers (topk_test convention) — F::abs() etc. are #[cube] fns that panic on the host"

patterns-established:
  - "Verbatim spike-shape transcription: copy the VALIDATED 001/002 kernel bodies exactly, only adding cpu-MLIR-contract doc comments"
  - "Launch smoke test proves non-zero correct read-back (catches the 002-A loud zero-readback) + index-identity self-drop + adversarial dup-point"

requirements-completed: [PRIM-11]

# Metrics
duration: 12min
completed: 2026-06-23
---

# Phase 13 Plan 02: KNN-Graph Device Kernels Summary

**The four genuinely-new PRIM-11 device kernels — direct pairwise Manhattan/Chebyshev/Minkowski-p (with in-kernel STATIC `F::powf`) plus the per-row index-identity `self_drop_gather` — transcribed verbatim from the VALIDATED Phase-13 spikes, re-exported from `mlrs-kernels`, and PROVEN to launch (not just compile) under cpu-MLIR via a non-zero-readback smoke test green on cpu(f32+f64) and rocm(f32).**

## Performance

- **Duration:** 12 min
- **Started:** 2026-06-23T13:08:06+09:00
- **Completed:** 2026-06-23T13:19:55+09:00
- **Tasks:** 2
- **Files modified/created:** 3 (2 source modified, 1 test created)

## Accomplishments

- Landed the three direct pairwise feature-loop distance kernels (`manhattan_dist`, `chebyshev_dist`, `minkowski_dist`) in `crates/mlrs-kernels/src/distance.rs`, transcribing the VALIDATED spike-001 verbatim shapes: 2D per-element launch (`ABSOLUTE_POS_{X,Y}`), bounded `while kk < cols` feature loop, `F`/`u32` accumulators only. Manhattan `acc += |diff|`; Chebyshev STATEMENT-form running max (`if diff > acc { acc = diff; }`) seeded at 0; Minkowski uses STATIC `F::powf` for BOTH the per-term power and the final `1/p` root (the named cpu-MLIR feasibility unknown, now landed). No SharedMemory/Atomic/F::INFINITY/mutable-bool; the only instance form is `.abs()` (jacobi-proven).
- Landed `self_drop_gather` (verbatim spike-002): per-row `CUBE_POS_X` / `UNIT_POS_X == 0u32` launch shape (avoids the 002-A loud MLIR pass failure), with the per-output-slot shift recomputed via a self-contained nested count inside the consuming `while` (avoids the 002-B silent miscompile from a cross-sibling-loop accumulator). Drops self by INDEX IDENTITY (D-02/R-3) with the documented last-column fallback when self is absent.
- Re-exported all four kernels from `mlrs-kernels/src/lib.rs` and added a launch smoke test in `mlrs-backend/tests/self_drop_gather_test.rs` that proves the kernel ACTUALLY LAUNCHED (non-zero correct read-back — the 002-A failure reads back all zeros), drops self by index identity, and preserves a genuine distance-0 duplicate neighbour (the adversarial R-9 gate). Green on cpu f32+f64 AND rocm f32 (f64-on-rocm correctly skips-with-log).

## Task Commits

Each task was committed atomically:

1. **Task 1: three direct pairwise distance kernels (manhattan/chebyshev/minkowski)** - `f0eccf3` (feat)
2. **Task 2: self_drop_gather kernel + lib.rs re-export + launch smoke test** - `4c33765` (feat)

_Note: these tasks are `tdd="true"`; the kernels' value-correctness is gated end-to-end by the Plan 01 per-metric oracle (turns GREEN in Plan 03), and Task 2's launch smoke test is the in-plan executable proof that the new self-drop kernel launches and computes correctly under cpu-MLIR._

## Files Created/Modified

- `crates/mlrs-kernels/src/distance.rs` - Added the four `#[cube(launch)]` kernels (manhattan/chebyshev/minkowski direct pairwise + self_drop_gather GATHER) with per-kernel cpu-MLIR-contract doc comments; bodies are verbatim from spikes 001/002.
- `crates/mlrs-kernels/src/lib.rs` - Added `pub use distance::{chebyshev_dist, manhattan_dist, minkowski_dist, self_drop_gather};`.
- `crates/mlrs-backend/tests/self_drop_gather_test.rs` (NEW) - Launch smoke test (f32+f64) proving non-zero read-back, index-identity self-drop, and the adversarial dup-point neighbour preservation; uses `capability::skip_f64_with_log()` for the f64-on-rocm skip.

## Decisions Made

- **Smoke test placement:** the launch proof lives in `mlrs-backend/tests/`, not `mlrs-kernels/tests/`. The `mlrs-kernels` crate is intentionally backend-feature-free (no `cpu`/`rocm` feature; `cubecl` with no runtime), so it cannot resolve `ActiveRuntime` or launch a kernel. This mirrors `spike_test.rs`, which OWNS the concrete-runtime launch proof for the same reason.
- **Build verify command:** ran `cargo build -p mlrs-kernels` bare instead of the plan's `cargo build -p mlrs-kernels --features cpu`. The crate has no `cpu` feature (it would error). This is the same Rule-3 correction recorded in 13-01 SUMMARY deviation #2; the real cpu-MLIR lowering proof is `cargo test -p mlrs-backend --features cpu`, which builds the kernels through `mlrs-backend`'s concrete runtime and runs the launch test.
- **Host-side value comparison:** the test compares device read-back via byte-cast `host_to_f64`/`from_f64` helpers (the `topk_test.rs` convention) rather than `F::abs()`/arithmetic, because CubeCL `Float` ops are `#[cube]` functions that panic ("Unexpanded Cube functions should not be called") when invoked on the host.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Build-verify command corrected (`mlrs-kernels` has no `cpu` feature)**
- **Found during:** Task 1 (build verification)
- **Issue:** The plan's `<verify>`/`<acceptance_criteria>` ran `cargo build -p mlrs-kernels --features cpu`, but `mlrs-kernels` is backend-feature-free — that command errors with "the package 'mlrs-kernels' does not contain this feature: cpu". (Identical to 13-01 SUMMARY dev #2.)
- **Fix:** Built `mlrs-kernels` bare (it is generic-over-runtime; bare build proves the `#[cube]` macro expands and the re-export resolves), and ran the cpu-MLIR lowering + launch proof via `cargo test -p mlrs-backend --features cpu` (which compiles the kernels through the concrete runtime). All gates pass.
- **Files modified:** none (verification-command adjustment only)
- **Verification:** `cargo build -p mlrs-kernels` exits 0; `cargo test -p mlrs-backend --features cpu --test self_drop_gather_test` is green (f32+f64).
- **Committed in:** N/A (no code change)

**2. [Rule 3 - Blocking] Launch smoke test relocated to `mlrs-backend/tests` (the only crate with a concrete runtime)**
- **Found during:** Task 2 (writing the launch smoke test)
- **Issue:** The plan's action allowed the smoke test "in the kernels crate's existing test convention". The kernels crate cannot launch a kernel (no runtime feature → no `ActiveRuntime`), so a launch test there is impossible to compile.
- **Fix:** Placed the launch smoke test in `crates/mlrs-backend/tests/self_drop_gather_test.rs`, where `ActiveRuntime` and `--features cpu`/`--features rocm` exist (mirroring `spike_test.rs`). Still satisfies the AGENTS.md tests-separated rule (dedicated test file in `tests/`).
- **Files modified:** `crates/mlrs-backend/tests/self_drop_gather_test.rs` (new)
- **Verification:** Test compiles and runs green under cpu (f32+f64) and rocm (f32; f64 skips-with-log).
- **Committed in:** `4c33765` (Task 2 commit)

**3. [Rule 1 - Bug] Host value comparison used cube-only `F` ops → runtime panic**
- **Found during:** Task 2 (first smoke-test run)
- **Issue:** The initial assertion computed `(got_val[slot] - want).abs()` and seeded values with `F::from_int`/`F::new` on the host. CubeCL `Float` arithmetic/`.abs()` are `#[cube]` functions and panic at runtime ("Unexpanded Cube functions should not be called") when called outside a kernel — both f32 and f64 tests failed.
- **Fix:** Switched all host comparisons/seed construction to the byte-cast `host_to_f64`/`from_f64` helpers (the `topk_test.rs` convention); all comparisons now happen in plain `f64`.
- **Files modified:** `crates/mlrs-backend/tests/self_drop_gather_test.rs`
- **Verification:** Both tests now pass under cpu; same code green under rocm f32.
- **Committed in:** `4c33765` (Task 2 commit; fixed before the task was committed)

---

**Total deviations:** 3 (2 Rule-3 blocking — build-command + test-location, both forced by the backend-feature-free kernels crate; 1 Rule-1 test bug). 2 of the 3 are inherited from / consistent with the 13-01 architecture (kernels crate cannot launch).
**Impact on plan:** No scope change. The kernel bodies are verbatim from the VALIDATED spikes exactly as planned; the deviations are entirely about WHERE the launch proof can physically live and HOW the host test reads device values — both dictated by the established crate layout, not by any change to the kernels.

## Issues Encountered

- The `<verify>` block's `grep -nE '\.powf\(...'` acceptance gate excludes doc-comment lines; the doc comments deliberately avoid bare instance-`powf` tokens (per the plan's "keep doc prose free of bare instance-powf tokens that would self-invalidate grep gates" instruction), so the code-line scan returns clean.

## Threat Surface

Per the plan's threat register: T-13-03 (self_drop correctness) is mitigated by transcribing the VALIDATED 002 shape exactly (self-contained nested count, no cross-sibling accumulator) AND the Task-2 launch smoke test catching the 002-A loud zero-readback; the load-bearing VALUE oracle is the Plan 01/03 duplicate-point gate. T-13-04 (OOB device read) is mitigated by the in-kernel bounds checks (`i<rows_x && j<rows_y` / feature loop `kk<cols` / `row<rows`). T-13-05 (instance-vs-static powf numeric integrity) is mitigated by STATIC `F::powf` only, enforced by the acceptance grep (no instance `.powf(` in code lines) and the per-metric oracle in Plan 03. No new security-relevant surface introduced.

## Threat Flags

None — no new network endpoint, auth path, file access, or schema change. Pure device-kernel compute.

## Next Phase Readiness

- All four kernels exist, are generic over `F`, contain no cpu-MLIR-forbidden constructs, and the self-drop kernel is launch-proven under cpu (f32+f64) and rocm (f32). Plan 13-03 can now land `Metric` + `knn_graph` in `prims/knn_graph.rs`, composing `distance`/these new kernels → `top_k` → `self_drop_gather`, turning the RED `knn_graph_test.rs` harness GREEN.
- No blockers.

## Self-Check: PASSED

All claimed files present on disk; both task commits (`f0eccf3`, `4c33765`) exist in git history; the self_drop_gather launch smoke test is green under `--features cpu` (f32+f64) and `--features rocm` (f32 green / f64 skip-with-log).

---
*Phase: 13-knn-graph-primitive-feasibility-keystone*
*Completed: 2026-06-23*
