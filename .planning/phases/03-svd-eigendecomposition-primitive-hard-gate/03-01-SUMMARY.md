---
phase: 03-svd-eigendecomposition-primitive-hard-gate
plan: 01
subsystem: infra
tags: [rocm, hip, cubecl, gfx1100, capability-gate, backend-selection, f64]

# Dependency graph
requires:
  - phase: 02-core-compute-primitives
    provides: runtime.rs ActiveRuntime selection, capability.rs f64 gate, spike_test saxpy gate
provides:
  - A compiling AND running rocm feature (cubecl-hip on gfx1100) — first real ROCm GPU execution in the project
  - spike_saxpy_runs_on_active_backend passes under --features rocm (real HIP kernel on gfx1100)
  - Empirically confirmed capability split: rocm f32=true/f64=false, cpu f64=true
  - ROADMAP/PROJECT/03-CONTEXT reconciled to the cpu+rocm gate (f64-on-cpu / f32-on-rocm)
affects: [03-02, 03-03, 03-04, 03-05, all Phase 3+ GPU validation]

# Tech tracking
tech-stack:
  added: [cubecl-hip (transitive via cubecl/rocm; now actually linked+run)]
  patterns:
    - "rocm = [cubecl/rocm, cubecl/std, cubecl/default] — cubecl/std propagates multi_threading cfg into cubecl-hip"
    - "cfg(feature rocm) re-exports cubecl::hip::{AmdDevice as ActiveDevice, HipRuntime as ActiveRuntime} (no cubecl::rocm module exists in 0.10)"
    - "f64-on-rocm SKIPS-with-log via the unchanged skip_f64_with_log gate (cubecl-cpp 0.10 does not register F64 for HIP)"

key-files:
  created: []
  modified:
    - crates/mlrs-backend/src/runtime.rs
    - crates/mlrs-backend/Cargo.toml
    - .planning/ROADMAP.md
    - .planning/PROJECT.md
    - .planning/phases/03-svd-eigendecomposition-primitive-hard-gate/03-CONTEXT.md

key-decisions:
  - "Gate = cpu(f64) + rocm(f32) from Phase 3; f64-on-rocm skips-with-log (cubecl-cpp 0.10 F64 unregistered), NOT a defect"
  - "rocm feature adds cubecl/std + cubecl/default only; default-features=false on cubecl/cubek pins left intact (RESEARCH Security V12)"

patterns-established:
  - "ROCm/HIP bring-up: cubecl::hip path + cubecl/std feature is the verified two-fix recipe"
  - "f64/f32 backend split is explicit in gate docs; capability.rs/skip_f64_with_log is the reused portable mechanism"

requirements-completed: [PRIM-05]

# Metrics
duration: 10min
completed: 2026-06-12
---

# Phase 3 Plan 01: ROCm/HIP Bring-up + cpu+rocm Gate Reconciliation Summary

**First real ROCm/HIP GPU execution in the project — a `#[cube]` saxpy kernel compiles to HIP and runs on gfx1100 via two verified fixes (cubecl::hip re-export + cubecl/std feature), with the project gate reconciled to cpu(f64)+rocm(f32) after empirically confirming f64 is unsupported on the HIP backend.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-06-12
- **Completed:** 2026-06-12
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- `--features rocm` now compiles cubecl-hip (no MultiStream/ResolvedStreams/EventStreamBackend E0432) and `spike_saxpy_runs_on_active_backend` runs a real HIP saxpy kernel on gfx1100 with correct read-back — Phase 3 GPU gate prerequisite is GREEN.
- Empirically confirmed the capability split on hardware: `capability backend=rocm f32_supported=true f64_supported=false` and `capability backend=cpu f32_supported=true f64_supported=true`. capability_test passes on both backends (rocm logs the f64 skip-with-log; cpu runs f64).
- Reconciled ROADMAP (Overview + Phase 3 success criteria 1 & 3), PROJECT (constraints + Key Decisions), and 03-CONTEXT (D-07) to the cpu+rocm gate with the explicit f64-on-cpu / f32-on-rocm split. No stale "cpu+wgpu" / "f64-runs-on-rocm" wording remains in Phase 3 success criteria 1 & 3.
- cpu build re-verified green — no cross-feature regression from the rocm feature change.

## Task Commits

Each task was committed atomically:

1. **Task 1: Apply the two verified ROCm bring-up fixes + prove saxpy gate on gfx1100** - `62fd51c` (fix)
2. **Task 2: Confirm rocm capability split + reconcile ROADMAP/PROJECT/CONTEXT wording** - `0b2ccd7` (docs)

## Files Created/Modified
- `crates/mlrs-backend/src/runtime.rs` - cfg(feature rocm) re-export now `cubecl::hip::{AmdDevice as ActiveDevice, HipRuntime as ActiveRuntime}` (was the non-existent `cubecl::rocm::{RocmDevice, RocmRuntime}`).
- `crates/mlrs-backend/Cargo.toml` - `rocm = ["cubecl/rocm", "cubecl/std", "cubecl/default"]` so the `multi_threading` cfg reaches cubecl-hip; default-features=false on the workspace cubecl/cubek pins untouched.
- `.planning/ROADMAP.md` - Overview sentence + Phase 3 success criteria 1 & 3 now state the cpu+rocm gate with the f64-on-cpu / f32-on-rocm split.
- `.planning/PROJECT.md` - Test/CI-target constraint + Key Decisions row reconciled to cpu+rocm from Phase 3.
- `.planning/phases/03-.../03-CONTEXT.md` - D-07 clause appended with the RESEARCH correction that f64-on-rocm is empirically unsupported at the CubeCL layer and validates on cpu.

## Capability Probe Results (recorded per Task 2 action)

| Backend | active_backend_name() | supports_type(F32) | supports_type(F64) | f64 path |
|---------|-----------------------|--------------------|--------------------|----------|
| rocm    | rocm                  | true               | **false**          | skip-with-log (EXPECTED — cubecl-cpp 0.10 F64 unregistered) |
| cpu     | cpu                   | true               | **true**           | runs |

## Decisions Made
- **Gate = cpu(f64) + rocm(f32) from Phase 3.** f64-on-rocm SKIPS-with-log via the existing `skip_f64_with_log` gate because cubecl-cpp 0.10 does not register F64 for the HIP backend. This is the safe behavior the capability gate already implements — documented as expected, not a defect (RESEARCH 03 CRITICAL FINDING 2 / Pitfall 1).
- **Minimal feature change.** Only the `rocm` feature line gained `cubecl/std` + `cubecl/default`; `default-features = false` on the workspace cubecl / cubek-matmul / cubek-std pins was left intact (RESEARCH Security V12 / threat T-03-01-01).

## Deviations from Plan

None - plan executed exactly as written. (No package installs; cubecl-hip/cubecl-hip-sys are transitive via the existing cubecl pin and were already built. The rocm build finished from cache, then the test recompiled and ran the real kernel on gfx1100.)

## Issues Encountered
None. The first `cargo build --features rocm` finished from cache (cubecl-hip was already compiled during research); the subsequent `cargo test` recompiled mlrs-backend and ran the saxpy kernel on gfx1100 (`test result: ok. 1 passed`).

## User Setup Required
None - no external service configuration required. `/opt/rocm/bin` is already on PATH and the gfx1100 device (`/dev/kfd` + `/dev/dri/renderD128`) is present.

## Next Phase Readiness
- The Phase 3 GPU gate prerequisite is satisfied: a real HIP kernel runs on gfx1100, so plans 03-02 through 03-05 (oracle scaffold + Jacobi SVD/eig kernels + tests) are unblocked on the rocm gate backend.
- f64 SVD/eig validation must target the **cpu** backend; rocm validates **f32** and f64-on-rocm skips-with-log — every downstream SVD/eig oracle test should mirror the existing gemm_test.rs skip pattern.
- No blockers.

## Self-Check: PASSED

All modified files exist on disk; both task commits (`62fd51c`, `0b2ccd7`) are present in git history.

---
*Phase: 03-svd-eigendecomposition-primitive-hard-gate*
*Completed: 2026-06-12*
