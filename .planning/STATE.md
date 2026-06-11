---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Plan 02-01 complete (GEMM PRIM-01 via cubek-matmul wrap + Wave-0 infra) — Phase 02 plan 1/5
last_updated: "2026-06-12T00:00:00.000Z"
last_activity: 2026-06-12 -- Plan 02-01 executed (GEMM substrate + Wave-0 hooks)
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 10
  completed_plans: 6
  percent: 18
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-11)

**Core value:** Correct, memory-efficient ML algorithms that match scikit-learn within 1e-5, running on any CubeCL backend from a single generic codebase.
**Current focus:** Phase 02 — core-compute-primitives

## Current Position

Phase: 02 (core-compute-primitives) — EXECUTING
Plan: 2 of 5
Status: Executing Phase 02 (plan 01 complete)
Last activity: 2026-06-12 -- Plan 02-01 executed (GEMM substrate + Wave-0 hooks)
Resume file: .planning/phases/02-core-compute-primitives/02-02-PLAN.md

Progress: [██░░░░░░░░] 18% (1/6 phases; 6/10 plans)

## Performance Metrics

**Velocity:**

- Total plans completed: 5
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 5 | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
| Phase 01 P02 | 25 | 2 tasks | 12 files |
| Phase 01 P03 | 7 | 2 tasks | 4 files |
| Phase 01 P04 | 5 | 2 tasks | 3 files |
| Phase 01 P05 | 18 | 3 tasks | 6 files |
| Phase 02 P01 | 35 | 5 tasks | 14 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Primitive-first horizontal-layer build order — primitives validated standalone before estimators (SVD/eig gates 4 estimators; distance gates 3).
- [Roadmap]: SVD/eig gets a dedicated phase (Phase 3) as the single hardest, highest-leverage primitive.
- [Roadmap]: Closed-form estimators (Phase 4) precede iterative (Phase 5) to de-risk the Arrow/oracle pipeline before convergence-sensitive solvers.
- [PROJECT.md]: scikit-learn (not cuML) is the oracle; sklearn-matching defaults (OLS=svd, KMeans=k-means++, TSVD=arpack, PCA with svd_flip) — not cuML defaults.
- [Phase ?]: [01-02]: NEAR_ZERO_FLOOR=1e-8 chosen below the 1e-5 abs tolerance so the near-zero guard never loosens the absolute check
- [Phase ?]: [01-02]: BridgeError lives in mlrs-core so Plan 03 bridge consumes it without a reverse dependency
- [Phase ?]: [01-03]: Sliced-array rejection detects the slice at the BUFFER level (ScalarBuffer ptr_offset + inner length) because arrow 59 PrimitiveArray::offset() always returns 0
- [Phase ?]: [01-03]: Bridge ingress has NO unsafe block (bytemuck::try_cast_slice is the only, safe, reinterpretation) — stronger than "// SAFETY: on every unsafe"
- [Phase ?]: [01-03]: upload() is honest "validated single-upload" (one host copy, A3), not literal zero-copy
- [Phase ?]: [01-03]: f64 skip mechanism = logged early-return (skip_f64_with_log); on this env's wgpu adapter SHADER_F64 is present so f64 runs
- [Phase ?]: [01-04]: Pool counters are LOGGED ONLY in Phase 1 (D-05) — tests assert the counters API increments, not a reuse-rate threshold (hard memory assertions deferred to Phase 2)
- [Phase ?]: [01-04]: mlrs-level HashMap free-list keyed by exact byte size, on top of client.empty — no CubeCL MemoryConfiguration tuning in Phase 1 (RESEARCH Open Question 4)
- [Phase ?]: [01-04]: DeviceArray::from_host meters the byte footprint through the pool then uploads via client.create (cubecl 0.10 has no in-place write into an empty handle)
- [Phase ?]: [01-05]: f32 oracle near-zero floor raised to 1e-2 (in pipeline_test only) — cross-backend f32 saxpy rounding (~1 ULP, abs_err ~3e-8) exceeds the strict 1e-5 *relative* bound on near-cancellation results; the 1e-5 *absolute* bound stays enforced. Core compare.rs (Plan 02) left untouched.
- [Phase ?]: [01-05]: mimalloc #[global_allocator] defined exactly once in the mlrs-py cdylib (src/allocator.rs), never in a library crate; activation proven by exercising it (no public introspection symbol in the mimalloc crate)
- [Phase ?]: [01-05]: f64 oracle cases stay capability-gated via skip_f64_with_log (skip-with-log, never fail) for backend portability — pattern for all future f64 oracle tests
- [Phase 02]: [02-01]: GEMM substrate = WRAP cubek-matmul 0.2.0 (Task-1 checkpoint). cubecl-matmul 0.9-pre / cubecl-linalg 0.5 are abandoned on incompatible cubecl lines; cubek-matmul 0.2.0 pins cubecl ^0.10 and unifies cleanly. mlrs-kernels stays feature-free (NO hand-written gemm_kernel); wrap lives in mlrs-backend/src/prims/gemm.rs.
- [Phase 02]: [02-01]: f64 GEMM accumulates in f64 via MatmulElems::from_globals (acc kept at f64 global dtype for non-f16/bf16 out), sidestepping cubek-matmul's default f32-stage MatmulPrecision<f64>. Passed 1e-5 oracle gate on cpu AND wgpu (SHADER_F64 present).
- [Phase 02]: [02-01]: Subgroup-query symbol RESOLVED = client.features().plane.contains(Plane::Ops) (cubecl::ir::features::Plane); plane width via properties().hardware.plane_size_{min,max}. Facade: capability::supports_plane / plane_supported. Plan 02 plane-path gates on this (no attempt-launch-and-skip fallback needed).
- [Phase 02]: [02-01]: GEMM host API validates geometry (PrimError::ShapeMismatch/DimMismatch) and returns Result before any unsafe launch (D-04 / T-0201-02). Transpose = InputBinding::swap_dims logical swap, no transpose buffer (D-06).

### Pending Todos

[From .planning/todos/pending/ — ideas captured during sessions]

None yet.

### Blockers/Concerns

[Issues that affect future work]

- Phase 3 (SVD/eig) needs `/gsd-plan-phase --research-phase 3` before coding — no pre-built CubeCL Jacobi SVD primitive.
- Phase 5 (LogisticRegression sub-task) needs `/gsd-plan-phase --research-phase 5` — QN/L-BFGS convergence parity with sklearn `lbfgs` is the highest correctness risk.
- f64 absence on wgpu (the primary CI gate): capability-gating must be in place from Phase 1 or f64 tests silently skip/fail.
- Maturin per-backend distribution naming (Phase 6) is undocumented first-party — small build-system spike expected.
- [02-01] ROADMAP Criterion 1 / REQUIREMENTS wording "wraps cubecl-matmul" must be updated to "wraps cubek-matmul" (the cubecl algorithm crates were split into tracel-ai/cubek and renamed cubek-*). Orchestrator action.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-12T00:00:00.000Z
Stopped at: Plan 02-01 complete (GEMM PRIM-01 via cubek-matmul wrap + Wave-0 infra: read_backs, subgroup probe, GEMM npz fixtures)
Resume file: .planning/phases/02-core-compute-primitives/02-02-PLAN.md (next: Plan 02-02)
