---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Plan 03-01 complete (ROCm/HIP bring-up — runtime.rs cubecl::hip re-export + Cargo rocm feature += cubecl/std,cubecl/default; spike_saxpy_runs_on_active_backend runs a real HIP kernel on gfx1100; capability split CONFIRMED rocm f32=true/f64=false, cpu f64=true; ROADMAP/PROJECT/03-CONTEXT reconciled to cpu+rocm gate per D-07)."
last_updated: "2026-06-12T03:14:45.655Z"
last_activity: 2026-06-12 -- Plan 03-01 complete (ROCm bring-up + gate reconciliation)
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 15
  completed_plans: 12
  percent: 36
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-11)

**Core value:** Correct, memory-efficient ML algorithms that match scikit-learn within 1e-5, running on any CubeCL backend from a single generic codebase.
**Current focus:** Phase 3 — svd-eigendecomposition-primitive-hard-gate

## Current Position

Phase: 3 (svd-eigendecomposition-primitive-hard-gate) — EXECUTING
Plan: 2 of 5 (03-01 complete)
Status: Executing Phase 3
Last activity: 2026-06-12 -- Plan 03-01 complete (ROCm bring-up + gate reconciliation)
Resume file: .planning/phases/03-svd-eigendecomposition-primitive-hard-gate/03-02-PLAN.md

Progress: [████░░░░░░] 36% (2/6 phases; 12/15 plans)

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
| Phase 02 P02 | 95 | 3 tasks | 8 files |
| Phase 02 P03 | 7 | 3 tasks | 8 files |
| Phase 02 P04 | 12 | 2 tasks | 9 files |
| Phase 02 P05 | 18 | 1 task | 1 file |
| Phase 03 P01 | 10 | 2 tasks | 5 files |

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
- [Phase 02]: [02-02]: Plane reduction kernels combine per-plane shuffle partials IN-CUBE to one partial per cube (NOT per-plane) — PLANE_DIM is runtime-variable on wgpu (min=32,max=64) so the host can't pre-size a per-plane output; in-cube combine makes plane/shared output layouts identical and the host plane-width-agnostic
- [Phase 02]: [02-02]: Host clamps the plane-path launch cube to >= plane width (else planes_per_cube=CUBE_DIM_X/PLANE_DIM rounds to 0 and the combine reads nothing → got=0)
- [Phase 02]: [02-02]: f32 large-magnitude reductions compare abs-OR-rel (numpy allclose) — strict abs-AND-rel is impossible when the f32 ULP (|x|·2⁻²³) exceeds 1e-5; both bounds stay 1e-5, f64 keeps strict assert_slice_close; sum/mean sweep uses non-negative data to avoid the near-zero relative artifact
- [Phase 02]: [02-02]: row-L2-norm (Plan 03 distance) = prims::reduce::row_reduce(.., ScalarOp::L2Norm, ..); column-mean (Plan 04 covariance) = column_reduce(.., ScalarOp::Mean, ..); per-row argmin (Plan 05 KMeans) = argmin_rows
- [Phase 02]: [02-03]: distance needs the SQUARED row norm ‖x‖² (not sqrt), so added ScalarOp::SumSq (Σxᵢ², no sqrt finalize) to prims::reduce — distinct from L2Norm which sqrt-finalizes; row_reduce(.., ScalarOp::SumSq, ..) is the ‖x‖² term of the GEMM-expansion
- [Phase 02]: [02-03]: scale kernel signature (Plan 04 covariance consumes) = mlrs_kernels::scale::launch::<F, R>(client, count, dim, input: ArrayArg, output: ArrayArg, factor: F) — factor a scalar F passed BY VALUE (no ScalarArg wrapper, cubecl 0.10)
- [Phase 02]: [02-03]: dist_combine_clamp uses a 2D launch (i=ABSOLUTE_POS_X rows, j=ABSOLUTE_POS_Y cols, 16×16 cube); the max(d²,0) clamp is the STATEMENT form so no negative squared distance escapes under f32 cancellation (distance_min_nonnegative pins min>=0)
- [Phase 02]: [02-03]: distance pipeline stays device-resident in distance.rs (grep to_host == 0); the optional Euclidean sqrt (D-08) runs in place over the already-clamped buffer so sqrt never sees a negative argument
- [Phase ?]: [01-04]: DeviceArray::from_host meters the byte footprint through the pool then uploads via client.create (cubecl 0.10 has no in-place write into an empty handle)
- [Phase ?]: [01-05]: f32 oracle near-zero floor raised to 1e-2 (in pipeline_test only) — cross-backend f32 saxpy rounding (~1 ULP, abs_err ~3e-8) exceeds the strict 1e-5 *relative* bound on near-cancellation results; the 1e-5 *absolute* bound stays enforced. Core compare.rs (Plan 02) left untouched.
- [Phase ?]: [01-05]: mimalloc #[global_allocator] defined exactly once in the mlrs-py cdylib (src/allocator.rs), never in a library crate; activation proven by exercising it (no public introspection symbol in the mimalloc crate)
- [Phase ?]: [01-05]: f64 oracle cases stay capability-gated via skip_f64_with_log (skip-with-log, never fail) for backend portability — pattern for all future f64 oracle tests
- [Phase 02]: [02-01]: GEMM substrate = WRAP cubek-matmul 0.2.0 (Task-1 checkpoint). cubecl-matmul 0.9-pre / cubecl-linalg 0.5 are abandoned on incompatible cubecl lines; cubek-matmul 0.2.0 pins cubecl ^0.10 and unifies cleanly. mlrs-kernels stays feature-free (NO hand-written gemm_kernel); wrap lives in mlrs-backend/src/prims/gemm.rs.
- [Phase 02]: [02-01]: f64 GEMM accumulates in f64 via MatmulElems::from_globals (acc kept at f64 global dtype for non-f16/bf16 out), sidestepping cubek-matmul's default f32-stage MatmulPrecision<f64>. Passed 1e-5 oracle gate on cpu AND wgpu (SHADER_F64 present).
- [Phase 02]: [02-01]: Subgroup-query symbol RESOLVED = client.features().plane.contains(Plane::Ops) (cubecl::ir::features::Plane); plane width via properties().hardware.plane_size_{min,max}. Facade: capability::supports_plane / plane_supported. Plan 02 plane-path gates on this (no attempt-launch-and-skip fallback needed).
- [Phase 02]: [02-01]: GEMM host API validates geometry (PrimError::ShapeMismatch/DimMismatch) and returns Result before any unsafe launch (D-04 / T-0201-02). Transpose = InputBinding::swap_dims logical swap, no transpose buffer (D-06).
- [Phase 02]: [02-04]: Covariance GEMM-output-buffer reuse (D-10 gate 3, LOAD-BEARING for Plan 05): covariance() drives the internal gemm(transa=true) into a SINGLE out buffer (caller's `out` when supplied D-11, else gemm's own pool.acquire) and launches scale with that SAME handle as input AND output (gram.handle()==in==out), normalising 1/(n-ddof) IN PLACE; returns Ok(gram) — the returned handle IS the GEMM output handle, no parallel Gram allocation. Plan 05 gate 3: pass a GEMM output DeviceArray as covariance's out; allocations does not bump for a fresh Gram.
- [Phase 02]: [02-04]: center_columns elementwise kernel added (Rule 2 — D-05) so covariance.rs keeps grep -c to_host == 0; out[tid]=a[tid]-mean[tid%cols], same per-element class as scale/clamp_nonneg, feature-free. NOT a new Gram kernel (AᵀA still composes GEMM); zero external deps (T-0204-SC held).
- [Phase 02]: [02-04]: ddof folded into the scale factor 1/(n_samples-ddof): ddof=0 population (1/n), ddof=1 sample (1/(n-1)). Fixtures pin np.cov(A, rowvar=False, ddof) — features as COLUMNS, matching the (n_samples, n_features) row-major contract; A 7×4, C 4×4. Direct centred-AᵀA/(n-ddof) host ref independent of the GEMM(transa) algebra.
- [Phase 02]: [02-05]: D-10 memory gate is build-failing (memory_gate_test.rs) — three HARD PoolStats assertions activate Phase-1 D-05's deferred gate. Gate 1 (reuse_bounded): allocations<=FIRST_ITER_ALLOCS for iters 2..N AND reuses>=N-1 (observed FIRST_ITER_ALLOCS=15, reuses=62, N=5). Gate 2 (read_backs==1): GEMM→reduce→distance chain + ONE terminal to_host_metered; the reduction's internal plain to_host does NOT bump read_backs (only the metered path does), so the counter isolates the terminal read. Gate 3 (Gram reuses GEMM buffer): free-list PROBE, not handle identity — CubeCL Handle has no PartialEq, so detect a parallel Gram by acquiring gram_bytes after covariance and checking reuse-vs-fresh; observed 0 parallel Gram allocations. Identical figures cpu==wgpu (counters backend-agnostic). Zero new deps, zero source-symbol changes.
- [Phase 02]: [02-05]: Pattern — free-list probe for allocation-identity when the handle type is not comparable: a reused-in-place buffer stays LIVE (off the free-list); only a parallel allocate-then-release lands on it, so acquire-and-check-reuse distinguishes the two.
- [Phase 03]: [03-01]: ROCm/HIP bring-up = two fixes — runtime.rs re-exports cubecl::hip::{AmdDevice as ActiveDevice, HipRuntime as ActiveRuntime} (no cubecl::rocm module in cubecl 0.10; rocm=["hip"]), and Cargo rocm feature = ["cubecl/rocm","cubecl/std","cubecl/default"] (cubecl/std propagates the multi_threading cfg into cubecl-hip, else 3× E0432 MultiStream/ResolvedStreams/EventStreamBackend). default-features=false on cubecl/cubek pins left intact (Security V12). spike_saxpy_runs_on_active_backend now runs a real HIP kernel on gfx1100.
- [Phase 03]: [03-01]: GATE = cpu(f64) + rocm(f32) project-wide from Phase 3 (D-07, supersedes cpu+wgpu). CONFIRMED empirically: capability backend=rocm f32_supported=true f64_supported=false; cpu f64_supported=true. f64-on-rocm SKIPS-with-log via the unchanged skip_f64_with_log gate because cubecl-cpp 0.10 does NOT register F64 for the HIP backend — EXPECTED, not a defect. f64 SVD/eig validates on cpu; rocm validates f32. wgpu opportunistic only. ROADMAP/PROJECT/03-CONTEXT reconciled.

### Pending Todos

[From .planning/todos/pending/ — ideas captured during sessions]

None yet.

### Blockers/Concerns

[Issues that affect future work]

- Phase 5 (LogisticRegression sub-task) needs `/gsd-plan-phase --research-phase 5` — QN/L-BFGS convergence parity with sklearn `lbfgs` is the highest correctness risk.
- [03-01 RESOLVED] f64 absence on the GPU gate: the gate is now cpu(f64) + rocm(f32) from Phase 3 (D-07). f64 validates on cpu; f64-on-rocm skips-with-log (cubecl-cpp 0.10 F64 unregistered for HIP) — capability gate (skip_f64_with_log) already in place since Phase 1. Every Phase 3+ f64 oracle test must mirror the gemm_test.rs skip pattern.
- Maturin per-backend distribution naming (Phase 6) is undocumented first-party — small build-system spike expected.
- [02-01] ROADMAP Criterion 1 / REQUIREMENTS wording "wraps cubecl-matmul" must be updated to "wraps cubek-matmul" (the cubecl algorithm crates were split into tracel-ai/cubek and renamed cubek-*). Orchestrator action.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-12 -- Plan 03-01 complete
Stopped at: Plan 03-01 complete — ROCm/HIP bring-up (runtime.rs cubecl::hip re-export + Cargo rocm feature += cubecl/std,cubecl/default). spike_saxpy_runs_on_active_backend runs a real HIP kernel on gfx1100. Capability split confirmed: rocm f32=true/f64=false, cpu f64=true. ROADMAP/PROJECT/03-CONTEXT reconciled to the cpu+rocm gate (D-07) — GPU work in plans 03-02..05 unblocked.
Resume file: .planning/phases/03-svd-eigendecomposition-primitive-hard-gate/03-02-PLAN.md (Wave-0 scaffold: PrimError NotSquare/NotConverged + gen_oracle.py svd/eigh fixtures + test skeletons).
