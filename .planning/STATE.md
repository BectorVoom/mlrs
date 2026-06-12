---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 04-02-PLAN.md
last_updated: "2026-06-12T07:06:44.000Z"
last_activity: 2026-06-12 -- Completed Phase 04 Plan 02 (Cholesky/SPD-solve primitive)
progress:
  total_phases: 6
  completed_phases: 3
  total_plans: 20
  completed_plans: 18
  percent: 55
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-11)

**Core value:** Correct, memory-efficient ML algorithms that match scikit-learn within 1e-5, running on any CubeCL backend from a single generic codebase.
**Current focus:** Phase 04 — closed-form-estimators

## Current Position

Phase: 04 (closed-form-estimators) — EXECUTING
Plan: 3 of 5
Status: Executing Phase 04
Last activity: 2026-06-12 -- Completed Phase 04 Plan 02 (Cholesky/SPD-solve primitive)
Resume file: .planning/phases/04-closed-form-estimators/04-03-PLAN.md

Progress: [█████░░░░░] 55% (3/6 phases; 18/20 plans)

## Performance Metrics

**Velocity:**

- Total plans completed: 10
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 5 | - | - |
| 3 | 5 | - | - |

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
| Phase 03 P02 | 8 | 2 tasks | 9 files |
| Phase 03 P03 | 38 | 3 tasks | 5 files |
| Phase 03 P04 | 45 | 3 tasks | 5 files |
| Phase 03 P05 | 18 | 2 tasks | 1 file |
| Phase 04 P01 | 9 | 3 tasks | 13 files |
| Phase 04 P02 | 5 | 2 tasks | 5 files |

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
- [Phase 03]: [03-02]: PrimError extended with NotSquare { operand, rows, cols } (D-06 eig squareness, ASVS V5) + NotConverged { operand, max_sweeps, residual } (D-12 Jacobi sweep cap) — thiserror, one variant per violation class. gen_oracle.py gen_svd uses np.linalg.svd(full_matrices=False) (descending S, D-02/D-04) saving A/U/S/Vt; gen_eigh uses np.linalg.eigh on a (M+Mᵀ)/2 symmetric matrix and stores w/V REVERSED to descending AT GENERATION (D-04) so the future test compares directly with no re-sort. Five fixtures committed: svd_tall f32+f64 (8×4), svd_wide f32 (4×8, Aᵀ-swap path), eigh f32+f64 (4×4).
- [Phase 03]: [03-02]: Nyquist Wave-0 scaffold pattern — svd_test.rs (7 fns) + eig_test.rs (4 fns) carry all VALIDATION.md test names as #[ignore] stubs that assert fixture load + shape well-formedness only (NO reference to non-existent prims::svd/eig symbols), so the test crate compiles today; 03-03/04 remove #[ignore] and wire the real svd()/eig() + invariants. f64 fixture tests gate on skip_f64_with_log (cpu runs, rocm skips) verbatim from gemm_test.rs.
- [Phase 03]: [03-03]: Jacobi convergence needs TWO thresholds, not one — a tiny rotation-skip bound (ε·‖A‖_F, so rotations are essentially never skipped) SEPARATE from a noise-floor-aware convergence-break bound (8·ε·‖A‖_F·√(n(n-1)/2)). Conflating them stalled the 256×64 f32 case at the 30-sweep cap (off-diag residual 7.9e-4 ≫ a single-ε 7e-5 bound) despite recon already within 1e-5; the √pairs scaling clears the accumulated f32 dot-product noise floor (≈4e-4) so it converges in ~7 sweeps. Reusable for the 03-04 two-sided eig kernel.
- [Phase 03]: [03-03]: A1/LDS budget REALIZED — the all-shared 256×64 f32 Jacobi tile (82176 B: A 64KiB + V 16KiB + accumulator) overflowed gfx1100's 65536 B LDS and HIP rejected the launch. Fix: keep A column-major in the GLOBAL a_out handle throughout the sweep; only V (≤16 KiB) + the off-diagonal accumulator stay in SharedMemory. The single-cube convergence loop is still fully in-kernel (global handle is cube-private for a single-cube launch) so D-11 gate 3 (no host round-trip between sweeps) holds. MAX_ROWS is now a host-side problem-size cap, not a shared size. Pattern: when a shared tile overflows LDS, keep the large operand in global, shared only for the small accumulator.
- [Phase 03]: [03-03]: thin-U via the Phase-2 gemm A·V from the ORIGINAL A and the kernel's accumulated V (NOT the kernel's rotated A) — satisfies D-02 + the plan's gemm key-link AND independently validates V. S[j]=‖B[:,j]‖₂ via column L2-norm reduce; U[:,j]=B[:,j]/S[j] with a 1e-8 near-zero floor leaving rank-deficient null-space columns at 0 (Pitfall 4, validated by the reconstruction invariant not per-vector compare). Wide path (D-05) materializes Aᵀ once into pooled scratch then relabels U=V', Vᵀ=U'ᵀ on the host.
- [Phase 03]: [03-03]: svd.rs has 10 plain to_host calls (criterion-4 literal grep wanted 0) — these are the post-convergence host-side thin-U normalize + descending sort + permute (D-04/A4 blessed) + one-time pre-launch ‖A‖_F estimate, on the PLAIN (non-metered) path so they do NOT bump read_backs (same precedent as reduce.rs internal per-row to_host). The CONVERGENCE LOOP (criterion-4's intent) is fully device-resident in-kernel; the D-11 read_backs==1 gate (03-05) is unaffected.
- [Phase 03]: [03-04]: Two-sided Jacobi eig pairs MUST be processed SEQUENTIALLY (single acting unit per pair), NOT the SVD's disjoint-parallel column schedule. A two-sided rotation Jᵀ·A·J touches the FULL rows AND columns p,q — including cross entries belonging to another index-disjoint pair — so index-disjoint pairs are NOT footprint-disjoint and a distributed/parallel shared-memory update RACES (first attempt stuck at residual ~0.1–0.8 → NotConverged on both cpu and rocm). Fix: unit 0 performs the whole rotation per pair, others idle, sync_cube after each pair (mirrors the SVD kernel's "acting unit does the whole rotation"). n is small (≤MAX_DIM=64, typically 4) so the serialization is cheap. Converges to ~1e-6 f32 / machine-precision f64.
- [Phase 03]: [03-04]: eig convergence uses a TRUE post-sweep off-diagonal-norm measured from the live matrix (each unit sums a_ij² over j≠i, log₂-tree reduce → 2·Σ_{i<j} a_ij²; the sqrt(2) double-count makes the break marginally stricter = more accurate), NOT an in-sweep per-pair accumulator (which underestimates because a later rotation in the same sweep refills an off-diagonal, stopping early). conv_thr keeps the 03-03 8·ε·‖A‖_F·√pairs form.
- [Phase 03]: [03-04]: eig() validate_geometry rejects a.len()!=n*n and n>MAX_DIM with PrimError::NotSquare BEFORE any unsafe launch (D-06 trusts symmetry, no (A+Aᵀ)/2); threads the covariance/GEMM out buffer straight through as the kernel working input (D-11 gate 2, no parallel n² alloc); descending eigenvalue sort + column permute host-side (D-04); NotConverged on cap (D-12). Small post-convergence w/V/info read-backs only — convergence loop fully in-kernel (D-11 gate 3), same precedent as svd.rs. The Task-2 `grep to_host==0` acceptance literally contradicts the mandated host descending sort; resolved in favor of the action/done + svd sibling.

- [Phase 04]: [04-01]: AlgoError is estimator-LOCAL in mlrs-algos (not mlrs-core), wrapping PrimError via #[from] — the n_components/alpha hyperparameter guards (T-04-01-01) are estimator-specific so the primitive layer never depends on them; ? stays ergonomic across prim calls. New PrimError::NotPositiveDefinite { operand, pivot_index, pivot_value } lives in mlrs-core for the 04-02 Cholesky negative-pivot guard (T-04-01-02).
- [Phase 04]: [04-01]: Fit/Predict/Transform trait surface (D-04) generic over <F: Float + CubeElement + Pod>; fit returns &mut self; Transform::inverse_transform has a default impl returning AlgoError::Unsupported so the surface stays total (PCA overrides, TruncatedSVD keeps default). mlrs-algos Cargo forwards cpu/wgpu/cuda/rocm to mlrs-backend (owns ActiveRuntime). lib.rs owns the module index; linear/decomposition mod.rs are stubs with commented future pub mod lines so 04-03/04/05 stay file-disjoint.
- [Phase 04]: [04-01]: Nyquist Wave-0 scaffold — five #[ignore] test stubs (cholesky + 4 estimators, 30 fns) assert fixture load+shape only (no non-existent symbol refs) so the crates compile today; 04-02/03/04/05 remove #[ignore] and wire the real assertion. cholesky_test::fixture_loads loads cholesky_f64 via mlrs_core::load_npz and validates A/b/x/L keys+shapes (Task 2's --ignored verify target). gen_oracle.py gained gen_cholesky (scipy SPD solve + L factor), gen_linear_regression (full-rank + near-collinear small-σ case), gen_ridge (cholesky solver alpha sweep), gen_pca (svd_solver=full tall/wide + canonical alias), gen_truncated_svd (DETERMINISTIC algorithm='arpack', NOT randomized). 14 f32/f64 .npz blobs committed; regen needs /tmp venv with numpy+scipy+scikit-learn (PEP 668).
- [Phase 04]: [04-02]: Cholesky/SPD-solve primitive = single-cube, all-shared-memory mlrs-kernels::cholesky_solve #[cube] kernel (feature-free, D-13) doing factor + forward + back triangular solve in ONE launch (D-11 gate 3, no host round-trip). UNIT-0-DOES-ALL serial schedule (jacobi_eig "acting unit" idiom) because the Cholesky-Banachiewicz recurrence is inherently sequential and n≤64 makes serialization cheap. Writes the lower factor L to a DEDICATED l_out buffer so the host checks ‖L·Lᵀ−A‖ from the KERNEL-EMITTED L (cholesky_solve_with_factor returns (x,L)), never re-derived. Diagonal sqrt guard (≤1e-12 floor) → info_out flag, never NaN (Pitfall 4). info array is LENGTH 3 [flag, pivot_index, pivot_value] (a length-2 encoded form reported the wrong pivot sign — fixed). prims::cholesky::cholesky_solve(pool,a,b,n,rhs,out) validates n*n/n≤MAX_DIM (NotSquare) + n*rhs (ShapeMismatch) BEFORE the unsafe launch (ASVS V5), threads out=Some Gram buffer through (D-11 gate 2, no parallel n² alloc), returns PrimError::NotPositiveDefinite on a non-positive pivot. ‖A·x−b‖ + ‖L·Lᵀ−A‖ + non-SPD all pass cpu(f64+f32)+rocm(f32; f64 skip-with-log). NOTE: mlrs-kernels has no cpu/rocm feature (feature-free by design) — the plan's `cargo build -p mlrs-kernels --features cpu` verb is wrong; the real launch-codegen gate is the backend build + cholesky_test under each feature.

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

Last session: 2026-06-12T07:06:44.000Z
Stopped at: Completed 04-02-PLAN.md (Cholesky/SPD-solve primitive)
Resume file: .planning/phases/04-closed-form-estimators/04-03-PLAN.md (LinearRegression estimator — SVD pseudo-inverse lstsq, centering for intercept; consumes the Phase-3 thin SVD primitive, no new kernel).
