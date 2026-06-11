---
phase: 02-core-compute-primitives
plan: 01
subsystem: compute-primitives
tags: [gemm, cubek-matmul, cubecl, blas, oracle, buffer-pool, subgroup, f64-gating]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: DeviceArray (pool-routed upload/read-back), BufferPool + PoolStats, capability gating (supports_type/skip_f64_with_log), mlrs-core oracle loader (load_npz) + assert_slice_close + tolerances, generic #[cube] launch idiom (saxpy)
provides:
  - "GEMM host API (PRIM-01): prims::gemm::gemm with BLAS-style transa/transb, device-resident in/out, pool-routed output buffer"
  - "cubek-matmul 0.2.0 wrap as the cubecl-0.10-compatible GEMM substrate (Task-1 checkpoint decision)"
  - "PoolStats.read_backs counter + BufferPool::record_read_back + DeviceArray::to_host_metered (D-10 memory-gate hook for Plan 02)"
  - "Subgroup-capability probe: capability::supports_plane / plane_supported (client.features().plane.contains(Plane::Ops)) + plane width via properties().hardware.plane_size_{min,max}"
  - "PrimError (ShapeMismatch / DimMismatch) in mlrs-core for D-04 geometry rejection"
  - "GEMM npz convention fixtures gemm_{f32,f64}_seed42.npz (A/B/C, non-square 5x4x3)"
  - "DeviceArray::from_raw (wrap a populated handle as a device-resident array)"
affects: [02-02-reduction-plane-path, 02-03-distance, 02-04-covariance, 02-05]

# Tech tracking
tech-stack:
  added: [cubek-matmul 0.2.0, cubek-std 0.2.0]
  patterns: ["wrap cubek-matmul launch_ref via TensorHandle::new_contiguous bindings + InputBinding + MatmulElems::from_globals + Strategy::default", "transpose = InputBinding::swap_dims logical (shape,strides) swap (no transpose buffer, D-06)", "f32 oracle near-zero floor helper (F32_GEMM_NEAR_ZERO_FLOOR=1e-2) reused from pipeline_test precedent", "host orchestration in mlrs-backend, kernels crate stays feature-free (D-13)"]

key-files:
  created:
    - crates/mlrs-backend/src/prims/mod.rs
    - crates/mlrs-backend/src/prims/gemm.rs
    - crates/mlrs-backend/tests/gemm_test.rs
    - tests/fixtures/gemm_f32_seed42.npz
    - tests/fixtures/gemm_f64_seed42.npz
  modified:
    - crates/mlrs-backend/src/pool.rs
    - crates/mlrs-backend/src/device_array.rs
    - crates/mlrs-backend/src/capability.rs
    - crates/mlrs-backend/tests/spike_test.rs
    - crates/mlrs-backend/src/lib.rs
    - crates/mlrs-backend/Cargo.toml
    - crates/mlrs-core/src/error.rs
    - crates/mlrs-core/src/lib.rs
    - scripts/gen_oracle.py

key-decisions:
  - "[02-01] Task-1 substrate RESOLVED: WRAP cubek-matmul 0.2.0 (NOT hand-write, NOT cubecl-matmul/cubecl-linalg). cubecl-matmul 0.9.0-pre.5 and cubecl-linalg 0.5.0 pin incompatible cubecl-core lines; cubek-matmul 0.2.0 pins cubecl ^0.10.0 and unifies cleanly on cubecl-core 0.10.0."
  - "[02-01] ROADMAP Criterion 1 / REQUIREMENTS wording 'wraps cubecl-matmul' must be updated to 'wraps cubek-matmul' (the cubecl algorithm crates were split into the tracel-ai/cubek repo and renamed cubek-*). Surfaced to orchestrator — do not silently diverge."
  - "[02-01] A3 subgroup-query symbol RESOLVED: client.features().plane.contains(Plane::Ops) (cubecl::ir::features::Plane), plane width via properties().hardware.plane_size_{min,max}. Stable property query EXISTS, so Plan 02's plane-path can gate on capability::plane_supported() (no attempt-launch-and-skip fallback needed)."
  - "[02-01] f64 GEMM accumulates in f64 via MatmulElems::from_globals (keeps acc at the f64 global dtype for non-f16/bf16 outputs), sidestepping cubek-matmul's default MatmulPrecision<f64> f32-stage profile (Pitfall 2). Verified within 1e-5 on cpu AND wgpu (this wgpu adapter has SHADER_F64)."
  - "[02-01] GEMM shape validation returns Result<DeviceArray, PrimError> (ShapeMismatch/DimMismatch) before any launch (D-04 / T-0201-02), rather than panicking."
  - "[02-01] f32 GEMM near-zero floor = 1e-2 (test-local, F32_GEMM_NEAR_ZERO_FLOOR), reusing the pipeline_test precedent: abs-only below the floor (still <=1e-5 abs), never loosens the 1e-5 abs bound; f64 keeps strict assert_slice_close."

patterns-established:
  - "Primitive host API pattern: validate geometry (PrimError) -> build cubek bindings from DeviceArray handles -> pool-routed out buffer -> launch_ref -> DeviceArray::from_raw (device-resident result, D-05)"
  - "Metered read-back: terminal reads go through DeviceArray::to_host_metered(&mut pool) so PoolStats.read_backs is a real runtime quantity (D-10 gate), not a code-review claim"

requirements-completed: [PRIM-01]

# Metrics
duration: 35min
completed: 2026-06-12
---

# Phase 02 Plan 01: GEMM Substrate + Wave-0 Infra Summary

**Backend-portable GEMM (PRIM-01) wrapping cubek-matmul 0.2.0 with BLAS-style transpose flags, oracle-validated within 1e-5 for f32 (always) and f64 (capability-gated) on cpu and wgpu, plus the Wave-0 hooks (PoolStats.read_backs, subgroup probe, GEMM npz fixtures) the rest of the phase composes on.**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-06-12
- **Tasks:** 6 (1 checkpoint pre-resolved, 5 executed)
- **Files modified:** 14 (5 created, 9 modified)

## Accomplishments
- PRIM-01 GEMM: `prims::gemm::gemm` wraps `cubek-matmul` `launch_ref`; device-resident in/out, pool-routed output buffer, transa/transb via `InputBinding::swap_dims` (no transpose buffer, D-06). 4/4 oracle tests green on cpu AND wgpu.
- Resolved the GEMM substrate blocker: `cubek-matmul 0.2.0` is the only cubecl-0.10-compatible matmul source (cubecl-matmul/cubecl-linalg are abandoned on older cubecl lines).
- Wave-0 infra for Plans 02–05: `PoolStats.read_backs` + `record_read_back` + `DeviceArray::to_host_metered` (D-10 memory gate), subgroup-capability probe (`supports_plane`/`plane_supported`), `PrimError` geometry-rejection enum, GEMM npz fixtures.
- f64 GEMM meets the 1e-5 gate by accumulating in f64 (`MatmulElems::from_globals`), avoiding the cubek-matmul default f32-accumulation precision caveat.

## Task Commits

1. **Task 1: GEMM substrate decision** — pre-resolved by orchestrator/user (WRAP cubek-matmul 0.2.0); no commit (decision recorded here).
2. **Task 2: PoolStats.read_backs + subgroup probe** — `eec3aed` (feat)
3. **Task 3: GEMM npz convention fixtures** — `e29d73e` (feat)
4. **Task 4: GEMM test scaffold + prims module + PrimError (RED)** — `2c98135` (test)
5. **Task 5: GEMM host API via cubek-matmul wrap (GREEN)** — `9c7a84a` (feat)
6. **Task 6: validate GEMM vs host ref + npz fixture** — `4cef2df` (test)

_TDD tasks 4 (RED) and 5 (GREEN) + 6 (validation) form the RED→GREEN→validate cycle for the GEMM feature._

## Files Created/Modified
- `crates/mlrs-backend/src/prims/gemm.rs` (created) — GEMM host API: geometry validation (PrimError), cubek-matmul binding construction, pool-routed out buffer, launch_ref, device-resident DeviceArray result.
- `crates/mlrs-backend/src/prims/mod.rs` (created) — prims barrel (`pub mod gemm`).
- `crates/mlrs-backend/tests/gemm_test.rs` (created) — 4 oracle tests (f32/f64 host-ref sweep incl. large-K, transpose, npz fixture); f32 near-zero floor helper.
- `tests/fixtures/gemm_{f32,f64}_seed42.npz` (created) — A(5x4)/B(4x3)/C named-array convention fixtures.
- `crates/mlrs-backend/src/pool.rs` — `read_backs` field + `record_read_back`.
- `crates/mlrs-backend/src/device_array.rs` — `to_host_metered` + `from_raw`.
- `crates/mlrs-backend/src/capability.rs` — `supports_plane` / `plane_supported`.
- `crates/mlrs-backend/tests/spike_test.rs` — `spike_subgroup_query_reports_support` probe.
- `crates/mlrs-core/src/error.rs` + `lib.rs` — `PrimError` enum + re-export.
- `crates/mlrs-backend/src/lib.rs` — `pub mod prims`.
- `crates/mlrs-backend/Cargo.toml` — `cubek-matmul` + `cubek-std` 0.2.0 deps (default-features=false).
- `scripts/gen_oracle.py` — `gen_gemm` case.

## Decisions Made
See `key-decisions` frontmatter. Headlines:
- **Substrate = cubek-matmul 0.2.0 (WRAP).** Verified clean resolution against cubecl-core 0.10.0; cubecl-matmul (0.9-pre) and cubecl-linalg (0.5) are incompatible.
- **ROADMAP/REQUIREMENTS wording update flagged:** "wraps cubecl-matmul" → "wraps cubek-matmul" (surface to orchestrator).
- **Subgroup-query symbol pinned:** `client.features().plane.contains(Plane::Ops)`; a stable query exists so Plan 02 can gate on `capability::plane_supported()` (no attempt-launch-and-skip fallback required).
- **f64 accumulates in f64** via `MatmulElems::from_globals` (Pitfall 2 mitigation).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking / Task-1 checkpoint consequence] WRAP path replaces hand-written kernel in mlrs-kernels**
- **Found during:** Tasks 4–5 (GEMM scaffold + implementation)
- **Issue:** The plan's `files_modified` frontmatter lists `crates/mlrs-kernels/src/gemm.rs` (hand-written `gemm_kernel`) and a `pub use gemm::gemm_kernel` in `mlrs-kernels/src/lib.rs`. The Task-1 checkpoint (pre-resolved) selected the WRAP path, which omits both.
- **Fix:** No `gemm.rs` created in `mlrs-kernels`; `mlrs-kernels` stays feature-free (D-13 verified — no backend feature in its Cargo.toml). The matmul wrap lives entirely in `mlrs-backend/src/prims/gemm.rs`; `cubek-matmul`/`cubek-std` added to `mlrs-backend/Cargo.toml` (NOT mlrs-kernels).
- **Verification:** `cargo build -p mlrs-kernels` (feature-free) + `cargo build -p mlrs-backend --features cpu/wgpu` all green; 4/4 GEMM oracle tests pass on both backends.
- **Committed in:** `9c7a84a` (Task 5).

**2. [Rule 2 - Missing critical] GEMM host API returns Result<_, PrimError> with DimMismatch**
- **Found during:** Task 4/5
- **Issue:** Plan describes `rows*cols==len` asserts; for a correct, recoverable D-04 input-validation boundary (T-0201-02, ASVS V5) the contraction-dimension disagreement (`k != k'`) also needs rejection.
- **Fix:** Added `PrimError::DimMismatch` alongside `ShapeMismatch`; `gemm` returns `Result` and validates all operand geometries before any unsafe launch.
- **Verification:** `validate_geometry` covers lhs/rhs/out/k; all GEMM tests pass.
- **Committed in:** `2c98135` (Task 4) / `9c7a84a` (Task 5).

---

**Total deviations:** 2 (1 blocking checkpoint-consequence, 1 missing-critical). **Impact:** No scope creep — deviation 1 is the direct consequence of the pre-resolved Task-1 decision; deviation 2 strengthens the D-04 trust boundary.

## Issues Encountered
- **cubek-matmul has no simple `matmul(client, handle, shape, ...)` helper** — the public entry is the low-level `launch_ref(&Strategy, client, InputBinding, InputBinding, TensorBinding, &mut MatmulElems)`. Resolved by reading the crate source (`launch/base.rs`, `launch/args.rs`, `cpu_reference.rs`, `definition/{spec,base}.rs`) to learn the construction: `TensorHandle::new_contiguous(shape, handle, StorageType)` → `.binding()` → `InputBinding::new(binding, dtype)`; `MatmulElems::from_globals`; `Strategy::default()`.
- **f64 precision guardrail (checkpoint STOP condition):** cubek-matmul's default `MatmulPrecision<f64>` uses an f32 stage/register. Mitigated by `MatmulElems::from_globals` (acc stays f64 for f64 output). The f64 oracle gate passed within 1e-5 (abs AND rel) on cpu AND wgpu — **guardrail NOT tripped**, substrate retained.
- **No CubeCL build errors** were encountered (clean first compile on cpu + wgpu), so the AGENTS.md §4 cubecl_error_guideline.md protocol was not invoked.

## Threat Flags
None — GEMM is a numerical compute kernel with no auth/session/network/PII surface. The two trust boundaries in the plan threat model (host slice → device buffer length; caller `(rows,cols)` → kernel index math) are mitigated by `DeviceArray.len` as the read-back source of truth and the pre-launch `PrimError` geometry validation (T-0201-01 / T-0201-02). The wrap path's new dependency (T-0201-SC) was gated behind the Task-1 checkpoint and verified (cubek-matmul 0.2.0, repo tracel-ai/cubek).

## Next Phase Readiness
- **Plan 02 (reduction / plane path):** `capability::plane_supported()` is ready; plane width available via `properties().hardware.plane_size_{min,max}`. Stable query exists, so the plane-path skip gates on capability (no launch-and-catch needed).
- **Plan 02 (memory gate):** `PoolStats.read_backs` + `to_host_metered` ready for the D-10 read-back assertion.
- **Plans 03/04 (distance, covariance):** GEMM (`prims::gemm::gemm`) ready to compose; covariance reuses `transa`/`transb` for XᵀX without a transpose buffer (D-06).
- **ACTION FOR ORCHESTRATOR:** update ROADMAP Criterion 1 / REQUIREMENTS wording from "wraps cubecl-matmul" to "wraps cubek-matmul".

---
*Phase: 02-core-compute-primitives*
*Completed: 2026-06-12*

## Self-Check: PASSED
- All 5 created files verified present on disk.
- All 5 task commits (eec3aed, e29d73e, 2c98135, 9c7a84a, 4cef2df) verified in git log.
- `cargo test -p mlrs-backend --features cpu gemm` and `--features wgpu gemm`: 4/4 green each.
- `cargo build -p mlrs-kernels` (feature-free) + `cargo build -p mlrs-backend --features cpu/wgpu`: green.
