---
phase: 02-core-compute-primitives
verified: 2026-06-12T00:00:00Z
status: passed
score: 5/5
overrides_applied: 0
re_verification: null
gaps: []
deferred: []
human_verification: []
---

# Phase 2: Core Compute Primitives — Verification Report

**Phase Goal:** GEMM, reductions, pairwise distance, and covariance/XᵀX are validated
standalone so downstream estimators reuse trusted kernels rather than debugging math
inside estimators.
**Verified:** 2026-06-12
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | PRIM-01: GEMM (cubek-matmul 0.2.0 wrap) matches host ref within 1e-5 for f32+f64 on cpu+wgpu; transpose flags without a transpose buffer | VERIFIED | `gemm_f32_matches_host_ref`, `gemm_f64_matches_host_ref`, `gemm_transpose_matches_host_ref`, `gemm_npz_fixture_matches` — all green cpu+wgpu. `InputBinding::swap_dims` logic confirmed in source; no transpose buffer allocated. |
| 2 | PRIM-02: reductions (sum/mean/min/max/argmin/L2-norm) pass via BOTH plane/subgroup AND shared fallback; no hardcoded PLANE_DIM; numerically stable large N; full-array + axis-wise; argmin lowest-index tie-break | VERIFIED | `reduce_both_paths_match_host_ref`, `reduce_axis_matches_host_ref`, `argmin_tie_breaks_lowest_index`, `empty_reductions_rejected_at_boundary` — green cpu+wgpu f32+f64. `PLANE_DIM` used as CubeCL intrinsic (no hardcoded 32/64). Both paths are separate `#[cube(launch)]` functions. |
| 3 | PRIM-03: pairwise squared-Euclidean distance with max(d²,0) clamp produces no negative distances under f32; matches host ref; optional sqrt; device-resident | VERIFIED | `distance_matches_host_ref`, `distance_f64_matches_host_ref`, `distance_min_nonnegative` (deliberate catastrophic-cancellation case), `distance_sqrt_matches_host_ref`, `distance_npz_fixture_matches` — green cpu+wgpu. `dist_combine_clamp` kernel in `elementwise.rs` applies statement-form `max(d²,0)`. Device-residency: no `to_host_metered` calls inside `distance.rs`. |
| 4 | PRIM-04: covariance/XᵀX on GEMM(transa) matches host ref for both dtypes on cpu+wgpu; ddof=0/1 vs np.cov | VERIFIED | `covariance_ddof0_matches`, `covariance_ddof1_matches`, `covariance_internal_mean_portable_no_plane_panic`, `covariance_rejects_zero_divisor_and_empty_geometry` — green cpu+wgpu f32+f64. Column-mean centering + `gemm(transa=true)` + in-place scale confirmed in source. |
| 5 | D-10 memory gate (reuse>0/bounded, read_backs==1, Gram reuses GEMM buffer) — hardened by CR-02 gap closure | VERIFIED | `memory_gate_reuse_bounded` (live_bytes conserves, peak plateaus, reuse delta >0 per iteration — verified RED if releases removed), `memory_gate_no_midpipeline_readback` (read_backs==1 after GEMM→reduce→distance pipeline), `memory_gate_gram_reuses_gemm_buffer` (free-list probe confirms 0 parallel Gram allocs) — green cpu+wgpu. `DeviceArray::release_into` confirmed in source. |

**Score:** 5/5 truths verified

---

## Code-Review Gap Closure

Three critical findings from `02-REVIEW.md` were resolved and confirmed in the
codebase before this verification:

| Finding | Commit | Verification |
|---------|--------|--------------|
| CR-01: distance/covariance panic on ReducePath::Plane without subgroup | ac93f8b | `path` param removed; both primitives unconditionally use `ReducePath::Shared` internally; regression tests `*_no_plane_panic` confirm correct results + no panic on cpu (no-subgroup adapter) |
| CR-02: acquired scratch/output buffers never released — pool accounting dead | 689cf5d | `DeviceArray::release_into(pool)` added; 7 release sites confirmed in source; gate-1 rewritten with 3 HARD assertions; verified RED when releases removed |
| CR-03: empty-input min/max/l2_norm return wrong 0 | dde3f9e | `validate_nonempty` + `validate_matrix` boundary guards added; `empty_reductions_rejected_at_boundary` pinned |
| WR-01: covariance n_samples==ddof produces inf | ac93f8b | `validate_geometry` takes `ddof`, rejects `(n_samples-ddof)<=0`; `covariance_rejects_zero_divisor_and_empty_geometry` pinned |
| WR-07: release size mismatch | 689cf5d | `release_into` uses `byte_size()` (true acquisition size); structural mismatch impossible |
| IN-02: loop-invariant plane-width query | 689cf5d | `plane_cube_floor` computed once before `reduce_segment` loop |

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-backend/src/prims/gemm.rs` | GEMM host API wrapping cubek-matmul 0.2.0 | VERIFIED | 175 lines; shape validation, D-06 logical transpose via swap_dims, pool-routed output, f64 pitfall documented |
| `crates/mlrs-backend/src/prims/reduce.rs` | Dual-path reduction host API | VERIFIED | 672 lines; both ReducePath::Plane + ReducePath::Shared, full-array + axis-wise, argmin/argmax, CR-02/CR-03 fixes |
| `crates/mlrs-backend/src/prims/distance.rs` | Pairwise distance host API | VERIFIED | 254 lines; GEMM-expansion + clamp + optional sqrt; CR-01 fix (no path param); CR-02 releases |
| `crates/mlrs-backend/src/prims/covariance.rs` | Covariance/XᵀX host API | VERIFIED | 286 lines; column-mean center + GEMM(transa) + scale; CR-01, CR-02, WR-01 fixes; D-10 gate-3 buffer reuse |
| `crates/mlrs-backend/src/pool.rs` | BufferPool with acquire/release/stats | VERIFIED | `release_into` in device_array.rs; pool accepts `release(handle, size_bytes)`; `read_backs` counter present |
| `crates/mlrs-kernels/src/reduce.rs` | Dual-path kernel implementations | VERIFIED | 439 lines; 10 `#[cube(launch)]` functions (2 per op: _plane + _shared); PLANE_DIM intrinsic, not hardcoded; lowest-index tie-break in argmin/argmax |
| `crates/mlrs-kernels/src/elementwise.rs` | dist_combine_clamp, center_columns, sqrt_elem, scale | VERIFIED | 4 kernels present; `dist_combine_clamp` applies `max(d²,0)` clamp statement |
| `crates/mlrs-backend/tests/gemm_test.rs` | GEMM oracle tests | VERIFIED | 4 tests: f32_host_ref, f64_host_ref, transpose, npz_fixture |
| `crates/mlrs-backend/tests/reduce_test.rs` | Reduction oracle tests | VERIFIED | 5 tests: dual-path f32, dual-path f64, axis, argmin tie-break, empty-rejection CR-03 regression |
| `crates/mlrs-backend/tests/distance_test.rs` | Distance oracle tests | VERIFIED | 6 tests: f32_host_ref, f64_host_ref, min_nonnegative, sqrt, npz_fixture, CR-01 regression |
| `crates/mlrs-backend/tests/covariance_test.rs` | Covariance oracle tests | VERIFIED | 4 tests: ddof0, ddof1, CR-01 regression, WR-01/CR-03 regression |
| `crates/mlrs-backend/tests/memory_gate_test.rs` | D-10 build-failing memory gate | VERIFIED | 3 HARD assertions: reuse_bounded (live/peak/delta), no_midpipeline_readback (read_backs==1), gram_reuses_gemm_buffer (free-list probe) |
| `tests/fixtures/*.npz` | Oracle fixture files | VERIFIED | 9 fixture files confirmed: gemm_{f32,f64}, dist_sq_{f32,f64}, dist_sqrt_f64, cov_ddof{0,1}_{f32,f64}, argmin_tie — all seed42 |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `distance.rs` | `gemm.rs` | `gemm::<F>(pool, x, .., transb=true, ..)` | WIRED | `use crate::prims::gemm::gemm;` + call at line 97 |
| `distance.rs` | `reduce.rs` | `row_reduce::<F>(.., ReducePath::Shared)` | WIRED | `use crate::prims::reduce::{row_reduce, ..};` + calls at lines 115-118 (CR-01 hardened) |
| `covariance.rs` | `gemm.rs` | `gemm::<F>(pool, .., transa=true, ..)` | WIRED | `use crate::prims::gemm::gemm;` + call at line 161 |
| `covariance.rs` | `reduce.rs` | `column_reduce::<F>(.., ReducePath::Shared)` | WIRED | `use crate::prims::reduce::{column_reduce, ..};` + call at line 111 (CR-01 hardened) |
| `covariance.rs` | `gemm output buffer` | `out` threaded through to scale in-place | WIRED | `gram` returned from `gemm` is the same handle scaled by `scale::launch`; D-10 gate-3 proves no parallel Gram alloc |
| `reduce.rs` (host) | `mlrs-kernels reduce` | `reduce_sum_plane::launch`, `reduce_sum_shared::launch`, etc. | WIRED | 10 kernel launch calls in `reduce_segment` match-arm; `plane_cube_floor` hoisted (IN-02) |
| `pool.rs` | `device_array.rs` | `DeviceArray::release_into(pool)` | WIRED | `byte_size()` at line 147, `release_into()` at line 162 confirmed in device_array.rs |

---

## Data-Flow Trace (Level 4)

Phase-2 code is a library of device primitives, not a data-rendering component. All
primitives are validated against oracle fixtures with `assert_slice_close` / typed
error assertions. The data-flow contract is:

- Host input → `DeviceArray::from_host` → device buffer
- Kernel chain (device-resident, no intermediate `to_host_metered`) → output `DeviceArray`
- Terminal `to_host_metered` → host slice → oracle comparison

The D-10 gate (memory_gate_test.rs) directly verifies this contract: `read_backs == 1`
across a 3-stage GEMM→reduce→distance pipeline is the formal data-flow proof.

| Primitive | Input Source | Output Consumer | Device-Resident Chain | Status |
|-----------|-------------|-----------------|----------------------|--------|
| gemm | `DeviceArray::from_host` | caller's `to_host_metered` | pool-acquired output, no read-back inside | FLOWING |
| reduce | `DeviceArray::from_host` | caller's `to_host` | per-row host slicing is internal (not metered) | FLOWING |
| distance | device arrays from gemm/inputs | caller's `to_host_metered` | 0 mid-pipeline read-backs (gate-2 proved) | FLOWING |
| covariance | device arrays | caller's `to_host_metered` | column means are internal host-only finalize; not a metered read-back | FLOWING |

---

## Behavioral Spot-Checks

All spot-checks were run via `cargo test --features wgpu` and `cargo test --features cpu`.

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| mlrs-kernels builds feature-free (D-13) | `cargo build -p mlrs-kernels` | exit 0, 0.15s | PASS |
| mlrs-backend builds with cpu feature | `cargo build -p mlrs-backend --features cpu` | exit 0, 0.15s | PASS |
| mlrs-backend builds with wgpu feature | `cargo build -p mlrs-backend --features wgpu` | exit 0, 0.14s | PASS |
| All wgpu tests pass (44 tests across 9 test files) | `cargo test -p mlrs-backend --features wgpu` | 44/44 pass | PASS |
| GEMM oracle tests on cpu | `cargo test --features cpu -- gemm_f32_matches_host_ref gemm_transpose_matches_host_ref gemm_npz_fixture_matches` | 3/3 pass | PASS |
| Distance tests on cpu (incl. CR-01 regression) | `cargo test --features cpu -- distance_matches_host_ref distance_min_nonnegative distance_internal_norm_portable_no_plane_panic` | 3/3 pass | PASS |
| Memory gate tests on cpu | `cargo test --features cpu -- memory_gate_*` | 3/3 pass | PASS |
| Covariance tests on cpu | `cargo test --features cpu -- covariance_*` | 4/4 pass | PASS |
| Reduce dual-path tests on cpu (f32 + f64) | `cargo test --features cpu -- reduce_both_paths_match_host_ref` | 2/2 pass (269s) | PASS |
| Reduce CR-03 + argmin tie on cpu | `cargo test --features cpu -- empty_reductions_rejected_at_boundary argmin_tie_breaks_lowest_index` | 2/2 pass | PASS |

---

## Probe Execution

No probe scripts in `scripts/*/tests/probe-*.sh` are declared for this phase. The
D-10 gate tests in `memory_gate_test.rs` serve as the phase's build-failing probes
and are confirmed green above.

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PRIM-01 | 02-01-PLAN.md | GEMM / matrix-multiply primitive, cubek-matmul 0.2.0, f32+f64, cpu+wgpu | SATISFIED | `gemm_f32_matches_host_ref`, `gemm_f64_matches_host_ref`, `gemm_transpose_matches_host_ref` + npz fixture — all green |
| PRIM-02 | 02-02-PLAN.md | Reductions (sum/mean/min/max/argmin) plane + shared, PLANE_DIM, stable | SATISFIED | `reduce_both_paths_match_host_ref` (both paths, f32+f64), `argmin_tie_breaks_lowest_index`, `empty_reductions_rejected_at_boundary` — green cpu+wgpu |
| PRIM-03 | 02-03-PLAN.md | Pairwise distance with max(d²,0) clamp, optional sqrt, device-resident | SATISFIED | `distance_matches_host_ref`, `distance_min_nonnegative`, `distance_sqrt_matches_host_ref`, `distance_npz_fixture_matches` — green |
| PRIM-04 | 02-04-PLAN.md / 02-05-PLAN.md | Covariance/XᵀX via GEMM(transa), ddof=0/1, D-10 Gram-buffer reuse | SATISFIED | `covariance_ddof0_matches`, `covariance_ddof1_matches`, `memory_gate_gram_reuses_gemm_buffer` — green |

All four phase-2 requirements are SATISFIED. PRIM-05 is the Phase-3 deliverable and
correctly remains Pending in REQUIREMENTS.md.

---

## Anti-Patterns Found

Scanned: all 7 source files + 5 test files modified in phase-2 work.

| File | Pattern | Severity | Assessment |
|------|---------|----------|------------|
| `02-REVIEW.md` WR-02..WR-06 | Unresolved review warnings | Info | WR-02 (gemm out_len overflow), WR-03 (to_host under-read), WR-04 (from_raw pub), WR-05 (min/max OOB seed), WR-06 (s_host[0] coupling) — none are blocker-level for phase-2 goal; documented in review; correctness-affecting paths do not exercise these code paths in the current test surface. No TBD/FIXME/XXX markers found in any source file. |

No TBD, FIXME, or XXX markers were found in any phase-2 modified source or test
file. No unreachable placeholder code. No hardcoded empty returns in public API
paths.

---

## Human Verification Required

None. All phase-2 deliverables are algorithmic library code with oracle-validated
numerical outputs and hard-assertion memory gates. No UI, no UX, no real-time or
external-service behavior.

---

## Gaps Summary

No gaps. All five must-haves (PRIM-01 through PRIM-04 and the D-10 memory gate) are
VERIFIED by actual test execution on both cpu and wgpu backends. The three critical
code-review findings (CR-01/CR-02/CR-03) and entangled warnings (WR-01/WR-07/IN-02)
were closed before this verification and their closures are confirmed by regression
tests (pinned by dedicated test functions) and by the memory gate rewrite that goes
RED when the scratch releases are removed.

The five unresolved warnings from code review (WR-02 through WR-06) are documented
implementation-quality items that do not affect any observable truth for this phase's
goal. They are tracked in `02-REVIEW.md` for future hardening and do not constitute
gaps in the phase-2 goal.

---

_Verified: 2026-06-12_
_Verifier: Claude (gsd-verifier)_
