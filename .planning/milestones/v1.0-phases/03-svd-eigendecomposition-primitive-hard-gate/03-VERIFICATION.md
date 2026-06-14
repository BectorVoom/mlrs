---
phase: 03-svd-eigendecomposition-primitive-hard-gate
verified: 2026-06-12T00:00:00Z
status: passed
score: 14/14 must-haves verified
overrides_applied: 0
---

# Phase 3: SVD / Eigendecomposition Primitive (Hard Gate) Verification Report

**Phase Goal:** Deliver a validated SVD + symmetric-eigendecomposition primitive (PRIM-05) — the single hardest, highest-leverage primitive — written once in CubeCL, generic over f32/f64 and the active runtime, matching numpy/scikit-learn within 1e-5, with the cpu(f64)+rocm(f32) hard gate and memory discipline (bounded scratch, buffer reuse, no host round-trip in the convergence loop) enforced by build-failing tests.
**Verified:** 2026-06-12
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A real #[cube] kernel compiles and runs end-to-end on gfx1100 via --features rocm (ROCm bring-up gate). | VERIFIED | `runtime.rs` wires `cubecl::hip::{AmdDevice as ActiveDevice, HipRuntime as ActiveRuntime}` under `cfg(feature = "rocm")`. Full 55-test suite green on rocm. |
| 2 | capability: active_backend_name()==rocm, supports_type(F32)==true, supports_type(F64)==false on rocm. | VERIFIED | `capability_test` on rocm logs `f32_supported=true`, `f64_supported=false`; skip path confirmed. |
| 3 | supports_type(F64)==false on rocm is documented as expected (f64 validates on cpu), not a defect. | VERIFIED | ROADMAP SC1/SC3, 03-CONTEXT.md D-07 all carry the explicit `f64-on-rocm SKIPS-with-log` note. No stale `f64 runs natively on rocm` wording. |
| 4 | ROADMAP/PROJECT/CONTEXT document the cpu+rocm gate with the f64-on-cpu/f32-on-rocm split; no stale cpu+wgpu or f64-runs-on-rocm wording for Phase 3+. | VERIFIED | ROADMAP.md Phase 3 SC1 and SC3 state `cpu + rocm` gate with the split. Overview sentence says `from Phase 3 the gate moves to cpu + rocm`. 03-CONTEXT.md D-07 carries the research correction. |
| 5 | PrimError carries NotSquare and NotConverged variants. | VERIFIED | `crates/mlrs-core/src/error.rs:109,126` — both variants with `#[error(...)]` messages and named fields. |
| 6 | Committed .npz fixtures exist for svd tall (f32+f64), svd wide (f32), svd tall ODD (f32+f64), and eigh (f32+f64). | VERIFIED | `tests/fixtures/`: `svd_tall_f32_seed42.npz`, `svd_tall_f64_seed42.npz`, `svd_wide_f32_seed42.npz`, `svd_tall_odd_f32_seed42.npz`, `svd_tall_odd_f64_seed42.npz`, `eigh_f32_seed42.npz`, `eigh_f64_seed42.npz`. All 7 present. |
| 7 | A Jacobi SVD of a tall matrix (including ODD thin dimensions — CR-01 fix) matches numpy svd(full_matrices=False) within tolerance after svd_flip on cpu (f32+f64) and rocm (f32); f64 SKIPS on rocm. | VERIFIED | 10/10 svd_test green on cpu (including `svd_tall_odd_f32_fixture`, `svd_tall_odd_f64_fixture`); 10/10 on rocm (f64 fixtures log "SKIPPED" and return early — `test ... ok`). Ghost-padded schedule in `jacobi_svd.rs:177-188` visits all n(n-1)/2 pairs for both even and odd cols. |
| 8 | Wide matrices (m<n) handled by the Aᵀ-and-swap path and hold the same tolerance. | VERIFIED | `svd_wide_f32_fixture` passes on cpu and rocm. `svd.rs:138-157` implements the wide path materalizing Aᵀ and calling `svd_tall` with `swap_uv=true`. |
| 9 | Reconstruction ‖U·diag(S)·Vᵀ−A‖/‖A‖ ≤ 1e-5 and orthonormality ‖UᵀU−I‖/‖VᵀV−I‖ invariants hold. | VERIFIED | `svd_reconstruction_invariant` (relative Frobenius ≤ 1e-5, WR-04 tightened) and `svd_orthonormality_invariant` pass on cpu and rocm. Post-sweep clean measurement (WR-01 fix) in `jacobi_svd.rs:270-298` ensures the convergence norm reflects the returned matrix. |
| 10 | NotConverged is reachable and surfaces correctly (WR-05 fix). | VERIFIED | `svd_not_converged_on_low_sweep_cap` passes: `svd_with_max_sweeps` with cap=1 on an 8×5 matrix returns `Err(PrimError::NotConverged{operand="svd", max_sweeps=1, residual>0})`. Same matrix converges under production cap. |
| 11 | A symmetric eigendecomposition matches np.linalg.eigh (reversed to descending) within tolerance after sign alignment on cpu (f32+f64) and rocm (f32). | VERIFIED | 4/4 eig_test green on cpu (f32+f64); 4/4 on rocm (f64 SKIPS). `eig_symmetric_f32_fixture` and `eig_symmetric_f64_fixture` pass. `eig_residual_invariant` and `eig_clustered_invariant` pass. |
| 12 | Eigenvalues descending (D-04); non-square rejected with PrimError::NotSquare before any unsafe launch (D-06). | VERIFIED | `eig.rs:210-239` — `validate_geometry` returns `NotSquare` when `a.len() != n*n`. Host descending sort at `eig.rs:185-198`. No (A+Aᵀ)/2 step. |
| 13 | The three D-11 memory gates are HARD and build-failing (not weakened): memory_gate_jacobi_scratch_bounded, memory_gate_eig_reuses_gram_buffer, memory_gate_svd_no_midsweep_readback (read_backs==0 after svd(), ==1 after terminal to_host_metered). | VERIFIED | All 6 memory gates (3 Phase-2 + 3 Phase-3 D-11) green on cpu and rocm. `memory_gate_svd_no_midsweep_readback` asserts `read_backs==0` after `svd()` (unmetered `to_host` used internally) then `==1` after `to_host_metered`. `memory_gate_eig_reuses_gram_buffer` asserts `peak_rise < 2*n2_bytes` (no parallel n² copy). `memory_gate_jacobi_scratch_bounded` asserts alloc delta==0 + live/peak conserved from iteration 2 onward. |
| 14 | The convergence sweep loop runs entirely in-kernel (single cube) — no host round-trip between sweeps. | VERIFIED | `memory_gate_svd_no_midsweep_readback` is the build-failing proof. `jacobi_svd.rs:172-320` — the sweep loop, convergence test, and V write-back are all inside one `#[cube(launch)]` kernel. `svd.rs` reads only the 2-element `info` array post-launch (plain `to_host`, unmetered). |

**Score:** 14/14 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-kernels/src/jacobi_svd.rs` | One-sided Jacobi SVD #[cube(launch)] kernel, generic over F | VERIFIED | 371 lines. Ghost-padded schedule (CR-01 fix), post-sweep clean convergence measurement (WR-01 fix), `SharedMemory` for V, A in global (LDS budget). No in-source mod tests. |
| `crates/mlrs-kernels/src/jacobi_eig.rs` | Two-sided cyclic Jacobi eig #[cube(launch)] kernel, generic over F | VERIFIED | 325 lines. Sequential pairs (race-safe), in-kernel convergence, A+V in shared. No in-source mod tests. |
| `crates/mlrs-backend/src/prims/svd.rs` | svd() host orchestration: validate, wide Aᵀ-swap, thin-U, descending sort | VERIFIED | 462 lines. `pub fn svd` + `pub fn svd_with_max_sweeps`. validate_geometry pre-launch. Wide path via Aᵀ materialization + swap_uv. Thin-U via Phase-2 GEMM + column_reduce. Descending sort. No in-API to_host of working buffers. |
| `crates/mlrs-backend/src/prims/eig.rs` | eig() host orchestration: validate-square, launch, descending sort, buffer reuse | VERIFIED | 295 lines. `pub fn eig`. NotSquare pre-launch. `out` parameter threads covariance/GEMM buffer through as kernel input. Descending sort. No in-API to_host of working buffers. |
| `crates/mlrs-backend/tests/svd_test.rs` | SVD oracle + invariant test suite, 10 test functions | VERIFIED | 541 lines. 10 tests: tall f32/f64 fixture, wide f32, odd f32/f64 (CR-01), NotConverged, reconstruction, orthonormality, degenerate, moderate 256x64. All 10 green cpu (132s) and rocm (5s). |
| `crates/mlrs-backend/tests/eig_test.rs` | Eig oracle + residual invariant test suite, 4 test functions | VERIFIED | 306 lines. 4 tests: f32/f64 fixture, residual invariant, clustered invariant. All 4 green cpu and rocm. |
| `crates/mlrs-backend/tests/memory_gate_test.rs` | 3 new D-11 SVD/eig PoolStats assertions extending the Phase-2 gate | VERIFIED | Three new HARD gates present and green on cpu and rocm. No assertions weakened. |
| `tests/fixtures/svd_tall_odd_f32_seed42.npz` | Odd-dim fixture (9×5) for CR-01 regression gate | VERIFIED | File present. Used by `svd_tall_odd_f32_fixture` and `svd_tall_odd_f64_fixture`. |
| `crates/mlrs-core/src/error.rs` | NotSquare + NotConverged PrimError variants | VERIFIED | Both variants present with thiserror `#[error(...)]` and named fields. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `crates/mlrs-kernels/src/lib.rs` | `jacobi_svd` + `jacobi_eig` | `pub mod` | WIRED | Lines 9-10: `pub mod jacobi_eig; pub mod jacobi_svd;` |
| `crates/mlrs-backend/src/prims/svd.rs` | `mlrs_kernels::jacobi_svd_sweep` | `use` + `jacobi_svd_sweep::launch` | WIRED | Line 49: `use mlrs_kernels::jacobi_svd_sweep;` Line 212: `jacobi_svd_sweep::launch::<F, ActiveRuntime>(...)` |
| `crates/mlrs-backend/src/prims/svd.rs` | `crate::prims::gemm::gemm` | thin-U extraction | WIRED | Line 54: `use crate::prims::gemm::gemm;` Line 258: `let b = gemm::<F>(...)` |
| `crates/mlrs-backend/src/prims/eig.rs` | `mlrs_kernels::jacobi_eig_sweep` | `use` + `jacobi_eig_sweep::launch` | WIRED | Line 42: `use mlrs_kernels::jacobi_eig_sweep;` Line 132: `jacobi_eig_sweep::launch::<F, ActiveRuntime>(...)` |
| `crates/mlrs-backend/src/prims/eig.rs` | covariance/GEMM buffer reuse via `out` param | `D-11 gate 2` | WIRED | Lines 102-105: `out.map(|buf| (buf.handle().clone(), Some(buf)))` threads the buffer as `a_in_handle` without allocating a fresh n² copy. |
| `crates/mlrs-backend/tests/svd_test.rs` | `mlrs_backend::prims::svd::{svd, svd_with_max_sweeps}` | `use` + test calls | WIRED | Line 28: both imported; used in `run_svd`, `svd_orthonormality_invariant`, `svd_not_converged_on_low_sweep_cap`. |
| `crates/mlrs-backend/tests/eig_test.rs` | `mlrs_backend::prims::eig::eig` | `use` + test calls | WIRED | Line 30: `use mlrs_backend::prims::eig::eig;` used in `run_eig`. |
| `crates/mlrs-backend/tests/memory_gate_test.rs` | `mlrs_backend::prims::{svd::svd, eig::eig}` | `use` + gate calls | WIRED | Lines 45-46: both imported; used in all three D-11 gate tests. |
| `crates/mlrs-backend/src/runtime.rs` | `cubecl::hip::{AmdDevice, HipRuntime}` | `cfg(feature = "rocm") pub use` | WIRED | Line 24: `pub use cubecl::hip::{AmdDevice as ActiveDevice, HipRuntime as ActiveRuntime};` |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `svd_test.rs::svd_tall_f32_fixture` | `case` (OracleCase) | `load_npz(fixture("svd_tall_f32_seed42.npz"))` | Yes — committed numpy reference fixture | FLOWING |
| `svd_test.rs::svd_tall_odd_f32_fixture` | `case` (OracleCase) | `load_npz(fixture("svd_tall_odd_f32_seed42.npz"))` | Yes — committed odd-dim fixture (CR-01 gate) | FLOWING |
| `svd_test.rs::svd_not_converged_on_low_sweep_cap` | `res` (Result) | `svd_with_max_sweeps` with cap=1 | Yes — Err(NotConverged) verified | FLOWING |
| `eig_test.rs::eig_symmetric_f32_fixture` | `case` (OracleCase) | `load_npz(fixture("eigh_f32_seed42.npz"))` | Yes — committed numpy reference fixture | FLOWING |
| `memory_gate_test.rs::memory_gate_svd_no_midsweep_readback` | `pool.stats().read_backs` | `BufferPool` counter after `svd()` + `to_host_metered` | Yes — runtime counter, not mocked | FLOWING |
| `memory_gate_test.rs::memory_gate_eig_reuses_gram_buffer` | `eig_peak_rise` (u64) | `pool.stats().peak_bytes` before/after `eig(out=Some)` | Yes — byte accounting, not assertion-only | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| SVD 10/10 tests on CPU (including odd-dim, NotConverged) | `cargo test -p mlrs-backend --features cpu --test svd_test` | 10 passed, 0 failed, 0 ignored; 131.98s | PASS |
| Eig 4/4 tests on CPU (f32+f64) | `cargo test -p mlrs-backend --features cpu --test eig_test` | 4 passed, 0 failed; 1.02s | PASS |
| Memory gate 6/6 on CPU | `cargo test -p mlrs-backend --features cpu --test memory_gate_test` | 6 passed, 0 failed; 2.95s | PASS |
| SVD 10/10 tests on ROCm (f32; f64 SKIPS-with-log) | `cargo test -p mlrs-backend --features rocm --test svd_test` | 10 passed, 0 failed, 0 ignored; 5.16s | PASS |
| Eig 4/4 tests on ROCm (f32; f64 SKIPS-with-log) | `cargo test -p mlrs-backend --features rocm --test eig_test` | 4 passed, 0 failed; 0.88s | PASS |
| Memory gate 6/6 on ROCm | `cargo test -p mlrs-backend --features rocm --test memory_gate_test` | 6 passed, 0 failed; 4.10s | PASS |
| Full mlrs-backend suite on ROCm | `cargo test -p mlrs-backend --features rocm` | All test files green, 0 failures | PASS |

---

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared for this phase. The behavioral spot-checks above serve as the runnable verification gate.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PRIM-05 | 03-01 through 03-05 | SVD / eigendecomposition primitive validated against oracle within tolerance | SATISFIED | 10/10 svd_test + 4/4 eig_test green on cpu+rocm. D-11 memory gate green. NotConverged reachable. Odd-dim coverage. |

REQUIREMENTS.md line 28: `[x] PRIM-05` — complete. Line 112: `PRIM-05 | Phase 3 | Complete`.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/mlrs-backend/tests/svd_test.rs` | 456 | `maxdev < 1e-4` in `assert_identity` (orthonormality check) | Info | Looser than the 1e-5 contract; the tighter 1e-5 relative Frobenius reconstruction invariant is the primary contract check (`svd_reconstruction_invariant` line 408 asserts `rel <= 1e-5`). Not a blocking concern — orthonormality at 1e-4 is documented in the review as the SVD family bound. |
| `.planning/ROADMAP.md` | 164 | Progress table shows `Phase 3 | 2/5 | In progress` — stale metadata | Info | The plan list at lines 95-108 correctly marks all 5 plans `[x]`. The header also marks Phase 3 `[x]` complete. Only the progress table is stale. No code impact. |

No TBD, FIXME, or XXX debt markers found in any Phase 3 source files.

---

### Human Verification Required

None. All observable truths are verified programmatically by running the test suite on the actual cpu and rocm backends with real hardware (gfx1100).

---

## Gaps Summary

No gaps. All 14 must-haves are verified against the actual codebase and test execution.

**CR-01 (odd-dim schedule bug):** Confirmed fixed. The ghost-padding in `jacobi_svd.rs:181-184` (`players = cols+1` when cols is odd) correctly visits all `n(n-1)/2` pairs. The odd-dim fixtures (`svd_tall_odd_f32_seed42.npz`, `svd_tall_odd_f64_seed42.npz`) and the `svd_tall_odd_f32_fixture` / `svd_tall_odd_f64_fixture` / `svd_not_converged_on_low_sweep_cap` tests (using an 8×5 matrix with odd k=5) enforce this as a build-failing regression gate.

**WR-01 (within-sweep convergence mixture):** Confirmed fixed. The kernel header at `jacobi_svd.rs:58-69` documents the clean post-sweep measurement. The rotation block at lines 228-231 explicitly does NOT accumulate γ² during rotation. The dedicated post-sweep pass at lines 270-298 measures the convergence norm from the final matrix state. The `memory_gate_svd_no_midsweep_readback` and `svd_reconstruction_invariant` (relative ≤ 1e-5) gates confirm correctness.

**WR-04 (1e-4 reconstruction tolerance):** Partially addressed. `svd_reconstruction_invariant` uses relative Frobenius ≤ 1e-5, meeting the project contract. The `assert_identity` helper for orthonormality still uses 1e-4, which is acceptable given the basis-invariant reconstruction is the primary gate.

**Lower-severity review items (WR-02, WR-03, IN-01 through IN-04):** Advisory; do not block phase pass per the verification context.

---

_Verified: 2026-06-12_
_Verifier: Claude (gsd-verifier)_
