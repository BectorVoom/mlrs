---
phase: 08-kernel-family
verified: 2026-06-21T10:30:00Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
human_verification_result: passed (user approved 2026-06-21 — Python FFI smoke test 4/4 from 08-05 execution accepted; see 08-UAT.md)
human_verification:
  - test: "Run Python smoke test: cd <repo> && maturin develop --features cpu -m crates/mlrs-py/Cargo.toml && pytest crates/mlrs-py/tests/test_kernel.py -v"
    expected: "4 tests pass (test_kernel_ridge_predict[f32], test_kernel_ridge_predict[f64], test_kernel_density_score_samples[f32], test_kernel_density_score_samples[f64]); all assertions pass with the correct shapes and log-density finiteness"
    why_human: "The Python smoke test requires maturin to build the extension and a live Python + pyarrow/numpy/sklearn environment. Cannot run without those dependencies; the SUMMARY reports 4/4 green but the verifier cannot re-run it in this environment."
---

# Phase 8: Kernel-Family Verification Report

**Phase Goal:** A data scientist can fit kernel-based regression and density estimators built on a new keystone kernel-matrix primitive (linear/RBF/poly/sigmoid) that Phase 9 and future kernel-SVM reuse; introduces the `ScoreSamples` trait.
**Verified:** 2026-06-21T10:30:00Z
**Status:** human_needed (all Rust oracle tests pass; Python FFI smoke test needs human verification)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `prims/kernel_matrix.rs` validated standalone against host reference within tolerance for f32 and f64, with PoolStats memory gate; SharedMemory-free, no atomics, large n×n in global memory | ✓ VERIFIED | `cargo test --features cpu -p mlrs-backend --test kernel_matrix_test` → 3 passed (all_kernels_f32, all_kernels_f64, memory_gate). Grep gates: zero `F::INFINITY`, zero `SharedMemory` in elementwise.rs. Dispatches `distance::` (RBF) and `gemm::` (linear/poly/sigmoid) then launches `*_map`. |
| 2 | A user can fit `KernelRidge` (4 kernels: linear/rbf/polynomial/sigmoid, gamma/degree/coef0) and predict, matching sklearn within 1e-5 | ✓ VERIFIED | `cargo test --features cpu -p mlrs-algos --test kernel_ridge_test` → 4 passed (all_kernels_f64, all_kernels_f32, multi_target_f64, multi_target_f32). Observed max_abs f64 = 5.55e-16 (well within 1e-5), f32 = 3.58e-7. No x_mean/y_mean/intercept_ present. |
| 3 | A user can fit `KernelDensity` (kernels + bandwidth) and call `score_samples` for log-density using numerically-stable log-sum-exp, matching sklearn within tolerance | ✓ VERIFIED | `cargo test --features cpu -p mlrs-algos --test kernel_density_test` → 5 passed (all_kernels_f64, all_kernels_f32, bandwidth_rules_f64, bandwidth_rules_f32, score_samples_shape_f32). Observed f64 max_abs ≤ 1.6e-8; f32 ≤ 1e-4. Composes `distance` + `row_reduce` directly (D-08), never `kernel_matrix`. |
| 4 | `ScoreSamples<F>` trait added next to existing traits; `KernelDensity` implements it (length-n log-densities, NOT Predict semantics) | ✓ VERIFIED | `pub trait ScoreSamples<F>` at `crates/mlrs-algos/src/traits.rs:231`. Re-exported from `lib.rs` line 59. `impl<F> ScoreSamples<F> for KernelDensity<F>` at `density/kernel_density.rs:252`. Returns `DeviceArray<ActiveRuntime, F>` of length-n_samples log-densities (distinct from Predict). |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-algos/src/traits.rs` | `ScoreSamples<F>` trait | ✓ VERIFIED | `pub trait ScoreSamples<F>` at line 231; one method `score_samples` returning length-n `DeviceArray` |
| `crates/mlrs-algos/src/error.rs` | `InvalidBandwidth`, `InvalidDegree`, `InvalidKernel` + (fix) `InvalidGamma` | ✓ VERIFIED | All four variants present (lines 225, 240, 275; InvalidGamma added by CR-01 fix commit 56f7ecc) |
| `crates/mlrs-algos/src/lib.rs` | `pub mod kernel_ridge`, `pub mod density`, re-export `ScoreSamples` | ✓ VERIFIED | Lines 45, 47 (pub mod), line 59 (re-export) |
| `crates/mlrs-backend/src/prims/kernel_matrix.rs` | `Kernel<F>` enum + `kernel_matrix` host fn (full compute body) | ✓ VERIFIED | `pub enum Kernel<F>` at line 64 with Linear/Rbf/Poly/Sigmoid variants; `pub fn kernel_matrix<F>` at line 121; full dispatch body present (no todo!() in code, only in stale doc comments, which is the IN-01 finding) |
| `crates/mlrs-backend/src/prims/mod.rs` | `pub mod kernel_matrix` | ✓ VERIFIED | Line 24 |
| `crates/mlrs-kernels/src/elementwise.rs` | `rbf_map`, `poly_map`, `sigmoid_map`, 6 KD density maps | ✓ VERIFIED | `rbf_map` line 115, `poly_map` line 131, `sigmoid_map` line 153; `kde_gaussian_map` line 175, `kde_epanechnikov_map` line 195, `kde_tophat_map` line 217, `kde_exponential_map` line 239, `kde_linear_map` line 254, `kde_cosine_map` line 277 |
| `crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs` | `KernelRidge` struct + fit + predict | ✓ VERIFIED | `pub struct KernelRidge<F>` at line 98; `pub fn fit` at line 173; `pub fn predict` at line 366; calls `kernel_matrix` + `cholesky_solve` |
| `crates/mlrs-algos/src/density/kernel_density.rs` | `KernelDensity` struct + fit + `ScoreSamples` impl | ✓ VERIFIED | `pub struct KernelDensity<F>` at line 128; `impl<F> ScoreSamples<F> for KernelDensity<F>` at line 252; composes `distance` + `row_reduce`, never `kernel_matrix` |
| `crates/mlrs-py/src/estimators/kernel.rs` | `PyKernelRidge` + `PyKernelDensity` via `any_estimator!` | ✓ VERIFIED | Two `crate::any_estimator!` invocations at lines 100 and 297; `score_samples_f32/_f64` methods present; `guard_f64()` on F64 arms; `py.detach` GIL release |
| `crates/mlrs-py/src/lib.rs` | `add_class::<PyKernelRidge>` + `add_class::<PyKernelDensity>` | ✓ VERIFIED | Lines 171-172 |
| `crates/mlrs-py/tests/test_kernel.py` | Python smoke test: fit/predict/score_samples, f32/f64 | ✓ VERIFIED (existence + content) | File exists; contains `score_samples`, `float32`/`float64` dispatch; 4 parametrized cases. SUMMARY claims 4/4 green but human must re-confirm. |
| `tests/fixtures/kernel_matrix_*.npz` (f32 + f64) | Oracle fixtures | ✓ VERIFIED | `kernel_matrix_f32_seed42.npz`, `kernel_matrix_f64_seed42.npz` present |
| `tests/fixtures/kernel_ridge_*.npz` (f32 + f64) | Oracle fixtures | ✓ VERIFIED | `kernel_ridge_f32_seed42.npz`, `kernel_ridge_f64_seed42.npz` present |
| `tests/fixtures/kernel_density_*.npz` (f32 + f64) | Oracle fixtures | ✓ VERIFIED | `kernel_density_f32_seed42.npz`, `kernel_density_f64_seed42.npz` present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `kernel_matrix.rs` | `prims::distance` / `prims::gemm` | base-op dispatch on `Kernel` enum | ✓ WIRED | `use crate::prims::distance::distance` (line 50), `use crate::prims::gemm::gemm` (line 51); dispatched at lines 160, 165, 174, 185 |
| `kernel_matrix.rs` | `mlrs_kernels::{rbf_map, poly_map, sigmoid_map}` | in-place map launch over base buffer | ✓ WIRED | `use mlrs_kernels::{poly_map, rbf_map, sigmoid_map}` (line 46); `rbf_map::launch` at line 167, `poly_map::launch` at line 176, `sigmoid_map::launch` at line 187 |
| `kernel_ridge.rs` | `mlrs_backend::prims::kernel_matrix` | K = kernel_matrix(X,X) at fit; K_test = kernel_matrix(X_test,X_fit_) at predict | ✓ WIRED | `use mlrs_backend::prims::kernel_matrix::{kernel_matrix, Kernel}` (line 57); called at `fit` line 269 and `predict` line 415 |
| `kernel_ridge.rs` | `mlrs_backend::prims::cholesky::cholesky_solve` | (K+αI) multi-RHS dual solve | ✓ WIRED | `use mlrs_backend::prims::cholesky::cholesky_solve` (line 55); called at line 304 |
| `kernel_density.rs` | `mlrs_backend::prims::distance` | D = distance(Q, X_fit_, sqrt=per-kernel) | ✓ WIRED | `use mlrs_backend::prims::distance::distance` (line 57); called at line 305 |
| `kernel_density.rs` | `mlrs_backend::prims::reduce::row_reduce` | per-query row log-sum-exp | ✓ WIRED | `use mlrs_backend::prims::reduce::{row_reduce, ReducePath, ScalarOp}` (line 58); called at line 330 |
| `lib.rs` (mlrs-py) | `PyKernelRidge` / `PyKernelDensity` | `m.add_class` registration | ✓ WIRED | `use estimators::kernel::{PyKernelDensity, PyKernelRidge}` + `m.add_class::<PyKernelRidge>()` and `m.add_class::<PyKernelDensity>()` at lines 171-172 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `kernel_ridge.rs` | `dual_coef_` | `cholesky_solve` over `kernel_matrix(X, X, kernel)` | Yes — full dual solve with DB query (device computation) | ✓ FLOWING |
| `kernel_density.rs` | log-density (score_samples) | `distance(Q, X_fit_)` → KD map → `row_reduce(Sum)` → host `log + log_norm − log(N)` | Yes — real device pipeline | ✓ FLOWING |
| `kernel_matrix.rs` | K(X,Y) output | `distance` or `gemm` base op → in-place `*_map` | Yes — real device computation | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| kernel_matrix 4-kernel f32+f64 value tests | `cargo test --features cpu -p mlrs-backend --test kernel_matrix_test` | 3 passed in 3.13s | ✓ PASS |
| kernel_matrix PoolStats memory gate | included in above run | kernel_matrix_memory_gate PASSED | ✓ PASS |
| KernelRidge 4-kernel + multi-target oracle | `cargo test --features cpu -p mlrs-algos --test kernel_ridge_test` | 4 passed in 2.85s | ✓ PASS |
| KernelDensity 6-kernel + scott/silverman + shape | `cargo test --features cpu -p mlrs-algos --test kernel_density_test` | 5 passed in 3.69s | ✓ PASS |
| Python FFI fit/predict/score_samples smoke test | `maturin develop + pytest test_kernel.py` | Cannot run (no maturin/Python env) | ? SKIP |

### Probe Execution

No probe scripts defined for this phase (`scripts/tests/probe-*.sh` not applicable). Step 7c: SKIPPED.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PRIM-08 | 08-02 | Kernel-matrix primitive (linear/RBF/poly/sigmoid) validated against host reference, f32+f64 | ✓ SATISFIED | `kernel_matrix_all_kernels_f32`, `kernel_matrix_all_kernels_f64`, `kernel_matrix_memory_gate` all pass. SharedMemory-free, atomics-free. |
| KERNEL-01 | 08-03 | KernelRidge fit/predict matching sklearn within 1e-5 | ✓ SATISFIED | `kernel_ridge_all_kernels_f64` (5.55e-16), `kernel_ridge_all_kernels_f32`, `kernel_ridge_multi_target_f64/f32` all pass. 4 kernels, multi-target, gamma=None and explicit. |
| KERNEL-02 | 08-04 | KernelDensity score_samples matching sklearn within tolerance | ✓ SATISFIED | `kernel_density_all_kernels_f64/f32`, `kernel_density_bandwidth_rules_f64/f32`, `score_samples_shape_f32` all pass. 6 kernels, forced-exact fixtures, scott/silverman. |

**Orphaned requirements check:** REQUIREMENTS.md maps PRIM-08, KERNEL-01, KERNEL-02 to Phase 8. All three appear in PLAN frontmatter. No orphans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `kernel_matrix.rs` | 35, 117 | Stale "Wave-0 stub / todo!()" in DOC COMMENTS ONLY — code is fully implemented (IN-01) | ℹ️ Info | Misleads readers; no functional impact |
| `kernel_matrix_test.rs` | 3, 169, 185, 199 | Stale "#[ignore] scaffold" framing in test module doc (IN-02) | ℹ️ Info | Tests are live and passing; documentation is outdated |
| `kernel.rs` (mlrs-py) | 54 | `"polynomial"` alias in `parse_kernel_kind` — undocumented, untested (IN-03) | ℹ️ Info | Harmless alias inconsistency |
| `kernel.rs` (mlrs-py) | 452-471 | `log_density_f32/_f64` are aliases of `score_samples_*`, misleading as "cheap accessors" (IN-04) | ℹ️ Info | Caller confusion risk; no correctness impact |
| `kernel_density.rs` | 339 | `.expect("shared path is never plane-gated to None")` — brittle panic surface on future change to reduce prim (WR-01, deferred) | ⚠️ Warning | Acceptable today (Shared path never returns None per current reduce.rs); deliberately deferred |
| `elementwise.rs` | 302 | `div_by_row` exported but never called — dead code (WR-04, deferred) | ⚠️ Warning | Dead API; no functional impact; kept for potential future use |
| `kernel.rs` (mlrs-py) | 266-280 | `dual_coef_f32/_f64` accessors do `to_host` without `py.detach` GIL release (WR-06, deferred) | ⚠️ Warning | Blocks GIL during device→host copy; deliberate deferral; small fixed-size read for typical n |

**Debt-marker gate:** Zero `TBD`, `FIXME`, or `XXX` markers found in any file modified by this phase. Gate: CLEAR.

**Review blocker status (CR-01):** The code review blocker (unvalidated gamma + silent NaN poly predictions) was fixed in commit `56f7ecc`. The fix adds `AlgoError::InvalidGamma` and rejects non-finite resolved gamma before launch, plus a post-solve finiteness check on `dual_coef_` that surfaces `PrimError::NotPositiveDefinite` instead of storing NaN duals. Verified present in `kernel_ridge.rs` lines 246-250 and 306-320.

**WR-02 fix (u32 cast guard):** `u32::try_from((n + block - 1) / block).expect(...)` present in both `kernel_density.rs:418` and `kernel_matrix.rs:238`.

**WR-03 fix (non-finite bandwidth):** `if !(bandwidth > 0.0 && bandwidth.is_finite())` at `kernel_density.rs:227`.

**WR-05 fix (NaN-loud KernelRidge test):** Non-finite fail-loud check added to `kernel_ridge_test.rs` (`70af5b9`).

**WR-07 fix (re-fit buffer release):** `if let Some(old) = self.x_fit_.take() { old.release_into(pool); }` and `if let Some(old) = self.dual_coef_.take() { old.release_into(pool); }` present in both estimators.

### Human Verification Required

#### 1. Python FFI Smoke Test

**Test:** Build the cpu extension (`maturin develop --features cpu` targeting the `mlrs-py` crate) into a Python environment with numpy/pyarrow/scikit-learn/pytest, then run `pytest crates/mlrs-py/tests/test_kernel.py -v`.

**Expected:** 4 tests pass — `test_kernel_ridge_predict[f32]`, `test_kernel_ridge_predict[f64]`, `test_kernel_density_score_samples[f32]`, `test_kernel_density_score_samples[f64]`. Predictions match sklearn smoke band; `score_samples(Q)` returns length-`nq` vector of finite log-densities; dtype dispatch (f32/f64) works; f64 path uses `backend_supports_f64()` skip guard.

**Why human:** Requires a live maturin build environment and Python runtime with numpy/pyarrow/scikit-learn/pytest installed. The verifier environment cannot run maturin. The SUMMARY reports 4/4 passing (section "Verification Evidence"), and the test file content + wiring are verified by code inspection, but the actual maturin+pytest execution cannot be confirmed without the build tool.

### Gaps Summary

No gaps found. All four success criteria are verified in the codebase:

1. `kernel_matrix` is fully implemented (not a stub), dispatches the correct base ops and maps, passes 4-kernel f32+f64 value tests plus PoolStats memory gate.
2. `KernelRidge` fit/predict achieves ≤ 5.55e-16 (f64) across all 4 kernels, multi-target, and both gamma paths — well within 1e-5.
3. `KernelDensity.score_samples` covers 6 kernels, scott/silverman bandwidths, forced-exact fixtures (f64 ≤ 1.6e-8).
4. `ScoreSamples<F>` trait is in `traits.rs`, re-exported from `lib.rs`, and implemented by `KernelDensity<F>` returning length-n log-densities (not Predict semantics).

The code review blocker (CR-01) is fixed. Warnings WR-01/WR-04/WR-06 are confirmed intentionally deferred (the problem statement documents this explicitly). Info items (IN-01 through IN-04) are documentation/naming quality issues with no correctness impact.

The only remaining item requiring human action is re-running the Python FFI smoke test in an environment with maturin and a Python runtime — an environment constraint, not a code gap.

---

_Verified: 2026-06-21T10:30:00Z_
_Verifier: Claude (gsd-verifier)_
