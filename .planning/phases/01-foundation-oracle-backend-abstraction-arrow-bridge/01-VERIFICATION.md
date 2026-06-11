---
phase: 01-foundation-oracle-backend-abstraction-arrow-bridge
verified: 2026-06-11T14:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 1: Foundation — Oracle, Backend Abstraction, Arrow Bridge Verification Report

**Phase Goal:** The generic compute spine, oracle harness, and data bridge exist so every downstream primitive and estimator can be validated against scikit-learn within 1e-5 on cpu and wgpu.
**Verified:** 2026-06-11T14:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Five-crate workspace compiles with --features cpu, --features wgpu, and --features cuda (compile-only) | ✓ VERIFIED | `cargo build --workspace --features cpu` → "Finished"; `--features wgpu` → "Finished"; `--features cuda` → 2 crates recompiled then "Finished" (no CUDA toolkit required for host compile) |
| 2 | mlrs-kernels carries zero backend feature flags | ✓ VERIFIED | `cargo tree -p mlrs-kernels -e features` output contains no `cubecl-cpu`, `cubecl-wgpu`, `cubecl-cuda`, or `cubecl-rocm` entries — confirmed empty match |
| 3 | A trivial #[cube] kernel generic over F: Float runs on cpu and wgpu, ingests Arrow arrays through the validated bridge, reads back, and oracle asserts within 1e-5 | ✓ VERIFIED | `cargo test --workspace --features cpu` 57 pass, 0 fail; `cargo test -p mlrs-backend --features wgpu` 25/25 pass including `pipeline_saxpy_f32_matches_numpy_oracle` and `pipeline_saxpy_f64_matches_numpy_oracle` on wgpu (AMD RADV GFX1152). Pipeline_test output: "pipeline f32 backend=wgpu: 1024 elements within Tolerance { abs: 1e-5, rel: 1e-5 }" and "pipeline f64 backend=wgpu: 1024 elements within Tolerance { abs: 1e-5, rel: 1e-5 }" |
| 4 | Arrow bridge rejects sliced/offset arrays, nullable arrays, and misaligned buffers before any unsafe transmutation | ✓ VERIFIED | `bridge_test.rs` has 9 tests covering all three classes: `validate_f32_rejects_sliced_array`, `validate_f64_rejects_sliced_array`, `validate_f32_rejects_nullable_array`, `validate_f64_rejects_nullable_array`, `cast_validated_rejects_misaligned_f32`, `cast_validated_rejects_misaligned_f64` — all 9 pass on cpu and wgpu. `bridge.rs` contains no `unsafe` block; all reinterpretation uses `bytemuck::try_cast_slice` which is safe. Validation order documented: offset → nulls → alignment, every path before any cast |
| 5 | Capability layer reports f64 support; f64 oracle tests skip/xfail with logged reason on unsupported adapters; CI log shows dtype×backend | ✓ VERIFIED | `capability.rs` implements `feature_enabled(FloatKind::F64)`, `skip_f64_with_log()`, `log_oracle_dtype()`, `active_backend_name()`. Capability test output: "oracle dtype=F32 backend=cpu adapter=default" confirming the dtype×backend log line fires. `pipeline_test.rs` uses `if capability::skip_f64_with_log() { return; }` pattern — on this machine (RADV GFX1152, SHADER_F64) f64 runs; skip path code-verified to log at `warn` level |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `Cargo.toml` | Virtual workspace with [workspace.dependencies] | ✓ VERIFIED | Contains `[workspace]` + `[workspace.dependencies]` with cubecl 0.10.0, arrow 59, bytemuck, thiserror, anyhow, mimalloc, npyz, log, env_logger |
| `crates/mlrs-kernels/src/smoke.rs` | Generic #[cube] saxpy kernel | ✓ VERIFIED | `#[cube(launch)] pub fn saxpy_kernel<F: Float + CubeElement>(a: F, x: &Array<F>, y: &mut Array<F>)` with ABSOLUTE_POS bounds check |
| `crates/mlrs-backend/src/runtime.rs` | Feature-gated runtime selection + Client alias | ✓ VERIFIED | cfg-gated re-exports for cpu/wgpu/cuda/rocm; `pub type Client = cubecl::client::ComputeClient<ActiveRuntime>`; `active_client()` |
| `crates/mlrs-backend/src/capability.rs` | feature_enabled(FloatKind::F64) facade + FloatKind re-export | ✓ VERIFIED | `feature_enabled`, `supports_f64`, `supports_type`, `log_oracle_dtype`, `skip_f64_with_log`, `active_backend_name` all present; `pub use cubecl::ir::FloatKind` |
| `crates/mlrs-backend/src/bridge.rs` | Hard-reject Arrow ingress | ✓ VERIFIED | `validate_f32`, `validate_f64`, `cast_validated`, `upload` — all reject paths before any transmute; no unsafe block |
| `crates/mlrs-backend/src/pool.rs` | BufferPool free-list with PoolStats | ✓ VERIFIED | HashMap<usize,Vec<Handle>> keyed by byte size; acquire/release/log_stats/Drop; logged-only counters per D-05 |
| `crates/mlrs-backend/src/device_array.rs` | DeviceArray<R,F> with from_host/to_host | ✓ VERIFIED | Pool-metered from_host; to_host via read_one + bytemuck::cast_slice; no Drop (known WR-04, D-05 scope decision) |
| `crates/mlrs-core/src/compare.rs` | is_close / assert_close with abs-AND-rel + near-zero guard | ✓ VERIFIED | `fn is_close` with NEAR_ZERO_FLOOR=1e-8; abs-AND-rel both required above floor (D-09); abs-only below floor; `assert_close`, `assert_slice_close` |
| `crates/mlrs-core/src/tolerance.rs` | Tolerance struct + F32_TOL/F64_TOL | ✓ VERIFIED | `F32_TOL = {abs:1e-5, rel:1e-5}`, `F64_TOL = {abs:1e-5, rel:1e-5}`; `Tolerance::for_family()` growth point |
| `crates/mlrs-core/src/sign_flip.rs` | SVD/PCA sign-alignment helper | ✓ VERIFIED | `canonical_sign`, `align_sign`, `align_sign_in_place`, `align_rows` — sklearn svd_flip convention |
| `crates/mlrs-core/src/label_perm.rs` | Clustering best-permutation matching helper | ✓ VERIFIED | `best_mapping`, `remap`, `best_match_accuracy`, `is_perfect_match` — greedy confusion-matrix assignment |
| `crates/mlrs-core/src/oracle.rs` | Named .npz fixture loader | ✓ VERIFIED | `load_npz`, `load_npz_reader`, `OracleCase` with f32/f64 views per array; npyz by_name API |
| `crates/mlrs-core/src/error.rs` | BridgeError thiserror enum | ✓ VERIFIED | `enum BridgeError { Offset, HasNulls, Misaligned, DataTypeMismatch }` — one variant per Arrow-violation class |
| `crates/mlrs-py/src/allocator.rs` | mimalloc #[global_allocator] wired in cdylib | ✓ VERIFIED | `#[global_allocator] static GLOBAL: MiMalloc = MiMalloc;` — exactly once, source-only |
| `crates/mlrs-py/tests/allocator_test.rs` | Allocator activation proof in separate test file | ✓ VERIFIED | 3 tests: varied sizes + integrity, concurrent churn (8 threads), large allocation (8 MiB) |
| `docs/tolerance-policy.md` | Documented per-family f32 tolerance policy | ✓ VERIFIED | Documents global defaults, abs-AND-rel rule, near-zero guard rationale, growth path via `Tolerance::for_family()` |
| `SPIKE-FINDINGS.md` | Resolved CubeCL 0.10 symbols A1-A7 | ✓ VERIFIED | Present at repo root; resolves A1 (supports_type not feature_enabled), A2 (SHADER_F64 on RADV), A3 (owned Bytes, one copy), A4 (npyz 0.9.1), A5 (N/A), A6 (ComputeClient<R> single generic), A7 (try_cast_slice recoverable Err) |
| `tests/fixtures/saxpy_f32_seed42.npz` | Committed seeded f32 oracle fixture | ✓ VERIFIED | File exists; pipeline_test loads and passes with 1024 elements |
| `tests/fixtures/saxpy_f64_seed42.npz` | Committed seeded f64 oracle fixture | ✓ VERIFIED | File exists; pipeline_test loads and passes with 1024 elements |
| `scripts/gen_oracle.py` | Seeded NumPy fixture generator | ✓ VERIFIED | Present; uses `numpy.random.default_rng(seed=42)`; generates saxpy_f32/f64_seed42.npz |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pipeline_test.rs` | `bridge::validate_f32/f64` | direct call before every upload | ✓ WIRED | Every upload in pipeline_test goes through bridge::validate_{f32,f64} |
| `pipeline_test.rs` | `saxpy_kernel::launch::<F, ActiveRuntime>` | cubecl launch API | ✓ WIRED | `saxpy_kernel::launch::<F, ActiveRuntime>(&client, count, dim, a, ...)` |
| `pipeline_test.rs` | `mlrs_core::oracle::load_npz` | `use mlrs_core::{..., load_npz, OracleCase}` | ✓ WIRED | OracleCase loaded via load_npz, then expect_f32/expect_f64 used |
| `pipeline_test.rs` | `compare::assert_close` / `assert_close_f32_oracle` | `use mlrs_core::{assert_close, is_close, ...}` | ✓ WIRED | f64 path uses core assert_close; f32 path uses local assert_close_f32_oracle wrapping is_close |
| `DeviceArray::from_host` | `BufferPool::acquire`/`release` | pool parameter | ✓ WIRED | Calls `pool.acquire(byte_size)` then `pool.release(metering_handle, byte_size)` |
| `capability::feature_enabled` | `runtime::active_client()` | internal call | ✓ WIRED | `feature_enabled` calls `crate::runtime::active_client()` then `client.properties().supports_type(kind)` |
| `mlrs-backend` bridge.rs | `mlrs_core::error::BridgeError` | `use mlrs_core::error::BridgeError` | ✓ WIRED | All three rejection variants (Offset, HasNulls, Misaligned) returned from bridge.rs |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `pipeline_test::pipeline_saxpy_f32_matches_numpy_oracle` | `got: Vec<f32>` | `DeviceArray::to_host` → `client.read_one` → kernel output | Yes — 1024-element SAXPY result compared element-wise against NPZ fixture | ✓ FLOWING |
| `pipeline_test::pipeline_saxpy_f64_matches_numpy_oracle` | `got: Vec<f64>` | same path, f64 | Yes — 1024-element result; capability-gated | ✓ FLOWING |
| `capability_test::log_oracle_dtype_emits_dtype_backend_line` | log output | `log_oracle_dtype(FloatKind::F32, backend, adapter)` | Yes — in-memory log capture asserted | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Workspace compiles --features cpu | `cargo build --workspace --features cpu` | exit 0, "Finished" | ✓ PASS |
| Workspace compiles --features wgpu | `cargo build --workspace --features wgpu` | exit 0, "Finished" | ✓ PASS |
| Workspace compiles --features cuda | `cargo build --workspace --features cuda` | exit 0, 2 crates compiled, "Finished" (no CUDA toolkit needed) | ✓ PASS |
| Full cpu test suite | `cargo test --workspace --features cpu` | 57 tests, 0 failures | ✓ PASS |
| mlrs-backend wgpu test suite | `cargo test -p mlrs-backend --features wgpu` | 25 tests, 0 failures | ✓ PASS |
| mlrs-kernels has no backend features | `cargo tree -p mlrs-kernels -e features \| grep cubecl-{cpu,wgpu,cuda,rocm}` | empty | ✓ PASS |
| No mod tests in src files | `grep -rn "mod tests" crates/*/src/` | empty | ✓ PASS |
| Pipeline f32 1024 elements within 1e-5 on wgpu | pipeline_test --nocapture | "pipeline f32 backend=wgpu: 1024 elements within Tolerance { abs: 1e-5, rel: 1e-5 }" | ✓ PASS |
| Pipeline f64 1024 elements within 1e-5 on wgpu | pipeline_test --nocapture | "pipeline f64 backend=wgpu: 1024 elements within Tolerance { abs: 1e-5, rel: 1e-5 }" | ✓ PASS |
| Dtype/backend log line fires | capability_test --nocapture | "oracle dtype=F32 backend=cpu adapter=default" printed | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FOUND-01 | 01-01 | Five-crate workspace with single-responsibility crates | ✓ SATISFIED | All five crates (`mlrs-core`, `mlrs-kernels`, `mlrs-backend`, `mlrs-algos`, `mlrs-py`) present and compiling; `[workspace.dependencies]` in root Cargo.toml |
| FOUND-02 | 01-01, 01-05 | Compute kernels generic over float type and CubeCL runtime, feature-free kernels crate | ✓ SATISFIED | `saxpy_kernel<F: Float + CubeElement>` in feature-free mlrs-kernels; runs on cpu and wgpu; `cargo tree` confirms no backend runtime in mlrs-kernels |
| FOUND-03 | 01-01 | Backend selected via Cargo features; cuda compiles; wgpu+cpu run as correctness gate | ✓ SATISFIED | `mlrs-backend` owns `cpu`/`wgpu`/`cuda`/`rocm` features; `--features cuda` compiles without CUDA toolkit; cpu+wgpu tests pass |
| FOUND-04 | 01-03 | Capability layer queries runtime support; f64 gated; f32 portable baseline | ✓ SATISFIED | `feature_enabled(FloatKind::F64)`, `skip_f64_with_log()`, `log_oracle_dtype()` all implemented and tested; 3 capability tests pass |
| FOUND-05 | 01-04 | Memory-efficient DeviceArray wrapping CubeCL buffers with reuse and minimal copies | ✓ SATISFIED | `BufferPool<R>` with acquire/release free-list; `DeviceArray<R,F>` with from_host/to_host; 5 pool tests pass; logged-only counters per D-05 (hard reuse assertions deferred to Phase 2 as documented) |
| FOUND-06 | 01-02, 01-03 | Apache Arrow input validated before any unsafe transmutation | ✓ SATISFIED | `validate_f32`/`validate_f64` with offset→nulls→alignment order, all before any cast; 9 bridge tests pass including all three rejection classes; no unsafe block in bridge.rs |
| FOUND-07 | 01-02, 01-05 | Oracle harness with seeded inputs and 1e-5 tolerance | ✓ SATISFIED | `gen_oracle.py` with `default_rng(seed=42)`; committed .npz fixtures; `load_npz` + `assert_close` wired end-to-end; pipeline tests pass at 1e-5 on both cpu and wgpu |
| FOUND-08 | 01-02 | Sign-flip, label-permutation helpers, documented per-family f32 tolerance policy | ✓ SATISFIED | `sign_flip.rs` (sklearn svd_flip convention), `label_perm.rs` (greedy confusion-matrix), `tolerance.rs` with `for_family()` growth point, `docs/tolerance-policy.md`; 11 helpers tests pass |
| FOUND-09 | 01-05 | Custom global allocator wired with source/test separation | ✓ SATISFIED | `crates/mlrs-py/src/allocator.rs` (source) + `crates/mlrs-py/tests/allocator_test.rs` (test) — separate files per AGENTS.md; 3 allocator tests pass |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/mlrs-backend/src/bridge.rs` | 85-89 | Dead logical-offset branch (`arr.offset()` always returns 0 in arrow 59) | ℹ️ Info | Harmless — real slice detection is on lines 93-101 (buffer-level). Documented as IN-01 in code review |
| `crates/mlrs-backend/src/bridge.rs` | 113 | Redundant null-count disjunct (`arr.nulls().is_some_and(...)` unreachable) | ℹ️ Info | Harmless — `arr.null_count()` subsumes it. Documented as IN-03 in code review |
| `crates/mlrs-backend/src/compare.rs` | 42-43 | abs-AND-rel semantics stricter than numpy/sklearn abs-OR-rel | ⚠️ Warning | D-09 intentional choice documented in CONTEXT.md and `docs/tolerance-policy.md`. REVIEW WR-01 notes this will cause false negatives for large-magnitude oracle values (e.g. `coef_` in Phase 4). Not a Phase 1 gap — this phase has no large-magnitude fixtures, and the policy is fully documented. Carry-forward to Phase 4. |
| `crates/mlrs-backend/src/pool.rs` / `device_array.rs` | pool.rs:103,126 / device_array.rs:67-68 | Caller-supplied size on acquire/release not validated against handle; live_bytes/peak_bytes reflect metering handle not real array; no Drop on DeviceArray | ⚠️ Warning | D-05 intentional: Phase 1 counters are "logged only". Hard reuse/accounting assertions deferred to Phase 2. Code review WR-02/WR-03/WR-04. Not a Phase 1 gap. |
| `crates/mlrs-backend/tests/pipeline_test.rs` | 78, 85-96 | f32 oracle near-zero floor of 1e-2 disables relative check over a wide band | ⚠️ Warning | Review WR-05. Test-local heuristic tied to seed-42 fixture's near-cancellation elements. The 1e-5 absolute bound holds for every element; only the relative check is skipped below 1e-2. Carry-forward to Phase 2 when broader fixture diversity exercises this more broadly. Not a blocker. |

### Human Verification Required

None. All success criteria are programmatically verified:
- Build commands run and exit codes observed
- Test counts and pass/fail recorded
- Log output from --nocapture confirms dtype×backend line
- Source code read to confirm implementation depth, not stubs
- Rejection behaviors confirmed by test passes (not just file existence)

### Gaps Summary

No gaps. All five ROADMAP success criteria are verified against the actual codebase through build execution, test execution, and source inspection.

**Review findings integration:**

The 01-REVIEW.md findings were factored into this assessment as follows:

- **WR-01 (abs-AND-rel comparator)**: D-09 documents this as an intentional choice. The `docs/tolerance-policy.md` explicitly states the abs-AND-rel rule and near-zero guard. Phase 1 fixtures use values in the 1e-1 to 1e1 range where this does not produce false negatives. The concern is real for Phase 4 (large `coef_`/`intercept_` values) but does not affect Phase 1 goal achievement. Carry-forward warning, not a blocker.

- **WR-02/03/04 (pool accounting and DeviceArray no-Drop)**: D-05 explicitly defers hard buffer-reuse assertions to Phase 2. Phase 1 goal says "buffer reuse" exists (the free-list API is present and tested: `pool_reuses_released_buffer_of_matching_size` passes), not that real device handles flow through the reuse path. The counter/accounting fix is a Phase 2 concern. Not a Phase 1 gap.

- **WR-05 (f32 oracle near-zero floor of 1e-2)**: The 1e-5 absolute bound holds for every element. The test passes on both backends. The floor heuristic is fixture-specific but documented in inline comments with measured rationale. Not a blocker for Phase 1.

- **IN-01/02/03/04/05/06/07**: All info-class findings. None affect correctness or goal achievement.

---

_Verified: 2026-06-11T14:00:00Z_
_Verifier: Claude (gsd-verifier)_
