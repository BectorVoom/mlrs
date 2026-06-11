---
phase: 01-foundation-oracle-backend-abstraction-arrow-bridge
plan: 01
subsystem: infra
tags: [rust, cargo-workspace, cubecl, wgpu, cpu, bytemuck, arrow, npyz, spike]

# Dependency graph
requires: []
provides:
  - "Five-crate virtual Cargo workspace (mlrs-core, mlrs-kernels, mlrs-backend, mlrs-algos, mlrs-py) compiling on cpu/wgpu/cuda"
  - "Generic #[cube(launch)] saxpy_kernel<F: Float + CubeElement> in feature-free mlrs-kernels, runs on cpu + wgpu (f32)"
  - "mlrs_backend::runtime::{ActiveRuntime, ActiveDevice, active_client, Client} feature-gated runtime selection (resolves A6)"
  - "mlrs_backend::capability::{supports_f64, supports_type, feature_enabled, FloatKind} f64 capability facade (resolves A1/A2)"
  - "Compiling module stubs mlrs_backend::{bridge, device_array, pool} for Wave-1 plans to fill"
  - "SPIKE-FINDINGS.md — resolved CubeCL 0.10 / Arrow / npz symbols (A1–A7), the input contract for Plans 02/03/04/05"
affects: [01-02-oracle-harness, 01-03-arrow-bridge, 01-04-device-array-pool, 01-05-pipeline-mimalloc, "Phase 2+ (all kernels)"]

# Tech tracking
tech-stack:
  added: [cubecl 0.10.0, arrow 59.0.0, bytemuck 1.25.0, npyz 0.9.1, thiserror 2.0.18, anyhow 1.0.102, mimalloc 0.1.52, log 0.4, env_logger 0.11]
  patterns:
    - "Virtual workspace with [workspace.dependencies] single-source versioning"
    - "Backend feature ownership lives in mlrs-backend; mlrs-kernels stays backend-feature-free"
    - "Active-runtime cfg re-export (exactly-one-backend-feature contract)"
    - "Symbol-insulation facade (Client alias, capability::FloatKind re-export) so downstream never imports cubecl::ir directly"
    - "Source/test separation: no #[cfg(test)] mod tests in src; integration tests own the backend feature"

key-files:
  created:
    - "Cargo.toml (virtual workspace manifest + [workspace.dependencies])"
    - "rust-toolchain.toml"
    - "crates/mlrs-core/src/lib.rs (+ host-module stubs for Plan 02)"
    - "crates/mlrs-kernels/src/smoke.rs (generic saxpy #[cube] kernel)"
    - "crates/mlrs-kernels/src/lib.rs"
    - "crates/mlrs-backend/src/runtime.rs (ActiveRuntime/Client, A6)"
    - "crates/mlrs-backend/src/capability.rs (supports_f64/feature_enabled, A1/A2)"
    - "crates/mlrs-backend/src/{bridge,device_array,pool}.rs (Wave-1 stubs)"
    - "crates/mlrs-backend/tests/spike_test.rs (5 live spike tests)"
    - "crates/mlrs-algos/src/lib.rs, crates/mlrs-py/src/lib.rs"
    - "SPIKE-FINDINGS.md"
  modified:
    - ".gitignore"

key-decisions:
  - "cubecl 0.10 capability query is client.properties().supports_type(FloatKind::F64) — there is NO feature_enabled/Feature enum (A1 corrected vs RESEARCH guess)"
  - "ComputeClient<R> takes a single generic parameter, not <Server, Channel> (A6)"
  - "cubecl::bytes::Bytes constructors own their allocation — no borrow/no-copy path; bridge guarantees validated single-upload, not literal host zero-copy (A3)"
  - "npz reader = npyz 0.9.1 (chosen over ndarray-npy to avoid the ndarray dep); numpy absent so fixture generated in pure Rust via npyz writer (A4)"
  - "bytemuck::try_cast_slice returns a recoverable Err on alignment/size violation — no panic, no manual ptr%align check needed (A7)"
  - "saxpy_kernel<F> requires F: Float + CubeElement (CubeElement needed for the scalar arg to implement LaunchArg)"

patterns-established:
  - "Spike-gated symbol resolution: Wave-0 resolves unverified upstream API against installed crates and records it in SPIKE-FINDINGS.md before downstream plans depend on it"
  - "Facade insulation: downstream call sites import mlrs_backend facades, never cubecl::ir/client paths directly"

requirements-completed: [FOUND-01, FOUND-02, FOUND-03]

# Metrics
duration: ~2h (Wave-0 spike including checkpoint)
completed: 2026-06-11
---

# Phase 01 Plan 01: Foundation Workspace + CubeCL 0.10 Spike Summary

**Five-crate virtual Cargo workspace compiling on cpu/wgpu/cuda with a generic `#[cube]` saxpy kernel running on cpu+wgpu, plus SPIKE-FINDINGS.md resolving the unverified CubeCL 0.10 / Arrow / npz symbols (A1–A7) for all downstream Wave-1 plans.**

## Performance

- **Duration:** ~2h (Wave-0, includes blocking human-verify checkpoint)
- **Completed:** 2026-06-11
- **Tasks:** 3 implementation tasks + 1 human-verify checkpoint (approved)
- **Files modified:** ~20 (5 crate manifests, source files, SPIKE-FINDINGS.md, .gitignore, rust-toolchain.toml)

## Accomplishments

- Stood up the five-crate virtual workspace (`mlrs-core`, `mlrs-kernels`, `mlrs-backend`, `mlrs-algos`, `mlrs-py`) with `[workspace.dependencies]` as the single version source — satisfies ROADMAP Criterion 1 (FOUND-01).
- Wrote a generic `#[cube(launch)] saxpy_kernel<F: Float + CubeElement>` in the backend-feature-free `mlrs-kernels` crate; it launches and matches a host reference on both cpu and wgpu (f32) — kernel half of Criterion 2 (FOUND-02).
- Implemented feature-gated runtime selection in `mlrs_backend::runtime` (`ActiveRuntime`/`ActiveDevice`/`active_client`/`Client` alias) with an exactly-one-backend-feature contract (FOUND-03).
- Implemented the f64 capability facade in `mlrs_backend::capability` and proved it returns a value on cpu and wgpu.
- Resolved every unverified upstream symbol assumption (A1–A7) against the *installed* crates and recorded them in `SPIKE-FINDINGS.md` — the input contract for Plans 02–05.
- Confirmed via the wgpu adapter (AMD RADV GFX1152, reporting `SHADER_F64`) that **f64-on-wgpu oracle tests will RUN in this environment** — a key input for Plan 05's skip/xfail design.

## Task Commits

1. **Task 1: Scaffold five-crate virtual workspace** - `d80dd29` (feat)
2. **Task 2: Generic saxpy #[cube] kernel + active runtime (resolve A6)** - `17d6f6a` (feat)
3. **Task 3: Capability query + Bytes/npz spike; SPIKE-FINDINGS.md (A1–A7)** - `a5bbbc4` (feat)

**Plan metadata:** this SUMMARY + tracking, committed separately as `docs(01-01): plan summary` / `docs(01-01): update tracking`.

_Task 2 was TDD-style: the integration test in `spike_test.rs` is the live RED→GREEN proof the kernel runs on the active backend._

## Files Created/Modified

- `Cargo.toml` — virtual workspace manifest with `[workspace.dependencies]` single-source versions
- `rust-toolchain.toml` — pins stable toolchain (rustc/cargo 1.95.0)
- `.gitignore` — ignores `target/` but NOT `tests/fixtures/*.npz`
- `crates/mlrs-core/src/lib.rs` — module stubs for the host-side oracle modules Plan 02 fills
- `crates/mlrs-kernels/src/smoke.rs` — generic `saxpy_kernel<F: Float + CubeElement>` `#[cube(launch)]`
- `crates/mlrs-kernels/src/lib.rs` — re-exports `smoke`
- `crates/mlrs-backend/src/runtime.rs` — `ActiveRuntime`/`ActiveDevice`/`active_client`/`Client` (A6)
- `crates/mlrs-backend/src/capability.rs` — `supports_f64`/`supports_type`/`feature_enabled` + `FloatKind` re-export (A1/A2)
- `crates/mlrs-backend/src/{bridge,device_array,pool}.rs` — compiling `//! TODO(plan NN)` stubs for Wave 1
- `crates/mlrs-backend/tests/spike_test.rs` — 5 integration tests (saxpy, f64 capability, Bytes probe, npz round-trip, try_cast_slice recoverability)
- `crates/mlrs-algos/src/lib.rs`, `crates/mlrs-py/src/lib.rs` — minimal compiling skeletons
- `SPIKE-FINDINGS.md` — resolved symbols A1–A7

## Spike Resolutions (A1–A7)

- **A1 — f64 capability query:** RESEARCH's `feature_enabled(Feature::Type(...))` does NOT exist in cubecl 0.10. The real form is `client.properties().supports_type(FloatKind::F64)` (`FloatKind` from `cubecl::ir`). Exposed via the `mlrs_backend::capability` facade.
- **A2 — wgpu f64 / `SHADER_F64`:** `supports_type(FloatKind::F64)` returns `true` on the local wgpu adapter (AMD Radeon RADV GFX1152, Vulkan/Mesa 25.2.8), whose feature list includes `SHADER_F64`. The capability query is the correct gate; no need to query raw wgpu adapter features. **f64-on-wgpu works here.**
- **A3 — `cubecl::bytes::Bytes`:** Both 0.10 constructors (`from_elems`, `from_bytes_vec`) own their allocation; there is no borrow/no-copy path. Honest semantics: the bridge (Plan 03) guarantees **validated single-upload**, not literal host zero-copy.
- **A4 — npz reader:** `npyz` 0.9.1 (chosen over `ndarray-npy`). `numpy` is absent, so the round-trip fixture was generated in pure Rust via npyz's own writer. `NpzArchive::by_name(...).into_vec::<f64>()` is the loader API Plan 02 uses.
- **A5 — seeded RNG:** N/A in Wave 0. No Rust-side RNG; oracle inputs come from Python `numpy.random.default_rng(seed)` (Plan 02). `rand` not yet a dependency.
- **A6 — `ComputeClient` signature:** Single generic parameter `ComputeClient<R: Runtime>` — NOT the `<Server, Channel>` form. Insulated once via `pub type Client = ComputeClient<ActiveRuntime>`. Also resolved: scalar kernel arg passed by value, `ArrayArg::from_raw_parts(handle, length)` (2 args, handle by value), read-back via `client.read_one(handle)`.
- **A7 — `bytemuck::try_cast_slice`:** Returns a **recoverable `Err(PodCastError)`** on alignment/size violations (does not panic), so the Arrow bridge can map `Err` → `BridgeError::Misaligned` before any `unsafe` transmute.

## Decisions Made

See `key-decisions` frontmatter and the spike resolutions above. Headline: capability query is `supports_type` (not `feature_enabled`), `ComputeClient` is single-generic, `Bytes` owns its allocation (validated single-upload, not literal zero-copy), npz reader is `npyz`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added `CubeElement` bound to `saxpy_kernel<F>`**
- **Found during:** Task 2
- **Issue:** `<F: Float>` alone does not let the scalar kernel argument implement `LaunchArg`, so `launch::<f32, R>` failed to compile.
- **Fix:** Tightened the bound to `<F: Float + CubeElement>` (matches the axpy / half-precision manuals).
- **Files modified:** `crates/mlrs-kernels/src/smoke.rs`
- **Verification:** `cargo test -p mlrs-backend --features cpu saxpy` and `--features wgpu saxpy` pass.

**2. [Rule 3 - Blocking] Chose `npyz` over `ndarray-npy` and generated the npz fixture in pure Rust**
- **Found during:** Task 3
- **Issue:** `numpy` is not installed in this environment, so the planned `python3 -c "np.savez(...)"` fixture generation was impossible.
- **Fix:** Selected `npyz` (no `ndarray` dependency) and generated the round-trip fixture with npyz's own writer in Rust, giving a full f32+f64 `by_name` round-trip without numpy.
- **Files modified:** `crates/mlrs-backend/tests/spike_test.rs`, root `Cargo.toml` (`npyz` dep with `features = ["npz"]`)
- **Verification:** `cargo test -p mlrs-backend --features cpu spike` — npz round-trip test passes.

**3. [Rule 3 - Blocking] Added `rlib` crate-type to `mlrs-py`**
- **Found during:** Task 1
- **Issue:** A `cdylib`-only crate cannot be depended on or exercised by other crates / integration paths during workspace builds.
- **Fix:** Added `rlib` alongside `cdylib` in `crates/mlrs-py/Cargo.toml`.
- **Files modified:** `crates/mlrs-py/Cargo.toml`
- **Verification:** `cargo build --workspace --features cpu` exits 0.

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking). **Impact:** all necessary for correctness/build; no scope creep. The `npyz`-vs-numpy substitution is recorded as a tentative crate choice with a formal package-legitimacy gate still owed in Plan 02 (threat T-01-SC).

## Issues Encountered

- The dominant Wave-0 risk (CubeCL 0.10 API fragility) materialized exactly as anticipated: several RESEARCH assumptions (A1 capability query, A6 client signature, A3 Bytes copy semantics) were *wrong* vs. the installed crate. Resolving them against the real crates and recording the corrections in SPIKE-FINDINGS.md is precisely the purpose this Wave-0 plan served — downstream plans now build on verified symbols.

## User Setup Required

None — no external service configuration required. (CUDA toolkit is absent; `--features cuda` is compile-only by contract, which is satisfied.)

## Next Phase Readiness

- **Wave 1 released** (checkpoint approved). Plans 02 (oracle harness), 03 (Arrow bridge + f64 gate), and 04 (DeviceArray + pool) can begin; each reads SPIKE-FINDINGS.md for resolved symbols and edits only its own stubbed module.
- **Carry-forward for Plan 02:** formal `npyz` package-legitimacy checkpoint (threat T-01-SC) before the oracle loader's first real use.
- **Carry-forward for Plan 03:** use "validated single-upload" wording in `bridge.rs`, not literal zero-copy (A3); map `try_cast_slice` `Err` → `BridgeError::Misaligned` (A7).
- **Carry-forward for Plan 05:** f64-on-wgpu RUNS here (`SHADER_F64` present); the skip/xfail path is still required for adapters lacking it but is not exercised in this environment — always log `dtype=… backend=…`.

## Self-Check: PASSED

Verified on disk (orchestrator independently re-verified the same green results):

- Commits present: `d80dd29`, `17d6f6a`, `a5bbbc4` (all `git log --grep="01-01"`).
- `SPIKE-FINDINGS.md` present at repo root and resolves A1–A7.
- All crate source files present (`Cargo.toml`, `rust-toolchain.toml`, all five `crates/*/src/lib.rs`, `smoke.rs`, `runtime.rs`, `capability.rs`, `spike_test.rs`).
- `cargo build --workspace --features cpu` and `--features wgpu` exit 0; `--features cuda` compiles (host-side).
- `cargo test -p mlrs-backend --features cpu spike` = 5/5 pass (`f32_supported=true`, `f64_supported=true`).
- `cargo tree -p mlrs-kernels -e features` shows only cubecl (no `cubecl-{cpu,wgpu,cuda,rocm}`) — kernels crate is backend-feature-free.
- `grep -rn "mod tests" crates/*/src/` returns nothing (AGENTS.md source/test separation).

---
*Phase: 01-foundation-oracle-backend-abstraction-arrow-bridge*
*Completed: 2026-06-11*
