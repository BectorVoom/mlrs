# Phase 1: Foundation — Oracle, Backend Abstraction, Arrow Bridge - Pattern Map

**Mapped:** 2026-06-11
**Files analyzed:** 24 (5 crate manifests + workspace manifest + 17 source/script files + fixtures)
**Analogs found:** 0 in-repo Rust analogs (greenfield) / 24 — all references are external manuals, RESEARCH.md patterns, or cuML behavioral references

> **GREENFIELD — read first.** No Rust workspace exists yet (verified: repo contains only `cuml-main/`, `.planning/`, `AGENTS.md`, `.serena/`). There are **no in-repo Rust files to copy from**. Every "analog" below is one of:
> - **MANUAL** — a CubeCL or optimisor manual section. Code-pattern reference, **NOT verbatim-portable**: the manuals embed `#[cfg(test)] mod tests { … }` in source files, which AGENTS.md strictly prohibits. Copy the *logic*, never the test-module structure.
> - **RESEARCH** — a pattern already written out in `01-RESEARCH.md` (concrete, version-current code). This is the primary, closest reference for most files.
> - **CUML** — `cuml-main/` for *behavioral / API-surface* reference only (sklearn-compatible defaults). **Never code to port** (C++/CUDA → Rust rewrite).
> - **NONE** — no useful reference beyond RESEARCH.md prose; stated explicitly rather than padded.

---

## File Classification

| File to create | Role | Data Flow | Closest Reference | Match Quality |
|----------------|------|-----------|-------------------|---------------|
| `Cargo.toml` (workspace) | config | — | RESEARCH §"Recommended Project Structure" + §"Standard Stack" install block | research-exact |
| `rust-toolchain.toml` | config | — | RESEARCH §Environment (rustc 1.95) | research-note |
| `crates/mlrs-kernels/src/smoke.rs` | kernel | transform (SAXPY) | MANUAL `Cubecl_axpy.md` + `Cubecl_generics.md` + RESEARCH Pattern 1 | research-exact |
| `crates/mlrs-kernels/Cargo.toml` | config | — | RESEARCH Pitfall 5 (feature-free, `default-features=false`) | research-exact |
| `crates/mlrs-backend/src/runtime.rs` | provider | request-response | RESEARCH Pattern 2 + MANUAL (CpuRuntime/WgpuRuntime usage) | research-good (symbols spike) |
| `crates/mlrs-backend/src/capability.rs` | service | request-response | MANUAL `HALF_PRECISION_CUBECL.md` (F16→F64) + RESEARCH Pattern 4 | research-good (symbols spike) |
| `crates/mlrs-backend/src/bridge.rs` | service | transform / file-I/O (Arrow→device) | MANUAL `ZERO_COPY_ARROW_CUBECL.md` + `ARROW_NUMERIC_BRANCHING.md` + RESEARCH Pattern 3 | research-exact |
| `crates/mlrs-backend/src/device_array.rs` | model | CRUD (alloc/read-back) | MANUAL `Backend-Agnostic_Buffer_Slicing…md` + RESEARCH §DeviceArray | research-partial |
| `crates/mlrs-backend/src/pool.rs` | store | CRUD (acquire/release) | RESEARCH Pattern 6 + MANUAL `Tuning_ExclusivePages_Allocator…md` | research-exact |
| `crates/mlrs-backend/Cargo.toml` | config | — | RESEARCH §"Recommended Project Structure" (owns cpu/wgpu/cuda/rocm features) | research-exact |
| `crates/mlrs-core/src/compare.rs` | utility | transform | RESEARCH Pattern 5 (`assert_close` + near-zero guard) | research-exact |
| `crates/mlrs-core/src/tolerance.rs` | config/model | — | RESEARCH Pattern 5 (`Tolerance`, `F32_TOL`/`F64_TOL`) | research-exact |
| `crates/mlrs-core/src/sign_flip.rs` | utility | transform | CUML `svd_flip` behavior (PCA) + RESEARCH §Comparison Helpers | cuml-behavioral |
| `crates/mlrs-core/src/label_perm.rs` | utility | transform | NONE (standard Hungarian/best-permutation; RESEARCH prose only) | none |
| `crates/mlrs-core/src/oracle.rs` | utility | file-I/O (npz load) | RESEARCH §"Don't Hand-Roll" (`ndarray-npy`/`npyz` `by_name`) | research-good (crate spike) |
| `crates/mlrs-core/src/error.rs` | utility | — | RESEARCH Pattern 3 (`BridgeError` thiserror enum) | research-exact |
| `crates/mlrs-core/Cargo.toml` | config | — | RESEARCH §"Standard Stack" + structure | research-exact |
| `crates/mlrs-algos/src/lib.rs` | (skeleton) | — | NONE (empty in Phase 1; estimators are Phase 4+) | none |
| `crates/mlrs-algos/Cargo.toml` | config | — | RESEARCH structure | research-note |
| `crates/mlrs-py/src/lib.rs` | provider | event-driven (allocator) | MANUAL `MIMALLOC_MANUAL.md` + RESEARCH Pattern 7 | research-exact |
| `crates/mlrs-py/Cargo.toml` | config | — | RESEARCH structure (`crate-type=["cdylib"]`, anyhow) | research-good |
| `scripts/gen_oracle.py` | utility (build-time) | file-I/O (write npz) | RESEARCH §"`scripts/gen_oracle.py` shape" + CUML sklearn API | research-exact |
| `tests/fixtures/saxpy_*_seed*.npz` | test fixture | — | RESEARCH §gen_oracle (output of the script) | research-exact |
| `crates/*/tests/*.rs` (7 test files) | test | — | RESEARCH §"Code Examples" (pipeline_test) + §"Wave 0 Gaps" | research-good |

---

## Pattern Assignments

### `crates/mlrs-kernels/src/smoke.rs` (kernel, transform)

**Reference:** MANUAL `Cubecl_axpy.md` + `Cubecl_generics.md`; codified in RESEARCH Pattern 1.
**Caveat:** Reference manuals append `#[cfg(test)] mod tests` to the same file — DO NOT. Tests for this kernel live in `mlrs-backend/tests/` (which owns a runtime feature), since `mlrs-kernels` is feature-free and cannot instantiate a client.

Core pattern (RESEARCH Pattern 1, lines 237-249):
```rust
use cubecl::prelude::*;

#[cube(launch)]
pub fn saxpy_kernel<F: Float>(a: F, x: &Array<F>, y: &mut Array<F>) {
    let tid = ABSOLUTE_POS;
    if tid < x.len() {
        y[tid] = a * x[tid] + y[tid];
    }
}
// Launch generic over <F, R>: kernel params (F) first, then Runtime (R).
```

**Macro rewrite rules to apply (RESEARCH Anti-Patterns / Pitfall 2):**
- No `let v = if c { a } else { b };` — use `let mut v = default; if c { v = a; }`.
- Math as associated fns: `F::exp(x)`, not `x.exp()` (this kernel needs none, but later ones will).
- No `usize` in device code — use `u32`/`i32`/`f32`/`f64`.
- On ANY cubecl build error: STOP, consult `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/` per AGENTS.md before fixing.

---

### `crates/mlrs-kernels/Cargo.toml` (config)

**Reference:** RESEARCH Pitfall 5 (line 442-446).
**Rule (Criterion 1):** zero backend feature flags. Depend on `cubecl` with `default-features = false` and no runtime feature. CI gate: `cargo tree -p mlrs-kernels -e features` must show no `cubecl-wgpu`/`cubecl-cuda`/`cubecl-cpu`/`cubecl-rocm`.

---

### `crates/mlrs-backend/src/runtime.rs` (provider, request-response)

**Reference:** RESEARCH Pattern 2 (lines 254-273).
**Spike-gated (A6):** `ComputeClient` generic signature in 0.10 (`<R>` vs `<Server, Channel>`) is unverified — insulate behind a `pub type Client = …` alias resolved once here.

Core pattern:
```rust
#[cfg(feature = "cpu")]
pub use cubecl::cpu::{CpuRuntime as ActiveRuntime, CpuDevice as ActiveDevice};
#[cfg(feature = "wgpu")]
pub use cubecl::wgpu::{WgpuRuntime as ActiveRuntime, WgpuDevice as ActiveDevice};
#[cfg(feature = "cuda")]
pub use cubecl::cuda::{CudaRuntime as ActiveRuntime, CudaDevice as ActiveDevice};
// rocm analogous. Exactly one backend feature active.

pub fn active_client() -> /* type alias */ {
    let device = ActiveDevice::default();
    ActiveRuntime::client(&device)
}
```

---

### `crates/mlrs-backend/src/capability.rs` (service, request-response)

**Reference:** MANUAL `HALF_PRECISION_CUBECL.md` (F16 capability query, directly transferable to F64); codified in RESEARCH Pattern 4 (lines 313-332).
**Spike-gated (A1/A2):** exact 0.10 symbol path and the f64 feature variant for wgpu SHADER_F64. Two candidate forms shown — expose a thin façade so the call site is stable.

Core pattern:
```rust
use cubecl::ir::FloatKind;
use cubecl::Feature;        // verify path: cubecl::Feature vs cubecl::ir::Feature
use cubecl::ir::Elem;       // verify path

pub fn supports_f64<R: cubecl::Runtime>(client: &ClientFor<R>) -> bool {
    client.properties().feature_enabled(Feature::Type(Elem::Float(FloatKind::F64)))
    // alt form (verify which compiles): client.properties().features.supports_type(FloatKind::F64)
}
```

**Skip/xfail mechanics (Claude's discretion, RESEARCH lines 330-332):** logged early-return skip:
```rust
if !supports_f64(&client) {
    log::warn!("skipping f64 oracle on {backend}: SHADER_F64 unsupported");
    return;
}
// Always log at test start (Criterion 4):
log::info!("oracle dtype=f64 backend=wgpu adapter={adapter}");
```

---

### `crates/mlrs-backend/src/bridge.rs` (service, transform / file-I/O)

**Reference:** MANUAL `ZERO_COPY_ARROW_CUBECL.md` + `ARROW_NUMERIC_BRANCHING.md`; codified in RESEARCH Pattern 3 (lines 277-309).
**Security-critical (V5 input validation):** reject offset/nulls/misalignment BEFORE any `unsafe` transmute (D-06). Use `bytemuck::try_cast_slice` (recoverable Err) not `cast_slice` (panic).

Validation pattern:
```rust
use arrow::array::Float32Array;

pub fn validate_f32(arr: &Float32Array) -> Result<&[f32], BridgeError> {
    if arr.offset() != 0 { return Err(BridgeError::Offset(arr.offset())); }
    if arr.null_count() != 0 || arr.nulls().is_some() {
        return Err(BridgeError::HasNulls(arr.null_count()));
    }
    let slice: &[f32] = arr.values();           // O(1) view into ScalarBuffer
    bytemuck::try_cast_slice::<f32, u8>(slice)   // alignment/size check before transmute
        .map_err(|_| BridgeError::Misaligned { dtype: "f32" })?;
    Ok(slice)
}
```

**Zero-copy nuance (Pitfall 6, A3):** the manuals' `Bytes::from_bytes_vec(slice.to_vec())` copies. Investigate `cubecl::bytes::Bytes` 0.10 constructors for a no-copy path; if none, document the honest semantics (validated single-upload), do not overclaim. Duplicate `validate_f64` for `Float64Array`.
**Negative tests required** (Criterion 3): sliced array (`arr.slice(1, n-1)`), nullable array (`Float64Array::from(vec![Some(1.0), None])`), misaligned buffer → each must return `Err`, not panic.

---

### `crates/mlrs-backend/src/device_array.rs` (model, CRUD)

**Reference:** MANUAL `Backend-Agnostic_Buffer_Slicing_and_Multi-Logical_Array_Allocation.md` (slice/slice_mut, `read_one`, `CubeDim::new_1d`); RESEARCH §DeviceArray (lines 44, 68).
**Match quality: partial** — manuals show `client.create`/`empty`/`read` primitives but not a `DeviceArray<R,F>` wrapper type; the wrapper design is new. Wrap CubeCL handles, carry length + dtype, and route allocation through `pool.rs` (D-04). Host read-back via `client.read_one(handle)` → `bytemuck::cast_slice`.

---

### `crates/mlrs-backend/src/pool.rs` (store, CRUD)

**Reference:** RESEARCH Pattern 6 (lines 355-367); context from MANUAL `Tuning_ExclusivePages_Allocator…md`.
**Phase-1 scope (D-05):** counters logged only, NO hard reuse assertions (deferred to Phase 2).

Design sketch:
```rust
pub struct PoolStats { pub allocations: u64, pub reuses: u64, pub peak_bytes: u64, pub live_bytes: u64 }
pub struct BufferPool<R: cubecl::Runtime> {
    client: /* ComputeClient */,
    free: std::collections::HashMap<usize, Vec<Handle>>,  // size_bytes -> reusable handles
    stats: PoolStats,
}
// acquire(size): pop free[size] (reuses += 1) else client.empty(size) (allocations += 1)
// release(handle, size): push back to free[size]
// drop/phase boundary: log::info!("pool stats: {:?}", stats);   // logged-only (D-05)
```
Simplest correct pool (mlrs-level HashMap free-list) suffices; do NOT tune CubeCL `MemoryConfiguration` in Phase 1 (Open Question 4).

---

### `crates/mlrs-core/src/compare.rs` (utility, transform)

**Reference:** RESEARCH Pattern 5 (lines 336-351). Direct, exact.

```rust
pub fn is_close(got: f64, expected: f64, tol: &Tolerance) -> bool {
    let abs_err = (got - expected).abs();
    if expected.abs() < NEAR_ZERO_FLOOR {        // near-zero guard (D-09 ⚠)
        return abs_err <= tol.abs;               // fall back to abs-only
    }
    let rel_err = abs_err / expected.abs();
    abs_err <= tol.abs && rel_err <= tol.rel     // BOTH must pass (D-09)
}
```
`NEAR_ZERO_FLOOR` value is Claude's discretion (RESEARCH suggests `1e-8`) — document the choice. `assert_close` wraps `is_close` and is the shared comparison entry point that sign-flip / label-perm feed into.

---

### `crates/mlrs-core/src/tolerance.rs` (config/model)

**Reference:** RESEARCH Pattern 5 (lines 337-339).
```rust
pub struct Tolerance { pub abs: f64, pub rel: f64 }
pub const F32_TOL: Tolerance = Tolerance { abs: 1e-5, rel: 1e-5 };
pub const F64_TOL: Tolerance = Tolerance { abs: 1e-5, rel: 1e-5 };
```
**D-08:** single global tolerance now; structure must be *growable* into a per-family table later, but do NOT build the table.

---

### `crates/mlrs-core/src/sign_flip.rs` (utility, transform)

**Reference:** CUML behavioral — `svd_flip` semantics (PCA/SVD sign convention); RESEARCH §Specific Ideas (line 142). FOUND-08.
**No code analog** (cuML is C++/CUDA). The behavior to match: align the sign of each singular vector/component to a deterministic convention (e.g. largest-abs element positive) before comparing to the sklearn oracle, so an equivalent-but-sign-flipped result doesn't spuriously fail `assert_close`. Feeds into `compare::assert_close`.

---

### `crates/mlrs-core/src/label_perm.rs` (utility, transform)

**Reference:** NONE beyond RESEARCH prose (FOUND-08, line 142).
Clustering label assignments are permutation-invariant; map predicted labels to oracle labels via best-permutation matching (e.g. confusion-matrix + Hungarian / greedy) before `assert_close`. No manual or cuML code reference applies — this is standard clustering-evaluation logic.

---

### `crates/mlrs-core/src/oracle.rs` (utility, file-I/O)

**Reference:** RESEARCH §"Don't Hand-Roll" (line 394) + §Standard Stack supporting table.
**Crate choice (Claude's discretion, A4):** `ndarray-npy` `NpzReader::by_name` OR `npyz` `NpzArchive::by_name`. Both read named-array `.npz`; `npyz` avoids the `ndarray` dependency. Load named arrays (`X`, `y`, `coef_`, `intercept_`, or `a`/`x`/`y`/`expected` for saxpy) into a case struct consumed by the pipeline test. Do NOT hand-roll zip+npy-header parsing.

---

### `crates/mlrs-core/src/error.rs` (utility)

**Reference:** RESEARCH Pattern 3 (lines 283-291). `BridgeError` thiserror enum (variant names are Claude's discretion):
```rust
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("array has non-zero offset ({0}); slice/offset not supported (compact first)")]
    Offset(usize),
    #[error("array has {0} null(s); nullable input not supported")]
    HasNulls(usize),
    #[error("buffer is misaligned for {dtype}")]
    Misaligned { dtype: &'static str },
}
```

---

### `crates/mlrs-py/src/lib.rs` (provider, event-driven)

**Reference:** MANUAL `MIMALLOC_MANUAL.md`; RESEARCH Pattern 7 (lines 370-377).
```rust
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
```
**Rules:** `#[global_allocator]` exactly once, only in this cdylib (NOT in any library crate). `crate-type = ["cdylib"]`; use `anyhow` at the boundary (D-10). Activation test goes in `mlrs-py/tests/allocator_test.rs`, NOT a `mod tests`.

---

### `scripts/gen_oracle.py` (utility, build-time, file-I/O)

**Reference:** RESEARCH §"`scripts/gen_oracle.py` shape" (lines 502-515) + CUML sklearn API surface for later estimator fixtures.
```python
import numpy as np
def gen_saxpy(seed=42, n=1024, dtype=np.float32):
    rng = np.random.default_rng(seed)
    a = dtype(2.5)
    x = rng.standard_normal(n).astype(dtype)
    y = rng.standard_normal(n).astype(dtype)
    expected = (a * x + y).astype(dtype)
    np.savez("tests/fixtures/saxpy_f32_seed42.npz", a=a, x=x, y=y, expected=expected)
```
**NOT run in the CI test job (D-03)** — regenerates committed `.npz` blobs on demand. NumPy `default_rng(seed)` is the authoritative seeded RNG (avoid Rust-side RNG, Pitfall 7). Fixture naming encodes case+dtype+seed (`linreg_f64_seed42.npz`). For Phase 4+ estimator fixtures, this script will `import sklearn`, fit, and `savez` `coef_`/`intercept_`/etc.

---

### `tests/fixtures/saxpy_*_seed*.npz` (test fixtures)

**Reference:** output of `gen_oracle.py` above. Committed binary blobs. Ensure `.gitignore` does NOT exclude `tests/fixtures/*.npz` (RESEARCH Runtime State Inventory, line 414).

---

### `crates/*/tests/*.rs` (test files — 7 total)

**Reference:** RESEARCH §"Code Examples" pipeline_test (lines 462-499) + §"Wave 0 Gaps" (lines 609-617).
**Mandatory placement (AGENTS.md):** every test in `crates/<crate>/tests/*.rs` — NEVER `#[cfg(test)] mod tests` in `src/`. CI grep gate: `grep -rn "mod tests" crates/*/src/` must return nothing.

Pipeline integration test pattern (FOUND-02/07, lives in `mlrs-backend/tests/pipeline_test.rs`):
```rust
let client = runtime::active_client();
let case = load_npz("tests/fixtures/saxpy_f32_seed42.npz");
let arr = Float32Array::from(case.x.clone());
let x_slice = bridge::validate_f32(&arr).expect("conforming input");
// upload → launch saxpy_kernel::launch::<f32, runtime::ActiveRuntime>(…) → read_one → assert_close
for (g, e) in got.iter().zip(case.expected.iter()) {
    assert!(is_close(*g as f64, *e as f64, &F32_TOL));
}
```
Test→requirement map and per-file commands are in RESEARCH §"Phase Requirements → Test Map" (lines 591-601).

---

## Shared Patterns

### Error handling (cross-cutting, D-07/D-10)
**Source:** RESEARCH Pattern 3 + MEMORY.md error-handling convention.
**Apply to:** all library crates (`mlrs-core`/`-kernels`/`-backend`/`-algos`) use `thiserror` typed enums; `mlrs-py` + `scripts`-driven Rust use `anyhow` at the boundary. All deps track latest (no pinning to old versions).

### Source/test separation (cross-cutting, AGENTS.md — OVERRIDES manuals)
**Source:** AGENTS.md; RESEARCH Pitfall 1 (line 418).
**Apply to:** every source file. The single highest-frequency mistake risk: all CubeCL/optimisor manuals model `#[cfg(test)] mod tests` in-file — copying that violates the rule. Add CI grep gate.

### CubeCL macro rewrite rules (cross-cutting, applies to every `#[cube]` file)
**Source:** MANUAL `cubecl_error_solution_guide/mismatched types.md`; RESEARCH Anti-Patterns (lines 381-383).
**Apply to:** all kernels (Phase 1: only `smoke.rs`; many in later phases): no `if`-expr-as-value, math via associated fns (`F::exp`), no `usize`/host types in device code. On ANY build error, consult the error-solution guide first (AGENTS.md mandate).

### Feature-flag ownership (cross-cutting, Criterion 1 / FOUND-03)
**Source:** RESEARCH §Architectural Responsibility Map + Pitfall 5.
**Apply to:** Cargo manifests. Backend features (`cpu`/`wgpu`/`cuda`/`rocm`) live ONLY in `mlrs-backend`. `mlrs-kernels` is feature-free. Verify with `cargo tree`.

### Workspace dependency single-source (cross-cutting, FOUND-01 / D-10)
**Source:** RESEARCH §"Recommended Project Structure" + install block (lines 108-119).
**Apply to:** all crate manifests use `{ workspace = true }`; the virtual root `Cargo.toml` `[workspace.dependencies]` is the only place versions are written (latest of each).

### Logging for dtype/backend visibility (cross-cutting, Criterion 4 / D-05)
**Source:** RESEARCH Pattern 4 + Pattern 6.
**Apply to:** capability gate (`dtype=… backend=…` at every oracle test start), pool (counters at drop/phase boundary). CI sets `RUST_LOG=info`; test harness inits a logger.

---

## No Analog Found

Files with no useful reference beyond RESEARCH.md prose (planner uses RESEARCH directly; do not expect a copy-from source):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/mlrs-core/src/label_perm.rs` | utility | transform | Standard clustering best-permutation matching; no manual/cuML code analog; RESEARCH gives only prose |
| `crates/mlrs-algos/src/lib.rs` | skeleton | — | Intentionally empty in Phase 1 (estimators are Phase 4+); nothing to model |
| `crates/mlrs-backend/src/device_array.rs` | model | CRUD | Wrapper *type* is new design; manuals show only the underlying `client` primitives, not a `DeviceArray<R,F>` analog |
| `rust-toolchain.toml` | config | — | Trivial toolchain pin (rustc 1.95); no pattern needed |

---

## Spike-Gated Items (resolve in Wave 0 before full implementation)

These files depend on CubeCL 0.10 symbols that RESEARCH flags as `[ASSUMED]` — the planner should front-load a toolchain spike (compile hello-world kernel + capability query + npz load on cpu and wgpu):

| File | Assumption | Risk |
|------|-----------|------|
| `capability.rs` | A1/A2 — `feature_enabled(Feature::Type(Elem::Float(FloatKind::F64)))` symbol path; wgpu SHADER_F64 mapping | f64 gate won't compile; alt `features.supports_type` form may be needed |
| `bridge.rs` | A3/A7 — `cubecl::bytes::Bytes` no-copy constructor; `try_cast_slice` Err-not-panic | "zero-copy" claim weakens; may need manual alignment check |
| `runtime.rs` | A6 — `ComputeClient<R>` vs `<Server,Channel>` signature | type aliases won't compile; insulate with alias/`impl` |
| `oracle.rs` | A4 — `ndarray-npy`/`npyz` `by_name` for f32 & f64 | swap npz crate (low risk) |

---

## Metadata

**Analog search scope:** entire repo (`/home/user/Documents/workspace/mlrs`) — confirmed no Rust workspace exists; external manual dirs `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` (28 files) and `/home/user/Documents/workspace/optimisor/manual/` (9 files) verified present; `cuml-main/cpp/include/cuml/linear_model/` checked for behavioral reference.
**Files scanned:** 0 in-repo Rust source (none exist); all references external manuals + RESEARCH.md patterns.
**Pattern extraction date:** 2026-06-11
