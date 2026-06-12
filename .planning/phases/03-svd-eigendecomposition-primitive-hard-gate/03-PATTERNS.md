# Phase 3: SVD / Eigendecomposition Primitive (Hard Gate) - Pattern Map

**Mapped:** 2026-06-12
**Files analyzed:** 11 new/modified (2 kernels, 2 prims, 2 tests, 1 test-extend, 1 runtime, 1 Cargo.toml, 1 error, 1 oracle script, +fixtures)
**Analogs found:** 11 / 11 (every new file has a strong in-repo analog; nothing is greenfield)

This phase is almost entirely **new kernel logic layered on already-validated primitives** (RESEARCH §"Don't Hand-Roll"). The only genuinely novel device code is the two Jacobi sweep kernels; every surrounding concern (launch idiom, geometry validation, pool metering, oracle harness, fixtures, sign-alignment, capability gating) has a direct Phase-1/2 analog to copy verbatim.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-kernels/src/jacobi_svd.rs` (NEW) | kernel | iterative / shared-mem transform | `crates/mlrs-kernels/src/reduce.rs` (`reduce_sumsq_shared`) | role-match (shared-mem `#[cube]`; no existing iterative kernel) |
| `crates/mlrs-kernels/src/jacobi_eig.rs` (NEW) | kernel | iterative / shared-mem transform | `crates/mlrs-kernels/src/reduce.rs` (`reduce_sumsq_shared`) | role-match |
| `crates/mlrs-backend/src/prims/svd.rs` (NEW) | prim / launch wrapper | request-response orchestration | `crates/mlrs-backend/src/prims/covariance.rs` | exact (multi-step host orchestration over pooled scratch) |
| `crates/mlrs-backend/src/prims/eig.rs` (NEW) | prim / launch wrapper | request-response orchestration | `crates/mlrs-backend/src/prims/gemm.rs` + `covariance.rs` | exact |
| `crates/mlrs-backend/src/prims/mod.rs` (MODIFY) | module index | — | itself (existing `pub mod` list) | exact |
| `crates/mlrs-kernels/src/lib.rs` (MODIFY) | module index | — | itself (existing `pub mod` + `pub use`) | exact |
| `crates/mlrs-backend/src/runtime.rs` (MODIFY) | config / runtime facade | — | the existing `#[cfg]` re-export block (cpu/wgpu/cuda lines) | exact (fix the rocm line) |
| `crates/mlrs-backend/Cargo.toml` (MODIFY) | build config | — | the existing `[features]` table | exact (extend the `rocm` line) |
| `crates/mlrs-core/src/error.rs` (MODIFY) | model / error enum | — | the existing `PrimError` enum | exact (add variants) |
| `crates/mlrs-backend/tests/svd_test.rs` (NEW) | test | oracle + invariant | `crates/mlrs-backend/tests/gemm_test.rs` | exact |
| `crates/mlrs-backend/tests/eig_test.rs` (NEW) | test | oracle + invariant | `crates/mlrs-backend/tests/gemm_test.rs` | exact |
| `crates/mlrs-backend/tests/memory_gate_test.rs` (MODIFY) | test | memory gate | the 3 existing Phase-2 gates in that file | exact (add 3 assertions) |
| `scripts/gen_oracle.py` (MODIFY) | fixture generator | batch / file-I/O | the existing `gen_gemm` / `gen_cov` functions | exact |
| `crates/mlrs-core/examples/gen_fixture.rs` (MODIFY) | fixture generator | file-I/O | the existing `write_arrays` | role-match (npz writer; svd/eigh content new) |

**Read-only reuse (no edit):** `crates/mlrs-core/src/sign_flip.rs`, `crates/mlrs-backend/src/{device_array.rs, pool.rs, capability.rs}`, `crates/mlrs-backend/src/prims/{gemm.rs, reduce.rs}`.

---

## Pattern Assignments

### `crates/mlrs-kernels/src/jacobi_svd.rs` + `jacobi_eig.rs` (kernel, iterative shared-mem)

**Analog:** `crates/mlrs-kernels/src/reduce.rs` (`reduce_sumsq_shared`, lines 161-186) for the shared-mem tree + `sync_cube` convergence reduction; `crates/mlrs-kernels/src/elementwise.rs` (`scale`, `center_columns`, lines 64-93) for the scalar-arg launch signature.

**Imports pattern** — every kernel file is one line (reduce.rs:39, elementwise.rs:29):
```rust
use cubecl::prelude::*;
```

**`#[cube(launch)]` skeleton + module doc** — copy the reduce.rs header convention (reduce.rs:1-38): a `//!` doc block stating the kernel is generic over `<F: Float + CubeElement>`, carries NO backend feature (D-13), and that tests live in `crates/mlrs-backend/tests/` (AGENTS.md §2 — never an in-source `mod tests`).

**Shared-memory tile + `sync_cube` + log₂ tree** — the off-diagonal-norm convergence reduction inside the sweep loop copies `reduce_sumsq_shared` (reduce.rs:161-186) verbatim in structure:
```rust
#[cube(launch)]
pub fn reduce_sumsq_shared<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let mut shared = SharedMemory::<F>::new(256usize);   // COMPILE-TIME size; runtime-bounded
    let tid = UNIT_POS_X;
    let gid = ABSOLUTE_POS_X;
    let v = if (gid as usize) < input.len() { input[gid as usize] } else { F::from_int(0i64) };
    shared[tid as usize] = v * v;
    sync_cube();
    let mut s = CUBE_DIM_X / 2u32;
    while s > 0u32 {
        if tid < s { let val = shared[(tid + s) as usize]; shared[tid as usize] += val; }
        sync_cube();
        s /= 2u32;
    }
    if tid == 0u32 { output[CUBE_POS_X as usize] = shared[0usize]; }
}
```
Carry forward exactly: `SharedMemory::<F>::new(256usize)` is a **compile-time** size capped at the max supported `n`; the active region is bounded by a runtime `n` via `if (gid as usize) < input.len()` (RESEARCH Pattern 6 + reduce.rs:166). Size the Jacobi tile to a comptime cap and guard the active region with a runtime `n` — **do not** size shared memory from a runtime value.

**Generic constants + Float methods inside the rotation** — copy the elementwise idiom (elementwise.rs:43, 65, 120): `F::from_int(0i64)`, `F::new(2.0)`, `F::sqrt(x)`, and the `Float` bound's `.abs()` / `.sqrt()`. RESEARCH §"Code Examples" gives the rotation:
```rust
let two  = F::from_int(2);
let one  = F::from_int(1);
let zeta = (beta - alpha) / (two * gamma);
let c    = one / (one + t * t).sqrt();
let s    = c * t;
```

**Conditionals as STATEMENTS, never expressions** (elementwise.rs:14-17, 42-47): the `max(d,0)` clamp is written `let zero = F::from_int(0i64); if d < zero { d = zero; }`, NOT an `if`-expression or `max()` call (the CubeCL conditionals manual warns `if`-expressions can mis-lower). The Jacobi "skip below-threshold pair" must likewise be `if |gamma| > thresh { ...rotate... }` — **`continue` is NOT supported in `#[cube]`** (RESEARCH Pattern 6 / Anti-Patterns); wrap the skip in an `if`.

**Scalar-by-value launch arg** (elementwise.rs:65, 82-87): a scalar (e.g. `n: u32`, sweep cap, threshold `F`) is passed by value, exactly like `scale(.., factor: F)` and `center_columns(.., cols: u32)`. The `CubeElement` bound on `F` is mandatory for an `F`-by-value arg to implement `LaunchArg` (elementwise.rs:18-23).

**Do NOT hardcode the plane width / 32** (reduce.rs:46-58, RESEARCH Anti-Patterns): the single-cube design should use the `SharedMemory` tree (not a plane path) for the off-diagonal norm to avoid plane-width portability concerns (carried Phase-2 D-03). If a plane path is used anywhere, use `PLANE_DIM`, never `32`.

---

### `crates/mlrs-backend/src/prims/svd.rs` (prim, request-response orchestration)

**Analog:** `crates/mlrs-backend/src/prims/covariance.rs` (multi-step host orchestration: validate → pool-acquire scratch → launch → release scratch → compose with GEMM → return device-resident). Secondary: `gemm.rs` for the launch-wrapper signature shape and `validate_geometry`.

**Imports pattern** (covariance.rs:42-52):
```rust
use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::{/* jacobi_svd_sweep, etc. */};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::gemm::gemm;                // A·V, reconstruction, residual (D-02/Pattern 3)
use crate::runtime::ActiveRuntime;
```

**Host-API signature** — copy the `gemm`/`covariance` shape (gemm.rs:54-66, covariance.rs:89-98): `pool: &mut BufferPool<ActiveRuntime>` first, the input `&DeviceArray<ActiveRuntime, F>` + explicit `(rows, cols)` tuple, an `out: Option<DeviceArray<...>>` (D-11 caller-out), returning `Result<DeviceArray<...>, PrimError>`, bound `where F: Float + CubeElement + Pod`:
```rust
pub fn svd<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    (rows, cols): (usize, usize),
    out: Option<DeviceArray<ActiveRuntime, F>>,   // optional caller buffer (D-11)
) -> Result<(/* U */, /* S */, /* Vt */), PrimError>
where
    F: Float + CubeElement + Pod,
```

**Geometry validation BEFORE any unsafe launch** (gemm.rs:67-68, 132-174; covariance.rs:99-101, 212-262): a private `validate_geometry(a_len, (rows,cols), out_len) -> Result<(), PrimError>` called first thing; returns `PrimError::ShapeMismatch` on `rows*cols != a.len()` using `checked_mul(...).map(...).unwrap_or(true)`. For the SVD wide path (D-05), this is where the `rows < cols` Aᵀ-and-swap dispatch lives (pure host label-swap, reuse the GEMM transpose flag — covariance.rs:151-172 shows `transa=true` driving AᵀA with no transpose buffer).

**Launch idiom: `ArrayArg::from_raw_parts` from validated lengths** (covariance.rs:123-139) — the load-bearing `unsafe` pattern with the SAFETY comment:
```rust
let client = pool.client().clone();
let (count, dim) = launch_dims_1d(a_len);
// SAFETY: lengths are the carried/validated element counts; the kernel
// bounds-checks tid < input.len() (mitigates the OOB-read threat, ASVS V5).
let a_arg = unsafe { ArrayArg::from_raw_parts(a.handle().clone(), a_len) };
let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };
jacobi_svd_sweep::launch::<F, ActiveRuntime>(&client, count, dim, a_arg, out_arg, /*scalars*/);
```
Lengths are ALWAYS derived from validated `DeviceArray::len()` / carried element counts, NEVER from raw caller geometry (covariance.rs:126, ASVS V5 / T-04-01).

**Pool-acquired scratch + caller-out reuse** (gemm.rs:95-100; covariance.rs:119-121): output buffer is `match &out { Some(o) => o.handle().clone(), None => pool.acquire(out_bytes) }`; transient scratch is `pool.acquire(bytes)` then **`scratch_dev.release_into(pool)`** at its true byte size once its consuming kernel is launched (covariance.rs:149, 180; device_array.rs:162-165). This is exactly the D-11 bounded-scratch discipline — the sweep loop must reuse a fixed set of pool-drawn buffers, not allocate per-sweep.

**Device-resident return** (gemm.rs:124-127; covariance.rs:202-204): wrap the result handle with `DeviceArray::from_raw(handle, len)` and return it WITHOUT any host read-back inside the API (D-05) — the convergence loop is in-kernel; only the test reads back.

**`launch_dims_1d` helper** (covariance.rs:264-273): copy verbatim for the post-step element-wise passes (thin-U normalize, sort permute):
```rust
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = ((n as u32) + block - 1) / block;
    (CubeCount::Static(cubes.max(1), 1, 1), CubeDim { x: block, y: 1, z: 1 })
}
```

**Thin-U / S extraction reuses Phase-2 GEMM + reduce** (RESEARCH Pattern 3, "Don't Hand-Roll"): `S[j] = ‖(A·V)[:,j]‖₂` via the Phase-2 column L2-norm reduction; `U[:,j] = (A·V)/S[j]` via `scale`/element-wise. Guard `S[j] ≈ 0` against a near-zero floor (Pitfall 4) — do not divide by zero on rank-deficient columns.

---

### `crates/mlrs-backend/src/prims/eig.rs` (prim, request-response orchestration)

**Analog:** identical to `svd.rs` above (covariance.rs / gemm.rs idiom). Eig-specific differences:

- **Squareness validation** — extend `validate_geometry` to reject non-square (D-06): emit the new `PrimError::NotSquare` variant before any launch (ASVS V5, RESEARCH Security §V5). Trusts symmetry — NO `(A+Aᵀ)/2` step (D-06).
- **Covariance is the only feeder + buffer-reuse target** (covariance.rs:19-29, D-11 gate 2): eig should reuse the covariance/GEMM output buffer rather than allocate a parallel matrix — mirror covariance's own "Gram reuses the GEMM buffer" reuse (covariance.rs:151-204) by threading `out` straight through.
- **Descending sort** (RESEARCH Pattern 5, D-04): eigenvalues sorted **descending** (NB `np.linalg.eigh` is ascending — the test reverses the numpy reference). A host-side selection sort of the converged diagonal after read-back is acceptable (the D-11 device-resident gate concerns the *convergence loop*, not the final O(n) sort — RESEARCH A4).

---

### `crates/mlrs-backend/src/runtime.rs` (config, MODIFY — the rocm fix)

**Analog:** the existing cpu/wgpu/cuda `#[cfg]` re-export lines (runtime.rs:10-20). **Current line 20 is WRONG** (`cubecl::rocm::{RocmDevice, RocmRuntime}` — no such module). Replace with the verified path (RESEARCH Pattern 1, verified by a saxpy run on gfx1100):
```rust
#[cfg(feature = "rocm")]
pub use cubecl::hip::{AmdDevice as ActiveDevice, HipRuntime as ActiveRuntime};
```
Everything else (`Client` alias runtime.rs:30, `active_client()` runtime.rs:37-42) already handles rocm via the existing `any(...)` cfg — no other change.

---

### `crates/mlrs-backend/Cargo.toml` (build config, MODIFY — the rocm feature)

**Analog:** the existing `[features]` table (Cargo.toml:35-39). **Current `rocm = ["cubecl/rocm"]` fails to compile** `cubecl-hip` (unresolved `cubecl_runtime::stream::MultiStream`, Pitfall 2). Extend to add the `std`/`default` propagation (RESEARCH Pattern 1, verified):
```toml
rocm = ["cubecl/rocm", "cubecl/std", "cubecl/default"]
```
Do NOT weaken `default-features = false` on the workspace `cubecl` or on `cubek-matmul`/`cubek-std` elsewhere (Cargo.toml:10, 20, 24; RESEARCH Security §V12) — only the `rocm` feature line changes.

---

### `crates/mlrs-core/src/error.rs` (error enum, MODIFY)

**Analog:** the existing `PrimError` enum (error.rs:65-100) with `thiserror`. Add new variants following the same form — `#[error("...")]` with named fields and per-field doc comments (error.rs:70-83):
```rust
/// The eig primitive requires a square matrix; the caller supplied rows != cols.
#[error("primitive '{operand}' must be square: rows({rows}) != cols({cols})")]
NotSquare { operand: &'static str, rows: usize, cols: usize },

/// The iterative Jacobi sweep did not reach the off-diagonal threshold within
/// the max-sweep cap (D-12 internal constants).
#[error("primitive '{operand}' did not converge within {max_sweeps} sweeps (off-diagonal norm {residual:e})")]
NotConverged { operand: &'static str, max_sweeps: u32, residual: f64 },
```
Variant naming is Claude's discretion (CONTEXT D-13 / RESEARCH §Discretion); keep "one variant per violation class" (error.rs:64). `thiserror` in libs is the carried convention (CONTEXT D-13).

---

### `crates/mlrs-backend/tests/svd_test.rs` + `eig_test.rs` (test, oracle + invariant)

**Analog:** `crates/mlrs-backend/tests/gemm_test.rs` (the full oracle harness).

**Imports + module doc** (gemm_test.rs:1-29): a `//!` header naming each test + stating tests live in `tests/` per AGENTS.md §2, then:
```rust
use std::path::PathBuf;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::svd::svd;          // (eig::eig for eig_test)
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{assert_slice_close, is_close, load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};
use mlrs_core::sign_flip::align_rows;        // D-03: sign-align at comparison ONLY
```

**Fixture-path resolver** (gemm_test.rs:77-84) — copy verbatim (resolves `<workspace>/tests/fixtures/<name>`):
```rust
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent().and_then(|p| p.parent()).expect("workspace root");
    workspace_root.join("tests").join("fixtures").join(name)
}
```

**Per-case device runner** (gemm_test.rs:120-149) — generic over `F`, builds a pool, uploads via `DeviceArray::from_host`, calls the prim, reads back via `to_host_metered`:
```rust
let client = runtime::active_client();
let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
let a_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, a_host);
let (u, s, vt) = svd::<F>(&mut pool, &a_dev, (rows, cols), None).expect("svd valid shape");
let s_host = s.to_host_metered(&mut pool);   // terminal metered read-back
```

**f64 capability split — copy the gemm skip pattern EXACTLY** (gemm_test.rs:175-194, RESEARCH Pitfall 1): f32 tests always run; f64 tests gate on `capability::skip_f64_with_log()` and `return` early. On rocm, f64 SKIPS-with-log (correct, not a bug — CubeCL HIP backend has F64 unregistered); f64 coverage comes from the **cpu** backend:
```rust
let backend = capability::active_backend_name();
capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
if capability::skip_f64_with_log() {
    println!("svd f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
    return;
}
```

**npz fixture compare** (gemm_test.rs:228-261): `load_npz(fixture("svd_tall_f32_seed42.npz"))` → `case.expect_f32("U"/"S"/"Vt")`, compare after `align_rows` sign-alignment (D-03). Reuse the `assert_slice_close` / `is_close` / `Tolerance` core API; for f32 large-magnitude / near-cancellation results, copy the per-test near-zero-floor helper `assert_slice_close_f32_gemm` (gemm_test.rs:41-74) which keeps the strict `1e-5` abs bound but raises the abs-only fallback floor (D-10 — do NOT pre-loosen; per-family looser bound only if a real case forces it).

**Reference-free invariants are NEW in-test Rust** (RESEARCH §Validation Architecture, D-09) — no analog beyond the host-ref helper shape (gemm_test.rs:91-115 `host_gemm_ref`): reconstruction `‖U·diag(S)·Vᵀ − A‖`, orthonormality `‖UᵀU − I‖` / `‖VᵀV − I‖`, eig residual `‖A·v − λ·v‖`. These are basis-invariant and are the primary check for the degenerate/clustered/rank-deficient D-08 cases (Pitfall 3/4) where per-vector fixture compare is ill-conditioned. Build them with the Phase-2 `gemm` for the matrix products.

---

### `crates/mlrs-backend/tests/memory_gate_test.rs` (test, MODIFY — 3 D-11 assertions)

**Analog:** the 3 existing Phase-2 gates in the same file (memory_gate_test.rs:80-433). Copy their exact assertion forms:

**Gate "bounded Jacobi scratch"** — mirror `memory_gate_reuse_bounded` (lines 80-190): thread ONE `BufferPool`, run the sweep, snapshot `pool.stats()` per iteration into `Vec`s, assert `live_after[iter] == live_baseline` (scratch released, not stacked) and `peak_after[iter] == peak_baseline` (bounded). The D-11 twist: allocation count must NOT grow with **sweep/iteration** count — assert the per-call allocation delta is flat after warmup.

**Gate "eig reuses covariance/GEMM buffer"** — copy `memory_gate_gram_reuses_gemm_buffer` (lines 299-433) verbatim in structure, including the `count_gram_sized_fresh_allocs` free-list probe (lines 409-433): run a GEMM producing the `n_features²` buffer, pass it as eig's `out`, assert eig allocated NO fresh buffer of that byte size (`served_as_reuse == false`).

**Gate "no host round-trip between sweeps"** — copy `memory_gate_no_midpipeline_readback` (lines 207-285): assert `pool.stats().read_backs == 0` after the device-resident sweep, then exactly `== 1` after the single terminal `to_host_metered`. The convergence loop is in-kernel (single cube), so no `to_host_metered` happens between sweeps.

The counter API is `pool.stats()` → `PoolStats { allocations, reuses, peak_bytes, live_bytes, read_backs }` (pool.rs:35-52); `read_backs` is bumped ONLY by `to_host_metered` (device_array.rs:133-136), plain `to_host` does not. These gates are HARD/build-failing and must NOT be weakened to pass (memory_gate_test.rs:28-29).

---

### `scripts/gen_oracle.py` + `crates/mlrs-core/examples/gen_fixture.rs` (fixture generators, MODIFY)

**Analog:** the existing `gen_gemm` / `gen_cov` functions in `gen_oracle.py` (lines 84+) and the shape constants block (lines 47-59). Add `gen_svd` / `gen_eigh` following the same shape:
- shape constants (e.g. `SVD_TALL = (8, 4)`, `SVD_WIDE = (4, 8)`, `EIG_N = 4`) alongside `GEMM_M/K/N` (lines 47-59);
- `rng = np.random.default_rng(seed)` seeded RNG (line 67) — authoritative, no Rust-side RNG (Pitfall 7);
- `U, S, Vt = np.linalg.svd(A, full_matrices=False)` (D-02/D-09); `w, V = np.linalg.eigh(Asym)` (reverse for descending, D-04);
- `np.savez(out_path, A=A, U=U, S=S, Vt=Vt)` named arrays (line 80) under `case_dtype_seed` naming (e.g. `svd_tall_f32_seed42.npz`, `eigh_f32_seed42.npz`);
- write into `_FIXTURE_DIR = <repo>/tests/fixtures` (lines 35-36).

Fixtures are committed blobs — CI never runs the script (gen_oracle.py:5-8); regen needs a `/tmp` venv with numpy (PEP 668, per project memory). The `gen_fixture.rs` Rust npz-writer analog (`write_arrays`, gen_fixture.rs:26-65) is only needed if a fixture must be produced without numpy; the SVD/eig fixtures should go through `gen_oracle.py` (numpy is the reference per D-09).

---

## Shared Patterns

### ROCm bring-up (FIRST TASK — D-07)
**Source:** RESEARCH Pattern 1 (verified by `spike_saxpy_runs_on_active_backend` on gfx1100).
**Apply to:** `runtime.rs` (line 20 → `cubecl::hip::{AmdDevice as ActiveDevice, HipRuntime as ActiveRuntime}`) + `Cargo.toml` (`rocm = ["cubecl/rocm", "cubecl/std", "cubecl/default"]`).
**Smoke-test analog:** `crates/mlrs-backend/tests/spike_test.rs` `spike_saxpy_runs_on_active_backend` (lines 31-79) — `runtime::active_client()` → `client.create(Bytes::from_elems(..))` → `saxpy_kernel::launch::<f32, ActiveRuntime>(..)` → `client.read_one(..)`. This test already exists and passes on rocm after the two fixes; it is the bring-up acceptance gate before any SVD work.

### Geometry validation → typed PrimError (ASVS V5)
**Source:** `gemm.rs::validate_geometry` (lines 132-174), `covariance.rs::validate_geometry` (lines 212-262).
**Apply to:** `svd.rs`, `eig.rs` — validate `(rows,cols)` against `DeviceArray::len()` (and squareness for eig) BEFORE any `unsafe` launch; return `PrimError::{ShapeMismatch, NotSquare}`. `DeviceArray::len()` is the single source of truth for read-back size (device_array.rs:43-48, T-04-01).

### Launch wrapper `unsafe { ArrayArg::from_raw_parts }`
**Source:** `covariance.rs` (lines 123-139, 198-200) with the SAFETY comment.
**Apply to:** both Jacobi launch wrappers — lengths from validated `DeviceArray::len()`, kernels bounds-check `tid < input.len()`.

### Pool-metered scratch + caller-out reuse (D-11)
**Source:** `BufferPool` (pool.rs:107-133), `DeviceArray::release_into` / `to_host_metered` / `byte_size` (device_array.rs:133-165).
**Apply to:** `svd.rs`, `eig.rs` — `pool.acquire(bytes)` for scratch, `release_into(pool)` once consumed (at true `byte_size`), `out: Option<..>` for the caller buffer, `to_host_metered` only at the terminal read in tests.

### Sign-alignment at comparison ONLY (D-03)
**Source:** `mlrs-core/src/sign_flip.rs` (`align_rows` lines 60-62, `align_sign` 40-43, `canonical_sign` 21-36).
**Apply to:** `svd_test.rs`, `eig_test.rs` — `align_rows(&components)` before fixture compare. NO device-side flip kernel; the kernel returns raw-sign output.

### f64 capability gate (D-07 / Pitfall 1)
**Source:** `capability::skip_f64_with_log` (capability.rs:146-154), `log_oracle_dtype` (132-134), `active_backend_name` (106-124).
**Apply to:** every f64 oracle test — gate-and-`return`. f64 runs on **cpu**, SKIPS on **rocm** (CubeCL HIP F64 unregistered — correct, not a bug).

### Source/test separation (AGENTS.md §2)
**Apply to:** ALL new source files — NO in-source `#[cfg(test)] mod tests`. Kernel/prim source files carry only a `//!` doc note pointing at the `tests/` file (reduce.rs:36-37, covariance.rs:40, elementwise.rs:25-27).

---

## No Analog Found

| File / Concern | Role | Reason |
|------|------|--------|
| Reference-free algebraic invariants (reconstruction / orthonormality / eig-residual) | test logic | NEW in-test Rust (D-09); closest shape is `host_gemm_ref` (gemm_test.rs:91-115) for the matrix-product helpers, but the invariant assertions themselves are new — no existing test computes `‖UΣVᵀ−A‖`. |
| Iterative in-kernel sweep loop (round-robin rotation schedule, in-kernel convergence break) | kernel control flow | NO existing iterative `#[cube]` kernel — all Phase-1/2 kernels are single-pass. The shared-mem + `sync_cube` machinery is borrowed from `reduce.rs`, but the multi-sweep `while` loop with an in-kernel convergence test (D-11 gate 3) has no analog. RESEARCH Pattern 2/5/6 is the design source. |

These two are the genuinely novel surfaces; the planner should reference RESEARCH §Architecture Patterns (Patterns 2-6) and §Validation Architecture directly for them, while copying every surrounding concern from the analogs above.

## Metadata

**Analog search scope:** `crates/mlrs-kernels/src/`, `crates/mlrs-backend/src/{prims,}/`, `crates/mlrs-backend/tests/`, `crates/mlrs-core/src/`, `crates/mlrs-core/examples/`, `scripts/`.
**Files scanned:** reduce.rs, elementwise.rs, lib.rs (kernels), gemm.rs, covariance.rs, runtime.rs, pool.rs, device_array.rs, capability.rs, error.rs, sign_flip.rs, gemm_test.rs, memory_gate_test.rs, spike_test.rs, gen_oracle.py, gen_fixture.rs, prims/mod.rs, Cargo.toml.
**Pattern extraction date:** 2026-06-12

## PATTERN MAPPING COMPLETE
