# Phase 2: Core Compute Primitives - Pattern Map

**Mapped:** 2026-06-12
**Files analyzed:** 14 (3 kernel src, 5 backend-src, 1 backend-src modify, 5 integration tests, 1 fixture-gen path)
**Analogs found:** 14 / 14 (all in-tree from Phase 1)

> Every new file in this phase has a strong in-tree analog. The ONLY genuinely
> new pattern is the GEMM *algorithm body* in `gemm.rs`, which is gated by the
> `cubecl-matmul` version blocker (RESEARCH Open Question 1) — see "No Analog
> Found". Everything else copies a Phase-1 idiom directly.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-kernels/src/gemm.rs` | kernel | transform (tiled matmul) | `mlrs-kernels/src/smoke.rs` | role-match (idiom only; algo is new) |
| `crates/mlrs-kernels/src/reduce.rs` | kernel | transform (reduction) | `mlrs-kernels/src/smoke.rs` | role-match |
| `crates/mlrs-kernels/src/elementwise.rs` | kernel | transform (map) | `mlrs-kernels/src/smoke.rs` | exact (per-element, like saxpy) |
| `crates/mlrs-kernels/src/lib.rs` (modify) | config/barrel | — | `mlrs-kernels/src/lib.rs` (self) | exact |
| `crates/mlrs-backend/src/prims/mod.rs` | config/barrel | — | `mlrs-backend/src/lib.rs` | exact |
| `crates/mlrs-backend/src/prims/gemm.rs` | service (host orchestration) | transform | `mlrs-backend/tests/spike_test.rs` (launch idiom) + `device_array.rs` (pool-routed alloc) | role-match |
| `crates/mlrs-backend/src/prims/reduce.rs` | service | transform | spike launch idiom + `device_array.rs` | role-match |
| `crates/mlrs-backend/src/prims/distance.rs` | service (composition) | transform (multi-stage device-resident) | spike launch idiom + `device_array.rs` | role-match |
| `crates/mlrs-backend/src/prims/covariance.rs` | service (composition over gemm) | transform | spike launch idiom + `device_array.rs` | role-match |
| `crates/mlrs-backend/src/pool.rs` (modify) | model (counter state) | event-driven (counter bump) | `mlrs-backend/src/pool.rs` (self — `PoolStats`/`acquire`) | exact |
| `crates/mlrs-backend/tests/gemm_test.rs` | test | request-response (oracle) | `mlrs-backend/tests/pipeline_test.rs` + `spike_test.rs` | exact |
| `crates/mlrs-backend/tests/reduce_test.rs` | test | request-response | `pipeline_test.rs` + `spike_test.rs` | exact |
| `crates/mlrs-backend/tests/distance_test.rs` | test | request-response | `pipeline_test.rs` | exact |
| `crates/mlrs-backend/tests/covariance_test.rs` | test | request-response | `pipeline_test.rs` | exact |
| `crates/mlrs-backend/tests/memory_gate_test.rs` | test | request-response (counter assert) | `mlrs-backend/tests/pool_test.rs` | exact |
| `tests/fixtures/*.npz` + `scripts/gen_oracle.py` (extend) | fixture/migration | file-I/O | `scripts/gen_oracle.py` + `mlrs-core/examples/gen_fixture.rs` | exact |

---

## Pattern Assignments

### `crates/mlrs-kernels/src/elementwise.rs` (kernel, per-element map)

**Analog:** `crates/mlrs-kernels/src/smoke.rs` (EXACT — `clamp_nonneg`/`sqrt_elem` are the
same `ABSOLUTE_POS`-indexed bounds-checked per-element shape as saxpy).

**Imports + kernel pattern** (`smoke.rs:13-30`):
```rust
use cubecl::prelude::*;

#[cube(launch)]
pub fn saxpy_kernel<F: Float + CubeElement>(a: F, x: &Array<F>, y: &mut Array<F>) {
    let tid = ABSOLUTE_POS;
    if tid < x.len() {
        y[tid] = a * x[tid] + y[tid];
    }
}
```
Copy literally for `clamp_nonneg` / `sqrt_elem`. The `<F: Float + CubeElement>`
bound is MANDATORY (D-13): `CubeElement` is required because any scalar `F` arg
must implement `LaunchArg` for the generated `launch` fn (smoke.rs:20-23). For
the clamp use the **statement form** (RESEARCH Code Examples / `Cubecl_conditionals.md`):
`let zero = F::from_int(0i64); if d < zero { d = zero; }` — NOT an expression.

**RESEARCH dist-combine clamp** to implement here (RESEARCH §Code Examples, lines 395-411):
`dist_combine_clamp<F>(xy, xnorm, ynorm, out, rows, cols)` with `F::new(2.0)` and `F::from_int(0i64)`.

---

### `crates/mlrs-kernels/src/reduce.rs` (kernel, reduction — dual path)

**Analog:** `crates/mlrs-kernels/src/smoke.rs` for the `#[cube(launch)]` generic shell;
the *bodies* come from RESEARCH §"Dual-Path Reduction Mechanics" (lines 313-352, CITED
from `Cubecl_plane.md` / `Cubecl_shared_memory.md`).

**Two SEPARATE kernel functions** (RESEARCH recommends this over a `#[comptime]` branch so
each path is a distinct named launch the test can exercise — line 315):
- Path A `reduce_*_plane<F>`: `PLANE_DIM`-folded `plane_shuffle_xor` — **no hardcoded 32**
  (D-03). RESEARCH lines 318-330.
- Path B `reduce_*_shared<F>`: `SharedMemory::<F>::new(256usize)` + `sync_cube()` log₂ tree
  (pairwise-stable). RESEARCH lines 333-348.

**Constants/idioms locked by Phase 1 + RESEARCH:** zero-init `F::from_int(0i64)`; index
`as usize`; `PLANE_DIM`/`UNIT_POS_PLANE`/`CUBE_DIM_X`/`CUBE_POS_X`. argmin carries
`(value, index)` with **lowest-index tie-break in every combine** (D-02, RESEARCH Pitfall 4
lines 295-299).

---

### `crates/mlrs-kernels/src/gemm.rs` (kernel, tiled matmul) — SEE "No Analog Found"

**Analog (idiom only):** `smoke.rs` for the `#[cube(launch)] pub fn k<F: Float + CubeElement>`
shell. The tiled-GEMM *algorithm* has no in-tree analog (Phase 1 has no multi-stage
shared-memory kernel). RESEARCH Pattern 3 (lines 224-251) gives the hand-written sketch with
`#[comptime] trans_a/trans_b` index-swap (`A[i*K+k]` vs `A[k*M+i]`) — NO transpose buffer (D-06).
**Whether this file exists at all is gated by the GEMM-substrate DECISION task** (Open Question 1):
if a cubecl-0.10-compatible `cubecl-matmul` is pinned, GEMM is a backend-side wrapper and this
file is omitted; the recommended default is to hand-write it here.

---

### `crates/mlrs-kernels/src/lib.rs` (modify, barrel) + `prims/mod.rs`

**Analog:** the existing `mlrs-kernels/src/lib.rs` (EXACT):
```rust
pub mod smoke;
pub use smoke::saxpy_kernel;
```
Add `pub mod gemm; pub mod reduce; pub mod elementwise;` and re-export the new kernels the
same way. Keep the crate-level doc note: **MUST NOT depend on any backend runtime feature**
(lib.rs:1-6, D-13). `prims/mod.rs` mirrors `mlrs-backend/src/lib.rs` (a plain `pub mod X;` list).

---

### `crates/mlrs-backend/src/prims/{gemm,reduce,distance,covariance}.rs` (service, host orchestration)

**Analog:** the launch idiom in `tests/spike_test.rs:60-75` (the launch call) + the
pool-routed allocation/read-back pattern in `src/device_array.rs` (the device-resident I/O).
These are NEW host-API modules but every mechanic they need is one of these two analogs.

**Launch-config helper** (copy from `spike_test.rs:21-28`, also `pipeline_test.rs:109-116`):
```rust
fn launch_dims(n: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = ((n as u32) + block - 1) / block;
    (CubeCount::Static(cubes.max(1), 1, 1), CubeDim { x: block, y: 1, z: 1 })
}
```
GEMM/distance need a **2D** variant (map `(rows, cols)` onto `ABSOLUTE_POS_X/Y`) — extend this
helper; the ceiling-division shape is the template.

**Launch idiom** (CRITICAL 0.10 pins — `spike_test.rs:60-72`):
```rust
saxpy_kernel::launch::<f32, ActiveRuntime>(
    &client, count, dim,
    a,                                              // scalar by value (A6 — no ScalarArg)
    unsafe { ArrayArg::from_raw_parts(x_handle, n) }, // 2 args, Handle by value
    unsafe { ArrayArg::from_raw_parts(y_handle, n) },
);
let bytes = client.read_one(y_read).expect("read-back"); // consumes handle → clone before launch
```
For primitives, take handles from `DeviceArray::handle().clone()` (see `pipeline_test.rs:147-151`)
because launch consumes its handle args and `read_one` consumes the read handle.

**Pool-routed scratch + caller-out-buffer (D-11)** — copy the metering pattern from
`device_array.rs:59-83`:
```rust
let metering_handle = pool.acquire(byte_size);
pool.release(metering_handle, byte_size);    // 0.10 has no in-place write into empty (A3)
// ...kernel-produced outputs DO write into a pool-acquired empty handle (RESEARCH Pitfall 6):
//   let out = pool.acquire(out_bytes); pass `out` as the kernel's &mut output target.
```
**Host API signature** (RESEARCH §Per-Primitive line 372): `gemm(pool, a: &DeviceArray, (m,k), b, (k,n), transa, transb, out: Option<DeviceArray>) -> DeviceArray`. Assert `rows*cols == len` per operand (D-04).
When `out` is `None`, allocate fresh and return; when supplied, reuse it as the kernel output (D-11).

**Shape-assert source of truth** — `DeviceArray.len()` (`device_array.rs:108-110`); the
`rows*cols == len` check (D-04) mirrors `pipeline_test.rs:142` (`assert_eq!(n, y.len(), ...)`).

**Distance composition** (`distance.rs`): GEMM(transb) → row-L2-norm reduce → `dist_combine_clamp`
→ optional `sqrt_elem`, all `DeviceArray`→`DeviceArray` **no `to_host` between stages** (D-05/D-10
gate 2). **Covariance** (`covariance.rs`): column-mean reduce → `Aᵀ·A` via GEMM(transa) → scale by
`1/(n-ddof)`, reusing the GEMM output buffer (D-10 gate 3) by passing it as `out`.

---

### `crates/mlrs-backend/src/pool.rs` (MODIFY — add `read_backs` counter, D-10 gate 2)

**Analog:** `pool.rs` itself (EXACT — extend the existing `PoolStats` + `acquire`/`release`
counter machinery). RESEARCH §D-10 Memory-Gate Strategy (lines 354-364) flags this as the one
small Wave-0 addition.

**Existing counter struct to extend** (`pool.rs:35-48`):
```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    pub allocations: u64,
    pub reuses: u64,
    pub peak_bytes: u64,
    pub live_bytes: u64,
    // ADD: pub read_backs: u64,   // bumped on each to_host/read_one (D-10 gate 2)
}
```
**Existing bump pattern to copy** (`pool.rs:103-119` `acquire`): a counter is `self.stats.X += 1`
inside the method that performs the operation. For `read_backs`, the cleanest hook (RESEARCH
line 360) is to route `DeviceArray::to_host` read-backs through a pool method that bumps the
counter — instrument `to_host` (`device_array.rs:91-105`) to call it, OR add a
`pool.record_read_back()` the primitives/tests call. `PoolStats` is `Copy + Default + Eq` so the
new field needs no other change. Keep the `Drop`/`log_stats` surfacing (`pool.rs:131-145`).

---

### `crates/mlrs-backend/tests/{gemm,reduce,distance,covariance}_test.rs` (test, oracle)

**Analog:** `tests/pipeline_test.rs` (EXACT — the canonical oracle test: load fixture → build
inputs → device path → read-back → `assert_close`/`assert_slice_close`) + `spike_test.rs` for the
raw launch.

**Test file header + imports** (copy the shape from `pipeline_test.rs:35-47`):
```rust
use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{assert_slice_close, load_npz, OracleCase, F32_TOL, F64_TOL};
```
Each test starts `let _ = env_logger::builder().is_test(true).try_init();` and logs the dtype line
`capability::log_oracle_dtype(FloatKind::F32, backend, "default");` (`pipeline_test.rs:177-180`).
**AGENTS.md §2:** all tests in `tests/`, NEVER an in-source `mod tests` (`smoke.rs:9-11`,
`pool.rs:21-22`).

**f64 capability gate** (copy `pipeline_test.rs:226-236`, RESEARCH lines 414-423):
```rust
#[test]
fn gemm_f64_matches_host_ref() {
    let _ = env_logger::builder().is_test(true).try_init();
    if mlrs_backend::capability::skip_f64_with_log() { return; } // SKIP, never fail
    // build DeviceArray<_, f64> inputs, run gemm, to_host, assert_slice_close(&F64_TOL)
}
```

**Host-reference sweep (D-12 primary)** — NEW in-test Rust loops (triple-loop matmul, Σ reductions,
direct distance) compared via `assert_slice_close(&got, &expected, &F64_TOL)` (`compare.rs:70-93`).
Build inputs with a deterministic seeded host RNG; compare device output to the host loop. No
in-tree analog for the loop bodies themselves (they're trivial), but the **compare call** is exactly
`pipeline_test.rs:262-265`.

**f32 near-zero oracle nuance:** for f32 distance with catastrophic cancellation, the
`F32_ORACLE_NEAR_ZERO_FLOOR` pattern in `pipeline_test.rs:78-106` is the precedent if the strict
relative bound spuriously fails near zero — reuse that local helper shape rather than loosening
`F32_TOL`.

**reduce_test.rs dual-path:** assert BOTH the plane and shared kernels (D-03), gating the plane path
with a subgroup-capability skip mirroring `skip_f64_with_log` (RESEARCH line 352 / Open Question 3 —
probe the symbol in Wave 0, analog `spike_capability_query_reports_f64` at `spike_test.rs:86-117`).

**distance_test.rs property test:** assert `min(result) >= 0` over random data (D-07 clamp,
RESEARCH Pitfall 5).

---

### `crates/mlrs-backend/tests/memory_gate_test.rs` (test, counter assert — D-10 HARD gate)

**Analog:** `tests/pool_test.rs` (EXACT — it already asserts on `PoolStats` fields; Phase 2 turns
the same assertions from logged-only into build-failing).

**Counter-assert pattern to copy** (`pool_test.rs:22-55`):
```rust
let h0 = pool.acquire(size);
assert_eq!(pool.stats().allocations, 1, "first acquire allocates");
pool.release(h0, size);
let _h1 = pool.acquire(size);
assert_eq!(pool.stats().reuses, 1, "second acquire of same size reuses");
assert_eq!(pool.stats().allocations, 1, "reuse must NOT increment allocations");
```
**The three D-10 HARD assertions** (RESEARCH lines 356-362):
1. **Reuse > 0, allocations bounded:** run a primitive N times at the same shape threading ONE pool;
   `assert!(stats.reuses >= N - 1)` and `assert!(stats.allocations <= FIRST_ITER_ALLOCS)`.
2. **No mid-pipeline host round-trip:** run GEMM→reduce→distance `DeviceArray`→`DeviceArray`, then
   `assert_eq!(stats.read_backs, 1)` (the single terminal compare). Requires the `read_backs`
   counter added to `pool.rs`.
3. **Gram reuses GEMM buffer:** pass the GEMM output `DeviceArray` as covariance's `out`; assert
   `stats.allocations` does not rise beyond the GEMM's own (reuse bumps, allocations does not).

**Chained device-resident pipeline reference** — the multi-stage device path shape comes from
`pipeline_test.rs` `run_saxpy` (`:134-171`); chain primitive calls the same way (handle clones,
no intermediate `to_host`).

---

### `tests/fixtures/*.npz` + `scripts/gen_oracle.py` (fixture, file-I/O — D-12 convention fixtures)

**Analog:** `scripts/gen_oracle.py` (EXACT — extend it) + `mlrs-core/examples/gen_fixture.rs`
(the Rust-side npz writer alternative).

**Naming + contract convention to copy** (`gen_oracle.py:38-65`): `case_dtype_seed.npz`
(e.g. `gemm_f64_seed42.npz`, `cov_ddof1_f64_seed42.npz`, `dist_sq_f32_seed42.npz`), named arrays,
seeded `np.random.default_rng(seed)`. The module docstring (`gen_oracle.py:11-20`) already names
Phase-4 extension; Phase 2 adds GEMM, distance squared/sqrt, and `np.cov` ddof=0/1 cases. Regen
needs a /tmp venv with numpy (PEP 668 — MEMORY.md `oracle-fixture-regen-needs-venv`); committed
blobs are read at test time with **no Python** (`oracle.rs:1-10`).

**Loader the fixtures must satisfy** — `mlrs_core::oracle::load_npz` → `OracleCase` with
`.expect_f64(name)` / `.expect_f32(name)` / `.shape(name)` (`oracle.rs:33-83`). The committed `.npz`
must carry 4- or 8-byte float arrays only (`oracle.rs:115-135`).

**Pure-Rust fixture alternative** (if numpy regen is unavailable) — the npyz writer pattern in
`gen_fixture.rs:26-64` (`NpzWriter::new` + `.array(name).default_dtype().shape(&[len]).begin_nd().extend(iter)`),
proven round-trip in `spike_test.rs:166-220`. Use this ONLY where the reference is computable in
Rust; D-12 convention fixtures (`np.cov` ddof) should come from numpy to pin the exact convention.

---

## Shared Patterns

### Generic feature-free `#[cube(launch)]` kernel (the locked idiom — D-13)
**Source:** `crates/mlrs-kernels/src/smoke.rs:24-30`
**Apply to:** every new kernel (`gemm.rs`, `reduce.rs`, `elementwise.rs`)
```rust
use cubecl::prelude::*;
#[cube(launch)]
pub fn k<F: Float + CubeElement>(/* &Array<F> in, &mut Array<F> out, scalars by value */) {
    let tid = ABSOLUTE_POS;
    if tid < x.len() { /* ... */ }   // ALWAYS bounds-check (over-provisioned launch)
}
```
`<F: Float + CubeElement>` is non-negotiable; `mlrs-kernels` carries NO backend feature (lib.rs:1-6).

### Launch + read-back (cubecl 0.10 pins)
**Source:** `crates/mlrs-backend/tests/spike_test.rs:60-75`; also `pipeline_test.rs:152-168`
**Apply to:** all `prims/*.rs` host wrappers and all `*_test.rs` files
- scalar args by value (no `ScalarArg`) — A6
- `ArrayArg::from_raw_parts(handle, len)` — 2 args, Handle by value
- `read_one`/launch CONSUME the handle → `.clone()` before (handles are cheap ref-counted)
- zero-init `F::from_int(0i64)`; index with `as usize`

### Pool-routed allocation + D-11 scratch/out-buffer
**Source:** `crates/mlrs-backend/src/device_array.rs:59-105`, `src/pool.rs:103-129`
**Apply to:** all `prims/*.rs` and `memory_gate_test.rs`
- `pool.acquire(bytes)` / `pool.release(handle, bytes)` meter every buffer
- kernel-produced outputs DO write into a pool-acquired empty handle (RESEARCH Pitfall 6);
  the in-place limitation only bit host uploads (A3)
- caller-provided `out: Option<DeviceArray>` reuses the handle as the kernel output target (D-11)

### Tolerance comparison (D-13)
**Source:** `crates/mlrs-core/src/compare.rs:50-93`, `tolerance.rs:26-38`
**Apply to:** every `*_test.rs`
`assert_close(got, expected, &F64_TOL)` / `assert_slice_close(&got, &expected, &F32_TOL)` — abs AND
rel with the `NEAR_ZERO_FLOOR` guard. Compare device output against an **f64 host reference** even
for f32 device output (RESEARCH Pitfall 3). Never hand-roll float-eq.

### f64 capability gate (skip-with-log, never fail — D-13)
**Source:** `crates/mlrs-backend/src/capability.rs:101-109`; usage `pipeline_test.rs:232-236`
**Apply to:** every f64 test arm; mirror the same mechanism for the reduction plane/subgroup gate
```rust
if mlrs_backend::capability::skip_f64_with_log() { return; }
```

### Oracle fixture load (no Python at test time — D-12)
**Source:** `crates/mlrs-core/src/oracle.rs:77-83`; usage `pipeline_test.rs:182-188`
**Apply to:** every convention-fixture test
`let case = load_npz(fixture("gemm_f64_seed42.npz")).expect(...); let x = case.expect_f64("X");`

### thiserror error variants (libs) — extension point
**Source:** `crates/mlrs-core/src/error.rs:16-56` (`BridgeError`, `#[derive(Debug, Error)]` + `#[error("...")]`)
**Apply to:** any new primitive error variants (D-13 / Claude's discretion). Follow the one-variant-
per-violation-class shape with a self-describing `#[error("...")]` message (e.g. a shape-mismatch
variant for the `rows*cols != len` / incompatible-GEMM-dims case, D-04).

### Source/test separation (AGENTS.md §2)
**Source:** stated in `smoke.rs:9-11`, `pool.rs:21-22`, `device_array.rs:29`
**Apply to:** ALL files — production `src/*.rs` carry NO `#[cfg(test)] mod tests`; every test lives
in `crates/mlrs-backend/tests/*_test.rs`.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/mlrs-kernels/src/gemm.rs` (algorithm body) | kernel | transform | No multi-stage shared-memory/tiled kernel exists in Phase 1 (smoke.rs is per-element). Use RESEARCH Pattern 3 (lines 224-251) + `Cubecl_shared_memory.md`/`Cubecl_transpose.md`. **Existence gated by the GEMM-substrate DECISION task** (Open Question 1 — if `cubecl-matmul` 0.10 is pinned, GEMM is a backend wrapper instead and this file is omitted). |
| reduction kernel bodies (plane/shared/argmin) in `reduce.rs` | kernel | transform | `#[cube(launch)]` shell is the smoke analog, but the dual-path reduction + `(value,index)` argmin combine have no Phase-1 precedent — bodies come from RESEARCH "Dual-Path Reduction Mechanics" (lines 313-352, CITED to the plane + shared-memory manuals). |
| host-reference loops inside `*_test.rs` (D-12 primary) | test | transform | New in-test Rust (triple-loop matmul / Σ / direct distance). Trivial to write; only the `assert_slice_close` *compare* call has an analog (`pipeline_test.rs:262-265`). |

> For all three, the planner should use the RESEARCH §"Architecture Patterns" / §"Dual-Path
> Reduction Mechanics" / §"Code Examples" excerpts as the body source, and the `smoke.rs`/`spike_test.rs`
> idioms for the surrounding shell. No external code to port.

---

## Metadata

**Analog search scope:** `crates/mlrs-kernels/{src}`, `crates/mlrs-backend/{src,tests}`,
`crates/mlrs-core/{src,examples}`, `scripts/`, `tests/fixtures/`
**Files scanned:** 14 source/test files (full read of smoke.rs, lib.rs ×2, pool.rs, device_array.rs,
runtime.rs, capability.rs, compare.rs, tolerance.rs, oracle.rs, error.rs, gen_fixture.rs,
gen_oracle.py, spike_test.rs, pool_test.rs, pipeline_test.rs) + AGENTS.md
**Pattern extraction date:** 2026-06-12
```
