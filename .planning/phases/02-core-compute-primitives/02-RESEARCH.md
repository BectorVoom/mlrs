# Phase 2: Core Compute Primitives - Research

**Researched:** 2026-06-12
**Domain:** Backend-portable CubeCL compute primitives (GEMM, reductions, pairwise distance, covariance/XᵀX) generic over `<F: Float>` and `<R: Runtime>`, validated f32+f64 on cpu+wgpu.
**Confidence:** HIGH on the version/API blocker and dual-path reduction mechanics (verified against cached cubecl source + crates.io + git tags); MEDIUM on exact `cubecl-matmul` 0.9-pre.5 `launch` signature details (cached source is 0.9-pre.5, not the 0.10 the workspace pins — and **no 0.10 matmul exists**, which is itself the headline finding).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Reductions expose **full-array (1D total) AND axis-wise (2D row-reduce + column-reduce)** for sum/mean/min/max/L2-norm.
- **D-02:** **argmin/argmax** deliver **full-array AND per-row argmin** over a 2D matrix. **Tie-break = lowest index** (numpy/sklearn convention). Index-reduction kernel built and validated here, not deferred to KMeans.
- **D-03:** Each reduction must pass on wgpu via **both** a plane/subgroup path **AND** a shared-memory fallback, **no hardcoded plane width** (use `PLANE_DIM`), numerically stable on large inputs. Dual-path ⇒ hand-written kernels.
- **D-04:** **Matrix shape passed explicitly as `(rows, cols)` per call.** `DeviceArray<R,F>` stays the flat 1D buffer from Phase 1; carried `len` remains the single source of truth for read-back. Caller-side `rows*cols == len` asserted. DeviceArray is **not** extended with 2D shape state.
- **D-05:** **Primitives take and return `DeviceArray` (device-resident in/out).** Chained calls never round-trip to host. Thin host-slice helpers exist **only in tests**.
- **D-06:** **GEMM exposes BLAS-style `transa`/`transb` flags** so XᵀX reuses GEMM directly without materializing a transpose buffer — **IF** `cubecl-matmul` supports transposed/strided operands (Open Question 1; **resolved below**). Fallback = transpose kernel + row-major multiply.
- **D-07:** Distance via **GEMM-expansion** `‖x‖² + ‖y‖² − 2·XYᵀ`, reusing GEMM (D-06) + row-L2-norm reduction (D-01), then **`max(d², 0)` clamp**. Direct difference-accumulation rejected as default.
- **D-08:** Distance returns **squared distance** as core output, with **optional sqrt** at the boundary for KNN (NEIGH-01). One validated kernel serves KMeans/DBSCAN/KNN.
- **D-09:** Covariance/XᵀX built **on GEMM** as `Aᵀ·A` via D-06 transpose flags. Convention pinned by committed numpy fixture (D-12). Must reuse GEMM output buffer (D-10).
- **D-10:** Memory gate is **build-failing** with three HARD assertions: (1) Reuse > 0, allocations bounded; (2) No mid-pipeline host round-trip; (3) Gram reuses GEMM buffer.
- **D-11:** Primitives accept an **optional caller-provided output `DeviceArray`** (reused across iterations) and **draw internal scratch from the `BufferPool`**. No out-buffer supplied ⇒ fresh array allocated and returned.
- **D-12:** **Hybrid reference.** Primary = live Rust host reference (naive seeded-random CPU loops). Supplemented by committed numpy `.npz` convention fixtures for: covariance normalization (`np.cov` ddof=1 vs population ddof=0), distance squared-vs-sqrt semantics, GEMM.
- **D-13 (carried):** `assert_close` with `F32_TOL`/`F64_TOL` = 1e-5 (abs AND rel, near-zero guard). f64 capability-gated via `skip_f64_with_log`. `mlrs-kernels` feature-free; launch wrappers in `mlrs-backend`. `thiserror` in libs / `anyhow` at boundaries; deps track latest.

### Claude's Discretion
- Module/file layout within `mlrs-kernels` and `mlrs-backend` (e.g. `prims` module vs per-primitive files). Honor AGENTS.md source/test separation.
- Internal kernel tiling/block sizes, launch-config helpers, shared-memory tile dimensions (subject to `PLANE_DIM` / no-hardcoded-width, D-03).
- Specific `cubecl-matmul` API surface used and the precise transpose-flag plumbing (subject to D-06 + Open Question).
- Naming of new primitive error variants (extend `thiserror` enums).
- Random shapes/seeds for the host-reference sweep, and which exact cases get committed numpy convention fixtures.

### Deferred Ideas (OUT OF SCOPE)
- Direct difference-accumulation distance kernel — only if f32 GEMM-expansion can't hold 1e-5.
- Extending `DeviceArray` to carry 2D shape.
- GEMM library fallback (transpose kernel + row-major multiply) — **only build if `cubecl-matmul` lacks transposed-operand support** (see resolution below).
- Per-estimator-family tolerance tables — deferred to Phase 3/4/5; global 1e-5 TOL governs Phase 2.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PRIM-01 | Backend-portable GEMM / matrix-multiply primitive, oracle-validated | "Standard Stack" + Open Question 1 resolution (matmul version blocker + transpose-via-strides). The GEMM approach decision (wrap `cubecl-matmul` 0.9 via git-pin vs hand-write) is the gating plan decision. |
| PRIM-02 | Numerically-stable reductions (sum/mean/min/max/argmin) with plane/subgroup path AND shared-memory fallback, both pass on wgpu | "Dual-Path Reduction Mechanics" — `plane_sum`/`plane_shuffle_xor` + `PLANE_DIM` path and `SharedMemory` tree-reduction path, both hand-written, selected by a runtime flag. Numerical stability = two-pass mean / Welford for variance. |
| PRIM-03 | Pairwise-distance primitive (Euclidean/squared) with `max(d²,0)` clamp serving KMeans/DBSCAN/KNN | "Per-Primitive Approach §Distance" — GEMM-expansion `‖x‖²+‖y‖²−2XYᵀ` reusing GEMM + row-L2-norm, clamp kernel, optional sqrt. |
| PRIM-04 | Covariance / XᵀX primitive serving PCA + linear closed-form solvers | "Per-Primitive Approach §Covariance" — `Aᵀ·A` via GEMM transpose flag (`swap_dims`), ddof normalization pinned by fixture, buffer reuse for D-10 gate 3. |
</phase_requirements>

## Summary

Phase 2 builds four device-resident primitives on top of the Phase 1 spine (`DeviceArray`, `BufferPool`, `assert_close`, `skip_f64_with_log`, the `#[cube(launch)]` generic-over-`F` pattern from `smoke.rs`). Three of the four (GEMM, reductions, distance, covariance) are well-grounded by existing CubeCL manuals and the Phase 1 launch idiom. **The single highest-leverage finding is a version/availability blocker on the GEMM substrate** that gates the entire phase and must be resolved by a decision before planning the GEMM tasks.

**The blocker, stated precisely:** the workspace pins `cubecl = "0.10.0"` (verified: builds today on cpu). But **`cubecl-matmul` has no 0.10 release** — its latest is `0.9.0-pre.5` (Dec 2025), which **exactly pins `cubecl-core =0.9.0-pre.5`** and therefore cannot link against `cubecl-core 0.10.0`. There is **no `cubecl-matmul` crate in the `tracel-ai/cubecl` repo at the `v0.10.0` git tag** (the algorithm crates were removed from the monorepo; the crate's own `repository` metadata is stale). Roadmap Criterion 1 says GEMM "wraps `cubecl-matmul`" — that wrapping is **not satisfiable at cubecl 0.10 today**. The planner must pick a resolution path (downgrade to cubecl 0.9-pre, git-pin matmul to a 0.10-compatible commit if one exists, or hand-write GEMM) — see Open Question 1.

The good news on the API: once a compatible `cubecl-matmul` is available, **transposed/strided operands ARE supported zero-copy** via `MatmulInputHandle::swap_dims(dim0, dim1)` (swaps shape + strides, no transpose buffer — `[VERIFIED: cubecl-matmul-0.9.0-pre.5 src/base.rs:369]`), and **f64 IS a supported `MatmulPrecision`** (`impl MatmulPrecision for f64 { type Lhs = (f64, f32) … }` — note: f32 accumulation registers, a precision caveat). So D-06's transpose-flag plumbing and D-09's `Aᵀ·A` reuse are sound *if* the version issue is resolved. The dual-path reduction (D-03) is fully hand-writable from the plane + shared-memory manuals and does not depend on any external algorithm crate.

**Primary recommendation:** Plan GEMM as the **first wave** and front-load a `checkpoint:human-verify` decision task resolving the `cubecl-matmul` version blocker (Open Question 1) before any GEMM kernel/wrapper code is written. Build the dependency-ordered chain GEMM → reductions → distance → covariance, with the D-10 memory gate as a dedicated final verification wave that asserts on the existing `PoolStats` counters.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| GEMM kernel/algorithm | `mlrs-kernels` (if hand-written) OR external `cubecl-matmul` | `mlrs-backend` (launch wrapper + host API) | Kernels are feature-free + generic (D-13). If wrapping matmul, the wrapper lives in backend (owns runtime). |
| GEMM host API (`transa`/`transb`, out-buffer) | `mlrs-backend` | — | Owns `ActiveRuntime`, `BufferPool`, `DeviceArray` (D-05/D-11). |
| Reduction kernels (plane + shared-mem paths) | `mlrs-kernels` | `mlrs-backend` (launch + path selection) | Hand-written `#[cube]` kernels, feature-free (D-03/D-13). |
| Distance (GEMM-expansion + clamp) | `mlrs-backend` (orchestration) | `mlrs-kernels` (clamp + sqrt elementwise kernels) | Composition of GEMM + reduction + a small clamp kernel; orchestration owns the device-resident chaining (D-05/D-07). |
| Covariance/XᵀX | `mlrs-backend` (orchestration over GEMM) | — | `Aᵀ·A` is GEMM-with-transpose + ddof scale; no new kernel beyond an optional scale kernel (D-09). |
| Memory gate (D-10) | `mlrs-backend/tests/` | `BufferPool::stats()` | Hard assertions on the existing `PoolStats` API; test-only (AGENTS.md). |
| Host references + convention fixtures (D-12) | `mlrs-backend/tests/` + `mlrs-core` (npz loader, `assert_close`) | `scripts/gen_oracle.py` | Reuses Phase 1 oracle infra; host-reference loops are new in-test Rust. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` (umbrella) | `0.10.0` | Device kernels, runtime, client, `SharedMemory`, plane intrinsics, `Bytes`, `Handle` | Already the workspace spine (Phase 1); builds on cpu today `[VERIFIED: cargo build -p mlrs-backend --features cpu]`. |
| `cubecl-matmul` | **⚠ NO 0.10 — latest `0.9.0-pre.5`** | GEMM engine (`launch`, `Strategy`, `MatmulInputHandle`, `swap_dims`) | Roadmap-named substrate for GEMM, BUT incompatible with cubecl 0.10 (see Open Question 1). `[VERIFIED: crates.io API — 0.10.0 does not exist]` |
| `bytemuck` | `1` | `cast_slice` for host↔device reinterpretation | Phase 1 pattern (`device_array.rs`, `spike_test.rs`). |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `cubecl-std` | `0.10.0` (transitive via umbrella? **NO — not re-exported**) | `TensorHandle`, `into_contiguous` | Needed by the matmul example's handle wrapping. `cubecl-std 0.10.0` exists in registry and exposes `TensorHandle::new/empty/zeros` `[VERIFIED: cubecl-std-0.10.0 src/tensor/handle.rs]`, but the `cubecl` umbrella only re-exports it under `feature = "stdlib"` as `cubecl::std`. Confirm it's reachable before relying on it. |
| `npyz` | `0.9` | `.npz` convention-fixture loader (D-12) | Already in Phase 1 (`spike_test.rs` proves read+write); reuse `mlrs-core` oracle loader. |
| `log` / `env_logger` | `0.4` / `0.11` | Pool-counter + dtype/backend log lines | Phase 1 pattern; reuse `capability::log_oracle_dtype`. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `cubecl-matmul` (blocked at 0.10) | **Hand-written tiled GEMM in `mlrs-kernels`** (shared-memory tiling per `Cubecl_shared_memory.md` + `Cubecl_transpose.md`) | Full control, no external dep, no version coupling, naturally feature-free + generic over `<F: Float>`, and `transa`/`transb` become trivial index-swap (`i*cols+j` vs `j*rows+i`) rather than relying on `swap_dims`. Cost: more kernel code + must hit 1e-5 vs host ref on its own. **Strongly recommended fallback if Open Question 1 can't pin a 0.10-compatible matmul.** |
| `cubecl-matmul` | Downgrade workspace to `cubecl 0.9.0-pre.x` to match matmul | Re-validates Phase 1 against an older cubecl; risky regression of completed work. Not recommended. |
| `cubecl-matmul` | Git-pin matmul to a tracel-ai commit compatible with cubecl-core 0.10 | **Only if such a commit exists** — at the `v0.10.0` tag the crate is absent and cubecl `main` is already `0.11.0-pre.1` with matmul still external. Requires locating the real 0.10-line matmul source. Uncertain. |
| `cubecl-reduce` for reductions | Hand-written dual-path kernels | D-03 mandates BOTH a plane path AND a shared-mem fallback as *separately exercisable* paths — a library that hides one path can't satisfy this. Hand-write (already the locked decision). `cubecl-reduce` is also not in the registry cache and tracks the 0.9 line. |

**Installation (conditional on Open Question 1 resolution):**
```bash
# ONLY if a cubecl-0.10-compatible matmul is located (see Open Question 1):
#   cubecl-matmul = { git = "https://github.com/<org>/<repo>", rev = "<sha>" }
# RECOMMENDED DEFAULT (no new dependency): hand-write tiled GEMM in mlrs-kernels.
# Reductions/distance/covariance add NO external crates — pure #[cube] kernels.
```

**Version verification performed:**
- `cubecl 0.10.0` — exists, published 2026-05-07, workspace builds on cpu. `[VERIFIED: crates.io + cargo build]`
- `cubecl-matmul` — latest `0.9.0-pre.5` (2025-12-05); **no 0.10**; pins `cubecl-core =0.9.0-pre.5`. `[VERIFIED: crates.io API + cached Cargo.toml]`
- `cubecl` repo `v0.10.0` tag — contains NO `cubecl-matmul`/`cubecl-reduce`/`cubecl-random`/`cubecl-linalg` crates. `[VERIFIED: git ls-tree v0.10.0]`
- `cubecl` `main` — `0.11.0-pre.1`, still no in-repo matmul/reduce. `[VERIFIED: git show main:Cargo.toml]`

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `cubecl` | crates.io | published 2026-05-07 (0.10.0) | established (Burn ecosystem) | github.com/tracel-ai/cubecl | not run (offline) | Approved — already a workspace dep, builds today |
| `cubecl-matmul` | crates.io | 0.9.0-pre.5, 2025-12-05 | low (pre-release) | github.com/tracel-ai/cubecl (stale metadata) | not run (offline) | **BLOCKED on version, not legitimacy** — legit crate, wrong version line |
| `npyz` | crates.io | Phase-1 dep | — | github | n/a | Approved (carried from Phase 1) |
| `bytemuck` | crates.io | Phase-1 dep | very high | github | n/a | Approved (carried) |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*slopcheck was not run (no network/pip available this session). All packages here are pre-existing workspace deps (Phase 1) or first-party tracel-ai crates verified by direct crates.io API + cached source inspection. The only "new" candidate, `cubecl-matmul`, is gated behind a `checkpoint:human-verify` decision (Open Question 1) regardless.*

## Architecture Patterns

### System Architecture Diagram

```text
                       ┌─────────────────────────────────────────────┐
   host seeded-random  │  mlrs-backend/tests/  (oracle + memory gate) │
   inputs + .npz       │  host-ref loops │ assert_close │ PoolStats    │
   convention fixtures └───────┬───────────────────────────▲──────────┘
                               │ DeviceArray::from_host     │ to_host (TESTS ONLY, D-05)
                               ▼                            │
   ┌───────────────────────────────────────────────────────┴────────────┐
   │  mlrs-backend  (host orchestration — owns ActiveRuntime + BufferPool)│
   │                                                                       │
   │   GEMM(transa,transb, out?) ──┐                                       │
   │      │ (matmul wrap OR hand-  │                                       │
   │      │  written tiled kernel) │                                       │
   │      ▼                        ▼                                       │
   │   covariance = Aᵀ·A ◄─reuse── distance = ‖x‖²+‖y‖²−2XYᵀ ─► clamp d²≥0│
   │      │ (swap_dims/idx-swap)   │  └─ row-L2-norm reduction ◄──┐        │
   │      ▼                        ▼                              │        │
   │   ddof scale            optional sqrt (KNN)                  │        │
   └────────────┬──────────────────────────────────────────────┬┘        │
                │ launch::<F, ActiveRuntime>                    │ launch  │
                ▼                                               ▼         │
   ┌──────────────────────────────────────────────────────────────────────┐
   │  mlrs-kernels  (feature-free #[cube] kernels, generic <F: Float>)      │
   │   • tiled_gemm (if hand-written)   • reduce_plane (PLANE_DIM path)     │
   │   • reduce_shared (SharedMemory)   • argmin_rowwise (tie=lowest idx)   │
   │   • clamp_nonneg / sqrt_elem       • (optional) transpose / scale      │
   └──────────────────────────────────────────────────────────────────────┘
                │ CubeCount/CubeDim + ArrayArg::from_raw_parts
                ▼
        CubeCL runtime (cpu | wgpu)  ── BufferPool.acquire/release meters every buffer
```

Trace the primary use case (distance for KMeans): host uploads X via `DeviceArray::from_host` → GEMM computes `XXᵀ` (device-resident) → row-L2-norm reduction computes `‖x_i‖²` → backend combines `‖x_i‖²+‖x_j‖²−2·XXᵀ` and launches the `clamp_nonneg` kernel → result stays on device as a `DeviceArray` (no host round-trip, D-05/D-10 gate 2) → only the test reads it back to compare against the host reference.

### Recommended Project Structure (Claude's discretion — D-13/Discretion)
```
crates/mlrs-kernels/src/
├── lib.rs              # re-export prim kernels
├── smoke.rs            # (existing) saxpy
├── gemm.rs             # tiled GEMM kernel IF hand-written (else omitted)
├── reduce.rs           # reduce_plane + reduce_shared + argmin_rowwise kernels
└── elementwise.rs      # clamp_nonneg, sqrt_elem, (optional) scale

crates/mlrs-backend/src/
├── prims/
│   ├── mod.rs
│   ├── gemm.rs         # GEMM host API: transa/transb, out-buffer (D-06/D-11)
│   ├── reduce.rs       # axis-wise + full-array dispatch, path selection (D-01/D-03)
│   ├── distance.rs     # GEMM-expansion + clamp + optional sqrt (D-07/D-08)
│   └── covariance.rs   # Aᵀ·A + ddof scale, GEMM-buffer reuse (D-09)
└── (existing: runtime.rs, device_array.rs, pool.rs, capability.rs, bridge.rs)

crates/mlrs-backend/tests/
├── gemm_test.rs        # host-ref sweep + npz GEMM fixture, f32/f64, cpu/wgpu
├── reduce_test.rs      # dual-path: BOTH plane and shared paths asserted
├── distance_test.rs    # clamp ≥0 property + host ref + sqrt fixture
├── covariance_test.rs  # ddof=0 and ddof=1 fixtures + host ref
└── memory_gate_test.rs # D-10 three HARD assertions on PoolStats
```

### Pattern 1: Generic `#[cube(launch)]` kernel (the locked idiom)
**What:** Every kernel follows `smoke.rs` — `#[cube(launch)] pub fn k<F: Float + CubeElement>(...)`, launched `k::launch::<F, ActiveRuntime>(&client, count, dim, args...)`.
**When to use:** All kernels (D-13 mandates feature-free generics).
**Example:**
```rust
// Source: crates/mlrs-kernels/src/smoke.rs (existing, verified-compiling)
#[cube(launch)]
pub fn saxpy_kernel<F: Float + CubeElement>(a: F, x: &Array<F>, y: &mut Array<F>) {
    let tid = ABSOLUTE_POS;
    if tid < x.len() { y[tid] = a * x[tid] + y[tid]; }
}
// launched in tests/spike_test.rs:
//   saxpy_kernel::launch::<f32, ActiveRuntime>(&client, count, dim, a,
//       unsafe { ArrayArg::from_raw_parts(x_handle, n) },
//       unsafe { ArrayArg::from_raw_parts(y_handle, n) });
```
Critical 0.10 idioms already pinned by Phase 1: scalar args passed **by value** (no `ScalarArg` wrapper, A6); `ArrayArg::from_raw_parts(handle, len)` (2 args, Handle by value); `client.read_one(handle)` consumes the handle so `.clone()` before launch; index with `as usize`; zero-init with `F::from_int(0i64)`.

### Pattern 2: GEMM via `cubecl-matmul` (IF version resolved) — transpose is zero-copy
**What:** Wrap operands in `MatmulInputHandle::Normal(TensorHandle::new(handle, shape, strides))`, call `swap_dims(0,1)` to transpose without a copy, then `launch(&Strategy::Auto, &client, lhs, rhs, out, dtypes)`.
**When to use:** Only after Open Question 1 yields a cubecl-0.10-compatible matmul.
**Example (⚠ 0.9-pre.5 surface — 0.10 signature may differ):**
```rust
// Source: [VERIFIED: cubecl-matmul-0.9.0-pre.5 src/base.rs:369, :545]
// transpose = swap shape + strides; NO transpose buffer (resolves D-06):
let mut lhs = MatmulInputHandle::Normal(TensorHandle::<R>::new(a_handle, vec![m, k], a_strides));
if transa { lhs.swap_dims(0, 1); }   // now reads Aᵀ strided, zero-copy
// 0.9-pre.5 launch takes a MatmulElems and returns Result:
//   pub fn launch<R: Runtime>(strategy, client, lhs, rhs, out, dtypes: MatmulElems)
//       -> Result<(), MatmulSetupError>
// f64 supported: impl MatmulPrecision for f64 { type Lhs = (f64, f32); ... }
//   ⚠ NOTE: f64 input but f32 accumulation registers — precision caveat vs a
//   pure-f64 host reference; watch the 1e-5 bound on large-K reductions.
```
**⚠ The matmul manual in `cubecl_manual/.../cubecl_matmul_gemm_example.md` shows `launch::<WgpuRuntime, f32>(&Strategy::Auto, client, lhs, rhs, out)` — a DIFFERENT, older signature with no `dtypes` arg and using `client.create_tensor`/`empty_tensor`. Treat the manual as conceptual only; the real signature must be pinned against whatever matmul version Open Question 1 selects.**

### Pattern 3: GEMM hand-written (RECOMMENDED DEFAULT) — transpose is index arithmetic
**What:** Tiled shared-memory GEMM kernel; `transa`/`transb` handled by choosing the index expression (`A[i*K+k]` vs `A[k*M+i]`) — no buffer, no library.
**When to use:** If Open Question 1 cannot pin a 0.10-compatible matmul (likely).
**Example sketch (compose from the shared-memory + transpose manuals):**
```rust
// Source: synthesized from Cubecl_shared_memory.md + Cubecl_transpose.md (CITED)
#[cube(launch)]
pub fn gemm_kernel<F: Float + CubeElement>(
    a: &Array<F>, b: &Array<F>, c: &mut Array<F>,
    m: u32, k: u32, n: u32,
    #[comptime] trans_a: bool, #[comptime] trans_b: bool,  // comptime ⇒ no per-thread branch cost
) {
    let row = ABSOLUTE_POS_X; let col = ABSOLUTE_POS_Y;
    if row < m && col < n {
        let mut acc = F::from_int(0i64);
        let mut kk = 0u32;
        while kk < k {
            let a_idx = if trans_a { kk * m + row } else { row * k + kk };
            let b_idx = if trans_b { col * k + kk } else { kk * n + col };
            acc += a[a_idx as usize] * b[b_idx as usize];
            kk += 1u32;
        }
        c[(row * n + col) as usize] = acc;
    }
}
// Tiled version loads A/B tiles into SharedMemory + sync_cube() between stages
// (per the shared-memory manual) for the large-shape numerical-stability cases.
```

### Anti-Patterns to Avoid
- **Putting a backend feature on `mlrs-kernels`:** breaks D-13/FOUND-02; kernels stay feature-free.
- **In-source `#[cfg(test)] mod tests`:** forbidden by AGENTS.md §2 — all tests in `tests/`.
- **Materializing a transpose buffer for XᵀX:** D-06 forbids it; use `swap_dims` (matmul) or index-swap (hand-written). A transpose kernel is the *deferred* fallback only.
- **Host round-trip mid-pipeline:** D-05/D-10 gate 2 forbids it; chain `DeviceArray`→`DeviceArray` on device.
- **Hardcoding plane width (e.g. `32`):** D-03 mandates `PLANE_DIM`.
- **Naive single-pass mean on large inputs:** accumulation error breaks the "numerically stable on large inputs" criterion — use two-pass / pairwise / Welford (see Pitfall 3).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tolerance comparison | Custom float-eq | `mlrs-core::compare::assert_close` / `assert_slice_close` (abs+rel + near-zero guard) | Phase 1, verified; D-13. |
| Buffer accounting | New counter struct | `BufferPool::stats()` → `PoolStats{allocations,reuses,peak_bytes,live_bytes}` | D-10 gate asserts on these exact fields. |
| Device array + read-back | New wrapper | `DeviceArray::from_host` / `to_host` (len-carrying, T-04-01 mitigation) | Phase 1; D-04/D-05. |
| f64 backend gating | Ad-hoc cfg | `capability::skip_f64_with_log()` early-return | Phase 1; D-13. |
| npz fixture loading | New parser | `mlrs-core` oracle npz loader (`npyz` `by_name`) | Phase 1; D-12. |
| Transpose for XᵀX | Transpose kernel | `swap_dims` (matmul) or index-swap (hand-written GEMM) | D-06; transpose kernel is the deferred fallback. |
| Plane/subgroup reduction primitive | Custom warp shuffle from scratch | `plane_sum(v)` / `plane_shuffle_xor(v,mask)` + `PLANE_DIM` | CubeCL plane manual provides hardware-portable intrinsics. |

**Key insight:** Everything except the GEMM *algorithm itself* is already in-tree from Phase 1. The only genuinely new external-library question is GEMM — and that question is currently a version blocker, which is exactly why the recommended default is to hand-write GEMM (zero new dependencies, naturally generic + feature-free, transpose-by-index).

## Common Pitfalls

### Pitfall 1: Assuming `cubecl-matmul` 0.10 exists / the umbrella re-exports it
**What goes wrong:** Plan writes `cubecl-matmul = "0.10"` or `use cubecl::matmul::launch` — both fail. No 0.10 release; the `cubecl` umbrella does not depend on or re-export matmul/reduce/linalg at all.
**Why it happens:** The roadmap and the cached manual example predate the matmul/cubecl version split; the manual's `cubecl_matmul_gemm_example.md` uses an even older signature.
**How to avoid:** Resolve Open Question 1 with a `checkpoint:human-verify` decision before any GEMM code. Default to hand-written GEMM.
**Warning signs:** `cargo` error "no matching package named `cubecl-matmul` found" at version 0.10, or `unresolved import cubecl::matmul`.

### Pitfall 2: f64 on wgpu silently skipping (or matmul f32-accumulating f64)
**What goes wrong:** f64 GEMM/distance/covariance either skip on adapters lacking `SHADER_F64`, OR (with cubecl-matmul) compute f64 with f32 accumulation registers (`type Lhs = (f64, f32)`), drifting past 1e-5 on large-K dot products.
**Why it happens:** wgpu f64 is adapter-dependent (Phase 1 D-13); matmul's mixed precision trades accuracy for tensor-core speed.
**How to avoid:** Wrap every f64 primitive test in `skip_f64_with_log()`. For matmul f64, prefer a non-CMMA `Strategy::Naive`/`SimpleUnit` (pure f64 accumulation) — or hand-write GEMM with `F`-typed accumulator. On this env's adapter (AMD RADV GFX1152) `SHADER_F64` is present so f64 runs.
**Warning signs:** f64 test passes on cpu, fails-by-a-hair on wgpu only; abs_err ~1e-4 on large K.

### Pitfall 3: Numerically-unstable reductions on large inputs (Criterion 2)
**What goes wrong:** Single-pass naive `Σ` accumulation loses precision on large N; mean/variance for covariance drift past 1e-5.
**Why it happens:** Float accumulation error grows with N; the criterion explicitly says "numerically stable on large inputs."
**How to avoid:** Tree/pairwise reduction (the shared-memory manual's `log₂` tree is already pairwise-stable); two-pass mean (sum, divide, then re-center for variance) or Welford for covariance ddof. Compare against an f64 host reference even for f32 device output.
**Warning signs:** error scales with input length in the host-ref sweep.

### Pitfall 4: argmin tie-break diverging from numpy (D-02)
**What goes wrong:** Parallel argmin reduction picks an arbitrary index on ties; KMeans label parity (Phase 5) breaks.
**Why it happens:** Plane/tree reductions combine pairs in non-left-to-right order.
**How to avoid:** Carry `(value, index)` pairs and on equal values keep the **lower index** in every combine step (both the plane-shuffle path and the shared-mem path). Pin with a fixture that has a deliberate tie.
**Warning signs:** intermittent label mismatch only on tied distances.

### Pitfall 5: Distance producing negative `d²` under f32 (Criterion 3)
**What goes wrong:** `‖x‖²+‖y‖²−2XYᵀ` cancels to a small negative for near-identical rows in f32.
**Why it happens:** Catastrophic cancellation in the GEMM-expansion form (the whole reason the clamp exists, D-07).
**How to avoid:** Apply the `max(d²,0)` clamp kernel unconditionally (even for f64). Add a property test asserting `min(result) >= 0` over random data.
**Warning signs:** sqrt-of-negative → NaN in the KNN sqrt path.

### Pitfall 6: D-11 scratch + cubecl 0.10 has no in-place write into `empty`
**What goes wrong:** Trying to write a computed result into a pool-acquired `empty` handle in place; 0.10 has no host-write-into-empty API (Phase 1 A3/STATE finding).
**Why it happens:** `empty` returns an uninitialized device handle; population happens via kernel output or `client.create` (which is a fresh allocation).
**How to avoid:** For *device-produced* outputs (kernel writes), `pool.acquire(bytes)` then pass the handle as the kernel's `&mut` output — that's fine (the kernel writes it). The in-place limitation only bit *host uploads* (Phase 1). For the optional caller-provided out-buffer (D-11), reuse the handle as the kernel output target directly.
**Warning signs:** garbage/zero output when expecting in-place host data in a scratch buffer.

## Dual-Path Reduction Mechanics (D-03 / Open Question 3 — RESOLVED)

Both paths are **hand-written `#[cube]` kernels** (no external crate) and made **independently selectable** so tests exercise each on wgpu. Recommended selection mechanism: a `#[comptime] use_plane: bool` kernel parameter OR two separate kernel functions dispatched by a host-side `enum ReducePath { Plane, Shared }`. Two separate functions is cleaner (no comptime branch, each path is a distinct launch the test names explicitly).

**Path A — Plane/subgroup (`PLANE_DIM`, no hardcoded width):**
```rust
// Source: CITED Cubecl_plane.md §3 (plane_shuffle_xor) — power-of-2 PLANE_DIM fold
#[cube(launch)]
pub fn reduce_sum_plane<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let mut acc = if ABSOLUTE_POS < input.len() { input[ABSOLUTE_POS] } else { F::from_int(0i64) };
    let mut i = 1u32;
    while i < PLANE_DIM {            // adapts to 4/8/16/32/64/128 — NO hardcoded 32
        acc += plane_shuffle_xor(acc, i);
        i *= 2u32;
    }
    if UNIT_POS_PLANE == 0u32 { output[PLANE_POS as usize] = acc; }
}
// Or the higher-level intrinsic: plane_sum(val) where available.
```
**Path B — Shared-memory tree fallback (works where subgroups unsupported):**
```rust
// Source: CITED Cubecl_shared_memory.md §3 — log₂ tree, pairwise-stable
#[cube(launch)]
pub fn reduce_sum_shared<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let mut shared = SharedMemory::<F>::new(256usize);   // comptime size = max CubeDim.x
    let tid = UNIT_POS_X; let gid = ABSOLUTE_POS_X;
    shared[tid as usize] = if (gid as usize) < input.len() { input[gid as usize] } else { F::from_int(0i64) };
    sync_cube();
    let mut s = CUBE_DIM_X / 2u32;
    while s > 0u32 {
        if tid < s { let v = shared[(tid + s) as usize]; shared[tid as usize] += v; }
        sync_cube();
        s /= 2u32;
    }
    if tid == 0u32 { output[CUBE_POS_X as usize] = shared[0usize]; }
}
```
**Both paths produce per-cube partials → a second pass (or host fold in tests) finalizes the total.** Axis-wise reductions (D-01) map rows/cols onto `CUBE_POS`/grid dims (one cube per row for row-reduce; transpose the launch geometry for column-reduce, reusing the row kernel). argmin (D-02) uses the same two structures but carries `(value, index)` and a lowest-index tie-break in the combine.

**wgpu subgroup caveat (Pitfall, plane manual §6):** plane intrinsics require the wgpu adapter's `subgroups` feature. The shared-memory path is the guaranteed-portable fallback. The test for the plane path should query subgroup support (analogous to `capability::supports_f64`) and `skip_with_log` if absent — mirroring the f64 gate — so "both paths pass on wgpu" means "both pass where the adapter supports them, and the unsupported one is logged-skipped, never failed." **Confirm the exact subgroup capability-query symbol in cubecl 0.10 during Wave 0** (analogous to `client.properties().supports_type` for f64; the plane/subgroup feature query is a separate property — verify the symbol, it was not needed in Phase 1).

## D-10 Memory-Gate Assertion Strategy

The gate is a single test file (`memory_gate_test.rs`) asserting on the **existing** `BufferPool::stats()` API (`PoolStats{allocations, reuses, peak_bytes, live_bytes}` — verified in `pool.rs`/`pool_test.rs`). Three HARD assertions (D-10), build-failing:

1. **Reuse > 0, allocations bounded.** Run a primitive (e.g. distance) N times at the *same shape*, threading the SAME `BufferPool` and reusing the caller-provided out-buffer + scratch (D-11). Assert `stats.reuses > 0` AND `stats.allocations` is **bounded** (≈ constant, not `∝ N`). Concretely: `assert!(stats.allocations <= FIRST_ITER_ALLOCS)` after iteration 2..N, and `assert!(stats.reuses >= N - 1)`. *Mechanism that makes this true:* primitives must `pool.acquire`/`pool.release` every scratch buffer (so released scratch of a given byte-size is reused next iteration), and the out-buffer is supplied by the caller (so it isn't reallocated). This is the realistic allocation pattern Phase 1 D-05 was waiting for.

2. **No mid-pipeline host round-trip.** Build a chained pipeline GEMM→reduce→distance entirely through `DeviceArray`→`DeviceArray` (no `to_host` between stages). Assert correctness via a single terminal `to_host` only. To make "zero host read-backs between stages" *checkable*, the cleanest mechanism is a read-back counter on the pool/client wrapper: add a `read_backs: u64` counter incremented in a `to_host`/`read_one` wrapper, and assert it equals exactly 1 (the terminal compare) across the whole pipeline. **This requires a tiny addition** — either instrument `DeviceArray::to_host` to bump a pool counter, or assert structurally that no intermediate `to_host` call exists. Recommend the counter (a `read_backs` field on `PoolStats`) so the gate is a real runtime assertion, not just a code-review claim.

3. **Gram reuses GEMM buffer.** Covariance (`Aᵀ·A`) must write into / reuse the GEMM output buffer rather than allocating a parallel one. Assert that computing covariance after a GEMM of the same output shape does **not** increase `stats.allocations` beyond the GEMM's own (i.e. the Gram output handle is the reused GEMM handle, or drawn from the free-list at the same byte size → `reuses` bumps, `allocations` does not). The D-11 caller-provided-out-buffer mechanism makes this directly testable: pass the GEMM's output `DeviceArray` as covariance's out-buffer.

**Planner note:** Assertions 1 and 3 are satisfiable today with the existing pool API. Assertion 2's runtime form needs a small `read_backs` counter addition (Claude's discretion under D-11/D-10) — surface this as an explicit Wave-0 task so it isn't discovered mid-implementation.

## Per-Primitive Implementation Approach (dependency order)

**Wave order is forced by D-07/D-09:** GEMM is the substrate for both distance and covariance, so it is built and validated first; reductions are independent but feed distance (row-L2-norm), so they come second; distance and covariance compose both.

### 1. GEMM (PRIM-01) — FIRST, gated by Open Question 1
- **Decide** (checkpoint): wrap `cubecl-matmul` (only if a 0.10-compatible source is pinned) vs hand-write tiled GEMM (recommended default).
- **Host API** (`mlrs-backend/src/prims/gemm.rs`): `gemm(pool, a: &DeviceArray, (m,k), b: &DeviceArray, (k,n), transa, transb, out: Option<DeviceArray>) -> DeviceArray`. Asserts `rows*cols == len` per operand (D-04). Scratch + out via pool/out-buffer (D-11).
- **Transpose:** `swap_dims(0,1)` (matmul) OR `#[comptime]` index-swap (hand-written) — NO transpose buffer (D-06).
- **Validate:** host-ref triple-loop sweep over random shapes/seeds + one committed GEMM npz fixture (D-12), f32 always, f64 behind `skip_f64_with_log`, cpu+wgpu.

### 2. Reductions (PRIM-02) — SECOND, dual-path
- **Kernels** (`mlrs-kernels/src/reduce.rs`): `reduce_*_plane` + `reduce_*_shared` for sum/min/max/L2-norm; `argmin/argmax` row-wise + full (lowest-index tie-break). mean = sum then scale (two-pass for stability).
- **Host API** (`mlrs-backend/src/prims/reduce.rs`): full-array + row-reduce + column-reduce (D-01); a `ReducePath` selector so tests exercise BOTH paths (D-03). Subgroup-capability skip for the plane path (mirror f64 gate).
- **Validate:** host-ref Σ/min/max/argmin loops over random shapes incl. large-N stability cases + a deliberate-tie argmin fixture; BOTH paths asserted on wgpu.

### 3. Distance (PRIM-03) — THIRD, composes GEMM + reduction
- **Orchestration** (`mlrs-backend/src/prims/distance.rs`): `XYᵀ` via GEMM(transb=true) → `‖x_i‖²`,`‖y_j‖²` via row-L2-norm reduction → combine `‖x‖²+‖y‖²−2XYᵀ` → `clamp_nonneg` kernel (`mlrs-kernels/src/elementwise.rs`) → optional `sqrt_elem` (D-08). All device-resident (D-05).
- **Validate:** host-ref direct distance loop; property test `min >= 0` (Criterion 3); squared-vs-sqrt npz fixture (D-12); f32 cancellation case.

### 4. Covariance / XᵀX (PRIM-04) — FOURTH, GEMM + ddof
- **Orchestration** (`mlrs-backend/src/prims/covariance.rs`): center columns (column-mean reduction, D-01) → `Aᵀ·A` via GEMM(transa=true) → scale by `1/(n-ddof)` (scale kernel or fold into a kernel). Reuse GEMM output buffer (D-10 gate 3).
- **Validate:** `np.cov` ddof=1 AND population ddof=0 npz fixtures (D-12) + host-ref; f32/f64, cpu/wgpu.

### 5. Memory gate (D-10) — FINAL verification wave
- `memory_gate_test.rs` with the three assertions above; add the `read_backs` counter in Wave 0.

## Code Examples

### Distance combine + clamp (the visible D-07 signature)
```rust
// Source: synthesized from D-07 + Cubecl_conditionals.md (CITED) — clamp as statement, not expr
#[cube(launch)]
pub fn dist_combine_clamp<F: Float + CubeElement>(
    xy: &Array<F>,        // -2·XYᵀ already? or raw XYᵀ — pass scaled
    xnorm: &Array<F>, ynorm: &Array<F>,
    out: &mut Array<F>, rows: u32, cols: u32,
) {
    let i = ABSOLUTE_POS_X; let j = ABSOLUTE_POS_Y;
    if i < rows && j < cols {
        let idx = (i * cols + j) as usize;
        let mut d = xnorm[i as usize] + ynorm[j as usize] - F::new(2.0) * xy[idx];
        let zero = F::from_int(0i64);
        if d < zero { d = zero; }      // max(d², 0) — Criterion 3, statement form
        out[idx] = d;
    }
}
```

### f64-gated oracle test skeleton (reuse Phase 1 gate)
```rust
// Source: CITED capability.rs (skip_f64_with_log) + spike_test.rs launch idiom
#[test]
fn gemm_f64_matches_host_ref() {
    let _ = env_logger::builder().is_test(true).try_init();
    if mlrs_backend::capability::skip_f64_with_log() { return; } // skip, never fail
    // ... build DeviceArray<_, f64> inputs, run gemm, to_host, assert_slice_close(F64_TOL)
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `cubecl-matmul` lives in the cubecl monorepo (`tracel-ai/cubecl/crates/cubecl-matmul`) | matmul/reduce/linalg crates removed from the monorepo by the `v0.10.0` tag; no 0.10 matmul release | cubecl 0.10 line (2026) | The roadmap's "wrap `cubecl-matmul`" is not directly satisfiable at cubecl 0.10 — hand-write or pin a compatible source. |
| matmul manual `launch::<R, f32>(strategy, client, lhs, rhs, out)` | 0.9-pre.5 `launch::<R>(strategy, client, lhs, rhs, out, dtypes: MatmulElems) -> Result` | 0.9 line | Manual example is stale; signature carries a precision spec and returns `Result`. |

**Deprecated/outdated:**
- `cubecl_matmul_gemm_example.md` `launch` signature (no `dtypes`, `client.create_tensor`) — superseded; conceptual reference only.
- Assuming the `cubecl` umbrella pulls matmul/reduce — it does not (verified: umbrella deps + lib.rs re-exports).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `cubecl-matmul`'s 0.10-line `swap_dims`/`launch`/`MatmulElems` surface matches the cached 0.9.0-pre.5 source | Patterns/Standard Stack | If a 0.10 matmul is later located, its API may differ — re-pin against the actual chosen version before coding GEMM. (Mitigated by recommending hand-written GEMM, which has no such dependency.) |
| A2 | `cubecl-std 0.10` `TensorHandle` is reachable from the workspace (via `feature="stdlib"`/`cubecl::std`) | Supporting stack | If not re-exported as needed, the matmul-wrap path needs `cubecl-std` added explicitly; hand-written GEMM avoids `TensorHandle` entirely. |
| A3 | cubecl 0.10 exposes a subgroup/plane capability query analogous to `supports_type` | Dual-Path Reduction | If the symbol differs/absent, the plane-path skip gate needs another mechanism — verify in Wave 0 (the plane intrinsics themselves are confirmed in the manual). |
| A4 | matmul f64 (`type Lhs=(f64,f32)`) may breach 1e-5 on large K due to f32 accumulation | Pitfall 2 | If it does, use `Strategy::Naive`/hand-written f64-accumulator GEMM. Low risk on cpu, watch on wgpu. |
| A5 | No cubecl-0.10-compatible `cubecl-matmul` source exists anywhere | Open Question 1 | If one DOES exist (e.g. an unreleased tracel-ai branch), git-pinning it becomes viable and the hand-written fallback is unnecessary. Needs human/network confirmation. |

## Open Questions

1. **`cubecl-matmul` for cubecl 0.10 — does a compatible source exist? (HIGHEST LEVERAGE — gates GEMM + covariance + distance)**
   - What we know: No crates.io `cubecl-matmul` 0.10; latest `0.9.0-pre.5` pins `cubecl-core =0.9.0-pre.5`; the crate is absent from `cubecl` repo `v0.10.0` tag and from `main` (now 0.11.0-pre.1). The workspace is firmly on `cubecl 0.10.0` and builds. `swap_dims` (transpose-by-strides) and f64 `MatmulPrecision` are confirmed in the 0.9-pre.5 source.
   - What's unclear: whether tracel-ai publishes/branches a 0.10-line matmul anywhere reachable by git-pin; whether downgrading cubecl to 0.9-pre is acceptable (re-validates Phase 1).
   - **Recommendation:** Make this a `checkpoint:human-verify` **decision task** in GEMM Wave 0. **Default to hand-writing a tiled GEMM in `mlrs-kernels`** (Pattern 3): zero new deps, no version coupling, naturally `<F: Float>`-generic + feature-free, transpose becomes index arithmetic (cleaner D-06), and the host-ref oracle already provides the correctness gate the roadmap wants. Only pursue `cubecl-matmul` if a human confirms a 0.10-compatible source. **Update REQUIREMENTS/ROADMAP wording** ("wraps `cubecl-matmul`") if hand-written GEMM is chosen — note this for the orchestrator.

2. **f64 GEMM precision via matmul mixed registers (only if matmul path chosen)**
   - What we know: `impl MatmulPrecision for f64 { type Lhs = (f64, f32) }` — f64 input, f32 accumulation.
   - What's unclear: whether f32 accumulation holds 1e-5 on large-K f64 GEMM.
   - Recommendation: if wrapping matmul, validate f64 against an f64 host ref on a large-K case early; fall back to `Strategy::Naive` or hand-written f64-accumulator. Hand-written GEMM with an `F`-typed accumulator sidesteps this entirely.

3. **Subgroup capability-query symbol in cubecl 0.10 (Wave-0 spike for the plane path)**
   - What we know: plane intrinsics (`plane_shuffle_xor`, `plane_sum`, `PLANE_DIM`) are documented and require adapter subgroup support; f64 uses `client.properties().supports_type`.
   - What's unclear: the exact property/query for subgroup support (separate from f64).
   - Recommendation: short Wave-0 probe (mirror `spike_capability_query_reports_f64`); if absent, gate the plane-path test by attempting the launch and skip-with-log on the documented unsupported-feature error.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `cubecl` (cpu) | All kernels (CI gate) | ✓ | 0.10.0 | — |
| `cubecl` (wgpu) | Criteria 1–4 (primary gate) | ✓ (Phase 1 ran wgpu f64 on AMD RADV GFX1152) | 0.10.0 | cpu still gates correctness |
| wgpu subgroup feature | reduction plane path (D-03) | unverified this session | — | shared-memory path (always available) |
| `cubecl-matmul` 0.10-compatible | GEMM-via-matmul (D-06 wrap path) | ✗ | none exists | **hand-written tiled GEMM (recommended)** |
| numpy (`gen_oracle.py`) | regenerating D-12 npz fixtures | ✗ at test time by design (D-03/Phase 1: hermetic) | — | committed `.npz` blobs; regen via /tmp venv (PEP 668, per MEMORY.md) |
| slopcheck / pip | package legitimacy audit | ✗ (offline) | — | first-party crates verified via crates.io API + cached source |

**Missing dependencies with no fallback:** none that block — the matmul gap has the hand-written-GEMM fallback.
**Missing dependencies with fallback:** `cubecl-matmul` 0.10 → hand-written GEMM; wgpu subgroups → shared-memory reduction path; numpy → committed npz fixtures.

## Validation Architecture

> `.planning/config.json` not inspected for `workflow.nyquist_validation`; treating as enabled (absent ⇒ enabled). Phase 1's test layout is the template.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (cargo test), integration tests in `crates/*/tests/` |
| Config file | none — standard cargo layout (Phase 1 convention) |
| Quick run command | `cargo test -p mlrs-backend --features cpu <test_name>` |
| Full suite command | `cargo test -p mlrs-backend --features cpu && cargo test -p mlrs-backend --features wgpu` |
| Phase gate | full suite green on cpu AND wgpu, f32 always + f64 where `skip_f64_with_log` permits |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PRIM-01 | GEMM matches host ref + fixture, f32/f64, cpu/wgpu, transpose flags | integration | `cargo test -p mlrs-backend --features cpu gemm` | ❌ Wave 0 |
| PRIM-02 | reductions both paths pass on wgpu, stable large-N, argmin tie=lowest | integration | `cargo test -p mlrs-backend --features wgpu reduce` | ❌ Wave 0 |
| PRIM-03 | distance `min>=0` + host ref + sqrt fixture | integration | `cargo test -p mlrs-backend --features cpu distance` | ❌ Wave 0 |
| PRIM-04 | covariance ddof=0/1 fixtures + host ref | integration | `cargo test -p mlrs-backend --features cpu covariance` | ❌ Wave 0 |
| D-10 | reuse>0, allocations bounded, no mid-pipeline read-back, Gram reuses GEMM buf | integration | `cargo test -p mlrs-backend --features cpu memory_gate` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p mlrs-backend --features cpu <prim>_test`
- **Per wave merge:** `cargo test -p mlrs-backend --features cpu && --features wgpu`
- **Phase gate:** full suite green on both backends before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `tests/gemm_test.rs`, `tests/reduce_test.rs`, `tests/distance_test.rs`, `tests/covariance_test.rs`, `tests/memory_gate_test.rs` — cover PRIM-01..04 + D-10
- [ ] `PoolStats.read_backs` counter (or a read-back-instrumented `to_host`) — enables D-10 gate 2 as a runtime assertion
- [ ] Subgroup capability-query probe (mirror `spike_capability_query_reports_f64`) for the plane-path skip gate
- [ ] D-12 convention `.npz` fixtures (GEMM, distance squared/sqrt, cov ddof=0/1) via `gen_oracle.py` (/tmp venv, PEP 668)
- [ ] **GEMM substrate DECISION task** (Open Question 1) before any GEMM code — `checkpoint:human-verify`
- [ ] Framework install: none — cargo test already in use.

## Security Domain

Not applicable in the conventional sense — this is a numerical compute-kernel phase with no auth/session/network/PII surface. The only `unsafe` is `ArrayArg::from_raw_parts(handle, len)` where `len` is derived from validated `DeviceArray.len` (T-04-01 mitigation, Phase 1). V5 (Input Validation) maps to the `rows*cols == len` shape assertion (D-04). No new threat surface; `security_enforcement` config not located this session — flag if the project expects this section populated.

## Sources

### Primary (HIGH confidence)
- `crates/mlrs-kernels/src/smoke.rs`, `crates/mlrs-backend/{src,tests}/*` — verified existing patterns (launch idiom, pool API, capability gate, device array).
- `cargo build -p mlrs-backend --features cpu` — workspace builds on cubecl 0.10 today.
- `cubecl-matmul-0.9.0-pre.5/src/base.rs` (cached) — `swap_dims` (:369), `launch`/`launch_ref` (:545/:571), `Strategy`, `MatmulInputHandle`; `src/components/spec.rs` — `MatmulElems`, `impl MatmulPrecision for f64`.
- `git ls-tree v0.10.0` / `git show main:Cargo.toml` (tracel-ai/cubecl) — matmul/reduce absent at 0.10 tag; main is 0.11.0-pre.1.
- crates.io API (`/api/v1/crates/cubecl`, `/cubecl-matmul`) — version + date confirmation.
- CubeCL manuals (`Cubecl_plane.md`, `Cubecl_shared_memory.md`, `Cubecl_transpose.md`, `cubecl_reduce_sum.md`, `Cubecl_dot.md`, `cubecl_matmul_gemm_example.md`) — kernel patterns.
- `.planning/phases/02-core-compute-primitives/02-CONTEXT.md`, `01-CONTEXT.md`, `REQUIREMENTS.md`, `STATE.md`, `ROADMAP.md`, `AGENTS.md`, `CLAUDE.md`.

### Secondary (MEDIUM confidence)
- `cubecl-std-0.10.0/src/tensor/handle.rs` (cached) — `TensorHandle` 0.10 API (reachability from umbrella unconfirmed → A2).

### Tertiary (LOW confidence)
- Inference that matmul/reduce were "merged into Burn" — burn consumes them as deps; their canonical 0.10 home was not definitively located (feeds A5/Open Question 1).

## Metadata

**Confidence breakdown:**
- GEMM substrate blocker (Open Question 1): HIGH — multiple independent verifications (crates.io, git tags, cached Cargo.toml pins, live build).
- matmul transpose/f64 API: MEDIUM — verified in cached 0.9-pre.5; 0.10 surface unconfirmed (no 0.10 exists). Mitigated by recommending hand-written GEMM.
- Dual-path reduction mechanics: HIGH — directly from plane + shared-memory manuals + verified launch idiom.
- D-10 memory gate: HIGH — asserts on the existing `PoolStats` API (verified in `pool.rs`/`pool_test.rs`); one small `read_backs` addition flagged.
- Distance/covariance composition: HIGH — pure composition of the above with a small clamp/scale kernel.

**Research date:** 2026-06-12
**Valid until:** ~2026-06-26 (14 days — cubecl/matmul are fast-moving pre-release; re-verify the matmul version situation before GEMM Wave 0).

## RESEARCH COMPLETE
