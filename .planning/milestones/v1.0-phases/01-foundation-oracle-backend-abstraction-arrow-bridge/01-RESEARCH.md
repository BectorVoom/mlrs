# Phase 1: Foundation — Oracle, Backend Abstraction, Arrow Bridge - Research

**Researched:** 2026-06-11
**Domain:** Rust/CubeCL generic GPU compute infrastructure, Apache Arrow zero-copy bridge, scikit-learn oracle harness, custom global allocator, Cargo workspace architecture
**Confidence:** HIGH (CubeCL patterns, Arrow validation, allocator, workspace) / MEDIUM (exact CubeCL 0.10 capability-API symbol names, npz reader choice)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** scikit-learn reference values are **pre-generated as committed fixtures**, not computed live at test time. A checked-in `scripts/gen_oracle.py` produces reference outputs from seeded inputs; Rust tests load the committed files and compare.
- **D-02:** Fixture format is **NumPy `.npz`** (bundled named arrays per case, e.g. `linreg_f64_seed42.npz` carrying `X`, `y`, `coef_`, `intercept_`). Read in Rust via an npy/npz reader crate (latest).
- **D-03:** Fixtures are **committed binary blobs**; `scripts/gen_oracle.py` regenerates on demand. **CI runs Rust tests against the committed files — no Python/sklearn needed in the test job.**
- **D-04:** Build the **buffer-reuse/pool layer in Phase 1** (free-list/arena over CubeCL buffers), not just a thin wrapper. `DeviceArray<R,F>` over the pool; zero-copy ingest from the validated Arrow bridge; host read-back.
- **D-05:** The pool exposes a **stats/counters API** (allocations, reuses, peak bytes). In Phase 1 these counters are **logged only** — hard reuse assertions are **deferred to Phase 2**.
- **D-06:** **Hard-reject only.** Zero-copy ingest is the *only* path. Non-conforming input (non-zero offset / slice, set null bits, misaligned buffer) returns a typed `Err` **before any unsafe transmute**. No compacting-copy escape hatch in Phase 1.
- **D-07:** Bridge errors are a typed **`thiserror`** enum (`BridgeError`) with variants for each violation class (e.g. `HasNulls`, `Offset`, `Misaligned`).
- **D-08:** Start with a **single global tolerance** (`F32_TOL` and `F64_TOL`, both `abs = 1e-5, rel = 1e-5`) rather than a per-family table. FOUND-08's "per-family policy" is satisfied by a policy *structure* that can grow rows.
- **D-09:** `assert_close` requires **both abs AND rel error to pass** (the stricter form), not numpy-style abs-OR-rel. ⚠ **Implementation consideration:** include a **near-zero guard** (fall back to abs-only when `|expected|` is below a small floor) so genuinely-correct near-zero results don't spuriously fail. Design this into `assert_close` from the start.
- **D-10:** Use **`thiserror`** for typed error enums in library crates (`mlrs-core`/`mlrs-kernels`/`mlrs-backend`/`mlrs-algos`) and **`anyhow`** at application / PyO3-binding boundaries (`mlrs-py`, binaries, `scripts`-driven Rust). **All Cargo dependencies track latest versions** — do not pin old versions.

### Claude's Discretion
- Choice of the trivial smoke-test kernel (e.g. SAXPY / elementwise add) — any minimal `#[cube]` kernel generic over `<F: Float>` that exercises the full pipeline.
- Specific npy/npz reader crate and Arrow crate versions (latest of each).
- Exact `BridgeError` variant names and the pool's internal data structure (free-list vs arena).
- f64 capability-gate skip-vs-xfail mechanics on wgpu adapters lacking `SHADER_F64`.
- Near-zero guard floor value in `assert_close`.

### Deferred Ideas (OUT OF SCOPE)
- **Per-estimator-family tolerance tables** — deferred from D-08; introduce in Phase 3/4/5.
- **Compacting-copy Arrow ingest path** — deferred from D-06; revisit in Phase 6 if PyCapsule path surfaces non-conforming inputs.
- **Hard buffer-reuse assertions** — deferred from D-05 to Phase 2.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FOUND-01 | Cargo workspace splits kernels/backend/algorithms/bindings into single-responsibility crates (`mlrs-core`, `mlrs-kernels`, `mlrs-backend`, `mlrs-algos`, `mlrs-py`) | Workspace Layout section: virtual manifest, `[workspace.dependencies]` for shared version pinning, dependency DAG, feature propagation |
| FOUND-02 | Compute kernels generic over float (`f32`/`f64`) AND over the CubeCL runtime, in a feature-free kernels crate | CubeCL Generic Kernel pattern: `#[cube(launch)] fn k<F: Float>(...)`, launch `::<F, R>`; `mlrs-kernels` depends only on `cubecl` core (no runtime feature flags) |
| FOUND-03 | Backend selected via Cargo features (`cuda`, `rocm`, `wgpu`, `cpu`); `cuda` compiles (no run); `wgpu`+`cpu` run | Feature-Flag Propagation pattern; `mlrs-backend` owns runtime selection; cargo `--features` matrix in CI |
| FOUND-04 | Capability layer queries runtime support (f64/plane/subgroup), gates/skips paths the backend can't run | Capability Layer section: `client.properties().feature_enabled(Feature::Type(Elem::Float(FloatKind::F64)))`; `feature_enabled(FloatKind::F64)` façade; skip/xfail mechanics |
| FOUND-05 | Memory-efficient device-array abstraction wraps CubeCL buffers with reuse and minimal copies | DeviceArray + Buffer Pool section; CubeCL `client.create`/`empty`/`read`; counters API |
| FOUND-06 | Arrow buffers feed CubeCL zero-copy with validation of offsets/nulls/alignment before unsafe transmute | Arrow Zero-Copy Bridge section: `offset()`, `nulls()`, alignment check, `bytemuck::cast_slice` |
| FOUND-07 | Oracle harness: seeded random inputs → sklearn reference → asserts abs/rel ≤ 1e-5 | Oracle Harness section: `scripts/gen_oracle.py` + npz fixtures + `assert_close` |
| FOUND-08 | Sign-flip (SVD/PCA) + label-permutation (clustering) helpers + documented per-estimator f32 tolerance policy | Comparison Helpers section + Tolerance Policy structure (D-08) |
| FOUND-09 | Custom global allocator (mimalloc) wired in; source/test in separate files per AGENTS.md | mimalloc section: `#[global_allocator]` in `mlrs-py`; test in `tests/` not `mod tests` |
</phase_requirements>

## Summary

Phase 1 is greenfield Rust infrastructure. Nothing exists yet (verified: repo has only `cuml-main/` reference, `.planning/`, no `Cargo.toml`). The work is to stand up a five-crate Cargo workspace where a single `#[cube]` kernel generic over `<F: Float>` and `<R: Runtime>` compiles once in a backend-feature-free `mlrs-kernels` crate and runs on cpu + wgpu, fed zero-copy from validated Apache Arrow buffers, with a hermetic scikit-learn oracle (committed `.npz` fixtures, no Python at test time), an f64 capability gate, a buffer-reuse pool with counters, and the mimalloc global allocator.

The dominant technical risk is **CubeCL `#[cube]` macro fragility**, not algorithmic complexity. The project's mandatory CubeCL manuals (read in full) and the error-solution guide give concrete, version-current patterns: generic kernels use `<F: Float>` (or `<N: Numeric>`) with `F::new(literal)` / `N::from_int(int)`; `if`-expressions-as-values are prohibited (must be `let mut` + `if` statement); math methods like `.exp()`/`.sqrt()` must be called as associated functions (`F::exp(x)`); device code must avoid `usize` and host-only types. The capability-query API pattern (from the half-precision manual, directly transferable to f64) is `client.properties().feature_enabled(Feature::Type(Elem::Float(FloatKind::F64)))`.

The second-largest risk is a **hard contradiction between the manuals and AGENTS.md**: every CubeCL/optimisor manual example embeds `#[cfg(test)] mod tests` at the bottom of the source file, but AGENTS.md *strictly prohibits* this. The implementation MUST follow AGENTS.md (tests in `tests/` or `*_test.rs`), using the manual code as logic reference only, never copying the test-module structure.

**Primary recommendation:** Build the workspace as a virtual manifest with a `[workspace.dependencies]` table (single source of truth for latest versions per D-10). Put the generic kernel in `mlrs-kernels` (depends only on `cubecl` with NO runtime features). Put runtime selection, capability layer, DeviceArray, buffer pool, and Arrow bridge in `mlrs-backend` (owns the `cpu`/`wgpu`/`cuda`/`rocm` Cargo features). Put `assert_close`, sign-flip/label-permutation helpers, tolerance constants, and npz loading in `mlrs-core` (consumed by every crate's tests). Wire mimalloc only in `mlrs-py`. Use `arrow` 59, `bytemuck` 1.25, `cubecl` 0.10, `mimalloc` 0.1.52, `thiserror` 2, `anyhow` 1, `rand` 0.9.x (see version note), and `ndarray-npy` 0.10 (npz with named arrays) for the npz reader.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Generic `#[cube]` smoke kernel | `mlrs-kernels` | — | FOUND-02: feature-free, generic over `<F: Float>` and `<R: Runtime>`; no runtime selection here |
| Runtime/backend selection (cpu/wgpu/cuda/rocm) | `mlrs-backend` | Cargo features | FOUND-03: features live where the concrete `Runtime` type is chosen |
| Capability query (f64/plane support) | `mlrs-backend` | `mlrs-core` (FloatKind enum) | FOUND-04: needs a live `ComputeClient`, which only `mlrs-backend` instantiates |
| DeviceArray + buffer-reuse pool | `mlrs-backend` | `mlrs-core` (stats types) | FOUND-05: wraps CubeCL `client` handles; pool lives next to the client |
| Arrow zero-copy bridge + validation | `mlrs-backend` | `mlrs-core` (`BridgeError`) | FOUND-06: bridge produces device buffers, so it sits with the device layer; error enum can live in core |
| `assert_close`, sign-flip, label-perm, tolerances | `mlrs-core` | — | FOUND-07/08: pure host-side comparison logic consumed by every crate's tests |
| npz fixture loading | `mlrs-core` (test-support) | — | D-02/D-03: shared test utility; `scripts/gen_oracle.py` is the Python generator |
| mimalloc global allocator | `mlrs-py` | — | FOUND-09: a binary/cdylib must own `#[global_allocator]`; libraries must not |
| Trivial end-to-end pipeline test | `mlrs-algos` or `mlrs-backend` integration test | `mlrs-core` (oracle), `mlrs-kernels` (kernel) | Success Criterion 2: exercises Arrow→device→kernel→read-back→oracle across crates |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.10.0 | Generic GPU compute language; `#[cube]` kernels, runtimes (cpu/wgpu/cuda/rocm), `ComputeClient` | The project's mandated and only allowed device-kernel layer (CLAUDE.md constraint) `[VERIFIED: cargo search → cubecl = "0.10.0"]` |
| `arrow` | 59.0.0 | Apache Arrow Rust impl; `Float32Array`/`Float64Array`, `PrimitiveArray`, buffer access | Mandated data interchange (CLAUDE.md); 59 is current latest `[VERIFIED: cargo search → arrow = "59.0.0"]` |
| `bytemuck` | 1.25.0 | Safe zero-copy `&[T] ↔ &[u8]` transmutation for Arrow→CubeCL handoff | Used by every CubeCL/optimisor manual; validates alignment+size before recast `[VERIFIED: cargo search → bytemuck = "1.25.0"]` |
| `thiserror` | 2.0.18 | Typed error enums in library crates (`BridgeError`, `CapabilityError`, etc.) | Mandated by D-07/D-10; v2 is current `[VERIFIED: cargo search → thiserror = "2.0.18"]` |
| `anyhow` | 1.0.102 | Boundary error handling in `mlrs-py` and `scripts`-driven Rust | Mandated by D-10 `[VERIFIED: cargo search → anyhow = "1.0.102"]` |
| `mimalloc` | 0.1.52 | Custom global allocator wired in `mlrs-py` | Mandated by FOUND-09 + MIMALLOC_MANUAL.md; 0.1.52 is current `[VERIFIED: cargo search → mimalloc = "0.1.52"]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `ndarray-npy` | 0.10.0 | Read `.npz` archives with **named** arrays via `NpzReader::by_name` | Primary recommendation for D-02 npz fixtures (named arrays per case) `[VERIFIED: cargo search → ndarray-npy = "0.10.0"]` |
| `ndarray` | latest (0.16.x family — verify) | Backing array type for `ndarray-npy` output | Required transitively if using `ndarray-npy` `[ASSUMED]` |
| `npyz` | 0.9.1 | Alternative npz reader (`NpzArchive`, `by_name`) with `npz` feature; no `ndarray` dependency | Use instead of `ndarray-npy` if avoiding the `ndarray` dependency is preferred `[VERIFIED: cargo search → npyz = "0.9.1"]` |
| `rand` | 0.9.x (NOT 0.10 — see note) | Seeded RNG for oracle fixtures (Rust side, if any) and any Rust-side random fixtures | FOUND-07 seeded fixtures `[VERIFIED: cargo search shows rand = "0.10.1"; see version note below]` |
| `log` + `env_logger` (or `tracing` + `tracing-subscriber`) | latest | Logging the f64 dtype/backend selection (Criterion 4) and pool counters (D-05) | Criterion 4 requires "CI log shows which dtype ran on which backend"; D-05 logs counters `[ASSUMED]` |
| `half` | 2.x | Only if f16/bf16 ever needed (NOT a v1 deliverable) | Out of scope for Phase 1; listed for completeness `[CITED: HALF_PRECISION_CUBECL.md]` |

> **`rand` version note (IMPORTANT):** `cargo search` reports `rand = "0.10.1"` as newest, but `rand` 0.8→0.9→0.10 has had repeated breaking API changes (`thread_rng`→`rng`, `gen`→`random`, `SeedableRng` paths). D-10 says "track latest," but the planner should add a `checkpoint:human-verify` or a small spike task to confirm the exact seeded-RNG API for the chosen `rand` version, since the oracle's reproducibility depends on it. Most Rust-side randomness in Phase 1 is minimal — the *authoritative* seeded RNG is NumPy's in `gen_oracle.py` (Python side), so Rust `rand` may only be needed for non-oracle fixtures. **Prefer generating all reference inputs in Python (`numpy.random.default_rng(seed)`) and committing them**, minimizing Rust RNG surface. `[ASSUMED]`

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `ndarray-npy` | `npyz` (+`npz` feature) | `npyz` avoids the `ndarray` dependency and reads into plain `Vec<T>`; `ndarray-npy` is more ergonomic if you already want `ndarray` 2D arrays. Both support named-array `.npz` via `by_name`. Pick `npyz` for a leaner dependency tree, `ndarray-npy` for richer array ops in tests. `[VERIFIED: docs.rs/npyz npz module + docs.rs/ndarray-npy NpzReader]` |
| `mimalloc` | `jemalloc` (`tikv-jemallocator`) | FOUND-09 allows either; PROJECT.md/D-04 context and the optimisor manuals lean mimalloc (drop-in, predictable latency, used by `uv`). Recommend **mimalloc** for simplicity. `[CITED: MIMALLOC_MANUAL.md / JEMALLOC_MANUAL.md]` |
| `log`/`env_logger` | `tracing`/`tracing-subscriber` | `tracing` is richer (structured spans) and pairs well with pool counters; `log` is simpler. Either satisfies Criterion 4. `[ASSUMED]` |
| Single global tolerance constants | Per-family table now | D-08 explicitly locks single-global-now, structure-that-can-grow. Do NOT build the table. |

**Installation (illustrative `[workspace.dependencies]` — verify each version at implementation time per D-10):**
```toml
[workspace.dependencies]
cubecl       = { version = "0.10.0", default-features = false }
arrow        = "59"
bytemuck     = { version = "1", features = ["derive"] }
thiserror    = "2"
anyhow       = "1"
mimalloc     = "0.1"
ndarray-npy  = "0.10"   # or: npyz = { version = "0.9", features = ["npz"] }
log          = "0.4"
env_logger   = "0.11"   # verify latest
```

**Version verification performed:** `cargo search` run 2026-06-11 against crates.io for `cubecl`, `arrow`, `bytemuck`, `mimalloc`, `ndarray-npy`, `npyz`, `thiserror`, `anyhow`, `pyo3`, `maturin`, `rand`. Results recorded inline above. Re-verify at implementation time (D-10: track latest).

## Package Legitimacy Audit

> slopcheck was not available in this research environment (`pip install slopcheck` not run/confirmed). Per protocol, packages discovered via search/training are tagged `[ASSUMED]` even though `cargo search` confirms registry existence; the planner must gate any non-obvious install behind a `checkpoint:human-verify` task. All packages below are first-party, well-known crates with long histories, which substantially lowers risk.

| Package | Registry | Notes | Source Repo | slopcheck | Disposition |
|---------|----------|-------|-------------|-----------|-------------|
| `cubecl` | crates.io | Project-mandated; tracel-ai | github.com/tracel-ai/cubecl | n/a | Approved (mandated) |
| `arrow` | crates.io | Apache Arrow official Rust | github.com/apache/arrow-rs | n/a | Approved (mandated) |
| `bytemuck` | crates.io | Lokathor; ubiquitous | github.com/Lokathor/bytemuck | n/a | Approved |
| `thiserror` | crates.io | dtolnay; ubiquitous | github.com/dtolnay/thiserror | n/a | Approved (mandated) |
| `anyhow` | crates.io | dtolnay; ubiquitous | github.com/dtolnay/anyhow | n/a | Approved (mandated) |
| `mimalloc` | crates.io | purpleprotocol; mature | github.com/purpleprotocol/mimalloc_rust | n/a | Approved |
| `ndarray-npy` | crates.io | jturner314/ndarray ecosystem | github.com/jturner314/ndarray-npy | n/a | Approved |
| `npyz` | crates.io | ExpHP; fork of npy-rs | github.com/ExpHP/npyz | n/a | Approved (alt) |
| `rand` | crates.io | rust-random; ubiquitous | github.com/rust-random/rand | n/a | Approved (version TBD) |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none
**Cross-ecosystem caution:** `ndarray-npy` (correct) vs the unrelated `ndarray_npz` and `veks-io` crates that also appear in search; confirm the exact crate name `ndarray-npy` (hyphen) before install.

## Architecture Patterns

### System Architecture Diagram

```text
                         ┌──────────────────────────────────────────────┐
  scripts/gen_oracle.py  │  PYTHON (build/regen time only — NOT in CI    │
  numpy.random.default   │  test job)                                    │
  _rng(seed) + sklearn   │  writes  linreg_f64_seed42.npz {X,y,coef_,…}  │
                         └───────────────────────┬──────────────────────┘
                                                 │ committed binary blob
                                                 ▼
   tests/fixtures/*.npz  ───────────────────────────────────────────────┐
                                                                         │
   ┌─────────────────────────────────────────────────────────────────┐ │
   │ RUST TEST JOB (cpu + wgpu; NO python)                            │ │
   │                                                                 │ │
   │  Arrow input              mlrs-core::oracle                     │ │
   │  Float32Array /           load_npz(by_name) ◄──────────────────┼─┘
   │  Float64Array                  │                                │
   │       │                        │ reference arrays               │
   │       ▼                        ▼                                │
   │  mlrs-backend::bridge     assert_close(got, expected,          │
   │   validate offset/nulls/   F32_TOL|F64_TOL)                     │
   │   alignment  ──Err? reject │  ▲  + sign_flip / label_perm       │
   │       │ ok                  │  │                                 │
   │       ▼  bytemuck::cast_slice  │ host read-back                  │
   │  mlrs-backend::pool ──► client.create(Bytes)                    │
   │   (free-list, counters)        │                                │
   │       │ DeviceArray<R,F>       │                                │
   │       ▼                        │                                │
   │  mlrs-kernels::smoke_kernel<F: Float>                           │
   │   launch::<F, R>  ──────────► client.read ─► &[F]               │
   │       ▲                                                         │
   │  mlrs-backend::capability                                       │
   │   feature_enabled(FloatKind::F64)?  ──no──► skip/xfail f64 +log │
   └─────────────────────────────────────────────────────────────────┘
        R = CpuRuntime  (--features cpu)  |  WgpuRuntime (--features wgpu)
        mlrs-py: #[global_allocator] static GLOBAL: MiMalloc = MiMalloc;
```

### Recommended Project Structure
```
mlrs/
├── Cargo.toml                  # virtual manifest: [workspace] + [workspace.dependencies]
├── rust-toolchain.toml         # pin toolchain (rustc 1.95 available); optional but recommended
├── scripts/
│   └── gen_oracle.py           # Python: numpy default_rng(seed) + sklearn → *.npz (NOT run in CI test job)
├── crates/
│   ├── mlrs-core/              # NO backend features. Tolerances, assert_close, sign_flip,
│   │   ├── Cargo.toml          #   label_perm, BridgeError, FloatKind, npz loader (test-support)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── tolerance.rs    # F32_TOL, F64_TOL, Tolerance struct (growable policy, D-08)
│   │   │   ├── compare.rs      # assert_close (abs AND rel + near-zero guard, D-09)
│   │   │   ├── sign_flip.rs    # FOUND-08 SVD/PCA helper
│   │   │   ├── label_perm.rs   # FOUND-08 clustering helper
│   │   │   ├── oracle.rs       # npz fixture loader
│   │   │   └── error.rs        # BridgeError + other typed enums (thiserror)
│   │   └── tests/              # ALL tests here (AGENTS.md rule) — e.g. compare_test.rs
│   ├── mlrs-kernels/           # ZERO backend feature flags (Criterion 1). cubecl core only.
│   │   ├── Cargo.toml          #   cubecl with default-features=false
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   └── smoke.rs        # #[cube(launch)] fn smoke<F: Float>(...) — saxpy/elementwise
│   │   └── tests/
│   ├── mlrs-backend/           # Owns cpu/wgpu/cuda/rocm Cargo features (Criterion 1, FOUND-03)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── runtime.rs      # active Runtime/Device/client selection per feature
│   │   │   ├── capability.rs   # feature_enabled(FloatKind::F64) façade (FOUND-04)
│   │   │   ├── bridge.rs       # Arrow validation + zero-copy ingest (FOUND-06)
│   │   │   ├── device_array.rs # DeviceArray<R,F> (FOUND-05)
│   │   │   └── pool.rs         # free-list/arena + counters (D-04/D-05)
│   │   └── tests/              # integration: arrow→device→kernel→readback→oracle (Criterion 2)
│   ├── mlrs-algos/             # Empty/skeleton in Phase 1 (estimators are Phase 4+)
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   └── mlrs-py/                # PyO3 + maturin (skeleton in Phase 1). Owns mimalloc.
│       ├── Cargo.toml          #   crate-type = ["cdylib"]; anyhow at boundary
│       ├── src/
│       │   ├── lib.rs          # #[global_allocator] static GLOBAL: MiMalloc = MiMalloc; (FOUND-09)
│       │   └── allocator.rs    # (optional) allocator wiring separated
│       └── tests/              # allocator activation test (NOT mod tests)
└── tests/fixtures/            # committed *.npz oracle blobs (or per-crate tests/fixtures)
```

### Pattern 1: Generic `#[cube]` smoke kernel (FOUND-02, Criterion 2)
**What:** A single kernel generic over the float type, launched generic over the runtime.
**When to use:** The whole Phase-1 pipeline proof; reused as the verification vehicle in later phases.
```rust
// Source: cubecl_manual Cubecl_generics.md + Cubecl_axpy.md + ZERO_COPY_ARROW_CUBECL.md
// Lives in mlrs-kernels/src/smoke.rs — NO #[cfg(test)] mod tests in this file (AGENTS.md).
use cubecl::prelude::*;

/// SAXPY-style smoke kernel: y = a*x + y, generic over F.
#[cube(launch)]
pub fn saxpy_kernel<F: Float>(a: F, x: &Array<F>, y: &mut Array<F>) {
    let tid = ABSOLUTE_POS;
    if tid < x.len() {
        y[tid] = a * x[tid] + y[tid];
    }
}
// Launch generic over <F, R>: saxpy_kernel::launch::<F, R>(&client, count, dim, a, x_arg, y_arg)
// Generic-param order in generated launch: kernel params (F) first, then Runtime (R).
```

### Pattern 2: Backend selection by Cargo feature (FOUND-03, Criterion 1)
**What:** `mlrs-backend` resolves the concrete `Runtime`/`Device`/`client` from the active feature.
**When to use:** Any code that needs a live `ComputeClient`.
```rust
// Source: cubecl manuals (CpuRuntime/WgpuRuntime usage) — symbol paths verify at impl time.
// mlrs-backend/src/runtime.rs
#[cfg(feature = "cpu")]
pub use cubecl::cpu::{CpuRuntime as ActiveRuntime, CpuDevice as ActiveDevice};
#[cfg(feature = "wgpu")]
pub use cubecl::wgpu::{WgpuRuntime as ActiveRuntime, WgpuDevice as ActiveDevice};
#[cfg(feature = "cuda")]
pub use cubecl::cuda::{CudaRuntime as ActiveRuntime, CudaDevice as ActiveDevice};
// rocm analogous. Exactly one backend feature must be active.

pub fn active_client() -> cubecl::client::ComputeClient<
    <ActiveRuntime as cubecl::Runtime>::Server,
    <ActiveRuntime as cubecl::Runtime>::Channel,
> {
    let device = ActiveDevice::default();
    ActiveRuntime::client(&device)
}
```
> **ComputeClient type-parameter caveat:** CubeCL's `ComputeClient` generics changed across versions (some examples write `ComputeClient<R>`, others `ComputeClient<Server, Channel>`). Verify the exact signature against the installed 0.10 docs/source; prefer returning `impl`-bound or a type alias to insulate the rest of the code. `[ASSUMED]`

### Pattern 3: Arrow zero-copy bridge with hard-reject validation (FOUND-06, Criterion 3, D-06/D-07)
**What:** Validate offset==0, no nulls, and buffer alignment BEFORE any `bytemuck` transmute; return typed `BridgeError` otherwise.
```rust
// Source: ZERO_COPY_ARROW_CUBECL.md + arrow-rs PrimitiveArray/ArrayData docs.
// mlrs-backend/src/bridge.rs
use arrow::array::Float32Array;
use bytemuck;

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("array has non-zero offset ({0}); slice/offset not supported (compact first)")]
    Offset(usize),
    #[error("array has {0} null(s); nullable input not supported")]
    HasNulls(usize),
    #[error("buffer is misaligned for {dtype}")]
    Misaligned { dtype: &'static str },
}

pub fn validate_f32(arr: &Float32Array) -> Result<&[f32], BridgeError> {
    if arr.offset() != 0 {
        return Err(BridgeError::Offset(arr.offset()));
    }
    if arr.null_count() != 0 || arr.nulls().is_some() {
        return Err(BridgeError::HasNulls(arr.null_count()));
    }
    let slice: &[f32] = arr.values(); // O(1) view into ScalarBuffer
    // Alignment check BEFORE bytemuck transmute — bytemuck::try_cast_slice surfaces
    // misalignment as a recoverable Err instead of a panic:
    bytemuck::try_cast_slice::<f32, u8>(slice)
        .map_err(|_| BridgeError::Misaligned { dtype: "f32" })?;
    Ok(slice)
}
// Then: client.create(cubecl::bytes::Bytes::from_bytes_vec(bytemuck::cast_slice(slice).to_vec()))
```
> **Zero-copy nuance (flag for planner):** The manuals' example does `.to_vec()` when handing bytes to `cubecl::bytes::Bytes`, which is a *copy*, not strictly zero-copy on the host. True host-side zero-copy into CubeCL depends on whether 0.10's `Bytes`/`client.create` can borrow an existing allocation. Criterion 2/3 require "ingest zero-copy through the validated bridge"; the validation (reject before transmute) is unambiguous, but the literal zero-copy handoff may need a `client.create` API that accepts a borrowed/owned aligned buffer. **Research the exact `cubecl::bytes::Bytes` constructors at implementation time** and prefer the one that avoids the `.to_vec()` copy; if none exists in 0.10, document that the "zero-copy" guarantee is the validated-no-extra-copy-beyond-upload semantics. `[ASSUMED]`

### Pattern 4: f64 capability gate (FOUND-04, Criterion 4)
**What:** Query the live client for f64 support; skip/xfail f64 oracle tests with a logged reason on adapters lacking it.
```rust
// Source: HALF_PRECISION_CUBECL.md (f16 pattern — directly transferable to f64).
// mlrs-backend/src/capability.rs
use cubecl::ir::FloatKind;
use cubecl::Feature;          // verify exact path: cubecl::Feature vs cubecl::ir::Feature
use cubecl::ir::{Elem};       // verify exact path

pub fn supports_f64<R: cubecl::Runtime>(client: &ComputeClientFor<R>) -> bool {
    // Pattern from the half-precision manual (F16 → F64):
    client.properties().feature_enabled(Feature::Type(Elem::Float(FloatKind::F64)))
    // NOTE: the manual also shows an alt form: client.properties().features.supports_type(FloatKind::F64)
    // The two forms differ across 0.10 point releases — verify which compiles.
}

/// Public façade required by Criterion 4 wording: feature_enabled(FloatKind::F64)
pub fn feature_enabled(kind: FloatKind) -> bool { /* dispatch on active client */ }
```
**f64 skip/xfail mechanics (Claude's discretion, D-08 → recommend):**
- Standard Rust `#[test]` has no native xfail/skip. Recommended pattern: at the top of each f64 oracle test, `if !supports_f64(&client) { log::warn!("skipping f64 oracle on {backend}: SHADER_F64 unsupported"); return; }` — this is a **logged early-return skip**, satisfying "skip/xfail with a logged reason."
- For the "CI log shows which dtype ran on which backend" requirement: emit a `log::info!("oracle dtype=f64 backend=wgpu adapter={}", ...)` (or f32/cpu) at the start of every oracle test, and ensure the CI test command initializes a logger (`env_logger::init()` via a test harness, or `RUST_LOG=info`). `[ASSUMED on exact wgpu SHADER_F64 feature symbol — verify against cubecl-wgpu 0.10]`

### Pattern 5: `assert_close` with abs AND rel + near-zero guard (FOUND-07/08, D-09)
```rust
// Source: D-09 spec + standard numerical-comparison practice. mlrs-core/src/compare.rs
pub struct Tolerance { pub abs: f64, pub rel: f64 }
pub const F32_TOL: Tolerance = Tolerance { abs: 1e-5, rel: 1e-5 };
pub const F64_TOL: Tolerance = Tolerance { abs: 1e-5, rel: 1e-5 };
const NEAR_ZERO_FLOOR: f64 = 1e-8; // Claude's discretion (D-08); document choice

pub fn is_close(got: f64, expected: f64, tol: &Tolerance) -> bool {
    let abs_err = (got - expected).abs();
    if expected.abs() < NEAR_ZERO_FLOOR {
        // near-zero guard: rel term explodes; fall back to abs-only (D-09 ⚠)
        return abs_err <= tol.abs;
    }
    let rel_err = abs_err / expected.abs();
    abs_err <= tol.abs && rel_err <= tol.rel   // BOTH must pass (D-09)
}
```

### Pattern 6: Buffer-reuse pool with counters (FOUND-05, D-04/D-05)
**What:** A free-list keyed by byte-size over reclaimed CubeCL handles, with `allocations`/`reuses`/`peak_bytes` counters logged (not asserted) in Phase 1.
```rust
// mlrs-backend/src/pool.rs — design sketch
pub struct PoolStats { pub allocations: u64, pub reuses: u64, pub peak_bytes: u64, pub live_bytes: u64 }
pub struct BufferPool<R: cubecl::Runtime> {
    client: /* ComputeClient */,
    free: std::collections::HashMap<usize, Vec<Handle>>, // size_bytes -> reusable handles
    stats: PoolStats,
}
// acquire(size): pop from free[size] (reuses += 1) else client.empty(size) (allocations += 1)
// release(handle, size): push back to free[size]
// On drop / phase boundary: log::info!("pool stats: {:?}", stats);  // logged-only (D-05)
```
> CubeCL also has its own memory pools (`MemoryConfiguration::ExclusivePages`, see Tuning manual). The D-04 pool is an **mlrs-level reuse layer on top of** `client.empty`/`client.create`; the planner should decide whether to additionally tune CubeCL's `MemoryConfiguration` or keep the default `SubSlices` allocator and reuse at the mlrs layer. Phase 1 only logs counters, so the simplest correct pool (HashMap free-list at mlrs level) is sufficient. `[CITED: Tuning_ExclusivePages_Allocator manual]`

### Pattern 7: mimalloc global allocator in `mlrs-py` only (FOUND-09)
```rust
// Source: MIMALLOC_MANUAL.md. mlrs-py/src/lib.rs (or allocator.rs).
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
// Test goes in mlrs-py/tests/allocator_test.rs — NOT a mod tests in lib.rs (AGENTS.md).
```
> `#[global_allocator]` must be defined exactly once in the final artifact. Define it in `mlrs-py` (the cdylib). Do NOT put it in library crates (`mlrs-core`/`-kernels`/`-backend`/`-algos`) — that would conflict if the binary also sets one. `[VERIFIED: Rust reference — single global_allocator per binary]`

### Anti-Patterns to Avoid
- **`#[cfg(test)] mod tests` in source files** — every manual does this, but AGENTS.md *strictly prohibits* it. Use `tests/` or `*_test.rs`. (Highest-frequency mistake risk because the reference code models the wrong pattern.)
- **`let v = if cond { a } else { b };` inside `#[cube]`** — produces `ExpandElementTyped` vs `{float}` mismatch (E0308). Use `let mut v = default; if cond { v = a; }`. `[CITED: cubecl_error_solution_guide/mismatched types.md]`
- **`.exp()`/`.sqrt()`/`.abs()` method calls inside `#[cube]`** — use associated functions `F::exp(x)`, `F::sqrt(x)`. (E0599 `__expand_*_method` not found.) `[CITED: cubecl_error_solution_guide]`
- **`usize`/host-only types inside `#[cube]` kernels** — use `u32`/`i32`/`f32`/`f64`. `[CITED: cubecl_error_solution_guide §3]`
- **Backend feature flags in `mlrs-kernels`** — violates Criterion 1 ("`mlrs-kernels` carries zero backend feature flags"). The kernels crate depends on `cubecl` with `default-features = false` and stays runtime-agnostic.
- **`#[global_allocator]` in a library crate** — must live only in `mlrs-py`.
- **`bytemuck::cast_slice` (panicking) on unvalidated Arrow buffers** — use `try_cast_slice` so misalignment becomes a `BridgeError`, not a panic (D-06 "before any unsafe transmutation").
- **Ignoring Arrow `offset()` / `nulls()`** — a sliced `Float64Array` has `offset != 0` and `values()` still points at the *parent* buffer start; must reject (Criterion 3).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| `&[f32]`/`&[f64]` → `&[u8]` transmute | Manual `std::slice::from_raw_parts` | `bytemuck::try_cast_slice` | Validates alignment + size divisibility; surfaces misalignment as recoverable Err instead of UB/panic |
| Reading `.npz` named arrays | Manual zip + npy header parser | `ndarray-npy` (`NpzReader::by_name`) or `npyz` (`NpzArchive`) | npy header parsing (dtype, fortran_order, shape) and zip handling are error-prone edge cases |
| GPU device kernels | Raw wgpu/CUDA/WGSL | `cubecl` `#[cube]` | Mandated; also gives the generic-over-runtime guarantee for free |
| Typed library errors | Manual `enum` + `Display`/`Error` impls | `thiserror` | D-07/D-10 mandate; less boilerplate, correct `source()` chaining |
| Custom allocator plumbing | Manual `GlobalAlloc` impl | `mimalloc` crate | FOUND-09; drop-in `#[global_allocator]` |
| f64/feature detection | Hardcoding "wgpu has no f64" | `client.properties().feature_enabled(...)` | Adapter-dependent; some wgpu adapters DO support SHADER_F64; query at runtime |
| Float comparison | `==` or hand-rolled epsilon | `assert_close` with abs AND rel + near-zero guard | D-09 semantics; near-zero guard prevents spurious failures |
| Seeded reference data | Rust-side RNG to mimic NumPy | NumPy `default_rng(seed)` in `gen_oracle.py`, committed `.npz` | NumPy and Rust RNGs differ; only NumPy's is the sklearn-matching oracle (D-01/D-03) |

**Key insight:** In this domain, the "deceptively hard" parts are (1) the `#[cube]` macro's IR translation quirks (mitigated only by following the error-solution guide's rewrite rules), (2) Arrow's offset/null/alignment invariants that make naive `values()` unsafe to upload, and (3) cross-RNG reproducibility (solved by committing Python-generated fixtures). None of these should be reimplemented.

## Runtime State Inventory

> N/A for the most part — this is a **greenfield** phase. No prior runtime state exists. The categories below are answered explicitly for completeness.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — verified no Rust workspace, no datastores exist (repo has only `cuml-main/` reference + `.planning/`) | None |
| Live service config | None — no services | None |
| OS-registered state | None | None |
| Secrets/env vars | None for Phase 1. (`RUST_LOG` is a *non-secret* runtime knob the CI test job should set to surface dtype/backend logs per Criterion 4) | Set `RUST_LOG=info` in CI test job |
| Build artifacts | None yet. Note: `.gitignore` exists; ensure `target/` is ignored and committed `.npz` fixtures are NOT ignored | Verify `.gitignore` does not exclude `tests/fixtures/*.npz` |

## Common Pitfalls

### Pitfall 1: Copying the manuals' in-file test modules (AGENTS.md violation)
**What goes wrong:** Implementer copies a manual example verbatim, including its `#[cfg(test)] mod tests { ... }`, into a source file.
**Why it happens:** All CubeCL/optimisor manual examples and the zero-copy docs embed tests in source.
**How to avoid:** Treat manual code as logic reference only. Put every test in `crates/<crate>/tests/*.rs` or a sibling `*_test.rs` module file. Add a CI grep gate: fail if `mod tests` appears in any `src/**/*.rs`.
**Warning signs:** `grep -rn "mod tests" crates/*/src/` returns hits.

### Pitfall 2: `#[cube]` `if`-expression and method-call macro errors
**What goes wrong:** `let s = if x < 0.0 { -1.0 } else { 1.0 };` → E0308; `(-y*y).exp()` → E0599.
**Why it happens:** `#[cube]` macro IR can't unify `if`-expression branches; math is exposed as associated functions, not methods.
**How to avoid:** `let mut s = 1.0; if x < 0.0 { s = -1.0; }`; use `F::exp(arg)`. **On ANY cubecl build error, STOP and consult `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/` per AGENTS.md §4 before attempting fixes.**
**Warning signs:** errors mentioning `ExpandElementTyped` or `__expand_*_method`.

### Pitfall 3: Uploading a sliced/nullable Arrow array (Criterion 3 failure)
**What goes wrong:** `arr.values()` on a sliced array returns a slice into the *parent* buffer, and `bytemuck::cast_slice` may silently upload wrong/aliased data; nullable arrays carry meaningless values at null positions.
**Why it happens:** Arrow shares buffers across slices; `offset` is logical metadata, not applied to `values()` start in all access paths; `null_count` is independent of the values buffer.
**How to avoid:** Reject `offset() != 0`, `null_count() != 0` / `nulls().is_some()` BEFORE transmute (Pattern 3). Add explicit negative tests constructing a sliced array (`arr.slice(1, n-1)`) and a nullable array (`Float64Array::from(vec![Some(1.0), None])`) and assert `Err`.
**Warning signs:** oracle mismatch only on subset/filtered inputs; non-deterministic device values.

### Pitfall 4: f64 oracle silently passing/failing on wgpu without SHADER_F64
**What goes wrong:** wgpu adapter lacks f64; kernel either fails to compile at runtime or produces garbage, and the test reports a confusing failure (or worse, the test is skipped silently with no record).
**Why it happens:** f64 support is adapter-dependent; "skip" without a log hides which dtype actually ran.
**How to avoid:** Gate every f64 path on `supports_f64(&client)`; on unsupported, log a warning and early-return; always log `dtype=… backend=…` at test start (Criterion 4). STATE.md flags this as a known concern.
**Warning signs:** CI green but no f64 line in logs; runtime compile errors mentioning `f64`/`float64` on wgpu.

### Pitfall 5: `mlrs-kernels` accidentally pulling backend features
**What goes wrong:** Adding `cubecl = { workspace = true, features = ["wgpu"] }` to `mlrs-kernels` violates Criterion 1.
**Why it happens:** Convenience; wanting to test the kernel directly in the kernels crate.
**How to avoid:** `mlrs-kernels` uses `cubecl` with `default-features = false` and no runtime feature. Kernel *tests* that need a concrete runtime live in `mlrs-backend`'s integration tests (which own the features) or are dev-dependency-gated. Add a CI check that `cargo tree -p mlrs-kernels` shows no `cubecl-wgpu`/`cubecl-cuda`/`cubecl-cpu`.
**Warning signs:** `cargo tree -p mlrs-kernels -e features` lists a backend runtime crate.

### Pitfall 6: `.to_vec()` defeating "zero-copy"
**What goes wrong:** `Bytes::from_bytes_vec(slice.to_vec())` copies the host buffer, contradicting the zero-copy intent.
**Why it happens:** The manual examples use `.to_vec()` for simplicity.
**How to avoid:** Investigate `cubecl::bytes::Bytes` constructors in 0.10 for a borrow/owned-without-copy path; if unavailable, document the actual semantics honestly (validated, single upload copy). See Pattern 3 nuance.
**Warning signs:** profiling shows an extra host allocation per ingest.

### Pitfall 7: `rand` 0.10 API churn breaking seeded fixtures
**What goes wrong:** Rust-side seeded RNG using an outdated `rand` API fails to compile or produces different sequences than expected.
**Why it happens:** `rand` 0.9→0.10 renamed core APIs; D-10 says "latest."
**How to avoid:** Minimize Rust RNG — generate all oracle inputs in Python. If Rust RNG is truly needed, add a spike/verify task pinning the exact 0.10 seeded-RNG calls. Prefer `rand` 0.9.x if a stable, well-documented seeded API matters more than absolute-latest. `[ASSUMED]`

## Code Examples

### End-to-end pipeline test (Criterion 2) — lives in `mlrs-backend/tests/`
```rust
// Source: composed from cubecl generics + ZERO_COPY_ARROW_CUBECL + oracle harness.
// crates/mlrs-backend/tests/pipeline_test.rs   (NOT a mod tests in src)
use arrow::array::Float32Array;
use mlrs_backend::{bridge, runtime, /* DeviceArray, pool */};
use mlrs_kernels::saxpy_kernel;
use mlrs_core::oracle::load_npz;
use mlrs_core::compare::{is_close, F32_TOL};
use cubecl::prelude::*;

#[test]
fn saxpy_f32_matches_numpy_reference() {
    let client = runtime::active_client();
    let case = load_npz("tests/fixtures/saxpy_f32_seed42.npz"); // {a, x, y, expected}
    let arr = Float32Array::from(case.x.clone());

    let x_slice = bridge::validate_f32(&arr).expect("conforming input");
    let x_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
        bytemuck::cast_slice(x_slice).to_vec()));
    let y_handle = client.create(cubecl::bytes::Bytes::from_elems(case.y.clone()));
    let n = x_slice.len();

    saxpy_kernel::launch::<f32, runtime::ActiveRuntime>(
        &client,
        CubeCount::Static(((n + 255) / 256) as u32, 1, 1),
        CubeDim { x: 256, y: 1, z: 1 },
        case.a,
        unsafe { ArrayArg::from_raw_parts(x_handle, n) },
        unsafe { ArrayArg::from_raw_parts(y_handle.clone(), n) },
    );

    let bytes = client.read_one(y_handle).unwrap();
    let got: &[f32] = bytemuck::cast_slice(&bytes);
    for (g, e) in got.iter().zip(case.expected.iter()) {
        assert!(is_close(*g as f64, *e as f64, &F32_TOL), "got {g} expected {e}");
    }
}
```

### `scripts/gen_oracle.py` shape (D-01/D-02/D-03)
```python
# Source: D-01..D-03 spec. NOT run in CI test job; regenerates committed .npz blobs.
import numpy as np
def gen_saxpy(seed=42, n=1024, dtype=np.float32):
    rng = np.random.default_rng(seed)
    a = dtype(2.5)
    x = rng.standard_normal(n).astype(dtype)
    y = rng.standard_normal(n).astype(dtype)
    expected = (a * x + y).astype(dtype)
    np.savez("tests/fixtures/saxpy_f32_seed42.npz", a=a, x=x, y=y, expected=expected)
# For estimator fixtures later (Phase 4+): import sklearn, fit, savez coef_/intercept_/etc.
if __name__ == "__main__":
    gen_saxpy()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `thiserror` 1.x | `thiserror` 2.x | 2.0 (2024) | Use v2 syntax; mostly compatible `[VERIFIED: cargo search]` |
| `arrow` 58 (manuals) | `arrow` 59 | 59.0.0 current | Manuals say 58/58.3; bump to 59 per D-10 `[VERIFIED: cargo search]` |
| `npy-rs` (unmaintained) | `npyz` (fork) / `ndarray-npy` | — | Use a maintained npz reader `[VERIFIED: docs.rs]` |
| `rand` 0.8 `thread_rng()`/`gen()` | `rand` 0.9/0.10 `rng()`/`random()` | 0.9, 0.10 | API churn — verify seeded API (see note) `[ASSUMED]` |
| In-file `mod tests` | Separate `tests/` files | Project rule (AGENTS.md) | Mandatory for this project; overrides manual examples |

**Deprecated/outdated:**
- `npy-rs`: superseded by `npyz`.
- Manuals pin `cubecl = "0.10.0"`, `arrow = "58.3.0"`, `bytemuck = "1"`, `half = "2"` — treat versions as floors; bump to latest per D-10 but keep the **API patterns** (those are current for 0.10).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `client.properties().feature_enabled(Feature::Type(Elem::Float(FloatKind::F64)))` is the correct 0.10 f64-capability call (transferred from documented F16 pattern) | Pattern 4 / Capability | f64 gate (FOUND-04, Criterion 4) won't compile; need the alt `features.supports_type(FloatKind::F64)` form — both shown, one will work |
| A2 | wgpu f64 support maps to a queryable CubeCL feature equivalent to `SHADER_F64` | Pattern 4 | f64 oracle skip logic mis-targets; may need to query wgpu adapter features directly |
| A3 | `cubecl::bytes::Bytes` in 0.10 requires owned bytes (`.to_vec()` copy) for `client.create` | Pattern 3 / Pitfall 6 | "Zero-copy" claim weakens to "validated single-upload"; affects Criterion 2/3 wording, not correctness |
| A4 | `ndarray-npy` 0.10 / `npyz` 0.9 read named `.npz` arrays via `by_name` for f32 and f64 | Standard Stack | Need a different npz reader; low risk (both documented) |
| A5 | `rand` latest seeded-RNG API; minimal Rust RNG needed because oracle inputs are Python-generated | Standard Stack / Pitfall 7 | Fixture reproducibility; mitigated by Python-side generation |
| A6 | `ComputeClient` generic signature in 0.10 (single `<R>` vs `<Server, Channel>`) | Pattern 2 | Type aliases/return types won't compile; insulate with `impl`/alias |
| A7 | `bytemuck::try_cast_slice` returns `Err` (not panic) on misalignment, usable for `BridgeError::Misaligned` | Pattern 3 | If it panics instead, need manual alignment check via `ptr as usize % align_of::<T>()` |
| A8 | mimalloc 0.1.52 `MiMalloc` unit struct + `#[global_allocator]` API unchanged | Pattern 7 | Allocator wiring (FOUND-09) — very low risk, stable API |
| A9 | `ndarray` (transitive) version compatible with `ndarray-npy` 0.10 | Supporting | Build break; resolve by letting cargo pick compatible versions |

**These `[ASSUMED]` items should be resolved by a small implementation spike (compile a hello-world kernel + capability query + npz load on cpu and wgpu) before the full phase plan executes. Recommend the planner front-load a "toolchain spike / Wave 0" task.**

## Open Questions

1. **Exact CubeCL 0.10 capability-query symbol path and the f64 feature variant**
   - What we know: F16 detection uses `client.properties().feature_enabled(Feature::Type(Elem::Float(FloatKind::F16)))` *and* an alt `client.properties().features.supports_type(FloatKind::F16)` (manual shows both).
   - What's unclear: which compiles in the installed 0.10; whether F64 is the right variant for wgpu's SHADER_F64.
   - Recommendation: spike-verify; expose a thin `feature_enabled(FloatKind::F64)` façade so the call site is stable regardless of the internal form.

2. **True host-side zero-copy into `cubecl::bytes::Bytes`**
   - What we know: manuals use `.to_vec()` (a copy).
   - What's unclear: whether 0.10 offers a borrow/owned-no-copy constructor.
   - Recommendation: inspect `cubecl::bytes` API; document actual semantics; don't overclaim "zero-copy."

3. **`ComputeClient` type signature and `Runtime` associated types in 0.10**
   - Recommendation: use type alias `pub type Client = ComputeClient<...>;` resolved once in `mlrs-backend`.

4. **Whether to tune CubeCL `MemoryConfiguration` (ExclusivePages) or reuse at the mlrs pool layer**
   - Recommendation: Phase 1 = mlrs-level free-list pool with logged counters (D-04/D-05); defer CubeCL allocator tuning unless profiling demands it.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (rustc/cargo) | All crates | ✓ | rustc 1.95.0 / cargo 1.95.0 | — |
| Python 3 | `scripts/gen_oracle.py` (regen only, NOT CI test job) | ✓ | 3.12.3 | — |
| numpy + scikit-learn | `gen_oracle.py` (build-time only) | ✗ (not verified installed) | — | Install in a dev/regen environment; NOT needed at test time (D-03) |
| wgpu-capable adapter | `--features wgpu` correctness gate | ? (not probed) | — | CPU runtime is the always-available gate; wgpu may run via software adapter (lavapipe/llvmpipe) in CI |
| CUDA toolkit/driver | `--features cuda` (compile-only) | ✗ (untestable here per constraints) | — | cuda compiles only; not run (PROJECT.md) |

**Missing dependencies with no fallback:** none that block Phase 1 (cpu runtime is always available; cuda is compile-only by design).
**Missing dependencies with fallback:**
- numpy/sklearn: needed only to (re)generate fixtures; committed `.npz` blobs make the CI test job hermetic (D-03). Generate fixtures once in a dev env.
- wgpu adapter: if no hardware GPU in CI, a software Vulkan adapter (lavapipe) can run wgpu; otherwise cpu remains the gate. **Probe wgpu availability during the Wave 0 spike** — note: f64 on a software adapter may be absent, exercising the capability-gate path (Criterion 4).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]` + `cargo test`); integration tests in `tests/` per crate (AGENTS.md) |
| Config file | none — standard cargo; per-crate `tests/` directories. Logger init for dtype/backend logs (Wave 0). |
| Quick run command | `cargo test -p mlrs-core --features cpu` (host-only logic: assert_close, sign-flip, label-perm) |
| Full suite command | `cargo test --workspace --features cpu` then `cargo test --workspace --features wgpu` (both gates); `cargo build --workspace --features cuda` (compile-only) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FOUND-01 | 5-crate workspace compiles cpu+wgpu, cuda compiles, kernels feature-free | build/smoke | `cargo build --workspace --features cpu && --features wgpu && --features cuda`; `cargo tree -p mlrs-kernels -e features \| grep -v cubecl-wgpu` | ❌ Wave 0 |
| FOUND-02 | Generic `<F: Float>` kernel runs f32 & f64 on cpu+wgpu | integration | `cargo test -p mlrs-backend --features cpu saxpy`; `--features wgpu` | ❌ Wave 0 |
| FOUND-03 | Backend selected by feature | build matrix | per-feature `cargo build` (above) | ❌ Wave 0 |
| FOUND-04 | Capability layer reports f64; f64 tests skip/xfail+log on no-SHADER_F64 | integration | `cargo test -p mlrs-backend --features wgpu f64 -- --nocapture` (inspect log) | ❌ Wave 0 |
| FOUND-05 | DeviceArray + pool reuse; counters logged | unit/integration | `cargo test -p mlrs-backend --features cpu pool` | ❌ Wave 0 |
| FOUND-06 | Bridge rejects offset/nulls/misaligned before transmute | unit (negative) | `cargo test -p mlrs-backend --features cpu bridge_reject` | ❌ Wave 0 |
| FOUND-07 | Oracle: load npz, assert ≤1e-5 vs reference | integration | `cargo test -p mlrs-backend --features cpu saxpy_f32_matches` | ❌ Wave 0 (needs committed fixture) |
| FOUND-08 | sign-flip + label-perm helpers; tolerance policy structure | unit | `cargo test -p mlrs-core --features cpu sign_flip label_perm tolerance` | ❌ Wave 0 |
| FOUND-09 | mimalloc global allocator active; test in separate file | unit | `cargo test -p mlrs-py --features cpu allocator` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p <crate> --features cpu` for the touched crate (fast, no GPU).
- **Per wave merge:** `cargo test --workspace --features cpu` + `cargo test --workspace --features wgpu`.
- **Phase gate:** both feature suites green + `cargo build --workspace --features cuda` succeeds + `cargo tree` kernels-feature-free check passes, before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/mlrs-core/tests/compare_test.rs` — covers FOUND-08 (assert_close, near-zero guard, tolerances)
- [ ] `crates/mlrs-core/tests/helpers_test.rs` — covers FOUND-08 (sign-flip, label-permutation)
- [ ] `crates/mlrs-backend/tests/bridge_test.rs` — covers FOUND-06 (negative: offset/nulls/misaligned reject)
- [ ] `crates/mlrs-backend/tests/pipeline_test.rs` — covers FOUND-02/07 (Arrow→device→kernel→read-back→oracle)
- [ ] `crates/mlrs-backend/tests/capability_test.rs` — covers FOUND-04 (f64 gate + dtype/backend logging)
- [ ] `crates/mlrs-backend/tests/pool_test.rs` — covers FOUND-05 (reuse + counters logged)
- [ ] `crates/mlrs-py/tests/allocator_test.rs` — covers FOUND-09 (mimalloc active, separate file)
- [ ] `tests/fixtures/saxpy_*_seed*.npz` — committed oracle blobs (generated by `scripts/gen_oracle.py`)
- [ ] Logger init in test harness (`env_logger`/`RUST_LOG=info`) so Criterion 4 dtype/backend lines appear in CI
- [ ] **Toolchain/API spike** resolving Assumptions A1–A7 (capability call, Bytes constructor, ComputeClient signature, npz API) before full implementation

## Security Domain

> `security_enforcement: true`, `security_asvs_level: 1`, `security_block_on: high` (verified in config.json). This is internal compute infrastructure with **no network, no auth, no user sessions, no persistence of secrets** — the standard web-app ASVS categories largely do not apply. The relevant surface is **memory safety around `unsafe` transmutation and FFI**.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface in Phase 1 |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No multi-user access |
| V5 Input Validation | **yes** | The Arrow bridge IS input validation: reject offset/nulls/misaligned before `unsafe` transmute (D-06/FOUND-06). `bytemuck::try_cast_slice` enforces alignment/size invariants. |
| V6 Cryptography | no | No crypto in scope |
| V12/V13 Files & API (FFI) | **partially (later)** | PyO3 Arrow PyCapsule boundary (Phase 6) needs ownership/lifetime care; Phase 1 only stubs `mlrs-py` |

### Known Threat Patterns for {Rust + CubeCL + Arrow + unsafe transmute}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Unsafe transmute of misaligned/short Arrow buffer → UB / OOB read | Tampering / DoS | `bytemuck::try_cast_slice` (alignment+size checked) + explicit offset/null/alignment rejection before any `unsafe` (Pattern 3) |
| Sliced/offset Arrow array uploading aliased parent data | Information disclosure / Tampering | Reject `offset() != 0` (Criterion 3) |
| Nullable array uploading meaningless null-slot values | Tampering (silent wrong results) | Reject `null_count() != 0` (Criterion 3) |
| `unsafe { ArrayArg::from_raw_parts(...) }` with wrong length | Memory safety / OOB | Always derive length from validated slice `.len()`; bounds-check in kernel (`if tid < x.len()`) |
| `#[global_allocator]` double-definition / library setting allocator | Build integrity | Define mimalloc only in `mlrs-py` cdylib (Pattern 7) |
| Committed `.npz` fixture as untrusted deserialization input | Tampering | Fixtures are first-party, committed, reviewed; npz reader (`ndarray-npy`/`npyz`) is memory-safe Rust — low risk |

**Security verification steps for the planner to include:**
- Negative tests proving the bridge returns `Err` (not panic, not UB) for each violation class (Criterion 3 directly).
- A grep/CI gate ensuring every `unsafe` block in `mlrs-backend` is preceded by validation (or carries a `// SAFETY:` comment justifying the invariant).
- Confirm `bytemuck` is used in preference to raw `from_raw_parts` for all host transmutes.

## Sources

### Primary (HIGH confidence)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/Cubecl_generics.md` — generic kernel definition, `<N: Numeric>`/`<F: Float>`, `launch::<N, R>`, trait bounds
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/Cubecl_axpy.md` — SAXPY kernel (smoke-kernel candidate), scalar arg ordering
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/Cubecl_multi_threading.md` — ABSOLUTE_POS, CubeCount/CubeDim, ceiling-division launch config
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/Cubecl_plane.md` — PLANE_DIM, plane ops (relevant Phase 2; subgroup capability context)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/ZERO_COPY_ARROW_CUBECL.md` — Arrow `values()` → `bytemuck::cast_slice` → `cubecl::bytes::Bytes` → `client.create`
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/ZERO_COPY_TRANSMUTATION_CUBECL.md` — bytemuck Pod/Zeroable, alignment validation semantics
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/mismatched types.md` — `if`-expr (E0308) and `.exp()` (E0599) fixes, device-type rules (AGENTS.md mandatory ref)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/Tuning_ExclusivePages_Allocator_and_Pool_Configurations_in_CubeCL.md` — MemoryConfiguration, pool tuning context for D-04
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/Backend-Agnostic_Buffer_Slicing_and_Multi-Logical_Array_Allocation.md` — slice/slice_mut, CubeDim::new_1d, read_one
- `/home/user/Documents/workspace/optimisor/manual/HALF_PRECISION_CUBECL.md` — **capability query API** (`client.properties().feature_enabled(Feature::Type(Elem::Float(FloatKind::_)))`), feature-gated launch (transferred F16→F64)
- `/home/user/Documents/workspace/optimisor/manual/MIMALLOC_MANUAL.md` — `#[global_allocator] static GLOBAL: MiMalloc = MiMalloc;`, crate `mimalloc = "0.1.52"`
- `/home/user/Documents/workspace/optimisor/manual/ARROW_NUMERIC_BRANCHING.md` — arrow-rs v58/59 ScalarBuffer/NullBuffer, contiguous layout
- `cargo search` (crates.io, 2026-06-11) — version verification for all stack crates
- `./AGENTS.md`, `./CLAUDE.md`, `.planning/*` — project constraints, locked decisions, success criteria

### Secondary (MEDIUM confidence)
- docs.rs/npyz npz module + docs.rs/ndarray-npy `NpzReader::by_name` — named-array `.npz` reading
- docs.rs/arrow `PrimitiveArray` / `ArrayData` — `offset()`, `nulls()`, `null_count()`, `align_buffers`
- GitHub tracel-ai/cubecl releases + architecture overview — ComputeClient/FeatureSet model

### Tertiary (LOW confidence — flagged for spike verification)
- Exact CubeCL 0.10 capability symbol path (A1/A2), `Bytes` no-copy constructor (A3), `ComputeClient` signature (A6) — resolve in Wave 0 spike

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions `cargo search`-verified; crates are first-party and project-mandated
- Architecture / workspace layout: HIGH — standard Rust virtual-manifest pattern; directly satisfies Criterion 1 and AGENTS.md
- CubeCL kernel patterns: HIGH — drawn from project's own mandatory manuals + error-solution guide
- f64 capability API exact symbols: MEDIUM — pattern proven for F16 in the manual; F64 variant + wgpu SHADER_F64 mapping needs spike
- Arrow validation: HIGH — offset/null checks documented; alignment via bytemuck try_cast_slice
- npz reader: MEDIUM — two viable maintained crates; `by_name` API documented
- Pitfalls: HIGH — error-solution guide is authoritative; AGENTS.md contradiction with manuals is concrete

**Research date:** 2026-06-11
**Valid until:** 2026-07-11 (30 days) for versions; CubeCL is fast-moving — re-verify 0.10 API symbols at implementation time (7-day currency on cubecl specifics)
