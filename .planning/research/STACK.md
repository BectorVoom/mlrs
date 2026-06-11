# Stack Research

**Domain:** GPU-accelerated, sklearn-compatible ML library in Rust (CubeCL kernels generic over float + runtime; Apache Arrow zero-copy interchange; PyO3 Python bindings; multi-backend via Cargo features)
**Researched:** 2026-06-11
**Confidence:** HIGH for compute/binding/Arrow core (Context7 + crates.io + provided manuals all agree on CubeCL 0.10.0 and surrounding versions); MEDIUM for the oracle-test tooling and the per-backend wheel packaging mechanics (verified via official docs + community sources, but no first-party "do it this way" prescription exists for this exact combination).

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `cubecl` | `0.10.0` | Single-source GPU/CPU compute kernels; the `#[cube]` / `#[cube(launch)]` macro, `Numeric`/`Float`/`CubeElement` trait bounds, `Array`/`Tensor`, `ComputeClient`, `Runtime` trait | Mandated by the project. Confirmed current: crates.io max_version is `0.10.0` (released 2026-05-07) and every provided manual pins `cubecl = "0.10.0"`. One crate exposes all backends behind Cargo features, which is exactly the `cuda`/`rocm`/`wgpu`/`cpu` toggle model the project requires. |
| `pyo3` | `0.28` (`0.28.3`) | Rust↔Python FFI; defines `#[pyclass]` estimators, `#[pymethods]` for `fit`/`predict`/`transform`, GIL handling | Current de-facto standard for Python extension modules in Rust. `0.28.3` (2026-04-02) supports Python 3.12/3.13/3.14 via `abi3-py312`, which matches the "Python ≥ 3.12" constraint. Pairs natively with maturin. Requires Rust ≥ 1.83. |
| `maturin` | `1.13` (`1.13.3`) | Build backend + CLI that compiles the PyO3 crate into per-backend wheels (`pip install`) | The standard PyO3 packaging tool. Drives `pyproject.toml` with `build-backend = "maturin"`. Supports `abi3` wheels (one wheel per platform across 3.12+) and Cargo feature passthrough (`--features`) so each backend wheel is a distinct build. Released 2026-05-11. |
| `arrow` (arrow-rs) | `59` (`59.0.0`) | Host-side columnar data: `Float32Array`/`Float64Array` backed by contiguous, aligned `ScalarBuffer<T>`; the zero-copy entry point into CubeCL | Official Rust Apache Arrow implementation. `array.values()` returns a packed `&[T]` slice that `bytemuck::cast_slice` reinterprets as `&[u8]` for `client.create(...)` with no host copy — the exact pattern in `ZERO_COPY_ARROW_CUBECL.md`. `arrow2` is unmaintained (see What NOT to Use). Note: the provided manual shows `arrow = "58.3.0"`; `59.0.0` (2026-06-09) is the current release and is API-compatible for the primitive-array path used here. |
| `bytemuck` | `1` (with `derive` feature) | Zero-copy `&[T] ⇄ &[u8]` transmutation between Arrow buffers, host `Vec`s, and `cubecl::bytes::Bytes` | The single load-bearing glue crate in all three zero-copy/half-precision manuals. `Pod`/`Zeroable` bounds give compile-time + runtime alignment/size validation. Already a transitive CubeCL dependency, so no version conflict risk. |

### CubeCL backend runtime crates (selected by Cargo feature)

CubeCL exposes backends as **features of the umbrella `cubecl` crate**, not separate top-level deps. The project's `cuda`/`rocm`/`wgpu`/`cpu` Cargo features map onto CubeCL features:

| Project feature | CubeCL feature | Runtime type | CI/test status |
|-----------------|----------------|--------------|----------------|
| `cpu` | `cubecl/cpu` | `cubecl::cpu::CpuRuntime` (`CpuDevice`) | Primary CI gate. No GPU needed; used in generics manual examples. |
| `wgpu` | `cubecl/wgpu` | `cubecl::wgpu::WgpuRuntime` (`WgpuDevice`) | Primary CI gate. Runs on Vulkan/Metal/DX12/WebGPU on commodity CI hardware. |
| `cuda` | `cubecl/cuda` | `cubecl::cuda::CudaRuntime` | Compile-only here; verified opportunistically on CUDA hardware. |
| `rocm` | `cubecl/rocm` (HIP) | ROCm/HIP runtime | Compile + opportunistic; AMD hardware. |

> Do **not** add `cubecl-wgpu` / `cubecl-cuda` as separate direct dependencies for normal use. Use the `cubecl` umbrella crate's features so the prelude, macro, and runtime versions stay locked in lockstep. The exception is the lower-level `cubecl-matmul` / `cubecl-std` / `cubecl-runtime` crates, which the GEMM example imports directly when you need pre-built linalg primitives (see Supporting Libraries).

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `cubecl-std` | matched to `0.10.0` | `TensorHandle<R, F>` and tensor helpers used by the matmul example | When wrapping device allocations as typed tensor handles for the prebuilt linalg kernels. |
| `cubecl-matmul` | matched to `0.10.0` | Pre-tuned GEMM (`launch::<R, F>(&Strategy::Auto, ...)`, `MatmulInputHandle`) | Linear models (normal equations / Gram matrix), PCA covariance, KNN distance blocks. Prefer this over hand-rolling GEMM; the matmul example shows `Strategy::Auto` picking a backend-appropriate kernel. |
| `cubecl-reduce` (or the reduce pattern from the manual) | matched to `0.10.0` | Sum/mean/min/max reductions | KMeans centroid sums, variance/mean for standardization, norms. The `cubecl_reduce_sum` manual documents the tree-reduction + `sync_cube` pattern if a custom reduce is needed. |
| `arrow-pyarrow` | feature of `arrow` `59` | Pass Arrow arrays across the Python↔Rust boundary via the Arrow C Data Interface / PyCapsule protocol, zero-copy | The PyO3 binding layer. PyArrow / Polars / pandas-via-Arrow inputs arrive as PyCapsules; convert to arrow-rs `ArrayRef` with no copy, then run the bytemuck→CubeCL path. Higher-level `pyo3-arrow` wraps this if you want ergonomic `PyArray` extractors. |
| `numpy` (rust-numpy) | `0.28` | Zero-copy `&[f64]`/`&[f32]` views of NumPy arrays in PyO3 | Fallback/secondary input path for users who pass plain `np.ndarray` rather than Arrow. Version tracks PyO3 `0.28`. Keep Arrow as the primary, memory-efficiency-first path; numpy is a convenience adapter. |
| `half` | `2` (features `["num-traits", "bytemuck"]`) | Host-side `f16`/`bf16` representation | Infrastructure only for v1 (half-precision is explicitly out of v1 scope). Wire the type plumbing now so kernels stay genuinely generic-over-float; gate execution behind `client.properties().feature_enabled(Feature::Type(Elem::Float(FloatKind::F16)))` per `HALF_PRECISION_CUBECL.md`. |
| `rand` | `0.9` | RNG core for oracle test input generation | Test-only (`[dev-dependencies]`). Seeded RNG (`StdRng::seed_from_u64`) so the same random matrix is fed to both mlrs and the scikit-learn oracle for ≤1e-5 comparison. |
| `rand_distr` | `0.5` (matched to `rand 0.9`) | Normal/uniform/cluster-blob distributions for realistic test fixtures | Test-only. Generate Gaussian clusters for KMeans/DBSCAN, correlated features for PCA, etc. |
| `approx` | `0.5` | `assert_abs_diff_eq!` / `assert_relative_eq!` with explicit `epsilon`/`max_relative` | Test-only. Encodes the 1e-5 abs/rel tolerance gate cleanly instead of hand-written float comparisons. |
| `mimalloc` | `0.1.52` | Global allocator | **Recommended default.** The optimisor manual pins `0.1.52` and the project prioritizes memory efficiency + low fragmentation for high allocation churn (per-fit temporaries, Arrow buffers). Drop-in `#[global_allocator]`. Set in the top-level binary/cdylib crate only. |
| `tikv-jemallocator` | `0.7.0` | Alternative global allocator | Choose **instead of** mimalloc if you want heap profiling (`jeprof`, `MALLOC_CONF=prof:true`) to diagnose memory growth across phases. Note: the manual's `jemallocator = "0.5"` crate is the older unmaintained line; use the maintained `tikv-jemallocator 0.7.0` fork. Pick one allocator, not both. |
| `smallvec` | `1.15.1` | Small-vector optimization for short, hot collections | Shapes, strides, axis lists, per-iteration KMeans/DBSCAN bookkeeping, K-neighbor index lists — almost always ≤ a handful of elements. Avoids heap allocs on the fit/predict hot path. |
| `compact_str` | `0.8` | Small-string optimization | Feature names, class labels, estimator-parameter keys, Arrow dictionary/category keys. Inline storage ≤24 bytes avoids per-string heap allocs. |
| `thiserror` | `2` | Ergonomic error enums for the core/kernels crates | Library error types (`MlrsError`) with `#[from]` conversions; PyO3 layer maps these to Python exceptions. |
| `anyhow` | `1` | Application-level error context | Examples, the matmul-style glue, and test harnesses (the GEMM manual example uses `anyhow::Result`). Not for library public APIs. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `maturin develop` / `maturin build` | Build + install the extension into a venv; produce wheels | Use `maturin build --features wgpu` etc. to produce one wheel per backend. `abi3-py312` gives a single wheel covering 3.12+. |
| `uv` | Fast venv + Python dependency management for the oracle | Pulls `scikit-learn`, `numpy`, `pyarrow` into the test environment reproducibly; faster than pip in CI. |
| `cargo nextest` | Test runner | Faster, better isolation than `cargo test`; respects the AGENTS.md rule that tests live in `tests/` and `*_test.rs` (no `mod tests` in source). |
| `pytest` | Python-side estimator API tests + sklearn-compat checks | Runs `sklearn.utils.estimator_checks` against the installed wheel and the random-data oracle comparisons. |
| `cargo clippy` + `rustfmt` | Lint/format | Standard gate. Be aware `#[cube]`-expanded code can trip some clippy lints; allow at the kernel module level if needed. |

## Installation

```toml
# --- Workspace root Cargo.toml (workspace.dependencies; versions pinned once) ---
[workspace.dependencies]
cubecl        = { version = "0.10.0", default-features = false }
cubecl-std    = "0.10.0"
cubecl-matmul = "0.10.0"
bytemuck      = { version = "1", features = ["derive"] }
arrow         = "59"
half          = { version = "2", features = ["num-traits", "bytemuck"] }
smallvec      = "1.15"
compact_str   = "0.8"
thiserror     = "2"
pyo3          = { version = "0.28", features = ["extension-module", "abi3-py312"] }
numpy         = "0.28"

# --- Backend selection lives in the leaf crate that owns the runtime ---
# mlrs-backend/Cargo.toml
[features]
default = ["cpu"]
cpu  = ["cubecl/cpu"]
wgpu = ["cubecl/wgpu"]
cuda = ["cubecl/cuda"]
rocm = ["cubecl/rocm"]

# --- Global allocator: choose ONE, set in the cdylib/bin crate only ---
[dependencies]
mimalloc = "0.1.52"
# OR (not both):
# tikv-jemallocator = "0.7.0"

# --- Test/oracle tooling ---
[dev-dependencies]
rand      = "0.9"
rand_distr = "0.5"
approx    = "0.5"
```

```bash
# Python oracle environment (CI, no GPU needed)
uv venv && uv pip install scikit-learn numpy pyarrow pytest

# Build a backend-specific wheel
maturin build --release --features wgpu     # CI/test wheel
maturin build --release --features cuda     # compile-only here
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `cubecl` umbrella crate + features | Direct `cubecl-cuda`/`cubecl-wgpu`/`cubecl-runtime` deps | Only when you need low-level primitives the umbrella doesn't re-export (e.g., constructing `TensorHandle` for `cubecl-matmul`). Keep them at the same `0.10.0` version to avoid macro/runtime skew. |
| `arrow` (arrow-rs) `59` | `arrow2` | Never for new code — `arrow2` is unmaintained/archived. Only relevant if integrating a legacy `arrow2`/`polars`-old dependency you cannot upgrade. |
| `mimalloc` | `tikv-jemallocator` | When you need production heap profiling / leak hunting across phases (`jeprof`). jemalloc's introspection beats mimalloc's. Otherwise mimalloc's lower fragmentation + predictable latency wins for the per-fit churn pattern. |
| `arrow-pyarrow` C Data Interface | `numpy` (rust-numpy) only | numpy alone is fine for a NumPy-only API, but it does not give the columnar/Arrow zero-copy path the project mandates. Keep numpy as a secondary adapter, Arrow as primary. |
| `pyo3-arrow` | raw `arrow-pyarrow` | Use `pyo3-arrow` for ergonomic auto-conversion of PyCapsule/buffer-protocol objects; drop to raw `arrow-pyarrow` if you need fine control over the FFI lifetime/ownership. |
| `cubecl-matmul`/`cubecl-reduce` | Hand-written `#[cube]` GEMM/reduce | Hand-roll only when the algorithm needs a fused/custom kernel the prebuilt ops can't express. Prefer prebuilt for correctness and backend-portability of the heavy linalg. |
| `rand` `0.9` | `fastrand` | Use `fastrand` only for trivial non-reproducible jitter. Oracle tests need `rand`'s seeded, distribution-rich generation for sklearn parity. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `arrow2` | Unmaintained/archived; the ecosystem (incl. Polars) consolidated on `arrow-rs`. Mixing it breaks the documented zero-copy `values()`→`bytemuck` path. | `arrow` (arrow-rs) `59` |
| `jemallocator` `0.5` (original crate) | Original line is unmaintained; the manual's example is stale on this point. | `tikv-jemallocator` `0.7.0` (maintained fork) |
| `ndarray` / `nalgebra` as the device-compute layer | Both are **host/CPU** linear algebra. Using them for the actual algorithm math would bypass CubeCL and break the generic-over-runtime requirement (kernels must run on GPU backends). | CubeCL kernels (`Array`/`Tensor`, `cubecl-matmul`, `cubecl-reduce`). `ndarray` is acceptable *only* in test/oracle code for host-side reference math or shaping fixtures, never in the compute path. |
| Raw literals (`2.0`, `10`) inside `#[cube]` generic kernels | Won't compile against a generic `N`/`F`; the basic-ops manual is explicit. | `N::from_int(2)` / `F::cast_from(0.5)` / `F::new(...)`. |
| `mod tests { ... }` inside source files | Violates AGENTS.md mandatory source/test separation (critical-failure rule). | Separate `tests/foo_tests.rs` or `src/foo_test.rs` files. |
| Two global allocators at once | Only one `#[global_allocator]` may exist; linking both mimalloc and jemalloc fails. | Pick exactly one, declared in the top-level cdylib/bin crate. |
| Bit-exact cuML reproduction as a target | Out of scope per PROJECT.md; the oracle is scikit-learn, not cuML. | Match scikit-learn within abs/rel ≤ 1e-5 via the random-data oracle. |
| Blind fixes on any CubeCL build error | AGENTS.md strictly prohibits this. | Read `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md` first, then follow its template. |

## The Generic-over-Runtime + Generic-over-Float Pattern (grounded in manuals)

This is the architectural spine of the codebase. Two independent generics:

1. **Generic over float `F`** — kernel functions are written `fn k<F: Float>(...)` (or `N: Numeric` when integers are also valid). `Float` unlocks `sin`/`exp`/`sqrt`/`powf`/`erf` (basic-ops manual); `Numeric` gives `+ - * / %`. Constants MUST use `F::from_int(i)` / `F::cast_from(f64)` / `F::new(...)`, never raw literals. Add `+ CubeElement` for anything stored in `Array`/`Tensor`, and `+ bytemuck::Pod` for host-side transfer. (Generics manual, lines 67–71.)

2. **Generic over runtime `R`** — host-side driver functions are written `fn run<R: Runtime>(device: &R::Device)`, obtaining `let client = R::client(device);`. The same body then runs on `CpuRuntime`, `WgpuRuntime`, `CudaRuntime`, etc. (Generics manual `run_with_type::<N, R>`; zero-copy manuals' `run_*::<R: Runtime>()`; matmul example.)

3. **Launch ordering** — `#[cube(launch)]` generates a `launch` fn whose generic params are **kernel generics first, then `R`**: `kernel::launch::<F, R>(&client, cube_count, cube_dim, args...)`. For dynamic vectorization (`N: Size`), the vectorization factor is passed as a runtime `usize` argument after `CubeDim`. (Generics manual lines 24–35; dynamic-vectorization manual lines 48–63.)

4. **Feature-gated specialization** — query `client.features()` / `client.properties().feature_enabled(...)` before using optional capabilities (plane ops, f16); branch with `#[comptime]` flags or `#[cube]` trait impls (`SumBasic`/`SumPlane`) so a portable fallback always exists. This is how the project supports wgpu (limited f64/plane) and CUDA (full) from one codebase. (Context7 CubeCL docs; half-precision manual lines 28–32.)

5. **Zero-copy ingest** — `arrow::Float32Array::values()` → `&[f32]` → `bytemuck::cast_slice` → `&[u8]` → `client.create(cubecl::bytes::Bytes::from_bytes_vec(...))`; read back with `client.read_one(handle)` → `bytemuck::cast_slice::<u8, F>`. (`ZERO_COPY_ARROW_CUBECL.md`, `ZERO_COPY_TRANSMUTATION_CUBECL.md`.)

## Recommended Cargo Workspace Layout

A modular, single-responsibility workspace (PROJECT.md "Foundation" requirement). Compute, backend, bindings, and algorithms are separate crates so backend features and the Python cdylib stay decoupled:

```
mlrs/                          (workspace root: workspace.dependencies, shared lints)
├── Cargo.toml                 (workspace)
├── crates/
│   ├── mlrs-core/             # backend-agnostic types: Estimator traits (fit/predict/transform),
│   │                          #   shapes/strides (smallvec), error types (thiserror), no CubeCL runtime dep
│   ├── mlrs-kernels/          # all #[cube]/#[cube(launch)] kernels, generic <F: Float>;
│   │                          #   depends on `cubecl` (default-features=false) + cubecl-matmul/-reduce/-std.
│   │                          #   NO backend feature enabled here — stays runtime-agnostic.
│   ├── mlrs-backend/          # owns the cpu/wgpu/cuda/rocm Cargo features (-> cubecl/<backend>);
│   │                          #   device/client management, buffer pool / reuse, the Arrow<->Bytes
│   │                          #   zero-copy bridge (arrow + bytemuck). Generic-over-R glue lives here.
│   ├── mlrs-algos/            # algorithm orchestration: LinearRegression/Ridge/Lasso/ElasticNet/
│   │                          #   LogisticRegression, KMeans/DBSCAN, PCA/TruncatedSVD, NearestNeighbors/KNN.
│   │                          #   Composes mlrs-kernels over mlrs-backend; generic over <F, R>.
│   └── mlrs-py/               # cdylib: #[pyclass] sklearn-compatible estimators (pyo3 + maturin),
│                              #   arrow-pyarrow / numpy adapters, #[global_allocator] = mimalloc.
│                              #   One built wheel per backend feature.
├── tests/                     # workspace-level integration + oracle tests (rand, rand_distr, approx)
│   ├── fixtures/              #   generated reference outputs OR sklearn invoked via the venv
│   └── *_tests.rs             #   per AGENTS.md: tests separated from source
├── pyproject.toml             # maturin build-backend; abi3-py312
└── python/mlrs/               # thin Python package: re-exports, docstrings, sklearn-style __init__
```

**Crate dependency direction (acyclic):**
`mlrs-py` → `mlrs-algos` → {`mlrs-kernels`, `mlrs-backend`} → `mlrs-core`; everything → `cubecl`.

**Feature flow:** only `mlrs-backend` (and transitively `mlrs-py`) declares `cpu`/`wgpu`/`cuda`/`rocm`. `mlrs-kernels` is feature-free so kernels compile once and are reused by every backend.

## Oracle Test Strategy (scikit-learn reference, no GPU in CI)

Two viable mechanisms — recommend **(A) precomputed fixtures** as the default CI gate, with **(B) live PyO3 invocation** available for exploratory parity checks:

- **(A) Precomputed fixture files (recommended default).** A Python script (run via `uv`) seeds `numpy`/`rand`-equivalent inputs, fits the scikit-learn estimator, and serializes inputs + reference outputs to Arrow IPC / Parquet / `.npy` under `tests/fixtures/`. Rust tests load the fixture (zero-copy via arrow-rs), run the mlrs algorithm on `cpu`+`wgpu`, and assert with `approx` at `epsilon = 1e-5`. Deterministic, fast, hermetic, no Python at `cargo test` time. The **same seed** must drive both sides so inputs are identical.
- **(B) Live sklearn via PyO3 in a `#[test]`.** Use `pyo3::Python::with_gil` to import `sklearn` from the venv, fit on the identical random matrix, pull predictions back as NumPy/Arrow, and compare. More flexible for ad-hoc tolerance exploration, but couples `cargo test` to a configured Python env — keep it behind a `--features oracle-live` flag so the default CI run stays pure-Rust against fixtures.

Generate inputs with `rand` (`StdRng::seed_from_u64`) + `rand_distr` (Gaussian blobs for KMeans/DBSCAN, correlated features for PCA, linear-with-noise for regression). Run both `f32` and `f64` (PROJECT.md: both validated in v1); f64 makes the 1e-5 tolerance comfortable.

## Stack Patterns by Variant

**If targeting CI (wgpu + cpu):**
- Build `mlrs-py` (or run `cargo nextest`) with `--features wgpu` and `--features cpu`.
- Guard f64 paths: some wgpu adapters lack f64 — query `client.features()` and skip/xfail rather than fail (mirror the half-precision feature-gate pattern).

**If targeting CUDA/ROCm (compile-only here):**
- `cargo build --features cuda` must succeed; runtime tests are opportunistic on real hardware.
- Keep CUDA-specific code behind `#[cfg(feature = "cuda")]` only in `mlrs-backend`; algorithms and kernels stay untouched.

**If memory pressure is observed in a phase:**
- Switch `mimalloc` → `tikv-jemallocator` temporarily and profile with `jeprof` (`MALLOC_CONF=prof:true`).
- Audit buffer reuse in `mlrs-backend`: prefer `client.empty(...)` pooling and CubeCL's `ExclusivePages` allocator tuning (see CubeCL `Tuning_ExclusivePages_Allocator` manual) over per-call allocation.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `cubecl 0.10.0` | `cubecl-std`/`cubecl-matmul`/`cubecl-reduce 0.10.0` | Must match exactly — these are workspace-versioned together by tracel-ai. Mismatched versions cause macro/ABI errors (consult the CubeCL error guideline if so). |
| `cubecl 0.10.0` | `bytemuck 1.x` | bytemuck is a transitive cubecl dep; use the same major line. `half 2.x` interops via bytemuck. |
| `pyo3 0.28` | `numpy 0.28`, `maturin 1.13` | rust-numpy must match the pyo3 minor (`0.28`↔`0.28`). maturin 1.13 requires Rust ≥ 1.89 to build; pyo3 0.28 requires Rust ≥ 1.83 to compile against. |
| `pyo3 0.28` (`abi3-py312`) | Python 3.12 / 3.13 / 3.14 | One abi3 wheel per platform covers all ≥3.12 — satisfies the project's Python constraint with minimal wheel matrix. |
| `arrow 59` | `arrow-pyarrow` (same version) / PyArrow | Cross-language FFI uses the stable Arrow C Data Interface / PyCapsule protocol; arrow-rs and PyArrow versions need only agree on the C ABI, which is stable. |

## Sources

- `/tracel-ai/cubecl` (Context7) — Runtime trait (`R::client(device)`), `client.features()` / `feature_enabled`, `#[comptime]` specialization, `#[cube]` trait polymorphism, `launch::<F, R>` ordering. HIGH confidence.
- crates.io API — verified current versions: `cubecl 0.10.0` (2026-05-07), `pyo3 0.28.3` (2026-04-02), `maturin 1.13.3` (2026-05-11), `arrow 59.0.0` (2026-06-09), `tikv-jemallocator 0.7.0` (2026-05-25), `numpy 0.28.0` (2026-02-08). HIGH confidence.
- Provided CubeCL manuals (`Cubecl_generics.md`, `Cubecl_basic_operations.md`, `Cubecl_dynamic_vectorization.md`, `cubecl_matmul_gemm_example.md`, `INDEX.md`) — kernel generics, trait bounds, constants, launch signatures, GEMM via `cubecl-matmul`/`cubecl-std`. HIGH confidence (authoritative project-pinned docs).
- Provided optimisor manuals (`ZERO_COPY_ARROW_CUBECL.md`, `ZERO_COPY_TRANSMUTATION_CUBECL.md`, `HALF_PRECISION_CUBECL.md`, `MIMALLOC_MANUAL.md`, `JEMALLOC_MANUAL.md`, `SMALLVEC_MANUAL.md`, `COMPACT_STR_OPTIMIZATION_EN.md`) — Arrow→bytemuck→CubeCL path, allocator pinning, smallvec 1.15.1 / compact_str 0.8. HIGH confidence; jemallocator crate updated to maintained `tikv-jemallocator 0.7.0`.
- https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html — Arrow PyCapsule Interface for zero-copy Python↔Rust. MEDIUM confidence.
- https://docs.rs/pyo3-arrow / https://docs.rs/arrow-pyarrow — Rust-side PyArrow FFI conversion. MEDIUM confidence.
- `.planning/PROJECT.md`, `.planning/codebase/STACK.md`, `AGENTS.md` — scope, constraints, test/source separation, oracle = scikit-learn. HIGH confidence (project canon).

---
*Stack research for: GPU-accelerated sklearn-compatible ML library in Rust (CubeCL + PyO3 + Arrow)*
*Researched: 2026-06-11*
