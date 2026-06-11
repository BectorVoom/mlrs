# Architecture Research

**Domain:** GPU-accelerated, sklearn-compatible ML library in Rust (CubeCL kernels generic over float + runtime; Apache Arrow zero-copy interchange; PyO3 per-backend Python packages; multi-backend via Cargo features)
**Researched:** 2026-06-11
**Confidence:** HIGH for crate layout, the generic-over-runtime/float boundary, and the Arrow→CubeCL zero-copy flow (grounded in the CubeCL generics/slicing/matmul/allocator manuals + the two zero-copy optimisor manuals, which all agree). MEDIUM for the precise estimator-trait shape and the per-backend PyO3 packaging mechanics (no first-party prescription for this exact combination; design synthesized from sklearn conventions + the cuML `Base`/`CumlArray`/`@reflect` reference).

> This document **refines and extends** the 5-crate layout proposed in `.planning/research/STACK.md`. The headline change: the STACK draft put "Estimator traits" in `mlrs-core` but the device array/buffer/client management in `mlrs-backend`. That creates a dependency tension — estimators in `mlrs-algos` need both, and the `<F, R>` generic must thread through cleanly. This doc keeps the 5 crates but **sharpens each crate's single responsibility and the exact placement of the `R: Runtime` / `F: Float` bounds** so the boundary is unambiguous. The layout is **adopted with refinements**, not replaced.

---

## Standard Architecture

### System Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│  PYTHON LAYER  (per-backend wheel: mlrs-cpu / mlrs-wgpu / mlrs-cuda…)   │
│  python/mlrs/  thin pkg: re-exports, docstrings, sklearn-style __init__ │
│  estimators: LinearRegression, KMeans, PCA, …  (sklearn API surface)    │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │  PyArrow PyCapsule / numpy view  (zero-copy)
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  mlrs-py   (cdylib)  ── ONE BUILD PER BACKEND FEATURE ──                │
│  #[pyclass] estimators · #[pymethods] fit/predict/transform/score       │
│  arrow-pyarrow + numpy adapters · #[global_allocator]=mimalloc          │
│  dtype dispatch (f32/f64) · NotFittedError mapping · concrete R picked   │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │  Estimator<F> trait, concrete R = backend
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  mlrs-algos   (generic over <F: Float, R: Runtime>)                     │
│  estimator orchestration: fit-loop, solver selection, fitted-state      │
│  Linear(OLS/Ridge/Lasso/ENet/Logistic) · KMeans/DBSCAN · PCA/TSVD · KNN │
│  composes primitives; holds NO kernels and NO backend feature           │
└───────┬───────────────────────────────────────────────┬───────────────┘
        │ calls primitives                                │ uses device-array + client
        ▼                                                 ▼
┌──────────────────────────────┐      ┌─────────────────────────────────┐
│  mlrs-kernels                │      │  mlrs-backend                    │
│  (generic <F: Float>, NO     │      │  OWNS cpu/wgpu/cuda/rocm features │
│   backend feature)           │      │  → cubecl/<backend>              │
│  #[cube]/#[cube(launch)]      │      │  device + client mgmt            │
│  primitives: GEMM, reduce,   │      │  DeviceArray<R,F> buffer wrapper │
│  pairwise-dist, SVD/eig, CD, │◀─────│  Arrow⇄Bytes zero-copy bridge    │
│  QN, top-k, scatter-mean     │ launch│  buffer pool / ExclusivePages    │
│  + cubecl-matmul/-reduce/-std│::<F,R>│  feature-gate queries            │
└──────────────┬───────────────┘      └────────────────┬─────────────────┘
               │                                        │
               └──────────────┬─────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────────────────┐
│  mlrs-core   (NO cubecl runtime dep, NO backend feature)                │
│  Estimator/Fit/Predict/Transform/Score traits · Params (get/set)        │
│  Shape/Strides (smallvec) · MlrsError (thiserror) · dtype enum          │
│  sign-flip / label-permutation comparison contracts (shared w/ tests)   │
└───────────────────────────────────────────────────────────────────────┘
        ▲
        │ everything depends on mlrs-core
        │
┌───────────────────────────────────────────────────────────────────────┐
│  tests/  (workspace integration + oracle)  rand · rand_distr · approx   │
│  + Python oracle script (uv): sklearn → Arrow IPC fixtures              │
└───────────────────────────────────────────────────────────────────────┘
```

**Dependency direction (strictly acyclic):**

```
mlrs-py ─▶ mlrs-algos ─▶ { mlrs-kernels, mlrs-backend } ─▶ mlrs-core
                              │                │
                           cubecl          cubecl/<backend feature>
                       (+matmul/reduce/std)
```

`mlrs-core` depends on nothing internal. Every other crate depends on it. The backend feature flags live in exactly one crate (`mlrs-backend`) and are re-exported by `mlrs-py`. `mlrs-kernels` is feature-free so kernels compile once.

### Component Responsibilities

| Crate | Single Responsibility | Depends On | Generic Bounds | Backend Feature? |
|-------|----------------------|-----------|----------------|------------------|
| **mlrs-core** | Backend-agnostic vocabulary: estimator traits, params, shape/stride types, error enum, dtype tag, oracle-comparison contracts. Pure types + traits, zero compute. | (std + smallvec, compact_str, thiserror only) | none — traits are generic over `F` but bind no `R` | **No** |
| **mlrs-kernels** | All `#[cube]`/`#[cube(launch)]` compute primitives, written once, generic `<F: Float>`. GEMM (via cubecl-matmul), reductions, pairwise distance, SVD/eig, coordinate-descent step, quasi-Newton step, top-k, scatter-mean. | mlrs-core, `cubecl` (default-features=false), cubecl-matmul/-reduce/-std | `<F: Float (+ CubeElement + Pod)>` in kernels; `launch::<F, R>` ordering | **No** (compiles once, runtime-agnostic) |
| **mlrs-backend** | The one place runtime is bound. Owns `cpu/wgpu/cuda/rocm` Cargo features → `cubecl/<backend>`. Device/client lifecycle, `DeviceArray<R, F>` buffer abstraction, Arrow⇄`Bytes` zero-copy bridge, buffer pool + `MemoryConfiguration` tuning, `client.features()` capability queries. | mlrs-core, `cubecl` (with backend feature), `arrow`, `bytemuck` | exposes a concrete `R` via the selected feature; helpers generic `<R: Runtime>` | **Yes** (sole owner) |
| **mlrs-algos** | Estimator logic: fit loops, solver dispatch, fitted-state assembly, convergence checks. Composes kernels over backend buffers. Implements `mlrs-core` traits. | mlrs-core, mlrs-kernels, mlrs-backend | `<F: Float, R: Runtime>` throughout; no feature flags | **No** (inherits via mlrs-backend) |
| **mlrs-py** | cdylib. `#[pyclass]` sklearn estimators, `#[pymethods]` fit/predict/transform/score, get/set_params, Arrow PyCapsule + numpy adapters, f32/f64 dispatch, Python-exception mapping, `#[global_allocator]`. Picks the **concrete `R`** at build time from the enabled feature. | mlrs-algos (+re-exports backend feature), pyo3, numpy, arrow-pyarrow, mimalloc | monomorphizes `R` to the built backend; dispatches `F` by input dtype | **Yes** (passthrough to mlrs-backend; one wheel per feature) |

---

## Recommended Project Structure

```
mlrs/                                   # workspace root
├── Cargo.toml                          # [workspace] members, workspace.dependencies (pin versions once)
├── pyproject.toml                      # maturin build-backend, abi3-py312
├── crates/
│   ├── mlrs-core/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── estimator.rs            # Estimator / Fit / Predict / Transform / Score traits
│   │       ├── params.rs               # Params trait: get_params/set_params (typed param map)
│   │       ├── fitted.rs               # FittedState marker + NotFitted error path
│   │       ├── shape.rs                # Shape/Strides via smallvec (≤ handful of dims)
│   │       ├── dtype.rs                # DType { F32, F64 } tag + float-kind dispatch enum
│   │       ├── error.rs                # MlrsError (thiserror) with #[from] conversions
│   │       └── oracle.rs               # sign-flip / label-permutation comparison *contracts*
│   ├── mlrs-kernels/
│   │   └── src/
│   │       ├── lib.rs                  # NO #[cfg(feature=...)]; pure <F: Float> kernels
│   │       ├── gemm.rs                 # wraps cubecl-matmul launch::<R,F>(Strategy::Auto,…)
│   │       ├── reduce.rs               # sum/mean/argmin/L2 (cubecl-reduce or custom tree-reduce)
│   │       ├── distance.rs             # pairwise Euclidean/cosine (GEMM-of-XXᵀ + norm trick)
│   │       ├── decomp.rs               # SVD (Jacobi) / symmetric-eig of covariance
│   │       ├── coord_descent.rs        # CD step + soft-threshold (Lasso/ElasticNet)
│   │       ├── quasi_newton.rs         # L-BFGS / OWL-QN inner kernels (LogisticRegression)
│   │       ├── select.rs               # top-k selection (KNN), scatter-mean (KMeans update)
│   │       └── elementwise.rs          # center/scale/whiten/sigmoid/softmax primitives
│   ├── mlrs-backend/
│   │   ├── Cargo.toml                  # [features] cpu/wgpu/cuda/rocm → cubecl/<backend>
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── runtime.rs              # active-runtime selection behind cfg(feature)
│   │       ├── client.rs               # ComputeClient::load, MemoryConfiguration tuning
│   │       ├── device_array.rs         # DeviceArray<R,F>: handle + shape + strides + client
│   │       ├── arrow_bridge.rs         # Float{32,64}Array.values() → bytemuck → client.create
│   │       ├── pool.rs                 # buffer reuse / ExclusivePages config helpers
│   │       └── caps.rs                 # client.features() / feature_enabled gates (f64, plane)
│   ├── mlrs-algos/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── linear/                 # ols.rs ridge.rs lasso_enet.rs logistic.rs
│   │       ├── cluster/                # kmeans.rs dbscan.rs
│   │       ├── decomp/                 # pca.rs truncated_svd.rs
│   │       └── neighbors/              # nearest.rs knn_classifier.rs knn_regressor.rs
│   └── mlrs-py/
│       ├── Cargo.toml                  # crate-type=["cdylib"]; features passthrough to backend
│       └── src/
│           ├── lib.rs                  # #[pymodule]; #[global_allocator]=mimalloc
│           ├── convert.rs              # PyCapsule/numpy → arrow ArrayRef → DeviceArray
│           ├── dispatch.rs             # f32/f64 codepath selection by input dtype
│           └── estimators/             # one #[pyclass] per estimator wrapping mlrs-algos
├── python/mlrs/                        # thin Python package (re-exports, sklearn-style __init__)
├── tests/                              # workspace integration + oracle (AGENTS.md: tests separated)
│   ├── fixtures/                       # Arrow IPC reference outputs from sklearn (committed or generated)
│   ├── oracle/                         # Python script (uv) that produces fixtures
│   ├── primitives_test.rs              # standalone primitive validation (gemm/svd/distance/cd/qn)
│   ├── linear_test.rs cluster_test.rs decomp_test.rs neighbors_test.rs
│   └── helpers/                        # sign-flip + label-permutation impls of mlrs-core::oracle
└── .github/workflows/ci.yml            # matrix: --features cpu, --features wgpu
```

### Structure Rationale

- **`mlrs-core/oracle.rs` defines comparison *contracts*, not the harness.** The 1e-5 comparison with sign-flip (PCA/SVD components) and label-permutation (KMeans/DBSCAN) invariances is a first-class correctness concern, so the *trait/contract* (`ComponentCompare`, `LabelCompare`) lives in core where both `mlrs-algos` and `tests/` can see it; the concrete `approx`-based impls live in `tests/helpers/`. This prevents the "oracle helpers as afterthought" pitfall flagged in FEATURES.md.
- **`mlrs-kernels` is split by primitive, not by estimator.** Because pairwise-distance gates 3 families and SVD/eig gates 2 (per FEATURES.md dependency backbone), organizing kernels by *primitive* makes the reuse explicit and forces "build the primitive once, validate standalone" discipline.
- **`mlrs-algos/linear/lasso_enet.rs` is one file** — Lasso is the `l1_ratio==1` special case of ElasticNet sharing the CD kernel (FEATURES.md). Co-locating them prevents divergent solver code.
- **`mlrs-backend` is the only crate with `#[cfg(feature)]`.** Everything CUDA-specific stays here behind `#[cfg(feature = "cuda")]`; algorithms and kernels never see a feature gate. This is what makes "compile once on CI for wgpu+cpu, opportunistically for cuda/rocm" actually work.
- **`tests/` at workspace root, never `mod tests` in source** — enforces the AGENTS.md source/test separation rule. Note the optimisor manuals' inline `#[cfg(test)] mod tests` examples are illustrative only; mlrs source must keep tests external.

---

## The Generic-over-Runtime + Generic-over-Float Boundary (the architectural spine)

This is the single most important design decision and the question the milestone explicitly asks. There are **two independent generics** and they are bound at **different crates**:

### Where each generic is introduced and bound

| Generic | Introduced (written as) | Stays open through | Bound to a concrete type at |
|---------|------------------------|--------------------|----------------------------|
| `F: Float` (float type) | `mlrs-kernels` (`fn k<F: Float>(…)`) and `mlrs-core` traits (`Estimator<F>`) | mlrs-kernels → mlrs-algos → mlrs-py | **mlrs-py at runtime**, by input dtype: f32 array → `F = f32` codepath, f64 → `F = f64` |
| `R: Runtime` (backend) | `mlrs-backend` (`fn run<R: Runtime>(…)`), `mlrs-algos` (`impl<F, R> …`) | mlrs-backend → mlrs-algos → mlrs-py | **mlrs-py at compile time**, by the enabled Cargo feature: `wgpu` feature → `R = WgpuRuntime` |

**Key asymmetry (this is the crux):** `R` is resolved at **build time** (one wheel per backend, monomorphized to a single `R`), while `F` is resolved at **runtime** (every wheel carries both `f32` and `f64` monomorphizations and picks per-call by dtype). This matches PROJECT.md exactly: "users install the package matching their backend" (compile-time `R`) and "both f32 and f64 validated in v1" (runtime `F`).

### How it threads through the code (grounded in the CubeCL manuals)

1. **Kernel definition (mlrs-kernels)** — written once, generic over float only:
   ```rust
   #[cube(launch)]
   fn pairwise_sq_l2<F: Float>(x: &Array<F>, out: &mut Array<F>) { … }
   ```
   `Float` unlocks `sqrt`/`exp`/`powf` etc.; constants MUST be `F::from_int(2)` / `F::new(…)` / `F::cast_from(…)`, never raw literals (generics manual, lines 60–64, 117–122). Add `+ CubeElement` for anything stored in `Array`/`Tensor` and `+ bytemuck::Pod` for host transfer (generics manual lines 67–71). **No `R` appears here.**

2. **Launch ordering (kernels manual)** — `#[cube(launch)]` generates `launch` whose generics are **kernel-generics first, then `R`**: `pairwise_sq_l2::launch::<F, R>(&client, cube_count, cube_dim, args…)` (generics manual lines 24–35, 85). The matmul example confirms the same `launch::<R, F>` family and `TensorHandle::<R, F>::new(…)` (matmul manual lines 36–53). mlrs adopts the `::<F, R>` convention from the generics manual for hand-written kernels and the `::<R, F>` form for the cubecl-matmul prebuilt path — both are just the macro-generated ordering for their respective sources.

3. **Backend driver (mlrs-backend)** — generic over runtime only, obtains the client:
   ```rust
   pub fn client_for<R: Runtime>(device: &R::Device) -> ComputeClient<R> { R::client(device) }
   ```
   The matmul manual uses `ComputeClient::load(&device)` after `init_setup`/`init_device`; the allocator manual uses the same to inject `MemoryConfiguration::ExclusivePages`. mlrs-backend wraps both behind one `client()` helper so the rest of the code never touches `init_*` directly. **No `F` appears in the client lifecycle** — buffers are byte-typed until a kernel reinterprets them.

4. **Estimator orchestration (mlrs-algos)** — both generics open:
   ```rust
   impl<F: Float + CubeElement + bytemuck::Pod, R: Runtime> Fit<F> for KMeans<F, R> {
       fn fit(&mut self, client: &ComputeClient<R>, x: &DeviceArray<R, F>, …) { … }
   }
   ```
   The fit body calls `pairwise_sq_l2::launch::<F, R>(client, …)` and `gemm::launch::<R, F>(…)`. This is the only layer where *both* `F` and `R` are simultaneously open — it is the composition point.

5. **Monomorphization (mlrs-py)** — `R` is fixed by feature, `F` chosen at runtime:
   ```rust
   // R is fixed to the built backend:
   #[cfg(feature = "wgpu")] type Rt = cubecl::wgpu::WgpuRuntime;
   #[cfg(feature = "cpu")]  type Rt = cubecl::cpu::CpuRuntime;

   // F dispatched by input dtype inside #[pymethods] fit:
   match input_dtype {
       DType::F32 => self.inner_f32.fit::<Rt>(…),
       DType::F64 => self.inner_f64.fit::<Rt>(…),
   }
   ```

6. **Capability gating (mlrs-backend `caps.rs`)** — before using f64 or plane ops, query `client.properties().feature_enabled(Feature::Type(Elem::Float(FloatKind::F64)))` (per HALF_PRECISION/generics guidance). Some wgpu adapters lack f64; mlrs-backend exposes `supports_f64()` so mlrs-py can xfail/skip rather than crash. This is how one codebase serves wgpu (limited) and CUDA (full).

**Why bind `R` in mlrs-py and not deeper:** keeping `R` generic all the way to the cdylib boundary means `mlrs-algos` and `mlrs-kernels` build *once* per workspace and are reused by every backend wheel; only `mlrs-py` (and the `cubecl/<backend>` it pulls in) recompiles per backend. This minimizes build cost and guarantees the algorithm code is genuinely backend-agnostic — it literally cannot reference a concrete runtime.

---

## Data Flow

### Primary fit/predict path: Python → Arrow → Rust → CubeCL device buffer → kernel → result → Python

```
estimator.fit(X, y)          # X is pyarrow/polars/pandas-Arrow or numpy
   ↓  (Python)
mlrs-py: convert.rs
   X arrives as PyCapsule (Arrow C Data Interface)  →  arrow-rs ArrayRef   [zero-copy]
   (or numpy ndarray → rust-numpy &[f64] view        →  fallback path)     [zero-copy]
   ↓
mlrs-py: dispatch.rs
   read dtype → choose F = f32 | f64 ;  R already fixed by build feature
   ↓
mlrs-backend: arrow_bridge.rs
   Float64Array.values()  →  &[f64]                                        [O(1), ScalarBuffer]
   bytemuck::cast_slice::<f64,u8>(&[f64])  →  &[u8]                         [zero-copy reinterpret]
   client.create(Bytes::from_bytes_vec(bytes))  →  device handle           [single host→device upload]
   wrap in DeviceArray<R,F> { handle, shape, strides, client }
   ↓
mlrs-algos: e.g. kmeans.rs fit loop
   pairwise_sq_l2::launch::<F,R>(client, …)   →  distance buffer (reused across iters)
   argmin reduce::launch::<F,R>(…)            →  labels
   scatter_mean::launch::<F,R>(…)             →  new centroids (in-place into pooled buffer)
   convergence check (read tiny scalar back)  ← client.read_one(scalar_handle)
   ↓  (fitted state stays on device; centroids handle retained in estimator struct)
mlrs-algos → mlrs-py
   on attribute access (.cluster_centers_):
   client.read_one(centroids_handle)  →  Bytes  →  bytemuck::cast_slice::<u8,F>  →  Arrow array
   ↓
mlrs-py: convert.rs
   Arrow array → PyArrow (or numpy) returned to caller                     [output dtype = input dtype]
```

**Key flows:**

1. **Zero-copy ingest (the memory-efficiency spine).** The Arrow `values()` → `bytemuck::cast_slice` → `client.create(Bytes::…)` path is verbatim from `ZERO_COPY_ARROW_CUBECL.md` (lines 67–79) and `ZERO_COPY_TRANSMUTATION_CUBECL.md`. There is exactly **one** host→device copy (the unavoidable upload); no host-side element iteration, no intermediate `Vec` re-pack on the hot path. Read-back is the mirror: `client.read_one(handle)` → `bytemuck::cast_slice::<u8, F>` (zero-copy reinterpret of the returned `Bytes`).

2. **Buffer reuse within a fit.** Iterative algorithms (KMeans Lloyd loop, CD, QN) allocate working buffers once and reuse across iterations via `client.empty(...)` handles retained in the estimator, plus `MemoryConfiguration::ExclusivePages` tuning (allocator manual) for the high-frequency-allocation pattern. Multiple logical arrays (e.g. distance block + label block) can be carved from one allocation using in-kernel `slice_mut` (slicing manual, lines 40–68) to cut `client.create`/`empty` calls.

3. **Fitted state lives on device.** Mirroring cuML's `CumlArrayDescriptor` (codebase ARCHITECTURE.md), fitted attributes (`coef_`, `cluster_centers_`, `components_`) are kept as `DeviceArray<R,F>` handles inside the estimator and only materialized host-side (device→host copy) lazily on Python attribute access — not at end of `fit`. This avoids a copy for attributes the user never reads.

4. **Output dtype mirrors input.** f32 in → f32 out, f64 in → f64 out (FEATURES.md table-stakes), preserving the 1e-5 budget. This is mlrs's analog of cuML's `@reflect` output-type mirroring, but simplified to dtype (not container-type) mirroring.

### State management

```
Estimator struct (mlrs-algos)
   ├── params: stored unchanged at construction (sklearn rule: __init__ does no work)
   ├── fitted: Option<FittedState<R,F>>   # None until fit(); access-before-fit → NotFittedError
   │     └── device handles: coef_/centers_/components_… (retained, lazily materialized)
   └── client: ComputeClient<R>           # carries the buffer pool / allocator config
```

`get_params`/`set_params` operate on the param struct only (never touch fitted state), enabling sklearn `clone()` round-trips and grid-search compatibility (FEATURES.md cross-cutting table stakes).

---

## Architectural Patterns

### Pattern 1: Float-generic kernel, runtime-generic driver, both-generic estimator

**What:** Three nested generic scopes — kernels bind only `F`, backend drivers bind only `R`, estimators bind both. Concrete `R` is chosen at build (feature), concrete `F` at runtime (dtype).
**When to use:** Every compute path in mlrs.
**Trade-offs:** (+) kernels/algos compile once and are reused by all backends; genuine backend-agnosticism is compiler-enforced. (−) every wheel carries 2× monomorphization (f32+f64); estimator structs are generic, so the PyO3 layer needs a small dtype-dispatch match per method. Acceptable: the duplication is bounded and the dispatch is shallow.

**Example:**
```rust
// mlrs-kernels — F only
#[cube(launch)]
fn axpy<F: Float>(a: F, x: &Array<F>, y: &mut Array<F>) {
    if ABSOLUTE_POS < x.len() { y[ABSOLUTE_POS] = a * x[ABSOLUTE_POS] + y[ABSOLUTE_POS]; }
}
// mlrs-algos — F and R; calls launch::<F, R>
fn step<F: Float + CubeElement + Pod, R: Runtime>(c: &ComputeClient<R>, …) {
    axpy::launch::<F, R>(c, count, dim, /*args*/);
}
```

### Pattern 2: DeviceArray<R, F> — the buffer abstraction

**What:** A thin owner of `(ServerHandle, Shape, Strides, ComputeClient<R>)`. It is the mlrs analog of cuML's `CumlArray`: it knows its dtype (`F`), layout (C/F-contiguous via strides), and which client owns the memory. Constructed zero-copy from Arrow via `arrow_bridge.rs`; read back via `bytemuck::cast_slice`.
**When to use:** Anything that crosses host↔device or is passed between primitives.
**Trade-offs:** (+) tracks layout so GEMM/SVD primitives stay layout-correct (a top correctness pitfall per FEATURES.md C/F-contiguity note); (+) centralizes reuse/pooling. (−) one more wrapper type; must be careful that `DeviceArray` never outlives its `client`.

**Example:**
```rust
pub struct DeviceArray<R: Runtime, F: Float + CubeElement> {
    handle: cubecl::server::Handle,
    shape: Shape,      // smallvec
    strides: Strides,  // smallvec; encodes C vs F order
    client: ComputeClient<R>,
    _f: PhantomData<F>,
}
```

### Pattern 3: Prebuilt-primitive-first, hand-rolled-only-when-fused

**What:** Use `cubecl-matmul` (`launch::<R,F>(&Strategy::Auto, …)`, `TensorHandle`, `MatmulInputHandle`) and `cubecl-reduce` for the heavy linalg; hand-write `#[cube]` kernels only for fused/custom ops the prebuilt ops can't express (soft-threshold CD step, scatter-mean, top-k, sign-flip).
**When to use:** GEMM, covariance/Gram, reductions → prebuilt. Distance/CD/QN/top-k/scatter → custom kernels composed over prebuilt GEMM/reduce.
**Trade-offs:** (+) backend-portable, tuned heavy math for free (matmul manual `Strategy::Auto` picks per-backend kernel). (−) prebuilt crates must stay pinned at the exact cubecl `0.10.0` version (STACK.md version-compat table) or macro/ABI errors result.

### Pattern 4: Per-backend wheel via single feature axis

**What:** `mlrs-backend` declares `cpu/wgpu/cuda/rocm`; `mlrs-py` re-exports them; `maturin build --features wgpu` produces `mlrs-wgpu`, `--features cuda` produces `mlrs-cuda`, etc. `abi3-py312` → one wheel per (backend × platform) covers Python ≥3.12.
**When to use:** Release/packaging.
**Trade-offs:** (+) clean separation, user installs exactly the backend they have. (−) N wheels to build/publish; CI builds cpu+wgpu, cuda/rocm are compile-only here.

---

## Suggested Build Order (driven by component dependencies)

Ordered so each component is buildable and testable before its dependents. Consistent with the FEATURES.md primitive backbone (GEMM → distance/covariance/SVD → estimators).

```
Phase 0  FOUNDATION
  mlrs-core (traits, params, shape, dtype, error, oracle contracts)
  mlrs-backend skeleton: client(), DeviceArray<R,F>, arrow_bridge (Arrow→Bytes→handle→back)
  tests/ scaffolding + oracle harness (sign-flip + label-permutation helpers, approx 1e-5)
  CI matrix (--features cpu, --features wgpu); one trivial kernel end-to-end to prove the spine
        │  (proves: generic R/F flow, zero-copy ingest, read-back, oracle compare all work)
        ▼
Phase 1  CORE PRIMITIVES  (mlrs-kernels — validate each standalone before any estimator)
  GEMM (cubecl-matmul wrap)  +  reductions (sum/mean/argmin/L2)
  pairwise distance (built on GEMM + norms)      ← gates KMeans/DBSCAN/KNN
  covariance / Gram (XᵀX, built on GEMM)         ← gates OLS-eig/Ridge/PCA
        │  each primitive gets its own oracle test vs a host reference (ndarray ok in tests only)
        ▼
Phase 2  DECOMPOSITION PRIMITIVE  (the hard gate)
  SVD (Jacobi) and/or symmetric-eig of covariance ← gates PCA, TruncatedSVD, OLS-svd, Ridge-svd
        │  highest-risk primitive; validate standalone with sign-flip oracle helper
        ▼
Phase 3  CLOSED-FORM ESTIMATORS  (cheapest assembly once primitives exist)
  LinearRegression (svd path for sklearn match; eig as fast option)
  Ridge (regularized eig/svd — same primitive)
  PCA (center + eig/SVD of covariance) ; TruncatedSVD (SVD, no centering)  ← build PCA+TSVD together
        ▼
Phase 4  DISTANCE-BASED ESTIMATORS  (reuse pairwise distance)
  NearestNeighbors (distance + top-k select)
  KNeighborsClassifier / KNeighborsRegressor (gather + weighted vote/mean)
  KMeans (Lloyd: distance + argmin + scatter-mean ; k-means++ init for oracle match)
  DBSCAN (range query + connected-components)   ← highest clustering complexity
        ▼
Phase 5  ITERATIVE-SOLVER ESTIMATORS  (own solver kernels; convergence-parity risk)
  Coordinate-descent solver → Lasso + ElasticNet (one feature; ENet, Lasso = l1_ratio==1)
  Quasi-Newton (L-BFGS/OWL-QN) → LogisticRegression  ← highest correctness risk; research-flag
        ▼
Phase 6  PYTHON SURFACE  (mlrs-py — can begin in parallel after Phase 3 with one estimator)
  #[pyclass] wrappers, get/set_params, fit-returns-self, Arrow PyCapsule + numpy adapters,
  f32/f64 dispatch, NotFittedError mapping, per-backend wheel build, sklearn.estimator_checks
```

**Ordering rationale:**
- **Primitives before estimators** — FEATURES.md is explicit that SVD/eig gates two families and pairwise-distance gates three; building/validating them standalone turns each estimator into "mostly assembly."
- **Oracle harness in Phase 0, not later** — without sign-flip/label-permutation helpers, PCA/SVD/KMeans/DBSCAN tests fail at 1e-5 for non-bugs (FEATURES.md "oracle helpers are a prerequisite").
- **Closed-form (Phase 3) before iterative (Phase 5)** — closed-form estimators exercise the full Arrow→kernel→read-back→oracle pipeline with no convergence subtleties, de-risking the spine before the delicate CD/QN convergence-parity work.
- **mlrs-py can overlap from Phase 3** — once one estimator and the backend bridge exist, the Python wrapping + per-backend wheel mechanics can be validated against a single estimator while remaining estimators are still being built.

---

## Anti-Patterns

### Anti-Pattern 1: Binding `R` (or `F`) too deep in the stack

**What people do:** Make `mlrs-kernels` or `mlrs-algos` reference a concrete `WgpuRuntime`/`CpuRuntime`, or enable a backend Cargo feature in `mlrs-kernels`.
**Why it's wrong:** It defeats the "compile kernels once, reuse across backends" guarantee, reintroduces feature flags into the compute layer, and makes the generic-over-runtime requirement unverifiable. It also forces recompilation of all algorithm code per backend.
**Do this instead:** Keep `R` fully generic until `mlrs-py`. Only `mlrs-backend` names a runtime, and only behind `#[cfg(feature)]`. `mlrs-kernels` has **no** `#[cfg(feature = "...")]` at all.

### Anti-Pattern 2: Using ndarray/nalgebra in the compute path

**What people do:** Reach for host linear-algebra crates for GEMM/SVD because they're familiar.
**Why it's wrong:** They run on CPU only, bypass CubeCL, and break the generic-over-runtime contract — the math would never reach a GPU backend.
**Do this instead:** All compute math goes through CubeCL kernels (`cubecl-matmul`, `cubecl-reduce`, hand-written `#[cube]`). `ndarray` is permitted **only** in `tests/` for host-side reference math, never in `crates/*` compute code (STACK.md "What NOT to Use").

### Anti-Pattern 3: Per-iteration allocation in iterative solvers

**What people do:** Call `client.create`/`empty` every Lloyd/CD/QN iteration.
**Why it's wrong:** GPU allocation is the most expensive operation; per-iteration alloc churns the pool and tanks performance, undermining the memory-efficiency-first mandate.
**Do this instead:** Allocate working buffers once, retain handles in the estimator, reuse across iterations; carve multiple logical arrays from one buffer via in-kernel `slice_mut` (slicing manual); tune `MemoryConfiguration::ExclusivePages` for the high-frequency pattern (allocator manual).

### Anti-Pattern 4: Materializing fitted attributes eagerly at end of fit

**What people do:** Copy every `coef_`/`components_`/`centers_` to host the moment `fit` returns.
**Why it's wrong:** Forces device→host copies for attributes the caller may never read, wasting bandwidth — the exact copy the zero-copy mandate targets.
**Do this instead:** Keep fitted state as device handles (cuML `CumlArrayDescriptor` analog); materialize lazily on Python attribute access only.

### Anti-Pattern 5: Re-packing Arrow data into a `Vec` before upload

**What people do:** Iterate the Arrow array element-by-element into a `Vec<F>`, then upload.
**Why it's wrong:** Adds a full host copy + iteration that the contiguous `ScalarBuffer` layout makes unnecessary.
**Do this instead:** `array.values()` → `bytemuck::cast_slice` → `Bytes` directly (zero-copy manuals). Only fall to a copy when the source is genuinely non-contiguous (e.g., a sliced/offset Arrow view), and document it.

### Anti-Pattern 6: Matching cuML's default solver instead of sklearn's

**What people do:** Use cuML's defaults (OLS=`eig`, KMeans=`k-means||`, PCA=`jacobi`) because cuML is the porting reference.
**Why it's wrong:** The oracle is **scikit-learn**, not cuML. sklearn OLS uses SVD-based lstsq, KMeans defaults to `k-means++`, PCA `full` — picking cuML defaults causes spurious 1e-5 failures (FEATURES.md "biggest correctness risk").
**Do this instead:** Default each estimator to the variant that matches sklearn within 1e-5; offer cuML's faster variants as opt-in differentiators.

---

## Integration Points

### External Services / Libraries

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| CubeCL backends (cpu/wgpu/cuda/rocm) | `cubecl` umbrella crate features, bound in `mlrs-backend` only | Keep cubecl + cubecl-matmul/-reduce/-std all at exactly `0.10.0` or macro/ABI errors. On any build error consult the cubecl error guideline first (AGENTS.md). |
| Apache Arrow (arrow-rs) | `values()` → `bytemuck::cast_slice` → `Bytes` (host); `arrow-pyarrow`/PyCapsule at Python boundary | Primary data path; numpy (rust-numpy) is the secondary adapter. |
| scikit-learn (oracle) | Out-of-process Python script via `uv` → Arrow IPC fixtures under `tests/fixtures/`; default CI is pure-Rust against fixtures | Live PyO3 sklearn comparison gated behind `--features oracle-live` for ad-hoc parity (STACK.md). |
| PyO3 / maturin | `#[pymodule]` in `mlrs-py`; `maturin build --features <backend>` → per-backend wheel; `abi3-py312` | One wheel per (backend × platform) covers Python ≥3.12. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| mlrs-algos ↔ mlrs-kernels | direct `launch::<F, R>` calls | algos pass the client + DeviceArray handles; kernels are pure functions |
| mlrs-algos ↔ mlrs-backend | `ComputeClient<R>` + `DeviceArray<R,F>` | backend owns lifecycle/pooling; algos borrow |
| mlrs-py ↔ mlrs-algos | generic estimator + dtype-dispatch match | `R` monomorphized by feature; `F` by runtime dtype |
| everything ↔ mlrs-core | trait impls + shared types | the only crate with no internal deps; the dependency root |

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Small data (fits in one allocation) | Default `SubSlices` allocator; single `client.create` per array; no special handling. |
| Medium / iterative (many fit iterations) | Retain + reuse working buffers; `ExclusivePages` allocator tuning; in-kernel `slice_mut` to reduce allocation count. |
| Large data (approaching VRAM) | Batched distance computation (`max_mbytes_per_batch` analog for DBSCAN/KNN per FEATURES.md); `ExclusivePages` to prevent fragmentation-driven OOM; consider streaming chunks through a reused staging buffer. |
| Multi-device | **Out of scope for this milestone** (PROJECT.md defers multi-GPU); the single-device `R`-generic design does not preclude a later device-pool layer in `mlrs-backend`. |

### Scaling priorities
1. **First bottleneck: host↔device copies and per-iteration allocation.** Addressed by zero-copy ingest + buffer reuse from Phase 0/1 (memory-efficiency is per-phase, not deferred).
2. **Second bottleneck: f64 on wgpu.** Some adapters lack f64; `caps.rs` gating lets f64 paths skip gracefully rather than fail.

## Sources

- CubeCL `Cubecl_generics.md` — generic kernel definition, `Numeric`/`Float`/`CubeElement`/`Pod` bounds, `F::from_int`/`new`/`cast_from` constants, `launch::<N, R>` ordering, `run_with_type::<N, R>` driver pattern, f64 backend-support caveat. HIGH (authoritative project-pinned manual).
- CubeCL `Backend-Agnostic_Buffer_Slicing_…md` — in-kernel `slice_mut` to carve multiple logical arrays from one allocation; `execute_slicing<R: Runtime>` driver; alloc-reduction rationale. HIGH.
- CubeCL `Choreographing_Parallel_Execution_…md` — `ABSOLUTE_POS`/`UNIT_POS`/`CUBE_POS`, bounds checks, `execute_*<R: Runtime, N: Numeric + CubeElement + Pod>` driver shape. HIGH.
- CubeCL `Tuning_ExclusivePages_Allocator_…md` — `MemoryConfiguration::ExclusivePages`, `RuntimeOptions`, `init_setup`/`init_device`, `ComputeClient` construction for buffer-reuse tuning. HIGH.
- CubeCL `cubecl_matmul_gemm_example.md` — `ComputeClient::load`, `create_tensor`/`empty_tensor`/`read_tensor` with strides, `TensorHandle::<R,F>::new`, `MatmulInputHandle::Normal`, `launch::<R,F>(&Strategy::Auto,…)`. HIGH.
- optimisor `ZERO_COPY_ARROW_CUBECL.md` — `Float32Array.values()` → `bytemuck::cast_slice` → `client.create(Bytes::from_bytes_vec(…))`; `run_arrow_ingestion_test<R: Runtime>`; read-back via `client.read_one` + `cast_slice::<u8,f32>`. HIGH.
- optimisor `ZERO_COPY_TRANSMUTATION_CUBECL.md` — `bytemuck` Pod/Zeroable invariants, host↔device byte reinterpretation, `Bytes::from_bytes_vec`. HIGH.
- `.planning/research/STACK.md` — proposed 5-crate layout (refined here), version pins, feature-flow isolation, oracle strategy, "what not to use." HIGH (project canon, refined).
- `.planning/research/FEATURES.md` — primitive dependency backbone (GEMM→distance/SVD→estimators), per-estimator solver-vs-sklearn risks, oracle sign-flip/label-permutation requirement, MVP build order. HIGH.
- `.planning/codebase/ARCHITECTURE.md` & `STRUCTURE.md` — cuML `handle_t`/`CumlArray`/`Base`/`CumlArrayDescriptor`/`@reflect` layered design being ported (collapsed into core+kernels+backend+algos+py). HIGH (read directly).
- `.planning/PROJECT.md` — v1 scope, constraints (compile-time backend, runtime dtype, 1e-5 oracle, memory-first, tests separated). HIGH (project canon).

---
*Architecture research for: sklearn-compatible GPU ML library in Rust (CubeCL + Arrow + PyO3)*
*Researched: 2026-06-11*
