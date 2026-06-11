# Pitfalls Research

**Domain:** GPU-accelerated, sklearn-compatible ML library in Rust (CubeCL kernels generic over float + runtime; Apache Arrow zero-copy; PyO3 per-backend wheels; scikit-learn 1e-5 oracle; wgpu+cpu CI gate)
**Researched:** 2026-06-11
**Confidence:** HIGH for CubeCL/Arrow/wgpu/sklearn-oracle pitfalls (grounded in the provided CubeCL error guideline, the CubeCL/optimisor manuals, the cuML codebase concerns, and verified wgpu/maturin facts). MEDIUM for the precise PyO3 per-backend packaging mechanics (verified against maturin docs + the WebGPU spec, but no first-party "do it this way" prescription exists for this exact multi-backend-wheel combination).

---

## How to read this document

Pitfalls are grouped by the six risk areas the milestone called out, ordered roughly by how early they bite and how expensive they are to fix late. The "Phase to address" field uses the natural phase backbone implied by FEATURES.md's dependency graph:

- **P0 Foundation** — workspace, backend abstraction, Arrow↔CubeCL bridge, device buffer/allocator, oracle harness
- **P1 Compute primitives** — GEMM, reductions, pairwise distance, SVD/eig, CD solver, QN solver (each validated standalone)
- **P2 Estimators** — linear models, clustering, decomposition, KNN assembled on the primitives
- **P3 Python packaging** — PyO3 estimators, Arrow PyCapsule ingest, per-backend wheels

Several pitfalls are cross-cutting and must be designed-in at P0 even though they only *manifest* in a later phase — those are flagged explicitly, because retrofitting them is what causes rewrites.

---

## Critical Pitfalls

### Pitfall 1: Treating the 1e-5 oracle as a kernel-correctness check when it is really a *solver-and-defaults-equivalence* check

**What goes wrong:**
Teams write a numerically correct kernel, compare to scikit-learn, and get errors of 1e-2 to 1e-1 — not because the kernel is wrong, but because sklearn's *default* solver/init differs from the one implemented (which is often cuML's default). The result is days spent "debugging" a kernel that is actually correct, or worse, distorting the kernel to chase sklearn's numerical fingerprint. FEATURES.md already enumerates the specific mismatches; they are the single biggest correctness risk in the project:
- **OLS:** sklearn uses SVD-based `lstsq` (`gelsd`); cuML defaults to `eig` on XᵀX. On ill-conditioned `X`, the eig path drifts well past 1e-5.
- **KMeans:** sklearn defaults to `k-means++`; cuML to scalable `k-means||`. These converge to *statistically* similar but numerically different centroids — never 1e-5.
- **TruncatedSVD:** sklearn defaults to `algorithm='randomized'` (stochastic, non-deterministic). Comparing against it at 1e-5 is impossible by construction.
- **LogisticRegression:** sklearn `lbfgs` multinomial vs. cuML softmax-`qn`; the converged coefficients depend on penalty normalization and the multinomial formulation.
- **PCA:** sign ambiguity of components (sklearn applies `svd_flip`); two correct implementations differ by per-column sign.

**Why it happens:**
The oracle is scikit-learn, not cuML (a deliberate PROJECT decision so CI runs without a GPU), but the *reference algorithm* being ported is cuML. Engineers naturally port cuML's default solver and assume "matches cuML ⇒ matches sklearn." The 1e-5 gate then fails for non-bug reasons and erodes trust in the harness.

**How to avoid:**
- For every estimator, pin the v1 solver to *whatever variant reproduces sklearn's default*, per the table in FEATURES.md "Competitor Feature Analysis" — OLS=`svd`, KMeans=`k-means++` (deterministic seed or explicit-array init for the strictest tests), TruncatedSVD oracle compared against sklearn `algorithm='arpack'` (deterministic) not the `randomized` default, PCA with `svd_flip` sign convention.
- Build **sign-flip and label-permutation comparison helpers into the oracle harness in P0**, before any estimator exists. cuML ships `assert_dbscan_equal` and a fuzzy `array_equal(unit_tol, total_tol)` exactly because raw element-wise comparison fails spuriously (see TESTING.md). Mirror both.
- Encode tolerance with `approx` (`epsilon`/`max_relative`), and run **both f32 and f64**; reserve the strict 1e-5 for f64 and use a documented, looser-but-justified tolerance for f32 (see Pitfall 2).
- Seed both sides from the *same* RNG so inputs are bit-identical (`StdRng::seed_from_u64`), and persist the seed on failure (cuML's `failure_logger` pattern).

**Warning signs:**
- Errors clustered around 1e-2..1e-1 (solver mismatch) rather than 1e-7..1e-5 (accumulation noise).
- A test that passes on f64 but fails f32 by a *constant factor per column* → sign/permutation issue, not precision.
- "Fixing" a kernel makes one estimator pass and another regress → you are fitting to sklearn's quirks, not computing correctly.

**Phase to address:** P0 (oracle harness with sign/permutation helpers + per-estimator solver-choice table is a prerequisite, not an afterthought). Re-checked in every P2 estimator phase.

---

### Pitfall 2: f32 accumulation drift silently blowing the 1e-5 budget on reductions, GEMM, and distance kernels

**What goes wrong:**
Naive parallel summation in f32 (centroid sums, XᵀX, variance/mean centering, pairwise squared distances, dot products) accumulates rounding error proportional to N and to the summation order. A 10⁵-element f32 reduction can lose 3–4 significant digits, putting results at ~1e-3 — 100× over budget. Pairwise Euclidean distance via the `‖a‖² + ‖b‖² − 2a·b` expansion (the GEMM-friendly form everyone uses) catastrophically cancels for nearby points in f32, producing small-negative "distances" that become NaN under `sqrt`.

**Why it happens:**
GPUs make f32 the fast/default path, and the obvious reduction (flat atomic-add or a single-pass tree) is order-dependent and low-precision. The `-2a·b` distance trick is taught everywhere but is numerically unstable in f32. cuML's own QN softmax path (CONCERNS.md) shows even the reference accumulates in stages for stability.

**How to avoid:**
- **Accumulate reductions in higher precision than the storage type.** For an f32 kernel, carry the accumulator in f32 but use a numerically stable *tree reduction* (the `Cubecl_shared_memory.md` `sync_cube` tree pattern), not a flat atomic add; for the tightest paths, accumulate in f64 and downcast the result. Document this per-primitive.
- For pairwise distance, **clamp the squared distance to ≥ 0 before `sqrt`** (`d2 = max(d2, 0)`), and prefer computing `‖a−b‖²` directly for the small-k / DBSCAN range-query path where the GEMM trick's cancellation hurts most.
- Make the **f32 tolerance explicit and separate** from f64 in the harness. f64 carries the strict 1e-5; f32 gets a documented, justified band (cuML itself uses `unit_tol=1e-4` fuzzy comparisons in TESTING.md). PROJECT says both are validated — that does not mean both at the identical epsilon.
- Validate each primitive's accumulation behavior **standalone in P1** against a high-precision host reference (ndarray/f64) *before* it is buried inside an estimator, so drift is attributed to the primitive, not the algorithm.

**Warning signs:**
- f64 oracle passes, f32 fails by ~10⁻³ and the gap grows with `n_samples` → accumulation order/precision.
- NaNs appearing only in KMeans/DBSCAN/KNN on tightly clustered data → negative squared distance under `sqrt`.
- Results that change when you change `CubeDim`/cube count → order-dependent reduction (also a correctness smell, not just precision).

**Phase to address:** P1 (every reduction/distance/GEMM primitive carries a documented accumulation strategy and standalone high-precision test). Tolerance policy fixed in P0 harness.

---

### Pitfall 3: Violating CubeCL's `#[cube]` IR constraints — the AGENTS.md "consult the error guideline first" rule exists because these fail in non-obvious ways

**What goes wrong:**
Kernels fail to compile (or compile but lower incorrectly) due to CubeCL `#[cube]`-specific restrictions that look like ordinary Rust but are not. The provided error guideline documents the exact recurring failures:
- **Calling a plain Rust helper from inside `#[cube]`** → `E0433 "<fn> is not a crate or module"`. Helpers must themselves be `#[cube]` or be inlined.
- **`let x = if cond { a } else { b };` as an expression** → `E0308 mismatched types: expected ExpandElementTyped<_> found {float}`. Must be rewritten as a mutable-binding + `if` *statement*.
- **Method-style math calls `x.exp()` / `x.sqrt()`** → `E0599 no method named __expand_exp_method`. Must use associated-function form `F::exp(x)` / `F::sqrt(x)` (or trait `Exp::exp`).
- **Raw numeric literals (`2.0`, `10`) in a generic-over-float kernel** → won't unify with `F`; must use `F::from_int(i)` / `F::cast_from(f64)` / `F::new(...)`.
- **`u64`/`usize` arithmetic inside kernels** → unsupported device types; use `u32`/`i32` for indices, cast `as usize` only at the indexer.

**Why it happens:**
`#[cube]` is a procedural macro that rewrites the function into CubeCL IR; it does not execute as Rust. Idiomatic Rust (expression-`if`, method calls, free helper functions, `usize` indices) is the natural thing to write and the natural thing to "fix" blindly when it errors — which AGENTS.md *strictly prohibits* (a critical-failure rule).

**How to avoid:**
- Adopt the guideline's patterns as **lint-level conventions from the first kernel**: mutable-binding + `if`-statement instead of `if`-expression; associated-function math; `#[cube]` (or inlined) helpers only; `F::from_int`/`F::cast_from` for all constants; `u32`/`i32` indices.
- On *any* CubeCL build/feature/toolchain error, **read `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md` and the `cubecl_error_solution_guide/` entries before touching code** (mandatory per AGENTS.md), and document root cause + resolution + prevention in the guideline's template.
- Keep `mlrs-kernels` **feature-free and backend-agnostic** (STACK.md layout) so a `#[cube]` mistake surfaces once at compile time for all backends, not separately per wheel.

**Warning signs:**
- `E0433`/`E0308`/`E0599` mentioning `ExpandElementTyped`, `__expand_*`, or "not a crate or module" — these are CubeCL-IR signatures, not ordinary Rust errors.
- Type-inference errors on numeric literals inside a `<F: Float>` kernel.
- A kernel that compiles on `cpu` but errors on `wgpu` (or vice versa) → feature/IR-lowering issue; do not paper over per-backend.

**Phase to address:** P1 (first kernel sets the conventions). Enforced continuously; AGENTS.md protocol is a standing rule across all kernel work.

---

### Pitfall 4: Building the codebase around f64, then discovering wgpu (the CI gate) has no f64 on most adapters and none in browser WebGPU

**What goes wrong:**
PROJECT requires both f32 and f64 validated, and f64 is what makes the 1e-5 tolerance comfortable — so the path of least resistance is to develop and test on f64 first. But the **WebGPU specification has no 64-bit float in WGSL at all**, and wgpu's `SHADER_F64` is a *native-only* feature that many Vulkan/Metal/DX12 adapters do not expose. The result: f64 oracle tests that pass on the `cpu` backend (and on CUDA) **fail or are silently skipped on the `wgpu` CI gate**, the very gate PROJECT designates as primary. If the architecture hard-codes f64 expectations, large parts of the suite can't run on the gate that's supposed to protect them.

**Why it happens:**
CubeCL's generics manual itself warns "some hardware backends might have limited or no support for `f64`." Developers see f64 working on `CpuRuntime` and assume portability. WebGPU's lack of f64 is non-obvious and adapter-dependent, so it passes on the dev machine's native Vulkan adapter and breaks in browser/CI/headless adapters.

**How to avoid:**
- **Feature-gate f64 at runtime, mirror the half-precision pattern.** Before launching an f64 kernel, query `client.properties()` (the half-precision manual uses `feature_enabled(Feature::Type(Elem::Float(FloatKind::F16)))`; the f64 analogue is the same query for `FloatKind::F64`). If absent, **skip/xfail with a clear reason** rather than fail — exactly as STACK.md's "Guard f64 paths" note prescribes.
- Make **f32 the portable correctness baseline** that *must* pass on wgpu, and treat f64 as "validated where supported (cpu always; wgpu/CUDA when the adapter exposes it)." This keeps the wgpu gate green and honest.
- Decide the f64-on-wgpu policy in **P0** (backend abstraction + capability query lives in `mlrs-backend`), not when the first f64 estimator test mysteriously skips.
- In CI, log which backend/adapter ran which dtype so a "passing" run that actually skipped all f64 on wgpu is visible.

**Warning signs:**
- f64 tests green on `cpu`, skipped/red on `wgpu`.
- `feature_enabled(... F64)` returns false on the CI adapter but true locally.
- A "100% pass" CI run where the f64 count silently dropped to zero on the wgpu job.

**Phase to address:** P0 (capability-query + dtype-skip policy in the backend layer and harness). Validated again the first time an f64 estimator test runs on wgpu (early P2).

---

### Pitfall 5: Plane/subgroup and workgroup-limit assumptions that pass on CUDA but break or skip on wgpu

**What goes wrong:**
A reduction or scan written with plane (warp/subgroup) intrinsics — `plane_shuffle_xor`, `plane_inclusive_sum`, `plane_elect` — runs on CUDA (warp=32, subgroups always available) but on wgpu **requires the `subgroups` feature** that the browser/hardware may not support (per `Cubecl_plane.md` §6). Worse, code that hard-codes a plane size of 32 (`while i < 32`) gives *wrong answers* on adapters with plane size 16/64/128 instead of failing loudly. Separately, shared-memory tile sizes and `CubeDim` chosen for an NVIDIA SM can exceed wgpu's lower `maxComputeWorkgroupStorageSize` / `maxComputeInvocationsPerWorkgroup` limits and fail device creation or kernel dispatch on the CI gate.

**Why it happens:**
cuML's CUDA heritage bakes in 32-thread-warp assumptions — CONCERNS.md counts **17 warp-size-dependent sites** (`warpSize`, `laneId`, shuffles) in the reference. Porting that mental model directly to CubeCL plane ops reproduces the assumption. The plane manual is explicit ("Never assume a plane size of 32. Always use `PLANE_DIM`"), but the CUDA reference makes 32 feel safe.

**How to avoid:**
- **Always use `PLANE_DIM` / `UNIT_POS_PLANE`, never the literal 32**, and write plane loops as `while i < PLANE_DIM` (power-of-two fold) per the plane manual's portable reduction example.
- **Feature-gate plane ops** behind a `client.features()` query and provide a **shared-memory tree-reduction fallback** (the `Cubecl_shared_memory.md` `sync_cube` pattern) for adapters without subgroups. STACK.md's `#[comptime]`/trait-specialization (`SumBasic`/`SumPlane`) is exactly this — design both paths so wgpu always has a working route.
- Keep shared-memory tile sizes and `CubeDim` **conservative and queryable**; do not hard-code an NVIDIA-sized tile. Validate against wgpu's lower workgroup-storage/invocation limits in P1 when the reduction/distance primitives are built.
- For ROCm specifically, note `Handling_Interleaved_Complex_Numbers..._ROCm` and the warp-size caveat: AMD wavefronts are 32 or 64 — another reason `PLANE_DIM` is mandatory.

**Warning signs:**
- Reduction correct on CUDA, wrong (off by a factor) on a wgpu adapter with non-32 plane size.
- `subgroups`-feature-not-supported error, or kernel silently producing partial sums on wgpu.
- Device-creation / dispatch failure citing workgroup storage or invocation limits only on the CI adapter.

**Phase to address:** P1 (reduction, scan, distance primitives — build plane path + shared-memory fallback together, both tested on wgpu). Re-touched whenever a new collective kernel is added.

---

### Pitfall 6: Arrow zero-copy transmutation that is unsound across alignment, lifetime, ownership, and validity-bitmap boundaries

**What goes wrong:**
The documented zero-copy path is `Float32Array::values()` → `&[f32]` → `bytemuck::cast_slice` → `&[u8]` → `client.create(Bytes::from_bytes_vec(...))`. Subtle failures:
- **Sliced/offset Arrow arrays:** `array.values()` returns the *full* backing buffer ignoring the array's logical offset/length unless you account for it; a sliced `Float32Array` uploads the wrong window. Reading `values()[i]` is not `array.value(offset+i)` once an offset exists.
- **Nullability:** Arrow arrays carry a validity bitmap. `values()` exposes the raw values buffer including garbage at null positions; uploading it straight to the device silently feeds NaN/garbage into the math. sklearn rejects NaN by default — so "zero-copy" can both differ from the oracle *and* hide a data-quality bug.
- **Lifetime/ownership:** `cast_slice` borrows; the moment you do `.to_vec()` you copy (acceptable), but holding a `&[u8]` view into an Arrow buffer that the Python side may drop/realloc (esp. across the PyCapsule FFI boundary) is a use-after-free. The borrow must outlive the upload, or the bytes must be owned.
- **Alignment / dtype mismatch:** `bytemuck::cast_slice` *panics* (not UB) on misaligned or non-divisible length — fine in a test, a hard crash in production if an upstream buffer (e.g., an offset slice, or an f64 buffer reinterpreted as f32) violates it.

**Why it happens:**
The manuals demonstrate the *happy path* on a freshly-constructed, full, non-null, aligned `Float32Array`. Real inputs from pandas/Polars/PyArrow arrive sliced, nullable, and owned by Python. CONCERNS.md shows cuML itself wraps all of this behind `CumlArray` + `check_inputs()` precisely because raw buffer handling is error-prone.

**How to avoid:**
- **Centralize the Arrow→Bytes bridge in `mlrs-backend`** (one audited place, per STACK.md layout) and validate at that boundary: reject or densify sliced/offset arrays (account for offset before `values()`), reject arrays with non-empty validity bitmaps (or surface a clear "NaN/null not supported" error matching sklearn semantics), and assert dtype/alignment *before* `cast_slice` so you get a typed error not a panic.
- For the **Python↔Rust FFI**, use the Arrow C Data Interface / PyCapsule protocol (`arrow-pyarrow`/`pyo3-arrow`, STACK.md) which transfers *ownership* of the buffer with a release callback — do not hold a bare `&[u8]` into a Python-owned buffer past the call. Keep the `ArrayRef` alive for the duration of the upload.
- Treat the device upload as **owning** the bytes for the kernel's lifetime (the manuals' `Bytes::from_bytes_vec(...to_vec())` copies once and is sound); reserve true borrow-only zero-copy for the host-resident `cpu` backend where the lifetime is controllable.
- Add a contiguity/finiteness validation step at the PyO3 boundary (FEATURES.md "Input validation" table stakes) mirroring cuML's `check_inputs()`.

**Warning signs:**
- Results correct for freshly-built arrays, wrong for `df["col"][a:b]` slices → ignored Arrow offset.
- Sporadic NaN in outputs traced to null positions in input → validity bitmap ignored.
- Segfault / freed-memory crash under load, only across the Python boundary → lifetime/ownership bug.
- `bytemuck` panic on cast → alignment or length-divisibility violation.

**Phase to address:** P0 (the bridge + validation in `mlrs-backend`). FFI ownership specifics revisited in P3 (PyO3 PyCapsule ingest).

---

### Pitfall 7: Per-backend wheels that collide on PyPI, ship the wrong driver, or import-fail at runtime

**What goes wrong:**
The plan is one wheel per backend (`cuda`/`rocm`/`wgpu`/`cpu`) built via `maturin build --features <backend>`. Failure modes:
- **Name collision:** maturin derives the wheel/distribution name from the Cargo *package* name. Building four feature variants of the same package produces **four wheels with the same name and version** — they overwrite each other locally and cannot coexist on PyPI. A user `pip install mlrs` gets whichever was uploaded last, not their backend.
- **Driver loading:** the `cuda`/`rocm` wheels link against CUDA/HIP runtimes that **must exist on the user's machine**; importing the extension with no driver gives a cryptic dynamic-link error at `import`, not a friendly "no GPU backend." cuML's own stack (CONCERNS.md) hard-pins CUDA Toolkit 13.x for this reason.
- **GIL/initialization:** device/client creation and the first kernel JIT can be slow and must not happen under contention; long compute must release the GIL or it serializes all Python threads and can appear to hang.
- **abi3 vs. native:** an `abi3-py312` wheel is portable across 3.12+ *for the Python ABI*, but the **backend driver ABI is orthogonal** — abi3 does not make a CUDA wheel run without CUDA.

**Why it happens:**
maturin's docs cover the single-package happy path; the multi-backend-wheel pattern is unusual and undocumented (confirmed: maturin names the wheel from the package name, with no first-party recipe for feature-variant distributions). Driver-loading failures only appear on machines unlike the build machine, so CI (which builds `wgpu`+`cpu`) never sees the CUDA import failure.

**How to avoid:**
- **Give each backend its own distribution name** (e.g., `mlrs-cpu`, `mlrs-wgpu`, `mlrs-cuda`, `mlrs-rocm`), or a single package with optional native components selected at install time — but do *not* publish same-name/same-version wheels per feature. Decide this naming scheme in P0 so the workspace/`pyproject.toml` is structured for it, even though wheels ship in P3.
- **Catch driver-load failure at import and re-raise a clear error** ("mlrs-cuda requires a CUDA 13.x runtime; none found") instead of leaking the linker error. Provide a tiny `available_backends()` probe.
- **Release the GIL around device compute** (`Python::allow_threads`) and do device/client init lazily-but-once; warm the JIT on first `fit`. Map `MlrsError` (thiserror) to Python exceptions at the boundary (STACK.md).
- CI builds only `wgpu`+`cpu`; **smoke-test the `cuda`/`rocm` wheels' *import-and-clear-error-without-driver* behavior** even though their compute can't run here — that's the failure mode the gate can actually catch.

**Warning signs:**
- Two backend wheels with identical filename → name-collision.
- `ImportError`/`OSError` referencing `libcuda.so`/`libamdhip64.so` on user machines.
- Python multithreaded callers serializing or hanging during `fit` → GIL not released.

**Phase to address:** P3 (packaging), but the **distribution-naming scheme and `available_backends()` contract must be fixed in P0** to avoid restructuring the workspace late.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Develop/validate on `cpu` backend only, defer `wgpu` | Fast iteration, f64 always works, no adapter quirks | wgpu is the *real* CI gate; plane/f64/workgroup-limit bugs (Pitfalls 4,5) surface in a batch at the end and force kernel rewrites | Only for the very first spike of a brand-new kernel; wgpu must run in CI from P1 onward |
| Port cuML's default solver/init verbatim | Faithful to the reference; fastest to write | Fails the sklearn oracle (Pitfall 1) — wrong default for OLS/KMeans/TruncatedSVD | Only as a *differentiator* solver alongside the sklearn-matching default, never as the v1 default |
| Hard-code constants/tile sizes/plane=32 in kernels | Compiles, runs on dev GPU | Breaks generic-over-float (literals) and generic-over-runtime (plane/tile) portability | Never for literals (use `F::from_int`); never for plane size (use `PLANE_DIM`) |
| Use the `‖a‖²+‖b‖²−2a·b` distance everywhere | One GEMM, fast | f32 cancellation → NaN on near points (Pitfall 2) | Acceptable for the large-block KNN/KMeans path *with* `max(d2,0)` clamp; use direct `‖a−b‖²` for DBSCAN range queries |
| `Bytes::from_bytes_vec(...to_vec())` copy on every upload | Sound, simple, matches the manuals | One host copy per fit — contradicts the "memory efficiency first-class" mandate at scale | Fine for v1 correctness; P1+ should add buffer reuse/pooling (`client.empty` reuse, `ExclusivePages`) for hot temporaries |
| One `mod tests` next to source "just for now" | Convenient | **Violates AGENTS.md critical-failure rule** (source/test separation) | Never — use `tests/` or `*_test.rs` |
| Skip f64 capability query, assume present | Less boilerplate | Silent skips/failures on wgpu CI (Pitfall 4) | Never; the query is one line and the half-precision manual already models it |

---

## Integration Gotchas

Common mistakes when connecting CubeCL, Arrow, PyO3, and the sklearn oracle.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| CubeCL `#[cube]` ↔ Rust helpers | Calling a plain `fn` from inside a kernel (`E0433`) | Mark helper `#[cube]` or inline it (error guideline §1) |
| CubeCL math | `x.exp()` / `x.sqrt()` method calls (`E0599`) | `F::exp(x)` / `F::sqrt(x)` associated functions (error guideline §2) |
| CubeCL control flow | `let v = if c {a} else {b};` (`E0308 ExpandElementTyped`) | Mutable binding + `if` statement (error guideline §1) |
| CubeCL launch generics | Wrong generic order on `::launch` | Kernel generics first, then `R`: `kernel::launch::<F, R>(...)`; dynamic-vectorization factor passed as runtime `usize` after `CubeDim` |
| Arrow → CubeCL | `values()` on a sliced/nullable array | Account for offset/length; reject/handle validity bitmap; validate alignment before `cast_slice` |
| PyArrow → Rust FFI | Holding `&[u8]` into a Python-owned buffer | Use Arrow C Data Interface / PyCapsule (`pyo3-arrow`) ownership transfer with release callback |
| PyO3 ↔ device compute | Long `fit` holding the GIL | `Python::allow_threads` around compute; lazy-once client init |
| maturin per-backend | Same package name for all feature variants | Distinct distribution names (`mlrs-cpu`/`mlrs-wgpu`/…) |
| scikit-learn oracle | Element-wise compare of PCA/KMeans/DBSCAN output | Sign-flip + label-permutation helpers (mirror `assert_dbscan_equal`, fuzzy `array_equal`) |
| cubecl crate versions | Mixing `cubecl`/`cubecl-matmul`/`-reduce`/`-std` versions | Pin all to the same `0.10.0`; mismatch → macro/ABI errors (consult error guideline) |

---

## Performance Traps

Patterns that work at small oracle-test scale but fail as data grows (memory efficiency is a first-class PROJECT requirement).

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Allocate a fresh device buffer every fit/iteration | Growing latency, allocator churn, fragmentation | Buffer reuse / pooling; `client.empty` reuse; tune `ExclusivePages` (CubeCL allocator manual) for iterative KMeans/CD/QN loops | High-frequency iterative algorithms (KMeans, coordinate descent, QN) — accumulates fast |
| Hidden host↔device round-trips per iteration | Throughput far below PCIe peak; CPU pegged | Keep data device-resident across iterations; read back only final results; batch reads (the zero-copy manuals keep data on-device) | cuML hit exactly this in genetic/TSNE (CONCERNS.md "excessive implicit host-device transfers") |
| f32 reduction over large N without stable accumulation | Error grows with `n_samples`, oracle fails at scale | Tree reduction; higher-precision accumulator (Pitfall 2) | Large-sample regression/PCA/KMeans (10⁴–10⁶ rows) |
| Materializing the full N×N pairwise distance matrix | OOM on the GPU; fails on memory-limited wgpu/CI adapters | Tile/batch the distance computation (cuML's `max_mbytes_per_batch` for DBSCAN); double-buffer staging (CubeCL double-buffering manual) | DBSCAN/KNN/KMeans at N ≳ 10⁴ where N² exceeds VRAM |
| int32 indices for `n_samples * n_clusters`/N² | Silent overflow → wrong labels/garbage | Use i64/u64 index space where products can exceed INT32_MAX | cuML had a real KMeans int32 overflow bug (CONCERNS.md); N²·k products overflow fast |
| Two global allocators or allocator set in a library crate | Link failure / undefined behavior | Exactly one `#[global_allocator]` (mimalloc), declared only in the `mlrs-py` cdylib (STACK.md) | At link time of the Python extension |

---

## Security / Soundness Mistakes

Domain-specific issues beyond general practice (this is a numeric/FFI library, so "security" is mostly memory soundness + supply chain).

| Mistake | Risk | Prevention |
|---------|------|------------|
| `unsafe ArrayArg::from_raw_parts` / `cast_slice` with mismatched length or alignment | Out-of-bounds device read, panic, or UB | Validate length == buffer size / `size_of::<F>()` and alignment before the unsafe call; the manuals always pass exact `typed_slice.len()` |
| Borrowing into a Python-owned Arrow buffer past the call | Use-after-free across the FFI boundary | PyCapsule ownership transfer with release callback; keep `ArrayRef` alive (Pitfall 6) |
| Uploading Arrow values buffer that includes null/garbage positions | Garbage/NaN silently enters the model; diverges from sklearn | Honor the validity bitmap; reject or document null handling (sklearn rejects NaN by default) |
| Trusting any-bit-pattern transmutes on non-`Pod` types | UB | `bytemuck::Pod`/`Zeroable` bounds (compile + runtime checked); never hand-transmute |
| Unpinned/unhashed transitive GPU-driver deps | Supply-chain drift (cuML pins to a single RAPIDS tag with no hash; CONCERNS.md) | Pin `cubecl`/`cubecl-*` to exact `0.10.0`; lockfile committed; review driver-crate updates |

---

## Looks Done But Isn't

Verification checklist — passing these is what separates "demo on my GPU" from "holds the PROJECT contract."

- [ ] Each estimator passes the oracle on **both `cpu` and `wgpu`**, not just `cpu`.
- [ ] f64 tests **actually ran** (not silently skipped) on whatever backend reports f64 support; the skip count is logged, not hidden.
- [ ] f32 *and* f64 both validated, with **separate, documented tolerances** (strict 1e-5 reserved for f64).
- [ ] Sign-flip handled for PCA/TSVD; label-permutation handled for KMeans/DBSCAN — tested, not assumed.
- [ ] Distance kernels clamp negative squared distances; no NaN on tightly-clustered fixtures.
- [ ] Sliced and nullable Arrow inputs are handled or cleanly rejected (not silently mis-uploaded).
- [ ] Every primitive (GEMM, reduce, distance, SVD/eig, CD, QN) has a **standalone** high-precision test before its estimator depends on it.
- [ ] No `mod tests` inside any source file (AGENTS.md).
- [ ] Every CubeCL build error resolved via the error guideline, with a documented root-cause/prevention note.
- [ ] Plane-using kernels have a shared-memory fallback and use `PLANE_DIM`, never literal 32.
- [ ] Per-backend wheels have distinct distribution names and a clear no-driver import error.
- [ ] `fit` releases the GIL around device compute.
- [ ] Exactly one `#[global_allocator]`, only in the cdylib crate.

---

## Pitfall → Phase Map (for roadmap construction)

| Pitfall | Primary phase | Designed-in earlier? |
|---------|---------------|----------------------|
| 1. Solver/defaults vs. sklearn oracle | P0 harness + each P2 estimator | Sign/permutation helpers + solver table in **P0** |
| 2. f32 accumulation drift | P1 primitives | Tolerance policy in **P0** |
| 3. CubeCL `#[cube]` IR constraints | P1 (first kernel) | Conventions set at first kernel; AGENTS.md standing rule |
| 4. f64 absent on wgpu | P0 backend capability query | Yes — **P0** |
| 5. Plane/subgroup/workgroup limits | P1 collective kernels | Fallback designed with the primitive |
| 6. Arrow zero-copy soundness | P0 bridge in `mlrs-backend` | FFI ownership re-checked in P3 |
| 7. Per-backend wheel packaging | P3 packaging | **Distribution-naming + `available_backends()` decided in P0** |

**Roadmap implication:** P0 is unusually load-bearing here — the oracle harness (with sign/permutation helpers and tolerance policy), the backend capability-query layer (f64/plane gating), the Arrow bridge with validation, and the packaging naming scheme must all be established before estimators are built. Five of the seven critical pitfalls are either prevented in P0 or require a P0 design decision. Phasing that defers any of these to "when we package" or "when we add wgpu" converts them from one-line guards into rewrites.

---

## Sources

- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/` (`mismatched types.md`, `calling a "normal" Rust function...md`) — `#[cube]` IR failure modes (E0433/E0308/E0599), `if`-statement and associated-function fixes, type restrictions. HIGH confidence (authoritative project-mandated guideline).
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` — `Cubecl_generics.md` (f64 backend-support warning, `F::from_int`, launch generic order), `Cubecl_plane.md` (PLANE_DIM, subgroup feature requirement, "never assume 32"), `Cubecl_shared_memory.md` (`sync_cube` tree reduction, static sizing), `Cubecl_dynamic_vectorization.md` (vectorization factor as runtime arg, alignment), `Staging_..._Double_Buffering...md`, `Tuning_ExclusivePages_Allocator...md`, `Handling_Interleaved_Complex_Numbers..._ROCm.md` (warp-size/wavefront caveats). HIGH confidence.
- `/home/user/Documents/workspace/optimisor/manual/` — `ZERO_COPY_ARROW_CUBECL.md` & `ZERO_COPY_TRANSMUTATION_CUBECL.md` (Arrow `values()`→bytemuck→`Bytes` path, `cast_slice` panic-on-misalignment), `HALF_PRECISION_CUBECL.md` (`feature_enabled`/`supports_type` capability-query pattern reused for f64), `MIMALLOC_MANUAL.md` (single global allocator). HIGH confidence.
- `/home/user/Documents/workspace/mlrs/AGENTS.md` — mandatory CubeCL error-guideline protocol, generics-over-float rule, source/test separation critical-failure rule. HIGH confidence (project canon).
- `.planning/PROJECT.md`, `.planning/research/STACK.md`, `.planning/research/FEATURES.md` — scope, solver-default mismatch table, primitive dependency graph, wheel/backend layout, f64-on-wgpu guard note. HIGH confidence.
- `.planning/codebase/CONCERNS.md` & `TESTING.md` — cuML int32-overflow bug, 17 warp-size sites, host↔device transfer debt, `assert_dbscan_equal` / fuzzy `array_equal` oracle helpers, CUDA-driver pinning. HIGH confidence (read directly).
- [wgpu `Features` (SHADER_F64 is native-only)](https://docs.rs/wgpu/latest/wgpu/struct.Features.html), [WebGPU issue #2805 — no f64 in WebGPU/WGSL](https://github.com/gpuweb/gpuweb/issues/2805), [wgpu `DownlevelFlags`](https://docs.rs/wgpu/latest/wgpu/struct.DownlevelFlags.html) — confirms f64 absent on browser WebGPU and many adapters. MEDIUM-HIGH confidence.
- [maturin Distribution guide](https://www.maturin.rs/distribution.html), [maturin Configuration](https://www.maturin.rs/config) — wheel name derives from Cargo package name; no first-party recipe for per-feature distributions (basis for the distinct-name recommendation). MEDIUM confidence.

---
*Pitfalls research for: CubeCL/Rust GPU + sklearn-port ML library (mlrs)*
*Researched: 2026-06-11*
