# Stack Research

**Domain:** ML estimator breadth sweep (v2.0) on the existing mlrs Rust/CubeCL stack
**Researched:** 2026-06-14
**Confidence:** HIGH

## Bottom Line

**No new runtime crate dependency is required for any of the ~16 v2 estimators.** Every one
composes from v1's validated primitives plus new *feature-free* CubeCL kernels. The three
decisions the downstream roadmapper needs:

1. **Dependencies:** Add **nothing** new to `[workspace.dependencies]` for compute. All v2
   estimators are assembly over existing prims (covariance, SVD, eig, Cholesky, distance,
   GEMM, reductions, L-BFGS, coordinate-descent) + new kernels. The one *new primitive* per
   phase (RNG-matrix, kernel-matrix, graph-Laplacian, SGD) is mlrs-authored CubeCL, not a crate.
2. **RNG:** **Keep the host-side SplitMix64** (already in `prims/kmeans.rs`); promote it to a
   shared `prims/rng.rs` module. **Do NOT add `cubek-random`.** It violates two hard
   constraints (no caller seed → fails ASVS-V6 reproducibility; shared-memory/atomic Tausworthe
   → fails cpu-MLIR). Host-generate-then-upload the projection / shuffle indices.
3. **Sparse input:** **Do NOT add a sparse path to the Arrow bridge in v2.** Accept sparse
   input by **densifying at the Python ingress layer** (sklearn-compatible: `MultinomialNB`,
   `SparseRandomProjection.transform` accept sparse but mlrs materializes dense). A real
   CSR device path is a v3 line item, explicitly out of scope here.

## Recommended Stack

### Core Technologies (UNCHANGED — all already pinned)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| cubecl | 0.10.0 (`default-features=false`) | device-kernel layer, generic over float + runtime | Latest stable (crates.io updated 2026-05-07); the entire v2 surface is feature-free CubeCL kernels over the existing pattern. No change. |
| cubek-matmul | 0.2.0 | GEMM backing the GEMM prim | Used by KernelRidge (K = XXᵀ via GEMM), RandomProjection (X·R), SGD (Xβ). Already wired. No change. |
| cubek-reduce | 0.2.0 | reductions backing sum/mean/min/max/L2/SumSq | The entire Naive-Bayes family + LedoitWolf shrinkage + SGD gradients are reductions. Already wired. No change. |
| arrow | 59 (`pyarrow`) | zero-copy host↔device interchange | Dense Float32/Float64 ingress is sufficient for v2 (sparse densified at ingress, see below). Latest stable (2026-06-09). No change. |
| pyo3 | 0.28 (`abi3-py312`, `extension-module`) | Python estimator bindings | **Stays PINNED at 0.28 even though 0.29 is now latest** (crates.io 2026-06-11). arrow-59's `pyarrow` feature transitively pins pyo3 0.28.x; mixing ABIs links two PyInit symbols and crashes the wheel at import (D-09/PY-05). The v2 estimators add `#[pyclass]` wrappers only — no pyo3 version pressure. |
| mimalloc | 0.1 (`local_dynamic_tls`) | global allocator in mlrs-py | dlopen-safe wheels. No change. |
| bytemuck | 1 | zero-copy `&[T]`↔`&[u8]` for Arrow→CubeCL | No change. |
| thiserror / anyhow | 2 / 1 | typed errors in libs / boundary errors | No change. |
| npyz | 0.9 (`npz`) | `.npz` oracle-fixture reader (dev/test) | New v2 fixtures are committed `.npz` blobs read by the same harness. No change. |

### New Primitives (mlrs-authored CubeCL — NOT crates)

These are the per-phase "one new shared primitive" from the seed roadmap. Each is a new module
under `crates/mlrs-backend/src/prims/` (+ kernel in `mlrs-kernels`), written feature-free and
generic over `F: Float`, obeying the GATHER idiom (single-owner outputs, u32/F accumulators,
if-guards, no `bool`/no `F::INFINITY`/no descending-shift loops, no SharedMemory, no cross-unit
atomics).

| Primitive | Phase | Composes from | New kernel work | cpu-MLIR / f64-rocm OK? |
|-----------|-------|---------------|-----------------|--------------------------|
| `prims/rng.rs` (RNG-matrix generator) | 1 | host SplitMix64 (promote from kmeans.rs) + `DeviceArray::from_host` | None on device — host generates the (n_features × n_components) Gaussian / sparse-±1 matrix and uploads once. | Yes — generation is host-side; only GEMM touches the device. f64-on-rocm follows existing skip-with-log. |
| incremental-SVD update | 1 | existing Jacobi `svd` prim | Small: concatenate [Σ·Vᵀ ; new batch] and re-run existing `svd` on the (n_components+batch)×n_features panel; mean-correction reduction. **No new device kernel if panel re-SVD is acceptable** (it is at v2 sizes). | Yes — reuses the gated Jacobi SVD; no new atomics. f32-on-rocm stability: the merge re-SVD inherits Jacobi's f32 band. |
| `prims/kernel_matrix.rs` | 2 | existing `distance` (sq-Euclidean) + `gemm` + reductions | Elementwise map over pairwise distance / Gram: RBF `exp(-γd²)`, poly `(γ·G+c0)^d`, sigmoid `tanh(γ·G+c0)`, linear `G`. Pure elementwise GATHER over an already-computed N×N matrix. | Yes — elementwise, single-owner per cell, no SharedMemory/atomics. f64-on-rocm skip-with-log. |
| `prims/laplacian.rs` (graph Laplacian) | 3 | `distance` + reductions (degree = row-sum) + existing `eig` | Build affinity (RBF over distance), degree reduction, normalized `L = I − D^{-1/2} W D^{-1/2}` elementwise. Then existing Jacobi `eig` for the full spectrum; **take smallest nontrivial eigenvectors host-side** (full-spectrum-then-slice is fine at v2 sizes — no Lanczos/shift-invert needed). | Yes — elementwise + reductions + gated eig. No new atomics. |
| `prims/sgd.rs` (minibatch SGD solver) | 4 | `gemm` (Xβ), reductions (gradient accumulation), host SplitMix64 (epoch shuffle) | Per-minibatch: forward GEMM, loss-gradient elementwise (hinge / log / squared / ε-insensitive), parameter-update reduction. Each output weight is single-owner → fits GATHER; minibatch sum is a reduction, **not** a cross-unit atomic. Epoch shuffle = host SplitMix64 permutation of row indices. | Yes — the one genuinely new solver, but it is reductions+GEMM, the exact pattern v1 already runs on cpu-MLIR. No SharedMemory, no atomics. f64-on-rocm skip-with-log. |

### Development Tools (UNCHANGED)

| Tool | Purpose | Notes |
|------|---------|-------|
| scikit-learn (oracle) | reference outputs for ≤1e-5 gate | Every v2 estimator has a sklearn reference; `gen_oracle.py` regen needs a `/tmp` venv with numpy (PEP 668) — fixtures are committed blobs. |
| maturin | per-backend abi3-py312 wheels | Four wheels (`mlrs_cpu`/`wgpu`/`cuda`/`rocm`). No change. |

## Installation

```bash
# No new crate installs. The v2 work is additive modules in existing crates:
#   crates/mlrs-kernels/src/{rng,kernel_matrix,laplacian,sgd}.rs   (feature-free kernels)
#   crates/mlrs-backend/src/prims/{rng,kernel_matrix,laplacian,sgd}.rs
#   crates/mlrs-algos/src/{covariance,decomposition,random_projection,kernel,
#                          cluster,svm,linear_model,naive_bayes}/...
#   crates/mlrs-py/src/estimators/...   (#[pyclass] wrappers only)
#
# Workspace Cargo.toml [workspace.dependencies] is UNCHANGED.
```

## RNG Decision (explicit — ASVS-V6 + cpu-MLIR)

**Decision: host-side SplitMix64, generate-then-upload. Reject `cubek-random`.**

| Option | Verdict | Reason |
|--------|---------|--------|
| **Host SplitMix64 (chosen)** | ✓ ADOPT | Already shipped and gated in `prims/kmeans.rs:664` with unbiased rejection sampling (`next_below`). Caller-controlled `u64` seed → ASVS-V6 reproducible, identical indices/matrix across runs **and across backends** (host-side, backend-independent). For RandomProjection: generate the full (n_features × n_components) Gaussian / sparse-±1 matrix on host, upload once, then device GEMM. For SGD: host permutation of row indices per epoch. Promote `SplitMix64` to a shared `prims/rng.rs`. |
| `cubek-random` 0.2.0 (device RNG) | ✗ REJECT | Verified via Context7 (`/tracel-ai/cubek`): public API (`random_uniform/normal/bernoulli`) takes **no caller seed argument** → cannot guarantee ASVS-V6 reproducibility from a documented `u64`. Uses a **hybrid Tausworthe/LCG with shared-memory optimizations** → fails the cpu-MLIR no-SharedMemory constraint. Adds a new crate for negative value. |
| `rand`/`rand_chacha` crate | ✗ REJECT | New dependency for what ~30 lines of SplitMix64 already do, reproducibly. `getrandom`/OsRng is explicitly forbidden by ASVS-V6. No justification under the "no new heavy deps" rule. |

**Why host-generate is acceptable for projection matrices:** the matrix is O(n_features × n_components),
generated once at `fit`, uploaded once, then consumed by device GEMM for every transform. The
host-generation cost is one-time and dwarfed by the GEMM; a device RNG kernel would buy nothing
while breaking reproducibility and cpu-MLIR. (Mirrors the v1 k-means++ host-PRNG decision.)

**Sparse-projection density:** sklearn `SparseRandomProjection` draws each entry from
{−√(1/(s·k)), 0, +√(1/(s·k))} with density `1/s` (default `s=√n_features`). SplitMix64 `next_below(s)`
per entry reproduces the Achlioptas/Li scheme exactly. Still materialized **dense** on device
(see sparse decision) — the *matrix* is sparse-valued but stored dense; that is fine at v2 sizes
and keeps the GEMM on the existing dense path.

## Sparse-Input Decision (explicit)

**Decision: densify at Python ingress; NO sparse device path in v2.**

The current Arrow bridge accepts only dense `Float32Array`/`Float64Array` (verified in
`mlrs-py/src/ingress.rs` + `errors.rs`). The estimators where sklearn commonly takes sparse
input are `MultinomialNB`/`ComplementNB`/`BernoulliNB` (term-count matrices) and
`SparseRandomProjection.transform`.

| Option | Verdict | Reason |
|--------|---------|--------|
| **Densify at ingress (chosen)** | ✓ ADOPT | sklearn API compatibility = *accept* sparse, not *stay* sparse internally. At the Python wrapper, call `.toarray()`/densify on incoming scipy-sparse before the existing dense Arrow→device path. Naive-Bayes math (per-class feature sums) and projection GEMM are identical on a densified matrix; the ≤1e-5 gate is unaffected. Zero stack change. |
| Add Arrow CSR + device CSR-GEMM/segmented-reduce | ✗ REJECT (v3) | Real value only at large/very-sparse text scale, which is out of v2's "low-risk breadth" scope. CSR segmented reductions on cpu-MLIR (no cross-unit atomics) is genuine new infrastructure — exactly the kind of Tier-3 lift v2 defers. Flag as a v3 backlog item ("native sparse interchange"). |

**Memory caveat for the roadmapper:** densifying a large sparse term-count matrix can blow the
per-phase memory gate. Mitigation: document the densify cost in the estimator docstring and let
the existing `BufferPool` memory-gate catch regressions; do **not** silently chunk in v2.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Host SplitMix64 RNG-matrix | `cubek-random` device RNG | Only if a future milestone needs RNG inside a device loop where host round-trips dominate AND reproducibility is dropped AND cpu is no longer a gate — none true in v2. |
| Full-spectrum Jacobi eig then slice smallest (Spectral) | Lanczos / shift-invert smallest-eigenpair solver | Only at large n_samples where computing the full N×N spectrum is wasteful. v2 problem sizes don't justify the new solver; revisit in v3 if SpectralClustering is pushed to large graphs. |
| Panel re-SVD for IncrementalPCA merge | Dedicated rank-1/QR SVD-update kernel | Only if streaming batches are huge and re-SVD of the (k+batch)×features panel becomes the bottleneck. At v2 sizes the existing Jacobi SVD on the merged panel is simplest and reuses the gated prim. |
| Densify sparse at ingress | Native CSR device path | Large-scale text/NLP workloads with extreme sparsity — defer to v3. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `cubek-random` 0.2.0 | No caller seed (breaks ASVS-V6 reproducibility); shared-memory Tausworthe (breaks cpu-MLIR no-SharedMemory) | Host SplitMix64 in `prims/rng.rs` |
| `rand` / `rand_chacha` / `getrandom` | New dependency; `getrandom` pulls OsRng (explicitly forbidden by ASVS-V6) | Host SplitMix64 (already present) |
| pyo3 0.29 | Links a second PyInit ABI alongside arrow-59's transitive pyo3 0.28 → wheel crashes at import (D-09/PY-05) | Stay on pyo3 0.28 |
| `ndarray` / `nalgebra` (host linear algebra) | The math runs on device via existing prims; a host linalg crate would duplicate and split the codepath | Existing GEMM/SVD/eig/Cholesky prims |
| SharedMemory / `Atomic*` / `bool` outputs / `F::INFINITY` / descending-shift loops in new kernels | cpu MLIR backend panics at launch (no SharedMemory, no cross-unit atomics) | GATHER idiom: single-owner outputs, u32/F accumulators, if-guards |
| Native sparse Arrow interchange | Out of v2 scope; CSR segmented-reduce on cpu-MLIR is Tier-3 infrastructure | Densify sparse at the Python ingress wrapper |

## Stack Patterns by Variant

**If the estimator needs random draws (RandomProjection, MBSGD shuffle):**
- Use host SplitMix64 → generate full matrix / permutation on host → single metered upload via `DeviceArray::from_host`
- Because it is reproducible (ASVS-V6), backend-independent, and keeps the device path GEMM-only (cpu-MLIR safe)

**If the estimator needs a kernel matrix (KernelRidge, KernelDensity, future kernel-SVM):**
- Use the new `prims/kernel_matrix.rs` = existing `distance`/`gemm` + elementwise map
- Because one prim covers linear/RBF/poly/sigmoid and is pure elementwise GATHER (no new atomics)

**If the estimator needs smallest eigenpairs (SpectralEmbedding/Clustering):**
- Use existing `eig` for the full spectrum, slice the smallest nontrivial vectors host-side
- Because v2 sizes don't justify a Lanczos/shift-invert solver and `eig` is already gated

**If the estimator accepts sparse input (MultinomialNB family, SparseRandomProjection):**
- Densify at the Python wrapper before the dense Arrow ingress
- Because sklearn compatibility means *accept* sparse, and the device math is dense-identical

## Integration Points (for the roadmapper)

| New code | Existing seam it plugs into |
|----------|-----------------------------|
| `prims/rng.rs` matrix output | `DeviceArray::from_host` (single metered upload via `BufferPool`) → `gemm` prim |
| `prims/kernel_matrix.rs` | consumes `distance`/`gemm` outputs; feeds `cholesky_solve` (KernelRidge) and reductions (KernelDensity) |
| `prims/laplacian.rs` | consumes `distance` + reductions; feeds existing `eig`; eigenvectors → `kmeans` (SpectralClustering) |
| `prims/sgd.rs` | consumes `gemm` + reductions; host SplitMix64 for shuffle |
| Naive-Bayes / LedoitWolf | pure `covariance` + reductions, no new prim |
| IncrementalPCA | existing `svd` on merged panel + mean reductions |
| All estimators | `mlrs-py` dtype-dispatch (`FloatDtype::{F32,F64}`) + dense Arrow ingress + `egress` 2-D/1-D shape path — unchanged; sparse densified before this point |

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| arrow 59 (`pyarrow`) | pyo3 0.28.x | HARD: arrow-59 transitively pins pyo3 0.28; only one pyo3 ABI may link the cdylib. Do not bump pyo3. |
| cubecl 0.10 | cubek-matmul/reduce 0.2.0 | Existing working pin; cubek 0.2.0 built against cubecl 0.10. Verified latest 2026-05-07. |
| cubek-random 0.2.0 | cubecl 0.10 | Compatible, but **not adopted** (see RNG decision). |
| cubecl 0.10 cpu (MLIR) | new v2 kernels | Must be SharedMemory-free / atomic-free (GATHER idiom) or the cpu backend panics at launch. |
| cubecl-cpp 0.10 (rocm/HIP) | f64 | F64 NOT registered for HIP → f64-on-rocm oracle cases skip-with-log; v2 kernels inherit this gate. |

## Sources

- Context7 `/tracel-ai/cubek` — cubek-random API (`random_uniform/normal/bernoulli`, no seed arg; hybrid Tausworthe/LCG with shared-memory optimization). HIGH (curated library docs).
- Context7 `/tracel-ai/cubecl` — confirmed CubeCL 0.10 multi-runtime (CUDA/ROCm/Metal/Vulkan/WebGPU/CPU). HIGH.
- crates.io API (fetched 2026-06-14) — cubecl 0.10.0 (2026-05-07), cubek-matmul/reduce/random 0.2.0 (2026-05-07), arrow 59.0.0 (2026-06-09), pyo3 newest 0.29.0 (2026-06-11, NOT adopted). HIGH (registry of record).
- mlrs source `crates/mlrs-backend/src/prims/kmeans.rs:656-710` — existing SplitMix64 with unbiased `next_below`. HIGH (direct read).
- mlrs source `crates/mlrs-py/src/ingress.rs`, `errors.rs` — dense Float32/Float64-only Arrow bridge. HIGH (direct read).
- mlrs `Cargo.toml [workspace.dependencies]` — current pins + rationale comments (D-09/D-10). HIGH (direct read).
- Project memory `cubecl-cpu-no-shared-memory.md`, `rocm-is-runnable-gpu-gate.md`, `cubecl algo crates moved to cubek` — constraint corroboration. HIGH.

---
*Stack research for: mlrs v2.0 estimator breadth sweep*
*Researched: 2026-06-14*
