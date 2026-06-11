# mlrs — cuML in Rust

## What This Is

mlrs is a ground-up rewrite of RAPIDS cuML's machine-learning algorithms in Rust.
Compute kernels are written once in [CubeCL](https://github.com/tracel-ai/cubecl) and made
generic over both the floating-point type (`f32`/`f64`) and the CubeCL runtime, so the same
algorithm runs on CUDA, ROCm, wgpu, or CPU selected at build time via Cargo features. It ships
sklearn-compatible Python estimators (via PyO3) so data scientists on Python ≥ 3.12 can `pip
install` the package for their backend and use familiar `fit`/`predict`/`transform` APIs.

## Core Value

**Correct, memory-efficient ML algorithms that match scikit-learn within 1e-5, running on any
CubeCL backend from a single generic codebase.** If everything else fails, the numerical results
must be right and the backend abstraction must hold.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — greenfield rewrite; ship to validate)

### Active

<!-- Current scope. Building toward these. All are hypotheses until shipped and validated. -->

**Foundation**
- [ ] Cargo workspace with modular, single-responsibility crates (clear core / kernels / backend / bindings separation)
- [ ] CubeCL compute layer generic over float type (`f32`/`f64`) and over runtime
- [ ] Backend selection via Cargo features: `cuda` (compile-only, untestable here), `rocm`, `wgpu`, `cpu`
- [ ] Apache Arrow zero-copy data interchange into CubeCL device buffers
- [ ] Memory-efficient device array / buffer abstraction (buffer reuse, minimal host↔device copies, custom allocator)
- [ ] Oracle test harness: randomly generated inputs, compared to scikit-learn, abs/rel error ≤ 1e-5
- [ ] sklearn-compatible PyO3 binding layer; per-backend Python package install (Python ≥ 3.12)

**v1 Algorithms (sklearn-compatible)**
- [ ] Linear models: LinearRegression (OLS), Ridge, Lasso, ElasticNet, LogisticRegression
- [ ] Clustering: KMeans, DBSCAN
- [ ] Decomposition: PCA, TruncatedSVD
- [ ] Neighbors: NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Multi-GPU / distributed (cuml.dask, NCCL/UCX, `*_mg` paths) — single-device first; distribution is a separate milestone
- cuml.accel transparent acceleration (sklearn import-hook proxying) — magic-proxy layer adds large surface area with no algorithmic value for v1
- Tree / ensemble algorithms (RandomForest, decision trees, FIL, gradient boosting) — high complexity; later milestone
- SVM, manifold (UMAP/t-SNE), time-series (ARIMA/Holt-Winters), genetic/symbolic regression, explainers (SHAP) — defer to later milestones
- Bit-exact reproduction of cuML internals — goal is numerical agreement with scikit-learn (≤1e-5), not kernel-for-kernel parity with cuML
- Half-precision (f16/bf16) validated paths — infrastructure may allow it but not a v1 deliverable

## Context

- **Reference source:** `cuml-main/` is RAPIDS cuML v26.08.00 (C++/CUDA + Cython + Python). It is read-only reference material for algorithm behavior and APIs — not code we maintain. Codebase map lives in `.planning/codebase/`.
- **cuML architecture being ported:** C++/CUDA `libcuml++` kernels → thin Cython bindings → sklearn-compatible Python estimators (`Base`, `CumlArray`, `@reflect` output-type mirroring). mlrs collapses this into Rust core + CubeCL kernels + PyO3 bindings.
- **CubeCL guidance:** Manuals at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` (INDEX.md, generics, plane, shared memory, matmul/gemm, reduce, dynamic vectorization, error guideline). Kernels MUST be written per these manuals and MUST use generics-over-float (per project AGENTS.md protocol).
- **Memory-optimisation guidance:** Manuals at `/home/user/Documents/workspace/optimisor/manual/` (zero-copy Arrow↔CubeCL, zero-copy transmutation, half-precision CubeCL, jemalloc/mimalloc, smallvec, compact_str, Arrow dictionary/numeric handling).
- **Oracle:** scikit-learn on CPU produces reference outputs from identical random inputs — runs in CI without a GPU.
- **Build protocol (AGENTS.md):** Source and test code strictly separated (no `mod tests` in source files; use `tests/` or `*_test.rs`). On any CubeCL build error, consult the CubeCL error guideline before attempting fixes.

## Constraints

- **Language**: Rust (core, kernels, bindings) — full rewrite, no C++/CUDA from cuML retained
- **Compute**: CubeCL only for device kernels; generic over float type and runtime
- **Backends**: Cargo features `cuda` (compile-only / untestable in this environment), `rocm`, `wgpu`, `cpu`
- **Test/CI target**: wgpu + cpu are the primary correctness gates; cuda/rocm compile but are verified opportunistically
- **Precision**: kernels generic over `f32` and `f64`; both validated in v1
- **Correctness**: abs/rel error ≤ 1e-5 vs scikit-learn on random-data oracle tests
- **Python**: ≥ 3.12; sklearn-compatible API surface; users install the package matching their backend
- **Data interchange**: Apache Arrow, zero-copy into CubeCL buffers
- **Memory**: efficiency is first-class — zero-copy, buffer reuse, minimal copies, custom allocator — verified per phase, not deferred
- **Code structure**: tests separated from source files (project AGENTS.md rule)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Broad multi-family v1 (linear + clustering + decomposition + KNN) | Validates the backend/Arrow/test stack across several kernel patterns, not just one | — Pending |
| sklearn-compatible API via PyO3 | Lets existing scikit-learn/cuML users adopt with minimal code change | — Pending |
| scikit-learn as oracle (not cuML) | CPU reference runs in CI without a GPU; cuML reference would require CUDA hardware | — Pending |
| Apache Arrow zero-copy interchange | Memory-efficiency priority; avoids host-side copies crossing the Python↔Rust↔device boundary | — Pending |
| wgpu + cpu as primary CI gate | CUDA untestable in this environment; wgpu runs on commodity hardware/CI | — Pending |
| Generic over float (`f32`/`f64`) | f64 makes 1e-5 tolerance comfortable; mirrors cuML's float/double symmetry | — Pending |
| Memory efficiency as per-phase requirement | Stated high priority; retrofitting zero-copy/allocators later is costly | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-11 after initialization*
