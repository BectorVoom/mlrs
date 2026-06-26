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

## Latest Milestone: v3.0 Manifold Algorithms & Rust-Native API — ✅ SHIPPED 2026-06-26

**Status:** Shipped (Phases 12–16, 34 plans, 16/16 requirements complete). Archive: [milestones/v3.0-ROADMAP.md](milestones/v3.0-ROADMAP.md) + [milestones/v3.0-REQUIREMENTS.md](milestones/v3.0-REQUIREMENTS.md).
**Next:** v4.0 to be defined via `/gsd-new-milestone` (candidate scope: Tier-3 backlog — RandomForest→FIL→TreeSHAP, ARIMA, kernel-SVM/SMO — see `notes/v3-hard-algorithm-backlog.md`).

**Goal (delivered):** Added the UMAP + HDBSCAN manifold/clustering pair on a shared multi-metric KNN-graph primitive, and established a Rust-native builder/typestate API additively retrofitted across the whole 32-estimator surface plus a pure-Python sklearn shim.

**Delivered features:**
- **KNN-graph primitive** — the shared, feasibility-critical, **multi-metric** prim (built on the v1 distance + top-k path, plus new direct pairwise distance kernels) returning directed `(indices, distances)`; metrics: euclidean, manhattan (L1), cosine, chebyshev (L∞), minkowski-p; under the cpu-MLIR no-SharedMemory/no-atomics constraint; consumed by both UMAP and HDBSCAN
- **UMAP** — fuzzy simplicial set → SGD-based low-dim layout (oracle: `umap-learn`; stochastic layout → structural/property gate à la RandomProjection D-12, not element-wise 1e-5)
- **HDBSCAN** — mutual-reachability → MST → condensed cluster tree → stability extraction (oracle: `hdbscan` / `sklearn.cluster.HDBSCAN`; exact labels up to permutation as the hard gate)
- **Rust-native builder-pattern API** — establish the idiomatic Rust builder convention (typed builder, fit/unfit typestate, error surface) and retrofit all 30 existing estimators plus the new v3 estimators (today's surface is sklearn-mirror, consumed mainly via PyO3)
- **Pure-Python sklearn shim** — `get_params`/`set_params`/`check_estimator` across the estimator surface; PyO3-wrap UMAP/HDBSCAN

**Key context:** Same gate (cpu f64 + rocm f32) as v1/v2. Oracle broadens beyond scikit-learn for UMAP (`umap-learn`), and the core-value 1e-5 relaxes to a property/structural gate for the stochastic UMAP layout (HDBSCAN keeps exact-label gating). Keeps primitive-first discipline — land the KNN-graph prim standalone before UMAP/HDBSCAN consume it. Deliberately avoids GPU tree-construction atomics risk: RandomForest→FIL→TreeSHAP, ARIMA, and kernel-SVM/SMO stay deferred. Live FFI `estimator_checks` re-triage stays deferred (needs a maturin+pyarrow host this env lacks). Phase numbering continues from v2.0 (next phase = 12). Backlog: `notes/v3-hard-algorithm-backlog.md`.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

**Foundation — v1.0**
- ✓ Modular single-responsibility crate workspace (`mlrs-core`/`-kernels`/`-backend`/`-algos`/`-py`) — v1.0
- ✓ CubeCL compute layer generic over float (`f32`/`f64`) and over runtime — v1.0
- ✓ Backend selection via Cargo features (`cuda` compile-only, `rocm`, `wgpu`, `cpu`) — v1.0
- ✓ Apache Arrow zero-copy data interchange into CubeCL buffers (hard-reject validation) — v1.0
- ✓ Memory-efficient device array / buffer-reuse abstraction (per-phase build-failing memory gates) — v1.0
- ✓ scikit-learn oracle harness, abs/rel error ≤ 1e-5 (sign-flip + label-perm helpers, f32 tolerance policy) — v1.0
- ✓ sklearn-compatible PyO3 binding layer; four per-backend wheels (Python ≥ 3.12, abi3) — v1.0

**v1 Algorithms (12, sklearn-compatible) — v1.0**
- ✓ Linear: LinearRegression (OLS/SVD), Ridge (Cholesky), Lasso, ElasticNet (coordinate descent), LogisticRegression (L-BFGS) — v1.0
- ✓ Clustering: KMeans (k-means++), DBSCAN — v1.0
- ✓ Decomposition: PCA, TruncatedSVD (Jacobi SVD/eig) — v1.0
- ✓ Neighbors: NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor (top-k) — v1.0

**v2 Algorithms (18, sklearn-compatible) — v2.0**
- ✓ New compute primitives: RNG-matrix (PRIM-06), incremental-SVD (PRIM-07), kernel-matrix (PRIM-08), graph-Laplacian (PRIM-09), SGD solver (PRIM-10) — all feature-free, GATHER-idiom, standalone-validated — v2.0
- ✓ Covariance & projection: EmpiricalCovariance, LedoitWolf, IncrementalPCA, GaussianRandomProjection, SparseRandomProjection (PartialFit trait; property-gated projections) — v2.0
- ✓ Kernel family: KernelRidge, KernelDensity (kernel-matrix prim; ScoreSamples trait) — v2.0
- ✓ Spectral family: SpectralEmbedding, SpectralClustering (graph-Laplacian + v1 eig + v1 KMeans) — v2.0
- ✓ SGD / linear-SVM: MBSGDClassifier, MBSGDRegressor, LinearSVC, LinearSVR (PRIM-10 two-pass GATHER solver, pinned deterministic sklearn oracle, exact-label hard gate) — v2.0
- ✓ Naive Bayes: GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB (reductions-only; exact-label hard gate) — v2.0
- ✓ PY-06: all v2 estimators `#[pyclass]`-backed with sklearn-named hyperparameters, f32/f64 dispatch, GIL release, shipped in the four per-backend wheels — v2.0

**v3.0 Manifold Algorithms & Rust-Native API — v3.0**
- ✓ KNN-graph primitive (shared, cpu-MLIR-safe, multi-metric: euclidean/manhattan/cosine/chebyshev/minkowski-p; built on v1 distance + top-k + new direct distance kernels) — Phase 13
- ✓ UMAP (fuzzy simplicial set → SGD layout; `umap-learn` oracle; property gate) — Phase 14; full umap_test oracle 35/35 GREEN confirmed Phase 16 UAT
- ✓ HDBSCAN (mutual-reach → MST → condensed tree → stability; exact-label gate) — Phase 15
- ✓ Rust-native builder-pattern API convention + additive retrofit across all 32 estimators (single `typestate` trait surface; `traits.rs` deleted; predict-before-fit is a compile error) — Phases 12, 16 (BLDR-03)
- ✓ Pure-Python sklearn shim (verbatim `__init__`, get_params/set_params/clone, AST-purity gate, 32-shim matrix) + PyO3-wrapped UMAP/HDBSCAN — Phase 16 (SHIM-01/02/03)

### Active

<!-- Current scope. Building toward these. All are hypotheses until shipped and validated. -->

_v3.0 milestone fully shipped (Phases 12–16). Next scope to be detailed via `/gsd-new-milestone`._

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Multi-GPU / distributed (cuml.dask, NCCL/UCX, `*_mg` paths) — single-device first; distribution is a separate milestone
- cuml.accel transparent acceleration (sklearn import-hook proxying) — magic-proxy layer adds large surface area with no algorithmic value
- Tree / ensemble (RandomForest → FIL → TreeSHAP) — GPU tree construction fights the cpu-MLIR no-SharedMemory/no-atomics constraint; deferred past v3 (the keystone lift; needs a feasibility spike first — see `notes/v3-hard-algorithm-backlog.md`)
- Time-series (ARIMA/AutoARIMA), kernel SVM (SVC/SVR via SMO), genetic/symbolic regression, explainers (SHAP) — deferred to a later milestone (`notes/v3-hard-algorithm-backlog.md`). NOTE: UMAP + HDBSCAN moved into v3.0 Active (manifold/cluster pair on a shared KNN-graph prim, dodging tree-atomics risk); *linear* SVM (LinearSVC/SVR) and SpectralClustering/Embedding shipped in v2.
- Live FFI `estimator_checks` re-triage — needs a maturin+pyarrow host this environment lacks; deferred (the pure-Python sklearn shim IS in v3 scope)
- Bit-exact reproduction of cuML internals — goal is numerical agreement with scikit-learn (≤1e-5), not kernel-for-kernel parity with cuML
- Half-precision (f16/bf16) validated paths — infrastructure may allow it but not a near-term deliverable

## Context

- **Reference source:** `cuml-main/` is RAPIDS cuML v26.08.00 (C++/CUDA + Cython + Python). It is read-only reference material for algorithm behavior and APIs — not code we maintain. Codebase map lives in `.planning/codebase/`.
- **cuML architecture being ported:** C++/CUDA `libcuml++` kernels → thin Cython bindings → sklearn-compatible Python estimators (`Base`, `CumlArray`, `@reflect` output-type mirroring). mlrs collapses this into Rust core + CubeCL kernels + PyO3 bindings.
- **CubeCL guidance:** Manuals at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` (INDEX.md, generics, plane, shared memory, matmul/gemm, reduce, dynamic vectorization, error guideline). Kernels MUST be written per these manuals and MUST use generics-over-float (per project AGENTS.md protocol).
- **Memory-optimisation guidance:** Manuals at `/home/user/Documents/workspace/optimisor/manual/` (zero-copy Arrow↔CubeCL, zero-copy transmutation, half-precision CubeCL, jemalloc/mimalloc, smallvec, compact_str, Arrow dictionary/numeric handling).
- **Oracle:** scikit-learn on CPU produces reference outputs from identical random inputs — runs in CI without a GPU.
- **Build protocol (AGENTS.md):** Source and test code strictly separated (no `mod tests` in source files; use `tests/` or `*_test.rs`). On any CubeCL build error, consult the CubeCL error guideline before attempting fixes.

### Current State (after v3.0)

- **Shipped v3.0** (2026-06-26): UMAP + HDBSCAN added on a shared multi-metric KNN-graph primitive, plus a Rust-native builder/typestate API retrofitted across all 32 estimators and a pure-Python sklearn shim. 34 plans over Phases 12–16, 248 commits, built 2026-06-23 → 2026-06-26 (426 files, +45,450 / −4,160). All 16 v3 requirements complete. Total estimator surface now 32 (+ the shared KNN-graph prim).
- **New in v3.0:** the multi-metric KNN-graph primitive (euclidean/manhattan/cosine/chebyshev/minkowski-p, cpu-MLIR-safe); UMAP (umap-learn property gate for the stochastic layout, ≤1e-5 for deterministic stages, 35/35 oracle); HDBSCAN (exact-label gate + GLOSH `outlier_scores_` + `store_centers`); the builder + compile-time fit/unfit typestate convention (predict-before-fit is a compile error; `traits.rs` deleted); the 32-estimator pure-Python shim with an AST-purity gate; PyO3-wrapped UMAP/HDBSCAN.
- **Resolved at close:** the carried live-Python-boundary gate (live FFI `estimator_checks`/capsule path) — verified end-to-end this session by building the cpu wheel (`maturin develop`) and exercising UMAP/HDBSCAN through a real interpreter + pyarrow capsule (22/22 assertions, f32+f64). The live FFI is runnable here whenever PyPI is reachable; it is no longer an automatic deferral.

### Current State (after v2.0)

- **Shipped v2.0** (2026-06-22): 18 sklearn-compatible estimators added across five families (covariance/projection, kernel, spectral, SGD/linear-SVM, Naive Bayes), 27 plans over Phases 7–11, ~192 commits, built 2026-06-20 → 2026-06-22. All 24 v2 requirements complete. Total estimator surface now 30.
- **Shipped v1.0** (2026-06-14): 12 sklearn-compatible estimators across 5 crates, 38 plans over Phases 1–6, 233 commits. All 27 v1 requirements complete.
- **Validated primitive base** (v1 + v2): GEMM, reductions, pairwise distance, covariance/Gram, Jacobi SVD + symmetric eig, Cholesky/triangular-solve, top-k, L-BFGS, coordinate descent, RNG-matrix, incremental-SVD merge, kernel-matrix, graph-Laplacian, two-pass SGD solver — all feature-free and standalone-validated.
- **Packaging:** four per-backend abi3-py312 wheels (`mlrs_cpu`/`wgpu`/`cuda`/`rocm`), each `import mlrs`; cpu imports without LD_PRELOAD (mimalloc local_dynamic_tls). pyo3 pinned 0.28; v2 added zero compute dependencies.
- **Known tech debt / deferred:** (v2) live Python FFI smoke + sklearn `estimator_checks` re-triage (need maturin+pyarrow host), pure-Python sklearn shim for get_params/set_params/check_estimator not built (consistent across Phases 8–11); (v1) CUDA-host-only checks, Phase-5 follow-ups. See STATE.md Deferred Items.
- **Next:** start the next milestone via `/gsd-new-milestone` — candidate scope is the Tier-3 backlog (`notes/v3-hard-algorithm-backlog.md`) and the carried-forward Python-surface work.

## Constraints

- **Language**: Rust (core, kernels, bindings) — full rewrite, no C++/CUDA from cuML retained
- **Compute**: CubeCL only for device kernels; generic over float type and runtime
- **Backends**: Cargo features `cuda` (compile-only / untestable in this environment), `rocm`, `wgpu`, `cpu`
- **Test/CI target**: through Phase 2 the primary gate was wgpu + cpu; **from Phase 3 the correctness gate is cpu + rocm** (D-07) — f64 validates on cpu, f32 validates on rocm, and f64-on-rocm SKIPS-with-log (cubecl-cpp 0.10 does not register F64 for the HIP backend; empirical, not a defect). cuda compiles only (untestable here); wgpu is opportunistic from Phase 3.
- **Precision**: kernels generic over `f32` and `f64`; both validated in v1
- **Correctness**: abs/rel error ≤ 1e-5 vs scikit-learn on random-data oracle tests
- **Python**: ≥ 3.12; sklearn-compatible API surface; users install the package matching their backend
- **Data interchange**: Apache Arrow, zero-copy into CubeCL buffers
- **Memory**: efficiency is first-class — zero-copy, buffer reuse, minimal copies, custom allocator — verified per phase, not deferred
- **Code structure**: tests separated from source files (project AGENTS.md rule)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Broad multi-family v1 (linear + clustering + decomposition + KNN) | Validates the backend/Arrow/test stack across several kernel patterns, not just one | ✓ Good — 12 estimators across 4 families shipped, stack proven |
| sklearn-compatible API via PyO3 | Lets existing scikit-learn/cuML users adopt with minimal code change | ✓ Good — estimator_checks 475 passed / 0 unexpected |
| scikit-learn as oracle (not cuML) | CPU reference runs in CI without a GPU; cuML reference would require CUDA hardware | ✓ Good — every estimator gated at 1e-5 vs sklearn, no GPU needed for the oracle |
| Apache Arrow zero-copy interchange | Memory-efficiency priority; avoids host-side copies crossing the Python↔Rust↔device boundary | ✓ Good — PyCapsule ingest, validated single-upload, per-phase memory gates |
| Gate = cpu + rocm from Phase 3 (was wgpu + cpu through Phase 2) | gfx1100/ROCm 7.1.1 runs real GPU kernels here; cpu runs f64. f32 validates on rocm, f64 on cpu; f64-on-rocm skips-with-log (cubecl-cpp 0.10 F64 unregistered). CUDA untestable; wgpu opportunistic. (D-07) | ✓ Good — held through Phases 3–6 |
| Generic over float (`f32`/`f64`) | f64 makes 1e-5 tolerance comfortable; mirrors cuML's float/double symmetry | ✓ Good — f64 strict 1e-5, f32 documented epsilon bands per family |
| Memory efficiency as per-phase requirement | Stated high priority; retrofitting zero-copy/allocators later is costly | ✓ Good — build-failing PoolStats gates every phase, never deferred |
| Primitive-first horizontal build order | Validate GEMM/reduce/distance/SVD/eig/Cholesky standalone before estimators consume them | ✓ Good — estimators were "mostly assembly"; SVD/eig gated 4, distance gated 3 |
| LogReg keeps symmetric over-parameterized multinomial for all K (D-12) | Single objective for binary+multiclass; gauge-fixed predict_proba is the correctness witness | ⚠️ Revisit — binary differs ~3.6e-3 from sklearn's binomial under L2 (user-approved); a binomial path may be wanted later |
| v2 reuses the shipped PyO3 binding layer (no dedicated Python phase) | v1's `any_estimator!` machinery + ingress/egress/capability/errors are general; each v2 phase wraps its own estimators incrementally | ✓ Good — v2 added zero new binding infrastructure across 18 estimators; pyo3 stayed 0.28 |
| Property-gate (not 1e-5 value match) for RandomProjection (D-12) | mlrs SplitMix64 ≠ NumPy MT19937, so the projection matrix can't match element-wise; JL distortion + distribution stats + seed-reproducibility are the meaningful contract | ✓ Good — `johnson_lindenstrauss_min_dim` value-matched exactly; JL ratio concentration gated via 50-trial averaging |
| Exact predicted labels as the hard correctness gate for classifiers (SGD/SVM/NB) | Iterative/host-order solvers agree with sklearn only to a band on coefficients, but argmax/label decisions are integer-exact | ✓ Good — every v2 classifier passes exact-label gate on cpu f32+f64; coef bands documented |
| cpu-MLIR-safe GATHER idiom for all new kernels (no SharedMemory, no cross-unit atomics) | cubecl-cpu MLIR lowering panics on SharedMemory + mutable-bool/INFINITY/shift-loops; single-owner GATHER launches first try | ✓ Good — all five v2 prims (incl. the highest-risk SGD solver) launched on cpu-MLIR without rework |
| Additive builder/typestate retrofit (builder constructs existing config; fit path untouched) with single `typestate` trait surface | Lets all 32 estimators adopt one idiomatic convention + a compile-time predict-before-fit guard without perturbing any shipped 1e-5/exact-label numeric path | ✓ Good — Phase 16: `traits.rs` deleted, 13/13 plans, 12 oracle suites + UMAP 35/35 all GREEN; byte-identical fit bodies confirmed |
| `lock_pool()` (poison-recovering) as the single sanctioned pool-lock path for the PyO3 layer (WR-02/WR-04) | One surviving `global_pool().lock().expect()` re-panics on a poisoned mutex, turning a recoverable device fault into a process-wide brick | ✓ Good — Phase 16 secure-phase audit caught covariance.rs as the last legacy holder; remediated (quick 260626-ktm); zero estimator modules now use the panicking lock |

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
*Last updated: 2026-06-26 after v3.0 milestone close — UMAP + HDBSCAN on the shared KNN-graph prim, builder/typestate retrofit across all 32 estimators, pure-Python shim, live PyO3 FFI sign-off (22/22). v3.0 archived + tagged; next milestone via `/gsd-new-milestone`.*
