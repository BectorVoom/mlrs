# Project Research Summary

**Project:** mlrs — Rust rewrite of RAPIDS cuML (CubeCL + Arrow + PyO3)
**Domain:** GPU-accelerated, sklearn-compatible ML library; Rust CubeCL kernels generic over float type and runtime backend; Apache Arrow zero-copy data interchange; PyO3 per-backend Python wheels
**Researched:** 2026-06-11
**Confidence:** HIGH

## Executive Summary

mlrs is a greenfield, ground-up Rust rewrite of RAPIDS cuML's core ML algorithms. Compute kernels are written once using CubeCL 0.10.0 and are generic over two independent dimensions: the floating-point type (`F: f32|f64`, resolved per-call at runtime by input dtype) and the compute runtime (`R: CpuRuntime|WgpuRuntime|CudaRuntime|...`, fixed at compile time per backend wheel). The codebase is structured as a five-crate Cargo workspace — `mlrs-core` (traits/types), `mlrs-kernels` (all `#[cube]` kernels, runtime-agnostic), `mlrs-backend` (the sole owner of Cargo backend features and the Arrow zero-copy bridge), `mlrs-algos` (estimator orchestration, generic `<F, R>`), and `mlrs-py` (cdylib, PyO3 bindings, monomorphizes `R` at build time). Python users `pip install` the wheel matching their hardware backend and use a fully sklearn-compatible `fit`/`predict`/`transform` API.

The recommended build strategy is strictly primitive-first: oracle harness and Arrow bridge in Phase 0, compute primitives (GEMM, reduce, pairwise distance, SVD/eig) validated standalone in Phase 1-2, then closed-form estimators (OLS, Ridge, PCA, TruncatedSVD) in Phase 3, distance-based and iterative-solver estimators in Phase 4, and the full PyO3 surface plus per-backend wheels in Phase 5. The SVD/eig primitive is the single highest-leverage gate: it blocks PCA, TruncatedSVD, OLS-svd, and Ridge-svd — build and validate it standalone before touching those estimators. The pairwise-distance primitive blocks three families (KMeans, DBSCAN, all KNN) and must be validated standalone with the `max(d2, 0)` clamp for f32 safety.

The dominant correctness risk is solver/defaults mismatch: the oracle is scikit-learn on CPU, not cuML, yet the reference code is cuML — and their defaults differ on OLS (sklearn=SVD-based lstsq, cuML=eig), KMeans init (sklearn=k-means++, cuML=k-means||), TruncatedSVD algorithm (sklearn=randomized/stochastic, cuML=full), and PCA sign convention (sklearn applies `svd_flip`). Matching cuML defaults will fail the 1e-5 oracle gate. The mitigation is to encode the correct sklearn-matching solver for each estimator in the oracle harness at P0, and to build sign-flip and label-permutation comparison helpers before any estimator exists. A secondary risk is f64 absence on wgpu (the primary CI gate): capability-gate f64 paths at runtime using `client.properties().feature_enabled(...)` from Phase 0 so the wgpu CI job never hard-fails on adapter limitations.

## Key Findings

### Recommended Stack

The core stack is: `cubecl 0.10.0` (umbrella crate with `cpu`/`wgpu`/`cuda`/`rocm` features), `cubecl-matmul`/`cubecl-reduce`/`cubecl-std` all pinned to `0.10.0` (must match exactly to avoid macro/ABI errors), `pyo3 0.28` with `abi3-py312` (Python 3.12+ from one wheel per platform), `maturin 1.13` as the build backend, `arrow 59` (arrow-rs, not arrow2 which is archived), and `bytemuck 1` as the zero-copy glue between Arrow buffers and CubeCL `Bytes`. Supporting libraries: `mimalloc 0.1.52` as the global allocator (declared only in `mlrs-py` cdylib), `smallvec 1.15` for shapes/strides, `compact_str 0.8` for class labels and param keys, `thiserror 2` for library error types, and `rand 0.9` + `rand_distr 0.5` + `approx 0.5` as dev-only oracle-harness dependencies.

**Core technologies:**
- `cubecl 0.10.0`: single-source GPU/CPU compute kernels via `#[cube]`/`#[cube(launch)]`; `Numeric`/`Float`/`CubeElement` trait bounds; backend selection via Cargo features — mandated by project, confirmed current
- `pyo3 0.28` + `maturin 1.13`: Rust-Python FFI and wheel build; `abi3-py312` produces one wheel per (backend x platform) covering Python 3.12/3.13/3.14; Rust >=1.83 required
- `arrow 59` (arrow-rs) + `bytemuck 1`: zero-copy Arrow->CubeCL bridge; `Float32/64Array::values()` -> `cast_slice::<T,u8>` -> `Bytes` is the load-bearing path from the zero-copy manuals
- `mimalloc 0.1.52`: global allocator for low-fragmentation per-fit allocation churn; declared once in cdylib; switch to `tikv-jemallocator 0.7.0` when heap profiling is needed
- `rand 0.9` + `approx 0.5`: seeded deterministic oracle inputs and `assert_abs_diff_eq!`/`assert_relative_eq!` with explicit epsilon; test-only dev dependencies

### Expected Features

The v1 feature surface spans four algorithm families: linear models (LinearRegression/OLS, Ridge, Lasso, ElasticNet, LogisticRegression), clustering (KMeans, DBSCAN), decomposition (PCA, TruncatedSVD), and neighbors (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor). Every estimator implements the full sklearn API contract: `fit`/`predict`/`transform` with `fit` returning `self`, `get_params`/`set_params`, trailing-underscore fitted attributes lazily materialized on access, `n_features_in_`, and mixin `score` semantics.

**Must have (table stakes):**
- Oracle harness with sign-flip helpers (PCA/TSVD components) and label-permutation helpers (KMeans/DBSCAN labels) — gates all correctness tests; P0 prerequisite before any estimator
- Arrow zero-copy ingest with offset/nullability/alignment validation — memory-efficiency spine; P0
- f64 capability gating per backend (`feature_enabled(FloatKind::F64)`) with xfail/skip on unsupported adapters — P0; without this wgpu CI silently hides f64 failures
- Per-backend wheel distribution naming (distinct names: `mlrs-cpu`, `mlrs-wgpu`, etc.) — decided P0 even though wheels ship in Phase 5
- GEMM (via `cubecl-matmul`), reductions (sum/mean/argmin/L2-norm), pairwise Euclidean distance — underpin all four estimator families
- SVD (full, Jacobi) and symmetric eigendecomposition — gates PCA, TruncatedSVD, OLS-svd, Ridge-svd; highest-priority single primitive
- Coordinate-descent solver (Lasso and ElasticNet share it; Lasso = `l1_ratio==1` special case)
- Quasi-Newton L-BFGS/OWL-QN solver (LogisticRegression; highest correctness risk in the project)
- All 11 v1 estimators with sklearn-matching default solvers/init — not cuML defaults

**Should have (v1.x differentiators after oracle passes):**
- Additional solver variants: `jacobi` SVD, `lsmr`, `qr` for OLS; L1/elasticnet penalties for LogisticRegression
- Distance-weighted KNN (`weights='distance'`)
- Additional distance metrics: cosine, Minkowski-p, Manhattan across KMeans/DBSCAN/KNN
- `class_weight='balanced'` for LogisticRegression
- k-means|| scalable init (cuML fidelity, once k-means++ oracle path passes)
- `rbc` (random ball cover) acceleration for DBSCAN/KNN

**Defer (v2+):**
- Approximate KNN (ivfflat/ivfpq) — breaks exact 1e-5 oracle; needs separate approximate-tolerance test design
- Sparse-input paths (TruncatedSVD on TF-IDF, sparse Lasso)
- IncrementalPCA, MiniBatchKMeans, multi-task linear models
- f16/bf16 validated precision paths
- Multi-GPU/distributed (cuml.dask, NCCL/UCX, `*_mg` paths)
- cuml.accel transparent acceleration (import-hook proxying)

### Architecture Approach

The architecture is a five-crate Cargo workspace with a strictly acyclic dependency chain and a clean two-generic boundary. The key architectural insight: `R: Runtime` (backend) is resolved at compile time — one wheel per backend, `R` monomorphized in `mlrs-py` by the enabled Cargo feature — while `F: Float` (float type) is resolved at runtime by the input array's dtype, so every wheel ships both f32 and f64 monomorphizations. The backend feature flags live in exactly one crate (`mlrs-backend`); `mlrs-kernels` has zero feature flags so kernels compile once and are reused across all backends. Fitted state (`coef_`, `components_`, `cluster_centers_`) is stored as device `ServerHandle` values inside the estimator struct and only materialized host-side lazily on Python attribute access, mirroring cuML's `CumlArrayDescriptor` pattern to avoid unnecessary device->host copies.

**Major components:**
1. `mlrs-core` — backend-agnostic vocabulary: `Estimator`/`Fit`/`Predict`/`Transform`/`Score` traits, `Params` (get/set), `Shape`/`Strides` (smallvec), `MlrsError` (thiserror), `DType` enum, sign-flip/label-permutation comparison contracts shared between `mlrs-algos` and `tests/`
2. `mlrs-kernels` — all `#[cube]`/`#[cube(launch)]` compute primitives generic over `<F: Float>` only: GEMM wrapper (cubecl-matmul), reduce, pairwise distance, SVD/eig (`decomp.rs`), coordinate descent, quasi-Newton, top-k selection, scatter-mean, elementwise ops; no backend features; compiles once
3. `mlrs-backend` — sole owner of `cpu`/`wgpu`/`cuda`/`rocm` Cargo features; `DeviceArray<R,F>` buffer abstraction; `arrow_bridge.rs` (Arrow->Bytes zero-copy with offset/nullability/alignment validation); buffer pool / `ExclusivePages` tuning; `caps.rs` capability queries (f64, plane/subgroup support)
4. `mlrs-algos` — estimator orchestration generic over `<F: Float, R: Runtime>`; composes primitives from `mlrs-kernels` over device arrays from `mlrs-backend`; implements `mlrs-core` traits; all four algorithm families
5. `mlrs-py` — cdylib; `#[pyclass]`/`#[pymethods]` sklearn estimators; Arrow PyCapsule + numpy adapters; f32/f64 dispatch by input dtype; concrete `R` fixed at compile time by feature; `#[global_allocator]` = mimalloc; one wheel per backend

**Dependency direction (strictly acyclic):**
`mlrs-py` -> `mlrs-algos` -> {`mlrs-kernels`, `mlrs-backend`} -> `mlrs-core`

### Critical Pitfalls

1. **Solver/defaults mismatch vs. sklearn oracle** — The oracle is scikit-learn, not cuML. Porting cuML's defaults (OLS=eig, KMeans=k-means||, TruncatedSVD=randomized, PCA without svd_flip) produces 1e-2..1e-1 errors that look like kernel bugs but are not. Mitigation: encode sklearn-matching defaults per the FEATURES.md solver table from P0; build sign-flip and label-permutation oracle helpers before any estimator.

2. **f64 absent on wgpu (the primary CI gate)** — WebGPU/WGSL has no 64-bit float; wgpu's `SHADER_F64` is native-only, absent on many Vulkan/Metal/DX12 adapters and all browser WebGPU. Building on f64 and assuming wgpu support causes CI to silently skip or hard-fail all f64 tests. Mitigation: capability-gate f64 via `feature_enabled(FloatKind::F64)` in P0; f32 is the portable correctness baseline.

3. **SVD/eig primitive gates two estimator families** — PCA, TruncatedSVD, OLS-svd, and Ridge-svd are unbuildable until a validated SVD/eig exists. This primitive is also the hardest single kernel in the project. Mitigation: build and validate SVD standalone with sign-flip oracle comparison before touching any of those estimators; give it a dedicated phase.

4. **Arrow zero-copy soundness on sliced/nullable/offset inputs** — `Float32Array::values()` returns the full backing buffer ignoring the array's logical offset; sliced arrays upload the wrong data window. Nullable arrays silently feed garbage/NaN into math. `bytemuck::cast_slice` panics on misaligned buffers. Mitigation: centralize the bridge in `mlrs-backend/arrow_bridge.rs`; validate offset, nullability, and alignment before upload; use PyCapsule ownership transfer (not bare `&[u8]` borrows) across the Python FFI boundary.

5. **LogisticRegression QN convergence parity is the highest correctness risk among estimators** — Quasi-Newton (L-BFGS/OWL-QN) convergence to match sklearn `lbfgs` requires careful penalty normalization and multinomial formulation. Mitigation: flag for deeper research before Phase 4 implementation; do not attempt without a research phase.

6. **Per-backend wheel naming collision** — maturin derives wheel name from the Cargo package name; four backend feature variants produce four wheels with identical names that overwrite each other on PyPI. Mitigation: assign distinct distribution names (`mlrs-cpu`, `mlrs-wgpu`, `mlrs-cuda`, `mlrs-rocm`) in workspace/pyproject.toml at P0.

7. **CubeCL `#[cube]` IR constraints fail in non-obvious ways** — `#[cube]` is a proc-macro rewriting to CubeCL IR: calling plain Rust helpers (E0433), `if`-expressions (E0308 ExpandElementTyped), method-style math (`x.exp()` -> E0599), raw numeric literals in generic kernels, and `u64`/`usize` device arithmetic all fail. AGENTS.md mandates reading the CubeCL error guideline before any fix. Mitigation: adopt guideline conventions from the first kernel; never apply a blind fix.

## Implications for Roadmap

Based on the combined research, the primitive dependency graph, and the pitfall phase-to-address map, the recommended phase structure is:

### Phase 0: Foundation — Oracle Harness, Backend Abstraction, Arrow Bridge
**Rationale:** Five of the seven critical pitfalls require a P0 design decision or are prevented here. Nothing downstream can be tested correctly without the oracle harness (sign-flip/label-permutation helpers), the Arrow bridge with validation, and the f64 capability-gate policy. The per-backend wheel naming scheme must also be locked now to avoid restructuring the workspace later.
**Delivers:** Complete workspace scaffolding; `mlrs-core` (all traits, types, oracle comparison contracts in `oracle.rs`); `mlrs-backend` skeleton with `DeviceArray<R,F>`, `arrow_bridge.rs` (offset/nullability/alignment validation), `caps.rs` (f64/plane capability gates), buffer pool skeleton; `mlrs-py` architecture with concrete `R` type alias per feature; oracle test harness with sign-flip and label-permutation helpers, seeded RNG fixtures (`StdRng::seed_from_u64`), `approx` 1e-5 (f64) and documented f32 tolerance policy; CI matrix (`--features cpu`, `--features wgpu`); one trivial end-to-end kernel to prove the generic R/F spine, zero-copy ingest, read-back, and oracle comparison all work; distinct distribution names and `available_backends()` contract.
**Features from FEATURES.md:** Cross-cutting sklearn API skeleton; Arrow zero-copy ingest infrastructure; oracle-test contract; f32/f64 dtype dispatch infrastructure.
**Avoids:** Pitfall 1 (solver table + oracle helpers codified before any estimator), Pitfall 4 (f64 capability gate policy), Pitfall 6 (Arrow offset/nullable/alignment validation), Pitfall 7 (wheel naming decided before packaging phase).
**Research flag:** Standard patterns — no deeper research needed for Phase 0.

### Phase 1: Core Compute Primitives — GEMM, Reductions, Pairwise Distance, Covariance
**Rationale:** Every estimator depends on at least one of these primitives. Validating them standalone means each downstream estimator is "mostly assembly." The f32 accumulation-drift risk (Pitfall 2) and plane/subgroup portability risk (Pitfall 5) must be addressed here or they force kernel rewrites later.
**Delivers:** `mlrs-kernels` with: GEMM wrapper around `cubecl-matmul` (`Strategy::Auto`); reductions (sum/mean/argmin/L2-norm) with stable tree-reduction pattern and shared-memory fallback for adapters without subgroups (never hardcode plane=32; always use `PLANE_DIM`); pairwise squared Euclidean distance with `max(d2, 0)` clamp (GEMM-based for large N; direct `||a-b||^2` for DBSCAN range-query path); covariance/Gram (XtX via GEMM); scatter-mean (KMeans centroid update); top-k selection. Each primitive has a standalone oracle test vs. a host reference, validating both f32 and f64, on both `cpu` and `wgpu`.
**Uses:** `cubecl-matmul`, `cubecl-reduce`, `cubecl-std` (all at 0.10.0); `bytemuck` for read-back; `approx` for tolerance checks.
**Avoids:** Pitfall 2 (f32 accumulation drift, stable tree reduction, distance clamp), Pitfall 3 (CubeCL IR conventions established on first kernel), Pitfall 5 (plane gating + shared-memory fallback).
**Research flag:** Standard patterns for GEMM and tree reductions (fully documented in provided CubeCL manuals). Flag the wgpu workgroup-storage limit if hit during distance kernel development.

### Phase 2: Decomposition Primitive — SVD/Eig (the Hard Gate)
**Rationale:** SVD (full + Jacobi) and/or symmetric eigendecomposition of the covariance matrix is the single highest-risk, highest-leverage primitive. It gates PCA, TruncatedSVD, OLS-svd path, and Ridge-svd path — four estimators in two families. Building and validating it standalone (with sign-flip oracle helpers from Phase 0) before any estimator exists means those estimators are cheap to assemble. Giving it a dedicated phase prevents it from being rushed inside a larger estimator phase.
**Delivers:** `mlrs-kernels/decomp.rs`: Jacobi SVD for general matrices; symmetric eigendecomposition of the covariance matrix (PCA `full` solver path); standalone oracle tests with sign-flip normalization (svd_flip convention); validated on both `cpu` and `wgpu`, both f32 and f64 (capability-gate skip on wgpu adapters lacking f64).
**Avoids:** Pitfall 1 (SVD sign ambiguity — `svd_flip` convention applied and oracle-tested here before any estimator uses it); Pitfall 3 (most complex kernel, highest CubeCL IR risk — requires the error guideline).
**Research flag:** NEEDS DEEPER RESEARCH — Jacobi SVD on GPU in CubeCL is not a pre-built `cubecl-matmul` primitive; the iterative Jacobi rotation kernel design for `#[cube]` requires domain research. Run `/gsd-plan-phase --research-phase 2` before writing any Phase 2 code.

### Phase 3: Closed-Form Estimators — LinearRegression, Ridge, PCA, TruncatedSVD
**Rationale:** These four estimators are "mostly assembly" once GEMM, covariance/XtX, and SVD/eig from Phases 1-2 are validated. They exercise the full pipeline (Arrow->kernel->device state->lazy materialize->oracle compare) with no convergence subtlety, de-risking the spine before the delicate iterative-solver work. PCA and TruncatedSVD are built together since TSVD is PCA minus centering over the same SVD primitive.
**Delivers:** `mlrs-algos/linear/ols.rs` (svd path as v1 default for sklearn match; eig as fast option); `mlrs-algos/linear/ridge.rs` (regularized eig/svd); `mlrs-algos/decomp/pca.rs` (center + eig/SVD + svd_flip sign convention; `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `mean_`; transform/inverse_transform); `mlrs-algos/decomp/truncated_svd.rs` (SVD without centering; oracle compared against sklearn `algorithm='arpack'` not the stochastic `randomized` default). Full fitted attributes, oracle tests with sign-flip normalization, both dtypes.
**Avoids:** Pitfall 1 (OLS=svd default, not eig; TruncatedSVD oracle vs sklearn `arpack`; PCA with `svd_flip`).
**Research flag:** Standard patterns — closed-form assembly on validated primitives is straightforward.

### Phase 4: Distance-Based Estimators + Iterative-Solver Estimators
**Rationale:** Two sub-tracks that can proceed in parallel after Phase 3 validates the pipeline. Distance-based estimators (KNN, KMeans, DBSCAN) all reuse the pairwise-distance primitive from Phase 1. KMeans must use k-means++ init (not k-means||) to match sklearn within 1e-5; labels require permutation-invariant comparison. Iterative-solver estimators (Lasso/ElasticNet via CD; LogisticRegression via QN) own their solver kernels and carry the convergence-parity risk. Lasso and ElasticNet are one feature — they share the CD kernel; Lasso = `l1_ratio==1`.
**Delivers:**
- `mlrs-kernels/coord_descent.rs` (CD step + soft-threshold); `mlrs-algos/linear/lasso_enet.rs` (single file, Lasso and ElasticNet together); both validated vs. sklearn coordinate-descent default at 1e-5.
- `mlrs-kernels/quasi_newton.rs` (L-BFGS inner kernels); `mlrs-algos/linear/logistic.rs` (QN solver, predict/predict_proba, `coef_`/`intercept_`/`classes_`/`n_iter_`, binary + softmax multiclass).
- `mlrs-algos/neighbors/nearest.rs` (brute exact kNN, Euclidean, `kneighbors` returning sorted distance+index, `kneighbors_graph`); `knn_classifier.rs` (uniform + distance-weighted vote, `predict_proba`); `knn_regressor.rs` (uniform + distance-weighted mean).
- `mlrs-algos/cluster/kmeans.rs` (Lloyd + k-means++ init; `cluster_centers_`/`labels_`/`inertia_`/`n_iter_`; predict/transform; label-permutation oracle compare); `dbscan.rs` (brute range query + BFS/union-find connected-components; `labels_` with -1 noise; `core_sample_indices_`).
**Avoids:** Pitfall 1 (KMeans k-means++ not k-means||; label-permutation oracle for KMeans/DBSCAN); Pitfall 2 (distance clamp in range-query path).
**Research flag:** LogisticRegression QN convergence parity NEEDS DEEPER RESEARCH — matching sklearn `lbfgs` within 1e-5 across penalty types and multinomial formulations is the highest correctness risk in the project. Run `/gsd-plan-phase --research-phase 4` for the LogisticRegression sub-task before implementation. CD convergence for Lasso/ElasticNet is medium-risk; validate tolerance during implementation.

### Phase 5: Python Surface — PyO3 Estimators, Per-Backend Wheels, sklearn Checks
**Rationale:** `mlrs-py` scaffolding can begin during Phase 3 (one estimator as the test scaffold), but the full surface is completed here. Finalizing Arrow PyCapsule ingest, GIL release, NotFittedError mapping, and per-backend wheel build completes the user-facing product.
**Delivers:** All `#[pyclass]`/`#[pymethods]` sklearn estimators; f32/f64 dispatch by input dtype; `fit` returns `self`; `get/set_params` for all estimators; `NotFittedError` mapping; Arrow PyCapsule ownership transfer + numpy adapter; `Python::allow_threads` around device compute; per-backend wheel build (`mlrs-cpu`, `mlrs-wgpu`, `mlrs-cuda`, `mlrs-rocm`) via `maturin build --features <backend>`; `abi3-py312`; `available_backends()` probe with clear no-driver import errors for cuda/rocm; `pytest` oracle tests + `sklearn.utils.estimator_checks` pass.
**Avoids:** Pitfall 6 (PyCapsule ownership transfer, lifetime soundness; no bare `&[u8]` borrows into Python-owned buffers); Pitfall 7 (distinct distribution names, no-driver clear error, GIL released around compute).
**Research flag:** Maturin per-feature distribution naming may need a small spike — the multi-distribution pattern is undocumented in maturin's first-party docs. Otherwise standard patterns.

### Phase Ordering Rationale

- **Primitives before estimators** (Phases 0-2 before Phases 3-4): SVD/eig gates two families, pairwise-distance gates three; validating standalone turns each downstream estimator into "assembly" rather than a debugging exercise.
- **Oracle harness in Phase 0, not Phase 1+**: without sign-flip and label-permutation helpers, PCA/SVD/KMeans/DBSCAN tests fail at 1e-5 for non-bugs; this is a P0 prerequisite per both FEATURES.md and PITFALLS.md.
- **SVD in its own phase (Phase 2)**: the single hardest primitive and the single most critical gate; a dedicated phase prevents it from being rushed inside a larger estimator phase and makes the decomposition-family unblocker explicit.
- **Closed-form estimators (Phase 3) before iterative (Phase 4)**: closed-form paths exercise the full pipeline with no convergence risk, catching Arrow bridge and oracle harness bugs cheaply before the delicate CD/QN convergence work.
- **Distance-based and iterative estimators in the same phase (Phase 4)**: they share no primitive dependencies with each other and can proceed in parallel sub-tracks; grouping them minimizes phase count without creating blocking dependencies.
- **Memory-efficiency is per-phase, not deferred**: buffer reuse (`client.empty` retain across iterations), `ExclusivePages` allocator tuning, and lazy fitted-state materialization are implemented in each phase where the relevant estimator is built — PROJECT.md constraint.

### Research Flags

Phases needing `/gsd-plan-phase --research-phase` before coding:
- **Phase 2 (SVD/eig primitive):** Jacobi SVD on GPU in CubeCL is not a pre-built `cubecl-matmul` primitive; the iterative Jacobi rotation kernel design for `#[cube]` requires dedicated domain research. This is the highest-risk single deliverable in the project.
- **Phase 4 — LogisticRegression sub-task:** Quasi-Newton (L-BFGS) convergence parity with sklearn `lbfgs` within 1e-5 across penalty types and multinomial formulations; requires research into penalty normalization, step-size schedule, and convergence criteria matching sklearn's internal implementation.

Phases with well-documented patterns (skip research-phase):
- **Phase 0:** Cargo workspace setup, CubeCL generics spine, Arrow->bytemuck->Bytes bridge, PyO3 architecture — fully documented in provided manuals and crates.io docs.
- **Phase 1:** GEMM via cubecl-matmul, tree reductions via shared-memory manual, pairwise distance — well-documented CubeCL patterns with authoritative manual coverage.
- **Phase 3:** Closed-form assembly (OLS, Ridge, PCA, TSVD) on validated primitives — straightforward once primitives exist.
- **Phase 5:** PyO3 + maturin packaging — well-documented; the multi-distribution spike is small.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Versions verified via crates.io API (cubecl 0.10.0, pyo3 0.28.3, maturin 1.13.3, arrow 59.0.0); provided CubeCL and optimisor manuals are authoritative. Maturin multi-backend wheel naming is MEDIUM — no first-party recipe exists. |
| Features | HIGH | Grounded in cuML v26.08 source signatures read directly + sklearn public API conventions. Solver-default mismatch table is the load-bearing correctness claim and is high-confidence documented risk. |
| Architecture | HIGH | Five-crate layout, generic R/F boundary, and Arrow zero-copy flow are grounded in provided CubeCL generics/matmul/slicing/allocator and zero-copy manuals which all agree. Estimator-trait exact shape is MEDIUM — synthesized from sklearn + cuML, no first-party prescription. |
| Pitfalls | HIGH | CubeCL error guideline is authoritative project canon; wgpu f64 absence confirmed via WebGPU spec and wgpu docs; cuML CONCERNS.md read directly (17 warp-size sites, int32 overflow, host-device debt). Arrow slicing/nullability pitfall grounded in `values()` semantics documentation. |

**Overall confidence:** HIGH

### Gaps to Address

- **Jacobi SVD CubeCL implementation strategy**: no pre-built cubecl-matmul SVD kernel exists; the specific iterative Jacobi rotation design for `#[cube]` needs research during Phase 2 planning before any code is written.
- **LogisticRegression QN convergence**: the exact penalty normalization and multinomial formulation that matches sklearn `lbfgs` within 1e-5 is not documented with sklearn-parity focus in the cuML source; needs research before Phase 4 implementation.
- **Maturin per-feature distribution naming**: the multi-distribution wheel pattern is undocumented in maturin first-party docs; a small build-system spike in Phase 0 or Phase 5 planning should validate the `pyproject.toml` structure for four distinct package names from one workspace.
- **f32 tolerance policy per estimator family**: f64 maps cleanly to 1e-5; the justified f32 tolerance for each algorithm family needs to be decided and documented in the oracle harness during Phase 0. cuML uses `unit_tol=1e-4` as a reference point.
- **wgpu f64 adapter coverage in CI**: the CI hardware's wgpu adapter may or may not expose `SHADER_F64`; Phase 0 CI setup should log which dtypes actually ran on which backend to make partial f64 coverage visible rather than silently passing.

## Sources

### Primary (HIGH confidence)
- Provided CubeCL manuals (`Cubecl_generics.md`, `Cubecl_basic_operations.md`, `Cubecl_plane.md`, `Cubecl_shared_memory.md`, `Cubecl_dynamic_vectorization.md`, `cubecl_matmul_gemm_example.md`, `Tuning_ExclusivePages_Allocator.md`, `Backend-Agnostic_Buffer_Slicing.md`, `cubecl_error_guideline.md`, error solution guides) — kernel generics, trait bounds, launch ordering, GEMM, reductions, buffer slicing, allocator tuning, IR failure modes
- Provided optimisor manuals (`ZERO_COPY_ARROW_CUBECL.md`, `ZERO_COPY_TRANSMUTATION_CUBECL.md`, `HALF_PRECISION_CUBECL.md`, `MIMALLOC_MANUAL.md`, `JEMALLOC_MANUAL.md`, `SMALLVEC_MANUAL.md`, `COMPACT_STR_OPTIMIZATION_EN.md`) — Arrow->bytemuck->CubeCL path, allocator pinning, smallvec/compact_str patterns
- `/tracel-ai/cubecl` (Context7) — Runtime trait, `client.features()`/`feature_enabled`, `#[comptime]` specialization
- crates.io API — verified current versions: cubecl 0.10.0 (2026-05-07), pyo3 0.28.3 (2026-04-02), maturin 1.13.3 (2026-05-11), arrow 59.0.0 (2026-06-09), tikv-jemallocator 0.7.0 (2026-05-25)
- `.planning/PROJECT.md`, `AGENTS.md` — scope, constraints, 1e-5 oracle, source/test separation, CubeCL error-guideline protocol
- `.planning/codebase/ARCHITECTURE.md`, `CONCERNS.md`, `TESTING.md` — cuML `CumlArray`/`Base`/`@reflect` patterns, int32 overflow bug, 17 warp-size sites, `assert_dbscan_equal`, fuzzy `array_equal`
- cuML v26.08 estimator sources (read directly) — estimator signatures, hyperparameters, fitted attributes, `_get_param_names`

### Secondary (MEDIUM confidence)
- https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html — Arrow PyCapsule Interface for zero-copy Python-Rust ownership transfer
- https://docs.rs/pyo3-arrow / https://docs.rs/arrow-pyarrow — Rust-side PyArrow FFI conversion patterns
- wgpu `Features` docs, WebGPU issue #2805 — confirms SHADER_F64 native-only, absent in browser/many adapters
- maturin Distribution guide — wheel name from Cargo package name; no first-party multi-feature recipe

### Tertiary (LOW confidence)
- scikit-learn default-solver behavior (training-data knowledge, cross-validated against cuML CONCERNS/FEATURES research) — needs runtime oracle validation to confirm 1e-5 achievable for each estimator family

---
*Research completed: 2026-06-11*
*Ready for roadmap: yes*
