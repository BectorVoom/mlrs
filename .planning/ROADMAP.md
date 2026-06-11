# Roadmap: mlrs — cuML in Rust

## Overview

mlrs is built primitive-first along a strictly acyclic five-crate workspace. The foundation phase lands the spine that makes everything else testable: the scikit-learn oracle harness (sign-flip + label-permutation helpers + f32 tolerance policy), the Arrow zero-copy bridge with validation, the f64 capability gate, and per-backend wheel naming. From there, compute primitives are validated standalone before any estimator exists — GEMM/reductions/distance/covariance, then the SVD/eig hard gate in its own phase. Estimators are then "mostly assembly": closed-form models (OLS, Ridge, PCA, TruncatedSVD) first to exercise the full pipeline with no convergence risk, then distance-based and iterative-solver estimators, then the complete PyO3 Python surface and per-backend wheels. Correctness against scikit-learn within 1e-5 on both the `cpu` and `wgpu` backends, for both `f32` and `f64` (capability-gated), is the gate at every phase.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation — Oracle, Backend Abstraction, Arrow Bridge** - Workspace, generic R/F spine, oracle harness, Arrow zero-copy bridge, f64 capability gate, allocator
- [ ] **Phase 2: Core Compute Primitives** - GEMM, reductions, pairwise distance, covariance/XᵀX validated standalone on cpu+wgpu
- [ ] **Phase 3: SVD / Eigendecomposition Primitive (Hard Gate)** - GPU Jacobi SVD + symmetric eig, sign-flip oracle-validated, gates four estimators
- [ ] **Phase 4: Closed-Form Estimators** - LinearRegression, Ridge, PCA, TruncatedSVD assembled on validated primitives
- [ ] **Phase 5: Distance-Based & Iterative-Solver Estimators** - KMeans, DBSCAN, KNN×3, Lasso, ElasticNet, LogisticRegression
- [ ] **Phase 6: Python Surface — PyO3 Estimators & Per-Backend Wheels** - sklearn-compatible pyclass estimators, Arrow PyCapsule, maturin per-backend wheels

## Phase Details

### Phase 1: Foundation — Oracle, Backend Abstraction, Arrow Bridge

**Goal**: The generic compute spine, oracle harness, and data bridge exist so every downstream primitive and estimator can be validated against scikit-learn within 1e-5 on cpu and wgpu.
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04, FOUND-05, FOUND-06, FOUND-07, FOUND-08, FOUND-09
**Success Criteria** (what must be TRUE):

  1. The five-crate workspace (`mlrs-core`, `mlrs-kernels`, `mlrs-backend`, `mlrs-algos`, `mlrs-py`) compiles with `--features cpu` and `--features wgpu`, and `--features cuda` compiles (without running); `mlrs-kernels` carries zero backend feature flags.
  2. A trivial end-to-end `#[cube]` kernel generic over `<F: Float>` runs on both cpu and wgpu, ingests an Arrow `Float32Array`/`Float64Array` zero-copy through the validated bridge, reads back, and the oracle harness asserts equality vs. a NumPy reference within 1e-5.
  3. The Arrow bridge rejects (does not silently upload) sliced/offset arrays, nullable arrays with set null bits, and misaligned buffers before any unsafe transmutation.
  4. The capability layer reports whether the active backend supports f64 (`feature_enabled(FloatKind::F64)`); f64 oracle tests skip/xfail with a logged reason on wgpu adapters lacking `SHADER_F64`, and the CI log shows which dtype ran on which backend.
  5. The oracle harness provides seeded-RNG fixtures, sign-flip and label-permutation comparison helpers, and a documented per-estimator-family f32 tolerance policy; the mimalloc global allocator is wired in `mlrs-py` with source/test code in separate files.

**Plans**: 5 plans
Plans:

- [x] 01-01-PLAN.md — Wave 0: scaffold five-crate workspace + toolchain/API spike (resolve CubeCL 0.10 symbols A1–A7)
- [x] 01-02-PLAN.md — mlrs-core oracle harness: assert_close, sign-flip, label-perm, npz loader, tolerance policy, BridgeError
- [x] 01-03-PLAN.md — Arrow zero-copy bridge (hard-reject validation) + f64 capability gate
- [x] 01-04-PLAN.md — DeviceArray + buffer-reuse pool with logged counters
- [ ] 01-05-PLAN.md — end-to-end pipeline test (Arrow→kernel→oracle) + gen_oracle.py fixtures + mimalloc allocator

### Phase 2: Core Compute Primitives

**Goal**: GEMM, reductions, pairwise distance, and covariance/XᵀX are validated standalone so downstream estimators reuse trusted kernels rather than debugging math inside estimators.
**Depends on**: Phase 1
**Requirements**: PRIM-01, PRIM-02, PRIM-03, PRIM-04
**Success Criteria** (what must be TRUE):

  1. A GEMM primitive (wrapping `cubecl-matmul`) matches a host reference within tolerance for f32 and f64 on both cpu and wgpu.
  2. Reduction primitives (sum/mean/min/max/argmin/L2-norm) pass on wgpu via both a plane/subgroup path and a shared-memory fallback, with no hardcoded plane width (uses `PLANE_DIM`), numerically stable on large inputs.
  3. A pairwise squared-Euclidean distance primitive with a `max(d², 0)` clamp produces no negative distances under f32 and matches the host reference within tolerance.
  4. A covariance / XᵀX (Gram) primitive built on GEMM matches the host reference within tolerance for both dtypes on cpu and wgpu.

**Plans**: TBD

### Phase 3: SVD / Eigendecomposition Primitive (Hard Gate)

**Goal**: A validated SVD / symmetric-eigendecomposition primitive exists with the `svd_flip` sign convention, unblocking PCA, TruncatedSVD, and the OLS/Ridge SVD solver paths.
**Depends on**: Phase 2
**Requirements**: PRIM-05
**Success Criteria** (what must be TRUE):

  1. A Jacobi SVD for general matrices matches a host/NumPy reference within tolerance after sign-flip normalization, for f32 and f64 (f64 capability-gated on wgpu).
  2. A symmetric eigendecomposition of a covariance matrix (PCA `full` solver path) matches the reference eigenvalues/eigenvectors within tolerance after sign alignment.
  3. The SVD/eig oracle tests pass on both cpu and wgpu (with documented f32 tolerance), proving the primitive before any estimator consumes it.

**Plans**: TBD
**Research flag**: NEEDS DEEPER RESEARCH — Jacobi SVD on GPU in CubeCL is not a pre-built `cubecl-matmul` primitive; the iterative Jacobi-rotation kernel design for `#[cube]` requires domain research. Run `/gsd-plan-phase --research-phase 3` before writing any code.

### Phase 4: Closed-Form Estimators

**Goal**: A data scientist can fit LinearRegression, Ridge, PCA, and TruncatedSVD and get results matching scikit-learn within 1e-5, exercising the full Arrow→kernel→device-state→materialize→oracle pipeline with no convergence risk.
**Depends on**: Phase 3
**Requirements**: LINEAR-01, LINEAR-02, DECOMP-01, DECOMP-02
**Success Criteria** (what must be TRUE):

  1. `LinearRegression` (SVD-based, matching sklearn's default lstsq) fits and exposes `coef_`/`intercept_`, predicting within 1e-5 of scikit-learn on random data via cpu and wgpu.
  2. `Ridge` with an `alpha` penalty produces `coef_`/`intercept_` matching scikit-learn within tolerance.
  3. `PCA` with `n_components` exposes `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `mean_`, and `transform`/`inverse_transform`, matching scikit-learn after `svd_flip` sign alignment.
  4. `TruncatedSVD` (no centering) exposes `components_`/`explained_variance_`/`singular_values_`/`transform`, matching scikit-learn's deterministic `arpack` path after sign alignment.

**Plans**: TBD

### Phase 5: Distance-Based & Iterative-Solver Estimators

**Goal**: A data scientist can fit the clustering, neighbors, and iterative-solver linear models matching scikit-learn within tolerance (up to label permutation where applicable), completing the v1 algorithm surface in Rust.
**Depends on**: Phase 4
**Requirements**: LINEAR-03, LINEAR-04, LINEAR-05, CLUSTER-01, CLUSTER-02, NEIGH-01, NEIGH-02, NEIGH-03
**Success Criteria** (what must be TRUE):

  1. `KMeans` (k-means++ init, sklearn default) exposes `cluster_centers_`/`labels_`/`inertia_` and predicts new points, matching scikit-learn up to label permutation; `DBSCAN` (`eps`/`min_samples`) exposes `labels_` (noise = -1) and `core_sample_indices_`, matching scikit-learn up to label permutation.
  2. `NearestNeighbors` (brute-force) returns k nearest distances and indices within 1e-5; `KNeighborsClassifier` (`predict`/`predict_proba`) and `KNeighborsRegressor` (`predict`) match scikit-learn within tolerance.
  3. `Lasso` and `ElasticNet` share a coordinate-descent kernel (Lasso = `l1_ratio==1`) and produce `coef_` matching scikit-learn's CD solver within tolerance.
  4. `LogisticRegression` (quasi-Newton/L-BFGS) with stable softmax handles binary and multiclass, with `predict`/`predict_proba` matching scikit-learn's `lbfgs` solver within tolerance.

**Plans**: TBD
**Research flag**: NEEDS DEEPER RESEARCH (LogisticRegression sub-task) — matching sklearn `lbfgs` within 1e-5 across penalty types and multinomial formulations is the highest correctness risk in the project; penalty normalization, step-size schedule, and convergence criteria need research. Run `/gsd-plan-phase --research-phase 5` for the LogisticRegression sub-task before implementation. CD convergence for Lasso/ElasticNet is medium-risk; validate tolerance during implementation.

### Phase 6: Python Surface — PyO3 Estimators & Per-Backend Wheels

**Goal**: A Python ≥ 3.12 data scientist can `pip install` the wheel matching their backend and use all 11 v1 estimators through a sklearn-compatible API with zero-copy Arrow ingest and the GIL released during compute.
**Depends on**: Phase 5
**Requirements**: PY-01, PY-02, PY-03, PY-04, PY-05
**Success Criteria** (what must be TRUE):

  1. All 11 v1 estimators are `#[pyclass]` objects with sklearn-compatible `fit`/`predict`/`transform`/`score` (`fit` returns `self`) and pass `pytest` oracle tests plus relevant `sklearn.utils.estimator_checks`.
  2. Estimators support `get_params`/`set_params` with constructor hyperparameters matching scikit-learn naming, and accept f32 and f64 NumPy/Arrow inputs via runtime dtype dispatch.
  3. NumPy/Arrow inputs cross the boundary via the Arrow PyCapsule interface with correct ownership/lifetime handling (no bare `&[u8]` borrows into Python-owned buffers), and `Python::allow_threads` releases the GIL around device compute.
  4. Per-backend wheels build via `maturin build --features <backend>` under distinct distribution names (`mlrs-cpu`, `mlrs-wgpu`, `mlrs-cuda`, `mlrs-rocm`) with `abi3-py312`; importing a wheel whose driver is absent fails with a clear error.

**Plans**: TBD
**Research flag**: Maturin per-feature distribution naming may need a small build-system spike — the multi-distribution pattern is undocumented in maturin's first-party docs. Otherwise standard patterns.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation — Oracle, Backend Abstraction, Arrow Bridge | 4/5 | In Progress|  |
| 2. Core Compute Primitives | 0/TBD | Not started | - |
| 3. SVD / Eigendecomposition Primitive (Hard Gate) | 0/TBD | Not started | - |
| 4. Closed-Form Estimators | 0/TBD | Not started | - |
| 5. Distance-Based & Iterative-Solver Estimators | 0/TBD | Not started | - |
| 6. Python Surface — PyO3 Estimators & Per-Backend Wheels | 0/TBD | Not started | - |
