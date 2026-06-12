# Requirements: mlrs — cuML in Rust

**Defined:** 2026-06-11
**Core Value:** Correct, memory-efficient ML algorithms that match scikit-learn within 1e-5, running on any CubeCL backend from a single generic codebase.

## v1 Requirements

Requirements for the initial release. Each maps to roadmap phases. The estimator-facing requirements are written from the perspective of a data scientist using the sklearn-compatible Python API; foundation/primitive requirements are written from the perspective of the library developer building on the workspace.

### Foundation

- [x] **FOUND-01**: A Cargo workspace splits compute kernels, backend/runtime selection, algorithms, and Python bindings into separate single-responsibility crates (`mlrs-core`, `mlrs-kernels`, `mlrs-backend`, `mlrs-algos`, `mlrs-py`)
- [x] **FOUND-02**: Compute kernels are generic over float type (`f32`/`f64`) and over the CubeCL runtime, compiled once in a feature-free kernels crate
- [x] **FOUND-03**: A backend is selected via Cargo features (`cuda`, `rocm`, `wgpu`, `cpu`); `cuda` compiles in this environment even though it cannot be run, and `wgpu`+`cpu` run as the correctness gate
- [x] **FOUND-04**: A backend capability layer queries runtime support (notably f64 / plane / subgroup) and gates or skips paths the active backend cannot run, so f32 stays the portable baseline on wgpu
- [x] **FOUND-05**: A memory-efficient device-array abstraction wraps CubeCL buffers with reuse and minimal host↔device copies
- [x] **FOUND-06**: Input data crosses into Rust as Apache Arrow buffers and feeds CubeCL device buffers zero-copy, with validation of offsets, null bitmaps, and alignment before any unsafe transmutation
- [x] **FOUND-07**: An oracle test harness generates seeded random inputs, runs scikit-learn to produce reference outputs, and asserts results match within abs/rel error ≤ 1e-5
- [x] **FOUND-08**: The oracle harness provides sign-flip (for SVD/PCA components) and label-permutation (for clustering) comparison helpers, plus a documented per-estimator f32 tolerance policy
- [x] **FOUND-09**: A custom global allocator (mimalloc or jemalloc) is wired in, with source and test code kept in separate files per the project CubeCL protocol

### Compute Primitives

- [x] **PRIM-01**: A backend-portable GEMM / matrix-multiply primitive is available to estimators and validated against an oracle (Plan 02-01: cubek-matmul 0.2.0 wrap; transpose flags D-06; f32/f64 oracle within 1e-5 on cpu+wgpu)
- [x] **PRIM-02**: Numerically-stable reduction primitives (sum/mean/min/max/argmin) work with a plane/subgroup path and a shared-memory fallback that both pass on wgpu
- [x] **PRIM-03**: A pairwise-distance primitive (Euclidean/squared) with a `max(d², 0)` clamp serves KMeans, DBSCAN, and KNN
- [x] **PRIM-04**: A covariance / XᵀX primitive serves PCA and the linear-model closed-form solvers (Plan 02-04: column-mean center + GEMM(transa) AᵀA + 1/(n-ddof) scale, ddof=0/1 match np.cov on cpu+wgpu; Plan 02-05 D-10 gate proves the GEMM-output-buffer reuse)
- [ ] **PRIM-05**: An SVD / eigendecomposition primitive (GPU Jacobi or equivalent) serves PCA, TruncatedSVD, and the OLS/Ridge SVD solver paths, validated against an oracle within tolerance

### Linear Models

- [ ] **LINEAR-01**: User can fit `LinearRegression` (OLS, SVD-based to match sklearn's default) and read `coef_` and `intercept_`, predicting within 1e-5 of scikit-learn
- [ ] **LINEAR-02**: User can fit `Ridge` with an `alpha` penalty and obtain `coef_`/`intercept_` matching scikit-learn
- [ ] **LINEAR-03**: User can fit `Lasso` (coordinate-descent) with `alpha` and obtain a sparse `coef_` matching scikit-learn within tolerance
- [ ] **LINEAR-04**: User can fit `ElasticNet` (`alpha`, `l1_ratio`, shared coordinate-descent with Lasso) matching scikit-learn within tolerance
- [ ] **LINEAR-05**: User can fit `LogisticRegression` (quasi-Newton/L-BFGS) for binary and multiclass classification with stable softmax, `predict`/`predict_proba` matching scikit-learn's `lbfgs` solver within tolerance

### Clustering

- [ ] **CLUSTER-01**: User can fit `KMeans` with k-means++ initialization (sklearn default), read `cluster_centers_`, `labels_`, `inertia_`, and `predict` new points, matching scikit-learn up to label permutation
- [ ] **CLUSTER-02**: User can fit `DBSCAN` with `eps`/`min_samples`, read `labels_` (including noise as -1) and `core_sample_indices_`, matching scikit-learn up to label permutation

### Decomposition

- [ ] **DECOMP-01**: User can fit `PCA` with `n_components`, read `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `mean_`, and `transform`/`inverse_transform`, matching scikit-learn after sign alignment
- [ ] **DECOMP-02**: User can fit `TruncatedSVD` with `n_components` (no centering), read `components_`/`explained_variance_`/`singular_values_` and `transform`, matching scikit-learn's deterministic `arpack` path after sign alignment

### Neighbors

- [ ] **NEIGH-01**: User can fit `NearestNeighbors` (brute-force) and call `kneighbors` to get the k nearest distances and indices matching scikit-learn within 1e-5
- [ ] **NEIGH-02**: User can use `KNeighborsClassifier` (`fit`/`predict`/`predict_proba`) matching scikit-learn
- [ ] **NEIGH-03**: User can use `KNeighborsRegressor` (`fit`/`predict`) matching scikit-learn within tolerance

### Python Surface

- [ ] **PY-01**: All v1 estimators are exposed as PyO3 `#[pyclass]` objects with sklearn-compatible `fit`/`predict`/`transform`/`score` methods, `fit` returning `self`
- [ ] **PY-02**: Estimators support `get_params`/`set_params` and constructor hyperparameters matching scikit-learn naming
- [ ] **PY-03**: NumPy / Arrow inputs cross the Python↔Rust boundary via the Arrow PyCapsule interface with correct ownership/lifetime handling and the GIL released during compute
- [ ] **PY-04**: Per-backend Python wheels build via maturin under distinct distribution names (e.g. `mlrs-cpu`, `mlrs-wgpu`) so a user installs the package matching their backend; importing a wheel whose driver is absent fails with a clear error
- [ ] **PY-05**: Estimators support f32 and f64 inputs (runtime dtype dispatch), targeting Python ≥ 3.12

## v2 Requirements

Deferred to future release. Tracked but not in the current roadmap.

### Algorithms

- **V2-01**: Tree / ensemble algorithms (RandomForest, decision trees, FIL, gradient boosting)
- **V2-02**: SVM (SVC, SVR)
- **V2-03**: Manifold learning (UMAP, t-SNE)
- **V2-04**: Time-series (ARIMA, AutoARIMA, Holt-Winters)
- **V2-05**: Additional decomposition/solvers (IncrementalPCA, randomized SVD path, LARS)

### Platform

- **V2-06**: Multi-GPU / distributed training (Dask-equivalent, NCCL/UCX)
- **V2-07**: cuml.accel-style transparent acceleration of scikit-learn via import hooks
- **V2-08**: Validated half-precision (f16/bf16) compute paths

## Out of Scope

Explicitly excluded for v1. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Multi-GPU / Dask distribution | Single-device first; distribution is a separate milestone (V2-06) |
| cuml.accel transparent acceleration | Large surface area, no algorithmic value for v1 (V2-07) |
| Tree / ensemble / SVM / manifold / time-series | High complexity; deferred to later milestones (V2-01..04) |
| Bit-exact reproduction of cuML internals | Goal is numerical agreement with scikit-learn (≤1e-5), not kernel-for-kernel cuML parity |
| Validated f16/bf16 paths | Infrastructure may allow it but not a v1 deliverable (V2-08) |
| Runnable/validated CUDA & ROCm CI | `cuda` is compile-only here; `rocm` opportunistic — wgpu+cpu are the v1 gate |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Complete |
| FOUND-02 | Phase 1 | Complete |
| FOUND-03 | Phase 1 | Complete |
| FOUND-04 | Phase 1 | Complete |
| FOUND-05 | Phase 1 | Complete |
| FOUND-06 | Phase 1 | Complete |
| FOUND-07 | Phase 1 | Complete |
| FOUND-08 | Phase 1 | Complete |
| FOUND-09 | Phase 1 | Complete |
| PRIM-01 | Phase 2 | Complete (Plan 02-01) |
| PRIM-02 | Phase 2 | Complete (Plan 02-02) |
| PRIM-03 | Phase 2 | Complete (02-03) |
| PRIM-04 | Phase 2 | Complete (02-04; D-10 gate 02-05) |
| PRIM-05 | Phase 3 | Pending |
| LINEAR-01 | Phase 4 | Pending |
| LINEAR-02 | Phase 4 | Pending |
| DECOMP-01 | Phase 4 | Pending |
| DECOMP-02 | Phase 4 | Pending |
| LINEAR-03 | Phase 5 | Pending |
| LINEAR-04 | Phase 5 | Pending |
| LINEAR-05 | Phase 5 | Pending |
| CLUSTER-01 | Phase 5 | Pending |
| CLUSTER-02 | Phase 5 | Pending |
| NEIGH-01 | Phase 5 | Pending |
| NEIGH-02 | Phase 5 | Pending |
| NEIGH-03 | Phase 5 | Pending |
| PY-01 | Phase 6 | Pending |
| PY-02 | Phase 6 | Pending |
| PY-03 | Phase 6 | Pending |
| PY-04 | Phase 6 | Pending |
| PY-05 | Phase 6 | Pending |

**Coverage:**

- v1 requirements: 27 total
- Mapped to phases: 27 ✓
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-11*
*Last updated: 2026-06-11 after roadmap creation (traceability populated, 27/27 mapped)*
