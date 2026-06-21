# Requirements: mlrs — cuML in Rust

**Defined:** 2026-06-14
**Milestone:** v2.0 Breadth Sweep
**Core Value:** Correct, memory-efficient ML algorithms that match scikit-learn within 1e-5, running on any CubeCL backend from a single generic codebase.

## v2 Requirements

Requirements for the v2.0 Breadth Sweep milestone — ~18 additional scikit-learn-compatible estimators built as assembly on v1's validated primitive base plus five new feature-free CubeCL primitives. Estimator-facing requirements are written from the perspective of a data scientist using the sklearn-compatible Python API; primitive requirements from the perspective of the library developer. **Oracle = scikit-learn (NOT cuML); gate = cpu(f64) + rocm(f32), f64-on-rocm skips-with-log.** Every f64 oracle case uses `skip_f64_with_log`; f32-on-rocm uses a documented per-family tolerance band with exact labels/argmax as the hard gate. All new kernels follow the cpu-MLIR-safe GATHER idiom (no SharedMemory, no cross-unit atomics).

### Compute Primitives (new)

- [x] **PRIM-06**: A reproducible seeded RNG-matrix primitive (host SplitMix64 promoted to `prims/rng.rs`, no `OsRng` per ASVS V6) generates Gaussian and Achlioptas-sparse projection matrices and shuffle permutations, validated for distribution + seed-reproducibility
- [x] **PRIM-07**: An incremental-SVD merge primitive composes over the existing Jacobi `svd` to update a running decomposition with a new batch (mean-correction row, `svd_flip(u_based_decision=False)`, ddof=1), serving IncrementalPCA
- [x] **PRIM-08**: A kernel-matrix primitive (linear / RBF / polynomial / sigmoid) composes over the existing pairwise-distance/GEMM prims, validated against a host reference within tolerance for f32/f64 — serving KernelRidge, KernelDensity, and the spectral affinity
- [x] **PRIM-09**: A graph-Laplacian primitive (normalized Laplacian with GATHER degree-normalization, no atomics) composes the kernel/affinity matrix, serving the spectral estimators
- [ ] **PRIM-10**: A minibatch SGD solver primitive (hinge / log / squared / squared-hinge / epsilon-insensitive losses; learning-rate schedules; GATHER two-pass margin+gradient update, cpu-MLIR-safe) is validated standalone on a convex objective before any estimator consumes it

### Covariance

- [x] **COV-01**: A data scientist can fit `EmpiricalCovariance` and get `covariance_`, `location_`, and `precision_` matching scikit-learn within 1e-5 (MLE / ddof=0)
- [x] **COV-02**: A data scientist can fit `LedoitWolf` and get the shrinkage-regularized `covariance_` and `shrinkage_` (clipped to [0,1]) matching scikit-learn within 1e-5

### Decomposition

- [x] **DECOMP-03**: A data scientist can fit `IncrementalPCA` (including via `partial_fit` over batches) and get `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `mean_`, `var_`, and `transform`/`inverse_transform` matching scikit-learn within 1e-5 after `svd_flip` sign alignment

### Random Projection

- [x] **PROJ-01**: A data scientist can fit `GaussianRandomProjection` (`n_components='auto'` via `johnson_lindenstrauss_min_dim`) and `transform` inputs — **property-gated** (JL distance-preservation/distortion bound, matrix-distribution stats, seed-reproducibility, `transform == X·componentsᵀ` self-consistency), NOT a 1e-5 value match (mlrs RNG ≠ NumPy MT19937); `johnson_lindenstrauss_min_dim` itself is value-matched
- [x] **PROJ-02**: A data scientist can fit `SparseRandomProjection` (Achlioptas sparse matrix with configurable `density`) and `transform`, property-gated as PROJ-01; sparse input is accepted by densifying at the Python ingress boundary

### Kernel

- [x] **KERNEL-01**: A data scientist can fit `KernelRidge` (dual-coefficient solve of `(K + αI)` via the v1 Cholesky primitive; kernels linear/rbf/polynomial/sigmoid with `gamma`/`degree`/`coef0`) and `predict`, matching scikit-learn within 1e-5
- [x] **KERNEL-02**: A data scientist can fit `KernelDensity` (kernels + `bandwidth`) and call `score_samples` for log-density, using a numerically-stable log-sum-exp, matching scikit-learn within tolerance

### Spectral

- [x] **SPECTRAL-01**: A data scientist can fit `SpectralEmbedding` (affinity → normalized graph Laplacian → smallest non-trivial eigenvectors via the v1 `eig`, dropping the ≈0 eigenvector, deterministic sign canonicalization) and get `embedding_` matching scikit-learn within tolerance after sign alignment
- [x] **SPECTRAL-02**: A data scientist can fit `SpectralClustering` (spectral embedding → KMeans) and get `labels_` matching scikit-learn up to label permutation

### SGD / Linear SVM

- [ ] **SGDSVM-01**: A data scientist can fit `MBSGDClassifier` (sklearn `SGDClassifier` objectives: hinge / log / squared-hinge; learning-rate schedules incl. `optimal`) with `predict`/`predict_proba`, matching scikit-learn within tolerance under a pinned deterministic oracle (`shuffle=False`, fixed schedule/epochs)
- [ ] **SGDSVM-02**: A data scientist can fit `MBSGDRegressor` (squared-loss / epsilon-insensitive; `invscaling` default) with `predict`, matching scikit-learn within tolerance under the pinned deterministic oracle
- [ ] **SGDSVM-03**: A data scientist can fit `LinearSVC` (`loss='squared_hinge'` default, `penalty`, `dual='auto'`, `intercept_scaling`) with `predict`, matching scikit-learn within tolerance
- [ ] **SGDSVM-04**: A data scientist can fit `LinearSVR` (`loss='squared_epsilon_insensitive'` default, `epsilon`) with `predict`, matching scikit-learn within tolerance

### Naive Bayes

- [ ] **NB-01**: A data scientist can fit `GaussianNB` (per-class Gaussian likelihood with `var_smoothing`, log-sum-exp) with `predict`/`predict_proba` matching scikit-learn within tolerance
- [ ] **NB-02**: A data scientist can fit `MultinomialNB` (multinomial likelihood, `alpha` smoothing) with `predict`/`predict_proba` matching scikit-learn within tolerance; sparse input accepted via ingress densify
- [ ] **NB-03**: A data scientist can fit `BernoulliNB` (binary likelihood with the `(1−x)·log(1−p)` non-occurrence term, `binarize`) matching scikit-learn within tolerance
- [ ] **NB-04**: A data scientist can fit `ComplementNB` (complement-class weights, argmin decision) matching scikit-learn within tolerance
- [ ] **NB-05**: A data scientist can fit `CategoricalNB` (per-feature categorical likelihood, `alpha`) on categorical-encoded integer features, matching scikit-learn within tolerance

### Python Surface

- [ ] **PY-06**: All new v2 estimators are `#[pyclass]`-backed with sklearn-compatible `fit`/`predict`/`transform`/`score` (+ `partial_fit` for IncrementalPCA/MBSGD, `score_samples` for KernelDensity), `get_params`/`set_params` with sklearn-named hyperparameters, f32/f64 runtime dispatch, GIL release during compute, and ship inside the existing four per-backend wheels

## Future Requirements (deferred)

Tracked in `.planning/notes/v3-hard-algorithm-backlog.md`: RandomForest → FIL → TreeSHAP, UMAP, HDBSCAN, ARIMA/AutoARIMA, kernel SVC/SVR (SMO), explainer/SHAP, genetic, cuml.accel, Dask multi-GPU, native sparse Arrow/CSR interchange, smallest-eigenpair (Lanczos/shift-invert) solver.

## Out of Scope (this milestone)

- **Native sparse (CSR) device interchange** — v2 accepts sparse by densifying at ingress; a CSR device path (segmented reduce on cpu-MLIR) is Tier-3 v3 infrastructure
- **Shift-invert / Lanczos smallest-eigenpair solver** — v2 spectral uses full-spectrum Jacobi `eig` + host-slice; a dedicated solver is deferred unless spectral scales up
- **A dedicated incremental-SVD update kernel** — v2 composes the existing Jacobi `svd` per batch; a bespoke device update kernel is deferred
- **cuML-faithful behavior** — scikit-learn remains the sole oracle; cuML's divergent SGD loss set and LinearSVC solver are explicitly NOT matched
- **New ABI / dependency** — v2 adds zero compute dependencies; no `cubek-random`, no pyo3 bump (stays 0.28)

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PRIM-06 | Phase 7 | Done (07-02) |
| PRIM-07 | Phase 7 | Complete (07-03) |
| COV-01 | Phase 7 | Complete (07-04) |
| COV-02 | Phase 7 | Complete (07-04) |
| DECOMP-03 | Phase 7 | Complete (07-05) |
| PROJ-01 | Phase 7 | Complete |
| PROJ-02 | Phase 7 | Complete |
| PRIM-08 | Phase 8 | Complete (08-02) |
| KERNEL-01 | Phase 8 | Complete |
| KERNEL-02 | Phase 8 | Complete (08-04) |
| PRIM-09 | Phase 9 | Complete (09-02) |
| SPECTRAL-01 | Phase 9 | Complete |
| SPECTRAL-02 | Phase 9 | Complete (09-04) |
| PRIM-10 | Phase 10 | Pending |
| SGDSVM-01 | Phase 10 | Pending |
| SGDSVM-02 | Phase 10 | Pending |
| SGDSVM-03 | Phase 10 | Pending |
| SGDSVM-04 | Phase 10 | Pending |
| NB-01 | Phase 11 | Pending |
| NB-02 | Phase 11 | Pending |
| NB-03 | Phase 11 | Pending |
| NB-04 | Phase 11 | Pending |
| NB-05 | Phase 11 | Pending |
| PY-06 | Phase 11 (cross-cutting; estimators wrapped incrementally per phase) | Pending |

**Coverage:** 24/24 v2 requirements mapped, each to exactly one phase. No orphans, no duplicates.

---
*Requirements defined: 2026-06-14 for v2.0 Breadth Sweep (research-backed: .planning/research/SUMMARY.md)*
*Traceability populated by roadmapper: 2026-06-14 (Phases 7–11)*
