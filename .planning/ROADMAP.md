# Roadmap: mlrs — cuML in Rust

## Milestones

- ✅ **v1.0 Core ML Library** — Phases 1–6 (shipped 2026-06-14) → [archive](milestones/v1.0-ROADMAP.md)
- 🚧 **v2.0 Breadth Sweep** — Phases 7–11 (planned 2026-06-14; seed: [v2-breadth-roadmap](seeds/v2-breadth-roadmap.md), research: [SUMMARY](research/SUMMARY.md))

## Phases

<details>
<summary>✅ v1.0 Core ML Library (Phases 1–6) — SHIPPED 2026-06-14 — 38 plans, 12 estimators</summary>

- [x] Phase 1: Foundation — Oracle, Backend Abstraction, Arrow Bridge (5/5 plans) — completed 2026-06-11
- [x] Phase 2: Core Compute Primitives (5/5 plans) — completed 2026-06-12
- [x] Phase 3: SVD / Eigendecomposition Primitive — Hard Gate (5/5 plans) — completed 2026-06-12
- [x] Phase 4: Closed-Form Estimators (5/5 plans) — completed 2026-06-12
- [x] Phase 5: Distance-Based & Iterative-Solver Estimators (11/11 plans) — completed 2026-06-13
- [x] Phase 6: Python Surface — PyO3 Estimators & Per-Backend Wheels (6/6 plans) — completed 2026-06-14

Full phase detail, plans, and per-plan notes: [milestones/v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)

</details>

### 🚧 v2.0 Breadth Sweep (Phases 7–11)

~16 sklearn-compatible estimators across five families, built as assembly on v1's validated primitive base plus five new feature-free CubeCL primitives — one (or zero) per phase. **No new compute dependency** (workspace `Cargo.toml` unchanged; pyo3 stays 0.28). Oracle = scikit-learn ≤ 1e-5; gate = cpu(f64) + rocm(f32), f64-on-rocm skips-with-log. Build order **7 → 8 → 9 → 10 → 11** is dependency-correct (P9 hard-depends on P8's kernel-matrix prim). Each phase keeps the v1 primitive-first shape: land + standalone-validate the new prim with its build-failing PoolStats memory gate, then assemble estimators on it.

- [x] **Phase 7: Covariance & Projection** — RNG-matrix + incremental-SVD prims, PartialFit trait; EmpiricalCovariance, LedoitWolf, IncrementalPCA, Gaussian/SparseRandomProjection (completed 2026-06-20)
- [x] **Phase 8: Kernel Family** — kernel-matrix prim (linear/RBF/poly/sigmoid), ScoreSamples trait; KernelRidge, KernelDensity (completed 2026-06-21; verified 4/4 must-haves, UAT passed)
- [ ] **Phase 9: Spectral Family** — graph-Laplacian prim (hard dep on Phase 8 kernel-matrix); SpectralEmbedding, SpectralClustering
- [ ] **Phase 10: SGD / Linear-SVM** — SGD solver prim (the one new device solver, highest cpu-MLIR risk); MBSGDClassifier, MBSGDRegressor, LinearSVC, LinearSVR
- [ ] **Phase 11: Naive Bayes** — reductions-only closing bookend; GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB

## Phase Details

### Phase 7: Covariance & Projection

**Goal**: A data scientist can fit covariance estimators and projection transformers that reuse v1's covariance + SVD prims — the lowest-risk opener that lands the reusable host RNG-matrix primitive and the incremental-SVD merge, and introduces the `PartialFit` trait.
**Depends on**: Nothing new (assembles on v1 covariance prim, Jacobi SVD, GEMM)
**Requirements**: PRIM-06, PRIM-07, COV-01, COV-02, DECOMP-03, PROJ-01, PROJ-02
**Success Criteria** (what must be TRUE):

  1. `prims/rng.rs` (host SplitMix64, promoted from kmeans++, no `OsRng` per ASVS-V6) generates Gaussian and Achlioptas-sparse matrices + permutations, validated for distribution stats and seed-reproducibility (same seed → identical matrix across runs/backends), with its PoolStats memory gate.
  2. `prims/incremental_svd.rs` (glue over v1 `svd`) merges a running decomposition with a new batch (mean-correction row, `svd_flip(u_based_decision=False)`, ddof=1), validated standalone against a **2+ batch** host reference.
  3. A user can fit `EmpiricalCovariance` (ddof=0 MLE) and `LedoitWolf`, getting `covariance_`/`location_`/`precision_` and `shrinkage_` (clipped to [0,1]) matching scikit-learn within 1e-5 — both `shrinkage_` and `covariance_` gated across two `n`.
  4. A user can fit `IncrementalPCA` via `partial_fit` over batches and get `components_`/`explained_variance_`/`explained_variance_ratio_`/`singular_values_`/`mean_`/`var_` + `transform`/`inverse_transform` matching scikit-learn within 1e-5 after `svd_flip` (V-based) sign alignment.
  5. A user can fit `GaussianRandomProjection` and `SparseRandomProjection` (`n_components='auto'` via `johnson_lindenstrauss_min_dim`) and `transform` — **property-gated** (JL distortion bound, matrix-distribution stats, seed-reproducibility, `transform == X·componentsᵀ` self-consistency), NOT a 1e-5 value match; `johnson_lindenstrauss_min_dim` itself value-matched. Sparse input densified at the Python ingress.

**Recurring gates**: `skip_f64_with_log` on every f64 oracle case; documented f32-on-rocm band for LedoitWolf/IncrementalPCA (components band + sign; explained_variance band); **RandomProjection property-gate exception** (the one v2 estimator whose correctness gate is structurally not the 1e-5 value oracle); per-prim PoolStats memory gate.
**Research flag**: `[v2-P1]` incremental-SVD merge — settle "full Jacobi per batch vs dedicated rank-update kernel" and the f32-on-rocm stability of the stacked re-SVD. Run a research spike before planning.
**Plans**: 7 plans (4 waves)Plans:
**Wave 1**

- [x] 07-01-PLAN.md — Wave-0 scaffold: PartialFit trait + AlgoError guards + module index + 6 #[ignore] tests + 4 oracle generators

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 07-02-PLAN.md — PRIM-06 rng.rs (promote SplitMix64, Gaussian/Achlioptas/permutation) + PoolStats gate
- [x] 07-03-PLAN.md — PRIM-07 incremental_svd.rs (stacked re-SVD merge over v1 svd, ddof=1, svd_flip) + PoolStats gate
- [x] 07-04-PLAN.md — COV-01 EmpiricalCovariance (ddof=0, eig-pinvh precision_) + COV-02 LedoitWolf

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 07-05-PLAN.md — DECOMP-03 IncrementalPCA (PartialFit + sklearn-faithful fit + whiten + transform/inverse)
- [x] 07-06-PLAN.md — PROJ-01/02 Gaussian/SparseRandomProjection + johnson_lindenstrauss_min_dim (property-gated)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 07-07-PLAN.md — PyO3 wrappers for all 5 estimators + IncrementalPCA partial_fit + jl_min_dim pyfunction + Python shims

### Phase 8: Kernel Family

**Goal**: A data scientist can fit kernel-based regression and density estimators built on a new keystone kernel-matrix primitive (linear/RBF/poly/sigmoid) that Phase 9 and future kernel-SVM reuse; introduces the `ScoreSamples` trait.
**Depends on**: Phase 7 (shared crate seam; reuses v1 distance/GEMM/Cholesky prims)
**Requirements**: PRIM-08, KERNEL-01, KERNEL-02
**Success Criteria** (what must be TRUE):

  1. `prims/kernel_matrix.rs` (one small elementwise map over the v1 distance/Gram prims → linear/RBF/poly/sigmoid; NO SharedMemory, NO atomics) is validated standalone against a host reference within tolerance for f32 and f64, with its PoolStats memory gate; large `n×n` operands kept in global memory (gfx1100 LDS ≤ 65536 B).
  2. A user can fit `KernelRidge` (dual-coefficient solve of `(K + αI)` via the v1 Cholesky prim; kernels linear/rbf/polynomial/sigmoid with `gamma`/`degree`/`coef0`) and `predict`, matching scikit-learn within 1e-5.
  3. A user can fit `KernelDensity` (kernels + `bandwidth`) and call `score_samples` for log-density using a numerically-stable log-sum-exp, matching scikit-learn within tolerance.
  4. The `ScoreSamples<F>` trait is added next to the existing traits and `KernelDensity` implements it (length-`n` log-densities, not `Predict` semantics).

**Recurring gates**: `skip_f64_with_log` on every f64 oracle case; documented f32-on-rocm band for KernelRidge (predictions) and KernelDensity (log-density — large dynamic range); LDS-budget audit on any SharedMemory tile; per-prim PoolStats memory gate.
**Research flag**: None — kernel-matrix is a known elementwise map over distance/Gram; design settled in research. Standard pattern, research-phase can be skipped.
**Plans**: 5 plans (4 waves)

Plans:
**Wave 0**
- [x] 08-01-PLAN.md — Wave-0 scaffold: ScoreSamples<F> trait + 3 AlgoError guards + Kernel<F> enum/kernel_matrix signature + kernel_ridge//density/ module homes + 3 #[ignore] test scaffolds + 3 oracle generators

**Wave 1** *(blocked on Wave 0)*
- [x] 08-02-PLAN.md — PRIM-08 kernel_matrix.rs keystone prim (linear/rbf/poly/sigmoid map over v1 distance/gemm) + PoolStats memory gate [3/3 tasks; f64 ≤2.2e-16, f32 ≤2.4e-7 vs sklearn; memory gate green; wave gate satisfied]

**Wave 2** *(blocked on Wave 1)*
- [x] 08-03-PLAN.md — KERNEL-01 KernelRidge (dual (K+αI) Cholesky multi-RHS solve over kernel_matrix; no centering/intercept) [2/2 tasks; f64 ≤5.6e-16, f32 ≤3.6e-7 vs sklearn across 4 kernels + multi-target + both gamma paths]
- [x] 08-04-PLAN.md — KERNEL-02 KernelDensity (6 KD kernels + scott/silverman; device log-sum-exp over v1 distance/reduce; ScoreSamples<F> impl) [3/3 tasks; f64 ≤1.6e-8 (cosine series), other 5 kernels ≤1e-12, f32 ≤1e-4 vs sklearn forced-exact; Open Q1 resolved — plain reduce-sum, no rescale]

**Wave 3** *(blocked on Wave 2)*
- [x] 08-05-PLAN.md — PY-06 (share) PyO3 wrappers PyKernelRidge/PyKernelDensity (any_estimator! + score_samples) + py smoke test [2/2 tasks; zero new binding infra; both pyclasses registered in _mlrs; smoke test 4/4 green (f32+f64 × predict + score_samples) via maturin develop --release on cpu]

### Phase 9: Spectral Family

**Goal**: A data scientist can fit spectral embedding and clustering that cash in v1's hardest-won prim (`eig`) plus KMeans cheaply — the graph affinity *is* `kernel_matrix(Rbf)` from Phase 8, so the order is mandatory.
**Depends on**: **Phase 8 (HARD DEPENDENCY** — graph-Laplacian affinity is `kernel_matrix(Rbf)`); also reuses v1 `eig` and v1 KMeans
**Requirements**: PRIM-09, SPECTRAL-01, SPECTRAL-02
**Success Criteria** (what must be TRUE):

  1. `prims/laplacian.rs` (normalized Laplacian: affinity → single-owner row-reduction degree → `d_inv_sqrt` with a typed-zero guard, **NO `F::INFINITY`**, no edge-scatter) is validated standalone with no NaN/inf on zero-degree nodes, with its PoolStats memory gate.
  2. A user can fit `SpectralEmbedding` (affinity → normalized Laplacian → **smallest** non-trivial eigenvectors via v1 `eig`, sorted ascending, dropping the trivial ≈0 eigenvector, deterministic `_deterministic_vector_sign_flip` canonicalization) and get `embedding_` matching scikit-learn within tolerance after sign alignment (subspace test for degenerate spectra).
  3. A user can fit `SpectralClustering` (spectral embedding → v1 KMeans) and get `labels_` matching scikit-learn up to label permutation (sign-immune via `label_perm`).

**Recurring gates**: `skip_f64_with_log` on every f64 oracle case; documented f32-on-rocm band for SpectralEmbedding (embedding band + sign, or subspace test), **exact labels** the hard gate for SpectralClustering; LDS-budget audit on dense Laplacian; per-prim PoolStats memory gate.
**Research flag**: `[v2-P3]` smallest-eigenpair extraction — confirm full-spectrum-then-slice is acceptable at v2 sizes (vs Lanczos/shift-invert) and document the `n_samples` problem-size cap. Run a research spike before planning.
**Plans**: 4 plans (4 waves)

Plans:
**Wave 0**
- [x] 09-01-wave0-scaffold-PLAN.md — Wave-0 scaffold: AlgoError::NSamplesExceedsMaxDim (D-06) + laplacian prim/kernel stubs + 2 estimator homes + PyO3 spectral.rs stub + 5 #[ignore] test scaffolds + gen_spectral_embedding/clustering oracle generators (committed .npz, default constructors per D-01) — DONE 2026-06-21 (5c5e763, 2b1e4bd)

**Wave 1** *(blocked on Wave 0)*
- [ ] 09-02-laplacian-prim-PLAN.md — PRIM-09 laplacian.rs (zero-diag → row_reduce(Sum) degree GATHER → typed-zero dd guard → SharedMemory-free laplacian_map: L = I − D^-1/2 A D^-1/2) standalone-validated f32+f64, zero-degree no-NaN/inf, PoolStats memory gate

**Wave 2** *(blocked on Wave 1)*
- [ ] 09-03-spectral-embedding-PLAN.md — SPECTRAL-01 SpectralEmbedding (rbf + nearest_neighbors-default affinity → laplacian → eig reverse→ascending → /dd recovery → sign-flip → drop-trivial; gamma None→1/n_features D-04; degenerate subspace test D-09; reject n_samples>64 D-06)

**Wave 3** *(blocked on Wave 2)*
- [ ] 09-04-spectral-clustering-pyo3-PLAN.md — SPECTRAL-02 SpectralClustering (rbf default + drop_first=FALSE + n_components=n_clusters D-11 → KMeans::new exact labels up to perm on well-separated fixture D-10) + PyO3 PySpectralEmbedding/PySpectralClustering (any_estimator! ×2, GIL release, f64 guard) + smoke test

### Phase 10: SGD / Linear-SVM

**Goal**: A data scientist can fit minibatch-SGD and linear-SVM estimators built on the single genuinely-new device solver of v2 — the highest cpu-MLIR risk, validated standalone before any of the four estimators consume it.
**Depends on**: Phase 9 (shared crate seam; reuses v1 GEMM, reductions, host SplitMix64 shuffle; LinearSVC/SVR may reuse v1 CD for the converged optimum)
**Requirements**: PRIM-10, SGDSVM-01, SGDSVM-02, SGDSVM-03, SGDSVM-04
**Success Criteria** (what must be TRUE):

  1. `prims/sgd.rs` (hinge / log / squared / squared-hinge / epsilon-insensitive losses; l1/l2/elasticnet penalty; LR schedules incl. `optimal` + Bottou t0) is validated **standalone on a convex objective** before any estimator is wired, using the **two-pass GATHER kernel** (one thread per weight coordinate, ascending scan, F/u32 accumulators, no SharedMemory, no cross-unit atomics) and **passing the `--features cpu` launch** (not just compile), with its PoolStats memory gate.
  2. A user can fit `MBSGDClassifier` (hinge / log / squared-hinge; schedules incl. `optimal`) with `predict`/`predict_proba`, matching scikit-learn within tolerance under a **pinned deterministic oracle** (`shuffle=False`, fixed `eta0`/schedule, fixed `max_iter`, `tol=0`, sklearn NOT cuML).
  3. A user can fit `MBSGDRegressor` (squared-loss / epsilon-insensitive; `invscaling` default) with `predict`, matching scikit-learn within tolerance under the pinned deterministic oracle.
  4. A user can fit `LinearSVC` (`loss='squared_hinge'` default, `penalty`, `dual='auto'`, `intercept_scaling`) and `LinearSVR` (`loss='squared_epsilon_insensitive'` default, `epsilon`) with `predict`, matching scikit-learn within tolerance.

**Recurring gates**: `skip_f64_with_log` on every f64 oracle case; documented f32-on-rocm band for weights, **exact predicted labels** the hard gate for the classifiers; pinned-deterministic oracle (shuffle off, fixed schedule/iters, sklearn ref) per Pitfall 7; **GATHER idiom + cpu-launch verification** per Pitfall 1; per-prim PoolStats memory gate.
**Research flag**: `[v2-P4]` SGD under cpu-MLIR — spike the two-pass GATHER kernel and the pinned deterministic oracle before wiring any of the four estimators. Run a research spike before planning. **Highest-risk phase.**
**Plans**: TBD

### Phase 11: Naive Bayes

**Goal**: A data scientist can fit the five Naive Bayes variants — a wide-but-shallow, reductions-only closing bookend with the highest coverage per unit effort and five mutually-independent, parallel-buildable estimators (no new prim).
**Depends on**: Phase 10 (shared crate seam; reuses v1 reduce prim only)
**Requirements**: NB-01, NB-02, NB-03, NB-04, NB-05, PY-06
**Success Criteria** (what must be TRUE):

  1. A user can fit `GaussianNB` (per-class Gaussian likelihood with `var_smoothing` from global feature variance, ddof=0 population variance, log-sum-exp) with `predict`/`predict_proba` matching scikit-learn within tolerance; the per-class sufficient statistics use a **one-owner-per-(class,feature) GATHER kernel** (no scatter-add) that passes the `--features cpu` launch.
  2. A user can fit `MultinomialNB`, `BernoulliNB` (`(1−x)·log(1−p)` non-occurrence term, `binarize`), `ComplementNB` (complement weights, argmin decision), and `CategoricalNB` (per-feature categorical likelihood) with the correct per-variant `alpha` smoothing/denominator, matching scikit-learn within tolerance; sparse input densified at ingress for MultinomialNB.
  3. Every NB `predict_proba` row sums to 1 (computed via log-sum-exp, no underflow), with `predict` labels exact.
  4. **PY-06 (cross-cutting):** all v2 estimators are `#[pyclass]`-backed with sklearn-compatible `fit`/`predict`/`transform`/`score` (+ `partial_fit` for IncrementalPCA/MBSGD, `score_samples` for KernelDensity), `get_params`/`set_params` with sklearn-named hyperparameters, f32/f64 runtime dispatch, GIL release during compute, and ship inside the existing four per-backend wheels.

**Recurring gates**: `skip_f64_with_log` on every f64 oracle case; documented f32-on-rocm band for GaussianNB (log-proba), **exact labels** the hard gate for all five; **GATHER idiom + cpu-launch verification** per Pitfall 2; log-sum-exp + `var_smoothing` per Pitfall 9; PoolStats memory gate per estimator.
**Research flag**: None — Naive Bayes is reductions-only over the validated v1 reduce prim; per-variant math fully specified in FEATURES.md. Standard pattern, research-phase can be skipped.
**PY-06 placement decision**: PY-06 spans all v2 estimators. **Each phase wraps its own estimators incrementally** (reusing the shipped PyO3 `any_estimator!` machinery — v2 adds zero binding infrastructure), and PY-06 is formally assigned to Phase 11 as the final cross-cutting Python-surface sign-off (all v2 `#[pyclass]` estimators registered, dtype-suffixed accessors complete, the two new methods `partial_fit`/`score_samples` exposed, and `estimator_checks` re-triaged across the full v2 surface). This differs from v1's dedicated Phase-6 Python phase because v2 reuses the shipped binding layer rather than building it.
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation — Oracle, Backend Abstraction, Arrow Bridge | v1.0 | 5/5 | Complete | 2026-06-11 |
| 2. Core Compute Primitives | v1.0 | 5/5 | Complete | 2026-06-12 |
| 3. SVD / Eigendecomposition Primitive (Hard Gate) | v1.0 | 5/5 | Complete | 2026-06-12 |
| 4. Closed-Form Estimators | v1.0 | 5/5 | Complete | 2026-06-12 |
| 5. Distance-Based & Iterative-Solver Estimators | v1.0 | 11/11 | Complete | 2026-06-13 |
| 6. Python Surface — PyO3 Estimators & Per-Backend Wheels | v1.0 | 6/6 | Complete | 2026-06-14 |
| 7. Covariance & Projection | v2.0 | 7/7 | Complete    | 2026-06-20 |
| 8. Kernel Family | v2.0 | 1/5 | Executing | - |
| 9. Spectral Family | v2.0 | 0/? | Not started | - |
| 10. SGD / Linear-SVM | v2.0 | 0/? | Not started | - |
| 11. Naive Bayes | v2.0 | 0/? | Not started | - |
