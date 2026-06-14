---
title: v2 milestone — breadth sweep of sklearn-core estimators
trigger_condition: when v1.0 milestone closes / v2 kickoff
planted_date: 2026-06-14
source: /gsd-explore "investigate unimplemented of mlrs comparing with cuml"
---

# v2 Milestone — Breadth Sweep (prioritized roadmap)

## Strategy

Maximize sklearn API coverage with **low-risk estimators that reuse v1's validated
primitives**, deferring the hard Tier-3 algorithms (RandomForest, UMAP, HDBSCAN, ARIMA,
kernel-SVM) to v3. Keeps v1's primitive-first discipline: each phase lands one reusable
primitive, then the estimators that consume it. Oracle stays **scikit-learn** (every
estimator below has a sklearn reference, so the 1e-5 gate holds). Backend gate stays
cpu(f64) + rocm(f32); f64-on-rocm skips-with-log.

## Phased roadmap (ordered)

| Phase | New shared primitive | Estimators | Leverages |
|---|---|---|---|
| **1. Covariance & projection** | RNG-matrix generator; incremental SVD update | EmpiricalCovariance, LedoitWolf, IncrementalPCA, GaussianRandomProjection, SparseRandomProjection | covariance, SVD |
| **2. Kernel family** | kernel-matrix prim (linear/RBF/poly/sigmoid) | KernelRidge, KernelDensity | distance, Cholesky |
| **3. Spectral family** | graph Laplacian prim | SpectralEmbedding, SpectralClustering | **eig**, KMeans |
| **4. SGD / linear-SVM** | SGD solver (hinge / log / squared / ε-insensitive losses) | MBSGDClassifier, MBSGDRegressor, LinearSVC, LinearSVR | reductions, GEMM |
| **5. Naive Bayes** | (none — reductions only) | GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB | reductions |
| **6. Stretch (defer-able to v3)** | per-algo | exact TSNE, AgglomerativeClustering, HoltWinters | distance, L-BFGS |

~16 estimators across 5 firm phases + 1 stretch.

## Ordering rationale

- **Phases 1–2** are pure assembly on existing prims → fastest confidence-builders; the
  kernel-matrix prim from Phase 2 is reused in later kernel work.
- **Phase 3** cashes in the hardest-won v1 primitive (`eig`) for two estimators cheaply
  (Laplacian → eig → KMeans).
- **Phase 4** is the one genuinely new solver (SGD); it unblocks four estimators at once.
- **Phase 5** is wide-but-shallow (reductions only) — high coverage per unit effort.

## Explicitly out of scope (→ v3)

RandomForest (→ FIL → TreeSHAP), UMAP, HDBSCAN, ARIMA/AutoARIMA, kernel SVC/SVR, and whole
subsystems (explainer/SHAP, genetic, cuml.accel, Dask multi-GPU). See
[[v3-hard-algorithm-backlog]].

## Pre-planning research required

See [[../research/questions.md]] — incremental SVD in CubeCL, SGD solver under the
cpu-MLIR no-SharedMemory/no-atomics constraint, and kernel-matrix prim design are the
genuine unknowns. Run a research pass before planning Phases 1, 2, and 4.

Related: [[cuml-mlrs-gap-inventory]]
