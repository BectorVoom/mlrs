# Roadmap: mlrs — cuML in Rust

## Milestones

- ✅ **v1.0 Core ML Library** — Phases 1–6 (shipped 2026-06-14) → [archive](milestones/v1.0-ROADMAP.md)
- ✅ **v2.0 Breadth Sweep** — Phases 7–11 (shipped 2026-06-22) → [archive](milestones/v2.0-ROADMAP.md)
- ✅ **v3.0 Manifold Algorithms & Rust-Native API** — Phases 12–16 (shipped 2026-06-26) → [archive](milestones/v3.0-ROADMAP.md)
- 📋 **v4.0 (next)** — TBD; start via `/gsd-new-milestone` (candidate scope: Tier-3 backlog — RandomForest→FIL→TreeSHAP, ARIMA, kernel-SVM/SMO — see `notes/v3-hard-algorithm-backlog.md`)

## Overview

All three shipped milestones grew one sklearn-compatible ML library on a single CubeCL-generic codebase (cpu f64 + rocm f32 gate, scikit-learn ≤1e-5 oracle): v1.0 stood up the foundation + 12 estimators, v2.0 swept 18 more across five families, and v3.0 added the UMAP + HDBSCAN manifold/clustering pair on a shared KNN-graph primitive plus a Rust-native builder/typestate API and pure-Python sklearn shim. Full per-phase detail for each shipped milestone lives in its archive (linked above). The next milestone's phases will be defined via `/gsd-new-milestone`.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)
- Phase numbering is continuous across milestones (never restarts); the next milestone continues from Phase 17.

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

<details>
<summary>✅ v2.0 Breadth Sweep (Phases 7–11) — SHIPPED 2026-06-22 — 27 plans, 18 estimators</summary>

~18 sklearn-compatible estimators across five families, built as assembly on v1's validated primitive base plus five new feature-free CubeCL primitives (RNG-matrix, incremental-SVD, kernel-matrix, graph-Laplacian, SGD solver). No new compute dependency. Oracle = scikit-learn ≤ 1e-5; gate = cpu(f64) + rocm(f32), f64-on-rocm skips-with-log.

- [x] Phase 7: Covariance & Projection — RNG-matrix + incremental-SVD prims, PartialFit trait; EmpiricalCovariance, LedoitWolf, IncrementalPCA, Gaussian/SparseRandomProjection (7/7 plans) — completed 2026-06-20
- [x] Phase 8: Kernel Family — kernel-matrix prim (linear/RBF/poly/sigmoid), ScoreSamples trait; KernelRidge, KernelDensity (5/5 plans) — completed 2026-06-21
- [x] Phase 9: Spectral Family — graph-Laplacian prim (hard dep on Phase 8); SpectralEmbedding, SpectralClustering (4/4 plans) — completed 2026-06-21
- [x] Phase 10: SGD / Linear-SVM — SGD solver prim (two-pass GATHER, cpu-MLIR-safe); MBSGDClassifier, MBSGDRegressor, LinearSVC, LinearSVR (6/6 plans) — completed 2026-06-21
- [x] Phase 11: Naive Bayes — reductions-only closing bookend; GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB + PY-06 final PyO3 sign-off (5/5 plans) — completed 2026-06-22

Full phase detail, plans, and per-plan notes: [milestones/v2.0-ROADMAP.md](milestones/v2.0-ROADMAP.md)

</details>

<details>
<summary>✅ v3.0 Manifold Algorithms & Rust-Native API (Phases 12–16) — SHIPPED 2026-06-26 — 34 plans, UMAP + HDBSCAN + builder/typestate retrofit</summary>

UMAP + HDBSCAN on a single shared, multi-metric KNN-graph primitive (primitive-first, standalone-gated before either consumer), plus a Rust-native builder/typestate API additively retrofitted across the full 32-estimator surface and a pure-Python sklearn shim. Zero new compute dependencies. Oracle broadened to umap-learn 0.5.12 (property gate for UMAP's stochastic SGD layout; ≤1e-5 value gates for the deterministic stages); HDBSCAN keeps an exact-label hard gate. Same gate as v1/v2 (cpu f64 + rocm f32, f64-on-rocm skips-with-log).

- [x] Phase 12: Builder + Typestate Convention Foundation (4/4 plans) — completed 2026-06-23
- [x] Phase 13: KNN-Graph Primitive (feasibility keystone) (3/3 plans) — completed 2026-06-23
- [x] Phase 14: UMAP (7/7 plans) — completed 2026-06-24
- [x] Phase 15: HDBSCAN (7/7 plans) — completed 2026-06-24
- [x] Phase 16: Builder Retrofit Sweep + Shim Coverage (13/13 plans) — completed 2026-06-24

Full phase detail, plans, and per-plan notes: [milestones/v3.0-ROADMAP.md](milestones/v3.0-ROADMAP.md)

</details>

### 📋 v4.0 (Next — Planned)

To be defined via `/gsd-new-milestone` (questioning → research → requirements → roadmap). Phase numbering continues from Phase 17.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation — Oracle, Backend Abstraction, Arrow Bridge | v1.0 | 5/5 | Complete | 2026-06-11 |
| 2. Core Compute Primitives | v1.0 | 5/5 | Complete | 2026-06-12 |
| 3. SVD / Eigendecomposition Primitive (Hard Gate) | v1.0 | 5/5 | Complete | 2026-06-12 |
| 4. Closed-Form Estimators | v1.0 | 5/5 | Complete | 2026-06-12 |
| 5. Distance-Based & Iterative-Solver Estimators | v1.0 | 11/11 | Complete | 2026-06-13 |
| 6. Python Surface — PyO3 Estimators & Per-Backend Wheels | v1.0 | 6/6 | Complete | 2026-06-14 |
| 7. Covariance & Projection | v2.0 | 7/7 | Complete | 2026-06-20 |
| 8. Kernel Family | v2.0 | 5/5 | Complete | 2026-06-21 |
| 9. Spectral Family | v2.0 | 4/4 | Complete | 2026-06-21 |
| 10. SGD / Linear-SVM | v2.0 | 6/6 | Complete | 2026-06-21 |
| 11. Naive Bayes | v2.0 | 5/5 | Complete | 2026-06-22 |
| 12. Builder + Typestate Convention Foundation | v3.0 | 4/4 | Complete | 2026-06-23 |
| 13. KNN-Graph Primitive (feasibility keystone) | v3.0 | 3/3 | Complete | 2026-06-23 |
| 14. UMAP | v3.0 | 7/7 | Complete | 2026-06-24 |
| 15. HDBSCAN | v3.0 | 7/7 | Complete | 2026-06-24 |
| 16. Builder Retrofit Sweep + Shim Coverage | v3.0 | 13/13 | Complete | 2026-06-24 |
