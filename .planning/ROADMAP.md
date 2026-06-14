# Roadmap: mlrs — cuML in Rust

## Milestones

- ✅ **v1.0 Core ML Library** — Phases 1–6 (shipped 2026-06-14) → [archive](milestones/v1.0-ROADMAP.md)
- 📋 **v2.0 Breadth Sweep** — planned (run `/gsd-new-milestone`; seed: [v2-breadth-roadmap](seeds/v2-breadth-roadmap.md))

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

### 📋 v2.0 Breadth Sweep (Planned)

Defined via `/gsd-new-milestone`. Scope sketched in [seeds/v2-breadth-roadmap.md](seeds/v2-breadth-roadmap.md):
Covariance/projection → Kernel family → Spectral family → SGD/linear-SVM → Naive Bayes (+ stretch).

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation — Oracle, Backend Abstraction, Arrow Bridge | v1.0 | 5/5 | Complete | 2026-06-11 |
| 2. Core Compute Primitives | v1.0 | 5/5 | Complete | 2026-06-12 |
| 3. SVD / Eigendecomposition Primitive (Hard Gate) | v1.0 | 5/5 | Complete | 2026-06-12 |
| 4. Closed-Form Estimators | v1.0 | 5/5 | Complete | 2026-06-12 |
| 5. Distance-Based & Iterative-Solver Estimators | v1.0 | 11/11 | Complete | 2026-06-13 |
| 6. Python Surface — PyO3 Estimators & Per-Backend Wheels | v1.0 | 6/6 | Complete | 2026-06-14 |
