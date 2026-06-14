# Retrospective — mlrs

Living retrospective across milestones.

## Milestone: v1.0 — Core ML Library

**Shipped:** 2026-06-14
**Phases:** 6 | **Plans:** 38 | **Commits:** 233 | **Timeline:** 3 days (2026-06-11 → 2026-06-14)

### What Was Built

A ground-up Rust rewrite of cuML's scikit-learn-core surface: 12 sklearn-compatible estimators on a single CubeCL codebase generic over `f32`/`f64` and runtime, validated against scikit-learn within 1e-5 on the cpu(f64) + rocm(f32) gate, shipped as four per-backend abi3 wheels with zero-copy Arrow ingest.

### What Worked

- **Primitive-first horizontal build order.** Validating GEMM/reduce/distance/SVD/eig/Cholesky standalone before any estimator meant estimators were "mostly assembly." The SVD/eig hard gate (Phase 3) unblocked 4 estimators; distance unblocked 3.
- **Build-failing memory gates per phase.** Encoding PoolStats assertions (bounded reuse, GEMM-output reuse, read-back counts) as compile/test gates kept memory efficiency from being deferred — it was provable at every phase.
- **scikit-learn as the oracle.** A CPU reference that runs without a GPU made the 1e-5 contract checkable in CI and decoupled correctness from hardware availability.
- **Committed .npz fixtures.** Oracle fixtures generated once (via a /tmp venv) and committed as blobs kept tests hermetic and fast.

### What Was Inefficient

- **cpu-MLIR kernel constraints discovered reactively.** Several kernels compiled but panicked at launch on the cpu backend (SharedMemory + mutable bool / F::INFINITY / descending-shift loops; cross-unit atomics). Each cost a rewrite to the GATHER/no-SharedMemory idiom. Knowing the constraint up front would have saved iterations.
- **Fortran-vs-C contiguity bug latency.** sklearn PCA `components_` is Fortran-order; the scaffold only asserted fixture *length*, so a transposed-read bug stayed latent until Phase 4.
- **f64-on-rocm reality surfaced mid-project.** The gate moved from wgpu+cpu to cpu+rocm at Phase 3 once it was confirmed cubecl-cpp 0.10 doesn't register F64 for HIP.

### Patterns Established

- GATHER kernels (single-owner outputs, F/u32 accumulators, if-guards, no SharedMemory/atomics) as the cpu-MLIR-safe default.
- Validate-geometry-before-unsafe-launch returning typed `PrimError`/`AlgoError` (ASVS V5).
- Estimator-side `svd_flip`/`align_rows` keeping primitives raw.
- f64 oracle cases capability-gated via `skip_f64_with_log` (skip, never fail).
- Two-threshold Jacobi convergence (rotation-skip vs noise-floor break with √pairs scaling).

### Key Lessons

1. Encode environment constraints (cpu-MLIR, f64-on-rocm) as explicit gates/notes early — they reshape kernel design more than algorithm choice does.
2. Scaffold tests that assert *shape and contiguity*, not just length — length-only checks hide layout bugs.
3. For gauge-redundant objectives (symmetric softmax LogReg), pick the correctness witness deliberately (gauge-fixed predict_proba), and document the divergence from sklearn's native parameterization.

### Cost Observations

- Sessions: multi-session over 3 days; model mix not instrumented.
- Notable: primitive-first ordering front-loaded the hard work (Phase 3 SVD/eig) so later estimator phases were fast and low-risk.

## Cross-Milestone Trends

| Milestone | Phases | Plans | Days | Estimators |
|-----------|--------|-------|------|------------|
| v1.0 | 6 | 38 | 3 | 12 |
