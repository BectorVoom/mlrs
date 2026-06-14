# Milestones

Historical record of shipped versions.

## v1.0 — Core ML Library

**Shipped:** 2026-06-14
**Phases:** 6 (1–6) | **Plans:** 38 | **Commits:** 233 | **Timeline:** 2026-06-11 → 2026-06-14 (3 days)
**Archive:** [v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md) · [v1.0-REQUIREMENTS.md](milestones/v1.0-REQUIREMENTS.md)

**Delivered:** A ground-up Rust rewrite of cuML's scikit-learn-core algorithm surface — 12 sklearn-compatible estimators on a single CubeCL codebase generic over `f32`/`f64` and runtime, validated against scikit-learn within 1e-5 on the cpu(f64) + rocm(f32) gate.

**Requirements:** 27/27 v1 requirements complete (0 unchecked, 27/27 mapped to phases).

### Key Accomplishments

1. **Foundation (Phase 1)** — Five-crate generic R/F workspace (`mlrs-core`/`-kernels`/`-backend`/`-algos`/`-py`), scikit-learn oracle harness (sign-flip + label-permutation helpers + documented f32 tolerance policy), Arrow zero-copy bridge with hard-reject validation (offset/null/alignment), f64 capability gate, mimalloc allocator.
2. **Core primitives (Phase 2)** — GEMM (cubek-matmul 0.2.0 wrap), dual-path reductions (plane + shared-memory fallback), pairwise squared-Euclidean distance with `max(d²,0)` clamp, covariance/Gram — all validated standalone with a D-10 build-failing memory gate (bounded reuse, GEMM-output-buffer reuse).
3. **SVD/eig hard gate (Phase 3)** — GPU one-sided Jacobi SVD + two-sided symmetric eigendecomposition written from scratch in CubeCL with the `svd_flip` sign convention; ROCm/HIP bring-up on gfx1100; D-11 memory gate (bounded Jacobi scratch, no mid-sweep read-back).
4. **Closed-form estimators (Phase 4)** — LinearRegression (SVD pseudo-inverse), Ridge (Cholesky normal-equations), PCA, TruncatedSVD matching scikit-learn within 1e-5; new Cholesky/triangular-solve primitive.
5. **Distance & iterative-solver estimators (Phase 5)** — KMeans (k-means++), DBSCAN, NearestNeighbors + KNeighborsClassifier/Regressor, Lasso, ElasticNet, LogisticRegression (L-BFGS over symmetric softmax); new top-k, coordinate-descent, and L-BFGS primitives.
6. **Python surface (Phase 6)** — 12 PyO3 `#[pyclass]` sklearn-compatible estimators with Arrow PyCapsule zero-copy ingest and GIL release during compute; four per-backend abi3-py312 wheels (`mlrs_cpu`/`wgpu`/`cuda`/`rocm`), each `import mlrs`; `estimator_checks` triage (475 passed / 102 by-design xfailed / 0 unexpected).

### Estimators Shipped (12)

LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, PCA, TruncatedSVD, KMeans, DBSCAN, NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor.

### Known Deferred Items (carried to v2+)

CUDA-hardware checks (live `import mlrs` from the cuda wheel, cross-hardware foreign-driver-absent ImportError, two-wheel-namespace-overwrite) — deferred opportunistically to a CUDA host (user-approved, recorded honestly, not fabricated). Phase-5 follow-ups: estimator-level empty-cluster KMeans fixture; deferred code-review items WR-05/06/07 + IN-01..05; no SECURITY.md for Phase 5. See STATE.md Deferred Items.
