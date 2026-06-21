---
phase: 09-spectral-family
plan: 04
subsystem: ml-algorithms
tags: [spectral-clustering, kmeans, laplacian, eig, pyo3, sklearn-oracle, cubecl]

# Dependency graph
requires:
  - phase: 09-03
    provides: SpectralEmbedding.fit (full _spectral_embedding pipeline) + the /dd recovery host math + the kNN-connectivity affinity builder
  - phase: 09-02
    provides: validated `laplacian` prim (normalized Laplacian + dd degree vector)
  - phase: 09-01
    provides: SpectralClustering struct + new(), both PyO3 any_estimator! stubs registered on _mlrs, the #[ignore] algo + smoke test scaffolds
  - phase: 08
    provides: kernel_matrix(Rbf) affinity prim, any_estimator! macro, py.detach + guard_f64 dispatch idiom
  - phase: 05
    provides: v1 KMeans (KMeans::new, kmeans++, n_init=1) reused unchanged on the embedding
provides:
  - SpectralClustering.fit (rbf affinity → laplacian → eig → /dd recovery drop_first=FALSE → KMeans::new → labels_)
  - SpectralClustering labels_ / fit_predict accessors over v1 KMeans
  - exact-label-up-to-permutation oracle gate (best_match_accuracy==1.0) for f32+f64 on the well-separated fixture
  - both spectral estimators wired into the _mlrs Python surface (device fit/accessor smoke green for f32+f64)
affects: [phase-11-python-signoff, spectral-family, clustering]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SpectralClustering = SpectralEmbedding-recovery (drop_first=FALSE) → v1 KMeans::new; no new trait, composes Fit"
    - "drop_first=FALSE recovery variant: keep ALL n_components eigenvectors incl. the trivial ≈0 one as the KMeans `maps`"
    - "exact-label gate via a WELL-SEPARATED fixture (D-10) makes the init RNG gap immaterial — no init-injection"
    - "device fit/accessor smoke through the algos estimators the py.detach fit shells delegate to (cargo-level, 08-05 precedent)"

key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/cluster/spectral_clustering.rs
    - crates/mlrs-algos/tests/spectral_clustering_test.rs
    - crates/mlrs-py/tests/spectral_smoke_test.rs
    - crates/mlrs-py/Cargo.toml

key-decisions:
  - "drop_first=FALSE for SpectralClustering: the `maps` is n × n_components INCLUDING the trivial eigenvector (row 0), pinned to sklearn's k_means input (D-11)"
  - "n_components defaults to n_clusters (D-11); gamma is the literal 1.0 default (D-04, NOT 1/n_features)"
  - "KMeans::new (kmeans++, n_init=1) — NOT with_init; the well-separated fixture (D-10) makes the SplitMix64-vs-MT19937 init gap immaterial to the labels"
  - "validate-before-launch: n_samples>64 → NSamplesExceedsMaxDim, n_clusters via InvalidK, gamma via InvalidGamma, all BEFORE any device work (D-06)"
  - "PyO3 wrapper bodies + accessors were already complete in the Wave-0 stub; this plan only filled the algos fit they delegate to + un-ignored the smoke; pyo3 stays 0.28, zero new binding infra"
  - "added cubecl + env_logger dev-deps to mlrs-py for the generic device-fit smoke trait bounds (Rule 3 blocking)"

patterns-established:
  - "Pattern: a recover_maps host helper mirroring recover_embedding but with drop_first=FALSE (keep row 0) for the clustering path"
  - "Pattern: cargo-level device smoke for PyO3 estimators drives the algos fit directly (the py.detach shell delegate), f64 gated by skip_f64_with_log"

requirements-completed: [SPECTRAL-02, PRIM-09, SPECTRAL-01]

# Metrics
duration: 5min
completed: 2026-06-21
---

# Phase 9 Plan 04: SpectralClustering + PyO3 Wrapping Summary

**SpectralClustering.fit (rbf → laplacian → eig → /dd recovery with drop_first=FALSE → v1 KMeans::new) landing labels_ that match sklearn EXACTLY up to permutation on the well-separated fixture, plus the live device fit/accessor smoke for both spectral estimators in the _mlrs Python surface.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-06-21T03:20:34Z
- **Completed:** 2026-06-21T03:25:35Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Filled `SpectralClustering.fit`: validate-before-launch → rbf affinity (gamma=1.0 literal, D-04) → `laplacian` → v1 `eig` → `/dd` recovery with `drop_first=FALSE` (keep the trivial eigenvector, D-11) → `KMeans::new(n_clusters, seed)` (kmeans++, n_init=1, D-10) → `labels_`; plus `labels`/`fit_predict` accessors.
- Un-ignored `spectral_clustering_test`: `labels_` exact up to permutation (`best_match_accuracy == 1.0`) on the well-separated fixture for **both f32 and f64**, plus a live `reject_oversize` (n=65 → `NSamplesExceedsMaxDim` pre-launch).
- Un-ignored the PyO3 `spectral_fit_accessors` smoke: drives `SpectralEmbedding.fit → embedding_` and `SpectralClustering.fit → labels_` on a live device for f32 always + f64 gated by `skip_f64_with_log`, asserting shape, finiteness, and a clean 2-cluster split.
- The PyO3 `any_estimator!` wrappers (both estimators, `guard_f64` on the F64 arm before upload, dtype-suffixed accessors) were already complete from the Wave-0 stub and now compile+run against the filled algos bodies; pyo3 stays 0.28, zero new binding infra.

## Task Commits

1. **Task 1: SpectralClustering.fit + label_perm test** - `9e04a75` (feat)
2. **Task 2: un-ignore spectral PyO3 device fit/accessor smoke** - `c5befb0` (feat)
3. **Cargo.lock sync for dev-deps** - `ca23537` (chore)

_Task 1 carries TDD intent (the failing test was the Wave-0 `#[ignore]` scaffold un-ignored here); the green test + impl landed in a single commit since the test file already existed as a compiling scaffold._

## Files Created/Modified
- `crates/mlrs-algos/src/cluster/spectral_clustering.rs` - filled `fit` (affinity → laplacian → eig → recover_maps drop_first=FALSE → KMeans), `labels`/`fit_predict` accessors, `knn_connectivity_affinity`, and the `recover_maps` host helper
- `crates/mlrs-algos/tests/spectral_clustering_test.rs` - un-ignored; live fit + `best_match_accuracy==1.0` (f32+f64) + `reject_oversize`
- `crates/mlrs-py/tests/spectral_smoke_test.rs` - un-ignored `spectral_fit_accessors`; device fit→`embedding_`/`labels_` smoke (f32+f64-gated)
- `crates/mlrs-py/Cargo.toml` - added `cubecl` + `env_logger` dev-deps for the generic smoke trait bounds

## Decisions Made
- **drop_first=FALSE** for SC: `recover_maps` keeps every one of the `n_components` smallest eigenvectors (including the trivial ≈0 row 0) as the KMeans `maps`, pinned to sklearn's `k_means` input (D-11). This is the only structural difference from `SpectralEmbedding`'s `recover_embedding` (which drops row 0).
- **n_components = n_components.unwrap_or(n_clusters)** and **gamma literal 1.0** (D-11 / D-04), matching sklearn's `SpectralClustering` defaults — NOT the `1/n_features` gamma of `SpectralEmbedding`.
- **KMeans::new, not with_init** (D-10): the well-separated fixture makes the partition unique up to permutation, so the init RNG gap between SplitMix64 and sklearn's MT19937 is immaterial; init-injection is rejected for SC.
- The `recover_maps` helper was written locally in `spectral_clustering.rs` (rather than factoring `recover_embedding` out of `spectral_embedding.rs`) because the two differ only by `drop_first` and the embedding helper is private + file-disjoint from this plan's surface; replicating the pinned order keeps the two files independent.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added cubecl + env_logger dev-deps to mlrs-py**
- **Found during:** Task 2 (PyO3 smoke test)
- **Issue:** The generic device-fit smoke needs the `Float`/`CubeElement` trait bounds (from `cubecl`) and test-scoped `env_logger`; neither was a dev-dependency of `mlrs-py`, so the test crate failed to compile (E0433 unresolved `cubecl`/`env_logger`).
- **Fix:** Added `cubecl = { workspace = true }` (default-features=false) and `env_logger = { workspace = true }` to `[dev-dependencies]`; the backend feature still arrives through the `mlrs-backend` dep chain.
- **Files modified:** crates/mlrs-py/Cargo.toml (+ Cargo.lock sync in `ca23537`)
- **Verification:** `cargo test -p mlrs-py --features cpu --test spectral_smoke_test` green (2 passed)
- **Committed in:** c5befb0 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The dev-dep addition is dev/test-only (no wheel impact, no new runtime binding infra); it was the minimal change to run the requested f32+f64 device smoke at the Rust test level per the 08-05 precedent. No scope creep.

## Issues Encountered
- `cargo build -p mlrs-py` with **no backend feature** fails inside `mlrs-backend` (the documented rust-analyzer/no-feature false-positive seam from memory). Verified the build + tests with `--features cpu` (the actual correctness gate), which compiles cleanly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- SPECTRAL-02 label gate is green (exact up to permutation, f32+f64); the full Phase-9 Python surface (`SpectralEmbedding` + `SpectralClustering`) is wired and device-smoke-verified.
- PY-06 final sign-off (the end-to-end pytest interpreter+capsule path) remains a Phase-11 task per the plan's incremental-wrap scope.

---
*Phase: 09-spectral-family*
*Completed: 2026-06-21*

## Self-Check: PASSED

- All modified files exist on disk (spectral_clustering.rs, spectral_clustering_test.rs, spectral_smoke_test.rs, 09-04-SUMMARY.md).
- All commits present in git history: `9e04a75` (Task 1), `c5befb0` (Task 2), `ca23537` (Cargo.lock sync).
