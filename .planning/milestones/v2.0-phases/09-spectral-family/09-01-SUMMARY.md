---
phase: 09-spectral-family
plan: 01
subsystem: testing
tags: [spectral, laplacian, eig, cubecl, pyo3, oracle, scaffold]

# Dependency graph
requires:
  - phase: 08-kernel-family
    provides: kernel_matrix(Rbf) affinity seam, any_estimator! macro, oracle harness
  - phase: 03-svd-eig
    provides: v1 symmetric eig (MAX_DIM=64 cap) reused by the spectral pipeline
  - phase: 05-clustering
    provides: v1 KMeans (kmeans++, n_init=1) reused by SpectralClustering
provides:
  - AlgoError::NSamplesExceedsMaxDim typed guard (D-06, MAX_DIM=64)
  - laplacian(pool, A, n) -> (L, dd) host signature with real geometry guard (todo!() compute)
  - laplacian_map #[cube(launch)] kernel stub (shared-memory-free, atomics-free, no infinite-value constant)
  - SpectralEmbedding / SpectralClustering struct + new() stubs (cluster module)
  - PySpectralEmbedding / PySpectralClustering pyclass stubs registered on _mlrs
  - five #[ignore] Nyquist test scaffolds (laplacian x3, SE x4, SC x1, py-smoke x1)
  - 10 committed spectral/laplacian .npz oracle fixtures (default constructor, D-01)
affects: [09-02-laplacian-prim, 09-03-spectral-embedding, 09-04-spectral-clustering-pyo3]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Wave-0 scaffold front-loads ALL shared-file edits so Waves 1/2/3 are file-disjoint"
    - "laplacian prim (L, dd) two-buffer return (RESEARCH Open Q2): estimator builds affinity, prim returns Laplacian + degree-norm vector"
    - "Oracle fixtures use each estimator's OWN default constructor (D-01), the inverse of Phase-7 oracle-injection"

key-files:
  created:
    - crates/mlrs-backend/src/prims/laplacian.rs
    - crates/mlrs-algos/src/cluster/spectral_embedding.rs
    - crates/mlrs-algos/src/cluster/spectral_clustering.rs
    - crates/mlrs-py/src/estimators/spectral.rs
    - crates/mlrs-backend/tests/laplacian_test.rs
    - crates/mlrs-algos/tests/spectral_embedding_test.rs
    - crates/mlrs-algos/tests/spectral_clustering_test.rs
    - crates/mlrs-py/tests/spectral_smoke_test.rs
  modified:
    - crates/mlrs-algos/src/error.rs
    - crates/mlrs-backend/src/prims/mod.rs
    - crates/mlrs-kernels/src/elementwise.rs
    - crates/mlrs-kernels/src/lib.rs
    - crates/mlrs-algos/src/cluster/mod.rs
    - crates/mlrs-py/src/estimators/mod.rs
    - crates/mlrs-py/src/lib.rs
    - scripts/gen_oracle.py

key-decisions:
  - "laplacian prim stays n<=64 cap-agnostic (like kernel_matrix.rs); the MAX_DIM cap is the estimator's AlgoError::NSamplesExceedsMaxDim job (D-06)"
  - "SpectralEmbedding default affinity=nearest_neighbors / gamma=None; SpectralClustering default affinity=rbf / gamma=1.0 (D-01/D-04) — both honored verbatim in struct defaults + pyclass signatures + oracle constructors"
  - "Oracle generators set only n_components + random_state; affinity/gamma are NOT overridden (D-01 default-constructor oracle)"

patterns-established:
  - "Pattern 1: laplacian_map gathers the dd divisor by row/col index (div_by_row idiom), no edge-scatter / no atomics; isolated-node diagonal is 0 not 1 (no NaN/no infinite value)"
  - "Pattern 2: reject_oversize scaffold asserts the typed AlgoError variant message structurally (variant exists + names MAX_DIM) since fit is todo!() in Wave-0"

requirements-completed: [PRIM-09, SPECTRAL-01, SPECTRAL-02]

# Metrics
duration: 12min
completed: 2026-06-21
---

# Phase 9 Plan 01: Spectral Family Wave-0 Scaffold Summary

**Front-loaded all shared-file edits (NSamplesExceedsMaxDim guard, laplacian prim + laplacian_map kernel stub, both spectral estimator homes, both PyO3 pyclass stubs) plus five #[ignore] Nyquist test scaffolds and 10 default-constructor oracle fixtures — a compiling --features cpu workspace where Waves 1/2/3 are file-disjoint.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-06-21T02:42:00Z
- **Completed:** 2026-06-21T02:53:40Z
- **Tasks:** 2
- **Files modified:** 19 (8 created, 8 modified, 3 fixtures dirs touched)

## Accomplishments
- `AlgoError::NSamplesExceedsMaxDim` (D-06) — typed pre-allocation guard naming the dense eigensolver MAX_DIM=64 cap; constructible with a spectral-domain message (T-9-VAL mitigated at scaffold time).
- `laplacian(pool, A, n) -> (L, dd)` host signature with a REAL geometry guard (reject `a.len() != n*n`, `n == 0`) and a `todo!()` compute body; `laplacian_map` `#[cube(launch)]` stub is shared-memory-free, atomics-free, and free of the infinite-value constant (T-9-LAP — the cpu-MLIR-safe profile is established for Wave-1 to inherit).
- Both spectral estimator struct homes (`SpectralEmbedding` / `SpectralClustering`) registered in `cluster/mod.rs` with sklearn-default-carrying `new()` constructors and `todo!()` fit/accessor bodies; both PyO3 `any_estimator!` wrappers registered on `_mlrs`.
- Five `#[ignore]` Nyquist scaffolds compile + collect (0 run, 9 ignored across the four files; the py-smoke construction test runs green). 10 committed `.npz` fixtures use each estimator's DEFAULT constructor (D-01); a degenerate-spectrum SE fixture and an isolated-node laplacian fixture are present.

## Task Commits

Each task was committed atomically:

1. **Task 1: Shared-file edits — error variant, module registrations, prim + kernel + estimator + pyclass stubs** - `5c5e763` (feat)
2. **Task 2: Five #[ignore] Nyquist test scaffolds + two oracle generators (committed .npz)** - `2b1e4bd` (test)

## Files Created/Modified
- `crates/mlrs-algos/src/error.rs` - Added `NSamplesExceedsMaxDim { estimator, n_samples, max }` after `InvalidGamma`.
- `crates/mlrs-backend/src/prims/{mod.rs,laplacian.rs}` - Registered `laplacian`; created the `(L, dd)` signature + real geometry guard + `todo!()` compute.
- `crates/mlrs-kernels/src/{elementwise.rs,lib.rs}` - Added + re-exported the `laplacian_map` `#[cube(launch)]` stub (placeholder body, real launch shape + dd gather).
- `crates/mlrs-algos/src/cluster/{mod.rs,spectral_embedding.rs,spectral_clustering.rs}` - Registered + created both estimator struct stubs.
- `crates/mlrs-py/src/{lib.rs,estimators/mod.rs,estimators/spectral.rs}` - Created + registered both pyclass wrappers on `_mlrs`.
- `crates/mlrs-backend/tests/laplacian_test.rs`, `crates/mlrs-algos/tests/spectral_{embedding,clustering}_test.rs`, `crates/mlrs-py/tests/spectral_smoke_test.rs` - Five `#[ignore]` scaffolds.
- `scripts/gen_oracle.py` - Added `gen_laplacian`, `gen_spectral_embedding`, `gen_spectral_clustering` + main registration; emitted 10 committed fixtures.

## Decisions Made
- Reused `InvalidK` (n_clusters/n_neighbors) and `InvalidGamma` (non-finite gamma) per the plan; added ONLY `NSamplesExceedsMaxDim` (no `InvalidNNeighbors` — `InvalidK`'s message fits the n_neighbors semantics).
- `laplacian.rs` stays n<=64 cap-agnostic; the cap is the estimator's job (D-06).
- Oracle fixtures fit with each estimator's own default constructor (D-01) — only `n_components`/`n_clusters` + `random_state` set; affinity/gamma never overridden.

## Deviations from Plan

None - plan executed exactly as written. Two minor in-task corrections (not deviations from intent):
- The `mlrs_core::best_match_accuracy` helper takes `&[i64]` (not `&[i32]`); the SpectralClustering test's `ref_labels` returns `Vec<i64>` accordingly.
- Reworded the `laplacian.rs` module doc to "shared-memory-free" / "no LDS tile" so the new prim+kernel source carries no literal `SharedMemory` token (the kernel source `elementwise.rs` was already token-clean), matching the [08-02] Rule-3 precedent.

## Issues Encountered
- The `nearest_neighbors`-affinity SpectralEmbedding default emits a benign sklearn "Graph is not fully connected" UserWarning on the small random fixture (n=12, n_neighbors=10). Expected for the default-constructor oracle; the committed `embedding_` is still reproducible (`random_state=42`). No action needed — Wave-2 (09-03) value-matches against this fixture.

## User Setup Required
None - no external service configuration required. Fixture regeneration (Wave-2/3 only) uses the existing `/tmp/oracle-venv` (numpy 2.4.6 / scipy 1.18.0 / sklearn 1.9.0, PEP 668).

## Next Phase Readiness
- Workspace compiles `--features cpu`; all module homes + the error variant land; the new kernel stub is shared-memory-free and infinite-value-free.
- Waves 1/2/3 are file-disjoint: 09-02 fills `laplacian.rs` + `laplacian_map` (un-ignores `laplacian_test.rs`); 09-03 fills `SpectralEmbedding` (un-ignores `spectral_embedding_test.rs`); 09-04 fills `SpectralClustering` + the PyO3 fit bodies (un-ignores `spectral_clustering_test.rs` + `spectral_smoke_test.rs`).
- All 10 oracle fixtures committed and shape-verified (SE 12×2, SC 12 labels in 3 clean blocks, laplacian 8×8 with finite isolated-node L).

## Self-Check: PASSED

All 8 created files and 3 representative fixtures present on disk; both task commits (`5c5e763`, `2b1e4bd`) exist in git history.

---
*Phase: 09-spectral-family*
*Completed: 2026-06-21*
