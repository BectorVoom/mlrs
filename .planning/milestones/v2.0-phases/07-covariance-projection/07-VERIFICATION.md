---
phase: 07-covariance-projection
verified: 2026-06-20T00:00:00Z
status: passed
score: 7/7
overrides_applied: 0
---

# Phase 7: Covariance & Projection Verification Report

**Phase Goal:** A data scientist can fit covariance estimators and projection transformers that reuse v1's covariance + SVD prims — the lowest-risk opener that lands the reusable host RNG-matrix primitive and the incremental-SVD merge, and introduces the `PartialFit` trait.
**Verified:** 2026-06-20
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `prims/rng.rs` generates Gaussian + Achlioptas-sparse matrices + permutations, validated for distribution stats and seed-reproducibility (same seed → identical matrix), with PoolStats memory gate | VERIFIED | `rng.rs` 296 lines with `SplitMix64` (pub), `gaussian_matrix`, `sparse_achlioptas_matrix`, `permutation`; 5/5 rng tests pass including `rng_seed_reproducible` (byte-identical) and `rng_memory_gate`; no OsRng; no biased modulo (3 occurrences in doc-comments only) |
| 2 | `prims/incremental_svd.rs` merges a running decomposition with a new batch (mean-correction row, svd_flip u_based_decision=False, ddof=1), validated against a 2+ batch host reference | VERIFIED | `incremental_svd.rs` 349 lines with `align_rows` (4 hits), `MAX_ROWS`/`MAX_COLS` cap validation (4 hits), `svd::` composition (3 hits), zero `#[cube]`/`SharedMemory`; 3/3 tests pass: two_batch_merge (f64 1e-5), two_batch_merge_f32 (1e-4 band), memory_gate |
| 3 | A user can fit `EmpiricalCovariance` (ddof=0 MLE) and `LedoitWolf`, getting covariance_/location_/precision_ and shrinkage_ (clipped [0,1]) matching scikit-learn within 1e-5 — both gated across two n | VERIFIED | `empirical_covariance.rs` 301 lines with `eig::` (2 hits), `covariance::` (2 hits); `ledoit_wolf.rs` 283 lines with `shrinkage.clamp(0.0, 1.0)`; 4/4 empirical_covariance tests pass (full-rank + rank-deficient, f32+f64); 4/4 ledoit_wolf tests pass (n=12, n=40, f32+f64) |
| 4 | A user can fit `IncrementalPCA` via partial_fit over batches and get all attributes + transform/inverse_transform matching scikit-learn within 1e-5 after svd_flip sign alignment | VERIFIED | `incremental_pca.rs` 513 lines with `impl.*PartialFit` (1 hit), `incremental_svd` (3 hits), `whiten` (25 hits); `n_samples_seen` accumulation test (0→10→20→30); 14/14 tests pass including partial_fit, fit, whiten on/off, transform, inverse_transform (f32 @ 1e-4 band, f64 @ 1e-5) |
| 5 | A user can fit `GaussianRandomProjection` and `SparseRandomProjection` (n_components='auto' via johnson_lindenstrauss_min_dim) and transform — property-gated (JL distortion, distribution stats, seed-reproducibility, transform == X·componentsᵀ), johnson_lindenstrauss_min_dim value-matched; sparse input densified at Python ingress | VERIFIED | `gaussian.rs` 283 lines with `johnson_lindenstrauss_min_dim` (10 hits), `rng::` (2 hits); `sparse.rs` 214 lines; 8/8 tests pass: jl_min_dim (integer-exact oracle), moments, self-consistency, JL distortion averaged ≈1 (JL_TRIALS=50), seed-repro; `_densify` in random_projection.py calls `X.toarray()` on `issparse(X)` |
| 6 | The `PartialFit` trait was introduced and re-exported from lib.rs | VERIFIED | `trait PartialFit` in traits.rs (1 hit); `pub use traits::{Fit, KNeighbors, PartialFit, …}` in lib.rs (line 51) |
| 7 | Five estimators exposed as PyO3 #[pyclass] objects with Python shims | VERIFIED | `covariance.rs` (PyEmpiricalCovariance + PyLedoitWolf), `projection.rs` (PyGaussianRandomProjection + PySparseRandomProjection + johnson_lindenstrauss_min_dim pyfunction), `decomposition.rs` extended with PyIncrementalPCA + partial_fit; covariance.py + random_projection.py + decomposition.py Python shims present; 3/3 smoke tests pass including `five_phase7_estimators_construct_unfit` |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-backend/src/prims/rng.rs` | PRIM-06 host RNG-matrix primitive with SplitMix64 | VERIFIED | 296 lines, substantive, wired via `use` from kmeans.rs (3 hits) and called from projection/sparse.rs |
| `crates/mlrs-backend/src/prims/incremental_svd.rs` | PRIM-07 incremental-SVD merge | VERIFIED | 349 lines, `align_rows`, `svd::`, MAX_ROWS/MAX_COLS, no new kernel |
| `crates/mlrs-algos/src/covariance/empirical_covariance.rs` | EmpiricalCovariance (COV-01) | VERIFIED | 301 lines, eig-based pinvh, ddof=0 |
| `crates/mlrs-algos/src/covariance/ledoit_wolf.rs` | LedoitWolf (COV-02) | VERIFIED | 283 lines, β/δ/μ closed form, shrinkage clipped [0,1] |
| `crates/mlrs-algos/src/decomposition/incremental_pca.rs` | IncrementalPCA (DECOMP-03) | VERIFIED | 513 lines, PartialFit + Fit + Transform + inverse_transform |
| `crates/mlrs-algos/src/projection/gaussian.rs` | GaussianRandomProjection (PROJ-01) | VERIFIED | 283 lines, johnson_lindenstrauss_min_dim, rng::gaussian_matrix |
| `crates/mlrs-algos/src/projection/sparse.rs` | SparseRandomProjection (PROJ-02) | VERIFIED | 214 lines, rng::sparse_achlioptas_matrix, density |
| `crates/mlrs-algos/src/traits.rs` | PartialFit<F> trait (D-01) | VERIFIED | `trait PartialFit` present, re-exported from lib.rs |
| `crates/mlrs-py/src/estimators/covariance.rs` | PyEmpiricalCovariance + PyLedoitWolf #[pyclass] | VERIFIED | Exists, guard_f64 =3, dtype-suffixed accessors |
| `crates/mlrs-py/src/estimators/projection.rs` | PyGaussianRP + PySparseRP + jl_min_dim #[pyfunction] | VERIFIED | Exists, guard_f64 =3 |
| `crates/mlrs-py/python/mlrs/covariance.py` | Python shim EmpiricalCovariance + LedoitWolf | VERIFIED | 2 classes present |
| `crates/mlrs-py/python/mlrs/random_projection.py` | Python shim GRP + SRP + jl_min_dim | VERIFIED | 2 classes + _densify + issparse path |
| `tests/fixtures/*.npz` | 14 oracle blobs (empirical_covariance ×4, ledoit_wolf ×4, incremental_pca ×4, jl_min_dim ×2) | VERIFIED | All 14 .npz files confirmed in tests/fixtures/ |
| `scripts/gen_oracle.py` | 4 new generators wired into main() | VERIFIED | `def gen_empirical_covariance`, `def gen_ledoit_wolf`, `def gen_incremental_pca`, `def gen_jl_min_dim` all present (grep count = 4) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `crates/mlrs-algos/src/lib.rs` | `PartialFit` | `pub use traits::PartialFit` | WIRED | Line 51: `pub use traits::{Fit, KNeighbors, PartialFit, …}` |
| `crates/mlrs-backend/src/prims/mod.rs` | `rng`, `incremental_svd` | `pub mod rng; pub mod incremental_svd;` | WIRED | Both registrations confirmed at lines 18-19 |
| `crates/mlrs-algos/src/decomposition/mod.rs` | `incremental_pca` | `pub mod incremental_pca;` | WIRED | Line 21 confirmed (uncommented, not a stub) |
| `crates/mlrs-algos/src/lib.rs` | `covariance`, `projection` | `pub mod covariance; pub mod projection;` | WIRED | Lines 38 and 43 confirmed |
| `prims/rng.rs` | `DeviceArray::from_host` | single upload of host-generated matrix | WIRED | `from_host` found in rng.rs implementation |
| `crates/mlrs-backend/src/prims/kmeans.rs` | `prims::rng::SplitMix64` | `use crate::prims::rng::SplitMix64` | WIRED | 3 hits for `prims::rng\|use crate::prims::rng` in kmeans.rs |
| `prims/incremental_svd.rs` | `prims::svd::svd` | re-SVD of stacked matrix | WIRED | `svd::` appears 3 times in incremental_svd.rs |
| `prims/incremental_svd.rs` | `mlrs_core::sign_flip::align_rows` | svd_flip(u_based_decision=False) | WIRED | `align_rows` appears 4 times in incremental_svd.rs |
| `covariance/empirical_covariance.rs` | `prims::covariance::covariance` | ddof=0 centered Gram | WIRED | `covariance::` appears 2 times |
| `covariance/empirical_covariance.rs` | `prims::eig::eig` | pinvh precision_ | WIRED | `eig::` appears 2 times |
| `decomposition/incremental_pca.rs` | `prims::incremental_svd::merge` | partial_fit calls merge | WIRED | `incremental_svd` appears 3 times in estimator |
| `projection/gaussian.rs` | `prims::rng::gaussian_matrix` | N(0,1/nc) matrix generation | WIRED | `rng::` appears 2 times in gaussian.rs |
| `projection/sparse.rs` | `prims::rng::sparse_achlioptas_matrix` | Achlioptas matrix generation | WIRED | `rng::sparse_achlioptas_matrix` found in sparse.rs |
| `crates/mlrs-py/src/estimators/decomposition.rs` | `IncrementalPCA::partial_fit` | `fn partial_fit` | WIRED | 1 `fn partial_fit` implementation present; guard_f64 =6 |
| `random_projection.py` | scipy sparse densify | `_densify(X)` → `X.toarray()` | WIRED | `_densify`, `issparse`, `toarray` all present (12 hits) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `empirical_covariance_test.rs` | `covariance_`, `precision_` | prims::covariance + prims::eig → oracle .npz blobs | Yes — eig-based pinvh assembly from device compute | FLOWING |
| `ledoit_wolf_test.rs` | `shrinkage_`, `covariance_` | β/δ/μ closed form on host from device Gram → oracle .npz | Yes — computed from real batch data | FLOWING |
| `incremental_pca_test.rs` | `components_`, `explained_variance_`, transforms | prims::incremental_svd::merge → oracle .npz blobs | Yes — 3-batch streaming over real X | FLOWING |
| `random_projection_test.rs` | `components_`, projected X | prims::rng::gaussian_matrix / sparse_achlioptas_matrix → property assertions | Yes — host-generated matrices via SplitMix64 + GEMM transform | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| PRIM-06: 5 rng tests pass | `cargo test -p mlrs-backend --features cpu --test rng_test` | 5 passed, 0 failed, 0 ignored | PASS |
| PRIM-07: incremental_svd 3 tests pass | `cargo test -p mlrs-backend --features cpu --test incremental_svd_test` | 3 passed, 0 failed, 0 ignored | PASS |
| COV-01: empirical covariance 4 tests pass | `cargo test -p mlrs-algos --features cpu --test empirical_covariance_test` | 4 passed, 0 failed, 0 ignored | PASS |
| COV-02: LedoitWolf 4 tests pass | `cargo test -p mlrs-algos --features cpu --test ledoit_wolf_test` | 4 passed, 0 failed, 0 ignored | PASS |
| DECOMP-03: IncrementalPCA 14 tests pass | `cargo test -p mlrs-algos --features cpu --test incremental_pca_test` | 14 passed, 0 failed, 0 ignored | PASS |
| PROJ-01/02: random projection 8 tests pass | `cargo test -p mlrs-algos --features cpu --test random_projection_test` | 8 passed, 0 failed, 0 ignored | PASS |
| PyO3 smoke: 3 tests pass | `cargo test -p mlrs-py --features cpu --test pyclass_smoke_test` | 3 passed, 0 failed (incl. `five_phase7_estimators_construct_unfit`) | PASS |

**Total: 41/41 tests passed**

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PRIM-06 | 07-02 | Host SplitMix64 RNG: Gaussian + Achlioptas-sparse + permutation, seed-reproducible, memory-gated | SATISFIED | `rng.rs` 296 lines, 5/5 tests green, byte-identical seed-repro confirmed |
| PRIM-07 | 07-03 | Incremental-SVD merge over v1 svd: stacked matrix, mean-correction, svd_flip, ddof=1 | SATISFIED | `incremental_svd.rs` 349 lines, 3/3 tests green at 1e-5 (f64) / 1e-4 (f32) |
| COV-01 | 07-04 | EmpiricalCovariance: covariance_/location_/precision_ within 1e-5 (MLE/ddof=0), eig-pinvh | SATISFIED | 4/4 tests green, rank-deficient precision_ case included |
| COV-02 | 07-04 | LedoitWolf: shrinkage-regularized covariance_, shrinkage_ ∈ [0,1], two n | SATISFIED | 4/4 tests green (n=12, n=40, f32+f64) |
| DECOMP-03 | 07-05 | IncrementalPCA: partial_fit + fit, all attributes, transform/inverse_transform, svd_flip | SATISFIED | 14/14 tests green; PartialFit trait implemented and wired |
| PROJ-01 | 07-06 | GaussianRandomProjection: n_components='auto' via jl_min_dim, property-gated | SATISFIED | 8/8 tests green; jl_min_dim integer-exact oracle; JL distortion averaged ≈1 over 50 trials |
| PROJ-02 | 07-06 | SparseRandomProjection: Achlioptas, property-gated, sparse input densified at Python ingress | SATISFIED | Achlioptas density+±v test green; `_densify` → `X.toarray()` wired in random_projection.py |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `rng.rs` | 20, 83, 237 | `next_u64() % n` (in doc-comments only) | None | Doc-comments warning AGAINST biased modulo; no actual biased modulo in any generator (permutation uses `next_below`); T-07-03 met |

No `TBD`, `FIXME`, or `XXX` markers found in any Phase-7 modified files. No `#[ignore]` remaining on active test functions in any test file (two files have `#[ignore]` appearing only in doc-comment text referencing the Wave-0 scaffold they were promoted from). No stub implementations, empty handlers, or hardcoded empty data detected.

### Human Verification Required

None. All correctness claims are programmatically verified via the 41/41 cpu test suite. The rocm f32 gate is an opportunistic phase-level check (gfx1100/ROCm 7.1.1 is the environment's GPU) and is not required for this verification — the REQUIREMENTS.md and ROADMAP.md both specify cpu(f64) as the primary correctness gate.

### Gaps Summary

No gaps. All 7 must-have truths are VERIFIED with behavioral evidence. All 7 requirement IDs (PRIM-06, PRIM-07, COV-01, COV-02, DECOMP-03, PROJ-01, PROJ-02) are satisfied. The PartialFit trait and all five PyO3 #[pyclass] wrappers with Python shims are present and wired.

**Notable implementation deviations from plan (all auto-fixed during execution, no gaps):**
- `InvalidEps` renamed to `InvalidEpsDistortion` (DBSCAN already owns `InvalidEps`)
- LedoitWolf `delta_` required `/n²` normalization per actual sklearn 1.7.1 source (RESEARCH pattern omitted it)
- Incremental-SVD subsequent-batch centering uses batch's OWN mean, not running mean (sklearn's actual algorithm)
- `johnson_lindenstrauss_min_dim` returns `Result<usize, AlgoError>` (not bare `usize`) to surface typed eps rejection

---

_Verified: 2026-06-20T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
