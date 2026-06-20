---
phase: 07-covariance-projection
plan: 01
subsystem: testing
tags: [scaffold, traits, error-handling, oracle-fixtures, nyquist, covariance, random-projection, incremental-pca, splitmix64]

# Dependency graph
requires:
  - phase: 04-closed-form-estimators
    provides: "Fit/Predict/Transform trait surface, AlgoError, gen_oracle.py gen_pca skeleton, pca_test.rs oracle scaffolding"
  - phase: 05-distance-iterative-estimators
    provides: "Wave-0 file-disjoint scaffold precedent (05-01), AlgoError hyperparameter-guard style, SplitMix64 in prims/kmeans.rs"
  - phase: 03-svd-eig
    provides: "thin SVD prim + MAX_ROWS/MAX_COLS caps, skip_f64_with_log f64 capability gate, Nyquist #[ignore] scaffold pattern (03-02)"
provides:
  - "PartialFit<F> trait (D-01) re-exported from lib.rs, alongside Fit/Predict/Transform"
  - "AlgoError::InvalidDensity / InvalidBatchSize / InvalidEpsDistortion hyperparameter guards (ASVS V5)"
  - "covariance/ + projection/ module-index stubs in mlrs-algos; prims::rng + prims::incremental_svd empty compiling stubs in mlrs-backend"
  - "4 gen_oracle.py generators (empirical_covariance, ledoit_wolf, incremental_pca, jl_min_dim) + 14 committed .npz blobs"
  - "6 #[ignore] Nyquist test scaffolds (fixture-load + shape only, no non-existent symbol refs)"
affects: [07-02-rng-prim, 07-03-incremental-svd-prim, 07-04-covariance-estimators, 07-05-incremental-pca, 07-06-random-projection]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PartialFit<F> mirrors Fit<F> exactly (same bound/pool/DeviceArray/explicit-shape) with the Option<&DeviceArray> y-slot retained for Phase-10 MBSGD reuse (D-01)"
    - "Wave-0 file-disjoint scaffold: this plan OWNS every shared index file so downstream Wave-1/2 plans stay strictly file-disjoint and parallel-safe"
    - "Nyquist #[ignore] stubs assert fixture-load/shape ONLY, never a not-yet-existent symbol — the test crate compiles today; downstream plans un-#[ignore] and wire the real call"

key-files:
  created:
    - crates/mlrs-algos/src/covariance/mod.rs
    - crates/mlrs-algos/src/projection/mod.rs
    - crates/mlrs-backend/src/prims/rng.rs
    - crates/mlrs-backend/src/prims/incremental_svd.rs
    - crates/mlrs-backend/tests/rng_test.rs
    - crates/mlrs-backend/tests/incremental_svd_test.rs
    - crates/mlrs-algos/tests/empirical_covariance_test.rs
    - crates/mlrs-algos/tests/ledoit_wolf_test.rs
    - crates/mlrs-algos/tests/incremental_pca_test.rs
    - crates/mlrs-algos/tests/random_projection_test.rs
  modified:
    - crates/mlrs-algos/src/traits.rs
    - crates/mlrs-algos/src/error.rs
    - crates/mlrs-algos/src/lib.rs
    - crates/mlrs-algos/src/decomposition/mod.rs
    - crates/mlrs-backend/src/prims/mod.rs
    - scripts/gen_oracle.py

key-decisions:
  - "InvalidEps name collision resolved: the planned JL-distortion guard is named InvalidEpsDistortion (NOT InvalidEps) because DBSCAN's InvalidEps variant already exists in AlgoError — a duplicate variant is a compile error (Rule 1)"
  - "LedoitWolf oracle uses a correlated low-rank+noise design (shrinkage_ ~0.18/0.12) instead of pure standard_normal (which degenerates shrinkage_ to 1.0) so the closed-form beta/delta arithmetic is genuinely exercised (Rule 2)"
  - "decomposition::incremental_pca registration left COMMENTED in decomposition/mod.rs (the file is created by plan 07-05) — keeps the index file-disjoint from the estimator plan"
  - "rng.rs / incremental_svd.rs created as empty doc-comment stub FILES (not just mod registrations) so THIS plan compiles; plans 07-02/03 fill the bodies"

patterns-established:
  - "Wave-0 scaffold owns shared index files (traits/error/lib + 3 mod.rs + prims/mod.rs) — 05-01 precedent"
  - "Oracle fixtures regenerated from system Python (sklearn 1.9.0 present) — no /tmp venv needed this run; 14 .npz blobs committed as binaries, CI never runs the script"

requirements-completed: [PRIM-06, PRIM-07, COV-01, COV-02, DECOMP-03, PROJ-01, PROJ-02]

# Metrics
duration: 14min
completed: 2026-06-20
---

# Phase 7 Plan 01: Covariance & Projection Wave-0 Scaffold Summary

**Landed the full Phase-7 shared interface surface (PartialFit<F> trait, three new AlgoError hyperparameter guards, covariance/projection/rng/incremental_svd module-index stubs) plus the complete Nyquist test scaffold (six #[ignore] test files) and four gen_oracle.py generators with 14 committed .npz oracle blobs — every downstream Wave-1/2 plan now receives its contracts in-place and stays strictly file-disjoint.**

## Performance

- **Duration:** ~14 min
- **Started:** 2026-06-20 (Phase 07 execution)
- **Completed:** 2026-06-20
- **Tasks:** 3 of 3
- **Files modified:** 16 (10 created, 6 modified)

## Accomplishments

### Task 1 — PartialFit trait + AlgoError guards + module-index stubs (commit fd655d0)
- Added `PartialFit<F>` to `traits.rs` (D-01), mirroring `Fit<F>`'s shape EXACTLY (same `F: Float + CubeElement + Pod` bound, `pool`/`DeviceArray`/explicit `(rows, cols)` convention, returns `&mut Self`), with the `Option<&DeviceArray>` `y`-slot RETAINED per the D-01 cross-cutting contract for Phase-10 MBSGD reuse. Documented that IncrementalPCA passes `y: None`.
- Re-exported `PartialFit` from `lib.rs` alongside the other traits.
- Extended `AlgoError` with three struct-variants in the existing `InvalidAlpha` style: `InvalidDensity { estimator, density: f64 }` (density ∈ (0,1]), `InvalidBatchSize { estimator, batch_size: usize }` (≥1), and `InvalidEpsDistortion { estimator, eps: f64 }` (eps ∈ (0,1) for JL). Reused `InvalidNComponents`/`NotFitted`/`Unsupported`/`Prim(#[from] PrimError)` as-is.
- Registered the module index: created empty doc-comment `covariance/mod.rs` + `projection/mod.rs`; added `pub mod covariance;` / `pub mod projection;` to `lib.rs`; left `// pub mod incremental_pca;` commented in `decomposition/mod.rs` (created by 07-05); added `pub mod rng;` / `pub mod incremental_svd;` to `prims/mod.rs` with empty stub FILES so the plan compiles.
- Gate: `cargo build -p mlrs-algos --features cpu` and `cargo build -p mlrs-backend --features cpu` both exit 0.

### Task 2 — gen_oracle.py: 4 generators + committed fixtures (commit a976bd0)
- Added `gen_empirical_covariance` (COV-01): full-rank (16×5) + RANK-DEFICIENT (4×6, n≤p) cases storing `X`/`covariance_`/`location_`/`precision_`; the rank-deficient case exercises the eig-based pinvh `precision_` floor.
- Added `gen_ledoit_wolf` (COV-02): two sample counts (n=12, n=40), `X`/`covariance_`/`shrinkage_`; correlated low-rank+noise design so `shrinkage_` is interior (≈0.18/≈0.12).
- Added `gen_incremental_pca` (DECOMP-03): whiten on/off, all attributes + `transform`/`inverse_transform`/`n_samples_seen_`, C-contiguous `components_` (Fortran-order pitfall), stacked SVD matrix sized `nc+bs+1=14 ≤ 256` and `n_features=6 ≤ 64`.
- Added `gen_jl_min_dim` (PROJ-01/02, D-12): the single value oracle over a `(n_samples, eps∈(0,1))` 3×3 grid. No matrix/transform RP oracle (RNG ≠ MT19937).
- Wired all four into `main()`'s dual-dtype loop; committed 14 `.npz` blobs (f32+f64). Regenerated from system Python (sklearn 1.9.0 present — no /tmp venv needed this run).

### Task 3 — Six #[ignore] Nyquist test scaffolds (commit cfd322a)
- Created six test files, each carrying every VALIDATION.md test name as an `#[ignore]` stub asserting fixture-load + shape ONLY, with ZERO references to not-yet-existent prim/estimator symbols.
- `rng_test.rs`: `rng_gaussian_distribution` / `rng_seed_reproducible` / `rng_achlioptas_density` / `rng_permutation_bijection` / `rng_memory_gate`; `RNG_TRIALS=50`; f64 capability gate.
- `incremental_svd_test.rs`: `incremental_svd_two_batch_merge` (asserts the stacked SVD matrix clears MAX_ROWS/MAX_COLS) + `incremental_svd_memory_gate`; f64 gate.
- `empirical_covariance_test.rs`: attrs f32/f64 + rank-deficient `precision_` f32/f64.
- `ledoit_wolf_test.rs`: two n (12, 40) f32/f64; `shrinkage_ ∈ [0,1]` invariant.
- `incremental_pca_test.rs`: whiten on/off f32/f64 (partial_fit + fit); `skip_f64_with_log` gate present.
- `random_projection_test.rs`: JL distortion / matrix moments / seed-repro / `transform==X·componentsᵀ` property stubs + `random_projection_jl_min_dim` value oracle; `JL_TRIALS=50` (D-11).
- Gate: `cargo test -p mlrs-backend --features cpu --no-run` and `cargo test -p mlrs-algos --features cpu --no-run` both exit 0; all stubs recognized as `ignored` by the test runner (5/5, 2/2, etc.).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] InvalidEps variant name collision**
- **Found during:** Task 1
- **Issue:** The plan/patterns specified a new `InvalidEps { estimator, eps: f64 }` JL-distortion guard, but `AlgoError` ALREADY contains an `InvalidEps` variant (DBSCAN neighborhood radius, added in Phase 5 / commit 05-01). Adding a second `InvalidEps` to the same enum is a duplicate-variant compile error.
- **Fix:** Named the new variant `InvalidEpsDistortion` (documenting the distinction from DBSCAN's radius `InvalidEps`, which has a different valid range and meaning). All three required guard concepts (density, batch_size, JL eps) exist and compile.
- **Files modified:** crates/mlrs-algos/src/error.rs
- **Commit:** fd655d0
- **Acceptance-criterion note:** the plan's Task-1 grep criterion `grep -Ec "InvalidDensity|InvalidBatchSize|InvalidEps" == 3` was written assuming `InvalidEps` was a brand-new name; given the pre-existing DBSCAN `InvalidEps`, a literal-3 line count is unsatisfiable. The three NEW variants are present (`grep -Ec "InvalidDensity|InvalidBatchSize|InvalidEpsDistortion"` over the variant lines == 3).

**2. [Rule 2 - Missing critical functionality] LedoitWolf oracle was a degenerate shrinkage_=1.0**
- **Found during:** Task 2
- **Issue:** Generating LedoitWolf fixtures from pure `standard_normal` data (identity true covariance) drove sklearn's `shrinkage_` to exactly `1.0` (full shrink to the scaled-identity target) for BOTH n — a weak oracle that does not exercise the closed-form β/δ shrink arithmetic the estimator must reproduce.
- **Fix:** Switched `gen_ledoit_wolf` to a CORRELATED low-rank-plus-noise design (2 latent factors + 0.3·isotropic noise), landing `shrinkage_` strictly inside (0,1) (≈0.18 at n=12, ≈0.12 at n=40). The COV-02 `shrinkage_ ∈ [0,1]` test invariant still holds; the value oracle is now discriminating.
- **Files modified:** scripts/gen_oracle.py
- **Commit:** a976bd0

### Other notes
- Oracle regen used the SYSTEM Python (sklearn 1.9.0, scipy, numpy all importable) rather than a fresh /tmp venv — the project memory item `oracle-fixture-regen-needs-venv` applies when the system interpreter lacks numpy (PEP 668), which was not the case here. The committed blobs are byte-reproducible (seed=42).
- The plan's `<output>` block has a duplicated `</output>` closing tag (cosmetic, in the PLAN.md source); no impact on execution.

## Known Stubs

All six test files are intentional Wave-0 `#[ignore]` scaffolds (fixture-load + shape assertions only). They are NOT data-flow stubs — they are the Nyquist test contract that downstream plans 07-02..06 un-`#[ignore]` and wire to the real prim/estimator calls. `crates/mlrs-backend/src/prims/rng.rs` and `incremental_svd.rs` are intentional empty doc-comment module stubs (bodies filled by plans 07-02/03). `crates/mlrs-algos/src/covariance/mod.rs` and `projection/mod.rs` are intentional empty index stubs (estimator files added by 07-04/06). This matches the documented 04-01 / 05-01 / 03-02 Wave-0 scaffold precedent and is the explicit objective of this plan; no stub blocks the plan's goal.

## Verification

- `cargo build -p mlrs-algos --features cpu` → exit 0.
- `cargo build -p mlrs-backend --features cpu` → exit 0.
- `cargo test -p mlrs-backend --features cpu --no-run` → exit 0 (rng_test + incremental_svd_test build).
- `cargo test -p mlrs-algos --features cpu --no-run` → exit 0 (all four new estimator test binaries build).
- `gen_oracle.py` parses as valid Python; 4 generators wired; 14 `.npz` blobs committed across all four families (f32+f64).
- All six scaffolds recognized as `ignored` by the cargo test runner (no symbol references resolved at compile time beyond `capability`/`load_npz`/`OracleCase`).
- `kmeanspp_test.rs` UNAFFECTED (this plan did not touch `kmeans.rs`; the SplitMix64 promotion is plan 07-03).

## Self-Check: PASSED

All 10 created source/test files + the SUMMARY exist on disk; all three task commits (fd655d0, a976bd0, cfd322a) are present in git history; 14 oracle `.npz` blobs are tracked under `tests/fixtures/`.
