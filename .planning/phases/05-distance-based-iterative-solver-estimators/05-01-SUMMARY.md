---
phase: 05-distance-based-iterative-solver-estimators
plan: 01
subsystem: testing
tags: [scaffold, traits, algoerror, gen_oracle, sklearn-fixtures, cubecl, nyquist, i32-deviced-array]

# Dependency graph
requires:
  - phase: 04-closed-form-estimators
    provides: "Fit/Predict/Transform trait surface, AlgoError, gen_oracle.py (gen_ridge/gen_cholesky), cholesky_test.rs/ridge_test.rs Wave-0 scaffold precedent"
  - phase: 03-svd-eig
    provides: "cpu(f64)+rocm(f32) gate, skip_f64_with_log capability pattern, Nyquist #[ignore] scaffold precedent"
provides:
  - "PredictLabels / KNeighbors / PredictProba traits (D-05/D-07) in mlrs-algos"
  - "AlgoError variants InvalidK/InvalidEps/InvalidMinSamples/InvalidL1Ratio/InvalidC/NotConverged (T-05-01-01)"
  - "mlrs_algos::cluster / mlrs_algos::neighbors module index stubs"
  - "5 kernel stub modules (mlrs_kernels::topk/kmeans/dbscan/coordinate/lbfgs) + 5 prim stub modules (mlrs_backend::prims::topk/kmeans/dbscan/coordinate_descent/lbfgs) — file-disjoint enabler for plans 02-06"
  - "6 gen_oracle.py generators (gen_kmeans w/ injected init, gen_dbscan, gen_knn, gen_lasso, gen_elastic_net, gen_logistic) + 14 committed .npz fixtures"
  - "14 #[ignore] Wave-0 oracle test stubs (6 prim + 8 estimator) + i32 DeviceArray round-trip confirmation (D-06)"
affects: [05-02, 05-03, 05-04, 05-05, 05-06, 05-07, 05-08, 05-09, 05-10]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Wave-0 scaffold owns lib.rs/mod.rs registrations; primitive/estimator plans fill only their own stub file (file-disjoint, parallel-safe)"
    - "Empty (doc-comment-only) Rust module body is a valid compiling stub — registered via pub mod with no pub use until the symbol exists"
    - "KMeans injected-init oracle (D-09): fixture carries an init array so Lloyd is deterministic across numpy/Rust"
    - "i32 DeviceArray rides the byte-keyed pool with zero pool/bridge changes (D-06); -1 noise sentinel representable"

key-files:
  created:
    - "crates/mlrs-algos/src/cluster/mod.rs"
    - "crates/mlrs-algos/src/neighbors/mod.rs"
    - "crates/mlrs-kernels/src/{topk,kmeans,dbscan,coordinate,lbfgs}.rs"
    - "crates/mlrs-backend/src/prims/{topk,kmeans,dbscan,coordinate_descent,lbfgs}.rs"
    - "14 tests/fixtures/*.npz blobs (kmeans/dbscan/knn/lasso/elastic_net/logistic_binary/logistic_multi × f32/f64)"
    - "6 mlrs-backend test stubs + 8 mlrs-algos test stubs"
  modified:
    - "crates/mlrs-algos/src/traits.rs (3 new traits)"
    - "crates/mlrs-algos/src/error.rs (6 new variants)"
    - "crates/mlrs-algos/src/lib.rs (module index + re-exports)"
    - "crates/mlrs-kernels/src/lib.rs (5 pub mod)"
    - "crates/mlrs-backend/src/prims/mod.rs (5 pub mod)"
    - "scripts/gen_oracle.py (6 generators + main registrations)"

key-decisions:
  - "i32 DeviceArray confirmed (D-06) via a non-ignored round-trip test incl. the -1 DBSCAN noise value — zero pool/bridge changes needed, so no later plan is surprised"
  - "logistic fixtures: predict/predict_proba are the PRIMARY gauge-invariant gate (Pitfall 5); coef_ is the looser secondary — multi fixture stores 3×n_features symmetric over-parameterized softmax weights"
  - "one KNN fixture serves NEIGH-01/02/03 (NearestNeighbors + classifier + regressor) with distinct distances (Pitfall 8)"
  - "stub test bodies reference ONLY load_npz + shape asserts (no non-existent prim/estimator symbol) so the test crates compile today; #[ignore] strings name the activating plan"

patterns-established:
  - "Nyquist Wave-0 scaffold: registrations centralized in the scaffold plan, stub files empty/ignored, downstream plans remove #[ignore] + add their symbol"
  - "gen_oracle generator shape: import sklearn inside fn, default_rng(SEED), wrap every stored array with c() C-contiguous, register f32(rocm)+f64(cpu) in main()"

requirements-completed: [LINEAR-03, LINEAR-04, LINEAR-05, CLUSTER-01, CLUSTER-02, NEIGH-01, NEIGH-02, NEIGH-03]

# Metrics
duration: 16min
completed: 2026-06-12
---

# Phase 5 Plan 01: Wave-0 Scaffold Summary

**Phase-5 trait/error surface (PredictLabels/KNeighbors/PredictProba + 6 hyperparameter-guard AlgoError variants), 10 file-disjoint kernel/prim stub modules, 6 sklearn oracle generators with 14 committed fixtures, and 14 #[ignore] Wave-0 oracle test stubs — with the i32 DeviceArray round-trip (D-06) confirmed.**

## Performance

- **Duration:** 16 min
- **Started:** 2026-06-12T21:44:44Z
- **Completed:** 2026-06-12T22:01:00Z
- **Tasks:** 4
- **Files modified:** 46 (across the 4 task commits: 6 source-modified, 7 module/index, 14 fixtures, 14 test stubs, gen_oracle.py)

## Accomplishments
- Extended the estimator trait surface with `PredictLabels<F>` (i32 labels, D-05/D-06), `KNeighbors<F>` (distances + i32 indices, D-07), `PredictProba<F>` (per-class fractions, D-07), re-exported from `lib.rs`.
- Extended `AlgoError` with the six Phase-5 hyperparameter-guard variants (`InvalidK`/`InvalidEps`/`InvalidMinSamples`/`InvalidL1Ratio`/`InvalidC`/`NotConverged`, T-05-01-01) in the existing `InvalidAlpha` struct-variant style.
- Created the `cluster`/`neighbors` module index stubs + 10 empty compiling kernel/prim stub modules (registered in `lib.rs`/`prims/mod.rs`) so plans 02-06 fill only their own file — file-disjoint, parallel-safe.
- Added six sklearn oracle generators and committed 14 `.npz` fixtures (KMeans carries an INJECTED init per D-09; DBSCAN gives cluster + noise(-1) + border; Lasso has 5 exact-zero coefficients; one KNN fixture serves all three neighbor estimators; LogReg binary + multiclass).
- Created 14 `#[ignore]` Wave-0 oracle test stubs (6 prim + 8 estimator) that compile today against the empty stubs, plus a non-ignored `i32_device_array_roundtrips` test that PASSES — confirming D-06 needs zero pool/bridge changes (including the `-1` noise value).

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend traits.rs/error.rs/lib.rs + cluster/neighbors module stubs** - `a1f392e` (feat)
2. **Task 2: Create 5 kernel + 5 prim stub module files** - `8ca4ac2` (feat)
3. **Task 3: Extend gen_oracle.py with 6 generators + commit 14 fixtures** - `7f835a5` (feat)
4. **Task 4: Create 14 Wave-0 oracle test stubs + i32 DeviceArray confirmation** - `745465c` (test)

**Plan metadata:** _(final docs commit below)_

## Files Created/Modified
- `crates/mlrs-algos/src/traits.rs` - +PredictLabels/KNeighbors/PredictProba traits (mirror Predict bound + signature shape)
- `crates/mlrs-algos/src/error.rs` - +6 hyperparameter-guard variants
- `crates/mlrs-algos/src/lib.rs` - pub mod cluster/neighbors + re-export the 3 new traits
- `crates/mlrs-algos/src/cluster/mod.rs`, `neighbors/mod.rs` - doc-commented index stubs (no pub mod <estimator> yet)
- `crates/mlrs-kernels/src/{topk,kmeans,dbscan,coordinate,lbfgs}.rs` + `lib.rs` - empty kernel stubs + registrations
- `crates/mlrs-backend/src/prims/{topk,kmeans,dbscan,coordinate_descent,lbfgs}.rs` + `mod.rs` - empty prim stubs + registrations
- `scripts/gen_oracle.py` - gen_kmeans/gen_dbscan/gen_knn/gen_lasso/gen_elastic_net/gen_logistic + main() registrations
- `tests/fixtures/{kmeans,dbscan,knn,lasso,elastic_net,logistic_binary,logistic_multi}_{f32,f64}_seed42.npz` - 14 committed blobs
- `crates/mlrs-backend/tests/{topk,kmeanspp,lloyd,dbscan_mask,cd,lbfgs}_test.rs` - 6 prim oracle stubs (topk also carries the i32 round-trip)
- `crates/mlrs-algos/tests/{kmeans,dbscan,nearest_neighbors,knn_classifier,knn_regressor,lasso,elastic_net,logistic}_test.rs` - 8 estimator oracle stubs

## Decisions Made
- **i32 DeviceArray (D-06) proven, not assumed:** the round-trip test is NON-ignored and passes on cpu, including `-1`. This was the load-bearing confirmation the scaffold owed plans 05-04/05-10.
- **One KNN fixture for three estimators (NEIGH-01/02/03):** the blob carries `distances`/`indices` (NearestNeighbors), `predict_class`/`predict_proba`/`y_class` (classifier), `predict_reg`/`y_reg` (regressor), with distinct distances (Pitfall 8).
- **LogReg predict_proba is the primary gate:** multi fixture stores the full symmetric over-parameterized softmax (3×n_features), `coef_` is secondary (gauge-freedom, Pitfall 5).

## Deviations from Plan

None - plan executed exactly as written. (No deviation rules were triggered; the trait signatures, error variants, stub modules, generators, and test stubs all landed as specified, and both feature builds + the i32 round-trip pass.)

## Issues Encountered

- **Transient disk-full on the temp filesystem during the first `cargo test --no-run`.** The build's temporary files (plus the freshly-created `/tmp/oracle-venv` pip cache) filled the temp partition, surfacing as `ENOSPC` on the harness output dir. Resolved by removing the now-unneeded `/tmp/oracle-venv` (fixtures are already committed blobs) and clearing stale temp files, then re-running the test build with a project-local `TMPDIR` (`./.build-tmp`, untracked and removed afterward). No source impact — purely an environment cleanup.

## User Setup Required

None - no external service configuration required. (Fixture regeneration needs a `/tmp` venv with numpy/scipy/scikit-learn per PEP 668, but the committed `.npz` blobs are the test-time artifact — gen_oracle.py never runs in CI.)

## Next Phase Readiness
- **Plans 02-06 (primitives) are unblocked and file-disjoint:** each fills exactly one kernel file + one prim file + removes `#[ignore]` from its oracle stub; none touch `lib.rs`/`prims/mod.rs`.
- **Plans 07-10 (estimators) are unblocked:** each adds its `pub mod <estimator>;` line to `cluster/mod.rs`/`neighbors/mod.rs`/`linear/mod.rs` and wires its `#[ignore]` test stub against the committed fixture.
- **D-06 confirmed**, so no downstream plan needs pool/bridge changes for i32 labels/indices.
- No blockers. The cpu(f64)+rocm(f32) gate holds: both crates and both test targets build on `--features cpu` and `--features rocm`.

## Self-Check: PASSED

- All created files verified present (cluster/neighbors mods, 10 kernel/prim stubs, gen_oracle.py, 14 test stubs, 14 fixtures, SUMMARY.md).
- All 4 task commits verified in git history (`a1f392e`, `8ca4ac2`, `7f835a5`, `745465c`).
- `cargo build -p mlrs-algos --features cpu/rocm` green; `cargo build -p mlrs-kernels` + `-p mlrs-backend --features cpu/rocm` green; both crates' `--tests` build on cpu+rocm; `i32_device_array_roundtrips` passes; all 14 `#[ignore]` `fixture_loads` stubs load + shape-assert correctly.

---
*Phase: 05-distance-based-iterative-solver-estimators*
*Completed: 2026-06-12*
