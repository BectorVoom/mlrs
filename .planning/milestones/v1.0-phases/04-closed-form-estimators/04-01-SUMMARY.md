---
phase: 04-closed-form-estimators
plan: 01
subsystem: testing
tags: [estimator-traits, thiserror, oracle-fixtures, sklearn, cholesky, pca, ridge, truncated-svd, nyquist-scaffold]

# Dependency graph
requires:
  - phase: 03-svd-eigendecomposition-primitive-hard-gate
    provides: thin SVD primitive, sign_flip::align_rows, skip_f64_with_log gate, oracle load_npz/OracleCase, PrimError enum
  - phase: 02-core-compute-primitives
    provides: gemm/covariance/reduce primitives, DeviceArray/BufferPool, memory gate
  - phase: 01-foundation-oracle-backend-abstraction-arrow-bridge
    provides: oracle harness, gen_oracle.py infra, capability gating, tolerance policy
provides:
  - "mlrs-algos dependency graph (mlrs-core/mlrs-backend/cubecl/bytemuck) compiling on cpu + rocm"
  - "Fit/Predict/Transform estimator traits (D-04), generic over <F: Float + CubeElement + Pod>"
  - "AlgoError estimator-facing error enum (InvalidNComponents/InvalidAlpha/NotFitted/Unsupported + #[from] PrimError)"
  - "PrimError::NotPositiveDefinite variant for the Cholesky negative-pivot guard"
  - "linear/ + decomposition/ module-index stubs (estimator plans edit only their own files)"
  - "gen_oracle.py generators for cholesky/linear_regression/ridge/pca/truncated_svd + 14 committed f32/f64 .npz fixtures"
  - "Five #[ignore] Nyquist test stubs enumerating every Test-Map function"
affects: [04-02-cholesky-primitive, 04-03-linear-regression, 04-04-pca-truncated-svd, 04-05-ridge, 06-pyo3-bindings]

# Tech tracking
tech-stack:
  added: [scipy (oracle venv only), scikit-learn (oracle venv only)]
  patterns:
    - "Estimator-local AlgoError in mlrs-algos (not mlrs-core) wrapping PrimError via #[from]"
    - "Feature-passthrough Cargo block: mlrs-algos forwards cpu/wgpu/cuda/rocm to mlrs-backend (owns ActiveRuntime)"
    - "Module-index stubs with commented future pub mod lines so estimator plans stay file-disjoint"

key-files:
  created:
    - crates/mlrs-algos/src/traits.rs
    - crates/mlrs-algos/src/error.rs
    - crates/mlrs-algos/src/linear/mod.rs
    - crates/mlrs-algos/src/decomposition/mod.rs
    - crates/mlrs-backend/tests/cholesky_test.rs
    - crates/mlrs-algos/tests/linear_regression_test.rs
    - crates/mlrs-algos/tests/ridge_test.rs
    - crates/mlrs-algos/tests/pca_test.rs
    - crates/mlrs-algos/tests/truncated_svd_test.rs
  modified:
    - crates/mlrs-algos/Cargo.toml
    - crates/mlrs-algos/src/lib.rs
    - crates/mlrs-core/src/error.rs
    - scripts/gen_oracle.py

key-decisions:
  - "AlgoError is estimator-local in mlrs-algos (D-08 discretion), not a mlrs-core enum, so the primitive layer never depends on it; wraps PrimError via #[from] for ergonomic ? in estimator methods"
  - "Transform::inverse_transform has a default impl returning AlgoError::Unsupported so the uniform trait surface stays total (PCA overrides; TruncatedSVD keeps the default)"
  - "PCA tall case also written under the canonical kind-less pca_{dtype}_seed42.npz so a consumer loads the default PCA fixture without knowing the tall/wide split"
  - "Cholesky fixture carries rhs=2 to exercise the multi-column triangular solve even though Ridge consumes a single RHS"

patterns-established:
  - "Nyquist Wave-0 scaffold: #[ignore] test stubs assert fixture load + shape only (no non-existent symbol refs) so the test crate compiles before any estimator lands; later plans remove #[ignore] and wire the real assertion"
  - "fixture_loads load-not-just-present check (mlrs_core::load_npz + key/shape asserts) proves committed blobs are well-formed, not merely present on disk"

requirements-completed: [LINEAR-01, LINEAR-02, DECOMP-01, DECOMP-02]

# Metrics
duration: 9min
completed: 2026-06-12
---

# Phase 4 Plan 01: Closed-Form Estimators Wave-0 Scaffold Summary

**mlrs-algos dependency graph + Fit/Predict/Transform trait surface (D-04), AlgoError/NotPositiveDefinite error variants, gen_oracle.py sklearn/scipy generators with 14 committed f32/f64 fixtures, and five compiling #[ignore] Nyquist test stubs — the contracts every downstream Phase-4 estimator/primitive plan builds against.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-06-12T06:48:16Z
- **Completed:** 2026-06-12T06:56:50Z
- **Tasks:** 3
- **Files modified:** 13 (4 modified, 9 created) + 14 committed .npz fixtures

## Accomplishments

- `mlrs-algos` now depends on `mlrs-core`/`mlrs-backend`/`cubecl`/`bytemuck` and compiles on BOTH the cpu and rocm gates (D-07), with a feature-passthrough block forwarding `cpu`/`wgpu`/`cuda`/`rocm` to the backend that owns `ActiveRuntime`.
- The uniform `Fit`/`Predict`/`Transform` trait surface (D-04) exists and is re-exported, generic over `<F: Float + CubeElement + Pod>` exactly as the prims (D-08); `fit` returns `&mut self` (sklearn convention).
- Two new error classes: `PrimError::NotPositiveDefinite { operand, pivot_index, pivot_value }` (Cholesky negative-pivot guard, T-04-01-02) and the estimator-facing `AlgoError` with an `InvalidNComponents` variant (untrusted-hyperparameter guard, T-04-01-01) plus `InvalidAlpha`/`NotFitted`/`Unsupported`/`#[from] PrimError`.
- `lib.rs` owns the Phase-4 module index; `linear/mod.rs` + `decomposition/mod.rs` are module-index stubs with commented future `pub mod` lines, so 04-03/04/05 edit only their own estimator files (file-disjoint, parallel-safe).
- `gen_oracle.py` gained `gen_cholesky`/`gen_linear_regression`/`gen_ridge`/`gen_pca`/`gen_truncated_svd`; 14 `.npz` blobs (f32+f64) committed under `tests/fixtures/`, including the deterministic `algorithm='arpack'` TruncatedSVD fixture and the near-collinear LinearRegression case.
- Five `#[ignore]` test stubs compile and enumerate every Test-Map function (30 functions total); `cholesky_test::fixture_loads` actually loads a committed blob via `mlrs_core::load_npz` and validates `A`/`b`/`x`/`L` keys+shapes.

## Task Commits

Each task was committed atomically:

1. **Task 1: mlrs-algos deps + traits + module stubs + error variants** - `d4fdcf1` (feat)
2. **Task 2: gen_oracle.py Phase-4 generators + committed fixtures** - `43cd95c` (feat)
3. **Task 3: Five Nyquist #[ignore] test stubs** - `09345e1` (test)

**Plan metadata:** _(final docs commit — this SUMMARY + STATE + ROADMAP)_

_Note: Task 3 is a `tdd="true"` scaffold task. Because the estimators/primitive do not exist yet, the stubs assert fixture load + shape only (which passes today as an `#[ignore]` scaffold) rather than a RED→GREEN cycle; the RED gate lands in 04-02/03/04/05 when `#[ignore]` is removed and the real assertion is wired. See TDD Gate Compliance below._

## Files Created/Modified

- `crates/mlrs-algos/Cargo.toml` - Added mlrs-core/mlrs-backend/cubecl/bytemuck deps + cpu/wgpu/cuda/rocm feature passthrough; mlrs-core/env_logger dev-deps
- `crates/mlrs-algos/src/lib.rs` - Phase-4 module index (pub mod traits/error/linear/decomposition) + Fit/Predict/Transform/AlgoError re-exports (owned by 04-01)
- `crates/mlrs-algos/src/traits.rs` - Fit/Predict/Transform traits (D-04), inverse_transform default = Unsupported
- `crates/mlrs-algos/src/error.rs` - AlgoError estimator enum (InvalidNComponents/InvalidAlpha/NotFitted/Unsupported/#[from] PrimError)
- `crates/mlrs-algos/src/linear/mod.rs` - linear module-index stub (04-03 linear_regression, 04-05 ridge)
- `crates/mlrs-algos/src/decomposition/mod.rs` - decomposition module-index stub (04-04 pca, truncated_svd)
- `crates/mlrs-core/src/error.rs` - Added PrimError::NotPositiveDefinite variant
- `scripts/gen_oracle.py` - Five Phase-4 generators + Phase-4 main() calls; canonical pca alias
- `crates/mlrs-backend/tests/cholesky_test.rs` - fixture_loads + solve/factor/non-SPD #[ignore] stubs
- `crates/mlrs-algos/tests/{linear_regression,ridge,pca,truncated_svd}_test.rs` - estimator #[ignore] stubs
- `tests/fixtures/*.npz` - 14 committed f32/f64 oracle blobs (cholesky, linear_regression, ridge, pca tall/wide + canonical, truncated_svd)

## Decisions Made

- **AlgoError is estimator-local** in `mlrs-algos` rather than added to `mlrs-core::PrimError` — the n_components/alpha guards are estimator-specific and the primitive layer must not depend on them. `#[from] PrimError` keeps `?` ergonomic across prim calls. (Plan offered this as D-08 discretion.)
- **`Transform::inverse_transform` has a default impl** returning `AlgoError::Unsupported` so the trait surface is total (PCA overrides it; TruncatedSVD keeps the default, matching D-01 where only PCA reconstructs in v1).
- **Canonical PCA fixture alias:** the tall case is also written as `pca_{dtype}_seed42.npz` (kind-less) so the plan's literal verify path and a default consumer both resolve, while tall/wide variants stay available for the truncation sweep in 04-04.

## Deviations from Plan

None - plan executed exactly as written. Fixture naming follows the plan's `case_dtype_seed.npz` convention; the PCA tall/wide split adds a kind tag (mirroring the Phase-3 `svd_tall`/`svd_wide` precedent) with a canonical kind-less alias so the verify path `pca_f32_seed42.npz` resolves.

## Issues Encountered

None. cpu and rocm builds were clean on the first attempt; all five test crates compiled and listed without iteration; the `fixture_loads --ignored` check passed, confirming the committed blobs load via `mlrs_core::load_npz`.

## TDD Gate Compliance

Plan/Task 3 carries `tdd="true"` but is a **Nyquist scaffold** task, not a behavior-adding implementation: the estimators and the Cholesky primitive do not exist yet, so the test bodies assert only fixture load + shape and are marked `#[ignore]`. This is the deliberate Wave-0 pattern (carried from Phase 3's `03-02` svd/eig scaffold) — the RED gate is satisfied later when 04-02/03/04/05 remove `#[ignore]` and wire the real oracle/invariant assertions against the new symbols. No `test→feat` cycle applies within 04-01 because no production behavior is added here. The committed test commit is `09345e1`.

## Known Stubs

The five test files are intentional `#[ignore]` stubs (the Nyquist scaffold). They are NOT data-flow stubs that mask a broken UI/path — they assert fixture well-formedness and carry a `// 04-0X removes #[ignore] and wires the real <assertion>` comment naming the activating plan. The `linear/mod.rs` and `decomposition/mod.rs` commented `pub mod` lines are intentional module-index placeholders the estimator plans uncomment. All are explicitly scoped to downstream plans (04-02 through 04-05) and documented in those module/test doc-comments.

## Self-Check: PASSED

- Created files verified present (see below).
- Task commits verified in git log (d4fdcf1, 43cd95c, 09345e1).
- `cargo build -p mlrs-algos --features cpu` and `--features rocm` both exit 0.
- All five test crates list (exit 0); `fixture_loads` passes `--ignored`; estimator tests report 0 run / N ignored.

## Next Phase Readiness

- **04-02 (Cholesky primitive)** ready: `PrimError::NotPositiveDefinite` exists, `cholesky_test.rs` stubs + cholesky f32/f64 fixtures are in place; remove `#[ignore]` and wire `prims::cholesky` + the ‖A·x−b‖/‖L·Lᵀ−A‖/non-SPD assertions.
- **04-03 (LinearRegression)** ready: `Fit`/`Predict` traits, `AlgoError`, `linear/mod.rs` stub, linreg fixtures (incl. near-collinear) in place.
- **04-04 (PCA/TruncatedSVD)** ready: `Fit`/`Transform`(+inverse) traits, `AlgoError::InvalidNComponents`, `decomposition/mod.rs` stub, pca tall/wide + tsvd arpack fixtures in place.
- **04-05 (Ridge)** ready: consumes the 04-02 Cholesky primitive + `Fit`/`Predict`, ridge alpha-sweep fixtures in place.
- No blockers. The estimator plans are file-disjoint (each edits only its estimator file + the relevant module-index stub; `lib.rs` is owned by 04-01).

---
*Phase: 04-closed-form-estimators*
*Completed: 2026-06-12*
