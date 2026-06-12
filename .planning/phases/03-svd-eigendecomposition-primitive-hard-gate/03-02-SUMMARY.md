---
phase: 03-svd-eigendecomposition-primitive-hard-gate
plan: 02
subsystem: testing
tags: [svd, eig, numpy-oracle, nyquist, fixtures, primerror, thiserror]

# Dependency graph
requires:
  - phase: 02-core-prims
    provides: "gemm/covariance oracle harness, load_npz/OracleCase, capability::skip_f64_with_log, align_rows sign-flip, PrimError(ShapeMismatch/DimMismatch), gen_oracle.py generator infra"
  - phase: 03-01
    provides: "ROCm bring-up so the f32 rocm gate is runnable (this plan touches disjoint files; no hard runtime dep)"
provides:
  - "PrimError::NotSquare and PrimError::NotConverged typed variants (D-06, D-12) for SVD/eig pre-launch validation + convergence failure"
  - "gen_oracle.py gen_svd (np.linalg.svd full_matrices=False) + gen_eigh (np.linalg.eigh reversed to descending) generators with SVD_TALL/SVD_WIDE/EIG_N shape constants"
  - "Five committed .npz fixtures: svd_tall (f32+f64), svd_wide (f32), eigh (f32+f64)"
  - "svd_test.rs (7 Nyquist fns) + eig_test.rs (4 Nyquist fns) compiling Wave-0 scaffolds, ignored until plans 03-03/03-04 land the prims"
affects: [03-03, 03-04, 03-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Nyquist Wave-0 scaffold: named test fns exist as #[ignore] stubs asserting fixture load + well-formedness BEFORE the prim, so 03-03/04 have failing tests to drive implementation"
    - "Oracle fixture stores numpy reference in the primitive's emitted order (eigh reversed ascending->descending, D-04) so the test compares directly with no re-sort"

key-files:
  created:
    - crates/mlrs-backend/tests/svd_test.rs
    - crates/mlrs-backend/tests/eig_test.rs
    - tests/fixtures/svd_tall_f32_seed42.npz
    - tests/fixtures/svd_tall_f64_seed42.npz
    - tests/fixtures/svd_wide_f32_seed42.npz
    - tests/fixtures/eigh_f32_seed42.npz
    - tests/fixtures/eigh_f64_seed42.npz
  modified:
    - crates/mlrs-core/src/error.rs
    - scripts/gen_oracle.py

key-decisions:
  - "eigh fixture stores eigenvalues/eigenvectors REVERSED to descending at generation time (D-04) so it matches the primitive's emitted order — the test compares directly rather than reversing in-test."
  - "Scaffold stubs reference NO prims::svd / prims::eig symbols (those don't exist yet); each stub asserts only fixture load + shape well-formedness so the test crate compiles, with #[ignore] marking the not-yet-implemented prim call."

patterns-established:
  - "Wave-0 #[ignore] scaffold: every VALIDATION.md-named test fn present and runnable (reported ignored) before its prim exists."
  - "f64 capability split copied verbatim from gemm_test.rs: f64 fixture tests gate on capability::skip_f64_with_log (cpu runs, rocm skips-with-log)."

requirements-completed: [PRIM-05]

# Metrics
duration: 8min
completed: 2026-06-12
---

# Phase 3 Plan 02: SVD/eig Nyquist Wave-0 Scaffold Summary

**Typed NotSquare/NotConverged PrimError variants, numpy SVD/eigh oracle generators, five committed .npz fixtures, and two compiling SVD/eig test scaffolds carrying all 11 VALIDATION.md Nyquist test functions (ignored until the kernels land in 03-03/04).**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-06-12T12:23:00Z
- **Completed:** 2026-06-12T12:27:08Z
- **Tasks:** 2
- **Files modified:** 9 (2 modified, 7 created)

## Accomplishments
- Extended `PrimError` with `NotSquare { operand, rows, cols }` (D-06 squareness validation, ASVS V5) and `NotConverged { operand, max_sweeps, residual }` (D-12 Jacobi sweep cap), following the existing thiserror form with per-field doc comments — one variant per violation class.
- Added `gen_svd` (`np.linalg.svd(full_matrices=False)`, descending S, D-02/D-04) and `gen_eigh` (`np.linalg.eigh`, reversed to descending to match the primitive's emitted order, D-04) to `gen_oracle.py` with `SVD_TALL=(8,4)`, `SVD_WIDE=(4,8)`, `EIG_N=4` shape constants, wired into `main()` so one run regenerates all five fixtures.
- Generated and committed the five `.npz` blobs (numpy via `/tmp` venv, PEP 668): svd tall f32+f64, svd wide f32, eigh f32+f64.
- Created `svd_test.rs` (7 fns) and `eig_test.rs` (4 fns) as compiling Wave-0 scaffolds — all 11 VALIDATION.md-named test functions present, fixture-load + shape-well-formedness asserted, prim calls marked `#[ignore]` for 03-03/04.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend PrimError + add SVD/eig fixture generators** - `603678a` (feat)
2. **Task 2: Generate/commit fixtures + create svd_test.rs/eig_test.rs scaffolds** - `9432d82` (test)

**Plan metadata:** (this commit — docs: complete plan)

## Files Created/Modified
- `crates/mlrs-core/src/error.rs` - Added `PrimError::NotSquare` + `PrimError::NotConverged` variants (D-06, D-12).
- `scripts/gen_oracle.py` - Added `gen_svd`/`gen_eigh` + `SVD_TALL`/`SVD_WIDE`/`EIG_N` constants; wired into `main()`.
- `crates/mlrs-backend/tests/svd_test.rs` - 7 Nyquist SVD test fns + `fixture()` resolver + `assert_svd_fixture_well_formed` helper.
- `crates/mlrs-backend/tests/eig_test.rs` - 4 Nyquist eig test fns + `fixture()` resolver + `assert_eigh_fixture_well_formed` helper.
- `tests/fixtures/svd_tall_f32_seed42.npz`, `svd_tall_f64_seed42.npz`, `svd_wide_f32_seed42.npz`, `eigh_f32_seed42.npz`, `eigh_f64_seed42.npz` - Committed numpy oracle blobs.

## Decisions Made
- **eigh order stored at generation, not reversed in-test:** `np.linalg.eigh` returns ascending; the device eig primitive sorts descending (D-04). The fixture stores `w`/`V` already reversed so the future test compares directly with no re-sort — keeps the test logic minimal and pins exactly the order the primitive must emit.
- **Scaffold avoids referencing non-existent prim symbols:** stubs assert only `load_npz` + shape well-formedness (plus a `runtime::active_client()` smoke for the device-only `svd_moderate_256x64`), so the test crate compiles today; `#[ignore]` flags the prim call that 03-03/04 wire.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. All five fixtures generated cleanly via the `/tmp/oraclevenv` numpy venv; both test crates compiled and ran on cpu with every stub correctly reported as `ignored` (svd: 7 ignored, eig: 4 ignored).

## User Setup Required
None - no external service configuration required. (Fixture regeneration is a build-time-only `/tmp` numpy venv; the committed `.npz` blobs are the trusted reference and CI never runs the generator.)

## Next Phase Readiness
- **03-03 (SVD kernel):** fixtures (`svd_tall_f32/f64`, `svd_wide_f32`), `PrimError::{NotSquare,NotConverged}`, and the 7 named svd_test stubs are ready — 03-03 removes `#[ignore]` and wires `svd::<F>` + reconstruction/orthonormality invariants.
- **03-04 (eig kernel):** `eigh_f32/f64` fixtures + the 4 eig_test stubs are ready — 03-04 wires `eig::<F>` + the residual invariant.
- **No blockers.** f64 fixture tests are capability-gated (cpu runs, rocm skips-with-log per the unchanged CubeCL-HIP F64 finding).

---
*Phase: 03-svd-eigendecomposition-primitive-hard-gate*
*Completed: 2026-06-12*

## Self-Check: PASSED

All created/modified files present on disk; both task commits (`603678a`, `9432d82`) exist in git history.
