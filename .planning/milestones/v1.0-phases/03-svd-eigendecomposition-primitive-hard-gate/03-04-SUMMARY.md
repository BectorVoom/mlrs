---
phase: 03-svd-eigendecomposition-primitive-hard-gate
plan: 04
subsystem: primitives
tags: [jacobi-eig, symmetric-eigendecomposition, cubecl, prim-05, rocm, covariance, pca]

# Dependency graph
requires:
  - phase: 03-01
    provides: ROCm/HIP bring-up (ActiveRuntime=HipRuntime on gfx1100), cpu(f64)+rocm(f32) gate, skip_f64_with_log
  - phase: 03-02
    provides: PrimError::NotSquare + NotConverged, eigh_{f32,f64}_seed42 fixtures, eig_test #[ignore] scaffold
  - phase: 03-03
    provides: jacobi_svd_sweep single-cube kernel idioms (shared-mem tile, sync_cube, two-threshold convergence, host descending sort), svd() host orchestration pattern
  - phase: 02-01
    provides: gemm() (used for the eig residual invariant A·V)
  - phase: 02-04
    provides: covariance() Gram buffer-reuse precedent (D-11 gate 2 target)
provides:
  - jacobi_eig_sweep — two-sided cyclic Jacobi symmetric-eig #[cube(launch)] kernel (generic <F: Float + CubeElement>, in-kernel convergence, single-acting-unit rotation)
  - eig() host orchestration — validate-square → NotSquare, launch, descending sort (D-04), covariance/GEMM out-buffer reuse (D-11 gate 2), NotConverged on cap
  - green eig oracle + residual + clustered invariant suite (cpu f32+f64, rocm f32; f64 skip-with-log)
affects: [03-05, pca-full-path, phase-4-estimators]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-sided symmetric Jacobi: single acting unit performs the whole Jᵀ·A·J rotation per pair (cyclic sequential pairs), parallel units idle — avoids cross-unit shared-memory aliasing"
    - "Post-sweep off-diagonal-norm measured directly from the live matrix (per-row Σa_ij² + log₂-tree), not an in-sweep estimate"
    - "eig() reuses the covariance/GEMM out buffer as the kernel working input (D-11 gate 2), mirroring covariance's own Gram-reuse"

key-files:
  created:
    - crates/mlrs-kernels/src/jacobi_eig.rs
    - crates/mlrs-backend/src/prims/eig.rs
  modified:
    - crates/mlrs-kernels/src/lib.rs
    - crates/mlrs-backend/src/prims/mod.rs
    - crates/mlrs-backend/tests/eig_test.rs

key-decisions:
  - "Two-sided eig pairs are processed SEQUENTIALLY (single acting unit per pair), NOT the SVD's disjoint-parallel column schedule: a two-sided rotation touches the full rows AND columns p,q, so index-disjoint pairs are NOT footprint-disjoint and a distributed/parallel update races on cross entries"
  - "Convergence uses a TRUE post-sweep off-diagonal-norm measurement from the live matrix (per-row sums double-count → 2·Σ_{i<j} a_ij², sqrt makes the break marginally stricter = more accurate), replacing the first-attempt in-sweep accumulator that underestimated and stopped early"
  - "NotSquare validation rejects a.len() != n*n (and n > MAX_DIM) BEFORE any unsafe launch (D-06 trusts symmetry but validates squareness); no (A+Aᵀ)/2"
  - "eig.rs keeps only small post-convergence read-backs (w, V, info) for the host descending sort + convergence check — the convergence LOOP is fully in-kernel (D-11 gate 3), mirroring the SVD sibling's precedent"

patterns-established:
  - "Single-acting-unit rotation for two-sided in-place shared-memory updates (proven on cpu + rocm gfx1100)"
  - "Clustered/degenerate eigenproblems validated by the basis-invariant residual ‖A·v−λ·v‖ only, never per-vector fixture compare (Pitfall 3)"

requirements-completed: [PRIM-05]

# Metrics
duration: 45min
completed: 2026-06-12
---

# Phase 3 Plan 04: Two-Sided Cyclic Jacobi Symmetric Eigendecomposition Summary

**Two-sided cyclic Jacobi symmetric-eig (`jacobi_eig_sweep` single-cube kernel + `eig()` host orchestration) that matches `np.linalg.eigh` within 1e-5, returns descending eigenvalues, reuses the covariance/GEMM buffer (D-11 gate 2), and is green on cpu (f32+f64) and rocm gfx1100 (f32).**

## Performance

- **Duration:** ~45 min
- **Completed:** 2026-06-12
- **Tasks:** 3
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments

- `jacobi_eig_sweep` — a `#[cube(launch)]` two-sided cyclic Jacobi sweep generic over `<F: Float + CubeElement>`, applying each rotation on BOTH rows and columns (`A ← Jᵀ·A·J`), accumulating eigenvectors in `V`, with the sweep loop + off-diagonal-norm convergence test fully in-kernel (D-11 gate 3) and the eigenvalue diagonal written UNSORTED (host sorts).
- `eig()` host orchestration: validate-square → `NotSquare` before any unsafe launch (D-06), covariance/GEMM `out`-buffer reuse threaded as the kernel working input (D-11 gate 2), descending eigenvalue sort + eigenvector-column permute (D-04), `NotConverged` on a sweep-cap hit (D-12), device-resident eigenpairs.
- Full eig validation suite green: `eig_symmetric_f32/f64_fixture` (vs `np.linalg.eigh` reversed to descending, sign-aligned with `align_rows`), `eig_residual_invariant` (`‖A·v−λ·v‖` via Phase-2 `gemm`), `eig_clustered_invariant` (repeated eigenvalues {5,2,2,2} via residual only) — all 4 pass on cpu (f32+f64) and rocm (f32; f64 skip-with-log). The 4 plan-02 `#[ignore]` stubs removed.

## Task Commits

1. **Task 1: Two-sided Jacobi symmetric-eig sweep kernel** - `8e05f3b` (feat)
2. **Task 2: eig() host orchestration** - `a6e5712` (feat)
3. **Task 3: Green eig oracle + residual + clustered invariants** - `2c5978f` (test; includes the Rule-1 kernel convergence/rotation fix)
4. **Convergence-doc correction** - `c710367` (docs)

_TDD note: Tasks 1 and 2 are tdd="true"; the kernel landed first (Task 1) and the host (Task 2) compiled against it, but the live RED→GREEN evidence is the Task-3 test suite, where the kernel's correctness bug surfaced and was fixed. See Deviations._

## Files Created/Modified

- `crates/mlrs-kernels/src/jacobi_eig.rs` - two-sided cyclic Jacobi `#[cube(launch)]` sweep kernel (single acting unit per pair, in-kernel post-sweep off-diagonal-norm convergence, D-12 constants documented)
- `crates/mlrs-kernels/src/lib.rs` - `pub mod jacobi_eig;` + re-exports `jacobi_eig_sweep`, `MAX_DIM`
- `crates/mlrs-backend/src/prims/eig.rs` - `pub fn eig<F>(pool, a, n, out) -> Result<(w, V), PrimError>`; squareness validation, buffer reuse, descending sort, NotConverged
- `crates/mlrs-backend/src/prims/mod.rs` - `pub mod eig;`
- `crates/mlrs-backend/tests/eig_test.rs` - the green oracle + residual + clustered suite (plan-02 `#[ignore]`s removed)

## Decisions Made

- **Sequential single-acting-unit two-sided rotation.** The one-sided SVD kernel rotates index-disjoint COLUMN pairs concurrently because each rotation's write footprint is its two columns. A two-sided eig rotation of plane (p,q) touches the ENTIRE rows AND columns p,q — including cross entries belonging to another index-disjoint pair — so index-disjoint pairs are NOT footprint-disjoint and cannot rotate concurrently. The kernel therefore visits the n(n-1)/2 upper-triangle pairs one at a time, with a single acting unit (unit 0) performing the whole rotation (mirroring the SVD kernel's "acting unit does the whole rotation" idiom, proven on the CPU + HIP backends). n is small (≤ MAX_DIM, typically 4), so this is cheap.
- **Direct post-sweep off-diagonal-norm measurement.** Convergence is measured from the live matrix state after each sweep (each unit sums `a_ij²` over `j≠i` for its row, then a log₂-tree reduction), not from an in-sweep per-pair accumulator (which underestimates because a later rotation in the same sweep can refill an off-diagonal, causing premature termination).
- **D-06 squareness, no symmetrization.** `validate_geometry` rejects `a.len() != n*n` and `n > MAX_DIM` with `PrimError::NotSquare` before any unsafe launch; symmetry is trusted (no `(A+Aᵀ)/2`), since the only v1 feeder is the symmetric covariance Gram.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Two-sided rotation race + premature convergence (kernel did not converge)**
- **Found during:** Task 3 (greening the oracle + residual tests)
- **Issue:** The first kernel implementation (a) accumulated the off-diagonal norm DURING the sweep (an estimate a later rotation could refill, so it stopped early), and (b) attempted to distribute each two-sided rotation ACROSS units (each unit updating its own row/column index). Because a two-sided rotation touches the full rows AND columns p,q, index-disjoint units' writes aliased this plane's footprint through shared memory — the rotations raced and the matrix never diagonalised (residuals stuck at ~0.1–0.8 after the 30-sweep cap → `NotConverged`).
- **Fix:** (a) Replaced the in-sweep accumulator with a TRUE post-sweep off-diagonal-norm measurement from the live matrix. (b) Switched to a single-acting-unit sequential cyclic-pair rotation (unit 0 performs the entire `Jᵀ·A·J` for each pair, all other units idle, `sync_cube` after each pair), mirroring the SVD kernel's proven idiom. Now converges to ~1e-6 (f32) / machine precision (f64).
- **Files modified:** crates/mlrs-kernels/src/jacobi_eig.rs
- **Verification:** All 4 eig tests pass on cpu (f32+f64) and rocm (f32); the sibling `svd_test` remains 7/7 green (shared kernel crate, no regression).
- **Committed in:** `2c5978f` (Task 3 commit), with a follow-up doc-sync in `c710367`.

---

**Total deviations:** 1 auto-fixed (1 bug).
**Impact on plan:** The fix was essential for correctness (the kernel was non-convergent without it). The artifact shape, host API, and tests are exactly as planned. No scope creep.

## Issues Encountered

- **Acceptance criterion `grep to_host == 0` is incompatible with the mandated host-side descending sort.** Task 2's `<acceptance_criteria>` literally requires `grep to_host == 0` in `eig.rs`, but the same task's `<action>` and `<done>` mandate a host-side DESCENDING eigenvalue sort + eigenvector-column permute (D-04), which inherently reads back the small `w`/`V`. The named sibling to mirror (`prims/svd.rs`) does exactly this — it reads back small `V`/`S`/`info` for its host sort and is explicitly the proven pattern. Resolved in favor of the `<action>`/`<done>`/sibling precedent: `eig.rs` keeps only the small post-convergence read-backs (`w`, `V`, `info`) plus the one pre-launch `‖A‖_F` estimate; the CONVERGENCE LOOP is fully in-kernel (D-11 gate 3, the criterion's true intent). This matches the 03-03 STATE decision recorded for `svd.rs` ("10 plain to_host calls… the convergence loop is fully device-resident").

## Known Stubs

None — `eig()` is fully wired to the kernel and validated end-to-end; no placeholder data paths.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- PRIM-05's eig half is proven standalone (PRIM-05 SVD half landed in 03-03). The PCA `full` solver path can now consume `eig()` for true signed eigenvalues.
- **Ready for 03-05 (D-11 memory gate):** `eig()` threads the covariance/GEMM `out` buffer straight through to the kernel working input (D-11 gate 2 — no parallel n² allocation) and keeps the convergence loop in-kernel (D-11 gate 3 — no host round-trip between sweeps). Both gate prerequisites hold, mirroring covariance's Gram-reuse and the SVD sibling's residency.
- No blockers.

---
*Phase: 03-svd-eigendecomposition-primitive-hard-gate*
*Completed: 2026-06-12*

## Self-Check: PASSED

- Created files verified on disk: `jacobi_eig.rs`, `prims/eig.rs`, `03-04-SUMMARY.md`.
- Task commits verified in git history: `8e05f3b`, `a6e5712`, `2c5978f`, `c710367`.
- eig_test green on cpu (4/4) and rocm (4/4, f64 skip-with-log); svd_test 7/7 (no regression).
