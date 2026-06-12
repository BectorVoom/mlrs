---
phase: 04-closed-form-estimators
plan: 02
subsystem: backend-primitive
tags: [cholesky, spd-solve, triangular-solve, cubecl-kernel, ridge-prereq, prim]

# Dependency graph
requires:
  - phase: 04-closed-form-estimators
    plan: 01
    provides: "PrimError::NotPositiveDefinite variant, cholesky_test.rs #[ignore] stubs, cholesky f32/f64 fixtures, Fit/Predict/Transform traits"
  - phase: 03-svd-eigendecomposition-primitive-hard-gate
    provides: "jacobi_eig single-cube blueprint, MAX_DIM, skip_f64_with_log gate, eig/svd host-wrapper pattern, DeviceArray/BufferPool, host_to_f64 bytemuck helpers"
provides:
  - "mlrs-kernels::cholesky_solve — feature-free #[cube(launch)] Cholesky factor + forward/back triangular solve, fully in-kernel (D-11 gate 3)"
  - "mlrs-backend::prims::cholesky::cholesky_solve(pool,a,b,n,rhs,out) -> Result<DeviceArray,PrimError> — validate-before-launch SPD solve for Ridge (D-02)"
  - "cholesky_solve_with_factor — returns the kernel-emitted L for the standalone ‖L·Lᵀ−A‖ invariant"
  - "Non-SPD rejection via PrimError::NotPositiveDefinite (negative-pivot flag, never NaN)"
affects: [04-05-ridge]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-cube unit-0-does-all serial factor+solve (jacobi_eig 'acting unit' idiom) for the inherently-sequential Cholesky-Banachiewicz recurrence"
    - "Length-3 info array [flag, pivot_index, pivot_value] so the host surfaces the exact non-positive pivot value, not a re-encoded magnitude"
    - "Dedicated l_out factor buffer so the host checks ‖L·Lᵀ−A‖ from the kernel-emitted L (unambiguous L source), never re-deriving the factor on the host"

key-files:
  created:
    - crates/mlrs-kernels/src/cholesky.rs
    - crates/mlrs-backend/src/prims/cholesky.rs
  modified:
    - crates/mlrs-kernels/src/lib.rs
    - crates/mlrs-backend/src/prims/mod.rs
    - crates/mlrs-backend/tests/cholesky_test.rs

key-decisions:
  - "Unit-0-does-all serial schedule (RESEARCH Open Q2) — n ≤ 64 makes serialization cheap and the Cholesky recurrence is inherently sequential; sidesteps cross-unit shared-memory aliasing a distributed triangular update would create"
  - "info array length 3 [flag, pivot_index, pivot_value] rather than length 2 with an encoded pivot — the host reports the true non-positive √ argument unambiguously (fixed an encoding bug found by the non-SPD test)"
  - "cholesky_solve_with_factor (returns (x, L)) is the public L-source for the ‖L·Lᵀ−A‖ test; the plain cholesky_solve returns only x and releases L — resolves the prior 'Claude's Discretion' L-source ambiguity"
  - "mlrs-kernels build verified feature-free (the crate has NO cpu/rocm feature by design, D-13); the real cpu/rocm launch-codegen gate is the backend build + cholesky_test under each feature"

requirements-completed: [LINEAR-02]

# Metrics
duration: 5min
completed: 2026-06-12
---

# Phase 4 Plan 02: Cholesky/Triangular-Solve Primitive Summary

**The one genuinely-new device primitive of Phase 4 — a single-cube, all-shared-memory `cholesky_solve` `#[cube]` kernel that factors a small SPD `A = L·Lᵀ` and solves `A·x = b` entirely in-kernel, plus a validate-before-launch `prims::cholesky` host wrapper that rejects non-SPD input as `PrimError::NotPositiveDefinite` — validated standalone (`‖A·x−b‖`, `‖L·Lᵀ−A‖`, non-SPD) on cpu(f64)+rocm(f32), ready for Ridge (04-05).**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-06-12T07:01:32Z
- **Completed:** 2026-06-12T07:06:44Z
- **Tasks:** 2
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments

- `mlrs-kernels::cholesky_solve` is a feature-free `#[cube(launch)]` kernel implementing all three phases — Cholesky-Banachiewicz factor, forward solve `L·z = b`, back solve `Lᵀ·x = z` — inside ONE launch with `sync_cube()` between phases and NO host round-trip (D-11 gate 3). It writes the lower factor `L` to a dedicated `l_out` buffer (for the host invariant check), the solution to `x_out`, and a non-SPD flag to `info_out`.
- The diagonal sqrt argument is GUARDED (`≤ 1e-12` near-zero floor): a non-positive pivot writes a negative flag + the failing index + the actual pivot value into `info_out` and NEVER emits `√(negative) = NaN` (RESEARCH Pitfall 4 / T-04-02-02).
- `mlrs-backend::prims::cholesky::cholesky_solve` validates geometry BEFORE any unsafe launch (ASVS V5 / T-04-02-01): `a.len() == n*n` and `n ≤ MAX_DIM` → `NotSquare`, `b.len() == n*rhs` → `ShapeMismatch`. It threads the optional `out` Gram buffer through as the kernel's working input (D-11 gate 2 — no parallel n² allocation for Ridge), reads the info array back via the bytemuck helper, and surfaces `PrimError::NotPositiveDefinite { operand, pivot_index, pivot_value }` on a non-positive pivot.
- `cholesky_test.rs` is fully activated (all `#[ignore]` removed): the `‖A·x−b‖` solve invariant (vs the committed `scipy.linalg.solve(...,assume_a="pos")` x, within 1e-5), the `‖L·Lᵀ−A‖` factor invariant (reading the KERNEL-EMITTED L, not re-derived), and the non-SPD guard (`PrimError::NotPositiveDefinite` on an indefinite matrix) all pass.
- **cpu gate (f64+f32): 6/6 pass. rocm gate (f32 runs; f64 skips-with-log): 6/6 pass.** Both backends build clean under `--features cpu` and `--features rocm`.

## Task Commits

1. **Task 1: cholesky_solve #[cube] kernel (single-cube, all-LDS, in-kernel solve)** — `482da30` (feat)
2. **Task 2: prims::cholesky host wrapper + activate cholesky_test.rs** — `b91c5d3` (feat)

**Plan metadata:** _(final docs commit — this SUMMARY + STATE + ROADMAP)_

## Files Created/Modified

- `crates/mlrs-kernels/src/cholesky.rs` (NEW) — feature-free `#[cube(launch)] cholesky_solve<F>` (factor + forward + back, in-kernel; writes `x_out`/`l_out`/`info_out`; negative-pivot guard; no plane width/32; no `continue`; SharedMemory sized to comptime `MAX_DIM`).
- `crates/mlrs-kernels/src/lib.rs` — `pub mod cholesky;` + `pub use cholesky::cholesky_solve;`.
- `crates/mlrs-backend/src/prims/cholesky.rs` (NEW) — host `cholesky_solve` + `cholesky_solve_with_factor` (returns kernel-emitted L), `validate_geometry` (NotSquare/ShapeMismatch), info-array read → `NotPositiveDefinite`.
- `crates/mlrs-backend/src/prims/mod.rs` — `pub mod cholesky;`.
- `crates/mlrs-backend/tests/cholesky_test.rs` — removed all `#[ignore]`; wired the real `‖A·x−b‖` / `‖L·Lᵀ−A‖` / non-SPD assertions (f64 skip-with-log gate retained for the f64 cases).

## Decisions Made

- **Unit-0-does-all serial schedule** (RESEARCH Open Q2): the Cholesky-Banachiewicz recurrence is inherently sequential and `n ≤ 64` makes serialization cheap, so unit 0 performs the entire factor + both solves (the eig "acting unit does the whole operation" idiom) while other units idle — sidestepping the cross-unit shared-memory aliasing a distributed triangular update would create.
- **Length-3 info array `[flag, pivot_index, pivot_value]`** instead of a 2-slot array with an encoded pivot magnitude — the host then reports the EXACT non-positive √ argument. (This corrected a sign-decoding bug; see Deviations.)
- **`cholesky_solve_with_factor` exposes the kernel-emitted L** as the unambiguous source for the `‖L·Lᵀ−A‖` test; `cholesky_solve` returns only `x` (releasing L) for the production Ridge path. This resolves the plan's flagged "Claude's Discretion" L-source ambiguity.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Non-SPD pivot value was reported as a positive magnitude**
- **Found during:** Task 2 (the `cholesky_rejects_non_spd` test failed: expected a non-positive pivot value, got `4e0`).
- **Issue:** The Task-1 kernel encoded the non-SPD flag as `info_out[0] = -(|diag| + 1)` and the host decoded `pivot_value = -(flag + 1.0)`, which recovers `|diag|` — a POSITIVE value — so the surfaced `PrimError::NotPositiveDefinite.pivot_value` was the wrong sign and the diagnostic was misleading.
- **Fix:** Widened the info array to length 3 `[flag, pivot_index, pivot_value]`: the kernel now writes a strict `-1` flag, the failing index, and the ACTUAL non-positive pivot value into separate slots; the host reads `info[2]` directly. Verified the non-SPD test now reports `value=-4e0`.
- **Files modified:** `crates/mlrs-kernels/src/cholesky.rs`, `crates/mlrs-backend/src/prims/cholesky.rs`, `crates/mlrs-backend/tests/cholesky_test.rs`.
- **Commit:** `b91c5d3` (folded into the Task-2 commit, since the kernel info contract and the host decode are one logical change validated by the same test).

### Note on the Task-1 verify command

The plan's Task-1 `<automated>` verify was `cargo build -p mlrs-kernels --features cpu && cargo build -p mlrs-kernels --features rocm`. `mlrs-kernels` is feature-free BY DESIGN (D-13 — it must not depend on any backend runtime feature), so it has no `cpu`/`rocm` feature and that exact command errors. The kernel was instead verified feature-free (`cargo build -p mlrs-kernels`, exit 0) AND through the meaningful launch-codegen gate: `cargo build -p mlrs-backend --features cpu` and `--features rocm` both exit 0, which is where the `cholesky_solve::launch` macro actually expands under each concrete runtime. This is consistent with how every other `mlrs-kernels` kernel (jacobi_eig/svd) is gated.

## Issues Encountered

Beyond the non-SPD pivot-sign bug above (auto-fixed), none. The solve and factor invariants passed first-try on both cpu (f64+f32) and rocm (f32); the f64 cases correctly skip-with-log on rocm (D-07).

## TDD Gate Compliance

Both tasks carry `tdd="true"`. The RED gate is the Wave-0 `#[ignore]` scaffold from 04-01 (`cholesky_test.rs` asserting fixture load only — the real assertions could not pass because `prims::cholesky` did not exist). Task 1 (`482da30`) added the kernel; Task 2 (`b91c5d3`) added the host wrapper AND removed `#[ignore]`, wiring the real `‖A·x−b‖`/`‖L·Lᵀ−A‖`/non-SPD assertions that now pass (GREEN). The intermediate failing run (non-SPD pivot-sign) is a genuine RED→GREEN within Task 2. No separate `refactor` commit was needed. Note: the two commits are both `feat` (kernel, then host+tests) rather than a literal `test`-then-`feat` pair because the failing test scaffold (the RED commit) already landed in 04-01 (`09345e1`) — this plan's job is to make it GREEN.

## Known Stubs

None. The `cholesky_solve` / `cholesky_solve_with_factor` wrappers are fully wired to the kernel; no hardcoded/placeholder data paths. The `out=Some(...)` Gram-buffer-reuse path (D-11 gate 2) is implemented and validated by `validate_geometry` but is first exercised end-to-end by Ridge in 04-05 (the standalone tests use `out=None`); this is the intended consumer split, not a stub.

## Threat Flags

None. The files introduce no security surface beyond the plan's `<threat_model>`: the only trust boundary is the host caller → kernel geometry, mitigated by `validate_geometry` before the unsafe launch (T-04-02-01), and the sqrt-pivot guard (T-04-02-02), both implemented. No new network/auth/file-access surface; no new cargo packages (T-04-02-SC, accept).

## Self-Check: PASSED

- `crates/mlrs-kernels/src/cholesky.rs` and `crates/mlrs-backend/src/prims/cholesky.rs` present on disk.
- Task commits `482da30` and `b91c5d3` present in `git log`.
- `cargo test -p mlrs-backend --features cpu --test cholesky_test`: 6 passed / 0 failed.
- `cargo test -p mlrs-backend --features rocm --test cholesky_test`: 6 passed / 0 failed (f64 cases print SKIPPED).
- `cargo build -p mlrs-backend --features cpu` and `--features rocm` both exit 0.

## Next Phase Readiness

- **04-05 (Ridge)** ready: `prims::cholesky::cholesky_solve(pool, a, b, n, rhs, out)` is registered and validated; Ridge solves `(XᵀX + αI)·coef = Xᵀy` by threading the Gram buffer through `out` (D-11 gate 2) and reading back `coef`. Non-SPD/near-singular inputs surface `PrimError::NotPositiveDefinite` (which `AlgoError` wraps via `#[from]`). No blockers.

---
*Phase: 04-closed-form-estimators*
*Completed: 2026-06-12*
