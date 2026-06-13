---
phase: 05-distance-based-iterative-solver-estimators
plan: 05
subsystem: prims
tags: [coordinate-descent, lasso, elastic-net, soft-threshold, duality-gap, cubecl-cpu, host-loop, d10, oracle, primitive-first]

# Dependency graph
requires:
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 01
    provides: "coordinate.rs/prims/coordinate_descent.rs/cd_test.rs stubs + lib.rs/prims/mod.rs registrations; lasso_{f32,f64}_seed42.npz + elastic_net_{f32,f64}_seed42.npz fixtures; AlgoError::NotConverged/InvalidL1Ratio"
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 02
    provides: "cubecl-cpu MLIR-safe #[cube] idiom (no SharedMemory/bool/F::INFINITY/atomics; F/u32 accumulators + if-guards) — reused for col_dot/residual_axpy/enet_gap"
  - phase: 04-closed-form-estimators
    provides: "prims::cholesky validate→launch→scalar-readback wrapper shape; DeviceArray from_host/to_host/release_into; ridge centering precedent"
provides:
  - "mlrs_kernels::coordinate::{cd_col_dot, cd_residual_axpy, cd_enet_gap} — feature-free #[cube] kernels (column dot, residual axpy, device-side ElasticNet duality gap → 1 scalar)"
  - "mlrs_backend::prims::coordinate_descent::cd_solve — host cyclic CD loop (D-10: R/x/y reused, host-side w soft-threshold, ONE scalar gap readback/iter, tol·‖y‖² stop, norm2_cols[j]==0 skip)"
  - "cd_test.rs standalone oracle GREEN on cpu(f64): coef + exact sparsity within 1e-5 vs sklearn on lasso + elastic_net fixtures (both f32/f64)"
affects: [05-09]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CD host-driven loop (D-10): the n-heavy work (column dot, residual axpy, duality gap) is device-side; the host owns the cyclic loop + per-coordinate soft-threshold scalar; the ENTIRE duality gap is assembled in ONE device kernel (enet_gap) so exactly one scalar crosses back per outer convergence check"
    - "cubecl-cpu-safe reduction-in-kernel: single-cube GATHER on unit 0 (forward while-loop, F-accumulator) replaces cross-unit scatter+atomic, which the cpu MLIR lowering does not lower — the modest CD sample/feature counts make single-unit accumulation acceptable"
    - "soft-threshold lives host-side (one scalar/coordinate): sign(t)·max(|t|−l1_reg,0)/(‖X_j‖²+l2_reg), so the kernel surface stays the n-heavy dot + axpy only"
    - "fit_intercept=True oracle: center X (per-column mean) + y (mean) BEFORE cd_solve, since sklearn runs enet_coordinate_descent on the centered design (the prim solves the centered problem; intercept recovery is the estimator's job in 05-09)"

key-files:
  created: []
  modified:
    - "crates/mlrs-kernels/src/coordinate.rs (filled the 05-01 stub: col_dot + residual_axpy + enet_gap #[cube] kernels; pub use inside file)"
    - "crates/mlrs-backend/src/prims/coordinate_descent.rs (filled the 05-01 stub: cd_solve host loop, validate-before-launch, 1 scalar gap/iter, buffer reuse)"
    - "crates/mlrs-backend/tests/cd_test.rs (de-#[ignore]d: real sklearn oracle on lasso + elastic_net {f32,f64} incl. exact-zero sparsity)"

key-decisions:
  - "The duality gap is computed ENTIRELY device-side in one enet_gap kernel (X.T@R, dual_norm_∞, R·R, R·y, ‖w‖₁, ‖w‖₂² → one scalar) rather than reading R/w back and combining on the host — this keeps the D-10 gate literal (exactly ONE scalar readback per outer check, no per-iteration array readback) and is gate-relevant for plan 05-11"
  - "w (coefficients) is HOST-side scalar state, not a device buffer: the soft-threshold is one scalar per coordinate (the plan permits host-side scalar math), and w is re-uploaded only at the gap check (length-d, reused-shape). R is the device-resident reused buffer that the axpy mutates in place"
  - "norm2_cols + tol·‖y‖² are computed ONCE on the host from a single pre-loop X/y read-back (setup read, not a per-iteration array readback — distinct from the D-10 in-loop constraint)"
  - "Lasso is exercised as the l1_ratio=1 (l2_reg=0) case of the SAME cd_solve path (D-03 shared kernel); the oracle covers both families × both dtypes"

patterns-established:
  - "Iterative-solver duality-gap-in-one-kernel: assemble every gap component device-side and emit a single scalar, so the host loop's only per-iteration readback is the convergence-deciding scalar (D-10 bounded-allocation analogue of the Phase-3 Jacobi scratch gate)"

requirements-completed: [LINEAR-03, LINEAR-04]

# Metrics
duration: 6min
completed: 2026-06-13
---

# Phase 5 Plan 05: Coordinate-Descent Step Primitive (D-03) Summary

**The genuinely-new iterative-solver core for Lasso + ElasticNet: feature-free `#[cube]` kernels for the column dot, the residual axpy, and a device-side ElasticNet duality gap, driven by a host cyclic coordinate-descent loop (`cd_solve`) that reuses the residual/design buffers across iterations and reads back EXACTLY ONE scalar (the duality gap) per outer convergence check (D-10) — reproducing sklearn's un-normalized soft-threshold update and `tol·‖y‖²` stop, GREEN on cpu(f64) within 1e-5 INCLUDING the exact sparsity pattern (Pitfall 1) before Lasso/ElasticNet (05-09) consume it (D-01 primitive-first).**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-06-13T03:07:33Z
- **Completed:** 2026-06-13T03:13:53Z
- **Tasks:** 2 (both TDD)
- **Files modified:** 3 (kernel + prim + test — all 05-01 stubs filled; zero shared-file edits)

## Accomplishments
- Filled `mlrs_kernels::coordinate` with three feature-free `#[cube]` kernels generic over `<F: Float + CubeElement>`:
  - `col_dot` — `t_dot = Σ_i X[i*cols+j]·R[i]` via a single-cube GATHER on unit 0 (no atomics, no SharedMemory).
  - `residual_axpy` — `R[i] += factor·X[i*cols+j]`, the `scale`-shaped per-element map specialised to one strided column (bounds-checked, over-provisionable).
  - `enet_gap` — the ElasticNet duality gap (formulation A) assembled ENTIRELY device-side (`X.T@R`, `dual_norm_∞`, `R·R`, `R·y`, `‖w‖₁`, `‖w‖₂²`) into ONE scalar, so the host loop's only per-iteration readback is the convergence-deciding gap (D-10).
- Filled `mlrs_backend::prims::coordinate_descent::cd_solve`: validates `n*d == x.len()` AND `y.len() == n` → `PrimError::ShapeMismatch` BEFORE any unsafe launch (T-05-05-01 / ASVS V5); acquires `R` (=y at w=0) + the reused scalar scratch ONCE and reuses every iteration (D-10 bounded allocation); runs the CYCLIC pass over columns (skipping `norm2_cols[j]==0`, T-05-05-02), computing `t` via `col_dot`, the host soft-threshold `sign(t)·max(|t|−l1_reg,0)/(‖X_j‖²+l2_reg)`, and the device residual axpy per coordinate; applies sklearn's cheap host gate (`d_w_max/w_max ≤ tol` or last iter) then the one-scalar `enet_gap` readback, stopping on `gap ≤ tol·‖y‖²` (`tol` scaled by `‖y‖²` ONCE, Pitfall 2); caps at `max_iter=1000`. No Gap-Safe screening (Anti-Pattern).
- De-`#[ignore]`d `cd_test.rs` with the real standalone oracle: centers `(X, y)` per `fit_intercept=True`, maps `(α, l1_ratio)` → `(l1_reg, l2_reg) = (α·l1_ratio·n, α·(1−l1_ratio)·n)`, runs `cd_solve`, and asserts every `coef[j]` matches the fixture `coef_[j]` within 1e-5 INCLUDING exact-zero entries (sparsity, Pitfall 1) — covering `lasso_{f32,f64}` and `elastic_net_{f32,f64}`, f64 cpu-gated via `skip_f64_with_log`.
- Verified the full gate: `cargo build -p mlrs-kernels` green; `cargo test --features cpu -p mlrs-backend --test cd_test` 5/5 green (incl. f64 + exact sparsity); `cargo build -p mlrs-backend --features rocm --tests` green; `lib.rs`/`prims/mod.rs` untouched.

## Task Commits

1. **Task 1: CD soft-threshold col-dot + residual axpy `#[cube]` kernel** — `26ed768` (feat)
2. **Task 2: cd_solve host CD loop + enet_gap kernel + cd oracle** — `b635383` (feat)

## Files Created/Modified
- `crates/mlrs-kernels/src/coordinate.rs` — `col_dot` / `residual_axpy` / `enet_gap` `#[cube]` kernels; `pub use self::{...}` re-exports INSIDE the file (lib.rs untouched).
- `crates/mlrs-backend/src/prims/coordinate_descent.rs` — `cd_solve` host loop + `validate_geometry` + `soft_threshold` helper + `host_to_f64`/`from_f64`. (prims/mod.rs untouched.)
- `crates/mlrs-backend/tests/cd_test.rs` — `check_cd` oracle body + `fixture_loads` + 4 sklearn-match tests (lasso/elastic_net × f32/f64).

## Decisions Made
- **Duality gap assembled in ONE device kernel:** rather than reading `R`/`w` back and computing the gap on the host (which would be a per-iteration array readback breaking the D-10 memory-gate exception), `enet_gap` computes every gap component device-side and emits a single scalar. The host loop's only per-iteration readback is that one gap value (metered via `to_host_metered` so plan 05-11's gate can assert `read_backs` grows by 1 per outer check).
- **`w` is host-side scalar state, `R` is the device reused buffer:** the soft-threshold is one scalar per coordinate (the plan explicitly permits host-side scalar math), so `w` lives on the host and is re-uploaded only at the gap check; `R` is the device-resident residual the `residual_axpy` mutates in place and reuses every iteration.
- **`fit_intercept=True` ⇒ center in the oracle:** sklearn runs `enet_coordinate_descent` on the centered design and recovers the intercept from the means; `cd_solve` solves the centered problem, so the oracle centers `(X, y)` the same way. Intercept recovery is deferred to the estimator (05-09), keeping the prim a pure centered CD solve.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Added a device-side `enet_gap` kernel to honor the D-10 single-scalar-readback gate**
- **Found during:** Task 2 (wiring the duality-gap convergence check)
- **Issue:** The plan's action text allows the gap to be computed from "device dots → 1 scalar," but a straightforward host assembly (read `R`·`R`, `R`·`y`, `XᵀR` back and combine) would perform several length-1 readbacks AND, for `XᵀR`, effectively read residual-derived state per iteration — risking a violation of the literal D-10 contract (exactly ONE scalar per outer check, no per-iteration array readback). T-05-05-03 names this exact tampering risk.
- **Fix:** Added a third `#[cube]` kernel, `enet_gap`, that assembles the entire formulation-A gap device-side (`X.T@R`, `dual_norm_∞`, `R·R`, `R·y`, `‖w‖₁`, `‖w‖₂²`) and writes ONE scalar; the host reads back only that value (metered). This is within scope (the prim's own gap computation, no architectural change — the public `cd_solve` signature and the kernel module surface are exactly as planned) and strengthens the memory-gate posture that plan 05-11 will assert.
- **Files modified:** `crates/mlrs-kernels/src/coordinate.rs` (kernel), `crates/mlrs-backend/src/prims/coordinate_descent.rs` (launch + single readback)
- **Commit:** `b635383` (rides with Task 2 since it was discovered while wiring the gap)

The kernel constructs stay cubecl-cpu MLIR-safe (single-cube unit-0 forward scans, `F`/`u32` accumulators, `if`-guarded `|·|`/max — no SharedMemory, no `bool`, no `F::INFINITY`, no atomics), so the cpu(f64) primary gate runs the gap at launch without the `failed to run pass` panic that plan 05-02 documented.

## Known Stubs

None. All three kernels and `cd_solve` are fully implemented; the oracle exercises real device output (the asserted `coef` flows from `cd_solve`, not a hardcoded value), and the exact-zero sparsity is checked against genuine device results.

## Issues Encountered
- The kernel re-exports live at `mlrs_kernels::coordinate::{...}` (module path), NOT the crate root, because `lib.rs` (owned by the 05-01 scaffold) only re-exports a fixed symbol set at the root and this plan must not edit it. The prim imports via the module path — no shared-file edit needed.

## Next Phase Readiness
- **Plan 05-09 (Lasso/ElasticNet estimators) unblocked:** `cd_solve(pool, x, y, n, d, l1_reg, l2_reg, tol, max_iter) -> DeviceArray<F>` returns the device-resident centered-problem `coef`; the estimator centers `(X, y)`, maps `(α, l1_ratio)` → `(l1_reg, l2_reg)`, calls `cd_solve`, and recovers the intercept from the means (the `ridge.rs` center-then-solve precedent). The standalone oracle is GREEN within 1e-5 + exact sparsity on both families, satisfying the D-01 primitive-first gate.
- **Plan 05-11 (memory gate) ready:** `cd_solve` reads back exactly ONE scalar per outer convergence check (metered via `to_host_metered`) and reuses `R`/`x`/`y` across iterations — the D-10 bounded-allocation contract the gate asserts.
- No blockers. cpu(f64) full + rocm(f32) test-target build both green; `lib.rs`/`prims/mod.rs` untouched so the sibling Wave-2 prim plans stay file-disjoint.

## Threat Flags

None — no new network/auth/file surface. The only trust boundary is the validated `cd_solve(n, d, l1_reg, l2_reg)` geometry, mitigated exactly as the threat register specified: T-05-05-01 (validate `n*d==x.len()` & `y.len()==n` → `ShapeMismatch` before unsafe launch), T-05-05-02 (`norm2_cols[j]==0` skip avoids divide-by-zero, `max_iter` cap avoids silent NaN), T-05-05-03 (exactly one scalar gap readback per outer check, buffers reused).

## Self-Check: PASSED

- All modified files verified present (coordinate.rs kernels, prims/coordinate_descent.rs, cd_test.rs, this SUMMARY).
- Both task commits verified in git history (`26ed768`, `b635383`).
- `cargo test --features cpu -p mlrs-backend --test cd_test` 5/5 green (incl. f64 + exact sparsity on lasso + elastic_net); `cargo build -p mlrs-kernels` + `-p mlrs-backend --features rocm --tests` green; `lib.rs`/`prims/mod.rs` untouched.
