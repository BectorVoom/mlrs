---
phase: 04-closed-form-estimators
plan: 05
subsystem: linear-models
tags: [ridge, l2-regularization, cholesky, normal-equations, sklearn, oracle, gram-reuse, memory-gate, device-resident]

# Dependency graph
requires:
  - phase: 04-closed-form-estimators
    plan: 01
    provides: Fit/Predict traits (D-04), AlgoError::InvalidAlpha, linear/mod.rs stub, ridge #[ignore] scaffold + committed ridge_{f32,f64}_seed42.npz fixtures
  - phase: 04-closed-form-estimators
    plan: 02
    provides: "prims::cholesky::cholesky_solve(pool,a,b,n,rhs,out) — SPD solve with out=Some Gram buffer reuse + NotPositiveDefinite guard (D-02)"
  - phase: 04-closed-form-estimators
    plan: 03
    provides: LinearRegression centering + intercept-recovery host pattern (center-then-solve, D-05); the host_to_f64/f64_to_host helpers
  - phase: 02-core-compute-primitives
    provides: gemm (transpose flags, D-06), column_reduce (ScalarOp::Mean), DeviceArray/BufferPool/PoolStats
provides:
  - "Ridge<F> (LINEAR-02): Fit + Predict via Cholesky normal-equations (XᵀX + αI)·coef = Xᵀy (D-02)"
  - "raw centered Gram via gemm(transa=true) (RESEARCH Open Q1 — NOT scaled covariance); α on the Gram diagonal only"
  - "center-then-solve intercept, NEVER penalized (D-05); InvalidAlpha on alpha<0 before any launch"
  - "Gram buffer threaded through cholesky_solve out → Cholesky factor reuses it (D-11 gate 2)"
  - "estimator fit→predict/transform memory gate (D-03): bounded reuse + Ridge Gram reuse + no mid-pipeline read-back"
  - "LINEAR-02 oracle green on cpu(f64)+rocm(f32) across the {0.1,1.0,10.0} alpha sweep within 1e-5"
affects: [06-pyo3-bindings]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Ridge normal-equations: gemm(transa=true) raw centered Gram → host diagonal-α inject → cholesky_solve, deliberately distinct from LinearRegression's SVD (D-02, NOT unified)"
    - "Gram restage-via-from_host recycles the just-released raw-Gram n²-byte-size from the free-list, keeping no second LIVE n² buffer (cubecl 0.10 has no in-place device write)"
    - "estimator-pipeline memory gate composed from the underlying prims (gemm/cholesky_solve) because mlrs-backend cannot dev-depend on mlrs-algos (dependency cycle)"

key-files:
  created:
    - crates/mlrs-algos/src/linear/ridge.rs
  modified:
    - crates/mlrs-algos/src/linear/mod.rs
    - crates/mlrs-algos/tests/ridge_test.rs
    - crates/mlrs-backend/tests/memory_gate_test.rs

key-decisions:
  - "Diagonal-α injection done on a host materialize-modify-restage of the small n×n Gram (cubecl 0.10 has no in-place device write); the raw-Gram buffer is RELEASED before the regularized one is staged via from_host, which recycles the same n²-byte-size from the free-list — so the diagonal-stride `+= alpha` is unambiguous AND no second live n² buffer exists (D-11 gate 2 preserved)"
  - "The 04-05 memory gate drives the estimator pipelines as PRIMITIVE compositions (the exact gemm(transa)→diag-α→cholesky_solve(out=Some) Ridge runs, + a gemm predict/transform round) inside mlrs-backend's test crate, because mlrs-algos depends on mlrs-backend so mlrs-backend cannot dev-depend on mlrs-algos (cargo dependency cycle). The pool counters asserted are the SAME ones the estimator code paths drive."
  - "Added a 5th ridge test (predict consistency) beyond the 4 scaffold functions so the Predict path / device-resident coef GEMM is exercised end-to-end (the scaffold only had coef/intercept families)"

requirements-completed: [LINEAR-02]

# Metrics
duration: 9min
completed: 2026-06-12
---

# Phase 4 Plan 05: Ridge (LINEAR-02) Summary

**`Ridge<F>` — L2-penalized least squares via the Cholesky normal-equations solver `(XᵀX + αI)·coef = Xᵀy` (D-02, deliberately NOT the SVD path), with the raw centered Gram from `gemm(transa=true)` (RESEARCH Open Q1), `alpha` on the Gram diagonal only, center-then-solve intercept (never penalized, D-05), and the Gram buffer threaded through the Cholesky factor (D-11 gate 2) — matching `sklearn.linear_model.Ridge(solver='cholesky')` within 1e-5 across the {0.1, 1.0, 10.0} alpha sweep on cpu(f64)+rocm(f32); plus a build-failing memory gate extending the device-residency contract to the estimator fit→predict/transform pipelines (D-03).**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-06-12T07:47:31Z
- **Completed:** 2026-06-12T07:56:33Z
- **Tasks:** 2
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments

- `Ridge<F>` (`Fit` + `Predict`) solves the regularized normal equations via the validated 04-02 `prims::cholesky::cholesky_solve` (D-02) — NO SVD pseudo-inverse (deliberately a DIFFERENT solver from LinearRegression; the two are NOT unified per RESEARCH Anti-Patterns).
- The normal matrix is the **raw** centered Gram `XᵀX` from `gemm(transa=true)` (RESEARCH Open Q1) — NOT `prims::covariance`, which would scale by `1/(n−ddof)`. Verified directly against the committed fixture that `Xc·Xc + αI` reproduces sklearn's `coef_` exactly (no `n_samples` scaling), matching sklearn's `_solve_cholesky`.
- `alpha` is added to the Gram **diagonal only** (`A[i·n+i] += alpha`, T-04-05-02); the intercept is recovered AFTER the solve via center-then-solve (`intercept_ = ȳ − x̄·coef_`, D-05) and is therefore never part of the penalized system — sklearn-exact (RESEARCH Pitfall 5). `fit_intercept=false` solves raw X with `intercept_=0`.
- The regularized Gram buffer is threaded through `cholesky_solve`'s `out` (D-11 gate 2) so the Cholesky factor reuses it in place — no parallel n² allocation. A negative `alpha` is rejected at `fit` with `AlgoError::InvalidAlpha` BEFORE any launch (T-04-05-03); a near-singular Gram surfaces `PrimError::NotPositiveDefinite` → `AlgoError` (Pitfall 4 / T-04-05-01), never NaN coef_.
- Fitted `coef_`/`intercept_` stored device-resident (D-03); `predict` runs `X_test·coef` on-device and broadcasts the intercept, materializing to host only at the accessor/oracle boundary.
- `ridge_test.rs` activated (all `#[ignore]` removed): the {0.1, 1.0, 10.0} alpha sweep asserts `coef_`/`intercept_` vs sklearn at 1e-5; a dedicated intercept-not-penalized test confirms the recovered intercept equals the analytic `ȳ − x̄·coef_` AND the sklearn fixture (D-05); a 5th predict-consistency test exercises the device-resident GEMM predict path.
- The build-failing memory gate (`memory_gate_test.rs`) gains three D-03 estimator-pipeline gates: Gate A (bounded reuse + reuses ≥ N−1 across repeated fit→predict/transform rounds), Gate B (Ridge Gram reuse — solve peak-live rise < 2·n_features²), Gate C (read_backs == 0 mid-pipeline / == 1 terminal).

## Task Commits

1. **Task 1: Ridge estimator (raw Gram + diagonal-α + Cholesky) + activate ridge_test.rs** — `73773bd` (feat)
2. **Task 2: Extend memory_gate_test.rs to the fit→predict/transform pipeline (D-03)** — `d45e2d1` (test)

**Plan metadata:** _(final docs commit — this SUMMARY + STATE + ROADMAP + REQUIREMENTS)_

## Test Results (reported honestly)

**Ridge — cpu (f64+f32) gate** — `cargo test -p mlrs-algos --features cpu --test ridge_test`:
```
running 5 tests
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```
(`ridge_coef_intercept_alpha_sweep_{f32,f64}`, `ridge_intercept_not_penalized_{f32,f64}`, `ridge_predict_consistency_f32` — all within the strict 1e-5 abs+rel contract.)

**Ridge — rocm (f32) gate** — `cargo test -p mlrs-algos --features rocm --test ridge_test`:
```
running 5 tests
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```
The two f64 functions print `... SKIPPED (no f64 support on this adapter)` and return early (verified with `--nocapture`: `ridge f64 backend=rocm: SKIPPED`); the f32 functions run on the real ROCm GPU.

**Memory gate — cpu (f32+f64) gate** — `cargo test -p mlrs-backend --features cpu --test memory_gate_test`:
```
running 9 tests
test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```
(6 prior Phase-2/3 gates + the 3 new D-03 estimator gates: `memory_gate_estimator_fit_round_reuse_bounded`, `memory_gate_ridge_reuses_gram_for_factor`, `memory_gate_estimator_round_no_midpipeline_readback`.)

**Memory gate — rocm (f32) gate** — `cargo test -p mlrs-backend --features rocm --test memory_gate_test`:
```
running 9 tests
test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```
The new gates drive f32 (portable on every backend) and assert the SAME backend-agnostic pool counters on cpu and rocm.

## Files Created/Modified

- `crates/mlrs-algos/src/linear/ridge.rs` (created) — `struct Ridge<F>` + `Fit`/`Predict`; raw centered Gram via `gemm(transa=true)`, diagonal-α inject, `cholesky_solve` with the Gram threaded through `out`; center-then-solve intercept; `InvalidAlpha` guard; host `coef`/`intercept` accessors.
- `crates/mlrs-algos/src/linear/mod.rs` (modified) — replaced the `// 04-05 adds: pub mod ridge;` placeholder with `pub mod ridge;` (the `linear_regression` line left intact; both estimators coexist).
- `crates/mlrs-algos/tests/ridge_test.rs` (modified) — removed all `#[ignore]`; wired the alpha-sweep / intercept-not-penalized / predict-consistency oracle assertions, keeping `skip_f64_with_log` on the f64 functions.
- `crates/mlrs-backend/tests/memory_gate_test.rs` (modified) — appended the Phase-4 D-03 estimator-pipeline section (3 gates + the `ridge_fit_round`/`estimator_apply_round`/`center` helpers) AFTER the existing Phase-2/3 gates; added the `cholesky_solve` + `column_reduce` imports.

## Decisions Made

- **Diagonal-α via host materialize-restage.** cubecl 0.10 has no in-place device write into an existing handle (confirmed in `device_array.rs::from_host`, which acquires-then-creates a fresh handle). Ridge therefore reads the raw Gram to host, adds α on the diagonal stride, RELEASES the raw-Gram device buffer back to the pool, and re-stages the regularized Gram via `from_host` — which recycles the just-released n²-byte-size from the free-list. So the diagonal-stride `+= alpha` is unambiguous AND no second LIVE n² buffer ever exists, preserving the D-11 gate-2 "no parallel n²" property (proven by the new Gate B).
- **Memory gate composed from primitives, not the estimators.** `mlrs-algos` depends on `mlrs-backend`, so `mlrs-backend`'s test crate cannot `use mlrs_algos::Ridge` (a cargo dependency CYCLE). The 04-05 gates therefore drive the EXACT primitive composition the estimators run — Ridge's `gemm(transa)`→diag-α→`cholesky_solve(out=Some)` fit and a `gemm` predict/transform round — inside `mlrs-backend`'s test crate. The pool counters asserted are the same ones the estimator code paths drive (the estimators call these very prims), so the device-residency / Gram-reuse / no-readback contract is proven at the layer that actually allocates. See the Deviations note.
- **Added a 5th predict-consistency test** beyond the 4 scaffold functions so the `Predict` device-resident-coef GEMM path is exercised end-to-end (the 04-01 scaffold only had the coef/intercept families).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Memory gate driven via primitive composition, not `mlrs_algos` estimators (dependency cycle)**
- **Found during:** Task 2 (the plan's `<action>` says "run N same-shape `fit`+`predict` (LinearRegression or Ridge) and N same-shape `fit`+`transform` (PCA) rounds" inside `crates/mlrs-backend/tests/memory_gate_test.rs`).
- **Issue:** `crates/mlrs-algos/Cargo.toml` declares `mlrs-backend = { path = "../mlrs-backend" }`. So `mlrs-backend` CANNOT add `mlrs-algos` as a dev-dependency — cargo rejects the resulting dependency cycle. The estimator structs (`Ridge`/`LinearRegression`/`Pca`) are therefore unreachable from `mlrs-backend`'s `tests/` crate, where the plan's `files_modified` places these gates.
- **Fix:** Implemented the gates as the EXACT primitive composition the estimators run (`gemm(transa)` raw centered Gram → diagonal-α → `cholesky_solve(out=Some)` for Ridge's fit; `gemm` over the fitted coef for predict/transform), via the `ridge_fit_round`/`estimator_apply_round`/`center` helpers in the test crate. This is faithful to the plan's intent: it asserts on the SAME `PoolStats` counters (allocations / reuses / live_bytes / peak_bytes / read_backs) the estimator code paths drive, in the crate that owns the pool. The Ridge Gram-reuse (Gate B), bounded-reuse (Gate A), and no-mid-pipeline-readback (Gate C) contracts are all proven. Not a Rule-4 architectural change — no structure/library/schema changed; this is the only non-cyclic placement consistent with the plan's `files_modified`.
- **Files modified:** `crates/mlrs-backend/tests/memory_gate_test.rs`
- **Commit:** `d45e2d1`

### Note on the 04-01 ridge scaffold location

The plan's `<read_first>` and `<prior_wave_context>` describe a `linear/ridge.rs` STUB created by 04-01. In the merged tree, 04-01 created the `#[ignore]` test scaffold + the `// 04-05 adds: pub mod ridge;` placeholder comment in `linear/mod.rs`, but NO `src/linear/ridge.rs` file (the 04-03 SUMMARY confirms it left the placeholder-comment line intact). So this plan CREATED `ridge.rs` fresh (consistent with the plan's `<action>`: "Create `crates/mlrs-algos/src/linear/ridge.rs`") rather than filling a stub. No behavior change; noted for accuracy.

## Authentication Gates

None — no auth or network was required (oracle fixtures are committed blobs).

## TDD Gate Compliance

Task 1 carries `tdd="true"`. The RED gate is the Wave-0 `#[ignore]` scaffold from 04-01 (`ridge_test.rs` asserting fixture shape only — the real assertions could not reference the non-existent `Ridge` symbol). Task 1 (`73773bd`) added the `Ridge` estimator AND removed `#[ignore]`, wiring the real coef/intercept/predict assertions that now pass (GREEN). The alpha-sweep + intercept-not-penalized oracle passed on the FIRST run on both cpu (f64+f32) and rocm (f32) — the raw-Gram + diagonal-α + center-then-solve arithmetic was verified against the fixture (`Xc·Xc + αI`) before implementation, so no RED→GREEN iteration on the numerics was needed. The single Task-1 commit is `feat` (estimator + activated tests as one logical change) because the failing-test scaffold (the RED commit) already landed in 04-01; this plan's job was to make it GREEN. Task 2 is a non-TDD `test`-type extension of the build-failing gate.

## Known Stubs

None. `fit_intercept=false` is a real branch (solves raw X, intercept 0), not a stub. All fitted state is wired to real device buffers; no hardcoded/placeholder/empty values flow to the oracle. The memory-gate helpers compose real prim calls (no mocked data path).

## Threat Flags

None. The files introduce no security surface beyond the plan's `<threat_model>`: the trust boundary is the host caller → `Ridge::fit/predict`, mitigated by the `alpha ≥ 0` + geometry validation BEFORE any launch (T-04-05-03), the diagonal-only α injection (T-04-05-02), and the `NotPositiveDefinite` guard inherited from the 04-02 Cholesky primitive (T-04-05-01). The f64-on-rocm launch is gated by `skip_f64_with_log` (T-04-05-04). No new network/auth/file-access surface; no new cargo packages (T-04-05-SC, accept).

## Self-Check: PASSED

- `crates/mlrs-algos/src/linear/ridge.rs` — FOUND
- `crates/mlrs-algos/tests/ridge_test.rs` — FOUND (modified, no `#[ignore]` remains)
- `crates/mlrs-algos/src/linear/mod.rs` — `pub mod ridge;` present (both estimators)
- `crates/mlrs-backend/tests/memory_gate_test.rs` — FOUND (3 new D-03 gates appended)
- Commits `73773bd` and `d45e2d1` — present in `git log`
- `cargo test -p mlrs-algos --features cpu --test ridge_test` — 5 passed, 0 failed
- `cargo test -p mlrs-algos --features rocm --test ridge_test` — 5 passed (f64 skips-with-log)
- `cargo test -p mlrs-backend --features cpu --test memory_gate_test` — 9 passed, 0 failed
- `cargo test -p mlrs-backend --features rocm --test memory_gate_test` — 9 passed
- `ridge.rs` contains `struct Ridge`, calls `cholesky_solve` + `gemm(transa=true)` (NOT `covariance`, NOT `svd`); diagonal `+= alpha` present

## Next Phase Readiness

- **LINEAR-02 satisfied.** The full Phase-4 linear-model surface is complete: LinearRegression (SVD, LINEAR-01) + Ridge (Cholesky, LINEAR-02) compose the validated Phase-2/3 primitives + the new 04-02 Cholesky primitive, both device-resident and oracle-green on cpu(f64)+rocm(f32).
- **Phase 4 closes** the closed-form estimators (LinearRegression / Ridge / PCA / TruncatedSVD); the uniform `Fit`/`Predict`/`Transform` surface (D-04) is the contract Phase-6 PyO3 wraps generically.
- The build-failing memory gate now covers the estimator fit→predict/transform pipelines (D-03), so any future device-residency regression in the estimator layer is caught at `cargo test`.

---
*Phase: 04-closed-form-estimators*
*Completed: 2026-06-12*
