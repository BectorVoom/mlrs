---
phase: 04-closed-form-estimators
verified: 2026-06-12T00:00:00Z
status: passed
score: 4/4
overrides_applied: 0
re_verification: null
gaps: []
deferred: []
human_verification: []
---

# Phase 4: Closed-Form Estimators — Verification Report

**Phase Goal:** A data scientist can fit LinearRegression, Ridge, PCA, and TruncatedSVD and get results matching scikit-learn within 1e-5, exercising the full Arrow→kernel→device-state→materialize→oracle pipeline with no convergence risk.
**Verified:** 2026-06-12T00:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | LinearRegression (SVD-based) fits and exposes coef_/intercept_, predicting within 1e-5 of scikit-learn on random data (cpu(f64) gate; f64-on-rocm skips-with-log per D-07) | VERIFIED | `cargo test -p mlrs-algos --features cpu --test linear_regression_test` → 6/6 passed (f32+f64 coef/intercept/predict + near-collinear cutoff). Struct exists in `linear_regression.rs`, uses `svd()` + `gemm()` pseudo-inverse, RCOND=1e-6 cutoff matches sklearn.linear_model.LinearRegression.tol. Device-resident fitted state confirmed. |
| 2 | Ridge with alpha penalty produces coef_/intercept_ matching scikit-learn within 1e-5 (cpu(f64)+rocm(f32)) | VERIFIED | `cargo test -p mlrs-algos --features cpu --test ridge_test` → 5/5 passed (alpha sweep {0.1,1.0,10.0}, intercept-not-penalized, predict-consistency). `ridge.rs` calls `cholesky_solve()` (NOT svd), gemm(transa=true) for raw Gram, diagonal-only alpha inject. Cholesky primitive standalone green: `cargo test -p mlrs-backend --features cpu --test cholesky_test` → 6/6. |
| 3 | PCA with n_components exposes components_, explained_variance_, explained_variance_ratio_, singular_values_, mean_, transform, inverse_transform — matching scikit-learn after svd_flip sign alignment | VERIFIED | `cargo test -p mlrs-algos --features cpu --test pca_test` → 10/10 passed (tall+wide cases, f32+f64, all six attributes + transform + inverse_transform). `pca.rs`: centered-X SVD, S²/(n-1) formula, align_rows svd_flip, ratio over full spectrum before truncation. All six named attributes present as DeviceArray fields with host accessors. |
| 4 | TruncatedSVD (no centering) exposes components_/explained_variance_/singular_values_/transform, matching sklearn arpack path after sign alignment | VERIFIED | `cargo test -p mlrs-algos --features cpu --test truncated_svd_test` → 6/6 passed (f32+f64, all four attributes + transform). `truncated_svd.rs`: UNCENTERED SVD, var(transform cols) explained_variance_, per-feature variance denominator for ratio. Fixtures generated with algorithm='arpack'. |

**Score:** 4/4 truths verified

### Deferred Items

None.

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-algos/src/traits.rs` | Fit/Predict/Transform trait definitions (D-04) | VERIFIED | Contains `trait Fit`, `trait Predict`, `trait Transform` generic over `<F: Float + CubeElement + Pod>`; `inverse_transform` has default impl returning `Unsupported`. |
| `crates/mlrs-algos/src/error.rs` | AlgoError with InvalidNComponents, InvalidAlpha, NotFitted, Unsupported, #[from] PrimError | VERIFIED | All five variants present; thiserror; `#[from] PrimError` wrap. |
| `crates/mlrs-algos/src/linear/linear_regression.rs` | LinearRegression<F> estimator (Fit + Predict) | VERIFIED | 390 lines; struct + impl Fit + impl Predict; SVD pseudo-inverse; RCOND=1e-6; center-then-solve intercept; device-resident coef_/intercept_. |
| `crates/mlrs-algos/src/linear/ridge.rs` | Ridge<F> estimator (Fit + Predict, Cholesky) | VERIFIED | 394 lines; struct + impl Fit + impl Predict; cholesky_solve; raw Gram via gemm(transa=true); diagonal-alpha; center-then-solve intercept; device-resident. |
| `crates/mlrs-algos/src/decomposition/pca.rs` | Pca<F> estimator (Fit + Transform + inverse_transform) | VERIFIED | 405 lines; all six sklearn attributes; align_rows svd_flip; S²/(n-1); full-spectrum ratio denominator; transform + inverse_transform implemented. |
| `crates/mlrs-algos/src/decomposition/truncated_svd.rs` | TruncatedSvd<F> estimator (Fit + Transform) | VERIFIED | 331 lines; four attributes; UNCENTERED SVD; var(transform cols); inverse_transform left as trait default (Unsupported). |
| `crates/mlrs-kernels/src/cholesky.rs` | #[cube(launch)] cholesky_solve kernel | VERIFIED | Single-cube, all-shared-memory, in-kernel factor+forward+back. Negative-pivot guard writes info_out flag (never NaN). No continue keyword. SharedMemory sized to comptime MAX_DIM. |
| `crates/mlrs-backend/src/prims/cholesky.rs` | cholesky_solve host wrapper + validate-before-launch | VERIFIED | Validates geometry (NotSquare, ShapeMismatch) before any unsafe launch. out=Some Gram buffer reuse (D-11 gate 2). NotPositiveDefinite on non-SPD pivot. |
| `tests/fixtures/` | 14 committed f32/f64 .npz oracle fixtures (all five estimators/primitive) | VERIFIED | Files confirmed present: cholesky_{f32,f64}_seed42.npz, linear_regression_{f32,f64}_seed42.npz, ridge_{f32,f64}_seed42.npz, pca_{,tall_,wide_}{f32,f64}_seed42.npz (6 files), truncated_svd_{f32,f64}_seed42.npz. fixture_loads test confirms keys/shapes valid. |
| `crates/mlrs-algos/tests/{linear_regression,ridge,pca,truncated_svd}_test.rs` | Activated oracle tests (no #[ignore]) | VERIFIED | All four test files activated. 6+5+10+6=27 oracle test functions green. No #[ignore] remains in any of the four files. |
| `crates/mlrs-backend/tests/cholesky_test.rs` | Cholesky standalone validation (6 tests) | VERIFIED | All 6 tests pass: solve invariant, factor invariant, non-SPD guard, fixture_loads — on cpu (f32+f64) gate. |
| `crates/mlrs-backend/tests/memory_gate_test.rs` | D-03 estimator pipeline memory gate (3 new gates) | VERIFIED | 9/9 gates pass on cpu gate: Gate A (bounded reuse), Gate B (Ridge Gram reuse), Gate C (read_backs=0 mid-pipeline, ==1 terminal). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `linear_regression.rs` | `mlrs_backend::prims::svd` | `svd::<F>(pool, &x_c_dev, ...)` call at line 224 | VERIFIED | Thin SVD of centered X; NOT Cholesky (D-02 constraint honored). |
| `linear_regression.rs` | `mlrs_backend::prims::reduce` | `column_reduce(.., ScalarOp::Mean, ReducePath::Shared)` at line 209 | VERIFIED | Key-link prim call present (note: WR-04 advisory — result discarded; see Anti-Patterns). |
| `ridge.rs` | `mlrs_backend::prims::cholesky` | `cholesky_solve::<F>(pool, &gram, &xty, n_features, 1, Some(gram_out))` at line 280 | VERIFIED | Cholesky normal-equations solve; Gram buffer threaded through out (D-11 gate 2). |
| `ridge.rs` | `mlrs_backend::prims::gemm` | `gemm::<F>(pool, &x_c_dev, (n_features, n_samples), &x_c_dev, ..., true, false, None)` at line 228 | VERIFIED | Raw Gram via gemm(transa=true); NOT prims::covariance (RESEARCH Open Q1 honored). |
| `pca.rs` | `mlrs_backend::prims::svd` | `svd::<F>(pool, &x_c_dev, (n_samples, n_features))` at line 208 | VERIFIED | SVD of CENTERED X (D-01); NOT eig-of-covariance. |
| `pca.rs` | `mlrs_core::sign_flip::align_rows` | `align_rows(&vt_rows)` at line 235 | VERIFIED | Estimator-side svd_flip; primitive stays raw (D-01/D-03). |
| `truncated_svd.rs` | `mlrs_backend::prims::svd` | `svd::<F>(pool, x, (n_samples, n_features))` at line 174 | VERIFIED | SVD of UNCENTERED X (Pitfall 2 difference); sign canonicalized via align_rows. |
| `mlrs-backend::prims::cholesky` | `mlrs-kernels::cholesky_solve` | `cholesky_solve_kernel::launch(...)` | VERIFIED | cholesky.rs imports `use mlrs_kernels::cholesky_solve as cholesky_solve_kernel`. |
| `mlrs-backend::prims::cholesky` | `PrimError::NotPositiveDefinite` | info-array read, `info[0] < 0.0` guard | VERIFIED | NotPositiveDefinite returned on non-negative pivot (confirmed by cholesky_rejects_non_spd test passing). |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `linear_regression.rs` | `coef_` | SVD pseudo-inverse: `gemm(vt, (n_features, k), t2_dev, (k, 1), true, false, None)` | Yes — device computation from real fixture data; oracle test asserts within 1e-5 | FLOWING |
| `ridge.rs` | `coef_` | `cholesky_solve(pool, &gram, &xty, n_features, 1, Some(gram_out))` | Yes — Cholesky solve on real Gram+RHS; oracle test asserts within 1e-5 | FLOWING |
| `pca.rs` | `components_` / `explained_variance_` / `mean_` | `svd(pool, &x_c_dev, ...)` + host S²/(n-1) + `align_rows` | Yes — all six attributes derive from real SVD results; 10 oracle tests pass | FLOWING |
| `truncated_svd.rs` | `components_` / `explained_variance_` / `singular_values_` | `svd(pool, x, ...)` + host var(transform cols) + `align_rows` | Yes — all four attributes derive from real SVD results; 6 oracle tests pass | FLOWING |

### Behavioral Spot-Checks

Tests were run directly (runnable environment, cpu(f64) gate):

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| LinearRegression coef_/intercept_/predict oracle (f32+f64, incl. collinear) | `cargo test -p mlrs-algos --features cpu --test linear_regression_test` | 6 passed, 0 failed | PASS |
| Ridge coef_/intercept_ oracle (alpha sweep + predict) | `cargo test -p mlrs-algos --features cpu --test ridge_test` | 5 passed, 0 failed | PASS |
| PCA 10-test oracle (tall+wide, all attributes, transform+inverse) | `cargo test -p mlrs-algos --features cpu --test pca_test` | 10 passed, 0 failed | PASS |
| TruncatedSVD 6-test oracle (all attributes + transform) | `cargo test -p mlrs-algos --features cpu --test truncated_svd_test` | 6 passed, 0 failed | PASS |
| Cholesky standalone invariants (‖A·x-b‖, ‖L·Lᵀ-A‖, non-SPD guard) | `cargo test -p mlrs-backend --features cpu --test cholesky_test` | 6 passed, 0 failed | PASS |
| Memory gate (D-03 estimator pipeline: bounded reuse / Gram reuse / no mid-pipeline readback) | `cargo test -p mlrs-backend --features cpu --test memory_gate_test` | 9 passed, 0 failed | PASS |

### Probe Execution

No conventional `scripts/*/tests/probe-*.sh` probes declared for this phase. Verification via direct `cargo test` spot-checks above.

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| LINEAR-01 | 04-03 | User can fit LinearRegression (OLS, SVD-based) and read coef_/intercept_, predicting within 1e-5 of scikit-learn | SATISFIED | 6/6 oracle tests pass cpu(f64); RCOND=1e-6 sklearn-pinned cutoff; near-collinear case green; REQUIREMENTS.md updated with "Complete" status. |
| LINEAR-02 | 04-02, 04-05 | User can fit Ridge with alpha penalty and obtain coef_/intercept_ matching scikit-learn | SATISFIED | Cholesky primitive standalone: 6/6. Ridge oracle: 5/5 across alpha={0.1,1,10}. Intercept-not-penalized test confirms D-05. REQUIREMENTS.md updated with "DONE" status. |
| DECOMP-01 | 04-04 | User can fit PCA with n_components, read all six attributes, transform/inverse_transform, matching scikit-learn after sign alignment | SATISFIED | 10/10 oracle tests pass cpu(f64); all six attributes verified in code and oracle output; C-contiguous fixture fix applied. REQUIREMENTS.md updated with "Complete" status. |
| DECOMP-02 | 04-04 | User can fit TruncatedSVD, read components_/explained_variance_/singular_values_/transform, matching sklearn arpack path after sign alignment | SATISFIED | 6/6 oracle tests pass cpu(f64); arpack fixtures confirmed deterministic (algorithm='arpack' in gen_oracle.py); distinct from PCA (no centering, var(transform) formula). REQUIREMENTS.md updated with "Complete" status. |

---

## Anti-Patterns Found

| File | Location | Pattern | Severity | Impact |
|------|----------|---------|----------|--------|
| `linear_regression.rs` | Lines 209-219 | Dead `column_reduce` + `to_host` (WR-04) | WARNING | A `column_reduce(ScalarOp::Mean)` + `to_host` + `release_into` is called on the zero-mean centered design but the result is immediately discarded. This is an extra kernel launch + device→host copy per `fit`. It inflates runtime but does NOT inflate the `read_backs` PoolStats counter (which counts only metered `to_host_metered` calls, not plain `to_host`). Gate C (`read_backs==0 mid-pipeline`) still passes. The code comment explicitly labels it a "key-link prim call" retained for documentation purposes — the load-bearing means come from the host two-pass loop above. |
| `ridge.rs` | Lines 213-223 | Same dead `column_reduce` + `to_host` (WR-04 sibling) | WARNING | Identical pattern to linear_regression.rs. Same impact and same reason. |
| `pca.rs` | Lines 162-170 | `n_samples ≤ 1` guard misuses `PrimError::ShapeMismatch` with a non-shape failure (WR-02) | WARNING | Error message will say "rows*cols != len" when they DO match — misleading to callers. Does not affect correctness of the happy path. |
| `pca.rs` / `truncated_svd.rs` | Lines 219-223 / 230-234 | Zero-variance guard: `total_var.abs()` is redundant since `total_var >= 0`; degenerate input silently returns ratio=0 instead of error (WR-03) | WARNING | `.abs()` is dead code; all-constant-column input produces ratio=0 without surfacing an error. Degenerate inputs are not exercised in the oracle tests. |
| `mlrs-kernels/src/cholesky.rs` | ~line 107, 157 | Scale-blind absolute pivot floor `1e-12` (WR-01) | WARNING | The non-SPD guard uses an absolute threshold rather than `eps*max_diag`. All committed fixtures are well-conditioned (SPD by construction), so this is latent. Does not affect the oracle test outcomes. |
| `crates/mlrs-algos/src/error.rs` | Line 29 doc comment | `InvalidNComponents` doc cites "(D-06 — v1 takes an int...)" but D-06 in the phase summaries refers to the "no transpose buffers" decision (IN-03) | INFO | Wrong decision tag in a doc comment. Does not affect runtime behavior. |
| All estimator files | Various | `host_to_f64`/`f64_to_host` helpers duplicated verbatim in 6 files (IN-01) | INFO | Code duplication, no correctness risk. |

**Assessment of WR-04 against the D-03 device-residency goal:**

The dead `column_reduce + to_host` calls are on the HOT PATH (each `fit` call) and add a kernel launch + device→host copy that the phase's D-03 device-residency principle discourages. However, they do NOT:
- break the oracle results (all tests pass within 1e-5)
- inflate the `read_backs` PoolStats counter (unmetered path)
- prevent Gate C from passing

The impact is a performance overhead on every `fit` call, not a correctness failure. The REVIEW already flags this as WR-04 advisory. This verifier records it as a WARNING per the advisory instructions — NOT a phase blocker — and defers the fix to a later cleanup or Phase 5 estimator improvements.

**WR-01, WR-02, WR-03** are robustness and API-clarity warnings. None invalidate the 1e-5 oracle contract on the committed fixtures. The REVIEW explicitly notes all fixtures are well-conditioned, so WR-01's latent behavior is not exercised. These are also advisory — NOT blockers for the phase goal.

**No blockers were found.**

---

## No-Convergence Verification

The phase goal requires "no convergence risk." Verification confirms:

- **LinearRegression**: direct composition of `svd()` + `gemm()` — one-shot deterministic operations. No iterative loop at the estimator level. The SVD primitive itself uses Jacobi sweeps but was validated in Phase 3 (hard gate) and converges by definition within bounded sweeps.
- **Ridge**: `cholesky_solve()` — single-pass Cholesky-Banachiewicz + forward/back substitution; no iterative loop, no convergence test.
- **PCA**: `svd()` of centered X — same as LinearRegression re Jacobi; estimator level is one-shot.
- **TruncatedSVD**: `svd()` of uncentered X — same.

No `while` / `loop` / convergence-check constructs appear in any estimator source file at the estimator composition level.

---

## Human Verification Required

None. All four success criteria are machine-verifiable on the cpu(f64) gate (as specified in verification instructions), and all were verified by direct `cargo test` execution with confirmed pass results. The rocm(f32) gate was documented as green by the executor (SUMMARY.md) but is not independently re-run here because ROCm hardware is not confirmed in this environment — this is noted but does not affect the cpu(f64) gate status per phase D-07.

---

## Gaps Summary

No gaps. All four success criteria are verified:

1. LinearRegression: 6/6 oracle tests pass cpu(f64), including the near-collinear SVD cutoff case that distinguishes sklearn's RCOND=1e-6 from numpy's default.
2. Ridge: 5/5 oracle tests pass cpu(f64), Cholesky primitive standalone 6/6, memory gate 9/9.
3. PCA: 10/10 oracle tests pass cpu(f64), all six sklearn attributes confirmed in code and asserted in tests, transform + inverse_transform implemented.
4. TruncatedSVD: 6/6 oracle tests pass cpu(f64), four attributes confirmed, arpack determinism ensured by committed fixtures.

Advisory code review findings (WR-01 through WR-06, IN-01 through IN-05) are recorded above. None threaten the phase goal. The most material (WR-04 dead column_reduce + to_host) is a performance inefficiency, not a correctness defect, and does not affect the D-03 PoolStats gate outcome.

---

_Verified: 2026-06-12T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
