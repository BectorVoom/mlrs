---
phase: 10-sgd-linear-svm
plan: 04
subsystem: algos
tags: [linear-svm, lbfgs, squared-hinge, squared-epsilon-insensitive, intercept-scaling, oracle, builder-pattern]

requires:
  - phase: 05-iterative-linear
    provides: lbfgs_minimize prim (host L-BFGS over a closure), logistic.rs classes_ ±1 remap + precision-floor accept, elastic_net::predict_linear, gemm transa
  - phase: 10-sgd-linear-svm
    plan: 01
    provides: LinearSVC/SVR builder scaffold + SgdConfig lowering target + BuildError enum + 12 pinned oracle fixtures
provides:
  - LinearSVC (SGDSVM-03) — Fit (classes_ ±1 remap, dual='auto' internal, C→data-term weight) + PredictLabels, intercept via synthetic-feature intercept_scaling
  - LinearSVR (SGDSVM-04) — Fit (squared-eps-insensitive) + Predict via predict_linear, same synthetic-feature intercept
  - Shared pub(crate) svm_lbfgs_fit helper (squared-hinge AND squared-eps reuse via a per-sample margin-loss closure)
  - BuildError::InvalidC variant (construction-time C>0 guard, sibling of AlgoError::InvalidC)
  - Q1 RESOLUTION — the SVM losses are L-BFGS, NOT cd_fit (documented below)
affects: [10-05-pyo3-wrap]

tech-stack:
  added: []
  patterns:
    - "Linear-SVM primal solved by the validated 05-06 lbfgs_minimize prim host-orchestrated over the device matvec (the logistic.rs L-BFGS precedent), NOT cd_fit and NOT the SGD prim (Q1)"
    - "Synthetic-feature intercept_scaling: append a constant column = intercept_scaling, solve with no separate bias, recover intercept_ = intercept_scaling·w_last (Pitfall 5 — NOT center-then-solve)"
    - "f32 convex-minimum precision-floor accept (k·sqrt(eps_F)) mirrors logistic.rs — gtol unreachable in f32, accept the line-search/cap stop at the dtype floor"

key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/error.rs
    - crates/mlrs-algos/src/linear/linear_svc.rs
    - crates/mlrs-algos/src/linear/linear_svr.rs
    - crates/mlrs-algos/tests/linear_svc_test.rs
    - crates/mlrs-algos/tests/linear_svr_test.rs

key-decisions:
  - "Q1 RESOLVED — option (b): the SVM squared-hinge / squared-eps-insensitive primals are SMOOTH + CONVEX but are NOT the Lasso/ElasticNet soft-threshold CD objective (squared-error vs hinge loss), so they reuse the validated lbfgs_minimize prim, NOT cd_fit. A thin shared svm_lbfgs_fit helper host-orchestrates the margin matvec over the device GEMM."
  - "An early Python/scipy spike against the pinned fixtures confirmed the L2-regularized squared-hinge (SVC) / squared-eps-insensitive (SVR) primal reproduces sklearn's coef_/intercept_ and EXACT predict labels BEFORE writing Rust."
  - "f64 coef/intercept oracle uses a 2e-4 BAND (not strict 1e-5): liblinear (the oracle) stops at its own tol=1e-4 slightly short of the true optimum, so the deeper-converged L-BFGS optimum agrees to ~1e-4. EXACT predict labels are the strict hard gate (SVC); SVR predict within the same band."

requirements-completed: [SGDSVM-03, SGDSVM-04]

duration: 60min
completed: 2026-06-21
---

# Phase 10 Plan 04: Linear-SVM Estimators (LinearSVC / LinearSVR) Summary

**Wired both linear-SVM estimators by RESOLVING Open Question Q1 — the squared-hinge / squared-epsilon-insensitive primals are smooth+convex but NOT the Lasso/ElasticNet soft-threshold CD objective, so they reuse the validated 05-06 `lbfgs_minimize` prim (a thin shared `svm_lbfgs_fit` helper host-orchestrated over the device matvec, the `logistic.rs` L-BFGS precedent), with the intercept handled by the synthetic-feature `intercept_scaling` mechanism and `dual='auto'` resolved internally — LinearSVC matches its pinned sklearn oracle on EXACT predict labels (hard gate) and coef/intercept within a documented band, LinearSVR matches coef/intercept/predict within band.**

## Open Question Q1 — RESOLVED (option b: L-BFGS, not cd_fit)

The plan and the Wave-0 scaffold hypothesized that the shared Lasso/ElasticNet `cd_fit` (soft-threshold coordinate descent) could express the SVM objective via "CD-reuse on the synthetic-feature-augmented design." **It cannot.** `cd_fit`'s `cd_solve` primitive minimizes the SQUARED-ERROR data term `½‖y − Xβ‖²` with L1/L2 penalties; sklearn's `LinearSVC`/`LinearSVR` minimize the L2-regularized **squared-hinge** / **squared-epsilon-insensitive** primals:

- LinearSVC: `½‖w‖² + C·Σᵢ max(0, 1 − yᵢ·(xᵢ·w))²`
- LinearSVR: `½‖w‖² + C·Σᵢ max(0, |yᵢ − xᵢ·w| − ε)²`

These are a different per-coordinate update entirely (hinge / epsilon-tube, not least squares). However both are SMOOTH (C¹) and strictly CONVEX, so the natural converged-optimum solver is **option (b) from the RESEARCH recommendation — a thin SVM solver host-orchestrated over the device dot/axpy — realized on the already-validated 05-06 `lbfgs_minimize` primitive**, exactly the `LogisticRegression` (05-10) precedent (a smooth convex objective minimized by the generic host L-BFGS over a closure). NOT the SGD prim (which only approaches the optimum) and NOT `cd_fit`.

**Spike evidence (run BEFORE writing Rust, per the plan's SPIKE-Q1-FIRST instruction):** a Python/scipy `L-BFGS-B` minimization of the exact primal (with the synthetic-feature intercept) against each pinned fixture reproduced sklearn's results:
- LinearSVC: coef max-abs-diff 8.2e-5, intercept diff ~1e-6, **predict labels EXACT match**.
- LinearSVR: coef max-abs-diff 1.7e-5, intercept diff ~1e-6, predict max-abs-diff 3.3e-5.

The ~1e-4 coef residual is because liblinear (the oracle) stops at its own `tol=1e-4` slightly short of the true minimum; the deeper-converged L-BFGS optimum lands ~1e-4 away. This is why the f64 oracle uses a 2e-4 BAND for coef/intercept (both are valid near-optimum iterates of the same strictly-convex objective), while the EXACT predict labels are the strict hard gate.

## What Was Built

**Task 1 — LinearSVC (commit f1eeced):** the L2-regularized squared-hinge primal Fit + PredictLabels. `Fit` copies the `logistic.rs` `classes_` distinct-sorted ±1 remap (Pitfall 4, exactly-2-classes binary), resolves `dual='auto'` INTERNALLY (`n_samples < n_features` → dual else primal; the fixtures resolve to primal; never a builder knob, D-07), maps `C` to the data-term weight, and delegates to the new shared `svm_lbfgs_fit` helper with a squared-hinge per-sample `(loss_i, dloss/dmargin)` closure. `PredictLabels` is the margin-sign → `classes_[·]` roundtrip via the on-device matvec GEMM. The `svm_lbfgs_fit` helper (`pub(crate)`, reused by SVR): appends the synthetic `intercept_scaling` column (Pitfall 5), runs `lbfgs_minimize` over the augmented weight vector with the objective `½‖w‖² + C·Σℓ` (margin = `X̃·w` GEMM, gradient `w + C·X̃ᵀg` second GEMM transa), and recovers `intercept_ = intercept_scaling·w_last` (NOT center-then-solve). `build()` validates `C>0` (new `BuildError::InvalidC`), the classifier loss family, and l1/l2-only penalty. f32 carries a convex-minimum precision-floor accept (`0.5·sqrt(eps_F)`) mirroring `logistic.rs` (f32 gtol is unreachable; the line search breaks down at the dtype floor — accepted as converged).

**Task 2 — LinearSVR (commit ba376cf):** the L2-regularized squared-epsilon-insensitive primal Fit + Predict, REUSING the shared `svm_lbfgs_fit` with only a different per-sample margin-loss closure (residual `r = y − margin`, `viol = max(0, |r| − ε)`, `ℓ = viol²`, `dℓ/dmargin = −2·sign(r)·viol`). `Predict` delegates to the shared `elastic_net::predict_linear` `X·coef_ + intercept_` GEMM path (the regressor precedent). `build()` validates `C>0`, `epsilon>=0`, the regressor loss family, and l1/l2-only penalty.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan's cd_fit-reuse premise was incorrect — L-BFGS is the correct solver**
- **Found during:** Task 1 (the SPIKE-Q1-FIRST step)
- **Issue:** The plan's `key_links`, `artifacts_produced`, and `action` all specified "CD-reuse (cd_fit on the augmented design)". The spike proved `cd_fit`'s soft-threshold squared-error CD cannot express the squared-hinge / squared-eps-insensitive objective (a different loss). Proceeding with cd_fit would have produced a WRONG solution that fails the oracle.
- **Fix:** Resolved Q1 with option (b) from the RESEARCH recommendation — the smooth+convex SVM primals reuse the validated `lbfgs_minimize` prim via the shared `svm_lbfgs_fit` helper (host-orchestrated over the device matvec GEMM). This is the documented design choice the plan's Q1 spike was meant to settle.
- **Files modified:** crates/mlrs-algos/src/linear/linear_svc.rs, linear_svr.rs
- **Commits:** f1eeced, ba376cf

**2. [Rule 2 - Missing functionality] BuildError::InvalidC variant**
- **Found during:** Task 1
- **Issue:** The plan requires `build()` to reject `C<=0` with a `BuildError`, but `BuildError` had no `InvalidC` variant (only `InvalidAlpha`/`InvalidEta0`/`InvalidEpsilon`). The plan said "InvalidC-equivalent BuildError".
- **Fix:** Added a dedicated `BuildError::InvalidC` variant (the construction-time sibling of the existing `AlgoError::InvalidC`) to `error.rs`. Additive/safe in sequential execution. The single `build_err_to_py` mapper (10-05) covers it as a `PyValueError` like every other `BuildError`.
- **Files modified:** crates/mlrs-algos/src/error.rs
- **Commit:** f1eeced

**3. [Rule 3 - Blocking] f64 oracle uses a documented band, not strict 1e-5**
- **Found during:** Task 1
- **Issue:** liblinear (the oracle generator) stops at its own `tol=1e-4` slightly short of the true squared-hinge optimum, so the deeper-converged L-BFGS optimum disagrees with the fixture coef by ~8e-5 (>1e-5). A strict 1e-5 assert would false-fail a correct solver.
- **Fix:** The f64 coef/intercept oracle uses a 2e-4 BAND (both iterates are valid near-optima of the same strictly-convex objective); the EXACT predict labels are the strict hard gate for SVC, SVR predict is within the same band. Documented in the test file and per RESEARCH Pitfall 6 (coef band, exact labels strict).
- **Files modified:** crates/mlrs-algos/tests/linear_svc_test.rs, linear_svr_test.rs
- **Commits:** f1eeced, ba376cf

### Notes (not deviations)

- **The shared `svm_lbfgs_fit` lives in `linear_svc.rs` as `pub(crate)`** (not a new `svm.rs` file) so the plan's declared file set (the 4 estimator/test files + the additive `error.rs` variant) is honored — adding a new module file would require editing `linear/mod.rs` (out of the plan's file boundary).
- **`dual='auto'` is computed but unused for routing** — the primal optimum equals the dual optimum for this convex problem, so mlrs always solves the primal; the flag is resolved internally only for fidelity to sklearn's resolution rule (and a future sparse/dual extension), never exposed as a builder setter (D-07).

## Authentication Gates

None.

## Known Stubs

None. Both estimators are fully fitted/predicting against their pinned oracle. (The `#[pyclass]` registration on `_mlrs` is the Wave-3 / plan 10-05 job per the 10-01 scaffold, not a stub of this plan.)

## Verification Evidence

- `cargo build -p mlrs-algos --features cpu` — exit 0
- `cargo test -p mlrs-algos --features cpu --test linear_svc_test` — 6/6 green: `exact_labels` (HARD gate, f64) + `exact_labels_f32`, `oracle` (coef/intercept band, f64) + `oracle_f32`, `default_matches_sklearn` (D-03), `build_rejects_bad_hyperparams`
- `cargo test -p mlrs-algos --features cpu --test linear_svr_test` — 5/5 green: `oracle` (coef/intercept/predict band, f64) + `oracle_f32`, `default_matches_sklearn` (D-03), `fixture_loads`, `build_rejects_bad_hyperparams`
- `clippy -p mlrs-algos --features cpu --tests` — no warnings from linear_svc.rs / linear_svr.rs / their tests (the only clippy error is the pre-existing `FRAC_PI_2` in mlrs-kernels elementwise.rs:282, logged to deferred-items.md by [10-02], out of scope)
- Grep gates: `intercept_scaling` present in linear_svc.rs (22 occurrences) incl. the synthetic-feature recovery; `predict_linear` delegation present in linear_svr.rs; `exact_labels` present in the SVC test
- Every f64 oracle case references `skip_f64_with_log`; f32 cases carry a documented band

## Self-Check: PASSED

All modified files verified present on disk; both task commits (f1eeced, ba376cf) verified in git history.
