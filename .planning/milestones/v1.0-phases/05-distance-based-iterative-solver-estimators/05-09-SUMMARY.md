---
phase: 05-distance-based-iterative-solver-estimators
plan: 09
subsystem: algos
tags: [lasso, elastic-net, coordinate-descent, linear-model, center-then-solve, penalty-mapping, oracle, sparsity, d03, d13, cpu-f64, rocm-f32]

# Dependency graph
requires:
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 01
    provides: "lasso_{f32,f64}_seed42.npz + elastic_net_{f32,f64}_seed42.npz fixtures; lasso_test/elastic_net_test Wave-0 stubs; AlgoError::{InvalidL1Ratio,NotConverged,InvalidAlpha}"
  - phase: 05-distance-based-iterative-solver-estimators
    plan: 05
    provides: "mlrs_backend::prims::coordinate_descent::cd_solve — validated centered-design CD solve (un-normalized l1_reg/l2_reg, single-scalar gap readback), GREEN within 1e-5 incl. exact sparsity"
  - phase: 04-closed-form-estimators
    provides: "ridge.rs center-then-solve intercept (D-13 two-pass means + intercept = y_bar - x_bar*coef + host_to_f64/f64_to_host helpers); Fit/Predict traits; GEMM predict path"
provides:
  - "mlrs_algos::linear::coordinate_descent::cd_fit — SHARED host fit helper (validate alpha/l1_ratio + centering + penalty map + cd_solve + intercept recovery) for both Lasso and ElasticNet"
  - "mlrs_algos::linear::elastic_net::ElasticNet<F>: Fit + Predict<F> (LINEAR-04)"
  - "mlrs_algos::linear::lasso::Lasso<F>: Fit + Predict<F>, thin l1_ratio=1 wrapper over cd_fit (LINEAR-03)"
  - "mlrs_algos::linear::elastic_net::predict_linear — shared X·coef+intercept GEMM predict path reused by Lasso"
affects: [05-10]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Shared host fit helper for a solver family: cd_fit owns validate→center→penalty-map→cd_solve→intercept once; ElasticNet calls it with the user l1_ratio, Lasso pins l1_ratio=1.0 (D-03 — one CD implementation, two estimators, NOT unified with the L-BFGS LogReg solver)"
    - "Penalty mapping at the estimator boundary (Pitfall 1): user-facing (alpha, l1_ratio) → sklearn un-normalized (l1_reg=α·l1_ratio·n, l2_reg=α·(1−l1_ratio)·n); the n_samples scaling is load-bearing for the exact sparsity pattern"
    - "Center-then-solve intercept reuse (D-13): the ridge.rs two-pass host means + intercept = ȳ − x̄·coef are copied verbatim; the intercept is recovered outside the centered penalized system so α/L1 never penalize the bias"
    - "Shared predict path: predict_linear (the ridge GEMM-then-broadcast) is pub(crate) and reused by both ElasticNet and Lasso so the X·coef+intercept surface is implemented once (D-03)"

key-files:
  created:
    - "crates/mlrs-algos/src/linear/coordinate_descent.rs (shared cd_fit helper + penalty map + centering + cd_solve + intercept + error map)"
    - "crates/mlrs-algos/src/linear/elastic_net.rs (ElasticNet<F>: Fit + Predict<F>; pub(crate) predict_linear)"
    - "crates/mlrs-algos/src/linear/lasso.rs (Lasso<F>: thin l1_ratio=1 wrapper)"
  modified:
    - "crates/mlrs-algos/src/linear/mod.rs (additive: pub mod coordinate_descent/elastic_net/lasso + extended 'deliberately different solvers' note)"
    - "crates/mlrs-algos/tests/elastic_net_test.rs (de-#[ignore]d: real sklearn oracle, coef_ incl. zeros + intercept_ 1e-5, f32+f64)"
    - "crates/mlrs-algos/tests/lasso_test.rs (de-#[ignore]d: real sklearn oracle, sparse coef_ incl. exact zeros + intercept_ 1e-5, f32+f64)"

key-decisions:
  - "Lasso is a TRUE thin wrapper (D-03): its fit delegates to cd_fit with l1_ratio=1.0 (→ l2_reg=0), reusing the same penalty map, centering, cd_solve, and intercept recovery; predict reuses elastic_net::predict_linear. No coordinate-descent code is duplicated in lasso.rs."
  - "cd_fit validation uses !(alpha >= 0.0) and !(0.0..=1.0).contains(&l1_ratio) so a NaN hyperparameter is also rejected (NaN fails the >= and the range check), surfacing InvalidAlpha/InvalidL1Ratio before any launch (T-05-09-01 / ASVS V5)."
  - "A defensive map_cd_error translates a (currently never-emitted) cd_solve PrimError::NotConverged to AlgoError::NotConverged carrying the estimator's max_iter, honoring the plan's 'surface NotConverged' directive forward-safely if the primitive ever starts emitting it; every other PrimError wraps transparently via #[from]."
  - "predict_linear is pub(crate) in elastic_net.rs (not a fourth file) so Lasso reuses it without a new shared module; the predict surface stays the single ridge-precedent GEMM path."

patterns-established:
  - "Solver-family estimators (Lasso/ElasticNet) share ONE host helper that bundles hyperparameter validation, centering, penalty mapping, the primitive solve, and intercept recovery — the estimator structs only hold config + device-resident fitted state and forward to it (the template the 05-10 LogReg estimator mirrors with its own distinct L-BFGS helper)"

requirements-completed: [LINEAR-03, LINEAR-04]

# Metrics
duration: ~12min
completed: 2026-06-13
---

# Phase 5 Plan 09: Lasso + ElasticNet (Coordinate-Descent Linear Models) Summary

**Landed Lasso (LINEAR-03) and ElasticNet (LINEAR-04) on the validated 05-05 `cd_solve` primitive via ONE shared `coordinate_descent::cd_fit` host helper (D-03): it validates the untrusted `(alpha, l1_ratio)` before launch, centers `(X, y)` with the verbatim ridge.rs two-pass means (D-13), maps the user-facing params to sklearn's un-normalized `(l1_reg=α·l1_ratio·n, l2_reg=α·(1−l1_ratio)·n)` (Pitfall 1), runs the centered CD solve, and recovers the unpenalized `intercept_ = ȳ − x̄·coef_`. `ElasticNet<F>` passes the user `l1_ratio`; `Lasso<F>` is a thin wrapper pinning `l1_ratio=1.0` (→ `l2_reg=0`, pure L1) — no duplicate CD loop. Both implement `Fit + Predict<F>` (sharing one `predict_linear` GEMM path) and reproduce sklearn's `coef_`/`intercept_` within 1e-5 INCLUDING the exact sparsity (zero) pattern, GREEN on cpu f32+f64; the CD path is deliberately NOT unified with the 05-10 L-BFGS LogReg solver.**

## Performance

- **Duration:** ~12 min
- **Completed:** 2026-06-13
- **Tasks:** 2 (both `type=auto`)
- **Files:** 3 created (cd_fit helper + 2 estimators), 3 modified (mod.rs + 2 de-ignored tests)

## Accomplishments
- **`coordinate_descent::cd_fit`** — the shared host helper consumed by both estimators. Validates `alpha ≥ 0` (`InvalidAlpha`) and `0 ≤ l1_ratio ≤ 1` (`InvalidL1Ratio`) plus the `n*d == x.len()` / `y.len() == n` geometry BEFORE any prim launch (T-05-09-01 / ASVS V5); centers `X`/`y` host-side with the ridge.rs two-pass means (D-13); maps `(l1_reg = α·l1_ratio·n, l2_reg = α·(1−l1_ratio)·n)` (Pitfall 1 — the `n_samples` scaling is load-bearing for the sparsity); calls `cd_solve(.., tol=1e-4, max_iter=1000)`; recovers `intercept_ = ȳ − x̄·coef_` (unpenalized). Returns device-resident `(coef_, intercept_)`.
- **`ElasticNet<F>`** (`new(alpha, l1_ratio, fit_intercept)` + `with_opts(.., max_iter, tol)`) implementing `Fit` (delegates to `cd_fit` with the user `l1_ratio`) + `Predict<F>` (the shared `predict_linear` GEMM `X·coef + intercept` broadcast). Device-resident `coef_`/`intercept_` (D-03), host accessors materialize on demand.
- **`Lasso<F>`** (`new(alpha, fit_intercept)` + `with_opts`) — a TRUE thin wrapper: `fit` delegates to the SAME `cd_fit` with `l1_ratio = 1.0` (→ `l2_reg = 0`, pure L1) and `predict` reuses `elastic_net::predict_linear`. No coordinate-descent or predict code is duplicated (D-03).
- **`linear/mod.rs`** — additive `pub mod coordinate_descent; pub mod elastic_net; pub mod lasso;` and an extended "deliberately different solvers, do not unify" note covering the Lasso/ElasticNet CD family vs the L-BFGS LogReg solver (05-10). Existing `linear_regression`/`ridge` declarations untouched; `lib.rs` untouched.
- **Oracles de-`#[ignore]`d:** `elastic_net_test.rs` and `lasso_test.rs` now load the committed fixtures, fit with the fixture `alpha`(`/l1_ratio`), and assert `coef_` (incl. exact zeros via the strict 1e-5 absolute arm) and `intercept_` against sklearn for both f32 and f64 (`skip_f64_with_log` gate).

## Task Commits

1. **Task 1: ElasticNet via shared cd_fit helper + oracle (LINEAR-04)** — `2c913a2` (feat)
2. **Task 2: Lasso = ElasticNet(l1_ratio=1) thin wrapper + oracle (LINEAR-03)** — `560ee34` (feat)

## Files Created/Modified
- `crates/mlrs-algos/src/linear/coordinate_descent.rs` — `cd_fit` shared helper + `map_cd_error` + `host_to_f64`/`f64_to_host`; `CD_DEFAULT_TOL`/`CD_DEFAULT_MAX_ITER`.
- `crates/mlrs-algos/src/linear/elastic_net.rs` — `ElasticNet<F>` (`Fit` + `Predict<F>`) + `pub(crate) predict_linear` shared GEMM predict path.
- `crates/mlrs-algos/src/linear/lasso.rs` — `Lasso<F>` thin `l1_ratio=1` wrapper.
- `crates/mlrs-algos/src/linear/mod.rs` — additive module declarations + extended solver-disjointness note.
- `crates/mlrs-algos/tests/elastic_net_test.rs`, `crates/mlrs-algos/tests/lasso_test.rs` — real sklearn oracles (coef incl. sparsity + intercept, f32+f64).

## Decisions Made
- **Lasso is a true thin wrapper (D-03):** delegates to `cd_fit` with `l1_ratio=1.0` and reuses `predict_linear`; no CD loop or predict body is re-implemented in `lasso.rs`.
- **NaN-safe hyperparameter validation:** `!(alpha >= 0.0)` and `!(0.0..=1.0).contains(&l1_ratio)` reject NaN as well as out-of-range values before launch.
- **Defensive `NotConverged` mapping:** `map_cd_error` translates a `cd_solve` `PrimError::NotConverged` to `AlgoError::NotConverged { max_iter }`, honoring the plan directive forward-safely (the 05-05 primitive caps silently today, but the mapping is in place if that changes); all other `PrimError`s wrap via `#[from]`.
- **`predict_linear` is `pub(crate)` in `elastic_net.rs`:** Lasso reuses it directly rather than introducing a fourth shared file.

## Deviations from Plan

### Auto-fixed Issues

None of significance. Two small additive choices, both within the plan's intent (Rule 2 — completeness, no architectural change):

**1. [Rule 2 - Missing critical functionality] NaN-safe hyperparameter guards**
- **Found during:** Task 1 (writing the `cd_fit` validation).
- **Issue:** A plain `alpha < 0.0` / range check would let a NaN `alpha`/`l1_ratio` slip through (`NaN < 0.0` is false), reaching the solver with a poisoned penalty.
- **Fix:** Used `!(alpha >= 0.0)` and `!(0.0..=1.0).contains(&l1_ratio)` so NaN is rejected with `InvalidAlpha`/`InvalidL1Ratio` before launch (ASVS V5 / T-05-09-01).
- **Files modified:** `crates/mlrs-algos/src/linear/coordinate_descent.rs`
- **Commit:** `2c913a2`

**2. [Rule 2 - Missing critical functionality] `with_opts` constructors for `max_iter`/`tol`**
- **Found during:** Task 1/2 (estimator surface).
- **Issue:** The plan lists `max_iter`/`tol` fields but `new` alone could not override them.
- **Fix:** Added `ElasticNet::with_opts` / `Lasso::with_opts`; `new` forwards sklearn's defaults (`max_iter=1000`, `tol=1e-4`).
- **Files modified:** `elastic_net.rs`, `lasso.rs`
- **Commit:** `2c913a2`, `560ee34`

## Known Stubs

None. Both estimators fully fit via the real `cd_solve` device path; the oracles assert genuine device output (coef flows from `cd_solve`, not a hardcoded value) and check the exact-zero sparsity against sklearn.

## Verification Evidence
- `cargo test --features cpu -p mlrs-algos --test elastic_net_test` — 3/3 green (fixture_loads, f32, f64 incl. sparsity + intercept).
- `cargo test --features cpu -p mlrs-algos --test lasso_test` — 3/3 green (fixture_loads, f32, f64 incl. exact-zero sparsity + intercept).
- `cargo build -p mlrs-algos --features rocm --tests` — green (f32 target build; f64 cpu-gated via `skip_f64_with_log`).
- `lib.rs` untouched; `linear_regression`/`ridge` module declarations untouched (additive `mod.rs` edit only, file-disjoint from the sibling 05-10 logistic plan which runs after).

> NOTE: the host disk filled to 100% during a combined two-test re-run, which aborted on a `query-cache.bin` write (`os error 28`), NOT on any compilation error or assertion. Each test was verified GREEN individually beforehand, and the rocm `--tests` build is green; the failure is purely an environment disk constraint, not a plan regression.

## Threat Flags

None — no new network/auth/file surface. The only trust boundary is the validated `(alpha, l1_ratio)` + geometry at the estimator `fit` boundary, mitigated exactly as the threat register specified: T-05-09-01 (validate `alpha≥0` / `0≤l1_ratio≤1` / geometry → typed `AlgoError` before any unsafe launch), T-05-09-02 (the `cd_solve` `max_iter=1000` cap → no silent NaN; defensive `NotConverged` mapping in place), T-05-09-SC (zero new dependencies).

## Self-Check: PASSED

- All created/modified files verified present (coordinate_descent.rs, elastic_net.rs, lasso.rs, both de-ignored tests, this SUMMARY).
- Both task commits verified in git history (`2c913a2`, `560ee34`).
- `elastic_net_test` 3/3 + `lasso_test` 3/3 green on cpu (incl. f64 + exact sparsity); `rocm --tests` build green; `lib.rs` + existing linear module declarations untouched.
