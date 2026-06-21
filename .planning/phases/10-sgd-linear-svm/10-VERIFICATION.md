---
phase: 10-sgd-linear-svm
verified: 2026-06-21T00:00:00Z
status: gaps_found
score: 3/4 must-haves verified
overrides_applied: 0
gaps:
  - truth: "MBSGDClassifier with schedules INCLUDING `optimal` matches sklearn within tolerance under the pinned oracle"
    status: failed
    reason: "The `optimal`-schedule classifier fixtures (mbsgd_classifier_optimal_{f32,f64}_seed42.npz) are committed and generated, but no Rust test ever loads them or fits with LearningRate::Optimal against a sklearn reference. All active classifier oracle tests hard-code LearningRate::Constant. The schedule_constant_then_invscaling_then_optimal test in sgd_test.rs exercises only the host arithmetic formula (a tautology against itself), NOT against the sklearn oracle. The Bottou t0 subtleties could diverge from sklearn and the suite stays green. The `default_matches_sklearn` test confirms LearningRate::Optimal is the default, yet no oracle validates it."
    artifacts:
      - path: "crates/mlrs-algos/tests/mbsgd_classifier_test.rs"
        issue: "All three fit paths (fit_hinge, fit_log_proba, default_matches_sklearn) hard-code LearningRate::Constant or only assert the config field value — no test fits with LearningRate::Optimal and asserts predict labels against the optimal-schedule fixture"
      - path: "tests/fixtures/mbsgd_classifier_optimal_f32_seed42.npz"
        issue: "Fixture exists and is well-formed but is never loaded by any Rust test"
      - path: "tests/fixtures/mbsgd_classifier_optimal_f64_seed42.npz"
        issue: "Fixture exists and is well-formed but is never loaded by any Rust test"
    missing:
      - "An oracle_optimal test (both dtypes, skip_f64_with_log gated) that loads mbsgd_classifier_optimal_{f32,f64}_seed42.npz, fits with .learning_rate(LearningRate::Optimal) (no eta0 override), and asserts coef_/intercept_ within the documented band AND predict_labels exactly against the fixture's predict field"
      - "Resolution of the underlying WR-01 L2-penalty ordering discrepancy OR documented justification for the 5e-3 band (order of magnitude above the 1e-5 project contract)"
---

# Phase 10: SGD / Linear-SVM Verification Report

**Phase Goal:** A data scientist can fit minibatch-SGD and linear-SVM estimators built on the single genuinely-new device solver of v2 (SGD prim) — the highest cpu-MLIR risk, validated standalone before any of the four estimators consume it.
**Verified:** 2026-06-21T00:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | prims/sgd.rs (all losses; schedules incl. `optimal`+Bottou t0) validated STANDALONE on convex objective with two-pass GATHER, cpu-launch, PoolStats gate | VERIFIED | sgd_test: `sgd_cpu_launch`, `sgd_convex_objective`, `dloss_table_matches_research`, `schedule_constant_then_invscaling_then_optimal`, `sgd_margin_matches_host`, `sgd_weight_update_matches_host` all pass. `memory_gate_sgd_bounded` passes. SharedMemory/INFINITY/atomic/OsRng grep gates clean. |
| 2 | MBSGDClassifier (hinge/log/squared-hinge; schedules **INCLUDING `optimal`**) predict/predict_proba matches sklearn under pinned oracle | FAILED | oracle/exact_labels/proba tests all pass with LearningRate::Constant. The `optimal`-schedule oracle fixtures (mbsgd_classifier_optimal_*.npz) are committed but **no test loads them or fits with LearningRate::Optimal against the sklearn reference**. The `default_matches_sklearn` test confirms Optimal is the default but does not run a fit. The WR-01 penalty-ordering discrepancy (shrink BEFORE vs AFTER gradient step) explains the 5e-3 f64 band — an order of magnitude above the 1e-5 project contract. |
| 3 | MBSGDRegressor (squared-loss/epsilon-insensitive; invscaling default) predict matches sklearn under pinned oracle | VERIFIED | oracle/oracle_f32/oracle_epsilon_f32/default_matches_sklearn all pass. Bands documented (5e-3 f64, 2e-2 f32). Exact predict within band. |
| 4 | LinearSVC (squared_hinge, dual='auto', intercept_scaling) and LinearSVR (squared_eps_insensitive, epsilon) predict matches sklearn | VERIFIED | LinearSVC: exact_labels/exact_labels_f32 (HARD gate, exact integer labels) + oracle/oracle_f32 (coef band 2e-4 f64, 5e-3 f32) all pass. LinearSVR: oracle/oracle_f32 (band 2e-4 f64, 5e-3 f32) + fixture_loads pass. Q1 resolved (L-BFGS, not cd_fit), documented in 10-04-SUMMARY. |

**Score:** 3/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-kernels/src/sgd.rs` | sgd_margin + sgd_weight_update GATHER kernels | VERIFIED | Both #[cube(launch)] fns present; SharedMemory-free, INFINITY-free, atomic-free; ABSOLUTE_POS bounds-checked; real GATHER bodies not stubs |
| `crates/mlrs-backend/src/prims/sgd.rs` | sgd_solve host epoch loop: dloss/schedule/penalty/NotConverged | VERIFIED | Real implementation present: dloss (all 6 losses), optimal_t0 (Bottou), schedule_eta (all 4 schedules), full epoch loop, geometry guard, OsRng absent |
| `crates/mlrs-algos/src/linear/sgd_config.rs` | Loss/Penalty/LearningRate enums + SgdConfig + four builders | VERIFIED | All enums present with TryFrom<&str> sklearn spellings; SgdConfig 14-field struct; four builders with build() validation |
| `crates/mlrs-algos/src/linear/mbsgd_classifier.rs` | MBSGDClassifier + Fit + PredictLabels + PredictProba | VERIFIED | Full implementation; classes_ ±1 remap; Fit delegates to sgd_solve; PredictProba gated to log-loss (not all-losses) |
| `crates/mlrs-algos/src/linear/mbsgd_regressor.rs` | MBSGDRegressor + Fit + Predict | VERIFIED | Full implementation; reuses predict_linear |
| `crates/mlrs-algos/src/linear/linear_svc.rs` | LinearSVC + Fit (L-BFGS) + PredictLabels | VERIFIED | svm_lbfgs_fit; intercept_scaling synthetic-feature mechanism; dual='auto' internal |
| `crates/mlrs-algos/src/linear/linear_svr.rs` | LinearSVR + Fit (L-BFGS) + Predict | VERIFIED | Shares svm_lbfgs_fit; reuses predict_linear |
| `crates/mlrs-py/src/estimators/linear.rs` | Four #[pyclass] fit/predict wrappers | VERIFIED | All four wrappers present; builder-chain adaptation; TryFrom+build_err_to_py+algo_err_to_py; lock_pool(); f64 guard |
| `crates/mlrs-py/src/lib.rs` | Four pyclasses registered on _mlrs | VERIFIED | PyMBSGDClassifier, PyMBSGDRegressor, PyLinearSVC, PyLinearSVR all in add_class registrations (lines 231-234) |
| `tests/fixtures/mbsgd_classifier_optimal_f{32,64}_seed42.npz` | Optimal-schedule classifier fixtures | WIRED (EXISTS, NEVER LOADED) | Files present on disk; generator emits them correctly; no Rust test ever loads or asserts against them |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `mbsgd_classifier.rs::fit` | `sgd_solve` | lower_config + sgd_solve call | VERIFIED | Confirmed via sgd_solve import + call in fit body |
| `mbsgd_regressor.rs::predict` | `elastic_net::predict_linear` | shared GEMM path | VERIFIED | predict_linear reuse confirmed by 10-03 SUMMARY and test passing |
| `linear_svc.rs::fit` | `lbfgs_minimize` (via svm_lbfgs_fit) | L-BFGS over device GEMM | VERIFIED | Q1 resolved: cd_fit NOT reused; svm_lbfgs_fit confirmed in 10-04 SUMMARY; tests pass |
| `linear_svr.rs::predict` | `elastic_net::predict_linear` | shared GEMM path | VERIFIED | predict_linear confirmed |
| `mlrs-py::fit` | `Loss/Penalty/LearningRate::try_from + build_err_to_py` | TryFrom + BuildError → PyValueError | VERIFIED | sgd_smoke_test bad_enum_string_maps_to_value_error passes; 10-05-SUMMARY confirms |
| `mbsgd_classifier_optimal_*.npz` | any Rust test | load_npz | NOT_WIRED | No Rust code references these fixture paths in test assertions |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `sgd_margin` kernel | `p[]` margin vector | device GATHER over `x`, `w` | Yes — bounds-checked GATHER, verified against host reference | FLOWING |
| `sgd_weight_update` kernel | `w[]` weight update | device GATHER over `x`, `g` | Yes — bounds-checked GATHER, verified against host reference | FLOWING |
| `sgd_solve` | coef, intercept DeviceArrays | kernel launches + dloss + schedule + penalty | Yes — convex-objective test confirms convergence to OLS optimum | FLOWING |
| `MBSGDClassifier::predict_labels` | labels Vec<i32> | sgd_solve coef + margin GEMM + sign | Yes — exact-labels HARD gate passes with constant schedule | FLOWING (constant schedule only) |
| `MBSGDClassifier optimal-schedule` | labels Vec<i32> with Optimal schedule | optimal_t0 + schedule_eta + sgd_solve | Unverified against sklearn oracle | DISCONNECTED (no oracle assertion) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| SGD prim cpu-launch (both kernels launch, not just compile) | `cargo test -p mlrs-backend --features cpu --test sgd_test sgd_cpu_launch` | 6/6 tests pass in 14s | PASS |
| SGD convex-objective f64 strict 1e-5 | `cargo test -p mlrs-backend --features cpu --test sgd_test sgd_convex_objective` | PASS | PASS |
| PoolStats memory gate | `cargo test -p mlrs-backend --features cpu --test memory_gate_test memory_gate_sgd_bounded` | 1/1 pass | PASS |
| MBSGDClassifier exact labels HARD gate (f32+f64) | `cargo test -p mlrs-algos --features cpu --test mbsgd_classifier_test` | 8/8 pass (constant-schedule only) | PASS (constant only) |
| MBSGDRegressor oracle | `cargo test -p mlrs-algos --features cpu --test mbsgd_regressor_test` | 5/5 pass | PASS |
| LinearSVC exact labels HARD gate (f32+f64) | `cargo test -p mlrs-algos --features cpu --test linear_svc_test` | 6/6 pass | PASS |
| LinearSVR oracle | `cargo test -p mlrs-algos --features cpu --test linear_svr_test` | 5/5 pass | PASS |
| PyO3 smoke (fit+predict, f32+f64, bad-enum ValueError) | `cargo test -p mlrs-py --features cpu --test sgd_smoke_test` | 3/3 pass | PASS |
| optimal-schedule classifier vs sklearn oracle | No test exists | N/A — no test to run | FAIL |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PRIM-10 | 10-02 | SGD prim standalone validated on convex objective, two-pass GATHER, cpu-launch, PoolStats | SATISFIED | sgd_test 6/6, memory_gate_test 1/1 |
| SGDSVM-01 | 10-03 | MBSGDClassifier predict/predict_proba matching sklearn incl. `optimal` schedule | BLOCKED | Tests pass for constant/log schedules; the `optimal`-schedule coverage gap (CR-01) means the requirement is partially satisfied — the REQUIREMENTS.md explicitly calls out "schedules incl. `optimal`" |
| SGDSVM-02 | 10-03 | MBSGDRegressor predict matching sklearn, squared-loss/epsilon-insensitive | SATISFIED | mbsgd_regressor_test 5/5 |
| SGDSVM-03 | 10-04 | LinearSVC predict matching sklearn, exact labels hard gate | SATISFIED | linear_svc_test 6/6, exact-labels HARD gate passes |
| SGDSVM-04 | 10-04 | LinearSVR predict matching sklearn | SATISFIED | linear_svr_test 5/5 |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/mlrs-algos/tests/mbsgd_classifier_test.rs` | 132, 267 | `LearningRate::Constant` hard-coded in both oracle fit helpers; `_optimal` fixture never loaded | BLOCKER | The default `SGDClassifier()` runs an entirely unvalidated code path (Optimal schedule); the t0 Bottou arithmetic is exercised only by a tautological self-check |
| `crates/mlrs-backend/src/prims/sgd.rs` | 271-284 | L2 wscale shrink applied BEFORE the gradient step; sklearn applies it AFTER (WR-01) | WARNING | The 5e-3 f64 band (10x above the 1e-5 project contract) is a consequence; the exact-labels gate is the only strict check |
| `crates/mlrs-backend/src/prims/sgd.rs` | 436 | `optimal_t0` hard-codes `epsilon=0.1` in the dloss probe (IN-04) | Info | Harmless for hinge/log/squared-error losses; would produce wrong t0 if optimal+epsilon-insensitive were combined |
| `crates/mlrs-algos/src/linear/linear_svc.rs` | 346 | `let _dual` dead binding (IN-01) | Info | No functional impact; dead code |

### Human Verification Required

None — all checking is fully automated. The CR-01 gap is mechanically observable (grep for `_optimal` fixture loads = 0 hits in test code).

### Gaps Summary

**One blocking gap: CR-01 — `optimal` learning-rate schedule is unvalidated against the sklearn oracle (SGDSVM-01 partial)**

The phase goal explicitly states MBSGDClassifier must cover "schedules INCLUDING `optimal`." The REQUIREMENTS.md for SGDSVM-01 states: "learning-rate schedules incl. `optimal`". The ROADMAP Success Criterion #2 repeats: "schedules INCLUDING `optimal`."

The fixtures for the optimal-schedule case (`mbsgd_classifier_optimal_{f32,f64}_seed42.npz`) are committed and correctly generated by `gen_oracle.py` (both constant and optimal schedules are emitted, lines 1836-1864). The generator's docstring explicitly documents the intent: "a constant-schedule match with an optimal-schedule mismatch localizes the bug to `t0`."

Yet in `mbsgd_classifier_test.rs`, every active fit call uses `LearningRate::Constant` (lines 132 and 267). The `schedule_constant_then_invscaling_then_optimal` test in `sgd_test.rs` only verifies that the formula `schedule_eta(SgdSchedule::Optimal, t, ...)` equals its own recomputed value — a mathematical tautology, not a cross-reference against the sklearn oracle. If `optimal_t0` or the schedule clock diverges from sklearn's `_sgd_fast`, no test would catch it.

The `default_matches_sklearn` test confirms `LearningRate::Optimal` is the default — meaning a real `SGDClassifier()` with no overrides runs an entirely unexercised solver path. This is precisely the CR-01 concern raised in the code review: the artifact *looks* covered (fixture present, generator documents rationale) but the assertion that would catch a `t0` bug does not exist.

**Secondary concern: WR-01 (Warning, not blocker for this verification) — L2-penalty ordering discrepancy**

The `wscale` L2 shrink is applied BEFORE the gradient step in `sgd_solve`; sklearn's `_plain_sgd` does it AFTER. This is why the f64 coef band is 5e-3 rather than near 1e-5. The exact-labels gate compensates for this on well-separated blobs, but the underlying iterate is observably different from sklearn's. This is a warning per WR-01 in the code review.

---

_Verified: 2026-06-21T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
