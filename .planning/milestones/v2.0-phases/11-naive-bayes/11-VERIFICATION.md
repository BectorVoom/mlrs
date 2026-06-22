---
phase: 11-naive-bayes
verified: 2026-06-22T10:30:00Z
status: human_needed
score: 11/12 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run live Python smoke test: cd repo && maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml && pytest crates/mlrs-py/tests/test_naive_bayes.py -v"
    expected: "7 tests pass, including predict_proba rows sum to 1.0±1e-6 across the FFI, predict_log_proba == log(predict_proba), score in [0,1], CategoricalNB min_categories int+list ingress, and ValueError on bad hyperparameters (negative alpha/var_smoothing)"
    why_human: "maturin and pyarrow are not installed in this environment; the extension wheel cannot be built/installed to run pytest against the live FFI. The Rust algos-side correctness (all five estimators, all oracle cases) is fully verified by the cpu gate."
  - test: "Run sklearn estimator_checks re-triage: build cpu wheel, then run check_estimator against the five mlrs.GaussianNB / mlrs.MultinomialNB / mlrs.BernoulliNB / mlrs.ComplementNB / mlrs.CategoricalNB shim estimators (once shims exist)"
    expected: "No unexpected failures; absence of partial_fit on NB estimators is not flagged as a failure (D-10 scope exclusion); f64-incapable-backend skips are expected-skips; f32 proba band skips are expected-skips"
    why_human: "The pure-Python shim family module (naive_bayes.py in crates/mlrs-py/python/mlrs/) does not exist yet — consistent with all other v2 phases (8-10). check_estimator cannot be run against _mlrs low-level classes (no get_params/set_params/clone surface at that level). The shim is documented as future work."
  - test: "Verify get_params / set_params round-trip for all five NB estimators via the Python shim surface once naive_bayes.py shim is created"
    expected: "mlrs.GaussianNB().get_params() returns {'var_smoothing': 1e-9, 'priors': None}; mlrs.MultinomialNB().get_params() returns {'alpha': 1.0, 'force_alpha': True, 'fit_prior': True, 'class_prior': None}; set_params round-trips correctly"
    why_human: "The pure-Python shim does not exist; _mlrs.GaussianNB etc. are low-level Rust #[pyclass] objects without get_params/set_params. The PLAN required asserting these in test_naive_bayes.py but the live Python test was not executed and the assertion is absent from test_naive_bayes.py."
---

# Phase 11: Naive Bayes Verification Report

**Phase Goal:** A data scientist can fit and predict with all five sklearn Naive Bayes estimators (GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB), matching scikit-learn within 1e-5 (exact labels as a hard gate), and use them from Python as sklearn-named #[pyclass] estimators.
**Verified:** 2026-06-22T10:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC-1 | GaussianNB fits/predicts with global var_smoothing epsilon_, per-class mean/var via GATHER, log-sum-exp, matching sklearn | VERIFIED | `gaussian_nb_test`: 7/7 passed — exact_labels (f32+f64 integer eq), proba_band rows-sum-to-1, default_matches_sklearn, refit_releases_buffers |
| SC-2 | MultinomialNB, BernoulliNB (non-occurrence term, binarize), ComplementNB (complement weights, argmin), CategoricalNB (categorical per-feature) — all correct per-variant smoothing, matching sklearn | VERIFIED | `multinomial_nb_test` 8/8, `bernoulli_nb_test` 8/8, `complement_nb_test` 8/8, `categorical_nb_test` 9/9 — all green; force_alpha_clip/binarize_none/norm_true/min_categories cases pass |
| SC-3 | Every NB predict_proba row sums to 1 (log-sum-exp, no underflow); predict labels exact | VERIFIED | All oracle tests include rows-sum-to-1 assertion (1e-6 tolerance); exact labels asserted with integer assert_eq!, no band — all pass |
| SC-4 | PY-06: all five NB estimators #[pyclass]-backed with fit/predict/transform/score, get_params/set_params with sklearn-named hyperparameters, f32/f64 dispatch, GIL release, ship in per-backend wheels | PARTIAL | _mlrs registration: 5/5 add_class lines present (30 total confirmed); sklearn-named #[new] signatures confirmed; guard_f64 + py.detach + lock_pool pattern present; get_params/set_params NOT at the _mlrs level (live in future pure-Python shim — consistent with all other v2 phases 8-10); live FFI test not executed (no maturin/pyarrow in env) |

**Score:** 11/12 must-haves verified (see detailed breakdown below)

### Deferred Items

None — all identified gaps are in scope for this phase but require a human-verified environment gate.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-algos/src/naive_bayes/nb_common.rs` | Shared NB free functions incl. class_grouped_sum GATHER | VERIFIED | All 6 free functions present: log_sum_exp_normalize, empirical_class_log_prior, argmax_decode, argmin_decode, accuracy_score, class_grouped_sum (+sumsq); column_reduce wired at line 271 |
| `crates/mlrs-algos/src/naive_bayes/mod.rs` | Module index re-exporting five estimators + MinCategories | VERIFIED | pub mod nb_common + all five estimator mods declared; MinCategories re-exported |
| `crates/mlrs-algos/src/traits.rs` | PredictLogProba trait | VERIFIED | `pub trait PredictLogProba<F>` at line 288 |
| `crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs` | GaussianNB Fit + PredictLabels + PredictProba + PredictLogProba | VERIFIED | fn predict_log_proba at line 452; global epsilon_ at line 269; GATHER via class_grouped_sum/sumsq |
| `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs` | MultinomialNB with GEMM joint-LL | VERIFIED | gemm imported at line 18, called at line 280; alpha*n_features denominator confirmed at line 214 |
| `crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs` | BernoulliNB with non-occurrence term + binarize | VERIFIED | binarize field at line 47; neg_prob_sum_ at line 64; 2*alpha denominator at line 252 |
| `crates/mlrs-algos/src/naive_bayes/complement_nb.rs` | ComplementNB with complement weights + argmin | VERIFIED | argmin_decode called at line 352; complement weights; norm=false default |
| `crates/mlrs-algos/src/naive_bayes/categorical_nb.rs` | CategoricalNB with ragged Vec<Vec<f64>> + MinCategories | VERIFIED | Vec<Vec<f64>> at line 78 and 313; MinCategories at line 39; 9/9 oracle tests green |
| `crates/mlrs-algos/tests/nb_common_test.rs` | 9 standalone + GATHER-launch-witness tests | VERIFIED | 9/9 passed; class_grouped_sum/sumsq cpu launch witnessed |
| `crates/mlrs-algos/tests/gaussian_nb_test.rs` | Oracle: exact_labels, proba_band, default_matches_sklearn, build_rejects, refit_releases_buffers | VERIFIED | 7/7 passed, 0 ignored |
| `crates/mlrs-algos/tests/{multinomial,bernoulli,complement}_nb_test.rs` | 8 oracle cases each | VERIFIED | 8/8 each (24 total), 0 ignored |
| `crates/mlrs-algos/tests/categorical_nb_test.rs` | 9 oracle cases (incl. min_categories, fit_rejects_bad_input) | VERIFIED | 9/9 passed, 0 ignored |
| `tests/fixtures/*_nb_{f32,f64}_seed42.npz` | 10 committed oracle blobs | VERIFIED | `ls tests/fixtures/ | grep nb | wc -l` == 10; all 5 variants × f32/f64 present |
| `scripts/gen_oracle.py` | 5 NB generators | VERIFIED | def gen_gaussian_nb/multinomial_nb/bernoulli_nb/complement_nb/categorical_nb all present |
| `crates/mlrs-py/src/estimators/naive_bayes.rs` | Five #[pyclass] wrappers with accuracy_score + predict_log_proba | VERIFIED | any_estimator! ×5; accuracy_score at line 51; predict_log_proba ×15 occurrences |
| `crates/mlrs-py/src/estimators/mod.rs` | pub mod naive_bayes | VERIFIED | Line 36 |
| `crates/mlrs-py/src/lib.rs` | 5 add_class registrations; total 30 | VERIFIED | 5 NB add_class lines at 254-258; grep -c add_class == 30 |
| `crates/mlrs-py/tests/pyclass_smoke_test.rs` | five_naive_bayes_estimators_construct_unfit | VERIFIED | fn at line 89; 4/4 smoke tests pass |
| `crates/mlrs-py/tests/test_naive_bayes.py` | 7 def test_ functions, syntax-valid | VERIFIED | python3 ast.parse exits 0; grep -c "def test_" == 7 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `nb_common.rs` | `mlrs_backend::prims::reduce` | column_reduce over per-class row-gathered buffers | VERIFIED | `use mlrs_backend::prims::reduce::{column_reduce, ReducePath, ScalarOp}` at line 44; column_reduce called at line 270-271 |
| `crates/mlrs-algos/src/lib.rs` | naive_bayes module | `pub mod naive_bayes` | VERIFIED | Line 54 of lib.rs |
| `gaussian_nb.rs` | `nb_common::log_sum_exp_normalize` | predict_proba/predict_log_proba normalization | VERIFIED | imported and used in joint_log_likelihood evaluator |
| `gaussian_nb.rs` | `nb_common::class_grouped_sum` | per-class mean/variance sufficient statistics | VERIFIED | imported and called in fit body |
| `multinomial_nb.rs` | `mlrs_backend::prims::gemm` | X @ feature_log_prob_.T joint-LL matvec (transb=true) | VERIFIED | gemm import at line 18; gemm call at line 280 |
| `complement_nb.rs` | `nb_common::argmin_decode` | CNB argmin decision (D-08) | VERIFIED | argmin_decode used at line 352 |
| `categorical_nb.rs` | MinCategories enum | per-feature category padding | VERIFIED | MinCategories referenced 14+ times in categorical_nb.rs |
| `categorical_nb.rs` | `nb_common::log_sum_exp_normalize` | per-feature lookup sum normalize | VERIFIED | imported and used in joint_log_likelihood |
| `crates/mlrs-py/src/lib.rs` | PyGaussianNB ... PyCategoricalNB | m.add_class registration | VERIFIED | 5 add_class lines at 254-258; total 30 confirmed |
| `naive_bayes.rs` | `nb_common::accuracy_score` | score() method | VERIFIED | use mlrs_algos::naive_bayes::nb_common::accuracy_score at line 51 |

### Data-Flow Trace (Level 4)

The estimators are not web/UI components rendering dynamic data; they are in-process numeric algorithms. Data flow is the computation pipeline:

| Estimator | Data Variable | Source | Produces Real Data | Status |
|-----------|---------------|--------|--------------------|--------|
| GaussianNB | theta_, var_, epsilon_ | class_grouped_sum/sumsq over training X | Yes — oracle tests verify sklearn parity | FLOWING |
| MultinomialNB | feature_log_prob_ | class_grouped_sum + log formula | Yes — gemm matvec drives predict; oracle confirms | FLOWING |
| BernoulliNB | feature_log_prob_ + neg_prob_sum_ | class_grouped_sum + 2*alpha denominator | Yes — binarize path + non-occurrence term confirmed | FLOWING |
| ComplementNB | feature_log_prob_ (complement) | feature_all_ - per-class count + optional L1 norm | Yes — argmin_decode used; oracle 8/8 | FLOWING |
| CategoricalNB | feature_log_prob_ (ragged) | host per-feature category counts + alpha*n_categories_j | Yes — ragged Vec<Vec<f64>> confirmed; 9/9 oracle | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| nb_common free functions + GATHER cpu launch | `cargo test --features cpu -p mlrs-algos --test nb_common_test` | 9 passed, 0 failed, 0 ignored | PASS |
| GaussianNB exact labels + proba + refit | `cargo test --features cpu -p mlrs-algos --test gaussian_nb_test` | 7 passed, 0 failed, 0 ignored | PASS |
| MultinomialNB oracle (incl. force_alpha_clip) | `cargo test --features cpu -p mlrs-algos --test multinomial_nb_test` | 8 passed, 0 failed, 0 ignored | PASS |
| BernoulliNB oracle (incl. binarize_none) | `cargo test --features cpu -p mlrs-algos --test bernoulli_nb_test` | 8 passed, 0 failed, 0 ignored | PASS |
| ComplementNB oracle (incl. norm_true) | `cargo test --features cpu -p mlrs-algos --test complement_nb_test` | 8 passed, 0 failed, 0 ignored | PASS |
| CategoricalNB oracle (incl. min_categories, fit_rejects_bad_input) | `cargo test --features cpu -p mlrs-algos --test categorical_nb_test` | 9 passed, 0 failed, 0 ignored | PASS |
| PyO3 construct-unfit smoke (5 NB estimators) | `cargo test -p mlrs-py --features cpu --test pyclass_smoke_test` | 4 passed, 0 failed, 0 ignored (incl. five_naive_bayes_estimators_construct_unfit) | PASS |
| mlrs-py crate builds clean | `cargo build -p mlrs-py --features cpu` | Finished, 2 pre-existing spectral.rs warnings | PASS |
| test_naive_bayes.py syntax-valid | `python3 -c "import ast; ast.parse(...)"` | exits 0 | PASS |
| Live FFI fit/predict/proba/score round-trip | `pytest crates/mlrs-py/tests/test_naive_bayes.py` | SKIP — no maturin/pyarrow in this environment | SKIP |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared or found for this phase. Step 7c: SKIPPED (no probe files).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| NB-01 | 11-01, 11-02 | GaussianNB fit/predict matching sklearn | SATISFIED | gaussian_nb_test 7/7; exact labels f32+f64; var_smoothing epsilon_ global |
| NB-02 | 11-01, 11-03 | MultinomialNB with sparse densify | SATISFIED | multinomial_nb_test 8/8; gemm joint-LL; alpha*n_features denominator; sparse densify at ingress (line 447 naive_bayes.rs) |
| NB-03 | 11-01, 11-03 | BernoulliNB with non-occurrence term, binarize | SATISFIED | bernoulli_nb_test 8/8; binarize_none case; 2*alpha denominator; non-occurrence term folded into GEMM |
| NB-04 | 11-01, 11-03 | ComplementNB complement weights, argmin | SATISFIED | complement_nb_test 8/8; norm_true case; argmin_decode confirmed |
| NB-05 | 11-01, 11-04 | CategoricalNB categorical integer features | SATISFIED | categorical_nb_test 9/9; ragged Vec<Vec<f64>>; MinCategories padding; unseen-category guard; fit_rejects_bad_input |
| PY-06 | 11-05 | All v2 estimators #[pyclass] with sklearn surface + get_params/set_params | PARTIALLY SATISFIED | 5 add_class registrations (25→30) confirmed; sklearn-named #[new] signatures; guard_f64+GIL release+lock_pool present; get_params/set_params NOT at _mlrs level (shim deferred, consistent with Phases 8-10); live estimator_checks environment-gated |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/naive_bayes/complement_nb.rs` | 6 (doc comment) | `todo!()` mentioned in `//!` — historical doc about Wave-0 state | Info | Not an actual stub; doc-comment only. No actual `todo!()` call in source |
| `src/naive_bayes/bernoulli_nb.rs` | 6 (doc comment) | Same — historical Wave-0 doc | Info | Not an actual stub |
| `src/naive_bayes/multinomial_nb.rs` | 6 (doc comment) | Same | Info | Not an actual stub |
| `src/naive_bayes/gaussian_nb.rs` | 6 (doc comment) | Same | Info | Not an actual stub |
| `src/naive_bayes/categorical_nb.rs` | 8 (doc comment) | Same | Info | Not an actual stub |

No `TBD`, `FIXME`, or `XXX` markers in any NB-phase file. No `SharedMemory`/`F::INFINITY`/`Atomic` in executable NB code (doc-comment mentions only, confirmed by line-level inspection). No actual `todo!()` calls in source after Wave-0 was filled. No `#[ignore]` attributes in any test.

### Human Verification Required

#### 1. Live FFI Python Smoke Test

**Test:** In an environment with maturin and pyarrow installed, run:
```
maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml
pytest crates/mlrs-py/tests/test_naive_bayes.py -v
```
**Expected:** 7 tests pass — `test_gaussian_nb_predict`, `test_multinomial_nb_predict`, `test_bernoulli_nb_predict`, `test_complement_nb_predict`, `test_categorical_nb_predict`, `test_categorical_nb_min_categories_int_and_list`, `test_bad_hyperparameter_raises_value_error` (5 parametrizations). Each predict test asserts `predict_proba.sum(axis=1)` is close to 1.0±1e-6, `np.exp(log_proba)` matches proba within 1e-5, and score is in [0,1]. `test_bad_hyperparameter_raises_value_error` must confirm `ValueError` is raised at fit for negative alpha/var_smoothing.
**Why human:** maturin and pyarrow are absent from this environment; the extension `.so` cannot be built. The test has `pytest.importorskip("mlrs._mlrs")` guards that silently skip without the wheel.

#### 2. sklearn estimator_checks Re-Triage (PY-06 cross-cutting)

**Test:** Once a pure-Python `naive_bayes.py` shim module exists in `crates/mlrs-py/python/mlrs/` (mirroring `linear.py`), build the cpu wheel and run `check_estimator` against the five NB shim estimators.
**Expected:** No unexpected failures. The absence of `partial_fit` on NB estimators should NOT be flagged (D-10: PY-06 scopes partial_fit to IncrementalPCA/MBSGD only). f64-incapable-backend skips and f32 proba-band tolerance differences are expected-skips.
**Why human:** The pure-Python shim module (`mlrs/naive_bayes.py`) does not exist yet — this is a documented gap consistent across all v2 phases (8-10). The _mlrs low-level classes lack `get_params`/`set_params`/`clone` at the Rust level, so `check_estimator` must target the shim. This requires creating the shim first.

#### 3. get_params / set_params round-trip via Python shim

**Test:** Once `mlrs/naive_bayes.py` shim is created:
```python
import mlrs
m = mlrs.GaussianNB()
assert m.get_params() == {'var_smoothing': 1e-9, 'priors': None}
m2 = mlrs.MultinomialNB()
assert 'alpha' in m2.get_params()
assert m2.set_params(alpha=0.5).get_params()['alpha'] == 0.5
```
**Expected:** get_params returns the sklearn-named hyperparameters; set_params round-trips.
**Why human:** The PLAN (11-05-PLAN.md Task 2 acceptance criteria) required asserting `get_params`/`set_params` in `test_naive_bayes.py`, but this assertion is absent from the file — the test drives `_mlrs` directly and `_mlrs.GaussianNB` has no `get_params`. The shim (which inherits from `sklearn.BaseEstimator` via `MlrsBase`) is where this lives, and the shim does not exist.

### Gaps Summary

No BLOCKER gaps found. All five estimators are fully implemented and the Rust-side correctness gate (47 oracle tests, 0 failures, 0 ignored, across all five NB variants on cpu f32+f64) is green.

One documentation / acceptance-criteria gap exists for PY-06:
- The pure-Python `mlrs/naive_bayes.py` shim (which provides `get_params`/`set_params`/`clone`/`check_estimator` compatibility) was not created. This is consistent with the treatment of all other v2 phases (Phases 8-10 also have no shims for their estimators). The PLAN (11-05-PLAN.md) required asserting `get_params`/`set_params` in `test_naive_bayes.py`, but the live test was not run and the assertion is absent. This is a WARNING, not a BLOCKER: the `#[pyclass]` registration, dtype dispatch, GIL release, and sklearn-named constructor API are all verified; only the sklearn-subclassing shim surface is absent.

The three human verification items above (live Python smoke, estimator_checks re-triage, get_params round-trip) represent the maturin+pyarrow environment gate and the shim-creation prerequisite. Once those pass, this phase is fully complete.

---

_Verified: 2026-06-22T10:30:00Z_
_Verifier: Claude (gsd-verifier)_
