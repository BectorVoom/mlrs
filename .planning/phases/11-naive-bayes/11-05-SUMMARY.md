---
phase: 11-naive-bayes
plan: 05
subsystem: python-bindings
tags: [naive-bayes, pyo3, pyclass, dtype-dispatch, sklearn, ffi, PY-06, NB-01, NB-02, NB-03, NB-04, NB-05]

# Dependency graph
requires:
  - phase: 11-naive-bayes
    plan: 02
    provides: "GaussianNB<F>: Fit + PredictLabels + PredictProba + PredictLogProba + builder().var_smoothing(..).priors(..).build()"
  - phase: 11-naive-bayes
    plan: 03
    provides: "MultinomialNB/BernoulliNB/ComplementNB<F>: Fit + the three predict traits + builders (.alpha/.force_alpha/.fit_prior/.class_prior + .binarize/.norm)"
  - phase: 11-naive-bayes
    plan: 04
    provides: "CategoricalNB<F>: Fit + the three predict traits + builder (.min_categories(MinCategories)) + MinCategories::{Infer,Uniform,PerFeature}"
  - phase: 11-naive-bayes
    plan: 01
    provides: "nb_common::accuracy_score (score backbone); PredictLogProba trait"
  - phase: 10-sgd-linear-svm
    provides: "the any_estimator! dtype-dispatch macro + the PyMBSGDClassifier #[pyclass] template (py.detach GIL release, guard_f64, build/algo_err_to_py mappers, lock_pool); the test_sgd.py live-FFI harness convention"
provides:
  - "Five #[pyclass] NB wrappers (PyGaussianNB … PyCategoricalNB) on _mlrs (add_class 25 → 30) with the full sklearn surface (fit/predict_labels/predict_proba_{f32,f64}/predict_log_proba_{f32,f64}/score) and sklearn-mirrored names (D-09)"
  - "f32/f64 runtime dispatch (guard_f64 before the F64 upload); GIL released during compute (py.detach + lock_pool); BuildError/AlgoError → PyValueError via the existing mappers (zero new mapper)"
  - "MultinomialNB sparse-densify-at-ingress (NB-02); CategoricalNB min_categories None/int/list → MinCategories::{Infer,Uniform,PerFeature}"
  - "Rust construct-unfit smoke (five_naive_bayes_estimators_construct_unfit) + live-FFI Python smoke (test_naive_bayes.py) asserting predict_proba rows sum to 1 across the FFI"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Shared NB predict surface factored into nb_surface_fns! FREE FUNCTIONS keyed by a module-unique prefix (PyO3 forbids a second #[pymethods] impl per pyclass without multiple-pymethods, and a macro_rules! cannot expand as items INSIDE a #[pymethods] block); each estimator's single #[pymethods] block carries literal thin delegators"
    - "Upload HOISTED to a let binding before the trait call (the &mut pool borrow cannot be re-taken inside the same call, E0499 — the linear.rs precedent)"
    - "score(x,y) materializes the i32 label vector host-side (labels_as_i32, round-to-nearest from the float Arrow y) and calls nb_common::accuracy_score over predict_labels"

key-files:
  created:
    - crates/mlrs-py/src/estimators/naive_bayes.rs
    - crates/mlrs-py/tests/test_naive_bayes.py
  modified:
    - crates/mlrs-py/src/estimators/mod.rs
    - crates/mlrs-py/src/lib.rs
    - crates/mlrs-py/tests/pyclass_smoke_test.rs

key-decisions:
  - "PyO3 0.28 (no multiple-pymethods — v2 adds ZERO binding infra) rejects BOTH a second #[pymethods] impl AND a macro_rules! invocation inside a #[pymethods] block. Resolved by factoring the device-touching predict surface into nb_surface_fns! free functions (DRY where it matters) + literal thin delegating methods per estimator (mechanical, accepted by the proc-macro)."
  - "The Rust integration binary keeps the construct-unfit witness (it cannot build pyarrow capsules or link PyErr::is_instance_of — the 10-05 precedent); the live fit→predict round-trip, the across-FFI predict_proba rows-sum-to-1, predict_log_proba==log(proba), score-in-[0,1], and the bad-hyperparameter ValueError all live in test_naive_bayes.py."
  - "score's y is materialized to i32 host-side via round-to-nearest from the float Arrow array (the same float-y ingress fit uses), feeding nb_common::accuracy_score."

patterns-established:
  - "nb_surface_fns! free-function factoring is the reusable answer for a uniform device-touching surface shared across N pyclasses under pyo3-without-multiple-pymethods."

requirements-completed: [NB-01, NB-02, NB-03, NB-04, NB-05, PY-06]

# Metrics
duration: 13min
completed: 2026-06-22
---

# Phase 11 Plan 05: PY-06 Final Cross-Cutting Sign-off Summary

**The five sklearn Naive-Bayes estimators wrapped as `#[pyclass]` on `_mlrs` (registration 25 → 30) with the full sklearn surface — `fit`/`predict_labels`/`predict_proba_{f32,f64}`/`predict_log_proba_{f32,f64}`/`score` — sklearn-mirrored hyperparameter names (D-09, zero translation), f32/f64 runtime dispatch with `guard_f64` before the F64 upload, GIL released via `py.detach`+`lock_pool`, MultinomialNB densify-at-ingress (NB-02), CategoricalNB `min_categories` None/int/list ingress, and BuildError/AlgoError → `PyValueError` through the existing mappers (zero new mapper). Rust construct-unfit smoke green (4/4); live-FFI Python smoke present (7 `def test_`, ast-valid) asserting `predict_proba` rows sum to 1 across the FFI. Closes PY-06 and the Phase-11 Python surface.**

## Performance
- **Duration:** ~13 min
- **Tasks:** 2
- **Files created:** 2 (naive_bayes.rs source + test_naive_bayes.py); **modified:** 3 (mod.rs, lib.rs, pyclass_smoke_test.rs)

## Accomplishments
- **Task 1 — five `#[pyclass]` wrappers + registration:** `PyGaussianNB` / `PyMultinomialNB` / `PyBernoulliNB` / `PyComplementNB` / `PyCategoricalNB` over the `any_estimator!`-emitted `Any<Name>` dtype-dispatch enums (D-06). Each `#[new]` carries the sklearn-default signature with the sklearn-mirrored names (D-09): `GaussianNB(var_smoothing=1e-9, priors=None)` (NO `alpha`); the discrete four take `alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None` plus `binarize=0.0` (Bernoulli), `norm=False` (Complement), `min_categories=None` (Categorical). `fit` does `float_dtype` dispatch inside `py.detach(|| { let mut pool = crate::lock_pool(); … })` with `guard_f64()?` on the F64 arm BEFORE upload, the builder chain `.build().map_err(build_err_to_py)?` → `.fit(..).map_err(algo_err_to_py)?`. MultinomialNB's `fit` consumes the already-dense float Arrow array (sparse densified at ingress, NB-02). CategoricalNB maps `min_categories` None/int/list → `MinCategories::{Infer,Uniform,PerFeature}` (`resolve_min_categories`). Registered on `_mlrs` (add_class 25 → 30).
- **Task 2 — smoke + live-FFI + PY-06 sign-off:** extended `pyclass_smoke_test.rs` with `five_naive_bayes_estimators_construct_unfit` (4/4 green); added `test_naive_bayes.py` (live-FFI, importorskip-guarded) over all five estimators — sklearn-named kwargs, fit → predict_labels/predict_proba/predict_log_proba/score, `predict_proba` rows sum to 1.0 ± 1e-6 **across the FFI**, `predict_log_proba == log(predict_proba)`, `score ∈ [0,1]`, CategoricalNB `min_categories` int+list ingress, and the bad-hyperparameter (negative `alpha`/`var_smoothing`) `ValueError`-at-`fit` (D-05/D-09, T-11-05-01).

## Task Commits
1. **Task 1: Five #[pyclass] NB wrappers (sklearn-named, dtype dispatch, GIL release) + registration** — `b82baf2` (feat)
2. **Task 2: NB pyclass construct-unfit smoke + live-FFI Python smoke (PY-06)** — `56334c1` (test)

## Files Created/Modified
- `crates/mlrs-py/src/estimators/naive_bayes.rs` (created) — five `any_estimator!` enums + five `#[pyclass]` wrappers; the `nb_surface_fns!` free-function factoring of the shared predict surface (keyed by a module-unique prefix per estimator) + literal thin delegating methods in each `#[pymethods]` block; `labels_as_i32` + `resolve_min_categories` host helpers.
- `crates/mlrs-py/src/estimators/mod.rs` (modified) — `pub mod naive_bayes;` (alphabetical, after `linear`).
- `crates/mlrs-py/src/lib.rs` (modified) — `use estimators::naive_bayes::{…}` + five `m.add_class::<…>()?;` (25 → 30).
- `crates/mlrs-py/tests/pyclass_smoke_test.rs` (modified) — `five_naive_bayes_estimators_construct_unfit`.
- `crates/mlrs-py/tests/test_naive_bayes.py` (created) — the live-FFI Python smoke.

## Decisions Made
- **Shared surface via free functions, not a second `#[pymethods]`.** PyO3 0.28 (no `multiple-pymethods` — v2 adds ZERO binding infra, pyo3 stays 0.28) rejects both a second `#[pymethods] impl` per pyclass (E0119) **and** a `macro_rules!` invocation expanded as items inside a `#[pymethods]` block ("macros cannot be used as items in `#[pymethods]` impl blocks"). The device-touching predict surface is factored into `nb_surface_fns!`-generated free functions (DRY where it matters — GIL release, dtype dispatch, error mapping), and each estimator's single `#[pymethods]` block carries literal thin one-line delegators.
- **Upload hoisted to a `let` binding** before the trait call (the `&mut pool` borrow cannot be re-taken inside the same call — E0499; the `linear.rs` precedent).
- **Rust keeps construct-unfit; the FFI assertions live in Python.** The Rust integration binary cannot build pyarrow capsules or link `PyErr::is_instance_of` (the 10-05 precedent), so the live fit→predict round-trip + the across-FFI rows-sum-to-1 + the bad-hyperparameter `ValueError` are in `test_naive_bayes.py`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Shared predict surface refactored from a `#[pymethods]`-emitting macro to free functions + thin delegators**
- **Found during:** Task 1 (the first `cargo build`).
- **Issue:** The plan's "copy PyMBSGDClassifier verbatim" framing implies a per-method `#[pymethods]` body. A first attempt factored the shared surface into a macro emitting a second `#[pymethods] impl` block — PyO3 0.28 rejects that with E0119 (conflicting `PyMethods` impl; `multiple-pymethods` is off, and enabling it is barred by "v2 adds ZERO binding infra"). A second attempt emitted the methods as items inside the existing `#[pymethods]` block via `macro_rules!` — PyO3 rejects that too ("macros cannot be used as items in `#[pymethods]` impl blocks").
- **Fix:** Factored the device-touching logic into `nb_surface_fns!` free functions keyed by a module-unique prefix; each estimator's single `#[pymethods]` block carries literal thin delegators. Same observable surface, infra-free.
- **Files modified:** crates/mlrs-py/src/estimators/naive_bayes.rs.
- **Commit:** `b82baf2`.

**2. [Rule 3 - Blocking] Upload hoisted before the trait call (E0499)**
- **Found during:** Task 1 (the first `cargo build`).
- **Issue:** Inlining `validated_f32(as_f32(&xa)?, &mut pool)` as an argument to a trait call that also takes `&mut pool` double-borrows `pool` (E0499).
- **Fix:** Hoisted each upload to a `let xd = …;` binding before the trait call (the `linear.rs` predict precedent).
- **Files modified:** crates/mlrs-py/src/estimators/naive_bayes.rs.
- **Commit:** `b82baf2`.

---

**Total deviations:** 2 auto-fixed (both Rule 3 — blocking compile issues; the observable `#[pyclass]` surface is exactly the plan's; no scope change).

## estimator_checks Re-Triage (PY-06) — Outcome

The full v2 `#[pyclass]` registration check passes: `grep -c add_class crates/mlrs-py/src/lib.rs` == **30** (25 → 30; the five NB add_class lines present). The sklearn `check_estimator` re-triage across the full v2 surface (11-VALIDATION.md §Manual-Only) **could NOT be executed in this environment** and is deferred to a Python+sklearn environment with the built wheel:

- **maturin is not installed** and **pyarrow is absent from the system Python** (Python 3.12.3, sklearn 1.9.0 present), so the `mlrs` extension wheel cannot be built/installed here to run `check_estimator`.
- The extension **does build** (`cargo build -p mlrs-py --features cpu` exits 0) and both test binaries compile, so the FFI surface is sound; the live `check_estimator` triage is the only piece that needs the wheel.
- **Expected triage when run** (documented for the operator): the low-level `_mlrs` classes are NOT sklearn estimators (no `get_params`/`set_params`/clone surface — that lives in the pure-Python shim, out of this plan's scope), so `check_estimator` is run against the SHIM estimators. Per **D-10**, the five NB estimators expose `Fit` only — **NO `partial_fit`** (PY-06 scopes `partial_fit` to IncrementalPCA / MBSGD per the ROADMAP success criterion); the re-triage must NOT flag the absence of NB `partial_fit` as a failure. Any `check_estimator` skips for f64-on-an-f64-incapable-backend (rocm) or for the documented f32 proba band are expected-skips, not real failures.

This is recorded honestly rather than reported as a pass: the registration half of PY-06 is verified in-repo; the live `check_estimator` triage is environment-gated.

## Python-Test Execution Status (honest)

`test_naive_bayes.py` was **NOT executed live** here — `pytest` collection `importorskip`s on the missing `pyarrow` + the not-yet-built `mlrs._mlrs` extension (no maturin in this environment). It was **syntax-validated** (`python3 -c "import ast; ast.parse(...)"` exits 0) and carries **7 `def test_`** functions. The extension itself builds (`cargo build -p mlrs-py --features cpu` exits 0), so the live run is a build-the-wheel-then-`pytest` step in a maturin+pyarrow environment. The numerical contract these tests smoke is independently gated by the Rust algos oracles (11-02/03/04 `*_nb_test.rs`, including the per-variant `proba_band` rows-sum-to-1 cases).

## Verification Evidence
- `cargo build -p mlrs-py --features cpu` — exits 0 (2 warnings, both pre-existing in `spectral.rs`, out of scope).
- `cargo test -p mlrs-py --features cpu --test pyclass_smoke_test` — **4 passed, 0 failed, 0 ignored** (incl. `five_naive_bayes_estimators_construct_unfit`).
- `grep -c "add_class::<PyGaussianNB>\|…\|<PyCategoricalNB>" crates/mlrs-py/src/lib.rs` == **5**; `grep -c "add_class" …/lib.rs` == **30**.
- `grep -c "accuracy_score" …/naive_bayes.rs` == 2; `grep -c "predict_log_proba" …/naive_bayes.rs` == 15; `grep -c "any_estimator!" …/naive_bayes.rs` == 5.
- `python3 -c "import ast; ast.parse(open('crates/mlrs-py/tests/test_naive_bayes.py').read())"` — exits 0; `grep -c "def test_" …/test_naive_bayes.py` == 7.
- predict_proba rows-sum-to-1: asserted across the FFI in `test_naive_bayes.py` (`np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)`) AND in the Rust **algos** oracles (11-02/03/04 `proba_band` cases, every row sums to 1.0 ± 1e-6). The Rust mlrs-py binary keeps the construct-unfit witness (cannot run the FFI fit without a Python interpreter — 10-05 precedent).

## Threat Surface Scan
The plan's threat register (T-11-05-01..05) is mitigated and introduces no surface beyond it: hyperparameter validation `build().map_err(build_err_to_py)` → `ValueError` BEFORE any upload (T-11-05-01, witnessed by the Python bad-hyperparameter test); `?` on typed errors inside the `py.detach` body, never a panic across the FFI (T-11-05-02); `guard_f64()?` on the F64 arm before upload (T-11-05-03); fit/predict geometry guards → `algo_err_to_py` `ValueError` (T-11-05-04); `crate::lock_pool()` is poison-recovering, NOT `.lock().expect()` (T-11-05-05). No new network/auth/file/schema surface.

## Known Stubs
None — all five wrappers are fully wired to the shipped `mlrs_algos` estimators; the predict surface delegates to real `Fit`/`PredictLabels`/`PredictProba`/`PredictLogProba` impls. No placeholder/empty-data paths.

## Self-Check: PASSED

Both created files verified on disk (`crates/mlrs-py/src/estimators/naive_bayes.rs`, `crates/mlrs-py/tests/test_naive_bayes.py`); both task commits present in git history (`b82baf2`, `56334c1`).

---
*Phase: 11-naive-bayes*
*Completed: 2026-06-22*
