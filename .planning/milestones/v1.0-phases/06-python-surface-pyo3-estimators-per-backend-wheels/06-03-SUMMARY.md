---
phase: 06-python-surface-pyo3-estimators-per-backend-wheels
plan: 03
subsystem: bindings
tags: [pyo3, pyclass, dtype-dispatch, gil, capability, estimators, arrow]

# Dependency graph
requires:
  - phase: 06-python-surface-pyo3-estimators-per-backend-wheels
    plan: 02
    provides: ingress/egress/capability/errors helpers, global_pool(), any_estimator! macro skeleton, _mlrs pymodule
  - phase: 04-closed-form-estimators
    provides: LinearRegression/Ridge/PCA/TruncatedSVD Estimator<F> + Fit/Predict/Transform
  - phase: 05-iterative-estimators-clustering-neighbors
    provides: Lasso/ElasticNet/LogisticRegression/KMeans/DBSCAN/NearestNeighbors/KNN + PredictLabels/KNeighbors/PredictProba
provides:
  - "12 PyO3 #[pyclass] estimator wrappers registered on _mlrs (PY-01)"
  - "Per-estimator Any<Name> dtype-dispatch enum (Unfit + F32 + F64) via any_estimator! (D-06/PY-05)"
  - "sklearn-named #[new] constructors storing hyperparameters verbatim on the Unfit arm (PY-02); LogisticRegression C->c mapping"
  - "GIL-released device compute (py.detach + global_pool) in every fit/predict/transform/kneighbors body (PY-03)"
  - "f64 capability guard before the F64 fit arm (D-04)"
  - "host-materializing fitted-attr accessors (Vec<f32>/Vec<f64>/Vec<i32>) for the Python shim (D-03/D-06)"
  - "errors::not_fitted() for the unfit-arm accessor/predict path (T-06-09)"
  - "Rust-callable unfit_default()/is_unfit() + pyclass_smoke_test constructing all 12 without an interpreter"
affects: [06-04-python-shim-logic, 06-05-wheel-build-tests, 06-06-estimator-checks]

# Tech tracking
tech-stack:
  added: []
  patterns: [pyclass-over-dispatch-enum, gil-released-detach-compute, dtype-dispatch-on-arrow-float, rust-callable-test-ctor-for-extension-module-cdylib]

key-files:
  created:
    - crates/mlrs-py/src/estimators/mod.rs
    - crates/mlrs-py/src/estimators/linear.rs
    - crates/mlrs-py/src/estimators/cluster.rs
    - crates/mlrs-py/src/estimators/decomposition.rs
    - crates/mlrs-py/src/estimators/neighbors.rs
    - crates/mlrs-py/tests/pyclass_smoke_test.rs
  modified:
    - crates/mlrs-py/src/lib.rs
    - crates/mlrs-py/src/errors.rs

key-decisions:
  - "pymethods #[new] bodies are written explicitly per estimator ON TOP of the macro-emitted Any<Name> enum (the Plan 02 skeleton emits only the enum); the macro is the dispatch-enum generator, the trait-specific method bodies are hand-written because each estimator's trait set differs (Predict vs Transform vs PredictLabels vs KNeighbors vs PredictProba)"
  - "predict/transform/predict_proba accessors are split into dtype-suffixed methods (predict_f32/predict_f64, coef_f32/coef_f64) because a #[pyclass] method cannot return a type generic over F; the shim picks the suffix from the wrapper's dtype()"
  - "added pub unfit_default()/is_unfit() Rust-callable ctors (outside #[pymethods]) so the cross-crate smoke test constructs each wrapper WITHOUT a Python interpreter — the extension-module-mode rules from Plan 02 mean a live interpreter is not assumed in a Rust integration test"
  - "KMeans random_state:Option<u64> -> seed (None -> fixed DEFAULT_SEED=0) for deterministic v1; DBSCAN eps stays f64 regardless of input dtype; DBSCAN exposes labels_/core_sample_indices_ but NO standalone predict (algos D-08)"

requirements-completed: [PY-01, PY-02, PY-05]

# Metrics
duration: 12min
completed: 2026-06-13
---

# Phase 6 Plan 03: PyO3 #[pyclass] Estimator Wrappers Summary

**All 12 `mlrs-algos` estimators wrapped as PyO3 `#[pyclass]` objects registered on `_mlrs`, each driving a macro-emitted `Any<Name>` three-state dispatch enum (`Unfit` + `F32` + `F64`): sklearn-named constructors store hyperparameters verbatim (PY-02), `fit`/`predict`/`transform`/`kneighbors`/`predict_proba` dispatch f32/f64 on the incoming Arrow float dtype (D-06/PY-05), every device call runs inside `py.detach` with the f64 capability guard ahead of the F64 arm (PY-03/D-04), and fitted-attribute accessors materialize host `Vec<f32|f64|i32>` for the Python shim (D-03).**

## Performance

- **Duration:** ~12 min
- **Completed:** 2026-06-13
- **Tasks:** 3 executed
- **Files:** 8 (6 created, 2 modified)
- **Tests:** 16 Rust tests pass (3 allocator + 7 ingress + 4 probe pre-existing + 2 new smoke); no regressions

## Accomplishments

- **Task 1 — linear + decomposition wrappers (`f475c06`):**
  - `estimators/linear.rs`: `PyLinearRegression`, `PyRidge`, `PyLasso`, `PyElasticNet`, `PyLogisticRegression`. Regressors = `Fit` + `Predict` + `coef_`/`intercept_`; LogisticRegression = `Fit` + `PredictLabels` (i32) + `PredictProba` + `n_classes` + `coef_`/`intercept_`, exposing the sklearn `C` constructor param mapped to the Rust `c` field (`#[allow(non_snake_case)]`).
  - `estimators/decomposition.rs`: `PyPCA` (`Fit` unsupervised + `Transform` + `inverse_transform` + `components_`/`mean_`/`explained_variance_`/`explained_variance_ratio_`); `PyTruncatedSVD` (`Fit` + `Transform` + `components_`/`singular_values_`/`explained_variance_ratio_`, no `inverse_transform`).
  - `estimators/mod.rs`: module surface doc + submodule decls. `errors.rs`: added `not_fitted()` helper (the unfit-arm accessor/predict path → `PyValueError` the shim re-raises as `NotFittedError`, T-06-09).
- **Task 2 — cluster + neighbors wrappers (`5e84c26`):**
  - `estimators/cluster.rs`: `PyKMeans` (`Fit` + `PredictLabels` i32 + `cluster_centers_`/`labels_`/`inertia_`; `random_state`→`seed` via `with_opts(n_clusters, seed, max_iter, tol)`); `PyDBSCAN` (`Fit` + `labels_`/`core_sample_indices_` ONLY — no standalone `predict`, algos D-08; `eps` stays `f64`).
  - `estimators/neighbors.rs`: `PyNearestNeighbors` (`Fit` + `kneighbors` → `(Vec<F>, Vec<i32>)`, no `predict`); `PyKNeighborsClassifier` (`Fit` + `PredictLabels` i32 + `PredictProba` + `n_classes`); `PyKNeighborsRegressor` (`Fit` + `Predict`).
- **Task 3 — register all 12 + smoke test (`cfeb1cd`):**
  - `lib.rs`: `m.add_class::<…>()?` for all 12 wrappers (kept the `backend_supports_f64` `#[pyfunction]`).
  - `tests/pyclass_smoke_test.rs`: constructs every wrapper via the Rust-callable `unfit_default()` and asserts `is_unfit()` — proves the 12 `#[pyclass]` defs + macro expansion compile and instantiate without a Python interpreter or a live device. A purity/repeatability test confirms the ctor is side-effect-free (sklearn `__init__` contract the shim relies on).

## Task Commits

1. **Task 1: linear + decomposition wrappers (Predict/Transform)** — `f475c06` (feat)
2. **Task 2: cluster + neighbors wrappers (PredictLabels/KNeighbors)** — `5e84c26` (feat)
3. **Task 3: register all 12 + construction smoke test** — `cfeb1cd` (feat)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added Rust-callable `unfit_default()`/`is_unfit()` ctors for the smoke test**
- **Found during:** Task 3 (writing `pyclass_smoke_test.rs`).
- **Issue:** The plan's smoke test must "build the cdylib's pyclass types directly … without a Python interpreter". But a `#[pymethods]` `#[new]` is registered with the Python type object and is a *private* method on the impl — it is not callable from a separate test crate, and `Py::new`/`Bound::new` would require attaching a live interpreter (which the Plan-02 `extension-module`-as-feature rules deliberately avoid asserting in a Rust integration test).
- **Fix:** Added a `pub fn unfit_default() -> Self` and `pub fn is_unfit(&self) -> bool` on each of the 12 wrappers in a plain (non-`#[pymethods]`) `impl` block, mirroring the `#[new]` defaults. The smoke test calls these directly — pure-Rust construction, no interpreter, no device.
- **Files:** all 4 `estimators/*.rs`.
- **Commit:** `cfeb1cd`.

### Plan-text vs. reality notes (not deviations)

- **The `any_estimator!` macro emits ONLY the dispatch enum; the `#[pymethods]` bodies are hand-written.** Plan 02 shipped the macro as a documented skeleton (its own SUMMARY says "the `#[pymethods] fit` bodies are added by Plan 03"). So each wrapper invokes `any_estimator!` to generate its `Any<Name>` enum, then defines an explicit `#[pyclass]` struct + `#[pymethods]` with the trait-specific bodies. This is exactly the interface-first hand-off Plan 02 designed; the wrappers extend the bridge, they do not reinvent it.
- **Output methods are dtype-suffixed (`predict_f32`/`predict_f64`, `coef_f32`/`coef_f64`).** A `#[pyclass]` method cannot return a type generic over `F`, and PyO3 cannot overload by return type. The shim reads the wrapper's `dtype()` and calls the matching suffix. Label/index methods (`predict_labels`, `labels_`, `kneighbors` indices) are single-typed (`i32`) and need no suffix. This is the only practical shape for a non-generic `#[pyclass]` over a generic estimator.

## Threat-Model Outcomes

| Threat ID | Disposition | Evidence |
|-----------|-------------|----------|
| T-06-08 (untrusted hyperparameter → OOB device gather) | mitigated | The wrappers store params verbatim and pass them to the `mlrs_algos` `fit`, which validates `InvalidK`/`InvalidAlpha`/`InvalidEps`/`InvalidC`/… BEFORE any launch (unchanged); the wrapper maps the resulting `AlgoError` → `PyValueError` via `algo_err_to_py`. |
| T-06-09 (reading uninitialized fitted state) | mitigated | The `Unfit` arm holds no device buffers; every accessor / output method on `Unfit` returns `not_fitted()` (or the algos-level `NotFitted`) → `PyValueError` the shim maps to `NotFittedError`. `is_unfit()` is asserted by the smoke test. |
| T-06-10 (Rust panic unwinding into CPython) | mitigated | PyO3 catches panics at the `#[pymethods]` boundary; the validated bridge (`validated_f32/f64`, `float_dtype`) + `guard_f64()` make the expected error paths return mapped `PyErr`s, not panics. `mutex.expect()` is the only `panic!` site and only fires on a genuinely poisoned pool (an already-aborting condition). |

## Known Stubs

None. Every wrapper is fully wired to its `mlrs_algos` estimator (no hardcoded/placeholder fitted state, no mock data path). The dtype-suffixed accessor split is an API shape, not a stub — both arms call the real estimator accessor.

## Verification

- `cargo build -p mlrs-py --features cpu` — green (all 12 wrappers compile).
- `cargo build -p mlrs-py --features cpu,extension-module` — green (wheel link mode).
- `cargo test -p mlrs-py --features cpu --test pyclass_smoke_test` — 2 pass (all 12 construct `Unfit`).
- `cargo test -p mlrs-py --features cpu` — 16 pass (no regression to allocator/ingress/probe).
- `grep -c add_class crates/mlrs-py/src/lib.rs` → 12.
- `grep -c pyclass …/linear.rs` → 6 (5 classes), `…/decomposition.rs` → 3 (2), `…/cluster.rs` → 3 (2), `…/neighbors.rs` → 4 (3) = 12 total.
- LogisticRegression ctor exposes sklearn `C` (mapped to `c`); KMeans ctor maps `random_state`→`seed` via `with_opts`; DBSCAN has `labels_` but no `predict`.
- Every `fit`/`predict`/`transform`/`kneighbors`/`predict_proba` body contains `py.detach`; every `FloatDtype::F64` fit arm calls `guard_f64()?` before upload.
- clippy on the new estimator/errors/lib/test files is clean (remaining workspace clippy warnings are all pre-existing in mlrs-algos/mlrs-backend, out of scope).

## Next Phase Readiness

- **Plan 04 (pure-Python shim)** can now subclass sklearn and delegate to `mlrs._mlrs.{LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, KMeans, DBSCAN, PCA, TruncatedSVD, NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor}`. The low-level surface per class: `__init__(...)` (sklearn-named) → `fit(x, y, rows, cols)` → the relevant output method(s) (`predict_f32/f64`, `transform_f32/f64`, `predict_labels`, `predict_proba_f32/f64`, `kneighbors_f32/f64`) → fitted accessors (`coef_f32/f64`, `intercept_f32/f64`, `components_f32/f64`, `cluster_centers_f32/f64`, `labels_`, `inertia_f32/f64`, …). The shim reads `dtype()` to pick the f32/f64 suffix and `is_fitted()` for the `check_is_fitted` contract.
- No blockers.

## Self-Check: PASSED

- Files: `estimators/{mod,linear,cluster,decomposition,neighbors}.rs` FOUND; `tests/pyclass_smoke_test.rs` FOUND; `lib.rs` + `errors.rs` modified.
- Commits: `f475c06` FOUND, `5e84c26` FOUND, `cfeb1cd` FOUND.

---
*Phase: 06-python-surface-pyo3-estimators-per-backend-wheels*
*Completed: 2026-06-13*
