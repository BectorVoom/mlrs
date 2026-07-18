//! TASK-08 (PY-ENS-01/RF-IMP-02/RF-OOB-02) — construction smoke test for
//! `PyRandomForestClassifier` (this task); `PyRandomForestRegressor`
//! (TASK-09) is appended here next.
//!
//! Rust **integration test** (separate crate linking the `mlrs-py` rlib,
//! AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`), mirroring
//! `tests/pyclass_smoke_test.rs`'s `unfit_default()` + `is_unfit()` pattern
//! (Plan 06-03 precedent): it proves the `any_estimator_typestate!`-generated
//! enum shape + the `#[pyclass]` definition + the `MaxFeaturesArg`
//! constructor-argument plumbing COMPILE and INSTANTIATE — the compiled half
//! of PY-ENS-01 — without needing a Python interpreter or a live compute
//! device (no `fit` is run).
//!
//! `#[pymethods]`-annotated methods (`fit`, `predict_labels`,
//! `predict_proba_f32/_f64`, `classes_`, `feature_importances_f32/_f64`,
//! `oob_score_f32/_f64`, `is_fitted`, `dtype`) are crate-private by
//! convention across every existing `mlrs-py` estimator wrapper (mirrors
//! `PyGaussianNB`/`PyKMeans` — only the hand-written `unfit_default()` /
//! `is_unfit()` pair outside the `#[pymethods]` block is `pub`), so they are
//! NOT directly callable from this separate integration-test crate; the full
//! FFI-boundary behavior (`fit` → `predict_labels`/`predict_proba`, the
//! `max_features` bogus-string `ValueError`, the `oob_score=True,
//! bootstrap=False` `ValueError`, `NotFittedError`) requires a live Python
//! interpreter with the built `mlrs._mlrs` extension AND its registration
//! (TASK-10) — that surface is exercised by the pytest harness in
//! `crates/mlrs-py/tests/test_random_forest.py`, mirroring the
//! `test_naive_bayes.py` precedent (the 10-05/08-05 "concrete FFI assertions
//! live in the pytest harness" convention this crate already established:
//! see `tests/pyclass_smoke_test.rs`'s own module doc).

use mlrs_py::estimators::ensemble::{
    PyHistGradientBoostingClassifier, PyHistGradientBoostingRegressor, PyRandomForestClassifier,
    PyRandomForestRegressor,
};

/// The `#[pyclass]` wrapper constructs with default hyperparameters and
/// starts `Unfit` — proves `PyRandomForestClassifier` + the
/// `any_estimator_typestate!`-generated `AnyRandomForestClassifier` enum +
/// the `MaxFeaturesArg` default (`Sqrt`, sklearn's classifier default)
/// compile and instantiate.
#[test]
fn random_forest_classifier_constructs_unfit() {
    let est = PyRandomForestClassifier::unfit_default();
    assert!(est.is_unfit(), "RandomForestClassifier");
}

/// TASK-09: the `#[pyclass]` wrapper constructs with default hyperparameters
/// and starts `Unfit` — proves `PyRandomForestRegressor` + the
/// `any_estimator_typestate!`-generated `AnyRandomForestRegressor` enum +
/// the `MaxFeaturesArg` default (`All`, sklearn's regressor default, NOT
/// `Sqrt`) compile and instantiate.
#[test]
fn random_forest_regressor_constructs_unfit() {
    let est = PyRandomForestRegressor::unfit_default();
    assert!(est.is_unfit(), "RandomForestRegressor");
}

/// TASK-18 (PY-ENS-03, structural): the `#[pyclass]` wrapper constructs with
/// default hyperparameters and starts `Unfit` — proves
/// `PyHistGradientBoostingClassifier` + its `any_estimator_typestate!`-generated
/// `AnyHistGradientBoostingClassifier` enum compile and instantiate, with NO
/// `max_features`/`bootstrap`/`oob_score` fields (HGB has none of these —
/// mechanically distinct from the RF `unfit_default()`s above).
#[test]
fn hist_gradient_boosting_classifier_constructs_unfit() {
    let est = PyHistGradientBoostingClassifier::unfit_default();
    assert!(est.is_unfit(), "HistGradientBoostingClassifier");
}

/// TASK-19 (PY-ENS-04, structural): the `#[pyclass]` wrapper constructs with
/// default hyperparameters and starts `Unfit` — proves
/// `PyHistGradientBoostingRegressor` + its `any_estimator_typestate!`-generated
/// `AnyHistGradientBoostingRegressor` enum compile and instantiate, mirroring
/// `hist_gradient_boosting_classifier_constructs_unfit` above (no
/// `max_features`/`bootstrap`/`oob_score`/`classes_`-related fields — HGB has
/// none of these, and the regressor additionally has no `feature_importances_`/
/// `oob_score_` accessors, SPEC §2 non-goal).
#[test]
fn hist_gradient_boosting_regressor_constructs_unfit() {
    let est = PyHistGradientBoostingRegressor::unfit_default();
    assert!(est.is_unfit(), "HistGradientBoostingRegressor");
}
