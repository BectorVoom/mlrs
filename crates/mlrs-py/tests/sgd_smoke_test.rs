//! Plan 10-01 Wave-0 — SGD / linear-SVM construction + fit smoke `#[ignore]`
//! scaffold (SGDSVM-01..04, PY-06 incremental share).
//!
//! This is a Rust integration test (separate crate linking the `mlrs-py` rlib,
//! AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`). The Wave-0
//! scaffold lands ONLY the four `any_estimator!` Unfit{} dispatch enums in
//! `estimators/linear.rs`; the `#[pyclass]` wrappers + their registration on the
//! `_mlrs` module are owned by the Wave-3 plan (so this scaffold compiles WITHOUT
//! the estimator bodies). Until then this smoke test exercises the Wave-0
//! construction surface that IS available — the `mlrs_algos` builders — proving
//! the builder-fronted construction path (D-01) compiles and lowers; it is
//! `#[ignore]` because the end-to-end PyO3 construct+fit path lands in Wave 3.
//!
//! Wave 3 replaces the body with the registered-`#[pyclass]` construct+fit smoke
//! (mirroring `spectral_smoke_test.rs`) and removes the `#[ignore]`.

use mlrs_algos::linear::linear_svc::LinearSVC;
use mlrs_algos::linear::linear_svr::LinearSVR;
use mlrs_algos::linear::mbsgd_classifier::MBSGDClassifier;
use mlrs_algos::linear::mbsgd_regressor::MBSGDRegressor;

/// All four builder-fronted estimators construct from their sklearn defaults
/// (D-01/D-03). `#[ignore]` Wave-0: Wave 3 promotes this to the registered
/// `#[pyclass]` construct+fit smoke over a live device.
#[test]
#[ignore = "Wave-3 wires the #[pyclass] registration + construct+fit PyO3 smoke"]
fn sgd_estimators_construct_from_builder() {
    MBSGDClassifier::<f32>::builder()
        .build::<f32>()
        .expect("MBSGDClassifier default builder");
    MBSGDRegressor::<f32>::builder()
        .build::<f32>()
        .expect("MBSGDRegressor default builder");
    LinearSVC::<f32>::builder()
        .build::<f32>()
        .expect("LinearSVC default builder");
    LinearSVR::<f32>::builder()
        .build::<f32>()
        .expect("LinearSVR default builder");
}
