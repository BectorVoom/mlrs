//! Plan 06-03, Task 3 — construction smoke test for all 12 estimator
//! `#[pyclass]` wrappers (PY-01).
//!
//! This is a Rust **integration test** (separate crate linking the `mlrs-py`
//! rlib, AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`). It builds
//! each of the 12 macro-expanded wrappers with its default hyperparameters via
//! the Rust-callable `unfit_default()` constructor and asserts it lands in the
//! `Unfit` arm (`is_unfit()`). This proves the `any_estimator!`-generated enum
//! shape + all 12 `#[pyclass]` definitions COMPILE and INSTANTIATE — the
//! compiled half of PY-01/PY-02/PY-05 — without needing a Python interpreter or
//! a live compute device (no `fit` is run, so no kernel launch / driver probe).
//!
//! The end-to-end Python-interpreter path (constructing via the registered
//! `_mlrs` module, running `fit`/`predict` against the oracle) is exercised by
//! the pytest harness in the later plans, where a live interpreter + device
//! exist.

use mlrs_py::estimators::cluster::{PyDBSCAN, PyKMeans};
use mlrs_py::estimators::decomposition::{PyPCA, PyTruncatedSVD};
use mlrs_py::estimators::linear::{
    PyElasticNet, PyLasso, PyLinearRegression, PyLogisticRegression, PyRidge,
};
use mlrs_py::estimators::neighbors::{
    PyKNeighborsClassifier, PyKNeighborsRegressor, PyNearestNeighbors,
};

/// Every wrapper constructs with default hyperparameters and starts `Unfit`.
#[test]
fn all_twelve_estimators_construct_unfit() {
    // linear_model (5)
    assert!(PyLinearRegression::unfit_default().is_unfit(), "LinearRegression");
    assert!(PyRidge::unfit_default().is_unfit(), "Ridge");
    assert!(PyLasso::unfit_default().is_unfit(), "Lasso");
    assert!(PyElasticNet::unfit_default().is_unfit(), "ElasticNet");
    assert!(PyLogisticRegression::unfit_default().is_unfit(), "LogisticRegression");

    // cluster (2)
    assert!(PyKMeans::unfit_default().is_unfit(), "KMeans");
    assert!(PyDBSCAN::unfit_default().is_unfit(), "DBSCAN");

    // decomposition (2)
    assert!(PyPCA::unfit_default().is_unfit(), "PCA");
    assert!(PyTruncatedSVD::unfit_default().is_unfit(), "TruncatedSVD");

    // neighbors (3)
    assert!(PyNearestNeighbors::unfit_default().is_unfit(), "NearestNeighbors");
    assert!(PyKNeighborsClassifier::unfit_default().is_unfit(), "KNeighborsClassifier");
    assert!(PyKNeighborsRegressor::unfit_default().is_unfit(), "KNeighborsRegressor");
}

/// Re-constructing does not panic and yields an independent `Unfit` instance —
/// the `#[new]`/`unfit_default` constructors are pure (sklearn `__init__`
/// purity contract the shim relies on, RESEARCH §estimator_checks).
#[test]
fn construction_is_pure_and_repeatable() {
    for _ in 0..3 {
        let a = PyLinearRegression::unfit_default();
        let b = PyKMeans::unfit_default();
        let c = PyNearestNeighbors::unfit_default();
        assert!(a.is_unfit() && b.is_unfit() && c.is_unfit());
    }
}
