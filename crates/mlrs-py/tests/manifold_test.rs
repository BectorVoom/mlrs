//! Plan 12-04 (Wave-3, BLDR-04) — cross-crate smoke + runtime not-fitted analog
//! for the two PyO3 shells over the v3 TYPESTATE estimators (`PyUMAP`,
//! `PyHDBSCAN`).
//!
//! Rust **integration test** (separate crate linking the `mlrs-py` rlib,
//! AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`). It proves the
//! Phase-12 builder + typestate convention is invisible to Python WITHOUT a
//! Python interpreter or live device (per MEMORY "Python wheel untestable in
//! env" — the live `estimator_checks` / capsule FFI path is routed to UAT,
//! SHIM-03):
//!
//!   - `typestate_shells_construct_unfit` — both wrappers build via the
//!     Rust-callable `unfit_default()` and land in the `Unfit` arm, proving the
//!     `#[pyclass]` definitions + the `any_estimator_typestate!`-emitted enums
//!     COMPILE and INSTANTIATE (BLDR-04 smoke).
//!   - `not_fitted_before_fit` — calling the `embedding_`/`labels_` accessor on
//!     an `unfit_default()` instance returns an `Err` (the `not_fitted` runtime
//!     analog → `PyValueError`, D-13). The `Unfit` arm returns BEFORE touching
//!     the pool, so no live device is needed; the CONCRETE `PyValueError` class
//!     through the live PyO3 boundary is asserted in UAT (the interpreter is
//!     undefined at link for this Rust integration binary, sgd_smoke_test.rs
//!     precedent).

use mlrs_py::estimators::cluster::PyHDBSCAN;
use mlrs_py::estimators::manifold::PyUMAP;

/// Both typestate shells construct with default hyperparameters and start
/// `Unfit` (no Python interpreter / live device needed) — the BLDR-04 smoke that
/// the macro-expanded `Unfit/F32/F64` enum instantiates for a v3 estimator.
#[test]
fn typestate_shells_construct_unfit() {
    assert!(PyUMAP::unfit_default().is_unfit(), "UMAP starts Unfit");
    assert!(PyHDBSCAN::unfit_default().is_unfit(), "HDBSCAN starts Unfit");
}

/// The fitted accessor on an UNFIT shell returns the `not_fitted` runtime analog
/// (D-13) — the Python-boundary counterpart of Plan 03's compile-time gate (the
/// Python side has no compile guarantee, so the `Unfit` arm guards at runtime).
#[test]
fn not_fitted_before_fit() {
    // UMAP: embedding_ before fit is an Err (not_fitted → PyValueError).
    let umap = PyUMAP::unfit_default();
    assert!(
        umap.embedding_f32_for_test().is_err(),
        "UMAP embedding_ before fit must be a not_fitted error"
    );

    // HDBSCAN: labels_ before fit is an Err (not_fitted → PyValueError).
    let hdbscan = PyHDBSCAN::unfit_default();
    assert!(
        hdbscan.labels_for_test().is_err(),
        "HDBSCAN labels_ before fit must be a not_fitted error"
    );
}
