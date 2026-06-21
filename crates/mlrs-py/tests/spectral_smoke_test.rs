//! Plan 09-01 — SpectralEmbedding / SpectralClustering construction + smoke
//! scaffold (PY-06 incremental share).
//!
//! Rust **integration test** (separate crate linking the `mlrs-py` rlib,
//! AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`). Two parts:
//!
//!   - `spectral_estimators_construct_unfit` (runs today) — builds both wrappers
//!     via the Rust-callable `unfit_default()` and asserts they land in the
//!     `Unfit` arm, proving the two `any_estimator!`-generated enums + the two
//!     `#[pyclass]` definitions COMPILE and INSTANTIATE without a Python
//!     interpreter or a live device.
//!   - `spectral_fit_accessors` (Wave-0 `#[ignore]` scaffold) — the f32 + f64
//!     `fit` → `embedding_` / `labels_` smoke path, f64 gated by
//!     `backend_supports_f64()`. Un-ignored by the Wave-3 plan 09-04 once the
//!     algos `fit` bodies are filled (they are `todo!()` today).

use mlrs_py::estimators::spectral::{PySpectralClustering, PySpectralEmbedding};

/// Both spectral wrappers construct with default hyperparameters and start
/// `Unfit` (no Python interpreter / live device needed).
#[test]
fn spectral_estimators_construct_unfit() {
    assert!(
        PySpectralEmbedding::unfit_default().is_unfit(),
        "SpectralEmbedding"
    );
    assert!(
        PySpectralClustering::unfit_default().is_unfit(),
        "SpectralClustering"
    );
}

/// Wave-0 smoke scaffold: the `fit` → `embedding_` / `labels_` accessor path for
/// f32 + f64 (f64 gated by `backend_supports_f64()`). Un-ignored by 09-04 once
/// the algos `fit` bodies are filled — the device path is exercised through the
/// pytest harness with a live interpreter + device, mirroring the kernel-family
/// precedent (STATE.md [08-05]).
#[test]
#[ignore = "Wave-0 Nyquist scaffold; un-ignored by plan 09-04 once fit is filled"]
fn spectral_fit_accessors() {
    // The construction surface is proven by `spectral_estimators_construct_unfit`;
    // the live fit/accessor smoke (f32 always, f64 when `backend_supports_f64()`)
    // lands when 09-04 fills the algos `fit` bodies.
    assert!(PySpectralEmbedding::unfit_default().is_unfit());
    assert!(PySpectralClustering::unfit_default().is_unfit());
}
