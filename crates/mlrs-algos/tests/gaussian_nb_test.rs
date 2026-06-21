//! Plan 11-01 Wave-0 — GaussianNB (NB-01) oracle test SCAFFOLDS.
//!
//! These are `#[ignore]`-marked Wave-1 oracle scaffolds: they load the committed
//! fixture and assert its SHAPE only (the estimator `fit` body is `todo!()` in
//! Wave 0). Wave 1 (11-02) un-ignores them and fills the fit/predict assertions:
//!
//!   - `exact_labels` / `exact_labels_f32` — `predict(Xq)` match sklearn EXACTLY
//!     (the HARD gate, integer labels, no band).
//!   - `proba_band` — `predict_proba(Xq)` value-match within the documented band
//!     (GaussianNB log-proba gets the WIDEST f32 band, A4).
//!   - `default_matches_sklearn` — `builder().build()` reproduces sklearn's
//!     default `GaussianNB` (D-02 litmus).
//!   - `build_rejects_bad_var_smoothing` — `build()` rejects `var_smoothing < 0`
//!     (D-05 validate-at-build).
//!   - `refit_releases_buffers` — the PoolStats no-leak gate across a re-fit.
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// GaussianNB fixture geometry (gen_oracle.py `NB_N_SAMPLES` // `NB_N_CLASSES` ×
/// `NB_N_FEATURES`, `NB_N_QUERY` // `NB_N_CLASSES` query rows, 3 classes).
const N_SAMPLES: usize = 39;
const N_FEATURES: usize = 4;
const N_QUERY: usize = 6;
const N_CLASSES: usize = 3;

/// predict_log_proba / predict_proba bands (the WIDEST f32 band per A4 — Wave 1
/// pins the final values; declared here so the scaffold and Wave-1 test share
/// one constant).
#[allow(dead_code)]
const PROBA_BAND_F64: f64 = 1e-5;
#[allow(dead_code)]
const PROBA_BAND_F32: f64 = 1e-2;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Assert the fixture's array shapes match the pinned NB geometry (the Wave-0
/// scaffold check; Wave 1 adds the fit/predict assertions).
fn assert_fixture_shape(case: &OracleCase) {
    assert_eq!(
        case.expect_f64("X").len(),
        N_SAMPLES * N_FEATURES,
        "X is N_SAMPLES x N_FEATURES"
    );
    assert_eq!(case.expect_f64("y").len(), N_SAMPLES, "y is N_SAMPLES");
    assert_eq!(
        case.expect_f64("Xq").len(),
        N_QUERY * N_FEATURES,
        "Xq is N_QUERY x N_FEATURES"
    );
    assert_eq!(
        case.expect_f64("predict").len(),
        N_QUERY,
        "predict is N_QUERY labels"
    );
    assert_eq!(
        case.expect_f64("predict_proba").len(),
        N_QUERY * N_CLASSES,
        "predict_proba is N_QUERY x N_CLASSES"
    );
}

/// HARD GATE (Wave 1): predict labels match sklearn EXACTLY, f32.
#[test]
#[ignore = "Wave-1 (11-02): GaussianNB::fit is todo!() in Wave 0"]
fn exact_labels_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("gaussian_nb_f32_seed42.npz")).expect("load gaussian_nb_f32");
    assert_fixture_shape(&case);
    // Wave 1: let (labels, _proba) = fit_gaussian::<f32>(&case); assert_eq!(labels, predict_ref);
}

/// HARD GATE (Wave 1): predict labels match sklearn EXACTLY, f64 (cpu runs; rocm skips).
#[test]
#[ignore = "Wave-1 (11-02): GaussianNB::fit is todo!() in Wave 0"]
fn exact_labels() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("gaussian_nb f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("gaussian_nb_f64_seed42.npz")).expect("load gaussian_nb_f64");
    assert_fixture_shape(&case);
}

/// proba band (Wave 1): predict_proba value-match within the documented band.
#[test]
#[ignore = "Wave-1 (11-02): GaussianNB::fit is todo!() in Wave 0"]
fn proba_band() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("gaussian_nb proba f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("gaussian_nb_f64_seed42.npz")).expect("load gaussian_nb_f64");
    assert_fixture_shape(&case);
}

/// D-02 litmus (Wave 1): bare builder().build() reproduces sklearn's default.
#[test]
#[ignore = "Wave-1 (11-02): GaussianNB::fit is todo!() in Wave 0"]
fn default_matches_sklearn() {
    let case = load_npz(fixture("gaussian_nb_f64_seed42.npz")).expect("load gaussian_nb_f64");
    assert_fixture_shape(&case);
}

/// build()-rejection (Wave 1): var_smoothing < 0 → BuildError::InvalidVarSmoothing.
#[test]
#[ignore = "Wave-1 (11-02): build-validation assertion lands with the fit body"]
fn build_rejects_bad_var_smoothing() {
    // Wave 1: let bad = GaussianNB::<f64>::builder().var_smoothing(-1.0).build::<f64>().err();
    //         assert!(matches!(bad, Some(BuildError::InvalidVarSmoothing { .. })));
}

/// PoolStats no-leak gate (Wave 1): live_bytes unchanged across a re-fit.
#[test]
#[ignore = "Wave-1 (11-02): GaussianNB::fit is todo!() in Wave 0"]
fn refit_releases_buffers() {
    let case = load_npz(fixture("gaussian_nb_f64_seed42.npz")).expect("load gaussian_nb_f64");
    assert_fixture_shape(&case);
}
