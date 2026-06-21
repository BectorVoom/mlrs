//! Plan 11-01 Wave-0 — MultinomialNB (NB-02) oracle test SCAFFOLDS.
//!
//! `#[ignore]`-marked Wave-1 (11-03) scaffolds: load the committed fixture and
//! assert its SHAPE only (the `fit` body is `todo!()` in Wave 0). Wave 1
//! un-ignores them and fills the exact-label hard gate / proba band /
//! default-matches-sklearn / build-rejection / refit-no-leak assertions. f64
//! cases carry the `skip_f64_with_log` capability gate (cpu runs; rocm skips,
//! D-07). Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

const N_SAMPLES: usize = 39;
const N_FEATURES: usize = 4;
const N_QUERY: usize = 6;
const N_CLASSES: usize = 3;

#[allow(dead_code)]
const PROBA_BAND_F64: f64 = 1e-5;
#[allow(dead_code)]
const PROBA_BAND_F32: f64 = 1e-3;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn assert_fixture_shape(case: &OracleCase) {
    assert_eq!(case.expect_f64("X").len(), N_SAMPLES * N_FEATURES);
    assert_eq!(case.expect_f64("y").len(), N_SAMPLES);
    assert_eq!(case.expect_f64("Xq").len(), N_QUERY * N_FEATURES);
    assert_eq!(case.expect_f64("predict").len(), N_QUERY);
    assert_eq!(case.expect_f64("predict_proba").len(), N_QUERY * N_CLASSES);
}

/// HARD GATE (Wave 1): predict labels match sklearn EXACTLY, f32.
#[test]
#[ignore = "Wave-1 (11-03): MultinomialNB::fit is todo!() in Wave 0"]
fn exact_labels_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("multinomial_nb_f32_seed42.npz")).expect("load multinomial_nb_f32");
    assert_fixture_shape(&case);
}

/// HARD GATE (Wave 1): predict labels match sklearn EXACTLY, f64 (cpu; rocm skips).
#[test]
#[ignore = "Wave-1 (11-03): MultinomialNB::fit is todo!() in Wave 0"]
fn exact_labels() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("multinomial_nb f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("multinomial_nb_f64_seed42.npz")).expect("load multinomial_nb_f64");
    assert_fixture_shape(&case);
}

/// proba band (Wave 1).
#[test]
#[ignore = "Wave-1 (11-03): MultinomialNB::fit is todo!() in Wave 0"]
fn proba_band() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("multinomial_nb proba f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("multinomial_nb_f64_seed42.npz")).expect("load multinomial_nb_f64");
    assert_fixture_shape(&case);
}

/// D-02 litmus (Wave 1): bare builder().build() reproduces sklearn's default.
#[test]
#[ignore = "Wave-1 (11-03): MultinomialNB::fit is todo!() in Wave 0"]
fn default_matches_sklearn() {
    let case = load_npz(fixture("multinomial_nb_f64_seed42.npz")).expect("load multinomial_nb_f64");
    assert_fixture_shape(&case);
}

/// build()-rejection (Wave 1): alpha < 0 → BuildError::InvalidAlpha.
#[test]
#[ignore = "Wave-1 (11-03): build-validation assertion lands with the fit body"]
fn build_rejects_bad_alpha() {
    // Wave 1: assert MultinomialNB::<f64>::builder().alpha(-1.0).build::<f64>() is InvalidAlpha.
}

/// PoolStats no-leak gate (Wave 1): live_bytes unchanged across a re-fit.
#[test]
#[ignore = "Wave-1 (11-03): MultinomialNB::fit is todo!() in Wave 0"]
fn refit_releases_buffers() {
    let case = load_npz(fixture("multinomial_nb_f64_seed42.npz")).expect("load multinomial_nb_f64");
    assert_fixture_shape(&case);
}
