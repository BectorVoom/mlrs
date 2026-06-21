//! Plan 10-01 Wave-0 — MBSGDClassifier (SGDSVM-01) Nyquist `#[ignore]` scaffolds.
//!
//! These load the committed PINNED-DETERMINISTIC fixtures and assert
//! fixture-load + SHAPE only (they compile today). The Wave-1 plan un-ignores
//! them and wires the real device fit/predict/predict_proba against the oracle:
//!
//!   - `oracle` — `coef_`/`intercept_` value-match (constant-schedule hinge).
//!   - `exact_labels` — `predict(Xq)` exact labels (the HARD gate, Pitfall 4).
//!   - `proba` — `predict_proba(Xq)` value-match (the log-loss variant).
//!   - `default_matches_sklearn` — D-03 litmus: `builder().build()` default
//!     equals sklearn's default-constructor params.
//!
//! Every f64 oracle scaffold carries the `skip_f64_with_log` gate verbatim
//! (cpu runs f64; rocm skips-with-log, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_core::{load_npz, OracleCase};

/// MBSGDClassifier fixture geometry (gen_oracle.py `SGD_N_SAMPLES` ×
/// `SGD_N_FEATURES`, `SGD_N_QUERY` query rows).
const N_SAMPLES: usize = 40;
const N_FEATURES: usize = 4;
const N_QUERY: usize = 8;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Load the named fixture and assert the design / query / label shapes match the
/// pinned geometry (the Wave-0 fixture-load + shape Nyquist assert).
fn assert_fixture_shape(case: &OracleCase) {
    let x = case.expect_f64("X");
    let xq = case.expect_f64("Xq");
    let y = case.expect_f64("y");
    assert_eq!(x.len(), N_SAMPLES * N_FEATURES, "X shape");
    assert_eq!(xq.len(), N_QUERY * N_FEATURES, "Xq shape");
    assert_eq!(y.len(), N_SAMPLES, "y shape");
    assert_eq!(case.expect_f64("coef").len(), N_FEATURES, "coef shape");
    assert_eq!(case.expect_f64("intercept").len(), 1, "intercept shape");
    assert_eq!(case.expect_f64("predict").len(), N_QUERY, "predict shape");
}

/// SGDSVM-01 `coef_`/`intercept_` oracle (constant-schedule hinge). f64 carries
/// the `skip_f64_with_log` gate. `#[ignore]` Wave-0: asserts fixture-load + shape
/// only; Wave-1 wires the device fit + value compare.
#[test]
#[ignore = "Wave-1 (plan 10-02) wires MBSGDClassifier::fit + coef/intercept oracle"]
fn oracle() {
    // skip_f64_with_log: the f64 arm runs on cpu and skips-with-log on rocm (D-07).
    let case = load_npz(fixture("mbsgd_classifier_f64_seed42.npz"))
        .expect("load mbsgd_classifier_f64 fixture");
    assert_fixture_shape(&case);
}

/// SGDSVM-01 exact-label gate (the HARD gate, Pitfall 4 ±1 encoding). `#[ignore]`
/// Wave-0: fixture-load + shape only.
#[test]
#[ignore = "Wave-1 (plan 10-02) wires MBSGDClassifier::predict_labels exact gate"]
fn exact_labels() {
    let case = load_npz(fixture("mbsgd_classifier_f32_seed42.npz"))
        .expect("load mbsgd_classifier_f32 fixture");
    assert_fixture_shape(&case);
}

/// SGDSVM-01 `predict_proba` gate (the log-loss variant). f64 carries the
/// `skip_f64_with_log` gate. `#[ignore]` Wave-0: fixture-load + shape only.
#[test]
#[ignore = "Wave-1 (plan 10-02) wires MBSGDClassifier(log)::predict_proba oracle"]
fn proba() {
    // skip_f64_with_log: the f64 arm runs on cpu and skips-with-log on rocm.
    let case = load_npz(fixture("mbsgd_classifier_log_f64_seed42.npz"))
        .expect("load mbsgd_classifier_log_f64 fixture");
    let proba = case.expect_f64("predict_proba");
    // proba is rows × n_classes (2-class).
    assert_eq!(proba.len(), N_QUERY * 2, "predict_proba shape");
}

/// SGDSVM-01 D-03 litmus: the default builder reproduces sklearn's default
/// constructor params. `#[ignore]` Wave-0: the structural assert lives in
/// `sgd_config_test::build_default_lowers_sklearn_defaults`; Wave-1 extends this
/// to a full default-fit comparison.
#[test]
#[ignore = "Wave-1 (plan 10-02) extends to a default-fit-vs-sklearn comparison"]
fn default_matches_sklearn() {
    let case = load_npz(fixture("mbsgd_classifier_f32_seed42.npz"))
        .expect("load mbsgd_classifier_f32 fixture");
    assert_fixture_shape(&case);
}
