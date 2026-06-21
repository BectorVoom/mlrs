//! Plan 10-01 Wave-0 — LinearSVR (SGDSVM-04) Nyquist `#[ignore]` scaffold.
//!
//! Load the committed LinearSVR fixture and assert fixture-load + SHAPE only
//! (compile today). The Wave-2 plan un-ignores it and wires the real CD-solved
//! device fit/predict against the oracle (LinearSVR is liblinear CD — converged,
//! D-07; squared_epsilon_insensitive + epsilon; intercept via the
//! synthetic-feature `intercept_scaling`, Pitfall 5).
//!
//!   - `oracle` — `coef_`/`intercept_`/`predict(Xq)` value-match.
//!
//! The f64 oracle scaffold carries the `skip_f64_with_log` gate verbatim (D-07).
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_core::{load_npz, OracleCase};

/// LinearSVR fixture geometry (gen_oracle.py `SGD_N_SAMPLES` × `SGD_N_FEATURES`,
/// `SGD_N_QUERY` query rows).
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

fn assert_fixture_shape(case: &OracleCase) {
    assert_eq!(
        case.expect_f64("X").len(),
        N_SAMPLES * N_FEATURES,
        "X shape"
    );
    assert_eq!(
        case.expect_f64("Xq").len(),
        N_QUERY * N_FEATURES,
        "Xq shape"
    );
    assert_eq!(case.expect_f64("y").len(), N_SAMPLES, "y shape");
    assert_eq!(case.expect_f64("coef").len(), N_FEATURES, "coef shape");
    assert_eq!(case.expect_f64("intercept").len(), 1, "intercept shape");
    assert_eq!(case.expect_f64("predict").len(), N_QUERY, "predict shape");
}

/// SGDSVM-04 `coef_`/`intercept_`/`predict` oracle. f64 carries the
/// `skip_f64_with_log` gate. `#[ignore]` Wave-0: fixture-load + shape only.
#[test]
#[ignore = "Wave-2 (plan 10-04) wires LinearSVR::fit (CD reuse) + predict oracle"]
fn oracle() {
    // skip_f64_with_log: the f64 arm runs on cpu and skips-with-log on rocm (D-07).
    let case =
        load_npz(fixture("linear_svr_f64_seed42.npz")).expect("load linear_svr_f64 fixture");
    assert_fixture_shape(&case);
}
