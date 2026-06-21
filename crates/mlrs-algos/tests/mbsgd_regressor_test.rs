//! Plan 10-01 Wave-0 — MBSGDRegressor (SGDSVM-02) Nyquist `#[ignore]` scaffolds.
//!
//! Load the committed PINNED-DETERMINISTIC fixture and assert fixture-load +
//! SHAPE only (compile today). The Wave-1 plan un-ignores them and wires the real
//! device fit/predict against the oracle:
//!
//!   - `oracle` — `coef_`/`intercept_`/`predict(Xq)` value-match
//!     (squared_error + invscaling, pinned).
//!   - `default_matches_sklearn` — D-03 litmus: `builder().build()` default
//!     equals sklearn's `SGDRegressor` default-constructor params.
//!
//! Every f64 oracle scaffold carries the `skip_f64_with_log` gate verbatim
//! (D-07). Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_core::{load_npz, OracleCase};

/// MBSGDRegressor fixture geometry (gen_oracle.py `SGD_N_SAMPLES` ×
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

/// SGDSVM-02 `coef_`/`intercept_`/`predict` oracle. f64 carries the
/// `skip_f64_with_log` gate. `#[ignore]` Wave-0: fixture-load + shape only.
#[test]
#[ignore = "Wave-1 (plan 10-02) wires MBSGDRegressor::fit + predict oracle"]
fn oracle() {
    // skip_f64_with_log: the f64 arm runs on cpu and skips-with-log on rocm (D-07).
    let case = load_npz(fixture("mbsgd_regressor_f64_seed42.npz"))
        .expect("load mbsgd_regressor_f64 fixture");
    assert_fixture_shape(&case);
}

/// SGDSVM-02 D-03 litmus. `#[ignore]` Wave-0: fixture-load + shape only.
#[test]
#[ignore = "Wave-1 (plan 10-02) extends to a default-fit-vs-sklearn comparison"]
fn default_matches_sklearn() {
    let case = load_npz(fixture("mbsgd_regressor_f32_seed42.npz"))
        .expect("load mbsgd_regressor_f32 fixture");
    assert_fixture_shape(&case);
}
