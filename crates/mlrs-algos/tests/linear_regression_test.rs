//! Plan 04-01 — LinearRegression (LINEAR-01) Nyquist scaffold (`#[ignore]`).
//!
//! Wave-0 scaffold for the SVD-pseudo-inverse `LinearRegression` estimator
//! (D-02, pinned SVD-based by LINEAR-01). The estimator does not exist yet, so
//! every test here is a COMPILING `#[ignore]` stub that loads its committed
//! sklearn fixture and asserts only fixture shape/well-formedness — NO reference
//! to the `LinearRegression` symbol in any compiled body. Plan 04-03 removes the
//! `#[ignore]` markers and wires the real `coef_`/`intercept_`/`predict` vs
//! sklearn comparison (after sign handling) including the near-collinear
//! small-σ-cutoff case.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu
//! runs f64; rocm skips-with-log, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never as an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// LinearRegression fixture geometry (gen_oracle.py `LIN_N_SAMPLES` ×
/// `LIN_N_FEATURES`, `LIN_TEST_SAMPLES`).
const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 4;
const N_TEST: usize = 3;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn assert_len(case: &OracleCase, name: &str, len: usize) {
    let got = case.expect_f64(name).len();
    assert_eq!(
        got, len,
        "fixture array '{name}' should have {len} elements, got {got}"
    );
}

/// `coef_`/`intercept_` vs sklearn, f32 (04-03 wires the real estimator).
#[test]
#[ignore = "04-03 removes #[ignore] and wires coef_/intercept_ vs sklearn"]
fn linear_regression_coef_intercept_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("linear_regression_f32_seed42.npz")).expect("load linreg_f32");
    assert_len(&case, "X", N_SAMPLES * N_FEATURES);
    assert_len(&case, "y", N_SAMPLES);
    assert_len(&case, "coef", N_FEATURES);
    assert_len(&case, "intercept", 1);
}

/// `coef_`/`intercept_` vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
#[ignore = "04-03 removes #[ignore] and wires coef_/intercept_ vs sklearn"]
fn linear_regression_coef_intercept_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("linreg f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("linear_regression_f64_seed42.npz")).expect("load linreg_f64");
    assert_len(&case, "coef", N_FEATURES);
    assert_len(&case, "intercept", 1);
}

/// `predict(X_test)` vs sklearn, f32 (04-03 wires the device predict).
#[test]
#[ignore = "04-03 removes #[ignore] and wires predict(X_test) vs sklearn"]
fn linear_regression_predict_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("linear_regression_f32_seed42.npz")).expect("load linreg_f32");
    assert_len(&case, "X_test", N_TEST * N_FEATURES);
    assert_len(&case, "y_pred", N_TEST);
}

/// `predict(X_test)` vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
#[ignore = "04-03 removes #[ignore] and wires predict(X_test) vs sklearn"]
fn linear_regression_predict_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("linreg predict f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("linear_regression_f64_seed42.npz")).expect("load linreg_f64");
    assert_len(&case, "X_test", N_TEST * N_FEATURES);
    assert_len(&case, "y_pred", N_TEST);
}

/// Near-collinear small-σ-cutoff case, f32 (LINEAR-01 Pitfall 1): 04-03 asserts
/// the cutoff keeps `coef_col` finite + matching sklearn on the collinear X.
#[test]
#[ignore = "04-03 removes #[ignore] and wires the near-collinear small-σ cutoff case"]
fn linear_regression_collinear_cutoff_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("linear_regression_f32_seed42.npz")).expect("load linreg_f32");
    assert_len(&case, "X_coll", N_SAMPLES * N_FEATURES);
    assert_len(&case, "y_coll", N_SAMPLES);
    assert_len(&case, "coef_col", N_FEATURES);
    assert_len(&case, "intercept_col", 1);
}

/// Near-collinear small-σ-cutoff case, f64 (cpu runs; rocm skips-with-log).
#[test]
#[ignore = "04-03 removes #[ignore] and wires the near-collinear small-σ cutoff case"]
fn linear_regression_collinear_cutoff_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("linreg collinear f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("linear_regression_f64_seed42.npz")).expect("load linreg_f64");
    assert_len(&case, "X_coll", N_SAMPLES * N_FEATURES);
    assert_len(&case, "coef_col", N_FEATURES);
}
