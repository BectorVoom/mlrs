//! Regression metrics sklearn-oracle tests (TASK-12..14, METR-REG-01..03).
//!
//! Replays the committed `metrics_reg_{f32,f64}_seed42.npz` fixture (TASK-02)
//! against `mlrs_algos::metrics::regression::*`. Per AGENTS.md §2 tests live
//! in `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_algos::metrics::regression::{mean_absolute_error, mean_squared_error, r2_score};
use mlrs_core::{load_npz, OracleCase};

const TOL: f64 = 1e-5;
const ATOL_F32: f64 = 1e-4;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn load(name: &str) -> OracleCase {
    load_npz(fixture(name)).unwrap_or_else(|e| panic!("load {name}: {e}"))
}

fn f64_vec(case: &OracleCase, name: &str) -> Vec<f64> {
    case.expect_f64(name).to_vec()
}

fn f32_vec(case: &OracleCase, name: &str) -> Vec<f32> {
    case.expect_f32(name).to_vec()
}

fn scalar(case: &OracleCase, name: &str) -> f64 {
    case.expect_f64(name)[0]
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} (diff {})",
        (got - want).abs()
    );
}

// ==================== TASK-12 — METR-REG-01: r2_score ====================

#[test]
fn r2_score_constant_target_pins_sklearn_actual_value() {
    let case = load("metrics_reg_f64_seed42.npz");
    let y_true_const = f64_vec(&case, "y_true_const");
    let y_pred_const = f64_vec(&case, "y_pred_const");
    // Fixture-pinned value (read from the ACTUAL scikit-learn==1.9.0 output
    // at TASK-02 generation, not hand-derived — SPEC §5 REG note / §9 risk 5).
    let got = r2_score::<f64>(&y_true_const, &y_pred_const, None);
    assert_close(
        got,
        scalar(&case, "ref_r2_const"),
        TOL,
        "r2 constant-target",
    );
}

#[test]
fn r2_score_perfect_prediction_is_one() {
    let case = load("metrics_reg_f64_seed42.npz");
    let y_perfect = f64_vec(&case, "y_perfect");
    let got = r2_score::<f64>(&y_perfect, &y_perfect, None);
    assert_close(got, scalar(&case, "ref_r2_perfect"), TOL, "r2 perfect");
}

#[test]
fn r2_score_matches_sklearn_oracle_f64() {
    let case = load("metrics_reg_f64_seed42.npz");
    let y_true = f64_vec(&case, "y_true");
    let y_pred = f64_vec(&case, "y_pred");
    let got = r2_score::<f64>(&y_true, &y_pred, None);
    assert_close(got, scalar(&case, "ref_r2"), TOL, "r2 f64");
}

#[test]
fn r2_score_matches_sklearn_oracle_f32() {
    let case = load("metrics_reg_f32_seed42.npz");
    let y_true = f32_vec(&case, "y_true");
    let y_pred = f32_vec(&case, "y_pred");
    let got = r2_score::<f32>(&y_true, &y_pred, None);
    assert_close(got, scalar(&case, "ref_r2"), ATOL_F32, "r2 f32");
}

#[test]
fn r2_score_weighted_matches_sklearn_oracle() {
    let case = load("metrics_reg_f64_seed42.npz");
    let y_true = f64_vec(&case, "y_true");
    let y_pred = f64_vec(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    let got = r2_score::<f64>(&y_true, &y_pred, Some(&sw));
    assert_close(got, scalar(&case, "ref_r2_sw"), TOL, "r2 weighted");
}

// ==================== TASK-13 — METR-REG-02: mean_squared_error ====================

#[test]
fn mean_squared_error_perfect_prediction_is_zero() {
    let case = load("metrics_reg_f64_seed42.npz");
    let y_perfect = f64_vec(&case, "y_perfect");
    let got = mean_squared_error::<f64>(&y_perfect, &y_perfect, None);
    assert_close(got, scalar(&case, "ref_mse_perfect"), TOL, "mse perfect");
    assert_eq!(got, 0.0, "mse perfect must be exactly 0.0");
}

#[test]
fn mean_squared_error_matches_sklearn_oracle_f64() {
    let case = load("metrics_reg_f64_seed42.npz");
    let y_true = f64_vec(&case, "y_true");
    let y_pred = f64_vec(&case, "y_pred");
    let got = mean_squared_error::<f64>(&y_true, &y_pred, None);
    assert_close(got, scalar(&case, "ref_mse"), TOL, "mse f64");
}

#[test]
fn mean_squared_error_matches_sklearn_oracle_f32() {
    let case = load("metrics_reg_f32_seed42.npz");
    let y_true = f32_vec(&case, "y_true");
    let y_pred = f32_vec(&case, "y_pred");
    let got = mean_squared_error::<f32>(&y_true, &y_pred, None);
    assert_close(got, scalar(&case, "ref_mse"), ATOL_F32, "mse f32");
}

#[test]
fn mean_squared_error_weighted_matches_sklearn_oracle() {
    let case = load("metrics_reg_f64_seed42.npz");
    let y_true = f64_vec(&case, "y_true");
    let y_pred = f64_vec(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    let got = mean_squared_error::<f64>(&y_true, &y_pred, Some(&sw));
    assert_close(got, scalar(&case, "ref_mse_sw"), TOL, "mse weighted");
}

// ==================== TASK-14 — METR-REG-03: mean_absolute_error ====================

#[test]
fn mean_absolute_error_perfect_prediction_is_zero() {
    let case = load("metrics_reg_f64_seed42.npz");
    let y_perfect = f64_vec(&case, "y_perfect");
    let got = mean_absolute_error::<f64>(&y_perfect, &y_perfect, None);
    assert_close(got, scalar(&case, "ref_mae_perfect"), TOL, "mae perfect");
    assert_eq!(got, 0.0, "mae perfect must be exactly 0.0");
}

#[test]
fn mean_absolute_error_matches_sklearn_oracle_f64() {
    let case = load("metrics_reg_f64_seed42.npz");
    let y_true = f64_vec(&case, "y_true");
    let y_pred = f64_vec(&case, "y_pred");
    let got = mean_absolute_error::<f64>(&y_true, &y_pred, None);
    assert_close(got, scalar(&case, "ref_mae"), TOL, "mae f64");
}

#[test]
fn mean_absolute_error_matches_sklearn_oracle_f32() {
    let case = load("metrics_reg_f32_seed42.npz");
    let y_true = f32_vec(&case, "y_true");
    let y_pred = f32_vec(&case, "y_pred");
    let got = mean_absolute_error::<f32>(&y_true, &y_pred, None);
    assert_close(got, scalar(&case, "ref_mae"), ATOL_F32, "mae f32");
}

#[test]
fn mean_absolute_error_weighted_matches_sklearn_oracle() {
    let case = load("metrics_reg_f64_seed42.npz");
    let y_true = f64_vec(&case, "y_true");
    let y_pred = f64_vec(&case, "y_pred");
    let sw = f64_vec(&case, "sample_weight");
    let got = mean_absolute_error::<f64>(&y_true, &y_pred, Some(&sw));
    assert_close(got, scalar(&case, "ref_mae_sw"), TOL, "mae weighted");
}
