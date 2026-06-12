//! Plan 05-09 — LogisticRegression (LINEAR-05) Wave-0 oracle SCAFFOLD.
//!
//! Nyquist Wave-0 stub: every test below is `#[ignore]`d and asserts ONLY that
//! the committed `logistic_binary_{f32,f64}` AND `logistic_multi_{f32,f64}`
//! fixtures load and are shape-well-formed — referencing NO
//! `mlrs_algos::linear::logistic::LogisticRegression` symbol — so this crate
//! COMPILES today. Plan 05-09 removes `#[ignore]`, imports
//! `LogisticRegression`, fits via the L-BFGS host loop (stable softmax), and
//! asserts `predict`/`predict_proba` (PRIMARY gauge-invariant gate, Pitfall 5)
//! with `coef_` as a looser secondary check.
//!
//! f64 stubs carry the `skip_f64_with_log` gate (cpu runs f64; rocm skips, D-07).
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// LogReg fixture geometry (gen_oracle.py LOG_N_FEATURES; binary query 8,
/// multiclass query 6 = (8/3)*3).
const LOG_N_FEATURES: usize = 4;
const LOG_BINARY_N_QUERY: usize = 8;
const LOG_MULTI_N_QUERY: usize = 6;
const LOG_BINARY_N_CLASSES: usize = 2;
const LOG_MULTI_N_CLASSES: usize = 3;

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

/// LOAD-NOT-JUST-PRESENT: BOTH the binary and multiclass fixtures load with
/// well-formed predict/predict_proba/coef/intercept (binary coef 1×n_features,
/// multi 3×n_features — symmetric over-parameterized softmax). WAVE-0 STUB —
/// 05-09 wires the real LogReg oracle.
#[test]
#[ignore = "Wave-0 scaffold: LogisticRegression estimator not implemented until plan 05-09"]
fn fixture_loads() {
    let bin = load_npz(fixture("logistic_binary_f64_seed42.npz")).expect("load logistic_binary_f64");
    assert_len(&bin, "coef", LOG_N_FEATURES);
    assert_len(&bin, "intercept", 1);
    assert_len(&bin, "predict", LOG_BINARY_N_QUERY);
    assert_len(&bin, "predict_proba", LOG_BINARY_N_QUERY * LOG_BINARY_N_CLASSES);

    let multi = load_npz(fixture("logistic_multi_f64_seed42.npz")).expect("load logistic_multi_f64");
    assert_len(&multi, "coef", LOG_MULTI_N_CLASSES * LOG_N_FEATURES);
    assert_len(&multi, "intercept", LOG_MULTI_N_CLASSES);
    assert_len(&multi, "predict", LOG_MULTI_N_QUERY);
    assert_len(&multi, "predict_proba", LOG_MULTI_N_QUERY * LOG_MULTI_N_CLASSES);
}

/// binary predict/predict_proba (gauge-invariant PRIMARY gate) matches sklearn,
/// f32. WAVE-0 STUB.
#[test]
#[ignore = "Wave-0 scaffold: LogisticRegression estimator not implemented until plan 05-09"]
fn logistic_binary_predict_proba_match_sklearn_f32() {
    let case = load_npz(fixture("logistic_binary_f32_seed42.npz")).expect("load logistic_binary_f32");
    assert_len(&case, "predict_proba", LOG_BINARY_N_QUERY * LOG_BINARY_N_CLASSES);
}

/// multiclass predict/predict_proba (gauge-invariant PRIMARY gate) matches
/// sklearn, f64 (cpu runs; rocm skips). WAVE-0 STUB.
#[test]
#[ignore = "Wave-0 scaffold: LogisticRegression estimator not implemented until plan 05-09"]
fn logistic_multi_predict_proba_match_sklearn_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("logistic_multi_f64_seed42.npz")).expect("load logistic_multi_f64");
    assert_len(&case, "predict_proba", LOG_MULTI_N_QUERY * LOG_MULTI_N_CLASSES);
}
