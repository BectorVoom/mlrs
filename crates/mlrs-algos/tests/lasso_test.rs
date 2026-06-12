//! Plan 05-07 — Lasso (LINEAR-03) Wave-0 oracle SCAFFOLD.
//!
//! Nyquist Wave-0 stub: every test below is `#[ignore]`d and asserts ONLY that
//! the committed `lasso_{f32,f64}_seed42.npz` fixture loads and is
//! shape-well-formed — referencing NO `mlrs_algos::linear::lasso::Lasso` symbol —
//! so this crate COMPILES today. The fixture's `coef` carries genuine exact zeros
//! (Pitfall 1). Plan 05-07 removes `#[ignore]`, imports `Lasso` (= ElasticNet
//! `l1_ratio=1`), fits via coordinate descent, and asserts the sparse `coef_` +
//! `intercept_` vs sklearn within 1e-5.
//!
//! f64 stubs carry the `skip_f64_with_log` gate (cpu runs f64; rocm skips, D-07).
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// CD fixture geometry (gen_oracle.py CD_N_SAMPLES × CD_N_FEATURES).
const CD_N_SAMPLES: usize = 50;
const CD_N_FEATURES: usize = 8;

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

/// LOAD-NOT-JUST-PRESENT: the `lasso` fixture loads with well-formed
/// X/y/alpha/coef/intercept. WAVE-0 STUB — 05-07 wires the real Lasso oracle.
#[test]
#[ignore = "Wave-0 scaffold: Lasso estimator not implemented until plan 05-07"]
fn fixture_loads() {
    let case = load_npz(fixture("lasso_f64_seed42.npz")).expect("load lasso_f64");
    assert_len(&case, "X", CD_N_SAMPLES * CD_N_FEATURES);
    assert_len(&case, "y", CD_N_SAMPLES);
    assert_len(&case, "alpha", 1);
    assert_len(&case, "coef", CD_N_FEATURES);
    assert_len(&case, "intercept", 1);
}

/// sparse coef_ (exact zeros) matches sklearn, f32. WAVE-0 STUB.
#[test]
#[ignore = "Wave-0 scaffold: Lasso estimator not implemented until plan 05-07"]
fn lasso_sparse_coef_match_sklearn_f32() {
    let case = load_npz(fixture("lasso_f32_seed42.npz")).expect("load lasso_f32");
    assert_len(&case, "coef", CD_N_FEATURES);
}

/// coef_/intercept_ match sklearn, f64 (cpu runs; rocm skips). WAVE-0 STUB.
#[test]
#[ignore = "Wave-0 scaffold: Lasso estimator not implemented until plan 05-07"]
fn lasso_coef_intercept_match_sklearn_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("lasso_f64_seed42.npz")).expect("load lasso_f64");
    assert_len(&case, "intercept", 1);
}
