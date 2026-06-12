//! Plan 05-10 — KNeighborsRegressor (NEIGH-03) Wave-0 oracle SCAFFOLD.
//!
//! Nyquist Wave-0 stub: every test below is `#[ignore]`d and asserts ONLY that
//! the committed `knn_{f32,f64}_seed42.npz` fixture loads with its regressor
//! arrays (`y_reg`, `predict_reg`) shape-well-formed — referencing NO
//! `mlrs_algos::neighbors::KNeighborsRegressor` symbol — so this crate COMPILES
//! today. Plan 05-10 removes `#[ignore]`, imports `KNeighborsRegressor`, and
//! asserts `predict` (neighbor mean, via the `Predict<F>` surface) vs sklearn
//! within 1e-5.
//!
//! f64 stubs carry the `skip_f64_with_log` gate (cpu runs f64; rocm skips, D-07).
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// KNN fixture geometry (gen_oracle.py KNN_N_TRAIN / KNN_N_QUERY).
const KNN_N_TRAIN: usize = 30;
const KNN_N_QUERY: usize = 8;

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

/// LOAD-NOT-JUST-PRESENT: the `knn` fixture loads with well-formed y_reg
/// (n_train) + predict_reg (n_query). WAVE-0 STUB — 05-10 wires the real
/// regressor oracle.
#[test]
#[ignore = "Wave-0 scaffold: KNeighborsRegressor estimator not implemented until plan 05-10"]
fn fixture_loads() {
    let case = load_npz(fixture("knn_f64_seed42.npz")).expect("load knn_f64");
    assert_len(&case, "y_reg", KNN_N_TRAIN);
    assert_len(&case, "predict_reg", KNN_N_QUERY);
}

/// predict (neighbor mean) matches sklearn within 1e-5, f32. WAVE-0 STUB.
#[test]
#[ignore = "Wave-0 scaffold: KNeighborsRegressor estimator not implemented until plan 05-10"]
fn knn_regressor_predict_match_sklearn_f32() {
    let case = load_npz(fixture("knn_f32_seed42.npz")).expect("load knn_f32");
    assert_len(&case, "predict_reg", KNN_N_QUERY);
}

/// predict matches sklearn, f64 (cpu runs; rocm skips). WAVE-0 STUB.
#[test]
#[ignore = "Wave-0 scaffold: KNeighborsRegressor estimator not implemented until plan 05-10"]
fn knn_regressor_predict_match_sklearn_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("knn_f64_seed42.npz")).expect("load knn_f64");
    assert_len(&case, "predict_reg", KNN_N_QUERY);
}
