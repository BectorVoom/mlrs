//! Plan 05-08 — DBSCAN (CLUSTER-02) Wave-0 oracle SCAFFOLD.
//!
//! Nyquist Wave-0 stub: every test below is `#[ignore]`d and asserts ONLY that
//! the committed `dbscan_{f32,f64}_seed42.npz` fixture loads and is
//! shape-well-formed — referencing NO `mlrs_algos::cluster::DBSCAN` symbol — so
//! this crate COMPILES today. Plan 05-08 removes `#[ignore]`, imports `DBSCAN`,
//! runs the device core-mask + host DFS expansion, and asserts `labels_`
//! (noise=-1) + `core_sample_indices_` vs sklearn up to a label permutation
//! (`mlrs_core::best_match_accuracy == 1.0`, D-09).
//!
//! f64 stubs carry the `skip_f64_with_log` gate (cpu runs f64; rocm skips, D-07).
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// DBSCAN fixture geometry (gen_oracle.py DB_N_SAMPLES × DB_N_FEATURES).
const DB_N_SAMPLES: usize = 40;
const DB_N_FEATURES: usize = 2;

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

/// LOAD-NOT-JUST-PRESENT: the `dbscan` fixture loads with well-formed
/// X/labels/core_sample_indices (labels carry the -1 noise sentinel). WAVE-0 STUB
/// — 05-08 wires the real DBSCAN estimator oracle.
#[test]
#[ignore = "Wave-0 scaffold: DBSCAN estimator not implemented until plan 05-08"]
fn fixture_loads() {
    let case = load_npz(fixture("dbscan_f64_seed42.npz")).expect("load dbscan_f64");
    assert_len(&case, "X", DB_N_SAMPLES * DB_N_FEATURES);
    assert_len(&case, "labels", DB_N_SAMPLES);
    let core = case.expect_f64("core_sample_indices");
    assert!(
        core.len() <= DB_N_SAMPLES,
        "core_sample_indices length {} must be <= n_samples {DB_N_SAMPLES}",
        core.len()
    );
}

/// labels (noise=-1) match sklearn up to a permutation (D-09), f32. WAVE-0 STUB.
#[test]
#[ignore = "Wave-0 scaffold: DBSCAN estimator not implemented until plan 05-08"]
fn dbscan_labels_match_sklearn_f32() {
    let case = load_npz(fixture("dbscan_f32_seed42.npz")).expect("load dbscan_f32");
    assert_len(&case, "labels", DB_N_SAMPLES);
}

/// core_sample_indices_ match sklearn, f64 (cpu runs; rocm skips). WAVE-0 STUB.
#[test]
#[ignore = "Wave-0 scaffold: DBSCAN estimator not implemented until plan 05-08"]
fn dbscan_core_sample_indices_match_sklearn_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("dbscan_f64_seed42.npz")).expect("load dbscan_f64");
    let core = case.expect_f64("core_sample_indices");
    assert!(core.len() <= DB_N_SAMPLES);
}
