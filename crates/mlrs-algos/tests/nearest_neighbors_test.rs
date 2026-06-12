//! Plan 05-10 — NearestNeighbors (NEIGH-01) Wave-0 oracle SCAFFOLD.
//!
//! Nyquist Wave-0 stub: every test below is `#[ignore]`d and asserts ONLY that
//! the committed `knn_{f32,f64}_seed42.npz` fixture loads and its
//! `distances`/`indices` arrays are shape-well-formed — referencing NO
//! `mlrs_algos::neighbors::NearestNeighbors` symbol — so this crate COMPILES
//! today. Plan 05-10 removes `#[ignore]`, imports `NearestNeighbors`, calls
//! `kneighbors`, and asserts the k distances + indices vs sklearn within 1e-5.
//!
//! f64 stubs carry the `skip_f64_with_log` gate (cpu runs f64; rocm skips, D-07).
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// KNN fixture geometry (gen_oracle.py KNN_N_QUERY × KNN_K).
const KNN_N_QUERY: usize = 8;
const KNN_K: usize = 5;

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

/// LOAD-NOT-JUST-PRESENT: the `knn` fixture loads with well-formed
/// distances/indices (n_query × k). WAVE-0 STUB — 05-10 wires the real
/// `kneighbors` oracle.
#[test]
#[ignore = "Wave-0 scaffold: NearestNeighbors estimator not implemented until plan 05-10"]
fn fixture_loads() {
    let case = load_npz(fixture("knn_f64_seed42.npz")).expect("load knn_f64");
    assert_len(&case, "distances", KNN_N_QUERY * KNN_K);
    assert_len(&case, "indices", KNN_N_QUERY * KNN_K);
    assert_eq!(
        case.shape("distances"),
        Some([KNN_N_QUERY as u64, KNN_K as u64].as_slice())
    );
}

/// kneighbors distances match sklearn within 1e-5, f32. WAVE-0 STUB.
#[test]
#[ignore = "Wave-0 scaffold: NearestNeighbors estimator not implemented until plan 05-10"]
fn nearest_neighbors_distances_match_sklearn_f32() {
    let case = load_npz(fixture("knn_f32_seed42.npz")).expect("load knn_f32");
    assert_len(&case, "distances", KNN_N_QUERY * KNN_K);
}

/// kneighbors indices match sklearn, f64 (cpu runs; rocm skips). WAVE-0 STUB.
#[test]
#[ignore = "Wave-0 scaffold: NearestNeighbors estimator not implemented until plan 05-10"]
fn nearest_neighbors_indices_match_sklearn_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("knn_f64_seed42.npz")).expect("load knn_f64");
    assert_len(&case, "indices", KNN_N_QUERY * KNN_K);
}
