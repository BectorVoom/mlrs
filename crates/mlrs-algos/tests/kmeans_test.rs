//! Plan 05-07 — KMeans (CLUSTER-01) Wave-0 oracle SCAFFOLD.
//!
//! Nyquist Wave-0 stub (04-01 precedent): every test below is `#[ignore]`d and
//! asserts ONLY that the committed `kmeans_{f32,f64}_seed42.npz` fixture loads
//! with its INJECTED `init` key present (D-09) and shape-well-formed arrays —
//! referencing NO `mlrs_algos::cluster::KMeans` symbol — so this crate COMPILES
//! today. Plan 05-07 removes `#[ignore]`, imports `KMeans`, runs Lloyd from the
//! injected init, and asserts centers/labels/inertia vs sklearn up to a label
//! permutation (`mlrs_core::best_match_accuracy == 1.0`, D-09) within 1e-5.
//!
//! f64 stubs carry the `skip_f64_with_log` gate (cpu runs f64; rocm skips, D-07).
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// KMeans fixture geometry (gen_oracle.py KM_N_SAMPLES × KM_N_FEATURES, K=KM_K).
const KM_N_SAMPLES: usize = 30;
const KM_N_FEATURES: usize = 4;
const KM_K: usize = 3;

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

/// LOAD-NOT-JUST-PRESENT: the `kmeans` fixture loads with the INJECTED `init`
/// (D-09) and well-formed X/centers/labels/inertia. WAVE-0 STUB — 05-07 wires the
/// real KMeans estimator oracle.
#[test]
#[ignore = "Wave-0 scaffold: KMeans estimator not implemented until plan 05-07"]
fn fixture_loads() {
    let case = load_npz(fixture("kmeans_f64_seed42.npz")).expect("load kmeans_f64");
    assert_len(&case, "init", KM_K * KM_N_FEATURES);
    assert_len(&case, "X", KM_N_SAMPLES * KM_N_FEATURES);
    assert_len(&case, "centers", KM_K * KM_N_FEATURES);
    assert_len(&case, "labels", KM_N_SAMPLES);
    assert_len(&case, "inertia", 1);
}

/// centers/labels match sklearn up to a label permutation (D-09), f32.
/// WAVE-0 STUB — 05-07 wires the real assertion.
#[test]
#[ignore = "Wave-0 scaffold: KMeans estimator not implemented until plan 05-07"]
fn kmeans_centers_labels_match_sklearn_f32() {
    let case = load_npz(fixture("kmeans_f32_seed42.npz")).expect("load kmeans_f32");
    assert_len(&case, "labels", KM_N_SAMPLES);
}

/// inertia matches sklearn, f64 (cpu runs; rocm skips). WAVE-0 STUB.
#[test]
#[ignore = "Wave-0 scaffold: KMeans estimator not implemented until plan 05-07"]
fn kmeans_inertia_matches_sklearn_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("kmeans_f64_seed42.npz")).expect("load kmeans_f64");
    assert_len(&case, "inertia", 1);
}

/// `predict` assigns new points to the fitted centers (D-08). WAVE-0 STUB.
#[test]
#[ignore = "Wave-0 scaffold: KMeans estimator not implemented until plan 05-07"]
fn kmeans_predict_assigns_to_centers_f32() {
    let case = load_npz(fixture("kmeans_f32_seed42.npz")).expect("load kmeans_f32");
    assert_len(&case, "centers", KM_K * KM_N_FEATURES);
}
