//! Plan 05-03 — Lloyd update + inertia primitive Wave-0 oracle SCAFFOLD.
//!
//! Nyquist Wave-0 stub: every test referencing the not-yet-existing
//! `prims::kmeans` symbol is `#[ignore]`d and asserts ONLY that the committed
//! `kmeans_{f32,f64}_seed42.npz` fixture loads (with the `init` key present, D-09)
//! and is shape-well-formed — so this crate COMPILES today against the empty
//! `prims::kmeans` stub. Plan 05-03 removes `#[ignore]` and wires the real
//! centroid-sum-by-label + inertia oracle (centers/labels/inertia within 1e-5 up
//! to a label permutation, run from the injected init).
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

/// LOAD-NOT-JUST-PRESENT: the `kmeans` fixture loads with the injected `init`
/// (D-09) and well-formed centers/labels/inertia shapes. WAVE-0 STUB — 05-03
/// wires the real Lloyd-update oracle on `prims::kmeans`.
#[test]
#[ignore = "Wave-0 scaffold: prims::kmeans not implemented until plan 05-03"]
fn fixture_loads() {
    let case = load_npz(fixture("kmeans_f64_seed42.npz")).expect("load kmeans_f64");
    assert_len(&case, "init", KM_K * KM_N_FEATURES);
    assert_len(&case, "centers", KM_K * KM_N_FEATURES);
    assert_len(&case, "labels", KM_N_SAMPLES);
    assert_len(&case, "inertia", 1);
}

/// Centroid sum-by-label reproduces sklearn `cluster_centers_` (up to perm), f32.
/// WAVE-0 STUB — 05-03 wires the real assertion.
#[test]
#[ignore = "Wave-0 scaffold: prims::kmeans not implemented until plan 05-03"]
fn lloyd_centers_match_sklearn_f32() {
    let case = load_npz(fixture("kmeans_f32_seed42.npz")).expect("load kmeans_f32");
    assert_len(&case, "centers", KM_K * KM_N_FEATURES);
}

/// Inertia (Σ d²) reproduces sklearn `inertia_`, f64 (cpu runs; rocm skips).
/// WAVE-0 STUB — 05-03 wires the real assertion.
#[test]
#[ignore = "Wave-0 scaffold: prims::kmeans not implemented until plan 05-03"]
fn lloyd_inertia_matches_sklearn_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("kmeans_f64_seed42.npz")).expect("load kmeans_f64");
    assert_len(&case, "inertia", 1);
}
