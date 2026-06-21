//! Plan 09-01 — SpectralClustering (SPECTRAL-02) sklearn oracle scaffold.
//!
//! Wave-0 Nyquist `#[ignore]` scaffold: loads the committed fixture and asserts
//! fixture-load + SHAPE only (the estimator `fit` / `labels_` bodies are
//! `todo!()` until the Wave-3 plan 09-04, which un-ignores this and fills the
//! exact-labels-up-to-permutation assertion). Compiles + collects today against
//! the Wave-0 stubs.
//!
//! Case (un-ignored by 09-04):
//!   - `spectral_clustering` — `labels_` matches sklearn up to a label permutation
//!     on a WELL-SEPARATED fixture (D-10) via `mlrs_core::best_match_accuracy`.
//!     EXACT labels — no tolerance band (labels match or they don't).
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{best_match_accuracy, load_npz, OracleCase};

/// SpectralClustering fixture geometry (gen_oracle.py `SC_N_SAMPLES`).
const N_SAMPLES: usize = 12;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Read the reference `labels_` (stored as f64 in the `.npz`) into an `i64` slice
/// for the `best_match_accuracy` label-permutation compare.
fn ref_labels(case: &OracleCase) -> Vec<i64> {
    case.expect_f64("labels").iter().map(|&v| v as i64).collect()
}

/// SPECTRAL-02: `labels_` matches sklearn up to a label permutation on the
/// well-separated fixture (D-10) — EXACT labels, no band. (Wave-0 scaffold:
/// fixture-load + shape only; the live fit + `best_match_accuracy == 1.0`
/// assertion lands when 09-04 un-ignores.)
#[test]
#[ignore = "Wave-0 Nyquist scaffold; un-ignored + label-asserted by plan 09-04"]
fn spectral_clustering() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("spectral_clustering f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("spectral_clustering_f64_seed42.npz"))
        .expect("load spectral_clustering_f64");
    let labels_ref = ref_labels(&case);
    assert_eq!(labels_ref.len(), N_SAMPLES, "reference labels are length n");
    // Self-consistency of the permutation helper on the reference (a partition
    // always matches itself perfectly); the device-fitted compare lands in 09-04.
    let acc = best_match_accuracy(&labels_ref, &labels_ref);
    assert!(
        (acc - 1.0).abs() < 1e-12,
        "reference labels must perfectly self-match (best_match_accuracy {acc})"
    );
}
