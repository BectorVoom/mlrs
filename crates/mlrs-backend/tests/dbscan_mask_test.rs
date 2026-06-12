//! Plan 05-04 — DBSCAN eps-core-mask primitive Wave-0 oracle SCAFFOLD.
//!
//! Nyquist Wave-0 stub: every test referencing the not-yet-existing
//! `prims::dbscan` symbol is `#[ignore]`d and asserts ONLY that the committed
//! `dbscan_{f32,f64}_seed42.npz` fixture loads and is shape-well-formed — so this
//! crate COMPILES today against the empty `prims::dbscan` stub. Plan 05-04 removes
//! `#[ignore]` and wires the real eps-neighborhood core-mask oracle (core =
//! eps-neighbor-count incl. self ≥ min_samples; `core_sample_indices_` match).
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
/// X/eps/min_samples/labels/core_sample_indices arrays. The `labels` array
/// carries the noise sentinel `-1` (validated by the i32 round-trip in
/// `topk_test`). WAVE-0 STUB — 05-04 wires the real core-mask oracle on
/// `prims::dbscan`.
#[test]
#[ignore = "Wave-0 scaffold: prims::dbscan not implemented until plan 05-04"]
fn fixture_loads() {
    let case = load_npz(fixture("dbscan_f64_seed42.npz")).expect("load dbscan_f64");
    assert_len(&case, "X", DB_N_SAMPLES * DB_N_FEATURES);
    assert_len(&case, "eps", 1);
    assert_len(&case, "min_samples", 1);
    assert_len(&case, "labels", DB_N_SAMPLES);
    // core_sample_indices length is data-dependent (≤ n_samples); just require it
    // to be present and within bounds.
    let core = case.expect_f64("core_sample_indices");
    assert!(
        core.len() <= DB_N_SAMPLES,
        "core_sample_indices length {} must be <= n_samples {DB_N_SAMPLES}",
        core.len()
    );
}

/// Core-mask reproduces sklearn `core_sample_indices_`, f32. WAVE-0 STUB — 05-04
/// wires the real assertion.
#[test]
#[ignore = "Wave-0 scaffold: prims::dbscan not implemented until plan 05-04"]
fn dbscan_core_mask_matches_sklearn_f32() {
    let case = load_npz(fixture("dbscan_f32_seed42.npz")).expect("load dbscan_f32");
    assert_len(&case, "labels", DB_N_SAMPLES);
}

/// eps-neighborhood includes self (count ≥ 1), f64 (cpu runs; rocm skips).
/// WAVE-0 STUB — 05-04 wires the real assertion.
#[test]
#[ignore = "Wave-0 scaffold: prims::dbscan not implemented until plan 05-04"]
fn dbscan_eps_neighborhood_includes_self_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("dbscan_f64_seed42.npz")).expect("load dbscan_f64");
    assert_len(&case, "eps", 1);
}
