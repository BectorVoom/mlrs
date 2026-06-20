//! Plan 07-04 — LedoitWolf (COV-02) sklearn oracle tests.
//!
//! WAVE-0 SCAFFOLD (this file is created by plan 07-01). Every test function is
//! `#[ignore]` and asserts ONLY fixture-load + shape well-formedness — it makes
//! NO reference to the not-yet-existent `mlrs_algos::covariance::ledoit_wolf`
//! estimator (the module is an empty stub until plan 07-04). This is the 04-01 /
//! 05-01 Wave-0 pattern: the test crate must COMPILE today; plan 07-04 removes
//! the `#[ignore]`, wires the real `LedoitWolf::fit`, and turns each stub into
//! the 1e-5 oracle compare of `covariance_` + `shrinkage_`.
//!
//! Two sample counts `n` (ROADMAP criterion 3): `ledoit_wolf_n12_*` (12×5) and
//! `ledoit_wolf_n40_*` (40×5). The committed fixtures use a CORRELATED
//! low-rank-plus-noise design so `shrinkage_` lands strictly inside `(0, 1)`
//! (≈0.18 / ≈0.12) rather than the degenerate `1.0` an identity-covariance
//! Gaussian produces — the closed-form β/δ arithmetic is actually exercised.
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log, D-07). f32 stays a documented per-family band
//! (`LW_F32_BAND` — Claude's-discretion, pinned from the standalone-estimator
//! measurement in plan 07-04). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, Tolerance, F32_TOL, F64_TOL};

/// LedoitWolf fixture feature count (gen_oracle.py `LW_P` = 5).
const LW_P: usize = 5;
/// The two sample counts (gen_oracle.py `LW_N_SMALL` / `LW_N_LARGE`).
const LW_N_SMALL: usize = 12;
const LW_N_LARGE: usize = 40;

/// f32-on-rocm per-family tolerance band for LedoitWolf, pinned from the
/// standalone-estimator measurement in plan 07-04 (Claude's-discretion, D-08
/// growth point — LedoitWolf's host β/δ shrink accumulates f32 round-off). f64
/// stays strict `F64_TOL` (1e-5); this is the f32 placeholder the estimator plan
/// replaces with the measured band.
#[allow(dead_code)]
const LW_F32_BAND: Tolerance = F32_TOL;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Assert `shrinkage_` is a length-1 array in `[0, 1]` (COV-02 invariant).
fn assert_shrinkage_in_unit_interval(case: &mlrs_core::OracleCase) {
    let s = case.expect_f64("shrinkage_");
    assert_eq!(s.len(), 1, "shrinkage_ is a length-1 array");
    assert!(
        (0.0..=1.0).contains(&s[0]),
        "shrinkage_ must lie in [0, 1], got {}",
        s[0]
    );
}

/// `covariance_` + `shrinkage_` vs sklearn LedoitWolf, n=12, f32 (cpu + rocm).
///
/// WAVE-0 STUB. Plan 07-04 removes `#[ignore]` and wires `LedoitWolf::new().fit(X)`
/// → 1e-5/`LW_F32_BAND` compare of `covariance_` + `shrinkage_`.
#[test]
#[ignore = "wave-0 scaffold: covariance::ledoit_wolf lands in plan 07-04"]
fn ledoit_wolf_small_n_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ledoit_wolf_n12_f32_seed42.npz"))
        .expect("load ledoit_wolf_n12_f32");
    assert_eq!(case.shape("X").expect("X").to_vec(), vec![LW_N_SMALL as u64, LW_P as u64]);
    assert_eq!(case.expect_f64("covariance_").len(), LW_P * LW_P);
    assert_shrinkage_in_unit_interval(&case);
}

/// `covariance_` + `shrinkage_` vs sklearn, n=12, f64 (cpu runs; rocm skips).
///
/// WAVE-0 STUB. Plan 07-04 wires the f64 `LedoitWolf::fit` 1e-5 compare.
#[test]
#[ignore = "wave-0 scaffold: covariance::ledoit_wolf lands in plan 07-04"]
fn ledoit_wolf_small_n_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ledoit_wolf n=12 f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("ledoit_wolf_n12_f64_seed42.npz"))
        .expect("load ledoit_wolf_n12_f64");
    assert_eq!(case.shape("X").expect("X").to_vec(), vec![LW_N_SMALL as u64, LW_P as u64]);
    assert_eq!(case.expect_f64("covariance_").len(), LW_P * LW_P);
    assert_shrinkage_in_unit_interval(&case);
    let _ = &F64_TOL; // 1e-5 contract used by plan 07-04's compare.
}

/// `covariance_` + `shrinkage_` vs sklearn LedoitWolf, n=40, f32 (cpu + rocm) —
/// the SECOND sample count (ROADMAP criterion 3).
///
/// WAVE-0 STUB. Plan 07-04 wires the n=40 f32 compare.
#[test]
#[ignore = "wave-0 scaffold: covariance::ledoit_wolf lands in plan 07-04"]
fn ledoit_wolf_large_n_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ledoit_wolf_n40_f32_seed42.npz"))
        .expect("load ledoit_wolf_n40_f32");
    assert_eq!(case.shape("X").expect("X").to_vec(), vec![LW_N_LARGE as u64, LW_P as u64]);
    assert_shrinkage_in_unit_interval(&case);
}

/// `covariance_` + `shrinkage_` vs sklearn, n=40, f64 (cpu runs; rocm skips) —
/// the SECOND sample count.
///
/// WAVE-0 STUB. Plan 07-04 wires the n=40 f64 1e-5 compare.
#[test]
#[ignore = "wave-0 scaffold: covariance::ledoit_wolf lands in plan 07-04"]
fn ledoit_wolf_large_n_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ledoit_wolf n=40 f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("ledoit_wolf_n40_f64_seed42.npz"))
        .expect("load ledoit_wolf_n40_f64");
    assert_eq!(case.shape("X").expect("X").to_vec(), vec![LW_N_LARGE as u64, LW_P as u64]);
    assert_shrinkage_in_unit_interval(&case);
}
