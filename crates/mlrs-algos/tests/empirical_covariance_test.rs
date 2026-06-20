//! Plan 07-04 — EmpiricalCovariance (COV-01) sklearn oracle tests.
//!
//! WAVE-0 SCAFFOLD (this file is created by plan 07-01). Every test function is
//! `#[ignore]` and asserts ONLY fixture-load + shape well-formedness — it makes
//! NO reference to the not-yet-existent
//! `mlrs_algos::covariance::empirical_covariance` estimator (the module is an
//! empty stub until plan 07-04). This is the 04-01 / 05-01 Wave-0 pattern: the
//! test crate must COMPILE today; plan 07-04 removes the `#[ignore]`, wires the
//! real `EmpiricalCovariance::fit`, and turns each stub into the 1e-5 oracle
//! compare of `covariance_` / `location_` / `precision_`.
//!
//! Two size families: a well-conditioned FULL-RANK case (`empirical_covariance_
//! fullrank_*`, 16×5) and a RANK-DEFICIENT case (`empirical_covariance_rankdef_*`,
//! 4×6 — n ≤ p, so `covariance_` is singular and `precision_ = pinvh(cov)` via
//! the symmetric eig must hold without a Cholesky inverse, D-05).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log, D-07). f32 stays a documented per-family band. Per
//! AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, Tolerance, F32_TOL, F64_TOL};

/// Full-rank geometry (gen_oracle.py `EMPCOV_FULLRANK` = 16×5).
const FR_N: usize = 16;
const FR_P: usize = 5;
/// Rank-deficient geometry (gen_oracle.py `EMPCOV_RANKDEF` = 4×6, n ≤ p).
const RD_N: usize = 4;
const RD_P: usize = 6;

/// f32-on-rocm per-family tolerance band for EmpiricalCovariance, pinned from the
/// standalone-estimator measurement in plan 07-04 (Claude's-discretion, D-08
/// growth point). f64 stays strict `F64_TOL` (1e-5); this is the f32 placeholder
/// the estimator plan replaces with the measured band.
#[allow(dead_code)]
const EMPCOV_F32_BAND: Tolerance = F32_TOL;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// `covariance_`/`location_`/`precision_` vs sklearn EmpiricalCovariance, full
/// rank, f32 (cpu + rocm).
///
/// WAVE-0 STUB. Plan 07-04 removes `#[ignore]` and wires `EmpiricalCovariance::
/// new(false).fit(X)` → 1e-5/`EMPCOV_F32_BAND` compare.
#[test]
#[ignore = "wave-0 scaffold: covariance::empirical_covariance lands in plan 07-04"]
fn empirical_covariance_attrs_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("empirical_covariance_fullrank_f32_seed42.npz"))
        .expect("load empirical_covariance_fullrank_f32");
    assert_eq!(case.shape("X").expect("X").to_vec(), vec![FR_N as u64, FR_P as u64]);
    assert_eq!(
        case.shape("covariance_").expect("covariance_").to_vec(),
        vec![FR_P as u64, FR_P as u64]
    );
    assert_eq!(case.expect_f64("location_").len(), FR_P);
    assert_eq!(
        case.shape("precision_").expect("precision_").to_vec(),
        vec![FR_P as u64, FR_P as u64]
    );
}

/// `covariance_`/`location_`/`precision_` vs sklearn, full rank, f64 (cpu runs;
/// rocm skips-with-log).
///
/// WAVE-0 STUB. Plan 07-04 wires the f64 `EmpiricalCovariance::fit` 1e-5 compare.
#[test]
#[ignore = "wave-0 scaffold: covariance::empirical_covariance lands in plan 07-04"]
fn empirical_covariance_attrs_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("empirical_covariance f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("empirical_covariance_fullrank_f64_seed42.npz"))
        .expect("load empirical_covariance_fullrank_f64");
    assert_eq!(case.shape("X").expect("X").to_vec(), vec![FR_N as u64, FR_P as u64]);
    assert_eq!(case.expect_f64("covariance_").len(), FR_P * FR_P);
    assert_eq!(case.expect_f64("precision_").len(), FR_P * FR_P);
    let _ = &F64_TOL; // 1e-5 contract used by plan 07-04's compare.
}

/// `precision_ = pinvh(covariance_)` on a RANK-DEFICIENT (n ≤ p) covariance:
/// the eig-based pseudo-inverse floor must hold where a Cholesky inverse would
/// fail (D-05), f64 (cpu runs; rocm skips-with-log).
///
/// WAVE-0 STUB. Plan 07-04 wires the rank-deficient `precision_` 1e-5 compare.
#[test]
#[ignore = "wave-0 scaffold: covariance::empirical_covariance lands in plan 07-04"]
fn empirical_covariance_precision_rank_deficient_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("empirical_covariance rank-deficient f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("empirical_covariance_rankdef_f64_seed42.npz"))
        .expect("load empirical_covariance_rankdef_f64");
    assert_eq!(case.shape("X").expect("X").to_vec(), vec![RD_N as u64, RD_P as u64]);
    assert!(RD_N <= RD_P, "rank-deficient fixture must satisfy n <= p");
    assert_eq!(case.expect_f64("precision_").len(), RD_P * RD_P);
}

/// Rank-deficient `precision_`, f32 (cpu + rocm) — the f32 arm of the pinvh floor.
///
/// WAVE-0 STUB. Plan 07-04 wires the f32 rank-deficient `precision_` band compare.
#[test]
#[ignore = "wave-0 scaffold: covariance::empirical_covariance lands in plan 07-04"]
fn empirical_covariance_precision_rank_deficient_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("empirical_covariance_rankdef_f32_seed42.npz"))
        .expect("load empirical_covariance_rankdef_f32");
    assert_eq!(case.shape("X").expect("X").to_vec(), vec![RD_N as u64, RD_P as u64]);
    assert_eq!(case.expect_f64("precision_").len(), RD_P * RD_P);
}
