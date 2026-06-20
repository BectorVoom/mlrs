//! Plan 07-04 — LedoitWolf (COV-02) sklearn oracle tests.
//!
//! Activated from the 07-01 Nyquist `#[ignore]` scaffold: each function now loads
//! its committed `sklearn.covariance.LedoitWolf` fixture, fits the device
//! estimator, and asserts `covariance_` + `shrinkage_` against the sklearn
//! reference within the 1e-5 abs+rel contract (f64) or the documented
//! `LW_F32_BAND` (f32). `shrinkage_ ∈ [0, 1]` is verified directly.
//!
//! Two sample counts `n` (ROADMAP criterion 3): `ledoit_wolf_n12_*` (12×5) and
//! `ledoit_wolf_n40_*` (40×5). The committed fixtures use a CORRELATED
//! low-rank-plus-noise design so `shrinkage_` lands strictly inside `(0, 1)`
//! (≈0.18 / ≈0.12) rather than the degenerate `1.0` an identity-covariance
//! Gaussian produces — the closed-form β/δ arithmetic is actually exercised.
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log, D-07). f32 stays a documented per-family band
//! (`LW_F32_BAND`). Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::covariance::LedoitWolf;
use mlrs_algos::traits::Fit;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// LedoitWolf fixture feature count (gen_oracle.py `LW_P` = 5).
const LW_P: usize = 5;
/// The two sample counts (gen_oracle.py `LW_N_SMALL` / `LW_N_LARGE`).
const LW_N_SMALL: usize = 12;
const LW_N_LARGE: usize = 40;

/// f32-on-rocm per-family tolerance band for LedoitWolf, pinned from the
/// standalone-estimator measurement in plan 07-04. f64 stays strict `F64_TOL`
/// (1e-5). The β/δ host finalize is performed in f64 internally, so the only f32
/// error is the upload/readback rounding of the small `p × p` covariance; the
/// strict `F32_TOL` band holds (measured well within `F32_TOL` on cpu f32).
const LW_F32_BAND: Tolerance = F32_TOL;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("ledoit_wolf fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("ledoit_wolf fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ atol + rtol·|exp|`
/// (abs-OR-rel), the strict 1e-5 ABSOLUTE arm never loosened.
fn assert_close(got: &[f64], expected: &[f64], tol: &Tolerance, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        let allclose = abs_err <= tol.abs + tol.rel * e.abs();
        assert!(
            allclose,
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (atol={:e}, rtol={:e})",
            tol.abs, tol.rel
        );
    }
}

/// Fitted LedoitWolf host attributes for an oracle compare.
struct LwFit {
    covariance: Vec<f64>,
    shrinkage: f64,
}

/// Load the fixture `X`, fit `LedoitWolf(assume_centered=false)`, and return the
/// fitted `covariance_` + `shrinkage_` host-promoted to f64.
fn fit_lw<F>(case: &OracleCase, n_samples: usize, n_features: usize) -> LwFit
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_f64 = case.expect_f64("X").to_vec();
    let x_host: Vec<F> = x_f64.iter().map(|&v| f64_to::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    let mut est = LedoitWolf::<F>::new(false);
    est.fit(&mut pool, &x_dev, None, (n_samples, n_features))
        .expect("LedoitWolf::fit on a valid shape");

    let covariance = est
        .covariance_(&pool)
        .expect("covariance_ after fit")
        .iter()
        .map(|&x| host_to_f64(x))
        .collect();
    let shrinkage = est.shrinkage_().expect("shrinkage_ after fit");
    LwFit {
        covariance,
        shrinkage,
    }
}

/// Assert the fitted `shrinkage_` lies in `[0, 1]` (COV-02 invariant).
fn assert_shrinkage_in_unit_interval(s: f64) {
    assert!(
        (0.0..=1.0).contains(&s),
        "shrinkage_ must lie in [0, 1], got {s}"
    );
}

// ===========================================================================
// n = 12 (small)
// ===========================================================================

/// `covariance_` + `shrinkage_` vs sklearn LedoitWolf, n=12, f32 (cpu + rocm).
#[test]
fn ledoit_wolf_small_n_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case =
        load_npz(fixture("ledoit_wolf_n12_f32_seed42.npz")).expect("load ledoit_wolf_n12_f32");
    let fit = fit_lw::<f32>(&case, LW_N_SMALL, LW_P);

    assert_shrinkage_in_unit_interval(fit.shrinkage);
    assert_close(
        &[fit.shrinkage],
        case.expect_f64("shrinkage_"),
        &LW_F32_BAND,
        "shrinkage_ n12 f32",
    );
    assert_close(
        &fit.covariance,
        case.expect_f64("covariance_"),
        &LW_F32_BAND,
        "covariance_ n12 f32",
    );
}

/// `covariance_` + `shrinkage_` vs sklearn, n=12, f64 (cpu runs; rocm skips).
#[test]
fn ledoit_wolf_small_n_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ledoit_wolf n=12 f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case =
        load_npz(fixture("ledoit_wolf_n12_f64_seed42.npz")).expect("load ledoit_wolf_n12_f64");
    let fit = fit_lw::<f64>(&case, LW_N_SMALL, LW_P);

    assert_shrinkage_in_unit_interval(fit.shrinkage);
    assert_close(
        &[fit.shrinkage],
        case.expect_f64("shrinkage_"),
        &F64_TOL,
        "shrinkage_ n12 f64",
    );
    assert_close(
        &fit.covariance,
        case.expect_f64("covariance_"),
        &F64_TOL,
        "covariance_ n12 f64",
    );
}

// ===========================================================================
// n = 40 (large) — the SECOND sample count (ROADMAP criterion 3)
// ===========================================================================

/// `covariance_` + `shrinkage_` vs sklearn LedoitWolf, n=40, f32 (cpu + rocm).
#[test]
fn ledoit_wolf_large_n_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case =
        load_npz(fixture("ledoit_wolf_n40_f32_seed42.npz")).expect("load ledoit_wolf_n40_f32");
    let fit = fit_lw::<f32>(&case, LW_N_LARGE, LW_P);

    assert_shrinkage_in_unit_interval(fit.shrinkage);
    assert_close(
        &[fit.shrinkage],
        case.expect_f64("shrinkage_"),
        &LW_F32_BAND,
        "shrinkage_ n40 f32",
    );
    assert_close(
        &fit.covariance,
        case.expect_f64("covariance_"),
        &LW_F32_BAND,
        "covariance_ n40 f32",
    );
}

/// `covariance_` + `shrinkage_` vs sklearn, n=40, f64 (cpu runs; rocm skips).
#[test]
fn ledoit_wolf_large_n_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ledoit_wolf n=40 f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case =
        load_npz(fixture("ledoit_wolf_n40_f64_seed42.npz")).expect("load ledoit_wolf_n40_f64");
    let fit = fit_lw::<f64>(&case, LW_N_LARGE, LW_P);

    assert_shrinkage_in_unit_interval(fit.shrinkage);
    assert_close(
        &[fit.shrinkage],
        case.expect_f64("shrinkage_"),
        &F64_TOL,
        "shrinkage_ n40 f64",
    );
    assert_close(
        &fit.covariance,
        case.expect_f64("covariance_"),
        &F64_TOL,
        "covariance_ n40 f64",
    );
}
