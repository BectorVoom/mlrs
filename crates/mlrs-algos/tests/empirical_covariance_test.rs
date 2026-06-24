//! Plan 07-04 — EmpiricalCovariance (COV-01) sklearn oracle tests.
//!
//! Activated from the 07-01 Nyquist `#[ignore]` scaffold: each function now loads
//! its committed `sklearn.covariance.EmpiricalCovariance` fixture, fits the device
//! estimator, and asserts `covariance_` (ddof=0 MLE) / `location_` / `precision_`
//! (eig-based pinvh, D-05) against the sklearn reference within the 1e-5 abs+rel
//! contract (f64) or the documented `EMPCOV_F32_BAND` (f32).
//!
//! Two size families: a well-conditioned FULL-RANK case (16×5) and a
//! RANK-DEFICIENT case (4×6 — n ≤ p, so `covariance_` is singular and
//! `precision_ = pinvh(cov)` via the symmetric eig must hold without a Cholesky
//! inverse, D-05; the floored-eigenvalue pseudo-inverse must be finite).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log, D-07). f32 stays a documented per-family band. Per
//! AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::covariance::EmpiricalCovariance;
use mlrs_algos::typestate::Fit;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// Full-rank geometry (gen_oracle.py `EMPCOV_FULLRANK` = 16×5).
const FR_N: usize = 16;
const FR_P: usize = 5;
/// Rank-deficient geometry (gen_oracle.py `EMPCOV_RANKDEF` = 4×6, n ≤ p).
const RD_N: usize = 4;
const RD_P: usize = 6;

/// f32-on-rocm per-family tolerance band for EmpiricalCovariance, pinned from the
/// standalone-estimator measurement in plan 07-04. f64 stays strict `F64_TOL`
/// (1e-5). The covariance/precision host finalize accumulates only modest f32
/// round-off, so the strict `F32_TOL` band holds (measured max_abs well within
/// `F32_TOL` on cpu f32).
const EMPCOV_F32_BAND: Tolerance = F32_TOL;

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
        _ => unreachable!("empirical_covariance fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("empirical_covariance fixtures are f32/f64 only"),
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

/// Fitted EmpiricalCovariance host attributes for an oracle compare.
struct EmpCovFit {
    covariance: Vec<f64>,
    location: Vec<f64>,
    precision: Vec<f64>,
}

/// Load the fixture `X`, fit `EmpiricalCovariance(assume_centered,
/// store_precision=true)`, and return the fitted attributes host-promoted to f64.
fn fit_empcov_with<F>(
    case: &OracleCase,
    n_samples: usize,
    n_features: usize,
    assume_centered: bool,
) -> EmpCovFit
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_f64 = case.expect_f64("X").to_vec();
    let x_host: Vec<F> = x_f64.iter().map(|&v| f64_to::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);

    let est = EmpiricalCovariance::<F>::builder()
        .assume_centered(assume_centered)
        .store_precision(true)
        .build::<F>()
        .expect("EmpiricalCovariance::build is infallible")
        .fit(&mut pool, &x_dev, None, (n_samples, n_features))
        .expect("EmpiricalCovariance::fit on a valid shape");

    let promote = |v: Vec<F>| v.iter().map(|&x| host_to_f64(x)).collect::<Vec<f64>>();
    EmpCovFit {
        covariance: promote(est.covariance_(&pool)),
        location: promote(est.location_(&pool)),
        precision: promote(est.precision_(&pool).expect("precision_ after fit")),
    }
}

/// `assume_centered=false` convenience wrapper (the default fit path).
fn fit_empcov<F>(case: &OracleCase, n_samples: usize, n_features: usize) -> EmpCovFit
where
    F: Float + CubeElement + Pod,
{
    fit_empcov_with::<F>(case, n_samples, n_features, false)
}

// ===========================================================================
// Full-rank case (16×5)
// ===========================================================================

/// `covariance_`/`location_`/`precision_` vs sklearn EmpiricalCovariance, full
/// rank, f32 (cpu + rocm).
#[test]
fn empirical_covariance_attrs_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("empirical_covariance_fullrank_f32_seed42.npz"))
        .expect("load empirical_covariance_fullrank_f32");
    let fit = fit_empcov::<f32>(&case, FR_N, FR_P);

    assert_close(
        &fit.covariance,
        case.expect_f64("covariance_"),
        &EMPCOV_F32_BAND,
        "covariance_ f32",
    );
    assert_close(
        &fit.location,
        case.expect_f64("location_"),
        &EMPCOV_F32_BAND,
        "location_ f32",
    );
    assert_close(
        &fit.precision,
        case.expect_f64("precision_"),
        &EMPCOV_F32_BAND,
        "precision_ f32",
    );
}

/// `covariance_`/`location_`/`precision_` vs sklearn, full rank, f64 (cpu runs;
/// rocm skips-with-log).
#[test]
fn empirical_covariance_attrs_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("empirical_covariance f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("empirical_covariance_fullrank_f64_seed42.npz"))
        .expect("load empirical_covariance_fullrank_f64");
    let fit = fit_empcov::<f64>(&case, FR_N, FR_P);

    assert_close(
        &fit.covariance,
        case.expect_f64("covariance_"),
        &F64_TOL,
        "covariance_ f64",
    );
    assert_close(
        &fit.location,
        case.expect_f64("location_"),
        &F64_TOL,
        "location_ f64",
    );
    assert_close(
        &fit.precision,
        case.expect_f64("precision_"),
        &F64_TOL,
        "precision_ f64",
    );
}

// ===========================================================================
// Rank-deficient case (4×6, n ≤ p) — the eig-based pinvh floor (D-05)
// ===========================================================================

/// `precision_ = pinvh(covariance_)` on a RANK-DEFICIENT (n ≤ p) covariance:
/// the eig-based pseudo-inverse floor must be FINITE (no inf/NaN) and match
/// sklearn's pinvh where a Cholesky inverse would fail (D-05), f64 (cpu runs;
/// rocm skips-with-log).
#[test]
fn empirical_covariance_precision_rank_deficient_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("empirical_covariance rank-deficient f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    assert!(RD_N <= RD_P, "rank-deficient fixture must satisfy n <= p");
    let case = load_npz(fixture("empirical_covariance_rankdef_f64_seed42.npz"))
        .expect("load empirical_covariance_rankdef_f64");
    let fit = fit_empcov::<f64>(&case, RD_N, RD_P);

    // The floored pseudo-inverse must be finite (the whole point of D-05).
    for (i, &v) in fit.precision.iter().enumerate() {
        assert!(
            v.is_finite(),
            "rank-deficient precision_ must be finite, got {v} at {i}"
        );
    }
    assert_close(
        &fit.covariance,
        case.expect_f64("covariance_"),
        &F64_TOL,
        "covariance_ rank-deficient f64",
    );
    assert_close(
        &fit.precision,
        case.expect_f64("precision_"),
        &F64_TOL,
        "precision_ rank-deficient f64",
    );
}

/// Rank-deficient `precision_`, f32 (cpu + rocm) — the f32 arm of the pinvh floor.
#[test]
fn empirical_covariance_precision_rank_deficient_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    assert!(RD_N <= RD_P, "rank-deficient fixture must satisfy n <= p");
    let case = load_npz(fixture("empirical_covariance_rankdef_f32_seed42.npz"))
        .expect("load empirical_covariance_rankdef_f32");
    let fit = fit_empcov::<f32>(&case, RD_N, RD_P);

    for (i, &v) in fit.precision.iter().enumerate() {
        assert!(
            v.is_finite(),
            "rank-deficient precision_ must be finite, got {v} at {i}"
        );
    }
    assert_close(
        &fit.precision,
        case.expect_f64("precision_"),
        &EMPCOV_F32_BAND,
        "precision_ rank-deficient f32",
    );
}

// ===========================================================================
// assume_centered=True case (16×5) — the SEPARATE uncentered host-Gram branch
// (WR-02: `mle_gram_uncentered`, Xᵀ·X/n, location_ all-zero)
// ===========================================================================

/// `covariance_`/`location_`/`precision_` vs sklearn
/// `EmpiricalCovariance(assume_centered=True)`, f32 (cpu + rocm). This is the
/// only test that value-gates the uncentered host-Gram branch (WR-02).
#[test]
fn empirical_covariance_assume_centered_attrs_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "assume_centered");
    let case = load_npz(fixture("empirical_covariance_centered_f32_seed42.npz"))
        .expect("load empirical_covariance_centered_f32");
    let fit = fit_empcov_with::<f32>(&case, FR_N, FR_P, true);

    // assume_centered ⇒ location_ is the all-zero vector.
    for (i, &v) in fit.location.iter().enumerate() {
        assert!(
            v.abs() <= EMPCOV_F32_BAND.abs,
            "assume_centered location_ must be ~0, got {v} at {i}"
        );
    }
    assert_close(
        &fit.covariance,
        case.expect_f64("covariance_"),
        &EMPCOV_F32_BAND,
        "covariance_ assume_centered f32",
    );
    assert_close(
        &fit.location,
        case.expect_f64("location_"),
        &EMPCOV_F32_BAND,
        "location_ assume_centered f32",
    );
    assert_close(
        &fit.precision,
        case.expect_f64("precision_"),
        &EMPCOV_F32_BAND,
        "precision_ assume_centered f32",
    );
}

/// `covariance_`/`location_`/`precision_` vs sklearn
/// `EmpiricalCovariance(assume_centered=True)`, f64 (cpu runs; rocm
/// skips-with-log) — value-gates the uncentered host-Gram branch (WR-02).
#[test]
fn empirical_covariance_assume_centered_attrs_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "assume_centered");
    if capability::skip_f64_with_log() {
        println!("empirical_covariance assume_centered f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("empirical_covariance_centered_f64_seed42.npz"))
        .expect("load empirical_covariance_centered_f64");
    let fit = fit_empcov_with::<f64>(&case, FR_N, FR_P, true);

    for (i, &v) in fit.location.iter().enumerate() {
        assert!(
            v.abs() <= F64_TOL.abs,
            "assume_centered location_ must be ~0, got {v} at {i}"
        );
    }
    assert_close(
        &fit.covariance,
        case.expect_f64("covariance_"),
        &F64_TOL,
        "covariance_ assume_centered f64",
    );
    assert_close(
        &fit.location,
        case.expect_f64("location_"),
        &F64_TOL,
        "location_ assume_centered f64",
    );
    assert_close(
        &fit.precision,
        case.expect_f64("precision_"),
        &F64_TOL,
        "precision_ assume_centered f64",
    );
}

/// BLDR-01 defaults equality: the zero-arg `new(false, true)` (sklearn defaults
/// `assume_centered=false`, `store_precision=true`) reproduces every
/// hyperparameter of `builder().build()` — the single-source-of-defaults
/// invariant (D-08).
#[test]
fn defaults_equal() {
    let from_new = EmpiricalCovariance::<f64>::new(false, true);
    let from_builder = EmpiricalCovariance::<f64>::builder()
        .build::<f64>()
        .expect("default EmpiricalCovariance builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "EmpiricalCovariance::new(false, true) must equal builder().build()"
    );
}
