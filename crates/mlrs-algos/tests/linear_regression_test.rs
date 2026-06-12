//! Plan 04-03 — LinearRegression (LINEAR-01) sklearn oracle tests.
//!
//! Activated from the 04-01 Nyquist `#[ignore]` scaffold: each function now
//! loads its committed `LinearRegression(fit_intercept=True)` fixture (sklearn's
//! `scipy.linalg.lstsq` / gelsd contract, the exact LINEAR-01 pin), fits the
//! device estimator, materializes `coef_`/`intercept_`/`predict`, and asserts
//! against the sklearn reference within the 1e-5 abs+rel contract.
//!
//! Two case families per dtype:
//!   - **Full-rank** (`X`/`y`, `coef`/`intercept`, `X_test`/`y_pred`).
//!   - **Near-collinear** (`X_coll`/`y_coll`, `coef_col`/`intercept_col`): feature
//!     2 ≈ feature 0, so the design has a ~0 singular value. The SVD-pseudo-inverse
//!     small-σ cutoff (RESEARCH Pitfall 1 / T-04-03-01) must keep `coef_col`
//!     bounded and matching sklearn — a no-cutoff inverse explodes here.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm.
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::linear::linear_regression::LinearRegression;
use mlrs_algos::traits::{Fit, Predict};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// LinearRegression fixture geometry (gen_oracle.py `LIN_N_SAMPLES` ×
/// `LIN_N_FEATURES`, `LIN_TEST_SAMPLES`).
const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 4;
const N_TEST: usize = 3;

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
        _ => unreachable!("linreg fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ atol + rtol·|exp|`
/// (abs-OR-rel), the strict 1e-5 ABSOLUTE arm never loosened (the D-10 floored
/// precedent from `svd_test.rs`/`gemm_test.rs`).
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

/// Load the fixture, fit `LinearRegression(true)` on the `(x_key, y_key)` case,
/// and return host `(coef_, intercept_)`.
fn fit_coef_intercept<F>(case: &OracleCase, x_key: &str, y_key: &str) -> (Vec<f64>, f64)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case
        .expect_f64(x_key)
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let y_host: Vec<F> = case
        .expect_f64(y_key)
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);

    let mut reg = LinearRegression::<F>::new(true);
    reg.fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("LinearRegression::fit on a valid shape");

    let coef = reg
        .coef(&pool)
        .expect("coef_ after fit")
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    let intercept = host_to_f64(reg.intercept(&pool).expect("intercept_ after fit"));
    (coef, intercept)
}

/// Fit the full-rank case and return host `predict(X_test)`.
fn fit_predict<F>(case: &OracleCase) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let y_host: Vec<F> = case.expect_f64("y").iter().map(|&v| f64_to::<F>(v)).collect();
    let xt_host: Vec<F> = case
        .expect_f64("X_test")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);
    let xt_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xt_host);

    let mut reg = LinearRegression::<F>::new(true);
    reg.fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("fit full-rank");
    let pred = reg
        .predict(&mut pool, &xt_dev, (N_TEST, N_FEATURES))
        .expect("predict on X_test");
    pred.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect()
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("linreg fixtures are f32/f64 only"),
    }
}

/// `coef_`/`intercept_` vs sklearn, f32.
#[test]
fn linear_regression_coef_intercept_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("linear_regression_f32_seed42.npz")).expect("load linreg_f32");
    let (coef, intercept) = fit_coef_intercept::<f32>(&case, "X", "y");
    assert_close(&coef, case.expect_f64("coef"), &F32_TOL, "coef_ f32");
    assert_close(
        &[intercept],
        case.expect_f64("intercept"),
        &F32_TOL,
        "intercept_ f32",
    );
}

/// `coef_`/`intercept_` vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
fn linear_regression_coef_intercept_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("linreg f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("linear_regression_f64_seed42.npz")).expect("load linreg_f64");
    let (coef, intercept) = fit_coef_intercept::<f64>(&case, "X", "y");
    assert_close(&coef, case.expect_f64("coef"), &F64_TOL, "coef_ f64");
    assert_close(
        &[intercept],
        case.expect_f64("intercept"),
        &F64_TOL,
        "intercept_ f64",
    );
}

/// `predict(X_test)` vs sklearn, f32.
#[test]
fn linear_regression_predict_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("linear_regression_f32_seed42.npz")).expect("load linreg_f32");
    let pred = fit_predict::<f32>(&case);
    assert_close(&pred, case.expect_f64("y_pred"), &F32_TOL, "predict f32");
}

/// `predict(X_test)` vs sklearn, f64 (cpu runs; rocm skips-with-log).
#[test]
fn linear_regression_predict_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("linreg predict f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("linear_regression_f64_seed42.npz")).expect("load linreg_f64");
    let pred = fit_predict::<f64>(&case);
    assert_close(&pred, case.expect_f64("y_pred"), &F64_TOL, "predict f64");
}

/// Near-collinear small-σ-cutoff case, f32 (LINEAR-01 Pitfall 1): the cutoff
/// keeps `coef_col` finite + matching sklearn on the collinear `X_coll`.
#[test]
fn linear_regression_collinear_cutoff_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("linear_regression_f32_seed42.npz")).expect("load linreg_f32");
    let (coef, intercept) = fit_coef_intercept::<f32>(&case, "X_coll", "y_coll");
    // The cutoff must keep the coefficients bounded (a no-cutoff inverse explodes).
    assert!(
        coef.iter().all(|c| c.is_finite()),
        "collinear coef_col f32 must stay finite (cutoff active): {coef:?}"
    );
    assert_close(&coef, case.expect_f64("coef_col"), &F32_TOL, "coef_col f32");
    assert_close(
        &[intercept],
        case.expect_f64("intercept_col"),
        &F32_TOL,
        "intercept_col f32",
    );
}

/// Near-collinear small-σ-cutoff case, f64 (cpu runs; rocm skips-with-log).
#[test]
fn linear_regression_collinear_cutoff_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("linreg collinear f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("linear_regression_f64_seed42.npz")).expect("load linreg_f64");
    let (coef, intercept) = fit_coef_intercept::<f64>(&case, "X_coll", "y_coll");
    assert!(
        coef.iter().all(|c| c.is_finite()),
        "collinear coef_col f64 must stay finite (cutoff active): {coef:?}"
    );
    assert_close(&coef, case.expect_f64("coef_col"), &F64_TOL, "coef_col f64");
    assert_close(
        &[intercept],
        case.expect_f64("intercept_col"),
        &F64_TOL,
        "intercept_col f64",
    );
}
