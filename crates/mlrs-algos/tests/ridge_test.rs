//! Plan 04-05 — Ridge (LINEAR-02) sklearn oracle tests.
//!
//! Activated from the 04-01 Nyquist `#[ignore]` scaffold: each function now
//! loads its committed `Ridge(solver='cholesky', fit_intercept=True)` fixture
//! across a 3-alpha sweep {0.1, 1.0, 10.0}, fits the device estimator per alpha,
//! materializes `coef_`/`intercept_`, and asserts against the sklearn reference
//! within the 1e-5 abs+rel contract. Ridge solves `(XᵀX + αI)·coef = Xᵀy` via
//! the Phase-4 Cholesky primitive (D-02), with α on the Gram diagonal only and
//! the intercept recovered by centering (NEVER penalized, D-05).
//!
//! Two case families per dtype:
//!   - **alpha sweep** (`coef_`/`intercept_` across {0.1, 1.0, 10.0} vs sklearn).
//!   - **intercept-not-penalized** (D-05): the recovered intercept matches
//!     sklearn's (`ȳ − x̄·coef_`) — α applies only to `coef_`, never the bias —
//!     verified by reproducing sklearn's intercept analytically from the fitted
//!     `coef_` and the column means, and confirming it equals the fixture value.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm.
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::linear::ridge::Ridge;
use mlrs_algos::typestate::{Fit, Predict};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// Ridge fixture geometry (gen_oracle.py `LIN_N_SAMPLES` × `LIN_N_FEATURES`)
/// with a 3-alpha sweep {0.1, 1.0, 10.0}: coef is (n_alphas × n_features),
/// intercept is length n_alphas.
const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 4;
const N_ALPHAS: usize = 3;

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
        _ => unreachable!("ridge fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("ridge fixtures are f32/f64 only"),
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

/// Fit `Ridge(alpha, fit_intercept=true)` on the fixture `(X, y)` and return host
/// `(coef_, intercept_)`.
fn fit_coef_intercept<F>(case: &OracleCase, alpha: f64) -> (Vec<f64>, f64)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case
        .expect_f64("X")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let y_host: Vec<F> = case
        .expect_f64("y")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);

    let reg = Ridge::<F>::builder()
        .alpha(alpha)
        .fit_intercept(true)
        .build::<F>()
        .expect("Ridge builds with valid hyperparameters")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("Ridge::fit on a valid shape");

    let coef = reg.coef(&pool).iter().map(|&v| host_to_f64(v)).collect();
    let intercept = host_to_f64(reg.intercept(&pool));
    (coef, intercept)
}

/// Drive the full {0.1, 1.0, 10.0} alpha sweep, asserting `coef_`/`intercept_`
/// against the fixture's `(N_ALPHAS × N_FEATURES)` coef and length-`N_ALPHAS`
/// intercept.
fn run_alpha_sweep<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let alphas = case.expect_f64("alpha");
    let coef_ref = case.expect_f64("coef");
    let intercept_ref = case.expect_f64("intercept");
    assert_eq!(alphas.len(), N_ALPHAS, "fixture alpha sweep length");
    assert_eq!(coef_ref.len(), N_ALPHAS * N_FEATURES, "fixture coef length");
    assert_eq!(intercept_ref.len(), N_ALPHAS, "fixture intercept length");

    for (a_idx, &alpha) in alphas.iter().enumerate() {
        let (coef, intercept) = fit_coef_intercept::<F>(case, alpha);
        let expected_coef = &coef_ref[a_idx * N_FEATURES..(a_idx + 1) * N_FEATURES];
        assert_close(
            &coef,
            expected_coef,
            tol,
            &format!("{label} coef_ alpha={alpha}"),
        );
        assert_close(
            &[intercept],
            &[intercept_ref[a_idx]],
            tol,
            &format!("{label} intercept_ alpha={alpha}"),
        );
    }
}

/// `coef_`/`intercept_` across the alpha sweep vs sklearn, f32.
#[test]
fn ridge_coef_intercept_alpha_sweep_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ridge_f32_seed42.npz")).expect("load ridge_f32");
    run_alpha_sweep::<f32>(&case, &F32_TOL, "ridge f32");
}

/// `coef_`/`intercept_` across the alpha sweep vs sklearn, f64 (cpu runs; rocm skips).
#[test]
fn ridge_coef_intercept_alpha_sweep_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ridge f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("ridge_f64_seed42.npz")).expect("load ridge_f64");
    run_alpha_sweep::<f64>(&case, &F64_TOL, "ridge f64");
}

/// Verify the intercept is NOT penalized by α (D-05): for every alpha, the
/// recovered `intercept_` must equal the analytic center-then-solve form
/// `ȳ − x̄·coef_` computed from the fitted `coef_` and the (unpenalized) column
/// means — and that, in turn, equals the sklearn fixture value. If α leaked into
/// the intercept, the recovered bias would diverge from this analytic form.
fn run_intercept_not_penalized<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let alphas = case.expect_f64("alpha");
    let x = case.expect_f64("X");
    let y = case.expect_f64("y");
    let intercept_ref = case.expect_f64("intercept");

    // Unpenalized column means (the intercept is recovered from these, never the
    // penalized system).
    let mut x_mean = [0.0f64; N_FEATURES];
    let mut y_mean = 0.0f64;
    for r in 0..N_SAMPLES {
        for c in 0..N_FEATURES {
            x_mean[c] += x[r * N_FEATURES + c];
        }
        y_mean += y[r];
    }
    for m in x_mean.iter_mut() {
        *m /= N_SAMPLES as f64;
    }
    y_mean /= N_SAMPLES as f64;

    for (a_idx, &alpha) in alphas.iter().enumerate() {
        let (coef, intercept) = fit_coef_intercept::<F>(case, alpha);
        // Analytic unpenalized intercept from the fitted coef_ and the means.
        let analytic = y_mean
            - x_mean
                .iter()
                .zip(coef.iter())
                .map(|(m, c)| m * c)
                .sum::<f64>();
        assert_close(
            &[intercept],
            &[analytic],
            tol,
            &format!("{label} intercept==analytic(ȳ−x̄·coef) alpha={alpha}"),
        );
        // And the analytic (=recovered) intercept matches sklearn's fixture value.
        assert_close(
            &[intercept],
            &[intercept_ref[a_idx]],
            tol,
            &format!("{label} intercept==sklearn alpha={alpha}"),
        );
    }
}

/// Intercept-not-penalized check, f32 (D-05).
#[test]
fn ridge_intercept_not_penalized_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ridge_f32_seed42.npz")).expect("load ridge_f32");
    run_intercept_not_penalized::<f32>(&case, &F32_TOL, "ridge f32");
}

/// Intercept-not-penalized check, f64 (cpu runs; rocm skips-with-log).
#[test]
fn ridge_intercept_not_penalized_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("ridge intercept f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("ridge_f64_seed42.npz")).expect("load ridge_f64");
    run_intercept_not_penalized::<f64>(&case, &F64_TOL, "ridge f64");
}

/// Sanity: a fitted Ridge can `predict`, exercising the device-resident
/// `coef_`/`intercept_` GEMM path (the `Predict` import is load-bearing). Asserts
/// predictions on the training X reproduce `X·coef_ + intercept_` (consistency,
/// not a separate oracle — the coef/intercept oracle above is the strict gate).
#[test]
fn ridge_predict_consistency_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("ridge_f32_seed42.npz")).expect("load ridge_f32");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<f32> = case.expect_f64("X").iter().map(|&v| v as f32).collect();
    let y_host: Vec<f32> = case.expect_f64("y").iter().map(|&v| v as f32).collect();
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y_host);

    let reg = Ridge::<f32>::builder()
        .alpha(1.0)
        .fit_intercept(true)
        .build::<f32>()
        .expect("Ridge builds")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("Ridge::fit");
    let pred = reg
        .predict(&mut pool, &x_dev, (N_SAMPLES, N_FEATURES))
        .expect("Ridge::predict on training X");
    let pred_host: Vec<f64> = pred.to_host(&pool).iter().map(|&v| v as f64).collect();

    // Reference: X·coef_ + intercept_ from the materialized fitted state.
    let coef: Vec<f64> = reg.coef(&pool).iter().map(|&v| v as f64).collect();
    let intercept = reg.intercept(&pool) as f64;
    let x64 = case.expect_f64("X");
    let mut reference = vec![0.0f64; N_SAMPLES];
    for r in 0..N_SAMPLES {
        let mut acc = intercept;
        for c in 0..N_FEATURES {
            acc += x64[r * N_FEATURES + c] * coef[c];
        }
        reference[r] = acc;
    }
    assert_close(
        &pred_host,
        &reference,
        &F32_TOL,
        "ridge predict==X·coef+b f32",
    );
}

/// BLDR-01: `Ridge::new()` equals `Ridge::builder().build()?` on the
/// hyperparameter subset (sklearn defaults: `alpha = 1.0`, `fit_intercept =
/// true`). Pure host comparison — no device, so no f64 gate.
#[test]
fn defaults_equal() {
    let from_new = Ridge::<f64>::new();
    let from_builder = Ridge::<f64>::builder()
        .build::<f64>()
        .expect("default RidgeBuilder builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "Ridge::new() and builder().build()? must agree on hyperparameters (BLDR-01)"
    );
}
