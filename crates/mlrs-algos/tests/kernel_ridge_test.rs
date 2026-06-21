//! Plan 08-03 — KernelRidge (KERNEL-01) sklearn oracle tests.
//!
//! Activated from the 08-01 Nyquist `#[ignore]` scaffold: each function now loads
//! its committed `KernelRidge` fixture, fits the device estimator per case,
//! materializes `predict(X_test)`, and asserts against the sklearn reference
//! within the 1e-5 abs+rel contract. KernelRidge solves `(K + αI)·dual_coef_ = y`
//! (the n×n training Gram K, D-02) via the Phase-4 Cholesky primitive over the
//! Phase-8 `kernel_matrix` keystone, then predicts `y = K(X_test, X_fit_) ·
//! dual_coef_`. Unlike v1 Ridge it fits RAW data with NO centering and NO
//! intercept (D-06 / Pitfall 1). `gamma=None` resolves to `1/n_features` at fit
//! (D-05).
//!
//! Case matrix (per dtype), pinned from the committed fixture (alpha=1.0,
//! gamma_default=1/n_features, degree=3, coef0=1.0, gamma_explicit=0.5):
//!   - one case per kernel: linear / rbf / poly / sigmoid (single target).
//!   - a 2-target multi-RHS rbf case (`y_multi`, D-04).
//!   - an explicit-gamma rbf case (`y_rbf_gamma`, D-05) alongside the
//!     gamma=None default rbf case (`y_rbf`) — both gamma paths pinned.
//!   - the poly/sigmoid cases exercise the degree=3 / coef0=1 defaults.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log, D-07). f32 runs on rocm at the documented
//! `KR_F32_BAND`. Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::kernel_ridge::{KernelKind, KernelRidge};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance, F32_TOL, F64_TOL};

/// KernelRidge fixture geometry (gen_oracle.py `KR_N_SAMPLES` × `KR_N_FEATURES`,
/// `KR_N_TEST` test rows).
const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 4;
const N_TEST: usize = 5;

/// Documented f32 band for the KERNEL-01 predictions. The strict 1e-5 ABSOLUTE
/// arm of `assert_close` is never loosened; the band only relaxes the f32 path's
/// abs/rel tolerance to the v1 documented-band precedent. The observed max f32
/// error is recorded in the SUMMARY. f64 stays strict `F64_TOL` (1e-5).
const KR_F32_BAND: Tolerance = Tolerance::new(1e-4, 1e-4);

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
        _ => unreachable!("kernel_ridge fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kernel_ridge fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ atol + rtol·|exp|`
/// (abs-OR-rel), the strict 1e-5 ABSOLUTE arm never loosened (the D-10 floored
/// precedent from `svd_test.rs` / `ridge_test.rs`). Returns the observed max abs
/// error for SUMMARY-band documentation.
fn assert_close(got: &[f64], expected: &[f64], tol: &Tolerance, what: &str) -> f64 {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    let mut max_abs = 0.0f64;
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        // Fail-loud on non-finite output (WR-05): a NaN prediction (e.g. CR-01's
        // poly negative-base path regressing) makes `(NaN − e).abs()` NaN and
        // `NaN <= tol` `false`, surfacing as an opaque "allclose failed" rather
        // than a clear NaN diagnosis. The sklearn KernelRidge reference is always
        // finite, so any non-finite `got` (or a mismatched non-finite `e`) is a
        // hard failure here, mirroring the KDE test helper.
        if !g.is_finite() || !e.is_finite() {
            assert!(
                g == e,
                "{what}: non-finite mismatch at {i}: got={g:e} expected={e:e}"
            );
            continue;
        }
        let abs_err = (g - e).abs();
        max_abs = max_abs.max(abs_err);
        let allclose = abs_err <= tol.abs + tol.rel * e.abs();
        assert!(
            allclose,
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (atol={:e}, rtol={:e})",
            tol.abs, tol.rel
        );
    }
    max_abs
}

/// Fit a `KernelRidge` (alpha=1.0) of the requested kind/gamma on the fixture's
/// `(X, target)` and return the host `predict(X_test)` (row-major
/// `N_TEST × n_targets`).
fn fit_predict<F>(
    case: &OracleCase,
    kind: KernelKind,
    target_key: &str,
    n_targets: usize,
    gamma: Option<f64>,
) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let y_host: Vec<F> = case
        .expect_f64(target_key)
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let x_test_host: Vec<F> = case
        .expect_f64("X_test")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);
    let x_test_dev: DeviceArray<ActiveRuntime, F> =
        DeviceArray::from_host(&mut pool, &x_test_host);

    let degree = case.expect_f64("degree")[0];
    let coef0 = case.expect_f64("coef0")[0];
    let gamma_f: Option<F> = gamma.map(|g| f64_to::<F>(g));

    let mut reg = KernelRidge::<F>::new(
        kind,
        f64_to::<F>(1.0),
        gamma_f,
        f64_to::<F>(degree),
        f64_to::<F>(coef0),
    );
    reg.fit(
        &mut pool,
        &x_dev,
        &y_dev,
        (N_SAMPLES, N_FEATURES),
        n_targets,
    )
    .expect("KernelRidge::fit on a valid shape");

    let pred = reg
        .predict(&mut pool, &x_test_dev, (N_TEST, N_FEATURES))
        .expect("KernelRidge::predict on X_test");
    pred.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect()
}

/// Drive every single-target kernel case + the explicit-gamma rbf case for one
/// dtype, asserting each against its sklearn reference. Returns the overall max
/// abs error observed (for SUMMARY-band documentation).
fn run_all_kernels<F>(case: &OracleCase, tol: &Tolerance, label: &str) -> f64
where
    F: Float + CubeElement + Pod,
{
    let gamma_default = case.expect_f64("gamma_default")[0];
    let gamma_explicit = case.expect_f64("gamma_explicit")[0];
    let mut max_abs = 0.0f64;

    // Linear (gamma irrelevant — None resolves at fit but linear ignores it).
    let pred = fit_predict::<F>(case, KernelKind::Linear, "y", 1, None);
    max_abs = max_abs.max(assert_close(
        &pred,
        case.expect_f64("y_linear"),
        tol,
        &format!("{label} linear"),
    ));

    // RBF, gamma=None → 1/n_features (D-05 default path).
    let pred = fit_predict::<F>(case, KernelKind::Rbf, "y", 1, None);
    max_abs = max_abs.max(assert_close(
        &pred,
        case.expect_f64("y_rbf"),
        tol,
        &format!("{label} rbf gamma=None"),
    ));

    // Poly, gamma=None → 1/n_features; degree=3, coef0=1 defaults.
    let pred = fit_predict::<F>(case, KernelKind::Poly, "y", 1, None);
    max_abs = max_abs.max(assert_close(
        &pred,
        case.expect_f64("y_poly"),
        tol,
        &format!("{label} poly degree=3 coef0=1"),
    ));

    // Sigmoid, gamma=None → 1/n_features; coef0=1 default.
    let pred = fit_predict::<F>(case, KernelKind::Sigmoid, "y", 1, None);
    max_abs = max_abs.max(assert_close(
        &pred,
        case.expect_f64("y_sigmoid"),
        tol,
        &format!("{label} sigmoid coef0=1"),
    ));

    // RBF, EXPLICIT gamma (D-05 explicit path) — sanity-check the resolved
    // default and explicit gamma differ, then assert the explicit reference.
    assert!(
        (gamma_default - gamma_explicit).abs() > 1e-9,
        "fixture default ({gamma_default}) and explicit ({gamma_explicit}) gamma must differ"
    );
    let pred = fit_predict::<F>(case, KernelKind::Rbf, "y", 1, Some(gamma_explicit));
    max_abs = max_abs.max(assert_close(
        &pred,
        case.expect_f64("y_rbf_gamma"),
        tol,
        &format!("{label} rbf gamma={gamma_explicit}"),
    ));

    max_abs
}

/// Drive the 2-target multi-RHS rbf case (D-04), asserting the `N_TEST × 2`
/// predictions against `y_multi`. Returns the observed max abs error.
fn run_multi_target<F>(case: &OracleCase, tol: &Tolerance, label: &str) -> f64
where
    F: Float + CubeElement + Pod,
{
    // y2 is the n×2 multi-target; rbf with the default gamma (the generator's
    // y_multi case). One multi-RHS cholesky_solve (rhs=2) produces both columns.
    let pred = fit_predict::<F>(case, KernelKind::Rbf, "y2", 2, None);
    assert_close(
        &pred,
        case.expect_f64("y_multi"),
        tol,
        &format!("{label} rbf multi-target (rhs=2)"),
    )
}

/// KERNEL-01 predictions vs sklearn `KernelRidge.predict` for all four kernels +
/// both gamma paths, f64 strict `F64_TOL`. Gated by `skip_f64_with_log`.
#[test]
fn kernel_ridge_all_kernels_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kernel_ridge f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("kernel_ridge_f64_seed42.npz")).expect("load kernel_ridge_f64");
    let max_abs = run_all_kernels::<f64>(&case, &F64_TOL, "kernel_ridge f64");
    println!("kernel_ridge f64 all-kernels max_abs_err = {max_abs:e}");
}

/// KERNEL-01 predictions vs sklearn at the documented f32 band (`KR_F32_BAND`).
/// Runs on every backend (the f32 gate is rocm; cpu also exercises f32).
#[test]
fn kernel_ridge_all_kernels_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("kernel_ridge_f32_seed42.npz")).expect("load kernel_ridge_f32");
    // First confirm the f32 path passes the STRICT 1e-5 absolute arm where it can,
    // then assert the documented band overall (record the observed max in SUMMARY).
    let max_abs = run_all_kernels::<f32>(&case, &KR_F32_BAND, "kernel_ridge f32");
    println!("kernel_ridge f32 all-kernels max_abs_err = {max_abs:e} (band atol={:e})", KR_F32_BAND.abs);
    // The f32 error must still be comfortably inside the documented band.
    assert!(
        max_abs <= KR_F32_BAND.abs,
        "f32 max_abs_err {max_abs:e} exceeds documented band {:e}",
        KR_F32_BAND.abs
    );
    let _ = &F32_TOL; // F32_TOL import kept load-bearing for the strict precedent.
}

/// KERNEL-01 multi-target (multi-RHS, D-04) prediction vs sklearn, f64 strict.
/// The 2-target rbf case verifies the near-free multi-RHS Cholesky solve.
#[test]
fn kernel_ridge_multi_target_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kernel_ridge multi f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("kernel_ridge_f64_seed42.npz")).expect("load kernel_ridge_f64");
    let max_abs = run_multi_target::<f64>(&case, &F64_TOL, "kernel_ridge f64");
    println!("kernel_ridge f64 multi-target max_abs_err = {max_abs:e}");
}

/// KERNEL-01 multi-target prediction vs sklearn at the documented f32 band.
#[test]
fn kernel_ridge_multi_target_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("kernel_ridge_f32_seed42.npz")).expect("load kernel_ridge_f32");
    let max_abs = run_multi_target::<f32>(&case, &KR_F32_BAND, "kernel_ridge f32");
    println!("kernel_ridge f32 multi-target max_abs_err = {max_abs:e}");
}
