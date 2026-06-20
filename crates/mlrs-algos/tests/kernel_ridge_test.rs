//! Plan 08-03 — KernelRidge (KERNEL-01) sklearn oracle tests.
//!
//! **Wave-0 Nyquist scaffold (08-01).** Every function below is `#[ignore]`d and
//! asserts ONLY that its committed `kernel_ridge_*` oracle fixture loads and is
//! shape-well-formed (the `X`/`y`/`X_test` inputs and the per-kernel reference
//! predictions `y_linear`/`y_rbf`/`y_poly`/`y_sigmoid`, the 2-target `y_multi`,
//! and the explicit-gamma `y_rbf_gamma`). It does NOT reference the not-yet-written
//! `KernelRidge` estimator, so this test crate COMPILES today. The Wave-2 plan
//! (08-03) removes `#[ignore]`, fits the device `KernelRidge` per case,
//! materializes `predict(X_test)`, and asserts against the sklearn reference
//! within the 1e-5 abs+rel contract (f64 strict `F64_TOL`; f32 a documented
//! per-family band).
//!
//! KernelRidge solves `(K + αI)·dual_coef_ = y` (the n×n training Gram K, D-02)
//! via the Phase-4 Cholesky primitive, then predicts `y = K_test · dual_coef_`.
//! Unlike v1 Ridge it fits RAW data with NO centering and NO intercept (D-06 /
//! Pitfall 1). `gamma=None` resolves to `1/n_features` at fit (D-05).
//!
//! Case families per dtype: one per kernel (linear/rbf/poly/sigmoid), a 2-target
//! multi-RHS case (D-04), and an explicit-gamma rbf case (D-05).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log, D-07). f32 runs on rocm. Per AGENTS.md §2 tests live
//! in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase, Tolerance, F64_TOL};

/// KernelRidge fixture geometry (gen_oracle.py `KR_N_SAMPLES` × `KR_N_FEATURES`,
/// `KR_N_TEST` test rows).
const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 4;
const N_TEST: usize = 5;

/// Documented f32 band for the KERNEL-01 predictions (set FROM the measurement
/// printed by the Wave-2 value test). f64 stays strict `F64_TOL` (1e-5).
#[allow(dead_code)]
const KR_F32_BAND: Tolerance = Tolerance::new(1e-4, 1e-4);

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Load a kernel_ridge oracle blob and assert the inputs + all per-kernel
/// reference predictions are shape-well-formed (the Wave-0 contract).
fn assert_fixture_well_formed(name: &str) -> OracleCase {
    let case = load_npz(fixture(name)).expect("kernel_ridge fixture loads");
    assert_eq!(
        case.expect_f64("X").len(),
        N_SAMPLES * N_FEATURES,
        "X is n_samples × n_features"
    );
    assert_eq!(case.expect_f64("y").len(), N_SAMPLES, "y is length n_samples");
    assert_eq!(
        case.expect_f64("y2").len(),
        N_SAMPLES * 2,
        "y2 is n_samples × 2 (multi-target)"
    );
    assert_eq!(
        case.expect_f64("X_test").len(),
        N_TEST * N_FEATURES,
        "X_test is n_test × n_features"
    );
    for k in ["y_linear", "y_rbf", "y_poly", "y_sigmoid", "y_rbf_gamma"] {
        assert_eq!(case.expect_f64(k).len(), N_TEST, "{k} is length n_test");
    }
    assert_eq!(
        case.expect_f64("y_multi").len(),
        N_TEST * 2,
        "y_multi is n_test × 2"
    );
    case
}

/// KERNEL-01 predictions vs sklearn `KernelRidge.predict` for all four kernels,
/// f64 strict `F64_TOL`. Gated by `skip_f64_with_log` (cpu runs; rocm skips).
/// Wave-2 (08-03) removes `#[ignore]` and fits the device estimator.
#[test]
#[ignore = "Wave-0 scaffold: KernelRidge estimator lands in plan 08-03"]
fn kernel_ridge_all_kernels_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kernel_ridge f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let _ = (assert_fixture_well_formed("kernel_ridge_f64_seed42.npz"), &F64_TOL);
}

/// KERNEL-01 predictions vs sklearn at the documented f32 band (`KR_F32_BAND`).
/// Runs on every backend (the f32 gate is rocm; cpu also exercises f32). Wave-2
/// (08-03) removes `#[ignore]` and fits the device estimator.
#[test]
#[ignore = "Wave-0 scaffold: KernelRidge estimator lands in plan 08-03"]
fn kernel_ridge_all_kernels_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let _ = assert_fixture_well_formed("kernel_ridge_f32_seed42.npz");
}

/// KERNEL-01 multi-target (multi-RHS, D-04) prediction vs sklearn, f64 strict.
/// The 2-target rbf case verifies the near-free multi-RHS Cholesky solve. Wave-2
/// (08-03) removes `#[ignore]` and fits the device estimator on `y2`.
#[test]
#[ignore = "Wave-0 scaffold: KernelRidge multi-target lands in plan 08-03"]
fn kernel_ridge_multi_target_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kernel_ridge multi f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = assert_fixture_well_formed("kernel_ridge_f64_seed42.npz");
    let _ = (case.expect_f64("y_multi"), &F64_TOL);
}
