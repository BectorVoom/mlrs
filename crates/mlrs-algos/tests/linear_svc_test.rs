//! Plan 10-04 Wave-2 — LinearSVC (SGDSVM-03) sklearn oracle tests.
//!
//! Activated from the Wave-0 `#[ignore]` scaffold. The device estimator fits the
//! L2-regularized SQUARED-HINGE primal by the validated 05-06 L-BFGS primitive
//! (Open Question Q1 RESOLUTION — `cd_fit`'s soft-threshold squared-error CD does
//! NOT express the squared-hinge objective; the smooth+convex SVM primal reuses
//! `lbfgs_minimize`, the `logistic.rs` precedent — see the 10-04 SUMMARY), with the
//! intercept handled by the synthetic-feature `intercept_scaling` mechanism
//! (Pitfall 5 — NOT center-then-solve), and `dual='auto'` resolved INTERNALLY (the
//! `n_samples >= n_features` fixture → primal, D-07).
//!
//!   - `exact_labels` — predict labels match sklearn EXACTLY (HARD gate, integers).
//!   - `oracle` — `coef_`/`intercept_` value-match within tolerance (f64 a tight
//!     band — liblinear stops at tol=1e-4 slightly short of the true optimum, so
//!     the converged L-BFGS optimum agrees to ~1e-4, not strict 1e-5; f32 wider).
//!   - `default_matches_sklearn` — `builder().build()` with no setters reproduces
//!     the sklearn `LinearSVC` defaults (D-03 litmus).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs f64;
//! rocm skips per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm at a documented
//! band. Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an
//! in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::BuildError;
use mlrs_algos::linear::linear_svc::LinearSVC;
use mlrs_algos::linear::sgd_config::{LearningRate, Loss, Penalty};
use mlrs_algos::typestate::{Fit, PredictLabels};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// LinearSVC fixture geometry (gen_oracle.py `SGD_N_SAMPLES` × `SGD_N_FEATURES`,
/// `SGD_N_QUERY` query rows).
const N_SAMPLES: usize = 40;
const N_FEATURES: usize = 4;
const N_QUERY: usize = 8;

/// The pinned fixture hyperparameters (gen_oracle.py: C=1.0, intercept_scaling=1.0,
/// max_iter=1000, tol=1e-4, squared_hinge/l2). NOTE these are the EXPLICIT pinned
/// setters — distinct from the D-03 bare-default litmus below.
const SVM_C: f64 = 1.0;
const SVM_INTERCEPT_SCALING: f64 = 1.0;
const SVM_MAX_ITER: usize = 1000;

/// f64 coef/intercept band: liblinear (the oracle) stops at its own tol=1e-4
/// SLIGHTLY short of the true squared-hinge optimum, so the converged L-BFGS
/// optimum lands ~1e-4 away from the fixture's coefficients (not strict 1e-5).
/// This is NOT a solver-accuracy regression — both are valid near-optimum iterates
/// of the same strictly-convex objective; the EXACT predict labels (the hard gate)
/// confirm correctness. Documented band (RESEARCH Pitfall 6 — coef band, exact
/// labels strict).
const COEF_BAND_F64: f64 = 2e-4;
/// f32 coef/intercept band (round-off over the matvec accumulations + the liblinear
/// gap; the labels stay the exact hard gate).
const COEF_BAND_F32: f64 = 5e-3;

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
        _ => unreachable!("linear_svc fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("linear_svc fixtures are f32/f64 only"),
    }
}

fn assert_band(got: &[f64], expected: &[f64], band: f64, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= band + band * e.abs(),
            "{what}: band failed at {i}: got={g:e} expected={e:e} abs_err={abs_err:e} (band={band:e})"
        );
    }
}

/// Build + fit `LinearSVC` on the fixture with the EXPLICIT pinned hyperparameters
/// and return host `(coef_, intercept_, predict_labels(Xq))`.
fn fit_svc<F>(case: &OracleCase) -> (Vec<f64>, f64, Vec<i32>)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let y_host: Vec<F> = case.expect_f64("y").iter().map(|&v| f64_to::<F>(v)).collect();
    let xq_host: Vec<F> = case.expect_f64("Xq").iter().map(|&v| f64_to::<F>(v)).collect();

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_host);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq_host);

    // EXPLICIT pinned setters (Pitfall 7) — squared_hinge/l2, C, intercept_scaling,
    // max_iter; NOT the bare default (the D-03 litmus checks the default separately).
    let clf = LinearSVC::<F>::builder()
        .loss(Loss::SquaredHinge)
        .penalty(Penalty::L2)
        .c(SVM_C)
        .intercept_scaling(SVM_INTERCEPT_SCALING)
        .fit_intercept(true)
        .max_iter(SVM_MAX_ITER)
        .tol(1e-4)
        .build::<F>()
        .expect("LinearSVC builds with valid hyperparameters")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("LinearSVC::fit on a valid shape");

    let coef: Vec<f64> = clf
        .coef(&pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    let intercept = host_to_f64(clf.intercept(&pool));

    let labels_dev = clf
        .predict_labels(&mut pool, &xq_dev, (N_QUERY, N_FEATURES))
        .expect("predict_labels after fit");
    let labels = labels_dev.to_host(&pool);

    (coef, intercept, labels)
}

/// HARD GATE: predict labels match sklearn EXACTLY (integers, no band), f32.
#[test]
fn exact_labels_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("linear_svc_f32_seed42.npz")).expect("load linear_svc_f32");
    let predict_ref: Vec<i32> = case
        .expect_f64("predict")
        .iter()
        .map(|&v| v.round() as i32)
        .collect();
    let (_coef, _intercept, labels) = fit_svc::<f32>(&case);
    assert_eq!(labels, predict_ref, "LinearSVC f32 exact predict labels (HARD gate)");
}

/// HARD GATE: predict labels match sklearn EXACTLY, f64 (cpu runs; rocm skips).
#[test]
fn exact_labels() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("linear_svc f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("linear_svc_f64_seed42.npz")).expect("load linear_svc_f64");
    let predict_ref: Vec<i32> = case
        .expect_f64("predict")
        .iter()
        .map(|&v| v.round() as i32)
        .collect();
    let (_coef, _intercept, labels) = fit_svc::<f64>(&case);
    assert_eq!(labels, predict_ref, "LinearSVC f64 exact predict labels (HARD gate)");
}

/// coef_/intercept_ match sklearn within the documented band, f32.
#[test]
fn oracle_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("linear_svc_f32_seed42.npz")).expect("load linear_svc_f32");
    let coef_ref = case.expect_f64("coef");
    let intercept_ref = case.expect_f64("intercept");
    let (coef, intercept, _labels) = fit_svc::<f32>(&case);
    assert_band(&coef, coef_ref, COEF_BAND_F32, "LinearSVC f32 coef_");
    assert_band(&[intercept], &[intercept_ref[0]], COEF_BAND_F32, "LinearSVC f32 intercept_");
}

/// coef_/intercept_ match sklearn within the documented band, f64 (cpu; rocm skips).
#[test]
fn oracle() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("linear_svc f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("linear_svc_f64_seed42.npz")).expect("load linear_svc_f64");
    let coef_ref = case.expect_f64("coef");
    let intercept_ref = case.expect_f64("intercept");
    assert_eq!(coef_ref.len(), N_FEATURES, "fixture coef length");
    assert_eq!(intercept_ref.len(), 1, "fixture intercept length");
    let (coef, intercept, _labels) = fit_svc::<f64>(&case);
    assert_band(&coef, coef_ref, COEF_BAND_F64, "LinearSVC f64 coef_");
    assert_band(&[intercept], &[intercept_ref[0]], COEF_BAND_F64, "LinearSVC f64 intercept_");
}

/// D-03 litmus: `builder().build()` with NO setters reproduces sklearn's
/// `LinearSVC` defaults (loss=squared_hinge, penalty=l2, C=1.0,
/// intercept_scaling=1.0, max_iter=1000, tol=1e-4). The CD-free SVM has no
/// learning-rate schedule, so the lowered `SgdConfig.learning_rate` is the inert
/// `Constant` placeholder.
#[test]
fn default_matches_sklearn() {
    let clf = LinearSVC::<f64>::builder()
        .build::<f64>()
        .expect("default LinearSVC builds");
    let cfg = clf.config();
    assert_eq!(cfg.loss, Loss::SquaredHinge, "default loss");
    assert_eq!(cfg.penalty, Penalty::L2, "default penalty");
    assert_eq!(clf.c(), 1.0, "default C");
    assert_eq!(clf.intercept_scaling(), 1.0, "default intercept_scaling");
    assert!(cfg.fit_intercept, "default fit_intercept");
    assert_eq!(cfg.max_iter, 1000, "default max_iter");
    assert_eq!(cfg.tol, 1e-4, "default tol");
    // The CD/L-BFGS SVM has no schedule; the lowered field is the inert placeholder.
    assert_eq!(cfg.learning_rate, LearningRate::Constant, "inert schedule");
}

/// `build()` rejects `C <= 0` and a regression loss (`EpsilonInsensitive`) on the
/// classifier builder (T-10-04-01 validate-at-build).
#[test]
fn build_rejects_bad_hyperparams() {
    let bad_c = LinearSVC::<f64>::builder().c(0.0).build::<f64>().err();
    assert!(
        matches!(bad_c, Some(BuildError::InvalidC { .. })),
        "C <= 0 must be BuildError::InvalidC, got {bad_c:?}"
    );
    let bad_loss = LinearSVC::<f64>::builder()
        .loss(Loss::EpsilonInsensitive)
        .build::<f64>()
        .err();
    assert!(
        matches!(bad_loss, Some(BuildError::InvalidLossForEstimator { .. })),
        "a regression loss must be BuildError::InvalidLossForEstimator, got {bad_loss:?}"
    );
}
