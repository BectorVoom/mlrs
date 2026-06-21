//! Plan 10-04 Wave-2 — LinearSVR (SGDSVM-04) sklearn oracle tests.
//!
//! Activated from the Wave-0 `#[ignore]` scaffold. The device estimator fits the
//! L2-regularized SQUARED-EPSILON-INSENSITIVE primal by the SHARED 05-06 L-BFGS
//! path (`svm_lbfgs_fit`, Open Question Q1 RESOLUTION — the same smooth+convex SVM
//! solver as LinearSVC, only the per-sample margin-loss closure differs; NOT the
//! `cd_fit` soft-threshold CD — see the 10-04 SUMMARY), with the intercept handled
//! by the synthetic-feature `intercept_scaling` mechanism (Pitfall 5), and
//! `predict` via the shared `X·coef_ + intercept_` GEMM path.
//!
//!   - `oracle` — `coef_`/`intercept_`/`predict(Xq)` value-match within tolerance
//!     (f64 a tight band — liblinear stops at tol=1e-4 slightly short of the true
//!     optimum, so the converged L-BFGS optimum agrees to ~1e-4; f32 wider).
//!   - `default_matches_sklearn` — `builder().build()` reproduces the sklearn
//!     `LinearSVR` defaults (D-03 litmus).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs f64;
//! rocm skips per the CubeCL-HIP F64 gap, D-07). f32 runs on rocm at a documented
//! band. Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an
//! in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::BuildError;
use mlrs_algos::linear::linear_svr::LinearSVR;
use mlrs_algos::linear::sgd_config::{LearningRate, Loss, Penalty};
use mlrs_algos::traits::{Fit, Predict};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// LinearSVR fixture geometry (gen_oracle.py `SGD_N_SAMPLES` × `SGD_N_FEATURES`,
/// `SGD_N_QUERY` query rows).
const N_SAMPLES: usize = 40;
const N_FEATURES: usize = 4;
const N_QUERY: usize = 8;

/// The pinned fixture hyperparameters (gen_oracle.py: C=1.0, epsilon=0.1,
/// intercept_scaling=1.0, max_iter=1000, tol=1e-4, squared_epsilon_insensitive/l2).
const SVM_C: f64 = 1.0;
const SVR_EPSILON: f64 = 0.1;
const SVM_INTERCEPT_SCALING: f64 = 1.0;
const SVM_MAX_ITER: usize = 1000;

/// f64 band: liblinear (the oracle) stops at its own tol=1e-4 slightly short of the
/// true squared-eps-insensitive optimum, so the converged L-BFGS optimum lands
/// ~1e-4 away (not strict 1e-5). Documented band (RESEARCH Pitfall 6).
const BAND_F64: f64 = 2e-4;
/// f32 band (round-off over the matvec accumulations + the liblinear gap).
const BAND_F32: f64 = 5e-3;

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
        _ => unreachable!("linear_svr fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("linear_svr fixtures are f32/f64 only"),
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

/// Build + fit `LinearSVR` on the fixture with the EXPLICIT pinned hyperparameters
/// and return host `(coef_, intercept_, predict(Xq))`.
fn fit_svr<F>(case: &OracleCase) -> (Vec<f64>, f64, Vec<f64>)
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

    // EXPLICIT pinned setters (Pitfall 7) — squared_epsilon_insensitive/l2, C,
    // epsilon, intercept_scaling, max_iter; NOT the bare default (D-03 litmus
    // checks the default separately).
    let mut reg = LinearSVR::<F>::builder()
        .loss(Loss::SquaredEpsilonInsensitive)
        .penalty(Penalty::L2)
        .c(SVM_C)
        .epsilon(SVR_EPSILON)
        .intercept_scaling(SVM_INTERCEPT_SCALING)
        .fit_intercept(true)
        .max_iter(SVM_MAX_ITER)
        .tol(1e-4)
        .build::<F>()
        .expect("LinearSVR builds with valid hyperparameters");

    reg.fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("LinearSVR::fit on a valid shape");

    let coef: Vec<f64> = reg
        .coef(&pool)
        .expect("coef_ after fit")
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    let intercept = host_to_f64(reg.intercept(&pool).expect("intercept_ after fit"));

    let pred_dev = reg
        .predict(&mut pool, &xq_dev, (N_QUERY, N_FEATURES))
        .expect("predict after fit");
    let pred: Vec<f64> = pred_dev.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();

    (coef, intercept, pred)
}

/// LOAD-NOT-JUST-PRESENT: the `linear_svr` fixture loads with well-formed
/// X/Xq/y/coef/intercept/predict.
#[test]
fn fixture_loads() {
    let case = load_npz(fixture("linear_svr_f64_seed42.npz")).expect("load linear_svr_f64");
    assert_eq!(case.expect_f64("X").len(), N_SAMPLES * N_FEATURES);
    assert_eq!(case.expect_f64("Xq").len(), N_QUERY * N_FEATURES);
    assert_eq!(case.expect_f64("y").len(), N_SAMPLES);
    assert_eq!(case.expect_f64("coef").len(), N_FEATURES);
    assert_eq!(case.expect_f64("intercept").len(), 1);
    assert_eq!(case.expect_f64("predict").len(), N_QUERY);
}

/// coef_/intercept_/predict match sklearn within the documented band, f32.
#[test]
fn oracle_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("linear_svr_f32_seed42.npz")).expect("load linear_svr_f32");
    let coef_ref = case.expect_f64("coef");
    let intercept_ref = case.expect_f64("intercept");
    let predict_ref = case.expect_f64("predict");
    let (coef, intercept, pred) = fit_svr::<f32>(&case);
    assert_band(&coef, coef_ref, BAND_F32, "LinearSVR f32 coef_");
    assert_band(&[intercept], &[intercept_ref[0]], BAND_F32, "LinearSVR f32 intercept_");
    assert_band(&pred, predict_ref, BAND_F32, "LinearSVR f32 predict");
}

/// coef_/intercept_/predict match sklearn within the documented band, f64 (cpu
/// runs; rocm skips).
#[test]
fn oracle() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("linear_svr f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("linear_svr_f64_seed42.npz")).expect("load linear_svr_f64");
    let coef_ref = case.expect_f64("coef");
    let intercept_ref = case.expect_f64("intercept");
    let predict_ref = case.expect_f64("predict");
    assert_eq!(coef_ref.len(), N_FEATURES, "fixture coef length");
    assert_eq!(intercept_ref.len(), 1, "fixture intercept length");
    let (coef, intercept, pred) = fit_svr::<f64>(&case);
    assert_band(&coef, coef_ref, BAND_F64, "LinearSVR f64 coef_");
    assert_band(&[intercept], &[intercept_ref[0]], BAND_F64, "LinearSVR f64 intercept_");
    assert_band(&pred, predict_ref, BAND_F64, "LinearSVR f64 predict");
}

/// D-03 litmus: `builder().build()` with NO setters reproduces sklearn's
/// `LinearSVR` defaults (loss=squared_epsilon_insensitive, penalty=l2, C=1.0,
/// epsilon=0.0, intercept_scaling=1.0, max_iter=1000, tol=1e-4).
#[test]
fn default_matches_sklearn() {
    let reg = LinearSVR::<f64>::builder()
        .build::<f64>()
        .expect("default LinearSVR builds");
    let cfg = reg.config();
    assert_eq!(cfg.loss, Loss::SquaredEpsilonInsensitive, "default loss");
    assert_eq!(cfg.penalty, Penalty::L2, "default penalty");
    assert_eq!(reg.c(), 1.0, "default C");
    assert_eq!(cfg.epsilon, 0.0, "default epsilon");
    assert_eq!(reg.intercept_scaling(), 1.0, "default intercept_scaling");
    assert!(cfg.fit_intercept, "default fit_intercept");
    assert_eq!(cfg.max_iter, 1000, "default max_iter");
    assert_eq!(cfg.tol, 1e-4, "default tol");
    // The L-BFGS SVM has no schedule; the lowered field is the inert placeholder.
    assert_eq!(cfg.learning_rate, LearningRate::Constant, "inert schedule");
}

/// `build()` rejects `C <= 0`, `epsilon < 0`, and a classifier loss (`Hinge`) on
/// the regressor builder (T-10-04-01 validate-at-build).
#[test]
fn build_rejects_bad_hyperparams() {
    let bad_c = LinearSVR::<f64>::builder().c(0.0).build::<f64>().err();
    assert!(
        matches!(bad_c, Some(BuildError::InvalidC { .. })),
        "C <= 0 must be BuildError::InvalidC, got {bad_c:?}"
    );
    let bad_eps = LinearSVR::<f64>::builder().epsilon(-0.1).build::<f64>().err();
    assert!(
        matches!(bad_eps, Some(BuildError::InvalidEpsilon { .. })),
        "epsilon < 0 must be BuildError::InvalidEpsilon, got {bad_eps:?}"
    );
    let bad_loss = LinearSVR::<f64>::builder().loss(Loss::Hinge).build::<f64>().err();
    assert!(
        matches!(bad_loss, Some(BuildError::InvalidLossForEstimator { .. })),
        "a classifier loss must be BuildError::InvalidLossForEstimator, got {bad_loss:?}"
    );
}
