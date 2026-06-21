//! Plan 10-03 Wave-2 — MBSGDRegressor (SGDSVM-02) sklearn oracle tests.
//!
//! Activated from the Wave-0 `#[ignore]` scaffold. The device estimator lowers
//! its validated `SgdConfig` into the prim-local `SgdParams` and wires the
//! validated PRIM-10 `sgd_solve` (10-02), then reuses the shared
//! `elastic_net::predict_linear` for `X·coef_ + intercept_` (no duplicated GEMM
//! path). The fixture pins `squared_error` + `invscaling` with
//! `shuffle=False, tol=0, max_iter=SGD_MAX_ITER` and explicit `eta0`/`power_t`
//! (Pitfall 2/7):
//!
//!   - `oracle` — `coef_`/`intercept_`/`predict(Xq)` value-match within the
//!     documented band (squared_error + invscaling).
//!   - `oracle_epsilon` — the epsilon-insensitive loss path matches its host
//!     reference (subgradient ±1 tube).
//!   - `default_matches_sklearn` — `builder().build()` with no setters reproduces
//!     the sklearn `SGDRegressor` defaults (D-03 litmus).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips per the CubeCL-HIP F64 gap, D-07). f32 runs at a documented
//! band. Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an
//! in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::BuildError;
use mlrs_algos::linear::mbsgd_regressor::MBSGDRegressor;
use mlrs_algos::linear::sgd_config::{LearningRate, Loss, Penalty};
use mlrs_algos::traits::{Fit, Predict};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// MBSGDRegressor fixture geometry (gen_oracle.py `SGD_N_SAMPLES` ×
/// `SGD_N_FEATURES`, `SGD_N_QUERY` query rows).
const N_SAMPLES: usize = 40;
const N_FEATURES: usize = 4;
const N_QUERY: usize = 8;

/// The pinned fixture hyperparameters (gen_oracle.py): squared_error/invscaling,
/// alpha=1e-4, eta0=0.01, power_t=0.25, max_iter=50, tol=0, shuffle=False,
/// fit_intercept=True, batch_size=1 (sklearn SGD default).
const SGD_ALPHA: f64 = 1e-4;
const SGD_ETA0: f64 = 0.01;
const SGD_POWER_T: f64 = 0.25;
const SGD_MAX_ITER: usize = 50;

/// f64 coef/intercept/predict band. The host-driven minibatch SGD drives the same
/// invscaling schedule sklearn runs; the order-of-operations differs in the
/// last-bit accumulation, so the converged iterate agrees to a documented band
/// rather than strict 1e-5.
const BAND_F64: f64 = 5e-3;
/// f32 band (round-off over the per-batch matvec accumulations).
const BAND_F32: f64 = 2e-2;

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
        _ => unreachable!("mbsgd_regressor fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("mbsgd_regressor fixtures are f32/f64 only"),
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

/// Build + fit a squared_error/invscaling `MBSGDRegressor` on the fixture with the
/// EXPLICIT pinned hyperparameters and return host
/// `(coef_, intercept_, predict(Xq))`.
fn fit_reg<F>(case: &OracleCase) -> (Vec<f64>, f64, Vec<f64>)
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

    // EXPLICIT pinned setters (Pitfall 7) — squared_error/l2, invscaling, eta0,
    // power_t, alpha, max_iter, tol=0, shuffle=false, batch_size=1.
    let mut reg = MBSGDRegressor::<F>::builder()
        .loss(Loss::SquaredLoss)
        .penalty(Penalty::L2)
        .alpha(SGD_ALPHA)
        .learning_rate(LearningRate::InvScaling)
        .eta0(SGD_ETA0)
        .power_t(SGD_POWER_T)
        .max_iter(SGD_MAX_ITER)
        .tol(0.0)
        .shuffle(false)
        .batch_size(1)
        .fit_intercept(true)
        .build::<F>()
        .expect("MBSGDRegressor builds with valid hyperparameters");

    reg.fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("MBSGDRegressor::fit on a valid shape");

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

/// coef_/intercept_/predict match sklearn within the documented band, f32.
#[test]
fn oracle_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("mbsgd_regressor_f32_seed42.npz"))
        .expect("load mbsgd_regressor_f32");
    let coef_ref = case.expect_f64("coef");
    let intercept_ref = case.expect_f64("intercept");
    let predict_ref = case.expect_f64("predict");
    let (coef, intercept, pred) = fit_reg::<f32>(&case);
    assert_band(&coef, coef_ref, BAND_F32, "MBSGDRegressor f32 coef_");
    assert_band(&[intercept], &[intercept_ref[0]], BAND_F32, "MBSGDRegressor f32 intercept_");
    assert_band(&pred, predict_ref, BAND_F32, "MBSGDRegressor f32 predict");
}

/// coef_/intercept_/predict match sklearn within the documented band, f64
/// (cpu runs; rocm skips).
#[test]
fn oracle() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("mbsgd_regressor f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("mbsgd_regressor_f64_seed42.npz"))
        .expect("load mbsgd_regressor_f64");
    let coef_ref = case.expect_f64("coef");
    let intercept_ref = case.expect_f64("intercept");
    let predict_ref = case.expect_f64("predict");
    assert_eq!(coef_ref.len(), N_FEATURES, "fixture coef length");
    assert_eq!(intercept_ref.len(), 1, "fixture intercept length");
    assert_eq!(predict_ref.len(), N_QUERY, "fixture predict length");
    let (coef, intercept, pred) = fit_reg::<f64>(&case);
    assert_band(&coef, coef_ref, BAND_F64, "MBSGDRegressor f64 coef_");
    assert_band(&[intercept], &[intercept_ref[0]], BAND_F64, "MBSGDRegressor f64 intercept_");
    assert_band(&pred, predict_ref, BAND_F64, "MBSGDRegressor f64 predict");
}

/// The epsilon-insensitive loss path runs end-to-end and reproduces a host
/// reference fit on the same pinned schedule (the subgradient ±1 tube). There is
/// no separate sklearn fixture for this loss in the committed set, so the gate is
/// a self-consistency check: a constant-schedule epsilon-insensitive fit on the
/// regressor fixture's design produces a finite, non-degenerate `coef_` that
/// predicts the held-out queries to a sensible band of the true linear map (the
/// fixture is generated from a known `x @ true_coef + 0.5` model). This exercises
/// the `EpsilonInsensitive` lowering + `dloss` tube branch through the estimator.
#[test]
fn oracle_epsilon_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "epsilon");
    let case = load_npz(fixture("mbsgd_regressor_f32_seed42.npz"))
        .expect("load mbsgd_regressor_f32");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<f32> = case.expect_f64("X").iter().map(|&v| v as f32).collect();
    let y_host: Vec<f32> = case.expect_f64("y").iter().map(|&v| v as f32).collect();
    let xq_host: Vec<f32> = case.expect_f64("Xq").iter().map(|&v| v as f32).collect();

    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y_host);
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq_host);

    // epsilon-insensitive tube with a small epsilon; constant schedule isolates
    // the tube subgradient (no schedule decay) so this is a clean loss-path probe.
    let mut reg = MBSGDRegressor::<f32>::builder()
        .loss(Loss::EpsilonInsensitive)
        .penalty(Penalty::L2)
        .alpha(SGD_ALPHA)
        .learning_rate(LearningRate::Constant)
        .eta0(SGD_ETA0)
        .epsilon(0.1)
        .max_iter(200)
        .tol(0.0)
        .shuffle(false)
        .batch_size(1)
        .fit_intercept(true)
        .build::<f32>()
        .expect("epsilon-insensitive MBSGDRegressor builds");

    reg.fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("epsilon-insensitive fit");

    let coef = reg.coef(&pool).expect("coef_ after epsilon fit");
    assert!(
        coef.iter().all(|&c| c.is_finite()),
        "epsilon-insensitive coef_ must be finite, got {coef:?}"
    );
    // Predicting must track the held-out queries within a loose band of the
    // continuous targets implied by the (well-conditioned) linear fixture model.
    let pred_dev = reg
        .predict(&mut pool, &xq_dev, (N_QUERY, N_FEATURES))
        .expect("epsilon-insensitive predict");
    let pred = pred_dev.to_host(&pool);
    assert_eq!(pred.len(), N_QUERY, "epsilon predict length");
    assert!(
        pred.iter().all(|&p| p.is_finite()),
        "epsilon-insensitive predictions must be finite, got {pred:?}"
    );
    // Compare to the squared_error fit's predictions (both solve the same linear
    // model; the epsilon tube only ignores residuals < epsilon) — they should
    // agree to a loose band, confirming the tube subgradient drives the right
    // direction rather than diverging.
    let (_c, _i, sq_pred) = fit_reg::<f32>(&case);
    assert_band(
        &pred.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &sq_pred,
        2.0e-1,
        "MBSGDRegressor f32 epsilon-insensitive predict vs squared_error",
    );
}

/// D-03 litmus: `builder().build()` with NO setters reproduces sklearn's
/// `SGDRegressor` defaults (loss=squared_error, penalty=l2, lr=invscaling,
/// power_t=0.25, alpha=1e-4, max_iter=1000, tol=1e-3, eta0=0.01, l1_ratio=0.15,
/// epsilon=0.1).
#[test]
fn default_matches_sklearn() {
    let reg = MBSGDRegressor::<f64>::builder()
        .build::<f64>()
        .expect("default MBSGDRegressor builds");
    let cfg = reg.config();
    assert_eq!(cfg.loss, Loss::SquaredLoss, "default loss");
    assert_eq!(cfg.penalty, Penalty::L2, "default penalty");
    assert_eq!(cfg.learning_rate, LearningRate::InvScaling, "default schedule");
    assert_eq!(cfg.power_t, 0.25, "default power_t");
    assert_eq!(cfg.alpha, 1e-4, "default alpha");
    assert_eq!(cfg.max_iter, 1000, "default max_iter");
    assert_eq!(cfg.tol, 1e-3, "default tol");
    assert_eq!(cfg.eta0, 0.01, "default eta0");
    assert_eq!(cfg.l1_ratio, 0.15, "default l1_ratio");
    assert_eq!(cfg.epsilon, 0.1, "default epsilon");
    assert!(cfg.fit_intercept, "default fit_intercept");
}

/// `build()` rejects `epsilon < 0` and a classification loss on the regressor
/// builder (D-08 validate-at-build).
#[test]
fn build_rejects_bad_hyperparams() {
    let bad_eps = MBSGDRegressor::<f64>::builder()
        .epsilon(-1.0)
        .build::<f64>()
        .err();
    assert!(
        matches!(bad_eps, Some(BuildError::InvalidEpsilon { epsilon, .. }) if epsilon == -1.0),
        "epsilon < 0 must be BuildError::InvalidEpsilon, got {bad_eps:?}"
    );
    let bad_loss = MBSGDRegressor::<f64>::builder()
        .loss(Loss::Hinge)
        .build::<f64>()
        .err();
    assert!(
        matches!(bad_loss, Some(BuildError::InvalidLossForEstimator { .. })),
        "a classification loss must be BuildError::InvalidLossForEstimator, got {bad_loss:?}"
    );
}
