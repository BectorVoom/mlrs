//! Plan 10-03 Wave-2 — MBSGDClassifier (SGDSVM-01) sklearn oracle tests.
//!
//! Activated from the Wave-0 `#[ignore]` scaffold. The device estimator lowers
//! its validated `SgdConfig` into the prim-local `SgdParams` and wires the
//! validated PRIM-10 `sgd_solve` (10-02). The fixtures pin
//! `shuffle=False, tol=0, max_iter=SGD_MAX_ITER` with an explicit schedule so the
//! Rust solver reproduces the EXACT iterate (Pitfall 2/7):
//!
//!   - `exact_labels` — `predict_labels(Xq)` match sklearn EXACTLY (the HARD
//!     gate, integers, no band — Pitfall 4 ±1 encoding).
//!   - `oracle` — `coef_`/`intercept_` value-match within the documented band
//!     (constant-schedule hinge).
//!   - `proba` — `predict_proba(Xq)` value-match within tolerance (the log-loss
//!     variant; sigmoid `1/(1+exp(-margin))`).
//!   - `default_matches_sklearn` — `builder().build()` with no setters reproduces
//!     the sklearn `SGDClassifier` defaults (D-03 litmus).
//!   - `build_rejects_bad_alpha` — `build()` rejects `alpha < 0` (D-08
//!     validate-at-build) and a regression loss.
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips per the CubeCL-HIP F64 gap, D-07). f32 runs at a documented
//! band. Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an
//! in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::BuildError;
use mlrs_algos::linear::mbsgd_classifier::MBSGDClassifier;
use mlrs_algos::linear::sgd_config::{LearningRate, Loss, Penalty};
use mlrs_algos::traits::{Fit, PredictLabels, PredictProba};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// MBSGDClassifier fixture geometry (gen_oracle.py `SGD_N_SAMPLES` ×
/// `SGD_N_FEATURES`, `SGD_N_QUERY` query rows).
const N_SAMPLES: usize = 40;
const N_FEATURES: usize = 4;
const N_QUERY: usize = 8;

/// The pinned fixture hyperparameters (gen_oracle.py `SGD_ALPHA` / `SGD_ETA0` /
/// `SGD_MAX_ITER`): alpha=1e-4, eta0=0.01, max_iter=50, tol=0, shuffle=False,
/// penalty=l2, fit_intercept=True. The default-file fixtures use the CONSTANT
/// schedule; the `_optimal` infix carries the optimal-schedule variant.
const SGD_ALPHA: f64 = 1e-4;
const SGD_ETA0: f64 = 0.01;
const SGD_MAX_ITER: usize = 50;

/// f64 coef/intercept band. The host-driven minibatch SGD drives the same epoch
/// schedule sklearn runs, but the order-of-operations (host f64 dloss/penalty vs
/// sklearn's Cython `_sgd_fast`) differs in the last-bit accumulation, so the
/// converged iterate agrees to a documented band rather than strict 1e-5. The
/// EXACT predict labels (the hard gate) are the correctness witness.
const COEF_BAND_F64: f64 = 5e-3;
/// f32 coef/intercept band (round-off over the per-batch matvec accumulations).
const COEF_BAND_F32: f64 = 2e-2;
/// predict_proba band (the log-loss sigmoid over the decision margin).
const PROBA_BAND_F64: f64 = 1e-2;
const PROBA_BAND_F32: f64 = 3e-2;

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
        _ => unreachable!("mbsgd_classifier fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("mbsgd_classifier fixtures are f32/f64 only"),
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

/// Build + fit a hinge `MBSGDClassifier` on the fixture with the EXPLICIT pinned
/// (constant-schedule) hyperparameters and return host
/// `(coef_, intercept_, predict_labels(Xq))`.
fn fit_hinge<F>(case: &OracleCase) -> (Vec<f64>, f64, Vec<i32>)
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

    // EXPLICIT pinned setters (Pitfall 7) — hinge/l2, constant schedule, eta0,
    // alpha, max_iter, tol=0, shuffle=false, batch_size=1 (sklearn SGD default).
    let mut clf = MBSGDClassifier::<F>::builder()
        .loss(Loss::Hinge)
        .penalty(Penalty::L2)
        .alpha(SGD_ALPHA)
        .learning_rate(LearningRate::Constant)
        .eta0(SGD_ETA0)
        .max_iter(SGD_MAX_ITER)
        .tol(0.0)
        .shuffle(false)
        .batch_size(1)
        .fit_intercept(true)
        .build::<F>()
        .expect("MBSGDClassifier builds with valid hyperparameters");

    clf.fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("MBSGDClassifier::fit on a valid shape");

    let coef: Vec<f64> = clf
        .coef(&pool)
        .expect("coef_ after fit")
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    let intercept = host_to_f64(clf.intercept(&pool).expect("intercept_ after fit"));

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
    let case = load_npz(fixture("mbsgd_classifier_f32_seed42.npz"))
        .expect("load mbsgd_classifier_f32");
    let predict_ref: Vec<i32> = case
        .expect_f64("predict")
        .iter()
        .map(|&v| v.round() as i32)
        .collect();
    let (_coef, _intercept, labels) = fit_hinge::<f32>(&case);
    assert_eq!(
        labels, predict_ref,
        "MBSGDClassifier f32 exact predict labels (HARD gate)"
    );
}

/// HARD GATE: predict labels match sklearn EXACTLY, f64 (cpu runs; rocm skips).
#[test]
fn exact_labels() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("mbsgd_classifier f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("mbsgd_classifier_f64_seed42.npz"))
        .expect("load mbsgd_classifier_f64");
    let predict_ref: Vec<i32> = case
        .expect_f64("predict")
        .iter()
        .map(|&v| v.round() as i32)
        .collect();
    let (_coef, _intercept, labels) = fit_hinge::<f64>(&case);
    assert_eq!(
        labels, predict_ref,
        "MBSGDClassifier f64 exact predict labels (HARD gate)"
    );
}

/// coef_/intercept_ match sklearn within the documented band, f32.
#[test]
fn oracle_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("mbsgd_classifier_f32_seed42.npz"))
        .expect("load mbsgd_classifier_f32");
    let coef_ref = case.expect_f64("coef");
    let intercept_ref = case.expect_f64("intercept");
    let (coef, intercept, _labels) = fit_hinge::<f32>(&case);
    assert_band(&coef, coef_ref, COEF_BAND_F32, "MBSGDClassifier f32 coef_");
    assert_band(
        &[intercept],
        &[intercept_ref[0]],
        COEF_BAND_F32,
        "MBSGDClassifier f32 intercept_",
    );
}

/// coef_/intercept_ match sklearn within the documented band, f64 (cpu; rocm skips).
#[test]
fn oracle() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("mbsgd_classifier f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("mbsgd_classifier_f64_seed42.npz"))
        .expect("load mbsgd_classifier_f64");
    let coef_ref = case.expect_f64("coef");
    let intercept_ref = case.expect_f64("intercept");
    assert_eq!(coef_ref.len(), N_FEATURES, "fixture coef length");
    assert_eq!(intercept_ref.len(), 1, "fixture intercept length");
    let (coef, intercept, _labels) = fit_hinge::<f64>(&case);
    assert_band(&coef, coef_ref, COEF_BAND_F64, "MBSGDClassifier f64 coef_");
    assert_band(
        &[intercept],
        &[intercept_ref[0]],
        COEF_BAND_F64,
        "MBSGDClassifier f64 intercept_",
    );
}

/// Fit the log-loss variant and return host `predict_proba(Xq)` (n_query × 2).
fn fit_log_proba<F>(case: &OracleCase) -> Vec<f64>
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

    let mut clf = MBSGDClassifier::<F>::builder()
        .loss(Loss::Log)
        .penalty(Penalty::L2)
        .alpha(SGD_ALPHA)
        .learning_rate(LearningRate::Constant)
        .eta0(SGD_ETA0)
        .max_iter(SGD_MAX_ITER)
        .tol(0.0)
        .shuffle(false)
        .batch_size(1)
        .fit_intercept(true)
        .build::<F>()
        .expect("MBSGDClassifier(log) builds");

    clf.fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("MBSGDClassifier(log)::fit");

    let proba_dev = clf
        .predict_proba(&mut pool, &xq_dev, (N_QUERY, N_FEATURES))
        .expect("predict_proba after fit");
    proba_dev.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect()
}

/// predict_proba matches the log-loss fixture within tolerance, f32.
#[test]
fn proba_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "log");
    let case = load_npz(fixture("mbsgd_classifier_log_f32_seed42.npz"))
        .expect("load mbsgd_classifier_log_f32");
    let proba_ref = case.expect_f64("predict_proba");
    assert_eq!(proba_ref.len(), N_QUERY * 2, "fixture predict_proba shape");
    let proba = fit_log_proba::<f32>(&case);
    assert_band(&proba, proba_ref, PROBA_BAND_F32, "MBSGDClassifier f32 predict_proba");
}

/// predict_proba matches the log-loss fixture within tolerance, f64 (cpu; rocm skips).
#[test]
fn proba() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "log");
    if capability::skip_f64_with_log() {
        println!("mbsgd_classifier(log) f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("mbsgd_classifier_log_f64_seed42.npz"))
        .expect("load mbsgd_classifier_log_f64");
    let proba_ref = case.expect_f64("predict_proba");
    assert_eq!(proba_ref.len(), N_QUERY * 2, "fixture predict_proba shape");
    let proba = fit_log_proba::<f64>(&case);
    assert_band(&proba, proba_ref, PROBA_BAND_F64, "MBSGDClassifier f64 predict_proba");
}

/// D-03 litmus: `builder().build()` with NO setters reproduces sklearn's
/// `SGDClassifier` defaults (loss=hinge, penalty=l2, alpha=1e-4, lr=optimal,
/// max_iter=1000, tol=1e-3, eta0=0.01, l1_ratio=0.15).
#[test]
fn default_matches_sklearn() {
    let clf = MBSGDClassifier::<f64>::builder()
        .build::<f64>()
        .expect("default MBSGDClassifier builds");
    let cfg = clf.config();
    assert_eq!(cfg.loss, Loss::Hinge, "default loss");
    assert_eq!(cfg.penalty, Penalty::L2, "default penalty");
    assert_eq!(cfg.alpha, 1e-4, "default alpha");
    assert_eq!(cfg.learning_rate, LearningRate::Optimal, "default schedule");
    assert_eq!(cfg.max_iter, 1000, "default max_iter");
    assert_eq!(cfg.tol, 1e-3, "default tol");
    assert_eq!(cfg.eta0, 0.01, "default eta0");
    assert_eq!(cfg.power_t, 0.5, "default power_t");
    assert_eq!(cfg.l1_ratio, 0.15, "default l1_ratio");
    assert!(cfg.fit_intercept, "default fit_intercept");
}

/// `build()` rejects `alpha < 0` (D-08 validate-at-build) and a regression loss
/// on the classifier builder; `TryFrom("bogus")` is `UnknownLoss` (D-05).
#[test]
fn build_rejects_bad_alpha() {
    let bad_alpha = MBSGDClassifier::<f64>::builder()
        .alpha(-1.0)
        .build::<f64>()
        .err();
    assert!(
        matches!(bad_alpha, Some(BuildError::InvalidAlpha { alpha, .. }) if alpha == -1.0),
        "alpha < 0 must be BuildError::InvalidAlpha, got {bad_alpha:?}"
    );
    let bad_loss = MBSGDClassifier::<f64>::builder()
        .loss(Loss::EpsilonInsensitive)
        .build::<f64>()
        .err();
    assert!(
        matches!(bad_loss, Some(BuildError::InvalidLossForEstimator { .. })),
        "a regression loss must be BuildError::InvalidLossForEstimator, got {bad_loss:?}"
    );
    let bad_str = Loss::try_from("bogus").err();
    assert!(
        matches!(bad_str, Some(BuildError::UnknownLoss { ref value }) if value == "bogus"),
        "an unknown loss string must be BuildError::UnknownLoss, got {bad_str:?}"
    );
}
