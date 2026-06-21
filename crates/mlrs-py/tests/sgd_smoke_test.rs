//! Plan 10-05 (Wave-3) — SGD / linear-SVM construct + fit/predict smoke
//! (SGDSVM-01..04, PY-06 incremental share).
//!
//! Rust **integration test** (separate crate linking the `mlrs-py` rlib,
//! AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`). It has three
//! parts, mirroring the 08-05 kernel / 09-04 spectral precedent:
//!
//!   - `sgd_estimators_construct_unfit` — builds all four wrappers via the
//!     Rust-callable `unfit_default()` and asserts they land in the `Unfit` arm,
//!     proving the four `#[pyclass]` definitions + the `any_estimator!` enums
//!     COMPILE and INSTANTIATE without a Python interpreter or live device.
//!   - `sgd_fit_predict_smoke` — the f32 + f64 device `fit` → `predict`
//!     (classifier `predict_proba`) smoke. The PyO3 `fit` method is a thin
//!     `py.detach` shell over the same `mlrs_algos` estimators the wrappers
//!     delegate to (`MBSGDClassifier<F>` / `MBSGDRegressor<F>` / `LinearSVC<F>`
//!     / `LinearSVR<F>`), built through the SAME builder chain the wrapper uses
//!     (`Estimator::<F>::builder()...build()` after `Loss/Penalty/LearningRate
//!     ::try_from`), so this drives the wrapper's fit body end to end on a live
//!     device (no Python interpreter at the Rust test level — the full
//!     interpreter+capsule path runs in the pytest harness `test_sgd.py`, the
//!     08-05 kernel precedent). f64 is gated by `skip_f64_with_log()` (skip on a
//!     backend without f64, e.g. rocm; run on cpu).
//!   - `bad_enum_string_maps_to_value_error` — the construction-time D-05/D-09
//!     witness: a bogus `loss='bogus'` string is a `BuildError::UnknownLoss` at
//!     the typed layer, and `build_err_to_py` maps it to a `PyValueError`. The
//!     CONCRETE `ValueError` class is asserted end-to-end (with a live
//!     interpreter) in `test_sgd.py::test_*_bad_enum_raises_value_error`; the
//!     Rust integration binary cannot link `PyErr::is_instance_of` (libpython is
//!     undefined at link, ingress_test.rs §typed-layer note), so here we pin the
//!     typed source of that mapping.

use mlrs_algos::error::BuildError;
use mlrs_algos::linear::linear_svc::LinearSVC;
use mlrs_algos::linear::linear_svr::LinearSVR;
use mlrs_algos::linear::mbsgd_classifier::MBSGDClassifier;
use mlrs_algos::linear::mbsgd_regressor::MBSGDRegressor;
use mlrs_algos::linear::sgd_config::{LearningRate, Loss, Penalty};
use mlrs_algos::traits::{Fit, Predict, PredictLabels, PredictProba};

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

use mlrs_py::estimators::linear::{
    PyLinearSVC, PyLinearSVR, PyMBSGDClassifier, PyMBSGDRegressor,
};

/// A tiny well-separated binary problem (two clusters at ∓2 on feature 0) so the
/// classifier/SVC land a clean ±1 split and the regressor/SVR track a simple
/// linear target. Geometry only — the value gates live in the algos oracle tests
/// (10-03 / 10-04); this smoke proves the fit→predict surface runs end to end.
const N: usize = 8;
const D: usize = 2;

/// The shared `N × D` design rows (row-major).
fn x_rows() -> [[f64; D]; N] {
    [
        [-2.0, 0.1],
        [-1.9, -0.2],
        [-2.1, 0.0],
        [-1.8, 0.2],
        [2.0, -0.1],
        [1.9, 0.2],
        [2.1, 0.0],
        [1.8, -0.2],
    ]
}

/// Row-major `N × D` design as a flat host `Vec<F>`.
fn x_host<F>() -> Vec<F>
where
    F: bytemuck::Pod + cubecl::prelude::Float,
{
    let mut v = Vec::with_capacity(N * D);
    for r in &x_rows() {
        for &c in r {
            v.push(host::<F>(c));
        }
    }
    v
}

/// Binary ±1 labels matching the two clusters (0 for the −2 cluster, 1 for +2).
fn y_labels<F>() -> Vec<F>
where
    F: bytemuck::Pod + cubecl::prelude::Float,
{
    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        .iter()
        .map(|&c| host::<F>(c))
        .collect()
}

/// A smooth linear regression target `y = 3·x0 − x1` over the same design.
fn y_regression<F>() -> Vec<F>
where
    F: bytemuck::Pod + cubecl::prelude::Float,
{
    x_rows()
        .iter()
        .map(|r| host::<F>(3.0 * r[0] - r[1]))
        .collect()
}

/// Reinterpret an `f64` host scalar as `F` (`f32`/`f64` only).
fn host<F>(c: f64) -> F
where
    F: bytemuck::Pod + cubecl::prelude::Float,
{
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(c as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&c)),
        _ => unreachable!("smoke design is f32/f64 only"),
    }
}

/// `F` → `f64` for finiteness/value smoke checks.
fn to_f64<F>(v: &F) -> f64
where
    F: bytemuck::Pod,
{
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(v)),
        _ => unreachable!(),
    }
}

/// All four wrappers construct with default hyperparameters and start `Unfit`
/// (no Python interpreter / live device needed).
#[test]
fn sgd_estimators_construct_unfit() {
    assert!(PyMBSGDClassifier::unfit_default().is_unfit(), "MBSGDClassifier");
    assert!(PyMBSGDRegressor::unfit_default().is_unfit(), "MBSGDRegressor");
    assert!(PyLinearSVC::unfit_default().is_unfit(), "LinearSVC");
    assert!(PyLinearSVR::unfit_default().is_unfit(), "LinearSVR");
}

/// Device fit→predict smoke for all four estimators (the algos bodies the PyO3
/// `fit` shells delegate to, built through the identical builder chain), f32
/// always + f64 when `skip_f64_with_log()` is false.
#[test]
fn sgd_fit_predict_smoke() {
    let _ = env_logger::builder().is_test(true).try_init();

    // The wrapper construction surface is proven by the test above.
    assert!(PyMBSGDClassifier::unfit_default().is_unfit());

    // f32 always.
    classifier_smoke::<f32>();
    regressor_smoke::<f32>();
    svc_smoke::<f32>();
    svr_smoke::<f32>();

    // f64 gated by backend capability (skip on rocm, run on cpu).
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "smoke");
    if capability::skip_f64_with_log() {
        println!("sgd_fit_predict_smoke f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    classifier_smoke::<f64>();
    regressor_smoke::<f64>();
    svc_smoke::<f64>();
    svr_smoke::<f64>();
}

/// MBSGDClassifier (log loss) fit → predict_labels + predict_proba: clean ±1
/// split + per-row proba in [0,1] summing to 1.
fn classifier_smoke<F>()
where
    F: bytemuck::Pod + cubecl::prelude::Float + cubecl::prelude::CubeElement,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let xd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host::<F>());
    let yd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_labels::<F>());

    // Builder chain identical to the PyMBSGDClassifier::fit body (log loss so
    // predict_proba is exercised, constant schedule for a deterministic smoke).
    let mut est = MBSGDClassifier::<F>::builder()
        .loss(Loss::try_from("log_loss").unwrap())
        .penalty(Penalty::try_from("l2").unwrap())
        .alpha(1e-4)
        .learning_rate(LearningRate::try_from("constant").unwrap())
        .eta0(0.1)
        .max_iter(50)
        .seed(0)
        .build::<F>()
        .expect("MBSGDClassifier builder");
    est.fit(&mut pool, &xd, Some(&yd), (N, D)).expect("classifier fit");

    let labels = est
        .predict_labels(&mut pool, &xd, (N, D))
        .expect("predict_labels")
        .to_host(&mut pool);
    assert_eq!(labels.len(), N, "labels length N");
    assert!(labels.iter().all(|&l| l == 0 || l == 1), "labels are 0/1");
    // Clean split: the two clusters land in different classes.
    assert_eq!(labels[0], labels[1], "first cluster shares a class");
    assert_ne!(labels[0], labels[4], "the two clusters split");

    let proba = est
        .predict_proba(&mut pool, &xd, (N, D))
        .expect("predict_proba")
        .to_host(&mut pool);
    assert_eq!(proba.len(), N * 2, "proba is N×2");
    for r in 0..N {
        let p0 = to_f64(&proba[r * 2]);
        let p1 = to_f64(&proba[r * 2 + 1]);
        assert!((0.0..=1.0).contains(&p0) && (0.0..=1.0).contains(&p1), "proba in [0,1]");
        assert!((p0 + p1 - 1.0).abs() < 1e-4, "proba sums to 1");
    }
}

/// MBSGDRegressor fit → predict: finite predictions tracking the linear target's
/// sign separation across the two clusters.
fn regressor_smoke<F>()
where
    F: bytemuck::Pod + cubecl::prelude::Float + cubecl::prelude::CubeElement,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let xd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host::<F>());
    let yd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_regression::<F>());

    let mut est = MBSGDRegressor::<F>::builder()
        .loss(Loss::try_from("squared_error").unwrap())
        .penalty(Penalty::try_from("l2").unwrap())
        .alpha(1e-4)
        .learning_rate(LearningRate::try_from("constant").unwrap())
        .eta0(0.01)
        .max_iter(100)
        .seed(0)
        .build::<F>()
        .expect("MBSGDRegressor builder");
    est.fit(&mut pool, &xd, Some(&yd), (N, D)).expect("regressor fit");

    let pred = est.predict(&mut pool, &xd, (N, D)).expect("predict").to_host(&mut pool);
    assert_eq!(pred.len(), N, "predict length N");
    assert!(pred.iter().all(|v| to_f64(v).is_finite()), "predictions finite");
    // The −2 cluster (target ≈ −6) predicts below the +2 cluster (target ≈ +6).
    let lo = to_f64(&pred[0]);
    let hi = to_f64(&pred[4]);
    assert!(lo < hi, "regressor separates the clusters (lo={lo} < hi={hi})");
}

/// LinearSVC fit → predict_labels: clean ±1 split (no learning_rate string).
fn svc_smoke<F>()
where
    F: bytemuck::Pod + cubecl::prelude::Float + cubecl::prelude::CubeElement,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let xd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host::<F>());
    let yd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_labels::<F>());

    let mut est = LinearSVC::<F>::builder()
        .loss(Loss::try_from("squared_hinge").unwrap())
        .penalty(Penalty::try_from("l2").unwrap())
        .c(1.0)
        .build::<F>()
        .expect("LinearSVC builder");
    est.fit(&mut pool, &xd, Some(&yd), (N, D)).expect("svc fit");

    let labels = est
        .predict_labels(&mut pool, &xd, (N, D))
        .expect("predict_labels")
        .to_host(&mut pool);
    assert_eq!(labels.len(), N, "labels length N");
    assert!(labels.iter().all(|&l| l == 0 || l == 1), "labels are 0/1");
    assert_eq!(labels[0], labels[1], "first cluster shares a class");
    assert_ne!(labels[0], labels[4], "the two clusters split");
}

/// LinearSVR fit → predict: finite predictions separating the clusters.
fn svr_smoke<F>()
where
    F: bytemuck::Pod + cubecl::prelude::Float + cubecl::prelude::CubeElement,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let xd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host::<F>());
    let yd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_regression::<F>());

    let mut est = LinearSVR::<F>::builder()
        .loss(Loss::try_from("squared_epsilon_insensitive").unwrap())
        .penalty(Penalty::try_from("l2").unwrap())
        .c(1.0)
        .epsilon(0.0)
        .build::<F>()
        .expect("LinearSVR builder");
    est.fit(&mut pool, &xd, Some(&yd), (N, D)).expect("svr fit");

    let pred = est.predict(&mut pool, &xd, (N, D)).expect("predict").to_host(&mut pool);
    assert_eq!(pred.len(), N, "predict length N");
    assert!(pred.iter().all(|v| to_f64(v).is_finite()), "predictions finite");
    let lo = to_f64(&pred[0]);
    let hi = to_f64(&pred[4]);
    assert!(lo < hi, "svr separates the clusters (lo={lo} < hi={hi})");
}

/// D-05/D-09 construction-time witness: a bogus enum string is a typed
/// `BuildError::UnknownLoss` (the source `build_err_to_py` maps to a
/// `PyValueError`). The CONCRETE `ValueError` class — through the live PyO3 `fit`
/// boundary — is asserted in `test_sgd.py` (which links a real interpreter);
/// here we pin the typed source the wrapper's `Loss::try_from(...).map_err(
/// build_err_to_py)?` relies on (the ingress_test.rs typed-layer precedent).
#[test]
fn bad_enum_string_maps_to_value_error() {
    assert!(
        matches!(
            Loss::try_from("bogus"),
            Err(BuildError::UnknownLoss { .. })
        ),
        "an unknown loss string is BuildError::UnknownLoss (→ PyValueError via build_err_to_py)"
    );
    // Penalty + learning_rate strings funnel through the same typed path.
    assert!(Penalty::try_from("bogus").is_err(), "unknown penalty is a BuildError");
    assert!(LearningRate::try_from("bogus").is_err(), "unknown learning_rate is a BuildError");
}
