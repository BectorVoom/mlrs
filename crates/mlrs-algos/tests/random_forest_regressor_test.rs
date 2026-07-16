//! RandomForestRegressor (ENSEMBLE-01) sklearn oracle tests.
//!
//! Two tiers over the committed `rf_reg_{f32,f64}_seed42.npz` fixtures
//! (grid-valued features, piecewise-constant target — `gen_oracle.py`):
//!
//! - DETERMINISTIC tier (`bootstrap=false`, all features, depth 12): the
//!   generator ASSERTS sklearn reaches zero-variance leaves on the train set
//!   (train predictions == y), so the mlrs forest's train predictions must
//!   match `y` (== sklearn's) within 1e-5.
//! - STATISTICAL tier (bootstrap, 64 trees): held-out R² within 0.05 of the
//!   stored sklearn-defaults R².
//!
//! f64 functions carry the `skip_f64_with_log` capability gate. Per AGENTS.md
//! §2 tests live here, never in-source.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::ensemble::random_forest_regressor::RandomForestRegressor;
use mlrs_algos::ensemble::MaxFeatures;
use mlrs_algos::typestate::{Fit, Predict};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// RF fixture geometry (gen_oracle.py).
const RF_N_TRAIN: usize = 96;
const RF_N_TEST: usize = 48;
const RF_N_FEATURES: usize = 5;
const RF_DET_MAX_DEPTH: usize = 12;
const RF_STAT_N_ESTIMATORS: usize = 64;
const RF_STAT_MAX_DEPTH: usize = 8;

const PRED_TOL: f64 = 1e-5;
/// Statistical-tier margin (RNG streams differ; parity is statistical).
const R2_MARGIN: f64 = 0.05;

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
        _ => unreachable!("rf fixtures are f32/f64 only"),
    }
}

fn from_f64<F: Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("rf fixtures are f32/f64 only"),
    }
}

fn fixture_vec<F: Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    case.expect_f64(name)
        .iter()
        .map(|&x| from_f64::<F>(x))
        .collect()
}

/// Deterministic tier: train predictions == y (sklearn purity) within 1e-5.
fn check_deterministic_tier<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load rf_reg fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let y: Vec<F> = fixture_vec::<F>(&case, "y");
    let det_pred: Vec<f64> = case.expect_f64("det_pred_train").to_vec();
    assert_eq!(x.len(), RF_N_TRAIN * RF_N_FEATURES);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);

    let reg = RandomForestRegressor::<F>::builder()
        .n_estimators(2)
        .bootstrap(false)
        .max_features(MaxFeatures::All)
        .max_depth(RF_DET_MAX_DEPTH)
        .build::<F>()
        .expect("build deterministic-tier regressor")
        .fit(&mut pool, &x_dev, Some(&y_dev), (RF_N_TRAIN, RF_N_FEATURES))
        .expect("fit deterministic-tier regressor");

    let pred = reg
        .predict(&mut pool, &x_dev, (RF_N_TRAIN, RF_N_FEATURES))
        .expect("predict")
        .to_host(&pool);
    for (i, (&got, &want)) in pred.iter().zip(det_pred.iter()).enumerate() {
        let g = host_to_f64(got);
        assert!(
            (g - want).abs() <= PRED_TOL,
            "train prediction {i}: got {g}, sklearn {want}"
        );
    }
}

/// Statistical tier: held-out R² within `R2_MARGIN` of sklearn's.
fn check_statistical_tier<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load rf_reg fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let y: Vec<F> = fixture_vec::<F>(&case, "y");
    let xq: Vec<F> = fixture_vec::<F>(&case, "Xq");
    let yq: Vec<f64> = case.expect_f64("yq").to_vec();
    let sk_r2 = case.expect_f64("stat_r2_test")[0];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq);

    let reg = RandomForestRegressor::<F>::builder()
        .n_estimators(RF_STAT_N_ESTIMATORS)
        .max_depth(RF_STAT_MAX_DEPTH)
        .build::<F>()
        .expect("build statistical-tier regressor")
        .fit(&mut pool, &x_dev, Some(&y_dev), (RF_N_TRAIN, RF_N_FEATURES))
        .expect("fit statistical-tier regressor");

    let pred = reg
        .predict(&mut pool, &xq_dev, (RF_N_TEST, RF_N_FEATURES))
        .expect("predict")
        .to_host(&pool);
    let mean_y: f64 = yq.iter().sum::<f64>() / RF_N_TEST as f64;
    let ss_res: f64 = pred
        .iter()
        .zip(yq.iter())
        .map(|(&p, &t)| (host_to_f64(p) - t).powi(2))
        .sum();
    let ss_tot: f64 = yq.iter().map(|&t| (t - mean_y).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    assert!(
        r2 >= sk_r2 - R2_MARGIN,
        "held-out R² {r2} below sklearn {sk_r2} - {R2_MARGIN}"
    );
}

#[test]
fn deterministic_tier_matches_sklearn_exactly_f32() {
    check_deterministic_tier::<f32>("rf_reg_f32_seed42.npz");
}

#[test]
fn deterministic_tier_matches_sklearn_exactly_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_deterministic_tier::<f64>("rf_reg_f64_seed42.npz");
}

#[test]
fn statistical_tier_within_margin_f32() {
    check_statistical_tier::<f32>("rf_reg_f32_seed42.npz");
}

#[test]
fn statistical_tier_within_margin_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_statistical_tier::<f64>("rf_reg_f64_seed42.npz");
}

/// Builder-time validation (shared validator with the classifier).
#[test]
fn builder_rejects_invalid_hyperparameters() {
    assert!(RandomForestRegressor::<f32>::builder()
        .n_estimators(0)
        .build::<f32>()
        .is_err());
    assert!(RandomForestRegressor::<f32>::builder()
        .max_depth(17)
        .build::<f32>()
        .is_err());
    assert!(RandomForestRegressor::<f32>::builder()
        .n_bins(1)
        .build::<f32>()
        .is_err());
    assert!(RandomForestRegressor::<f32>::builder()
        .build::<f32>()
        .is_ok());
}
