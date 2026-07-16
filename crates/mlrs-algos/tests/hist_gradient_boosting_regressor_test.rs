//! HistGradientBoostingRegressor (GBT-01) sklearn oracle tests.
//!
//! Two tiers over the committed `hgb_reg_{f32,f64}_seed42.npz` fixtures
//! (grid-valued features, piecewise-constant target — `gen_oracle.py`):
//!
//! - DETERMINISTIC tier (`max_iter=20, lr=0.1, max_depth=6, n_bins=255,
//!   min_samples_leaf=5, l2=0`): sklearn is fitted with `max_leaf_nodes=None`
//!   + the same depth bound, which makes its leaf-wise growth ORDER-IRRELEVANT
//!   and its trees equal to the mlrs level-wise trees (identical midpoint
//!   candidate sets on grid data, identical gain rule, no RNG on either
//!   side) — so TRAIN predictions must match sklearn within tolerance.
//! - STATISTICAL tier (sklearn defaults vs mlrs defaults): held-out R² within
//!   0.05 of the stored sklearn-defaults R² (tree SHAPES differ there:
//!   sklearn grows leaf-wise with a 31-leaf budget).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate. Per AGENTS.md
//! §2 tests live here, never in-source.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::ensemble::hist_gradient_boosting_regressor::HistGradientBoostingRegressor;
use mlrs_algos::typestate::{Fit, Predict};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// HGB fixture geometry + deterministic-tier hyperparameters (gen_oracle.py).
const HGB_N_TRAIN: usize = 96;
const HGB_N_TEST: usize = 48;
const HGB_N_FEATURES: usize = 5;
const HGB_DET_MAX_ITER: usize = 20;
const HGB_DET_MAX_DEPTH: usize = 6;
const HGB_DET_MIN_SAMPLES_LEAF: usize = 5;
const HGB_DET_LEARNING_RATE: f64 = 0.1;

/// f64 runs the whole pipeline in f64 against sklearn's f64 histogram sums
/// (sklearn's per-sample gradients are f32 — the residual difference is far
/// below this gate). f32 accumulates 20 iterations of f32 leaf values, so its
/// gate is one decade looser.
const PRED_TOL_F64: f64 = 1e-5;
const PRED_TOL_F32: f64 = 1e-4;
/// Statistical-tier margin (tree shapes differ: leaf-wise vs level-wise).
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
        _ => unreachable!("hgb fixtures are f32/f64 only"),
    }
}

fn from_f64<F: Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("hgb fixtures are f32/f64 only"),
    }
}

fn fixture_vec<F: Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    case.expect_f64(name)
        .iter()
        .map(|&x| from_f64::<F>(x))
        .collect()
}

/// Deterministic tier: train predictions == sklearn's within tolerance.
fn check_deterministic_tier<F>(fixture_name: &str, tol: f64)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load hgb_reg fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let y: Vec<F> = fixture_vec::<F>(&case, "y");
    let det_pred: Vec<f64> = case.expect_f64("det_pred_train").to_vec();
    assert_eq!(x.len(), HGB_N_TRAIN * HGB_N_FEATURES);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);

    let reg = HistGradientBoostingRegressor::<F>::builder()
        .max_iter(HGB_DET_MAX_ITER)
        .learning_rate(HGB_DET_LEARNING_RATE)
        .max_depth(HGB_DET_MAX_DEPTH)
        .n_bins(255)
        .min_samples_leaf(HGB_DET_MIN_SAMPLES_LEAF)
        .l2_regularization(0.0)
        .build::<F>()
        .expect("build deterministic-tier regressor")
        .fit(&mut pool, &x_dev, Some(&y_dev), (HGB_N_TRAIN, HGB_N_FEATURES))
        .expect("fit deterministic-tier regressor");

    let pred = reg
        .predict(&mut pool, &x_dev, (HGB_N_TRAIN, HGB_N_FEATURES))
        .expect("predict")
        .to_host(&pool);
    let mut max_err = 0f64;
    for (i, (&got, &want)) in pred.iter().zip(det_pred.iter()).enumerate() {
        let g = host_to_f64(got);
        let err = (g - want).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err <= tol,
            "train prediction {i}: got {g}, sklearn {want} (err {err} > tol {tol})"
        );
    }
    println!("hgb_reg det tier[{fixture_name}]: max_abs_err = {max_err:e}");
}

/// Statistical tier: mlrs defaults' held-out R² within `R2_MARGIN` of the
/// stored sklearn-defaults R².
fn check_statistical_tier<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load hgb_reg fixture");
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

    let reg = HistGradientBoostingRegressor::<F>::builder()
        .build::<F>()
        .expect("build statistical-tier regressor")
        .fit(&mut pool, &x_dev, Some(&y_dev), (HGB_N_TRAIN, HGB_N_FEATURES))
        .expect("fit statistical-tier regressor");

    let pred = reg
        .predict(&mut pool, &xq_dev, (HGB_N_TEST, HGB_N_FEATURES))
        .expect("predict")
        .to_host(&pool);
    let mean_y: f64 = yq.iter().sum::<f64>() / HGB_N_TEST as f64;
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
fn deterministic_tier_matches_sklearn_f32() {
    check_deterministic_tier::<f32>("hgb_reg_f32_seed42.npz", PRED_TOL_F32);
}

#[test]
fn deterministic_tier_matches_sklearn_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_deterministic_tier::<f64>("hgb_reg_f64_seed42.npz", PRED_TOL_F64);
}

#[test]
fn statistical_tier_within_margin_f32() {
    check_statistical_tier::<f32>("hgb_reg_f32_seed42.npz");
}

#[test]
fn statistical_tier_within_margin_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_statistical_tier::<f64>("hgb_reg_f64_seed42.npz");
}

/// Builder-time validation (shared validator with the classifier).
#[test]
fn builder_rejects_invalid_hyperparameters() {
    assert!(HistGradientBoostingRegressor::<f32>::builder()
        .max_iter(0)
        .build::<f32>()
        .is_err());
    assert!(HistGradientBoostingRegressor::<f32>::builder()
        .learning_rate(0.0)
        .build::<f32>()
        .is_err());
    assert!(HistGradientBoostingRegressor::<f32>::builder()
        .learning_rate(f64::NAN)
        .build::<f32>()
        .is_err());
    assert!(HistGradientBoostingRegressor::<f32>::builder()
        .max_depth(17)
        .build::<f32>()
        .is_err());
    assert!(HistGradientBoostingRegressor::<f32>::builder()
        .n_bins(1)
        .build::<f32>()
        .is_err());
    assert!(HistGradientBoostingRegressor::<f32>::builder()
        .l2_regularization(-0.5)
        .build::<f32>()
        .is_err());
    assert!(HistGradientBoostingRegressor::<f32>::builder()
        .min_samples_leaf(0)
        .build::<f32>()
        .is_err());
    assert!(HistGradientBoostingRegressor::<f32>::builder()
        .build::<f32>()
        .is_ok());
}
