//! HistGradientBoostingClassifier (GBT-01) sklearn oracle tests.
//!
//! Two tiers over the committed `hgb_cls_{f32,f64}_seed42.npz` fixtures
//! (grid-valued features, CLEAN 3-class rule for the exact tier + its noisy
//! sibling for the statistical tier, plus the binarized `y == 0` target —
//! `gen_oracle.py`):
//!
//! - DETERMINISTIC tier (`max_iter=20, lr=0.1, max_depth=6, n_bins=255,
//!   min_samples_leaf=5, l2=0`, CLEAN labels): sklearn fitted with
//!   `max_leaf_nodes=None` + the same depth bound is growth-order-equivalent
//!   to the mlrs level-wise grower (see the regressor test header), and the
//!   clean rule keeps every informative gain tie-free (noisy labels create
//!   exact-tie gains whose float resolution differs between sklearn's
//!   histogram-SUBTRACTION sums and the mlrs direct sums — the generator
//!   documents this). TRAIN probabilities match within tolerance and TRAIN
//!   labels exactly, for BOTH class paths: 3-class (softmax, K = 3 batched
//!   trees/iteration) and binary (sigmoid, K = 1).
//! - STATISTICAL tier (sklearn defaults on the NOISY labels vs mlrs
//!   defaults): held-out accuracy within 0.05 of the stored sklearn value.
//!
//! f64 functions carry the `skip_f64_with_log` gate AND skip on wgpu: the
//! log-loss kernels use `F::exp`, and 64-bit `exp` SIGSEGVs this
//! environment's RADV shader compiler (see
//! `mlrs-backend/tests/hist_gradient_boosting_test.rs`). Per AGENTS.md §2
//! tests live here, never in-source.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::ensemble::hist_gradient_boosting_classifier::HistGradientBoostingClassifier;
use mlrs_algos::typestate::{Fit, PredictLabels, PredictProba};
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

/// Probability tolerances (see the regressor test: f32 accumulates 20
/// iterations of f32 leaf values before the link function).
const PROBA_TOL_F64: f64 = 1e-5;
const PROBA_TOL_F32: f64 = 1e-4;
/// Statistical-tier margin.
const ACC_MARGIN: f64 = 0.05;

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

/// Skip f64 log-loss (exp-using) cases on wgpu (RADV ACO lacks 64-bit fexp2 —
/// the backend test's documented landmine).
fn skip_f64_exp_on_wgpu() -> bool {
    if cfg!(feature = "wgpu") {
        eprintln!("skipping f64 exp kernel on wgpu: RADV ACO lacks 64-bit fexp2");
        return true;
    }
    false
}

fn det_builder<F>() -> HistGradientBoostingClassifier<F>
where
    F: Float + CubeElement + Pod,
{
    HistGradientBoostingClassifier::<F>::builder()
        .max_iter(HGB_DET_MAX_ITER)
        .learning_rate(HGB_DET_LEARNING_RATE)
        .max_depth(HGB_DET_MAX_DEPTH)
        .n_bins(255)
        .min_samples_leaf(HGB_DET_MIN_SAMPLES_LEAF)
        .l2_regularization(0.0)
        .build::<F>()
        .expect("build deterministic-tier classifier")
}

/// Deterministic tier for one target column: probas within `tol`, labels
/// exact. `y_key`/`proba_key`/`pred_key` select the 3-class or binary path.
fn check_deterministic_tier<F>(
    fixture_name: &str,
    y_key: &str,
    proba_key: &str,
    pred_key: &str,
    n_classes: usize,
    tol: f64,
) where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load hgb_cls fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let y: Vec<F> = fixture_vec::<F>(&case, y_key);
    let det_proba: Vec<f64> = case.expect_f64(proba_key).to_vec();
    let det_pred: Vec<f64> = case.expect_f64(pred_key).to_vec();
    assert_eq!(x.len(), HGB_N_TRAIN * HGB_N_FEATURES);
    assert_eq!(det_proba.len(), HGB_N_TRAIN * n_classes);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);

    let clf = det_builder::<F>()
        .fit(&mut pool, &x_dev, Some(&y_dev), (HGB_N_TRAIN, HGB_N_FEATURES))
        .expect("fit deterministic-tier classifier");
    assert_eq!(clf.n_classes(), n_classes);

    let proba = clf
        .predict_proba(&mut pool, &x_dev, (HGB_N_TRAIN, HGB_N_FEATURES))
        .expect("predict_proba")
        .to_host(&pool);
    let mut max_err = 0f64;
    for (i, (&got, &want)) in proba.iter().zip(det_proba.iter()).enumerate() {
        let g = host_to_f64(got);
        let err = (g - want).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err <= tol,
            "train proba {i}: got {g}, sklearn {want} (err {err} > tol {tol})"
        );
    }
    println!("hgb_cls det tier[{fixture_name}/{y_key}]: max_abs_err = {max_err:e}");

    let labels = clf
        .predict_labels(&mut pool, &x_dev, (HGB_N_TRAIN, HGB_N_FEATURES))
        .expect("predict_labels")
        .to_host(&pool);
    for (i, (&got, &want)) in labels.iter().zip(det_pred.iter()).enumerate() {
        assert_eq!(got, want as i32, "train label {i}");
    }
}

/// Statistical tier: mlrs defaults' held-out accuracy within `ACC_MARGIN` of
/// the stored sklearn-defaults accuracy (3-class path).
fn check_statistical_tier<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load hgb_cls fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let y: Vec<F> = fixture_vec::<F>(&case, "y_noisy");
    let xq: Vec<F> = fixture_vec::<F>(&case, "Xq");
    let yq: Vec<f64> = case.expect_f64("yq").to_vec();
    let sk_acc = case.expect_f64("stat_acc_test")[0];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq);

    let clf = HistGradientBoostingClassifier::<F>::builder()
        .build::<F>()
        .expect("build statistical-tier classifier")
        .fit(&mut pool, &x_dev, Some(&y_dev), (HGB_N_TRAIN, HGB_N_FEATURES))
        .expect("fit statistical-tier classifier");

    let labels = clf
        .predict_labels(&mut pool, &xq_dev, (HGB_N_TEST, HGB_N_FEATURES))
        .expect("predict_labels")
        .to_host(&pool);
    let correct = labels
        .iter()
        .zip(yq.iter())
        .filter(|&(&l, &t)| l == t as i32)
        .count();
    let acc = correct as f64 / HGB_N_TEST as f64;
    assert!(
        acc >= sk_acc - ACC_MARGIN,
        "held-out accuracy {acc} below sklearn {sk_acc} - {ACC_MARGIN}"
    );
}

#[test]
fn deterministic_multiclass_matches_sklearn_f32() {
    check_deterministic_tier::<f32>(
        "hgb_cls_f32_seed42.npz",
        "y",
        "det_proba_train",
        "det_pred_train",
        3,
        PROBA_TOL_F32,
    );
}

#[test]
fn deterministic_multiclass_matches_sklearn_f64() {
    if capability::skip_f64_with_log() || skip_f64_exp_on_wgpu() {
        return;
    }
    check_deterministic_tier::<f64>(
        "hgb_cls_f64_seed42.npz",
        "y",
        "det_proba_train",
        "det_pred_train",
        3,
        PROBA_TOL_F64,
    );
}

#[test]
fn deterministic_binary_matches_sklearn_f32() {
    check_deterministic_tier::<f32>(
        "hgb_cls_f32_seed42.npz",
        "y_bin",
        "det_proba_bin_train",
        "det_pred_bin_train",
        2,
        PROBA_TOL_F32,
    );
}

#[test]
fn deterministic_binary_matches_sklearn_f64() {
    if capability::skip_f64_with_log() || skip_f64_exp_on_wgpu() {
        return;
    }
    check_deterministic_tier::<f64>(
        "hgb_cls_f64_seed42.npz",
        "y_bin",
        "det_proba_bin_train",
        "det_pred_bin_train",
        2,
        PROBA_TOL_F64,
    );
}

#[test]
fn statistical_tier_within_margin_f32() {
    check_statistical_tier::<f32>("hgb_cls_f32_seed42.npz");
}

#[test]
fn statistical_tier_within_margin_f64() {
    if capability::skip_f64_with_log() || skip_f64_exp_on_wgpu() {
        return;
    }
    check_statistical_tier::<f64>("hgb_cls_f64_seed42.npz");
}

/// Builder-time validation (shared validator with the regressor).
#[test]
fn builder_rejects_invalid_hyperparameters() {
    assert!(HistGradientBoostingClassifier::<f32>::builder()
        .max_iter(0)
        .build::<f32>()
        .is_err());
    assert!(HistGradientBoostingClassifier::<f32>::builder()
        .learning_rate(-0.1)
        .build::<f32>()
        .is_err());
    assert!(HistGradientBoostingClassifier::<f32>::builder()
        .max_depth(0)
        .build::<f32>()
        .is_err());
    assert!(HistGradientBoostingClassifier::<f32>::builder()
        .n_bins(257)
        .build::<f32>()
        .is_err());
    assert!(HistGradientBoostingClassifier::<f32>::builder()
        .build::<f32>()
        .is_ok());
}
