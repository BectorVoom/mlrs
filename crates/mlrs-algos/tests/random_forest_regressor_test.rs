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
use mlrs_algos::error::BuildError;
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

/// RF-IMP-01: `feature_importances()` parity with sklearn on the
/// deterministic tier. **`atol=0.05`, NOT `1e-5`/`1e-4`** — mirrors
/// `random_forest_classifier_test.rs`'s resolution (TASK-02 Green time,
/// SPEC.md `spec_revision: 2`, 2026-07-18): `predict` exact-match on this
/// tier only proves outcome-equivalence, not split-choice-equivalence.
/// sklearn's Cython "best" splitter breaks near-tied candidate splits using
/// internal state independent of the public `random_state`-controlled
/// bootstrap/feature-subsample streams, so sklearn's own two
/// deterministic-tier trees are themselves NOT bit-identical to each other,
/// even though mlrs's two trees ARE bit-identical (zero RNG consumed at
/// `bootstrap=false, max_features=All`). `atol=0.05` is tight enough to
/// catch a real attribution bug while tolerant of legitimate tie-break
/// disagreement; f32 shares the same band as f64 since the dominant error
/// source is sklearn's tie-break nondeterminism, not f32-vs-f64 rounding.
/// The qualitative dominant-feature-ranking test below remains the PRIMARY
/// correctness signal for RF-IMP-01, not a fallback.
const IMPORTANCE_TOL: f64 = 0.05;
const IMPORTANCE_TOL_F32: f64 = 0.05;

fn check_feature_importances_deterministic_tier<F>(fixture_name: &str, tol: f64)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load rf_reg fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let y: Vec<F> = fixture_vec::<F>(&case, "y");
    let ref_importances: Vec<f64> = case.expect_f64("ref_feature_importances").to_vec();
    assert_eq!(ref_importances.len(), RF_N_FEATURES);

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

    let got = reg.feature_importances();
    assert_eq!(got.len(), RF_N_FEATURES);
    for (i, (&g, &want)) in got.iter().zip(ref_importances.iter()).enumerate() {
        let gf = host_to_f64(g);
        assert!(
            (gf - want).abs() <= tol,
            "feature_importances[{i}]: got {gf}, sklearn {want}"
        );
    }

    // Second, cheaper sanity check, independent of the sklearn-comparison
    // `atol` assertion above: the normalized vector must sum to 1 (see
    // random_forest_classifier_test.rs's identical rationale for the
    // dtype-aware tolerance split).
    let sum: f64 = got.iter().map(|&g| host_to_f64(g)).sum();
    let sum_tol = match std::mem::size_of::<F>() {
        4 => 1e-6,
        8 => 1e-9,
        _ => unreachable!("rf fixtures are f32/f64 only"),
    };
    assert!(
        (sum - 1.0).abs() <= sum_tol,
        "feature_importances sums to {sum}, expected 1.0 (tol {sum_tol})"
    );
}

#[test]
fn feature_importances_matches_sklearn_deterministic_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_feature_importances_deterministic_tier::<f64>("rf_reg_f64_seed42.npz", IMPORTANCE_TOL);
}

#[test]
fn feature_importances_matches_sklearn_deterministic_f32() {
    check_feature_importances_deterministic_tier::<f32>("rf_reg_f32_seed42.npz", IMPORTANCE_TOL_F32);
}

/// RF-IMP-01 qualitative tier: a hand-built dataset with one dominant
/// feature (continuous `y` strongly correlated with feature 0 only) plus
/// noise features uncorrelated with `y`. Asserts only the RANKING property
/// (dominant strictly beats every noise feature), never an exact match.
#[test]
fn feature_importances_dominant_feature_ranking() {
    const N: usize = 60;
    const D: usize = 4; // feature 0 dominant, 1-3 noise
    let mut x: Vec<f32> = Vec::with_capacity(N * D);
    let mut y: Vec<f32> = Vec::with_capacity(N);
    for i in 0..N {
        let f0 = (i % 10) as f32 / 9.0;
        let noise = |k: usize| -> f32 {
            let v = ((i * 2654435761usize + k * 40503 + 12345) % 1000) as f32 / 1000.0;
            v
        };
        x.push(f0);
        for k in 1..D {
            x.push(noise(k));
        }
        // Continuous target strongly correlated with feature 0 only.
        y.push(10.0 * f0);
    }

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

    let reg = RandomForestRegressor::<f32>::builder()
        .build::<f32>()
        .expect("build statistical-tier regressor")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N, D))
        .expect("fit statistical-tier regressor");

    let importances = reg.feature_importances();
    assert_eq!(importances.len(), D);
    let dominant = importances[0];
    for (k, &noise_importance) in importances.iter().enumerate().skip(1) {
        assert!(
            dominant > noise_importance,
            "dominant feature 0 importance {dominant} should exceed noise feature {k} importance {noise_importance}"
        );
    }
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

/// TASK-05 (RF-OOB-01): `oob_score=true, bootstrap=false` is rejected at
/// `build()` time, mirroring the classifier's cross-check.
#[test]
fn builder_rejects_oob_score_without_bootstrap() {
    let result = RandomForestRegressor::<f64>::builder()
        .oob_score(true)
        .bootstrap(false)
        .build::<f64>();
    assert!(
        matches!(result, Err(BuildError::OobRequiresBootstrap { .. })),
        "expected Err(BuildError::OobRequiresBootstrap)"
    );
}

/// The positive case (`oob_score=true, bootstrap=true`) must NOT be rejected
/// by the new cross-check.
#[test]
fn builder_accepts_oob_score_with_bootstrap() {
    assert!(RandomForestRegressor::<f64>::builder()
        .oob_score(true)
        .bootstrap(true)
        .build::<f64>()
        .is_ok());
}

/// TASK-07 (RF-OOB-01): `oob_score()` sklearn-parity oracle assertion
/// (regressor, R²-based), statistical tier only — mirrors
/// `random_forest_classifier_test.rs`'s TASK-06 exactly, with an
/// INDEPENDENTLY-tuned margin (not assumed equal to the classifier's
/// accuracy-based `OOB_MARGIN`/`OOB_MARGIN_F32`). Green-time observed
/// divergence (both dtypes, same `RF_STAT_N_ESTIMATORS=64,
/// RF_STAT_MAX_DEPTH=8` statistical-tier hyperparameters as the `R2_MARGIN`
/// tier above): `|got - ref_oob_score| ≈ 0.00694` for f64 (`0.99157230...
/// vs 0.99851104...`) and `≈ 0.00694` for f32 (`0.99157232... vs
/// 0.99851102...` — the two dtypes agree with each other to `~1e-8`,
/// confirming the divergence is almost entirely
/// `SplitMix64`-vs-sklearn's-`MT19937` RNG-stream disagreement, not
/// f32/f64 rounding — mirrors the classifier's own TASK-06 finding). The
/// `0.10` starting `OOB_MARGIN`/`OOB_MARGIN_F32` (≈14x the observed
/// regressor divergence) was NOT widened — it already comfortably covers
/// the actually-observed Green-time value for both dtypes, so per this
/// task's own "never silently widen past what Green-time shows is needed"
/// instruction both constants were left at their starting value.
const OOB_MARGIN: f64 = 0.10;
const OOB_MARGIN_F32: f64 = 0.10;

fn check_oob_score_statistical_tier<F>(fixture_name: &str, margin: f64)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load rf_reg fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let y: Vec<F> = fixture_vec::<F>(&case, "y");
    let ref_oob_score = case.expect_f64("ref_oob_score")[0];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);

    let reg = RandomForestRegressor::<F>::builder()
        .n_estimators(RF_STAT_N_ESTIMATORS)
        .max_depth(RF_STAT_MAX_DEPTH)
        .bootstrap(true)
        .oob_score(true)
        .build::<F>()
        .expect("build oob-tier regressor")
        .fit(&mut pool, &x_dev, Some(&y_dev), (RF_N_TRAIN, RF_N_FEATURES))
        .expect("fit oob-tier regressor");

    let got = reg
        .oob_score()
        .expect("oob_score() is Some when oob_score=true");
    let gf = host_to_f64(got);
    assert!(
        (gf - ref_oob_score).abs() <= margin,
        "oob_score: got {gf}, sklearn {ref_oob_score}, margin {margin}"
    );
}

#[test]
fn oob_score_within_statistical_band_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_oob_score_statistical_tier::<f64>("rf_reg_f64_seed42.npz", OOB_MARGIN);
}

#[test]
fn oob_score_within_statistical_band_f32() {
    check_oob_score_statistical_tier::<f32>("rf_reg_f32_seed42.npz", OOB_MARGIN_F32);
}

/// `oob_score(false)` (the default) must end-to-end return `None` through
/// the full `builder -> fit -> accessor` path (not just the prim-level check
/// TASK-04 already covers).
#[test]
fn oob_score_none_when_flag_false() {
    let case = load_npz(fixture("rf_reg_f32_seed42.npz")).expect("load rf_reg fixture");
    let x: Vec<f32> = fixture_vec::<f32>(&case, "X");
    let y: Vec<f32> = fixture_vec::<f32>(&case, "y");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

    let reg = RandomForestRegressor::<f32>::builder()
        .n_estimators(RF_STAT_N_ESTIMATORS)
        .max_depth(RF_STAT_MAX_DEPTH)
        .build::<f32>()
        .expect("build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (RF_N_TRAIN, RF_N_FEATURES))
        .expect("fit");

    assert!(
        reg.oob_score().is_none(),
        "oob_score() must be None when oob_score=false (the default)"
    );
}
