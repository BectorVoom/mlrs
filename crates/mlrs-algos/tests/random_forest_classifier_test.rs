//! RandomForestClassifier (ENSEMBLE-01) sklearn oracle tests.
//!
//! Two tiers over the committed `rf_cls_{f32,f64}_seed42.npz` fixtures
//! (grid-valued features, 3 classes, ~10% label noise — `gen_oracle.py`):
//!
//! - DETERMINISTIC tier (`bootstrap=false`, all features, depth 12): the
//!   fixture generator ASSERTS sklearn reaches pure leaves on the train set,
//!   and with `<< n_bins` distinct values per feature the mlrs binned
//!   candidate set equals sklearn's exact midpoint set — so `predict_labels`
//!   on the TRAIN set must match sklearn (== `y`) EXACTLY and `predict_proba`
//!   must be the same one-hot rows within 1e-5. Held-out points are NOT
//!   exact-gated (decision-equivalent thresholds may differ inside data gaps
//!   — the Phase-17 witness Open-Question-1 resolution).
//! - STATISTICAL tier (bootstrap + sqrt features, 64 trees): held-out
//!   accuracy within 0.05 of the stored sklearn-defaults accuracy.
//!
//! Also gates the `classes_` non-contiguous label contract (CR-03) and the
//! builder validation errors. f64 functions carry the `skip_f64_with_log`
//! capability gate. Per AGENTS.md §2 tests live here, never in-source.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::ensemble::random_forest_classifier::RandomForestClassifier;
use mlrs_algos::ensemble::MaxFeatures;
use mlrs_algos::error::BuildError;
use mlrs_algos::typestate::{Fit, PredictLabels, PredictProba};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// RF fixture geometry (gen_oracle.py).
const RF_N_TRAIN: usize = 96;
const RF_N_TEST: usize = 48;
const RF_N_FEATURES: usize = 5;
const RF_N_CLASSES: usize = 3;
const RF_DET_MAX_DEPTH: usize = 12;
const RF_STAT_N_ESTIMATORS: usize = 64;
const RF_STAT_MAX_DEPTH: usize = 8;

const PROBA_TOL: f64 = 1e-5;
/// Statistical-tier margin: the mlrs forest's held-out accuracy must land
/// within this of the stored sklearn-defaults accuracy (RNG streams differ,
/// so exact parity is not defined for the bootstrap tier).
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

/// Deterministic tier: exact train-set parity with sklearn.
fn check_deterministic_tier<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load rf_cls fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let y: Vec<F> = fixture_vec::<F>(&case, "y");
    let det_pred: Vec<f64> = case.expect_f64("det_pred_train").to_vec();
    let det_proba: Vec<f64> = case.expect_f64("det_proba_train").to_vec();
    assert_eq!(x.len(), RF_N_TRAIN * RF_N_FEATURES);
    assert_eq!(det_proba.len(), RF_N_TRAIN * RF_N_CLASSES);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);

    let clf = RandomForestClassifier::<F>::builder()
        .n_estimators(2)
        .bootstrap(false)
        .max_features(MaxFeatures::All)
        .max_depth(RF_DET_MAX_DEPTH)
        .build::<F>()
        .expect("build deterministic-tier classifier")
        .fit(&mut pool, &x_dev, Some(&y_dev), (RF_N_TRAIN, RF_N_FEATURES))
        .expect("fit deterministic-tier classifier");

    assert_eq!(clf.n_classes(), RF_N_CLASSES);
    assert_eq!(clf.classes(), &[0, 1, 2]);

    // predict_labels on the TRAIN set == sklearn's (== y, purity asserted at
    // fixture generation) — EXACT.
    let labels = clf
        .predict_labels(&mut pool, &x_dev, (RF_N_TRAIN, RF_N_FEATURES))
        .expect("predict_labels")
        .to_host(&pool);
    for (i, (&got, &want)) in labels.iter().zip(det_pred.iter()).enumerate() {
        assert_eq!(
            got, want as i32,
            "train prediction {i}: got {got}, sklearn {want}"
        );
    }

    // predict_proba on the TRAIN set: the same one-hot rows within 1e-5.
    let proba = clf
        .predict_proba(&mut pool, &x_dev, (RF_N_TRAIN, RF_N_FEATURES))
        .expect("predict_proba")
        .to_host(&pool);
    assert_eq!(proba.len(), RF_N_TRAIN * RF_N_CLASSES);
    for (i, (&got, &want)) in proba.iter().zip(det_proba.iter()).enumerate() {
        let g = host_to_f64(got);
        assert!(
            (g - want).abs() <= PROBA_TOL,
            "train proba[{i}]: got {g}, sklearn {want}"
        );
    }
}

/// Statistical tier: held-out accuracy within `ACC_MARGIN` of sklearn's.
fn check_statistical_tier<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load rf_cls fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let y: Vec<F> = fixture_vec::<F>(&case, "y");
    let xq: Vec<F> = fixture_vec::<F>(&case, "Xq");
    let yq: Vec<f64> = case.expect_f64("yq").to_vec();
    let sk_acc = case.expect_f64("stat_acc_test")[0];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq);

    let clf = RandomForestClassifier::<F>::builder()
        .n_estimators(RF_STAT_N_ESTIMATORS)
        .max_depth(RF_STAT_MAX_DEPTH)
        .build::<F>()
        .expect("build statistical-tier classifier")
        .fit(&mut pool, &x_dev, Some(&y_dev), (RF_N_TRAIN, RF_N_FEATURES))
        .expect("fit statistical-tier classifier");

    let labels = clf
        .predict_labels(&mut pool, &xq_dev, (RF_N_TEST, RF_N_FEATURES))
        .expect("predict_labels")
        .to_host(&pool);
    let correct = labels
        .iter()
        .zip(yq.iter())
        .filter(|&(&got, &want)| got == want as i32)
        .count();
    let acc = correct as f64 / RF_N_TEST as f64;
    assert!(
        acc >= sk_acc - ACC_MARGIN,
        "held-out accuracy {acc} below sklearn {sk_acc} - {ACC_MARGIN}"
    );

    // Proba rows must sum to 1 (device mean of leaf distributions).
    let proba = clf
        .predict_proba(&mut pool, &xq_dev, (RF_N_TEST, RF_N_FEATURES))
        .expect("predict_proba")
        .to_host(&pool);
    for r in 0..RF_N_TEST {
        let sum: f64 = (0..RF_N_CLASSES)
            .map(|c| host_to_f64(proba[r * RF_N_CLASSES + c]))
            .sum();
        assert!((sum - 1.0).abs() <= 1e-4, "proba row {r} sums to {sum}");
    }
}

#[test]
fn deterministic_tier_matches_sklearn_exactly_f32() {
    check_deterministic_tier::<f32>("rf_cls_f32_seed42.npz");
}

#[test]
fn deterministic_tier_matches_sklearn_exactly_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_deterministic_tier::<f64>("rf_cls_f64_seed42.npz");
}

#[test]
fn statistical_tier_within_margin_f32() {
    check_statistical_tier::<f32>("rf_cls_f32_seed42.npz");
}

#[test]
fn statistical_tier_within_margin_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_statistical_tier::<f64>("rf_cls_f64_seed42.npz");
}

/// CR-03: a NON-CONTIGUOUS label set (`{3, 7, 11}`) must round-trip through
/// `classes_` — predictions return the ORIGINAL labels, never dense indices.
#[test]
fn non_contiguous_labels_round_trip_f32() {
    let case = load_npz(fixture("rf_cls_f32_seed42.npz")).expect("load rf_cls fixture");
    let x: Vec<f32> = fixture_vec::<f32>(&case, "X");
    let y_dense: Vec<f64> = case.expect_f64("y").to_vec();
    // Remap class c → 3 + 4c (so {0,1,2} → {3,7,11}).
    let y_shift: Vec<f32> = y_dense.iter().map(|&c| (3.0 + 4.0 * c) as f32).collect();
    let det_pred: Vec<f64> = case.expect_f64("det_pred_train").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y_shift);

    let clf = RandomForestClassifier::<f32>::builder()
        .n_estimators(2)
        .bootstrap(false)
        .max_features(MaxFeatures::All)
        .max_depth(RF_DET_MAX_DEPTH)
        .build::<f32>()
        .expect("build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (RF_N_TRAIN, RF_N_FEATURES))
        .expect("fit");

    assert_eq!(clf.classes(), &[3, 7, 11]);
    let labels = clf
        .predict_labels(&mut pool, &x_dev, (RF_N_TRAIN, RF_N_FEATURES))
        .expect("predict_labels")
        .to_host(&pool);
    for (i, (&got, &want_dense)) in labels.iter().zip(det_pred.iter()).enumerate() {
        let want = 3 + 4 * (want_dense as i32);
        assert_eq!(got, want, "shifted train prediction {i}");
    }
}

/// RF-IMP-01: `feature_importances()` parity with sklearn on the
/// deterministic tier. **`atol=0.05`, NOT `1e-5`/`1e-4`** — resolved at
/// TASK-02 Green time (SPEC.md `spec_revision: 2`, 2026-07-18): `predict`/
/// `predict_proba` exact-match on this tier only proves outcome-equivalence,
/// not split-choice-equivalence. sklearn's Cython "best" splitter breaks
/// near-tied candidate splits using internal state independent of the public
/// `random_state`-controlled bootstrap/feature-subsample streams, so
/// sklearn's own two deterministic-tier trees are themselves NOT bit-identical
/// to each other (confirmed empirically), even though mlrs's two trees ARE
/// bit-identical (zero RNG consumed at `bootstrap=false, max_features=All`).
/// A genuine tied-split divergence was observed at a low-sample deep node,
/// producing a ~0.0022 per-feature divergence in `feature_importances_`.
/// `atol=0.05` (25x the observed divergence) is tight enough to catch a real
/// attribution bug while tolerant of legitimate tie-break disagreement; f32
/// shares the same band as f64 since the dominant error source is sklearn's
/// tie-break nondeterminism, not f32-vs-f64 rounding. The qualitative
/// dominant-feature-ranking test below remains the PRIMARY correctness
/// signal for RF-IMP-01, not a fallback.
const IMPORTANCE_TOL: f64 = 0.05;
const IMPORTANCE_TOL_F32: f64 = 0.05;

fn check_feature_importances_deterministic_tier<F>(fixture_name: &str, tol: f64)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load rf_cls fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let y: Vec<F> = fixture_vec::<F>(&case, "y");
    let ref_importances: Vec<f64> = case.expect_f64("ref_feature_importances").to_vec();
    assert_eq!(ref_importances.len(), RF_N_FEATURES);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);

    let clf = RandomForestClassifier::<F>::builder()
        .n_estimators(2)
        .bootstrap(false)
        .max_features(MaxFeatures::All)
        .max_depth(RF_DET_MAX_DEPTH)
        .build::<F>()
        .expect("build deterministic-tier classifier")
        .fit(&mut pool, &x_dev, Some(&y_dev), (RF_N_TRAIN, RF_N_FEATURES))
        .expect("fit deterministic-tier classifier");

    let got = clf.feature_importances();
    assert_eq!(got.len(), RF_N_FEATURES);
    for (i, (&g, &want)) in got.iter().zip(ref_importances.iter()).enumerate() {
        let gf = host_to_f64(g);
        assert!(
            (gf - want).abs() <= tol,
            "feature_importances[{i}]: got {gf}, sklearn {want}"
        );
    }

    // Second, cheaper sanity check, independent of the sklearn-comparison
    // `atol` assertion above: the normalized vector must sum to 1. The host
    // reduction normalizes in f64 BEFORE casting each element down to `F`
    // (`f64_to_host::<F>`, `random_forest.rs`), so f64 recovers the exact
    // `1.0 ± 1e-9` band; f32 loses per-element mantissa precision on that
    // final cast, so summing 5 f32-rounded elements back through
    // `host_to_f64` needs the correspondingly wider (still tight) f32 sum
    // tolerance (matches this file's own established f32/f64 split
    // convention, e.g. `IMPORTANCE_TOL`/`IMPORTANCE_TOL_F32` above).
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
    check_feature_importances_deterministic_tier::<f64>("rf_cls_f64_seed42.npz", IMPORTANCE_TOL);
}

#[test]
fn feature_importances_matches_sklearn_deterministic_f32() {
    check_feature_importances_deterministic_tier::<f32>("rf_cls_f32_seed42.npz", IMPORTANCE_TOL_F32);
}

/// RF-IMP-01 qualitative tier: a hand-built dataset with one dominant,
/// perfectly-separating feature plus noise features. Asserts only the
/// RANKING property (dominant strictly beats every noise feature), never an
/// exact match — the statistical/`bootstrap=true` tier has no cross-implementation
/// exact-match guarantee.
#[test]
fn feature_importances_dominant_feature_ranking() {
    const N: usize = 60;
    const D: usize = 4; // feature 0 dominant, 1-3 noise
    let mut x: Vec<f32> = Vec::with_capacity(N * D);
    let mut y: Vec<f32> = Vec::with_capacity(N);
    for i in 0..N {
        let f0 = if i % 2 == 0 { 0.0f32 } else { 1.0f32 };
        // Deterministic pseudo-noise via a simple LCG-like formula, bounded
        // to [0,1), uncorrelated with the label.
        let noise = |k: usize| -> f32 {
            let v = ((i * 2654435761usize + k * 40503 + 12345) % 1000) as f32 / 1000.0;
            v
        };
        x.push(f0);
        for k in 1..D {
            x.push(noise(k));
        }
        y.push(f0);
    }

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

    let clf = RandomForestClassifier::<f32>::builder()
        .build::<f32>()
        .expect("build statistical-tier classifier")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N, D))
        .expect("fit statistical-tier classifier");

    let importances = clf.feature_importances();
    assert_eq!(importances.len(), D);
    let dominant = importances[0];
    for (k, &noise_importance) in importances.iter().enumerate().skip(1) {
        assert!(
            dominant > noise_importance,
            "dominant feature 0 importance {dominant} should exceed noise feature {k} importance {noise_importance}"
        );
    }
}

/// Builder-time validation (D-08 data-independent split).
#[test]
fn builder_rejects_invalid_hyperparameters() {
    assert!(RandomForestClassifier::<f32>::builder()
        .n_estimators(0)
        .build::<f32>()
        .is_err());
    assert!(RandomForestClassifier::<f32>::builder()
        .max_depth(0)
        .build::<f32>()
        .is_err());
    assert!(RandomForestClassifier::<f32>::builder()
        .max_depth(17)
        .build::<f32>()
        .is_err());
    assert!(RandomForestClassifier::<f32>::builder()
        .n_bins(1)
        .build::<f32>()
        .is_err());
    assert!(RandomForestClassifier::<f32>::builder()
        .n_bins(257)
        .build::<f32>()
        .is_err());
    assert!(RandomForestClassifier::<f32>::builder()
        .max_features(MaxFeatures::Value(0))
        .build::<f32>()
        .is_err());
    assert!(RandomForestClassifier::<f32>::builder()
        .min_samples_split(1.0)
        .build::<f32>()
        .is_err());
    assert!(RandomForestClassifier::<f32>::builder()
        .min_samples_leaf(0.0)
        .build::<f32>()
        .is_err());
    assert!(RandomForestClassifier::<f32>::builder()
        .build::<f32>()
        .is_ok());
}

/// TASK-05 (RF-OOB-01): `oob_score=true, bootstrap=false` is rejected at
/// `build()` time, matching sklearn's `ValueError("Out of bag estimation
/// only available if bootstrap=True")`.
#[test]
fn builder_rejects_oob_score_without_bootstrap() {
    let result = RandomForestClassifier::<f64>::builder()
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
    assert!(RandomForestClassifier::<f64>::builder()
        .oob_score(true)
        .bootstrap(true)
        .build::<f64>()
        .is_ok());
}

/// TASK-06 (RF-OOB-01): `oob_score()` sklearn-parity oracle assertion,
/// statistical tier only (a stochastic quantity has no exact-match tier —
/// `SplitMix64` != sklearn's `MT19937`). Starting bands, documented and
/// independently tunable per dtype (never assumed equal, per this task's own
/// Implementation Steps). Green-time observed divergence (both dtypes, same
/// `RF_STAT_N_ESTIMATORS=64, RF_STAT_MAX_DEPTH=8` statistical-tier
/// hyperparameters as the `ACC_MARGIN` tier above): `|got - ref_oob_score|
/// ≈ 0.0104` (f64: `0.80208333... vs 0.8125`; f32: `0.80208331... vs
/// 0.8125` — the two dtypes land within `~3e-8` of each other, i.e. this
/// divergence is almost entirely `SplitMix64`-vs-`MT19937` RNG-stream
/// disagreement, not f32/f64 rounding). The `0.10` starting point (≈10x the
/// observed divergence) was NOT widened — it already comfortably covers the
/// actually-observed Green-time value for both dtypes, so no per-dtype
/// widening was needed per this task's own "never silently widen past what
/// Green-time shows is needed" instruction.
const OOB_MARGIN: f64 = 0.10;
const OOB_MARGIN_F32: f64 = 0.10;

fn check_oob_score_statistical_tier<F>(fixture_name: &str, margin: f64)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load rf_cls fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let y: Vec<F> = fixture_vec::<F>(&case, "y");
    let ref_oob_score = case.expect_f64("ref_oob_score")[0];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);

    let clf = RandomForestClassifier::<F>::builder()
        .n_estimators(RF_STAT_N_ESTIMATORS)
        .max_depth(RF_STAT_MAX_DEPTH)
        .bootstrap(true)
        .oob_score(true)
        .build::<F>()
        .expect("build oob-tier classifier")
        .fit(&mut pool, &x_dev, Some(&y_dev), (RF_N_TRAIN, RF_N_FEATURES))
        .expect("fit oob-tier classifier");

    let got = clf
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
    check_oob_score_statistical_tier::<f64>("rf_cls_f64_seed42.npz", OOB_MARGIN);
}

#[test]
fn oob_score_within_statistical_band_f32() {
    check_oob_score_statistical_tier::<f32>("rf_cls_f32_seed42.npz", OOB_MARGIN_F32);
}

/// `oob_score(false)` (the default) must end-to-end return `None` through
/// the full `builder -> fit -> accessor` path (not just the prim-level
/// check TASK-04 already covers).
#[test]
fn oob_score_none_when_flag_false() {
    let case = load_npz(fixture("rf_cls_f32_seed42.npz")).expect("load rf_cls fixture");
    let x: Vec<f32> = fixture_vec::<f32>(&case, "X");
    let y: Vec<f32> = fixture_vec::<f32>(&case, "y");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

    let clf = RandomForestClassifier::<f32>::builder()
        .n_estimators(RF_STAT_N_ESTIMATORS)
        .max_depth(RF_STAT_MAX_DEPTH)
        .build::<f32>()
        .expect("build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (RF_N_TRAIN, RF_N_FEATURES))
        .expect("fit");

    assert!(
        clf.oob_score().is_none(),
        "oob_score() must be None when oob_score=false (the default)"
    );
}
