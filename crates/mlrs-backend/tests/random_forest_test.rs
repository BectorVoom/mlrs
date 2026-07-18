//! Random Forest prim (ENSEMBLE-01) standalone hand-oracle tests.
//!
//! Exercises `mlrs_backend::prims::random_forest::{rf_fit_class, rf_fit_reg,
//! rf_predict_proba, rf_predict_reg}` on a tiny 8-point, 2-feature dataset
//! whose greedy binned tree is fully HAND-COMPUTED (split features, thresholds
//! = quantile-midpoint bin edges, leaf distributions, predictions), so the
//! whole kernel pipeline (bin → hist → cum → stats → scores → best-split →
//! partition → traverse → vote) is validated against exact expected VALUES —
//! never just non-panic (the spike verification discipline; a silent
//! cpu-MLIR miscompile reads back as wrong values here).
//!
//! Hand derivation (gini proxy `Σ_c l_c²/n_l + Σ_c r_c²/n_r`, maximize):
//! classes (4,2,2); root best = feature 0 @ 0.5 (proxy 6 vs ≤ 4.8 elsewhere,
//! unique). Left child pure class 0 → leaf. Right child (0,2,2): best proxy 4
//! tie among x1-edges {0.35, 0.5, 0.7}; strict-> lowest-(f,s) tie-break picks
//! threshold 0.35. Grandchildren pure. Regression (y = 1/2/3 per class):
//! root = feature 0 @ 0.5 (proxy 29); right child tie (26) between f0@0.75
//! (k=5) and f1@0.35 (k=8) → f0@0.75 wins by flat-k order.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64).
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)]` module.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::random_forest::{
    rf_fit_class, rf_fit_reg, rf_predict_proba, rf_predict_reg, RfFitOutcome, RfParams,
};
use mlrs_backend::runtime::{self, ActiveRuntime};

const TOL: f64 = 1e-5;

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("rf tests are f32/f64 only"),
    }
}

fn from_f64<F: Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("rf tests are f32/f64 only"),
    }
}

/// The hand-oracle dataset: 8 rows × 2 features on a 0.1 grid; class 0 iff
/// `x0 < 0.5`, else class 1 iff `x1 < ~0.5` else class 2. Includes duplicate
/// feature VALUES across rows (x1 has three 0.9s / two 0.1s) so the binning
/// dedup path is exercised (the spike duplicate-point discipline).
fn xdata() -> Vec<f64> {
    vec![
        0.1, 0.1, // y=0
        0.2, 0.9, // y=0
        0.3, 0.4, // y=0
        0.4, 0.6, // y=0
        0.9, 0.1, // y=1
        0.8, 0.3, // y=1
        0.7, 0.9, // y=2
        0.6, 0.8, // y=2
    ]
}

fn params_single_tree() -> RfParams {
    RfParams {
        n_trees: 1,
        max_depth: 2,
        n_bins: 8,
        max_features: 2,
        min_samples_split: 2.0,
        min_samples_leaf: 1.0,
        bootstrap: false,
        seed: 42,
        oob_score: false,
    }
}

fn upload<F>(pool: &mut BufferPool<ActiveRuntime>, v: &[f64]) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    let host: Vec<F> = v.iter().map(|&x| from_f64::<F>(x)).collect();
    DeviceArray::from_host(pool, &host)
}

fn check_classifier_hand_oracle<F>()
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev = upload::<F>(&mut pool, &xdata());
    let y_idx: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 2, 2];

    let RfFitOutcome { model, .. } =
        rf_fit_class::<F>(&mut pool, &x_dev, (8, 2), &y_idx, 3, &params_single_tree())
            .expect("fit hand-oracle classifier");

    // Root split: feature 0, threshold 0.5 (the (0.4+0.6)/2 midpoint edge).
    let feats = model.split_feature_host(&pool);
    let thrs = model.threshold_host(&pool);
    let leaves = model.is_leaf_host(&pool);
    assert_eq!(leaves[0], 0, "root must be interior");
    assert_eq!(feats[0], 0, "root split feature");
    assert!(
        (host_to_f64(thrs[0]) - 0.5).abs() <= TOL,
        "root threshold: got {}, want 0.5",
        host_to_f64(thrs[0])
    );
    // Node 1 (left child): pure class 0 → leaf. Node 2: interior; BOTH
    // f0@0.75 (class 2 has x0 ∈ {.6,.7}, class 1 has {.8,.9}) and f1@0.35
    // reach the pure proxy 4 — the flat-(f,s) tie-break picks feature 0 @ 0.75.
    assert_eq!(leaves[1], 1, "left child is a pure leaf");
    assert_eq!(leaves[2], 0, "right child is interior");
    assert_eq!(feats[2], 0, "right child split feature (flat-k tie-break)");
    assert!(
        (host_to_f64(thrs[2]) - 0.75).abs() <= TOL,
        "right-child threshold: got {}, want 0.75 (tie-break)",
        host_to_f64(thrs[2])
    );

    // Predictions on queries away from every bin edge: hand-traced classes.
    let xq = vec![0.2, 0.5, 0.85, 0.2, 0.65, 0.95];
    let xq_dev = upload::<F>(&mut pool, &xq);
    let proba = rf_predict_proba::<F>(&mut pool, &model, &xq_dev, (3, 2)).expect("predict_proba");
    let p: Vec<f64> = proba.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    assert_eq!(p.len(), 9);
    let expected = [
        [1.0, 0.0, 0.0], // (0.2, 0.5) → pure class-0 leaf
        [0.0, 1.0, 0.0], // (0.85, 0.2) → pure class-1 leaf
        [0.0, 0.0, 1.0], // (0.65, 0.95) → pure class-2 leaf
    ];
    for (r, row) in expected.iter().enumerate() {
        let mut sum = 0.0;
        for (c, &want) in row.iter().enumerate() {
            let got = p[r * 3 + c];
            assert!(
                (got - want).abs() <= TOL,
                "proba[{r},{c}]: got {got}, want {want}"
            );
            sum += got;
        }
        assert!((sum - 1.0).abs() <= TOL, "proba row {r} must sum to 1");
    }
}

fn check_regressor_hand_oracle<F>()
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev = upload::<F>(&mut pool, &xdata());
    let y = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    let y_dev = upload::<F>(&mut pool, &y);

    let RfFitOutcome { model, .. } =
        rf_fit_reg::<F>(&mut pool, &x_dev, (8, 2), &y_dev, &params_single_tree())
            .expect("fit hand-oracle regressor");

    // Root: feature 0 @ 0.5. Right child: feature 0 @ 0.75 (flat-k tie-break
    // over the equal-proxy f1@0.35 split).
    let feats = model.split_feature_host(&pool);
    let thrs = model.threshold_host(&pool);
    assert_eq!(feats[0], 0, "root split feature");
    assert!((host_to_f64(thrs[0]) - 0.5).abs() <= TOL, "root threshold");
    assert_eq!(feats[2], 0, "right-child split feature (tie-break)");
    assert!(
        (host_to_f64(thrs[2]) - 0.75).abs() <= TOL,
        "right-child threshold: got {}, want 0.75",
        host_to_f64(thrs[2])
    );

    let xq = vec![0.2, 0.5, 0.85, 0.2, 0.65, 0.95];
    let xq_dev = upload::<F>(&mut pool, &xq);
    let pred = rf_predict_reg::<F>(&mut pool, &model, &xq_dev, (3, 2)).expect("predict");
    let got: Vec<f64> = pred.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    let want = [1.0, 2.0, 3.0];
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() <= TOL, "pred[{i}]: got {g}, want {w}");
    }
}

#[test]
fn classifier_hand_oracle_f32() {
    check_classifier_hand_oracle::<f32>();
}

#[test]
fn classifier_hand_oracle_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_classifier_hand_oracle::<f64>();
}

#[test]
fn regressor_hand_oracle_f32() {
    check_regressor_hand_oracle::<f32>();
}

#[test]
fn regressor_hand_oracle_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_regressor_hand_oracle::<f64>();
}

/// Same seed → byte-identical predictions across two independent bootstrap
/// fits (the SplitMix64 stream is fully deterministic); a different seed on
/// this tiny forest is allowed to differ (not asserted).
#[test]
fn bootstrap_fit_is_seed_deterministic_f32() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev = upload::<f32>(&mut pool, &xdata());
    let y_idx: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 2, 2];
    let params = RfParams {
        n_trees: 7,
        max_depth: 3,
        n_bins: 8,
        max_features: 1,
        min_samples_split: 2.0,
        min_samples_leaf: 1.0,
        bootstrap: true,
        seed: 1234,
        oob_score: false,
    };
    let xq = vec![0.2, 0.5, 0.85, 0.2, 0.65, 0.95, 0.45, 0.45];
    let xq_dev = upload::<f32>(&mut pool, &xq);

    let RfFitOutcome { model: m1, .. } =
        rf_fit_class::<f32>(&mut pool, &x_dev, (8, 2), &y_idx, 3, &params).expect("fit 1");
    let p1 = rf_predict_proba::<f32>(&mut pool, &m1, &xq_dev, (4, 2))
        .expect("proba 1")
        .to_host(&pool);
    let RfFitOutcome { model: m2, .. } =
        rf_fit_class::<f32>(&mut pool, &x_dev, (8, 2), &y_idx, 3, &params).expect("fit 2");
    let p2 = rf_predict_proba::<f32>(&mut pool, &m2, &xq_dev, (4, 2))
        .expect("proba 2")
        .to_host(&pool);
    assert_eq!(p1.len(), p2.len());
    for (i, (a, b)) in p1.iter().zip(p2.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "proba[{i}] differs across same-seed fits");
    }
}

/// Geometry / hyperparameter guards surface typed errors BEFORE any launch.
#[test]
fn invalid_geometry_is_typed_error_f32() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev = upload::<f32>(&mut pool, &xdata());
    let y_idx: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 2, 2];
    let mut params = params_single_tree();

    // Wrong x geometry.
    assert!(rf_fit_class::<f32>(&mut pool, &x_dev, (7, 2), &y_idx, 3, &params).is_err());
    // y length mismatch.
    assert!(rf_fit_class::<f32>(&mut pool, &x_dev, (8, 2), &y_idx[..7], 3, &params).is_err());
    // Out-of-range class index.
    assert!(rf_fit_class::<f32>(&mut pool, &x_dev, (8, 2), &y_idx, 2, &params).is_err());
    // max_features > d.
    params.max_features = 3;
    assert!(rf_fit_class::<f32>(&mut pool, &x_dev, (8, 2), &y_idx, 3, &params).is_err());
    params.max_features = 2;
    // Depth over the cap.
    params.max_depth = 17;
    assert!(rf_fit_class::<f32>(&mut pool, &x_dev, (8, 2), &y_idx, 3, &params).is_err());
    params.max_depth = 2;

    // Predict-side: feature-count mismatch vs the fitted model.
    let RfFitOutcome { model, .. } =
        rf_fit_class::<f32>(&mut pool, &x_dev, (8, 2), &y_idx, 3, &params).expect("valid fit");
    let xq_bad = upload::<f32>(&mut pool, &[0.1, 0.2, 0.3]);
    assert!(rf_predict_proba::<f32>(&mut pool, &model, &xq_bad, (1, 3)).is_err());
}
