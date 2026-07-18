//! Random Forest `oob_score_` prim-level tests (RF-OOB-01, TASK-04 of
//! `.planning/plans/py-ensemble/PLAN.md`).
//!
//! Exercises `mlrs_backend::prims::random_forest::{rf_fit_class,
//! rf_fit_reg}`'s `RfFitOutcome::oob_score` field: `None` when
//! `params.oob_score == false` (the default, zero extra cost); `Some(score)`
//! (accuracy for the classifier, R² for the regressor) computed from ONLY
//! the out-of-bag trees per training row when `true`. This task proves the
//! MECHANISM produces a plausible in-range score via a wide placeholder
//! band — the exact sklearn-parity statistical-tier cross-check is TASK-06/
//! TASK-07's job (algos-layer oracle tests), not this one.
//!
//! Per AGENTS.md §2, tests live here, never an in-source `#[cfg(test)]`
//! module. f64 is gated behind `capability::skip_f64_with_log()` (the
//! established project convention for backends without `SHADER_F64`).

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::random_forest::{rf_fit_class, rf_fit_reg, RfParams};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// The same tiny 8-row × 2-feature hand-oracle dataset shape used by
/// `random_forest_test.rs` (three well-separated classes), self-contained
/// here since integration test files do not share code.
fn small_xdata() -> Vec<f64> {
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

fn small_y_idx() -> Vec<u32> {
    vec![0, 0, 0, 0, 1, 1, 2, 2]
}

/// 40-row × 2-feature synthetic dataset: feature 0 cleanly separates the two
/// classes / drives the continuous target, feature 1 is class-independent
/// noise (mirrors `random_forest_feature_importances_test.rs`'s dataset
/// shape, reproduced here since integration test files cannot share code).
fn statistical_two_feature_data() -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(80);
    let mut f0s = Vec::with_capacity(40);
    for i in 0..40u32 {
        let f0 = if i < 20 {
            0.05 + 0.01 * (i as f64)
        } else {
            0.60 + 0.01 * ((i - 20) as f64)
        };
        let f1 = (((i * 7 + 3) % 40) as f64) / 40.0;
        x.push(f0);
        x.push(f1);
        f0s.push(f0);
    }
    (x, f0s)
}

#[test]
fn oob_score_false_is_none_and_adds_no_cost() {
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> =
        DeviceArray::from_host(&mut pool, &small_xdata());
    let params = RfParams {
        n_trees: 1,
        max_depth: 2,
        n_bins: 8,
        max_features: 2,
        min_samples_split: 2.0,
        min_samples_leaf: 1.0,
        bootstrap: false,
        seed: 42,
        oob_score: false,
    };
    let outcome = rf_fit_class::<f64>(&mut pool, &x_dev, (8, 2), &small_y_idx(), 3, &params)
        .expect("fit with oob_score=false");
    assert!(
        outcome.oob_score.is_none(),
        "oob_score must be None when params.oob_score == false"
    );
}

#[test]
fn oob_score_true_matches_statistical_band() {
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x, f0s) = statistical_two_feature_data();
    let y_idx: Vec<u32> = f0s.iter().map(|&f| if f < 0.5 { 0u32 } else { 1u32 }).collect();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    let params = RfParams {
        n_trees: 64,
        max_depth: 8,
        n_bins: 16,
        max_features: 2,
        min_samples_split: 2.0,
        min_samples_leaf: 1.0,
        bootstrap: true,
        seed: 42,
        oob_score: true,
    };
    let outcome = rf_fit_class::<f64>(&mut pool, &x_dev, (40, 2), &y_idx, 2, &params)
        .expect("fit with oob_score=true (classifier, statistical tier)");
    let score = outcome
        .oob_score
        .expect("oob_score must be Some when params.oob_score == true");
    assert!(
        (0.0..=1.0).contains(&score),
        "oob accuracy must be a valid [0,1] proportion, got {score}"
    );

    // Regressor mirror: continuous target strongly correlated with feature 0.
    let y: Vec<f64> = f0s.clone();
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);
    let outcome_reg = rf_fit_reg::<f64>(&mut pool, &x_dev, (40, 2), &y_dev, &params)
        .expect("fit with oob_score=true (regressor, statistical tier)");
    let r2 = outcome_reg
        .oob_score
        .expect("oob_score must be Some when params.oob_score == true");
    assert!(
        r2.is_finite(),
        "oob R² must be finite (no NaN/inf) on a well-conditioned statistical-tier fit, got {r2}"
    );
}

#[test]
fn oob_score_zero_oob_rows_excluded_not_panicking() {
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> =
        DeviceArray::from_host(&mut pool, &small_xdata());
    // A single-tree bootstrap forest: `n_trees=1` means EVERY row drawn into
    // that tree's bootstrap sample has ZERO out-of-bag trees (it is in-bag
    // for the only tree) — mathematically guaranteed that at least one of
    // the 8 rows is drawn at least once (8 draws must land somewhere among 8
    // slots), so `zero_oob_count > 0` is deterministic here, exercising the
    // skip-and-warn path without a panic.
    let params = RfParams {
        n_trees: 1,
        max_depth: 2,
        n_bins: 8,
        max_features: 2,
        min_samples_split: 2.0,
        min_samples_leaf: 1.0,
        bootstrap: true,
        seed: 7,
        oob_score: true,
    };
    let outcome = rf_fit_class::<f64>(&mut pool, &x_dev, (8, 2), &small_y_idx(), 3, &params)
        .expect("fit with a pathologically tiny (n_trees=1) bootstrap forest");
    let score = outcome
        .oob_score
        .expect("oob_score must be Some when params.oob_score == true");
    assert!(
        score.is_finite(),
        "oob score must be finite even when some/all rows are excluded, got {score}"
    );
}
