//! Random Forest `feature_importances_` prim-level oracle tests (RF-IMP-01,
//! TASK-01 of `.planning/plans/py-ensemble/PLAN.md`).
//!
//! Exercises `mlrs_backend::prims::random_forest::{rf_fit_class, rf_fit_reg}`'s
//! `RfFitOutcome::feature_importances` field: a length-`n_features`,
//! non-negative vector summing to `1.0` (or all-zero in the degenerate
//! all-leaf-forest case), where an obviously dominant feature (one that
//! perfectly separates the classes / target) ranks materially above an
//! uncorrelated noise feature — a qualitative ranking assertion, not an
//! exact-match oracle (sklearn/mlrs impurity-importance values have no
//! cross-implementation exact-match guarantee outside the deterministic-tier
//! fixture, per SPEC.md §5 RF-IMP-01).
//!
//! Per AGENTS.md §2, tests live here, never an in-source `#[cfg(test)]`
//! module. f64 is gated behind `capability::skip_f64_with_log()` (the
//! established project convention for backends without `SHADER_F64`).

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::random_forest::{rf_fit_class, rf_fit_reg, RfParams};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// 40-row × 2-feature synthetic dataset: feature 0 is monotonic-per-class
/// (class 0 rows in `[0.05, 0.24]`, class 1 rows in `[0.60, 0.79]` — cleanly
/// separable anywhere in the `(0.24, 0.60)` gap), feature 1 is a
/// class-independent pseudo-random "noise" value in `[0, 1)` (a full-period
/// residue permutation mod 40, via `gcd(7, 40) == 1`).
///
/// Returns `(x flattened row-major n*2, f0 values)`; callers derive the
/// classifier label / regressor target from `f0` directly.
fn synthetic_two_feature_data() -> (Vec<f64>, Vec<f64>) {
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

fn params_dominant_feature() -> RfParams {
    RfParams {
        n_trees: 4,
        max_depth: 3,
        n_bins: 8,
        max_features: 2,
        min_samples_split: 2.0,
        min_samples_leaf: 1.0,
        bootstrap: false,
        seed: 42,
        oob_score: false,
    }
}

#[test]
fn feature_importances_dominant_feature_ranks_highest() {
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x, f0s) = synthetic_two_feature_data();
    let y_idx: Vec<u32> = f0s.iter().map(|&f| if f < 0.5 { 0u32 } else { 1u32 }).collect();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    let outcome = rf_fit_class::<f64>(
        &mut pool,
        &x_dev,
        (40, 2),
        &y_idx,
        2,
        &params_dominant_feature(),
    )
    .expect("fit dominant-feature classifier");

    let imp = &outcome.feature_importances;
    assert_eq!(imp.len(), 2, "feature_importances length must equal n_features");
    let sum: f64 = imp.iter().sum();
    assert!(
        (sum - 1.0).abs() <= 1e-9,
        "feature_importances must sum to 1.0, got {sum}"
    );
    assert!(
        imp[0] > imp[1],
        "dominant feature 0 must rank above noise feature 1: got {imp:?}"
    );
}

#[test]
fn feature_importances_regressor_dominant_feature() {
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x, f0s) = synthetic_two_feature_data();
    // Continuous target strongly (here, exactly) correlated with feature 0
    // only — feature 1 is the same class-independent noise column.
    let y: Vec<f64> = f0s.clone();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);

    let outcome = rf_fit_reg::<f64>(&mut pool, &x_dev, (40, 2), &y_dev, &params_dominant_feature())
        .expect("fit dominant-feature regressor");

    let imp = &outcome.feature_importances;
    assert_eq!(imp.len(), 2, "feature_importances length must equal n_features");
    let sum: f64 = imp.iter().sum();
    assert!(
        (sum - 1.0).abs() <= 1e-9,
        "feature_importances must sum to 1.0, got {sum}"
    );
    assert!(
        imp[0] > imp[1],
        "dominant feature 0 must rank above noise feature 1: got {imp:?}"
    );
}

#[test]
fn feature_importances_all_leaf_forest_is_all_zero() {
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x, f0s) = synthetic_two_feature_data();
    let y_idx: Vec<u32> = f0s.iter().map(|&f| if f < 0.5 { 0u32 } else { 1u32 }).collect();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    // `min_samples_split` far above the dataset size forces `tot < min_split`
    // at every node (starting at the root), so EVERY node in EVERY tree is
    // forced to a leaf regardless of `max_depth` — the degenerate
    // all-leaf-forest case (the divide-by-zero guard on the normalization
    // step).
    let params = RfParams {
        n_trees: 2,
        max_depth: 1,
        n_bins: 8,
        max_features: 2,
        min_samples_split: 1000.0,
        min_samples_leaf: 1.0,
        bootstrap: false,
        seed: 42,
        oob_score: false,
    };
    let outcome = rf_fit_class::<f64>(&mut pool, &x_dev, (40, 2), &y_idx, 2, &params)
        .expect("fit all-leaf-forced classifier");

    let imp = &outcome.feature_importances;
    assert_eq!(imp.len(), 2, "feature_importances length must equal n_features");
    assert!(
        imp.iter().all(|&v| v == 0.0),
        "all-leaf forest must yield all-zero feature_importances, got {imp:?}"
    );
}

/// 60-row × 5-feature dataset for exercising the CROSS-TREE aggregation of
/// `feature_importances_`. Feature 0 separates the classes strongly, feature 1
/// moderately, features 2-4 are class-independent noise. Combined with
/// `bootstrap=true` (the default), each tree sees a DIFFERENT resample, so its
/// root weighted impurity — and hence its total impurity decrease `S_t` —
/// genuinely differs from the other trees' (unlike a `bootstrap=false`
/// fully-separable forest, where every tree drives impurity to zero and all
/// `S_t` collapse to the identical root impurity, making the two aggregation
/// schemes coincide). Differing `S_t` is exactly the regime where sklearn's
/// per-tree-normalize-then-average scheme diverges from a naive global
/// `Σd/ΣS` normalization.
fn synthetic_five_feature_data() -> Vec<f64> {
    let mut x = Vec::with_capacity(60 * 5);
    for i in 0..60u32 {
        let (f0, f1) = if i < 30 {
            (0.01 * (i as f64), 0.20 + 0.01 * (i as f64))
        } else {
            (0.70 + 0.01 * ((i - 30) as f64), 0.50 + 0.01 * ((i - 30) as f64))
        };
        let f2 = (((i * 7 + 3) % 60) as f64) / 60.0;
        let f3 = (((i * 13 + 5) % 60) as f64) / 60.0;
        let f4 = (((i * 29 + 11) % 60) as f64) / 60.0;
        x.extend_from_slice(&[f0, f1, f2, f3, f4]);
    }
    x
}

/// Recompute feature importances from a fitted model's per-node arrays two
/// independent ways: `per_tree` = sklearn's mean-of-per-tree-normalized (then
/// renormalized) and `global` = the naive single global normalization `Σd/ΣS`.
/// Returns `(per_tree, global)`.
#[allow(clippy::too_many_arguments)]
fn recompute_importances(
    split_feature: &[u32],
    is_leaf: &[u32],
    node_decrease: &[f64],
    n_trees: usize,
    total_nodes: usize,
    n_features: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut per_tree = vec![0f64; n_features];
    let mut n_contributing = 0usize;
    let mut global = vec![0f64; n_features];
    for tr in 0..n_trees {
        let base = tr * total_nodes;
        let mut imp_t = vec![0f64; n_features];
        for node in 0..total_nodes {
            let i = base + node;
            if is_leaf[i] == 0 {
                let f = split_feature[i] as usize;
                imp_t[f] += node_decrease[i];
                global[f] += node_decrease[i];
            }
        }
        let s_t: f64 = imp_t.iter().sum();
        if s_t > 0.0 {
            for (acc, v) in per_tree.iter_mut().zip(imp_t.iter()) {
                *acc += *v / s_t;
            }
            n_contributing += 1;
        }
    }
    if n_contributing > 0 {
        for v in per_tree.iter_mut() {
            *v /= n_contributing as f64;
        }
        let s: f64 = per_tree.iter().sum();
        if s > 0.0 {
            for v in per_tree.iter_mut() {
                *v /= s;
            }
        }
    }
    let gsum: f64 = global.iter().sum();
    if gsum > 0.0 {
        for v in global.iter_mut() {
            *v /= gsum;
        }
    }
    (per_tree, global)
}

/// RF-IMP-01 aggregation lock: `feature_importances_` must match scikit-learn's
/// per-tree-normalize-then-average scheme (`mean_t(d_{t,f}/S_t)`), NOT a naive
/// global `Σ_t d_{t,f} / Σ_t S_t`. The two coincide only when every tree's
/// total decrease `S_t` is equal — true in the deterministic-tier fixture
/// (bit-identical trees) but FALSE for real (bootstrap / feature-subsampled)
/// forests, which is why this test deliberately builds a forest whose trees
/// differ. Guards against a regression to the old global-normalize behavior.
#[test]
fn feature_importances_uses_per_tree_normalization_not_global() {
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = synthetic_five_feature_data();
    let y_idx: Vec<u32> = (0..60u32).map(|i| if i < 30 { 0 } else { 1 }).collect();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    // bootstrap=true (different resample per tree) + max_features=2 (< 5) =>
    // the trees genuinely differ AND, crucially, their per-tree total
    // decreases S_t differ (different root weighted impurity per resample),
    // which is what makes the per-tree-average and global aggregations
    // diverge. Fully deterministic given `seed`.
    let params = RfParams {
        n_trees: 8,
        max_depth: 4,
        n_bins: 8,
        max_features: 2,
        min_samples_split: 2.0,
        min_samples_leaf: 1.0,
        bootstrap: true,
        seed: 42,
        oob_score: false,
    };
    let outcome = rf_fit_class::<f64>(&mut pool, &x_dev, (60, 5), &y_idx, 2, &params)
        .expect("fit 5-feature classifier");

    let got = &outcome.feature_importances;
    let model = &outcome.model;
    let sf = model.split_feature_host(&pool);
    let il = model.is_leaf_host(&pool);
    let nd = model.node_decrease_host(&pool);
    let (per_tree, global) = recompute_importances(
        &sf,
        &il,
        &nd,
        model.n_trees(),
        model.total_nodes(),
        model.n_features(),
    );

    // (1) The prim's own feature_importances MUST equal the per-tree scheme.
    for f in 0..5 {
        assert!(
            (got[f] - per_tree[f]).abs() <= 1e-9,
            "feature_importances[{f}] must match sklearn's per-tree-normalized \
             scheme: got {got:?}, expected {per_tree:?}"
        );
    }

    // (2) Regression guard: this config MUST actually exercise the
    // per-tree-vs-global distinction (else assertion (1) would pass even for a
    // naive global implementation). If this ever fails, the trees no longer
    // have differing S_t here — choose data/params that make them differ.
    let max_gap = (0..5)
        .map(|f| (per_tree[f] - global[f]).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_gap > 1e-6,
        "test data must make per-tree and global aggregation differ (max gap \
         {max_gap:e}); it no longer exercises the distinction"
    );
}
