//! FIL-01 (Phase 20) — batched forest-inference gates.
//!
//! Two tiers:
//!
//! 1. **Device-vs-host-walk equality (the Phase-20 goal):** fit NATIVE mlrs
//!    forests (RandomForest cls/reg, HistGradientBoosting cls/reg), read the
//!    complete-layout node arrays back through the debug accessors, and
//!    replay the documented traversal contract on the host
//!    (`x < threshold → 2i+1`, bounded `max_depth` walk, `is_leaf` stops
//!    advancement; RF = mean of reached-leaf values, HGB = baseline + Σ
//!    shrunk leaves + link). The device predictions must match the host walk
//!    to ≤1e-6 relative (identical routing; only the reduction association
//!    may differ in the last bits).
//!
//! 2. **`ForestInference` import (cuML FIL parity):** hand-built
//!    sklearn-layout trees (explicit children, `x <= t → left`) imported via
//!    `from_trees` must route EXACTLY per sklearn's comparator — including a
//!    query landing exactly ON a threshold (the `next_up` bump crux) — and
//!    validation must reject an empty forest, an over-deep tree, and
//!    kind-mismatched predict calls with typed errors.
//!
//! f64 cases carry the `skip_f64_with_log` capability gate. Per AGENTS.md §2
//! tests live in `crates/mlrs-algos/tests/`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::ensemble::forest_inference::{ForestInference, ForestKind, TreeSpec};
use mlrs_algos::ensemble::random_forest_classifier::RandomForestClassifier;
use mlrs_algos::ensemble::random_forest_regressor::RandomForestRegressor;
use mlrs_algos::error::AlgoError;
use mlrs_algos::typestate::{Fit, Predict, PredictProba};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

fn to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!(),
    }
}

fn from_f64<F: Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!(),
    }
}

/// The documented complete-layout walk: from the root, for exactly
/// `max_depth` steps, advance `x < thr → 2c+1` else `2c+2` while interior.
fn walk_leaf(
    x_row: &[f64],
    split_feature: &[u32],
    threshold: &[f64],
    is_leaf: &[u32],
    tree_base: usize,
    max_depth: usize,
) -> usize {
    let mut cur = 0usize;
    for _ in 0..max_depth {
        if is_leaf[tree_base + cur] == 0 {
            let f = split_feature[tree_base + cur] as usize;
            let t = threshold[tree_base + cur];
            cur = if x_row[f] < t { 2 * cur + 1 } else { 2 * cur + 2 };
        }
    }
    cur
}

/// Deterministic quasi-random design (no fixture needed — the host walk IS
/// the oracle here).
fn design(n: usize, d: usize, salt: usize) -> Vec<f64> {
    (0..n * d)
        .map(|k| (((k + salt) * 2654435761) % 1000) as f64 / 250.0 - 2.0)
        .collect()
}

// --- tier 1: native-forest device-vs-host-walk ------------------------------

fn rf_class_walk_case<F>()
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (n, d) = (60usize, 4usize);
    let x = design(n, d, 0);
    let y: Vec<f64> = (0..n).map(|i| ((x[i * d] > 0.0) as i64 + (x[i * d + 1] > 0.5) as i64) as f64).collect();
    let x_f: Vec<F> = x.iter().map(|&v| from_f64::<F>(v)).collect();
    let y_f: Vec<F> = y.iter().map(|&v| from_f64::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_f);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_f);

    let model = RandomForestClassifier::<F>::builder()
        .n_estimators(7)
        .max_depth(4)
        .build::<F>()
        .expect("valid build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (n, d))
        .expect("valid fit");

    // Device predict_proba on a fresh query set.
    let (q, _) = (25usize, d);
    let xq = design(q, d, 999);
    let xq_f: Vec<F> = xq.iter().map(|&v| from_f64::<F>(v)).collect();
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq_f);
    let proba_dev = model.predict_proba(&mut pool, &xq_dev, (q, d)).expect("predict_proba");
    let proba: Vec<f64> = proba_dev.to_host(&pool).iter().map(|&v| to_f64(v)).collect();
    proba_dev.release_into(&mut pool);

    // Host reference walk over the model arrays.
    let m = model.model();
    let nc = m.n_values();
    let t = m.n_trees();
    let total = m.total_nodes();
    let sf = m.split_feature_host(&pool);
    let th: Vec<f64> = m.threshold_host(&pool).iter().map(|&v| to_f64(v)).collect();
    let il = m.is_leaf_host(&pool);
    let ld: Vec<f64> = m.leaf_dist_host(&pool).iter().map(|&v| to_f64(v)).collect();
    // Recover max_depth from the layout (total = 2^(md+1) − 1).
    let md = (usize::BITS - (total + 1).leading_zeros() - 1) as usize - 1;

    for r in 0..q {
        for c in 0..nc {
            let mut acc = 0.0f64;
            for ti in 0..t {
                let leaf = walk_leaf(&xq[r * d..(r + 1) * d], &sf, &th, &il, ti * total, md);
                acc += ld[(ti * total + leaf) * nc + c];
            }
            let expect = acc / t as f64;
            let got = proba[r * nc + c];
            assert!(
                (got - expect).abs() <= 1e-6 * expect.abs().max(1.0),
                "rf cls device proba[{r},{c}] {got} != host walk {expect}"
            );
        }
    }
}

fn rf_reg_walk_case<F>()
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (n, d) = (50usize, 3usize);
    let x = design(n, d, 7);
    let y: Vec<f64> = (0..n).map(|i| x[i * d] * 1.5 - x[i * d + 2]).collect();
    let x_f: Vec<F> = x.iter().map(|&v| from_f64::<F>(v)).collect();
    let y_f: Vec<F> = y.iter().map(|&v| from_f64::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_f);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_f);

    let model = RandomForestRegressor::<F>::builder()
        .n_estimators(6)
        .max_depth(4)
        .build::<F>()
        .expect("valid build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (n, d))
        .expect("valid fit");

    let q = 20usize;
    let xq = design(q, d, 4321);
    let xq_f: Vec<F> = xq.iter().map(|&v| from_f64::<F>(v)).collect();
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq_f);
    let pred_dev = model.predict(&mut pool, &xq_dev, (q, d)).expect("predict");
    let pred: Vec<f64> = pred_dev.to_host(&pool).iter().map(|&v| to_f64(v)).collect();
    pred_dev.release_into(&mut pool);

    let m = model.model();
    let t = m.n_trees();
    let total = m.total_nodes();
    let sf = m.split_feature_host(&pool);
    let th: Vec<f64> = m.threshold_host(&pool).iter().map(|&v| to_f64(v)).collect();
    let il = m.is_leaf_host(&pool);
    let ld: Vec<f64> = m.leaf_dist_host(&pool).iter().map(|&v| to_f64(v)).collect();
    let md = (usize::BITS - (total + 1).leading_zeros() - 1) as usize - 1;

    for r in 0..q {
        let mut acc = 0.0f64;
        for ti in 0..t {
            let leaf = walk_leaf(&xq[r * d..(r + 1) * d], &sf, &th, &il, ti * total, md);
            acc += ld[ti * total + leaf];
        }
        let expect = acc / t as f64;
        let got = pred[r];
        assert!(
            (got - expect).abs() <= 1e-6 * expect.abs().max(1.0),
            "rf reg device pred[{r}] {got} != host walk {expect}"
        );
    }
}

#[test]
fn rf_device_matches_host_walk_f32() {
    rf_class_walk_case::<f32>();
    rf_reg_walk_case::<f32>();
}

#[test]
fn rf_device_matches_host_walk_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    rf_class_walk_case::<f64>();
    rf_reg_walk_case::<f64>();
}

// --- tier 2: ForestInference import -----------------------------------------

/// Two hand-built depth-2 sklearn-layout trees over 2 features.
fn tiny_classifier_trees() -> Vec<TreeSpec> {
    // Tree 0:            node0 (f0 <= 1.5)
    //                   /                \
    //          node1 (leaf [3,1])   node2 (f1 <= -0.5)
    //                               /              \
    //                        node3 [0,2]      node4 [2,2]
    let t0 = TreeSpec {
        children_left: vec![1, -1, 3, -1, -1],
        children_right: vec![2, -1, 4, -1, -1],
        feature: vec![0, -2, 1, -2, -2],
        threshold: vec![1.5, -2.0, -0.5, -2.0, -2.0],
        value: vec![5.0, 5.0, 3.0, 1.0, 2.0, 4.0, 0.0, 2.0, 2.0, 2.0],
        // Cover = subtree leaf-count sums: node0=10 (all leaves), node1=4
        // (leaf [3,1]), node2=6 (node3+node4), node3=2 (leaf [0,2]), node4=4
        // (leaf [2,2]).
        node_sample_weight: vec![10.0, 4.0, 6.0, 2.0, 4.0],
    };
    // Tree 1: a single split on f1 <= 0.0.
    let t1 = TreeSpec {
        children_left: vec![1, -1, -1],
        children_right: vec![2, -1, -1],
        feature: vec![1, -2, -2],
        threshold: vec![0.0, -2.0, -2.0],
        value: vec![4.0, 4.0, 4.0, 0.0, 0.0, 4.0],
        node_sample_weight: vec![8.0, 4.0, 4.0],
    };
    vec![t0, t1]
}

#[test]
fn forest_inference_classifier_exact_routing() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let fil = ForestInference::<f32>::from_trees(
        &mut pool,
        &tiny_classifier_trees(),
        ForestKind::Classifier { n_classes: 2 },
        2,
    )
    .expect("valid import");
    assert_eq!(fil.n_trees(), 2);

    // Query rows chosen to hit every routing case — INCLUDING x exactly ON a
    // threshold (sklearn `<=` must route LEFT; the next_up bump crux).
    let xq: Vec<f32> = vec![
        1.5, 0.0, // on BOTH thresholds → t0 left leaf [3,1]→[.75,.25]; t1 left [1,0]
        2.0, -0.5, // t0 right, f1 <= -0.5 → node3 [0,1]; t1 left [1,0]
        2.0, 0.5, // t0 right-right [.5,.5]; t1 right [0,1]
    ];
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq);
    let proba_dev = fil.predict_proba(&mut pool, &xq_dev, (3, 2)).expect("proba");
    let proba: Vec<f32> = proba_dev.to_host(&pool);
    proba_dev.release_into(&mut pool);

    let expect = [
        (0.75 + 1.0) / 2.0,
        (0.25 + 0.0) / 2.0, // row 0
        (0.0 + 1.0) / 2.0,
        (1.0 + 0.0) / 2.0, // row 1
        (0.5 + 0.0) / 2.0,
        (0.5 + 1.0) / 2.0, // row 2
    ];
    for (i, (&g, &e)) in proba.iter().zip(expect.iter()).enumerate() {
        assert!(
            (g as f64 - e).abs() < 1e-6,
            "proba[{i}] {g} != {e} (threshold-boundary routing must be sklearn's <=)"
        );
    }

    let labels = fil
        .predict_class_indices(&mut pool, &xq_dev, (3, 2))
        .expect("labels");
    // Row 1 is an exact 0.5/0.5 tie → lowest class index wins (sklearn rule).
    assert_eq!(labels, vec![0, 0, 1]);
}

#[test]
fn forest_inference_regressor() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let trees = vec![
        TreeSpec {
            children_left: vec![1, -1, -1],
            children_right: vec![2, -1, -1],
            feature: vec![0, -2, -2],
            threshold: vec![0.0, 0.0, 0.0],
            value: vec![0.0, -3.0, 5.0],
            node_sample_weight: vec![10.0, 4.0, 6.0],
        },
        TreeSpec {
            children_left: vec![-1],
            children_right: vec![-1],
            feature: vec![-2],
            threshold: vec![0.0],
            value: vec![1.0],
            node_sample_weight: vec![5.0],
        },
    ];
    let fil = ForestInference::<f32>::from_trees(&mut pool, &trees, ForestKind::Regressor, 1)
        .expect("valid import");
    let xq: Vec<f32> = vec![-1.0, 0.0, 2.0];
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq);
    let pred_dev = fil.predict(&mut pool, &xq_dev, (3, 1)).expect("predict");
    let pred: Vec<f32> = pred_dev.to_host(&pool);
    pred_dev.release_into(&mut pool);
    // x=0.0 is ON the threshold → LEFT (-3.0). Forest mean with the constant 1.0 tree.
    assert_eq!(pred, vec![(-3.0 + 1.0) / 2.0, (-3.0 + 1.0) / 2.0, (5.0 + 1.0) / 2.0]);
}

#[test]
fn forest_inference_validation() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // Empty forest → typed error.
    assert!(matches!(
        ForestInference::<f32>::from_trees(&mut pool, &[], ForestKind::Regressor, 1),
        Err(AlgoError::Build(_))
    ));

    // Depth over the cap (a 18-deep chain) → typed error.
    let deep = {
        let n = 19usize; // chain of 18 interior + terminal leaf
        let mut cl = vec![-1i64; 2 * n];
        let mut cr = vec![-1i64; 2 * n];
        let mut fe = vec![-2i64; 2 * n];
        let th = vec![0.0f64; 2 * n];
        let va = vec![0.0f64; 2 * n];
        for i in 0..n - 1 {
            cl[i] = (n + i) as i64; // left = a leaf
            cr[i] = (i + 1) as i64; // right = next chain node
            fe[i] = 0;
        }
        TreeSpec {
            children_left: cl,
            children_right: cr,
            feature: fe,
            threshold: th,
            value: va,
            node_sample_weight: Vec::new(),
        }
    };
    assert!(matches!(
        ForestInference::<f32>::from_trees(&mut pool, &[deep], ForestKind::Regressor, 1),
        Err(AlgoError::Build(_))
    ));

    // Kind-mismatched predict calls → typed Unsupported.
    let fil = ForestInference::<f32>::from_trees(
        &mut pool,
        &tiny_classifier_trees(),
        ForestKind::Classifier { n_classes: 2 },
        2,
    )
    .expect("valid import");
    let xq: Vec<f32> = vec![0.0, 0.0];
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq);
    assert!(matches!(
        fil.predict(&mut pool, &xq_dev, (1, 2)),
        Err(AlgoError::Unsupported { .. })
    ));
}
