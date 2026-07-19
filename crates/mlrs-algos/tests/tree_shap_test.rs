//! SHAP-01 (Phase 21) — path-dependent TreeSHAP oracle gates.
//!
//! Two tiers:
//!
//! 1. **`ForestInference` path (exact oracle)** — loads a committed
//!    `tree_shap_{classifier,regressor}_seed42.npz` fixture (a fitted
//!    sklearn `RandomForest*`'s per-tree node arrays, INCLUDING
//!    `tree_.weighted_n_node_samples`, plus `shap.TreeExplainer`'s
//!    `shap_values`/`expected_value` on the same fixture's query set —
//!    `scripts/gen_oracle.py::gen_tree_shap`), imports the SAME arrays via
//!    `ForestInference::from_trees`, and asserts `shap_values()` matches the
//!    recorded `shap.TreeExplainer` output ≤1e-5.
//! 2. **Additive efficiency (exact, both tiers)** — `Σ_f φ + expected_value
//!    == prediction` for every query row, both for the FIL-imported forest
//!    (vs `predict_proba`/`predict`) and for a NATIVELY fitted mlrs
//!    `RandomForestClassifier`/`Regressor` (self-consistency gate — no
//!    external oracle for mlrs's own split policy, see the `tree_shap`
//!    module docs).
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_algos::ensemble::forest_inference::{ForestInference, ForestKind, TreeSpec};
use mlrs_algos::ensemble::random_forest_classifier::RandomForestClassifier;
use mlrs_algos::ensemble::random_forest_regressor::RandomForestRegressor;
use mlrs_algos::typestate::{Fit, Predict, PredictProba};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Slice the fixture's flattened per-tree arrays into `TreeSpec`s (the SAME
/// convention `PyForestInference::load_from_arrays` uses).
fn trees_from_case(case: &OracleCase, n_values: usize) -> Vec<TreeSpec> {
    let cl: Vec<i64> = case.expect_f64("children_left").iter().map(|&v| v as i64).collect();
    let cr: Vec<i64> = case.expect_f64("children_right").iter().map(|&v| v as i64).collect();
    let fe: Vec<i64> = case.expect_f64("feature").iter().map(|&v| v as i64).collect();
    let th = case.expect_f64("threshold");
    let va = case.expect_f64("value");
    let nsw = case.expect_f64("node_sample_weight");
    let counts: Vec<usize> = case.expect_f64("node_counts").iter().map(|&v| v as usize).collect();

    let mut trees = Vec::with_capacity(counts.len());
    let mut off = 0usize;
    for &nc in &counts {
        trees.push(TreeSpec {
            children_left: cl[off..off + nc].to_vec(),
            children_right: cr[off..off + nc].to_vec(),
            feature: fe[off..off + nc].to_vec(),
            threshold: th[off..off + nc].to_vec(),
            value: va[off * n_values..(off + nc) * n_values].to_vec(),
            node_sample_weight: nsw[off..off + nc].to_vec(),
        });
        off += nc;
    }
    trees
}

fn assert_close(got: &[f64], expect: &[f64], what: &str) {
    assert_eq!(got.len(), expect.len(), "{what}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
        let tol = 1e-5 + 1e-5 * e.abs();
        assert!(
            (g - e).abs() <= tol,
            "{what}[{i}]: got {g}, expected {e} (diff {})",
            (g - e).abs()
        );
    }
}

// --- tier 1: ForestInference exact oracle -----------------------------------

#[test]
fn fil_shap_matches_shap_treeexplainer_classifier() {
    let case = load_npz(&fixture("tree_shap_classifier_seed42.npz")).expect("fixture loads");
    let n_values = case.expect_f64("n_values")[0] as usize;
    let n_features = case.expect_f64("n_features")[0] as usize;
    let trees = trees_from_case(&case, n_values);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let fil = ForestInference::<f32>::from_trees(
        &mut pool,
        &trees,
        ForestKind::Classifier { n_classes: n_values },
        n_features,
    )
    .expect("valid import");

    let xq = case.expect_f64("Xq");
    let n_query = xq.len() / n_features;
    let (phi, expected_value) = fil.shap_values(&pool, &xq, n_query).expect("cover present");

    assert_close(&expected_value, &case.expect_f64("expected_value"), "expected_value");
    assert_close(&phi, &case.expect_f64("shap_values"), "shap_values");

    // Additive efficiency vs the device predict_proba, exactly.
    let xq_f32: Vec<f32> = xq.iter().map(|&v| v as f32).collect();
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq_f32);
    let proba_dev = fil.predict_proba(&mut pool, &xq_dev, (n_query, n_features)).expect("proba");
    let proba: Vec<f64> = proba_dev.to_host(&pool).iter().map(|&v| v as f64).collect();
    proba_dev.release_into(&mut pool);

    for q in 0..n_query {
        for c in 0..n_values {
            let mut sum = expected_value[c];
            for f in 0..n_features {
                sum += phi[(q * n_features + f) * n_values + c];
            }
            let pred = proba[q * n_values + c];
            assert!(
                (sum - pred).abs() < 1e-4,
                "additive efficiency: row {q} class {c}: Σφ+E[f]={sum} != predict_proba={pred}"
            );
        }
    }
}

#[test]
fn fil_shap_matches_shap_treeexplainer_regressor() {
    let case = load_npz(&fixture("tree_shap_regressor_seed42.npz")).expect("fixture loads");
    let n_features = case.expect_f64("n_features")[0] as usize;
    let trees = trees_from_case(&case, 1);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let fil = ForestInference::<f32>::from_trees(&mut pool, &trees, ForestKind::Regressor, n_features)
        .expect("valid import");

    let xq = case.expect_f64("Xq");
    let n_query = xq.len() / n_features;
    let (phi, expected_value) = fil.shap_values(&pool, &xq, n_query).expect("cover present");

    assert_close(&expected_value, &case.expect_f64("expected_value"), "expected_value");
    assert_close(&phi, &case.expect_f64("shap_values"), "shap_values");

    let xq_f32: Vec<f32> = xq.iter().map(|&v| v as f32).collect();
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq_f32);
    let pred_dev = fil.predict(&mut pool, &xq_dev, (n_query, n_features)).expect("predict");
    let pred: Vec<f64> = pred_dev.to_host(&pool).iter().map(|&v| v as f64).collect();
    pred_dev.release_into(&mut pool);

    for q in 0..n_query {
        let sum: f64 = expected_value[0] + (0..n_features).map(|f| phi[q * n_features + f]).sum::<f64>();
        assert!(
            (sum - pred[q]).abs() < 1e-4,
            "additive efficiency: row {q}: Σφ+E[f]={sum} != predict={}",
            pred[q]
        );
    }
}

// --- tier 2: native mlrs forest, self-consistency (additive efficiency) ----

#[test]
fn native_rf_shap_additive_efficiency_classifier() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (n, d) = (60usize, 4usize);
    let x: Vec<f64> = (0..n * d).map(|k| ((k * 2654435761) % 1000) as f64 / 250.0 - 2.0).collect();
    let y: Vec<f64> = (0..n).map(|i| ((x[i * d] > 0.0) as i64 + (x[i * d + 1] > 0.5) as i64) as f64).collect();
    let x_f: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let y_f: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_f);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y_f);

    let model = RandomForestClassifier::<f32>::builder()
        .n_estimators(6)
        .max_depth(4)
        .build::<f32>()
        .expect("valid build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (n, d))
        .expect("valid fit");

    let q = 12usize;
    let xq: Vec<f64> = (0..q * d).map(|k| (((k + 999) * 2654435761) % 1000) as f64 / 250.0 - 2.0).collect();
    let xq_f: Vec<f32> = xq.iter().map(|&v| v as f32).collect();
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq_f);

    let (phi, expected_value) = model.shap_values(&pool, &x, n, &xq, q);
    let n_classes = model.n_classes();
    assert_eq!(expected_value.len(), n_classes);
    assert_eq!(phi.len(), q * d * n_classes);

    let proba_dev = model.predict_proba(&mut pool, &xq_dev, (q, d)).expect("proba");
    let proba: Vec<f64> = proba_dev.to_host(&pool).iter().map(|&v| v as f64).collect();
    proba_dev.release_into(&mut pool);

    for row in 0..q {
        for c in 0..n_classes {
            let mut sum = expected_value[c];
            for f in 0..d {
                sum += phi[(row * d + f) * n_classes + c];
            }
            let pred = proba[row * n_classes + c];
            assert!(
                (sum - pred).abs() < 1e-4,
                "native additive efficiency: row {row} class {c}: Σφ+E[f]={sum} != predict_proba={pred}"
            );
        }
    }
}

#[test]
fn native_rf_shap_additive_efficiency_regressor() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (n, d) = (50usize, 3usize);
    let x: Vec<f64> = (0..n * d).map(|k| ((k * 2654435761) % 1000) as f64 / 250.0 - 2.0).collect();
    let y: Vec<f64> = (0..n).map(|i| x[i * d] * 1.5 - x[i * d + 2]).collect();
    let x_f: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let y_f: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_f);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y_f);

    let model = RandomForestRegressor::<f32>::builder()
        .n_estimators(5)
        .max_depth(4)
        .build::<f32>()
        .expect("valid build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (n, d))
        .expect("valid fit");

    let q = 10usize;
    let xq: Vec<f64> = (0..q * d).map(|k| (((k + 4321) * 2654435761) % 1000) as f64 / 250.0 - 2.0).collect();
    let xq_f: Vec<f32> = xq.iter().map(|&v| v as f32).collect();
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq_f);

    let (phi, expected_value) = model.shap_values(&pool, &x, n, &xq, q);
    assert_eq!(expected_value.len(), 1);
    assert_eq!(phi.len(), q * d);

    let pred_dev = model.predict(&mut pool, &xq_dev, (q, d)).expect("predict");
    let pred: Vec<f64> = pred_dev.to_host(&pool).iter().map(|&v| v as f64).collect();
    pred_dev.release_into(&mut pool);

    for row in 0..q {
        let sum: f64 = expected_value[0] + (0..d).map(|f| phi[row * d + f]).sum::<f64>();
        assert!(
            (sum - pred[row]).abs() < 1e-4,
            "native additive efficiency: row {row}: Σφ+E[f]={sum} != predict={}",
            pred[row]
        );
    }
}

#[test]
fn shap_values_finite_with_zero_cover_branch() {
    // Regression (SHAP-01): an imported forest with a legitimately
    // zero-`node_sample_weight` interior BRANCH must not produce NaN SHAP
    // values (the recursion divides child covers by the node cover; a
    // zero-cover node is guarded to contribute nothing rather than 0/0 → NaN).
    // Tree: root splits on f0; its right child is a zero-cover interior node
    // (its whole subtree carried no training weight), the left child a leaf.
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let trees = vec![TreeSpec {
        children_left: vec![1, -1, 3, -1, -1],
        children_right: vec![2, -1, 4, -1, -1],
        feature: vec![0, -2, 1, -2, -2],
        threshold: vec![0.0, 0.0, 0.0, 0.0, 0.0],
        value: vec![2.0, 5.0, 7.0, 3.0, 9.0],
        // node2 (right subtree) + node3 + node4 all carry ZERO cover; only the
        // root and its left leaf saw training weight.
        node_sample_weight: vec![10.0, 10.0, 0.0, 0.0, 0.0],
    }];
    let fil = ForestInference::<f32>::from_trees(&mut pool, &trees, ForestKind::Regressor, 2)
        .expect("valid import");

    // Query rows routing into BOTH children, incl. the zero-cover branch.
    let xq: Vec<f64> = vec![-1.0, 0.0, 1.0, 0.5];
    let (phi, ev) = fil.shap_values(&pool, &xq, 2).expect("cover present");
    assert!(ev.iter().all(|v| v.is_finite()), "expected_value must be finite");
    assert!(
        phi.iter().all(|v| v.is_finite()),
        "SHAP values must be finite even when a reached branch has zero cover: {phi:?}"
    );
}
