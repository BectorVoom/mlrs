//! ENSEMBLE-PERSIST (prototype) — safetensors save/load round-trips for the four
//! tree ensembles: `RandomForestClassifier`, `RandomForestRegressor`,
//! `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`.
//!
//! (`StackingRegressor` has no Rust estimator to persist — mlrs keeps only its
//! structural helpers in Rust and composes the estimator itself on the Python
//! side, so there is no fitted state here to save.)
//!
//! A tree ensemble is where model formats usually get complicated, and this one
//! does not: mlrs stores every tree in the COMPLETE layout its traversal kernel
//! indexes directly, so the file is four flat arrays of equal length. What that
//! buys is a load with no expansion pass; what it costs is the unused slots of a
//! shallow tree. `the_file_is_a_complete_layout` measures the cost so it is a
//! known quantity rather than a surprise.
//!
//! Two gates carry the real weight here, because this is the family whose
//! traversal kernel indexes two different axes off STORED values:
//!
//!   - `a_non_complete_node_count_is_rejected` — the walk goes `2i+1`/`2i+2`
//!     with no bound beyond the depth counter, so it is in range only for a
//!     `2^(d+1) − 1` node count. A plausible-looking 100 would make every walk
//!     that reached the last level read past the end of the node table.
//!   - `an_out_of_range_split_feature_is_rejected` — each node reads
//!     `split_feature[node]` to index the QUERY row, so an out-of-range index
//!     reads past the end of a sample on the first prediction.
//!
//! Neither is visible in any single tensor; both are established before the
//! model exists. The file is untrusted input (T-04-01-01).
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::ensemble::ensemble_persist::{
    AlignedBytes, EnsembleFile, EnsembleWriter, LoadModel, PersistError, SaveModel, TensorRef,
};
use mlrs_algos::ensemble::hist_gradient_boosting_classifier::HistGradientBoostingClassifier;
use mlrs_algos::ensemble::hist_gradient_boosting_regressor::HistGradientBoostingRegressor;
use mlrs_algos::ensemble::random_forest_classifier::RandomForestClassifier;
use mlrs_algos::ensemble::random_forest_regressor::RandomForestRegressor;
use mlrs_algos::ensemble::MaxFeatures;
use mlrs_algos::preprocessing::MaxAbsScaler;
use mlrs_algos::typestate::{Fit, Fitted, Predict, PredictLabels, PredictProba};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

const N_SAMPLES: usize = 32;
const N_FEATURES: usize = 4;
const MAX_DEPTH: usize = 3;

fn fixture<F: Pod>() -> Vec<F> {
    (0..N_SAMPLES * N_FEATURES)
        .map(|i| {
            let v = ((i * 31) % 71) as f64 / 17.0 - 2.0;
            mlrs_core::f64_to_host::<F>(v)
        })
        .collect()
}

/// Three classes, assigned so the split structure is learnable — a forest fitted
/// on noise would produce degenerate trees whose round-trip proves less.
fn labels<F: Pod>() -> Vec<F> {
    let x = fixture::<f64>();
    (0..N_SAMPLES)
        .map(|i| {
            let v = x[i * N_FEATURES];
            let c = if v < -0.7 {
                0.0
            } else if v < 0.7 {
                1.0
            } else {
                2.0
            };
            mlrs_core::f64_to_host::<F>(c)
        })
        .collect()
}

fn targets<F: Pod>() -> Vec<F> {
    let x = fixture::<f64>();
    (0..N_SAMPLES)
        .map(|i| mlrs_core::f64_to_host::<F>(2.0 * x[i * N_FEATURES] - x[i * N_FEATURES + 1] + 0.5))
        .collect()
}

fn pool() -> BufferPool<ActiveRuntime> {
    BufferPool::new(runtime::active_client())
}

fn upload<F: Float + CubeElement + Pod>(
    p: &mut BufferPool<ActiveRuntime>,
) -> DeviceArray<ActiveRuntime, F> {
    DeviceArray::from_host(p, &fixture::<F>())
}

fn fit_rf_clf<F>(p: &mut BufferPool<ActiveRuntime>) -> RandomForestClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    let y: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(p, &labels::<F>());
    RandomForestClassifier::<F>::builder()
        .n_estimators(4)
        .max_depth(MAX_DEPTH)
        .max_features(MaxFeatures::Log2)
        .seed(7)
        .build::<F>()
        .expect("RandomForestClassifier builds")
        .fit(p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
        .expect("RandomForestClassifier fits the fixture")
}

fn fit_rf_reg<F>(p: &mut BufferPool<ActiveRuntime>) -> RandomForestRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    let y: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(p, &targets::<F>());
    RandomForestRegressor::<F>::builder()
        .n_estimators(4)
        .max_depth(MAX_DEPTH)
        .seed(7)
        .build::<F>()
        .expect("RandomForestRegressor builds")
        .fit(p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
        .expect("RandomForestRegressor fits the fixture")
}

fn fit_hgb_clf<F>(p: &mut BufferPool<ActiveRuntime>) -> HistGradientBoostingClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    let y: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(p, &labels::<F>());
    HistGradientBoostingClassifier::<F>::builder()
        .max_iter(3)
        .max_depth(MAX_DEPTH)
        .learning_rate(0.2)
        .build::<F>()
        .expect("HistGradientBoostingClassifier builds")
        .fit(p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
        .expect("HistGradientBoostingClassifier fits the fixture")
}

fn fit_hgb_reg<F>(p: &mut BufferPool<ActiveRuntime>) -> HistGradientBoostingRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    let y: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(p, &targets::<F>());
    HistGradientBoostingRegressor::<F>::builder()
        .max_iter(3)
        .max_depth(MAX_DEPTH)
        .learning_rate(0.2)
        .build::<F>()
        .expect("HistGradientBoostingRegressor builds")
        .fit(p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
        .expect("HistGradientBoostingRegressor fits the fixture")
}

// ---------------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------------

#[test]
fn random_forest_classifier_roundtrip_preserves_predictions() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("rfc.safetensors");
    let mut p = pool();

    let fitted = fit_rf_clf::<f32>(&mut p);
    let x = upload::<f32>(&mut p);
    let labels_before = fitted
        .predict_labels(&mut p, &x, (N_SAMPLES, N_FEATURES))
        .expect("predict_labels succeeds")
        .to_host(&p);
    let x = upload::<f32>(&mut p);
    let proba_before = fitted
        .predict_proba(&mut p, &x, (N_SAMPLES, N_FEATURES))
        .expect("predict_proba succeeds")
        .to_host(&p);

    fitted.save(&p, &path).expect("save succeeds");
    let loaded: RandomForestClassifier<f32, Fitted> =
        RandomForestClassifier::load(&mut p, &path).expect("load succeeds");

    assert_eq!(
        loaded.classes(),
        fitted.classes(),
        "classes_ must round-trip"
    );
    assert_eq!(
        loaded.feature_importances(),
        fitted.feature_importances(),
        "feature_importances_"
    );
    assert_eq!(loaded.n_features(), fitted.n_features(), "n_features_in_");

    // The node tables have no public accessor, so the predictions ARE the
    // comparison — and `predict_proba` is the stronger of the two, since it
    // exercises every leaf's whole distribution rather than only its argmax.
    let x = upload::<f32>(&mut p);
    assert_eq!(
        loaded
            .predict_labels(&mut p, &x, (N_SAMPLES, N_FEATURES))
            .expect("predict_labels succeeds")
            .to_host(&p),
        labels_before,
        "the reloaded forest must classify identically"
    );
    let x = upload::<f32>(&mut p);
    assert_eq!(
        loaded
            .predict_proba(&mut p, &x, (N_SAMPLES, N_FEATURES))
            .expect("predict_proba succeeds")
            .to_host(&p),
        proba_before,
        "and reproduce every leaf distribution exactly"
    );
}

#[test]
fn random_forest_regressor_roundtrip_preserves_predictions() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("rfr.safetensors");
    let mut p = pool();

    let fitted = fit_rf_reg::<f32>(&mut p);
    let x = upload::<f32>(&mut p);
    let before = fitted
        .predict(&mut p, &x, (N_SAMPLES, N_FEATURES))
        .expect("predict succeeds")
        .to_host(&p);

    fitted.save(&p, &path).expect("save succeeds");
    let loaded: RandomForestRegressor<f32, Fitted> =
        RandomForestRegressor::load(&mut p, &path).expect("load succeeds");

    assert_eq!(
        loaded.feature_importances(),
        fitted.feature_importances(),
        "feature_importances_"
    );
    let x = upload::<f32>(&mut p);
    assert_eq!(
        loaded
            .predict(&mut p, &x, (N_SAMPLES, N_FEATURES))
            .expect("predict succeeds")
            .to_host(&p),
        before,
        "the reloaded forest must predict identically"
    );
}

#[test]
fn hist_gradient_boosting_roundtrip_preserves_predictions() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();

    // The classifier here is MULTICLASS, which is the case that exercises the
    // `n_iters · k` tree axis — a binary booster has `k == 1` and would leave
    // the stride untested.
    let clf_path = dir.path().join("hgbc.safetensors");
    let clf = fit_hgb_clf::<f32>(&mut p);
    let x = upload::<f32>(&mut p);
    let clf_before = clf
        .predict_proba(&mut p, &x, (N_SAMPLES, N_FEATURES))
        .expect("predict_proba succeeds")
        .to_host(&p);
    clf.save(&p, &clf_path).expect("save succeeds");
    let loaded: HistGradientBoostingClassifier<f32, Fitted> =
        HistGradientBoostingClassifier::load(&mut p, &clf_path).expect("load succeeds");
    assert_eq!(loaded.classes(), clf.classes(), "classes_");
    let x = upload::<f32>(&mut p);
    assert_eq!(
        loaded
            .predict_proba(&mut p, &x, (N_SAMPLES, N_FEATURES))
            .expect("predict_proba succeeds")
            .to_host(&p),
        clf_before,
        "the reloaded booster must score identically across all k columns"
    );

    let reg_path = dir.path().join("hgbr.safetensors");
    let reg = fit_hgb_reg::<f32>(&mut p);
    let x = upload::<f32>(&mut p);
    let reg_before = reg
        .predict(&mut p, &x, (N_SAMPLES, N_FEATURES))
        .expect("predict succeeds")
        .to_host(&p);
    reg.save(&p, &reg_path).expect("save succeeds");
    let loaded: HistGradientBoostingRegressor<f32, Fitted> =
        HistGradientBoostingRegressor::load(&mut p, &reg_path).expect("load succeeds");
    let x = upload::<f32>(&mut p);
    assert_eq!(
        loaded
            .predict(&mut p, &x, (N_SAMPLES, N_FEATURES))
            .expect("predict succeeds")
            .to_host(&p),
        reg_before,
        "the reloaded booster must predict identically"
    );
}

#[test]
fn non_default_params_roundtrip() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");
    let mut p = pool();

    // save → load → save is byte-stable, which gates the nine private
    // hyperparameters at once. `max_features` is the interesting one: it is
    // enum-shaped with a payload variant, has no public accessor, and is
    // invisible in every fitted tensor.
    let fitted = fit_rf_clf::<f32>(&mut p);
    fitted.save(&p, &first).expect("save succeeds");
    let loaded: RandomForestClassifier<f32, Fitted> =
        RandomForestClassifier::load(&mut p, &first).expect("load succeeds");
    loaded.save(&p, &second).expect("re-save succeeds");
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "save→load→save must be byte-stable, or a hyperparameter was dropped"
    );
}

#[test]
fn every_max_features_policy_roundtrips() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();

    // `MaxFeatures` encodes four variants into one string, three of them
    // policies and one an explicit count. `All` renders as `"1.0"` and a count
    // as a bare integer, which is what keeps them from colliding — so all four
    // have to survive.
    for (i, policy) in [
        MaxFeatures::Sqrt,
        MaxFeatures::Log2,
        MaxFeatures::All,
        MaxFeatures::Value(2),
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.path().join(format!("mf{i}.safetensors"));
        let x = upload::<f32>(&mut p);
        let y: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &labels::<f32>());
        let fitted = RandomForestClassifier::<f32>::builder()
            .n_estimators(2)
            .max_depth(MAX_DEPTH)
            .max_features(policy)
            .seed(7)
            .build::<f32>()
            .expect("builds")
            .fit(&mut p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
            .expect("fits");
        fitted.save(&p, &path).expect("save succeeds");

        let raw = AlignedBytes::read(&path).expect("read succeeds");
        let file = EnsembleFile::parse(&raw, "random_forest_classifier").expect("parse succeeds");
        assert_eq!(
            file.scalar_str("param:max_features")
                .expect("the key is present"),
            policy.name(),
            "{policy:?} must be stored as its own spelling"
        );

        // And it loads back as the same variant, which the byte-stable re-save
        // confirms without needing an accessor.
        let second = dir.path().join(format!("mf{i}-b.safetensors"));
        let loaded: RandomForestClassifier<f32, Fitted> =
            RandomForestClassifier::load(&mut p, &path).expect("load succeeds");
        loaded.save(&p, &second).expect("re-save succeeds");
        assert_eq!(
            std::fs::read(&path).expect("read"),
            std::fs::read(&second).expect("read"),
            "{policy:?} must round-trip byte-identically"
        );
    }
}

// ---------------------------------------------------------------------------
// The complete layout, measured
// ---------------------------------------------------------------------------

#[test]
fn the_file_is_a_complete_layout() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("rfc.safetensors");
    let mut p = pool();
    fit_rf_clf::<f32>(&mut p)
        .save(&p, &path)
        .expect("save succeeds");

    // The size trade stated as a measurement rather than a comment: every tree
    // occupies `2^(max_depth+1) − 1` slots whether or not the fit used them, and
    // `max_depth` is NOT stored — it is recovered from that count. A ragged file
    // would be smaller and would need an expansion pass on every load.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = EnsembleFile::parse(&raw, "random_forest_classifier").expect("parse succeeds");
    let expected_nodes = (1usize << (MAX_DEPTH + 1)) - 1;
    for name in ["split_feature", "threshold", "is_leaf"] {
        let view = file.tensor(name).expect("the tensor is present");
        assert_eq!(
            view.shape(),
            &[4, expected_nodes],
            "'{name}' must be the complete layout for {} trees at depth {MAX_DEPTH}",
            4
        );
    }
    assert!(
        file.metadata().get("max_depth").is_none(),
        "max_depth must NOT be stored — it is recovered from the node count, so a \
         second copy could only ever contradict the first"
    );
}

#[test]
fn f32_model_writes_a_smaller_file() {
    if mlrs_backend::capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let narrow = dir.path().join("f32.safetensors");
    let wide = dir.path().join("f64.safetensors");
    let mut p = pool();

    fit_rf_reg::<f32>(&mut p)
        .save(&p, &narrow)
        .expect("save succeeds");
    fit_rf_reg::<f64>(&mut p)
        .save(&p, &wide)
        .expect("save succeeds");

    // The stored dtype is the MODEL's dtype for the float tables. The two
    // integer tables are `U64` at both widths, so the saving is over
    // `threshold` + `leaf_dist` + `node_decrease` only — a smaller fraction than
    // in the all-float families, and worth measuring rather than assuming.
    let narrow_len = std::fs::metadata(&narrow).expect("stat").len();
    let wide_len = std::fs::metadata(&wide).expect("stat").len();
    let nodes = 4 * ((1u64 << (MAX_DEPTH + 1)) - 1);
    assert!(
        wide_len - narrow_len >= nodes * 3 * 4,
        "an f32 file must be at least {} bytes smaller (f32 {narrow_len}, f64 {wide_len})",
        nodes * 3 * 4
    );
}

// ---------------------------------------------------------------------------
// Rejection — the file is untrusted input (T-04-01-01)
// ---------------------------------------------------------------------------

/// Build a syntactically valid forest file with the given node geometry, so the
/// rejection gates below differ from a good file in exactly one respect.
fn write_forest(
    path: &std::path::Path,
    n_trees: usize,
    total_nodes: usize,
    split_feature: &[u64],
    is_leaf: &[u64],
) {
    let n = n_trees * total_nodes;
    let threshold = vec![0.0f32; n];
    let leaf_dist = vec![0.5f32; n * 3];
    let decrease = vec![0.0f32; n];
    let classes = [0i64, 1, 2];
    let importances = [0.25f32; N_FEATURES];

    let mut w = EnsembleWriter::new("random_forest_classifier");
    w.scalar_usize("param:n_estimators", n_trees);
    w.scalar_usize("param:max_depth", MAX_DEPTH);
    w.scalar_usize("param:n_bins", 128);
    w.scalar_str("param:max_features", "sqrt");
    w.scalar_f64("param:min_samples_split", 2.0);
    w.scalar_f64("param:min_samples_leaf", 1.0);
    w.scalar_bool("param:bootstrap", true);
    w.scalar_bool("param:oob_score", false);
    w.scalar_u64("param:seed", 7);
    w.scalar_usize("n_features_in_", N_FEATURES);
    let shape = vec![n_trees, total_nodes];
    w.tensor(
        "split_feature",
        TensorRef::u64s(split_feature, shape.clone()).expect("well-formed"),
    );
    w.tensor(
        "threshold",
        TensorRef::floats(&threshold, shape.clone()).expect("well-formed"),
    );
    w.tensor(
        "is_leaf",
        TensorRef::u64s(is_leaf, shape.clone()).expect("well-formed"),
    );
    w.tensor(
        "leaf_dist",
        TensorRef::floats(&leaf_dist, vec![n_trees, total_nodes, 3]).expect("well-formed"),
    );
    w.tensor(
        "node_decrease",
        TensorRef::floats(&decrease, shape).expect("well-formed"),
    );
    w.tensor(
        "classes_",
        TensorRef::i64s(&classes, vec![3]).expect("well-formed"),
    );
    w.tensor(
        "feature_importances_",
        TensorRef::floats(&importances, vec![N_FEATURES]).expect("well-formed"),
    );
    w.write(path)
        .expect("the hand-written file is well-formed as a container");
}

#[test]
fn a_non_complete_node_count_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ragged.safetensors");
    let mut p = pool();

    // 100 is a perfectly plausible-looking node count and is not `2^(d+1) − 1`.
    // The traversal walks `2i+1`/`2i+2` with no bound beyond the depth counter,
    // so on a non-complete table every walk that reached the last level would
    // read past the end. Nothing else in the file could reveal it — the tables
    // are internally consistent at any width.
    write_forest(&path, 2, 100, &vec![0u64; 200], &vec![1u64; 200]);

    // `expect_err` is unavailable: the estimators deliberately do not derive
    // `Debug` (they hold device handles), so the Ok arm is rejected by hand.
    let err = match RandomForestClassifier::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a non-complete node count must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn an_out_of_range_split_feature_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bad-feature.safetensors");
    let mut p = pool();

    // Each node reads `split_feature[node]` to index the QUERY row, so an index
    // at or past `n_features` reads past the end of a sample on the first
    // prediction. The value is individually well-formed — it is a fine integer —
    // so only the cross-check against `n_features_in_` catches it.
    let total_nodes = (1usize << (MAX_DEPTH + 1)) - 1;
    let mut split = vec![0u64; 2 * total_nodes];
    let mut is_leaf = vec![1u64; 2 * total_nodes];
    // Node 5 must be INTERNAL for its split feature to be dereferenced — a
    // leaf's slot carries a sentinel the kernel never reads, so a leaf here
    // would make the gate vacuous.
    split[5] = N_FEATURES as u64;
    is_leaf[5] = 0;
    write_forest(&path, 2, total_nodes, &split, &is_leaf);

    let err = match RandomForestClassifier::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an out-of-range split feature must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn sibling_ensembles_do_not_cross_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("rfr.safetensors");
    let mut p = pool();

    // The classifier and regressor files hold the SAME node tables at the same
    // shapes and dtypes; what `leaf_dist` MEANS differs completely — a class
    // distribution against a regression value — and nothing in the geometry says
    // which. A cross-load would predict confident nonsense rather than error.
    fit_rf_reg::<f32>(&mut p)
        .save(&p, &path)
        .expect("save succeeds");

    let err = match RandomForestClassifier::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a regressor file must not load as a classifier"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "random_forest_classifier" && found == "random_forest_regressor"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn a_boosters_k_disagreeing_with_its_trees_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bad-k.safetensors");
    let mut p = pool();

    // The raw-score accumulation strides the tree axis by `k`, so a `k` that
    // disagrees with the table would read a different tree for every column
    // after the first. `n_iters * k` and the table's row extent are each
    // well-formed alone; only their product ties them together.
    let total_nodes = (1usize << (MAX_DEPTH + 1)) - 1;
    let n_trees = 6usize;
    let n = n_trees * total_nodes;
    let mut w = EnsembleWriter::new("hist_gradient_boosting_regressor");
    w.scalar_str("param:device", "auto");
    w.scalar_usize("param:max_iter", 3);
    w.scalar_f64("param:learning_rate", 0.2);
    w.scalar_usize("param:max_depth", MAX_DEPTH);
    w.scalar_usize("param:n_bins", 128);
    w.scalar_f64("param:l2_regularization", 0.0);
    w.scalar_usize("param:min_samples_leaf", 20);
    w.scalar_usize("n_features_in_", N_FEATURES);
    // 3 x 3 = 9, but the tables hold 6 trees.
    w.scalar_usize("n_iters_", 3);
    w.scalar_usize("k_", 3);
    w.scalar_usize("n_classes_", 1);
    let shape = vec![n_trees, total_nodes];
    let zeros_u = vec![0u64; n];
    let ones_u = vec![1u64; n];
    let zeros_f = vec![0.0f32; n];
    let baseline = [0.0f32; 3];
    w.tensor(
        "split_feature",
        TensorRef::u64s(&zeros_u, shape.clone()).expect("well-formed"),
    );
    w.tensor(
        "threshold",
        TensorRef::floats(&zeros_f, shape.clone()).expect("well-formed"),
    );
    w.tensor(
        "is_leaf",
        TensorRef::u64s(&ones_u, shape.clone()).expect("well-formed"),
    );
    w.tensor(
        "leaf_value",
        TensorRef::floats(&zeros_f, shape).expect("well-formed"),
    );
    w.tensor(
        "baseline",
        TensorRef::floats(&baseline, vec![3]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match HistGradientBoostingRegressor::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a k disagreeing with the tree count must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_preprocessing_file_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("scaler.safetensors");
    let mut p = pool();

    let x: DeviceArray<ActiveRuntime, f32> = upload::<f32>(&mut p);
    MaxAbsScaler::<f32>::new()
        .fit(&mut p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("MaxAbsScaler fits")
        .save(&p, &path)
        .expect("save succeeds");

    let err = match RandomForestRegressor::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an mlrs-prep file must not load as an ensemble"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::NotAnMlrsModel { expected, .. } if *expected == "mlrs-ensemble"
        ),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

#[test]
fn saving_twice_produces_an_identical_model() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");
    let mut p = pool();

    // RAW BYTES: a model file must be a deterministic function of the model, so
    // it can be content-addressed and deduplicated. This is also the gate on the
    // `third_party/safetensors` `BTreeMap` patch.
    let fitted = fit_rf_clf::<f32>(&mut p);
    fitted.save(&p, &first).expect("save succeeds");
    fitted.save(&p, &second).expect("save succeeds");
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "saving the same model twice must produce byte-identical files"
    );
}

#[test]
fn save_leaves_no_temporary_behind() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("rfc.safetensors");
    let mut p = pool();
    fit_rf_clf::<f32>(&mut p)
        .save(&p, &path)
        .expect("save succeeds");

    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .expect("the scratch directory is readable")
        .filter_map(Result::ok)
        .map(|e| e.file_name())
        .filter(|n| n.to_string_lossy().contains("mlrs-tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "a successful save must leave no temporary file, found {leftovers:?}"
    );
    assert!(path.exists(), "the model file must exist");
}
