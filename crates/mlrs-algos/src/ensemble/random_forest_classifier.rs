//! `RandomForestClassifier` (ENSEMBLE-01) — gini-split random forest over the
//! launch-only batched forest primitive (`prims::random_forest`).
//!
//! ## Surface (typestate, D-03/D-05)
//! Builder-fronted `RandomForestClassifier<F, S = Unfit>`; [`Fit::fit`]
//! consumes `self` and returns the `Fitted` sibling holding the
//! device-resident [`RfModel`]. `predict_proba` returns the sklearn
//! mean-of-leaf-distributions (`n_query × n_classes`, rows sum to 1);
//! `predict_labels` is its argmax (lowest-index tie-break) mapped back
//! through the DISTINCT sorted `classes_` (the sklearn `classes_` contract —
//! CR-03 sibling of the KNN classifier).
//!
//! ## Class space
//! `fit` gathers the integer-valued `F` targets host-side, validates them
//! (WR-02: finite, integer, i32-range), collects `classes_` as the distinct
//! sorted labels and remaps each sample to its DENSE class index before the
//! device fit — a non-contiguous label set (e.g. `{0, 2}`) round-trips
//! exactly.
//!
//! Tests live in `crates/mlrs-algos/tests/random_forest_classifier_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)]` module).

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::random_forest::{
    rf_fit_class, rf_predict_proba, RfFitOutcome, RfModel, RfParams, RF_MAX_DEPTH_CAP,
};
use mlrs_backend::runtime::ActiveRuntime;

use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, PredictProba, State, Unfit};

use super::ensemble_persist::{
    as_floats, as_i64, expect_len, read_leaf_table, read_node_tables, shape_1d, widen_nodes,
    write_node_tables, AlignedBytes, EnsembleFile, EnsembleWriter, LoadModel, PersistError,
    SaveModel, TensorRef, LEAF_DIST_NAME, NODE_DECREASE_NAME,
};
use super::MaxFeatures;

/// The `estimator` discriminator written into every `RandomForestClassifier`
/// file.
///
/// Load-bearing rather than decorative: the regressor's file holds the SAME four
/// node tables at the same shapes and dtypes, differing only by carrying no
/// `classes_` and by what its `leaf_dist` MEANS — a class distribution against a
/// regression value. A cross-load would produce a model that predicts confident
/// nonsense rather than an error.
const PERSIST_TAG: &str = "random_forest_classifier";

/// The tensor holding the distinct sorted training labels, `[n_classes]`.
const CLASSES_NAME: &str = "classes_";
/// The tensor holding the normalized per-feature importances, `[n_features]`.
const IMPORTANCES_NAME: &str = "feature_importances_";

/// sklearn defaults (single source, D-08): `n_estimators=100`,
/// `min_samples_split=2`, `min_samples_leaf=1`, `bootstrap=true`,
/// `max_features=sqrt`. `max_depth=10` and `n_bins=32` are the mlrs-bounded
/// histogram-builder defaults (documented deviation, `ensemble/mod.rs`).
const RF_CLF_DEFAULT_N_ESTIMATORS: usize = 100;
const RF_CLF_DEFAULT_MAX_DEPTH: usize = 10;
const RF_CLF_DEFAULT_N_BINS: usize = 32;
const RF_CLF_DEFAULT_MIN_SAMPLES_SPLIT: f64 = 2.0;
const RF_CLF_DEFAULT_MIN_SAMPLES_LEAF: f64 = 1.0;
const RF_CLF_DEFAULT_SEED: u64 = 42;

/// Random forest classifier (ENSEMBLE-01), generic over the float type and
/// lifecycle state. The fitted forest is device-resident (D-03); host
/// accessors materialize on demand and exist only on the `Fitted` sibling.
pub struct RandomForestClassifier<F, S = Unfit>
where
    F: Float + CubeElement + Pod,
    S: State,
{
    n_estimators: usize,
    max_depth: usize,
    n_bins: usize,
    max_features: MaxFeatures,
    min_samples_split: f64,
    min_samples_leaf: f64,
    bootstrap: bool,
    /// RF-OOB-01: compute `oob_score_` at fit time (requires `bootstrap`,
    /// enforced at `build()`). Default `false` — the common case pays no
    /// extra fit-time cost.
    oob_score: bool,
    seed: u64,
    /// The fitted device-resident forest, `None` until `fit`.
    model_: Option<RfModel<F>>,
    /// The DISTINCT sorted training labels; `predict_labels` maps the dense
    /// argmax column back through these (CR-03).
    classes_: Vec<i32>,
    /// `classes_.len()`, cached.
    n_classes_: usize,
    /// RF-IMP-01: normalized (sums to 1) length-`n_features` mean-decrease-in-
    /// impurity vector, empty until `fit`.
    feature_importances_: Vec<F>,
    /// RF-OOB-01: `Some(score)` once fitted with `oob_score=true`; `None`
    /// otherwise (including always on the `Unfit` state, before `fit`).
    oob_score_: Option<F>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> RandomForestClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfit classifier with the defaults above (D-08 single
    /// source; the builder `Default` re-derives from here).
    pub fn new() -> Self {
        Self {
            n_estimators: RF_CLF_DEFAULT_N_ESTIMATORS,
            max_depth: RF_CLF_DEFAULT_MAX_DEPTH,
            n_bins: RF_CLF_DEFAULT_N_BINS,
            max_features: MaxFeatures::Sqrt,
            min_samples_split: RF_CLF_DEFAULT_MIN_SAMPLES_SPLIT,
            min_samples_leaf: RF_CLF_DEFAULT_MIN_SAMPLES_LEAF,
            bootstrap: true,
            oob_score: false,
            seed: RF_CLF_DEFAULT_SEED,
            model_: None,
            classes_: Vec::new(),
            n_classes_: 0,
            feature_importances_: Vec::new(),
            oob_score_: None,
            _state: PhantomData,
        }
    }

    /// Start building from the defaults (D-08 single source).
    pub fn builder() -> RandomForestClassifierBuilder {
        RandomForestClassifierBuilder::default()
    }

    /// Decompose back into the builder (used by the builder `Default`).
    pub fn into_builder(self) -> RandomForestClassifierBuilder {
        RandomForestClassifierBuilder {
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            n_bins: self.n_bins,
            max_features: self.max_features,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            bootstrap: self.bootstrap,
            oob_score: self.oob_score,
            seed: self.seed,
        }
    }
}

impl<F> Default for RandomForestClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> RandomForestClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The number of distinct classes inferred at `fit`.
    pub fn n_classes(&self) -> usize {
        self.n_classes_
    }

    /// The DISTINCT sorted training labels (the sklearn `classes_` contract).
    pub fn classes(&self) -> &[i32] {
        &self.classes_
    }

    /// The fitted feature count.
    pub fn n_features(&self) -> usize {
        self.model_
            .as_ref()
            .expect("model_ is Some by construction on the Fitted state")
            .n_features()
    }

    /// Borrow the fitted device forest (for the perf harness / debugging).
    pub fn model(&self) -> &RfModel<F> {
        self.model_
            .as_ref()
            .expect("model_ is Some by construction on the Fitted state")
    }

    /// RF-IMP-01: the sklearn-equivalent normalized (sums to 1) mean-decrease-
    /// in-impurity `feature_importances_`, length `n_features()`. Always
    /// populated on any `Fitted` instance (no `oob_score`/`bootstrap`
    /// precondition, matching sklearn).
    pub fn feature_importances(&self) -> &[F] {
        &self.feature_importances_
    }

    /// RF-OOB-01: the out-of-bag score computed at fit time — accuracy of
    /// the OOB-tree-averaged class-distribution argmax vs. training `y`.
    /// `Some(..)` iff the builder's `oob_score` flag was `true`; `None`
    /// otherwise (matches `RfFitOutcome::oob_score`'s own contract).
    pub fn oob_score(&self) -> Option<F> {
        self.oob_score_
    }

    /// SHAP-01: path-dependent TreeSHAP values, self-consistency-gated (see
    /// `tree_shap` module docs — a native mlrs forest has no external
    /// oracle). `x_train_host`/`query_host` are host row-major `f64` buffers
    /// (`n_train`/`n_query` × `n_features()`). Returns `(phi, expected_value)`:
    /// `phi` is `n_query × n_features × n_classes`; `expected_value` is
    /// length `n_classes`. `Σ_f phi[q, f, :] + expected_value ==
    /// predict_proba(query)[q]` exactly for every row.
    pub fn shap_values(
        &self,
        pool: &BufferPool<ActiveRuntime>,
        x_train_host: &[f64],
        n_train: usize,
        query_host: &[f64],
        n_query: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        crate::ensemble::tree_shap::native_forest_shap_values(
            pool,
            self.model(),
            x_train_host,
            n_train,
            query_host,
            n_query,
        )
    }
}

/// Builder for [`RandomForestClassifier`] (D-01). `Default` re-derives the
/// defaults from [`RandomForestClassifier::new`] (D-08 single source).
#[derive(Debug, Clone, Copy)]
pub struct RandomForestClassifierBuilder {
    n_estimators: usize,
    max_depth: usize,
    n_bins: usize,
    max_features: MaxFeatures,
    min_samples_split: f64,
    min_samples_leaf: f64,
    bootstrap: bool,
    oob_score: bool,
    seed: u64,
}

impl Default for RandomForestClassifierBuilder {
    fn default() -> Self {
        RandomForestClassifier::<f64, Unfit>::new().into_builder()
    }
}

impl RandomForestClassifierBuilder {
    /// Set the tree count `n_estimators` (`>= 1`).
    pub fn n_estimators(mut self, v: usize) -> Self {
        self.n_estimators = v;
        self
    }

    /// Set the depth bound (`1..=16`; leaves are forced at this depth —
    /// documented deviation from sklearn's `None`).
    pub fn max_depth(mut self, v: usize) -> Self {
        self.max_depth = v;
        self
    }

    /// Set the histogram bin count per feature (`2..=256`).
    pub fn n_bins(mut self, v: usize) -> Self {
        self.n_bins = v;
        self
    }

    /// Set the per-node feature-subsample policy (sklearn `max_features`;
    /// classifier default [`MaxFeatures::Sqrt`]).
    pub fn max_features(mut self, v: MaxFeatures) -> Self {
        self.max_features = v;
        self
    }

    /// Set `min_samples_split` (`>= 2`, sklearn integer form as f64 — A5).
    pub fn min_samples_split(mut self, v: f64) -> Self {
        self.min_samples_split = v;
        self
    }

    /// Set `min_samples_leaf` (`>= 1`).
    pub fn min_samples_leaf(mut self, v: f64) -> Self {
        self.min_samples_leaf = v;
        self
    }

    /// Enable/disable per-tree bootstrap resampling (sklearn `bootstrap`).
    pub fn bootstrap(mut self, v: bool) -> Self {
        self.bootstrap = v;
        self
    }

    /// RF-OOB-01: enable/disable `oob_score_` computation at fit time
    /// (sklearn `oob_score`, default `false`). Requires `bootstrap = true`
    /// (enforced at `build()`, mirrors sklearn's `ValueError`).
    pub fn oob_score(mut self, v: bool) -> Self {
        self.oob_score = v;
        self
    }

    /// Set the host RNG seed (bootstrap + feature subsampling; fully
    /// deterministic across runs and backends).
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }

    /// Build the (unfit) estimator, validating every data-INDEPENDENT
    /// hyperparameter (D-08; `max_features <= n_features` is data-dependent
    /// and stays at `fit`).
    pub fn build<F>(self) -> Result<RandomForestClassifier<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        validate_forest_hyperparams(
            "random_forest_classifier",
            self.n_estimators,
            self.max_depth,
            self.n_bins,
            self.max_features,
            self.min_samples_split,
            self.min_samples_leaf,
        )?;
        if self.oob_score && !self.bootstrap {
            return Err(BuildError::OobRequiresBootstrap {
                estimator: "random_forest_classifier",
            });
        }
        Ok(RandomForestClassifier {
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            n_bins: self.n_bins,
            max_features: self.max_features,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            bootstrap: self.bootstrap,
            oob_score: self.oob_score,
            seed: self.seed,
            model_: None,
            classes_: Vec::new(),
            n_classes_: 0,
            feature_importances_: Vec::new(),
            oob_score_: None,
            _state: PhantomData,
        })
    }
}

/// Shared builder-time hyperparameter validation (classifier + regressor).
pub(crate) fn validate_forest_hyperparams(
    estimator: &'static str,
    n_estimators: usize,
    max_depth: usize,
    n_bins: usize,
    max_features: MaxFeatures,
    min_samples_split: f64,
    min_samples_leaf: f64,
) -> Result<(), BuildError> {
    if n_estimators == 0 {
        return Err(BuildError::InvalidNEstimators {
            estimator,
            n_estimators,
        });
    }
    if max_depth == 0 || max_depth > RF_MAX_DEPTH_CAP {
        return Err(BuildError::InvalidMaxDepth {
            estimator,
            max_depth,
        });
    }
    if n_bins < 2 || n_bins > 256 {
        return Err(BuildError::InvalidNBins { estimator, n_bins });
    }
    if let MaxFeatures::Value(0) = max_features {
        return Err(BuildError::InvalidMaxFeatures {
            estimator,
            max_features: 0,
        });
    }
    if !min_samples_split.is_finite() || min_samples_split < 2.0 {
        return Err(BuildError::InvalidMinSamplesForest {
            estimator,
            which: "min_samples_split",
            value: min_samples_split,
        });
    }
    if !min_samples_leaf.is_finite() || min_samples_leaf < 1.0 {
        return Err(BuildError::InvalidMinSamplesForest {
            estimator,
            which: "min_samples_leaf",
            value: min_samples_leaf,
        });
    }
    Ok(())
}

/// Shared label ingestion (the KNN classifier's WR-02/CR-03 discipline):
/// validate integer-valued finite i32-range labels, collect DISTINCT sorted
/// `classes_`, and remap each sample to its dense class index.
pub(crate) fn ingest_labels<F>(
    estimator: &'static str,
    y_host: &[F],
) -> Result<(Vec<i32>, Vec<u32>), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let mut raw_class: Vec<i32> = Vec::with_capacity(y_host.len());
    for &v in y_host.iter() {
        let lf = host_to_f64(v);
        let lr = lf.round();
        if !lr.is_finite() || (lr - lf).abs() > 1e-6 || i32::try_from(lr as i64).is_err() {
            return Err(AlgoError::InvalidLabels {
                estimator,
                reason: format!("labels must be i32-range integers (got {lf})"),
            });
        }
        raw_class.push(lr as i32);
    }
    let mut classes: Vec<i32> = raw_class.clone();
    classes.sort_unstable();
    classes.dedup();
    if classes.len() < 2 {
        return Err(AlgoError::InvalidLabels {
            estimator,
            reason: format!("need at least 2 distinct classes (got {})", classes.len()),
        });
    }
    let y_idx: Vec<u32> = raw_class
        .iter()
        .map(|&l| {
            classes
                .binary_search(&l)
                .expect("every raw label is in classes_ by construction") as u32
        })
        .collect();
    Ok((classes, y_idx))
}

impl<F> SaveModel for RandomForestClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted forest to `path` as a safetensors file.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `split_feature` / `is_leaf` | `U64` | `[n_trees, total_nodes]` |
    /// | `threshold` | `F` (`F32`/`F64`) | `[n_trees, total_nodes]` |
    /// | `leaf_dist` | `F` | `[n_trees, total_nodes, n_classes]` |
    /// | `node_decrease` | `F` | `[n_trees, total_nodes]` |
    /// | `classes_` | `I64` | `[n_classes]` |
    /// | `feature_importances_` | `F` | `[n_features]` |
    /// | `oob_score_` | `__metadata__` scalar, optional | — |
    /// | nine `param:*` scalars | `__metadata__` | — |
    ///
    /// The trees are stored in the COMPLETE layout the traversal kernel indexes
    /// directly — see [`ensemble_persist`](super::ensemble_persist) for the size
    /// trade that makes, and why a ragged encoding would move unbounded work
    /// onto the read path.
    ///
    /// `node_decrease` is written even though `feature_importances_` (the
    /// reduced, normalized form callers actually read) is written beside it.
    /// The two are not redundant: the per-node array is what a future per-tree
    /// or per-subset importance query would need, and it is what
    /// [`RfModel::from_saved_parts`] restores so a reloaded forest is
    /// indistinguishable from the fitted one rather than merely equivalent at
    /// predict time.
    ///
    /// `max_depth` is NOT stored — it is recovered from `total_nodes`, which
    /// must be `2^(d+1) − 1` for the child arithmetic to be in bounds at all.
    /// Storing it would be a second copy of the same fact that a hand-edited
    /// header could contradict.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let model = self.model_.as_ref().ok_or(PersistError::MissingState {
            estimator: PERSIST_TAG,
            field: "model_",
        })?;
        // Bound BEFORE the writer: `EnsembleWriter` borrows every payload so it
        // can stream them out without a second copy.
        let split_feature = widen_nodes(&model.split_feature_host(pool));
        let is_leaf = widen_nodes(&model.is_leaf_host(pool));
        let threshold = model.threshold_host(pool);
        let leaf_dist = model.leaf_dist_host(pool);
        let node_decrease = model.node_decrease_host(pool);
        let classes: Vec<i64> = self.classes_.iter().map(|&v| i64::from(v)).collect();

        let (n_trees, total_nodes) = (model.n_trees(), model.total_nodes());
        let n_values = model.n_values();

        let mut w = EnsembleWriter::new(PERSIST_TAG);
        w.scalar_usize("param:n_estimators", self.n_estimators);
        w.scalar_usize("param:max_depth", self.max_depth);
        w.scalar_usize("param:n_bins", self.n_bins);
        w.scalar_str("param:max_features", &self.max_features.name());
        w.scalar_f64("param:min_samples_split", self.min_samples_split);
        w.scalar_f64("param:min_samples_leaf", self.min_samples_leaf);
        w.scalar_bool("param:bootstrap", self.bootstrap);
        w.scalar_bool("param:oob_score", self.oob_score);
        w.scalar_u64("param:seed", self.seed);
        w.scalar_usize("n_features_in_", model.n_features());
        w.scalar_opt_f64("oob_score_", self.oob_score_.map(host_to_f64));

        write_node_tables(
            &mut w,
            &split_feature,
            &threshold,
            &is_leaf,
            n_trees,
            total_nodes,
        )?;
        w.tensor(
            LEAF_DIST_NAME,
            TensorRef::floats(&leaf_dist, vec![n_trees, total_nodes, n_values])?,
        );
        w.tensor(
            NODE_DECREASE_NAME,
            TensorRef::floats(&node_decrease, vec![n_trees, total_nodes])?,
        );
        w.tensor(CLASSES_NAME, TensorRef::i64s(&classes, vec![classes.len()])?);
        w.tensor(
            IMPORTANCES_NAME,
            TensorRef::floats(
                &self.feature_importances_,
                vec![self.feature_importances_.len()],
            )?,
        );
        w.write(path)
    }
}

impl<F> LoadModel for RandomForestClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read the forest back from `path`, re-uploading every node table to
    /// `pool`.
    ///
    /// The file is untrusted input (T-04-01-01), and this estimator has the
    /// strictest checks in the crate because its traversal kernel indexes two
    /// different axes off stored values: `2i+1`/`2i+2` into the node table, and
    /// `split_feature[node]` into the query row. So
    /// [`read_node_tables`] establishes both the complete-layout invariant and
    /// every split feature's range before the model exists, and `leaf_dist`'s
    /// per-node width is cross-checked against `classes_`.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<RandomForestClassifier<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = EnsembleFile::parse(&raw, PERSIST_TAG)?;
        let n_features = file.scalar_usize("n_features_in_")?;
        if n_features == 0 {
            return Err(PersistError::InconsistentGeometry {
                reason: "'n_features_in_' is 0; a fitted forest has at least one feature"
                    .to_string(),
            });
        }
        let tables = read_node_tables::<F>(&file, n_features)?;

        let classes_v = file.tensor(CLASSES_NAME)?;
        let n_classes = shape_1d(&classes_v, CLASSES_NAME)?;
        if n_classes < 2 {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "tensor '{CLASSES_NAME}' holds {n_classes} labels; a fitted classifier \
                     has at least 2"
                ),
            });
        }
        let classes_: Vec<i32> = as_i64(&classes_v, CLASSES_NAME)?
            .iter()
            .map(|&v| {
                i32::try_from(v).map_err(|_| PersistError::InconsistentGeometry {
                    reason: format!(
                        "tensor '{CLASSES_NAME}' holds the label {v}, which does not fit the \
                         i32 the forest's kernels consume"
                    ),
                })
            })
            .collect::<Result<_, _>>()?;

        // `leaf_dist` holds one entry per CLASS per node, so its length is the
        // cross-check that ties the node tables to the label table.
        let leaf_dist = read_leaf_table::<F>(
            &file,
            LEAF_DIST_NAME,
            tables.n_trees,
            tables.total_nodes,
            n_classes,
        )?;
        let node_decrease = read_leaf_table::<F>(
            &file,
            NODE_DECREASE_NAME,
            tables.n_trees,
            tables.total_nodes,
            1,
        )?;

        let importances_v = file.tensor(IMPORTANCES_NAME)?;
        expect_len(
            IMPORTANCES_NAME,
            shape_1d(&importances_v, IMPORTANCES_NAME)?,
            n_features,
            "entries",
        )?;
        let feature_importances_ = as_floats::<F>(&importances_v, IMPORTANCES_NAME)?.into_owned();

        let model_ = RfModel::from_saved_parts(
            pool,
            &tables.split_feature,
            &tables.threshold,
            &tables.is_leaf,
            &leaf_dist,
            &node_decrease,
            tables.n_trees,
            tables.max_depth,
            n_features,
            n_classes,
        )
        .map_err(|e| PersistError::InconsistentGeometry {
            reason: format!("the node tables do not assemble into a forest: {e}"),
        })?;

        Ok(RandomForestClassifier {
            n_estimators: file.scalar_usize("param:n_estimators")?,
            max_depth: file.scalar_usize("param:max_depth")?,
            n_bins: file.scalar_usize("param:n_bins")?,
            max_features: MaxFeatures::from_name(file.scalar_str("param:max_features")?).ok_or(
                PersistError::BadMetadata {
                    key: "param:max_features",
                },
            )?,
            min_samples_split: file.scalar_f64("param:min_samples_split")?,
            min_samples_leaf: file.scalar_f64("param:min_samples_leaf")?,
            bootstrap: file.scalar_bool("param:bootstrap")?,
            oob_score: file.scalar_bool("param:oob_score")?,
            seed: file.scalar_u64("param:seed")?,
            model_: Some(model_),
            classes_,
            n_classes_: n_classes,
            feature_importances_,
            oob_score_: file.scalar_opt_f64("oob_score_")?.map(f64_to_host::<F>),
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for RandomForestClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = RandomForestClassifier<F, Fitted>;

    /// Grow the forest on `(x, y)` (y = integer-valued `F` class labels),
    /// CONSUMING `self`. The device fit loop is launch-only; the single host
    /// sync is the bin-edge quantile pass (see `prims::random_forest`).
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<RandomForestClassifier<F, Fitted>, AlgoError> {
        let (n, d) = shape;
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "random_forest_classifier",
            operation: "fit (requires y)",
        })?;
        if y.len() != n {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n,
                cols: 1,
                len: y.len(),
            }));
        }

        let y_host = y.to_host(pool);
        let (classes, y_idx) = ingest_labels::<F>("random_forest_classifier", &y_host)?;
        let n_classes = classes.len();

        let params = RfParams {
            n_trees: self.n_estimators,
            max_depth: self.max_depth,
            n_bins: self.n_bins,
            max_features: self.max_features.resolve(d),
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            bootstrap: self.bootstrap,
            seed: self.seed,
            oob_score: self.oob_score,
        };
        let RfFitOutcome {
            model,
            feature_importances,
            oob_score: oob_score_,
        } = rf_fit_class::<F>(pool, x, shape, &y_idx, n_classes, &params)?;

        Ok(RandomForestClassifier {
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            n_bins: self.n_bins,
            max_features: self.max_features,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            bootstrap: self.bootstrap,
            oob_score: self.oob_score,
            seed: self.seed,
            model_: Some(model),
            feature_importances_: feature_importances,
            oob_score_,
            classes_: classes,
            n_classes_: n_classes,
            _state: PhantomData,
        })
    }
}

impl<F> PredictProba<F> for RandomForestClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// `n_query × n_classes` mean of the reached leaves' class distributions
    /// (device-computed, rows sum to 1) — the sklearn `predict_proba` form.
    fn predict_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        validate_geometry(x, shape)?;
        let model = self
            .model_
            .as_ref()
            .expect("model_ is Some by construction on the Fitted state");
        Ok(rf_predict_proba::<F>(pool, model, x, shape)?)
    }
}

impl<F> PredictLabels<F> for RandomForestClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// `predict = argmax(predict_proba)` with the lowest-class-index
    /// tie-break, mapped back through `classes_` (CR-03).
    ///
    /// The argmax runs HOST-side over ONE metered proba readback (`n_query ×
    /// n_classes` floats). The per-row `argmax_rows` prim is deliberately NOT
    /// used here: it uploads + launches + reads back PER ROW, which made
    /// predict sync-bound (~100 µs/row — the exact disease the launch-only
    /// fit loop avoids).
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        let (n_query, _) = shape;
        let nc = self.n_classes_;
        let proba = self.predict_proba(pool, x, shape)?;
        let proba_host = proba.to_host_metered(pool);
        proba.release_into(pool);

        let mut labels_i32: Vec<i32> = Vec::with_capacity(n_query);
        for r in 0..n_query {
            let row = &proba_host[r * nc..(r + 1) * nc];
            // Strict `>` keeps the FIRST maximum — the lowest-class-index
            // tie-break (the argmax_rows / sklearn convention).
            let mut best = 0usize;
            let mut best_v = host_to_f64(row[0]);
            for (c, &v) in row.iter().enumerate().skip(1) {
                let vf = host_to_f64(v);
                if vf > best_v {
                    best_v = vf;
                    best = c;
                }
            }
            labels_i32.push(self.classes_[best]);
        }
        Ok(DeviceArray::from_host(pool, &labels_i32))
    }
}
