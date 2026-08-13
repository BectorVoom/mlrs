//! `HistGradientBoostingClassifier` (GBT-01) — log-loss gradient boosting
//! over the launch-only histogram-tree primitive
//! (`prims::hist_gradient_boosting`).
//!
//! ## Surface (typestate, D-03/D-05)
//! Builder-fronted `HistGradientBoostingClassifier<F, S = Unfit>`;
//! [`Fit::fit`] consumes `self` and returns the `Fitted` sibling holding the
//! device-resident [`HgbModel`]. Binary targets use ONE sigmoid raw-score
//! column (sklearn `n_trees_per_iteration_ = 1`); multiclass uses
//! `n_classes` softmax columns whose trees grow batched per iteration.
//! `predict_proba` is the sklearn link (sigmoid / softmax of the raw scores);
//! `predict_labels` is its argmax (lowest-index tie-break) mapped back
//! through the DISTINCT sorted `classes_` (the sklearn `classes_` contract —
//! the Random Forest CR-03 sibling).
//!
//! ## Class space
//! `fit` gathers the integer-valued `F` targets host-side, validates them
//! (WR-02: finite, integer, i32-range), collects `classes_` as the distinct
//! sorted labels and remaps each sample to its DENSE class index before the
//! device fit (shared `ingest_labels` with the forest classifier).
//!
//! Fits are fully deterministic: no bootstrap, no feature subsampling, no RNG.
//!
//! Tests live in
//! `crates/mlrs-algos/tests/hist_gradient_boosting_classifier_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)]` module).

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device::Device;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::hist_gradient_boosting::{
    hgb_fit_class, hgb_predict_proba, HgbModel, HgbParams,
};
use mlrs_backend::runtime::ActiveRuntime;

use mlrs_core::{host_to_f64, PrimError};

use super::ensemble_persist::{
    as_floats, as_i64, expect_len, read_leaf_table, read_node_tables, shape_1d, widen_nodes,
    write_node_tables, AlignedBytes, EnsembleFile, EnsembleWriter, LoadModel, PersistError,
    SaveModel, TensorRef, BASELINE_NAME, LEAF_VALUE_NAME,
};

/// The tensor holding the distinct sorted training labels, `[n_classes]`.
const CLASSES_NAME: &str = "classes_";

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, PredictProba, State, Unfit};

use super::hist_gradient_boosting_regressor::validate_hgb_hyperparams;
use super::random_forest_classifier::ingest_labels;

/// sklearn defaults (single source, D-08) — see the regressor's constants for
/// the `max_depth=6` level-wise and `n_bins=64` histogram-lattice deviation
/// rationales.
const HGB_CLF_DEFAULT_MAX_ITER: usize = 100;
const HGB_CLF_DEFAULT_LEARNING_RATE: f64 = 0.1;
const HGB_CLF_DEFAULT_MAX_DEPTH: usize = 6;
const HGB_CLF_DEFAULT_N_BINS: usize = 64;
const HGB_CLF_DEFAULT_L2: f64 = 0.0;
const HGB_CLF_DEFAULT_MIN_SAMPLES_LEAF: usize = 20;

/// HistGradientBoosting classifier (GBT-01), generic over the float type and
/// lifecycle state. The fitted ensemble is device-resident (D-03).
pub struct HistGradientBoostingClassifier<F, S = Unfit>
where
    F: Float + CubeElement + Pod,
    S: State,
{
    /// Where to run the heavy phase (DEVICE-PARAM-01).
    device: Device,
    max_iter: usize,
    learning_rate: f64,
    max_depth: usize,
    n_bins: usize,
    l2_regularization: f64,
    min_samples_leaf: usize,
    /// The fitted device-resident ensemble, `None` until `fit`.
    model_: Option<HgbModel<F>>,
    /// The DISTINCT sorted training labels; `predict_labels` maps the dense
    /// argmax column back through these (CR-03).
    classes_: Vec<i32>,
    /// `classes_.len()`, cached.
    n_classes_: usize,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> HistGradientBoostingClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfit classifier with the defaults above (D-08 single
    /// source; the builder `Default` re-derives from here).
    pub fn new() -> Self {
        Self {
            max_iter: HGB_CLF_DEFAULT_MAX_ITER,
            device: Device::Auto,
            learning_rate: HGB_CLF_DEFAULT_LEARNING_RATE,
            max_depth: HGB_CLF_DEFAULT_MAX_DEPTH,
            n_bins: HGB_CLF_DEFAULT_N_BINS,
            l2_regularization: HGB_CLF_DEFAULT_L2,
            min_samples_leaf: HGB_CLF_DEFAULT_MIN_SAMPLES_LEAF,
            model_: None,
            classes_: Vec::new(),
            n_classes_: 0,
            _state: PhantomData,
        }
    }

    /// Start building from the defaults (D-08 single source).
    pub fn builder() -> HistGradientBoostingClassifierBuilder {
        HistGradientBoostingClassifierBuilder::default()
    }

    /// Decompose back into the builder (used by the builder `Default`).
    pub fn into_builder(self) -> HistGradientBoostingClassifierBuilder {
        HistGradientBoostingClassifierBuilder {
            max_iter: self.max_iter,
            device: self.device,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            n_bins: self.n_bins,
            l2_regularization: self.l2_regularization,
            min_samples_leaf: self.min_samples_leaf,
        }
    }
}

impl<F> Default for HistGradientBoostingClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> HistGradientBoostingClassifier<F, Fitted>
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

    /// Borrow the fitted device ensemble (for the perf harness / debugging).
    pub fn model(&self) -> &HgbModel<F> {
        self.model_
            .as_ref()
            .expect("model_ is Some by construction on the Fitted state")
    }
}

/// Builder for [`HistGradientBoostingClassifier`] (D-01). `Default`
/// re-derives the defaults from [`HistGradientBoostingClassifier::new`]
/// (D-08 single source).
#[derive(Debug, Clone, Copy)]
pub struct HistGradientBoostingClassifierBuilder {
    device: Device,
    max_iter: usize,
    learning_rate: f64,
    max_depth: usize,
    n_bins: usize,
    l2_regularization: f64,
    min_samples_leaf: usize,
}

impl Default for HistGradientBoostingClassifierBuilder {
    fn default() -> Self {
        HistGradientBoostingClassifier::<f64, Unfit>::new().into_builder()
    }
}

impl HistGradientBoostingClassifierBuilder {

    /// Pin the execution arm (DEVICE-PARAM-01). [`Device::Auto`] keeps the
    /// existing gate and its `MLRS_*` A/B flag; `Cpu`/`Gpu` override its PERF
    /// half only — each prim keeps its own capability checks inside.
    pub fn device(mut self, v: Device) -> Self {
        self.device = v;
        self
    }
    /// Set the boosting iteration count `max_iter` (`>= 1`).
    pub fn max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }

    /// Set the shrinkage `learning_rate` (finite, `> 0`).
    pub fn learning_rate(mut self, v: f64) -> Self {
        self.learning_rate = v;
        self
    }

    /// Set the depth bound (`1..=16`; documented deviation from sklearn's
    /// leaf-wise `max_leaf_nodes` growth).
    pub fn max_depth(mut self, v: usize) -> Self {
        self.max_depth = v;
        self
    }

    /// Set the histogram bin count per feature (`2..=256`; sklearn
    /// `max_bins = 255`).
    pub fn n_bins(mut self, v: usize) -> Self {
        self.n_bins = v;
        self
    }

    /// Set the leaf-value L2 penalty `l2_regularization` (finite, `>= 0`).
    pub fn l2_regularization(mut self, v: f64) -> Self {
        self.l2_regularization = v;
        self
    }

    /// Set `min_samples_leaf` (`>= 1`, a sample COUNT — the sklearn HGB
    /// contract).
    pub fn min_samples_leaf(mut self, v: usize) -> Self {
        self.min_samples_leaf = v;
        self
    }

    /// Build the (unfit) estimator, validating every data-INDEPENDENT
    /// hyperparameter (D-08).
    pub fn build<F>(self) -> Result<HistGradientBoostingClassifier<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        validate_hgb_hyperparams(
            "hist_gradient_boosting_classifier",
            self.max_iter,
            self.learning_rate,
            self.max_depth,
            self.n_bins,
            self.l2_regularization,
            self.min_samples_leaf,
        )?;
        Ok(HistGradientBoostingClassifier {
            max_iter: self.max_iter,
            device: self.device,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            n_bins: self.n_bins,
            l2_regularization: self.l2_regularization,
            min_samples_leaf: self.min_samples_leaf,
            model_: None,
            classes_: Vec::new(),
            n_classes_: 0,
            _state: PhantomData,
        })
    }
}

/// The `estimator` discriminator written into every `HistGradientBoostingClassifier` file.
///
/// Load-bearing rather than decorative: the sibling booster's file holds the
/// SAME node tables at the same shapes and dtypes. The classifier's `k` is the
/// class count and the regressor's is always 1, but a binary classifier also has
/// `k == 1` — so on the commonest shape the two files are structurally identical
/// and only the tag separates them.
const PERSIST_TAG: &str = "hist_gradient_boosting_classifier";

impl<F> SaveModel for HistGradientBoostingClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted booster to `path` as a safetensors file.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `split_feature` / `is_leaf` | `U64` | `[n_iters * k, total_nodes]` |
    /// | `threshold` / `leaf_value` | `F` (`F32`/`F64`) | `[n_iters * k, total_nodes]` |
    /// | `baseline` | `F` | `[k]` |
    /// | `n_features_in_` / `n_iters_` / `k_` | `__metadata__` scalar | — |
    /// | seven `param:*` scalars | `__metadata__` | — |
    ///
    /// The tree axis is `n_iters · k`, not `n_iters`: a multiclass booster fits
    /// `k` trees per iteration, one per raw-score column. Both extents are
    /// therefore stored as scalars rather than inferred from the flat count —
    /// their PRODUCT is what the node tables' row extent gives, and a product
    /// does not determine its factors. Each is cross-checked against that row
    /// extent on load.
    ///
    /// See [`ensemble_persist`](super::ensemble_persist) for the complete-layout
    /// decision and the size trade it makes.
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
        let leaf_value = model.leaf_value_host(pool);
        let baseline = model.baseline_host(pool);
        let classes: Vec<i64> = self.classes_.iter().map(|&v| i64::from(v)).collect();
        let (n_iters, k) = (model.n_iters(), model.k());
        let n_trees = n_iters * k;
        let total_nodes = model.total_nodes();

        let mut w = EnsembleWriter::new(PERSIST_TAG);
        w.scalar_str("param:device", self.device.name());
        w.scalar_usize("param:max_iter", self.max_iter);
        w.scalar_f64("param:learning_rate", self.learning_rate);
        w.scalar_usize("param:max_depth", self.max_depth);
        w.scalar_usize("param:n_bins", self.n_bins);
        w.scalar_f64("param:l2_regularization", self.l2_regularization);
        w.scalar_usize("param:min_samples_leaf", self.min_samples_leaf);
        w.scalar_usize("n_features_in_", model.n_features());
        w.scalar_usize("n_iters_", n_iters);
        w.scalar_usize("k_", k);
        w.scalar_usize("n_classes_", model.n_classes());

        write_node_tables(
            &mut w,
            &split_feature,
            &threshold,
            &is_leaf,
            n_trees,
            total_nodes,
        )?;
        w.tensor(
            LEAF_VALUE_NAME,
            TensorRef::floats(&leaf_value, vec![n_trees, total_nodes])?,
        );
        w.tensor(BASELINE_NAME, TensorRef::floats(&baseline, vec![k])?);
        w.tensor(CLASSES_NAME, TensorRef::i64s(&classes, vec![classes.len()])?);
        w.write(path)
    }
}

impl<F> LoadModel for HistGradientBoostingClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read the booster back from `path`, re-uploading every node table to
    /// `pool`.
    ///
    /// `n_iters · k` must equal the node tables' row extent, and `baseline` must
    /// hold exactly `k` entries. The file is untrusted input (T-04-01-01), and
    /// the raw-score accumulation strides the tree axis by `k` — so a `k` that
    /// disagreed with the table would read a different tree for every class
    /// after the first.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<HistGradientBoostingClassifier<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = EnsembleFile::parse(&raw, PERSIST_TAG)?;
        let n_features = file.scalar_usize("n_features_in_")?;
        if n_features == 0 {
            return Err(PersistError::InconsistentGeometry {
                reason: "'n_features_in_' is 0; a fitted booster has at least one feature"
                    .to_string(),
            });
        }
        let tables = read_node_tables::<F>(&file, n_features)?;

        let n_iters = file.scalar_usize("n_iters_")?;
        let k = file.scalar_usize("k_")?;
        if k == 0 || n_iters == 0 || n_iters * k != tables.n_trees {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "'n_iters_' is {n_iters} and 'k_' is {k}, but the node tables hold {} \
                     trees",
                    tables.n_trees
                ),
            });
        }

        let leaf_value = read_leaf_table::<F>(
            &file,
            LEAF_VALUE_NAME,
            tables.n_trees,
            tables.total_nodes,
            1,
        )?;
        let baseline_v = file.tensor(BASELINE_NAME)?;
        expect_len(
            BASELINE_NAME,
            shape_1d(&baseline_v, BASELINE_NAME)?,
            k,
            "entries",
        )?;
        let baseline = as_floats::<F>(&baseline_v, BASELINE_NAME)?.into_owned();

        // `classes_` is what turns a raw-score argmax back into the label the
        // caller trained with. Its length is cross-checked against `k`: a
        // multiclass booster fits one tree column per class, and a binary one
        // fits a single column for two classes — the same asymmetry
        // `RidgeClassifier` has, and the only relation that ties the label table
        // to the tree axis.
        let classes_v = file.tensor(CLASSES_NAME)?;
        let n_classes = shape_1d(&classes_v, CLASSES_NAME)?;
        let expected_k = if n_classes == 2 { 1 } else { n_classes };
        if n_classes < 2 || k != expected_k {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "tensor '{CLASSES_NAME}' holds {n_classes} labels, which implies \
                     {expected_k} raw-score column(s), but 'k_' is {k}"
                ),
            });
        }
        let classes_: Vec<i32> = as_i64(&classes_v, CLASSES_NAME)?
            .iter()
            .map(|&v| {
                i32::try_from(v).map_err(|_| PersistError::InconsistentGeometry {
                    reason: format!(
                        "tensor '{CLASSES_NAME}' holds the label {v}, which does not fit the \
                         i32 the booster's kernels consume"
                    ),
                })
            })
            .collect::<Result<_, _>>()?;

        let model_ = HgbModel::from_parts(
            pool,
            &tables.split_feature,
            &tables.threshold,
            &tables.is_leaf,
            &leaf_value,
            &baseline,
            n_iters,
            k,
            file.scalar_usize("n_classes_")?,
            tables.max_depth,
            n_features,
        )
        .map_err(|e| PersistError::InconsistentGeometry {
            reason: format!("the node tables do not assemble into a booster: {e}"),
        })?;

        Ok(HistGradientBoostingClassifier {
            device: Device::from_name(file.scalar_str("param:device")?).ok_or(
                PersistError::BadMetadata {
                    key: "param:device",
                },
            )?,
            max_iter: file.scalar_usize("param:max_iter")?,
            learning_rate: file.scalar_f64("param:learning_rate")?,
            max_depth: file.scalar_usize("param:max_depth")?,
            n_bins: file.scalar_usize("param:n_bins")?,
            l2_regularization: file.scalar_f64("param:l2_regularization")?,
            min_samples_leaf: file.scalar_usize("param:min_samples_leaf")?,
            model_: Some(model_),
            classes_,
            n_classes_: n_classes,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for HistGradientBoostingClassifier<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = HistGradientBoostingClassifier<F, Fitted>;

    /// Boost on `(x, y)` (y = integer-valued `F` class labels), CONSUMING
    /// `self`. The device loop is launch-only; the host syncs are the
    /// bin-edge quantile pass and the label ingestion.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<HistGradientBoostingClassifier<F, Fitted>, AlgoError> {
        let (n, _d) = shape;
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "hist_gradient_boosting_classifier",
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
        let (classes, y_idx) = ingest_labels::<F>("hist_gradient_boosting_classifier", &y_host)?;
        let n_classes = classes.len();

        let params = HgbParams {
            max_iter: self.max_iter,
            max_depth: self.max_depth,
            n_bins: self.n_bins,
            learning_rate: self.learning_rate,
            l2_regularization: self.l2_regularization,
            min_samples_leaf: self.min_samples_leaf,
        };
        let model = hgb_fit_class::<F>(pool, x, shape, &y_idx, n_classes, &params, self.device)?;

        Ok(HistGradientBoostingClassifier {
            max_iter: self.max_iter,
            device: self.device,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            n_bins: self.n_bins,
            l2_regularization: self.l2_regularization,
            min_samples_leaf: self.min_samples_leaf,
            model_: Some(model),
            classes_: classes,
            n_classes_: n_classes,
            _state: PhantomData,
        })
    }
}

impl<F> PredictProba<F> for HistGradientBoostingClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// `n_query × n_classes` device probabilities (sigmoid of the binary raw
    /// score / softmax of the multiclass raw scores — the sklearn
    /// `predict_proba` link functions; rows sum to 1).
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
        Ok(hgb_predict_proba::<F>(pool, model, x, shape)?)
    }
}

impl<F> PredictLabels<F> for HistGradientBoostingClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// `predict = argmax(predict_proba)` with the lowest-class-index
    /// tie-break, mapped back through `classes_` (CR-03).
    ///
    /// The argmax runs HOST-side over ONE metered proba readback — never the
    /// per-row `argmax_rows` prim (the RF predict sync-bound lesson).
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
