//! `RandomForestRegressor` (ENSEMBLE-01) — variance-reduction random forest
//! over the launch-only batched forest primitive (`prims::random_forest`).
//!
//! ## Surface (typestate, D-03/D-05)
//! Builder-fronted `RandomForestRegressor<F, S = Unfit>`; [`Fit::fit`]
//! consumes `self` and returns the `Fitted` sibling holding the
//! device-resident [`RfModel`]. [`Predict::predict`] is the forest MEAN of
//! the reached leaves' stored mean targets (the sklearn averaging form).
//!
//! Split quality is the sklearn MSE proxy `(Σ_l y)²/n_l + (Σ_r y)²/n_r`
//! (maximized), computed from a two-slot (`Σw`, `Σwy`) cumulative histogram.
//! Defaults mirror sklearn's regressor (`max_features = 1.0` → all features)
//! with the mlrs-bounded `max_depth = 10` / `n_bins = 32` deviations
//! (documented in `ensemble/mod.rs`).
//!
//! Tests live in `crates/mlrs-algos/tests/random_forest_regressor_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)]` module).

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::random_forest::{
    rf_fit_reg, rf_predict_reg, RfFitOutcome, RfModel, RfParams,
};
use mlrs_backend::runtime::ActiveRuntime;

use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, State, Unfit};

use super::ensemble_persist::{
    as_floats, expect_len, read_leaf_table, read_node_tables, shape_1d, widen_nodes,
    write_node_tables, AlignedBytes, EnsembleFile, EnsembleWriter, LoadModel, PersistError,
    SaveModel, TensorRef, LEAF_DIST_NAME, NODE_DECREASE_NAME,
};
use super::random_forest_classifier::validate_forest_hyperparams;
use super::MaxFeatures;

/// The `estimator` discriminator written into every `RandomForestRegressor`
/// file.
///
/// Load-bearing rather than decorative: the classifier's file holds the SAME
/// node tables at the same shapes and dtypes, and the only structural difference
/// is that it also carries a `classes_`. What `leaf_dist` MEANS differs
/// completely — a class distribution against a regression value — and nothing in
/// the geometry says which.
const PERSIST_TAG: &str = "random_forest_regressor";

/// The tensor holding the normalized per-feature importances, `[n_features]`.
const IMPORTANCES_NAME: &str = "feature_importances_";

/// sklearn defaults (single source, D-08); `max_depth=10` / `n_bins=32` are
/// the mlrs histogram-builder deviations.
const RF_REG_DEFAULT_N_ESTIMATORS: usize = 100;
const RF_REG_DEFAULT_MAX_DEPTH: usize = 10;
const RF_REG_DEFAULT_N_BINS: usize = 32;
const RF_REG_DEFAULT_MIN_SAMPLES_SPLIT: f64 = 2.0;
const RF_REG_DEFAULT_MIN_SAMPLES_LEAF: f64 = 1.0;
const RF_REG_DEFAULT_SEED: u64 = 42;

/// Random forest regressor (ENSEMBLE-01), generic over the float type and
/// lifecycle state. The fitted forest is device-resident (D-03).
pub struct RandomForestRegressor<F, S = Unfit>
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
    /// RF-IMP-01: normalized (sums to 1) length-`n_features` mean-decrease-in-
    /// impurity vector, empty until `fit`.
    feature_importances_: Vec<F>,
    /// RF-OOB-01: `Some(score)` once fitted with `oob_score=true`; `None`
    /// otherwise (including always on the `Unfit` state, before `fit`).
    oob_score_: Option<F>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> RandomForestRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfit regressor with the defaults above (D-08 single
    /// source; the builder `Default` re-derives from here).
    pub fn new() -> Self {
        Self {
            n_estimators: RF_REG_DEFAULT_N_ESTIMATORS,
            max_depth: RF_REG_DEFAULT_MAX_DEPTH,
            n_bins: RF_REG_DEFAULT_N_BINS,
            max_features: MaxFeatures::All,
            min_samples_split: RF_REG_DEFAULT_MIN_SAMPLES_SPLIT,
            min_samples_leaf: RF_REG_DEFAULT_MIN_SAMPLES_LEAF,
            bootstrap: true,
            oob_score: false,
            seed: RF_REG_DEFAULT_SEED,
            model_: None,
            feature_importances_: Vec::new(),
            oob_score_: None,
            _state: PhantomData,
        }
    }

    /// Start building from the defaults (D-08 single source).
    pub fn builder() -> RandomForestRegressorBuilder {
        RandomForestRegressorBuilder::default()
    }

    /// Decompose back into the builder (used by the builder `Default`).
    pub fn into_builder(self) -> RandomForestRegressorBuilder {
        RandomForestRegressorBuilder {
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

impl<F> Default for RandomForestRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> RandomForestRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
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

    /// RF-OOB-01: the out-of-bag score computed at fit time — R² of the
    /// OOB-tree-averaged prediction vs. training `y`. `Some(..)` iff the
    /// builder's `oob_score` flag was `true`; `None` otherwise (matches
    /// `RfFitOutcome::oob_score`'s own contract).
    pub fn oob_score(&self) -> Option<F> {
        self.oob_score_
    }

    /// SHAP-01: path-dependent TreeSHAP values, self-consistency-gated (see
    /// `tree_shap` module docs). `x_train_host`/`query_host` are host
    /// row-major `f64` buffers. Returns `(phi, expected_value)`: `phi` is
    /// `n_query × n_features × 1`; `expected_value` is length `1`. `Σ_f
    /// phi[q, f, 0] + expected_value[0] == predict(query)[q]` exactly.
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

/// Builder for [`RandomForestRegressor`] (D-01). `Default` re-derives the
/// defaults from [`RandomForestRegressor::new`] (D-08 single source).
#[derive(Debug, Clone, Copy)]
pub struct RandomForestRegressorBuilder {
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

impl Default for RandomForestRegressorBuilder {
    fn default() -> Self {
        RandomForestRegressor::<f64, Unfit>::new().into_builder()
    }
}

impl RandomForestRegressorBuilder {
    /// Set the tree count `n_estimators` (`>= 1`).
    pub fn n_estimators(mut self, v: usize) -> Self {
        self.n_estimators = v;
        self
    }

    /// Set the depth bound (`1..=16`; documented deviation from sklearn).
    pub fn max_depth(mut self, v: usize) -> Self {
        self.max_depth = v;
        self
    }

    /// Set the histogram bin count per feature (`2..=256`).
    pub fn n_bins(mut self, v: usize) -> Self {
        self.n_bins = v;
        self
    }

    /// Set the per-node feature-subsample policy (regressor default
    /// [`MaxFeatures::All`], sklearn `max_features=1.0`).
    pub fn max_features(mut self, v: MaxFeatures) -> Self {
        self.max_features = v;
        self
    }

    /// Set `min_samples_split` (`>= 2`).
    pub fn min_samples_split(mut self, v: f64) -> Self {
        self.min_samples_split = v;
        self
    }

    /// Set `min_samples_leaf` (`>= 1`).
    pub fn min_samples_leaf(mut self, v: f64) -> Self {
        self.min_samples_leaf = v;
        self
    }

    /// Enable/disable per-tree bootstrap resampling.
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

    /// Set the host RNG seed (fully deterministic across runs and backends).
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }

    /// Build the (unfit) estimator, validating every data-INDEPENDENT
    /// hyperparameter (D-08).
    pub fn build<F>(self) -> Result<RandomForestRegressor<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        validate_forest_hyperparams(
            "random_forest_regressor",
            self.n_estimators,
            self.max_depth,
            self.n_bins,
            self.max_features,
            self.min_samples_split,
            self.min_samples_leaf,
        )?;
        if self.oob_score && !self.bootstrap {
            return Err(BuildError::OobRequiresBootstrap {
                estimator: "random_forest_regressor",
            });
        }
        Ok(RandomForestRegressor {
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
            feature_importances_: Vec::new(),
            oob_score_: None,
            _state: PhantomData,
        })
    }
}

impl<F> SaveModel for RandomForestRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted forest to `path` as a safetensors file.
    ///
    /// The same layout as the classifier minus `classes_`, and with `leaf_dist`
    /// carrying ONE value per node rather than one per class:
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `split_feature` / `is_leaf` | `U64` | `[n_trees, total_nodes]` |
    /// | `threshold` / `leaf_dist` / `node_decrease` | `F` (`F32`/`F64`) | `[n_trees, total_nodes]` |
    /// | `feature_importances_` | `F` | `[n_features]` |
    /// | `oob_score_` | `__metadata__` scalar, optional | — |
    /// | nine `param:*` scalars | `__metadata__` | — |
    ///
    /// See [`ensemble_persist`](super::ensemble_persist) for the complete-layout
    /// decision and the size trade it makes.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let model = self.model_.as_ref().ok_or(PersistError::MissingState {
            estimator: PERSIST_TAG,
            field: "model_",
        })?;
        // Bound BEFORE the writer, which borrows every payload.
        let split_feature = widen_nodes(&model.split_feature_host(pool));
        let is_leaf = widen_nodes(&model.is_leaf_host(pool));
        let threshold = model.threshold_host(pool);
        let leaf_dist = model.leaf_dist_host(pool);
        let node_decrease = model.node_decrease_host(pool);
        let (n_trees, total_nodes) = (model.n_trees(), model.total_nodes());

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
            TensorRef::floats(&leaf_dist, vec![n_trees, total_nodes, model.n_values()])?,
        );
        w.tensor(
            NODE_DECREASE_NAME,
            TensorRef::floats(&node_decrease, vec![n_trees, total_nodes])?,
        );
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

impl<F> LoadModel for RandomForestRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read the forest back from `path`, re-uploading every node table to
    /// `pool`.
    ///
    /// `n_values` is recovered from `leaf_dist`'s length rather than assumed to
    /// be 1: a multi-output regression forest holds one value per target per
    /// node, and the traversal gathers by that stride.
    fn load(
        pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<RandomForestRegressor<F, Fitted>, PersistError> {
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

        // `n_values` comes off the leaf table's own shape — its trailing extent
        // when rank-3, or 1 when the fit was single-target.
        let leaf_v = file.tensor(LEAF_DIST_NAME)?;
        let n_values = match leaf_v.shape() {
            [_, _, v] => *v,
            [_, _] => 1,
            other => {
                return Err(PersistError::InconsistentGeometry {
                    reason: format!(
                        "tensor '{LEAF_DIST_NAME}' declares shape {other:?}; a leaf table is                          rank-2 or rank-3"
                    ),
                })
            }
        };
        if n_values == 0 {
            return Err(PersistError::InconsistentGeometry {
                reason: format!("tensor '{LEAF_DIST_NAME}' declares 0 outputs per leaf"),
            });
        }
        let leaf_dist = read_leaf_table::<F>(
            &file,
            LEAF_DIST_NAME,
            tables.n_trees,
            tables.total_nodes,
            n_values,
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
            n_values,
        )
        .map_err(|e| PersistError::InconsistentGeometry {
            reason: format!("the node tables do not assemble into a forest: {e}"),
        })?;

        Ok(RandomForestRegressor {
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
            feature_importances_,
            oob_score_: file.scalar_opt_f64("oob_score_")?.map(f64_to_host::<F>),
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for RandomForestRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = RandomForestRegressor<F, Fitted>;

    /// Grow the forest on `(x, y)` (continuous `F` target), CONSUMING `self`.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<RandomForestRegressor<F, Fitted>, AlgoError> {
        let (n, d) = shape;
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "random_forest_regressor",
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
        // RF-IMP-01 (TASK-03) / RF-OOB-01 (TASK-07): `rf_fit_reg` returns
        // `RfFitOutcome<F>` (TASK-01); destructure `feature_importances`
        // AND `oob_score` alongside `model`, mirroring the classifier's
        // `fit()` (TASK-02/TASK-06).
        let RfFitOutcome {
            model,
            feature_importances,
            oob_score: oob_score_,
        } = rf_fit_reg::<F>(pool, x, shape, y, &params)?;

        Ok(RandomForestRegressor {
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
            _state: PhantomData,
        })
    }
}

impl<F> Predict<F> for RandomForestRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Length-`n_query` forest mean of the reached leaves' mean targets
    /// (device-computed).
    fn predict(
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
        Ok(rf_predict_reg::<F>(pool, model, x, shape)?)
    }
}
