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

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::random_forest::{
    rf_fit_reg, rf_predict_reg, RfFitOutcome, RfModel, RfParams,
};
use mlrs_backend::runtime::ActiveRuntime;

use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, State, Unfit};

use super::random_forest_classifier::validate_forest_hyperparams;
use super::MaxFeatures;

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
