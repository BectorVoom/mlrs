//! `HistGradientBoostingRegressor` (GBT-01) — squared-error gradient boosting
//! over the launch-only histogram-tree primitive
//! (`prims::hist_gradient_boosting`).
//!
//! ## Surface (typestate, D-03/D-05)
//! Builder-fronted `HistGradientBoostingRegressor<F, S = Unfit>`; [`Fit::fit`]
//! consumes `self` and returns the `Fitted` sibling holding the
//! device-resident [`HgbModel`]. `predict` returns the raw ensemble scores
//! (squared error has the identity link): `baseline mean + Σ shrunk leaf
//! values`.
//!
//! Fits are fully deterministic: HistGradientBoosting has no bootstrap, no
//! feature subsampling and no RNG (unlike the forest there is no `seed`).
//!
//! Tests live in
//! `crates/mlrs-algos/tests/hist_gradient_boosting_regressor_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)]` module).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::hist_gradient_boosting::{
    hgb_fit_reg, hgb_predict_reg, HgbModel, HgbParams,
};
use mlrs_backend::runtime::ActiveRuntime;

use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, State, Unfit};

/// sklearn defaults (single source, D-08): `learning_rate=0.1`,
/// `max_iter=100`, `l2_regularization=0`, `min_samples_leaf=20`.
/// Two documented mlrs deviations: `max_depth=6` (sklearn's default tree is
/// leaf-wise with `max_leaf_nodes=31`; a depth-6 complete tree has up to 64
/// leaves — the closest depth-bounded analogue) and `n_bins=64` (sklearn
/// `max_bins=255`; the device pays zero/cumulate/score work over the FULL
/// `nodes × features × bins` lattice each level, so the histogram default is
/// kept moderate — the RF `n_bins=32` precedent. Set `.n_bins(255)` for
/// sklearn-exact candidate sets, as the oracle tests do).
const HGB_REG_DEFAULT_MAX_ITER: usize = 100;
const HGB_REG_DEFAULT_LEARNING_RATE: f64 = 0.1;
const HGB_REG_DEFAULT_MAX_DEPTH: usize = 6;
const HGB_REG_DEFAULT_N_BINS: usize = 64;
const HGB_REG_DEFAULT_L2: f64 = 0.0;
const HGB_REG_DEFAULT_MIN_SAMPLES_LEAF: usize = 20;

/// HistGradientBoosting regressor (GBT-01), generic over the float type and
/// lifecycle state. The fitted ensemble is device-resident (D-03).
pub struct HistGradientBoostingRegressor<F, S = Unfit>
where
    F: Float + CubeElement + Pod,
    S: State,
{
    max_iter: usize,
    learning_rate: f64,
    max_depth: usize,
    n_bins: usize,
    l2_regularization: f64,
    min_samples_leaf: usize,
    /// The fitted device-resident ensemble, `None` until `fit`.
    model_: Option<HgbModel<F>>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> HistGradientBoostingRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an unfit regressor with the defaults above (D-08 single
    /// source; the builder `Default` re-derives from here).
    pub fn new() -> Self {
        Self {
            max_iter: HGB_REG_DEFAULT_MAX_ITER,
            learning_rate: HGB_REG_DEFAULT_LEARNING_RATE,
            max_depth: HGB_REG_DEFAULT_MAX_DEPTH,
            n_bins: HGB_REG_DEFAULT_N_BINS,
            l2_regularization: HGB_REG_DEFAULT_L2,
            min_samples_leaf: HGB_REG_DEFAULT_MIN_SAMPLES_LEAF,
            model_: None,
            _state: PhantomData,
        }
    }

    /// Start building from the defaults (D-08 single source).
    pub fn builder() -> HistGradientBoostingRegressorBuilder {
        HistGradientBoostingRegressorBuilder::default()
    }

    /// Decompose back into the builder (used by the builder `Default`).
    pub fn into_builder(self) -> HistGradientBoostingRegressorBuilder {
        HistGradientBoostingRegressorBuilder {
            max_iter: self.max_iter,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            n_bins: self.n_bins,
            l2_regularization: self.l2_regularization,
            min_samples_leaf: self.min_samples_leaf,
        }
    }
}

impl<F> Default for HistGradientBoostingRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> HistGradientBoostingRegressor<F, Fitted>
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

    /// Borrow the fitted device ensemble (for the perf harness / debugging).
    pub fn model(&self) -> &HgbModel<F> {
        self.model_
            .as_ref()
            .expect("model_ is Some by construction on the Fitted state")
    }
}

/// Builder for [`HistGradientBoostingRegressor`] (D-01). `Default` re-derives
/// the defaults from [`HistGradientBoostingRegressor::new`] (D-08).
#[derive(Debug, Clone, Copy)]
pub struct HistGradientBoostingRegressorBuilder {
    max_iter: usize,
    learning_rate: f64,
    max_depth: usize,
    n_bins: usize,
    l2_regularization: f64,
    min_samples_leaf: usize,
}

impl Default for HistGradientBoostingRegressorBuilder {
    fn default() -> Self {
        HistGradientBoostingRegressor::<f64, Unfit>::new().into_builder()
    }
}

impl HistGradientBoostingRegressorBuilder {
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
    /// contract, not the forest's weighted form).
    pub fn min_samples_leaf(mut self, v: usize) -> Self {
        self.min_samples_leaf = v;
        self
    }

    /// Build the (unfit) estimator, validating every data-INDEPENDENT
    /// hyperparameter (D-08).
    pub fn build<F>(self) -> Result<HistGradientBoostingRegressor<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        validate_hgb_hyperparams(
            "hist_gradient_boosting_regressor",
            self.max_iter,
            self.learning_rate,
            self.max_depth,
            self.n_bins,
            self.l2_regularization,
            self.min_samples_leaf,
        )?;
        Ok(HistGradientBoostingRegressor {
            max_iter: self.max_iter,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            n_bins: self.n_bins,
            l2_regularization: self.l2_regularization,
            min_samples_leaf: self.min_samples_leaf,
            model_: None,
            _state: PhantomData,
        })
    }
}

/// Shared builder-time hyperparameter validation (regressor + classifier).
pub(crate) fn validate_hgb_hyperparams(
    estimator: &'static str,
    max_iter: usize,
    learning_rate: f64,
    max_depth: usize,
    n_bins: usize,
    l2_regularization: f64,
    min_samples_leaf: usize,
) -> Result<(), BuildError> {
    if max_iter == 0 {
        return Err(BuildError::InvalidMaxIter {
            estimator,
            max_iter,
        });
    }
    if !(learning_rate.is_finite() && learning_rate > 0.0) {
        return Err(BuildError::InvalidLearningRate {
            estimator,
            learning_rate,
        });
    }
    if max_depth == 0 || max_depth > mlrs_backend::prims::random_forest::RF_MAX_DEPTH_CAP {
        return Err(BuildError::InvalidMaxDepth {
            estimator,
            max_depth,
        });
    }
    if n_bins < 2 || n_bins > 256 {
        return Err(BuildError::InvalidNBins { estimator, n_bins });
    }
    if !(l2_regularization.is_finite() && l2_regularization >= 0.0) {
        return Err(BuildError::InvalidL2Regularization {
            estimator,
            l2_regularization,
        });
    }
    if min_samples_leaf == 0 {
        return Err(BuildError::InvalidMinSamplesForest {
            estimator,
            which: "min_samples_leaf",
            value: 0.0,
        });
    }
    Ok(())
}

impl<F> Fit<F> for HistGradientBoostingRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = HistGradientBoostingRegressor<F, Fitted>;

    /// Boost on `(x, y)` (continuous `F` target), CONSUMING `self`. The
    /// device loop is launch-only; the host syncs are the bin-edge quantile
    /// pass and the baseline mean (see `prims::hist_gradient_boosting`).
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<HistGradientBoostingRegressor<F, Fitted>, AlgoError> {
        let (n, _d) = shape;
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "hist_gradient_boosting_regressor",
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

        let params = HgbParams {
            max_iter: self.max_iter,
            max_depth: self.max_depth,
            n_bins: self.n_bins,
            learning_rate: self.learning_rate,
            l2_regularization: self.l2_regularization,
            min_samples_leaf: self.min_samples_leaf,
        };
        let model = hgb_fit_reg::<F>(pool, x, shape, y, &params)?;

        Ok(HistGradientBoostingRegressor {
            max_iter: self.max_iter,
            learning_rate: self.learning_rate,
            max_depth: self.max_depth,
            n_bins: self.n_bins,
            l2_regularization: self.l2_regularization,
            min_samples_leaf: self.min_samples_leaf,
            model_: Some(model),
            _state: PhantomData,
        })
    }
}

impl<F> Predict<F> for HistGradientBoostingRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Length-`n_query` raw ensemble scores (device-computed): baseline mean
    /// plus the sum of every stage tree's shrunk leaf value.
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
        Ok(hgb_predict_reg::<F>(pool, model, x, shape)?)
    }
}
