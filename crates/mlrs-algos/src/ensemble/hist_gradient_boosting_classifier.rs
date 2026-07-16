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

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::hist_gradient_boosting::{
    hgb_fit_class, hgb_predict_proba, HgbModel, HgbParams,
};
use mlrs_backend::runtime::ActiveRuntime;

use mlrs_core::{host_to_f64, PrimError};

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
        let model = hgb_fit_class::<F>(pool, x, shape, &y_idx, n_classes, &params)?;

        Ok(HistGradientBoostingClassifier {
            max_iter: self.max_iter,
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
