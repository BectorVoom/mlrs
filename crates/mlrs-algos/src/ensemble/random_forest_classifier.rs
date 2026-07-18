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

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::random_forest::{
    rf_fit_class, rf_predict_proba, RfFitOutcome, RfModel, RfParams, RF_MAX_DEPTH_CAP,
};
use mlrs_backend::runtime::ActiveRuntime;

use mlrs_core::{host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, PredictLabels, PredictProba, State, Unfit};

use super::MaxFeatures;

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
