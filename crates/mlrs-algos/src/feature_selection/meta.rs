//! The four META-selectors (FSEL-01) — `SelectFromModel`, `RFE`, `RFECV`,
//! `SequentialFeatureSelector` — and the two traits that stand in for sklearn's
//! `estimator` parameter.
//!
//! ## The design problem, and the answer
//! sklearn's meta-selectors take `estimator` — any duck-typed object with `fit`
//! and either `coef_`/`feature_importances_` (for `SelectFromModel`/`RFE`) or a
//! `score` (for `RFECV`/`SequentialFeatureSelector`). Rust has no duck typing, so
//! the parameter has to become a trait. Two traits, matching the two things the
//! selectors actually ASK of the estimator:
//!
//! * [`ImportanceEstimator`] — "fit on this column subset and tell me each
//!   column's importance". This is `SelectFromModel` and `RFE`'s entire
//!   requirement, and it corresponds one-to-one with sklearn's
//!   `_get_feature_importances(estimator, importance_getter)`.
//! * [`FoldScorer`] — "fit on this train split, score on this test split". This
//!   is `RFECV` and `SequentialFeatureSelector`'s entire requirement, and it
//!   corresponds to sklearn's `scoring` + `cv` pair collapsed into the one
//!   operation they are always used to perform together.
//!
//! [`FnImportance`] / [`FnScorer`] wrap plain closures, so a caller does not have
//! to declare a type to plug in a model — which keeps the surface as open as
//! sklearn's duck typing, without a blanket `impl` over every `Fn` (which would
//! make the traits un-implementable for any callable type a user owns).
//!
//! `importance_getter` becomes [`ImportanceGetter`]: sklearn's `"auto"` /
//! attribute-path / callable choice, where the Rust equivalent of "which
//! attribute" is "which variant of [`Importances`] the estimator returned".
//!
//! ## Everything here is HOST code, and drives DEVICE estimators
//! A meta-selector's own arithmetic is a handful of `argsort`s and comparisons
//! over `n_features` values; all of its cost is in the inner estimator's `fit`,
//! which is whatever the caller plugged in — typically an mlrs estimator running
//! its own device kernels. So the selectors are host drivers by construction,
//! exactly as sklearn's are Python drivers over compiled estimators. The column
//! subsetting they do between fits is a host slice of the `f64` design.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{host_to_f64, PrimError};

use crate::error::AlgoError;
use crate::typestate::{validate_geometry, Fit, Fitted, Transform, Unfit};

use super::selector::{inverse_transform_selected, transform_selected, Selector};

// ===========================================================================
// Importances — sklearn's `coef_` / `feature_importances_` / importance_getter
// ===========================================================================

/// What a fitted inner estimator reports about its features — the Rust stand-in
/// for reading `coef_` or `feature_importances_` off a sklearn estimator.
#[derive(Debug, Clone, PartialEq)]
pub enum Importances {
    /// A 1-D `coef_` (a regressor, or a binary classifier's single row) or a
    /// `feature_importances_` vector: one value per column.
    Flat(Vec<f64>),
    /// A 2-D `coef_`, `rows × n_features` row-major — a multiclass or
    /// multi-output linear model.
    ///
    /// The distinction from [`Importances::Flat`] is NOT cosmetic: sklearn's
    /// `transform_func` reduces the two differently
    /// (`abs` vs a column NORM for `"norm"`, `x²` vs a column SUM OF SQUARES for
    /// `"square"`), so collapsing a `(1, d)` coef into a flat one would change
    /// the reduction that follows — and a `(1, d)` coef is exactly what
    /// sklearn's binary `LogisticRegression` produces.
    Rows {
        /// Number of coefficient rows (classes or outputs).
        rows: usize,
        /// `rows × n_features` row-major values.
        values: Vec<f64>,
    },
}

impl Importances {
    /// Column count, whichever shape this is.
    fn n_features(&self) -> usize {
        match self {
            Self::Flat(v) => v.len(),
            Self::Rows { rows, values } => {
                if *rows == 0 {
                    0
                } else {
                    values.len() / rows
                }
            }
        }
    }

    /// sklearn's `_get_feature_importances(transform_func="norm",
    /// norm_order=..)` — `SelectFromModel`'s reduction.
    ///
    /// 1-D: elementwise `abs`. 2-D: the `norm_order`-norm DOWN each column
    /// (`np.linalg.norm(importances, axis=0, ord=norm_order)`).
    fn to_norm(&self, norm_order: f64) -> Vec<f64> {
        match self {
            Self::Flat(v) => v.iter().map(|x| x.abs()).collect(),
            Self::Rows { rows, values } => {
                let d = self.n_features();
                (0..d)
                    .map(|c| {
                        let col = (0..*rows).map(|r| values[r * d + c].abs());
                        if norm_order.is_infinite() {
                            // `ord=inf` is the max-abs norm.
                            col.fold(0.0f64, f64::max)
                        } else if norm_order == 1.0 {
                            col.sum()
                        } else {
                            col.map(|v| v.powf(norm_order))
                                .sum::<f64>()
                                .powf(1.0 / norm_order)
                        }
                    })
                    .collect()
            }
        }
    }

    /// sklearn's `_get_feature_importances(transform_func="square")` — `RFE`'s
    /// reduction.
    ///
    /// 1-D: elementwise `x²`. 2-D: `safe_sqr(importances).sum(axis=0)`, i.e. the
    /// column sum of squares.
    fn to_square(&self) -> Vec<f64> {
        match self {
            Self::Flat(v) => v.iter().map(|x| x * x).collect(),
            Self::Rows { rows, values } => {
                let d = self.n_features();
                (0..d)
                    .map(|c| (0..*rows).map(|r| values[r * d + c].powi(2)).sum())
                    .collect()
            }
        }
    }
}

/// sklearn's `importance_getter` — WHICH importance an inner estimator's report
/// should be read from.
#[derive(Clone)]
pub enum ImportanceGetter {
    /// sklearn's `"auto"`: take whatever the estimator reports.
    ///
    /// In sklearn this means "`coef_` if present, else `feature_importances_`,
    /// else raise". In Rust the estimator returns ONE [`Importances`] value, so
    /// the choice has already been made by the [`ImportanceEstimator`] impl —
    /// which is the same decision, relocated from the selector to the estimator
    /// where the type information is.
    Auto,
    /// sklearn's callable form: post-process the reported importances.
    ///
    /// This covers the attribute-PATH form too (`"named_steps.svc.coef_"`), which
    /// exists in sklearn only because the estimator may be a `Pipeline`; a Rust
    /// [`ImportanceEstimator`] impl for a pipeline reports its inner model's
    /// importances directly, so there is no path to walk.
    #[allow(clippy::type_complexity)]
    Custom(std::sync::Arc<dyn Fn(&Importances) -> Vec<f64> + Send + Sync>),
}

impl std::fmt::Debug for ImportanceGetter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Custom(_) => f.write_str("<custom importance_getter>"),
        }
    }
}

impl Default for ImportanceGetter {
    fn default() -> Self {
        Self::Auto
    }
}

impl ImportanceGetter {
    /// Apply the getter, then sklearn's `transform_func`.
    ///
    /// A `Custom` getter's output REPLACES the raw importances and the
    /// `transform_func` is applied to it as a flat vector — sklearn's order
    /// (`importances = getter(estimator)` then `transform_func(importances)`).
    fn resolve(&self, imp: &Importances, square: bool, norm_order: f64) -> Vec<f64> {
        match self {
            Self::Auto => {
                if square {
                    imp.to_square()
                } else {
                    imp.to_norm(norm_order)
                }
            }
            Self::Custom(f) => {
                let flat = Importances::Flat(f(imp));
                if square {
                    flat.to_square()
                } else {
                    flat.to_norm(norm_order)
                }
            }
        }
    }
}

// ===========================================================================
// The two estimator traits
// ===========================================================================

/// An inner estimator a meta-selector can fit on a column subset and ask for
/// per-feature importances — sklearn's `estimator` parameter for
/// `SelectFromModel` and `RFE`.
///
/// `x` is row-major `n × d` where `d` is the SUBSET width, so the returned
/// importances are indexed by position within the subset, not by original column.
/// The selector maps them back.
pub trait ImportanceEstimator: Send + Sync {
    /// Fit on `(x, y)` and report each of the `d` columns' importance.
    fn fit_importances(
        &self,
        x: &[f64],
        y: &[f64],
        n: usize,
        d: usize,
    ) -> Result<Importances, AlgoError>;
}

/// A closure as an [`ImportanceEstimator`].
///
/// A newtype rather than a blanket `impl<T: Fn(..)>`: a blanket impl over all
/// callables would make the trait un-implementable for any user type that also
/// happens to be callable, and would leak into every future impl's coherence.
pub struct FnImportance<T>(pub T);

impl<T> ImportanceEstimator for FnImportance<T>
where
    T: Fn(&[f64], &[f64], usize, usize) -> Result<Importances, AlgoError> + Send + Sync,
{
    fn fit_importances(
        &self,
        x: &[f64],
        y: &[f64],
        n: usize,
        d: usize,
    ) -> Result<Importances, AlgoError> {
        (self.0)(x, y, n, d)
    }
}

/// An inner estimator + metric a meta-selector can train on one split and score
/// on another — sklearn's `estimator` + `scoring` + `cv` triple, collapsed into
/// the single operation they are always used together to perform.
///
/// Higher is better, as every sklearn scorer is (`scoring='neg_mean_squared_error'`
/// exists precisely so that convention holds).
pub trait FoldScorer: Send + Sync {
    /// Fit on the train split and return the test-split score. Both matrices are
    /// row-major with the same `d` column subset.
    fn fit_score(
        &self,
        x_train: &[f64],
        y_train: &[f64],
        n_train: usize,
        x_test: &[f64],
        y_test: &[f64],
        n_test: usize,
        d: usize,
    ) -> Result<f64, AlgoError>;
}

/// A closure as a [`FoldScorer`] — the [`FnImportance`] pattern, same rationale.
pub struct FnScorer<T>(pub T);

impl<T> FoldScorer for FnScorer<T>
where
    T: Fn(&[f64], &[f64], usize, &[f64], &[f64], usize, usize) -> Result<f64, AlgoError>
        + Send
        + Sync,
{
    fn fit_score(
        &self,
        x_train: &[f64],
        y_train: &[f64],
        n_train: usize,
        x_test: &[f64],
        y_test: &[f64],
        n_test: usize,
        d: usize,
    ) -> Result<f64, AlgoError> {
        (self.0)(x_train, y_train, n_train, x_test, y_test, n_test, d)
    }
}

// ===========================================================================
// Cross-validation splitting
// ===========================================================================

/// sklearn's `cv` parameter, narrowed to what `check_cv` can produce.
#[derive(Debug, Clone, PartialEq)]
pub enum Cv {
    /// `cv=None` or `cv=<int>`: `KFold(n)` for a regressor, `StratifiedKFold(n)`
    /// for a classifier — which is what `check_cv(cv, y, classifier=..)` decides.
    /// `None` is `Folds(5)`, sklearn's default.
    Folds {
        /// Fold count.
        n_splits: usize,
        /// Stratify by `y` (sklearn's `classifier=is_classifier(estimator)`).
        stratified: bool,
    },
    /// An explicit `(train_indices, test_indices)` list — sklearn's
    /// "iterable yielding splits" form.
    Explicit(Vec<(Vec<usize>, Vec<usize>)>),
}

impl Default for Cv {
    fn default() -> Self {
        Self::Folds {
            n_splits: 5,
            stratified: false,
        }
    }
}

impl Cv {
    /// Materialise the `(train, test)` index pairs for `n` samples with target
    /// `y`.
    ///
    /// The UNSHUFFLED `KFold` and `StratifiedKFold` layouts are reproduced
    /// exactly, because they are deterministic and therefore comparable to
    /// sklearn without any RNG question:
    ///
    /// * `KFold(shuffle=False)` — `n % k` folds of size `n/k + 1` first, then
    ///   folds of size `n/k`, taken as CONTIGUOUS index blocks.
    /// * `StratifiedKFold(shuffle=False)` — each class's members, in index
    ///   order, dealt round-robin into the `k` folds. sklearn implements this by
    ///   assigning `np.arange(count) % k` within each class (via a per-class
    ///   `KFold` over that class's block), which for the unshuffled case is
    ///   exactly a round-robin deal.
    ///
    /// `shuffle=True` is NOT offered: it would need numpy's MT19937 permutation
    /// to be comparable, and unlike `mutual_info_*`'s noise (see
    /// [`super::numpy_rng`]) a shuffled CV split changes the SCORE rather than
    /// breaking a tie, so an approximate match would be misleading. A caller
    /// wanting a shuffled split passes [`Cv::Explicit`] with the indices it
    /// wants — which is also how the Python shim forwards a sklearn splitter
    /// object, so no capability is lost at the boundary a user sees.
    pub fn splits(&self, n: usize, y: &[f64]) -> Result<Vec<(Vec<usize>, Vec<usize>)>, AlgoError> {
        match self {
            Self::Explicit(v) => Ok(v.clone()),
            Self::Folds {
                n_splits,
                stratified,
            } => {
                if *n_splits < 2 || *n_splits > n {
                    return Err(AlgoError::InvalidSelectorParam {
                        estimator: "cv",
                        param: "n_splits",
                        value: *n_splits as f64,
                        reason: "must be in 2..=n_samples",
                    });
                }
                let fold_of: Vec<usize> = if *stratified {
                    let (labels, classes) = super::score::class_indices(y);
                    let mut per_class = vec![0usize; classes.len()];
                    labels
                        .iter()
                        .map(|&l| {
                            let f = per_class[l as usize] % n_splits;
                            per_class[l as usize] += 1;
                            f
                        })
                        .collect()
                } else {
                    // Contiguous blocks: the first `n % k` folds get one extra.
                    let base = n / n_splits;
                    let rem = n % n_splits;
                    let mut assign = Vec::with_capacity(n);
                    for f in 0..*n_splits {
                        let size = base + usize::from(f < rem);
                        assign.extend(std::iter::repeat_n(f, size));
                    }
                    assign
                };
                Ok((0..*n_splits)
                    .map(|f| {
                        let test: Vec<usize> = (0..n).filter(|&i| fold_of[i] == f).collect();
                        let train: Vec<usize> = (0..n).filter(|&i| fold_of[i] != f).collect();
                        (train, test)
                    })
                    .collect())
            }
        }
    }
}

/// Gather rows `idx` and columns `cols` out of a row-major `n × d` design.
fn subset(x: &[f64], d: usize, rows: &[usize], cols: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.len() * cols.len());
    for &r in rows {
        for &c in cols {
            out.push(x[r * d + c]);
        }
    }
    out
}

/// Gather all rows, columns `cols`, out of a row-major `n × d` design.
///
/// `pub` so a caller building a candidate column set by hand (e.g. to score one
/// outside a selector) uses the same gather the selectors do.
pub fn subset_cols(x: &[f64], n: usize, d: usize, cols: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(n * cols.len());
    for r in 0..n {
        for &c in cols {
            out.push(x[r * d + c]);
        }
    }
    out
}

/// Read a device design + target back as row-major `f64` host buffers, the form
/// every meta-selector's inner loop slices.
fn host_design<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: Option<&DeviceArray<ActiveRuntime, F>>,
    shape: (usize, usize),
    estimator: &'static str,
) -> Result<(Vec<f64>, Vec<f64>), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry(x, shape)?;
    let (n, _) = shape;
    let yd = y.ok_or(AlgoError::InvalidLabels {
        estimator,
        reason: "meta-selectors are supervised and require y".to_string(),
    })?;
    if yd.len() != n {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "y",
            rows: n,
            cols: 1,
            len: yd.len(),
        }));
    }
    Ok((
        x.to_host(pool).into_iter().map(host_to_f64).collect(),
        yd.to_host(pool).into_iter().map(host_to_f64).collect(),
    ))
}

// ===========================================================================
// SelectFromModel
// ===========================================================================

/// `SelectFromModel(threshold=..)`'s dual-typed threshold.
#[derive(Debug, Clone, PartialEq)]
pub enum Threshold {
    /// `threshold=None` — sklearn's estimator-dependent default. It resolves to
    /// `1e-5` for an L1-penalised model and `"mean"` otherwise, a decision
    /// sklearn makes by INSPECTING the estimator's class name and `penalty` /
    /// `l1_ratio` attributes (`_calculate_threshold`'s six `is_*` probes).
    ///
    /// Rust cannot introspect a trait object's class name, so the L1-ness is
    /// carried by the caller: [`Threshold::Default`] means `"mean"`, and an
    /// L1-penalised inner model passes [`Threshold::Value`]`(1e-5)`. The Python
    /// shim reproduces sklearn's probes exactly — it has the real estimator
    /// object to inspect — so a Python user sees no difference; a Rust caller
    /// gets an explicit choice instead of a hidden name-based one.
    Default,
    /// A numeric cutoff: keep features whose importance is `>=` it.
    Value(f64),
    /// `"mean"` / `"median"`, optionally scaled: `"1.25*mean"`.
    Scaled {
        /// The `<scale>*` prefix, `1.0` when absent.
        scale: f64,
        /// `true` for `median`, `false` for `mean`.
        median: bool,
    },
}

impl Default for Threshold {
    fn default() -> Self {
        Self::Default
    }
}

impl Threshold {
    /// Parse sklearn's string form: `"mean"`, `"median"`, `"<scale>*mean"`,
    /// `"<scale>*median"`.
    pub fn parse(s: &str) -> Result<Self, AlgoError> {
        let unknown = || AlgoError::UnknownSelectorOption {
            estimator: "select_from_model",
            param: "threshold",
            value: s.to_string(),
            expected: "mean, median, or '<scale>*mean' / '<scale>*median'",
        };
        if let Some((scale, reference)) = s.split_once('*') {
            let scale: f64 = scale.trim().parse().map_err(|_| unknown())?;
            let median = match reference.trim() {
                "median" => true,
                "mean" => false,
                _ => return Err(unknown()),
            };
            return Ok(Self::Scaled { scale, median });
        }
        match s {
            "mean" => Ok(Self::Scaled {
                scale: 1.0,
                median: false,
            }),
            "median" => Ok(Self::Scaled {
                scale: 1.0,
                median: true,
            }),
            _ => Err(unknown()),
        }
    }

    /// `_calculate_threshold` — resolve to the numeric cutoff for these scores.
    fn resolve(&self, scores: &[f64]) -> f64 {
        match self {
            // `None` → `"mean"` for a non-L1 estimator (see the variant docs).
            Self::Default => mean(scores),
            Self::Value(v) => *v,
            Self::Scaled { scale, median } => {
                scale
                    * if *median {
                        median_of(scores)
                    } else {
                        mean(scores)
                    }
            }
        }
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// `numpy.median` — the average of the two middle order statistics for an even
/// count, which is what sklearn's `np.median(importances)` computes.
fn median_of(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let m = s.len() / 2;
    if s.len() % 2 == 1 {
        s[m]
    } else {
        (s[m - 1] + s[m]) / 2.0
    }
}

/// `sklearn.feature_selection.SelectFromModel`.
pub struct SelectFromModel<F, E, S = Unfit> {
    estimator: E,
    threshold: Threshold,
    prefit: bool,
    norm_order: f64,
    max_features: Option<usize>,
    importance_getter: ImportanceGetter,
    /// `threshold_` — the resolved numeric cutoff, set at `fit`.
    threshold_value: f64,
    support: Vec<bool>,
    _state: PhantomData<(F, S)>,
}

impl<F, E> SelectFromModel<F, E, Unfit>
where
    F: Float + CubeElement + Pod,
    E: ImportanceEstimator,
{
    /// sklearn defaults: `threshold=None, prefit=False, norm_order=1,
    /// max_features=None, importance_getter='auto'`.
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            threshold: Threshold::Default,
            prefit: false,
            norm_order: 1.0,
            max_features: None,
            importance_getter: ImportanceGetter::Auto,
            threshold_value: f64::NAN,
            support: Vec::new(),
            _state: PhantomData,
        }
    }

    /// `threshold=..`.
    pub fn with_threshold(mut self, threshold: Threshold) -> Self {
        self.threshold = threshold;
        self
    }

    /// `prefit=..`.
    ///
    /// With `prefit=True` sklearn does NOT clone-and-refit; it reads the
    /// importances off the already-fitted estimator the caller passed. The Rust
    /// [`ImportanceEstimator`] is asked for importances either way, so `prefit`
    /// is carried for parameter parity and to let an impl short-circuit its own
    /// refit; the selector's behaviour is identical. The Python shim, which has
    /// a real estimator object, reproduces sklearn's `prefit` semantics fully,
    /// including its `NotFittedError`.
    pub fn with_prefit(mut self, prefit: bool) -> Self {
        self.prefit = prefit;
        self
    }

    /// `norm_order=..` — the norm applied down a 2-D `coef_`'s columns.
    pub fn with_norm_order(mut self, norm_order: f64) -> Self {
        self.norm_order = norm_order;
        self
    }

    /// `max_features=..`. sklearn also accepts a CALLABLE here (evaluated on `X`
    /// at `fit`); a Rust caller computes the number itself, which is the same
    /// thing without the deferred call.
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// `importance_getter=..`.
    pub fn with_importance_getter(mut self, getter: ImportanceGetter) -> Self {
        self.importance_getter = getter;
        self
    }
}

impl<F, E> Fit<F> for SelectFromModel<F, E, Unfit>
where
    F: Float + CubeElement + Pod,
    E: ImportanceEstimator,
{
    type Fitted = SelectFromModel<F, E, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError> {
        let (x_host, y_host) = host_design(pool, x, y, shape, "select_from_model")?;
        let (n, d) = shape;
        let imp = self.estimator.fit_importances(&x_host, &y_host, n, d)?;
        if imp.n_features() != d {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "importances",
                rows: 1,
                cols: d,
                len: imp.n_features(),
            }));
        }
        let scores = self.importance_getter.resolve(&imp, false, self.norm_order);
        let threshold_value = self.threshold.resolve(&scores);

        // sklearn's order, which is observable: `max_features` picks the top-N
        // by score FIRST, and the threshold then removes any of those that fall
        // below it. So the result can be SMALLER than `max_features` but never
        // larger, and a feature above the threshold can still be dropped for
        // being outside the top-N.
        let mut support = if let Some(maxf) = self.max_features {
            let mut order: Vec<usize> = (0..d).collect();
            // `argsort(-scores, kind="mergesort")` — descending, stable, so a
            // tie keeps the LOWER column index (the opposite of `SelectKBest`,
            // which sorts ascending and takes the tail).
            order.sort_by(|&a, &b| (-scores[a]).total_cmp(&-scores[b]));
            let mut m = vec![false; d];
            for &i in order.iter().take(maxf) {
                m[i] = true;
            }
            m
        } else {
            vec![true; d]
        };
        for c in 0..d {
            if scores[c] < threshold_value {
                support[c] = false;
            }
        }

        Ok(SelectFromModel {
            estimator: self.estimator,
            threshold: self.threshold,
            prefit: self.prefit,
            norm_order: self.norm_order,
            max_features: self.max_features,
            importance_getter: self.importance_getter,
            threshold_value,
            support,
            _state: PhantomData,
        })
    }
}

impl<F, E> SelectFromModel<F, E, Fitted> {
    /// `threshold_` — the resolved numeric cutoff.
    pub fn threshold_value(&self) -> f64 {
        self.threshold_value
    }

    /// `max_features_` — the resolved cap, `None` when unset.
    pub fn max_features(&self) -> Option<usize> {
        self.max_features
    }

    /// The inner estimator (`estimator_`).
    pub fn estimator(&self) -> &E {
        &self.estimator
    }
}

impl<F, E, S> Selector for SelectFromModel<F, E, S> {
    fn support_mask(&self) -> &[bool] {
        &self.support
    }
}

impl<F, E> Transform<F> for SelectFromModel<F, E, Fitted>
where
    F: Float + CubeElement + Pod,
    E: ImportanceEstimator,
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        transform_selected(self, pool, x, shape, "select_from_model")
    }

    fn inverse_transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        z: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        inverse_transform_selected(self, pool, z, shape)
    }
}

// ===========================================================================
// RFE
// ===========================================================================

/// `RFE(n_features_to_select=..)` — an absolute count, a FRACTION of
/// `n_features`, or sklearn's `None` (half).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NFeatures {
    /// `None` — `n_features // 2`.
    Half,
    /// An integer count. sklearn WARNS (and keeps everything) when it exceeds
    /// `n_features` rather than raising.
    Count(usize),
    /// A `float` in `(0, 1]` — `int(n_features * fraction)`.
    Fraction(f64),
}

impl Default for NFeatures {
    fn default() -> Self {
        Self::Half
    }
}

impl NFeatures {
    /// Resolve against `n_features`, sklearn's `_fit` prologue.
    fn resolve(&self, d: usize) -> usize {
        match self {
            Self::Half => d / 2,
            Self::Count(k) => *k,
            Self::Fraction(f) => (d as f64 * f) as usize,
        }
    }
}

/// The per-elimination-step record `RFECV` reads out of an `RFE` run —
/// sklearn's `step_scores_` / `step_support_` / `step_ranking_` /
/// `step_n_features_`, which exist only to be consumed by `RFECV`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RfeSteps {
    /// Features remaining at each step.
    pub n_features: Vec<usize>,
    /// The fold score at each step.
    pub scores: Vec<f64>,
    /// The support mask at each step.
    pub support: Vec<Vec<bool>>,
    /// The ranking vector at each step.
    pub ranking: Vec<Vec<usize>>,
}

/// `sklearn.feature_selection.RFE`.
pub struct Rfe<F, E, S = Unfit> {
    estimator: E,
    n_features_to_select: NFeatures,
    step: RfeStep,
    verbose: u32,
    importance_getter: ImportanceGetter,
    support: Vec<bool>,
    ranking: Vec<usize>,
    _state: PhantomData<(F, S)>,
}

/// `RFE(step=..)` — features removed per iteration, as a count or a fraction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RfeStep {
    /// An integer count (sklearn default `1`).
    Count(usize),
    /// A float in `(0, 1)` — `max(1, int(step * n_features))`.
    Fraction(f64),
}

impl Default for RfeStep {
    fn default() -> Self {
        Self::Count(1)
    }
}

impl RfeStep {
    fn resolve(&self, d: usize) -> Result<usize, AlgoError> {
        let v = match self {
            // sklearn's branch is `if 0.0 < step < 1.0` → fraction, `else`
            // → `int(step)`. A `step` of exactly `1.0` therefore takes the
            // INTEGER branch and removes one feature, not `n_features`.
            Self::Fraction(f) if *f > 0.0 && *f < 1.0 => ((*f * d as f64) as usize).max(1),
            Self::Fraction(f) => *f as usize,
            Self::Count(c) => *c,
        };
        if v == 0 {
            return Err(AlgoError::InvalidSelectorParam {
                estimator: "rfe",
                param: "step",
                value: v as f64,
                reason: "must remove at least one feature per iteration",
            });
        }
        Ok(v)
    }
}

impl<F, E> Rfe<F, E, Unfit>
where
    F: Float + CubeElement + Pod,
    E: ImportanceEstimator,
{
    /// sklearn defaults: `n_features_to_select=None, step=1, verbose=0,
    /// importance_getter='auto'`.
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            n_features_to_select: NFeatures::Half,
            step: RfeStep::Count(1),
            verbose: 0,
            importance_getter: ImportanceGetter::Auto,
            support: Vec::new(),
            ranking: Vec::new(),
            _state: PhantomData,
        }
    }

    /// `n_features_to_select=..`.
    pub fn with_n_features_to_select(mut self, n: NFeatures) -> Self {
        self.n_features_to_select = n;
        self
    }

    /// `step=..`.
    pub fn with_step(mut self, step: RfeStep) -> Self {
        self.step = step;
        self
    }

    /// `verbose=..`. Emits the per-iteration "Fitting estimator with N
    /// features." line sklearn `print`s, through `log::info!` rather than stdout
    /// — a library writing to stdout is not acceptable, and the log target is
    /// what the rest of this crate uses.
    pub fn with_verbose(mut self, verbose: u32) -> Self {
        self.verbose = verbose;
        self
    }

    /// `importance_getter=..`.
    pub fn with_importance_getter(mut self, getter: ImportanceGetter) -> Self {
        self.importance_getter = getter;
        self
    }

    /// The elimination loop, shared by `fit` and by [`Rfe::fit_with_steps`]
    /// (which `RFECV` drives).
    ///
    /// sklearn's `_fit`, transcribed. The two details that decide the result:
    ///
    /// * `ranks = np.argsort(importances, kind="stable")` then eliminate
    ///   `ranks[:threshold]` — an ASCENDING stable sort, so the LOWEST
    ///   importances go first and a tie eliminates the lower position within the
    ///   surviving-feature list.
    /// * `threshold = min(step, n_remaining − n_to_select)`, so the final
    ///   iteration removes only as many as needed and never overshoots.
    fn eliminate(
        &self,
        x: &[f64],
        y: &[f64],
        n: usize,
        d: usize,
        scorer: Option<&dyn FoldScorer>,
        train: &[usize],
        test: &[usize],
    ) -> Result<(Vec<bool>, Vec<usize>, RfeSteps), AlgoError> {
        let n_to_select = self.n_features_to_select.resolve(d).min(d);
        let step = self.step.resolve(d)?;
        let mut support = vec![true; d];
        let mut ranking = vec![1usize; d];
        let mut steps = RfeSteps::default();

        // One scoring pass, factored so the loop below reads as the elimination
        // it is. Returns `None` when no scorer was supplied (the plain `fit`).
        let score_at = |cols: &[usize]| -> Result<Option<f64>, AlgoError> {
            match scorer {
                None => Ok(None),
                Some(s) => {
                    let xtr = subset(x, d, train, cols);
                    let xte = subset(x, d, test, cols);
                    let ytr: Vec<f64> = train.iter().map(|&r| y[r]).collect();
                    let yte: Vec<f64> = test.iter().map(|&r| y[r]).collect();
                    Ok(Some(s.fit_score(
                        &xtr,
                        &ytr,
                        train.len(),
                        &xte,
                        &yte,
                        test.len(),
                        cols.len(),
                    )?))
                }
            }
        };

        while support.iter().filter(|&&s| s).count() > n_to_select {
            let features: Vec<usize> = (0..d).filter(|&c| support[c]).collect();
            if self.verbose > 0 {
                log::info!("Fitting estimator with {} features.", features.len());
            }
            let sub_n = if scorer.is_some() { train.len() } else { n };
            let sub_rows: Vec<usize> = if scorer.is_some() {
                train.to_vec()
            } else {
                (0..n).collect()
            };
            let x_sub = subset(x, d, &sub_rows, &features);
            let y_sub: Vec<f64> = sub_rows.iter().map(|&r| y[r]).collect();
            let imp = self
                .estimator
                .fit_importances(&x_sub, &y_sub, sub_n, features.len())?;
            if imp.n_features() != features.len() {
                return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                    operand: "importances",
                    rows: 1,
                    cols: features.len(),
                    len: imp.n_features(),
                }));
            }
            // sklearn records the step values BEFORE eliminating, "because
            // 'estimator' must use features that have not been eliminated yet".
            if scorer.is_some() {
                steps.n_features.push(features.len());
                steps.scores.push(score_at(&features)?.unwrap_or(f64::NAN));
                steps.support.push(support.clone());
                steps.ranking.push(ranking.clone());
            }

            let scores = self
                .importance_getter
                .resolve(&imp, true, self.norm_order());
            let mut ranks: Vec<usize> = (0..features.len()).collect();
            ranks.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
            let threshold = step.min(features.iter().len() - n_to_select);
            for &r in ranks.iter().take(threshold) {
                support[features[r]] = false;
            }
            for c in 0..d {
                if !support[c] {
                    ranking[c] += 1;
                }
            }
        }

        // The final refit on the surviving features, sklearn's `estimator_`.
        let features: Vec<usize> = (0..d).filter(|&c| support[c]).collect();
        let sub_rows: Vec<usize> = if scorer.is_some() {
            train.to_vec()
        } else {
            (0..n).collect()
        };
        let x_sub = subset(x, d, &sub_rows, &features);
        let y_sub: Vec<f64> = sub_rows.iter().map(|&r| y[r]).collect();
        self.estimator
            .fit_importances(&x_sub, &y_sub, sub_rows.len(), features.len())?;
        if scorer.is_some() {
            steps.n_features.push(features.len());
            steps.scores.push(score_at(&features)?.unwrap_or(f64::NAN));
            steps.support.push(support.clone());
            steps.ranking.push(ranking.clone());
        }
        Ok((support, ranking, steps))
    }

    /// `norm_order` is fixed at `1` for `RFE`: sklearn calls
    /// `_get_feature_importances(transform_func="square")` with the DEFAULT
    /// `norm_order`, and the square reduction ignores it entirely.
    fn norm_order(&self) -> f64 {
        1.0
    }

    /// Run the elimination on one CV fold, returning the per-step record —
    /// sklearn's `_rfe_single_fit`, which is `RFECV`'s inner loop.
    pub fn fit_with_steps(
        &self,
        x: &[f64],
        y: &[f64],
        n: usize,
        d: usize,
        scorer: &dyn FoldScorer,
        train: &[usize],
        test: &[usize],
    ) -> Result<RfeSteps, AlgoError> {
        let (_, _, steps) = self.eliminate(x, y, n, d, Some(scorer), train, test)?;
        Ok(steps)
    }
}

impl<F, E> Fit<F> for Rfe<F, E, Unfit>
where
    F: Float + CubeElement + Pod,
    E: ImportanceEstimator,
{
    type Fitted = Rfe<F, E, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError> {
        let (x_host, y_host) = host_design(pool, x, y, shape, "rfe")?;
        let (n, d) = shape;
        // sklearn validates with `ensure_min_features=2`: eliminating from a
        // single column has no meaning.
        if d < 2 {
            return Err(AlgoError::InvalidSelectorParam {
                estimator: "rfe",
                param: "n_features",
                value: d as f64,
                reason: "recursive elimination needs at least 2 features",
            });
        }
        let (support, ranking, _) = self.eliminate(&x_host, &y_host, n, d, None, &[], &[])?;
        Ok(Rfe {
            estimator: self.estimator,
            n_features_to_select: self.n_features_to_select,
            step: self.step,
            verbose: self.verbose,
            importance_getter: self.importance_getter,
            support,
            ranking,
            _state: PhantomData,
        })
    }
}

impl<F, E> Rfe<F, E, Fitted> {
    /// `ranking_` — `1` for a selected feature, then `2, 3, …` in reverse
    /// elimination order.
    pub fn ranking(&self) -> &[usize] {
        &self.ranking
    }

    /// `n_features_` — the selected count.
    pub fn n_features(&self) -> usize {
        self.support.iter().filter(|&&s| s).count()
    }

    /// The inner estimator (`estimator_`).
    pub fn estimator(&self) -> &E {
        &self.estimator
    }
}

impl<F, E, S> Selector for Rfe<F, E, S> {
    fn support_mask(&self) -> &[bool] {
        &self.support
    }
}

impl<F, E> Transform<F> for Rfe<F, E, Fitted>
where
    F: Float + CubeElement + Pod,
    E: ImportanceEstimator,
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        transform_selected(self, pool, x, shape, "rfe")
    }

    fn inverse_transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        z: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        inverse_transform_selected(self, pool, z, shape)
    }
}

// ===========================================================================
// RFECV
// ===========================================================================

/// `cv_results_` — the per-feature-subset cross-validation record `RFECV`
/// exposes.
///
/// Field-per-array rather than sklearn's `dict of ndarrays`, with the
/// `split{i}_*` keys as the outer index of a `Vec<Vec<_>>`. Same content, and a
/// caller building sklearn's dict does so by naming the fields.
///
/// Every array is ordered by ASCENDING feature count, the reverse of the
/// elimination order — sklearn reverses its arrays at the end ("reverse to stay
/// consistent with before") and that ordering is part of the public attribute.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CvResults {
    /// `n_features` — the feature count of each subset, ascending.
    pub n_features: Vec<usize>,
    /// `mean_test_score`.
    pub mean_test_score: Vec<f64>,
    /// `std_test_score` — the POPULATION std (`np.std` default `ddof=0`).
    pub std_test_score: Vec<f64>,
    /// `split{i}_test_score`, outer index `i` = fold.
    pub split_test_score: Vec<Vec<f64>>,
    /// `split{i}_ranking`.
    pub split_ranking: Vec<Vec<Vec<usize>>>,
    /// `split{i}_support`.
    pub split_support: Vec<Vec<Vec<bool>>>,
}

/// `sklearn.feature_selection.RFECV`.
pub struct Rfecv<F, E, C, S = Unfit> {
    estimator: E,
    scorer: C,
    step: RfeStep,
    min_features_to_select: usize,
    cv: Cv,
    verbose: u32,
    n_jobs: Option<usize>,
    importance_getter: ImportanceGetter,
    support: Vec<bool>,
    ranking: Vec<usize>,
    cv_results: CvResults,
    _state: PhantomData<(F, S)>,
}

impl<F, E, C> Rfecv<F, E, C, Unfit>
where
    F: Float + CubeElement + Pod,
    E: ImportanceEstimator + Clone,
    C: FoldScorer,
{
    /// sklearn defaults: `step=1, min_features_to_select=1, cv=None,
    /// scoring=None, verbose=0, n_jobs=None, importance_getter='auto'`.
    ///
    /// `scoring` is not a separate parameter here: it is inseparable from the
    /// estimator in the [`FoldScorer`] the caller supplies (see this module's
    /// docs), which is also why `cv=None`'s classifier-vs-regressor
    /// stratification choice is spelled out in [`Cv::Folds::stratified`] rather
    /// than inferred from the estimator.
    pub fn new(estimator: E, scorer: C) -> Self {
        Self {
            estimator,
            scorer,
            step: RfeStep::Count(1),
            min_features_to_select: 1,
            cv: Cv::default(),
            verbose: 0,
            n_jobs: None,
            importance_getter: ImportanceGetter::Auto,
            support: Vec::new(),
            ranking: Vec::new(),
            cv_results: CvResults::default(),
            _state: PhantomData,
        }
    }

    /// `step=..`.
    pub fn with_step(mut self, step: RfeStep) -> Self {
        self.step = step;
        self
    }

    /// `min_features_to_select=..`.
    pub fn with_min_features_to_select(mut self, m: usize) -> Self {
        self.min_features_to_select = m;
        self
    }

    /// `cv=..`.
    pub fn with_cv(mut self, cv: Cv) -> Self {
        self.cv = cv;
        self
    }

    /// `verbose=..`.
    pub fn with_verbose(mut self, verbose: u32) -> Self {
        self.verbose = verbose;
        self
    }

    /// `n_jobs=..`. The fold loop is embarrassingly parallel, but each fold runs
    /// the caller's estimator, which may itself hold a device queue — so folds
    /// run SEQUENTIALLY here and the parameter is carried for parity. Running
    /// them concurrently would multiply device-queue contention by the fold
    /// count for a `k`-times speedup only when the inner estimator is host-bound,
    /// which the selector cannot know. A caller who does know parallelises at
    /// its own level.
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// `importance_getter=..`.
    pub fn with_importance_getter(mut self, getter: ImportanceGetter) -> Self {
        self.importance_getter = getter;
        self
    }
}

impl<F, E, C> Fit<F> for Rfecv<F, E, C, Unfit>
where
    F: Float + CubeElement + Pod,
    E: ImportanceEstimator + Clone,
    C: FoldScorer,
{
    type Fitted = Rfecv<F, E, C, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError> {
        let (x_host, y_host) = host_design(pool, x, y, shape, "rfecv")?;
        let (n, d) = shape;
        if d < 2 {
            return Err(AlgoError::InvalidSelectorParam {
                estimator: "rfecv",
                param: "n_features",
                value: d as f64,
                reason: "recursive elimination needs at least 2 features",
            });
        }
        let splits = self.cv.splits(n, &y_host)?;

        // Phase 1: eliminate down to `min_features_to_select` on every fold,
        // recording the score at each subset size.
        let probe: Rfe<F, E, Unfit> = Rfe::new(self.estimator.clone())
            .with_n_features_to_select(NFeatures::Count(self.min_features_to_select.min(d)))
            .with_step(self.step)
            .with_verbose(self.verbose)
            .with_importance_getter(self.importance_getter.clone());
        let mut fold_steps = Vec::with_capacity(splits.len());
        for (train, test) in splits.iter() {
            fold_steps.push(probe.fit_with_steps(
                &x_host,
                &y_host,
                n,
                d,
                &self.scorer,
                train,
                test,
            )?);
        }

        // Phase 2: pick the subset size with the best SUMMED score, breaking a
        // tie toward the SMALLEST size. sklearn does this by reversing the arrays
        // before `argmax` ("Reverse order such that lowest number of features is
        // selected in case of tie") — reproduced by iterating the reversed order
        // and taking the first strict maximum.
        let n_steps = fold_steps[0].n_features.len();
        let mut best = (f64::NEG_INFINITY, self.min_features_to_select);
        for s in (0..n_steps).rev() {
            let total: f64 = fold_steps.iter().map(|f| f.scores[s]).sum();
            if total > best.0 {
                best = (total, fold_steps[0].n_features[s]);
            }
        }

        // Phase 3: re-run the elimination on the WHOLE design at that size.
        let final_rfe: Rfe<F, E, Unfit> = Rfe::new(self.estimator.clone())
            .with_n_features_to_select(NFeatures::Count(best.1))
            .with_step(self.step)
            .with_verbose(self.verbose)
            .with_importance_getter(self.importance_getter.clone());
        let (support, ranking, _) = final_rfe.eliminate(&x_host, &y_host, n, d, None, &[], &[])?;

        // `cv_results_`, reversed to ascending feature count.
        let rev = |v: &[f64]| -> Vec<f64> { v.iter().rev().copied().collect() };
        let split_test_score: Vec<Vec<f64>> = fold_steps.iter().map(|f| rev(&f.scores)).collect();
        let k = splits.len() as f64;
        let mean_test_score: Vec<f64> = (0..n_steps)
            .map(|s| split_test_score.iter().map(|f| f[s]).sum::<f64>() / k)
            .collect();
        let std_test_score: Vec<f64> = (0..n_steps)
            .map(|s| {
                let m = mean_test_score[s];
                (split_test_score
                    .iter()
                    .map(|f| (f[s] - m) * (f[s] - m))
                    .sum::<f64>()
                    / k)
                    .sqrt()
            })
            .collect();
        let cv_results = CvResults {
            n_features: fold_steps[0].n_features.iter().rev().copied().collect(),
            mean_test_score,
            std_test_score,
            split_test_score,
            split_ranking: fold_steps
                .iter()
                .map(|f| f.ranking.iter().rev().cloned().collect())
                .collect(),
            split_support: fold_steps
                .iter()
                .map(|f| f.support.iter().rev().cloned().collect())
                .collect(),
        };

        Ok(Rfecv {
            estimator: self.estimator,
            scorer: self.scorer,
            step: self.step,
            min_features_to_select: self.min_features_to_select,
            cv: self.cv,
            verbose: self.verbose,
            n_jobs: self.n_jobs,
            importance_getter: self.importance_getter,
            support,
            ranking,
            cv_results,
            _state: PhantomData,
        })
    }
}

impl<F, E, C> Rfecv<F, E, C, Fitted> {
    /// `ranking_`.
    pub fn ranking(&self) -> &[usize] {
        &self.ranking
    }

    /// `n_features_`.
    pub fn n_features(&self) -> usize {
        self.support.iter().filter(|&&s| s).count()
    }

    /// `cv_results_`.
    pub fn cv_results(&self) -> &CvResults {
        &self.cv_results
    }
}

impl<F, E, C, S> Selector for Rfecv<F, E, C, S> {
    fn support_mask(&self) -> &[bool] {
        &self.support
    }
}

impl<F, E, C> Transform<F> for Rfecv<F, E, C, Fitted>
where
    F: Float + CubeElement + Pod,
    E: ImportanceEstimator,
    C: FoldScorer,
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        transform_selected(self, pool, x, shape, "rfecv")
    }

    fn inverse_transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        z: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        inverse_transform_selected(self, pool, z, shape)
    }
}

// ===========================================================================
// SequentialFeatureSelector
// ===========================================================================

/// `SequentialFeatureSelector(direction=..)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// `'forward'` — grow the selected set.
    Forward,
    /// `'backward'` — shrink it by excluding.
    Backward,
}

impl Direction {
    /// Parse sklearn's string.
    pub fn parse(s: &str) -> Result<Self, AlgoError> {
        match s {
            "forward" => Ok(Self::Forward),
            "backward" => Ok(Self::Backward),
            other => Err(AlgoError::UnknownSelectorOption {
                estimator: "sequential_feature_selector",
                param: "direction",
                value: other.to_string(),
                expected: "forward, backward",
            }),
        }
    }
}

/// `SequentialFeatureSelector(n_features_to_select=..)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SfsTarget {
    /// `'auto'` — `n_features − 1` when `tol` is set (the loop stops on `tol`),
    /// else `n_features // 2`.
    Auto,
    /// An integer count, which sklearn REQUIRES to be `< n_features`.
    Count(usize),
    /// A float fraction — `int(n_features * fraction)`.
    Fraction(f64),
}

impl Default for SfsTarget {
    fn default() -> Self {
        Self::Auto
    }
}

/// `sklearn.feature_selection.SequentialFeatureSelector`.
pub struct SequentialFeatureSelector<F, C, S = Unfit> {
    scorer: C,
    n_features_to_select: SfsTarget,
    tol: Option<f64>,
    direction: Direction,
    cv: Cv,
    n_jobs: Option<usize>,
    support: Vec<bool>,
    n_features_selected: usize,
    _state: PhantomData<(F, S)>,
}

impl<F, C> SequentialFeatureSelector<F, C, Unfit>
where
    F: Float + CubeElement + Pod,
    C: FoldScorer,
{
    /// sklearn defaults: `n_features_to_select='auto', tol=None,
    /// direction='forward', scoring=None, cv=5, n_jobs=None`.
    pub fn new(scorer: C) -> Self {
        Self {
            scorer,
            n_features_to_select: SfsTarget::Auto,
            tol: None,
            direction: Direction::Forward,
            cv: Cv::Folds {
                n_splits: 5,
                stratified: false,
            },
            n_jobs: None,
            support: Vec::new(),
            n_features_selected: 0,
            _state: PhantomData,
        }
    }

    /// `n_features_to_select=..`.
    pub fn with_n_features_to_select(mut self, t: SfsTarget) -> Self {
        self.n_features_to_select = t;
        self
    }

    /// `tol=..`. Only meaningful with `n_features_to_select='auto'`, exactly as
    /// sklearn documents; a POSITIVE value is required for forward selection
    /// (sklearn raises "tol must be strictly positive when doing forward
    /// selection" for a negative one).
    pub fn with_tol(mut self, tol: Option<f64>) -> Self {
        self.tol = tol;
        self
    }

    /// `direction=..`.
    pub fn with_direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// `cv=..`.
    pub fn with_cv(mut self, cv: Cv) -> Self {
        self.cv = cv;
        self
    }

    /// `n_jobs=..`. Carried for parity; the candidate loop runs sequentially for
    /// the reason [`Rfecv::with_n_jobs`] documents.
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// The mean CV score of one candidate column set — sklearn's
    /// `cross_val_score(...).mean()`.
    fn cv_score(
        &self,
        x: &[f64],
        y: &[f64],
        d: usize,
        cols: &[usize],
        splits: &[(Vec<usize>, Vec<usize>)],
    ) -> Result<f64, AlgoError> {
        let mut total = 0.0;
        for (train, test) in splits {
            let xtr = subset(x, d, train, cols);
            let xte = subset(x, d, test, cols);
            let ytr: Vec<f64> = train.iter().map(|&r| y[r]).collect();
            let yte: Vec<f64> = test.iter().map(|&r| y[r]).collect();
            total += self.scorer.fit_score(
                &xtr,
                &ytr,
                train.len(),
                &xte,
                &yte,
                test.len(),
                cols.len(),
            )?;
        }
        Ok(total / splits.len() as f64)
    }
}

impl<F, C> Fit<F> for SequentialFeatureSelector<F, C, Unfit>
where
    F: Float + CubeElement + Pod,
    C: FoldScorer,
{
    type Fitted = SequentialFeatureSelector<F, C, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError> {
        let (x_host, y_host) = host_design(pool, x, y, shape, "sequential_feature_selector")?;
        let (n, d) = shape;
        if d < 2 {
            return Err(AlgoError::InvalidSelectorParam {
                estimator: "sequential_feature_selector",
                param: "n_features",
                value: d as f64,
                reason: "sequential selection needs at least 2 features",
            });
        }
        let is_auto = matches!(self.n_features_to_select, SfsTarget::Auto);
        let target = match self.n_features_to_select {
            SfsTarget::Auto => {
                if self.tol.is_some() {
                    d - 1
                } else {
                    d / 2
                }
            }
            SfsTarget::Count(k) => {
                if k >= d {
                    return Err(AlgoError::InvalidSelectorParam {
                        estimator: "sequential_feature_selector",
                        param: "n_features_to_select",
                        value: k as f64,
                        reason: "must be < n_features",
                    });
                }
                k
            }
            SfsTarget::Fraction(f) => (d as f64 * f) as usize,
        };
        if let Some(tol) = self.tol {
            if tol < 0.0 && self.direction == Direction::Forward {
                return Err(AlgoError::InvalidSelectorParam {
                    estimator: "sequential_feature_selector",
                    param: "tol",
                    value: tol,
                    reason: "must be strictly positive when doing forward selection",
                });
            }
        }
        let splits = self.cv.splits(n, &y_host)?;

        // `current_mask` marks features already SELECTED (forward) or already
        // EXCLUDED (backward) — sklearn's comment, and the reason the same loop
        // serves both directions.
        let mut current_mask = vec![false; d];
        let n_iterations = if is_auto || self.direction == Direction::Forward {
            target
        } else {
            d - target
        };
        let mut old_score = f64::NEG_INFINITY;
        let is_auto_select = self.tol.is_some() && is_auto;

        for _ in 0..n_iterations {
            let mut best: Option<(usize, f64)> = None;
            for candidate in 0..d {
                if current_mask[candidate] {
                    continue;
                }
                let mut trial = current_mask.clone();
                trial[candidate] = true;
                // Forward: score the SELECTED set. Backward: score everything
                // NOT yet excluded — so a high score means the excluded
                // candidate was the least useful.
                let cols: Vec<usize> = (0..d)
                    .filter(|&c| {
                        if self.direction == Direction::Forward {
                            trial[c]
                        } else {
                            !trial[c]
                        }
                    })
                    .collect();
                if cols.is_empty() {
                    continue;
                }
                let score = self.cv_score(&x_host, &y_host, d, &cols, &splits)?;
                // STRICTLY greater, so the FIRST candidate wins a tie — the
                // ascending-index tie-break `np.argmax` gives sklearn.
                if best.is_none_or(|(_, b)| score > b) {
                    best = Some((candidate, score));
                }
            }
            let (idx, score) = match best {
                Some(v) => v,
                None => break,
            };
            if is_auto_select && (score - old_score) < self.tol.unwrap_or(0.0) {
                break;
            }
            old_score = score;
            current_mask[idx] = true;
        }

        let support: Vec<bool> = if self.direction == Direction::Forward {
            current_mask
        } else {
            current_mask.iter().map(|&m| !m).collect()
        };
        let n_features_selected = support.iter().filter(|&&s| s).count();

        Ok(SequentialFeatureSelector {
            scorer: self.scorer,
            n_features_to_select: self.n_features_to_select,
            tol: self.tol,
            direction: self.direction,
            cv: self.cv,
            n_jobs: self.n_jobs,
            support,
            n_features_selected,
            _state: PhantomData,
        })
    }
}

impl<F, C> SequentialFeatureSelector<F, C, Fitted> {
    /// `n_features_to_select_` — the count actually selected, which with
    /// `'auto'` + `tol` is whatever the early stop landed on.
    pub fn n_features_to_select(&self) -> usize {
        self.n_features_selected
    }
}

impl<F, C, S> Selector for SequentialFeatureSelector<F, C, S> {
    fn support_mask(&self) -> &[bool] {
        &self.support
    }
}

impl<F, C> Transform<F> for SequentialFeatureSelector<F, C, Fitted>
where
    F: Float + CubeElement + Pod,
    C: FoldScorer,
{
    fn transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        transform_selected(self, pool, x, shape, "sequential_feature_selector")
    }

    fn inverse_transform(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        z: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        inverse_transform_selected(self, pool, z, shape)
    }
}
