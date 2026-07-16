//! `ensemble` — Random Forest (ENSEMBLE-01) and HistGradientBoosting (GBT-01)
//! estimators.
//!
//! Module index, all composed on the launch-only batched level-wise histogram
//! tree primitives (`mlrs_backend::prims::{random_forest,
//! hist_gradient_boosting}`):
//!
//! - `RandomForestClassifier` — gini-split forest; `predict_proba` is the
//!   sklearn mean-of-leaf-distributions, `predict_labels` its argmax mapped
//!   through `classes_` (i32).
//! - `RandomForestRegressor` — variance-reduction forest; `predict` is the
//!   forest mean of leaf means.
//! - `HistGradientBoostingClassifier` — log-loss gradient boosting (sigmoid /
//!   softmax link, `n_classes` trees per iteration on multiclass).
//! - `HistGradientBoostingRegressor` — squared-error gradient boosting.
//!
//! ## Deviations from sklearn (documented, ENSEMBLE-01 / GBT-01)
//! - Trees are HISTOGRAM-BINNED (quantile-midpoint candidate thresholds,
//!   `n_bins` per feature — the cuML design). When a feature has fewer than
//!   `n_bins` distinct values the candidate set equals sklearn's exact
//!   midpoints; otherwise thresholds are quantile approximations. (For
//!   HistGradientBoosting this matches sklearn's own `_BinMapper` midpoint
//!   rule exactly when distinct values fit the bin budget.)
//! - `max_depth` is REQUIRED-BOUNDED (`1..=16`; forest default 10, boosting
//!   default 6): the complete-tree device layout has no unbounded-depth form.
//!   For HistGradientBoosting this replaces sklearn's leaf-wise
//!   `max_leaf_nodes=31` growth: with `max_leaf_nodes=None` and a depth
//!   bound, sklearn's leaf-wise grower produces the SAME tree as the mlrs
//!   level-wise grower (growth order is irrelevant without a leaf budget) —
//!   the oracle-fixture equivalence.
//! - Forest tie-breaks between equal-quality splits follow the deterministic
//!   lowest-(feature-slot, bin) rule, not sklearn's RNG-shuffled feature
//!   order. (HistGradientBoosting has NO feature subsampling; its
//!   lowest-(feature, bin) strict-`>` scan matches sklearn exactly.)
//! - HistGradientBoosting has NO early stopping (sklearn's `early_stopping=
//!   'auto'` enables it above 10k samples; pass `early_stopping=False` when
//!   comparing).
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

pub mod hist_gradient_boosting_classifier;
pub mod hist_gradient_boosting_regressor;
pub mod random_forest_classifier;
pub mod random_forest_regressor;

/// The sklearn `max_features` policy for per-node feature subsampling
/// (resolved to a concrete count at `fit`, when `n_features` is known).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxFeatures {
    /// `max(1, floor(sqrt(n_features)))` — the sklearn CLASSIFIER default.
    Sqrt,
    /// `max(1, floor(log2(n_features)))`.
    Log2,
    /// All features (the sklearn REGRESSOR default, `max_features=1.0`).
    All,
    /// An explicit per-node feature count (`1 ..= n_features`, the upper
    /// half validated at `fit`).
    Value(usize),
}

impl MaxFeatures {
    /// Resolve the policy against the fitted feature count.
    pub(crate) fn resolve(self, n_features: usize) -> usize {
        match self {
            MaxFeatures::Sqrt => ((n_features as f64).sqrt().floor() as usize).max(1),
            MaxFeatures::Log2 => ((n_features as f64).log2().floor() as usize).max(1),
            MaxFeatures::All => n_features,
            MaxFeatures::Value(v) => v,
        }
    }
}
