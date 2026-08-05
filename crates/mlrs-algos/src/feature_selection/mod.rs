//! `feature_selection` — the complete `sklearn.feature_selection` public API
//! (FSEL-01).
//!
//! sklearn's `feature_selection.__init__` exports eighteen names. All eighteen
//! are here:
//!
//! | sklearn                     | mlrs                                              |
//! |-----------------------------|---------------------------------------------------|
//! | `SelectorMixin`             | [`Selector`] trait                                |
//! | `VarianceThreshold`         | [`VarianceThreshold`]                             |
//! | `SelectKBest`               | [`UnivariateFilter::k_best`]                       |
//! | `SelectPercentile`          | [`UnivariateFilter::percentile`]                   |
//! | `SelectFpr`                 | [`UnivariateFilter::fpr`]                          |
//! | `SelectFdr`                 | [`UnivariateFilter::fdr`]                          |
//! | `SelectFwe`                 | [`UnivariateFilter::fwe`]                          |
//! | `GenericUnivariateSelect`   | [`UnivariateFilter::generic`]                      |
//! | `SelectFromModel`           | [`SelectFromModel`]                               |
//! | `RFE`                       | [`Rfe`]                                           |
//! | `RFECV`                     | [`Rfecv`]                                         |
//! | `SequentialFeatureSelector` | [`SequentialFeatureSelector`]                     |
//! | `f_oneway`                  | [`f_oneway`]                                      |
//! | `f_classif`                 | [`f_classif`]                                     |
//! | `chi2`                      | [`chi2`]                                          |
//! | `r_regression`              | [`r_regression`]                                  |
//! | `f_regression`              | [`f_regression`]                                  |
//! | `mutual_info_classif`       | [`mutual_info_classif`]                           |
//! | `mutual_info_regression`    | [`mutual_info_regression`]                        |
//!
//! Every selector is `Fit` + `Transform` ([`crate::typestate`], D-01) and
//! implements [`Selector`], so `get_support` / `transform` /
//! `inverse_transform` come from one place, exactly as sklearn derives them all
//! from `SelectorMixin._get_support_mask`.
//!
//! ## Where the compute lives, and why
//! This module is deliberately explicit about its host/device split, because it
//! is not the usual one:
//!
//! * **A selector's `transform` is a DEVICE kernel** —
//!   `mlrs_kernels::feature_select::gather_columns`, launched by
//!   [`mlrs_backend::prims::feature_score::gather_columns`]. Pure data movement,
//!   exact in any float width, genuinely parallel.
//! * **A score's `O(n·d)` sweep is HOST `f64`** —
//!   [`mlrs_backend::prims::feature_score`]'s three parallel sweeps. Not a
//!   concession: the oracle contract is RELATIVE and these scores' p-values reach
//!   `1e-27`, so `f32` accumulation cannot meet it; cuda does not advertise
//!   `f64`; and the sweep runs once per `fit`. That module's docs give the full
//!   argument.
//! * **The p-values are HOST scalars** —
//!   [`mlrs_backend::prims::special`]'s `f_sf` / `chi2_sf`, i.e. the incomplete
//!   beta and gamma functions, `O(d)` work with data-dependent iteration counts.
//! * **The mutual-information estimators are HOST k-NN** —
//!   [`mutual_info`], for the reasons that module documents.
//! * **The meta-selectors are HOST drivers over the caller's estimator** —
//!   [`meta`], whose cost is entirely the inner estimator's `fit`, which is
//!   itself free to be device-resident.
//!
//! ## Reproducibility
//! Only `mutual_info_*` consumes randomness, and it is the one place in the
//! crate that matches numpy's MT19937 BIT-FOR-BIT rather than seeding a
//! SplitMix64 — see [`numpy_rng`] for why that inversion of the crate's usual
//! policy is the correct call there.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

pub mod meta;
pub mod mutual_info;
pub mod numpy_rng;
pub mod score;
pub mod selector;
pub mod univariate;
pub mod variance_threshold;

pub use meta::{
    Cv, CvResults, Direction, FnImportance, FnScorer, FoldScorer, ImportanceEstimator,
    ImportanceGetter, Importances, NFeatures, Rfe, RfeStep, RfeSteps, Rfecv, SelectFromModel,
    SequentialFeatureSelector, SfsTarget, Threshold,
};
pub use mutual_info::{
    mutual_info_classif, mutual_info_regression, DiscreteFeatures, MutualInfoParams,
};
pub use score::{
    chi2, f_classif, f_oneway, f_regression, r_regression, CustomScoreFunc, ScoreFunc, ScoreResult,
};
pub use selector::Selector;
pub use univariate::{GenericParam, KBest, SelectionMode, UnivariateFilter};
pub use variance_threshold::VarianceThreshold;
