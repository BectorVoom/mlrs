//! `preprocessing` — sklearn `preprocessing` scalers (PREP-01, Phase 24).
//!
//! Six stateless-hyperparameter / data-dependent-statistic transformers, all
//! `Fit` + `Transform` (`typestate`, D-01). Every column statistic (mean,
//! variance, min, max, median, quantile) is computed via the validated
//! `prims::reduce::column_reduce` (the same PCA/covariance/NB precedent —
//! `column_reduce` reads the whole matrix to host once per call, which is
//! correct and simple for a ONE-SHOT `fit`, not a repeated hot loop) or, for
//! the order-statistics `RobustScaler` needs (median / quantile), a host sort
//! (mirrors the ARIMA/BayesianRidge "inherently sequential → host arm"
//! precedent — sorting has no useful device parallelism at this scale).
//!
//! `transform`/`inverse_transform` apply a per-column AFFINE map
//! (`out = x * scale + shift`) or, for [`normalizer`], a per-ROW rescale.
//! Like [`crate::decomposition::pca`]'s column-mean centering, this is done as
//! a single host materialize → compute → re-upload pass (the fitted
//! scale/shift vectors are length-`n_features`, tiny) rather than a new
//! device kernel — consistent with PCA being this crate's most current
//! Transform-estimator template.
//!
//! - [`standard_scaler`] — `StandardScaler`: `(x − mean) / std`.
//! - [`min_max_scaler`] — `MinMaxScaler`: linear map into `feature_range`.
//! - [`max_abs_scaler`] — `MaxAbsScaler`: `x / max(|x|)`.
//! - [`robust_scaler`] — `RobustScaler`: `(x − median) / IQR`.
//! - [`normalizer`] — `Normalizer`: per-row L1/L2/max unit-norm rescale.
//! - [`binarizer`] — `Binarizer`: `x > threshold ? 1 : 0`.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

pub(crate) mod common;

pub mod binarizer;
pub mod max_abs_scaler;
pub mod min_max_scaler;
pub mod normalizer;
pub mod robust_scaler;
pub mod standard_scaler;

pub use binarizer::Binarizer;
pub use max_abs_scaler::MaxAbsScaler;
pub use min_max_scaler::MinMaxScaler;
pub use normalizer::{Norm, Normalizer};
pub use robust_scaler::RobustScaler;
pub use standard_scaler::StandardScaler;
