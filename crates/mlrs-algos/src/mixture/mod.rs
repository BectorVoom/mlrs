//! `mixture` — probabilistic mixture models (MIX-01).
//!
//! Module index for the `sklearn.mixture` family. Unlike `cluster`, whose
//! estimators return a hard partition, a mixture model returns a full posterior
//! over components, so these estimators implement the
//! [`PredictProba`](crate::typestate::PredictProba) /
//! [`PredictLogProba`](crate::typestate::PredictLogProba) /
//! [`ScoreSamples`](crate::typestate::ScoreSamples) accessors alongside
//! [`PredictLabels`](crate::typestate::PredictLabels).
//!
//! - [`GaussianMixture`](gaussian_mixture::GaussianMixture) (MIX-01) — the EM
//!   fit of a `k`-component Gaussian mixture over all four sklearn
//!   `covariance_type` parameterizations (`full` / `tied` / `diag` /
//!   `spherical`) and all four `init_params` routes (`kmeans` / `k-means++` /
//!   `random` / `random_from_data`). The compute engine is
//!   `mlrs_backend::prims::gmm_host`, which is host-resident on EVERY backend —
//!   see its module docs for why, and for the three structural wins it holds
//!   over sklearn's own implementation.
//!
//! Tests live in `crates/mlrs-algos/tests/gaussian_mixture_test.rs` and
//! `crates/mlrs-algos/tests/gaussian_mixture_perf_test.rs` (AGENTS.md §2).

pub mod gaussian_mixture;

pub use gaussian_mixture::{
    GaussianMixture, GaussianMixtureBuilder, InitParams, MixtureParams,
};
