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
//! - [`BayesianGaussianMixture`](bayesian_gaussian_mixture::BayesianGaussianMixture)
//!   (MIX-02) — the VARIATIONAL sibling of the above. Same four
//!   `covariance_type`s and same four `init_params` routes, but a conjugate
//!   prior on every block and a `weight_concentration_prior_type` that decides
//!   whether unneeded components are pruned (`dirichlet_process`) or merely
//!   smoothed (`dirichlet_distribution`). It runs on the SAME `gmm_host`
//!   engine: its E-step differs only by a vector of per-component constants,
//!   which `GmmHost::e_step_biased` takes as a parameter, so both estimators
//!   share one `O(n·k·d²)` inner nest and one set of initializations.
//!
//! Tests live in `crates/mlrs-algos/tests/gaussian_mixture_test.rs`,
//! `crates/mlrs-algos/tests/bayesian_mixture_test.rs` and their `*_perf_test`
//! siblings (AGENTS.md §2).

pub mod bayesian_gaussian_mixture;
pub mod gaussian_mixture;

pub use bayesian_gaussian_mixture::{
    BayesianGaussianMixture, BayesianGaussianMixtureBuilder, BayesianMixtureParams,
    MixturePriors, WeightConcentrationPriorType,
};
pub use gaussian_mixture::{
    GaussianMixture, GaussianMixtureBuilder, InitParams, MixtureParams,
};
