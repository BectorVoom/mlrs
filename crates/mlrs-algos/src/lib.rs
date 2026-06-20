//! `mlrs-algos` — sklearn-compatible ML estimators.
//!
//! Phase 4 fills this crate with the four closed-form estimators —
//! `LinearRegression` / `Ridge` (the `linear` module) and `PCA` /
//! `TruncatedSVD` (the `decomposition` module) — each generic over
//! `<F: Float + CubeElement + Pod>` and over the CubeCL runtime, composed on the
//! already-validated Phase-2/3 primitives (thin SVD, GEMM, covariance/Gram,
//! reductions) plus the new Phase-4 Cholesky/solve primitive in `mlrs-backend`.
//!
//! ## Module index (this file is OWNED by the Wave-0 scaffold plans 04-01 / 05-01)
//! - [`traits`] — the uniform estimator surface: `Fit` / `Predict` / `Transform`
//!   (D-04) plus the Phase-5 `PredictLabels` / `KNeighbors` / `PredictProba`
//!   (D-05/D-07), all re-exported below.
//! - [`error`] — the estimator-facing [`AlgoError`](error::AlgoError) (invalid
//!   hyperparameters; wraps `PrimError`).
//! - [`linear`] — `LinearRegression` (04-03) + `Ridge` (04-05) + the Phase-5
//!   iterative linear models `Lasso` / `ElasticNet` / `LogisticRegression`
//!   (05-07/08/09).
//! - [`decomposition`] — `PCA` + `TruncatedSVD` (04-04) + `IncrementalPCA`
//!   (DECOMP-03, 07-05).
//! - [`cluster`] — `KMeans` (CLUSTER-01) + `DBSCAN` (CLUSTER-02) (05-07/08).
//! - [`neighbors`] — `NearestNeighbors` / `KNeighborsClassifier` /
//!   `KNeighborsRegressor` (NEIGH-01/02/03) (05-10).
//! - [`covariance`] — `EmpiricalCovariance` (COV-01) + `LedoitWolf` (COV-02)
//!   (07-04). Registered as an empty stub here by the 07-01 Wave-0 scaffold.
//! - [`projection`] — `GaussianRandomProjection` / `SparseRandomProjection`
//!   (PROJ-01/02) (07-06). Registered as an empty stub here by 07-01.
//! - [`kernel_ridge`] — `KernelRidge` (KERNEL-01) (Wave-2, 08-03). Registered as
//!   an empty stub here by the 08-01 Wave-0 scaffold.
//! - [`density`] — `KernelDensity` (KERNEL-02) (Wave-2, 08-04). New `density/`
//!   home (RESEARCH Open Q2 — KD is not a neighbor estimator in mlrs's trait
//!   sense). Registered as an empty stub here by the 08-01 Wave-0 scaffold.
//!
//! The estimator plans edit ONLY their own estimator file and the relevant
//! module-index `mod.rs` (`linear/mod.rs` / `decomposition/mod.rs` /
//! `cluster/mod.rs` / `neighbors/mod.rs`) — never this `lib.rs` — so they stay
//! file-disjoint and parallel-safe.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2 — no in-source
//! `#[cfg(test)] mod tests`).

pub mod cluster;
pub mod covariance;
pub mod decomposition;
pub mod density;
pub mod error;
pub mod kernel_ridge;
pub mod linear;
pub mod neighbors;
pub mod projection;
pub mod traits;

// Re-export the estimator surface so downstream crates/tests write
// `use mlrs_algos::{Fit, PartialFit, Predict, Transform, PredictLabels,
// KNeighbors, PredictProba, AlgoError};` directly.
pub use error::AlgoError;
pub use traits::{
    Fit, KNeighbors, PartialFit, Predict, PredictLabels, PredictProba,
    ScoreSamples, Transform,
};
