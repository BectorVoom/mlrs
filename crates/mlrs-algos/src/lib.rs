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
//! - [`typestate`] — the SINGLE estimator trait surface: the consuming-`self`
//!   `Fit` / `Predict` / `Transform` / `PartialFit` lifecycle traits plus the
//!   `&self` accessor traits `PredictLabels` / `KNeighbors` / `ScoreSamples` /
//!   `PredictProba` / `PredictLogProba`. The legacy `&mut self` `traits` module
//!   was hard-deleted in Phase 16 (D-01); `typestate` is now the only surface.
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
//! - [`naive_bayes`] — the five Naive Bayes classifiers `GaussianNB` /
//!   `MultinomialNB` / `BernoulliNB` / `ComplementNB` / `CategoricalNB`
//!   (NB-01..05, Phase 11). Five mutually-independent builder-fronted structs
//!   sharing only the `nb_common` free functions (D-03 — NO `NbBase`).
//!   Registered here by the 11-01 Wave-0 scaffold.
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
// Random Forest estimators (ENSEMBLE-01): `RandomForestClassifier` /
// `RandomForestRegressor` over the launch-only batched forest primitive
// (`mlrs_backend::prims::random_forest`).
pub mod ensemble;
pub mod error;
pub mod kernel_ridge;
pub mod linear;
pub mod manifold;
pub mod naive_bayes;
pub mod neighbors;
pub mod projection;
// The SINGLE estimator trait surface (D-01). The legacy `&mut self` `traits`
// module was hard-deleted in Phase 16 once every estimator migrated; consumers
// reach the lifecycle/accessor traits via the explicit `mlrs_algos::typestate::`
// path (e.g. `use mlrs_algos::typestate::Fit;`).
pub mod typestate;

// Re-export the estimator-facing error so downstream crates/tests write
// `use mlrs_algos::AlgoError;` directly. The trait surface is NOT re-exported at
// the crate root — consumers import the consuming-`self` traits explicitly from
// `mlrs_algos::typestate::*` (D-01, single trait surface).
pub use error::AlgoError;
