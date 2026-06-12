//! `mlrs-algos` — sklearn-compatible ML estimators.
//!
//! Phase 4 fills this crate with the four closed-form estimators —
//! `LinearRegression` / `Ridge` (the `linear` module) and `PCA` /
//! `TruncatedSVD` (the `decomposition` module) — each generic over
//! `<F: Float + CubeElement + Pod>` and over the CubeCL runtime, composed on the
//! already-validated Phase-2/3 primitives (thin SVD, GEMM, covariance/Gram,
//! reductions) plus the new Phase-4 Cholesky/solve primitive in `mlrs-backend`.
//!
//! ## Module index (this file is OWNED by plan 04-01)
//! - [`traits`] — the uniform `Fit` / `Predict` / `Transform` estimator surface
//!   (D-04), re-exported below.
//! - [`error`] — the estimator-facing [`AlgoError`](error::AlgoError) (invalid
//!   hyperparameters; wraps `PrimError`).
//! - [`linear`] — `LinearRegression` (04-03) + `Ridge` (04-05).
//! - [`decomposition`] — `PCA` + `TruncatedSVD` (04-04).
//!
//! The estimator plans (04-03/04/05) edit ONLY their own estimator file and the
//! relevant `linear/mod.rs` / `decomposition/mod.rs` module index — never this
//! `lib.rs` — so they stay file-disjoint and parallel-safe.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2 — no in-source
//! `#[cfg(test)] mod tests`).

pub mod decomposition;
pub mod error;
pub mod linear;
pub mod traits;

// Re-export the estimator surface so downstream crates/tests write
// `use mlrs_algos::{Fit, Predict, Transform, AlgoError};` directly.
pub use error::AlgoError;
pub use traits::{Fit, Predict, Transform};
