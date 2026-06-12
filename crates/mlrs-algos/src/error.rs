//! Estimator-facing error type `AlgoError` (D-08, estimator-local).
//!
//! The Phase-2/3 primitives surface geometry/convergence failures as
//! [`mlrs_core::PrimError`]. The estimators add a second, higher-level failure
//! class: invalid *hyperparameters* supplied at the host → estimator boundary
//! (untrusted per the Phase-4 threat model — T-04-01-01). `AlgoError` lives in
//! `mlrs-algos` (not `mlrs-core`) because it is estimator-specific and must not
//! be a dependency of the primitive layer; it wraps `PrimError` via `#[from]`
//! so an estimator method can use `?` on a prim call directly.
//!
//! `thiserror` in libraries (D-08, project convention); `anyhow` is reserved for
//! the Phase-6 PyO3 boundary, never here.

use thiserror::Error;

use mlrs_core::PrimError;

/// Errors raised by an `mlrs-algos` estimator during `fit` / `predict` /
/// `transform`.
///
/// One variant per failure class: an out-of-range `n_components` (the chief
/// untrusted-hyperparameter guard, T-04-01-01 / RESEARCH Pitfall 6), a negative
/// Ridge `alpha`, an unfitted-estimator misuse, an unsupported operation (e.g.
/// `inverse_transform` on TruncatedSVD), and a transparent wrap of any
/// underlying [`PrimError`] from the primitive layer.
#[derive(Debug, Error)]
pub enum AlgoError {
    /// A decomposition was constructed/fitted with `n_components` outside the
    /// valid range `1 ..= min(n_samples, n_features)` (D-06 — v1 takes an int
    /// `k ≤ min(m, n)`). Rejected at `fit` *before* any kernel launch so an
    /// untrusted hyperparameter becomes a typed error, not an out-of-bounds
    /// device read (T-04-01-01 / ASVS V5). Carries the requested `k` and the
    /// `max = min(n_samples, n_features)` that was exceeded.
    #[error(
        "estimator '{estimator}': n_components = {requested} is out of range \
         (must be 1..={max} = min(n_samples, n_features))"
    )]
    InvalidNComponents {
        /// Which estimator rejected the value (e.g. `"pca"` / `"truncated_svd"`).
        estimator: &'static str,
        /// The `n_components` the caller requested.
        requested: usize,
        /// The inclusive upper bound `min(n_samples, n_features)`.
        max: usize,
    },

    /// A regularised estimator (Ridge) was given a negative `alpha`. Ridge
    /// requires `alpha ≥ 0` (α = 0 degenerates to ordinary least squares);
    /// a negative penalty makes the normal matrix indefinite and the Cholesky
    /// factorization undefined (D-02). Rejected at `fit`.
    #[error("estimator '{estimator}': alpha = {alpha} is invalid (must be >= 0)")]
    InvalidAlpha {
        /// Which estimator rejected the value (e.g. `"ridge"`).
        estimator: &'static str,
        /// The offending penalty value.
        alpha: f64,
    },

    /// A `predict` / `transform` (or an attribute accessor) was called before
    /// the estimator was `fit`. Carries the estimator and the attribute/method
    /// that was unavailable.
    #[error(
        "estimator '{estimator}': '{operation}' called before fit (no fitted state)"
    )]
    NotFitted {
        /// Which estimator was used unfitted (e.g. `"pca"`).
        estimator: &'static str,
        /// The method/attribute that required fitted state.
        operation: &'static str,
    },

    /// An optional trait method that this estimator does not implement was
    /// invoked — e.g. `inverse_transform` on `TruncatedSVD` (only PCA supports
    /// the reconstruction in v1, D-01). Surfaced rather than panicking so the
    /// uniform trait surface (D-04) stays total.
    #[error(
        "estimator '{estimator}': operation '{operation}' is not supported"
    )]
    Unsupported {
        /// Which estimator was asked for the unsupported operation.
        estimator: &'static str,
        /// The unsupported operation name.
        operation: &'static str,
    },

    /// A primitive-layer failure (geometry / squareness / convergence /
    /// non-SPD pivot) surfaced from a `mlrs-backend` prim call the estimator
    /// composed. Transparent `#[from]` so estimator methods can `?` a prim
    /// `Result<_, PrimError>` directly.
    #[error("estimator primitive error: {0}")]
    Prim(#[from] PrimError),
}
