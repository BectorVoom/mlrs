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
/// Ridge `alpha`, the Phase-5 distance-based / iterative-solver hyperparameter
/// guards (`InvalidK` / `InvalidEps` / `InvalidMinSamples` / `InvalidL1Ratio` /
/// `InvalidC`, T-05-01-01) and the iterative-solver `NotConverged` cap, an
/// unfitted-estimator misuse, an unsupported operation (e.g. `inverse_transform`
/// on TruncatedSVD), and a transparent wrap of any underlying [`PrimError`] from
/// the primitive layer.
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

    /// A distance-based estimator (KMeans / KNeighbors*) was given an invalid
    /// neighbor / cluster count `k`. The count must satisfy `1 ≤ k ≤ n_samples`
    /// (you cannot request more clusters / neighbors than there are training
    /// samples). Rejected at `fit` *before* any kernel launch so an untrusted
    /// hyperparameter becomes a typed error, not an out-of-bounds device read
    /// (T-05-01-01 / ASVS V5).
    #[error(
        "estimator '{estimator}': k = {k} is out of range \
         (must be 1..={n_samples} = n_samples)"
    )]
    InvalidK {
        /// Which estimator rejected the value (e.g. `"kmeans"` / `"knn"`).
        estimator: &'static str,
        /// The requested neighbor / cluster count.
        k: usize,
        /// The training sample count `k` must not exceed.
        n_samples: usize,
    },

    /// DBSCAN was given a non-positive neighborhood radius `eps`. The radius must
    /// be `eps ≥ 0` (a negative radius is geometrically meaningless and would
    /// make every point noise). Rejected at `fit` (T-05-01-01).
    #[error("estimator '{estimator}': eps = {eps} is invalid (must be >= 0)")]
    InvalidEps {
        /// Which estimator rejected the value (e.g. `"dbscan"`).
        estimator: &'static str,
        /// The offending radius.
        eps: f64,
    },

    /// DBSCAN was given an invalid `min_samples`. A core point requires at least
    /// one sample in its eps-neighborhood (itself), so `min_samples ≥ 1`.
    /// Rejected at `fit` (T-05-01-01).
    #[error(
        "estimator '{estimator}': min_samples = {min_samples} is invalid \
         (must be >= 1)"
    )]
    InvalidMinSamples {
        /// Which estimator rejected the value (e.g. `"dbscan"`).
        estimator: &'static str,
        /// The offending core-point threshold.
        min_samples: usize,
    },

    /// ElasticNet / Lasso was given an `l1_ratio` outside `[0, 1]`. The mixing
    /// parameter blends the L1 and L2 penalties (`l1_ratio = 1` is pure Lasso,
    /// `l1_ratio = 0` pure Ridge) so it must lie in the closed unit interval.
    /// Rejected at `fit` (T-05-01-01).
    #[error(
        "estimator '{estimator}': l1_ratio = {l1_ratio} is invalid \
         (must be 0 <= l1_ratio <= 1)"
    )]
    InvalidL1Ratio {
        /// Which estimator rejected the value (e.g. `"elastic_net"` / `"lasso"`).
        estimator: &'static str,
        /// The offending mixing parameter.
        l1_ratio: f64,
    },

    /// LogisticRegression was given a non-positive inverse-regularization `C`.
    /// `C` scales the data-fit term against the L2 penalty and must be `C > 0`
    /// (sklearn's contract); `C ≤ 0` makes the objective unbounded / degenerate.
    /// Rejected at `fit` (T-05-01-01).
    #[error("estimator '{estimator}': C = {c} is invalid (must be > 0)")]
    InvalidC {
        /// Which estimator rejected the value (e.g. `"logistic_regression"`).
        estimator: &'static str,
        /// The offending inverse-regularization strength.
        c: f64,
    },

    /// A random-projection estimator (Gaussian/Sparse) — specifically the sparse
    /// Achlioptas path — was given an out-of-range `density`. The density is the
    /// expected fraction of non-zero entries in the projection matrix and must lie
    /// in `(0, 1]` (`density = 1` is the fully-dense Gaussian-style limit; a
    /// non-positive or `> 1` density is meaningless). Rejected at `fit`/construction
    /// *before* any RNG matrix allocation so an untrusted hyperparameter becomes a
    /// typed error, not an out-of-bounds device write (T-07-01 / ASVS V5).
    #[error(
        "estimator '{estimator}': density = {density} is invalid \
         (must be 0 < density <= 1)"
    )]
    InvalidDensity {
        /// Which estimator rejected the value (e.g. `"sparse_random_projection"`).
        estimator: &'static str,
        /// The offending sparsity density.
        density: f64,
    },

    /// A streaming estimator (IncrementalPCA) was given an invalid `batch_size`.
    /// Each `partial_fit` batch must contain at least one sample, so
    /// `batch_size ≥ 1`. Rejected at `fit`/construction *before* any kernel launch
    /// (T-07-01 / ASVS V5).
    #[error(
        "estimator '{estimator}': batch_size = {batch_size} is invalid \
         (must be >= 1)"
    )]
    InvalidBatchSize {
        /// Which estimator rejected the value (e.g. `"incremental_pca"`).
        estimator: &'static str,
        /// The offending batch size.
        batch_size: usize,
    },

    /// The Johnson–Lindenstrauss `johnson_lindenstrauss_min_dim` helper (and the
    /// `n_components='auto'` random-projection path that calls it) was given an
    /// out-of-range distortion `eps`. The maximum-distortion parameter must lie in
    /// the open interval `(0, 1)` (sklearn's contract); `eps ≤ 0` or `eps ≥ 1`
    /// makes the JL minimum-dimension bound undefined. Rejected *before* any
    /// computation (T-07-01 / ASVS V5).
    ///
    /// NOTE: this is the projection-domain `eps` (distortion) — DISTINCT from the
    /// DBSCAN neighborhood-radius `InvalidEps` above, which has a different valid
    /// range (`eps ≥ 0`) and a different meaning. Both keep their own variant.
    #[error(
        "estimator '{estimator}': eps = {eps} is invalid \
         (must be 0 < eps < 1)"
    )]
    InvalidEpsDistortion {
        /// Which estimator rejected the value (e.g. `"random_projection"`).
        estimator: &'static str,
        /// The offending distortion bound.
        eps: f64,
    },

    /// A kernel-density estimator (KernelDensity) was given a non-positive
    /// `bandwidth`. The kernel bandwidth `h` scales the per-sample distances and
    /// must be `h > 0` (sklearn's contract — `Interval(Real, 0, None,
    /// closed='neither')`); a non-positive bandwidth makes the density
    /// normalization (which divides by a power of `h`) undefined. Rejected at
    /// `fit` *before* any kernel launch so an untrusted hyperparameter becomes a
    /// typed error, not a divide-by-zero / out-of-bounds device read (T-08-01-01 /
    /// ASVS V5).
    #[error(
        "estimator '{estimator}': bandwidth = {bandwidth} is invalid (must be > 0)"
    )]
    InvalidBandwidth {
        /// Which estimator rejected the value (e.g. `"kernel_density"`).
        estimator: &'static str,
        /// The offending bandwidth.
        bandwidth: f64,
    },

    /// A polynomial-kernel estimator (KernelRidge with `kernel="poly"`) was given
    /// a `degree` below 1. The polynomial kernel `(γ·⟨x,y⟩ + coef0)^degree`
    /// requires `degree ≥ 1` (sklearn's contract — `Interval(Real, 1, None,
    /// closed='left')`); a degree below 1 is not a valid polynomial-kernel order.
    /// Rejected at `fit` *before* any kernel launch (T-08-01-01 / ASVS V5).
    #[error(
        "estimator '{estimator}': degree = {degree} is invalid (must be >= 1)"
    )]
    InvalidDegree {
        /// Which estimator rejected the value (e.g. `"kernel_ridge"`).
        estimator: &'static str,
        /// The offending polynomial degree.
        degree: f64,
    },

    /// A kernel estimator (KernelRidge) was given a non-finite resolved `gamma`.
    /// The kernel coefficient `γ` scales the inner-product / squared-distance
    /// argument of every kernel that uses it (`rbf`/`poly`/`sigmoid`); sklearn's
    /// contract is `Interval(Real, 0, None, closed='neither')` (a positive finite
    /// value), and either an explicit user-supplied `gamma` or the resolved
    /// `1/n_features` default must be finite. A non-finite `gamma` (NaN / ±inf)
    /// reaches the device kernels unguarded and drives `powf`/`exp` to NaN — the
    /// same untrusted-hyperparameter class the validate-before-launch contract
    /// covers — so it is rejected at `fit` *before* any kernel launch (T-08-01-01 /
    /// ASVS V5).
    #[error(
        "estimator '{estimator}': gamma = {gamma} is invalid (must be finite)"
    )]
    InvalidGamma {
        /// Which estimator rejected the value (e.g. `"kernel_ridge"`).
        estimator: &'static str,
        /// The offending (non-finite) kernel coefficient.
        gamma: f64,
    },

    /// A kernel estimator (KernelRidge / KernelDensity) was given an unrecognised
    /// `kernel` name. Only the supported kernel families are accepted
    /// (KernelRidge: `linear`/`rbf`/`poly`/`sigmoid`; KernelDensity:
    /// `gaussian`/`tophat`/`epanechnikov`/`exponential`/`linear`/`cosine`); any
    /// other name is rejected at `fit` *before* any kernel launch so an untrusted
    /// string becomes a typed error, not a silent fall-through (T-08-01-01 /
    /// ASVS V5). Carries the offending (owned) name for diagnosis.
    #[error("estimator '{estimator}': kernel '{kernel}' is not supported")]
    InvalidKernel {
        /// Which estimator rejected the value (e.g. `"kernel_ridge"` /
        /// `"kernel_density"`).
        estimator: &'static str,
        /// The unrecognised kernel name the caller supplied.
        kernel: String,
    },

    /// An iterative solver (coordinate descent for Lasso/ElasticNet, L-BFGS for
    /// LogisticRegression) failed to reach its convergence tolerance within the
    /// `max_iter` cap. Surfaced as a typed error rather than silently returning a
    /// non-converged estimate (D-06), carrying the `max_iter` bound that was
    /// reached so the caller can raise it.
    #[error(
        "estimator '{estimator}': failed to converge within max_iter = {max_iter} \
         iterations"
    )]
    NotConverged {
        /// Which estimator's solver did not converge (e.g. `"lasso"` /
        /// `"logistic_regression"`).
        estimator: &'static str,
        /// The iteration cap that was reached without converging.
        max_iter: usize,
    },

    /// A primitive-layer failure (geometry / squareness / convergence /
    /// non-SPD pivot) surfaced from a `mlrs-backend` prim call the estimator
    /// composed. Transparent `#[from]` so estimator methods can `?` a prim
    /// `Result<_, PrimError>` directly.
    #[error("estimator primitive error: {0}")]
    Prim(#[from] PrimError),
}
