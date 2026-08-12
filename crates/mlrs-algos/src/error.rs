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
        "estimator '{estimator}': gamma = {gamma} is invalid \
         (must be a finite value > 0)"
    )]
    InvalidGamma {
        /// Which estimator rejected the value (e.g. `"kernel_ridge"`).
        estimator: &'static str,
        /// The offending (non-finite) kernel coefficient.
        gamma: f64,
    },

    /// A spectral estimator (SpectralEmbedding / SpectralClustering) was given
    /// more samples than the dense eigensolver cap (`n_samples > 64`). The
    /// normalized Laplacian is `n_samples × n_samples` and v1 `eig` caps
    /// `n ≤ MAX_DIM = 64` (the full-spectrum cyclic-Jacobi solver, D-05). The
    /// guard is applied at `fit` *before* any affinity / Laplacian / eig launch so
    /// the message names the SPECTRAL cap rather than deferring to `eig`'s generic
    /// [`mlrs_core::PrimError::NotSquare`] (D-06 / ASVS V5). Carries the requested
    /// `n_samples` and the `max = MAX_DIM` it exceeded.
    #[error(
        "estimator '{estimator}': n_samples = {n_samples} exceeds the dense \
         eigensolver cap (must be <= {max} = MAX_DIM)"
    )]
    NSamplesExceedsMaxDim {
        /// Which estimator rejected the value (e.g. `"spectral_embedding"` /
        /// `"spectral_clustering"`).
        estimator: &'static str,
        /// The training sample count the caller supplied.
        n_samples: usize,
        /// The inclusive cap `MAX_DIM` (`64`) that was exceeded.
        max: usize,
    },

    /// `LinearRegression::fit` was given more features than the large-`n_samples`
    /// Gram+eig path's dense eigensolver cap (`n_features > 64`). Above the
    /// direct-SVD size (`n_samples.max(n_features) > jacobi_svd::MAX_ROWS`), the
    /// estimator forms the `n_features × n_features` Gram (unbounded in
    /// `n_samples`, PRIM-01 GEMM) and eigendecomposes it (PRIM-05 `eig`, capped at
    /// `MAX_DIM = 64` — mirrors the spectral-family cap, D-06). A wide design
    /// that also exceeds the direct-SVD row cap falls in neither path's covered
    /// geometry and is rejected here *before* any Gram/eig launch, naming the
    /// LINEAR-01 cap rather than deferring to `eig`'s generic
    /// [`mlrs_core::PrimError::NotSquare`].
    #[error(
        "estimator '{estimator}': n_features = {n_features} exceeds the large-\
         n_samples Gram+eig cap (must be <= {max} = MAX_DIM); reduce n_features \
         or keep n_samples.max(n_features) within the direct-SVD path"
    )]
    NFeaturesExceedsMaxDim {
        /// Which estimator rejected the value (`"linear_regression"`).
        estimator: &'static str,
        /// The feature count the caller supplied.
        n_features: usize,
        /// The inclusive cap `MAX_DIM` (`64`) that was exceeded.
        max: usize,
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

    /// A mixture estimator was given an injected parameter whose CONTENT is
    /// invalid, where "invalid" could only be determined once `n_features` was
    /// known (the D-08 split puts the data-INDEPENDENT half at `build()`).
    ///
    /// For `GaussianMixture` (MIX-01) those are the three initializations
    /// (`weights_init` / `means_init` / `precisions_init`): a weight outside
    /// `[0, 1]`, a weight vector that does not sum to one, or a non-finite
    /// entry — sklearn's `_check_weights` / `_check_means` / `_check_precisions`
    /// `ValueError`s.
    ///
    /// For `BayesianGaussianMixture` (MIX-02) they are the PRIORS whose
    /// validity depends on the design: `degrees_of_freedom_prior` must exceed
    /// `n_features − 1`, `mean_prior` must have length `n_features`, and
    /// `covariance_prior` must have the `covariance_type`'s shape and be
    /// positive definite — sklearn's `_check_means_parameters` /
    /// `_check_precision_parameters` / `_checkcovariance_prior_parameter`.
    #[error("estimator '{estimator}': {param} is invalid — {reason}")]
    InvalidMixtureInit {
        /// Which estimator rejected the value (`"gaussian_mixture"` /
        /// `"bayesian_gaussian_mixture"`).
        estimator: &'static str,
        /// Which injected parameter or prior was rejected.
        param: &'static str,
        /// The content-validity reason.
        reason: String,
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

    /// A classifier (LinearSVC / MBSGDClassifier) was given a label vector that
    /// is not a valid CLASSIFICATION target: a non-integer label value, or a
    /// class count that does not match the estimator's task (the binary linear
    /// classifiers require EXACTLY 2 distinct classes). This is a data-VALIDITY
    /// failure, distinct from a geometry [`PrimError::ShapeMismatch`] (WR-07):
    /// the labels have the right SHAPE, their CONTENT is invalid. Rejected at
    /// `fit` *before* the solve, carrying an honest reason string rather than a
    /// fabricated row/col/len shape error.
    #[error("estimator '{estimator}': invalid labels — {reason}")]
    InvalidLabels {
        /// Which estimator rejected the labels (e.g. `"linear_svc"` /
        /// `"mbsgd_classifier"`).
        estimator: &'static str,
        /// The data-validity reason (e.g. `"labels must be integers"` or
        /// `"binary classifier needs exactly 2 classes, found 3"`).
        reason: String,
    },

    /// A count-based Naive Bayes variant (`MultinomialNB` / `ComplementNB`) was
    /// given a feature matrix outside its DOMAIN: the model reads `X` as
    /// occurrence counts, so a negative entry has no meaning and would drive
    /// `log` of a negative smoothed count.
    ///
    /// Distinct from [`AlgoError::InvalidLabels`], which this used to borrow:
    /// the labels are fine, it is `X` that is invalid, and saying "invalid
    /// labels" sent readers to the wrong operand.
    ///
    /// The `reason` reproduces scikit-learn's own `check_non_negative` wording
    /// ("Negative values in data passed to {name} (input X)") so a caller
    /// matching on sklearn's message — including sklearn's own
    /// `check_positive_only_tag_during_fit` — sees what it expects.
    #[error("estimator '{estimator}': invalid input X — {reason}")]
    InvalidFeatureInput {
        /// Which estimator rejected the matrix (e.g. `"multinomial_nb"`).
        estimator: &'static str,
        /// The data-validity reason, in sklearn's wording.
        reason: String,
    },

    /// `CategoricalNB` (Phase 11) was given a feature matrix that is not a valid
    /// non-negative-INTEGER categorical encoding: a negative value, a non-integer
    /// value, or a predict-time category index that exceeds the per-feature
    /// category count learned at `fit`. Like [`AlgoError::InvalidLabels`] this is
    /// a data-VALIDITY failure (the matrix has the right SHAPE, its CONTENT is
    /// invalid), distinct from a geometry [`PrimError::ShapeMismatch`]. Carries an
    /// honest reason string. Rejected at `fit` / `predict` (data-DEPENDENT, D-05).
    #[error("estimator '{estimator}': invalid categorical input — {reason}")]
    InvalidCategoricalInput {
        /// Which estimator rejected the input (always `"categorical_nb"`).
        estimator: &'static str,
        /// The data-validity reason (e.g. `"feature values must be non-negative
        /// integers"` or `"category index 5 >= n_categories 4 for feature 2"`).
        reason: String,
    },

    /// A cross-module invariant carried across a host round-trip was violated at
    /// a consumption boundary — e.g. a KNN neighbour index that should be `< n`
    /// (the Phase-13 prim guarantee) was found `>= n` when consumed by the UMAP
    /// fuzzy-graph affinity write. Surfaced as a typed error rather than a silent
    /// out-of-bounds host write / panic so a future KNN-prim regression (or a NaN
    /// float-encoded index whose `round()` yields a huge value) is caught at the
    /// boundary instead of corrupting memory (T-14-05 / ASVS V5). Carries an
    /// honest reason string describing the violated invariant.
    #[error("estimator '{estimator}': invalid graph input — {reason}")]
    InvalidGraphInput {
        /// Which estimator detected the violation (e.g. `"umap"`).
        estimator: &'static str,
        /// The invariant-violation reason (e.g. `"knn index 17 >= n_samples 16"`).
        reason: String,
    },

    /// TSNE was fitted with a `perplexity >= n_samples` (TSNE-01). sklearn
    /// raises `"perplexity must be less than n_samples"` — the conditional-
    /// Gaussian entropy target is unreachable otherwise. Data-DEPENDENT, so
    /// rejected at `fit` (the data-INDEPENDENT `perplexity > 0` bound is
    /// [`BuildError::InvalidPerplexity`]).
    #[error(
        "estimator '{estimator}': perplexity = {perplexity} must be less than \
         n_samples = {n_samples}"
    )]
    InvalidPerplexity {
        /// Which estimator rejected the value.
        estimator: &'static str,
        /// The offending perplexity.
        perplexity: f64,
        /// The training sample count the perplexity must stay under.
        n_samples: usize,
    },

    /// A `fit`-time `sample_weight` carried a NEGATIVE or non-finite entry.
    /// The weights are data (they arrive with `X`/`y`, not at `build()`), so
    /// this is an `AlgoError`. It is rejected rather than propagated because
    /// every weighted solver path either takes `√wᵢ` (NaN for a negative
    /// weight, which then silently poisons the Gram and every reduction
    /// downstream of it) or folds `wᵢ` straight into a gradient. Carries the
    /// FIRST offending index so the caller can find the row.
    #[error(
        "estimator '{estimator}': sample_weight[{index}] = {value} is invalid \
         (must be finite and >= 0)"
    )]
    InvalidSampleWeight {
        /// Which estimator rejected the weights (e.g. `"ridge"`).
        estimator: &'static str,
        /// Index of the first offending weight.
        index: usize,
        /// The offending weight value.
        value: f64,
    },

    /// A `fit`-time `sample_weight` was ALL ZERO. Every sample being weighted
    /// out leaves no data to fit, and the penalized solve would silently return
    /// the all-zero coefficient vector as if it were a real answer. sklearn's
    /// `check_all_zero_sample_weights_error` requires a `ValueError` here whose
    /// message mentions both "weight" and "zero" — the wording below satisfies
    /// its `r"(.*weight.*zero.*)|(.*zero.*weight.*)"` pattern.
    #[error(
        "estimator '{estimator}': sample_weight sums to zero — at least one \
         sample must carry a non-zero weight"
    )]
    ZeroSampleWeightSum {
        /// Which estimator rejected the weights (e.g. `"ridge"`).
        estimator: &'static str,
    },

    /// A `feature_selection` selector was given a hyperparameter outside its
    /// sklearn-documented domain (FSEL-01): `SelectPercentile`'s
    /// `percentile ∉ [0, 100]`, an `alpha ∉ [0, 1]` on
    /// `SelectFpr`/`SelectFdr`/`SelectFwe`, `GenericUnivariateSelect`'s
    /// `param < 0`, `RFE`'s non-positive `step`, `SequentialFeatureSelector`'s
    /// `n_features_to_select >= n_features`, or a negative
    /// `SelectFromModel(threshold=..)` scale factor.
    ///
    /// One variant rather than nine because the check is uniformly "this number
    /// is outside this interval" and the interesting content is WHICH parameter
    /// and WHY — carrying that as a `reason` string keeps the message as
    /// specific as sklearn's `_parameter_constraints` rejection without a
    /// variant per selector. Rejected at `build()` when the bound is
    /// data-INDEPENDENT and at `fit` when it depends on `n_features` (D-05).
    #[error("estimator '{estimator}': {param} = {value} is invalid ({reason})")]
    InvalidSelectorParam {
        /// Which selector rejected the value (e.g. `"select_percentile"`).
        estimator: &'static str,
        /// The sklearn parameter name (e.g. `"percentile"`, `"alpha"`, `"step"`).
        param: &'static str,
        /// The offending value.
        value: f64,
        /// The documented domain that was violated (e.g. `"must be in [0, 100]"`).
        reason: &'static str,
    },

    /// A `feature_selection` selector was given an unrecognised string option
    /// (FSEL-01): `GenericUnivariateSelect(mode=..)` outside
    /// `{percentile, k_best, fpr, fdr, fwe}`, `SequentialFeatureSelector`'s
    /// `direction` outside `{forward, backward}`, or a `SelectFromModel`
    /// `threshold` string that is neither `"mean"`, `"median"`, nor
    /// `"<scale>*mean"` / `"<scale>*median"`.
    ///
    /// Carries the offending value so the message names it, mirroring sklearn's
    /// `StrOptions` rejection. Distinct from [`AlgoError::UnknownAlgorithm`] and
    /// friends only in that it names the PARAMETER too, because these selectors
    /// have several string-valued parameters each.
    #[error(
        "estimator '{estimator}': unknown {param} '{value}' \
         (expected one of: {expected})"
    )]
    UnknownSelectorOption {
        /// Which selector rejected the option.
        estimator: &'static str,
        /// The sklearn parameter name (e.g. `"mode"`, `"direction"`).
        param: &'static str,
        /// The offending value, as given.
        value: String,
        /// The accepted set, in sklearn's order.
        expected: &'static str,
    },

    /// A univariate selector's `score_func` returned no p-values but the
    /// selection mode needs them (FSEL-01).
    ///
    /// `SelectFpr` / `SelectFdr` / `SelectFwe` (and `GenericUnivariateSelect` in
    /// those modes) threshold `pvalues_`, which `r_regression` and both
    /// `mutual_info_*` functions do not produce — sklearn's own docstrings say
    /// so ("Function taking two arrays X and y, and returning a pair of arrays
    /// (scores, pvalues)" for these three, versus "or a single array with
    /// scores" for `SelectKBest`/`SelectPercentile`). sklearn itself fails with
    /// a `TypeError` deep inside `_get_support_mask`; this reports the mismatch
    /// as a typed error at `fit`, naming both sides.
    #[error(
        "estimator '{estimator}': selection mode '{mode}' requires p-values, \
         but the score function returned scores only"
    )]
    ScoreFuncHasNoPValues {
        /// Which selector needed them (e.g. `"select_fdr"`).
        estimator: &'static str,
        /// The mode that needs them (`"fpr"` / `"fdr"` / `"fwe"`).
        mode: &'static str,
    },

    /// A construction-class hyperparameter violation raised by a NON-builder
    /// entry point (FIL-01: `ForestInference::from_trees` validates an
    /// imported forest — empty forest / depth over the complete-layout cap —
    /// with the SAME typed vocabulary the builders use, but surfaces it from
    /// a fallible constructor that also raises `Prim` geometry errors, so
    /// the two classes share one `Result` error type here). Transparent
    /// `#[from]`.
    #[error("estimator build error: {0}")]
    Build(#[from] BuildError),

    /// A primitive-layer failure (geometry / squareness / convergence /
    /// non-SPD pivot) surfaced from a `mlrs-backend` prim call the estimator
    /// composed. Transparent `#[from]` so estimator methods can `?` a prim
    /// `Result<_, PrimError>` directly.
    #[error("estimator primitive error: {0}")]
    Prim(#[from] PrimError),
}

/// Construction-time (data-INDEPENDENT) hyperparameter errors for the Phase-10
/// builder-fronted estimators (D-08 split validation / D-09 single PyO3 mapping).
///
/// `BuildError` is the second, sibling failure class to [`AlgoError`]: where
/// `AlgoError` carries the data-DEPENDENT failures raised at `fit` (geometry,
/// label integrality, convergence), `BuildError` carries the data-INDEPENDENT
/// hyperparameter validation that the Phase-10 builders perform at
/// `build() -> Result<Estimator, BuildError>` (D-01/D-08) BEFORE any data is
/// seen. It also folds the enum `TryFrom<&str>` failures
/// (`UnknownLoss`/`UnknownPenalty`/`UnknownLearningRate`) so a SINGLE
/// `build_err_to_py` mapper at the PyO3 boundary covers every construction
/// failure as a `PyValueError` (D-09 — mirrors the single-site `algo_err_to_py`
/// rationale; sklearn raises these at construction, mlrs at the first `fit`
/// because the Unfit arm stores the raw strings until then).
///
/// `thiserror` in libraries (project convention; `anyhow` is reserved for the
/// PyO3 boundary). The two enums are deliberately separate types so the prim /
/// fit layer cannot accidentally surface a construction error and vice versa.
#[derive(Debug, Error)]
pub enum BuildError {
    /// A penalized estimator was given a negative `alpha`. Every Phase-10
    /// estimator requires `alpha >= 0` (`alpha = 0` degenerates to the
    /// unpenalized objective); a negative penalty is undefined. Rejected at
    /// `build()` (T-10-01-01).
    #[error("estimator '{estimator}': alpha = {alpha} is invalid (must be >= 0)")]
    InvalidAlpha {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending penalty value.
        alpha: f64,
    },

    /// An ElasticNet-penalty estimator was given an `l1_ratio` outside `[0, 1]`.
    /// The mixing parameter blends the L1 and L2 penalties so it must lie in the
    /// closed unit interval. Rejected at `build()` (T-10-01-01).
    #[error(
        "estimator '{estimator}': l1_ratio = {l1_ratio} is invalid \
         (must be 0 <= l1_ratio <= 1)"
    )]
    InvalidL1Ratio {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending mixing parameter.
        l1_ratio: f64,
    },

    /// DBSCAN was given a non-positive (or non-finite) neighborhood radius `eps`.
    /// The radius must be `eps >= 0` (a negative radius is geometrically
    /// meaningless and would make every point noise). Rejected at `build()`
    /// (data-INDEPENDENT, the D-08 split, T-05-07-01) — the construction-time
    /// sibling of [`AlgoError::InvalidEps`].
    #[error("estimator '{estimator}': eps = {eps} is invalid (must be >= 0)")]
    InvalidEps {
        /// Which estimator's builder rejected the value (e.g. `"dbscan"`).
        estimator: &'static str,
        /// The offending radius.
        eps: f64,
    },

    /// A linear-SVM estimator (LinearSVC / LinearSVR) was given a non-positive
    /// inverse-regularization `C`. `C` scales the data-fit (hinge / epsilon-tube)
    /// term against the L2 penalty and must be `C > 0` (sklearn's contract); a
    /// non-positive `C` makes the regularized objective degenerate. Rejected at
    /// `build()` (T-10-04-01) — the construction-time (data-INDEPENDENT) sibling
    /// of [`AlgoError::InvalidC`], which the fit-time solvers raise.
    #[error("estimator '{estimator}': C = {c} is invalid (must be > 0)")]
    InvalidC {
        /// Which estimator's builder rejected the value (e.g. `"linear_svc"`).
        estimator: &'static str,
        /// The offending inverse-regularization strength.
        c: f64,
    },

    /// An SGD estimator was given a non-positive initial learning rate `eta0`.
    /// The `constant` / `invscaling` schedules require `eta0 > 0`; a non-positive
    /// initial rate makes the schedule degenerate. Rejected at `build()`
    /// (T-10-01-01).
    #[error("estimator '{estimator}': eta0 = {eta0} is invalid (must be > 0)")]
    InvalidEta0 {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending initial learning rate.
        eta0: f64,
    },

    /// An epsilon-insensitive estimator was given a negative `epsilon`. The
    /// insensitivity margin must be `epsilon >= 0`. Rejected at `build()`
    /// (T-10-01-01).
    #[error("estimator '{estimator}': epsilon = {epsilon} is invalid (must be >= 0)")]
    InvalidEpsilon {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending insensitivity margin.
        epsilon: f64,
    },

    /// An SGD estimator was given a non-finite inverse-scaling exponent
    /// `power_t` (NaN / ±inf). `power_t` flows into the `invscaling` schedule
    /// `eta0 / t^power_t`; a non-finite value drives the step rate to NaN/inf and
    /// diverges the solve. Rejected at `build()` (T-10-03-01 — the same
    /// untrusted-hyperparameter class as the sibling schedule scalars). NOTE: a
    /// NEGATIVE finite `power_t` is ACCEPTED but makes the step rate GROW with
    /// `t` (and `power_t = 0` degenerates `invscaling` to constant) — these are
    /// sklearn-divergent but well-defined, so they are documented rather than
    /// rejected.
    #[error(
        "estimator '{estimator}': power_t = {power_t} is invalid \
         (must be a finite value)"
    )]
    InvalidPowerT {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending (non-finite) inverse-scaling exponent.
        power_t: f64,
    },

    /// An unrecognised `loss` string was supplied (the
    /// [`TryFrom<&str>`](core::convert::TryFrom) enum-parse failure folded into
    /// `BuildError` so a single mapper covers it, D-09). Carries the offending
    /// (owned) name for diagnosis.
    #[error("unknown loss '{value}'")]
    UnknownLoss {
        /// The unrecognised loss name the caller supplied.
        value: String,
    },

    /// An unrecognised `init` string was supplied (the enum-parse failure
    /// folded into `BuildError`, D-09). Mirrors sklearn's `StrOptions`
    /// rejection for `KMeans(init=...)`, whose only string values are
    /// `'k-means++'` and `'random'` (an explicit `k × d` array is the third,
    /// non-string form).
    #[error("unknown init '{value}'")]
    UnknownInit {
        /// The unrecognised init name the caller supplied.
        value: String,
    },

    /// An unrecognised `n_init` string was supplied. sklearn's `n_init`
    /// constraint is `StrOptions({'auto'}) | Interval(Integral, 1, None)`, so
    /// `'auto'` is the ONLY legal string; every other string is rejected here
    /// rather than silently treated as a count.
    #[error("unknown n_init '{value}' (the only legal string is 'auto')")]
    UnknownNInit {
        /// The unrecognised `n_init` string the caller supplied.
        value: String,
    },

    /// An unrecognised `algorithm` string was supplied (the enum-parse failure
    /// folded into `BuildError`, D-09). Mirrors sklearn's `StrOptions`
    /// rejection for `KMeans(algorithm=...)` (`'lloyd'` / `'elkan'`).
    #[error("unknown algorithm '{value}'")]
    UnknownAlgorithm {
        /// The unrecognised algorithm name the caller supplied.
        value: String,
    },

    /// An unrecognised `penalty` string was supplied (the enum-parse failure
    /// folded into `BuildError`, D-09).
    #[error("unknown penalty '{value}'")]
    UnknownPenalty {
        /// The unrecognised penalty name the caller supplied.
        value: String,
    },

    /// An unrecognised `learning_rate` string was supplied (the enum-parse
    /// failure folded into `BuildError`, D-09).
    #[error("unknown learning_rate '{value}'")]
    UnknownLearningRate {
        /// The unrecognised learning-rate schedule name the caller supplied.
        value: String,
    },

    /// A `loss` value was supplied that is not valid for the target estimator
    /// (e.g. a regression loss on a classifier builder). The loss family must
    /// match the estimator's task. Rejected at `build()` (T-10-01-01).
    #[error(
        "estimator '{estimator}': loss '{loss}' is not valid for this estimator"
    )]
    InvalidLossForEstimator {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending (owned) loss name.
        loss: String,
    },

    /// `GaussianNB` was given a negative `var_smoothing` (Phase 11, T-11-01). The
    /// portion of the largest feature variance added to every variance for
    /// numerical stability must be `var_smoothing >= 0`; a negative value is
    /// undefined. Rejected at `build()` (data-INDEPENDENT, D-05).
    #[error(
        "estimator '{estimator}': var_smoothing = {var_smoothing} is invalid \
         (must be >= 0)"
    )]
    InvalidVarSmoothing {
        /// Which estimator's builder rejected the value (always `"gaussian_nb"`).
        estimator: &'static str,
        /// The offending smoothing value.
        var_smoothing: f64,
    },

    /// A Naive Bayes estimator was given a `priors` / `class_prior` vector with a
    /// non-finite or negative entry (Phase 11, T-11-01). Class priors are
    /// probabilities, so each entry must be finite and `>= 0` (the data-DEPENDENT
    /// length-`== n_classes` and sum-to-one checks stay at `fit`, D-05). Rejected
    /// at `build()` for the data-INDEPENDENT per-entry validity.
    #[error(
        "estimator '{estimator}': class prior entries must be finite and non-negative"
    )]
    InvalidClassPrior {
        /// Which estimator's builder rejected the prior vector.
        estimator: &'static str,
    },

    /// `CategoricalNB` was given a `min_categories` specification with a negative
    /// entry (Phase 11, T-11-01). The minimum number of categories per feature is
    /// a count, so every entry must be `>= 0`. Rejected at `build()`
    /// (data-INDEPENDENT, D-05) — the data-DEPENDENT length-`== n_features` check
    /// for the per-feature form stays at `fit`.
    #[error(
        "estimator '{estimator}': min_categories entries must be non-negative"
    )]
    InvalidMinCategories {
        /// Which estimator's builder rejected the value (always `"categorical_nb"`).
        estimator: &'static str,
    },

    /// `UMAP` was given a `min_dist` that is non-finite or exceeds `spread`
    /// (Phase 12, UMAP-01, T-12-02). `min_dist` controls how tightly UMAP packs
    /// points in the low-dimensional embedding and must be a finite value
    /// `<= spread`; a non-finite or larger-than-`spread` value is undefined.
    /// Rejected at `build()` (data-INDEPENDENT, the D-08 split) — never at `fit`.
    #[error(
        "estimator '{estimator}': min_dist = {min_dist} is invalid \
         (must be finite and <= spread)"
    )]
    InvalidMinDist {
        /// Which estimator's builder rejected the value (always `"umap"`).
        estimator: &'static str,
        /// The offending minimum-distance value.
        min_dist: f64,
    },

    /// `UMAP` was given an `n_components` of 0 — a degenerate embedding
    /// dimensionality (Phase 12, UMAP-01, T-12-02). umap-learn requires
    /// `n_components >= 1` (and `n_neighbors >= 1`); a value of 0 would produce a
    /// silently-empty embedding. Rejected at `build()` (data-INDEPENDENT, the
    /// D-08 split) — never at `fit`.
    #[error(
        "estimator '{estimator}': {param} = {value} is invalid (must be >= 1)"
    )]
    InvalidNComponents {
        /// Which estimator's builder rejected the value (always `"umap"`).
        estimator: &'static str,
        /// Which hyperparameter was rejected (`"n_components"` / `"n_neighbors"`).
        param: &'static str,
        /// The offending value (0).
        value: usize,
    },

    /// A neighbor estimator (`KNeighborsClassifier` / `KNeighborsRegressor` /
    /// `NearestNeighbors`) was given `n_neighbors == 0` (IN-02). The neighbor
    /// count must be `>= 1`; a value of 0 has no meaning. Rejected at `build()`
    /// (data-INDEPENDENT, the D-08 split) — the data-DEPENDENT `k > n_train`
    /// half stays in the `kneighbors` core as [`AlgoError::InvalidK`]. This is
    /// the construction-time, neighbor-honest sibling of `InvalidK`; it is
    /// distinct from `InvalidNComponents` so the variant name matches the
    /// hyperparameter it guards.
    #[error("estimator '{estimator}': n_neighbors = {n_neighbors} is invalid (must be >= 1)")]
    InvalidNNeighbors {
        /// Which estimator's builder rejected the value (e.g.
        /// `"knn_classifier"` / `"knn_regressor"` / `"nearest_neighbors"`).
        estimator: &'static str,
        /// The offending neighbor count (0).
        n_neighbors: usize,
    },

    /// AgglomerativeClustering was given `n_clusters == 0` (AGGLO-01). sklearn
    /// raises on `n_clusters <= 0`; the data-DEPENDENT `n_clusters <= n_samples`
    /// bound is a fit-body check ([`AlgoError::InvalidK`]). Rejected at
    /// `build()` (data-INDEPENDENT, the D-08 split).
    #[error("estimator '{estimator}': n_clusters = {n_clusters} is invalid (must be >= 1)")]
    InvalidNClusters {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending cluster count (0).
        n_clusters: usize,
    },

    /// A Random Forest estimator was given `n_estimators == 0` (ENSEMBLE-01).
    /// A forest must contain at least one tree. Rejected at `build()`
    /// (data-INDEPENDENT, the D-08 split).
    #[error("estimator '{estimator}': n_estimators = {n_estimators} is invalid (must be >= 1)")]
    InvalidNEstimators {
        /// Which estimator's builder rejected the value
        /// (`"random_forest_classifier"` / `"random_forest_regressor"`).
        estimator: &'static str,
        /// The offending tree count (0).
        n_estimators: usize,
    },

    /// A Random Forest estimator was given a `max_depth` outside
    /// `1 ..= 16` (ENSEMBLE-01). mlrs grows COMPLETE-layout trees whose node
    /// arrays scale as `2^(max_depth+1)`, so the depth is bounded (the cuML
    /// default cap) — a documented deviation from sklearn's unbounded
    /// `max_depth=None`. Rejected at `build()`.
    #[error(
        "estimator '{estimator}': max_depth = {max_depth} is invalid (must be in 1..=16; \
         mlrs trees are depth-bounded, unlike sklearn's max_depth=None)"
    )]
    InvalidMaxDepth {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending depth.
        max_depth: usize,
    },

    /// A Random Forest estimator was given an `n_bins` outside `2 ..= 256`
    /// (ENSEMBLE-01). Splits are histogram-binned (cuML-style); at least two
    /// bins are needed for one candidate threshold, and the per-level
    /// histogram memory scales linearly with bins. Rejected at `build()`.
    #[error("estimator '{estimator}': n_bins = {n_bins} is invalid (must be in 2..=256)")]
    InvalidNBins {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending bin count.
        n_bins: usize,
    },

    /// A Random Forest estimator was given `max_features = Value(0)`
    /// (ENSEMBLE-01). At least one feature must be sampled per node; the
    /// data-DEPENDENT `value <= n_features` half is validated at `fit`.
    /// Rejected at `build()`.
    #[error("estimator '{estimator}': max_features = {max_features} is invalid (must be >= 1)")]
    InvalidMaxFeatures {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending per-node feature count (0).
        max_features: usize,
    },

    /// A Random Forest estimator was given a non-finite or out-of-range
    /// `min_samples_split` / `min_samples_leaf` (ENSEMBLE-01): the split
    /// minimum must be `>= 2` and the leaf minimum `>= 1` (the sklearn
    /// integer-form contract). Rejected at `build()`.
    #[error(
        "estimator '{estimator}': {which} = {value} is invalid \
         (min_samples_split must be >= 2, min_samples_leaf >= 1, both finite)"
    )]
    InvalidMinSamplesForest {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// Which hyperparameter failed (`"min_samples_split"` /
        /// `"min_samples_leaf"`).
        which: &'static str,
        /// The offending value.
        value: f64,
    },

    /// A Random Forest estimator was given `oob_score = true` together with
    /// `bootstrap = false` (RF-OOB-01). Out-of-bag estimation requires each
    /// tree to be grown on a bootstrap resample so some training rows are
    /// held out per tree; with `bootstrap = false` every row is in-bag for
    /// every tree, so no OOB signal exists — mirrors sklearn's
    /// `ValueError("Out of bag estimation only available if
    /// bootstrap=True")`. Rejected at `build()` (data-INDEPENDENT, the D-08
    /// split).
    #[error(
        "estimator '{estimator}': oob_score = true requires bootstrap = true \
         (out-of-bag estimation only available if bootstrap=True)"
    )]
    OobRequiresBootstrap {
        /// Which estimator's builder rejected the value
        /// (`"random_forest_classifier"` / `"random_forest_regressor"`).
        estimator: &'static str,
    },

    /// TSNE was given a non-positive / non-finite `perplexity` (TSNE-01). The
    /// perplexity is the (soft) effective neighbor count of the conditional
    /// Gaussians and must be a finite positive number; the data-DEPENDENT
    /// `perplexity < n_samples` bound is a fit-body check
    /// ([`AlgoError::InvalidPerplexity`]). Rejected at `build()`.
    #[error(
        "estimator '{estimator}': perplexity = {perplexity} is invalid \
         (must be finite and > 0)"
    )]
    InvalidPerplexity {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending perplexity.
        perplexity: f64,
    },

    /// TSNE was given an `early_exaggeration < 1` or non-finite (TSNE-01).
    /// sklearn raises on `early_exaggeration < 1.0` (the P scale-up must not
    /// shrink the attractive forces). Rejected at `build()`.
    #[error(
        "estimator '{estimator}': early_exaggeration = {early_exaggeration} is \
         invalid (must be finite and >= 1)"
    )]
    InvalidEarlyExaggeration {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending exaggeration factor.
        early_exaggeration: f64,
    },

    /// A HistGradientBoosting estimator was given `max_iter == 0` (GBT-01).
    /// At least one boosting iteration is required. Rejected at `build()`
    /// (data-INDEPENDENT, the D-08 split).
    #[error("estimator '{estimator}': max_iter = {max_iter} is invalid (must be >= 1)")]
    InvalidMaxIter {
        /// Which estimator's builder rejected the value
        /// (`"hist_gradient_boosting_classifier"` /
        /// `"hist_gradient_boosting_regressor"`).
        estimator: &'static str,
        /// The offending iteration count (0).
        max_iter: usize,
    },

    /// A HistGradientBoosting estimator was given a non-positive or non-finite
    /// `learning_rate` (GBT-01). The shrinkage multiplies every leaf value, so
    /// it must be a finite positive number (the sklearn contract). Rejected at
    /// `build()`.
    #[error(
        "estimator '{estimator}': learning_rate = {learning_rate} is invalid \
         (must be finite and > 0)"
    )]
    InvalidLearningRate {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending shrinkage.
        learning_rate: f64,
    },

    /// A HistGradientBoosting estimator was given a negative or non-finite
    /// `l2_regularization` (GBT-01). The penalty enters every leaf-value
    /// denominator `H + λ`, so it must be finite and `>= 0` (`0` disables it —
    /// the sklearn default). Rejected at `build()`.
    #[error(
        "estimator '{estimator}': l2_regularization = {l2_regularization} is invalid \
         (must be finite and >= 0)"
    )]
    InvalidL2Regularization {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending penalty.
        l2_regularization: f64,
    },

    /// `HDBSCAN` was given a `min_cluster_size` below 2 (Phase 12, HDBS-01,
    /// T-12-02). The minimum number of samples in a cluster must be `>= 2`; a
    /// smaller value is undefined. Rejected at `build()` (data-INDEPENDENT, the
    /// D-08 split) — never at `fit`.
    #[error(
        "estimator '{estimator}': min_cluster_size = {min_cluster_size} is invalid \
         (must be >= 2)"
    )]
    InvalidMinClusterSize {
        /// Which estimator's builder rejected the value (always `"hdbscan"`).
        estimator: &'static str,
        /// The offending minimum-cluster-size value.
        min_cluster_size: usize,
    },

    /// A density estimator (`HDBSCAN` min_samples — Phase 15, HDBS-01,
    /// T-15-03-V5b; or `DBSCAN` min_samples — Phase 16, CLUSTER-02, T-05-07-01)
    /// was given a `min_samples` of 0. The core-point count must be `>= 1` (a
    /// core point counts at least itself); a value of 0 is undefined. Rejected at
    /// `build()` (data-INDEPENDENT, the D-08 split) — never at `fit`. The
    /// construction-time sibling of [`AlgoError::InvalidMinSamples`].
    #[error(
        "estimator '{estimator}': min_samples = {min_samples} is invalid \
         (must be >= 1 when Some)"
    )]
    InvalidMinSamples {
        /// Which estimator's builder rejected the value (`"hdbscan"` / `"dbscan"`).
        estimator: &'static str,
        /// The offending core-point smoothing count.
        min_samples: usize,
    },

    /// `HDBSCAN` was given a `max_cluster_size` that is neither `0` (unbounded)
    /// nor `>= min_cluster_size` (Phase 15, HDBS-01, T-15-03-V5b). A finite bound
    /// below `min_cluster_size` is contradictory (no cluster can satisfy both at
    /// once). Rejected at `build()` (data-INDEPENDENT, the D-08 split) — never at
    /// `fit`. Mirrors [`BuildError::InvalidMinClusterSize`].
    #[error(
        "estimator '{estimator}': max_cluster_size = {max_cluster_size} is invalid \
         (must be 0 = unbounded, or >= min_cluster_size = {min_cluster_size})"
    )]
    InvalidMaxClusterSize {
        /// Which estimator's builder rejected the value (always `"hdbscan"`).
        estimator: &'static str,
        /// The offending maximum-cluster-size value.
        max_cluster_size: usize,
        /// The `min_cluster_size` it failed to reach.
        min_cluster_size: usize,
    },

    /// `HDBSCAN` was given a non-positive `alpha` (Phase 15, HDBS-01,
    /// T-15-03-V5b). The robust-single-linkage distance scaling divides pairwise
    /// distances, so it must be `alpha > 0`; a non-positive value is undefined
    /// (a zero divides by zero, a negative flips distances). Rejected at
    /// `build()` (data-INDEPENDENT, the D-08 split) — never at `fit`.
    #[error("estimator '{estimator}': alpha = {alpha} is invalid (must be > 0)")]
    InvalidAlphaHdbscan {
        /// Which estimator's builder rejected the value (always `"hdbscan"`).
        estimator: &'static str,
        /// The offending scaling value.
        alpha: f64,
    },

    /// `HDBSCAN` was given a `Metric::Minkowski { p }` with `p < 1` (Phase 15,
    /// HDBS-01, T-15-03-V5b). The Minkowski exponent must be `p >= 1` for the
    /// metric to be a proper distance (the triangle inequality fails for `p < 1`);
    /// mirrors the `knn_graph` precedent. Rejected at `build()`
    /// (data-INDEPENDENT, the D-08 split) — never at `fit`.
    #[error(
        "estimator '{estimator}': minkowski p = {p} is invalid (must be >= 1)"
    )]
    InvalidMinkowskiP {
        /// Which estimator's builder rejected the value (always `"hdbscan"`).
        estimator: &'static str,
        /// The offending Minkowski exponent.
        p: f64,
    },

    /// `HDBSCAN` was given a tree `algorithm` (`kd_tree`/`ball_tree`) together
    /// with a metric that has no tree route (HDBS-PARAMS). A KD/ball tree prunes
    /// on per-axis box bounds, which requires a metric that aggregates monotonely
    /// over the feature axes; `cosine` is normalized (not an axis aggregate) and
    /// `precomputed` has no feature axes at all, so neither can be traversed.
    ///
    /// This mirrors sklearn 1.9.0 exactly, which raises
    /// `"<metric> is not a valid metric for a KDTree-based algorithm"` for the
    /// same two combinations. Rejected at `build()` (data-INDEPENDENT, the D-08
    /// split) — sklearn raises at `fit`, but the pair is knowable without data.
    #[error(
        "estimator '{estimator}': metric '{metric}' is not a valid metric for a \
         {algorithm}-based algorithm (use algorithm='brute' or 'auto')"
    )]
    InvalidAlgorithmMetric {
        /// Which estimator's builder rejected the pair (always `"hdbscan"`).
        estimator: &'static str,
        /// The tree algorithm requested (`"kd_tree"` / `"ball_tree"`).
        algorithm: &'static str,
        /// The metric that has no tree route (`"cosine"` / `"precomputed"`).
        metric: &'static str,
    },

    /// `HDBSCAN` was given a `leaf_size < 1` (HDBS-PARAMS). A leaf must hold at
    /// least one point or the tree build would never terminate. sklearn's
    /// `Interval(Integral, 1, None, closed='left')` enforces the same floor.
    /// Rejected at `build()` (data-INDEPENDENT, the D-08 split).
    #[error("estimator '{estimator}': leaf_size = {leaf_size} is invalid (must be >= 1)")]
    InvalidLeafSize {
        /// Which estimator's builder rejected the value (always `"hdbscan"`).
        estimator: &'static str,
        /// The offending leaf size.
        leaf_size: usize,
    },

    /// `HDBSCAN` was given `n_jobs = 0` (HDBS-PARAMS). joblib reads `n_jobs` as
    /// "this many workers", with negatives counting back from all cores
    /// (`-1` = all, `-2` = all but one); `0` names no worker count at all and
    /// joblib itself raises `ValueError` for it. Rejected at `build()`
    /// (data-INDEPENDENT, the D-08 split).
    #[error("estimator '{estimator}': n_jobs = 0 is invalid (use None, a positive count, or a negative joblib offset)")]
    InvalidNJobs {
        /// Which estimator's builder rejected the value (always `"hdbscan"`).
        estimator: &'static str,
    },

    /// An ARIMA order `(p, d, q)` component exceeded the bound `mlrs` enforces
    /// to keep the Kalman-filter state dimension `r = max(p, q+1)` and the
    /// differencing pass tractable (TSA-01). sklearn/statsmodels place no such
    /// cap; mlrs's is a documented v1 scope bound. Rejected at `build()`
    /// (data-INDEPENDENT).
    #[error(
        "estimator '{estimator}': order ({p}, {d}, {q}) has a component over the \
         mlrs bound (p, q <= {max_pq}, d <= {max_d})"
    )]
    InvalidArimaOrder {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The offending AR order.
        p: usize,
        /// The offending differencing order.
        d: usize,
        /// The offending MA order.
        q: usize,
        /// The `p`/`q` bound.
        max_pq: usize,
        /// The `d` bound.
        max_d: usize,
    },

    /// An iterative-solver estimator was given a negative or non-finite `tol`.
    /// sklearn's constraint is `Interval(Real, 0, None, closed="left")` — i.e.
    /// `tol >= 0` — so `0` is ACCEPTED (it just means "iterate to the cap").
    /// Rejected at `build()` (data-INDEPENDENT, the D-08 split).
    #[error("estimator '{estimator}': tol = {tol} is invalid (must be finite and >= 0)")]
    InvalidTol {
        /// Which estimator's builder rejected the value (e.g. `"ridge"`).
        estimator: &'static str,
        /// The offending stopping tolerance.
        tol: f64,
    },

    /// `TSNE` was given a Barnes-Hut `angle` outside `[0, 1]` (TSNE-PARAMS).
    /// θ is the angular size below which a quadtree cell may stand in for the
    /// points it contains; sklearn's constraint is
    /// `Interval(Real, 0, 1, closed='both')`, so both endpoints are legal —
    /// `0` degrades Barnes-Hut to the exact `O(n²)` summation and `1` is the
    /// coarsest summary. Rejected at `build()` (data-INDEPENDENT, the D-08
    /// split).
    #[error("estimator '{estimator}': angle = {angle} is invalid (must be in [0, 1])")]
    InvalidAngle {
        /// Which estimator's builder rejected the value (always `"tsne"`).
        estimator: &'static str,
        /// The offending summary angle.
        angle: f64,
    },

    /// An unrecognised `covariance_type` string was supplied to
    /// `GaussianMixture` (MIX-01). Mirrors sklearn's `StrOptions({'full',
    /// 'tied', 'diag', 'spherical'})` rejection. This is its OWN variant rather
    /// than a reuse of [`BuildError::UnknownInit`] because `covariance_type`
    /// selects the model's parameterization — a typo silently changing which
    /// family is fitted is exactly what the typed rejection prevents.
    #[error(
        "unknown covariance_type '{value}' \
         (must be one of 'full', 'tied', 'diag', 'spherical')"
    )]
    UnknownCovarianceType {
        /// The unrecognised covariance parameterization the caller supplied.
        value: String,
    },

    /// `GaussianMixture` was given a negative or non-finite `reg_covar`
    /// (MIX-01). The regularization is ADDED to the covariance diagonal to keep
    /// it positive definite, so a negative value works against the very
    /// invariant it exists to maintain; sklearn's constraint is
    /// `Interval(Real, 0.0, None, closed="left")`. Rejected at `build()`
    /// (data-INDEPENDENT, the D-08 split).
    #[error(
        "estimator '{estimator}': reg_covar = {reg_covar} is invalid \
         (must be finite and >= 0)"
    )]
    InvalidRegCovar {
        /// Which estimator's builder rejected the value (always
        /// `"gaussian_mixture"`).
        estimator: &'static str,
        /// The offending covariance regularization.
        reg_covar: f64,
    },

    /// An unrecognised `weight_concentration_prior_type` string was supplied to
    /// `BayesianGaussianMixture` (MIX-02). Mirrors sklearn's
    /// `StrOptions({'dirichlet_process', 'dirichlet_distribution'})` rejection.
    /// Its own variant rather than a reuse of
    /// [`BuildError::UnknownCovarianceType`] because it selects a different
    /// axis of the model — the PRIOR over the mixing weights, which decides
    /// whether unneeded components are shrunk away (the stick-breaking process)
    /// or merely smoothed (the symmetric Dirichlet).
    #[error(
        "unknown weight_concentration_prior_type '{value}' \
         (must be one of 'dirichlet_process', 'dirichlet_distribution')"
    )]
    UnknownWeightConcentrationPriorType {
        /// The unrecognised prior family the caller supplied.
        value: String,
    },

    /// A `BayesianGaussianMixture` prior hyperparameter was given a value
    /// outside its admissible range (MIX-02) — a non-positive
    /// `weight_concentration_prior` or `mean_precision_prior`, or a
    /// non-positive `covariance_prior` under `covariance_type='spherical'`.
    ///
    /// Only the priors whose validity is data-INDEPENDENT are rejected here
    /// (the D-08 split). `degrees_of_freedom_prior > n_features − 1`, the
    /// SHAPES of `mean_prior` / `covariance_prior`, and the positive-definiteness
    /// of a matrix `covariance_prior` all need `n_features`, so they are `fit`
    /// checks raised as [`AlgoError::InvalidMixtureInit`].
    #[error(
        "estimator '{estimator}': {param} = {value} is invalid (must be finite and > 0)"
    )]
    InvalidPrior {
        /// Which estimator's builder rejected the value (always
        /// `"bayesian_gaussian_mixture"`).
        estimator: &'static str,
        /// Which prior hyperparameter was rejected.
        param: &'static str,
        /// The offending value.
        value: f64,
    },

    /// An unrecognised `solver` string was supplied (the
    /// [`TryFrom<&str>`](core::convert::TryFrom) enum-parse failure folded into
    /// `BuildError` so a single mapper covers it, D-09). Mirrors sklearn's
    /// `StrOptions` rejection for `Ridge(solver=...)`.
    #[error("unknown solver '{value}'")]
    UnknownSolver {
        /// The unrecognised solver name the caller supplied.
        value: String,
    },

    /// An unrecognised `device` string was supplied (DEVICE-PARAM-01). The
    /// [`TryFrom<&str>`](core::convert::TryFrom) enum-parse failure folded into
    /// `BuildError` alongside [`BuildError::UnknownSolver`], so one mapper
    /// covers every `StrOptions`-shaped rejection.
    #[error("unknown device '{value}' (expected one of 'auto', 'cpu', 'gpu')")]
    UnknownDevice {
        /// The unrecognised placement name the caller supplied.
        value: String,
    },

    /// An unrecognised `gcv_mode` string was supplied to `RidgeCV` (the
    /// [`TryFrom<&str>`](core::convert::TryFrom) enum-parse failure folded into
    /// `BuildError`, exactly as [`BuildError::UnknownSolver`] is). Mirrors
    /// sklearn's `StrOptions({"auto", "svd", "eigen"})` rejection.
    #[error("unknown gcv_mode '{value}' (expected one of 'auto', 'svd', 'eigen')")]
    UnknownGcvMode {
        /// The unrecognised mode name the caller supplied.
        value: String,
    },

    /// `solver='lbfgs'` was requested with `positive=False`. sklearn's
    /// `Ridge.fit` raises exactly this combination (`"'lbfgs' solver can be used
    /// only when positive=True."`) because the L-BFGS arm exists solely to carry
    /// the non-negativity bound; every other solver is faster unconstrained.
    /// Rejected at `build()` (both operands are data-INDEPENDENT).
    #[error(
        "estimator '{estimator}': solver = 'lbfgs' can be used only when \
         positive = true"
    )]
    LbfgsRequiresPositive {
        /// Which estimator's builder rejected the combination (`"ridge"`).
        estimator: &'static str,
    },

    /// `positive=True` was requested with a solver that cannot carry the
    /// non-negativity bound. sklearn accepts only `'auto'` (which resolves to
    /// `'lbfgs'`) and `'lbfgs'`; every other solver raises. Rejected at
    /// `build()` (both operands are data-INDEPENDENT).
    #[error(
        "estimator '{estimator}': solver = '{solver}' does not support positive \
         fitting (use 'auto' or 'lbfgs', or set positive = false)"
    )]
    PositiveUnsupportedSolver {
        /// Which estimator's builder rejected the combination (`"ridge"`).
        estimator: &'static str,
        /// The offending solver name.
        solver: &'static str,
    },

    /// A scalar hyperparameter fell outside the half-line sklearn's
    /// `Interval(Real, 0, None, ...)` constraint allows.
    ///
    /// Carries the parameter NAME and the bound as text because the estimators
    /// that raise it (`BayesianRidge`'s four Gamma hyperpriors `alpha_1` /
    /// `alpha_2` / `lambda_1` / `lambda_2`, its two optional initial values
    /// `alpha_init` / `lambda_init`, and its `tol`) split across BOTH closures
    /// of that interval — the hyperpriors admit `0`, the initial values and
    /// `tol` do not — and a variant per parameter would be six variants saying
    /// the same thing. Rejected at `build()` (data-INDEPENDENT, the D-08 split).
    ///
    /// Despite the name it is the GENERIC named-scalar-bound rejection, and
    /// `SpectralClustering` reuses it for the `pairwise_kernels` extras it
    /// forwards (`degree >= 0`, `coef0` finite) for the same reason: the bound
    /// differs per parameter and the rendered message already names both.
    #[error("estimator '{estimator}': {param} = {value} is invalid (must be finite and {bound})")]
    InvalidHyperprior {
        /// Which estimator's builder rejected the value (e.g.
        /// `"bayesian_ridge"` / `"spectral_clustering"`).
        estimator: &'static str,
        /// The sklearn parameter name, so the message names what the caller set.
        param: &'static str,
        /// The offending value.
        value: f64,
        /// The violated bound, rendered into the message (`">= 0"` / `"> 0"`).
        bound: &'static str,
    },

    /// A two-ended range hyperparameter had its bounds out of order or outside
    /// its valid domain — `MinMaxScaler(feature_range=(min, max))` requires
    /// `min < max`; `RobustScaler(quantile_range=(q_min, q_max))` additionally
    /// requires both endpoints in `[0, 100]`. Rejected at `build()`
    /// (data-INDEPENDENT).
    #[error("estimator '{estimator}': {param} = ({min}, {max}) is invalid")]
    InvalidRange {
        /// Which estimator's builder rejected the value.
        estimator: &'static str,
        /// The sklearn parameter name (`"feature_range"` / `"quantile_range"`).
        param: &'static str,
        /// The offending lower bound.
        min: f64,
        /// The offending upper bound.
        max: f64,
    },

}
