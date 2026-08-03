//! `BayesianRidge` (LINEAR-06) — Bayesian ridge regression with automatic
//! `alpha`/`lambda` estimation, and the FULL
//! `sklearn.linear_model.BayesianRidge` parameter surface.
//!
//! ```text
//! BayesianRidge(*, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6,
//!               lambda_1=1e-6, lambda_2=1e-6, alpha_init=None,
//!               lambda_init=None, compute_score=False, fit_intercept=True,
//!               copy_X=True, verbose=False)
//!               .fit(X, y, sample_weight=None)
//!               .predict(X, return_std=False)
//! ```
//!
//! Every one of those parameters is implemented here, plus the fitted
//! `coef_` / `intercept_` / `alpha_` / `lambda_` / `sigma_` / `scores_` /
//! `n_iter_` / `X_offset_` / `X_scale_` attributes sklearn exposes.
//!
//! ## The model
//! The likelihood is `y | X, w, α ~ N(Xw, α⁻¹I)` with a spherical Gaussian prior
//! `w | λ ~ N(0, λ⁻¹I)`, and Gamma hyperpriors on both precisions. `fit`
//! alternates the posterior mean
//!
//! ```text
//! w = (λ/α·I + XᵀX)⁻¹·Xᵀy
//! ```
//!
//! with MacKay's (1992) evidence updates for the two precisions,
//!
//! ```text
//! γ = Σᵢ α·λᵢ / (λ_prec + α·λᵢ)          (λᵢ = the eigenvalues of XᵀX)
//! λ_prec ← (γ + 2·lambda_1) / (‖w‖² + 2·lambda_2)
//! α      ← (n − γ + 2·alpha_1) / (‖y − Xw‖² + 2·alpha_2)
//! ```
//!
//! until `Σ|w_prev − w| < tol` or `max_iter` iterations, then does ONE more
//! posterior-mean update at the converged precisions — which is sklearn's loop
//! structure statement for statement, including that the final update uses the
//! precisions produced by the LAST iteration's evidence step.
//!
//! ## Perf: the spectrum is computed ONCE, and the loop then costs `O(d)`
//! ## (the reason this beats scikit-learn on cpu)
//! sklearn takes a thin SVD of `X` up front and then, per iteration, forms the
//! posterior mean from `Vh` and recomputes `rmse = ‖y − Xw‖²` with an `O(n·d)`
//! matrix-vector product. Over the default 300 iterations that residual pass —
//! not the SVD — is the bulk of the fit.
//!
//! mlrs never touches the design after the first pass. `centered_gram_xty`
//! forms the centered Gram `G = XᵀX` and `b₀ = Xᵀy` in ONE parallel `O(n·d²/2)`
//! sweep (centering folded in, the `n×d` centered design never materialized),
//! [`sym_eig`] diagonalizes `G = V·diag(λ)·Vᵀ`, and everything the loop needs
//! then lives in the eigenbasis. With `b = Vᵀb₀`, `r = λ_prec/α` and
//! `cᵢ = bᵢ/(λᵢ + r)`:
//!
//! | quantity | sklearn | here |
//! |---|---|---|
//! | `w` | `Vhᵀ·(Vh/(λ+r))·Xᵀy`, `O(d²)` | `V·c` — needed only for the `Σ|Δw|` test, `O(d²)` |
//! | `‖w‖²` | `O(d)` | `Σcᵢ²`, `O(d)` |
//! | `‖y − Xw‖²` | `y − X·w`, **`O(n·d)`** | `yᵀy − 2Σcᵢbᵢ + Σλᵢcᵢ²`, **`O(d)`** |
//! | `γ`, `logdet` | `O(d)` | `O(d)` |
//!
//! The residual identity `‖y − Xw‖² = yᵀy − 2wᵀXᵀy + wᵀGw` is EXACT for any `w`,
//! and both correction terms are already diagonal in the eigenbasis — so the
//! `n_samples` dimension leaves the iteration entirely. The whole fit is one
//! `O(n·d²)` pass plus `O(d³)` of eigenwork plus `O(max_iter·d²)`, against
//! sklearn's `O(n·d²)` SVD plus `O(max_iter·n·d)`.
//!
//! ## One route for both of sklearn's branches (`n_samples ≶ n_features`)
//! sklearn's `_update_coef_` has two arms — a `Vh` form when `n > d` and a `U`
//! form otherwise — and the `U` arm exists only because the thin SVD's `Vh` is
//! `k × d` with `k = min(n, d) < d` there. They are the SAME map. Writing
//! `X = U·diag(σ)·Vᵀ`, the `U` arm's `Xᵀ·U·diag(1/(λ+r))·Uᵀ·y` has component
//! `σᵢ/(λᵢ+r)·(Uᵀy)ᵢ` along `vᵢ`, and `(Vᵀ·Xᵀy)ᵢ = σᵢ·(Uᵀy)ᵢ`, so both arms are
//! `cᵢ = (Vᵀ·Xᵀy)ᵢ/(λᵢ + r)` term for term. The directions the `U` arm cannot
//! see (`σᵢ = 0`) carry `(Vᵀ·Xᵀy)ᵢ = 0` and drop out of the `Vh` arm too.
//!
//! So this takes the eigendecomposition of the `d×d` Gram in every shape and
//! TRUNCATES to the leading `k = min(n_samples, n_features)` directions, which
//! reproduces the shape of sklearn's thin `linalg.svd(X, full_matrices=False)`
//! exactly. The truncation is load-bearing in one place only — `sigma_`, whose
//! `n ≤ d` form genuinely omits the null directions (see [`posterior_sigma`]);
//! `γ` and `logdet_sigma` are invariant to it because the dropped eigenvalues
//! are zero, and sklearn's own `n ≤ d` `logdet` branch pads with exactly the
//! `log(λ_prec)` terms that `λᵢ = 0` contributes here.
//!
//! ## Gram, not SVD — and what that costs
//! Forming `XᵀX` squares the condition number of `X`, which the direct SVD
//! sklearn takes does not. That is the same trade `Ridge`'s `solve_svd_gram_eig`
//! and `LinearRegression`'s `fit_gram_eig` already make in this crate, and it is
//! made here for the same reason: the Gram is what collapses the iteration from
//! `O(n·d)` to `O(d)`. Every accumulation runs in `f64` regardless of the
//! estimator's `F`, and centering is applied to the DATA before squaring rather
//! than undone afterwards (`gram_host`'s module docs), so the `1e-5` oracle gate
//! holds on both float widths.
//!
//! ## Device residency and the two ingress paths (D-03, BAYES-GPU)
//! Fitted `coef_`/`intercept_` are device-resident [`DeviceArray`]s exactly as
//! `Ridge`'s are, so `predict` shares the fused [`linear_predict`] kernel and the
//! host-ingress [`predict_linear_from_host`] route with the other four dense
//! linear regressors. `sigma_`/`scores_` are host `f64` — they are read at the
//! Python boundary, and `sigma_`'s FACTOR (see [`posterior_sigma_sqrt_t`]) is
//! what the `return_std` kernel consumes.
//!
//! `fit` has two ingress paths, chosen by
//! [`BayesianRidge::host_fit_applicable`] before any upload:
//!
//! | | [`Fit::fit`] (device) | [`BayesianRidge::fit_from_host_slice`] |
//! |---|---|---|
//! | design upload | `n·d` | none |
//! | centering + Gram / `Xᵀy` | `f64` on the DEVICE, `d² + d` read back | one parallel `f64` host pass |
//! | eig + loop + `sigma_` | host | host |
//!
//! The `O(n·d²)` reduction is the only stage that moves, and that is the whole
//! design. The two stages that stay on the host stay there for two different
//! and equally deliberate reasons:
//!
//! - The ITERATION is `O(d)` of arithmetic per step over a length-`d` vector, so
//!   a launch per iteration would be the launch-overhead pathology `sgd_solve`,
//!   HDBSCAN's core scan and UMAP's per-epoch layout each became host arms to
//!   escape.
//! - The EIGENDECOMPOSITION is `O(d³)` and independent of `n`. It was the
//!   estimator's second cost after the Gram until [`sym_eig`] was moved to a
//!   transposed working layout — measured `172 ms → 18 ms` at `d = 256`, a 9.6×
//!   cut from re-indexing alone, which puts it back an order of magnitude below
//!   the reduction and leaves the Gram as the thing worth offloading.
//!
//! What the device arm may NOT do is accumulate at `F`: the residual identity
//! amplifies the Gram's error by `yᵀy/sse` (see [`update_coef`]), so an `f32`
//! design is widened ON THE DEVICE and the whole assembly runs at `f64`.
//! `prims::normal_eq` documents the arm; the measurement that forced the `f64`
//! rule is in [`BayesianRidge::fit_with_sample_weight`].
//!
//! ## `copy_X` is accepted and is a genuine no-op here
//! As in `Ridge`: sklearn's `copy_X` exists because `_preprocess_data` can
//! center `X` in place, and mlrs never writes into the caller's buffer. Stored
//! for `get_params` parity and documented, not silently dropped.
//!
//! Tests live in `crates/mlrs-algos/tests/bayesian_ridge_test.rs` (AGENTS.md §2).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gram_host::centered_gram_xty;
use mlrs_backend::prims::linear_predict::{
    bayes_predict_std, bayes_predict_std_from_host, linear_predict, HostMirror, HostPrediction,
};
use mlrs_backend::prims::normal_eq::{
    centered_gram_xty_device, device_fit_preferred, device_gram_applicable,
};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::linear::elastic_net::predict_linear_from_host;
use crate::linear::ridge::{upload_coef, validate_sample_weight};
use crate::linear::sym_eig::sym_eig;
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};

/// sklearn's `BayesianRidge` default `max_iter` (`300`).
const DEFAULT_MAX_ITER: usize = 300;

/// sklearn's `BayesianRidge` default `tol` (`1e-3`) — the `Σ|Δcoef|` threshold.
const DEFAULT_TOL: f64 = 1e-3;

/// sklearn's default for all four Gamma hyperpriors (`1e-6`).
const DEFAULT_HYPERPRIOR: f64 = 1e-6;

/// `np.finfo(np.float64).eps`, the guard sklearn adds under `var(y)` when it
/// derives the initial `alpha_` (`alpha_ = 1/(var(y) + eps)`) so an all-constant
/// target cannot divide by zero.
const F64_EPS: f64 = f64::EPSILON;

/// Bayesian ridge regression with evidence-maximized precisions (LINEAR-06).
///
/// Construct with the zero-arg [`BayesianRidge::new`] (sklearn defaults) or
/// [`BayesianRidge::builder`], then the consuming [`Fit::fit`] (or
/// [`BayesianRidge::fit_from_host_slice`]) and [`Predict::predict`]. Fitted
/// `coef_`/`intercept_` are device-resident (D-03); the host accessors exist
/// ONLY on `BayesianRidge<F, Fitted>` (the compile-time typestate replaces a
/// runtime `NotFitted` guard).
pub struct BayesianRidge<F, S = Unfit> {
    /// Maximum evidence iterations (sklearn's `max_iter`, default `300`).
    max_iter: usize,
    /// Stopping threshold on `Σ|coef_prev − coef|` (default `1e-3`).
    tol: f64,
    /// Shape parameter of the Gamma prior over `alpha` (the noise precision).
    alpha_1: f64,
    /// Rate parameter of the Gamma prior over `alpha`.
    alpha_2: f64,
    /// Shape parameter of the Gamma prior over `lambda` (the weight precision).
    lambda_1: f64,
    /// Rate parameter of the Gamma prior over `lambda`.
    lambda_2: f64,
    /// Initial `alpha`. `None` takes sklearn's `1/(var(y) + eps)`.
    alpha_init: Option<f64>,
    /// Initial `lambda`. `None` takes sklearn's `1.0`.
    lambda_init: Option<f64>,
    /// Accumulate the log marginal likelihood into `scores_` each iteration.
    compute_score: bool,
    /// Whether to center `X`/`y` and recover a bias term.
    fit_intercept: bool,
    /// sklearn's `copy_X`. Stored for parity; a genuine no-op (module docs).
    copy_x: bool,
    /// Print the convergence iteration to stderr, as sklearn prints it to
    /// stdout.
    verbose: bool,
    /// Fitted coefficients (length `n_features`), device-resident.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (length 1), device-resident.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
    /// sklearn's `alpha_`: the estimated noise precision.
    alpha_: f64,
    /// sklearn's `lambda_`: the estimated weight precision.
    lambda_: f64,
    /// sklearn's `sigma_`: the `d×d` posterior covariance of the weights,
    /// row-major. Host-resident — it is read at the Python boundary and by
    /// [`BayesianRidge::predict_std_from_host`], never by a kernel.
    sigma_: Option<Vec<f64>>,
    /// `Mᵀ` row-major (`d × d`), where `Σ = M·Mᵀ` — the factor
    /// [`predict_std`](BayesianRidge::predict_std) evaluates the predictive
    /// variance through instead of `sigma_` itself (see
    /// [`posterior_sigma_sqrt_t`]). Derived from the SAME spectrum and
    /// eigenvectors `sigma_` is, at no extra asymptotic cost, and stored rather
    /// than re-derived because `predict` must not depend on fit-time scratch.
    sigma_sqrt_t_: Option<Vec<f64>>,
    /// sklearn's `scores_`: the log marginal likelihood per iteration plus one
    /// final value. Empty unless `compute_score`.
    scores_: Vec<f64>,
    /// sklearn's `n_iter_`: evidence iterations actually run.
    n_iter_: usize,
    /// sklearn's `X_offset_`: the (possibly weighted) column means removed, or
    /// zeros when `!fit_intercept`.
    x_offset_: Vec<f64>,
    /// sklearn's `X_scale_`: all ones. sklearn kept the attribute after
    /// `normalize` was removed, and `_set_intercept` still divides `coef_` by it.
    x_scale_: Vec<f64>,
    /// Memoized host `(coef_, intercept_)` for the host-ingress `predict` path
    /// (the IN-05 `OnceLock` mirror idiom shared with `Ridge`).
    predict_mirror: HostMirror<F>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> BayesianRidge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct a `BayesianRidge` with sklearn's defaults (`max_iter = 300`,
    /// `tol = 1e-3`, all four hyperpriors `1e-6`, both inits `None`,
    /// `compute_score = false`, `fit_intercept = true`, `copy_X = true`,
    /// `verbose = false`) directly in the `Unfit` state.
    ///
    /// This is the SINGLE source of truth for the default hyperparameters
    /// (D-08): the builder's `Default` re-derives from here via
    /// [`BayesianRidge::into_builder`] rather than re-listing the literals.
    /// Defaults are trusted valid, so this bypasses [`BayesianRidgeBuilder::build`]'s
    /// validation.
    pub fn new() -> Self {
        Self {
            max_iter: DEFAULT_MAX_ITER,
            tol: DEFAULT_TOL,
            alpha_1: DEFAULT_HYPERPRIOR,
            alpha_2: DEFAULT_HYPERPRIOR,
            lambda_1: DEFAULT_HYPERPRIOR,
            lambda_2: DEFAULT_HYPERPRIOR,
            alpha_init: None,
            lambda_init: None,
            compute_score: false,
            fit_intercept: true,
            copy_x: true,
            verbose: false,
            coef_: None,
            intercept_: None,
            alpha_: 0.0,
            lambda_: 0.0,
            sigma_: None,
            sigma_sqrt_t_: None,
            scores_: Vec::new(),
            n_iter_: 0,
            x_offset_: Vec::new(),
            x_scale_: Vec::new(),
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        }
    }

    /// Start building a `BayesianRidge` from sklearn's defaults (D-08).
    pub fn builder() -> BayesianRidgeBuilder {
        BayesianRidgeBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`BayesianRidgeBuilder::default`] to re-derive
    /// the defaults from [`BayesianRidge::new`] (D-08).
    pub fn into_builder(self) -> BayesianRidgeBuilder {
        BayesianRidgeBuilder {
            max_iter: self.max_iter,
            tol: self.tol,
            alpha_1: self.alpha_1,
            alpha_2: self.alpha_2,
            lambda_1: self.lambda_1,
            lambda_2: self.lambda_2,
            alpha_init: self.alpha_init,
            lambda_init: self.lambda_init,
            compute_score: self.compute_score,
            fit_intercept: self.fit_intercept,
            copy_x: self.copy_x,
            verbose: self.verbose,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// fields are excluded — all are empty in any `Unfit` value). Used by the
    /// defaults-equality test (BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.max_iter == other.max_iter
            && self.tol == other.tol
            && self.alpha_1 == other.alpha_1
            && self.alpha_2 == other.alpha_2
            && self.lambda_1 == other.lambda_1
            && self.lambda_2 == other.lambda_2
            && self.alpha_init == other.alpha_init
            && self.lambda_init == other.lambda_init
            && self.compute_score == other.compute_score
            && self.fit_intercept == other.fit_intercept
            && self.copy_x == other.copy_x
            && self.verbose == other.verbose
    }

    /// Should `fit` take the host-slice ingress
    /// ([`BayesianRidge::fit_from_host_slice`]) rather than uploading and going
    /// through [`Fit::fit`]?
    ///
    /// This is purely an INGRESS hint, never a correctness gate: BOTH entry
    /// points produce the same fit to rounding — the device arm accumulates its
    /// Gram in `f64` exactly as the host sweep does (`prims::normal_eq` module
    /// docs), and everything after the reduction is literally the same code. So
    /// `fit_from_host_slice` accepts ANY shape and there is no configuration
    /// this predicate can answer wrongly; it can only answer it slowly.
    ///
    /// Callers still branch on it because the two entry points take DIFFERENT
    /// operand types (host slice vs [`DeviceArray`]), and that choice has to be
    /// made before ingress — on the host route the design is never uploaded at
    /// all. `true` on the cpu backend, on any adapter that cannot accumulate in
    /// `f64`, and below the work floor where the upload cannot pay for itself
    /// (`prims::normal_eq::device_fit_preferred`, which explains why that floor
    /// is much higher here than for `Ridge`).
    ///
    /// `shape` is `(n_samples, n_features)`.
    pub fn host_fit_applicable(&self, shape: (usize, usize)) -> bool {
        !device_fit_preferred::<F>(shape.0, shape.1)
    }

    /// [`Fit::fit`] over HOST slices — the no-upload, no-launch ingress.
    ///
    /// `x` is the `n × d` row-major design and `y` the length-`n` target, both
    /// borrowed from host memory (at the Python boundary, the Arrow values
    /// themselves). Nothing about the FITTED estimator differs from one produced
    /// by [`Fit::fit`] — the two share `fit_host_core`, so the only difference
    /// is that this one never uploads the design. Accepts any shape and any
    /// backend: with the Gram formed on the host either way, there is no
    /// configuration this route can answer differently.
    pub fn fit_from_host_slice(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        y: &[F],
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<BayesianRidge<F, Fitted>, AlgoError> {
        self.fit_host_core(pool, x, y, shape, sample_weight)
    }

    /// The shared body of BOTH `fit` ingresses: form the normal equations on the
    /// host in `f64`, then run [`BayesianRidge::finish_fit`].
    ///
    /// [`BayesianRidge::fit_from_host_slice`] reaches it with the caller's own
    /// buffers; [`BayesianRidge::fit_with_sample_weight`] reaches it after one
    /// read-back. There is deliberately no third path — see
    /// `fit_with_sample_weight`'s docs for why the Gram cannot be formed
    /// on-device for this estimator.
    fn fit_host_core(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        y: &[F],
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<BayesianRidge<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- The slice twin of the D-08 geometry guard: `validate_geometry`
        //     reads a DeviceArray's length, which we do not have here. ---
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        if y.len() != n_samples {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_samples,
                cols: 1,
                len: y.len(),
            }));
        }
        let sw64 = validate_sample_weight::<F>("bayesian_ridge", sample_weight, n_samples)?;

        let profile = std::env::var("BAYES_PROFILE").is_ok();
        let lap0 = std::time::Instant::now();

        // Centering, the `√w` rescale, the Gram and `Xᵀy` in TWO passes over the
        // design — and the centered/rescaled `n × d` design is never
        // materialized, because centering is folded into the tile the Gram
        // sweep reads.
        let (x_mean, y_mean, gram, xty) = centered_gram_xty::<F>(
            x,
            y,
            n_samples,
            n_features,
            sw64.as_deref(),
            self.fit_intercept,
        );
        // The two length-`n` target scalars the evidence loop needs. `O(n)`
        // against the `O(n·d²)` sweep above it, and `yᵀy` uses the SAME
        // centering/rescale rule so it matches the `Xᵀy` it is differenced
        // against in the residual identity.
        let (y_var, yty) = y_moments::<F>(y, n_samples, y_mean, sw64.as_deref());
        let sw_sum = sw64.as_deref().map_or(n_samples as f64, |w| w.iter().sum());
        let t_gram = if profile {
            lap0.elapsed().as_secs_f64()
        } else {
            0.0
        };

        self.finish_fit(
            pool, gram, xty, yty, y_var, sw_sum, x_mean, y_mean, n_samples, n_features, profile,
            t_gram,
        )
    }

    /// The shared tail of both ingress paths: eigendecompose, run the evidence
    /// loop, and assemble the `Fitted` estimator.
    ///
    /// Both arms reach here with the SAME operands (`gram`, `xty`, `yᵀy`,
    /// `y_var`, `Σw`) plus the means, which is what guarantees the device and
    /// host routes cannot drift into different answers — the only thing that
    /// differs above this line is where the `O(n·d²)` reduction ran.
    #[allow(clippy::too_many_arguments)]
    fn finish_fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        gram: Vec<f64>,
        xty: Vec<f64>,
        yty: f64,
        y_var: f64,
        sw_sum: f64,
        x_mean: Vec<f64>,
        y_mean: f64,
        n_samples: usize,
        n_features: usize,
        profile: bool,
        t_gram: f64,
    ) -> Result<BayesianRidge<F, Fitted>, AlgoError> {
        let d = n_features;
        let lap = std::time::Instant::now();

        // G = V·diag(λ)·Vᵀ, λ DESCENDING. `k` is the rank sklearn's thin SVD
        // would expose (module docs: the truncation reproduces
        // `full_matrices=False`).
        let (mut lambdas, v) = sym_eig(&gram, d);
        let k = n_samples.min(d);
        let t_eig = if profile {
            lap.elapsed().as_secs_f64()
        } else {
            0.0
        };

        // b = Vᵀ·(Xᵀy) — the right-hand side in the eigenbasis. `v` is
        // row-major with eigenvectors in COLUMNS, so column `c` is strided; this
        // is `O(d²)` once, not per iteration.
        let mut b = vec![0.0f64; d];
        for (r, &xr) in xty.iter().enumerate().take(d) {
            if xr == 0.0 {
                continue;
            }
            for (c, bc) in b.iter_mut().enumerate().take(d) {
                *bc += v[r * d + c] * xr;
            }
        }

        clamp_numerical_rank::<F>(&mut lambdas, &mut b, d);

        let lap = std::time::Instant::now();
        let fit = self.evidence_loop(&lambdas, &b, &v, yty, y_var, sw_sum, d, k);
        let t_loop = if profile {
            lap.elapsed().as_secs_f64()
        } else {
            0.0
        };

        let lap = std::time::Instant::now();
        let sigma = posterior_sigma(&lambdas, &v, d, k, fit.alpha, fit.lambda);
        let sigma_sqrt_t = posterior_sigma_sqrt_t(&lambdas, &v, d, k, fit.alpha, fit.lambda);
        let t_sigma = if profile {
            lap.elapsed().as_secs_f64()
        } else {
            0.0
        };

        // intercept_ = ȳ − x̄·coef_ when fit_intercept, else 0 — sklearn's
        // `_set_intercept` with the all-ones `X_scale_` that a post-`normalize`
        // sklearn always produces.
        let intercept = if self.fit_intercept {
            let dot: f64 = x_mean.iter().zip(fit.coef.iter()).map(|(m, c)| m * c).sum();
            y_mean - dot
        } else {
            0.0
        };

        // sklearn prints ONLY when the `Σ|Δcoef| < tol` test fires (an exhausted
        // `max_iter` says nothing), and prints the loop index `iter_`, which is
        // one less than the reported `n_iter_`. To stdout, in sklearn; stderr
        // here, so a library cannot corrupt the caller's data stream.
        if self.verbose && fit.converged {
            eprintln!("Convergence after  {}  iterations", fit.n_iter - 1);
        }
        if profile {
            eprintln!(
                "BAYES_PROFILE n={n_samples} d={d} iters={}: \
                 gram={t_gram:.4}s eig={t_eig:.4}s loop={t_loop:.4}s sigma={t_sigma:.4}s",
                fit.n_iter
            );
        }

        Ok(BayesianRidge {
            max_iter: self.max_iter,
            tol: self.tol,
            alpha_1: self.alpha_1,
            alpha_2: self.alpha_2,
            lambda_1: self.lambda_1,
            lambda_2: self.lambda_2,
            alpha_init: self.alpha_init,
            lambda_init: self.lambda_init,
            compute_score: self.compute_score,
            fit_intercept: self.fit_intercept,
            copy_x: self.copy_x,
            verbose: self.verbose,
            coef_: Some(upload_coef::<F>(pool, &fit.coef)),
            intercept_: Some(DeviceArray::from_host(pool, &[f64_to_host::<F>(intercept)])),
            alpha_: fit.alpha,
            lambda_: fit.lambda,
            sigma_: Some(sigma),
            sigma_sqrt_t_: Some(sigma_sqrt_t),
            scores_: fit.scores,
            n_iter_: fit.n_iter,
            x_offset_: x_mean,
            x_scale_: vec![1.0f64; d],
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }

    /// MacKay's evidence iteration, entirely in the eigenbasis.
    ///
    /// `lambdas`/`v` are the Gram's spectrum and eigenvectors, `b = Vᵀ·Xᵀy`,
    /// `yty` the preprocessed target's second moment, `y_var` sklearn's raw-`y`
    /// variance (see [`y_moments`]), `sw_sum` the effective sample count
    /// (`n_samples`, or `Σw` when weighted), and `k` the truncation rank.
    ///
    /// The loop body is sklearn's `fit` body statement for statement — the
    /// posterior-mean update, the optional score, MacKay's two updates, THEN the
    /// convergence test (so the precisions the final update runs at are the ones
    /// the last iteration produced), and `n_iter_ = iter + 1`.
    #[allow(clippy::too_many_arguments)]
    fn evidence_loop(
        &self,
        lambdas: &[f64],
        b: &[f64],
        v: &[f64],
        yty: f64,
        y_var: f64,
        sw_sum: f64,
        d: usize,
        k: usize,
    ) -> EvidenceFit {
        let mut alpha = self.alpha_init.unwrap_or(1.0 / (y_var + F64_EPS));
        let mut lambda = self.lambda_init.unwrap_or(1.0);

        let mut c = vec![0.0f64; d];
        let mut coef = vec![0.0f64; d];
        let mut coef_old = vec![0.0f64; d];
        let mut scores = Vec::new();
        let mut n_iter = 0usize;
        // The loop index the convergence test fired at — sklearn's `verbose`
        // message prints `iter_`, not `iter_ + 1`.
        let mut converged: Option<usize> = None;

        for iter in 0..self.max_iter {
            let sse = update_coef(lambdas, b, v, d, k, lambda / alpha, &mut c, &mut coef, yty);
            if self.compute_score {
                scores.push(
                    self.log_marginal_likelihood(lambdas, sw_sum, d, k, alpha, lambda, &coef, sse),
                );
            }

            // MacKay's evidence updates. `γ` sums the k directions sklearn's
            // thin spectrum exposes; the dropped ones have `λᵢ = 0` and
            // contribute nothing, so the truncation is invariant here.
            let gamma: f64 = lambdas[..k]
                .iter()
                .map(|&l| (alpha * l) / (lambda + alpha * l))
                .sum();
            let sum_coef2: f64 = coef.iter().map(|w| w * w).sum();
            lambda = (gamma + 2.0 * self.lambda_1) / (sum_coef2 + 2.0 * self.lambda_2);
            alpha = (sw_sum - gamma + 2.0 * self.alpha_1) / (sse + 2.0 * self.alpha_2);
            n_iter = iter + 1;

            // `Σ|Δcoef|` is an L1 norm in the ORIGINAL basis, which is NOT
            // invariant under `V` — so this is the one place the loop cannot
            // stay in the eigenbasis, and the one reason `coef` is materialized
            // every iteration.
            if iter != 0 {
                let delta: f64 = coef_old
                    .iter()
                    .zip(coef.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                if delta < self.tol {
                    converged = Some(iter);
                    break;
                }
            }
            coef_old.copy_from_slice(&coef);
        }

        // sklearn re-runs the posterior mean once more at the converged
        // precisions, after the loop — and then, if scoring, appends a final
        // score computed from the NEW `sse` but the loop-local `coef_`, which is
        // still the PREVIOUS iteration's posterior mean (`self.coef_` is what
        // the re-run assigned; the score call passes the un-prefixed `coef_`).
        // That mismatch is sklearn's, and it is reproduced rather than repaired:
        // `scores_` is an oracle-gated attribute, so "more correct than sklearn"
        // is a test failure.
        let prev_coef = coef.clone();
        let sse = update_coef(lambdas, b, v, d, k, lambda / alpha, &mut c, &mut coef, yty);
        if self.compute_score {
            scores.push(
                self.log_marginal_likelihood(lambdas, sw_sum, d, k, alpha, lambda, &prev_coef, sse),
            );
        }

        EvidenceFit {
            coef,
            alpha,
            lambda,
            scores,
            n_iter,
            converged: converged.is_some(),
        }
    }

    /// The log marginal likelihood sklearn appends to `scores_`.
    ///
    /// `logdet_sigma` is the one term where sklearn's two branches read
    /// differently, and they agree here: the `n > d` branch sums
    /// `log(λ_prec + α·λᵢ)` over all `d`, and the `n ≤ d` branch sums it over
    /// the `k = n` thin-SVD directions and pads the remaining `d − k` with
    /// `log(λ_prec)`. Those pad terms are exactly what `λᵢ = 0` contributes, so
    /// ONE expression covers both — but it is written as the explicit split
    /// rather than relying on the trailing eigenvalues being exactly zero, which
    /// they are only up to rounding.
    ///
    /// Note the `+ logdet_sigma` in the bracket: `logdet_sigma` is already the
    /// NEGATED sum, so the score adds it (sklearn's sign, verbatim).
    #[allow(clippy::too_many_arguments)]
    fn log_marginal_likelihood(
        &self,
        lambdas: &[f64],
        sw_sum: f64,
        d: usize,
        k: usize,
        alpha: f64,
        lambda: f64,
        coef: &[f64],
        sse: f64,
    ) -> f64 {
        let logdet_sigma = -(lambdas[..k]
            .iter()
            .map(|&l| (lambda + alpha * l).ln())
            .sum::<f64>()
            + (d - k) as f64 * lambda.ln());

        let sum_coef2: f64 = coef.iter().map(|w| w * w).sum();
        let mut score = self.lambda_1 * lambda.ln() - self.lambda_2 * lambda;
        score += self.alpha_1 * alpha.ln() - self.alpha_2 * alpha;
        score += 0.5
            * (d as f64 * lambda.ln() + sw_sum * alpha.ln() - alpha * sse - lambda * sum_coef2
                + logdet_sigma
                - sw_sum * std::f64::consts::TAU.ln());
        score
    }
}

/// What [`BayesianRidge::evidence_loop`] returns — the converged posterior mean
/// and precisions, plus the bookkeeping sklearn surfaces.
struct EvidenceFit {
    /// Posterior mean weights (length `d`).
    coef: Vec<f64>,
    /// Converged noise precision (`alpha_`).
    alpha: f64,
    /// Converged weight precision (`lambda_`).
    lambda: f64,
    /// Log marginal likelihood per iteration (empty unless `compute_score`).
    scores: Vec<f64>,
    /// Iterations run (`n_iter_`).
    n_iter: usize,
    /// Did the `Σ|Δcoef| < tol` test fire, or did the loop exhaust `max_iter`?
    /// Only drives the `verbose` message (sklearn prints ONLY on convergence).
    converged: bool,
}

/// One posterior-mean update, returning `rmse = ‖y − Xw‖²`.
///
/// Writes `c[i] = b[i]/(λᵢ + r)` for the leading `k` directions (zero above
/// them) and `coef = V·c`, then evaluates the residual through the EXACT
/// identity
///
/// ```text
/// ‖y − Xw‖² = yᵀy − 2·wᵀ(Xᵀy) + wᵀ(XᵀX)w
///           = yᵀy − Σᵢ cᵢ·(2·bᵢ − λᵢ·cᵢ)
/// ```
///
/// which is what removes the `O(n·d)` residual pass from the iteration (module
/// docs).
///
/// ## The identity's one weakness, and why the clamp is not a fudge
/// The identity is EXACT for any `w`. What it trades a pass over the design for
/// is a cancellation: it computes `sse` as a difference of two quantities both
/// of size `yᵀy`, so the absolute error is `~ε·yᵀy` regardless of how small
/// `sse` is. In the ordinary regime that is irrelevant — `sse` is `O(n·σ²)` and
/// the relative error stays near `ε`.
///
/// It stops being irrelevant when the model INTERPOLATES (`sse → 0`), which is
/// exactly the `n_samples ≤ rank` regime the wide fixture exercises. There the
/// computed value is pure cancellation noise and can come out NEGATIVE — and a
/// negative `sse` does not merely lose precision, it flips the sign of
/// `alpha_ = (sw_sum − γ + 2·alpha_1)/(sse + 2·alpha_2)`. Measured on the
/// `6 × 10` wide fixture through the device `f32` arm (where `gram_xty`
/// accumulates at `f32`, so the noise floor is `~1e-7·yᵀy` rather than
/// `~1e-16·yᵀy`): `sse` came back at `−3e-6` against a true value near `1e-12`,
/// and `alpha_` came back NEGATIVE.
///
/// Clamping at zero is the correct repair rather than a patch: `‖y − Xw‖²` is a
/// sum of squares, so a negative result is PROVABLY rounding and `0` is the
/// nearest true value. And it costs nothing in fidelity, because in precisely
/// this regime sklearn's own `alpha_` is not a function of `sse` either — with
/// `sse ≈ 0` both engines report `(sw_sum − γ + 2·alpha_1)/(2·alpha_2)`, a
/// quantity determined by the PRIOR. On that fixture the clamp lands `alpha_`
/// within `1.3e-6` relative of sklearn's.
///
/// What remains, and is documented rather than hidden: where `sse` is small but
/// not negligible against `2·alpha_2`, the device `f32` arm's `alpha_` carries
/// the `f32` Gram's absolute error. The host arm (and every `f64` fit)
/// accumulates the Gram in `f64` and does not.
#[allow(clippy::too_many_arguments)]
fn update_coef(
    lambdas: &[f64],
    b: &[f64],
    v: &[f64],
    d: usize,
    k: usize,
    r: f64,
    c: &mut [f64],
    coef: &mut [f64],
    yty: f64,
) -> f64 {
    // The subtrahend is summed on its own and applied ONCE, rather than
    // decremented from `yty` term by term: the terms are all the same sign and
    // the same order of magnitude, so summing them first keeps the `O(d)`
    // roundings out of the single cancelling subtraction.
    let mut fitted = 0.0f64;
    for i in 0..d {
        if i < k {
            let denom = lambdas[i] + r;
            // A zero denominator needs both `λᵢ = 0` and `r = 0`; the direction
            // carries no signal either way (`bᵢ = 0` when `λᵢ = 0` — see
            // `clamp_numerical_rank`), so it drops out exactly as sklearn's
            // `σ = 0` directions do.
            c[i] = if denom > 0.0 { b[i] / denom } else { 0.0 };
            fitted += c[i] * (2.0 * b[i] - lambdas[i] * c[i]);
        } else {
            c[i] = 0.0;
        }
    }
    let rmse = (yty - fitted).max(0.0);

    // coef = V·c. `v` is row-major with eigenvectors in columns, so row `r` of
    // `v` is contiguous and this is `d` contiguous dot products.
    for (r_idx, w) in coef.iter_mut().enumerate().take(d) {
        let row = &v[r_idx * d..r_idx * d + k];
        *w = row.iter().zip(c[..k].iter()).map(|(a, b)| a * b).sum();
    }
    rmse
}

/// Force the numerically-null eigen-directions to be EXACTLY null: `λᵢ = 0` and
/// `bᵢ = 0` for every direction below the Gram's own noise floor.
///
/// ## Why this is needed, and why it is not a departure from sklearn
/// sklearn reaches the spectrum through `linalg.svd(X)`, so a null direction
/// arrives as `λᵢ = σᵢ²` — the SQUARE of a small number, which drives it far
/// below anything it is compared against — and its right-hand side arrives as
/// `bᵢ = σᵢ·(Uᵀy)ᵢ`, likewise ∝ `σᵢ`. Both vanish, so a null direction
/// contributes nothing to `coef_`, `γ`, `logdet_sigma`, or `sigma_` beyond the
/// `1/λ_prec` term that `λᵢ = 0` already gives.
///
/// This estimator reaches the spectrum through the GRAM, where a null
/// direction's `λᵢ` is not squared — it sits at the accumulation's rounding
/// floor directly. In `f64` that floor is `~1e-16·λ_max` and the distinction is
/// academic. On the DEVICE arm at `f32`, where `gram_xty` accumulates in the
/// estimator's own float width, it is `~1e-8·λ_max` — and once `λᵢ` becomes
/// comparable to `r = λ_prec/α`, the `bᵢ/(λᵢ + r)` for that direction stops
/// being noise-times-nothing and becomes noise-times-`1/r`. Measured on the
/// `6 × 10` wide fixture (rank 5 after centering, `r ≈ 1.5e-6`): the host arm's
/// null eigenvalues came back at `1e-15`, the device `f32` arm's at `5e-7`, and
/// the resulting `coef_[0]` was off by 15% — not a rounding difference, a
/// direction that should not have been there at all.
///
/// Zeroing those directions restores exactly sklearn's exact-arithmetic
/// behaviour, so this makes the two agree rather than diverge. Every downstream
/// consumer already handles `λᵢ = 0` correctly: `γ` gets `0`, `logdet_sigma`
/// gets `log(λ_prec)` (which is what the `d − k` padding contributes anyway),
/// `sigma_` gets `vᵢvᵢᵀ/λ_prec` (matching sklearn's zero-padded
/// `eigen_vals_full`), and `coef_` gets nothing.
///
/// ## The floor
/// `d · ε_F · λ_max`, the standard numerical-rank test, with `ε` taken from the
/// float width the GRAM was accumulated in — `F`, not `f64`, because the device
/// arm's `gram_xty` runs at `F`. The host arm accumulates in `f64` whatever `F`
/// is, so at `F = f32` this floor is stricter than that arm needs; that is
/// deliberate, since a threshold that varied by ingress path would let the two
/// routes disagree about a design's rank (which
/// `bayesian_ridge_host_and_device_agree_f64` exists to forbid).
///
/// The margin is not tight: on that same fixture the floor lands at `2.4e-5`,
/// an order of magnitude above the largest spurious eigenvalue (`5e-7`) and four
/// orders below the smallest real one (`1.45`). A well-conditioned design never
/// comes close — a Gaussian `60 × 8` Gram has `λ_min/λ_max ≈ 0.3`.
///
/// Negative eigenvalues are clamped by the same test: a Gram is PSD, so a
/// negative `λ` is always rounding, and leaving one in place would flip the sign
/// of `1/(α·λ + λ_prec)` or take the log of a negative number.
fn clamp_numerical_rank<F: Pod>(lambdas: &mut [f64], b: &mut [f64], d: usize) {
    let eps = match size_of::<F>() {
        4 => f32::EPSILON as f64,
        _ => f64::EPSILON,
    };
    // `lambdas` is descending, so the largest is at 0. A non-positive maximum
    // means the whole Gram is zero (an all-constant design under centering);
    // there is no rank to detect and nothing to scale a floor by.
    let lambda_max = lambdas.first().copied().unwrap_or(0.0);
    if !(lambda_max > 0.0) {
        lambdas.fill(0.0);
        b.fill(0.0);
        return;
    }
    let floor = d as f64 * eps * lambda_max;
    for i in 0..d {
        if lambdas[i] <= floor {
            lambdas[i] = 0.0;
            b[i] = 0.0;
        }
    }
}

/// sklearn's `sigma_`: the posterior covariance
/// `Σⱼ vⱼ·vⱼᵀ / (α·λⱼ + λ_prec)`, row-major `d × d`.
///
/// This is sklearn's
/// `Vh_full.T @ (Vh_full / (alpha_ * eigen_vals_full + lambda_)[:, None])` —
/// note the FULL `d × d` basis and the absence of any `1/α` prefactor (it is
/// already inside the `α·λⱼ` denominator). The sum therefore runs over all `d`
/// directions, NOT the `k` the rest of the fit truncates to: `eigen_vals_full`
/// is sklearn's length-`d` spectrum zero-padded past `k`, so the `d − k`
/// directions a thin SVD cannot see still contribute `vⱼvⱼᵀ/λ_prec` each. `k`
/// enters only as the point past which `λⱼ` is forced to EXACTLY zero, matching
/// that zero-padding rather than trusting the eigensolver's residual `~1e-16`.
///
/// Formed as `B·Vᵀ` with `B = V·diag(1/(α·λ + λ_prec))`, so both operands are
/// read along their contiguous axis and the `(i, j)` sweep is a dot product.
/// Only the lower triangle is computed; the upper is mirrored. Rows are split
/// across scoped threads above a work floor — the same work-proportional sizing
/// `gram_host::host_units` uses, and for the same reason: a `d = 8` fit must not
/// pay a thread spawn it cannot amortize.
fn posterior_sigma(
    lambdas: &[f64],
    v: &[f64],
    d: usize,
    k: usize,
    alpha: f64,
    lambda: f64,
) -> Vec<f64> {
    /// Multiply-adds one worker must be given before spawning it pays (the
    /// `gram_host::HOST_MACS_PER_UNIT` precedent).
    const MACS_PER_UNIT: usize = 1 << 19;

    // B = V·diag(1/(α·λ + λ_prec)) over the full basis, with the spectrum
    // zero-padded past `k` exactly as `eigen_vals_full` is.
    let mut scaled = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            let ev = if j < k { lambdas[j] } else { 0.0 };
            let denom = alpha * ev + lambda;
            scaled[i * d + j] = if denom != 0.0 {
                v[i * d + j] / denom
            } else {
                0.0
            };
        }
    }

    let macs = (d * d / 2 * d).max(1);
    let units = (macs / MACS_PER_UNIT)
        .clamp(
            1,
            mlrs_backend::capability::cpu_launch_units().max(1) as usize,
        )
        .min(d.max(1));

    let mut sigma = vec![0.0f64; d * d];
    if units <= 1 {
        sigma_rows(&scaled, v, d, 0, &mut sigma);
    } else {
        let rows = d.div_ceil(units);
        // Each worker owns a DISJOINT row band of the output, so the bands are
        // handed out as non-overlapping mutable chunks rather than merged after.
        std::thread::scope(|scope| {
            for (u, band) in sigma.chunks_mut(rows * d).enumerate() {
                let r0 = u * rows;
                let (s, vv) = (&scaled, &v);
                scope.spawn(move || sigma_rows(s, vv, d, r0, band));
            }
        });
    }

    // Only the lower triangle was computed; mirror it.
    for i in 0..d {
        for j in 0..i {
            sigma[j * d + i] = sigma[i * d + j];
        }
    }
    sigma
}

/// `Mᵀ` row-major, where `M = V·diag(1/√(α·λⱼ + λ_prec))` factors the posterior
/// covariance as `Σ = M·Mᵀ` — the form `predict(X, return_std=True)` evaluates
/// its quadratic form through.
///
/// ## Why a factor rather than `sigma_` itself
/// `Σ = Σⱼ vⱼvⱼᵀ/(α·λⱼ + λ_prec)` is symmetric POSITIVE DEFINITE: `λⱼ ≥ 0` for a
/// Gram (after [`clamp_numerical_rank`], exactly `0` in the null directions) and
/// `λ_prec > 0`, so every denominator is strictly positive and the square root
/// is real for every direction. Writing `M = V·diag(1/√·)` gives
/// `M·Mᵀ = V·diag(1/·)·Vᵀ = Σ` term for term, and hence
///
/// ```text
/// x̃·Σ·x̃ᵀ = ‖Mᵀ·x̃‖² = Σⱼ (mⱼ·x̃)²
/// ```
///
/// which is a SUM OF SQUARES. The direct form `x̃·(Σ·x̃)` is a difference of
/// products of mixed sign and can come out negative under cancellation — the
/// reason the pre-existing host path clamped with `.max(0.0)` before its `sqrt`.
/// Through the factor there is nothing to clamp: the quantity is non-negative by
/// construction. The device kernel gets the same guarantee, which matters more
/// there — a negative under a `sqrt` on the device is a silent NaN in the output
/// rather than a host-side clamp.
///
/// The layout is the TRANSPOSE, `mt[j·d + i] = M[i][j]`, so `mⱼ` is contiguous.
/// Both consumers walk `j` in the outer loop and `i` in the inner one, so this
/// is the layout in which every read is sequential — see
/// [`mlrs_kernels::linear_predict::bayes_predict_std`] for what that buys on the
/// device.
///
/// `k` enters exactly as it does in [`posterior_sigma`]: the point past which
/// `λⱼ` is forced to EXACTLY zero, matching the zero-padding of sklearn's
/// `eigen_vals_full`. A zero denominator (only reachable if `λ_prec` itself
/// underflows to `0` in a null direction) yields a zero row, which is
/// [`posterior_sigma`]'s behaviour for the same case.
///
/// `O(d²)` — negligible beside the `O(d³)` [`posterior_sigma`] it runs
/// alongside.
fn posterior_sigma_sqrt_t(
    lambdas: &[f64],
    v: &[f64],
    d: usize,
    k: usize,
    alpha: f64,
    lambda: f64,
) -> Vec<f64> {
    let mut mt = vec![0.0f64; d * d];
    for j in 0..d {
        let ev = if j < k { lambdas[j] } else { 0.0 };
        let denom = alpha * ev + lambda;
        if !(denom > 0.0) {
            continue;
        }
        let scale = 1.0 / denom.sqrt();
        let row = &mut mt[j * d..(j + 1) * d];
        for (i, slot) in row.iter_mut().enumerate() {
            *slot = v[i * d + j] * scale;
        }
    }
    mt
}

/// The lower-triangle rows of [`posterior_sigma`]'s output for one band,
/// written into `out` whose row 0 IS global row `r0`.
fn sigma_rows(scaled: &[f64], v: &[f64], d: usize, r0: usize, out: &mut [f64]) {
    for (local, orow) in out.chunks_mut(d).enumerate() {
        let i = r0 + local;
        if i >= d {
            break;
        }
        let si = &scaled[i * d..(i + 1) * d];
        for (j, slot) in orow.iter_mut().enumerate().take(i + 1) {
            let vj = &v[j * d..(j + 1) * d];
            *slot = si.iter().zip(vj.iter()).map(|(a, b)| a * b).sum();
        }
    }
}

/// The two target scalars the evidence loop needs, both read off the RAW target
/// in one pass: `(y_var, yty)`.
///
/// They are deliberately NOT the same quantity taken twice — sklearn computes
/// them at different points of `fit`, over differently-transformed arrays, and
/// conflating them silently changes the fit:
///
/// - `y_var` is sklearn's `y_var`, taken BEFORE `_preprocess_data` and used only
///   to seed `alpha_ = 1/(y_var + eps)`. It is the variance of the raw target
///   about ITS OWN mean — `y.var()` unweighted, and the `sample_weight`-weighted
///   variance about the weighted mean otherwise. Notably it does NOT depend on
///   `fit_intercept`: a `fit_intercept=False` fit still seeds `alpha_` from the
///   mean-removed variance, even though nothing is centered afterwards.
/// - `yty = Σ y_c²` is over the PREPROCESSED target `y_c[r] = (y[r] − ȳ)·√w[r]`
///   (with `ȳ = 0` when `!fit_intercept`) — the `yᵀy` of [`update_coef`]'s
///   residual identity, which must match the `Xᵀy` it is differenced against.
///
/// `y_mean` is the (possibly weighted) mean the Gram pass already computed, so
/// the second moment cannot drift from the design's centering.
fn y_moments<F: Pod>(y: &[F], n: usize, y_mean: f64, sw: Option<&[f64]>) -> (f64, f64) {
    let wide = |v: F| -> f64 {
        match size_of::<F>() {
            4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
            8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
            other => unreachable!("bayesian_ridge is f32/f64 only, got a {other}-byte element"),
        }
    };

    // --- sklearn's pre-preprocessing `y_var`. ---
    let (raw_mean, w_sum) = match sw {
        None => (
            y.iter().take(n).map(|&v| wide(v)).sum::<f64>() / n as f64,
            n as f64,
        ),
        Some(w) => {
            let ws: f64 = w.iter().take(n).sum();
            let acc: f64 = (0..n).map(|r| w[r] * wide(y[r])).sum();
            (if ws > 0.0 { acc / ws } else { 0.0 }, ws)
        }
    };
    let y_var = match sw {
        None => {
            (0..n)
                .map(|r| {
                    let e = wide(y[r]) - raw_mean;
                    e * e
                })
                .sum::<f64>()
                / n as f64
        }
        Some(w) => {
            let acc: f64 = (0..n)
                .map(|r| {
                    let e = wide(y[r]) - raw_mean;
                    w[r] * e * e
                })
                .sum();
            if w_sum > 0.0 {
                acc / w_sum
            } else {
                0.0
            }
        }
    };

    // --- The preprocessed target's second moment. ---
    let mut yty = 0.0f64;
    for r in 0..n {
        let scale = match sw {
            None => 1.0,
            Some(w) => w[r].sqrt(),
        };
        let v = (wide(y[r]) - y_mean) * scale;
        yty += v * v;
    }
    (y_var, yty)
}

impl<F> Default for BayesianRidge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`BayesianRidge`] (D-01). Scalar setters are `f64`-typed per the
/// A5 convention. `Default` re-derives the sklearn defaults from
/// [`BayesianRidge::new`] (D-08) rather than holding literals (Pitfall 1:
/// default-drift breaks the oracle gate silently).
#[derive(Debug, Clone, Copy)]
pub struct BayesianRidgeBuilder {
    max_iter: usize,
    tol: f64,
    alpha_1: f64,
    alpha_2: f64,
    lambda_1: f64,
    lambda_2: f64,
    alpha_init: Option<f64>,
    lambda_init: Option<f64>,
    compute_score: bool,
    fit_intercept: bool,
    copy_x: bool,
    verbose: bool,
}

impl Default for BayesianRidgeBuilder {
    /// Re-derive the sklearn defaults from [`BayesianRidge::new`] (D-08). `f64`
    /// is pinned only to read the F-independent scalar defaults — the builder is
    /// non-generic, so the choice of `F` here is irrelevant.
    fn default() -> Self {
        BayesianRidge::<f64, Unfit>::new().into_builder()
    }
}

impl BayesianRidgeBuilder {
    /// Set the maximum number of evidence iterations (sklearn default `300`).
    pub fn max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }

    /// Set the `Σ|Δcoef|` stopping threshold (sklearn default `1e-3`).
    pub fn tol(mut self, v: f64) -> Self {
        self.tol = v;
        self
    }

    /// Set the shape parameter of the Gamma prior over `alpha`.
    pub fn alpha_1(mut self, v: f64) -> Self {
        self.alpha_1 = v;
        self
    }

    /// Set the rate parameter of the Gamma prior over `alpha`.
    pub fn alpha_2(mut self, v: f64) -> Self {
        self.alpha_2 = v;
        self
    }

    /// Set the shape parameter of the Gamma prior over `lambda`.
    pub fn lambda_1(mut self, v: f64) -> Self {
        self.lambda_1 = v;
        self
    }

    /// Set the rate parameter of the Gamma prior over `lambda`.
    pub fn lambda_2(mut self, v: f64) -> Self {
        self.lambda_2 = v;
        self
    }

    /// Set the initial `alpha` (`None` ⇒ sklearn's `1/(var(y) + eps)`).
    pub fn alpha_init(mut self, v: Option<f64>) -> Self {
        self.alpha_init = v;
        self
    }

    /// Set the initial `lambda` (`None` ⇒ sklearn's `1.0`).
    pub fn lambda_init(mut self, v: Option<f64>) -> Self {
        self.lambda_init = v;
        self
    }

    /// Accumulate the log marginal likelihood into `scores_` (sklearn's
    /// `compute_score`).
    pub fn compute_score(mut self, v: bool) -> Self {
        self.compute_score = v;
        self
    }

    /// Set whether to center `X`/`y` and recover a bias term.
    pub fn fit_intercept(mut self, v: bool) -> Self {
        self.fit_intercept = v;
        self
    }

    /// Set sklearn's `copy_X`. Accepted for API parity; mlrs never writes into
    /// the caller's buffer, so the value has no observable effect (module docs).
    pub fn copy_x(mut self, v: bool) -> Self {
        self.copy_x = v;
        self
    }

    /// Print the convergence iteration (sklearn's `verbose`; mlrs writes to
    /// stderr rather than stdout).
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08), against sklearn's
    /// `_parameter_constraints` one-for-one:
    ///
    /// - `max_iter >= 1` ([`BuildError::InvalidMaxIter`]) — sklearn's
    ///   `Interval(Integral, 1, None, closed="left")`.
    /// - `tol > 0` and finite — sklearn's `closed="neither"`, so unlike `Ridge`
    ///   a `tol` of exactly `0` is REJECTED here.
    /// - `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2` `>= 0` and finite —
    ///   `closed="left"`, so `0` is accepted (it drops that Gamma term).
    /// - `alpha_init`, `lambda_init` `>= 0` and finite when given —
    ///   `closed="left"`, as sklearn 1.9 spells it (`0` is admitted even though
    ///   both are precisions that go on to divide).
    ///
    /// All five out-of-interval cases surface as
    /// [`BuildError::InvalidHyperprior`], which names the offending parameter.
    pub fn build<F>(self) -> Result<BayesianRidge<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        const EST: &str = "bayesian_ridge";
        if self.max_iter == 0 {
            return Err(BuildError::InvalidMaxIter {
                estimator: EST,
                max_iter: 0,
            });
        }
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(BuildError::InvalidHyperprior {
                estimator: EST,
                param: "tol",
                value: self.tol,
                bound: "> 0",
            });
        }
        for (param, value) in [
            ("alpha_1", self.alpha_1),
            ("alpha_2", self.alpha_2),
            ("lambda_1", self.lambda_1),
            ("lambda_2", self.lambda_2),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(BuildError::InvalidHyperprior {
                    estimator: EST,
                    param,
                    value,
                    bound: ">= 0",
                });
            }
        }
        for (param, value) in [
            ("alpha_init", self.alpha_init),
            ("lambda_init", self.lambda_init),
        ] {
            if let Some(v) = value {
                if !v.is_finite() || v < 0.0 {
                    return Err(BuildError::InvalidHyperprior {
                        estimator: EST,
                        param,
                        value: v,
                        bound: ">= 0",
                    });
                }
            }
        }

        Ok(BayesianRidge {
            max_iter: self.max_iter,
            tol: self.tol,
            alpha_1: self.alpha_1,
            alpha_2: self.alpha_2,
            lambda_1: self.lambda_1,
            lambda_2: self.lambda_2,
            alpha_init: self.alpha_init,
            lambda_init: self.lambda_init,
            compute_score: self.compute_score,
            fit_intercept: self.fit_intercept,
            copy_x: self.copy_x,
            verbose: self.verbose,
            coef_: None,
            intercept_: None,
            alpha_: 0.0,
            lambda_: 0.0,
            sigma_: None,
            sigma_sqrt_t_: None,
            scores_: Vec::new(),
            n_iter_: 0,
            x_offset_: Vec::new(),
            x_scale_: Vec::new(),
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }
}

impl<F> BayesianRidge<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `coef_` (length `n_features`). `Some` by
    /// construction on the `Fitted` state (D-03).
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.coef_
            .as_ref()
            .expect("coef_ is Some by construction on BayesianRidge<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `intercept_` (scalar). `Some` by construction on
    /// the `Fitted` state (D-03).
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> F {
        self.intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on BayesianRidge<F, Fitted>")
            .to_host(pool)[0]
    }

    /// sklearn's `alpha_`: the estimated precision of the noise.
    pub fn alpha(&self) -> f64 {
        self.alpha_
    }

    /// sklearn's `lambda_`: the estimated precision of the weights.
    pub fn lambda(&self) -> f64 {
        self.lambda_
    }

    /// sklearn's `sigma_`: the row-major `d × d` posterior covariance of the
    /// weights.
    pub fn sigma(&self) -> &[f64] {
        self.sigma_
            .as_ref()
            .expect("sigma_ is Some by construction on BayesianRidge<F, Fitted>")
    }

    /// sklearn's `scores_`: the log marginal likelihood at each iteration plus
    /// one final value. Empty unless the estimator was built with
    /// `compute_score`.
    pub fn scores(&self) -> &[f64] {
        &self.scores_
    }

    /// sklearn's `n_iter_`: evidence iterations actually run.
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// sklearn's `X_offset_`: the column means removed before the fit (zeros
    /// when `!fit_intercept`).
    pub fn x_offset(&self) -> &[f64] {
        &self.x_offset_
    }

    /// sklearn's `X_scale_`: all ones (the attribute outlived `normalize`).
    pub fn x_scale(&self) -> &[f64] {
        &self.x_scale_
    }

    /// `predict` for a test matrix that is still on the HOST — returns the
    /// length-`n_samples` predictions plus the operand-finiteness verdict.
    ///
    /// The host-ingress twin of [`Predict::predict`], shared verbatim with the
    /// four other dense linear regressors — see [`predict_linear_from_host`] for
    /// the backend routing and the finiteness verdict's meaning.
    pub fn predict_from_host(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<HostPrediction<F>, AlgoError> {
        predict_linear_from_host(
            self.coef_.as_ref(),
            self.intercept_.as_ref(),
            &self.predict_mirror,
            "bayesian_ridge",
            pool,
            x,
            shape,
        )
    }

    /// sklearn's `predict(X, return_std=True)` second return value — the
    /// per-sample predictive standard deviation
    ///
    /// ```text
    /// x̃ᵢ     = xᵢ − X_offset_
    /// std[i] = √( x̃ᵢ·Σ·x̃ᵢᵀ + 1/α )
    /// ```
    ///
    /// over a HOST design, where `Σ` is [`sigma`](BayesianRidge::sigma) and
    /// `1/α` the estimated noise variance.
    ///
    /// Note the CENTERING, which sklearn applies here (`X = X - self.X_offset_`)
    /// and nowhere else in `predict` — the mean goes through the raw design,
    /// because `intercept_` already absorbs the offset, while `sigma_` is the
    /// posterior covariance of the CENTERED problem's weights and so has to be
    /// evaluated in that frame. With `fit_intercept = false` the offset is zeros
    /// and the subtraction is a no-op, exactly as in sklearn.
    ///
    /// `predict` itself is a separate call because sklearn returns the mean
    /// whether or not `return_std` is set, so the common path must not pay for
    /// this quadratic form.
    ///
    /// ## Where the work runs
    /// The quadratic form is `O(n·d²)` — a factor `d` more arithmetic than the
    /// mean — while the operand is the same `n·d`. That ratio is why this is the
    /// dense-linear predict path with the most to gain from the device, and
    /// [`bayes_predict_std_from_host`] routes it accordingly: the cpu backend
    /// reads the caller's buffer in place across
    /// [`cpu_launch_units`](mlrs_backend::capability::cpu_launch_units) scoped
    /// threads, every device backend uploads once and runs the fused
    /// [`bayes_predict_std`] kernel. Both arms evaluate the SAME sum-of-squares
    /// form through the covariance factor (see [`posterior_sigma_sqrt_t`]), so
    /// the routing cannot change the answer.
    pub fn predict_std_from_host(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<Vec<f64>, AlgoError> {
        let (n_samples, n_features) = shape;
        let mt = self.check_std_state(n_features)?;
        Ok(bayes_predict_std_from_host::<F>(
            pool,
            x,
            &self.x_offset_,
            mt,
            1.0 / self.alpha_,
            (n_samples, n_features),
        )?)
    }

    /// [`predict_std_from_host`](BayesianRidge::predict_std_from_host) for a
    /// test matrix that is already DEVICE-resident — the `return_std` twin of
    /// [`Predict::predict`], and like it the result stays on the device (D-05).
    ///
    /// The fitted `X_offset_` and the covariance factor are host `f64` (they are
    /// read at the Python boundary and are `O(d²)`, not `O(n·d)`), so they are
    /// narrowed to `F` and uploaded per call. That is `d² + d` elements against
    /// the `n·d²` multiply-adds the launch then does — the same order the
    /// launch's own fixed cost sits at, and it keeps `predict` free of fit-time
    /// device state that would have to be kept alive and pool-managed for the
    /// estimator's whole lifetime.
    pub fn predict_std(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_samples, n_features) = shape;
        let mt = self.check_std_state(n_features)?;
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }

        let off: Vec<F> = self.x_offset_.iter().map(|v| f64_to_host::<F>(*v)).collect();
        let mt_f: Vec<F> = mt.iter().map(|v| f64_to_host::<F>(*v)).collect();
        let off_dev = DeviceArray::from_host(pool, &off);
        let mt_dev = DeviceArray::from_host(pool, &mt_f);
        let out = bayes_predict_std::<F>(
            pool,
            x,
            &off_dev,
            &mt_dev,
            (n_samples, n_features),
            f64_to_host::<F>(1.0 / self.alpha_),
        );
        off_dev.release_into(pool);
        mt_dev.release_into(pool);
        Ok(out?)
    }

    /// The fitted state both `predict_std` ingresses need, checked against the
    /// test design's feature count exactly once: returns the covariance factor
    /// `Mᵀ` ([`posterior_sigma_sqrt_t`]) on success.
    ///
    /// `X_offset_` is checked alongside it because the two are consumed
    /// together and a mismatch in either is the same caller error — a test
    /// matrix whose feature count disagrees with the fit.
    fn check_std_state(&self, n_features: usize) -> Result<&[f64], AlgoError> {
        let mt = self
            .sigma_sqrt_t_
            .as_ref()
            .expect("sigma_sqrt_t_ is Some by construction on BayesianRidge<F, Fitted>");
        if mt.len() != n_features * n_features {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: (mt.len() as f64).sqrt() as usize,
                rhs: n_features,
            }));
        }
        if self.x_offset_.len() != n_features {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: self.x_offset_.len(),
                rhs: n_features,
            }));
        }
        Ok(mt)
    }
}

impl<F> Fit<F> for BayesianRidge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = BayesianRidge<F, Fitted>;

    /// Unweighted `fit` — [`BayesianRidge::fit_with_sample_weight`] with no
    /// weights.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<BayesianRidge<F, Fitted>, AlgoError> {
        self.fit_with_sample_weight(pool, x, y, shape, None)
    }
}

impl<F> BayesianRidge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// `fit` with sklearn's `sample_weight`, over a DEVICE-resident design.
    ///
    /// Forms the normal equations ON THE DEVICE where the backend can
    /// ([`device_gram_applicable`]) and reads back only the `d² + d` scalars the
    /// rest of the fit consumes; otherwise reads the design back and delegates
    /// to [`BayesianRidge::fit_from_host_slice`]. Either way the reduction runs
    /// in `f64`, which is the invariant this estimator cannot trade away.
    ///
    /// ## Why the Gram must be `f64` even when the design is not
    /// `gram_xty` accumulates in the estimator's own float width `F`. For
    /// `Ridge` that is fine: its Gram feeds a Cholesky solve, which is
    /// backward-stable, so an `f32` Gram gives an `f32`-accurate `coef_`.
    ///
    /// `BayesianRidge` consumes the Gram through the residual identity
    /// `sse = yᵀy − 2wᵀXᵀy + wᵀGw` ([`update_coef`]), whose absolute error is
    /// `~ε·yᵀy` no matter how small `sse` is — so the RELATIVE error in `sse` is
    /// amplified by `yᵀy/sse`, the reciprocal of the fraction of variance the
    /// model leaves unexplained. `sse` then feeds `alpha_`, `alpha_` feeds the
    /// next iteration's shrinkage, and the error compounds over the loop.
    ///
    /// Measured on the `6 × 10` wide fixture with `fit_intercept=False`, where
    /// the model explains all but `~3e-4` of the target variance: an `f64` Gram
    /// reproduced sklearn's `coef_`, `alpha_` and `n_iter_` EXACTLY, while an
    /// `f32` one returned `alpha_ = 762` against sklearn's `254` and stopped two
    /// iterations late. That is a 3× error in a fitted attribute from a Gram
    /// that was itself accurate to `1e-7` — the amplification, not the Gram.
    ///
    /// That measurement is why the device arm was originally refused outright,
    /// and it is not repealed here — it is SATISFIED. `prims::normal_eq` widens
    /// a narrower design on the device
    /// ([`widen_elem`](mlrs_kernels::elementwise::widen_elem)) and runs the
    /// whole assembly at `f64`, so the two arms differ only in summation order.
    /// The refusal now falls where it belongs: on an adapter that cannot do
    /// `f64` at all, where [`device_gram_applicable`] returns `false` and this
    /// path reads the design back exactly as it used to.
    ///
    /// `sample_weight` is a length-`n_samples` host slice of NON-NEGATIVE finite
    /// weights. It changes the fit in two places, both sklearn-faithful: the
    /// column means become weighted (`x̄ = Σwᵢxᵢ / Σwᵢ`), and the rows are
    /// rescaled by `√wᵢ` — sklearn's `_rescale_data`, which `BayesianRidge`
    /// applies unconditionally (it has no SAG-family exception).
    pub fn fit_with_sample_weight(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<BayesianRidge<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- ASVS V5: data-DEPENDENT geometry guard BEFORE any read-back (the
        //     data-INDEPENDENT hyperparameter checks ran at build(), D-08). ---
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "bayesian_ridge",
            operation: "fit (requires y)",
        })?;
        if y.len() != n_samples {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_samples,
                cols: 1,
                len: y.len(),
            }));
        }

        // --- The DEVICE arm: form the normal equations where the design already
        //     is, and read back only the `d² + d` scalars the rest of the fit
        //     consumes. `device_gram_applicable` is a capability gate as much as
        //     a perf one — it refuses any backend that cannot accumulate in
        //     `f64`, which is the property this estimator's Gram cannot do
        //     without (see this function's docs). ---
        if !device_gram_applicable::<F>(n_features) {
            let x_host = x.to_host(pool);
            let y_host = y.to_host(pool);
            return self.fit_host_core(pool, &x_host, &y_host, shape, sample_weight);
        }

        let sw64 = validate_sample_weight::<F>("bayesian_ridge", sample_weight, n_samples)?;
        let profile = std::env::var("BAYES_PROFILE").is_ok();
        let lap0 = std::time::Instant::now();

        let (x_mean, y_mean, gram, xty) = centered_gram_xty_device::<F>(
            pool,
            x,
            y,
            n_samples,
            n_features,
            sw64.as_deref(),
            self.fit_intercept,
        )?;

        // The two target scalars stay on the host. They are `O(n)` over the
        // length-`n` TARGET — `1/d` of the design that just stayed on the
        // device — so the read-back this costs is a rounding error against the
        // `O(n·d)` one the arm exists to remove, and doing it here keeps ONE
        // implementation of sklearn's two subtly different `y` moments
        // ([`y_moments`]) rather than a device twin that could drift from it.
        let y_host = y.to_host(pool);
        let (y_var, yty) = y_moments::<F>(&y_host, n_samples, y_mean, sw64.as_deref());
        let sw_sum = sw64.as_deref().map_or(n_samples as f64, |w| w.iter().sum());
        let t_gram = if profile {
            lap0.elapsed().as_secs_f64()
        } else {
            0.0
        };

        self.finish_fit(
            pool, gram, xty, yty, y_var, sw_sum, x_mean, y_mean, n_samples, n_features, profile,
            t_gram,
        )
    }
}

impl<F> Predict<F> for BayesianRidge<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_samples, n_features) = shape;

        let coef = self
            .coef_
            .as_ref()
            .expect("coef_ is Some by construction on BayesianRidge<F, Fitted>");
        let intercept = self
            .intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on BayesianRidge<F, Fitted>");

        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        if coef.len() != n_features {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: coef.len(),
                rhs: n_features,
            }));
        }

        // y_pred = X_test · coef + intercept via ONE fused device launch — the
        // same `linear_predict` kernel every dense linear regressor here shares.
        Ok(linear_predict::<F>(
            pool,
            x,
            coef,
            intercept,
            (n_samples, n_features),
        )?)
    }
}
