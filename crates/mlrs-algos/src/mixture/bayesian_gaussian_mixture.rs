//! `BayesianGaussianMixture` (MIX-02) — variational Bayes for a Gaussian
//! mixture.
//!
//! The full `sklearn.mixture.BayesianGaussianMixture` surface: all seventeen
//! hyperparameters, every fitted attribute (including the six `*_prior_`
//! attributes and the four variational posteriors), and the `fit` /
//! `fit_predict` / `predict` / `predict_proba` / `predict_log_proba` / `score` /
//! `score_samples` / `sample` method set.
//!
//! ## What makes it Bayesian, in one paragraph
//! [`GaussianMixture`](super::gaussian_mixture::GaussianMixture) maximizes the
//! likelihood, so every component keeps a point estimate and `k` is a hard
//! choice the caller must get right. This estimator puts a conjugate prior on
//! each block — a Dirichlet (or a stick-breaking Dirichlet PROCESS) on the
//! mixing weights, and a Normal-Wishart on each component's mean/precision —
//! and maximizes an evidence lower bound over the posterior instead. Two
//! consequences drive the whole file:
//!
//! - The E-step no longer plugs in `ln π_c` and `ln|Λ_c|`; it uses their
//!   EXPECTATIONS under the posterior, which is where `ψ` (digamma) enters.
//!   Those expectations are per-COMPONENT constants, which is the fact that
//!   lets this estimator share
//!   [`GmmHost`](mlrs_backend::prims::gmm_host::GmmHost)'s entire inner nest —
//!   see [`e_step_biased`](mlrs_backend::prims::gmm_host::GmmHost::e_step_biased).
//! - `n_components` becomes an UPPER BOUND, not a count. With
//!   `weight_concentration_prior_type='dirichlet_process'` and a small
//!   concentration, unneeded components are driven to near-zero weight instead
//!   of splitting real clusters. That behaviour is the estimator's whole reason
//!   to exist, and it is why `weights_` here does not track `nk / n` the way
//!   the plain model's does.
//!
//! ## Where the compute lives
//! Exactly where `GaussianMixture`'s does:
//! [`mlrs_backend::prims::gmm_host`], host-resident on every backend for the
//! three structural reasons that module documents, and carrying the same three
//! wins over sklearn (triangular Mahalanobis, a hoisted `tied` E-step, one
//! fused sweep where sklearn makes five). The M-step's `nk` / `xk` / `sk` are
//! literally sklearn's `_estimate_gaussian_parameters` outputs, so what is
//! ADDED here over the plain model is `O(k·d²)` per iteration of conjugate
//! updates against an `O(n·k·d²)` sweep — the Bayesian machinery is nearly
//! free at any `n` a user runs.
//!
//! Like [`GaussianMixture`](super::gaussian_mixture::GaussianMixture), this
//! estimator ALSO has a device EM engine
//! ([`GmmDevice`](mlrs_backend::prims::gmm_device::GmmDevice)) it can take for
//! large fits on cuda/rocm. It reuses the identical mechanism: the
//! variational E-step's `_estimate_log_prob` differs from the plain model's
//! only by a per-component additive constant
//! ([`BayesianGaussianMixture::log_weight_term`]), so
//! [`GmmDevice::e_step_biased`](mlrs_backend::prims::gmm_device::GmmDevice::e_step_biased)
//! runs the SAME `O(n·k·d²)` Mahalanobis/normalize/reduce kernels
//! [`GmmDevice::e_step`](mlrs_backend::prims::gmm_device::GmmDevice::e_step)
//! does for the plain model, plus one small additional kernel
//! ([`mlrs_kernels::gmm::gmm_entropy_rows`]) for the `Σ r·ln r` term this
//! estimator's lower bound needs and plain EM does not. The variational
//! M-step ([`BayesianGaussianMixture::m_step`]) stays host-resident regardless
//! of engine — it is `O(k·d²)`, not `O(n·k·d²)`, same as
//! [`GaussianMixture`](super::gaussian_mixture::GaussianMixture)'s
//! `precisions_cholesky` tail.
//!
//! ## sklearn parity notes (the traps)
//! - **`lower_bound_` is NOT a log-likelihood.** It is the evidence lower bound
//!   with every constant term dropped (sklearn's `_compute_lower_bound`
//!   comment), so it is not comparable to `GaussianMixture`'s `lower_bound_`,
//!   and it is not even guaranteed negative. What it IS is monotone in the
//!   variational objective, which is all the convergence test needs.
//! - **The bound is computed AFTER the M-step, from the PRE-M-step
//!   responsibilities.** sklearn's loop is `e_step; m_step;
//!   compute_lower_bound(log_resp, …)` — the entropy term comes from the old
//!   responsibilities while every other term reads the new parameters. Tidying
//!   that would silently change the convergence point.
//! - **`degrees_of_freedom_` is a SCALAR under `covariance_type='tied'`** and a
//!   `k`-vector otherwise, because all components then share one Wishart. It is
//!   stored here as a length-1 `Vec` in that case and
//!   [`BayesianGaussianMixture::degrees_of_freedom_shape`] reports which.
//! - **`weight_concentration_` is a PAIR under `dirichlet_process`** (the two
//!   Beta parameters of each stick break) and a single `k`-vector under
//!   `dirichlet_distribution`.
//! - **Under `dirichlet_process` the component ORDER is part of the model.**
//!   Stick-breaking is not exchangeable: component `c`'s second Beta parameter
//!   sums the `nk` of every component AFTER it, so permuting components changes
//!   `weights_` (by `O(prior/n)`) and `lower_bound_`. Two runs that find the
//!   same clustering in a different order therefore agree on `means_` /
//!   `covariances_` / `degrees_of_freedom_` but NOT on the weights — which is
//!   why the oracle compares those two families differently.
//! - **There is no `bic` / `aic`.** sklearn defines them on `GaussianMixture`
//!   only; the variational model's free-parameter count is not well defined.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use std::marker::PhantomData;
use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device::Device;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gmm_device::{
    gmm_device_applicable, gmm_device_possible, GmmDevice,
};
use mlrs_backend::prims::gmm_host::{
    cholesky_lower_blocks, log_det_cholesky, logsumexp_rows, precisions_cholesky,
    precisions_from_cholesky, weighted_log_prob_biased, CovarianceType, GmmHost,
};
use mlrs_backend::prims::rng::SplitMix64;
use mlrs_backend::prims::special::{betaln, digamma, lgamma};
use mlrs_backend::runtime::ActiveRuntime;

use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::mixture::mixture_persist::{
    as_f64, as_i64, expect_len, read_device_arm, read_opt_vec, shape_1d, write_opt_vec,
    AlignedBytes, LoadModel, MixtureFile, MixtureWriter, PersistError, SaveModel, TensorRef,
    LOWER_BOUNDS_NAME, TRAIN_LABELS_NAME,
};

use crate::error::{AlgoError, BuildError};
use crate::mixture::gaussian_mixture::{
    argmax_rows, ill_conditioned, initial_responsibilities, standard_normal, InitParams,
};
use crate::typestate::{
    validate_geometry, Fit, Fitted, PredictLabels, PredictLogProba, PredictProba, ScoreSamples,
    State, Unfit,
};

/// The estimator name carried by every typed error this file raises.
const EST: &str = "bayesian_gaussian_mixture";

/// Seed used when the caller leaves `random_state = None` — the same fixed
/// default [`GaussianMixture`](super::gaussian_mixture::GaussianMixture) uses,
/// and for the same reason (a Rust-side entropy source would make `fit`
/// irreproducible; RESEARCH Pitfall 7).
const DEFAULT_SEED: u64 = 0x5EED_6D31;

// ---------------------------------------------------------------------------
// The prior family
// ---------------------------------------------------------------------------

/// sklearn's `weight_concentration_prior_type`: which prior sits over the
/// mixing weights.
///
/// This is the hyperparameter that decides whether the model can PRUNE
/// components. Both families are conjugate and cost the same per iteration;
/// they differ in what they believe about unused components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightConcentrationPriorType {
    /// `'dirichlet_process'` (sklearn's default) — the stick-breaking
    /// construction. Each component `c` gets a `Beta(1 + nk_c, γ + Σ_{j>c}
    /// nk_j)` posterior over its share of the remaining stick, so mass flows
    /// toward the low indices and a component that explains nothing keeps
    /// almost none. This is what turns `n_components` into an upper bound.
    DirichletProcess,
    /// `'dirichlet_distribution'` — the symmetric `Dir(γ + nk)` posterior. Every
    /// component is exchangeable, so the fit is invariant to their order, but
    /// nothing pushes an unused component's weight to zero: it settles at
    /// `γ / (kγ + n)` instead.
    DirichletDistribution,
}

impl WeightConcentrationPriorType {
    /// The sklearn string spelling.
    pub fn name(self) -> &'static str {
        match self {
            WeightConcentrationPriorType::DirichletProcess => "dirichlet_process",
            WeightConcentrationPriorType::DirichletDistribution => "dirichlet_distribution",
        }
    }
}

impl TryFrom<&str> for WeightConcentrationPriorType {
    type Error = BuildError;

    fn try_from(value: &str) -> Result<Self, BuildError> {
        match value {
            "dirichlet_process" => Ok(WeightConcentrationPriorType::DirichletProcess),
            "dirichlet_distribution" => Ok(WeightConcentrationPriorType::DirichletDistribution),
            other => Err(BuildError::UnknownWeightConcentrationPriorType {
                value: other.to_string(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Priors and posteriors
// ---------------------------------------------------------------------------

/// The RESOLVED priors — sklearn's `weight_concentration_prior_`,
/// `mean_precision_prior_`, `mean_prior_`, `degrees_of_freedom_prior_` and
/// `covariance_prior_` fitted attributes.
///
/// "Resolved" because four of the five have data-DEPENDENT defaults (the mean
/// prior is the design's mean, the covariance prior its empirical covariance),
/// so they cannot exist until `fit` sees `X`. sklearn exposes them as fitted
/// attributes for exactly that reason, and so does this file.
#[derive(Debug, Clone, PartialEq)]
pub struct MixturePriors {
    /// `weight_concentration_prior_` (γ) — defaults to `1 / n_components`.
    pub weight_concentration: f64,
    /// `mean_precision_prior_` (β₀) — defaults to `1`.
    pub mean_precision: f64,
    /// `mean_prior_` (m₀), length `n_features` — defaults to the column means.
    pub mean: Vec<f64>,
    /// `degrees_of_freedom_prior_` (ν₀) — defaults to `n_features`, and must
    /// exceed `n_features − 1` for the Wishart to be proper.
    pub degrees_of_freedom: f64,
    /// `covariance_prior_` (W₀⁻¹), flat in the `covariance_type` layout for ONE
    /// component (`d × d` for `full`/`tied`, `d` for `diag`, a scalar for
    /// `spherical`). Defaults to the design's empirical covariance/variance at
    /// `ddof = 1` — numpy's `np.cov` / `np.var(ddof=1)` default, NOT the MLE
    /// the E-step's own covariances use.
    pub covariance: Vec<f64>,
}

/// One variational posterior: what sklearn's `_get_parameters` /
/// `_set_parameters` move between `n_init` restarts.
///
/// Always `f64` regardless of the estimator's `F`, like
/// [`MixtureParams`](super::gaussian_mixture::MixtureParams) — the whole loop
/// is (`gmm_host` module docs); `F` appears only at the accessors.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianMixtureParams {
    /// First Beta parameter per component (`dirichlet_process`), or the whole
    /// Dirichlet concentration vector (`dirichlet_distribution`). Length `k`.
    pub weight_concentration_a: Vec<f64>,
    /// Second Beta parameter per component under `dirichlet_process`; EMPTY
    /// under `dirichlet_distribution`, which has no second parameter.
    pub weight_concentration_b: Vec<f64>,
    /// `mean_precision_` (β), length `k`.
    pub mean_precision: Vec<f64>,
    /// `means_` (m), `k × d` row-major.
    pub means: Vec<f64>,
    /// `degrees_of_freedom_` (ν) — length `k`, or length 1 under
    /// `covariance_type='tied'` where one Wishart is shared.
    pub degrees_of_freedom: Vec<f64>,
    /// `covariances_`, in the `covariance_type` layout. Note this is the
    /// posterior EXPECTED covariance (`W⁻¹/ν`), which is what sklearn stores.
    pub covariances: Vec<f64>,
    /// `precisions_cholesky_`, same layout.
    pub precisions_cholesky: Vec<f64>,
}

impl BayesianMixtureParams {
    /// `degrees_of_freedom_` for component `c`, collapsing the `tied` case
    /// where one scalar stands for all of them.
    #[inline]
    fn dof(&self, c: usize) -> f64 {
        self.degrees_of_freedom[c.min(self.degrees_of_freedom.len() - 1)]
    }
}

// ---------------------------------------------------------------------------
// The estimator
// ---------------------------------------------------------------------------

/// Variational-Bayes fit of a Gaussian mixture (MIX-02).
///
/// `S` is the [`State`] typestate marker: an `Unfit` value exposes only the fit
/// entry points, and every fitted attribute / scoring method lives on the
/// `Fitted` sibling, so a `predict`-before-`fit` is a compile error.
pub struct BayesianGaussianMixture<F, S = Unfit>
where
    F: Float + CubeElement + Pod,
    S: State,
{
    // ---- hyperparameters (sklearn ctor order) ----
    n_components: usize,
    covariance_type: CovarianceType,
    tol: f64,
    reg_covar: f64,
    max_iter: usize,
    n_init: usize,
    init_params: InitParams,
    weight_concentration_prior_type: WeightConcentrationPriorType,
    weight_concentration_prior: Option<f64>,
    mean_precision_prior: Option<f64>,
    mean_prior: Option<Vec<f64>>,
    degrees_of_freedom_prior: Option<f64>,
    covariance_prior: Option<Vec<f64>>,
    random_state: Option<u64>,
    warm_start: bool,
    verbose: usize,
    verbose_interval: usize,

    /// Posterior carried in from a previous `fit` when `warm_start = true` (the
    /// consuming typestate `fit` cannot keep it on the estimator the way
    /// sklearn does — see [`BayesianGaussianMixture::into_warm_start`]).
    warm: Option<BayesianMixtureParams>,
    /// Where to run the EM loop (DEVICE-PARAM-01). `Auto` keeps the
    /// `gmm_device_applicable` gate — backend, `f64` capability, `f64`
    /// transcendentals, the `MLRS_GMM_DEVICE` flag, then a size floor.
    device: Device,

    // ---- fitted state (`None` / zero while `Unfit`) ----
    params: Option<BayesianMixtureParams>,
    /// The EM engine that ACTUALLY ran (`"cpu"` / `"gpu"`), `None` until `fit`.
    device_: Option<&'static str>,
    priors: Option<MixturePriors>,
    converged_: bool,
    n_iter_: usize,
    lower_bound_: f64,
    lower_bounds_: Vec<f64>,
    n_features_in_: usize,
    n_samples_: usize,
    train_labels: Option<Vec<i32>>,

    _float: PhantomData<F>,
    _state: PhantomData<S>,
}

impl<F> Default for BayesianGaussianMixture<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Hand-written for the same reason
/// [`GaussianMixture`](super::gaussian_mixture::GaussianMixture)'s is: a derive
/// would demand `F: Debug` / `S: Debug` and would dump the fitted buffers into
/// any `{:?}`.
impl<F, S> std::fmt::Debug for BayesianGaussianMixture<F, S>
where
    F: Float + CubeElement + Pod,
    S: State,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BayesianGaussianMixture")
            .field("n_components", &self.n_components)
            .field("covariance_type", &self.covariance_type.name())
            .field("tol", &self.tol)
            .field("reg_covar", &self.reg_covar)
            .field("max_iter", &self.max_iter)
            .field("n_init", &self.n_init)
            .field("init_params", &self.init_params.name())
            .field(
                "weight_concentration_prior_type",
                &self.weight_concentration_prior_type.name(),
            )
            .field(
                "weight_concentration_prior",
                &self.weight_concentration_prior,
            )
            .field("mean_precision_prior", &self.mean_precision_prior)
            .field("degrees_of_freedom_prior", &self.degrees_of_freedom_prior)
            .field("random_state", &self.random_state)
            .field("warm_start", &self.warm_start)
            .field("fitted", &self.params.is_some())
            .field("n_iter_", &self.n_iter_)
            .field("converged_", &self.converged_)
            .field("lower_bound_", &self.lower_bound_)
            .finish()
    }
}

impl<F> BayesianGaussianMixture<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// A new estimator at sklearn's defaults (`n_components=1`,
    /// `covariance_type='full'`, `tol=1e-3`, `reg_covar=1e-6`, `max_iter=100`,
    /// `n_init=1`, `init_params='kmeans'`,
    /// `weight_concentration_prior_type='dirichlet_process'`, every prior
    /// `None` — i.e. derived from the data — `warm_start=False`, `verbose=0`,
    /// `verbose_interval=10`).
    ///
    /// SINGLE source of truth for the defaults:
    /// [`BayesianGaussianMixtureBuilder`]'s `Default` re-derives them through
    /// [`BayesianGaussianMixture::into_builder`].
    pub fn new() -> Self {
        Self {
            n_components: 1,
            covariance_type: CovarianceType::Full,
            tol: 1e-3,
            reg_covar: 1e-6,
            max_iter: 100,
            n_init: 1,
            init_params: InitParams::KMeans,
            weight_concentration_prior_type: WeightConcentrationPriorType::DirichletProcess,
            weight_concentration_prior: None,
            mean_precision_prior: None,
            mean_prior: None,
            degrees_of_freedom_prior: None,
            covariance_prior: None,
            random_state: None,
            warm_start: false,
            verbose: 0,
            verbose_interval: 10,
            warm: None,
            device: Device::Auto,
            params: None,
            device_: None,
            priors: None,
            converged_: false,
            n_iter_: 0,
            lower_bound_: f64::NEG_INFINITY,
            lower_bounds_: Vec::new(),
            n_features_in_: 0,
            n_samples_: 0,
            train_labels: None,
            _float: PhantomData,
            _state: PhantomData,
        }
    }

    /// Start from sklearn's defaults with the builder.
    pub fn builder() -> BayesianGaussianMixtureBuilder {
        BayesianGaussianMixtureBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter (BLDR-01, single source of the defaults).
    pub fn into_builder(self) -> BayesianGaussianMixtureBuilder {
        BayesianGaussianMixtureBuilder {
            device: self.device,
            n_components: self.n_components,
            covariance_type: self.covariance_type.name().to_string(),
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            n_init: self.n_init,
            init_params: self.init_params.name().to_string(),
            weight_concentration_prior_type: self
                .weight_concentration_prior_type
                .name()
                .to_string(),
            weight_concentration_prior: self.weight_concentration_prior,
            mean_precision_prior: self.mean_precision_prior,
            mean_prior: self.mean_prior,
            degrees_of_freedom_prior: self.degrees_of_freedom_prior,
            covariance_prior: self.covariance_prior,
            random_state: self.random_state,
            warm_start: self.warm_start,
            verbose: self.verbose,
            verbose_interval: self.verbose_interval,
            warm_params: None,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the
    /// defaults-equality gate, BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_components == other.n_components
            && self.covariance_type == other.covariance_type
            && self.tol == other.tol
            && self.reg_covar == other.reg_covar
            && self.max_iter == other.max_iter
            && self.n_init == other.n_init
            && self.init_params == other.init_params
            && self.weight_concentration_prior_type == other.weight_concentration_prior_type
            && self.weight_concentration_prior == other.weight_concentration_prior
            && self.mean_precision_prior == other.mean_precision_prior
            && self.mean_prior == other.mean_prior
            && self.degrees_of_freedom_prior == other.degrees_of_freedom_prior
            && self.covariance_prior == other.covariance_prior
            && self.random_state == other.random_state
            && self.warm_start == other.warm_start
            && self.verbose == other.verbose
            && self.verbose_interval == other.verbose_interval
            && self.device == other.device
    }

    /// Should `fit` take [`BayesianGaussianMixture::fit_from_host_slice`]
    /// rather than uploading and going through [`Fit::fit`]?
    ///
    /// ALWAYS `true` — mirrors
    /// [`GaussianMixture::host_fit_applicable`](super::gaussian_mixture::GaussianMixture::host_fit_applicable)
    /// exactly, for the same reason: `fit_from_host_slice` is a strict
    /// superset of what `Fit::fit` can reach. It never uploads `x` itself when
    /// the variational loop stays host-resident, or is uploaded exactly once
    /// by [`GmmDevice::new`](mlrs_backend::prims::gmm_device::GmmDevice::new)
    /// when [`BayesianGaussianMixture::device_fit_applicable`] takes the
    /// device EM engine — whereas `Fit::fit` always pays one upload up front.
    /// The predicate exists anyway because the two entry points take
    /// DIFFERENT operand types and a caller has to choose before ingress.
    pub fn host_fit_applicable(&self, _shape: (usize, usize)) -> bool {
        true
    }

    /// Does the DEVICE EM engine
    /// ([`GmmDevice`](mlrs_backend::prims::gmm_device::GmmDevice)) apply to
    /// this `(n_samples, n_features)` shape, given this estimator's
    /// `n_components`? Delegates entirely to
    /// [`gmm_device_applicable`](mlrs_backend::prims::gmm_device::gmm_device_applicable) —
    /// see [`GaussianMixture::device_fit_applicable`](super::gaussian_mixture::GaussianMixture::device_fit_applicable)'s
    /// docs, which this mirrors exactly. `fit_core` consults this ONCE per
    /// fit, before the `n_init` restart loop.
    pub fn device_fit_applicable(&self, shape: (usize, usize)) -> bool {
        // Phrased as "should the DEVICE arm run", so this takes
        // `prefers_device` rather than the negation of `prefers_host`: the host
        // EM engine is always available, and the only question is whether the
        // kernels are worth it.
        gmm_device_possible()
            && self
                .device
                .prefers_device(|| gmm_device_applicable(shape.0, shape.1, self.n_components))
    }

    /// `fit` over a HOST slice — the no-upload-by-default, Python-boundary
    /// ingress.
    ///
    /// `x` is the `n × d` row-major design borrowed from host memory. `pool`
    /// is only touched when
    /// [`BayesianGaussianMixture::device_fit_applicable`] holds for this
    /// shape — the common case (small/medium fits, or any cpu/wgpu-at-f64
    /// backend) never touches it and never uploads `x`.
    pub fn fit_from_host_slice(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<BayesianGaussianMixture<F, Fitted>, AlgoError> {
        let (n, d) = shape;
        if n == 0 || d == 0 || x.len() != n * d {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n,
                cols: d,
                len: x.len(),
            }));
        }
        let x64: Vec<f64> = x.iter().map(|&v| host_to_f64(v)).collect();
        self.fit_core(pool, &x64, shape)
    }

    /// The shared body of both ingresses (the design already widened to `f64`).
    ///
    /// Mirrors `BaseMixture.fit_predict` structurally — `n_init` restarts, each
    /// an initialization plus up to `max_iter` E/M iterations, keeping the
    /// restart with the highest `lower_bound_` — with the E and M steps
    /// replaced by their variational counterparts.
    fn fit_core(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[f64],
        shape: (usize, usize),
    ) -> Result<BayesianGaussianMixture<F, Fitted>, AlgoError> {
        let (n, d) = shape;
        let k = self.n_components;
        // sklearn: "Expected n_samples >= n_components". Data-DEPENDENT, so it
        // belongs at `fit`, not at `build()` (the D-08 split).
        if k > n {
            return Err(AlgoError::InvalidK {
                estimator: EST,
                k,
                n_samples: n,
            });
        }
        let priors = self.resolve_priors(x, n, d)?;

        let ct = self.covariance_type;
        let mut host = GmmHost::new(x, n, d, k, ct, self.reg_covar);
        let mut rng = SplitMix64::new(self.random_state.unwrap_or(DEFAULT_SEED));

        // The device EM engine, built ONCE (not per restart) when applicable —
        // mirrors `GaussianMixture::fit_core` exactly (module docs).
        // Recorded from the gate that actually built the engine, so `device_`
        // cannot describe an arm the fit did not take.
        let device_arm = if self.device_fit_applicable(shape) {
            "gpu"
        } else {
            "cpu"
        };
        let mut device: Option<GmmDevice> = if self.device_fit_applicable(shape) {
            Some(GmmDevice::new(pool, x, n, d, k, ct, self.reg_covar).map_err(AlgoError::Prim)?)
        } else {
            None
        };

        // sklearn's `do_init = not (warm_start and converged_)`: a warm-started
        // refit resumes from the carried posterior instead of re-initializing.
        let do_init = !(self.warm_start && self.warm.is_some());
        let n_init = if do_init { self.n_init } else { 1 };

        let mut best: Option<BayesianMixtureParams> = None;
        let mut best_bound = f64::NEG_INFINITY;
        let mut best_n_iter = 0usize;
        let mut best_converged = false;
        let mut best_trace: Vec<f64> = Vec::new();

        for _restart in 0..n_init {
            let (mut cur, mut lower_bound) = if do_init {
                (
                    self.initialize(&mut host, &mut rng, &priors, n, d, k)?,
                    f64::NEG_INFINITY,
                )
            } else {
                (
                    self.warm
                        .clone()
                        .expect("warm_start branch requires carried params"),
                    self.lower_bound_,
                )
            };

            let mut converged = false;
            let mut iters = 0usize;
            let mut trace: Vec<f64> = Vec::with_capacity(self.max_iter);
            for it in 1..=self.max_iter {
                let prev = lower_bound;
                // The E-step: ONE fused sweep producing the responsibilities,
                // `nk`, `xk` and the entropy, driven by the per-component
                // expected-log terms this model contributes — on whichever
                // engine this fit is using (mirrors `GaussianMixture`).
                let lw = self.log_weight_term(&cur, d);
                let (nk, means, resp_log_resp) = if let Some(dev) = device.as_mut() {
                    let (_mean_lpn, nk, means, resp_log_resp) = dev
                        .e_step_biased(pool, &lw, &cur.means, &cur.precisions_cholesky)
                        .map_err(AlgoError::Prim)?;
                    (nk, means, resp_log_resp)
                } else {
                    let est = host.e_step_biased(&lw, &cur.means, &cur.precisions_cholesky);
                    (est.nk, est.means, est.resp_log_resp)
                };
                // `sk` — sklearn's `_estimate_gaussian_parameters` third
                // output, identical to the plain model's M-step covariance.
                let sk = if let Some(dev) = device.as_mut() {
                    dev.covariances(pool, &nk, &means).map_err(AlgoError::Prim)?
                } else {
                    host.covariances(&nk, &means)
                };
                cur = self.m_step(&priors, &nk, &means, &sk, d, k)?;
                // sklearn computes the bound AFTER the M-step but from the
                // PRE-M-step responsibilities; `resp_log_resp` is exactly that
                // (see the module docs' second trap).
                lower_bound = self.compute_lower_bound(&cur, resp_log_resp, d, k);
                trace.push(lower_bound);
                iters = it;
                if self.verbose > 0 && it % self.verbose_interval.max(1) == 0 {
                    log::info!(
                        "bayesian_gaussian_mixture: iteration {it}, lower_bound = {lower_bound:e}"
                    );
                }
                if (lower_bound - prev).abs() < self.tol {
                    converged = true;
                    break;
                }
            }
            if best.is_none() || lower_bound > best_bound {
                best_bound = lower_bound;
                best = Some(cur);
                best_n_iter = iters;
                best_converged = converged;
                best_trace = trace;
            }
        }

        let params = best.expect("n_init >= 1 guarantees at least one restart");

        // sklearn's terminal E-step, so `fit_predict`'s labels come from the
        // FINAL posterior rather than the last iteration's. Its bound is
        // deliberately discarded (sklearn keeps `max_lower_bound`).
        let lw = self.log_weight_term(&params, d);
        let labels: Vec<i32> = if let Some(dev) = device.as_mut() {
            let _ = dev
                .e_step_biased(pool, &lw, &params.means, &params.precisions_cholesky)
                .map_err(AlgoError::Prim)?;
            argmax_rows(&dev.resp_to_host(pool), n, k)
        } else {
            let _ = host.e_step_biased(&lw, &params.means, &params.precisions_cholesky);
            argmax_rows(host.resp(), n, k)
        };
        if let Some(dev) = device {
            dev.release_into(pool);
        }

        Ok(BayesianGaussianMixture {
            device: self.device,
            device_: Some(device_arm),
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            n_init: self.n_init,
            init_params: self.init_params,
            weight_concentration_prior_type: self.weight_concentration_prior_type,
            weight_concentration_prior: self.weight_concentration_prior,
            mean_precision_prior: self.mean_precision_prior,
            mean_prior: self.mean_prior,
            degrees_of_freedom_prior: self.degrees_of_freedom_prior,
            covariance_prior: self.covariance_prior,
            random_state: self.random_state,
            warm_start: self.warm_start,
            verbose: self.verbose,
            verbose_interval: self.verbose_interval,
            warm: Some(params.clone()),
            params: Some(params),
            priors: Some(priors),
            converged_: best_converged,
            n_iter_: best_n_iter,
            lower_bound_: best_bound,
            lower_bounds_: best_trace,
            n_features_in_: d,
            n_samples_: n,
            train_labels: Some(labels),
            _float: PhantomData,
            _state: PhantomData,
        })
    }

    /// sklearn's `_check_parameters`: fill in every prior's data-DEPENDENT
    /// default and validate whatever the caller supplied instead.
    ///
    /// The four defaults that read `X` are why this cannot live at `build()`
    /// (the D-08 split): `mean_prior_` is the design's column mean and
    /// `covariance_prior_` its empirical covariance at `ddof = 1` — numpy's
    /// `np.cov` / `np.var(ddof=1)` convention, which is NOT the `ddof = 0` MLE
    /// the E-step's own covariances use. Getting that one wrong shifts every
    /// fitted covariance by a factor of `n/(n−1)`.
    fn resolve_priors(&self, x: &[f64], n: usize, d: usize) -> Result<MixturePriors, AlgoError> {
        let k = self.n_components;
        let ct = self.covariance_type;

        let weight_concentration = self.weight_concentration_prior.unwrap_or(1.0 / k as f64);
        let mean_precision = self.mean_precision_prior.unwrap_or(1.0);

        // Column means, needed for `mean_prior_` and for the empirical
        // covariance below.
        let mut col_mean = vec![0.0f64; d];
        for i in 0..n {
            for j in 0..d {
                col_mean[j] += x[i * d + j];
            }
        }
        for m in col_mean.iter_mut() {
            *m /= n as f64;
        }

        let mean = match &self.mean_prior {
            Some(m) => {
                if m.len() != d {
                    return Err(prior_error(
                        "mean_prior",
                        format!("expected length {d} (n_features), got {}", m.len()),
                    ));
                }
                if m.iter().any(|v| !v.is_finite()) {
                    return Err(prior_error(
                        "mean_prior",
                        "every entry must be finite".into(),
                    ));
                }
                m.clone()
            }
            None => col_mean.clone(),
        };

        let degrees_of_freedom = match self.degrees_of_freedom_prior {
            Some(v) => {
                // sklearn: "degrees_of_freedom_prior_ ... must be greater than
                // n_features - 1" — below that the Wishart is improper and
                // `lnΓ(½(ν − j))` is undefined for the last feature.
                if !v.is_finite() || v <= d as f64 - 1.0 {
                    return Err(prior_error(
                        "degrees_of_freedom_prior",
                        format!("must be finite and > n_features - 1 = {}, got {v}", d - 1),
                    ));
                }
                v
            }
            None => d as f64,
        };

        let covariance = match &self.covariance_prior {
            Some(c) => {
                let want = ct.param_len(1, d);
                if c.len() != want {
                    return Err(prior_error(
                        "covariance_prior",
                        format!(
                            "expected {want} entries for covariance_type='{}', got {}",
                            ct.name(),
                            c.len()
                        ),
                    ));
                }
                validate_covariance_prior(c, d, ct)?;
                c.clone()
            }
            None => empirical_covariance_prior(x, &col_mean, n, d, ct),
        };

        Ok(MixturePriors {
            weight_concentration,
            mean_precision,
            mean,
            degrees_of_freedom,
            covariance,
        })
    }

    /// sklearn's `_initialize(X, resp)`: build the first responsibilities from
    /// `init_params`, then run ONE variational M-step off them.
    ///
    /// Unlike the plain model there is no injected-parameter override to apply
    /// afterwards — `BayesianGaussianMixture` has no `weights_init` /
    /// `means_init` / `precisions_init`, because the priors play that role.
    fn initialize(
        &self,
        host: &mut GmmHost<'_>,
        rng: &mut SplitMix64,
        priors: &MixturePriors,
        n: usize,
        d: usize,
        k: usize,
    ) -> Result<BayesianMixtureParams, AlgoError> {
        let resp = initial_responsibilities(self.init_params, host, rng, n, k);
        host.set_resp(&resp);
        let (nk, xk) = host.nk_and_means_from_resp();
        let sk = host.covariances(&nk, &xk);
        self.m_step(priors, &nk, &xk, &sk, d, k)
    }

    /// The variational M-step: sklearn's `_estimate_weights` +
    /// `_estimate_means` + `_estimate_precisions`, in that order (the last two
    /// read what the previous one wrote).
    ///
    /// `nk` / `xk` / `sk` are `_estimate_gaussian_parameters`' three outputs,
    /// produced by the shared host engine.
    fn m_step(
        &self,
        pr: &MixturePriors,
        nk: &[f64],
        xk: &[f64],
        sk: &[f64],
        d: usize,
        k: usize,
    ) -> Result<BayesianMixtureParams, AlgoError> {
        let ct = self.covariance_type;

        // --- weights: the conjugate update of the chosen prior family ------ //
        let (wc_a, wc_b) = weight_concentration(
            self.weight_concentration_prior_type,
            pr.weight_concentration,
            nk,
        );

        // --- means: the Normal-Wishart mean posterior ---------------------- //
        let mean_precision: Vec<f64> = nk.iter().map(|v| pr.mean_precision + v).collect();
        let mut means = vec![0.0f64; k * d];
        for c in 0..k {
            let inv = 1.0 / mean_precision[c];
            for j in 0..d {
                means[c * d + j] = (pr.mean_precision * pr.mean[j] + nk[c] * xk[c * d + j]) * inv;
            }
        }

        // --- precisions: the Wishart posterior, per parameterization ------- //
        // Every branch is the same three-term sum — prior + scatter + the
        // mean-shift correction `nk·β₀/β · (x̄ − m₀)(x̄ − m₀)ᵀ` — divided by the
        // posterior degrees of freedom. Only the layout differs.
        let (degrees_of_freedom, covariances) = match ct {
            CovarianceType::Full => {
                let dof: Vec<f64> = nk.iter().map(|v| pr.degrees_of_freedom + v).collect();
                let mut cov = vec![0.0f64; k * d * d];
                for c in 0..k {
                    let scale = nk[c] * pr.mean_precision / mean_precision[c];
                    let inv_dof = 1.0 / dof[c];
                    let block = &mut cov[c * d * d..(c + 1) * d * d];
                    let sk_c = &sk[c * d * d..(c + 1) * d * d];
                    for a in 0..d {
                        let da = xk[c * d + a] - pr.mean[a];
                        for b in 0..d {
                            let db = xk[c * d + b] - pr.mean[b];
                            block[a * d + b] = (pr.covariance[a * d + b]
                                + nk[c] * sk_c[a * d + b]
                                + scale * da * db)
                                * inv_dof;
                        }
                    }
                }
                (dof, cov)
            }
            CovarianceType::Tied => {
                // One shared Wishart, so `nk` enters only through its SUM —
                // sklearn's `nk.sum() / n_components`.
                let nk_sum: f64 = nk.iter().sum();
                let dof = pr.degrees_of_freedom + nk_sum / k as f64;
                let inv_dof = 1.0 / dof;
                let mut cov = vec![0.0f64; d * d];
                let scale = nk_sum / k as f64;
                for a in 0..d {
                    for b in 0..d {
                        let mut cross = 0.0;
                        for c in 0..k {
                            let da = xk[c * d + a] - pr.mean[a];
                            let db = xk[c * d + b] - pr.mean[b];
                            cross += (nk[c] / mean_precision[c]) * da * db;
                        }
                        cov[a * d + b] = (pr.covariance[a * d + b]
                            + scale * sk[a * d + b]
                            + pr.mean_precision / k as f64 * cross)
                            * inv_dof;
                    }
                }
                (vec![dof], cov)
            }
            CovarianceType::Diag => {
                let dof: Vec<f64> = nk.iter().map(|v| pr.degrees_of_freedom + v).collect();
                let mut cov = vec![0.0f64; k * d];
                for c in 0..k {
                    let ratio = pr.mean_precision / mean_precision[c];
                    let inv_dof = 1.0 / dof[c];
                    for j in 0..d {
                        let diff = xk[c * d + j] - pr.mean[j];
                        cov[c * d + j] = (pr.covariance[j]
                            + nk[c] * (sk[c * d + j] + ratio * diff * diff))
                            * inv_dof;
                    }
                }
                (dof, cov)
            }
            CovarianceType::Spherical => {
                let dof: Vec<f64> = nk.iter().map(|v| pr.degrees_of_freedom + v).collect();
                let mut cov = vec![0.0f64; k];
                for c in 0..k {
                    let ratio = pr.mean_precision / mean_precision[c];
                    let mut mean_sq = 0.0;
                    for j in 0..d {
                        let diff = xk[c * d + j] - pr.mean[j];
                        mean_sq += diff * diff;
                    }
                    mean_sq /= d as f64;
                    cov[c] = (pr.covariance[0] + nk[c] * (sk[c] + ratio * mean_sq)) / dof[c];
                }
                (dof, cov)
            }
        };

        let precisions_cholesky =
            precisions_cholesky(&covariances, k, d, ct).map_err(ill_conditioned)?;

        Ok(BayesianMixtureParams {
            weight_concentration_a: wc_a,
            weight_concentration_b: wc_b,
            mean_precision,
            means,
            degrees_of_freedom,
            covariances,
            precisions_cholesky,
        })
    }

    /// The per-component additive log-term the variational E-step contributes,
    /// i.e. everything in sklearn's `_estimate_log_prob` +
    /// `_estimate_log_weights` that is NOT the Gaussian kernel the shared host
    /// engine already computes.
    ///
    /// Three pieces, all `O(d)` per component:
    /// - `E[ln π_c]`, from the stick-breaking or Dirichlet posterior;
    /// - `−½·d·ln ν_c`, sklearn's correction for storing the NORMALIZED
    ///   precision (`covariances_` already carries the `1/ν` factor, so the
    ///   Cholesky log-determinant the engine adds is off by exactly this);
    /// - `½(E[ln|Λ_c|] − d/β_c)`, the expected log-precision and the
    ///   uncertainty in the mean.
    fn log_weight_term(&self, p: &BayesianMixtureParams, d: usize) -> Vec<f64> {
        let k = self.n_components;
        let elw = expected_log_weights(
            self.weight_concentration_prior_type,
            &p.weight_concentration_a,
            &p.weight_concentration_b,
        );
        (0..k)
            .map(|c| {
                let nu = p.dof(c);
                let log_lambda = expected_log_det_precision(nu, d);
                elw[c] - 0.5 * d as f64 * nu.ln()
                    + 0.5 * (log_lambda - d as f64 / p.mean_precision[c])
            })
            .collect()
    }

    /// sklearn's `_compute_lower_bound`: the evidence lower bound with every
    /// term that does not depend on the parameters dropped.
    ///
    /// `resp_log_resp` is `Σ_i Σ_c r·ln r` from the E-step that PRECEDED the
    /// M-step whose output `p` is — see the module docs' second trap.
    fn compute_lower_bound(
        &self,
        p: &BayesianMixtureParams,
        resp_log_resp: f64,
        d: usize,
        k: usize,
    ) -> f64 {
        let ct = self.covariance_type;
        let log_det = log_det_cholesky(&p.precisions_cholesky, k, d, ct);
        // Same `−½·d·ln ν` normalization correction as in `log_weight_term`.
        let ldpc = |c: usize| log_det[c] - 0.5 * d as f64 * p.dof(c).ln();

        let log_wishart = if ct == CovarianceType::Tied {
            // ONE shared Wishart, counted `k` times — sklearn multiplies rather
            // than summing because there is only one value to sum.
            k as f64 * log_wishart_norm(p.dof(0), ldpc(0), d)
        } else {
            (0..k).map(|c| log_wishart_norm(p.dof(c), ldpc(c), d)).sum()
        };

        let log_norm_weight = match self.weight_concentration_prior_type {
            WeightConcentrationPriorType::DirichletProcess => -p
                .weight_concentration_a
                .iter()
                .zip(p.weight_concentration_b.iter())
                .map(|(&a, &b)| betaln(a, b))
                .sum::<f64>(),
            WeightConcentrationPriorType::DirichletDistribution => {
                let sum: f64 = p.weight_concentration_a.iter().sum();
                lgamma(sum)
                    - p.weight_concentration_a
                        .iter()
                        .map(|&a| lgamma(a))
                        .sum::<f64>()
            }
        };

        let log_beta: f64 = p.mean_precision.iter().map(|v| v.ln()).sum();
        -resp_log_resp - log_wishart - log_norm_weight - 0.5 * d as f64 * log_beta
    }
}

/// The `estimator` discriminator written into every `BayesianGaussianMixture`
/// file. See [`gaussian_mixture`](super::gaussian_mixture)'s tag for why it is
/// load-bearing.
const PERSIST_TAG: &str = "bayesian_gaussian_mixture";

/// The seven posterior blocks a variational mixture holds, in the order they are
/// written and read.
///
/// One list rather than two so the save and load sides cannot drift:
/// `weight_concentration_a`/`_b`, `mean_precision` and `degrees_of_freedom` are
/// all length-`k` `f64` vectors, so a name reordered on one side only would
/// produce a file that round-trips its own geometry perfectly and scores
/// everything wrongly — exactly the failure no length check can catch.
const POSTERIOR_NAMES: [&str; 7] = [
    "weight_concentration_a_",
    "weight_concentration_b_",
    "mean_precision_",
    "means_",
    "degrees_of_freedom_",
    "covariances_",
    "precisions_cholesky_",
];

/// The five PRIOR values, which are hyperparameters resolved at fit rather than
/// posterior state. Two are vectors and three are scalars, so they are staged
/// individually rather than through a shared helper.
const PRIOR_MEAN_NAME: &str = "mean_prior_";
/// See [`PRIOR_MEAN_NAME`].
const PRIOR_COVARIANCE_NAME: &str = "covariance_prior_";

impl<F> SaveModel for BayesianGaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted mixture to `path` as a safetensors file.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | the seven posterior blocks | `F64` | see [`POSTERIOR_NAMES`] |
    /// | `mean_prior_` | `F64` | `[n_features]` |
    /// | `covariance_prior_` | `F64` | flat, `param_len(1, d)` |
    /// | `weight_concentration_prior_` / `mean_precision_prior_` / `degrees_of_freedom_prior_` | `__metadata__` | — |
    /// | `lower_bounds_` / `train_labels` | `F64` / `I64` | as `GaussianMixture` |
    /// | seventeen `param:*` scalars | `__metadata__` | — |
    ///
    /// ## Why the RESOLVED priors are stored, not just the requests
    ///
    /// Every one of sklearn's five `*_prior` arguments defaults to `None` and
    /// resolves at fit against the DATA: `mean_prior` becomes the sample mean,
    /// `covariance_prior` the sample covariance, `degrees_of_freedom_prior` the
    /// feature count, and so on. The request and the outcome are therefore
    /// different facts, and both round-trip — the same split
    /// [`kernel_persist`](crate::kernel_persist) makes for the kernel
    /// coefficient, and here it matters more: a reloaded model that re-derived
    /// its priors would need the training data, which this file does not hold.
    ///
    /// `mean_prior_` and `covariance_prior_` are ARRAYS, so they ride as tensors
    /// (under their fitted names, no `param:` prefix); the three scalar priors
    /// ride in `__metadata__` beside them.
    ///
    /// Everything is `F64` regardless of the estimator's `F`, for the reason
    /// [`mixture_persist`](super::mixture_persist) gives.
    fn save(&self, _pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let absent = |field| PersistError::MissingState {
            estimator: PERSIST_TAG,
            field,
        };
        let params = self.params.as_ref().ok_or_else(|| absent("params"))?;
        let priors = self.priors.as_ref().ok_or_else(|| absent("priors"))?;
        let k = params.means.len() / self.n_features_in_.max(1);
        let param_len = self.covariance_type.param_len(k, self.n_features_in_);
        // Bound BEFORE the writer, which borrows every payload.
        let train_labels: Option<Vec<i64>> = self
            .train_labels
            .as_ref()
            .map(|l| l.iter().map(|&v| i64::from(v)).collect());

        let mut w = MixtureWriter::new(PERSIST_TAG);
        w.scalar_usize("param:n_components", self.n_components);
        w.scalar_str("param:covariance_type", self.covariance_type.name());
        w.scalar_f64("param:tol", self.tol);
        w.scalar_f64("param:reg_covar", self.reg_covar);
        w.scalar_usize("param:max_iter", self.max_iter);
        w.scalar_usize("param:n_init", self.n_init);
        w.scalar_str("param:init_params", self.init_params.name());
        w.scalar_str(
            "param:weight_concentration_prior_type",
            self.weight_concentration_prior_type.name(),
        );
        w.scalar_opt_f64(
            "param:weight_concentration_prior",
            self.weight_concentration_prior,
        );
        w.scalar_opt_f64("param:mean_precision_prior", self.mean_precision_prior);
        w.scalar_opt_f64(
            "param:degrees_of_freedom_prior",
            self.degrees_of_freedom_prior,
        );
        w.scalar_opt_u64("param:random_state", self.random_state);
        w.scalar_bool("param:warm_start", self.warm_start);
        w.scalar_usize("param:verbose", self.verbose);
        w.scalar_usize("param:verbose_interval", self.verbose_interval);
        w.scalar_str("param:device", self.device.name());

        // The three RESOLVED scalar priors — distinct from the `param:` requests
        // above, which are `Option` and usually absent.
        w.scalar_f64("weight_concentration_prior_", priors.weight_concentration);
        w.scalar_f64("mean_precision_prior_", priors.mean_precision);
        w.scalar_f64("degrees_of_freedom_prior_", priors.degrees_of_freedom);

        w.scalar_bool("converged_", self.converged_);
        w.scalar_usize("n_iter_", self.n_iter_);
        w.scalar_f64("lower_bound_", self.lower_bound_);
        w.scalar_usize("n_features_in_", self.n_features_in_);
        w.scalar_usize("n_samples_", self.n_samples_);
        if let Some(arm) = self.device_ {
            w.scalar_str("device_", arm);
        }

        // The seven posterior blocks, staged in `POSTERIOR_NAMES` order with
        // each one's own expected length.
        let blocks: [(&'static str, &[f64], usize); 7] = [
            (POSTERIOR_NAMES[0], &params.weight_concentration_a, k),
            (POSTERIOR_NAMES[1], &params.weight_concentration_b, k),
            (POSTERIOR_NAMES[2], &params.mean_precision, k),
            (
                POSTERIOR_NAMES[3],
                &params.means,
                k * self.n_features_in_,
            ),
            (POSTERIOR_NAMES[4], &params.degrees_of_freedom, k),
            (POSTERIOR_NAMES[5], &params.covariances, param_len),
            (
                POSTERIOR_NAMES[6],
                &params.precisions_cholesky,
                param_len,
            ),
        ];
        for (name, values, expected) in blocks {
            expect_len(name, values.len(), expected, "entries")?;
            // `means_` alone is rank-2 — it is the one block whose two extents
            // carry separate meaning, and storing it `[k, d]` is what lets a
            // Python reader index it by component the way sklearn's attribute
            // does. The rest are flat by construction (see `param_len`).
            let shape = if name == POSTERIOR_NAMES[3] {
                vec![k, self.n_features_in_]
            } else {
                vec![values.len()]
            };
            w.tensor(name, TensorRef::f64s(values, shape)?);
        }

        w.tensor(
            PRIOR_MEAN_NAME,
            TensorRef::f64s(&priors.mean, vec![priors.mean.len()])?,
        );
        w.tensor(
            PRIOR_COVARIANCE_NAME,
            TensorRef::f64s(&priors.covariance, vec![priors.covariance.len()])?,
        );
        write_opt_vec(&mut w, "param:mean_prior", self.mean_prior.as_ref())?;
        write_opt_vec(
            &mut w,
            "param:covariance_prior",
            self.covariance_prior.as_ref(),
        )?;
        w.tensor(
            LOWER_BOUNDS_NAME,
            TensorRef::f64s(&self.lower_bounds_, vec![self.lower_bounds_.len()])?,
        );
        if let Some(l) = train_labels.as_ref() {
            w.tensor(TRAIN_LABELS_NAME, TensorRef::i64s(l, vec![l.len()])?);
        }
        w.write(path)
    }
}

impl<F> LoadModel for BayesianGaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read the mixture back from `path`.
    ///
    /// `covariance_type` and `n_features_in_` are read first, because together
    /// they fix the flat length every parameter block must have. The file is
    /// untrusted input (T-04-01-01), so each of the seven posterior blocks is
    /// measured against that length before a single value is stored: they index
    /// each other component-wise inside the variational E-step, and a short one
    /// would read past its end on the first `score_samples`.
    ///
    /// The `warm` resumption block is NOT restored. Unlike `GaussianMixture`'s,
    /// it is a `BayesianMixtureParams` whose seven arrays would double this
    /// file, and a `warm_start` continuation of a variational fit re-derives it
    /// from the posterior the file already holds — so storing it would be a
    /// second copy of recoverable state, which this format does not do.
    fn load(
        _pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<BayesianGaussianMixture<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = MixtureFile::parse(&raw, PERSIST_TAG)?;

        let covariance_type = CovarianceType::try_from(file.scalar_str("param:covariance_type")?)
            .map_err(|_| PersistError::BadMetadata {
            key: "param:covariance_type",
        })?;
        let n_features_in_ = file.scalar_usize("n_features_in_")?;
        if n_features_in_ == 0 {
            return Err(PersistError::InconsistentGeometry {
                reason: "'n_features_in_' is 0; a fitted mixture has at least one feature"
                    .to_string(),
            });
        }

        // `k` comes off the first length-`k` posterior, and every other block is
        // measured against it.
        let first = file.tensor(POSTERIOR_NAMES[0])?;
        let k = shape_1d(&first, POSTERIOR_NAMES[0])?;
        if k == 0 {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "tensor '{}' is empty; a fitted mixture has at least one component",
                    POSTERIOR_NAMES[0]
                ),
            });
        }
        let param_len = covariance_type.param_len(k, n_features_in_);

        let mut blocks: Vec<Vec<f64>> = Vec::with_capacity(7);
        for (i, name) in POSTERIOR_NAMES.iter().enumerate() {
            let expected = match i {
                3 => k * n_features_in_,
                5 | 6 => param_len,
                _ => k,
            };
            let view = file.tensor(name)?;
            let len: usize = view.shape().iter().product();
            expect_len(name, len, expected, "entries")?;
            blocks.push(as_f64(&view, name)?.into_owned());
        }
        let mut drain = blocks.into_iter();
        let params = BayesianMixtureParams {
            weight_concentration_a: drain.next().expect("seven blocks were read"),
            weight_concentration_b: drain.next().expect("seven blocks were read"),
            mean_precision: drain.next().expect("seven blocks were read"),
            means: drain.next().expect("seven blocks were read"),
            degrees_of_freedom: drain.next().expect("seven blocks were read"),
            covariances: drain.next().expect("seven blocks were read"),
            precisions_cholesky: drain.next().expect("seven blocks were read"),
        };

        let mean_prior_v = file.tensor(PRIOR_MEAN_NAME)?;
        expect_len(
            PRIOR_MEAN_NAME,
            shape_1d(&mean_prior_v, PRIOR_MEAN_NAME)?,
            n_features_in_,
            "entries",
        )?;
        let cov_prior_v = file.tensor(PRIOR_COVARIANCE_NAME)?;
        expect_len(
            PRIOR_COVARIANCE_NAME,
            shape_1d(&cov_prior_v, PRIOR_COVARIANCE_NAME)?,
            covariance_type.param_len(1, n_features_in_),
            "entries",
        )?;
        let priors = MixturePriors {
            weight_concentration: file.scalar_f64("weight_concentration_prior_")?,
            mean_precision: file.scalar_f64("mean_precision_prior_")?,
            mean: as_f64(&mean_prior_v, PRIOR_MEAN_NAME)?.into_owned(),
            degrees_of_freedom: file.scalar_f64("degrees_of_freedom_prior_")?,
            covariance: as_f64(&cov_prior_v, PRIOR_COVARIANCE_NAME)?.into_owned(),
        };

        let lower_bounds_v = file.tensor(LOWER_BOUNDS_NAME)?;
        shape_1d(&lower_bounds_v, LOWER_BOUNDS_NAME)?;
        let lower_bounds_ = as_f64(&lower_bounds_v, LOWER_BOUNDS_NAME)?.into_owned();

        let n_samples_ = file.scalar_usize("n_samples_")?;
        let train_labels = match file.tensor_opt(TRAIN_LABELS_NAME) {
            None => None,
            Some(view) => {
                expect_len(
                    TRAIN_LABELS_NAME,
                    shape_1d(&view, TRAIN_LABELS_NAME)?,
                    n_samples_,
                    "entries",
                )?;
                Some(
                    as_i64(&view, TRAIN_LABELS_NAME)?
                        .iter()
                        .map(|&v| {
                            i32::try_from(v).map_err(|_| PersistError::InconsistentGeometry {
                                reason: format!(
                                    "tensor '{TRAIN_LABELS_NAME}' holds the component id {v}, \
                                     which does not fit an i32"
                                ),
                            })
                        })
                        .collect::<Result<Vec<i32>, _>>()?,
                )
            }
        };

        Ok(BayesianGaussianMixture {
            n_components: file.scalar_usize("param:n_components")?,
            covariance_type,
            tol: file.scalar_f64("param:tol")?,
            reg_covar: file.scalar_f64("param:reg_covar")?,
            max_iter: file.scalar_usize("param:max_iter")?,
            n_init: file.scalar_usize("param:n_init")?,
            init_params: InitParams::try_from(file.scalar_str("param:init_params")?).map_err(
                |_| PersistError::BadMetadata {
                    key: "param:init_params",
                },
            )?,
            weight_concentration_prior_type: WeightConcentrationPriorType::try_from(
                file.scalar_str("param:weight_concentration_prior_type")?,
            )
            .map_err(|_| PersistError::BadMetadata {
                key: "param:weight_concentration_prior_type",
            })?,
            weight_concentration_prior: file
                .scalar_opt_f64("param:weight_concentration_prior")?,
            mean_precision_prior: file.scalar_opt_f64("param:mean_precision_prior")?,
            mean_prior: read_opt_vec(&file, "param:mean_prior")?,
            degrees_of_freedom_prior: file.scalar_opt_f64("param:degrees_of_freedom_prior")?,
            covariance_prior: read_opt_vec(&file, "param:covariance_prior")?,
            random_state: file.scalar_opt_u64("param:random_state")?,
            warm_start: file.scalar_bool("param:warm_start")?,
            verbose: file.scalar_usize("param:verbose")?,
            verbose_interval: file.scalar_usize("param:verbose_interval")?,
            // Re-derived from the posterior on the next `warm_start` fit rather
            // than stored — see this impl's docs.
            warm: None,
            device: Device::from_name(file.scalar_str("param:device")?).ok_or(
                PersistError::BadMetadata {
                    key: "param:device",
                },
            )?,
            params: Some(params),
            device_: read_device_arm(&file, "device_")?,
            priors: Some(priors),
            converged_: file.scalar_bool("converged_")?,
            n_iter_: file.scalar_usize("n_iter_")?,
            lower_bound_: file.scalar_f64("lower_bound_")?,
            lower_bounds_,
            n_features_in_,
            n_samples_,
            train_labels,
            _float: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for BayesianGaussianMixture<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = BayesianGaussianMixture<F, Fitted>;

    /// Device ingress: validate the geometry, read the design back ONCE, and
    /// run the same host core [`BayesianGaussianMixture::fit_from_host_slice`]
    /// runs. `y` is ignored — a mixture model is unsupervised.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Self::Fitted, AlgoError> {
        validate_geometry(x, shape)?;
        let host: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        self.fit_core(pool, &host, shape)
    }
}

// ---------------------------------------------------------------------------
// Fitted state
// ---------------------------------------------------------------------------

impl<F> BayesianGaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn p(&self) -> &BayesianMixtureParams {
        self.params
            .as_ref()
            .expect("a Fitted BayesianGaussianMixture always carries parameters")
    }

    fn pr(&self) -> &MixturePriors {
        self.priors
            .as_ref()
            .expect("a Fitted BayesianGaussianMixture always carries priors")
    }

    /// `weights_` — the mixing proportions implied by the weight posterior.
    ///
    /// Under `dirichlet_distribution` this is just `α / Σα`. Under
    /// `dirichlet_process` it is the stick-breaking mean
    /// `E[v_c]·Π_{j<c}(1 − E[v_j])`, renormalized — which is why it is NOT
    /// `nk / n` and why it depends on the component order.
    pub fn weights(&self) -> Vec<F> {
        self.weights_f64().iter().map(|&v| f64_to_host(v)).collect()
    }

    /// `weights_` in full `f64` — the oracle's comparison target.
    pub fn weights_f64(&self) -> Vec<f64> {
        mixing_weights(
            self.weight_concentration_prior_type,
            &self.p().weight_concentration_a,
            &self.p().weight_concentration_b,
        )
    }

    /// `means_` — `n_components × n_features`, row-major.
    pub fn means(&self) -> Vec<F> {
        self.p().means.iter().map(|&v| f64_to_host(v)).collect()
    }

    /// `covariances_`, flat in the [`CovarianceType::param_shape`] layout.
    pub fn covariances(&self) -> Vec<F> {
        self.p()
            .covariances
            .iter()
            .map(|&v| f64_to_host(v))
            .collect()
    }

    /// `precisions_cholesky_`, same layout as `covariances_`.
    pub fn precisions_cholesky(&self) -> Vec<F> {
        self.p()
            .precisions_cholesky
            .iter()
            .map(|&v| f64_to_host(v))
            .collect()
    }

    /// `precisions_` — `precisions_cholesky_ · precisions_choleskyᵀ`.
    pub fn precisions(&self) -> Vec<F> {
        precisions_from_cholesky(
            &self.p().precisions_cholesky,
            self.n_components,
            self.n_features_in_,
            self.covariance_type,
        )
        .iter()
        .map(|&v| f64_to_host(v))
        .collect()
    }

    /// `weight_concentration_` — the pair of Beta parameters under
    /// `dirichlet_process` (second element non-empty), or the single Dirichlet
    /// concentration vector under `dirichlet_distribution` (second element
    /// empty).
    pub fn weight_concentration(&self) -> (&[f64], &[f64]) {
        (
            &self.p().weight_concentration_a,
            &self.p().weight_concentration_b,
        )
    }

    /// `mean_precision_` (β), length `n_components`.
    pub fn mean_precision(&self) -> &[f64] {
        &self.p().mean_precision
    }

    /// `degrees_of_freedom_` (ν) — length `n_components`, or length 1 under
    /// `covariance_type='tied'`.
    pub fn degrees_of_freedom(&self) -> &[f64] {
        &self.p().degrees_of_freedom
    }

    /// The sklearn SHAPE of `degrees_of_freedom_`: `[]` (a scalar) under
    /// `tied`, `[n_components]` otherwise. Exposed so the Python layer can
    /// reproduce sklearn's scalar-vs-array asymmetry without re-deriving it.
    pub fn degrees_of_freedom_shape(&self) -> Vec<usize> {
        if self.covariance_type == CovarianceType::Tied {
            Vec::new()
        } else {
            vec![self.n_components]
        }
    }

    /// The five resolved `*_prior_` attributes.
    pub fn priors(&self) -> &MixturePriors {
        self.pr()
    }

    /// The `f64` posterior block, for callers that must not lose precision (the
    /// oracle tests and the `warm_start` hand-off).
    pub fn params_f64(&self) -> &BayesianMixtureParams {
        self.p()
    }

    /// The shape of `covariances_` / `precisions_` for this parameterization.
    pub fn covariance_shape(&self) -> Vec<usize> {
        self.covariance_type
            .param_shape(self.n_components, self.n_features_in_)
    }

    /// `converged_` — did the best restart hit `|Δ lower_bound| < tol`?
    /// The EM engine that ACTUALLY ran, `"cpu"` or `"gpu"` (DEVICE-PARAM-01).
    ///
    /// `device='gpu'` overrides the SIZE half of `gmm_device_applicable`, not
    /// its capability half: a backend without `f64` transcendentals cannot run
    /// the device engine at all, and this reports the host fallback.
    pub fn device_arm(&self) -> Option<&'static str> {
        self.device_
    }

    pub fn converged(&self) -> bool {
        self.converged_
    }

    /// `n_iter_` — variational iterations used by the best restart.
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// `lower_bound_` — the best restart's evidence lower bound. NOT a
    /// log-likelihood; see the module docs.
    pub fn lower_bound(&self) -> f64 {
        self.lower_bound_
    }

    /// `lower_bounds_` — the per-iteration bound trace of the WINNING restart.
    pub fn lower_bounds(&self) -> &[f64] {
        &self.lower_bounds_
    }

    /// `n_features_in_`.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    /// Training-set row count.
    pub fn n_samples(&self) -> usize {
        self.n_samples_
    }

    /// The `covariance_type` this estimator was built with.
    pub fn covariance_type(&self) -> CovarianceType {
        self.covariance_type
    }

    /// The `weight_concentration_prior_type` this estimator was built with.
    pub fn weight_concentration_prior_type(&self) -> WeightConcentrationPriorType {
        self.weight_concentration_prior_type
    }

    /// The training-set assignment from `fit`'s terminal E-step — sklearn's
    /// `fit_predict(X)` return value, for free.
    pub fn labels(&self) -> &[i32] {
        self.train_labels
            .as_deref()
            .expect("a Fitted BayesianGaussianMixture always carries its training labels")
    }

    /// Turn a fitted estimator back into an `Unfit` one that RESUMES from this
    /// posterior — the typestate spelling of sklearn's `warm_start=True`.
    pub fn into_warm_start(self) -> BayesianGaussianMixture<F, Unfit> {
        BayesianGaussianMixture {
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            n_init: self.n_init,
            init_params: self.init_params,
            weight_concentration_prior_type: self.weight_concentration_prior_type,
            weight_concentration_prior: self.weight_concentration_prior,
            mean_precision_prior: self.mean_precision_prior,
            mean_prior: self.mean_prior,
            degrees_of_freedom_prior: self.degrees_of_freedom_prior,
            covariance_prior: self.covariance_prior,
            random_state: self.random_state,
            warm_start: self.warm_start,
            verbose: self.verbose,
            verbose_interval: self.verbose_interval,
            warm: self.warm,
            device: self.device,
            params: None,
            device_: None,
            priors: None,
            converged_: false,
            n_iter_: 0,
            lower_bound_: self.lower_bound_,
            lower_bounds_: Vec::new(),
            n_features_in_: 0,
            n_samples_: 0,
            train_labels: None,
            _float: PhantomData,
            _state: PhantomData,
        }
    }

    /// sklearn's `_estimate_weighted_log_prob` for a HOST design — the one
    /// quantity `predict` / `predict_proba` / `score_samples` all reduce.
    fn weighted_log_prob_host(&self, x: &[f64], n: usize) -> Vec<f64> {
        let p = self.p();
        let d = self.n_features_in_;
        let k = self.n_components;
        // The SAME per-component term the fit loop uses, recomputed from the
        // fitted posterior rather than cached: it is `O(k·d)` against this
        // call's `O(n·k·d²)`, and caching it would be a second source of truth.
        let elw = expected_log_weights(
            self.weight_concentration_prior_type,
            &p.weight_concentration_a,
            &p.weight_concentration_b,
        );
        let term: Vec<f64> = (0..k)
            .map(|c| {
                let nu = p.dof(c);
                elw[c] - 0.5 * d as f64 * nu.ln()
                    + 0.5 * (expected_log_det_precision(nu, d) - d as f64 / p.mean_precision[c])
            })
            .collect();
        weighted_log_prob_biased(
            x,
            n,
            d,
            k,
            self.covariance_type,
            &term,
            &p.means,
            &p.precisions_cholesky,
        )
    }

    /// Widen + geometry-check a host design against the fitted `n_features_in_`.
    fn host_design(&self, x: &[F], shape: (usize, usize)) -> Result<Vec<f64>, AlgoError> {
        let (n, d) = shape;
        if n == 0 || d != self.n_features_in_ || x.len() != n * d {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n,
                cols: self.n_features_in_,
                len: x.len(),
            }));
        }
        Ok(x.iter().map(|&v| host_to_f64(v)).collect())
    }

    /// `predict(X)` over a HOST design: the argmax component per row.
    pub fn predict_labels_host(
        &self,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<Vec<i32>, AlgoError> {
        let x64 = self.host_design(x, shape)?;
        let wlp = self.weighted_log_prob_host(&x64, shape.0);
        Ok(argmax_rows(&wlp, shape.0, self.n_components))
    }

    /// `predict_proba(X)` over a HOST design: the `n × k` posterior
    /// responsibilities, row-major.
    pub fn predict_proba_host(&self, x: &[F], shape: (usize, usize)) -> Result<Vec<F>, AlgoError> {
        Ok(self
            .log_resp_f64(x, shape)?
            .into_iter()
            .map(|v| f64_to_host(v.exp()))
            .collect())
    }

    /// `predict_log_proba(X)` over a HOST design.
    pub fn predict_log_proba_host(
        &self,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<Vec<F>, AlgoError> {
        Ok(self
            .log_resp_f64(x, shape)?
            .into_iter()
            .map(f64_to_host)
            .collect())
    }

    /// The `n × k` LOG responsibilities in full `f64`, kept un-narrowed so an
    /// `f32` estimator does not round twice on the way to a probability.
    fn log_resp_f64(&self, x: &[F], shape: (usize, usize)) -> Result<Vec<f64>, AlgoError> {
        let x64 = self.host_design(x, shape)?;
        let (n, k) = (shape.0, self.n_components);
        let mut wlp = self.weighted_log_prob_host(&x64, n);
        let norm = logsumexp_rows(&wlp, n, k);
        for i in 0..n {
            for c in 0..k {
                wlp[i * k + c] -= norm[i];
            }
        }
        Ok(wlp)
    }

    /// `score_samples(X)` over a HOST design: the per-row log-density.
    pub fn score_samples_host(
        &self,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<Vec<f64>, AlgoError> {
        let x64 = self.host_design(x, shape)?;
        let wlp = self.weighted_log_prob_host(&x64, shape.0);
        Ok(logsumexp_rows(&wlp, shape.0, self.n_components))
    }

    /// `score(X)` — the MEAN per-sample log-density.
    pub fn score_host(&self, x: &[F], shape: (usize, usize)) -> Result<f64, AlgoError> {
        let s = self.score_samples_host(x, shape)?;
        Ok(s.iter().sum::<f64>() / s.len() as f64)
    }

    /// `sample(n_samples)` — draw from the fitted mixture, returning the
    /// `n_samples × n_features` design and each row's component index.
    ///
    /// Identical in shape to `GaussianMixture::sample` (both come from
    /// sklearn's `BaseMixture`): the per-component counts come from a
    /// multinomial over `weights_` and each component's rows are emitted
    /// CONTIGUOUSLY, so `y` is sorted. `seed` replaces sklearn's
    /// `random_state`, whose numpy stream is not reproducible from Rust (D-09).
    pub fn sample(&self, n_samples: usize, seed: u64) -> Result<(Vec<F>, Vec<i32>), AlgoError> {
        if n_samples < 1 {
            return Err(AlgoError::InvalidK {
                estimator: EST,
                k: n_samples,
                n_samples: 1,
            });
        }
        let (k, d) = (self.n_components, self.n_features_in_);
        let p = self.p();
        let weights = self.weights_f64();
        let mut rng = SplitMix64::new(seed);

        let total: f64 = weights.iter().sum();
        let mut counts = vec![0usize; k];
        for _ in 0..n_samples {
            let t = rng.next_f64() * total;
            let mut acc = 0.0;
            let mut pick = k - 1;
            for (c, &w) in weights.iter().enumerate() {
                acc += w;
                if acc >= t {
                    pick = c;
                    break;
                }
            }
            counts[pick] += 1;
        }

        let chol = cholesky_lower_blocks(&p.covariances, k, d, self.covariance_type)
            .map_err(ill_conditioned)?;
        let mut out = Vec::with_capacity(n_samples * d);
        let mut y = Vec::with_capacity(n_samples);
        let mut z = vec![0.0f64; d];
        for c in 0..k {
            for _ in 0..counts[c] {
                for slot in z.iter_mut() {
                    *slot = standard_normal(&mut rng);
                }
                for a in 0..d {
                    let mut v = p.means[c * d + a];
                    match self.covariance_type {
                        CovarianceType::Full | CovarianceType::Tied => {
                            let block = if self.covariance_type == CovarianceType::Full {
                                &chol[c * d * d..(c + 1) * d * d]
                            } else {
                                &chol[..d * d]
                            };
                            for b in 0..=a {
                                v += block[a * d + b] * z[b];
                            }
                        }
                        CovarianceType::Diag => v += chol[c * d + a] * z[a],
                        CovarianceType::Spherical => v += chol[c] * z[a],
                    }
                    out.push(f64_to_host::<F>(v));
                }
                y.push(c as i32);
            }
        }
        Ok((out, y))
    }
}

impl<F> PredictLabels<F> for BayesianGaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        validate_geometry(x, shape)?;
        let host = x.to_host(pool);
        let labels = self.predict_labels_host(&host, shape)?;
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

impl<F> PredictProba<F> for BayesianGaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        validate_geometry(x, shape)?;
        let host = x.to_host(pool);
        let proba = self.predict_proba_host(&host, shape)?;
        Ok(DeviceArray::from_host(pool, &proba))
    }
}

impl<F> PredictLogProba<F> for BayesianGaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict_log_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        validate_geometry(x, shape)?;
        let host = x.to_host(pool);
        let lp = self.predict_log_proba_host(&host, shape)?;
        Ok(DeviceArray::from_host(pool, &lp))
    }
}

impl<F> ScoreSamples<F> for BayesianGaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn score_samples(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        validate_geometry(x, shape)?;
        let host = x.to_host(pool);
        let s = self.score_samples_host(&host, shape)?;
        let narrowed: Vec<F> = s.into_iter().map(f64_to_host).collect();
        Ok(DeviceArray::from_host(pool, &narrowed))
    }
}

// ---------------------------------------------------------------------------
// The weight-posterior arithmetic (public: the oracle pins it directly)
// ---------------------------------------------------------------------------

/// sklearn's `_estimate_weights`: the conjugate update of the weight posterior
/// from the component counts `nk`.
///
/// Returns `(a, b)`. Under `dirichlet_process` these are the two Beta
/// parameters of each stick break — `a_c = 1 + nk_c` and
/// `b_c = γ + Σ_{j>c} nk_j`, the count of everything that comes AFTER `c`,
/// which is what makes the construction order-dependent. Under
/// `dirichlet_distribution` `a = γ + nk` and `b` is empty.
///
/// Public because the oracle pins it directly, from a FIXED `nk` vector: the
/// stick-breaking recursion is the one piece of this estimator that a converged
/// end-to-end comparison cannot check exactly, precisely because it is
/// order-dependent and two engines with different RNGs find the same clustering
/// in different orders.
pub fn weight_concentration(
    prior_type: WeightConcentrationPriorType,
    prior: f64,
    nk: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    match prior_type {
        WeightConcentrationPriorType::DirichletProcess => {
            let k = nk.len();
            let a: Vec<f64> = nk.iter().map(|v| 1.0 + v).collect();
            // Reverse cumulative sum, EXCLUSIVE of the component itself — the
            // last entry is therefore `γ` alone (nothing follows it).
            let mut b = vec![0.0f64; k];
            let mut tail = 0.0;
            for c in (0..k).rev() {
                b[c] = prior + tail;
                tail += nk[c];
            }
            (a, b)
        }
        WeightConcentrationPriorType::DirichletDistribution => {
            (nk.iter().map(|v| prior + v).collect(), Vec::new())
        }
    }
}

/// sklearn's `_estimate_log_weights`: `E[ln π_c]` under the weight posterior.
///
/// The `dirichlet_process` branch is the stick-breaking expectation
/// `ψ(a_c) − ψ(a_c + b_c) + Σ_{j<c} (ψ(b_j) − ψ(a_j + b_j))`, written with the
/// running sum shifted by one so component 0 contributes nothing — sklearn's
/// `np.hstack((0, np.cumsum(...)[:-1]))`.
pub fn expected_log_weights(
    prior_type: WeightConcentrationPriorType,
    a: &[f64],
    b: &[f64],
) -> Vec<f64> {
    match prior_type {
        WeightConcentrationPriorType::DirichletProcess => {
            let k = a.len();
            let mut out = vec![0.0f64; k];
            let mut running = 0.0;
            for c in 0..k {
                let sum = digamma(a[c] + b[c]);
                out[c] = digamma(a[c]) - sum + running;
                running += digamma(b[c]) - sum;
            }
            out
        }
        WeightConcentrationPriorType::DirichletDistribution => {
            let total: f64 = a.iter().sum();
            let dg_total = digamma(total);
            a.iter().map(|&v| digamma(v) - dg_total).collect()
        }
    }
}

/// sklearn's `weights_` derivation inside `_set_parameters`: the posterior MEAN
/// mixing proportions, renormalized to sum to one.
pub fn mixing_weights(prior_type: WeightConcentrationPriorType, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut w: Vec<f64> = match prior_type {
        WeightConcentrationPriorType::DirichletProcess => {
            let k = a.len();
            let mut out = vec![0.0f64; k];
            // `remaining` is the expected fraction of the stick still unbroken
            // before component `c`: `Π_{j<c} b_j/(a_j + b_j)`.
            let mut remaining = 1.0;
            for c in 0..k {
                let total = a[c] + b[c];
                out[c] = a[c] / total * remaining;
                remaining *= b[c] / total;
            }
            out
        }
        WeightConcentrationPriorType::DirichletDistribution => a.to_vec(),
    };
    let sum: f64 = w.iter().sum();
    for v in w.iter_mut() {
        *v /= sum;
    }
    w
}

/// `E[ln|Λ|]` for a Wishart with `ν` degrees of freedom in `d` dimensions,
/// MINUS the `ln|W|` term sklearn folds into the Cholesky log-determinant
/// instead — i.e. sklearn's `log_lambda`.
fn expected_log_det_precision(nu: f64, d: usize) -> f64 {
    let mut acc = d as f64 * std::f64::consts::LN_2;
    for j in 0..d {
        acc += digamma(0.5 * (nu - j as f64));
    }
    acc
}

/// sklearn's `_log_wishart_norm`: the log normalization of a Wishart with `ν`
/// degrees of freedom whose scale matrix has `ln|W^{1/2}| = log_det_prec_chol`.
fn log_wishart_norm(nu: f64, log_det_prec_chol: f64, d: usize) -> f64 {
    let mut gam = 0.0;
    for j in 0..d {
        gam += lgamma(0.5 * (nu - j as f64));
    }
    -(nu * log_det_prec_chol + nu * d as f64 * 0.5 * std::f64::consts::LN_2 + gam)
}

// ---------------------------------------------------------------------------
// Prior resolution helpers
// ---------------------------------------------------------------------------

/// The typed error every fit-time prior rejection raises.
fn prior_error(param: &'static str, reason: String) -> AlgoError {
    AlgoError::InvalidMixtureInit {
        estimator: EST,
        param,
        reason,
    }
}

/// The `covariance_prior_` default: the design's empirical covariance at
/// `ddof = 1`.
///
/// `ddof = 1` is not a detail — sklearn spells this `np.cov(X.T)` /
/// `np.var(X, axis=0, ddof=1)`, whose divisor is `n − 1`, while every OTHER
/// covariance in the mixture (the E-step's `sk`, `covariances_`) is the
/// `ddof = 0` MLE. Using the MLE here would scale the prior by `(n−1)/n`,
/// which is invisible at large `n` and a real error at small `n` — exactly
/// where a prior matters most.
fn empirical_covariance_prior(
    x: &[f64],
    col_mean: &[f64],
    n: usize,
    d: usize,
    ct: CovarianceType,
) -> Vec<f64> {
    let denom = (n as f64) - 1.0;
    match ct {
        CovarianceType::Full | CovarianceType::Tied => {
            let mut cov = vec![0.0f64; d * d];
            for i in 0..n {
                let row = &x[i * d..(i + 1) * d];
                for a in 0..d {
                    let da = row[a] - col_mean[a];
                    for b in a..d {
                        cov[a * d + b] += da * (row[b] - col_mean[b]);
                    }
                }
            }
            for a in 0..d {
                for b in a..d {
                    let v = cov[a * d + b] / denom;
                    cov[a * d + b] = v;
                    cov[b * d + a] = v;
                }
            }
            cov
        }
        CovarianceType::Diag => {
            let mut var = vec![0.0f64; d];
            for i in 0..n {
                for j in 0..d {
                    let t = x[i * d + j] - col_mean[j];
                    var[j] += t * t;
                }
            }
            for v in var.iter_mut() {
                *v /= denom;
            }
            var
        }
        CovarianceType::Spherical => {
            let mut var = vec![0.0f64; d];
            for i in 0..n {
                for j in 0..d {
                    let t = x[i * d + j] - col_mean[j];
                    var[j] += t * t;
                }
            }
            // sklearn's `np.var(X, axis=0, ddof=1).mean()` — the MEAN of the
            // per-feature variances, not the variance of the flattened design.
            let mean_var = var.iter().map(|v| v / denom).sum::<f64>() / d as f64;
            vec![mean_var]
        }
    }
}

/// sklearn's `_check_precision_matrix` / `_check_precision_positivity` for a
/// caller-supplied `covariance_prior`.
fn validate_covariance_prior(c: &[f64], d: usize, ct: CovarianceType) -> Result<(), AlgoError> {
    if c.iter().any(|v| !v.is_finite()) {
        return Err(prior_error(
            "covariance_prior",
            "every entry must be finite".into(),
        ));
    }
    match ct {
        CovarianceType::Full | CovarianceType::Tied => {
            // Symmetry first: an asymmetric matrix can still factor, so a
            // Cholesky alone would silently accept one and then use only its
            // lower triangle.
            for a in 0..d {
                for b in (a + 1)..d {
                    let (u, l) = (c[a * d + b], c[b * d + a]);
                    if (u - l).abs() > 1e-10 * (1.0 + u.abs().max(l.abs())) {
                        return Err(prior_error(
                            "covariance_prior",
                            format!("must be symmetric, but [{a}][{b}] != [{b}][{a}]"),
                        ));
                    }
                }
            }
            // Positive definiteness, by the factorization that would fail later
            // anyway — reported HERE, where the offending parameter is named.
            cholesky_lower_blocks(c, 1, d, CovarianceType::Tied).map_err(|e| {
                prior_error(
                    "covariance_prior",
                    format!(
                        "must be positive definite (pivot {} was {:e})",
                        e.pivot_index, e.pivot_value
                    ),
                )
            })?;
            Ok(())
        }
        CovarianceType::Diag | CovarianceType::Spherical => {
            if c.iter().any(|&v| v <= 0.0) {
                return Err(prior_error(
                    "covariance_prior",
                    "every entry must be > 0".into(),
                ));
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for [`BayesianGaussianMixture`] (Phase-16 A5 convention: `f64`-typed
/// setters, string-valued hyperparameters staying strings until `build`, where
/// they become typed enums or a [`BuildError`]).
#[derive(Debug, Clone)]
pub struct BayesianGaussianMixtureBuilder {
    device: Device,
    n_components: usize,
    covariance_type: String,
    tol: f64,
    reg_covar: f64,
    max_iter: usize,
    n_init: usize,
    init_params: String,
    weight_concentration_prior_type: String,
    weight_concentration_prior: Option<f64>,
    mean_precision_prior: Option<f64>,
    mean_prior: Option<Vec<f64>>,
    degrees_of_freedom_prior: Option<f64>,
    covariance_prior: Option<Vec<f64>>,
    random_state: Option<u64>,
    warm_start: bool,
    verbose: usize,
    verbose_interval: usize,
    /// NOT a hyperparameter: the previous fit's posterior, carried in so a
    /// `warm_start` refit resumes. Excluded from `into_builder` /
    /// `hyperparams_eq` for that reason.
    warm_params: Option<BayesianMixtureParams>,
}

impl Default for BayesianGaussianMixtureBuilder {
    /// Re-derive the sklearn defaults from [`BayesianGaussianMixture::new`] —
    /// the SINGLE source of truth.
    fn default() -> Self {
        BayesianGaussianMixture::<f64, Unfit>::new().into_builder()
    }
}

impl BayesianGaussianMixtureBuilder {
    /// Pin the EM engine (DEVICE-PARAM-01). [`Device::Auto`] keeps the
    /// `gmm_device_applicable` gate; `Cpu`/`Gpu` override its SIZE decision.
    /// The capability half still applies — see [`BayesianGaussianMixture::device_arm`].
    pub fn device(mut self, v: Device) -> Self {
        self.device = v;
        self
    }

    /// Set `n_components` — an UPPER BOUND on the number of components, not a
    /// count (see the module docs).
    pub fn n_components(mut self, v: usize) -> Self {
        self.n_components = v;
        self
    }

    /// Set `covariance_type` (`"full"` / `"tied"` / `"diag"` / `"spherical"`).
    pub fn covariance_type(mut self, v: impl Into<String>) -> Self {
        self.covariance_type = v.into();
        self
    }

    /// Set the convergence threshold on `|Δ lower_bound|`.
    pub fn tol(mut self, v: f64) -> Self {
        self.tol = v;
        self
    }

    /// Set the value ADDED to each covariance diagonal to keep it positive
    /// definite.
    pub fn reg_covar(mut self, v: f64) -> Self {
        self.reg_covar = v;
        self
    }

    /// Set the per-restart iteration cap.
    pub fn max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }

    /// Set the number of independent restarts (best `lower_bound_` wins).
    pub fn n_init(mut self, v: usize) -> Self {
        self.n_init = v;
        self
    }

    /// Set `init_params` (`"kmeans"` / `"k-means++"` / `"random"` /
    /// `"random_from_data"`).
    pub fn init_params(mut self, v: impl Into<String>) -> Self {
        self.init_params = v.into();
        self
    }

    /// Set `weight_concentration_prior_type` (`"dirichlet_process"` /
    /// `"dirichlet_distribution"`) — see
    /// [`WeightConcentrationPriorType`] for what each believes.
    pub fn weight_concentration_prior_type(mut self, v: impl Into<String>) -> Self {
        self.weight_concentration_prior_type = v.into();
        self
    }

    /// Set `weight_concentration_prior` (γ). `None` means sklearn's
    /// `1 / n_components`. Lower values shrink more components away.
    pub fn weight_concentration_prior(mut self, v: Option<f64>) -> Self {
        self.weight_concentration_prior = v;
        self
    }

    /// Set `mean_precision_prior` (β₀). `None` means `1`.
    pub fn mean_precision_prior(mut self, v: Option<f64>) -> Self {
        self.mean_precision_prior = v;
        self
    }

    /// Set `mean_prior` (m₀), length `n_features`. `None` means the design's
    /// column means.
    pub fn mean_prior(mut self, v: Option<Vec<f64>>) -> Self {
        self.mean_prior = v;
        self
    }

    /// Set `degrees_of_freedom_prior` (ν₀). `None` means `n_features`; any
    /// supplied value must exceed `n_features − 1`.
    pub fn degrees_of_freedom_prior(mut self, v: Option<f64>) -> Self {
        self.degrees_of_freedom_prior = v;
        self
    }

    /// Set `covariance_prior`, flat in the `covariance_type` layout for ONE
    /// component (`d × d`, `d`, or a single scalar). `None` means the design's
    /// empirical covariance at `ddof = 1`.
    pub fn covariance_prior(mut self, v: Option<Vec<f64>>) -> Self {
        self.covariance_prior = v;
        self
    }

    /// Set the seed for the initialization RNG (sklearn's `random_state`).
    pub fn random_state(mut self, v: Option<u64>) -> Self {
        self.random_state = v;
        self
    }

    /// Set `warm_start` — resume from the previous fit's posterior instead of
    /// re-initializing.
    pub fn warm_start(mut self, v: bool) -> Self {
        self.warm_start = v;
        self
    }

    /// Set `verbose`. A library crate must not print, so a non-zero value emits
    /// `log::info!` records rather than writing to stdout.
    pub fn verbose(mut self, v: usize) -> Self {
        self.verbose = v;
        self
    }

    /// Set how many iterations separate two `verbose` records.
    pub fn verbose_interval(mut self, v: usize) -> Self {
        self.verbose_interval = v;
        self
    }

    /// Carry a previous fit's posterior in, so a `warm_start = true` build
    /// RESUMES from it (the builder-side spelling of
    /// [`BayesianGaussianMixture::into_warm_start`], for callers — like the
    /// PyO3 wrapper — that rebuild from stored hyperparameters at every `fit`).
    ///
    /// Ignored unless `warm_start` is also set.
    pub fn warm_params(mut self, v: Option<BayesianMixtureParams>) -> Self {
        self.warm_params = v;
        self
    }

    /// Build the (unfit) estimator, parsing the string hyperparameters into
    /// typed enums and rejecting every data-INDEPENDENT invalid value (D-08).
    pub fn build<F>(self) -> Result<BayesianGaussianMixture<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if self.n_components < 1 {
            return Err(BuildError::InvalidNComponents {
                estimator: EST,
                param: "n_components",
                value: self.n_components,
            });
        }
        if self.n_init < 1 {
            return Err(BuildError::InvalidNComponents {
                estimator: EST,
                param: "n_init",
                value: self.n_init,
            });
        }
        if !self.tol.is_finite() || self.tol < 0.0 {
            return Err(BuildError::InvalidTol {
                estimator: EST,
                tol: self.tol,
            });
        }
        if !self.reg_covar.is_finite() || self.reg_covar < 0.0 {
            return Err(BuildError::InvalidRegCovar {
                estimator: EST,
                reg_covar: self.reg_covar,
            });
        }
        // The two strictly-positive scalar priors are data-INDEPENDENT, so they
        // are rejected here. `degrees_of_freedom_prior` is NOT: its bound is
        // `n_features − 1`, so it waits for `fit` (the D-08 split).
        for (param, value) in [
            (
                "weight_concentration_prior",
                self.weight_concentration_prior,
            ),
            ("mean_precision_prior", self.mean_precision_prior),
        ] {
            if let Some(v) = value {
                if !v.is_finite() || v <= 0.0 {
                    return Err(BuildError::InvalidPrior {
                        estimator: EST,
                        param,
                        value: v,
                    });
                }
            }
        }
        let covariance_type =
            CovarianceType::try_from(self.covariance_type.as_str()).map_err(|()| {
                BuildError::UnknownCovarianceType {
                    value: self.covariance_type.clone(),
                }
            })?;
        let init_params = InitParams::try_from(self.init_params.as_str())?;
        let weight_concentration_prior_type =
            WeightConcentrationPriorType::try_from(self.weight_concentration_prior_type.as_str())?;
        Ok(BayesianGaussianMixture {
            device: self.device,
            device_: None,
            n_components: self.n_components,
            covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            n_init: self.n_init,
            init_params,
            weight_concentration_prior_type,
            weight_concentration_prior: self.weight_concentration_prior,
            mean_precision_prior: self.mean_precision_prior,
            mean_prior: self.mean_prior,
            degrees_of_freedom_prior: self.degrees_of_freedom_prior,
            covariance_prior: self.covariance_prior,
            random_state: self.random_state,
            warm_start: self.warm_start,
            verbose: self.verbose,
            verbose_interval: self.verbose_interval,
            warm: self.warm_params,
            params: None,
            priors: None,
            converged_: false,
            n_iter_: 0,
            lower_bound_: f64::NEG_INFINITY,
            lower_bounds_: Vec::new(),
            n_features_in_: 0,
            n_samples_: 0,
            train_labels: None,
            _float: PhantomData,
            _state: PhantomData,
        })
    }
}
