//! `GaussianMixture` (MIX-01) — EM for a `k`-component Gaussian mixture.
//!
//! The full `sklearn.mixture.GaussianMixture` surface: every hyperparameter,
//! every fitted attribute, and the `fit` / `fit_predict` / `predict` /
//! `predict_proba` / `predict_log_proba` / `score` / `score_samples` / `bic` /
//! `aic` / `sample` method set.
//!
//! ## Where the compute lives
//! This file is the ESTIMATOR — parameter validation, the `n_init` restart
//! loop, the convergence rule, the four initializations, and the fitted-state
//! bookkeeping. Every inner loop lives in
//! [`mlrs_backend::prims::gmm_host`], which is host-resident on every backend
//! and holds three structural wins over sklearn (triangular Mahalanobis, a
//! hoisted `tied` E-step, and a fused single-sweep pass A). Read that module's
//! docs for why a device arm would be the wrong shape here.
//!
//! ## Two ingresses, one fit
//! [`GaussianMixture::fit_from_host_slice`] takes the caller's own host buffer
//! and never uploads anything — the route the Python boundary takes, where the
//! design is already a numpy/Arrow block. [`Fit::fit`] takes a device-resident
//! [`DeviceArray`], reads it back once, and calls the same core. The two produce
//! bit-identical fits; the difference is purely who pays for a transfer.
//!
//! ## sklearn parity notes (the traps)
//! - **`weights_` does not sum to exactly 1.** sklearn's `nk` carries a
//!   `+10·eps` floor per component and `weights_ = nk / n_samples`, so the sum
//!   is `1 + 10·k·eps`. Reproduced exactly rather than renormalized.
//! - **`lower_bound_` lags the parameters by one M-step.** The bound reported
//!   for iteration `t` is the log-likelihood under iteration `t−1`'s
//!   parameters, because sklearn computes it from the E-step that PRECEDES the
//!   M-step. A "fix" here would silently shift the convergence test.
//! - **`n_iter_` counts the LAST restart that improved**, not the total.
//! - **Convergence is `|Δ lower_bound| < tol`, not a parameter norm.** With
//!   `tol = 0` the loop runs to `max_iter` (sklearn's `Interval(..., closed
//!   ="left")` admits `0`), and `converged_` stays `false`.
//! - **`precisions_init` uses the OTHER Cholesky convention** — see
//!   [`invert_spd`] for why that cannot just be copied through.
//! - **The component ORDER is init-dependent**, exactly as in sklearn. Two runs
//!   that reach the same optimum can label the components differently, so
//!   oracle comparisons match components before comparing them.
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
    cholesky_lower_blocks, invert_spd, logsumexp_rows, precisions_cholesky,
    precisions_from_cholesky, weighted_log_prob, CovarianceType, GmmHost, IllConditioned,
};
use mlrs_backend::prims::rng::SplitMix64;
use mlrs_backend::runtime::ActiveRuntime;

use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use super::mixture_persist::{
    as_f64, as_i64, read_device_arm, read_mixture_params, read_opt_vec, shape_1d,
    write_mixture_params, write_opt_vec, AlignedBytes, LoadModel, MixtureFile, MixtureWriter,
    PersistError, SaveModel, TensorRef, FITTED_NAMES, LOWER_BOUNDS_NAME, TRAIN_LABELS_NAME,
    WARM_NAMES,
};
use crate::error::{AlgoError, BuildError};
use crate::typestate::{
    validate_geometry, Fit, Fitted, PredictLabels, PredictLogProba, PredictProba, ScoreSamples,
    State, Unfit,
};

/// The estimator name carried by every typed error this file raises.
const EST: &str = "gaussian_mixture";

/// Seed used when the caller leaves `random_state = None`. sklearn draws from
/// the global numpy state there, which is not reproducible from Rust; a FIXED
/// default keeps `fit` deterministic (RESEARCH Pitfall 7 — never a Rust-side
/// entropy source), and a caller who wants variation passes `random_state`.
const DEFAULT_SEED: u64 = 0x5EED_6D31;

/// sklearn's `init_params`: how the FIRST responsibilities are produced, before
/// any E-step has run.
///
/// All four routes end at the same place — an `n × k` responsibility matrix fed
/// straight into an M-step — and differ only in how informative that matrix is.
/// The two `k-means`-family routes look at the data; the two random ones do not,
/// and converge to a worse optimum more often (which is what `n_init` exists
/// for).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitParams {
    /// Full Lloyd k-means (`n_init=1`), one-hot from its hard labels. sklearn's
    /// default, and the most expensive of the four.
    KMeans,
    /// k-means++ SEEDING only — no Lloyd iterations. `resp` is one-hot at the
    /// `k` chosen seed rows and ZERO everywhere else, so the first M-step sees
    /// `nk ≈ 1` per component.
    KMeansPlusPlus,
    /// `resp = uniform(n, k)`, row-normalized. Every component starts at
    /// roughly the global mean, which is why this route needs the most
    /// iterations.
    Random,
    /// One-hot at `k` DISTINCT uniformly-drawn rows — the random counterpart of
    /// `k-means++`.
    RandomFromData,
}

impl InitParams {
    /// The sklearn string spelling.
    pub fn name(self) -> &'static str {
        match self {
            InitParams::KMeans => "kmeans",
            InitParams::KMeansPlusPlus => "k-means++",
            InitParams::Random => "random",
            InitParams::RandomFromData => "random_from_data",
        }
    }
}

impl TryFrom<&str> for InitParams {
    type Error = BuildError;

    fn try_from(value: &str) -> Result<Self, BuildError> {
        match value {
            "kmeans" => Ok(InitParams::KMeans),
            "k-means++" => Ok(InitParams::KMeansPlusPlus),
            "random" => Ok(InitParams::Random),
            "random_from_data" => Ok(InitParams::RandomFromData),
            other => Err(BuildError::UnknownInit {
                value: other.to_string(),
            }),
        }
    }
}

/// One fitted parameter block: what sklearn's `_get_parameters` /
/// `_set_parameters` move around between `n_init` restarts.
///
/// Always `f64` regardless of the estimator's `F` — the whole EM loop is (see
/// the `gmm_host` module docs); `F` appears only at the accessors.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureParams {
    /// `weights_`, length `k`.
    pub weights: Vec<f64>,
    /// `means_`, `k × d` row-major.
    pub means: Vec<f64>,
    /// `covariances_`, in the `covariance_type` layout.
    pub covariances: Vec<f64>,
    /// `precisions_cholesky_`, same layout (upper-triangular `inv(L)ᵀ` blocks
    /// for `full`/`tied`).
    pub precisions_cholesky: Vec<f64>,
}

/// EM fit of a `k`-component Gaussian mixture (MIX-01).
///
/// `S` is the [`State`] typestate marker: an `Unfit` value exposes only the fit
/// entry points, and every fitted attribute / scoring method lives on the
/// `Fitted` sibling, so a `predict`-before-`fit` is a compile error.
pub struct GaussianMixture<F, S = Unfit>
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
    weights_init: Option<Vec<f64>>,
    means_init: Option<Vec<f64>>,
    precisions_init: Option<Vec<f64>>,
    random_state: Option<u64>,
    warm_start: bool,
    verbose: usize,
    verbose_interval: usize,

    /// Parameters carried in from a previous `fit` when `warm_start = true`
    /// (sklearn keeps them on the estimator; the consuming typestate `fit`
    /// cannot, so they ride here — see [`GaussianMixture::into_warm_start`]).
    warm: Option<MixtureParams>,
    /// Where to run the EM loop (DEVICE-PARAM-01). `Auto` keeps the
    /// `gmm_device_applicable` gate — backend, `f64` capability, `f64`
    /// transcendentals, the `MLRS_GMM_DEVICE` flag, then a size floor.
    device: Device,

    // ---- fitted state (`None` / zero while `Unfit`) ----
    params: Option<MixtureParams>,
    /// The EM engine that ACTUALLY ran (`"cpu"` / `"gpu"`), `None` until `fit`.
    device_: Option<&'static str>,
    converged_: bool,
    n_iter_: usize,
    lower_bound_: f64,
    /// sklearn's `lower_bounds_`: the per-iteration bound trace of the WINNING
    /// restart (length `n_iter_`), not of the last one run.
    lower_bounds_: Vec<f64>,
    n_features_in_: usize,
    n_samples_: usize,
    /// The training-set assignment from `fit`'s terminal E-step, so a caller
    /// asking for `fit_predict`'s labels does not pay for a second scoring pass.
    train_labels: Option<Vec<i32>>,

    _float: PhantomData<F>,
    _state: PhantomData<S>,
}

impl<F> Default for GaussianMixture<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Hand-written rather than derived: a derive would demand `F: Debug` and
/// `S: Debug` (neither marker is), and it would dump the `n × k` fitted buffers
/// into any `{:?}`. This prints the HYPERPARAMETERS plus a fitted-state summary,
/// which is what a `Result::expect_err` or an assertion message actually needs.
impl<F, S> std::fmt::Debug for GaussianMixture<F, S>
where
    F: Float + CubeElement + Pod,
    S: State,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaussianMixture")
            .field("n_components", &self.n_components)
            .field("covariance_type", &self.covariance_type.name())
            .field("tol", &self.tol)
            .field("reg_covar", &self.reg_covar)
            .field("max_iter", &self.max_iter)
            .field("n_init", &self.n_init)
            .field("init_params", &self.init_params.name())
            .field("random_state", &self.random_state)
            .field("warm_start", &self.warm_start)
            .field("fitted", &self.params.is_some())
            .field("n_iter_", &self.n_iter_)
            .field("converged_", &self.converged_)
            .field("lower_bound_", &self.lower_bound_)
            .finish()
    }
}

impl<F> GaussianMixture<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// A new estimator at sklearn's defaults (`n_components=1`,
    /// `covariance_type='full'`, `tol=1e-3`, `reg_covar=1e-6`, `max_iter=100`,
    /// `n_init=1`, `init_params='kmeans'`, no injected parameters,
    /// `warm_start=False`, `verbose=0`, `verbose_interval=10`).
    ///
    /// SINGLE source of truth for the defaults: [`GaussianMixtureBuilder`]'s
    /// `Default` re-derives them through [`GaussianMixture::into_builder`].
    pub fn new() -> Self {
        Self {
            n_components: 1,
            covariance_type: CovarianceType::Full,
            tol: 1e-3,
            reg_covar: 1e-6,
            max_iter: 100,
            n_init: 1,
            init_params: InitParams::KMeans,
            weights_init: None,
            means_init: None,
            precisions_init: None,
            random_state: None,
            warm_start: false,
            verbose: 0,
            verbose_interval: 10,
            warm: None,
            device: Device::Auto,
            params: None,
            device_: None,
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
    pub fn builder() -> GaussianMixtureBuilder {
        GaussianMixtureBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`GaussianMixtureBuilder::default`] to re-derive
    /// the defaults from [`GaussianMixture::new`] (BLDR-01, single source).
    pub fn into_builder(self) -> GaussianMixtureBuilder {
        GaussianMixtureBuilder {
            device: self.device,
            n_components: self.n_components,
            covariance_type: self.covariance_type.name().to_string(),
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            n_init: self.n_init,
            init_params: self.init_params.name().to_string(),
            weights_init: self.weights_init,
            means_init: self.means_init,
            precisions_init: self.precisions_init,
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
            && self.weights_init == other.weights_init
            && self.means_init == other.means_init
            && self.precisions_init == other.precisions_init
            && self.random_state == other.random_state
            && self.warm_start == other.warm_start
            && self.verbose == other.verbose
            && self.verbose_interval == other.verbose_interval
            && self.device == other.device
    }

    /// Should `fit` take [`GaussianMixture::fit_from_host_slice`] rather than
    /// uploading and going through [`Fit::fit`]?
    ///
    /// ALWAYS `true`. `fit_from_host_slice` is a strict superset of what
    /// `Fit::fit` can reach: it never uploads `x` itself (it stays a host
    /// slice all the way to [`GmmHost`], or is uploaded exactly once by
    /// [`GmmDevice::new`](mlrs_backend::prims::gmm_device::GmmDevice::new) when
    /// [`GaussianMixture::device_fit_applicable`] takes the device EM engine),
    /// whereas `Fit::fit` always pays one upload up front to obtain the
    /// `DeviceArray` it requires — so there is no shape where `Fit::fit` wins.
    /// The predicate exists anyway because the two entry points take DIFFERENT
    /// operand types and a caller has to choose before ingress.
    pub fn host_fit_applicable(&self, _shape: (usize, usize)) -> bool {
        true
    }

    /// Does the DEVICE EM engine
    /// ([`GmmDevice`](mlrs_backend::prims::gmm_device::GmmDevice)) apply to
    /// this `(n_samples, n_features)` shape, given this estimator's
    /// `n_components`?
    ///
    /// Delegates entirely to
    /// [`gmm_device_applicable`](mlrs_backend::prims::gmm_device::gmm_device_applicable) —
    /// see that function's docs for the gate order (backend, `f64` capability,
    /// `f64` transcendentals, the `MLRS_GMM_DEVICE` override, then a size
    /// floor). `fit_core` consults this ONCE per fit, before the `n_init`
    /// restart loop, since the design and its geometry do not change across
    /// restarts.
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
    /// `x` is the `n × d` row-major design borrowed from host memory (at the
    /// Python boundary, the caller's own numpy/Arrow block). Produces exactly
    /// the fit [`Fit::fit`] does. `pool` is consulted ONLY when
    /// [`GaussianMixture::device_fit_applicable`] holds for this shape — the
    /// common case (small/medium fits, or any cpu/wgpu-at-f64 backend) never
    /// touches it and never uploads `x`.
    pub fn fit_from_host_slice(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<GaussianMixture<F, Fitted>, AlgoError> {
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

    /// The shared body of BOTH ingresses (the design already widened to `f64`).
    ///
    /// Mirrors `BaseMixture.fit_predict` structurally: `n_init` restarts, each
    /// an initialization plus up to `max_iter` E/M iterations, keeping the
    /// restart with the highest `lower_bound_`.
    ///
    /// [`GaussianMixture::device_fit_applicable`] is consulted ONCE, before the
    /// restart loop (the design and its geometry are the same for every
    /// restart): when it holds, `x` is uploaded exactly once into a
    /// [`GmmDevice`] that stays alive for every restart's whole iteration loop,
    /// and every `e_step`/`covariances` call in the loop below routes through
    /// it instead of the host `host` engine. Initialization
    /// ([`GaussianMixture::initialize`]) ALWAYS runs on `host` regardless — see
    /// that method and the `gmm_host`/`gmm_device` module docs for why.
    fn fit_core(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[f64],
        shape: (usize, usize),
    ) -> Result<GaussianMixture<F, Fitted>, AlgoError> {
        let (n, d) = shape;
        let k = self.n_components;
        // sklearn: "Expected n_samples >= n_components". A data-DEPENDENT
        // check, so it belongs at `fit`, not at `build()` (the D-08 split).
        if k > n {
            return Err(AlgoError::InvalidK {
                estimator: EST,
                k,
                n_samples: n,
            });
        }
        self.validate_injected(d)?;

        let ct = self.covariance_type;
        // `host` always exists: `initialize` (the two k-means routes + the two
        // random draws) is a one-time, small-relative-to-`max_iter` cost that
        // stays host-resident regardless of which engine runs the E/M loop
        // (`gmm_device` module docs).
        let mut host = GmmHost::new(x, n, d, k, ct, self.reg_covar);
        let mut rng = SplitMix64::new(self.random_state.unwrap_or(DEFAULT_SEED));

        // The device EM engine, built ONCE (not per restart) when applicable —
        // `x` is uploaded exactly once and `resp` stays device-resident for
        // every restart's whole `max_iter` loop.
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

        // `warm_start` with a previous fit skips initialization entirely and
        // resumes from the carried parameters, so a second `fit` continues the
        // same ascent — sklearn's `do_init = not (warm_start and converged_)`.
        let do_init = !(self.warm_start && self.warm.is_some());
        let n_init = if do_init { self.n_init } else { 1 };

        let mut best: Option<MixtureParams> = None;
        let mut best_bound = f64::NEG_INFINITY;
        let mut best_n_iter = 0usize;
        let mut best_converged = false;
        // sklearn keeps the winning restart's per-iteration bound trace, NOT
        // the last restart's — `best_trace` is swapped in on the same condition
        // as `best_params`.
        let mut best_trace: Vec<f64> = Vec::new();

        for _restart in 0..n_init {
            let (mut cur, mut lower_bound) = if do_init {
                (self.initialize(&mut host, &mut rng, n, d, k)?, f64::NEG_INFINITY)
            } else {
                (
                    self.warm.clone().expect("warm_start branch requires carried params"),
                    self.lower_bound_,
                )
            };

            let mut converged = false;
            let mut iters = 0usize;
            let mut trace: Vec<f64> = Vec::with_capacity(self.max_iter);
            for it in 1..=self.max_iter {
                let prev = lower_bound;
                // ONE fused sweep: responsibilities + `nk` + `means` (win #3),
                // on whichever engine this fit is using.
                let (bound, nk, means) = if let Some(dev) = device.as_mut() {
                    dev.e_step(pool, &cur.weights, &cur.means, &cur.precisions_cholesky)
                        .map_err(AlgoError::Prim)?
                } else {
                    host.e_step(&cur.weights, &cur.means, &cur.precisions_cholesky)
                };
                let covariances = if let Some(dev) = device.as_mut() {
                    dev.covariances(pool, &nk, &means).map_err(AlgoError::Prim)?
                } else {
                    host.covariances(&nk, &means)
                };
                let prec_chol =
                    precisions_cholesky(&covariances, k, d, ct).map_err(ill_conditioned)?;
                let inv_n = 1.0 / n as f64;
                cur = MixtureParams {
                    weights: nk.iter().map(|v| v * inv_n).collect(),
                    means,
                    covariances,
                    precisions_cholesky: prec_chol,
                };
                lower_bound = bound;
                trace.push(bound);
                iters = it;
                if self.verbose > 0 && it % self.verbose_interval.max(1) == 0 {
                    log::info!(
                        "gaussian_mixture: iteration {it}, lower_bound = {lower_bound:e}"
                    );
                }
                if (lower_bound - prev).abs() < self.tol {
                    converged = true;
                    break;
                }
            }
            // `max_iter = 0` needs no special case: the loop never runs, so
            // `iters` stays `0` and `lower_bound` stays at its pre-loop value —
            // which is exactly sklearn's "report the initialization" behaviour.
            if best.is_none() || lower_bound > best_bound {
                best_bound = lower_bound;
                best = Some(cur);
                best_n_iter = iters;
                best_converged = converged;
                best_trace = trace;
            }
        }

        let params = best.expect("n_init >= 1 guarantees at least one restart");

        // sklearn's terminal E-step: `fit_predict` runs one more so the returned
        // labels come from the FINAL parameters rather than the last iteration's.
        // The bound it produces is deliberately NOT stored (sklearn keeps
        // `max_lower_bound`), which is what makes `lower_bound_` lag by one
        // M-step — see the module docs.
        let labels: Vec<i32> = if let Some(dev) = device.as_mut() {
            let (_final_bound, _, _) = dev
                .e_step(pool, &params.weights, &params.means, &params.precisions_cholesky)
                .map_err(AlgoError::Prim)?;
            argmax_rows(&dev.resp_to_host(pool), n, k)
        } else {
            let (_final_bound, _, _) = host.e_step(
                &params.weights,
                &params.means,
                &params.precisions_cholesky,
            );
            argmax_rows(host.resp(), n, k)
        };
        if let Some(dev) = device {
            dev.release_into(pool);
        }

        Ok(GaussianMixture {
            device: self.device,
            device_: Some(device_arm),
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            n_init: self.n_init,
            init_params: self.init_params,
            weights_init: self.weights_init,
            means_init: self.means_init,
            precisions_init: self.precisions_init,
            random_state: self.random_state,
            warm_start: self.warm_start,
            verbose: self.verbose,
            verbose_interval: self.verbose_interval,
            warm: Some(params.clone()),
            params: Some(params),
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

    /// Data-DEPENDENT validation of the three injected parameter buffers, which
    /// can only be length-checked once `n_features` is known (the D-08 split —
    /// their per-entry validity is a `build()` concern, their SHAPE is a `fit`
    /// one).
    fn validate_injected(&self, d: usize) -> Result<(), AlgoError> {
        let k = self.n_components;
        let check = |v: &Option<Vec<f64>>, want: usize, what: &'static str| -> Result<(), AlgoError> {
            if let Some(v) = v {
                if v.len() != want {
                    return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                        operand: what,
                        rows: k,
                        cols: want / k.max(1),
                        len: v.len(),
                    }));
                }
            }
            Ok(())
        };
        check(&self.weights_init, k, "weights_init")?;
        // sklearn's `_check_weights`: every entry in `[0, 1]`, and the vector
        // sums to one within `1e-6` (its `np.allclose` default `atol`).
        if let Some(w) = &self.weights_init {
            if w.iter().any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0) {
                return Err(AlgoError::InvalidMixtureInit {
                    estimator: EST,
                    param: "weights_init",
                    reason: "every weight must be finite and in [0, 1]".to_string(),
                });
            }
            let sum: f64 = w.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(AlgoError::InvalidMixtureInit {
                    estimator: EST,
                    param: "weights_init",
                    reason: format!("the weights must sum to 1, got {sum}"),
                });
            }
        }
        if let Some(m) = &self.means_init {
            if m.iter().any(|v| !v.is_finite()) {
                return Err(AlgoError::InvalidMixtureInit {
                    estimator: EST,
                    param: "means_init",
                    reason: "every entry must be finite".to_string(),
                });
            }
        }
        check(&self.means_init, k * d, "means_init")?;
        check(
            &self.precisions_init,
            self.covariance_type.param_len(k, d),
            "precisions_init",
        )?;
        Ok(())
    }

    /// sklearn's `_initialize_parameters` + `_initialize`: build the first
    /// responsibilities from `init_params`, run ONE M-step off them, then let
    /// any injected `weights_init` / `means_init` / `precisions_init` override
    /// the corresponding block.
    fn initialize(
        &self,
        host: &mut GmmHost<'_>,
        rng: &mut SplitMix64,
        n: usize,
        d: usize,
        k: usize,
    ) -> Result<MixtureParams, AlgoError> {
        let ct = self.covariance_type;
        let resp = initial_responsibilities(self.init_params, host, rng, n, k);

        host.set_resp(&resp);
        let (nk, means) = host.nk_and_means_from_resp();
        let covariances = host.covariances(&nk, &means);
        let inv_n = 1.0 / n as f64;

        let weights = match &self.weights_init {
            Some(w) => w.clone(),
            None => nk.iter().map(|v| v * inv_n).collect(),
        };
        let means = match &self.means_init {
            Some(m) => m.clone(),
            None => means,
        };
        let (covariances, precisions_cholesky) = match &self.precisions_init {
            // sklearn stores `cholesky(precisions_init, lower=True)` here, which
            // is the OTHER factor of the same precision. Round-tripping through
            // the covariance keeps this module's single upper-triangular
            // convention and gives numerically identical distances / log-dets.
            Some(p) => {
                let cov = invert_spd(p, k, d, ct).map_err(ill_conditioned)?;
                let chol = precisions_cholesky(&cov, k, d, ct).map_err(ill_conditioned)?;
                (cov, chol)
            }
            None => {
                let chol =
                    precisions_cholesky(&covariances, k, d, ct).map_err(ill_conditioned)?;
                (covariances, chol)
            }
        };

        Ok(MixtureParams {
            weights,
            means,
            covariances,
            precisions_cholesky,
        })
    }
}

/// The `estimator` discriminator written into every `GaussianMixture` file.
///
/// Load-bearing rather than decorative: a `BayesianGaussianMixture` file holds
/// `means_` and `covariances_` of the same shapes and dtypes under the same
/// names. Its density is parameterized by a POSTERIOR the frequentist mixture
/// has no notion of, so a cross-load would produce a model that scores every
/// sample differently with nothing structural to signal it.
const PERSIST_TAG: &str = "gaussian_mixture";

impl<F> SaveModel for GaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Write the fitted mixture to `path` as a safetensors file.
    ///
    /// | name | dtype | shape |
    /// |---|---|---|
    /// | `weights_` | `F64` | `[n_components]` |
    /// | `means_` | `F64` | `[n_components, n_features]` |
    /// | `covariances_` / `precisions_cholesky_` | `F64` | flat, `param_len(k, d)` |
    /// | `lower_bounds_` | `F64` | `[n_iter]` |
    /// | `train_labels` | `I64` | `[n_samples]`, optional |
    /// | `warm_*` | `F64` | the `warm_start` resumption block, optional |
    /// | `param:weights_init` / `_means_init` / `_precisions_init` | `F64` | optional |
    /// | `converged_` / `n_iter_` / `lower_bound_` / `device_` / … | `__metadata__` | — |
    ///
    /// Everything is `F64` regardless of the estimator's `F` — see
    /// [`mixture_persist`](super::mixture_persist) for why that is a model
    /// property here rather than a storage choice.
    ///
    /// `precisions_cholesky_` is stored alongside `covariances_` rather than
    /// re-derived, and `lower_bounds_` is stored in full rather than reduced to
    /// its last value: the trace is what a caller inspects to see whether the EM
    /// loop plateaued or was cut off by `max_iter`, and it is not recoverable
    /// from anything else in the file.
    ///
    /// `pool` is unused: the EM engine is host-resident on EVERY backend
    /// (MIX-01), so there is nothing device-resident to read back. The parameter
    /// is present because [`SaveModel`] is one signature for every estimator.
    fn save(&self, _pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError> {
        let params = self.params.as_ref().ok_or(PersistError::MissingState {
            estimator: PERSIST_TAG,
            field: "params",
        })?;
        // The training labels widen `i32 → i64`; bound BEFORE the writer, which
        // borrows every payload.
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
        w.scalar_opt_u64("param:random_state", self.random_state);
        w.scalar_bool("param:warm_start", self.warm_start);
        w.scalar_usize("param:verbose", self.verbose);
        w.scalar_usize("param:verbose_interval", self.verbose_interval);
        w.scalar_str("param:device", self.device.name());

        w.scalar_bool("converged_", self.converged_);
        w.scalar_usize("n_iter_", self.n_iter_);
        w.scalar_f64("lower_bound_", self.lower_bound_);
        w.scalar_usize("n_features_in_", self.n_features_in_);
        w.scalar_usize("n_samples_", self.n_samples_);
        if let Some(arm) = self.device_ {
            w.scalar_str("device_", arm);
        }

        write_mixture_params(
            &mut w,
            &FITTED_NAMES,
            &params.weights,
            &params.means,
            &params.covariances,
            &params.precisions_cholesky,
            self.covariance_type,
        )?;
        // The `warm_start` resumption block, when the model carries one. Storing
        // it is what makes a reloaded model a valid continuation rather than
        // merely a scorer — the same distinction `IncrementalPCA` draws.
        if let Some(warm) = self.warm.as_ref() {
            write_mixture_params(
                &mut w,
                &WARM_NAMES,
                &warm.weights,
                &warm.means,
                &warm.covariances,
                &warm.precisions_cholesky,
                self.covariance_type,
            )?;
        }
        write_opt_vec(&mut w, "param:weights_init", self.weights_init.as_ref())?;
        write_opt_vec(&mut w, "param:means_init", self.means_init.as_ref())?;
        write_opt_vec(&mut w, "param:precisions_init", self.precisions_init.as_ref())?;
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

impl<F> LoadModel for GaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Read the mixture back from `path`.
    ///
    /// `covariance_type` is read FIRST, because it determines the flat length
    /// both parameter blocks must have — the file is untrusted input
    /// (T-04-01-01), and a `tied` model whose `covariances_` is `full`-length
    /// would otherwise index past the end of the shared matrix on the first
    /// `score_samples`.
    ///
    /// `n_components` and `n_features_in_` are recovered from `weights_` and
    /// `means_`, and the `param:n_components` scalar is cross-checked against
    /// them rather than trusted.
    fn load(
        _pool: &mut BufferPool<ActiveRuntime>,
        path: &Path,
    ) -> Result<GaussianMixture<F, Fitted>, PersistError> {
        let raw = AlignedBytes::read(path)?;
        let file = MixtureFile::parse(&raw, PERSIST_TAG)?;

        let covariance_type = CovarianceType::try_from(file.scalar_str("param:covariance_type")?)
            .map_err(|_| PersistError::BadMetadata {
            key: "param:covariance_type",
        })?;
        let fitted = read_mixture_params(&file, &FITTED_NAMES, covariance_type)?;
        if file.scalar_usize("param:n_components")? != fitted.n_components {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "'param:n_components' is {}, but 'weights_' holds {} components",
                    file.scalar_usize("param:n_components")?,
                    fitted.n_components
                ),
            });
        }

        // The warm block is present only for a model saved mid-`warm_start`, and
        // is read under the same geometry rules as the fitted one.
        let warm = if file.tensor_opt(WARM_NAMES.weights).is_some() {
            let w = read_mixture_params(&file, &WARM_NAMES, covariance_type)?;
            if w.n_components != fitted.n_components || w.n_features != fitted.n_features {
                return Err(PersistError::InconsistentGeometry {
                    reason: format!(
                        "the warm-start block is [{}, {}] but the fitted block is [{}, {}]",
                        w.n_components, w.n_features, fitted.n_components, fitted.n_features
                    ),
                });
            }
            Some(MixtureParams {
                weights: w.weights,
                means: w.means,
                covariances: w.covariances,
                precisions_cholesky: w.precisions_cholesky,
            })
        } else {
            None
        };

        let lower_bounds_v = file.tensor(LOWER_BOUNDS_NAME)?;
        shape_1d(&lower_bounds_v, LOWER_BOUNDS_NAME)?;
        let lower_bounds_ = as_f64(&lower_bounds_v, LOWER_BOUNDS_NAME)?.into_owned();

        let n_samples_ = file.scalar_usize("n_samples_")?;
        let train_labels = match file.tensor_opt(TRAIN_LABELS_NAME) {
            None => None,
            Some(view) => {
                let len = shape_1d(&view, TRAIN_LABELS_NAME)?;
                if len != n_samples_ {
                    return Err(PersistError::InconsistentGeometry {
                        reason: format!(
                            "tensor '{TRAIN_LABELS_NAME}' holds {len} entries, but the model \
                             was fitted on {n_samples_} samples"
                        ),
                    });
                }
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

        Ok(GaussianMixture {
            n_components: fitted.n_components,
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
            weights_init: read_opt_vec(&file, "param:weights_init")?,
            means_init: read_opt_vec(&file, "param:means_init")?,
            precisions_init: read_opt_vec(&file, "param:precisions_init")?,
            random_state: file.scalar_opt_u64("param:random_state")?,
            warm_start: file.scalar_bool("param:warm_start")?,
            verbose: file.scalar_usize("param:verbose")?,
            verbose_interval: file.scalar_usize("param:verbose_interval")?,
            warm,
            device: Device::from_name(file.scalar_str("param:device")?).ok_or(
                PersistError::BadMetadata {
                    key: "param:device",
                },
            )?,
            params: Some(MixtureParams {
                weights: fitted.weights,
                means: fitted.means,
                covariances: fitted.covariances,
                precisions_cholesky: fitted.precisions_cholesky,
            }),
            device_: read_device_arm(&file, "device_")?,
            converged_: file.scalar_bool("converged_")?,
            n_iter_: file.scalar_usize("n_iter_")?,
            lower_bound_: file.scalar_f64("lower_bound_")?,
            lower_bounds_,
            n_features_in_: file.scalar_usize("n_features_in_")?,
            n_samples_,
            train_labels,
            _float: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for GaussianMixture<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = GaussianMixture<F, Fitted>;

    /// Device ingress: validate the geometry, read the design back ONCE, and run
    /// the same host core [`GaussianMixture::fit_from_host_slice`] runs.
    ///
    /// `y` is ignored — a mixture model is unsupervised.
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

impl<F> GaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn p(&self) -> &MixtureParams {
        self.params
            .as_ref()
            .expect("a Fitted GaussianMixture always carries parameters")
    }

    /// `weights_` — the mixing proportions, length `n_components`.
    ///
    /// Sums to `1 + 10·k·eps`, not to `1`; see the module docs.
    pub fn weights(&self) -> Vec<F> {
        self.p().weights.iter().map(|&v| f64_to_host(v)).collect()
    }

    /// `means_` — `n_components × n_features`, row-major.
    pub fn means(&self) -> Vec<F> {
        self.p().means.iter().map(|&v| f64_to_host(v)).collect()
    }

    /// `covariances_`, flat in the [`CovarianceType::param_shape`] layout for
    /// this estimator's `covariance_type`.
    pub fn covariances(&self) -> Vec<F> {
        self.p()
            .covariances
            .iter()
            .map(|&v| f64_to_host(v))
            .collect()
    }

    /// `precisions_cholesky_`, same layout as `covariances_`. The `full`/`tied`
    /// blocks are UPPER triangular (`inv(L)ᵀ`).
    pub fn precisions_cholesky(&self) -> Vec<F> {
        self.p()
            .precisions_cholesky
            .iter()
            .map(|&v| f64_to_host(v))
            .collect()
    }

    /// `precisions_` — `precisions_cholesky_ · precisions_choleskyᵀ`, same
    /// layout.
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

    /// The `f64` parameter block, for callers that must not lose precision (the
    /// oracle tests and the `warm_start` hand-off).
    pub fn params_f64(&self) -> &MixtureParams {
        self.p()
    }

    /// The shape of `covariances_` / `precisions_` for this parameterization.
    pub fn covariance_shape(&self) -> Vec<usize> {
        self.covariance_type
            .param_shape(self.n_components, self.n_features_in_)
    }

    /// `converged_` — did the LAST-BEST restart hit `|Δ lower_bound| < tol`?
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

    /// `n_iter_` — EM iterations used by the best restart.
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// `lower_bound_` — the best restart's mean per-sample log-likelihood.
    pub fn lower_bound(&self) -> f64 {
        self.lower_bound_
    }

    /// `lower_bounds_` — the per-iteration bound trace of the WINNING restart
    /// (length `n_iter_`).
    ///
    /// Note whose trace this is: with `n_init > 1` sklearn keeps the trace of
    /// the restart that produced `lower_bound_`, not of the last restart run,
    /// so a caller plotting convergence sees the ascent that was actually
    /// adopted. Empty when `max_iter == 0`.
    pub fn lower_bounds(&self) -> &[f64] {
        &self.lower_bounds_
    }

    /// `n_features_in_`.
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }

    /// Training-set row count, the `n` `bic` / `aic` are reported against.
    pub fn n_samples(&self) -> usize {
        self.n_samples_
    }

    /// The `covariance_type` this estimator was built with.
    pub fn covariance_type(&self) -> CovarianceType {
        self.covariance_type
    }

    /// The training-set component assignment computed by `fit`'s terminal
    /// E-step — sklearn's `fit_predict(X)` return value, for free.
    pub fn labels(&self) -> &[i32] {
        self.train_labels
            .as_deref()
            .expect("a Fitted GaussianMixture always carries its training labels")
    }

    /// Turn a fitted estimator back into an `Unfit` one that RESUMES from these
    /// parameters — the typestate spelling of sklearn's `warm_start=True`.
    ///
    /// sklearn keeps the fitted attributes on the object and checks for them at
    /// the next `fit`; a consuming typestate `fit` cannot, so the carried block
    /// travels explicitly. The returned estimator skips initialization and
    /// `n_init` entirely, exactly as sklearn's `do_init = False` branch does.
    pub fn into_warm_start(self) -> GaussianMixture<F, Unfit> {
        GaussianMixture {
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            n_init: self.n_init,
            init_params: self.init_params,
            weights_init: self.weights_init,
            means_init: self.means_init,
            precisions_init: self.precisions_init,
            random_state: self.random_state,
            warm_start: self.warm_start,
            verbose: self.verbose,
            verbose_interval: self.verbose_interval,
            warm: self.warm,
            device: self.device,
            params: None,
            device_: None,
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

    /// `log π_k + log N(x | μ_k, Σ_k)` for every row of a HOST design — the one
    /// quantity `predict` / `predict_proba` / `score_samples` all reduce.
    fn weighted_log_prob_host(&self, x: &[f64], n: usize) -> Vec<f64> {
        let p = self.p();
        weighted_log_prob(
            x,
            n,
            self.n_features_in_,
            self.n_components,
            self.covariance_type,
            &p.weights,
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
    pub fn predict_proba_host(
        &self,
        x: &[F],
        shape: (usize, usize),
    ) -> Result<Vec<F>, AlgoError> {
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

    /// The `n × k` LOG responsibilities in full `f64` — the shared body of
    /// `predict_proba` / `predict_log_proba`, kept un-narrowed so an `f32`
    /// estimator does not round twice on the way to a probability.
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

    /// sklearn's `_n_parameters()`: free parameters in the fitted model —
    /// `k − 1` mixing weights, `k·d` means, and the covariance count for this
    /// parameterization.
    pub fn n_parameters(&self) -> usize {
        let (k, d) = (self.n_components, self.n_features_in_);
        let cov = match self.covariance_type {
            CovarianceType::Full => k * d * (d + 1) / 2,
            CovarianceType::Diag => k * d,
            CovarianceType::Tied => d * (d + 1) / 2,
            CovarianceType::Spherical => k,
        };
        cov + k * d + k - 1
    }

    /// `bic(X)` — the Bayesian information criterion (LOWER is better).
    pub fn bic(&self, x: &[F], shape: (usize, usize)) -> Result<f64, AlgoError> {
        let score = self.score_host(x, shape)?;
        let n = shape.0 as f64;
        Ok(-2.0 * score * n + self.n_parameters() as f64 * n.ln())
    }

    /// `aic(X)` — the Akaike information criterion (LOWER is better).
    pub fn aic(&self, x: &[F], shape: (usize, usize)) -> Result<f64, AlgoError> {
        let score = self.score_host(x, shape)?;
        Ok(-2.0 * score * shape.0 as f64 + 2.0 * self.n_parameters() as f64)
    }

    /// `sample(n_samples)` — draw from the fitted mixture, returning the
    /// `n_samples × n_features` design and the component index of each row.
    ///
    /// sklearn draws the per-component counts from a multinomial and then emits
    /// each component's rows CONTIGUOUSLY (its `y` is sorted), which is
    /// reproduced here. `seed` replaces sklearn's `random_state`, which is not
    /// reproducible from Rust.
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
        let mut rng = SplitMix64::new(seed);

        // Multinomial counts by inverse-CDF over the (near-)normalized weights.
        let total: f64 = p.weights.iter().sum();
        let mut counts = vec![0usize; k];
        for _ in 0..n_samples {
            let t = rng.next_f64() * total;
            let mut acc = 0.0;
            let mut pick = k - 1;
            for (c, &w) in p.weights.iter().enumerate() {
                acc += w;
                if acc >= t {
                    pick = c;
                    break;
                }
            }
            counts[pick] += 1;
        }

        // A component's sampler needs the covariance's LOWER Cholesky factor,
        // which is the inverse-transpose of the stored precision factor; deriving
        // it from `covariances_` directly is simpler and equally exact.
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

impl<F> PredictLabels<F> for GaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Device-ingress `predict`: read the query back once, score on the host,
    /// and return the argmax component as a fresh device buffer.
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

impl<F> PredictProba<F> for GaussianMixture<F, Fitted>
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

impl<F> PredictLogProba<F> for GaussianMixture<F, Fitted>
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

impl<F> ScoreSamples<F> for GaussianMixture<F, Fitted>
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
// Builder
// ---------------------------------------------------------------------------

/// Builder for [`GaussianMixture`] (Phase-16 A5 convention: `f64`-typed setters,
/// narrowed to `F` inside `build::<F>()`; the string-valued hyperparameters stay
/// strings until `build`, where they become typed enums or a [`BuildError`]).
#[derive(Debug, Clone)]
pub struct GaussianMixtureBuilder {
    device: Device,
    n_components: usize,
    covariance_type: String,
    tol: f64,
    reg_covar: f64,
    max_iter: usize,
    n_init: usize,
    init_params: String,
    weights_init: Option<Vec<f64>>,
    means_init: Option<Vec<f64>>,
    precisions_init: Option<Vec<f64>>,
    random_state: Option<u64>,
    warm_start: bool,
    verbose: usize,
    verbose_interval: usize,
    /// NOT a hyperparameter: the previous fit's parameters, carried in so a
    /// `warm_start` refit resumes instead of re-initializing. Set only by
    /// [`GaussianMixtureBuilder::warm_params`] (the binding layer's spelling of
    /// [`GaussianMixture::into_warm_start`]), and excluded from
    /// `into_builder` / `hyperparams_eq` for exactly that reason.
    warm_params: Option<MixtureParams>,
}

impl Default for GaussianMixtureBuilder {
    /// Re-derive the sklearn defaults from [`GaussianMixture::new`] — the SINGLE
    /// source of truth. `f64` is pinned only to read the `F`-independent scalar
    /// defaults; the builder itself is non-generic.
    fn default() -> Self {
        GaussianMixture::<f64, Unfit>::new().into_builder()
    }
}

impl GaussianMixtureBuilder {
    /// Pin the EM engine (DEVICE-PARAM-01). [`Device::Auto`] keeps the
    /// `gmm_device_applicable` gate; `Cpu`/`Gpu` override its SIZE decision.
    /// The capability half still applies — see [`GaussianMixture::device_arm`].
    pub fn device(mut self, v: Device) -> Self {
        self.device = v;
        self
    }

    /// Set the number of mixture components `k`.
    pub fn n_components(mut self, v: usize) -> Self {
        self.n_components = v;
        self
    }

    /// Set `covariance_type` (`"full"` / `"tied"` / `"diag"` / `"spherical"`) —
    /// the parameterization of every component's covariance, and the single
    /// hyperparameter that changes the algorithm's complexity class.
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

    /// Set the per-restart EM iteration cap.
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

    /// Inject `weights_init` (length `n_components`), overriding the
    /// initialization's mixing proportions.
    pub fn weights_init(mut self, v: Option<Vec<f64>>) -> Self {
        self.weights_init = v;
        self
    }

    /// Inject `means_init` (`n_components × n_features`, row-major).
    pub fn means_init(mut self, v: Option<Vec<f64>>) -> Self {
        self.means_init = v;
        self
    }

    /// Inject `precisions_init`, flat in the `covariance_type` layout.
    pub fn precisions_init(mut self, v: Option<Vec<f64>>) -> Self {
        self.precisions_init = v;
        self
    }

    /// Set the seed for the initialization RNG (sklearn's `random_state`).
    pub fn random_state(mut self, v: Option<u64>) -> Self {
        self.random_state = v;
        self
    }

    /// Set `warm_start` — resume from the previous fit's parameters instead of
    /// re-initializing (see [`GaussianMixture::into_warm_start`]).
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

    /// Carry a previous fit's parameters in, so a `warm_start = true` build
    /// RESUMES from them (the builder-side spelling of
    /// [`GaussianMixture::into_warm_start`], for callers — like the PyO3
    /// wrapper — that rebuild the estimator from stored hyperparameters at
    /// every `fit` rather than moving the fitted value).
    ///
    /// Ignored unless `warm_start` is also set, exactly as sklearn ignores its
    /// retained attributes then.
    pub fn warm_params(mut self, v: Option<MixtureParams>) -> Self {
        self.warm_params = v;
        self
    }

    /// Build the (unfit) estimator, parsing the string hyperparameters into
    /// typed enums and rejecting every data-INDEPENDENT invalid value (D-08).
    pub fn build<F>(self) -> Result<GaussianMixture<F, Unfit>, BuildError>
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
        // sklearn's `max_iter` constraint is `Interval(Integral, 0, None,
        // closed="left")` — `0` is LEGAL and means "report the initialization".
        let covariance_type = CovarianceType::try_from(self.covariance_type.as_str()).map_err(
            |()| BuildError::UnknownCovarianceType {
                value: self.covariance_type.clone(),
            },
        )?;
        let init_params = InitParams::try_from(self.init_params.as_str())?;
        Ok(GaussianMixture {
            device: self.device,
            device_: None,
            n_components: self.n_components,
            covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            n_init: self.n_init,
            init_params,
            weights_init: self.weights_init,
            means_init: self.means_init,
            precisions_init: self.precisions_init,
            random_state: self.random_state,
            warm_start: self.warm_start,
            verbose: self.verbose,
            verbose_interval: self.verbose_interval,
            warm: self.warm_params,
            params: None,
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

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

/// sklearn's `_initialize_parameters`: the `n × k` responsibility matrix each
/// `init_params` route produces, BEFORE any M-step has run.
///
/// Shared verbatim with [`BayesianGaussianMixture`](super::bayesian_gaussian_mixture::BayesianGaussianMixture)
/// (MIX-02), which takes the same four routes — sklearn implements them once on
/// `BaseMixture` for exactly that reason. Keeping one copy is what makes the
/// oracle's `init_params` cross apply to both estimators: a divergence between
/// two transcriptions would show up as one estimator converging to a different
/// optimum, which is the hardest kind of bug to attribute.
pub(crate) fn initial_responsibilities(
    init: InitParams,
    host: &mut GmmHost<'_>,
    rng: &mut SplitMix64,
    n: usize,
    k: usize,
) -> Vec<f64> {
    let mut resp = vec![0.0f64; n * k];
    match init {
        InitParams::KMeans => {
            let labels = host.kmeans_labels(rng);
            for (i, &l) in labels.iter().enumerate() {
                resp[i * k + l as usize] = 1.0;
            }
        }
        InitParams::KMeansPlusPlus => {
            for (c, idx) in host.kmeans_plusplus(k, rng).into_iter().enumerate() {
                resp[idx * k + c] = 1.0;
            }
        }
        InitParams::Random => {
            for row in resp.chunks_mut(k) {
                let mut s = 0.0;
                for slot in row.iter_mut() {
                    let v = rng.next_f64();
                    *slot = v;
                    s += v;
                }
                // A degenerate all-zero draw is impossible in practice but
                // would divide by zero; fall back to the uniform row.
                let inv = if s > 0.0 { 1.0 / s } else { 1.0 / k as f64 };
                for slot in row.iter_mut() {
                    *slot *= inv;
                }
            }
        }
        InitParams::RandomFromData => {
            for (c, idx) in distinct_indices(n, k, rng).into_iter().enumerate() {
                resp[idx * k + c] = 1.0;
            }
        }
    }
    resp
}

/// Map a `gmm_host` factorization failure onto the crate's typed error.
///
/// This is sklearn's
/// `"Fitting the mixture model failed because some components have ill-defined
/// empirical covariance"` — raised for exactly the same reason (a component
/// collapsed onto fewer points than it has dimensions, or `reg_covar` is too
/// small for the data's conditioning), and carrying the offending pivot so the
/// caller can see WHICH component died and where.
pub(crate) fn ill_conditioned(e: IllConditioned) -> AlgoError {
    AlgoError::Prim(PrimError::NotPositiveDefinite {
        operand: "gaussian_mixture covariance",
        pivot_index: e.pivot_index,
        pivot_value: e.pivot_value,
    })
}

/// Row-wise argmax of an `n × k` matrix, ties going to the LOWEST index
/// (numpy's `argmax` convention).
pub(crate) fn argmax_rows(m: &[f64], n: usize, k: usize) -> Vec<i32> {
    (0..n)
        .map(|i| {
            let row = &m[i * k..(i + 1) * k];
            let mut best = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for (c, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = c;
                }
            }
            best as i32
        })
        .collect()
}

/// `k` DISTINCT uniformly-drawn row indices — numpy's
/// `choice(n, size=k, replace=False)`, by partial Fisher-Yates over a lazily
/// materialized index array (so `n` large with `k` small stays `O(n)` once).
fn distinct_indices(n: usize, k: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let j = i + (rng.next_below((n - i) as u64) as usize);
        idx.swap(i, j);
        out.push(idx[i]);
    }
    out
}

/// One standard normal draw by the Box-Muller transform. Used only by
/// [`GaussianMixture::sample`], which is not on any hot path.
pub(crate) fn standard_normal(rng: &mut SplitMix64) -> f64 {
    // `next_f64` is in [0, 1); guard the log against an exact 0.
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

