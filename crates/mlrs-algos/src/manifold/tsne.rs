//! `TSNE` (TSNE-01 / TSNE-PARAMS) — t-distributed Stochastic Neighbor
//! Embedding with sklearn 1.9.0's FULL parameter surface.
//!
//! ```text
//! TSNE(n_components=2, *, perplexity=30.0, early_exaggeration=12.0,
//!      learning_rate='auto', max_iter=1000, n_iter_without_progress=300,
//!      min_grad_norm=1e-7, metric='euclidean', metric_params=None,
//!      init='pca', verbose=0, random_state=None, method='barnes_hut',
//!      angle=0.5, n_jobs=None)
//! ```
//!
//! Every one of those is honoured. The three that change what the algorithm
//! DOES — rather than only when it stops — are `method`, `metric`, and `init`:
//!
//! - **`method`** picks the objective. `'barnes_hut'` (sklearn's DEFAULT) takes
//!   the `O(n log n)` route: a `min(n − 1, int(3·perplexity + 1))`-neighbour
//!   graph, a SPARSE `P` ([`tsne_knn`](super::tsne_knn)), and a quadtree-
//!   summarized negative force. `'exact'` takes the `O(n²)` route: a dense
//!   pairwise matrix, a dense `P`, and the full Student-t gradient. They are
//!   different algorithms with different asymptotics, not two spellings of one.
//! - **`metric`** selects among the 22 sklearn strings
//!   ([`TsneMetric`](super::tsne_metric::TsneMetric)), including `'precomputed'`
//!   and the six bool-cast set metrics; `metric_params` carries scipy's `p` /
//!   `V` / `VI`.
//! - **`init`** is `'pca'`, `'random'`, or a caller-supplied `(n,
//!   n_components)` array (sklearn accepts an `ndarray` there).
//!
//! ## Where the work happens
//! The gradient descent — both objectives — runs in the parallel host engine
//! [`mlrs_backend::prims::tsne_host`], which owns the justification for being
//! host-resident. The pre-existing exact DEVICE prim
//! ([`mlrs_backend::prims::tsne`]) is retained and still serves the exact
//! method on backends where it was measured to win; [`host_engine_applicable`]
//! is the dispatch, and `MLRS_TSNE_HOST` the A/B knob.
//!
//! ## Deliberate divergences from sklearn, all documented at their field
//! - `n_jobs = None` uses every core rather than joblib's one worker. Every
//!   parallel pass here reduces in POINT order, so the thread count cannot
//!   change a value — leaving cores idle would buy nothing.
//! - `verbose` prints sklearn's progress lines: the stage banners at `>= 1`
//!   and the per-check iteration report at `>= 2`. It is otherwise inert, and
//!   gated for that (`tsne_params_test::verbose_is_value_neutral`).
//! - The Barnes-Hut engine runs in `f64`; sklearn casts to `float32`.
//! - `random_state` seeds a SplitMix64, deliberately ≠ numpy's MT19937 (the
//!   milestone-wide stochastic-gate convention), so `init='random'` fits are
//!   compared by property, never by value.
//!
//! Tests live in `crates/mlrs-algos/tests/tsne_test.rs` and
//! `crates/mlrs-algos/tests/tsne_params_test.rs` (AGENTS.md §2).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device::Device;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::rng::SplitMix64;
use mlrs_backend::prims::tsne::{squared_distance, tsne_gradient, MACHINE_EPSILON};
use mlrs_backend::prims::tsne_host::{
    self, TsneDescentConfig, TsneP, BH_MAX_COMPONENTS, EXPLORATION_MAX_ITER, N_ITER_CHECK,
};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64};

use crate::error::{AlgoError, BuildError};
use crate::manifold::tsne_knn::{bh_n_neighbors, joint_probabilities_nn, knn_graph};
use crate::manifold::tsne_metric::{
    pairwise_squared, resolve_metric_params, validate_metric_geometry, MetricParams, TsneMetric,
};
use crate::typestate::{validate_geometry, Fit, Fitted, State, Unfit};

/// Embedding initialization (sklearn `init`).
#[derive(Debug, Clone, PartialEq)]
pub enum TsneInit {
    /// Deterministic PCA init (sklearn 1.9's default): project onto the top
    /// `n_components` principal axes, then scale so `std(y[:, 0]) = 1e-4`.
    /// sklearn REJECTS this with `metric='precomputed'` (there is no feature
    /// space to project), and so does [`Fit::fit`].
    Pca,
    /// Random init: `1e-4 · N(0, 1)` from the seeded SplitMix64.
    Random,
    /// A caller-supplied starting embedding, row-major `(n, n_components)` —
    /// sklearn's `init=<ndarray>`. Its length is checked against the design at
    /// `fit`.
    Array(Vec<f64>),
}

/// The learning-rate specification (sklearn `learning_rate`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearningRate {
    /// sklearn `'auto'`: `max(n_samples / early_exaggeration / 4, 50)`.
    Auto,
    /// An explicit positive step size.
    Value(f64),
}

/// The gradient objective (sklearn `method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsneMethod {
    /// `'barnes_hut'` — sklearn's DEFAULT. `O(n log n)` via a sparse k-NN `P`
    /// and a quadtree-summarized negative force. Requires `n_components <= 3`.
    BarnesHut,
    /// `'exact'` — the dense `O(n²)` objective.
    Exact,
}

impl TsneMethod {
    /// Parse a sklearn `method=` string.
    pub fn from_sklearn_name(s: &str) -> Option<Self> {
        match s {
            "barnes_hut" => Some(Self::BarnesHut),
            "exact" => Some(Self::Exact),
            _ => None,
        }
    }
}

/// Should the parallel HOST engine serve this fit, rather than the exact
/// DEVICE prim?
///
/// Barnes-Hut has no device arm at all (see
/// [`mlrs_backend::prims::tsne_host`]), so this only decides the EXACT method.
/// It is true on `cpu`, where the device prim's per-iteration upload + gradient
/// readback is the round-trip pathology [[mlrs-gpu-perf-root-cause]] names, and
/// false elsewhere — a perf path is gated on the backend it was MEASURED on and
/// never extrapolated onto another ([[mlrs-feedback-verify-on-target-hardware]]).
///
/// `MLRS_TSNE_HOST=1` forces the host arm on, `=0` forces it off, for on-target
/// A/B.
///
/// One case is NOT overridable by that knob: an `f64` fit on a backend that
/// cannot evaluate f64 transcendentals in device code. The device gradient's
/// Student-t weight is a `powf`, so there the device arm does not merely run
/// slowly — it cannot run at all (`wgpu` returns
/// [`PrimError::UnsupportedCapability`](mlrs_core::PrimError), and without the
/// guard the driver's shader compiler SEGFAULTs; see
/// [`mlrs_backend::capability::f64_transcendental_supported`] and
/// [[mlrs-wgpu-f64-eig-broken]]). The host engine has no such limit, so it is
/// the only way `method='exact'` can serve `f64` there, and the knob stays a
/// pure PERF switch rather than a correctness one — the
/// [`host_knn_applicable`](super::umap_host_knn::host_knn_applicable)
/// precedent.
/// Can the DEVICE engine run at all for this float width?
///
/// The CAPABILITY half of [`host_engine_applicable`], split out under
/// DEVICE-PARAM-01. `device='gpu'` may override the PERF half; overriding this
/// one is not a slowdown but a CRASH — on a backend without `f64`
/// transcendentals the device path's `powf`/`exp` does not fail at launch, the
/// driver's shader compiler segfaults (the measurement is on
/// `umap_host_knn::host_knn_applicable`, which shares this shape).
pub fn device_engine_possible<F>() -> bool {
    !(std::mem::size_of::<F>() == 8
        && !mlrs_backend::capability::f64_transcendental_supported())
}

pub fn host_engine_applicable<F>() -> bool {
    if std::mem::size_of::<F>() == 8
        && !mlrs_backend::capability::f64_transcendental_supported()
    {
        return true;
    }
    match mlrs_backend::abflag::var("MLRS_TSNE_HOST").as_deref() {
        Some("0") => false,
        Some(_) => true,
        None => mlrs_backend::capability::active_backend_name() == "cpu",
    }
}

/// t-SNE (TSNE-01), builder-fronted + typestate (`Tsne<F, S = Unfit>`).
/// No `Debug` derive — `DeviceArray` is not `Debug` (the family precedent).
pub struct Tsne<F, S = Unfit>
where
    S: State,
{
    /// Embedding dimensionality (sklearn `n_components`, default 2).
    n_components: usize,
    /// Target perplexity (sklearn `perplexity`, default 30).
    perplexity: f64,
    /// Early-exaggeration factor (sklearn default 12).
    early_exaggeration: f64,
    /// Learning rate (sklearn 1.9 default `'auto'`).
    learning_rate: LearningRate,
    /// Total gradient-descent iterations (sklearn `max_iter`, default 1000).
    max_iter: usize,
    /// Iterations without KL improvement before the MAIN phase gives up
    /// (sklearn `n_iter_without_progress`, default 300). The exploration phase
    /// always uses its own full length instead, which is what sklearn passes.
    n_iter_without_progress: usize,
    /// Convergence threshold on the (gains-scaled) gradient norm
    /// (sklearn `min_grad_norm`, default 1e-7).
    min_grad_norm: f64,
    /// Input-space metric (sklearn `metric`, default `'euclidean'`).
    metric: TsneMetric,
    /// scipy keyword payload for `metric` (sklearn `metric_params`).
    metric_params: MetricParams,
    /// Init strategy (sklearn 1.9 default `'pca'`).
    init: TsneInit,
    /// Progress verbosity (sklearn `verbose`, default 0). `>= 1` prints the
    /// stage banners sklearn prints, `>= 2` adds its per-check iteration
    /// report. Printing is the ONLY effect: the fit is bit-identical at every
    /// level (`tsne_params_test::verbose_is_value_neutral`).
    verbose: usize,
    /// Seed for the `init='random'` SplitMix64 (sklearn `random_state`).
    seed: u64,
    /// The engine that ACTUALLY ran (`"cpu"` / `"gpu"`), `None` until `fit`.
    device_: Option<&'static str>,
    /// Where to run the gradient descent (DEVICE-PARAM-01). `Auto` keeps the
    /// `MLRS_TSNE_HOST`-then-backend ladder; `Cpu`/`Gpu` override its PERF half
    /// only — [`device_engine_possible`] is never overridable.
    device: Device,
    /// The gradient objective (sklearn `method`, default `'barnes_hut'`).
    method: TsneMethod,
    /// Barnes-Hut summary angle θ (sklearn `angle`, default 0.5). A cell
    /// summarizes when `width² / dist² < θ²`, so LOWER is more accurate and
    /// slower. Inert for `method='exact'`.
    angle: f64,
    /// Host worker count (sklearn `n_jobs`, default `None`). joblib semantics
    /// for the numeric values: positive is exact, negative counts back from all
    /// cores (`-1` = all, `-2` = all but one), and an offset past zero clamps
    /// to one worker rather than erroring.
    ///
    /// `None` resolves to ALL cores, NOT joblib's single worker. Every parallel
    /// pass in this estimator reduces in point order, so the worker count
    /// cannot change a value — the divergence is pure wall clock, and it is
    /// gated by an exact-equality test across `n_jobs` settings.
    n_jobs: Option<i32>,
    /// Fitted embedding (`n × n_components`, row-major, device-resident).
    embedding_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Final KL divergence (sklearn `kl_divergence_`).
    kl_divergence_: Option<f64>,
    /// Iterations actually run (sklearn `n_iter_`).
    n_iter_: usize,
    /// Number of features seen at fit.
    n_features_in_: usize,
    _float: PhantomData<F>,
    _state: PhantomData<S>,
}

impl<F> Tsne<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// sklearn 1.9 defaults (D-08 single source).
    pub fn new() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            early_exaggeration: 12.0,
            learning_rate: LearningRate::Auto,
            max_iter: 1000,
            n_iter_without_progress: 300,
            min_grad_norm: 1e-7,
            metric: TsneMetric::Euclidean,
            metric_params: MetricParams::default(),
            init: TsneInit::Pca,
            verbose: 0,
            seed: 0,
            device: Device::Auto,
            device_: None,
            method: TsneMethod::BarnesHut,
            angle: 0.5,
            n_jobs: None,
            embedding_: None,
            kl_divergence_: None,
            n_iter_: 0,
            n_features_in_: 0,
            _float: PhantomData,
            _state: PhantomData,
        }
    }

    /// Start building from sklearn's defaults (D-08 single source).
    pub fn builder() -> TsneBuilder {
        TsneBuilder::default()
    }

    /// Fold this (unfit) estimator back into a builder (round-trip surface).
    pub fn into_builder(self) -> TsneBuilder {
        TsneBuilder {
            n_components: self.n_components,
            perplexity: self.perplexity,
            early_exaggeration: self.early_exaggeration,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            n_iter_without_progress: self.n_iter_without_progress,
            min_grad_norm: self.min_grad_norm,
            metric: self.metric,
            metric_params: self.metric_params,
            init: self.init,
            verbose: self.verbose,
            seed: self.seed,
            device: self.device,
            method: self.method,
            angle: self.angle,
            n_jobs: self.n_jobs,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the
    /// defaults-equality gate, BLDR-01).
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        self.n_components == other.n_components
            && self.perplexity == other.perplexity
            && self.early_exaggeration == other.early_exaggeration
            && self.learning_rate == other.learning_rate
            && self.max_iter == other.max_iter
            && self.n_iter_without_progress == other.n_iter_without_progress
            && self.min_grad_norm == other.min_grad_norm
            && self.metric == other.metric
            && self.metric_params == other.metric_params
            && self.init == other.init
            && self.verbose == other.verbose
            && self.seed == other.seed
            && self.method == other.method
            && self.angle == other.angle
            && self.n_jobs == other.n_jobs
            && self.device == other.device
    }

    /// Should the HOST t-SNE engine run, honouring `device` (DEVICE-PARAM-01)?
    ///
    /// Capability FIRST: with no `f64` transcendentals the device engine cannot
    /// run at all, so the host arm is forced regardless of preference. Only
    /// after that does `device` get to override the perf ladder.
    fn host_engine_arm(&self) -> bool {
        if !device_engine_possible::<F>() {
            return true;
        }
        self.device.prefers_host(host_engine_applicable::<F>)
    }

    /// `fit_transform`: fit to `x` and return the fitted embedding host buffer
    /// (row-major `(n, n_components)`) in one call — sklearn `fit_transform`.
    /// CONSUMES `self` (the `Fit::fit` contract).
    pub fn fit_transform(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<F>, AlgoError> {
        let fitted = self.fit(pool, x, None, shape)?;
        Ok(fitted.embedding(pool))
    }
}

impl<F> Default for Tsne<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F, S> Tsne<F, S>
where
    S: State,
{
    /// Resolve `n_jobs` into a worker count (see the [`Self::n_jobs`] docs for
    /// the `None` divergence).
    fn units(&self) -> usize {
        let all = mlrs_backend::capability::cpu_launch_units().max(1) as i64;
        match self.n_jobs {
            None => all as usize,
            Some(k) if k > 0 => k as usize,
            Some(k) => (all + 1 + i64::from(k)).max(1) as usize,
        }
    }
}

/// Builder for [`Tsne`] (data-INDEPENDENT validation at `build`, D-08).
#[derive(Debug, Clone)]
pub struct TsneBuilder {
    n_components: usize,
    perplexity: f64,
    early_exaggeration: f64,
    learning_rate: LearningRate,
    max_iter: usize,
    n_iter_without_progress: usize,
    min_grad_norm: f64,
    metric: TsneMetric,
    metric_params: MetricParams,
    init: TsneInit,
    verbose: usize,
    seed: u64,
    method: TsneMethod,
    angle: f64,
    n_jobs: Option<i32>,
    device: Device,
}

impl Default for TsneBuilder {
    /// Re-derive the sklearn defaults from [`Tsne::new`] (D-08 single source).
    fn default() -> Self {
        Tsne::<f64, Unfit>::new().into_builder()
    }
}

impl TsneBuilder {
    /// Pin the execution arm of the gradient descent (DEVICE-PARAM-01).
    /// [`Device::Auto`] keeps the existing `MLRS_TSNE_HOST`-then-backend ladder.
    pub fn device(mut self, v: Device) -> Self {
        self.device = v;
        self
    }

    /// Set the embedding dimensionality `n_components`.
    pub fn n_components(mut self, v: usize) -> Self {
        self.n_components = v;
        self
    }
    /// Set the target `perplexity`.
    pub fn perplexity(mut self, v: f64) -> Self {
        self.perplexity = v;
        self
    }
    /// Set the `early_exaggeration` factor.
    pub fn early_exaggeration(mut self, v: f64) -> Self {
        self.early_exaggeration = v;
        self
    }
    /// Set the learning rate (`Auto` or an explicit positive value).
    pub fn learning_rate(mut self, v: LearningRate) -> Self {
        self.learning_rate = v;
        self
    }
    /// Set the total iteration budget `max_iter`.
    pub fn max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }
    /// Set `n_iter_without_progress` (the MAIN phase's patience).
    pub fn n_iter_without_progress(mut self, v: usize) -> Self {
        self.n_iter_without_progress = v;
        self
    }
    /// Set the convergence threshold `min_grad_norm`.
    pub fn min_grad_norm(mut self, v: f64) -> Self {
        self.min_grad_norm = v;
        self
    }
    /// Set the input-space `metric`.
    pub fn metric(mut self, v: TsneMetric) -> Self {
        self.metric = v;
        self
    }
    /// Set the scipy keyword payload for `metric`.
    pub fn metric_params(mut self, v: MetricParams) -> Self {
        self.metric_params = v;
        self
    }
    /// Set the init strategy.
    pub fn init(mut self, v: TsneInit) -> Self {
        self.init = v;
        self
    }
    /// Set the progress `verbose` level.
    pub fn verbose(mut self, v: usize) -> Self {
        self.verbose = v;
        self
    }
    /// Set the `init='random'` seed (sklearn `random_state`).
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }
    /// Set the gradient objective `method`.
    pub fn method(mut self, v: TsneMethod) -> Self {
        self.method = v;
        self
    }
    /// Set the Barnes-Hut summary angle θ.
    pub fn angle(mut self, v: f64) -> Self {
        self.angle = v;
        self
    }
    /// Set the host worker count `n_jobs`.
    pub fn n_jobs(mut self, v: Option<i32>) -> Self {
        self.n_jobs = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08):
    /// - `n_components >= 1`, and `<= 3` for `method='barnes_hut'` (the tree is
    ///   a quad-/oct-tree; sklearn raises the same),
    /// - `perplexity` finite and `> 0`,
    /// - `early_exaggeration` finite and `>= 1`,
    /// - explicit `learning_rate` finite and `> 0`,
    /// - `max_iter >= 1`, `n_iter_without_progress >= 1`,
    /// - `min_grad_norm >= 0` and finite,
    /// - `angle` in `[0, 1]` (sklearn's `Interval(Real, 0, 1, closed='both')`),
    /// - `minkowski` `p` finite and `> 0`,
    /// - `n_jobs != 0`.
    pub fn build<F>(self) -> Result<Tsne<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if self.n_components < 1 {
            return Err(BuildError::InvalidNComponents {
                estimator: "tsne",
                param: "n_components",
                value: self.n_components,
            });
        }
        // sklearn: "'n_components' should be inferior to 4 for the barnes_hut
        // algorithm as it relies on quad-tree or oct-tree."
        if self.method == TsneMethod::BarnesHut && self.n_components > BH_MAX_COMPONENTS {
            return Err(BuildError::InvalidNComponents {
                estimator: "tsne (method='barnes_hut' needs n_components <= 3)",
                param: "n_components",
                value: self.n_components,
            });
        }
        if !(self.perplexity > 0.0) || !self.perplexity.is_finite() {
            return Err(BuildError::InvalidPerplexity {
                estimator: "tsne",
                perplexity: self.perplexity,
            });
        }
        if !(self.early_exaggeration >= 1.0) || !self.early_exaggeration.is_finite() {
            return Err(BuildError::InvalidEarlyExaggeration {
                estimator: "tsne",
                early_exaggeration: self.early_exaggeration,
            });
        }
        if let LearningRate::Value(lr) = self.learning_rate {
            if !(lr > 0.0) || !lr.is_finite() {
                return Err(BuildError::InvalidLearningRate {
                    estimator: "tsne",
                    learning_rate: lr,
                });
            }
        }
        if self.max_iter < 1 {
            return Err(BuildError::InvalidMaxIter {
                estimator: "tsne",
                max_iter: self.max_iter,
            });
        }
        if self.n_iter_without_progress < 1 {
            return Err(BuildError::InvalidMaxIter {
                estimator: "tsne (n_iter_without_progress)",
                max_iter: self.n_iter_without_progress,
            });
        }
        if !(self.min_grad_norm >= 0.0) || !self.min_grad_norm.is_finite() {
            return Err(BuildError::InvalidTol {
                estimator: "tsne",
                tol: self.min_grad_norm,
            });
        }
        // sklearn: Interval(Real, 0, 1, closed="both").
        if !(0.0..=1.0).contains(&self.angle) {
            return Err(BuildError::InvalidAngle {
                estimator: "tsne",
                angle: self.angle,
            });
        }
        if self.metric == TsneMetric::Minkowski {
            if let Some(p) = self.metric_params.p {
                if !(p > 0.0) || !p.is_finite() {
                    return Err(BuildError::InvalidMinkowskiP {
                        estimator: "tsne",
                        p,
                    });
                }
            }
        }
        if self.n_jobs == Some(0) {
            return Err(BuildError::InvalidNJobs { estimator: "tsne" });
        }
        Ok(Tsne {
            n_components: self.n_components,
            perplexity: self.perplexity,
            early_exaggeration: self.early_exaggeration,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            n_iter_without_progress: self.n_iter_without_progress,
            min_grad_norm: self.min_grad_norm,
            metric: self.metric,
            metric_params: self.metric_params,
            init: self.init,
            verbose: self.verbose,
            seed: self.seed,
            device: self.device,
            device_: None,
            method: self.method,
            angle: self.angle,
            n_jobs: self.n_jobs,
            embedding_: None,
            kl_divergence_: None,
            n_iter_: 0,
            n_features_in_: 0,
            _float: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for Tsne<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = Tsne<F, Fitted>;

    /// Fit: distances (dense or k-NN, under `metric`) → perplexity search → `P`
    /// → init → the two-phase gradient descent for the chosen `method`.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        _y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Tsne<F, Fitted>, AlgoError> {
        let (n, p) = shape;
        validate_geometry(x, shape)?;

        // sklearn: "perplexity must be less than n_samples" (data-DEPENDENT).
        if self.perplexity >= n as f64 {
            return Err(AlgoError::InvalidPerplexity {
                estimator: "tsne",
                perplexity: self.perplexity,
                n_samples: n,
            });
        }
        let d = self.n_components;
        // t-SNE needs at least 2 points to define pairwise affinities.
        if n < 2 {
            return Err(AlgoError::Prim(mlrs_core::PrimError::ShapeMismatch {
                operand: "x (tsne requires >= 2 samples)",
                rows: n,
                cols: p,
                len: x.len(),
            }));
        }
        validate_metric_geometry(n, p, self.metric)?;
        // sklearn: 'The parameter init="pca" cannot be used with
        // metric="precomputed".' — there is no feature space to project.
        if self.metric == TsneMetric::Precomputed && self.init == TsneInit::Pca {
            return Err(AlgoError::InvalidGraphInput {
                estimator: "tsne",
                reason: "the parameter init=\"pca\" cannot be used with \
                         metric=\"precomputed\""
                    .to_string(),
            });
        }
        // `init='pca'` projects onto `n_components` principal axes, and a
        // design of rank `min(n, p)` has no more than that many. sklearn's
        // `PCA(n_components=...)` raises for the same reason; rejecting here
        // keeps the failure at the parameter the user set rather than deep in
        // the projection.
        if self.init == TsneInit::Pca && d > n.min(p) {
            return Err(AlgoError::InvalidNComponents {
                estimator: "tsne (init='pca')",
                requested: d,
                max: n.min(p),
            });
        }
        if let TsneInit::Array(v) = &self.init {
            if v.len() != n * d {
                return Err(AlgoError::InvalidGraphInput {
                    estimator: "tsne",
                    reason: format!(
                        "init array has {} entries, expected n_samples · n_components = {}",
                        v.len(),
                        n * d
                    ),
                });
            }
        }

        let units = self.units();
        let x_host: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        let rp = resolve_metric_params(&x_host, n, p, self.metric, &self.metric_params)?;

        // --- Init embedding (host f64), BEFORE the P stage so a bad init is
        //     rejected without paying for the O(n²) / O(n·k) distance work when
        //     it is the caller's own array. ---
        let mut y: Vec<f64> = match &self.init {
            TsneInit::Array(v) => v.clone(),
            TsneInit::Pca => {
                let emb = pca_init(&x_host, n, p, d);
                // sklearn: X_embedded / np.std(X_embedded[:, 0]) * 1e-4
                // (population std, ddof=0).
                let mean0 = (0..n).map(|i| emb[i * d]).sum::<f64>() / n as f64;
                let var0 = (0..n).map(|i| (emb[i * d] - mean0).powi(2)).sum::<f64>() / n as f64;
                let std0 = var0.sqrt();
                let scale = if std0 > 0.0 { 1e-4 / std0 } else { 1e-4 };
                emb.iter().map(|&v| v * scale).collect()
            }
            TsneInit::Random => {
                // 1e-4 · N(0,1) via SplitMix64 Box–Muller (deliberately ≠
                // MT19937 — the milestone stochastic-gate convention).
                let mut rng = SplitMix64::new(self.seed);
                let mut out = vec![0.0f64; n * d];
                let mut k = 0usize;
                while k < out.len() {
                    let (z0, z1) = box_muller(&mut rng);
                    out[k] = 1e-4 * z0;
                    if k + 1 < out.len() {
                        out[k + 1] = 1e-4 * z1;
                    }
                    k += 2;
                }
                out
            }
        };

        let dof = (d as f64 - 1.0).max(1.0);
        let learning_rate = match self.learning_rate {
            LearningRate::Auto => (n as f64 / self.early_exaggeration / 4.0).max(50.0),
            LearningRate::Value(v) => v,
        };
        let cfg = TsneDescentConfig {
            n,
            d,
            dof,
            max_iter: self.max_iter,
            early_exaggeration: self.early_exaggeration,
            learning_rate,
            min_grad_norm: self.min_grad_norm,
            n_iter_without_progress: self.n_iter_without_progress,
            angle: self.angle,
            threads: units,
            verbose: self.verbose,
        };

        let (kl, n_iter) = match self.method {
            TsneMethod::BarnesHut => {
                let k = bh_n_neighbors(n, self.perplexity);
                if self.verbose > 0 {
                    println!("[t-SNE] Computing {k} nearest neighbors...");
                }
                let graph = knn_graph(&x_host, n, p, k, self.metric, &rp, units)?;
                let sp = joint_probabilities_nn(&graph, n, self.perplexity, units);
                let outcome = tsne_host::tsne_descent(
                    &mut y,
                    TsneP::Sparse {
                        indptr: &sp.indptr,
                        indices: &sp.indices,
                        data: &sp.data,
                    },
                    &cfg,
                );
                (outcome.kl_divergence, outcome.n_iter)
            }
            TsneMethod::Exact => {
                if self.verbose > 0 {
                    println!("[t-SNE] Computing pairwise distances...");
                }
                // The euclidean fast path stays on the DEVICE distance prim
                // when the device arm is serving this backend; every other
                // metric is host-only either way.
                let dsq = if self.metric == TsneMetric::Euclidean && !self.host_engine_arm() {
                    let dsq_dev = squared_distance::<F>(pool, x, n, p);
                    let v: Vec<f64> = dsq_dev
                        .to_host(pool)
                        .iter()
                        .map(|&val| host_to_f64(val))
                        .collect();
                    dsq_dev.release_into(pool);
                    v
                } else {
                    pairwise_squared(&x_host, n, p, self.metric, &rp, units)?
                };
                let p_joint = joint_probabilities(&dsq, n, self.perplexity, units);
                if self.host_engine_arm() {
                    let outcome =
                        tsne_host::tsne_descent(&mut y, TsneP::Dense(&p_joint), &cfg);
                    (outcome.kl_divergence, outcome.n_iter)
                } else {
                    device_exact_descent::<F>(pool, &mut y, &p_joint, &cfg)?
                }
            }
        };

        if self.verbose > 0 {
            println!("[t-SNE] KL divergence after {} iterations: {kl}", n_iter + 1);
        }

        let y_f: Vec<F> = y.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let embedding_ = DeviceArray::from_host(pool, &y_f);

        // Recorded from the SAME predicate the descent branched on, so
        // `device_` cannot name an engine the fit did not use.
        let arm = if self.host_engine_arm() { "cpu" } else { "gpu" };
        Ok(Tsne {
            n_components: self.n_components,
            perplexity: self.perplexity,
            early_exaggeration: self.early_exaggeration,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            n_iter_without_progress: self.n_iter_without_progress,
            min_grad_norm: self.min_grad_norm,
            metric: self.metric,
            metric_params: self.metric_params,
            init: self.init,
            verbose: self.verbose,
            seed: self.seed,
            device: self.device,
            device_: Some(arm),
            method: self.method,
            angle: self.angle,
            n_jobs: self.n_jobs,
            embedding_: Some(embedding_),
            kl_divergence_: Some(kl),
            n_iter_: n_iter,
            n_features_in_: p,
            _float: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<F> Tsne<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `embedding_` (`n × n_components` row-major).
    /// `Some` by construction on the `Fitted` state (D-03).
    pub fn embedding(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.embedding_
            .as_ref()
            .expect("embedding_ is Some by construction on Tsne<F, Fitted>")
            .to_host(pool)
    }

    /// The final KL divergence (sklearn `kl_divergence_`). `Some` by
    /// construction on the `Fitted` state.
    /// The engine that ACTUALLY ran, `"cpu"` or `"gpu"` (DEVICE-PARAM-01);
    /// `None` before `fit`.
    pub fn device_arm(&self) -> Option<&'static str> {
        self.device_
    }

    pub fn kl_divergence(&self) -> f64 {
        self.kl_divergence_
            .expect("kl_divergence_ is Some by construction on Tsne<F, Fitted>")
    }

    /// Iterations actually run (sklearn `n_iter_`).
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Number of features seen at fit (`n_features_in_`).
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
    }
}

// ===========================================================================
// Host pipeline stages (line-exact sklearn ports)
// ===========================================================================

/// sklearn `_utils.pyx::_binary_search_perplexity` + `_joint_probabilities`
/// (verified against the installed 1.9.0 source): the input squared distances
/// are rounded through **f32** (sklearn `distances.astype(np.float32)`), all
/// arithmetic and the P array are **f64**. Returns the DENSE row-major joint
/// `P` (diagonal 0; off-diagonal `max(p_ij/ΣP, MACHINE_EPSILON)`).
///
/// `dsq` is the dense row-major `n×n` SQUARED distance matrix; `perplexity`
/// must be positive (builder-validated).
///
/// Rows are INDEPENDENT (each bisects its own `beta`), so the search is split
/// over `threads` scoped workers on disjoint output rows. Splitting cannot
/// change a value; it is the same arithmetic in the same per-row order.
pub fn joint_probabilities(dsq: &[f64], n: usize, perplexity: f64, threads: usize) -> Vec<f64> {
    debug_assert_eq!(dsq.len(), n * n);

    // sklearn: distances.astype(np.float32) — the ONLY f32 rounding.
    let d32: Vec<f32> = dsq.iter().map(|&v| v as f32).collect();

    let desired_entropy = perplexity.ln();
    let mut cond = vec![0.0f64; n * n];

    {
        let d32 = &d32;
        let run = |row0: usize, block: &mut [f64]| {
            for (r, out) in block.chunks_exact_mut(n).enumerate() {
                let i = row0 + r;
                dense_search_one_row(&d32[i * n..i * n + n], out, i, n, desired_entropy);
            }
        };
        let units = threads.max(1).min(n.max(1));
        if units <= 1 {
            run(0, &mut cond);
        } else {
            let rows_per = n.div_ceil(units);
            std::thread::scope(|scope| {
                let run = &run;
                let mut rest: &mut [f64] = &mut cond;
                let mut row0 = 0usize;
                let mut first: Option<(usize, &mut [f64])> = None;
                while row0 < n {
                    let rows = rows_per.min(n - row0);
                    let (blk, tail) = rest.split_at_mut(rows * n);
                    rest = tail;
                    if first.is_none() {
                        first = Some((row0, blk));
                    } else {
                        scope.spawn(move || run(row0, blk));
                    }
                    row0 += rows;
                }
                if let Some((r0, blk)) = first {
                    run(r0, blk);
                }
            });
        }
    }

    // _joint_probabilities: P = cond + condᵀ, normalize by max(ΣP, eps),
    // clamp OFF-DIAGONAL at eps (sklearn clamps the condensed form; the
    // diagonal stays 0).
    let mut joint = vec![0.0f64; n * n];
    let mut sum_p = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let v = cond[i * n + j] + cond[j * n + i];
            joint[i * n + j] = v;
            sum_p += v;
        }
    }
    let sum_p = sum_p.max(MACHINE_EPSILON);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                joint[i * n + j] = (joint[i * n + j] / sum_p).max(MACHINE_EPSILON);
            }
        }
    }
    joint
}

/// One row of the DENSE perplexity search (sklearn's inner
/// `for l in range(n_steps)` with `using_neighbors = False`).
///
/// The `j != i` self-skip applies to the `exp` loop only; the normalization
/// pass divides EVERY entry, including the diagonal, exactly as sklearn does.
/// The diagonal stays 0 through both (it is never written by the `exp` loop and
/// `d²(i, i) = 0` contributes nothing to the entropy), and is forced to 0 at the
/// end so the value is structural rather than incidental.
#[inline]
fn dense_search_one_row(row32: &[f32], out: &mut [f64], i: usize, n: usize, desired_entropy: f64) {
    const EPSILON_DBL: f64 = 1e-8;
    const PERPLEXITY_TOLERANCE: f64 = 1e-5;
    const N_STEPS: usize = 100;

    let mut beta_min = f64::NEG_INFINITY;
    let mut beta_max = f64::INFINITY;
    let mut beta = 1.0f64;

    for _ in 0..N_STEPS {
        let mut sum_pi = 0.0f64;
        for j in 0..n {
            if j != i {
                let pij = (-(row32[j] as f64) * beta).exp();
                out[j] = pij;
                sum_pi += pij;
            }
        }
        if sum_pi == 0.0 {
            sum_pi = EPSILON_DBL;
        }
        let mut sum_disti_pi = 0.0f64;
        for j in 0..n {
            out[j] /= sum_pi;
            sum_disti_pi += (row32[j] as f64) * out[j];
        }
        let entropy_diff = sum_pi.ln() + beta * sum_disti_pi - desired_entropy;
        if entropy_diff.abs() <= PERPLEXITY_TOLERANCE {
            break;
        }
        if entropy_diff > 0.0 {
            beta_min = beta;
            if beta_max == f64::INFINITY {
                beta *= 2.0;
            } else {
                beta = (beta + beta_max) / 2.0;
            }
        } else {
            beta_max = beta;
            if beta_min == f64::NEG_INFINITY {
                beta /= 2.0;
            } else {
                beta = (beta + beta_min) / 2.0;
            }
        }
    }
    out[i] = 0.0;
}

/// The `init='pca'` projection: the top-`k` principal scores of `x`, host-side.
///
/// ## Why this is not [`Pca`]
/// The shared `Pca` estimator projects through the device Jacobi SVD prim,
/// which stages the tall dimension in shared memory and therefore rejects
/// `n_samples > mlrs_kernels::MAX_ROWS` (256). `init='pca'` is t-SNE's DEFAULT,
/// and t-SNE is run at thousands of samples, so routing the init through that
/// prim would make the default configuration fail on every realistic input.
///
/// The eigen route has no such cap: it forms the `p × p` feature covariance —
/// whose size is set by the FEATURE count, not the sample count — and
/// eigendecomposes that. It is the same subspace the SVD produces, since the
/// principal axes are the eigenvectors of `XᵀX`.
///
/// ## The sign convention does not matter, and here is why
/// An eigenvector is defined up to sign. That would normally demand a
/// `svd_flip`-style tie-break for reproducibility, and one is applied
/// (largest-magnitude loading made positive, deterministic for any input). But
/// it cannot affect any gated quantity: the t-SNE gradient depends on `y` only
/// through pairwise DIFFERENCES and DISTANCES, so the whole descent is
/// equivariant under an orthogonal transform of the embedding. Flipping an init
/// axis reflects the final embedding and leaves the KL divergence and every
/// neighbourhood statistic exactly unchanged.
fn pca_init(x: &[f64], n: usize, p: usize, k: usize) -> Vec<f64> {
    let k = k.min(p.min(n)).max(1);
    // Center (PCA is defined on the centered design; sklearn's PCA centers too).
    let mut means = vec![0.0f64; p];
    for j in 0..p {
        means[j] = (0..n).map(|i| x[i * p + j]).sum::<f64>() / n as f64;
    }
    let mut xc = vec![0.0f64; n * p];
    for i in 0..n {
        for j in 0..p {
            xc[i * p + j] = x[i * p + j] - means[j];
        }
    }

    // Which side to eigendecompose is a pure cost decision, not a modelling
    // one: the `p × p` feature covariance and the `n × n` Gram matrix have the
    // same non-zero spectrum, and the scores this returns are identical either
    // way (up to the per-axis sign, which the descent is equivariant to). Wide
    // designs — t-SNE on raw high-dimensional data, `p` in the thousands with
    // `n` in the hundreds — would otherwise pay an `O(p³)` Jacobi for a
    // rank-`n` problem.
    if p > n {
        return pca_init_gram(&xc, n, p, k);
    }

    // Feature covariance (up to the constant `n − 1`, which scales every
    // eigenvalue equally and so changes neither the axes nor their order).
    let mut cov = vec![0.0f64; p * p];
    for i in 0..n {
        let row = &xc[i * p..i * p + p];
        for a in 0..p {
            let va = row[a];
            if va == 0.0 {
                continue;
            }
            for b in a..p {
                cov[a * p + b] += va * row[b];
            }
        }
    }
    for a in 0..p {
        for b in a..p {
            let v = cov[a * p + b];
            cov[a * p + b] = v;
            cov[b * p + a] = v;
        }
    }

    let (eigvals, eigvecs) = jacobi_symmetric_eig(&cov, p);
    // Descending by eigenvalue — the principal axes in sklearn's order.
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| {
        eigvals[b]
            .partial_cmp(&eigvals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let mut out = vec![0.0f64; n * k];
    for (c, &axis) in order.iter().take(k).enumerate() {
        // `svd_flip`: make the largest-magnitude loading positive, so the same
        // input always yields the same signs.
        let mut best = 0usize;
        for r in 1..p {
            if eigvecs[r * p + axis].abs() > eigvecs[best * p + axis].abs() {
                best = r;
            }
        }
        let sign = if eigvecs[best * p + axis] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for i in 0..n {
            let row = &xc[i * p..i * p + p];
            let mut acc = 0.0;
            for (j, &v) in row.iter().enumerate() {
                acc += v * eigvecs[j * p + axis];
            }
            out[i * k + c] = acc * sign;
        }
    }
    out
}

/// The wide-design branch of [`pca_init`]: eigendecompose the `n × n` Gram
/// matrix `Xc·Xcᵀ` instead of the `p × p` covariance.
///
/// For a centered design, `Xc = U·S·Vᵀ`, so `Xc·Xcᵀ = U·S²·Uᵀ` and the
/// principal scores `U·S` come straight out of the Gram eigenvectors and
/// eigenvalues as `U·√Λ` — no `p`-sized matrix is ever formed. `xc` is the
/// already-centered design.
fn pca_init_gram(xc: &[f64], n: usize, p: usize, k: usize) -> Vec<f64> {
    let mut gram = vec![0.0f64; n * n];
    for i in 0..n {
        let ri = &xc[i * p..i * p + p];
        for j in i..n {
            let rj = &xc[j * p..j * p + p];
            let mut acc = 0.0;
            for t in 0..p {
                acc += ri[t] * rj[t];
            }
            gram[i * n + j] = acc;
            gram[j * n + i] = acc;
        }
    }
    let (eigvals, eigvecs) = jacobi_symmetric_eig(&gram, n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eigvals[b]
            .partial_cmp(&eigvals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let mut out = vec![0.0f64; n * k];
    for (c, &axis) in order.iter().take(k).enumerate() {
        // A tiny negative eigenvalue is rounding on a rank-deficient Gram;
        // clamp rather than take a NaN square root.
        let scale = eigvals[axis].max(0.0).sqrt();
        // The same `svd_flip` tie-break the covariance branch applies, so the
        // two routes agree on more than the subspace.
        let mut best = 0usize;
        for r in 1..n {
            if eigvecs[r * n + axis].abs() > eigvecs[best * n + axis].abs() {
                best = r;
            }
        }
        let sign = if eigvecs[best * n + axis] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for i in 0..n {
            out[i * k + c] = eigvecs[i * n + axis] * scale * sign;
        }
    }
    out
}

/// Cyclic Jacobi eigendecomposition of a symmetric `p × p` matrix. Returns
/// `(eigenvalues, eigenvectors)` with eigenvector `j` in COLUMN `j` of the
/// row-major output.
///
/// Jacobi rather than a tridiagonal reduction because `p` here is a FEATURE
/// count (single or double digits for a typical t-SNE input), and Jacobi is
/// both simpler and more accurate for small symmetric matrices — the same
/// choice the device `jacobi_eig` prim makes.
fn jacobi_symmetric_eig(a: &[f64], p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0f64; p * p];
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    if p <= 1 {
        return (vec![m.first().copied().unwrap_or(0.0)], v);
    }

    let frob: f64 = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    let tol = 1e-14 * frob.max(1.0);
    for _ in 0..100 {
        let mut off = 0.0f64;
        for r in 0..p {
            for c in (r + 1)..p {
                off += m[r * p + c] * m[r * p + c];
            }
        }
        if off.sqrt() <= tol {
            break;
        }
        for r in 0..p {
            for c in (r + 1)..p {
                let apq = m[r * p + c];
                if apq.abs() <= tol * 1e-3 {
                    continue;
                }
                let app = m[r * p + r];
                let aqq = m[c * p + c];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let cs = 1.0 / (1.0 + t * t).sqrt();
                let sn = t * cs;
                for j in 0..p {
                    let arj = m[r * p + j];
                    let acj = m[c * p + j];
                    m[r * p + j] = cs * arj - sn * acj;
                    m[c * p + j] = sn * arj + cs * acj;
                }
                for i in 0..p {
                    let air = m[i * p + r];
                    let aic = m[i * p + c];
                    m[i * p + r] = cs * air - sn * aic;
                    m[i * p + c] = sn * air + cs * aic;
                    let vir = v[i * p + r];
                    let vic = v[i * p + c];
                    v[i * p + r] = cs * vir - sn * vic;
                    v[i * p + c] = sn * vir + cs * vic;
                }
            }
        }
    }
    let eigvals = (0..p).map(|i| m[i * p + i]).collect();
    (eigvals, v)
}

/// One Box–Muller pair from two SplitMix64 uniforms (the rng.rs
/// `gaussian_matrix` idiom, without the `1/sqrt(k)` projection scale).
fn box_muller(rng: &mut SplitMix64) -> (f64, f64) {
    // Guard u1 = 0 (ln(0)) — the same open-interval nudge gaussian_matrix uses.
    let mut u1 = rng.next_f64();
    if u1 <= f64::MIN_POSITIVE {
        u1 = f64::MIN_POSITIVE;
    }
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

// ===========================================================================
// The exact DEVICE arm (retained; see `host_engine_applicable`)
// ===========================================================================

/// The two-phase schedule over the DEVICE exact-gradient prim
/// ([`tsne_gradient`]) — the pre-TSNE-PARAMS path, kept for the backends where
/// it was measured to win. Structurally identical to
/// [`tsne_host::tsne_descent`]; only the objective differs.
fn device_exact_descent<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    y: &mut [f64],
    p_joint: &[f64],
    cfg: &TsneDescentConfig,
) -> Result<(f64, usize), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let p_early: Vec<f64> = p_joint
        .iter()
        .map(|&v| v * cfg.early_exaggeration)
        .collect();
    let explore_iters = EXPLORATION_MAX_ITER.min(cfg.max_iter);
    let (_kl_early, it_early) = device_gradient_descent::<F>(
        pool,
        y,
        &p_early,
        cfg,
        0,
        explore_iters,
        0.5,
        EXPLORATION_MAX_ITER,
    )?;

    let mut it_final = it_early;
    let remaining = cfg.max_iter.saturating_sub(EXPLORATION_MAX_ITER);
    if it_early + 1 < explore_iters || remaining > 0 {
        let (_kl2, it2) = device_gradient_descent::<F>(
            pool,
            y,
            p_joint,
            cfg,
            it_early + 1,
            cfg.max_iter,
            0.8,
            cfg.n_iter_without_progress,
        )?;
        it_final = it2;
    }

    // `kl_divergence_` is ALWAYS against the UN-exaggerated `p_joint` at the
    // final embedding, in every branch — including a fit short enough to end
    // inside the exploration phase, whose own KL is inflated by ~the
    // exaggeration factor.
    let kl = device_kl_divergence::<F>(pool, y, p_joint, cfg.n, cfg.d, cfg.dof)?;
    Ok((kl, it_final))
}

/// Evaluate the KL divergence at `y` through one device objective evaluation.
fn device_kl_divergence<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    y: &[f64],
    p: &[f64],
    n: usize,
    d: usize,
    dof: f64,
) -> Result<f64, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let y_f: Vec<F> = y.iter().map(|&v| f64_to_host::<F>(v)).collect();
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y_f);
    let p_f: Vec<F> = p.iter().map(|&v| f64_to_host::<F>(v)).collect();
    let p_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &p_f);
    let step = tsne_gradient::<F>(pool, &y_dev, &p_dev, n, d, dof).map_err(AlgoError::Prim)?;
    let qnum: Vec<f64> = step
        .qnum
        .to_host(pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    let qsum = step.qsum;
    step.qnum.release_into(pool);
    p_dev.release_into(pool);
    y_dev.release_into(pool);
    let mut kl = 0.0f64;
    for r in 0..n {
        for c in 0..n {
            if r != c {
                let pv = p[r * n + c].max(MACHINE_EPSILON);
                let qv = (qnum[r * n + c] / qsum).max(MACHINE_EPSILON);
                kl += p[r * n + c] * (pv / qv).ln();
            }
        }
    }
    Ok(kl)
}

/// sklearn `_gradient_descent` over the DEVICE objective: per iteration the
/// TSNE-01 prim evaluates the KL gradient on device; the gains/momentum update
/// runs host-side in f64.
#[allow(clippy::too_many_arguments)]
fn device_gradient_descent<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    y: &mut [f64],
    p_host: &[f64],
    cfg: &TsneDescentConfig,
    it_start: usize,
    max_iter: usize,
    momentum: f64,
    n_iter_without_progress: usize,
) -> Result<(f64, usize), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let (n, d) = (cfg.n, cfg.d);
    let nd = n * d;
    let p_f: Vec<F> = p_host.iter().map(|&v| f64_to_host::<F>(v)).collect();
    let p_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &p_f);

    let mut update = vec![0.0f64; nd];
    let mut gains = vec![1.0f64; nd];
    let mut error = f64::MAX;
    let mut best_error = f64::MAX;
    let mut best_iter = it_start;
    let mut i = it_start;

    if it_start >= max_iter {
        p_dev.release_into(pool);
        return Ok((error, it_start.saturating_sub(1)));
    }

    for iter in it_start..max_iter {
        i = iter;
        let check_convergence = (iter + 1) % N_ITER_CHECK == 0 || iter == max_iter - 1;

        let y_f: Vec<F> = y.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y_f);
        let step =
            tsne_gradient::<F>(pool, &y_dev, &p_dev, n, d, cfg.dof).map_err(AlgoError::Prim)?;
        y_dev.release_into(pool);

        if check_convergence {
            let qnum: Vec<f64> = step
                .qnum
                .to_host(pool)
                .iter()
                .map(|&v| host_to_f64(v))
                .collect();
            let mut kl = 0.0f64;
            for r in 0..n {
                for c in 0..n {
                    if r != c {
                        let pv = p_host[r * n + c].max(MACHINE_EPSILON);
                        let qv = (qnum[r * n + c] / step.qsum).max(MACHINE_EPSILON);
                        kl += p_host[r * n + c] * (pv / qv).ln();
                    }
                }
            }
            error = kl;
        }
        step.qnum.release_into(pool);

        let mut grad_norm_sq = 0.0f64;
        for k in 0..nd {
            let g = host_to_f64(step.grad[k]);
            if update[k] * g < 0.0 {
                gains[k] += 0.2;
            } else {
                gains[k] *= 0.8;
            }
            if gains[k] < 0.01 {
                gains[k] = 0.01;
            }
            let gg = g * gains[k];
            grad_norm_sq += gg * gg;
            update[k] = momentum * update[k] - cfg.learning_rate * gg;
            y[k] += update[k];
        }
        let grad_norm = grad_norm_sq.sqrt();

        if check_convergence {
            if error < best_error {
                best_error = error;
                best_iter = iter;
            } else if iter - best_iter > n_iter_without_progress {
                break;
            }
            if grad_norm <= cfg.min_grad_norm {
                break;
            }
        }
    }

    p_dev.release_into(pool);
    Ok((error, i))
}
