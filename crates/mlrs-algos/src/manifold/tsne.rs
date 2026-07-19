//! `TSNE` (TSNE-01) — t-distributed Stochastic Neighbor Embedding,
//! sklearn/cuML-parity, EXACT method.
//!
//! The pipeline mirrors sklearn 1.9.0's `TSNE(method='exact')` stage-for-stage:
//!
//! 1. **Squared input distances** — the Phase-2 `distance(sqrt=false)` prim on
//!    the DEVICE (the O(n²p) input pass), read back once.
//! 2. **Joint probabilities P** — [`joint_probabilities`]: sklearn rounds the
//!    squared distances through **float32** (`distances.astype(np.float32)`)
//!    but runs the per-point perplexity binary search and stores P in
//!    **float64** (`_utils.pyx::_binary_search_perplexity` — verified against
//!    the installed 1.9.0 source). The port reproduces exactly that: f32-cast
//!    distances, f64 search (100 steps, tolerance 1e-5), symmetrize,
//!    normalize by `max(ΣP, MACHINE_EPSILON)`, clamp off-diagonal at
//!    `MACHINE_EPSILON`. This stage is DETERMINISTIC and gated ≤1e-5 against
//!    sklearn's `_joint_probabilities`.
//! 3. **Init** — `init='pca'` (deterministic; the mlrs full-SVD [`Pca`] vs
//!    sklearn's randomized solver — same subspace up to sign, absorbed by the
//!    band gate) scaled to `y / std(y[:, 0]) · 1e-4`, or `init='random'`
//!    (`1e-4 · N(0,1)` from the seeded SplitMix64 — deliberately ≠ MT19937,
//!    the milestone-wide stochastic-gate convention).
//! 4. **Gradient descent** — a verbatim port of sklearn `_gradient_descent`
//!    (gains ±0.2/×0.8 clipped at 0.01, momentum update, `grad_norm` checked
//!    AFTER the gains scaling, error every `n_iter_check=50`), two phases:
//!    early exaggeration (`P·early_exaggeration`, 250 iters, momentum 0.5)
//!    then the main phase (momentum 0.8, `n_iter_without_progress = 300`).
//!    The O(n²) per-iteration Q/gradient runs ON DEVICE via the TSNE-01 prim
//!    (`mlrs_backend::prims::tsne::tsne_gradient`); the O(n·d) update rule is
//!    host-side f64 (the sgd/lbfgs iterative-prim posture).
//!
//! The final embedding is stochastic-adjacent (1000 chaotic iterations), so
//! the end-to-end gate is a BAND (trustworthiness + KL divergence vs the
//! sklearn oracle — the UMAP property-gate convention); the deterministic P
//! stage carries the strict ≤1e-5 value gate.
//!
//! Scope (cuML parity): `metric='euclidean'`, `method='exact'`. Any
//! `n_components >= 1` is accepted (`degrees_of_freedom = max(nc-1, 1)`,
//! the sklearn formula); the default — and the only oracle-gated case — is 2.
//!
//! Tests live in `crates/mlrs-algos/tests/tsne_test.rs` (AGENTS.md §2).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::rng::SplitMix64;
use mlrs_backend::prims::tsne::{squared_distance, tsne_gradient, MACHINE_EPSILON};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64};

use crate::decomposition::pca::Pca;
use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, State, Transform, Unfit};

/// Embedding initialization (sklearn `init`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TsneInit {
    /// Deterministic PCA init (sklearn 1.9's default): project onto the top
    /// `n_components` principal axes, then scale so `std(y[:, 0]) = 1e-4`.
    Pca,
    /// Random init: `1e-4 · N(0, 1)` from the seeded SplitMix64.
    Random,
}

/// The learning-rate specification (sklearn `learning_rate`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearningRate {
    /// sklearn `'auto'`: `max(n_samples / early_exaggeration / 4, 50)`.
    Auto,
    /// An explicit positive step size.
    Value(f64),
}

/// sklearn `_EXPLORATION_MAX_ITER` — the early-exaggeration phase length.
const EXPLORATION_MAX_ITER: usize = 250;
/// sklearn `_N_ITER_CHECK` — error/convergence check cadence.
const N_ITER_CHECK: usize = 50;
/// sklearn `min_gain`.
const MIN_GAIN: f64 = 0.01;
/// sklearn `n_iter_without_progress` (main phase; the exploration phase uses
/// its own full length, i.e. never breaks on progress).
const N_ITER_WITHOUT_PROGRESS: usize = 300;

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
    /// Init strategy (sklearn 1.9 default `'pca'`).
    init: TsneInit,
    /// Seed for the `init='random'` SplitMix64 (sklearn `random_state`).
    seed: u64,
    /// Convergence threshold on the (gains-scaled) gradient norm
    /// (sklearn `min_grad_norm`, default 1e-7).
    min_grad_norm: f64,
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
    /// sklearn 1.9 defaults (D-08 single source): `n_components=2`,
    /// `perplexity=30`, `early_exaggeration=12`, `learning_rate='auto'`,
    /// `max_iter=1000`, `init='pca'`, `min_grad_norm=1e-7`.
    pub fn new() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            early_exaggeration: 12.0,
            learning_rate: LearningRate::Auto,
            max_iter: 1000,
            init: TsneInit::Pca,
            seed: 0,
            min_grad_norm: 1e-7,
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
            init: self.init,
            seed: self.seed,
            min_grad_norm: self.min_grad_norm,
        }
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

/// Builder for [`Tsne`] (data-INDEPENDENT validation at `build`, D-08).
#[derive(Debug, Clone, Copy)]
pub struct TsneBuilder {
    n_components: usize,
    perplexity: f64,
    early_exaggeration: f64,
    learning_rate: LearningRate,
    max_iter: usize,
    init: TsneInit,
    seed: u64,
    min_grad_norm: f64,
}

impl Default for TsneBuilder {
    /// Re-derive the sklearn defaults from [`Tsne::new`] (D-08 single source).
    fn default() -> Self {
        Tsne::<f64, Unfit>::new().into_builder()
    }
}

impl TsneBuilder {
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
    /// Set the init strategy.
    pub fn init(mut self, v: TsneInit) -> Self {
        self.init = v;
        self
    }
    /// Set the `init='random'` seed (sklearn `random_state`).
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }
    /// Set the convergence threshold `min_grad_norm`.
    pub fn min_grad_norm(mut self, v: f64) -> Self {
        self.min_grad_norm = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08):
    /// - `n_components >= 1` ([`BuildError::InvalidNComponents`], `max` is the
    ///   data-dependent bound so `usize::MAX` stands in here),
    /// - `perplexity` finite and `> 0` ([`BuildError::InvalidPerplexity`]),
    /// - `early_exaggeration` finite and `>= 1`
    ///   ([`BuildError::InvalidEarlyExaggeration`] — the sklearn check),
    /// - explicit `learning_rate` finite and `> 0`
    ///   ([`BuildError::InvalidLearningRate`]),
    /// - `max_iter >= 1` ([`BuildError::InvalidMaxIter`]).
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
        Ok(Tsne {
            n_components: self.n_components,
            perplexity: self.perplexity,
            early_exaggeration: self.early_exaggeration,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            init: self.init,
            seed: self.seed,
            min_grad_norm: self.min_grad_norm,
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

    /// Fit: device squared distances → f32-rounded f64 perplexity search → P →
    /// PCA/random init → two-phase device-gradient descent (see module docs).
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

        // --- 1. Squared input distances on device; single read-back. Direct
        //     GATHER (mlrs_backend::prims::tsne::squared_distance), NOT the
        //     GEMM-expansion `distance` prim — its row_reduce(Shared) norm
        //     term is pathologically slow under some host-threading contexts
        //     (see that prim's docs). ---
        let dsq_dev = squared_distance::<F>(pool, x, n, p);
        let dsq: Vec<f64> = dsq_dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        dsq_dev.release_into(pool);

        // --- 2. Joint probabilities P (sklearn-exact; see module docs). ---
        let p_joint = joint_probabilities(&dsq, n, self.perplexity);

        // --- 3. Init embedding (host f64). ---
        let mut y: Vec<f64> = match self.init {
            TsneInit::Pca => {
                let pca = Pca::<F>::builder()
                    .n_components(d)
                    .build::<F>()
                    .expect("PcaBuilder::build is infallible")
                    .fit(pool, x, None, (n, p))?;
                let emb_dev = pca.transform(pool, x, (n, p))?;
                let emb: Vec<f64> =
                    emb_dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
                emb_dev.release_into(pool);
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

        // --- 4. Two-phase gradient descent (sklearn `_tsne`). ---
        let dof = (d as f64 - 1.0).max(1.0);
        let learning_rate = match self.learning_rate {
            LearningRate::Auto => (n as f64 / self.early_exaggeration / 4.0).max(50.0),
            LearningRate::Value(v) => v,
        };

        // Phase 1: early exaggeration — P·ee, momentum 0.5, its own full
        // length as the no-progress window (sklearn passes
        // n_iter_without_progress = _EXPLORATION_MAX_ITER there).
        let p_early: Vec<f64> = p_joint.iter().map(|&v| v * self.early_exaggeration).collect();
        let explore_iters = EXPLORATION_MAX_ITER.min(self.max_iter);
        let (_kl_early, it_early) = gradient_descent::<F>(
            pool,
            &mut y,
            &p_early,
            n,
            d,
            dof,
            0,
            explore_iters,
            0.5,
            learning_rate,
            self.min_grad_norm,
            EXPLORATION_MAX_ITER,
        )?;

        // Phase 2: main — P un-exaggerated, momentum 0.8 (sklearn runs it when
        // the exploration ran to completion or budget remains). `_kl_early`
        // is the KL against the EXAGGERATED P and is NOT the reported value
        // (see the final `kl_divergence` recompute below).
        let mut it_final = it_early;
        let remaining = self.max_iter.saturating_sub(EXPLORATION_MAX_ITER);
        if it_early + 1 < explore_iters || remaining > 0 {
            let (_kl2, it2) = gradient_descent::<F>(
                pool,
                &mut y,
                &p_joint,
                n,
                d,
                dof,
                it_early + 1,
                self.max_iter,
                0.8,
                learning_rate,
                self.min_grad_norm,
                N_ITER_WITHOUT_PROGRESS,
            )?;
            it_final = it2;
        }

        // `kl_divergence_` is ALWAYS the KL against the UN-exaggerated `p_joint`
        // at the final embedding (sklearn's `kl_divergence_` contract). Phase
        // 2's own returned KL would serve when it runs, but when the whole fit
        // fits inside the exploration phase (`max_iter <= EXPLORATION_MAX_ITER`)
        // phase 2 is skipped and phase 1's KL is against `P·early_exaggeration`
        // — inflated by ~the exaggeration factor. Recomputing here (one extra
        // device objective evaluation) makes the reported value correct in
        // every branch.
        let kl = kl_divergence::<F>(pool, &y, &p_joint, n, d, dof)?;

        let y_f: Vec<F> = y.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let embedding_ = DeviceArray::from_host(pool, &y_f);

        Ok(Tsne {
            n_components: self.n_components,
            perplexity: self.perplexity,
            early_exaggeration: self.early_exaggeration,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            init: self.init,
            seed: self.seed,
            min_grad_norm: self.min_grad_norm,
            embedding_: Some(embedding_),
            kl_divergence_: Some(kl),
            n_iter_: it_final,
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
pub fn joint_probabilities(dsq: &[f64], n: usize, perplexity: f64) -> Vec<f64> {
    debug_assert_eq!(dsq.len(), n * n);
    const EPSILON_DBL: f64 = 1e-8;
    const PERPLEXITY_TOLERANCE: f64 = 1e-5;
    const N_STEPS: usize = 100;

    // sklearn: distances.astype(np.float32) — the ONLY f32 rounding.
    let d32: Vec<f32> = dsq.iter().map(|&v| v as f32).collect();

    let desired_entropy = perplexity.ln();
    let mut cond = vec![0.0f64; n * n];

    for i in 0..n {
        let mut beta_min = f64::NEG_INFINITY;
        let mut beta_max = f64::INFINITY;
        let mut beta = 1.0f64;

        for _ in 0..N_STEPS {
            let mut sum_pi = 0.0f64;
            for j in 0..n {
                if j != i {
                    let pij = (-(d32[i * n + j] as f64) * beta).exp();
                    cond[i * n + j] = pij;
                    sum_pi += pij;
                }
            }
            if sum_pi == 0.0 {
                sum_pi = EPSILON_DBL;
            }
            let mut sum_disti_pi = 0.0f64;
            for j in 0..n {
                cond[i * n + j] /= sum_pi;
                sum_disti_pi += (d32[i * n + j] as f64) * cond[i * n + j];
            }
            let entropy = sum_pi.ln() + beta * sum_disti_pi;
            let entropy_diff = entropy - desired_entropy;
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
        // The diagonal was never written this row (j != i skips it), but the
        // /= sum_pi pass touches it — force the sklearn zero.
        cond[i * n + i] = 0.0;
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

/// Evaluate the KL divergence `Σ_{i≠j} p_ij · log(p_ij / q_ij)` of the joint
/// probabilities `p` against the Student-t affinities `q` at embedding `y`
/// (one device objective evaluation via [`tsne_gradient`], reusing its
/// `qnum`/`qsum`). The same clamped form the in-loop `compute_error` uses, so
/// the standalone final value is consistent with the convergence checks.
fn kl_divergence<F>(
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
    let qnum: Vec<f64> = step.qnum.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
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

/// Verbatim port of sklearn `_gradient_descent` over the device objective:
/// per iteration the TSNE-01 prim evaluates the KL gradient on device; the
/// gains/momentum update runs host-side in f64. Returns `(error, iter)` —
/// the KL at the last error-check and the last iteration index run.
///
/// `p_host` is the (possibly exaggerated) dense joint P; it is uploaded to the
/// device ONCE per call. `y` is updated in place.
#[allow(clippy::too_many_arguments)]
fn gradient_descent<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    y: &mut [f64],
    p_host: &[f64],
    n: usize,
    d: usize,
    dof: f64,
    it_start: usize,
    max_iter: usize,
    momentum: f64,
    learning_rate: f64,
    min_grad_norm: f64,
    n_iter_without_progress: usize,
) -> Result<(f64, usize), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let nd = n * d;
    let p_f: Vec<F> = p_host.iter().map(|&v| f64_to_host::<F>(v)).collect();
    let p_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &p_f);

    // sklearn: update = zeros, gains = ones, best_error = MAX, best_iter = it.
    let mut update = vec![0.0f64; nd];
    let mut gains = vec![1.0f64; nd];
    let mut error = f64::MAX;
    let mut best_error = f64::MAX;
    let mut best_iter = it_start;
    let mut i = it_start;

    if it_start >= max_iter {
        // Nothing to run (a tiny max_iter budget) — keep sklearn's "return
        // current state" behavior.
        p_dev.release_into(pool);
        return Ok((error, it_start.saturating_sub(1)));
    }

    for iter in it_start..max_iter {
        i = iter;
        let check_convergence = (iter + 1) % N_ITER_CHECK == 0 || iter == max_iter - 1;

        // Upload the current embedding; evaluate the device objective.
        let y_f: Vec<F> = y.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y_f);
        let step = tsne_gradient::<F>(pool, &y_dev, &p_dev, n, d, dof)
            .map_err(AlgoError::Prim)?;
        y_dev.release_into(pool);

        // KL error only on check iterations (sklearn compute_error).
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

        // sklearn update rule (f64 host):
        //   inc = update * grad < 0; gains[inc] += .2; gains[!inc] *= .8;
        //   clip(min_gain); grad *= gains; update = momentum·update − lr·grad;
        //   p += update.  grad_norm is the norm of the GAINED grad.
        let mut grad_norm_sq = 0.0f64;
        for k in 0..nd {
            let g = host_to_f64(step.grad[k]);
            if update[k] * g < 0.0 {
                gains[k] += 0.2;
            } else {
                gains[k] *= 0.8;
            }
            if gains[k] < MIN_GAIN {
                gains[k] = MIN_GAIN;
            }
            let gg = g * gains[k];
            grad_norm_sq += gg * gg;
            update[k] = momentum * update[k] - learning_rate * gg;
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
            if grad_norm <= min_grad_norm {
                break;
            }
        }
    }

    p_dev.release_into(pool);
    Ok((error, i))
}
