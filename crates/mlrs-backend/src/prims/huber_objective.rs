//! Huber-regression primal objective evaluator (HUBER-01) — the one function
//! the `HuberRegressor` L-BFGS solve calls per iteration and line-search step.
//!
//! ## The objective
//! `HuberRegressor` minimizes, jointly over the coefficients `w`, the intercept
//! `c` and the SCALE `σ > 0` (sklearn `scale_`),
//!
//! ```text
//!   L(w, c, σ) = σ·Σᵢ swᵢ
//!              + Σ_{|rᵢ| ≤ ε·σ}  swᵢ·rᵢ²/σ
//!              + Σ_{|rᵢ| >  ε·σ} swᵢ·(2·ε·|rᵢ| − ε²·σ)
//!              + α·‖w‖²                            with rᵢ = yᵢ − x̃ᵢ·w̃.
//! ```
//!
//! This is the perspective transform of the Huber loss, so it is JOINTLY convex
//! on `σ > 0` and has a unique minimizer. The `σ` parameter is what makes the
//! estimator scale-equivariant in `y`: rescaling `y` rescales `σ` and leaves the
//! outlier classification alone, which is why `ε` can carry the fixed default
//! 1.35 (`sklearn/linear_model/_huber.py`).
//!
//! ## What one evaluation actually needs from the design
//! Exactly the same two things the linear-SVM objective needs
//! ([`svm_objective`](super::svm_objective), whose structure this file mirrors
//! deliberately):
//!
//! 1. the **margins** `m = x̃·w̃` (an `n`-length matvec), and
//! 2. the **data-term gradient** `x̃ᵀ·g` where `gᵢ = ∂L/∂mᵢ` (a `d_aug`-length
//!    transposed matvec),
//!
//! plus THREE extra scalar reductions the SVM does not have, because `σ` is a
//! fitted parameter rather than a constant: `Σ_{inlier} swᵢ·rᵢ²`,
//! `Σ_{outlier} swᵢ·|rᵢ|` and `Σ_{outlier} swᵢ`. All five come out of ONE pass
//! over the design ([`HuberEval`]); the estimator assembles the loss, the `α`
//! penalty and `∂L/∂σ` from them in `O(d)` host arithmetic.
//!
//! ## Why this beats scikit-learn's evaluation
//! `_huber_loss_and_gradient` is written in NumPy, and the vectorized form makes
//! it walk the `n × d` design **five** times per evaluation with two full-size
//! allocations in the middle:
//!
//! | # | sklearn step | cost |
//! |---|---|---|
//! | 1 | `safe_sparse_dot(X, w)` | one `n·d` pass (BLAS) |
//! | 2 | `axis0_safe_slice(X, ~outliers_mask, …)` | fancy-index COPY of the inlier rows |
//! | 3 | `safe_sparse_dot(weighted_non_outliers, X_non_outliers)` | pass over that copy |
//! | 4 | `axis0_safe_slice(X, outliers_mask, …)` | fancy-index COPY of the outlier rows |
//! | 5 | `safe_sparse_dot(sw_outliers, X_outliers)` | pass over that copy |
//!
//! Steps 2 and 4 together allocate and write another whole `n × d` of memory on
//! EVERY objective evaluation, and steps 3/5 then read it back. The row loop
//! here computes `mᵢ`, classifies the sample, and accumulates its gradient
//! contribution while the row is still in L1 — the design is streamed ONCE, with
//! no allocation, split across a persistent [`WorkerPool`].
//!
//! ### Measured
//! `HuberRegressor.fit` wall-clock, 16-core Zen5, f64, min over 5 fits, each
//! engine in its OWN process (`scripts/bench_huber.py --engine …` — the
//! OpenBLAS-spin caveat recorded in `mlrs-cpu-bench-separate-processes`):
//!
//! | n × d | mlrs | scikit-learn 1.9.0 | |
//! |---|---|---|---|
//! | 1 000 × 8 | **0.2 ms** | 16.3 ms | 81x |
//! | 10 000 × 8 | **1.4 ms** | 17.6 ms | 13x |
//! | 10 000 × 64 | **7.0 ms** | 388 ms | 55x |
//! | 100 000 × 16 | **15.7 ms** | 590 ms | 38x |
//! | 100 000 × 64 | **52.0 ms** | 681 ms | 13x |
//! | 50 000 × 128 | **221 ms** | 1 162 ms | 5.2x |
//! | 200 000 × 32 | **247 ms** | 1 525 ms | 6.2x |
//!
//! Both engines land on the same answer (max |Δcoef| 5e-8 … 1.5e-6, identical
//! outlier masks, `scale_` agreeing to six digits) while mlrs runs a FEW MORE
//! iterations — it stops at `ftol = 64·eps` where scikit-learn stops at scipy's
//! `factr = 1e7`. The extra steps buy accuracy; see `huber.rs`.
//!
//! ### What the fused pass actually achieves
//! With `MLRS_HUBER_PROBE=1` the solve reports its EVALUATION count, which is
//! what the time divides by (an outer iteration costs one evaluation plus the
//! line search's — two designs of the same geometry were measured 3x apart on
//! wall-clock at an identical `n_iter_` purely because one needed far more
//! line-search steps). Per evaluation, on the release build:
//!
//! | n × d | evals | µs/eval | design read | effective |
//! |---|---|---|---|---|
//! | 10 000 × 64 | 36 | 81 | 5.1 MB | 63 GB/s |
//! | 100 000 × 16 | 82 | 334 | 12.8 MB | 38 GB/s |
//! | 100 000 × 64 | 38 | 1 327 | 51 MB | 39 GB/s |
//! | 50 000 × 128 | 42 | 1 012 | 51 MB | 50 GB/s |
//!
//! i.e. the pass runs at DRAM bandwidth, which is the ceiling for a single
//! streaming read of the design — there is no arithmetic left to remove, only
//! passes, and the count is already one.
//!
//! ## Why the cpu backend does NOT go through the device GEMM
//! Verbatim the [`svm_objective`](super::svm_objective) reasoning: `cubecl-cpu`
//! JITs at LLVM `-O0` and spawns one OS thread per unit, so a GPU-shaped GEMM
//! launch costs three orders of magnitude more than the matvec it performs. The
//! cpu arm therefore evaluates on the host, compiled into this crate at the
//! release profile's `-O3`, split across [`host_units`] workers that are spawned
//! ONCE per solve rather than once per evaluation.
//!
//! ## Accumulation precision (host arm)
//! The host arm accumulates in **`f64` regardless of `F`**. Same reason as the
//! SVM objective — an `f32` gradient sum over `n` samples carries a round-off
//! floor near `√n·ε_f32`, which at `n ≥ 1000` sits above the `tol = 1e-5`
//! stopping gradient and would let an `f32` solve miss its own tolerance — with
//! one Huber-specific sharpening: `∂L/∂σ` is the difference of two `O(n)`
//! quantities that nearly cancel at the optimum, so it is the FIRST gradient
//! entry to be destroyed by narrow accumulation.
//!
//! Tests live in `crates/mlrs-backend/tests/huber_objective_test.rs`
//! (AGENTS.md §2), never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

#[cfg(not(feature = "cpu"))]
use std::marker::PhantomData;

use mlrs_core::PrimError;
// Both conversions are device-arm-only: the cpu arm's fused pass is
// monomorphized on the concrete `f32`/`f64` element and never round-trips a
// value through the generic `F` bit-cast.
#[cfg(not(feature = "cpu"))]
use mlrs_core::{f64_to_host, host_to_f64};

use crate::device_array::DeviceArray;
#[cfg(feature = "cpu")]
use crate::prims::host_pool::{Shared, WorkerPool};
#[cfg(feature = "cpu")]
use crate::prims::host_simd::avx2_available;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// What ONE objective evaluation produces — five reductions out of one pass.
///
/// The caller adds the `σ·Σsw` term, the `α‖w‖²` penalty and their derivatives:
/// those are `O(d)` host arithmetic over values the evaluator never sees.
#[derive(Debug, Clone)]
pub struct HuberEval {
    /// `Σ_{|rᵢ| ≤ ε·σ} swᵢ·rᵢ²` — sklearn's `weighted_loss`, i.e. its
    /// `squared_loss · σ`. Kept UNDIVIDED so the caller can form both
    /// `squared_loss = sq_sum/σ` and `∂/∂σ = −sq_sum/σ²` without a second
    /// rounding.
    pub sq_sum: f64,
    /// `Σ_{|rᵢ| > ε·σ} swᵢ·|rᵢ|` — the outliers' linear term.
    pub out_abs_sum: f64,
    /// `Σ_{|rᵢ| > ε·σ} swᵢ` — sklearn's `n_sw_outliers`.
    pub out_sw_sum: f64,
    /// Number of samples classified as outliers (sklearn's `num_outliers`).
    pub n_outliers: usize,
    /// `x̃ᵀ·g`, length `d_aug` (`n_features + 1` when fitting an intercept).
    /// Entry `n_features`, when present, is `Σᵢ gᵢ` — the intercept gradient.
    pub xtg: Vec<f64>,
}

/// Whether [`HuberDesign::Host`] is the CHEAPER operand form on this backend.
///
/// True exactly where the objective evaluates from host memory (cpu), which is
/// where handing the caller's slab over directly removes three full passes over
/// it. On the device backends the design has to reach the GEMM operand anyway.
/// The [`svm_host_ingress_preferred`](super::svm_objective::svm_host_ingress_preferred)
/// precedent, and the same contract: both forms are always ACCEPTED, this only
/// says which one to prefer.
pub fn huber_host_ingress_preferred() -> bool {
    cfg!(feature = "cpu")
}

/// Where the design the evaluator reads comes from — see
/// [`SvmDesign`](super::svm_objective::SvmDesign), whose contract this repeats.
pub enum HuberDesign<'a, F> {
    /// Device-resident, `n × d` row-major — the `Fit` trait's operand.
    Device(&'a DeviceArray<ActiveRuntime, F>),
    /// Host-resident, `n × d` row-major — the caller's own buffer.
    Host(&'a [F]),
}

impl<F: Float + CubeElement + Pod> HuberDesign<'_, F> {
    /// Element count, whichever form this is.
    fn len(&self) -> usize {
        match self {
            HuberDesign::Device(x) => x.len(),
            HuberDesign::Host(x) => x.len(),
        }
    }
}

/// The cpu arm's design: the caller's own buffer when it was already host
/// resident, an owned copy only when it had to be pulled off the device.
#[cfg(feature = "cpu")]
enum HostDesign<'a, F> {
    /// The caller's slab, read in place (the no-upload ingress).
    Borrowed(&'a [F]),
    /// Pulled off the device because that is where the caller had it.
    Owned(Vec<F>),
}

#[cfg(feature = "cpu")]
impl<F> HostDesign<'_, F> {
    #[inline]
    fn as_slice(&self) -> &[F] {
        match self {
            HostDesign::Borrowed(s) => s,
            HostDesign::Owned(v) => v,
        }
    }
}

/// The design matrix in whatever form the active backend evaluates against,
/// prepared ONCE per `fit` and reused by every L-BFGS iteration and line-search
/// step (the bounded-allocation iterative-solver shape).
///
/// - **cpu**: the host `n × d` slab, read in place (borrowed outright when the
///   caller supplied [`HuberDesign::Host`]), plus the [`WorkerPool`] the fused
///   pass runs on.
/// - **wgpu / cuda / rocm**: the device-resident `n × d_aug` augmented design
///   the two GEMMs read.
pub struct HuberObjective<'a, F> {
    /// Sample count.
    n: usize,
    /// UNaugmented feature count. cpu arm only: the device arms materialize the
    /// synthetic column into the operand at construction.
    #[cfg(feature = "cpu")]
    d: usize,
    /// Augmented weight length: `d + 1` when fitting an intercept, else `d`.
    d_aug: usize,
    /// Per-sample regression targets, length `n`, host-resident because the loss
    /// is classified on the host on every backend.
    targets: Vec<f64>,
    /// Per-sample weights, length `n`. `None` is the unweighted fit, which takes
    /// a separate monomorphization of the row loop rather than multiplying by a
    /// vector of ones (`WEIGHTED` const generic).
    weights: Option<Vec<f64>>,
    /// `Σᵢ swᵢ` — sklearn's `n_samples = np.sum(sample_weight)`, which is the
    /// `σ` coefficient in the loss and the leading term of `∂L/∂σ`. Summed once
    /// at construction because it never changes across the solve.
    sw_total: f64,
    /// cpu arm: the design read in place, unaugmented — BORROWED from the caller
    /// when it was already host-resident, owned only when it had to be pulled
    /// off the device.
    #[cfg(feature = "cpu")]
    x_host: HostDesign<'a, F>,
    /// cpu arm: the threads the fused pass is split across, spawned ONCE for the
    /// whole solve. `None` below the parallel knee ([`HUBER_ELEMS_PER_UNIT`]),
    /// where the pass runs inline on the calling thread.
    #[cfg(feature = "cpu")]
    workers: Option<WorkerPool>,
    /// device arms: the augmented `n × d_aug` design both GEMMs read.
    #[cfg(not(feature = "cpu"))]
    x_aug: DeviceArray<ActiveRuntime, F>,
    /// Binds `'a` on the device arms, which borrow nothing.
    #[cfg(not(feature = "cpu"))]
    _borrow: PhantomData<&'a [F]>,
}

impl<'a, F> HuberObjective<'a, F>
where
    F: Float + CubeElement + Pod,
{
    /// Prepare the evaluator for an `n × d` row-major design.
    ///
    /// `targets` is length `n`; `weights` is `None` (unweighted) or length `n`
    /// with every entry finite and non-negative — the estimator validates the
    /// VALUES, this validates the GEOMETRY. `fit_intercept` appends the
    /// synthetic unit column (sklearn's Huber intercept is a plain unit column,
    /// NOT the SVM's `intercept_scaling` mechanism, so the constant is exactly
    /// `1.0`).
    ///
    /// Geometry is validated before anything is allocated (ASVS V5): `n·d ==
    /// x.len()`, `targets.len() == n`, `weights` length `n` when present, both
    /// dims non-zero.
    pub fn new(
        pool: &mut BufferPool<ActiveRuntime>,
        x: HuberDesign<'a, F>,
        (n, d): (usize, usize),
        targets: Vec<f64>,
        weights: Option<Vec<f64>>,
        fit_intercept: bool,
    ) -> Result<Self, PrimError> {
        if n == 0 || d == 0 || n.checked_mul(d).map(|v| v != x.len()).unwrap_or(true) {
            return Err(PrimError::ShapeMismatch {
                operand: "x",
                rows: n,
                cols: d,
                len: x.len(),
            });
        }
        if targets.len() != n {
            return Err(PrimError::ShapeMismatch {
                operand: "targets",
                rows: n,
                cols: 1,
                len: targets.len(),
            });
        }
        if let Some(w) = weights.as_ref() {
            if w.len() != n {
                return Err(PrimError::ShapeMismatch {
                    operand: "sample_weight",
                    rows: n,
                    cols: 1,
                    len: w.len(),
                });
            }
        }
        let d_aug = if fit_intercept { d + 1 } else { d };
        let sw_total = match weights.as_ref() {
            Some(w) => w.iter().sum(),
            None => n as f64,
        };

        #[cfg(feature = "cpu")]
        {
            let x_host = match x {
                HuberDesign::Host(slab) => HostDesign::Borrowed(slab),
                HuberDesign::Device(dev) => HostDesign::Owned(dev.to_host(pool)),
            };
            // Spawned ONCE for the whole solve: the L-BFGS driver calls `eval`
            // dozens of times and `std::thread` setup dominated every one of
            // them (see [`HUBER_ELEMS_PER_UNIT`]).
            let units = host_units(n * d).min(n.max(1));
            Ok(Self {
                n,
                d,
                d_aug,
                targets,
                weights,
                sw_total,
                x_host,
                workers: (units > 1).then(|| WorkerPool::new(units)),
            })
        }
        #[cfg(not(feature = "cpu"))]
        {
            // The device arms feed `prims::gemm`, which reads one contiguous
            // operand — so the synthetic column has to be materialized here.
            let x_host: Vec<F> = match x {
                HuberDesign::Device(dev) => dev.to_host(pool),
                HuberDesign::Host(slab) => slab.to_vec(),
            };
            let mut aug: Vec<F> = vec![f64_to_host::<F>(0.0); n * d_aug];
            for r in 0..n {
                aug[r * d_aug..r * d_aug + d].copy_from_slice(&x_host[r * d..(r + 1) * d]);
                if fit_intercept {
                    aug[r * d_aug + d] = f64_to_host::<F>(1.0);
                }
            }
            Ok(Self {
                n,
                d_aug,
                targets,
                weights,
                sw_total,
                x_aug: DeviceArray::from_host(pool, &aug),
                _borrow: PhantomData,
            })
        }
    }

    /// The augmented weight length the caller's `w` must have (`n_features + 1`
    /// when fitting an intercept).
    pub fn d_aug(&self) -> usize {
        self.d_aug
    }

    /// `Σᵢ swᵢ` — sklearn's `n_samples`, the `σ` coefficient of the loss.
    pub fn sw_total(&self) -> f64 {
        self.sw_total
    }

    /// Sample count.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Evaluate the five reductions of [`HuberEval`] at the augmented weights
    /// `w` (length [`d_aug`](Self::d_aug)) and the scale `sigma`.
    ///
    /// `sigma` must be strictly positive: the loss divides by it. The estimator
    /// rejects the infeasible half-line with a `+∞` barrier BEFORE calling here
    /// (`huber.rs`), which is what keeps the line search off it, so this
    /// function does not re-check a value only its own caller can produce.
    ///
    /// Generic over nothing: unlike the SVM objective there is exactly one loss
    /// here, so the classification branch is written directly into the row loop
    /// where LLVM can turn it into a select rather than a call.
    pub fn eval(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        w: &[f64],
        sigma: f64,
        epsilon: f64,
    ) -> Result<HuberEval, PrimError> {
        if w.len() != self.d_aug {
            return Err(PrimError::DimMismatch {
                dim: "d_aug",
                lhs: w.len(),
                rhs: self.d_aug,
            });
        }
        #[cfg(feature = "cpu")]
        {
            let _ = pool;
            Ok(self.eval_host(w, sigma, epsilon))
        }
        #[cfg(not(feature = "cpu"))]
        {
            self.eval_device(pool, w, sigma, epsilon)
        }
    }

    /// Release the operand back to the pool, consuming the evaluator.
    pub fn release_into(self, pool: &mut BufferPool<ActiveRuntime>) {
        #[cfg(feature = "cpu")]
        {
            let _ = pool;
        }
        #[cfg(not(feature = "cpu"))]
        {
            self.x_aug.release_into(pool);
        }
    }
}

// ---------------------------------------------------------------------------
// cpu arm — the fused host pass
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu")]
impl<F> HuberObjective<'_, F>
where
    F: Float + CubeElement + Pod,
{
    /// One fused pass over the design (module docs), split across the pool.
    ///
    /// Two dispatches happen here, both for the same reason the SVM objective
    /// dispatches once: the row loop is monomorphized on the CONCRETE element
    /// type so it vectorizes (rather than going through `host_to_f64` element by
    /// element), and on whether the fit is WEIGHTED so the unweighted fit — the
    /// overwhelmingly common one — never loads a vector of ones.
    fn eval_host(&self, w: &[f64], sigma: f64, epsilon: f64) -> HuberEval {
        match (size_of::<F>(), self.weights.is_some()) {
            (4, false) => self.eval_host_typed::<f32, false>(w, sigma, epsilon),
            (4, true) => self.eval_host_typed::<f32, true>(w, sigma, epsilon),
            (8, false) => self.eval_host_typed::<f64, false>(w, sigma, epsilon),
            (8, true) => self.eval_host_typed::<f64, true>(w, sigma, epsilon),
            (other, _) => {
                unreachable!("huber is f32/f64 only, got a {other}-byte element")
            }
        }
    }

    fn eval_host_typed<T: HostElem, const WEIGHTED: bool>(
        &self,
        w: &[f64],
        sigma: f64,
        epsilon: f64,
    ) -> HuberEval {
        let x: &[T] = bytemuck::cast_slice(self.x_host.as_slice());
        let (n, d, d_aug) = (self.n, self.d, self.d_aug);
        // The synthetic column is a CONSTANT 1.0, so it contributes `w[d]` to
        // every margin and is hoisted out of the row loop entirely.
        let bias = if d_aug > d { w[d] } else { 0.0 };
        let wd = &w[..d];
        // The three scalars every row needs, formed once: the outlier threshold
        // and the two gradient prefactors.
        let cfg = RowCfg {
            thr: epsilon * sigma,
            inlier_scale: -2.0 / sigma,
            outlier_scale: 2.0 * epsilon,
            bias,
        };
        // `None` weights are never indexed under `WEIGHTED == false`; the empty
        // slice keeps the two monomorphizations sharing one signature.
        let sw: &[f64] = self.weights.as_deref().unwrap_or(&[]);

        let Some(workers) = self.workers.as_ref() else {
            let mut acc = Accum::new(d, d_aug);
            acc.rows_dispatch::<T, WEIGHTED>(x, wd, &self.targets, sw, &cfg);
            return acc.finish(d_aug > d);
        };

        // Contiguous row chunks: unit `u` owns rows `[u·rows, u·rows + k)`, so
        // its design slab, its target run and its weight run are all unbroken
        // ranges.
        let units = workers.units();
        let rows = n.div_ceil(units);
        let mut partials: Vec<Accum> = (0..units).map(|_| Accum::new(d, d_aug)).collect();
        {
            // SAFETY (`Shared`'s contract): unit `u` is the only writer of
            // `partials[u]` within the pass, and `run` does not return until
            // every unit has finished — the barrier release is what publishes
            // the writes to the reducing thread below.
            let slots = Shared::new(&mut partials);
            // Bound out of `self` rather than captured through it: the pool is
            // deliberately `!Sync` (one driver per pool), so a closure holding
            // `&self` — which owns the pool — could not itself be `Sync`.
            let targets = &self.targets;
            workers.run(&|u: usize| {
                let lo = (u * rows).min(n);
                let hi = (lo + rows).min(n);
                if lo == hi {
                    return;
                }
                let acc = unsafe { &mut slots.get_mut()[u] };
                let sw_chunk = if WEIGHTED { &sw[lo..hi] } else { sw };
                acc.rows_dispatch::<T, WEIGHTED>(
                    &x[lo * d..hi * d],
                    wd,
                    &targets[lo..hi],
                    sw_chunk,
                    &cfg,
                );
            });
        }

        let mut total = Accum::new(d, d_aug);
        for p in partials {
            total.sq_sum += p.sq_sum;
            total.out_abs_sum += p.out_abs_sum;
            total.out_sw_sum += p.out_sw_sum;
            total.n_outliers += p.n_outliers;
            total.gsum += p.gsum;
            for (t, v) in total.xtg.iter_mut().zip(p.xtg.iter()) {
                *t += *v;
            }
        }
        total.finish(d_aug > d)
    }

    /// The outlier mask sklearn exposes as `outliers_`: `|yᵢ − x̃ᵢ·w̃| > σ·ε`,
    /// evaluated at the FITTED parameters.
    ///
    /// This is the same predicate the row loop classifies on, but it is a
    /// separate (single, unfused) pass because it runs exactly ONCE per fit —
    /// folding a length-`n` mask write into the per-iteration evaluation would
    /// cost an `n`-byte store on every one of the dozens of evaluations to save
    /// one pass at the end.
    fn outlier_mask_host(&self, w: &[f64], sigma: f64, epsilon: f64) -> Vec<bool> {
        match size_of::<F>() {
            4 => self.outlier_mask_typed::<f32>(w, sigma, epsilon),
            8 => self.outlier_mask_typed::<f64>(w, sigma, epsilon),
            other => unreachable!("huber is f32/f64 only, got a {other}-byte element"),
        }
    }

    fn outlier_mask_typed<T: HostElem>(&self, w: &[f64], sigma: f64, epsilon: f64) -> Vec<bool> {
        let x: &[T] = bytemuck::cast_slice(self.x_host.as_slice());
        let (n, d, d_aug) = (self.n, self.d, self.d_aug);
        let bias = if d_aug > d { w[d] } else { 0.0 };
        let wd = &w[..d];
        let thr = epsilon * sigma;
        (0..n)
            .map(|r| {
                let margin = T::dot(&x[r * d..(r + 1) * d], wd) + bias;
                (self.targets[r] - margin).abs() > thr
            })
            .collect()
    }
}

/// The per-row constants the fused loop needs, formed once per evaluation.
#[cfg(feature = "cpu")]
struct RowCfg {
    /// `ε·σ` — the outlier threshold on `|rᵢ|`.
    thr: f64,
    /// `−2/σ` — an inlier's gradient prefactor (`gᵢ = −2·swᵢ·rᵢ/σ`).
    inlier_scale: f64,
    /// `2·ε` — an outlier's gradient magnitude (`gᵢ = −2·ε·swᵢ·sign(rᵢ)`).
    outlier_scale: f64,
    /// The hoisted synthetic-column contribution `w[d]` to every margin.
    bias: f64,
}

/// One worker's partial reductions, plus the separate `Σgᵢ` the synthetic
/// column's gradient entry IS (the column is a constant 1.0, so tracking the
/// plain sum keeps it out of the inner loop).
#[cfg(feature = "cpu")]
struct Accum {
    sq_sum: f64,
    out_abs_sum: f64,
    out_sw_sum: f64,
    n_outliers: usize,
    gsum: f64,
    /// Length `d_aug`; entry `d` (when present) is filled at [`Accum::finish`].
    xtg: Vec<f64>,
    /// Unaugmented feature count — where the synthetic entry lands.
    d: usize,
}

#[cfg(feature = "cpu")]
impl Accum {
    fn new(d: usize, d_aug: usize) -> Self {
        Self {
            sq_sum: 0.0,
            out_abs_sum: 0.0,
            out_sw_sum: 0.0,
            n_outliers: 0,
            gsum: 0.0,
            xtg: vec![0.0; d_aug],
            d,
        }
    }

    /// The fused row loop: `mᵢ = xᵢ·w + bias`, classify `rᵢ = tᵢ − mᵢ` against
    /// `ε·σ`, accumulate the matching scalar reduction, then `xtg += gᵢ·xᵢ`.
    /// Both uses of row `i` happen while it is in L1, so the design is streamed
    /// once (module docs).
    ///
    /// `WEIGHTED == false` never touches `sw` at all — the unweighted fit is the
    /// common one and a vector of ones would double the loop's load traffic for
    /// nothing. Accumulates in `f64` whatever `T` is (module docs).
    /// [`Accum::rows`] on the machine's REAL vector unit.
    ///
    /// The row loop's `axpy` half is a straight elementwise update over `d`
    /// independent accumulators, which is exactly the shape a wider register
    /// halves the instruction count of; the crate itself is compiled for the
    /// x86-64 baseline. Nothing about the arithmetic changes — see
    /// [`host_simd`](super::host_simd) for why, and for why this is an explicit
    /// twin rather than a closure helper.
    #[inline]
    fn rows_dispatch<T: HostElem, const WEIGHTED: bool>(
        &mut self,
        x: &[T],
        w: &[f64],
        targets: &[f64],
        sw: &[f64],
        cfg: &RowCfg,
    ) {
        #[cfg(target_arch = "x86_64")]
        if avx2_available() {
            // SAFETY: guarded by the runtime detection this branch tests; the
            // body is the ordinary `rows`, which contains nothing unsafe.
            unsafe {
                self.rows_avx2::<T, WEIGHTED>(x, w, targets, sw, cfg);
            }
            return;
        }
        self.rows::<T, WEIGHTED>(x, w, targets, sw, cfg);
    }

    /// [`Accum::rows`] compiled for AVX2 + FMA — see [`Accum::rows_dispatch`].
    ///
    /// # Safety
    /// The caller must have established that the CPU supports `avx2` and `fma`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn rows_avx2<T: HostElem, const WEIGHTED: bool>(
        &mut self,
        x: &[T],
        w: &[f64],
        targets: &[f64],
        sw: &[f64],
        cfg: &RowCfg,
    ) {
        self.rows::<T, WEIGHTED>(x, w, targets, sw, cfg);
    }

    #[inline(always)]
    fn rows<T: HostElem, const WEIGHTED: bool>(
        &mut self,
        x: &[T],
        w: &[f64],
        targets: &[f64],
        sw: &[f64],
        cfg: &RowCfg,
    ) {
        let d = self.d;
        let g_acc = &mut self.xtg[..d];
        for (r, &t) in targets.iter().enumerate() {
            let row = &x[r * d..(r + 1) * d];
            let margin = T::dot(row, w) + cfg.bias;
            let res = t - margin;
            let a = res.abs();
            let s = if WEIGHTED { sw[r] } else { 1.0 };
            // sklearn's mask is STRICTLY greater (`abs_linear_loss > epsilon *
            // sigma`), so a residual exactly ON the threshold is an inlier.
            let g = if a > cfg.thr {
                self.out_abs_sum += s * a;
                self.out_sw_sum += s;
                self.n_outliers += 1;
                // sign(rᵢ) with sign(0) = +1, matching sklearn's
                // `np.ones_like` + `< 0` mask (a zero residual can never be an
                // outlier anyway, so the tie only has to be well-defined).
                if res < 0.0 {
                    cfg.outlier_scale * s
                } else {
                    -cfg.outlier_scale * s
                }
            } else {
                self.sq_sum += s * res * res;
                cfg.inlier_scale * s * res
            };
            self.gsum += g;
            T::axpy(g_acc, row, g);
        }
    }

    /// Fold the hoisted synthetic-column entry in and hand back the result.
    fn finish(mut self, fit_intercept: bool) -> HuberEval {
        if fit_intercept {
            self.xtg[self.d] = self.gsum;
        }
        HuberEval {
            sq_sum: self.sq_sum,
            out_abs_sum: self.out_abs_sum,
            out_sw_sum: self.out_sw_sum,
            n_outliers: self.n_outliers,
            xtg: self.xtg,
        }
    }
}

/// The concrete host element types the fused pass is monomorphized over.
///
/// Like `linear_predict`'s `HostFloat` and `svm_objective`'s `HostElem` this
/// deliberately avoids `mul_add`: without `target-feature=+fma` (which the
/// default `x86-64` baseline lacks) `mul_add` lowers to a LIBRARY CALL, an order
/// of magnitude slower than the `mul`+`add` pair LLVM vectorizes here.
#[cfg(feature = "cpu")]
trait HostElem: Pod + Copy + Send + Sync {
    /// `Σⱼ row[j]·w[j]` in `f64`, over [`DOT_LANES`] independent accumulators.
    fn dot(row: &[Self], w: &[f64]) -> f64;

    /// `acc[j] += g·row[j]` in `f64`.
    fn axpy(acc: &mut [f64], row: &[Self], g: f64);
}

/// Design elements one worker must be given before splitting the pass pays.
///
/// The value is inherited from [`svm_objective`](super::svm_objective)'s
/// `SVM_ELEMS_PER_UNIT` — the same shape of pass, one fused streaming read of
/// the design behind a persistent pool — but it is RE-MEASURED here rather than
/// assumed to transfer, because the Huber row loop carries three extra scalar
/// reductions and a branch the SVM loop does not.
///
/// `HuberRegressor::fit` wall-clock, 16-core Zen5, f64, min over 3 fits after a
/// warm-up (`huber_perf_test::worker_knee_sweep`, which A/Bs
/// `MLRS_HUBER_ELEMS_PER_UNIT` through [`crate::abflag`]):
///
/// | n × d | `1<<11` | `1<<12` | `1<<13` | `1<<14` | `1<<15` | `1<<16` |
/// |---|---|---|---|---|---|---|
/// | 1 000 × 16 | 0.26 | 0.22 | 0.36 | 0.34 | 0.34 | 0.34 |
/// | 10 000 × 16 | 15.6 | 13.9 | 21.5 | **1.12** | 1.64 | 5.69 |
/// | 100 000 × 16 | 16.7 | 20.3 | 19.9 | **14.4** | 21.0 | 15.0 |
/// | 50 000 × 64 | 25.4 | 23.6 | 28.6 | **23.5** | 28.6 | 20.7 |
///
/// `1 << 14` is best or tied-best on every rung, and at `10 000 × 16` it is
/// best by more than an ORDER OF MAGNITUDE. That rung is the informative one:
/// 160 000 elements, so `1<<14` asks for 9 workers while `1<<13` and below all
/// ask for more than the machine has and clamp to 16. Handing every core to the
/// pass leaves none for the driver, and on a box with any other work on it the
/// barrier degrades from microseconds to scheduler timeslices — the same
/// oversubscription the [`Barrier`](super::host_pool)'s park backoff exists to
/// bound, here avoided outright by simply asking for fewer workers.
#[cfg(feature = "cpu")]
const HUBER_ELEMS_PER_UNIT: usize = 1 << 14;

/// Worker threads to split `elems` design elements across — see
/// [`HUBER_ELEMS_PER_UNIT`]. Never more than the machine offers
/// ([`crate::capability::cpu_launch_units`], which `MLRS_CPU_UNITS` overrides
/// for A/B), never fewer than one.
///
/// `MLRS_HUBER_ELEMS_PER_UNIT` overrides the knee itself for on-target A/B
/// (through [`crate::abflag`], never a raw `std::env::var` — see its docs).
#[cfg(feature = "cpu")]
fn host_units(elems: usize) -> usize {
    let knee = crate::abflag::var("MLRS_HUBER_ELEMS_PER_UNIT")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(HUBER_ELEMS_PER_UNIT);
    (elems / knee).clamp(1, crate::capability::cpu_launch_units().max(1) as usize)
}

/// Independent accumulators the dot product is split across — the natural SIMD
/// group. At `-O3` LLVM keeps the fixed-size `[f64; 8]` in AVX registers and
/// turns the body into multiply-add pairs, while the 8 independent chains hide
/// FP-add latency.
#[cfg(feature = "cpu")]
const DOT_LANES: usize = 8;

#[cfg(feature = "cpu")]
impl HostElem for f64 {
    #[inline]
    fn dot(row: &[f64], w: &[f64]) -> f64 {
        let mut acc = [0.0f64; DOT_LANES];
        let mut rows = row.chunks_exact(DOT_LANES);
        let mut ws = w.chunks_exact(DOT_LANES);
        for (xc, wc) in rows.by_ref().zip(ws.by_ref()) {
            for i in 0..DOT_LANES {
                acc[i] += xc[i] * wc[i];
            }
        }
        let mut sum = 0.0;
        for a in acc {
            sum += a;
        }
        for (xv, wv) in rows.remainder().iter().zip(ws.remainder()) {
            sum += *xv * *wv;
        }
        sum
    }

    #[inline]
    fn axpy(acc: &mut [f64], row: &[f64], g: f64) {
        for (a, &xv) in acc.iter_mut().zip(row) {
            *a += g * xv;
        }
    }
}

#[cfg(feature = "cpu")]
impl HostElem for f32 {
    #[inline]
    fn dot(row: &[f32], w: &[f64]) -> f64 {
        let mut acc = [0.0f64; DOT_LANES];
        let mut rows = row.chunks_exact(DOT_LANES);
        let mut ws = w.chunks_exact(DOT_LANES);
        for (xc, wc) in rows.by_ref().zip(ws.by_ref()) {
            for i in 0..DOT_LANES {
                acc[i] += xc[i] as f64 * wc[i];
            }
        }
        let mut sum = 0.0;
        for a in acc {
            sum += a;
        }
        for (xv, wv) in rows.remainder().iter().zip(ws.remainder()) {
            sum += *xv as f64 * *wv;
        }
        sum
    }

    #[inline]
    fn axpy(acc: &mut [f64], row: &[f32], g: f64) {
        for (a, &xv) in acc.iter_mut().zip(row) {
            *a += g * xv as f64;
        }
    }
}

// ---------------------------------------------------------------------------
// device arms — the two-GEMM evaluator
// ---------------------------------------------------------------------------

#[cfg(not(feature = "cpu"))]
impl<F> HuberObjective<'_, F>
where
    F: Float + CubeElement + Pod,
{
    /// `m = X̃·w` then `X̃ᵀ·g`, both as `prims::gemm` launches with the
    /// classification + the three scalar reductions done on the host between
    /// them (the shape a discrete GPU wants: the design never leaves the device
    /// and only `n + d_aug` floats cross).
    fn eval_device(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        w: &[f64],
        sigma: f64,
        epsilon: f64,
    ) -> Result<HuberEval, PrimError> {
        use crate::prims::gemm::gemm;

        let (n, d_aug) = (self.n, self.d_aug);
        let w_host: Vec<F> = w.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let w_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &w_host);
        let margins = match gemm::<F>(
            pool,
            &self.x_aug,
            (n, d_aug),
            &w_dev,
            (d_aug, 1),
            false,
            false,
            None,
        ) {
            Ok(m) => m,
            Err(e) => {
                w_dev.release_into(pool);
                return Err(e);
            }
        };
        let margins_host = margins.to_host(pool);
        margins.release_into(pool);
        w_dev.release_into(pool);

        let thr = epsilon * sigma;
        let inlier_scale = -2.0 / sigma;
        let outlier_scale = 2.0 * epsilon;
        let mut sq_sum = 0.0f64;
        let mut out_abs_sum = 0.0f64;
        let mut out_sw_sum = 0.0f64;
        let mut n_outliers = 0usize;
        let mut g: Vec<F> = vec![f64_to_host::<F>(0.0); n];
        for i in 0..n {
            let res = self.targets[i] - host_to_f64(margins_host[i]);
            let a = res.abs();
            let s = match self.weights.as_ref() {
                Some(sw) => sw[i],
                None => 1.0,
            };
            let gi = if a > thr {
                out_abs_sum += s * a;
                out_sw_sum += s;
                n_outliers += 1;
                if res < 0.0 {
                    outlier_scale * s
                } else {
                    -outlier_scale * s
                }
            } else {
                sq_sum += s * res * res;
                inlier_scale * s * res
            };
            g[i] = f64_to_host::<F>(gi);
        }
        let g_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &g);

        // The LOGICAL op is (d_aug × n)·(n × 1); the stored X̃ is (n × d_aug) so
        // `transa` presents the transposed view (gemm.rs:78).
        let xtg = match gemm::<F>(
            pool,
            &self.x_aug,
            (d_aug, n),
            &g_dev,
            (n, 1),
            true,
            false,
            None,
        ) {
            Ok(v) => v,
            Err(e) => {
                g_dev.release_into(pool);
                return Err(e);
            }
        };
        let xtg_host = xtg.to_host(pool);
        xtg.release_into(pool);
        g_dev.release_into(pool);

        Ok(HuberEval {
            sq_sum,
            out_abs_sum,
            out_sw_sum,
            n_outliers,
            xtg: xtg_host.iter().map(|&v| host_to_f64(v)).collect(),
        })
    }

    /// The device twin of the cpu outlier mask: one margin GEMM, then the same
    /// `|rᵢ| > σ·ε` predicate on the host.
    fn outlier_mask_device(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        w: &[f64],
        sigma: f64,
        epsilon: f64,
    ) -> Result<Vec<bool>, PrimError> {
        use crate::prims::gemm::gemm;

        let (n, d_aug) = (self.n, self.d_aug);
        let w_host: Vec<F> = w.iter().map(|&v| f64_to_host::<F>(v)).collect();
        let w_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &w_host);
        let margins = match gemm::<F>(
            pool,
            &self.x_aug,
            (n, d_aug),
            &w_dev,
            (d_aug, 1),
            false,
            false,
            None,
        ) {
            Ok(m) => m,
            Err(e) => {
                w_dev.release_into(pool);
                return Err(e);
            }
        };
        let margins_host = margins.to_host(pool);
        margins.release_into(pool);
        w_dev.release_into(pool);

        let thr = epsilon * sigma;
        Ok((0..n)
            .map(|i| (self.targets[i] - host_to_f64(margins_host[i])).abs() > thr)
            .collect())
    }
}

// ---------------------------------------------------------------------------
// backend-agnostic entry point for the one-shot outlier mask
// ---------------------------------------------------------------------------

impl<F> HuberObjective<'_, F>
where
    F: Float + CubeElement + Pod,
{
    /// sklearn's `outliers_`: `|yᵢ − x̃ᵢ·w̃| > σ·ε` at the FITTED parameters.
    ///
    /// Errors with [`PrimError::DimMismatch`] if `w` is not [`d_aug`](Self::d_aug)
    /// long.
    pub fn outlier_mask(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        w: &[f64],
        sigma: f64,
        epsilon: f64,
    ) -> Result<Vec<bool>, PrimError> {
        if w.len() != self.d_aug {
            return Err(PrimError::DimMismatch {
                dim: "d_aug",
                lhs: w.len(),
                rhs: self.d_aug,
            });
        }
        #[cfg(feature = "cpu")]
        {
            let _ = pool;
            Ok(self.outlier_mask_host(w, sigma, epsilon))
        }
        #[cfg(not(feature = "cpu"))]
        {
            self.outlier_mask_device(pool, w, sigma, epsilon)
        }
    }
}
