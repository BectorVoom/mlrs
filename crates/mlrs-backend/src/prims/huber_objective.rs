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
//! ## Two arms, chosen per FIT — not per build (HUBER-02)
//! There is a second evaluator, the device engine in [`mlrs_kernels::huber`],
//! and [`huber_device_applicable`] decides between them for each objective
//! rather than a `cfg` deciding once for the whole binary. Both are always
//! compiled; nothing about the fused host pass was ever cpu-specific
//! ([`WorkerPool`] and [`host_simd`](super::host_simd) carry no backend
//! feature), the same way [`gmm_host`](super::gmm_host) is available on every
//! backend and [`gmm_device`](super::gmm_device) is the opt-in above a floor.
//!
//! The reason it has to be a runtime choice is the SHAPE of the fit. A Huber
//! solve is L-BFGS: a few dozen evaluations, each of which must synchronize once
//! because the driver needs the loss and gradient on the host before it can pick
//! the next step. That stall is not removable — moving L-BFGS itself onto the
//! device would put its `O(d²)` two-loop recursion somewhere worse — so the
//! device arm carries a FIXED per-evaluation floor while the host arm's cost is
//! proportional to `n·d` from the first row. Below the crossover the host pass
//! finishes the whole fit before the device is done launching.
//!
//! The cpu backend is excluded outright rather than by size: `cubecl-cpu` JITs
//! at LLVM `-O0` and maps one OS THREAD PER UNIT, so a launch there costs orders
//! of magnitude more than the pass it performs
//! ([[mlrs-cubecl-cpu-execution-model]]) — the same
//! [`svm_objective`](super::svm_objective) reasoning, now expressed as a gate
//! instead of a `cfg`.
//!
//! ## Accumulation precision
//! The host arm accumulates in **`f64` regardless of `F`**. Same reason as the
//! SVM objective — an `f32` gradient sum over `n` samples carries a round-off
//! floor near `√n·ε_f32`, which at `n ≥ 1000` sits above the `tol = 1e-5`
//! stopping gradient and would let an `f32` solve miss its own tolerance — with
//! one Huber-specific sharpening: `∂L/∂σ` is the difference of two `O(n)`
//! quantities that nearly cancel at the optimum, so it is the FIRST gradient
//! entry to be destroyed by narrow accumulation.
//!
//! The device arm cannot have that for free — a kernel is monomorphized on one
//! element type, and the backends this arm exists for (rocm, cuda) do not offer
//! `f64` at all. It buys the accuracy back with SHAPE instead: a two-level
//! blocked fold at `nblocks ≈ rows_per_block ≈ √n`, whose random-walk error is
//! `O(n^¼·ε)` against a flat sum's `O(√n·ε)` — ~35·ε rather than ~316·ε at
//! `n = 100 000`. See [`quad_blocks`] and [`mlrs_kernels::huber`].
//!
//! Tests live in `crates/mlrs-backend/tests/huber_objective_test.rs` and
//! `huber_device_test.rs` (AGENTS.md §2), never an in-source `#[cfg(test)] mod
//! tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use cubecl::prelude::ArrayArg;

use mlrs_core::PrimError;
// Both conversions are device-arm-only: the host arm's fused pass is
// monomorphized on the concrete `f32`/`f64` element and never round-trips a
// value through the generic `F` bit-cast.
use mlrs_core::{f64_to_host, host_to_f64};

use mlrs_kernels::huber::{
    huber_classify_rows, huber_copy_into, huber_fold_partials, huber_margin_rows,
    huber_outlier_mask_rows, huber_quad_reduce_blocked, huber_row_pass, huber_xtg_blocked,
    HUBER_QUANTITIES,
};

use crate::device_array::DeviceArray;
use crate::prims::host_pool::{Shared, WorkerPool};
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

/// Whether [`HuberDesign::Host`] is the CHEAPER operand form for an `n × d`
/// fit on this backend.
///
/// True exactly where [`huber_device_applicable`] is false — i.e. where the
/// objective will evaluate from host memory, which is where handing the
/// caller's slab over directly removes three full passes over it (`from_host`
/// copies twice, `to_host` once) from a solve that re-reads the design dozens
/// of times. The
/// [`svm_host_ingress_preferred`](super::svm_objective::svm_host_ingress_preferred)
/// precedent, and the same contract: both forms are always ACCEPTED, this only
/// says which one to prefer.
///
/// Takes the geometry because the arm is now chosen per fit rather than per
/// build ([`HuberArm`]): a small fit on a GPU backend runs on the host and
/// wants the borrow, a large one runs on the device and wants the upload.
pub fn huber_host_ingress_preferred(n: usize, d: usize) -> bool {
    !huber_device_applicable(n, d)
}

/// Whether an `n × d` Huber fit should run on the DEVICE engine rather than the
/// fused host pass.
///
/// Four gates, in order:
///
/// 1. **The cpu backend never qualifies.** `cubecl-cpu` maps one OS thread per
///    unit and JITs at LLVM `-O0` ([[mlrs-cubecl-cpu-execution-model]]), so a
///    launch there costs orders of magnitude more than the pass it performs.
///    The host arm IS the cpu arm.
/// 2. **`MLRS_HUBER_ENGINE` override.** `"host"` forces the host pass on any
///    backend, `"device"` forces the engine past the size floor (but not past
///    gate 1, which is not a preference). Read through [`crate::abflag`] so a
///    test can scope it without an environment data race — the
///    `gmm_device_applicable` idiom.
/// 3. **Size floor.** See [`HUBER_DEVICE_MIN_WORK`].
///
/// ## Why there is a floor at all
/// A Huber fit is an L-BFGS solve: a few dozen objective evaluations, each of
/// which MUST synchronize once, because the driver needs the loss and gradient
/// on the host before it can choose the next step. That synchronization is not
/// removable without moving L-BFGS itself onto the device, where its `O(d²)`
/// two-loop recursion would be worse off. So the device arm's floor is
/// `n_evals × (one stall + five launches)` — a FIXED cost that does not shrink
/// with the problem — while the host arm's cost is proportional to `n·d` from
/// the first row. Below the crossover the host pass finishes the entire fit in
/// the time the device spends on its first few launches.
pub fn huber_device_applicable(n: usize, d: usize) -> bool {
    if crate::capability::active_backend_name() == "cpu" {
        return false;
    }
    match crate::abflag::var("MLRS_HUBER_ENGINE").as_deref() {
        Some("host") => return false,
        Some("device") => return true,
        _ => {}
    }
    n.saturating_mul(d.max(1)) >= HUBER_DEVICE_MIN_WORK
}

/// `n·d` floor below which [`huber_device_applicable`] keeps the fit on the
/// fused host pass.
///
/// **`usize::MAX` — the host pass wins at every size measured here, so by
/// default it always runs and the device engine is opt-in.** That is a
/// measurement, not a placeholder.
///
/// `huber_device_perf_test.rs::host_vs_device_crossover` A/Bs the two arms
/// through `MLRS_HUBER_ENGINE` on ONE build, interleaved min-of-5. rocm, `f32`,
/// gfx1151 iGPU, 2026-08-07, `loadavg 6.5` (the whole ladder was also run at
/// `loadavg 71`, which scaled both columns but left every ratio inside a factor
/// of two — the verdict is not a load artefact):
///
/// | n × d | n·d | host | device | |
/// |---|---|---|---|---|
/// | 1 000 × 8 | 8 K | **0.23 ms** | 11.8 ms | 52× |
/// | 10 000 × 8 | 80 K | **0.80 ms** | 22.6 ms | 28× |
/// | 10 000 × 64 | 640 K | **6.2 ms** | 51.8 ms | 8.4× |
/// | 100 000 × 16 | 1.6 M | **12.2 ms** | 122.8 ms | 10× |
/// | 100 000 × 64 | 6.4 M | **28.4 ms** | 595.9 ms | 21× |
/// | 50 000 × 128 | 6.4 M | **27.0 ms** | 3 345 ms | 124× |
///
/// The ratio does not trend to 1 as `n·d` grows, so there is no crossover
/// further up the ladder — it narrows to 8.4× and then WIDENS again, because
/// the forward row pass loses its cache locality once a wavefront's rows stop
/// fitting in L1 (see [`mlrs_kernels::huber::huber_margin_rows`]).
///
/// The host column is also the check that this table means anything: it matches
/// the cpu backend's own measured ladder (0.2 ms / 15.7 ms at the `1 000 × 8`
/// and `100 000 × 16` rungs, module docs) to within load noise, on a rocm
/// build. The host arm is the same code on every backend, and it shows.
///
/// Two things make this the hardest possible case for a device arm and the
/// reason the constant should NOT be read as "the GPU engine is useless":
///
/// 1. **The GPU is integrated.** It shares DRAM with the 16-core host, and the
///    host pass already runs at DRAM bandwidth (`huber_objective`'s own
///    38–63 GB/s table). There is no separate memory system to bring, so the
///    device can only lose ground on launch and synchronization.
/// 2. **The solve shape charges a stall per evaluation.** L-BFGS needs the loss
///    and gradient on the host to pick each step; at `n = 1 000` the entire host
///    fit takes 0.25 ms, which is less than the device spends on its first few
///    launches.
///
/// On a discrete card with its own HBM the balance is different and a crossover
/// plausibly exists. Re-running that test is how this constant should move;
/// `MLRS_HUBER_ENGINE=device` bypasses it entirely for the measurement, and the
/// oracle suite gates the device arm against scikit-learn either way
/// (`huber_test.rs::oracle_value_cases_{f32,f64}_device_engine`) so it cannot
/// rot while it is off by default.
pub const HUBER_DEVICE_MIN_WORK: usize = usize::MAX;

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
enum HostDesign<'a, F> {
    /// The caller's slab, read in place (the no-upload ingress).
    Borrowed(&'a [F]),
    /// Pulled off the device because that is where the caller had it.
    Owned(Vec<F>),
}

impl<F> HostDesign<'_, F> {
    #[inline]
    fn as_slice(&self) -> &[F] {
        match self {
            HostDesign::Borrowed(s) => s,
            HostDesign::Owned(v) => v,
        }
    }
}

/// The device arms' design: the caller's OWN device buffer when the ingress was
/// already device-resident, an owned upload only when it was not.
///
/// The borrow is what makes a device-resident `fit` upload nothing at all —
/// possible only because the synthetic intercept column is never materialized
/// (see [`mlrs_kernels::huber`]'s module docs), so the buffer both GEMMs read
/// is exactly the `n × d` slab the caller already has.
enum DevDesign<'a, F> {
    /// The caller's device array, read in place (the no-upload ingress).
    Borrowed(&'a DeviceArray<ActiveRuntime, F>),
    /// Uploaded because the caller supplied a host slab.
    Owned(DeviceArray<ActiveRuntime, F>),
}

impl<F: Float + CubeElement + Pod> DevDesign<'_, F> {
    #[inline]
    fn as_ref(&self) -> &DeviceArray<ActiveRuntime, F> {
        match self {
            DevDesign::Borrowed(d) => d,
            DevDesign::Owned(d) => d,
        }
    }

    /// Release only what this objective ALLOCATED — a borrowed design belongs
    /// to the caller and must outlive the evaluator untouched.
    fn release_into(self, pool: &mut BufferPool<ActiveRuntime>) {
        match self {
            DevDesign::Borrowed(_) => {}
            DevDesign::Owned(d) => d.release_into(pool),
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
    /// UNaugmented feature count — the width of the design as the caller
    /// supplied it, on EVERY arm: neither the host pass nor (since HUBER-02)
    /// the device engine materializes the synthetic intercept column.
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
    /// Which evaluator this objective was built for, chosen ONCE at
    /// construction by [`huber_device_applicable`].
    arm: HuberArm<'a, F>,
}

/// The two evaluators, and the state each needs for the whole solve.
///
/// A RUNTIME choice, not a `cfg` one. It used to be the latter — the host pass
/// existed only in a `cpu` build and the device engine only outside one — and
/// that was wrong in both directions. A GPU build was forced onto the device
/// arm for every fit including the tiny ones, where an L-BFGS solve's dozens of
/// one-synchronization-each evaluations are pure latency and the host pass
/// finishes the entire fit in the time the device spends on its first few
/// launches; meanwhile the cpu backend could never reach the device arm even to
/// A/B against it.
///
/// Nothing about the host pass was ever cpu-specific: [`WorkerPool`] and
/// [`host_simd`](super::host_simd) carry no backend feature, the same way
/// [`gmm_host`](super::gmm_host) is available everywhere and
/// [`gmm_device`](super::gmm_device) is the opt-in above a size floor. This is
/// that precedent, applied here.
enum HuberArm<'a, F> {
    /// The fused `-O3` host pass (module docs).
    Host {
        /// The design read in place, unaugmented — BORROWED from the caller
        /// when it was already host-resident, owned only when it had to be
        /// pulled off the device.
        x: HostDesign<'a, F>,
        /// The threads the fused pass is split across, spawned ONCE for the
        /// whole solve. `None` below the parallel knee
        /// ([`HUBER_ELEMS_PER_UNIT`]), where the pass runs inline on the
        /// calling thread.
        workers: Option<WorkerPool>,
    },
    /// The device engine ([`mlrs_kernels::huber`]).
    Device {
        /// The UNaugmented `n × d` design every kernel reads — BORROWED when
        /// the ingress was already device-resident.
        x: DevDesign<'a, F>,
        /// The targets, resident for the whole solve so the row pass forms
        /// `rᵢ = yᵢ − mᵢ − bias` without an `n`-length transfer.
        ///
        /// Held as `F`, which is LOSSLESS relative to
        /// [`HuberObjective::targets`]: those were widened from the caller's
        /// own `&[F]` at construction, so no bit the device copy could carry
        /// was ever there.
        y: DeviceArray<ActiveRuntime, F>,
        /// The sample weights, resident for the same reason as `y`. A length-1
        /// placeholder on an unweighted fit — the row pass's `weighted` flag is
        /// `0` and never indexes it, so no vector of ones is allocated or read
        /// (the host pass's `WEIGHTED` const generic, expressed as the runtime
        /// flag a kernel can carry).
        sw: DeviceArray<ActiveRuntime, F>,
    },
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

        let arm = if huber_device_applicable(n, d) {
            // WR-03: a dimension that does not fit the kernel-launch `u32`
            // would truncate into an out-of-bounds device read.
            guard_u32("n", n)?;
            guard_u32("d", d)?;

            // Every kernel reads the design EXACTLY as the caller supplied it —
            // the synthetic intercept column is folded into the row pass's
            // `bias` scalar and the fold's `Σgᵢ` quantity instead of being
            // materialized (see `mlrs_kernels::huber`). So a device-resident
            // ingress uploads nothing at all.
            let x_dev = match x {
                HuberDesign::Device(dev) => DevDesign::Borrowed(dev),
                HuberDesign::Host(slab) => DevDesign::Owned(DeviceArray::from_host(pool, slab)),
            };
            let y_host: Vec<F> = targets.iter().map(|&v| f64_to_host::<F>(v)).collect();
            let sw_host: Vec<F> = match weights.as_ref() {
                Some(w) => w.iter().map(|&v| f64_to_host::<F>(v)).collect(),
                // The length-1 placeholder the `weighted == 0` kernel never
                // indexes; a zero-length `Array` binding is not a valid launch
                // argument, so it cannot simply be empty.
                None => vec![f64_to_host::<F>(0.0)],
            };
            HuberArm::Device {
                x: x_dev,
                y: DeviceArray::from_host(pool, &y_host),
                sw: DeviceArray::from_host(pool, &sw_host),
            }
        } else {
            let x_host = match x {
                HuberDesign::Host(slab) => HostDesign::Borrowed(slab),
                HuberDesign::Device(dev) => HostDesign::Owned(dev.to_host(pool)),
            };
            // Spawned ONCE for the whole solve: the L-BFGS driver calls `eval`
            // dozens of times and `std::thread` setup dominated every one of
            // them (see [`HUBER_ELEMS_PER_UNIT`]).
            let units = host_units(n * d).min(n.max(1));
            HuberArm::Host {
                x: x_host,
                workers: (units > 1).then(|| WorkerPool::new(units)),
            }
        };
        Ok(Self {
            n,
            d,
            d_aug,
            targets,
            weights,
            sw_total,
            arm,
        })
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
        match &self.arm {
            HuberArm::Host { .. } => {
                let _ = pool;
                Ok(self.eval_host(w, sigma, epsilon))
            }
            HuberArm::Device { .. } => self.eval_device(pool, w, sigma, epsilon),
        }
    }

    /// Release the operand back to the pool, consuming the evaluator.
    pub fn release_into(self, pool: &mut BufferPool<ActiveRuntime>) {
        match self.arm {
            // The host arm allocated no device memory; a `HostDesign::Owned`
            // is a plain `Vec` and drops with `self`.
            HuberArm::Host { .. } => {}
            HuberArm::Device { x, y, sw } => {
                x.release_into(pool);
                y.release_into(pool);
                sw.release_into(pool);
            }
        }
    }

    /// The host arm's state. Panics on the device arm — every caller is a
    /// private method the [`Self::eval`] / [`Self::outlier_mask`] dispatch has
    /// already matched, so reaching it would be a bug in this file rather than
    /// anything a caller can provoke.
    fn host_parts(&self) -> (&HostDesign<'a, F>, Option<&WorkerPool>) {
        match &self.arm {
            HuberArm::Host { x, workers } => (x, workers.as_ref()),
            HuberArm::Device { .. } => {
                unreachable!("host evaluator reached on the device arm")
            }
        }
    }

    /// The device arm's state — the twin of [`Self::host_parts`], with the same
    /// invariant.
    #[allow(clippy::type_complexity)]
    fn device_parts(
        &self,
    ) -> (
        &DeviceArray<ActiveRuntime, F>,
        &DeviceArray<ActiveRuntime, F>,
        &DeviceArray<ActiveRuntime, F>,
    ) {
        match &self.arm {
            HuberArm::Device { x, y, sw } => (x.as_ref(), y, sw),
            HuberArm::Host { .. } => {
                unreachable!("device evaluator reached on the host arm")
            }
        }
    }
}

/// WR-03: reject a `usize` dimension that does not fit the kernel-launch `u32`
/// (an unguarded `as u32` truncation becomes an out-of-bounds device read).
/// The `gmm_device::guard_u32` shape.
fn guard_u32(operand: &'static str, dim: usize) -> Result<(), PrimError> {
    if dim > u32::MAX as usize {
        return Err(PrimError::ShapeMismatch {
            operand,
            rows: dim,
            cols: 0,
            len: u32::MAX as usize,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// host arm — the fused `-O3` pass
// ---------------------------------------------------------------------------

impl<'a, F> HuberObjective<'a, F>
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
        let (x_host, workers) = self.host_parts();
        let x: &[T] = bytemuck::cast_slice(x_host.as_slice());
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

        let Some(workers) = workers else {
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
        let x: &[T] = bytemuck::cast_slice(self.host_parts().0.as_slice());
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
const HUBER_ELEMS_PER_UNIT: usize = 1 << 14;

/// Worker threads to split `elems` design elements across — see
/// [`HUBER_ELEMS_PER_UNIT`]. Never more than the machine offers
/// ([`crate::capability::cpu_launch_units`], which `MLRS_CPU_UNITS` overrides
/// for A/B), never fewer than one.
///
/// `MLRS_HUBER_ELEMS_PER_UNIT` overrides the knee itself for on-target A/B
/// (through [`crate::abflag`], never a raw `std::env::var` — see its docs).
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
const DOT_LANES: usize = 8;

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
// device arm — the resident-`g` engine
// ---------------------------------------------------------------------------

impl<F> HuberObjective<'_, F>
where
    F: Float + CubeElement + Pod,
{
    /// ONE evaluation as `row pass → reduce → fold → transposed pass → fold`:
    /// five launches and a SINGLE readback of `d_aug + 5` floats.
    ///
    /// The per-sample gradient factor `g` is produced by
    /// [`huber_row_pass`] and consumed by [`huber_xtg_blocked`] without ever
    /// leaving the device; the five scalar reductions ride the same blocked
    /// fold and land in the same readback buffer. That is the whole HUBER-02
    /// change — see [`mlrs_kernels::huber`]'s module docs for the table of what
    /// the previous arm paid per evaluation (two `n`-length transfers, two
    /// pipeline stalls, an `O(n)` serial host loop).
    ///
    /// `MLRS_HUBER_DEVICE=0` routes to [`Self::eval_device_roundtrip`] instead,
    /// so the two arms can be A/B'd on the SAME build ([[mlrs-bench-verify-knob-is-live]]
    /// — a flat sweep means the knob is dead until proven otherwise).
    fn eval_device(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        w: &[f64],
        sigma: f64,
        epsilon: f64,
    ) -> Result<HuberEval, PrimError> {
        if crate::abflag::var("MLRS_HUBER_DEVICE").as_deref() == Some("0") {
            return self.eval_device_roundtrip(pool, w, sigma, epsilon);
        }
        let (n, d, d_aug) = (self.n, self.d, self.d_aug);
        let fit_intercept = d_aug > d;
        let bias = if fit_intercept { w[d] } else { 0.0 };

        // Whether to materialize the margins in their OWN launch instead of
        // fusing them into the row pass. `Some` on two routes:
        //
        // - `MLRS_HUBER_DEVICE=gemm`, which cannot fuse at all (its margins come
        //   out of the matmul substrate rather than a row scan);
        // - wide designs, where the fusion measured NEGATIVE — see
        //   [`HUBER_FUSE_MAX_D`].
        let split = via_gemm() || split_row_pass(d);
        let margins = split.then(|| self.margins_device(pool, w)).transpose()?;

        let client = pool.client().clone();
        // Only the fused row pass reads `w` on the device; the `gemm` route's
        // margins already carry it, so uploading it there would be a pointless
        // transfer on every evaluation.
        let w_dev: Option<DeviceArray<ActiveRuntime, F>> = margins.is_none().then(|| {
            let w_host: Vec<F> = w[..d].iter().map(|&v| f64_to_host::<F>(v)).collect();
            DeviceArray::from_host(pool, &w_host)
        });
        let quad_len = n * HUBER_QUANTITIES as usize;
        let (nb, rpb) = quad_blocks(n);
        let psums_len = nb * HUBER_QUANTITIES as usize;
        // `out` carries BOTH halves of the result so the evaluation reads back
        // once: `[0..d]` the transposed GEMM's gradient, `[d_aug..]` the folded
        // scalars. Index `d` (the intercept gradient) is assembled on the host
        // from the fold's `Σgᵢ`, so it is deliberately left unwritten.
        let out_len = d_aug + HUBER_QUANTITIES as usize;

        let elem = size_of::<F>();
        let g = pool.acquire(n * elem);
        let quad = pool.acquire(quad_len * elem);
        let psums = pool.acquire(psums_len * elem);
        let out = pool.acquire(out_len * elem);

        {
            // SAFETY: every element count below is the one the matching
            // `pool.acquire` reserved, and each kernel bounds-checks its unit
            // id (`i < n`, `tid < nblocks·nq`, `tid < len`).
            let (x_dev, y_dev, sw_dev) = self.device_parts();
            let y_arg =
                unsafe { ArrayArg::from_raw_parts(y_dev.handle().clone(), y_dev.len()) };
            let sw_arg =
                unsafe { ArrayArg::from_raw_parts(sw_dev.handle().clone(), sw_dev.len()) };
            let g_arg = unsafe { ArrayArg::from_raw_parts(g.clone(), n) };
            let q_arg = unsafe { ArrayArg::from_raw_parts(quad.clone(), quad_len) };
            let (count, dim) = super::launch_dims_1d_folded(n, super::PERF_TUNED_BLOCK);
            let weighted = u32::from(self.weights.is_some());
            let thr = f64_to_host::<F>(epsilon * sigma);
            let inlier = f64_to_host::<F>(-2.0 / sigma);
            let outlier = f64_to_host::<F>(2.0 * epsilon);
            let neg_outlier = f64_to_host::<F>(-2.0 * epsilon);
            match margins.as_ref() {
                Some(m) => {
                    let m_arg = unsafe { ArrayArg::from_raw_parts(m.handle().clone(), n) };
                    huber_classify_rows::launch::<F, ActiveRuntime>(
                        &client,
                        count,
                        dim,
                        m_arg,
                        y_arg,
                        sw_arg,
                        g_arg,
                        q_arg,
                        n as u32,
                        weighted,
                        f64_to_host::<F>(bias),
                        thr,
                        inlier,
                        outlier,
                        neg_outlier,
                    );
                }
                None => {
                    let wd = w_dev
                        .as_ref()
                        .expect("w is uploaded exactly when the fused row pass runs");
                    let x_arg =
                        unsafe { ArrayArg::from_raw_parts(x_dev.handle().clone(), n * d) };
                    let w_arg = unsafe { ArrayArg::from_raw_parts(wd.handle().clone(), d) };
                    huber_row_pass::launch::<F, ActiveRuntime>(
                        &client,
                        count,
                        dim,
                        x_arg,
                        w_arg,
                        y_arg,
                        sw_arg,
                        g_arg,
                        q_arg,
                        n as u32,
                        d as u32,
                        weighted,
                        f64_to_host::<F>(bias),
                        thr,
                        inlier,
                        outlier,
                        neg_outlier,
                    );
                }
            }
        }
        if let Some(wd) = w_dev {
            wd.release_into(pool);
        }
        {
            let q_arg = unsafe { ArrayArg::from_raw_parts(quad.clone(), quad_len) };
            let p_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
            let (count, dim) = super::launch_dims_1d_folded(psums_len, super::PERF_TUNED_BLOCK);
            huber_quad_reduce_blocked::launch::<F, ActiveRuntime>(
                &client,
                count,
                dim,
                q_arg,
                p_arg,
                n as u32,
                HUBER_QUANTITIES,
                nb as u32,
                rpb as u32,
            );

            let p_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
            let o_arg = unsafe { ArrayArg::from_raw_parts(out.clone(), out_len) };
            let (c2, d2) =
                super::launch_dims_1d_folded(HUBER_QUANTITIES as usize, super::PERF_TUNED_BLOCK);
            huber_fold_partials::launch::<F, ActiveRuntime>(
                &client,
                c2,
                d2,
                p_arg,
                o_arg,
                HUBER_QUANTITIES,
                nb as u32,
                d_aug as u32,
            );
        }

        let g_dev = DeviceArray::<ActiveRuntime, F>::from_raw(g, n);
        if let Err(e) = self.xtg_into(pool, &client, &g_dev, &out, out_len) {
            g_dev.release_into(pool);
            if let Some(m) = margins {
                m.release_into(pool);
            }
            pool.release(quad, quad_len * elem);
            pool.release(psums, psums_len * elem);
            pool.release(out, out_len * elem);
            return Err(e);
        }

        // THE one synchronization of the evaluation.
        let out_dev = DeviceArray::<ActiveRuntime, F>::from_raw(out, out_len);
        let out_host = out_dev.to_host(pool);
        out_dev.release_into(pool);
        g_dev.release_into(pool);
        if let Some(m) = margins {
            m.release_into(pool);
        }
        pool.release(quad, quad_len * elem);
        pool.release(psums, psums_len * elem);

        let mut xtg_out: Vec<f64> = out_host[..d].iter().map(|&v| host_to_f64(v)).collect();
        if fit_intercept {
            // The synthetic column is a constant 1.0, so its `x̃ᵀg` entry is
            // exactly `Σᵢ gᵢ` — quantity 4 of the fold, not a GEMM column.
            xtg_out.push(host_to_f64(out_host[d_aug + 4]));
        }
        Ok(HuberEval {
            sq_sum: host_to_f64(out_host[d_aug]),
            out_abs_sum: host_to_f64(out_host[d_aug + 1]),
            out_sw_sum: host_to_f64(out_host[d_aug + 2]),
            n_outliers: host_to_f64(out_host[d_aug + 3]).round().max(0.0) as usize,
            xtg: xtg_out,
        })
    }

    /// `m = X·w[..d]`, left DEVICE-resident. Shared by [`Self::eval_device`],
    /// [`Self::eval_device_roundtrip`] and [`Self::outlier_mask_device`], which
    /// differ only in what they then do with the margins.
    ///
    /// Reads `w[..d]` only: the intercept is the `bias` scalar the consuming
    /// kernel adds (`mlrs_kernels::huber` module docs), never a column of the
    /// operand.
    ///
    /// [`mlrs_kernels::huber::huber_margin_rows`] rather than `prims::gemm` —
    /// see that kernel's docs for the measured reason (a tiled matmul
    /// degenerated to `N = 1` cost 142 ms/iteration on a gfx1151 for 1.6 M
    /// MACs). `MLRS_HUBER_DEVICE=gemm` keeps the old route reachable so the
    /// comparison stays a measurement instead of a memory.
    fn margins_device(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        w: &[f64],
    ) -> Result<DeviceArray<ActiveRuntime, F>, PrimError> {
        let (n, d) = (self.n, self.d);
        let w_host: Vec<F> = w[..d].iter().map(|&v| f64_to_host::<F>(v)).collect();
        let w_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &w_host);

        if via_gemm() {
            let margins = crate::prims::gemm::gemm::<F>(
                pool,
                self.device_parts().0,
                (n, d),
                &w_dev,
                (d, 1),
                false,
                false,
                None,
            );
            w_dev.release_into(pool);
            return margins;
        }

        let client = pool.client().clone();
        let margins = pool.acquire(n * size_of::<F>());
        {
            // SAFETY: `margins` was reserved for `n` elements, `x` holds `n·d`
            // and `w_dev` holds `d`; the kernel bounds-checks `i < n` and its
            // inner scan is `j < d`.
            let x_arg = unsafe {
                ArrayArg::from_raw_parts(self.device_parts().0.handle().clone(), n * d)
            };
            let w_arg = unsafe { ArrayArg::from_raw_parts(w_dev.handle().clone(), d) };
            let m_arg = unsafe { ArrayArg::from_raw_parts(margins.clone(), n) };
            let (count, dim) = super::launch_dims_1d_folded(n, super::PERF_TUNED_BLOCK);
            huber_margin_rows::launch::<F, ActiveRuntime>(
                &client, count, dim, x_arg, w_arg, m_arg, n as u32, d as u32,
            );
        }
        w_dev.release_into(pool);
        Ok(DeviceArray::from_raw(margins, n))
    }

    /// `out[0..d] = Xᵀ·g` — the transposed design pass, written straight into
    /// the shared readback buffer so no separate synchronization is needed.
    ///
    /// [`mlrs_kernels::huber::huber_xtg_blocked`] + a fold, for the same
    /// measured reason [`Self::margins_device`] avoids `prims::gemm`; the
    /// `MLRS_HUBER_DEVICE=gemm` route takes the matmul and then a
    /// [`mlrs_kernels::huber::huber_copy_into`] to land in the same place.
    fn xtg_into(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        client: &cubecl::client::ComputeClient<ActiveRuntime>,
        g_dev: &DeviceArray<ActiveRuntime, F>,
        out: &cubecl::server::Handle,
        out_len: usize,
    ) -> Result<(), PrimError> {
        let (n, d) = (self.n, self.d);
        let elem = size_of::<F>();

        if via_gemm() {
            // The LOGICAL op is (d × n)·(n × 1); the stored X is (n × d) so
            // `transa` presents the transposed view (gemm.rs:78).
            let xtg = crate::prims::gemm::gemm::<F>(
                pool,
                self.device_parts().0,
                (d, n),
                g_dev,
                (n, 1),
                true,
                false,
                None,
            )?;
            let s_arg = unsafe { ArrayArg::from_raw_parts(xtg.handle().clone(), d) };
            let o_arg = unsafe { ArrayArg::from_raw_parts(out.clone(), out_len) };
            let (count, dim) = super::launch_dims_1d_folded(d, super::PERF_TUNED_BLOCK);
            huber_copy_into::launch::<F, ActiveRuntime>(
                client, count, dim, s_arg, o_arg, d as u32, 0u32,
            );
            xtg.release_into(pool);
            return Ok(());
        }

        let (nb, rpb) = xtg_blocks(n, d);
        let psums_len = nb * d;
        let psums = pool.acquire(psums_len * elem);
        {
            // SAFETY: `psums` was reserved for `nblocks·d` elements and the
            // kernel bounds-checks `tid < nblocks·d`, clamping its row range
            // to `n`.
            let x_arg = unsafe {
                ArrayArg::from_raw_parts(self.device_parts().0.handle().clone(), n * d)
            };
            let g_arg = unsafe { ArrayArg::from_raw_parts(g_dev.handle().clone(), n) };
            let p_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
            let (count, dim) = super::launch_dims_1d_folded(psums_len, super::PERF_TUNED_BLOCK);
            huber_xtg_blocked::launch::<F, ActiveRuntime>(
                &client.clone(),
                count,
                dim,
                x_arg,
                g_arg,
                p_arg,
                n as u32,
                d as u32,
                nb as u32,
                rpb as u32,
            );

            let p_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
            let o_arg = unsafe { ArrayArg::from_raw_parts(out.clone(), out_len) };
            let (c2, d2) = super::launch_dims_1d_folded(d, super::PERF_TUNED_BLOCK);
            huber_fold_partials::launch::<F, ActiveRuntime>(
                &client.clone(),
                c2,
                d2,
                p_arg,
                o_arg,
                d as u32,
                nb as u32,
                0u32,
            );
        }
        pool.release(psums, psums_len * elem);
        Ok(())
    }

    /// The PREVIOUS device arm, kept for A/B only (`MLRS_HUBER_DEVICE=0`):
    /// margins down, classify on the host, `g` back up.
    ///
    /// It exists so the HUBER-02 win can be measured on one build rather than
    /// inferred from two — this repo's rule after a stale `.so` made a sweep
    /// vacuous ([[mlrs-bench-verify-knob-is-live]]). Nothing in the product
    /// path selects it, and it is intentionally the ONLY caller of the host
    /// classification loop on a device backend.
    fn eval_device_roundtrip(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        w: &[f64],
        sigma: f64,
        epsilon: f64,
    ) -> Result<HuberEval, PrimError> {
        let (n, d, d_aug) = (self.n, self.d, self.d_aug);
        let fit_intercept = d_aug > d;
        let bias = if fit_intercept { w[d] } else { 0.0 };

        let margins = self.margins_device(pool, w)?;
        let margins_host = margins.to_host(pool);
        margins.release_into(pool);

        let thr = epsilon * sigma;
        let inlier_scale = -2.0 / sigma;
        let outlier_scale = 2.0 * epsilon;
        let mut sq_sum = 0.0f64;
        let mut out_abs_sum = 0.0f64;
        let mut out_sw_sum = 0.0f64;
        let mut gsum = 0.0f64;
        let mut n_outliers = 0usize;
        let mut g: Vec<F> = vec![f64_to_host::<F>(0.0); n];
        for i in 0..n {
            let res = self.targets[i] - host_to_f64(margins_host[i]) - bias;
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
            gsum += gi;
            g[i] = f64_to_host::<F>(gi);
        }
        let g_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &g);

        // Deliberately the SAME transposed pass the resident engine runs, so
        // the A/B isolates exactly one variable: whether `g` round-trips. An
        // arm that also swapped the design kernel would conflate the two and
        // attribute the whole ratio to the wrong change.
        let client = pool.client().clone();
        let out = pool.acquire(d * size_of::<F>());
        if let Err(e) = self.xtg_into(pool, &client, &g_dev, &out, d) {
            pool.release(out, d * size_of::<F>());
            g_dev.release_into(pool);
            return Err(e);
        }
        let out_dev = DeviceArray::<ActiveRuntime, F>::from_raw(out, d);
        let xtg_host = out_dev.to_host(pool);
        out_dev.release_into(pool);
        g_dev.release_into(pool);

        let mut xtg_out: Vec<f64> = xtg_host.iter().map(|&v| host_to_f64(v)).collect();
        if fit_intercept {
            xtg_out.push(gsum);
        }
        Ok(HuberEval {
            sq_sum,
            out_abs_sum,
            out_sw_sum,
            n_outliers,
            xtg: xtg_out,
        })
    }

    /// The device twin of the cpu outlier mask: one margin GEMM, then the same
    /// `|rᵢ| > σ·ε` predicate as a second launch — one readback, no host loop.
    fn outlier_mask_device(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        w: &[f64],
        sigma: f64,
        epsilon: f64,
    ) -> Result<Vec<bool>, PrimError> {
        let (n, d, d_aug) = (self.n, self.d, self.d_aug);
        let margins = self.margins_device(pool, w)?;

        let client = pool.client().clone();
        let mask = pool.acquire(n * size_of::<F>());
        {
            // SAFETY: `mask` was reserved for `n` elements and the kernel
            // bounds-checks `i < n`.
            let m_arg = unsafe { ArrayArg::from_raw_parts(margins.handle().clone(), n) };
            let (_, y_dev, _) = self.device_parts();
            let y_arg =
                unsafe { ArrayArg::from_raw_parts(y_dev.handle().clone(), y_dev.len()) };
            let o_arg = unsafe { ArrayArg::from_raw_parts(mask.clone(), n) };
            let (count, dim) = super::launch_dims_1d_folded(n, super::PERF_TUNED_BLOCK);
            huber_outlier_mask_rows::launch::<F, ActiveRuntime>(
                &client,
                count,
                dim,
                m_arg,
                y_arg,
                o_arg,
                n as u32,
                f64_to_host::<F>(if d_aug > d { w[d] } else { 0.0 }),
                f64_to_host::<F>(epsilon * sigma),
            );
        }
        let mask_dev = DeviceArray::<ActiveRuntime, F>::from_raw(mask, n);
        let mask_host = mask_dev.to_host(pool);
        mask_dev.release_into(pool);
        margins.release_into(pool);
        Ok(mask_host.iter().map(|&v| host_to_f64(v) != 0.0).collect())
    }
}

/// Row-block layout for [`huber_quad_reduce_blocked`]: `nblocks ≈
/// rows_per_block ≈ √n`.
///
/// BALANCED rather than "as many blocks as the device will take", which is what
/// every other blocked reduction in this crate wants. The reason is round-off,
/// not occupancy: this fold's inputs are `O(n)` arrays whose reduction is free
/// next to the `O(n·d)` GEMMs either side of it, so the only thing left to
/// optimize is accuracy — and a two-level sum with both levels at `√n` is the
/// minimum of the `√(nblocks) + √(rows_per_block)` random-walk error model
/// (`mlrs_kernels::huber` module docs). The `F`-width accumulation the device
/// arm is stuck with is exactly why that is worth spending the choice on.
///
/// Capped at [`QUAD_MAX_BLOCKS`] so the intermediate partial buffer stays
/// bounded at very large `n`.
fn quad_blocks(n: usize) -> (usize, usize) {
    let target = (n as f64).sqrt().ceil() as usize;
    let nb = target.clamp(1, QUAD_MAX_BLOCKS.min(n.max(1)));
    let rpb = n.div_ceil(nb);
    // Re-derive so the last block is never empty (a block whose whole row range
    // clamps away would still be launched and would write a zero partial —
    // harmless, but the count is then a lie the docs would carry).
    (n.div_ceil(rpb), rpb)
}

/// Block cap for [`quad_blocks`] — the intermediate `nblocks ·
/// HUBER_QUANTITIES` partial buffer, in elements.
///
/// `√n` only exceeds this above `n ≈ 67 million` rows, where the design itself
/// is far past device memory; the cap is a bound, not a tuning knob.
const QUAD_MAX_BLOCKS: usize = 8192;

/// Row-block layout for [`huber_xtg_blocked`]: enough blocks that `nblocks · d`
/// units keep the device busy, without letting the intermediate `nblocks · d`
/// partial buffer grow past [`XTG_PARTIAL_BUDGET`].
///
/// OCCUPANCY-driven, unlike [`quad_blocks`]: this reduction reads the whole
/// `n × d` design, so it is one of the two passes that actually costs, and
/// under-parallelizing it is far more expensive than the round-off a balanced
/// split would buy. The blocking still gives a two-level sum for free.
fn xtg_blocks(n: usize, d: usize) -> (usize, usize) {
    let cap = (XTG_PARTIAL_BUDGET / d.max(1)).max(1);
    let nb = n.div_ceil(64).clamp(1, cap.min(n.max(1)));
    let rpb = n.div_ceil(nb);
    (n.div_ceil(rpb), rpb)
}

/// Element budget for [`xtg_blocks`]' intermediate partial buffer.
///
/// The `gmm_device::REDUCE_BUDGET_DENSE` concern in miniature: the per-block
/// stride here is `d`, so a blind block count would allocate `nblocks · d`
/// elements — bounded rather than tuned.
const XTG_PARTIAL_BUDGET: usize = 1 << 20;

/// Whether the two `O(n·d)` design passes route through `prims::gemm` instead
/// of the dedicated kernels (`MLRS_HUBER_DEVICE=gemm`).
///
/// A/B only. The dedicated kernels are the product path because the matmul
/// substrate is catastrophically slow at this shape — see
/// [`mlrs_kernels::huber::huber_margin_rows`] for the measurement. Kept
/// reachable so the claim stays checkable on whatever hardware is in front of
/// you, rather than inherited from this machine's iGPU
/// ([[mlrs-feedback-verify-on-target-hardware]]).
fn via_gemm() -> bool {
    crate::abflag::var("MLRS_HUBER_DEVICE").as_deref() == Some("gemm")
}

/// Whether to run the margin as its OWN launch
/// ([`mlrs_kernels::huber::huber_margin_rows`]) and classify in a second
/// ([`huber_classify_rows`]), instead of the fused
/// [`mlrs_kernels::huber::huber_row_pass`].
///
/// `MLRS_HUBER_DEVICE=split` / `=fused` force it either way; otherwise the
/// width decides at [`HUBER_FUSE_MAX_D`].
fn split_row_pass(d: usize) -> bool {
    match crate::abflag::var("MLRS_HUBER_DEVICE").as_deref() {
        Some("split") => true,
        Some("fused") => false,
        _ => d > HUBER_FUSE_MAX_D,
    }
}

/// Feature width above which fusing the margin into the classify pass stops
/// paying, so the two run as separate launches.
///
/// Fusing removes a launch and an `n`-element store/reload, which is why it is
/// the default — but it also makes ONE kernel hold the row loop's accumulator
/// AND the classification's live state (targets, weights, threshold, the three
/// gradient prefactors, five per-sample outputs). Past some `d` that register
/// pressure costs more occupancy than the launch it saved.
///
/// MEASURED, on the same interleaved ladder that produced
/// [`HUBER_DEVICE_MIN_WORK`] (rocm, `f32`, gfx1151). The `50 000 × 128` rung is
/// where it shows: with the fusion forced it ran **3 083 ms**, against 549 ms at
/// `100 000 × 64` — 5.6× worse for the SAME `n·d = 6.4 M` of arithmetic, and
/// worse than both the round-trip arm (952 ms) and the `gemm` route (2 449 ms),
/// neither of which fuses. Splitting the pass at that rung brought it to
/// **1 804 ms**, a 1.7× recovery, while every rung at `d ≤ 64` still favours the
/// fusion. The threshold therefore sits at 64, i.e. exactly where the evidence
/// changes sign.
///
/// The recovery is partial, and that is worth saying: at `d = 128` the split
/// engine is still behind the round-trip arm, so register pressure was one
/// cause and not the only one — the margin kernel's own cache behaviour at wide
/// `d` is the other ([`mlrs_kernels::huber::huber_margin_rows`]). Chasing the
/// rest was not worth it while the host pass wins this geometry by 124×;
/// `MLRS_HUBER_DEVICE=split`/`fused` is how the threshold gets re-derived on
/// hardware where it would be.
const HUBER_FUSE_MAX_D: usize = 64;

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
        match &self.arm {
            HuberArm::Host { .. } => {
                let _ = pool;
                Ok(self.outlier_mask_host(w, sigma, epsilon))
            }
            HuberArm::Device { .. } => self.outlier_mask_device(pool, w, sigma, epsilon),
        }
    }
}
