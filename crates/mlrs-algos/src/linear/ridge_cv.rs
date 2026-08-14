//! `RidgeCV` (RIDGECV-01) — the numeric core of
//! `sklearn.linear_model.RidgeCV`, with its FULL parameter surface.
//!
//! ```text
//! RidgeCV(alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None,
//!         cv=None, gcv_mode=None, store_cv_results=False,
//!         alpha_per_target=False).fit(X, y, sample_weight=None)
//! ```
//!
//! sklearn's `RidgeCV` is two estimators wearing one name, and this module owns
//! both engines:
//!
//! | `cv` | engine | what it costs |
//! |---|---|---|
//! | `None` (the DEFAULT) | [`ridge_gcv`] — generalized (leave-one-out) CV in closed form off ONE symmetric eigendecomposition | one `O(n·d²)` Gram + one `O(d³)` eig + one `O(n·d²)` projection, then `O(n·d)` PER ALPHA |
//! | anything else | [`ridge_cv_grid`] — an explicit `GridSearchCV(Ridge(), {'alpha': alphas}, cv=cv)` | one `O(n_train·d²)` Gram PER SPLIT, then `O(d³/3)` per alpha |
//!
//! ## Where the work runs (RIDGECV-02)
//! The GCV engine's `n > d` route has TWO arms, chosen by [`gcv_device_arm`]
//! from the `device` hyperparameter and reported by the estimator's `device_`:
//!
//! | phase | host arm | device arm |
//! |---|---|---|
//! | means + `X̃ᵀX̃` + `X̃ᵀ[ỹ ǀ √w]`, `O(n·d²)` | `gram_host` | `prims::ridge_gcv::GcvDevice::normal_equations` |
//! | `sym_eig` + spectral weights + coefficients, `O(d³ + n_alphas·d²·n_y)` | HOST | HOST (the same code — [`spectral_weights`]) |
//! | the streaming sweep, `O(n·d² + n_alphas·n·d·(n_y+2))` | [`cov_sweep`] | `prims::ridge_gcv::GcvDevice::sweep` |
//!
//! The `O(d³)` middle runs on the host on BOTH arms, and not for lack of a
//! kernel: it is a serial scalar recurrence three to five orders below the two
//! passes around it, which is the one shape a GPU is worst at
//! (`bayesian_ridge.rs` makes the same split for the same reason). Because it is
//! literally the same code on both arms, the two cannot disagree about the LOO
//! algebra — only about summation order.
//!
//! Everything else stays host-only and says why: the `n ≤ d` route (its cost IS
//! a serial `O(n³)` eigendecomposition — see below), and the explicit-`cv` grid
//! engine (a per-split `O(d³/3)` Cholesky, not a data-parallel kernel). Both
//! report `device_ == "cpu"` rather than the preference they could not honour.
//!
//! `device='auto'` keeps the HOST arm on every backend measured so far — the
//! ingress upload dominates a device fit, and the host arm already beats sklearn
//! several times over. The ladders, the crossover and the reasons the crossover
//! is not shipped as a threshold are in
//! [`gcv_device_preferred`](mlrs_backend::prims::ridge_gcv::gcv_device_preferred).
//!
//! ## The LOO identity, and why the alpha loop is nearly free
//! For ridge, the leave-one-out residual has a closed form that needs no
//! refitting: with `G = X̃X̃ᵀ + αI`, `c = G⁻¹ỹ` and `d = diag(G⁻¹)`,
//!
//! ```text
//! looe = ỹ − ŷ₍₋ᵢ₎ = c / d
//! ```
//!
//! sklearn evaluates that per alpha through a **fresh `d×d` inverse and a fresh
//! `n×d` product**: its default dense route (`gcv_mode=None` on a tall `X`
//! resolves to `_solve_eigen_covariance`) forms `Hinv = V·diag(1/(λ+α))·Vᵀ` and
//! then `X @ Hinv` — an `O(n·d²)` GEMM — **inside the alpha loop**, so the whole
//! fit is `O(n_alphas · n · d²)`.
//!
//! mlrs pushes the eigenbasis one step further. Writing `W = X̃·V` (formed ONCE,
//! `O(n·d²)`), every per-alpha quantity is a `diag`-weighted contraction over
//! `W`'s rows:
//!
//! ```text
//! (X Hinv Xᵀ y)ᵢ = Σₖ Wᵢₖ·zₖ/(λₖ+α)      with z = Vᵀ·Xᵀy   (formed once)
//! diag(X Hinv Xᵀ)ᵢ = Σₖ Wᵢₖ²/(λₖ+α)
//! ```
//!
//! so the alpha loop drops from `O(n·d²)` to `O(n·d)` per alpha, and the gap
//! against sklearn GROWS with `len(alphas)`. Measured on a 16-core box at
//! `n = 100 000, d = 64`, each library in its own process
//! (`scripts/bench_ridge_cv.py`, min-of-7):
//!
//! | `len(alphas)` | sklearn | mlrs | |
//! |---|---|---|---|
//! | 1 | 79.7 ms | 24.5 ms | 3.3× |
//! | 3 (the default) | 146 ms | 30.6 ms | 4.8× |
//! | 10 | 377 ms | 45.3 ms | 8.3× |
//! | 30 | 1 056 ms | 85.5 ms | 12.3× |
//! | 100 | 3 376 ms | 229 ms | 14.8× |
//! | 200 | 6 770 ms | 437 ms | 15.5× |
//!
//! The shape of that column is the claim, not any single row: mlrs's time is
//! `24 ms + 2.1 ms/alpha` and sklearn's is `80 ms + 34 ms/alpha`. The run those
//! come from verified that other processes took 2.3% / 3.7% of the machine
//! while it was measured — `bench_ridge_cv.py` refuses to vouch for a run above
//! 10%, because on a contended box this comparison (a parallel engine against a
//! mostly single-threaded one) moves the RATIO, not just the variance.
//!
//! `W` is never materialized: the projection is fused into a row-blocked sweep
//! that consumes each block for every alpha before moving on, so the extra
//! footprint is `ROW_BLOCK × d` doubles PER WORKER regardless of `n_samples` —
//! 64 KiB at `d = 64`, against the `n × d` the naive form would allocate
//! (51 MiB at `n = 100 000`).
//!
//! ## Where this is SLOWER than sklearn, and why
//! The `n ≤ d` route has no such story. There the `O(n³)` symmetric
//! eigendecomposition of the `n × n` Gram dominates everything else, and mlrs's
//! [`sym_eig`] is a SERIAL Householder+QL pair while sklearn reaches LAPACK's
//! threaded `dsyevd`. Everything around it is parallel here — the Gram, the
//! per-alpha solve and the dual-to-primal back-projection all split across
//! workers — but the serial part is what is left, and it is `O(n³)`.
//! `ridge_cv_perf_test.rs::wide_ladder` extrapolates the design-INDEPENDENT
//! cost (measured on a quiet 16-core box, 30 alphas):
//!
//! | `n` | design-independent (≈ `sym_eig(n)`) | vs the whole fit at `d = 4n` |
//! |---|---|---|
//! | 100 | 1.5 ms | 2.1 ms |
//! | 200 | 6.8 ms | 12.6 ms |
//! | 400 | 51.6 ms | 92.1 ms |
//! | 800 | 383 ms | 734 ms |
//!
//! Clean `n³` scaling (7.6× and 7.4× per doubling) and over half the fit from
//! `n = 400` up, so the route LOSES to sklearn by several times there.
//!
//! ## One UNEXPLAINED rung: `n = 10 000, d = 64`
//! On the `scripts/bench_ridge_cv.py` ladder this single shape reads ~0.75× —
//! the only tall-design loss. It is documented rather than tuned away because
//! what it is NOT has been established and what it IS has not:
//!
//! * not contention (foreign cpu 0.5% on the run that shows it),
//! * not thread count in the usual sense — a warm sweep is NON-MONOTONIC
//!   (8.80 / 6.09 / 8.92 / 8.96 ms at 2 / 4 / 8 / 16 workers), while EVERY other
//!   shape on the same sweep prefers 16 monotonically,
//! * not the Python ingress (0.15 ms) nor the result read-back (~0 ms),
//! * not Python's GC (`gc.disable()` does not move it),
//! * not glibc malloc arenas (`MALLOC_ARENA_MAX=1` does not move it),
//! * and not the engine: [`ridge_gcv`] itself measures 2.6 ms at this shape
//!   whether or not a differently-shaped fit preceded it, while the surrounding
//!   `fit` measures 3.1 ms cold and ~9 ms warm.
//!
//! Raising [`MACS_PER_UNIT`] so this shape gets 4 workers would fix the cell and
//! cost ~8% at `n = 20 000, d = 64`, which prefers 16 — i.e. it would be
//! overfitting a constant to one unexplained measurement. It is left alone.
//!
//! ## That the wide-route loss is shared infrastructure
//! It is a shared-infrastructure gap, not a `RidgeCV` one: the same serial
//! `sym_eig` is on `BayesianRidge`'s critical path. Fixing it means
//! parallelizing the QL rotation accumulation (the `~6·d³` term — its rotations
//! depend only on the tridiagonal, so they can be recorded once and replayed
//! across column blocks with identical arithmetic), which is a change to a
//! numerically-vetted routine with its own oracle suite and belongs in its own
//! campaign. `ridge_cv_perf_test.rs::wide_ladder` prints the eig share so that
//! campaign starts from a measurement.
//!
//! ## `gcv_mode` (`None` / `'auto'` / `'svd'` / `'eigen'`) — accepted, validated,
//! ## and deliberately ONE code path
//! sklearn 1.9's `_check_gcv_mode` maps `None`/`'auto'`/`'eigen'` onto
//! `"gram"` (`n ≤ d`) or `"cov"` (`n > d`) and `'svd'` onto a reduced SVD of the
//! design. Those are three LAPACK calls, not three answers: the three routes are
//! algebraically identical (`U = X̃·V·diag(1/σ)` and `σ² = λ` turn every formula
//! in one into the corresponding formula in the other — the derivation is in
//! [`ridge_gcv`]'s doc), and sklearn's own `'auto'` and `'eigen'` have been the
//! same route since 1.9.
//!
//! mlrs therefore derives all three from the SAME symmetric eigendecomposition —
//! of `X̃ᵀX̃` when `n > d`, of `X̃X̃ᵀ` when `n ≤ d`, whichever is smaller — and the
//! three `gcv_mode` values return bit-identical results here. The parameter is
//! still real surface: it is parsed, an unknown string is rejected exactly as
//! sklearn's `StrOptions` rejects it ([`BuildError::UnknownGcvMode`]), and the
//! oracle suite compares EACH of the three against sklearn run in the SAME mode.
//! What it is not is a performance knob in mlrs, and the perf test says so with
//! a measurement rather than a claim.
//!
//! The cost of that choice is the cost of any Gram-then-eig emulation of an SVD:
//! it squares the condition number of `X̃`. That is the same trade
//! `bayesian_ridge.rs` already makes for the same reason, and it is bounded by
//! the `f64` accumulation both the Gram pass and the eigensolver use whatever
//! the estimator's `F` is. What it costs in practice is measured rather than
//! guessed at: `test_oracle_ridge_cv_params.py::
//! test_near_singular_design_is_close_but_looser` fits a rank-deficient design
//! (two exactly-duplicated columns in seven) and holds the PREDICTIONS to a
//! relative 1e-4 against sklearn instead of this suite's usual 1e-5 — close
//! enough to stay usable, and honestly looser. Every well-conditioned design in
//! that file meets the 1e-5 gate.
//!
//! ## `sample_weight`, and why `Xᵀ√w` is computed rather than assumed
//! sklearn preprocesses with `_preprocess_data(..., rescale_with_sw=True)`:
//! weighted column means, then a `√wᵢ` row rescale. The LOO denominator then
//! carries an extra term, `−(G⁻¹√w)ᵢ·√wᵢ / Σw`, which corrects for the intercept
//! being refit on each held-out round.
//!
//! Under `fit_intercept` that correction's operand `X̃ᵀ√w = Σᵢ wᵢ(xᵢ − x̄)` is
//! ZERO by construction of the weighted mean, so it could be skipped. It is not:
//! the term is accumulated for real, in the same `f64` pass shape as everything
//! else, so the arm carries whatever cancellation error the centering left
//! rather than an analytic zero that quietly disagrees with sklearn's numerics.
//!
//! ## `alpha_per_target`
//! With a 2-D `y`, each target column gets its own `alpha_` and its own column of
//! `coef_`. That is why [`GcvFit::coefs`] keeps EVERY alpha's coefficients rather
//! than only the winner's: the winner is not a single alpha, and the per-alpha
//! coefficient block is `O(d·n_y)` — three orders below the design it came from.
//!
//! ## `scoring`
//! `scoring=None` (the default) scores with `−mean(looe²)`, which this module
//! computes itself: no per-alpha `n`-vector ever leaves Rust. Any other scorer
//! is an arbitrary Python callable, so the engine instead returns the rescaled
//! LOO PREDICTIONS (`want_predictions`) and the shim applies the scorer — the
//! same split sklearn makes internally, and the same one
//! `model_selection::search` makes for the search drivers.
//!
//! Tests live in `crates/mlrs-algos/tests/ridge_cv_test.rs` and
//! `crates/mlrs-algos/tests/ridge_cv_perf_test.rs` (AGENTS.md §2), never an
//! in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability::cpu_launch_units;
use mlrs_backend::device::Device;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gram_host::centered_gram_multi_xty;
use mlrs_backend::prims::ridge_gcv::{gcv_device_possible, gcv_device_preferred, GcvDevice};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::{AlgoError, BuildError};
use crate::linear::ridge_solvers::{cholesky_ridge, gram_eig_ridge};
use crate::linear::sym_eig::sym_eig;

/// Rows a streaming worker projects into the eigenbasis at a time.
///
/// The block exists so `W = X̃·V` is never materialized at `n × d`: a worker
/// builds `ROW_BLOCK × d` of it, consumes those rows for EVERY alpha, and
/// reuses the buffer. 128 rows × `d ≤ 512` doubles is 512 KiB worst case and
/// a few KiB at the shapes this estimator is actually used at, so the block
/// stays resident across the alpha loop — which is the loop that has to be
/// cheap.
const ROW_BLOCK: usize = 128;

/// Multiply-accumulates one worker should be given before splitting further.
///
/// The streaming sweep is `n·d·(d + n_alphas·(n_y + 2))` MACs; below this a
/// `std::thread::scope` spawn costs more than the work it hands out (the
/// `gram_host::HOST_MACS_PER_UNIT` precedent, same order).
const MACS_PER_UNIT: usize = 1 << 18;

// ---------------------------------------------------------------------------
// gcv_mode
// ---------------------------------------------------------------------------

/// sklearn's `gcv_mode` selector — `{'auto', 'svd', 'eigen'}` (`None` at the
/// Python boundary maps to [`GcvMode::Auto`], as sklearn's own `_check_gcv_mode`
/// does).
///
/// A non-scalar hyperparameter, so the PyO3 boundary parses the sklearn STRING
/// into it via [`TryFrom<&str>`] and surfaces an unknown name as
/// [`BuildError::UnknownGcvMode`] (the `RidgeSolver` precedent, D-09).
///
/// See the module docs for why all three values take the same route in mlrs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcvMode {
    /// Pick the decomposition from the shape (sklearn's `None` / `'auto'`).
    Auto,
    /// sklearn's reduced-SVD route.
    Svd,
    /// sklearn's eigendecomposition route.
    Eigen,
}

impl GcvMode {
    /// The sklearn `gcv_mode` string.
    pub fn name(self) -> &'static str {
        match self {
            GcvMode::Auto => "auto",
            GcvMode::Svd => "svd",
            GcvMode::Eigen => "eigen",
        }
    }
}

impl TryFrom<&str> for GcvMode {
    type Error = BuildError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "auto" => Ok(GcvMode::Auto),
            "svd" => Ok(GcvMode::Svd),
            "eigen" => Ok(GcvMode::Eigen),
            other => Err(BuildError::UnknownGcvMode {
                value: other.to_string(),
            }),
        }
    }
}

/// Which symmetric matrix the GCV engine decomposes — the SMALLER of the two
/// Grams, which is what makes every `gcv_mode` affordable at any shape.
///
/// sklearn's `_check_gcv_mode` makes the same `n ≤ d` split for its
/// `"gram"`/`"cov"` routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcvRoute {
    /// `X̃X̃ᵀ`, `n × n` — taken when `n ≤ d` (the wide-design case).
    Gram,
    /// `X̃ᵀX̃`, `d × d` — taken when `n > d` (the usual case).
    Cov,
}

/// The route [`ridge_gcv`] will take for this shape.
///
/// `mode` is accepted for symmetry with sklearn's signature and is deliberately
/// not read: see the module docs for why the three modes coincide here.
pub fn resolve_route(_mode: GcvMode, n_samples: usize, n_features: usize) -> GcvRoute {
    if n_samples <= n_features {
        GcvRoute::Gram
    } else {
        GcvRoute::Cov
    }
}

// ---------------------------------------------------------------------------
// Element widening (the `gram_host::GramElem` idiom)
// ---------------------------------------------------------------------------

/// Host element types the engines dispatch to. The arithmetic is all `f64`
/// whatever the estimator's `F`, so the only thing the element type does is
/// widen — and stay `Pod` so the shared `gram_host` passes can take it.
trait Elem: Copy + Send + Sync + Pod {
    /// Widen one element to the `f64` the accumulators use.
    fn wide(self) -> f64;
}

impl Elem for f32 {
    #[inline]
    fn wide(self) -> f64 {
        self as f64
    }
}

impl Elem for f64 {
    #[inline]
    fn wide(self) -> f64 {
        self
    }
}

// ---------------------------------------------------------------------------
// GCV engine (cv = None)
// ---------------------------------------------------------------------------

/// Everything [`ridge_gcv`] produces. Every buffer is `f64` and host-resident;
/// the caller narrows at its own boundary.
#[derive(Debug, Clone, Default)]
pub struct GcvFit {
    /// `n_alphas × n_y` row-major PER-TARGET scores, `−mean_i(looeᵢₜ²)`.
    ///
    /// Empty when the caller asked for predictions instead (a non-`None`
    /// `scoring` means only the shim can score).
    pub scores: Vec<f64>,
    /// `n_alphas × d × n_y` row-major coefficients — every alpha's, because
    /// `alpha_per_target` picks a different winner per column.
    ///
    /// These are the coefficients of the CENTERED problem; the caller recovers
    /// `intercept_ = ȳ − x̄·coefᵀ` from [`GcvFit::x_offset`] /
    /// [`GcvFit::y_offset`] (D-05 — α never penalizes the intercept).
    pub coefs: Vec<f64>,
    /// `n × n_alphas × n_y` row-major (ROW-major in `n` so a worker owns a
    /// contiguous slice): squared LOO errors when scoring is `None`, rescaled
    /// LOO predictions when it is not. Empty unless the caller asked for it.
    pub cv_values: Vec<f64>,
    /// Weighted column means of `X` (`d`), or zeros when `fit_intercept` is
    /// false.
    pub x_offset: Vec<f64>,
    /// Weighted target means (`n_y`), or zeros when `fit_intercept` is false.
    pub y_offset: Vec<f64>,
    /// Which Gram was decomposed — reported so a caller (and the perf probe)
    /// can attribute a measurement to the route that produced it.
    pub route: GcvRoute,
}

impl Default for GcvRoute {
    fn default() -> Self {
        GcvRoute::Cov
    }
}

/// Generalized (leave-one-out) cross-validated ridge — sklearn's `_RidgeGCV`.
///
/// - `x` is the `n × d` row-major design and `y` the `n × n_y` row-major target,
///   both borrowed from host memory in the caller's element type.
/// - `sample_weight` is an optional length-`n` vector of non-negative finite
///   weights, already widened to `f64` (`ridge::validate_sample_weight` is the
///   shared validator).
/// - `alphas` must be strictly positive: the LOO identity divides by `α`, and
///   sklearn's own `_BaseRidgeCV.fit` rejects `alpha = 0` for `cv=None` with
///   `include_boundaries="neither"`.
/// - `want_predictions` switches the per-alpha output from squared errors to
///   rescaled LOO predictions (the non-`None` `scoring` arm).
/// - `store_cv_values` fills [`GcvFit::cv_values`]; `want_predictions` implies
///   it, because a Python-side scorer has nothing else to consume.
///
/// ## The derivation the three `gcv_mode` values share
/// Let `X̃` be the preprocessed design (weighted-centered, then `√w`-rescaled)
/// and `λ, V` the eigendecomposition of `X̃ᵀX̃`. sklearn's `"cov"` route computes,
/// per alpha, `Hinv = V·diag(1/(λ+α))·Vᵀ` and then `X̃·Hinv` — an `O(n·d²)` GEMM
/// per alpha. Substituting `W = X̃·V` (which satisfies `WᵀW = diag(λ)`) collapses
/// both quantities it needs onto `W`'s rows:
///
/// ```text
/// (X̃·Hinv·X̃ᵀ·ỹ)ᵢ  = Σₖ Wᵢₖ·(Vᵀ·X̃ᵀỹ)ₖ / (λₖ+α)
/// diag(X̃·Hinv·X̃ᵀ)ᵢ = Σₖ Wᵢₖ² / (λₖ+α)
/// coef            = V·diag(1/(λ+α))·Vᵀ·X̃ᵀỹ
/// ```
///
/// so `W` and `Vᵀ·X̃ᵀỹ` are formed once and the alpha loop is `O(n·d)`.
/// sklearn's `"svd"` route is the same expressions with `U = W·diag(1/σ)` and
/// `σ² = λ` substituted back in — `M = α/(σ²+α) − 1` obeys `M/σ² = −1/(λ+α)`,
/// which is exactly the weight above — and its `"gram"` route is the dual of it
/// for `n ≤ d`. All three therefore reduce to the two contractions this function
/// computes.
///
/// Returns [`PrimError::ShapeMismatch`] on inconsistent geometry and
/// [`BuildError::InvalidAlpha`] (through [`AlgoError`]) on a non-positive alpha.
#[allow(clippy::too_many_arguments)]
pub fn ridge_gcv<F: Pod>(
    x: &[F],
    y: &[F],
    n_samples: usize,
    n_features: usize,
    n_targets: usize,
    sample_weight: Option<&[f64]>,
    alphas: &[f64],
    fit_intercept: bool,
    mode: GcvMode,
    want_predictions: bool,
    store_cv_values: bool,
) -> Result<GcvFit, AlgoError> {
    match size_of::<F>() {
        4 => ridge_gcv_t::<f32>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            n_samples,
            n_features,
            n_targets,
            sample_weight,
            alphas,
            fit_intercept,
            mode,
            want_predictions,
            store_cv_values,
        ),
        8 => ridge_gcv_t::<f64>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            n_samples,
            n_features,
            n_targets,
            sample_weight,
            alphas,
            fit_intercept,
            mode,
            want_predictions,
            store_cv_values,
        ),
        other => unreachable!("ridge_cv is f32/f64 only, got a {other}-byte element"),
    }
}

/// Monomorphized body of [`ridge_gcv`].
#[allow(clippy::too_many_arguments)]
fn ridge_gcv_t<T: Elem>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    n_y: usize,
    sw: Option<&[f64]>,
    alphas: &[f64],
    fit_intercept: bool,
    mode: GcvMode,
    want_predictions: bool,
    store_cv_values: bool,
) -> Result<GcvFit, AlgoError> {
    let (sqrt_sw, sw_sum) = validate_gcv(x.len(), y.len(), n, d, n_y, sw, alphas)?;

    let emit_values = store_cv_values || want_predictions;
    match resolve_route(mode, n, d) {
        GcvRoute::Cov => gcv_cov::<T>(
            x,
            y,
            n,
            d,
            n_y,
            sw,
            &sqrt_sw,
            sw_sum,
            alphas,
            fit_intercept,
            want_predictions,
            emit_values,
        ),
        GcvRoute::Gram => gcv_gram::<T>(
            x,
            y,
            n,
            d,
            n_y,
            sw,
            &sqrt_sw,
            sw_sum,
            alphas,
            fit_intercept,
            want_predictions,
            emit_values,
        ),
    }
}

/// The shape/alpha/weight validation both `"cov"` arms run before either one
/// allocates, returning the two derived quantities the sweep needs: `√w` (ones
/// when unweighted, as sklearn's `sqrt_sw`) and `Σw`.
///
/// Extracted rather than duplicated because the device arm must reject exactly
/// what the host arm rejects — a `RidgeCV` that accepted `alpha = 0` on one
/// backend and not another would be a placement parameter changing behaviour,
/// which is the one thing `device` promises not to do.
fn validate_gcv(
    x_len: usize,
    y_len: usize,
    n: usize,
    d: usize,
    n_y: usize,
    sw: Option<&[f64]>,
    alphas: &[f64],
) -> Result<(Vec<f64>, f64), AlgoError> {
    validate_geometry(x_len, y_len, n, d, n_y)?;
    if alphas.is_empty() {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "alphas",
            rows: 1,
            cols: 1,
            len: 0,
        }));
    }
    for &a in alphas {
        if !(a > 0.0) || !a.is_finite() {
            return Err(AlgoError::Build(BuildError::InvalidAlpha {
                estimator: "ridge_cv",
                alpha: a,
            }));
        }
    }
    if let Some(w) = sw {
        if w.len() != n {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "sample_weight",
                rows: n,
                cols: 1,
                len: w.len(),
            }));
        }
    }

    // `√w` (or ones) and `Σw` — sklearn's `sqrt_sw` / `sw_sum`.
    let sqrt_sw: Vec<f64> = match sw {
        Some(w) => w.iter().map(|v| v.sqrt()).collect(),
        None => vec![1.0; n],
    };
    let sw_sum: f64 = sqrt_sw.iter().map(|s| s * s).sum();
    if !(sw_sum > 0.0) {
        return Err(AlgoError::ZeroSampleWeightSum {
            estimator: "ridge_cv",
        });
    }
    Ok((sqrt_sw, sw_sum))
}

// ---------------------------------------------------------------------------
// GCV engine — arm placement (RIDGECV-02)
// ---------------------------------------------------------------------------

/// Will [`ridge_gcv_auto`] take the DEVICE arm for this configuration?
///
/// Public so a caller can report `device_` and a perf probe can assert that the
/// arm it is timing is the arm it asked for (`mlrs-bench-verify-knob-is-live`:
/// a sweep against a gate that silently declined is vacuous).
///
/// Three questions in the order the `mlrs-device-param` rule requires —
/// CAPABILITY first, and `device = "gpu"` may override only the last:
///
/// 1. **Is this the `"cov"` route?** The `n ≤ d` route's cost is a serial
///    `O(n³)` eigendecomposition of the `n × n` Gram (module docs), which is the
///    shape a GPU is worst at and which no kernel here addresses. It has ONE
///    arm.
/// 2. **Can the device engine run at all?** `f64` device kernels, `d` inside the
///    sweep's shared-memory tile, a fused Gram at this `d`
///    ([`gcv_device_possible`]).
/// 3. **Is it worth the upload?** [`gcv_device_preferred`] — the only overridable
///    half, and the only one that reads `MLRS_RIDGECV_DEVICE`.
pub fn gcv_device_arm(
    device: Device,
    mode: GcvMode,
    n_samples: usize,
    n_features: usize,
    n_alphas: usize,
    n_targets: usize,
) -> bool {
    if resolve_route(mode, n_samples, n_features) != GcvRoute::Cov {
        return false;
    }
    if !gcv_device_possible(n_features) {
        return false;
    }
    device.prefers_device(|| gcv_device_preferred(n_samples, n_features, n_alphas, n_targets))
}

/// [`ridge_gcv`] with the arm chosen by [`gcv_device_arm`], reporting which one
/// ran.
///
/// The returned `&'static str` is what the estimator publishes as `device_`
/// (`"cpu"` / `"gpu"`). It is derived from the SAME boolean the branch below
/// takes, not recomputed, so the estimator cannot report one arm and run another
/// (`Device::resolved_name`'s contract).
///
/// Both arms produce the same [`GcvFit`] to within summation order — see
/// `prims::ridge_gcv`'s module docs for what that means and
/// `ridge_cv_device_test.rs` for the gate that holds it.
#[allow(clippy::too_many_arguments)]
pub fn ridge_gcv_auto<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[F],
    y: &[F],
    n_samples: usize,
    n_features: usize,
    n_targets: usize,
    sample_weight: Option<&[f64]>,
    alphas: &[f64],
    fit_intercept: bool,
    mode: GcvMode,
    want_predictions: bool,
    store_cv_values: bool,
    device: Device,
) -> Result<(GcvFit, &'static str), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let on_device = gcv_device_arm(
        device,
        mode,
        n_samples,
        n_features,
        alphas.len(),
        n_targets,
    );
    if !on_device {
        let fit = ridge_gcv::<F>(
            x,
            y,
            n_samples,
            n_features,
            n_targets,
            sample_weight,
            alphas,
            fit_intercept,
            mode,
            want_predictions,
            store_cv_values,
        )?;
        return Ok((fit, Device::resolved_name(true)));
    }

    let (sqrt_sw, sw_sum) = validate_gcv(
        x.len(),
        y.len(),
        n_samples,
        n_features,
        n_targets,
        sample_weight,
        alphas,
    )?;
    let dev = GcvDevice::from_host::<F>(
        pool,
        x,
        y,
        n_samples,
        n_features,
        n_targets,
        &sqrt_sw,
        sample_weight.is_some(),
    )
    .map_err(AlgoError::Prim)?;
    let fit = gcv_cov_device(
        pool,
        &dev,
        n_samples,
        n_features,
        n_targets,
        sw_sum,
        alphas,
        fit_intercept,
        want_predictions,
        store_cv_values || want_predictions,
    );
    dev.release_into(pool);
    Ok((fit?, Device::resolved_name(false)))
}

/// The device `"cov"` body, once the design is uploaded.
///
/// Split from [`ridge_gcv_auto`] so the upload has exactly ONE release path: an
/// error anywhere in here still returns the `n × d` device buffer to the pool.
///
/// Structurally it is [`gcv_cov`] with two substitutions — the normal-equation
/// sweep becomes [`GcvDevice::normal_equations`] and the streaming sweep becomes
/// [`GcvDevice::sweep`] — and the `O(d³)` middle is shared verbatim
/// ([`sym_eig`] + [`spectral_weights`]). That is deliberate: the LOO algebra
/// exists once, so the arms cannot disagree about anything except summation
/// order.
#[allow(clippy::too_many_arguments)]
fn gcv_cov_device(
    pool: &mut BufferPool<ActiveRuntime>,
    dev: &GcvDevice,
    n: usize,
    d: usize,
    n_y: usize,
    sw_sum: f64,
    alphas: &[f64],
    fit_intercept: bool,
    want_predictions: bool,
    emit_values: bool,
) -> Result<GcvFit, AlgoError> {
    let (x_offset, y_offset, gram, xty, xtsw) = dev
        .normal_equations(pool, fit_intercept)
        .map_err(AlgoError::Prim)?;

    let (lam, v) = sym_eig(&gram, d);
    let sp = spectral_weights(&lam, &v, &xty, &xtsw, alphas, d, n_y);

    let out = dev
        .sweep(
            pool,
            &x_offset,
            &y_offset,
            &v,
            &sp.g,
            &sp.gz,
            &sp.gzsw,
            alphas.len(),
            sw_sum,
            fit_intercept,
            want_predictions,
            emit_values,
        )
        .map_err(AlgoError::Prim)?;

    let inv_n = 1.0 / n as f64;
    let scores = if want_predictions {
        Vec::new()
    } else {
        out.score_sums.iter().map(|s| -s * inv_n).collect()
    };

    Ok(GcvFit {
        scores,
        coefs: sp.coefs,
        cv_values: out.cv_values,
        x_offset,
        y_offset,
        route: GcvRoute::Cov,
    })
}

/// The `n > d` route: eigendecompose the `d × d` Gram `X̃ᵀX̃`, then stream the
/// design once more to contract every alpha against `W = X̃·V` row-block by
/// row-block.
#[allow(clippy::too_many_arguments)]
fn gcv_cov<T: Elem>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    n_y: usize,
    sw: Option<&[f64]>,
    sqrt_sw: &[f64],
    sw_sum: f64,
    alphas: &[f64],
    fit_intercept: bool,
    want_predictions: bool,
    emit_values: bool,
) -> Result<GcvFit, AlgoError> {
    let n_alphas = alphas.len();

    // --- 1. Means + the `d × d` Gram and `Xᵀy`, in ONE parallel sweep. This is
    //        the SAME pass `Ridge`'s host arm runs (`gram_host`), so the two
    //        estimators cannot drift on weighted centering or on the `√w`
    //        rescale. ---
    let (x_offset, y_offset, gram, xty) =
        centered_gram_multi_xty::<T>(x, y, n, d, n_y, sw, fit_intercept);

    // `X̃ᵀ√w`, the operand of the intercept correction in the LOO denominator.
    // Zero by construction under `fit_intercept` (module docs) — accumulated
    // anyway rather than assumed.
    let xtsw = if fit_intercept {
        xt_sqrt_sw::<T>(x, n, d, &x_offset, sqrt_sw)
    } else {
        vec![0.0; d]
    };

    // --- 2. ONE eigendecomposition. `lam` descending, `v[j·d + k]` component
    //        `j` of eigenvector `k` (columns). ---
    let (lam, v) = sym_eig(&gram, d);

    // --- 3. Per-alpha spectral weights, all precomputed so the streaming sweep
    //        never divides. ---
    let Spectral {
        g,
        gz,
        gzsw,
        coefs,
    } = spectral_weights(&lam, &v, &xty, &xtsw, alphas, d, n_y);

    // --- 4. The streaming sweep: project each row block into the eigenbasis
    //        ONCE and contract it against every alpha. ---
    let mut cv_values = if emit_values {
        vec![0.0f64; n * n_alphas * n_y]
    } else {
        Vec::new()
    };
    let units = stream_units(n, d, n_alphas, n_y);
    let rows_per_unit = n.div_ceil(units).max(1);

    let mut scores = vec![0.0f64; n_alphas * n_y];
    let partials: Vec<Vec<f64>> = if units <= 1 {
        vec![cov_sweep::<T>(
            x,
            y,
            0,
            n,
            n,
            d,
            n_y,
            &x_offset,
            &y_offset,
            sqrt_sw,
            sw_sum,
            sw.is_some(),
            fit_intercept,
            &v,
            &g,
            &gz,
            &gzsw,
            n_alphas,
            want_predictions,
            if emit_values {
                Some(&mut cv_values)
            } else {
                None
            },
        )]
    } else {
        // Each worker owns a contiguous ROW range, so its slice of `cv_values`
        // (laid out `n`-major precisely for this) is one unbroken range and no
        // sharing beyond the read-only decomposition is needed.
        let chunk = rows_per_unit * n_alphas * n_y;
        let mut out_chunks: Vec<Option<&mut [f64]>> = if emit_values {
            cv_values.chunks_mut(chunk).map(Some).collect()
        } else {
            (0..units).map(|_| None).collect()
        };
        std::thread::scope(|scope| {
            let handles: Vec<_> = out_chunks
                .drain(..)
                .enumerate()
                .filter_map(|(u, out)| {
                    let r0 = u * rows_per_unit;
                    if r0 >= n {
                        return None;
                    }
                    let rows = rows_per_unit.min(n - r0);
                    let (x_offset, y_offset) = (&x_offset, &y_offset);
                    let (v, g, gz, gzsw) = (&v, &g, &gz, &gzsw);
                    Some(scope.spawn(move || {
                        cov_sweep::<T>(
                            x,
                            y,
                            r0,
                            rows,
                            n,
                            d,
                            n_y,
                            x_offset,
                            y_offset,
                            sqrt_sw,
                            sw_sum,
                            sw.is_some(),
                            fit_intercept,
                            v,
                            g,
                            gz,
                            gzsw,
                            n_alphas,
                            want_predictions,
                            out,
                        )
                    }))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };
    for part in &partials {
        for (dst, src) in scores.iter_mut().zip(part.iter()) {
            *dst += src;
        }
    }
    let inv_n = 1.0 / n as f64;
    for s in scores.iter_mut() {
        *s = -*s * inv_n;
    }

    Ok(GcvFit {
        scores: if want_predictions { Vec::new() } else { scores },
        coefs,
        cv_values,
        x_offset,
        y_offset,
        route: GcvRoute::Cov,
    })
}

/// The per-alpha operands both `"cov"` arms contract against `W`'s rows.
///
/// One struct rather than a four-tuple because the host sweep and the device
/// launch each take all four and would otherwise thread them positionally
/// through two long argument lists.
struct Spectral {
    /// `n_alphas × d`, `1/(λₖ + α)`.
    g: Vec<f64>,
    /// `n_alphas × d × n_y`, `g ⊙ (Vᵀ·X̃ᵀỹ)`.
    gz: Vec<f64>,
    /// `n_alphas × d`, `g ⊙ (Vᵀ·X̃ᵀ√w)`.
    gzsw: Vec<f64>,
    /// `n_alphas × d × n_y`, `V·diag(g)·Vᵀ·X̃ᵀỹ` — EVERY alpha's coefficients
    /// (`alpha_per_target` picks a different winner per column).
    coefs: Vec<f64>,
}

/// Everything the alpha loop needs, from the spectrum and the `y`-side
/// projections — the `O(n_alphas · d² · n_y)` block that sits between the
/// eigendecomposition and the sweep.
///
/// Shared verbatim by the host and device `"cov"` arms. It is the reason the
/// two arms cannot drift on the LOO algebra: only the SWEEP differs between
/// them, and the sweep consumes exactly this.
fn spectral_weights(
    lam: &[f64],
    v: &[f64],
    xty: &[f64],
    xtsw: &[f64],
    alphas: &[f64],
    d: usize,
    n_y: usize,
) -> Spectral {
    let n_alphas = alphas.len();

    // `z = Vᵀ·Xᵀy` (d × n_y) and `zsw = Vᵀ·Xᵀ√w` (d) — the alpha loop's only
    // `y`-dependent operands, formed once.
    let mut z = vec![0.0f64; d * n_y];
    for k in 0..d {
        for t in 0..n_y {
            let mut acc = 0.0;
            for j in 0..d {
                acc += v[j * d + k] * xty[j * n_y + t];
            }
            z[k * n_y + t] = acc;
        }
    }
    let mut zsw = vec![0.0f64; d];
    for k in 0..d {
        let mut acc = 0.0;
        for j in 0..d {
            acc += v[j * d + k] * xtsw[j];
        }
        zsw[k] = acc;
    }

    let mut g = vec![0.0f64; n_alphas * d];
    let mut gz = vec![0.0f64; n_alphas * d * n_y];
    let mut gzsw = vec![0.0f64; n_alphas * d];
    let mut coefs = vec![0.0f64; n_alphas * d * n_y];
    for (a, &alpha) in alphas.iter().enumerate() {
        for k in 0..d {
            let gk = 1.0 / (lam[k] + alpha);
            g[a * d + k] = gk;
            gzsw[a * d + k] = gk * zsw[k];
            for t in 0..n_y {
                gz[(a * d + k) * n_y + t] = gk * z[k * n_y + t];
            }
        }
        // coef = V·diag(1/(λ+α))·Vᵀ·Xᵀy — the eigenbasis form of
        // `(XᵀX + αI)⁻¹Xᵀy`, so the singular-Gram case degrades smoothly
        // instead of failing a Cholesky pivot.
        for j in 0..d {
            for t in 0..n_y {
                let mut acc = 0.0;
                for k in 0..d {
                    acc += v[j * d + k] * gz[(a * d + k) * n_y + t];
                }
                coefs[(a * d + j) * n_y + t] = acc;
            }
        }
    }

    Spectral {
        g,
        gz,
        gzsw,
        coefs,
    }
}

/// One worker's share of the `"cov"` streaming sweep.
///
/// Returns the per-`(alpha, target)` sums of `looe²` for its rows (the caller
/// combines and negates/normalizes), and fills its slice of `cv_values` when
/// asked.
#[allow(clippy::too_many_arguments)]
fn cov_sweep<T: Elem>(
    x: &[T],
    y: &[T],
    row0: usize,
    rows: usize,
    _n: usize,
    d: usize,
    n_y: usize,
    x_offset: &[f64],
    y_offset: &[f64],
    sqrt_sw: &[f64],
    sw_sum: f64,
    weighted: bool,
    fit_intercept: bool,
    v: &[f64],
    g: &[f64],
    gz: &[f64],
    gzsw: &[f64],
    n_alphas: usize,
    want_predictions: bool,
    mut out: Option<&mut [f64]>,
) -> Vec<f64> {
    let mut acc = vec![0.0f64; n_alphas * n_y];
    let mut xt = vec![0.0f64; ROW_BLOCK * d];
    let mut w = vec![0.0f64; ROW_BLOCK * d];
    let mut yt = vec![0.0f64; ROW_BLOCK * n_y];
    let mut tt = vec![0.0f64; n_y];

    let mut done = 0usize;
    while done < rows {
        let block = ROW_BLOCK.min(rows - done);
        let base = row0 + done;

        // Preprocess the block: X̃ = √w ⊙ (X − x̄), ỹ = √w ⊙ (y − ȳ).
        for r in 0..block {
            let s = sqrt_sw[base + r];
            let src = &x[(base + r) * d..(base + r) * d + d];
            let dst = &mut xt[r * d..r * d + d];
            for j in 0..d {
                dst[j] = s * (src[j].wide() - x_offset[j]);
            }
            for t in 0..n_y {
                yt[r * n_y + t] = s * (y[(base + r) * n_y + t].wide() - y_offset[t]);
            }
        }

        // Project into the eigenbasis: W = X̃·V. This is the block's `O(b·d²)`
        // term and the only one that does not repeat per alpha.
        //
        // Deliberately ONE ROW AT A TIME, which re-reads all of `V` per row —
        // 512 KiB at `d = 256`. Sharing one `V` read across four accumulators
        // (classic outer-product blocking) was implemented and MEASURED: 1.05x,
        // 0.98x, 1.01x, 0.99x at `d = 64/128/256/512` on a quiet 16-core box.
        // Flat, so the loop is latency/compute-bound at the ~21-25 GFLOP/s
        // `ridge_cv_perf_test.rs::shape_ladder` reports, not bandwidth-bound,
        // and the extra arm was removed rather than kept unjustified. Do not
        // re-add it without a measurement that says otherwise.
        for r in 0..block {
            let row = &xt[r * d..r * d + d];
            let wrow = &mut w[r * d..r * d + d];
            wrow.fill(0.0);
            for (j, &xj) in row.iter().enumerate() {
                let vcol = &v[j * d..j * d + d];
                for (k, wk) in wrow.iter_mut().enumerate() {
                    *wk += xj * vcol[k];
                }
            }
        }

        for r in 0..block {
            let i = base + r;
            let wrow = &w[r * d..r * d + d];
            let s_i = sqrt_sw[i];
            for a in 0..n_alphas {
                let ga = &g[a * d..a * d + d];
                let gza = &gz[a * d * n_y..a * d * n_y + d * n_y];
                let gzswa = &gzsw[a * d..a * d + d];

                tt.iter_mut().for_each(|v| *v = 0.0);
                let mut q = 0.0f64;
                let mut rsw = 0.0f64;
                for k in 0..d {
                    let wk = wrow[k];
                    q += ga[k] * wk * wk;
                    rsw += wk * gzswa[k];
                    let row = &gza[k * n_y..k * n_y + n_y];
                    for t in 0..n_y {
                        tt[t] += wk * row[t];
                    }
                }

                // alpha_d = 1 − diag(X·Hinv·Xᵀ) − (√w − X·Hinv·Xᵀ√w)·√w / Σw
                let mut denom = 1.0 - q;
                if fit_intercept {
                    denom -= (s_i - rsw) * s_i / sw_sum;
                }
                for t in 0..n_y {
                    let looe = (yt[r * n_y + t] - tt[t]) / denom;
                    if want_predictions {
                        // sklearn: predictions = y − looe, un-rescaled by √w
                        // ONLY when weights were given, then re-offset.
                        let mut p = yt[r * n_y + t] - looe;
                        if weighted {
                            p /= s_i;
                        }
                        p += y_offset[t];
                        if let Some(o) = out.as_deref_mut() {
                            o[(done + r) * n_alphas * n_y + a * n_y + t] = p;
                        }
                    } else {
                        let sq = looe * looe;
                        acc[a * n_y + t] += sq;
                        if let Some(o) = out.as_deref_mut() {
                            o[(done + r) * n_alphas * n_y + a * n_y + t] = sq;
                        }
                    }
                }
            }
        }
        done += block;
    }
    acc
}

/// The `n ≤ d` route: eigendecompose the `n × n` Gram `X̃X̃ᵀ` (sklearn's
/// `"gram"`), which is the SMALLER matrix there.
///
/// No streaming: `n ≤ d` bounds the preprocessed design at `d²` doubles, and
/// every per-alpha quantity is already `O(n²)`.
#[allow(clippy::too_many_arguments)]
fn gcv_gram<T: Elem>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    n_y: usize,
    sw: Option<&[f64]>,
    sqrt_sw: &[f64],
    sw_sum: f64,
    alphas: &[f64],
    fit_intercept: bool,
    want_predictions: bool,
    emit_values: bool,
) -> Result<GcvFit, AlgoError> {
    let n_alphas = alphas.len();
    let (x_offset, y_offset) = weighted_means::<T>(x, y, n, d, n_y, sw, fit_intercept);

    // Materialize the preprocessed design — `n·d ≤ d²` here, and both the Gram
    // and the `coef = X̃ᵀc` back-projection read it.
    let mut xt = vec![0.0f64; n * d];
    let mut yt = vec![0.0f64; n * n_y];
    for i in 0..n {
        let s = sqrt_sw[i];
        for j in 0..d {
            xt[i * d + j] = s * (x[i * d + j].wide() - x_offset[j]);
        }
        for t in 0..n_y {
            yt[i * n_y + t] = s * (y[i * n_y + t].wide() - y_offset[t]);
        }
    }

    // K = X̃X̃ᵀ (n × n) — the FULL matrix, not the triangle, across contiguous
    // row chunks.
    //
    // Computing both halves doubles the flops and is still far cheaper: a
    // triangular sweep has row `i` costing `i+1` dots, so equal ROW counts give
    // wildly unequal WORK and the last worker finishes ~2× after the first,
    // which on 16 threads wastes more than the half it saved. Equal-row chunks
    // of the full product are perfectly balanced and let each worker own a
    // contiguous, non-overlapping slice of `k` with no sharing at all.
    let mut k = vec![0.0f64; n * n];
    let units = units_for(n.saturating_mul(n).saturating_mul(d), n);
    if units <= 1 {
        gram_rows(&xt, 0, n, n, d, &mut k);
    } else {
        let rows_per_unit = n.div_ceil(units);
        std::thread::scope(|scope| {
            for (u, chunk) in k.chunks_mut(rows_per_unit * n).enumerate() {
                let r0 = u * rows_per_unit;
                let rows = chunk.len() / n;
                let xt = &xt;
                scope.spawn(move || gram_rows(xt, r0, rows, n, d, chunk));
            }
        });
    }
    let (lam, q) = sym_eig(&k, n);

    // Qᵀỹ (n × n_y) and Qᵀ√w (n).
    let mut qty = vec![0.0f64; n * n_y];
    let mut qtsw = vec![0.0f64; n];
    for kk in 0..n {
        for t in 0..n_y {
            let mut acc = 0.0;
            for i in 0..n {
                acc += q[i * n + kk] * yt[i * n_y + t];
            }
            qty[kk * n_y + t] = acc;
        }
        let mut acc = 0.0;
        for i in 0..n {
            acc += q[i * n + kk] * sqrt_sw[i];
        }
        qtsw[kk] = acc;
    }

    // Per-alpha spectral weights, precomputed together so the two threaded
    // passes below each cross a thread boundary ONCE instead of once per alpha
    // — at 30 alphas a per-alpha `thread::scope` would spend more on spawns
    // than on the arithmetic they carry.
    let mut dg = vec![0.0f64; n_alphas * n];
    for (a, &alpha) in alphas.iter().enumerate() {
        for kk in 0..n {
            dg[a * n + kk] = 1.0 / (lam[kk] + alpha);
        }
    }

    let mut coefs = vec![0.0f64; n_alphas * d * n_y];
    let mut cv_values = if emit_values {
        vec![0.0f64; n * n_alphas * n_y]
    } else {
        Vec::new()
    };
    // The dual coefficients for EVERY alpha (`n × n_alphas × n_y`), so the
    // back-projection below is one pass rather than one per alpha. Same size as
    // `cv_values`, three orders below the design.
    let mut c_all = vec![0.0f64; n * n_alphas * n_y];

    // --- Pass 1: rows in parallel, alphas inside. ---
    let units = units_for(
        n.saturating_mul(n_alphas)
            .saturating_mul(n)
            .saturating_mul(n_y + 2),
        n,
    );
    let rows_per_unit = n.div_ceil(units).max(1);
    let ctx = GramSolve {
        n,
        n_y,
        n_alphas,
        sw_sum,
        weighted: sw.is_some(),
        fit_intercept,
        want_predictions,
    };
    let partials: Vec<Vec<f64>> = if units <= 1 {
        vec![gram_rows_solve(
            &ctx,
            0,
            n,
            &q,
            &dg,
            &qty,
            &qtsw,
            &yt,
            sqrt_sw,
            &y_offset,
            &mut c_all,
            if emit_values {
                Some(&mut cv_values)
            } else {
                None
            },
        )]
    } else {
        let stride = rows_per_unit * n_alphas * n_y;
        let mut c_chunks: Vec<&mut [f64]> = c_all.chunks_mut(stride).collect();
        let mut out_chunks: Vec<Option<&mut [f64]>> = if emit_values {
            cv_values.chunks_mut(stride).map(Some).collect()
        } else {
            (0..c_chunks.len()).map(|_| None).collect()
        };
        std::thread::scope(|scope| {
            let handles: Vec<_> = c_chunks
                .drain(..)
                .zip(out_chunks.drain(..))
                .enumerate()
                .map(|(u, (cc, oc))| {
                    let r0 = u * rows_per_unit;
                    let rows = cc.len() / (n_alphas * n_y);
                    let (ctx, q, dg, qty, qtsw, yt, y_offset) =
                        (&ctx, &q, &dg, &qty, &qtsw, &yt, &y_offset);
                    scope.spawn(move || {
                        gram_rows_solve(
                            ctx, r0, rows, q, dg, qty, qtsw, yt, sqrt_sw, y_offset,
                            cc, oc,
                        )
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };
    let mut scores = vec![0.0f64; n_alphas * n_y];
    for p in &partials {
        for (dst, src) in scores.iter_mut().zip(p.iter()) {
            *dst += src;
        }
    }

    // --- Pass 2: `coef = X̃ᵀ·c` (the dual-to-primal step, sklearn's `XT_c`),
    //     parallel over FEATURE blocks with the alpha loop inside. `d ≥ n` on
    //     this route, so the features are the long axis and the right one to
    //     split. `coefs` is alpha-major, so a feature block is not contiguous
    //     in it: each worker fills its own strip and the strips are stitched
    //     back on the driving thread (an `O(n_alphas·d)` copy against an
    //     `O(n_alphas·n·d)` product). ---
    let feat_units = units_for(
        n_alphas.saturating_mul(d).saturating_mul(n).saturating_mul(n_y),
        d,
    );
    let feats_per_unit = d.div_ceil(feat_units).max(1);
    if feat_units <= 1 {
        gram_backproject(&xt, &c_all, 0, d, n, d, n_y, n_alphas, &mut coefs);
    } else {
        let strips: Vec<(usize, usize, Vec<f64>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..feat_units)
                .filter_map(|u| {
                    let j0 = u * feats_per_unit;
                    if j0 >= d {
                        return None;
                    }
                    let feats = feats_per_unit.min(d - j0);
                    let (xt, c_all) = (&xt, &c_all);
                    Some(scope.spawn(move || {
                        let mut strip = vec![0.0f64; n_alphas * feats * n_y];
                        gram_backproject(
                            xt, c_all, j0, feats, n, feats, n_y, n_alphas, &mut strip,
                        );
                        (j0, feats, strip)
                    }))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (j0, feats, strip) in strips {
            for a in 0..n_alphas {
                let src = &strip[a * feats * n_y..(a + 1) * feats * n_y];
                coefs[(a * d + j0) * n_y..(a * d + j0 + feats) * n_y]
                    .copy_from_slice(src);
            }
        }
    }

    let inv_n = 1.0 / n as f64;
    for s in scores.iter_mut() {
        *s = -*s * inv_n;
    }

    Ok(GcvFit {
        scores: if want_predictions { Vec::new() } else { scores },
        coefs,
        cv_values,
        x_offset,
        y_offset,
        route: GcvRoute::Gram,
    })
}

/// `rows` rows of `K = X̃X̃ᵀ`, starting at `row0` — one worker's share.
///
/// Writes the FULL row (both triangles), which is what makes equal-row chunks
/// equal-work; see the call site for why the triangular saving is a false
/// economy on a thread pool.
fn gram_rows(xt: &[f64], row0: usize, rows: usize, n: usize, d: usize, out: &mut [f64]) {
    for r in 0..rows {
        let a = &xt[(row0 + r) * d..(row0 + r) * d + d];
        for j in 0..n {
            let b = &xt[j * d..j * d + d];
            let mut acc = 0.0f64;
            for c in 0..d {
                acc += a[c] * b[c];
            }
            out[r * n + j] = acc;
        }
    }
}

/// The shape-and-flag bundle [`gram_rows_solve`] needs, so the worker closure
/// takes one reference instead of eleven scalars.
struct GramSolve {
    n: usize,
    n_y: usize,
    n_alphas: usize,
    sw_sum: f64,
    weighted: bool,
    fit_intercept: bool,
    want_predictions: bool,
}

/// One worker's rows of the `"gram"` route's per-alpha solve.
///
/// Fills its slice of `c_all` (the dual coefficients, `rows × n_alphas × n_y`)
/// and of `out` (squared errors or predictions), and returns its per-`(alpha,
/// target)` sums of `looe²`.
#[allow(clippy::too_many_arguments)]
fn gram_rows_solve(
    ctx: &GramSolve,
    row0: usize,
    rows: usize,
    q: &[f64],
    dg: &[f64],
    qty: &[f64],
    qtsw: &[f64],
    yt: &[f64],
    sqrt_sw: &[f64],
    y_offset: &[f64],
    c_all: &mut [f64],
    mut out: Option<&mut [f64]>,
) -> Vec<f64> {
    let (n, n_y, n_alphas) = (ctx.n, ctx.n_y, ctx.n_alphas);
    let mut acc = vec![0.0f64; n_alphas * n_y];
    for r in 0..rows {
        let i = row0 + r;
        let qrow = &q[i * n..i * n + n];
        let s_i = sqrt_sw[i];
        for a in 0..n_alphas {
            let dga = &dg[a * n..a * n + n];
            let mut dd = 0.0f64;
            let mut gsw = 0.0f64;
            for kk in 0..n {
                let qk = qrow[kk];
                let w = dga[kk];
                dd += w * qk * qk;
                gsw += qk * w * qtsw[kk];
            }
            if ctx.fit_intercept {
                dd -= gsw * s_i / ctx.sw_sum;
            }
            for t in 0..n_y {
                let mut ci = 0.0f64;
                for kk in 0..n {
                    ci += qrow[kk] * dga[kk] * qty[kk * n_y + t];
                }
                c_all[(r * n_alphas + a) * n_y + t] = ci;
                let looe = ci / dd;
                if ctx.want_predictions {
                    let mut p = yt[i * n_y + t] - looe;
                    if ctx.weighted {
                        p /= s_i;
                    }
                    p += y_offset[t];
                    if let Some(o) = out.as_deref_mut() {
                        o[(r * n_alphas + a) * n_y + t] = p;
                    }
                } else {
                    let sq = looe * looe;
                    acc[a * n_y + t] += sq;
                    if let Some(o) = out.as_deref_mut() {
                        o[(r * n_alphas + a) * n_y + t] = sq;
                    }
                }
            }
        }
    }
    acc
}

/// `coef = X̃ᵀ·c` for a block of `feats` features starting at `j0`, every alpha.
///
/// `out` is `n_alphas × out_d × n_y` row-major; `out_d` is `feats` for a
/// worker's private strip and `d` for the single-threaded whole-matrix call.
#[allow(clippy::too_many_arguments)]
fn gram_backproject(
    xt: &[f64],
    c_all: &[f64],
    j0: usize,
    feats: usize,
    n: usize,
    out_d: usize,
    n_y: usize,
    n_alphas: usize,
    out: &mut [f64],
) {
    let d = xt.len() / n;
    for a in 0..n_alphas {
        for jj in 0..feats {
            let j = j0 + jj;
            for t in 0..n_y {
                let mut sum = 0.0f64;
                for i in 0..n {
                    sum += xt[i * d + j] * c_all[(i * n_alphas + a) * n_y + t];
                }
                out[(a * out_d + jj) * n_y + t] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Grid engine (cv != None)
// ---------------------------------------------------------------------------

/// What [`ridge_cv_grid`] produces.
#[derive(Debug, Clone, Default)]
pub struct GridFit {
    /// `n_splits × n_alphas` row-major R² test scores — `GridSearchCV`'s
    /// DEFAULT scoring for a regressor, computed here so the common case never
    /// ships a prediction vector back to Python.
    pub scores: Vec<f64>,
    /// Per-split test predictions, laid out split-major:
    /// `[prefix[s]·n_alphas·n_y + a·(|testₛ|·n_y) + j·n_y + t]`, where
    /// `prefix[s]` is the running sum of test-fold sizes. Empty unless the
    /// caller asked for it (a non-`None` `scoring`).
    pub predictions: Vec<f64>,
}

/// `GridSearchCV(Ridge(fit_intercept=…), {'alpha': alphas}, cv=cv)` — sklearn's
/// `cv != None` arm of `RidgeCV`, with the Gram hoisted out of the alpha loop.
///
/// `splits` is the materialized index pair per fold, exactly as
/// `model_selection::check_cv` produces it (the shim passes sklearn-identical
/// rows; this function never invents a split).
///
/// ## Why this is not just a loop over `Ridge::fit`
/// sklearn refits from scratch for every `(alpha, split)` pair, and each refit
/// re-forms `XᵀX` — an `O(n_train·d²)` reduction — inside `_solve_cholesky`. The
/// Gram depends only on the TRAIN ROWS, never on `alpha`, so it is formed once
/// per split here and every alpha then costs one `O(d³/3)` factorization plus an
/// `O(n_test·d)` prediction. The saving is the same shape as the GCV engine's,
/// for the same reason.
///
/// `alpha = 0` is accepted on this arm (sklearn's `include_boundaries="left"`),
/// because a plain least-squares fit is a legitimate grid point when the fold is
/// genuinely held out.
///
/// ## `weighted_score` is not the same switch as `sample_weight`
/// sklearn forwards `sample_weight` to the TEST-fold scorer as well as to the
/// fit — but only when the scorer accepts it (`_check_scorers_accept_sample_
/// weight`), and the default `GridSearchCV` scorer for a regressor does. So the
/// held-out R² is WEIGHTED by default and unweighted for a scorer like
/// `max_error` that has no `sample_weight` parameter. Folding that into
/// `sample_weight.is_some()` would silently pick the wrong one of two answers
/// that differ in the fourth decimal — visible only as a different `alpha_` on
/// some datasets — so the caller decides and passes it explicitly.
#[allow(clippy::too_many_arguments)]
pub fn ridge_cv_grid<F: Pod>(
    x: &[F],
    y: &[F],
    n_samples: usize,
    n_features: usize,
    n_targets: usize,
    sample_weight: Option<&[f64]>,
    alphas: &[f64],
    fit_intercept: bool,
    splits: &[(Vec<usize>, Vec<usize>)],
    want_predictions: bool,
    weighted_score: bool,
) -> Result<GridFit, AlgoError> {
    match size_of::<F>() {
        4 => ridge_cv_grid_t::<f32>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            n_samples,
            n_features,
            n_targets,
            sample_weight,
            alphas,
            fit_intercept,
            splits,
            want_predictions,
            weighted_score,
        ),
        8 => ridge_cv_grid_t::<f64>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(y),
            n_samples,
            n_features,
            n_targets,
            sample_weight,
            alphas,
            fit_intercept,
            splits,
            want_predictions,
            weighted_score,
        ),
        other => unreachable!("ridge_cv is f32/f64 only, got a {other}-byte element"),
    }
}

/// Monomorphized body of [`ridge_cv_grid`].
#[allow(clippy::too_many_arguments)]
fn ridge_cv_grid_t<T: Elem>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    n_y: usize,
    sw: Option<&[f64]>,
    alphas: &[f64],
    fit_intercept: bool,
    splits: &[(Vec<usize>, Vec<usize>)],
    want_predictions: bool,
    weighted_score: bool,
) -> Result<GridFit, AlgoError> {
    validate_geometry(x.len(), y.len(), n, d, n_y)?;
    if alphas.is_empty() || splits.is_empty() {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: if alphas.is_empty() { "alphas" } else { "cv" },
            rows: 1,
            cols: 1,
            len: 0,
        }));
    }
    for &a in alphas {
        if !(a >= 0.0) || !a.is_finite() {
            return Err(AlgoError::Build(BuildError::InvalidAlpha {
                estimator: "ridge_cv",
                alpha: a,
            }));
        }
    }
    for (train, test) in splits {
        for &i in train.iter().chain(test.iter()) {
            if i >= n {
                return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                    operand: "cv split index",
                    rows: n,
                    cols: 1,
                    len: i,
                }));
            }
        }
    }

    let n_alphas = alphas.len();
    let mut prefix = Vec::with_capacity(splits.len() + 1);
    let mut total_test = 0usize;
    for (_, test) in splits {
        prefix.push(total_test);
        total_test += test.len();
    }

    let mut scores = vec![0.0f64; splits.len() * n_alphas];
    let mut predictions = if want_predictions {
        vec![0.0f64; total_test * n_alphas * n_y]
    } else {
        Vec::new()
    };

    // Scratch reused across splits: the gathered train/test rows.
    let mut xtr: Vec<T> = Vec::new();
    let mut ytr: Vec<T> = Vec::new();

    for (s, (train, test)) in splits.iter().enumerate() {
        if train.is_empty() || test.is_empty() {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "cv split",
                rows: n,
                cols: 1,
                len: 0,
            }));
        }
        xtr.clear();
        ytr.clear();
        xtr.reserve(train.len() * d);
        ytr.reserve(train.len() * n_y);
        for &i in train {
            xtr.extend_from_slice(&x[i * d..i * d + d]);
            ytr.extend_from_slice(&y[i * n_y..i * n_y + n_y]);
        }
        let swtr: Option<Vec<f64>> = sw.map(|w| train.iter().map(|&i| w[i]).collect());

        // ONE Gram per split, shared by every alpha.
        let (x_mean, y_mean, gram, xty) = centered_gram_multi_xty::<T>(
            &xtr,
            &ytr,
            train.len(),
            d,
            n_y,
            swtr.as_deref(),
            fit_intercept,
        );

        // The test fold's per-target mean, for R²'s denominator. Weighted iff
        // the scorer takes the weights (see the doc on `weighted_score`).
        let score_w: Option<&[f64]> = if weighted_score { sw } else { None };
        let mut y_test_mean = vec![0.0f64; n_y];
        let mut w_test_sum = 0.0f64;
        for &i in test {
            let wi = score_w.map_or(1.0, |w| w[i]);
            w_test_sum += wi;
            for t in 0..n_y {
                y_test_mean[t] += wi * y[i * n_y + t].wide();
            }
        }
        for m in y_test_mean.iter_mut() {
            *m /= w_test_sum;
        }

        let mut rhs = vec![0.0f64; d];
        for (a, &alpha) in alphas.iter().enumerate() {
            let mut coef = vec![0.0f64; d * n_y];
            let mut intercept = vec![0.0f64; n_y];
            for t in 0..n_y {
                for j in 0..d {
                    rhs[j] = xty[j * n_y + t];
                }
                // sklearn's `_solve_cholesky` with its `except LinAlgError:
                // solver = "svd"` retry — the same pair `Ridge`'s host arm uses.
                let w = match cholesky_ridge(&gram, &rhs, d, alpha) {
                    Some(w) => w,
                    None => gram_eig_ridge(&gram, &rhs, d, alpha),
                };
                let mut b = y_mean[t];
                for j in 0..d {
                    coef[j * n_y + t] = w[j];
                    b -= x_mean[j] * w[j];
                }
                intercept[t] = if fit_intercept { b } else { 0.0 };
            }

            // Predict the held-out fold and reduce to R² (uniform average over
            // targets, sklearn's `RegressorMixin.score` default).
            let mut ss_res = vec![0.0f64; n_y];
            let mut ss_tot = vec![0.0f64; n_y];
            for (j, &i) in test.iter().enumerate() {
                let row = &x[i * d..i * d + d];
                let wi = score_w.map_or(1.0, |w| w[i]);
                for t in 0..n_y {
                    let mut p = intercept[t];
                    for (c, &xv) in row.iter().enumerate() {
                        p += xv.wide() * coef[c * n_y + t];
                    }
                    let truth = y[i * n_y + t].wide();
                    ss_res[t] += wi * (truth - p) * (truth - p);
                    ss_tot[t] += wi * (truth - y_test_mean[t]) * (truth - y_test_mean[t]);
                    if want_predictions {
                        predictions
                            [prefix[s] * n_alphas * n_y + a * test.len() * n_y + j * n_y + t] = p;
                    }
                }
            }
            let mut r2 = 0.0f64;
            for t in 0..n_y {
                r2 += r2_score(ss_res[t], ss_tot[t]);
            }
            scores[s * n_alphas + a] = r2 / n_y as f64;
        }
    }

    Ok(GridFit {
        scores,
        predictions,
    })
}

/// `sklearn.metrics.r2_score` for one target, from its two sums of squares.
///
/// The two degenerate branches are sklearn's, not a simplification: a constant
/// `y_true` gives `1.0` when the prediction is exact and `0.0` when it is not,
/// rather than a division by zero.
fn r2_score(ss_res: f64, ss_tot: f64) -> f64 {
    if ss_tot != 0.0 {
        1.0 - ss_res / ss_tot
    } else if ss_res != 0.0 {
        0.0
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Shared host passes
// ---------------------------------------------------------------------------

/// Geometry guard shared by both engines (the slice twin of
/// `typestate::validate_geometry`, which reads a `DeviceArray` length these
/// entry points do not have).
fn validate_geometry(
    x_len: usize,
    y_len: usize,
    n: usize,
    d: usize,
    n_y: usize,
) -> Result<(), AlgoError> {
    if n == 0 || d == 0 || n_y == 0 || x_len != n * d {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x_len,
        }));
    }
    if y_len != n * n_y {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "y",
            rows: n,
            cols: n_y,
            len: y_len,
        }));
    }
    Ok(())
}

/// `X̃ᵀ√w` = `Σᵢ wᵢ·(xᵢ − x̄)`, accumulated over contiguous row chunks.
///
/// Analytically zero under weighted centering — see the module docs for why it
/// is computed rather than assumed.
fn xt_sqrt_sw<T: Elem>(
    x: &[T],
    n: usize,
    d: usize,
    x_offset: &[f64],
    sqrt_sw: &[f64],
) -> Vec<f64> {
    let units = stream_units(n, d, 1, 1).min(4);
    if units <= 1 {
        return xt_sqrt_sw_rows(x, 0, n, d, x_offset, sqrt_sw);
    }
    let rows_per_unit = n.div_ceil(units);
    let partials: Vec<Vec<f64>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..units)
            .filter_map(|u| {
                let r0 = u * rows_per_unit;
                if r0 >= n {
                    return None;
                }
                let rows = rows_per_unit.min(n - r0);
                Some(scope.spawn(move || xt_sqrt_sw_rows(x, r0, rows, d, x_offset, sqrt_sw)))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = vec![0.0f64; d];
    for p in partials {
        for (dst, src) in out.iter_mut().zip(p.iter()) {
            *dst += src;
        }
    }
    out
}

/// One worker's rows of [`xt_sqrt_sw`].
fn xt_sqrt_sw_rows<T: Elem>(
    x: &[T],
    row0: usize,
    rows: usize,
    d: usize,
    x_offset: &[f64],
    sqrt_sw: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0f64; d];
    for r in 0..rows {
        let i = row0 + r;
        let w = sqrt_sw[i] * sqrt_sw[i];
        let row = &x[i * d..i * d + d];
        for (j, o) in out.iter_mut().enumerate() {
            *o += w * (row[j].wide() - x_offset[j]);
        }
    }
    out
}

/// Weighted column means of `X` and per-target means of `y` — the `"gram"`
/// route's preprocessing, which does not want the `O(n·d²)` Gram
/// [`centered_gram_multi_xty`] would form alongside them.
fn weighted_means<T: Elem>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
    n_y: usize,
    sw: Option<&[f64]>,
    fit_intercept: bool,
) -> (Vec<f64>, Vec<f64>) {
    if !fit_intercept {
        return (vec![0.0; d], vec![0.0; n_y]);
    }
    let mut xm = vec![0.0f64; d];
    let mut ym = vec![0.0f64; n_y];
    let mut wsum = 0.0f64;
    for i in 0..n {
        let w = sw.map_or(1.0, |v| v[i]);
        wsum += w;
        let row = &x[i * d..i * d + d];
        for (j, m) in xm.iter_mut().enumerate() {
            *m += w * row[j].wide();
        }
        for (t, m) in ym.iter_mut().enumerate() {
            *m += w * y[i * n_y + t].wide();
        }
    }
    for m in xm.iter_mut() {
        *m /= wsum;
    }
    for m in ym.iter_mut() {
        *m /= wsum;
    }
    (xm, ym)
}

/// Worker count for the `"cov"` streaming sweep: enough MACs each to pay for
/// the spawn, never more than the machine offers, never more than there are row
/// blocks.
fn stream_units(n: usize, d: usize, n_alphas: usize, n_y: usize) -> usize {
    let per_row = d.saturating_mul(d + n_alphas.saturating_mul(n_y + 2)).max(1);
    units_for(n.saturating_mul(per_row), n.div_ceil(ROW_BLOCK))
}

/// Worker count for a pass whose MAC count the caller already knows.
///
/// The `"gram"` route's three passes have three DIFFERENT cost formulas
/// (`n²·d`, `n·n_alphas·n·(2+n_y)`, `n_alphas·d·n·n_y`), and sharing one
/// estimate across them is not a rounding error: reusing the streaming formula
/// for the back-projection asked for 16 workers to divide 200 000 MACs at
/// `n = 100`, which measured SLOWER than not threading at all. Each caller
/// passes its own count.
fn units_for(macs: usize, max_splits: usize) -> usize {
    (macs.max(1) / MACS_PER_UNIT)
        .clamp(1, cpu_launch_units().max(1) as usize)
        .min(max_splits.max(1))
}
