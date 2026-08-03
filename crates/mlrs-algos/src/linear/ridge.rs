//! `Ridge` (LINEAR-02) — L2-penalized least squares with the FULL
//! `sklearn.linear_model.Ridge` parameter surface.
//!
//! ```text
//! Ridge(alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None,
//!       tol=1e-4, solver='auto', positive=False, random_state=None)
//!       .fit(X, y, sample_weight=None)
//! ```
//!
//! Every one of those parameters is implemented here, plus the fitted
//! `coef_` / `intercept_` / `n_iter_` / `solver_` attributes sklearn exposes.
//!
//! ## The solver matrix (D-02)
//! sklearn's eight `solver` values split into direct factorizations (which map
//! onto validated device primitives) and iterative methods (which live in
//! [`crate::linear::ridge_solvers`] — see that module for why they run on the
//! host in `f64`, and for what "matches sklearn" means for an iterative solver):
//!
//! | `solver` | path | `n_iter_` |
//! |---|---|---|
//! | `auto` | resolves to `lbfgs` when `positive`, else `cholesky` (sklearn's `resolve_solver_for_numpy` for dense `X`) | — |
//! | `cholesky` | DEVICE: [`column_means`] + [`gram_xty_centered`] (centering FUSED) + [`cholesky_solve_reg`] (`α` on the diagonal IN-KERNEL) + [`ridge_intercept_device`] — no host round-trip at all; a fully HOST arm on cpu / small shapes | `None` |
//! | `svd` | DEVICE: [`svd`] then `coef = V·diag(σ/(σ²+α))·Uᵀy`; the Gram+[`eig`] form above the Jacobi caps | `None` |
//! | `sparse_cg` | HOST CG on the device-formed Gram (sklearn's own `Xᵀ(X·x) + αx` operator) | `None` |
//! | `lsqr` | HOST Paige–Saunders LSQR with `damp = √α` | `Some(itn)` |
//! | `sag` / `saga` | HOST stochastic average gradient, sklearn's `get_auto_step_size` | `Some(epochs)` |
//! | `lbfgs` | DEVICE: [`column_means`] + [`gram_xty_centered`] (centering FUSED into the Gram pass) + [`ridge_nnls`] — projected coordinate descent on the Gram, whole sweep loop in one cube (the `positive=True` arm); a fully HOST arm on cpu / small shapes / over-cap `d` | `None` |
//!
//! The `n_iter_` column is sklearn's, not an mlrs choice: `_ridge_regression`
//! initializes `n_iter = None` and only `_solve_lsqr` and the `sag`/`saga` arm
//! ever assign it, so `sparse_cg`/`cholesky`/`svd`/`lbfgs` genuinely report
//! `None` there.
//!
//! `auto` → `cholesky` keeps the pre-existing default path BYTE-IDENTICAL: an
//! unparameterized `Ridge::new()` still runs exactly the device Cholesky solve
//! described below, with no host arm and no extra readback.
//!
//! ## Cholesky, the default path (deliberately NOT SVD — that is
//! ## LinearRegression, D-02)
//! Ridge solves the regularized normal equations `(XᵀX + αI)·coef = Xᵀy`
//! via the validated Phase-4 [`cholesky_solve`] primitive (`A = L·Lᵀ`, then
//! forward and back substitution, all in-kernel — 04-02). It does NOT use the
//! SVD pseudo-inverse path by default (that is the LinearRegression
//! anti-pattern; the two DEFAULT solvers MUST NOT be unified — RESEARCH
//! Anti-Patterns / D-02). `solver='svd'` is an explicit, opt-in sklearn choice,
//! which is a different thing from silently unifying the defaults.
//!
//! ## Singular-Gram fallback (sklearn-faithful)
//! sklearn's `_ridge_regression` wraps `_solve_cholesky` in
//! `try/except LinAlgError` and RETRIES with the SVD solver, reporting the
//! fallback through `solver_`. This does the same: a non-SPD pivot from the
//! Cholesky primitive ([`PrimError::NotPositiveDefinite`], which a tiny α over a
//! collinear `X` can drive) re-solves via the SVD arm and sets `solver_` to
//! `"svd"`. NaN coefficients are still never emitted.
//!
//! ## Raw Gram, NOT scaled covariance (RESEARCH Open Q1)
//! The normal matrix is the **raw** Gram `XᵀX` formed by the row-blocked
//! [`gram_xty`] prim over the centered design — NOT `prims::covariance`,
//! which centers AND scales by `1/(n−ddof)`. sklearn's `_solve_cholesky` adds
//! `alpha` to the raw `XᵀX` diagonal directly (no `n_samples` scaling), so the
//! raw Gram is the sklearn-faithful normal matrix (verified against the
//! committed fixture: `Xc·Xc + αI` reproduces sklearn's `coef_` exactly).
//!
//! ## Perf: device-resident centering + row-blocked Gram (LINEAR-02, shared
//! ## with LinearRegression's `fit_gram_eig`)
//! Centering and Gram/Xty formation both run entirely on-device via
//! [`center_columns`] and [`gram_xty`] — the SAME primitives that fixed
//! `LinearRegression`'s large-`n_samples` path (`linear_regression.rs` module
//! docs): `center_columns` avoids an `O(n·d)` host round-trip of the full
//! design matrix (the original Ridge implementation shipped X/y to host,
//! recentered there, and re-uploaded — a PCIe-bound cost that scales with
//! `n_samples` and dominates at any realistic dataset size), and `gram_xty`
//! avoids the skinny-output/huge-K `gemm` pathology (`d×d` output over a
//! `n_samples`-sized reduction starves the GPU of independent output tiles —
//! see `mlrs_kernels::gram` module docs) by accumulating row-blocked partials
//! in shared memory instead. Ridge has no `LinearRegression`-style feature
//! cap: `gram_xty` itself falls back to the original two-`gemm` formation
//! whenever `d² > 4096`, so arbitrarily wide `X` stays correct, just without
//! the shared-memory speedup.
//!
//! A `sample_weight` (and only a `sample_weight`) forces a host preprocessing
//! pass, because the weighted column means and the `√w` row rescale are not
//! expressible through `center_columns`. The unweighted path — the default —
//! never pays it.
//!
//! ## alpha on the diagonal only; intercept never penalized (D-05)
//! `alpha` is added to the Gram DIAGONAL only (`A[i·n+i] += alpha`). The
//! intercept is recovered AFTER the solve via center-then-solve
//! (`intercept_ = ȳ − x̄·coef_`) and is therefore NEVER part of the penalized
//! system — sklearn-exact (RESEARCH Pitfall 5; α applies only to `coef_`).
//! This holds for EVERY solver, including `positive=True`: sklearn constrains
//! the CENTERED problem's coefficients and then recovers the (unconstrained,
//! possibly negative) intercept, so a non-negative fit can still have a
//! negative `intercept_`.
//!
//! ## `sample_weight` (sklearn's two-regime handling, reproduced)
//! `_preprocess_data` centers with WEIGHTED means `x̄ = Σwᵢxᵢ / Σwᵢ`, and then
//! `_ridge_regression` rescales the rows by `√wᵢ` — but ONLY for the non-SAG
//! solvers (`if solver not in ["sag", "saga"]: X, y = _rescale_data(...)`),
//! because the SAG family consumes the per-sample weight directly in its
//! stochastic gradient. Both regimes are implemented; the split is asserted by
//! the oracle tests, which compare every solver against sklearn under the same
//! weights.
//!
//! ## `copy_X` is accepted and is a genuine no-op here
//! sklearn's `copy_X` exists because `_preprocess_data` can center `X` IN
//! PLACE. mlrs never writes into the caller's buffer: the device path centers
//! into a fresh pooled allocation, and the host preprocessing path builds a new
//! `Vec`. The parameter is therefore stored for API/`get_params` parity and
//! documented as having no observable effect — not silently dropped.
//!
//! ## Gram threaded through the Cholesky factor (D-11 gate 2)
//! The Gram buffer `(XᵀX + αI)` is passed as the Cholesky primitive's `out`
//! working buffer, so the factor reuses it in place — no parallel `n²`
//! allocation (the memory gate, 04-05 Task 2, asserts this).
//!
//! ## Device residency (D-03)
//! Fitted `coef_` (length n) and `intercept_` (length 1) are stored as
//! device-resident [`DeviceArray`]s; `predict` runs the `X_test · coef_`
//! GEMM on-device and broadcasts the intercept, materializing to the host only
//! at a Rust accessor / oracle-comparison boundary. This is true whichever
//! solver produced them.
//!
//! ## What crosses the PCIe bus, per solver (the cuda/rocm/wgpu contract)
//! Every solver is correct on every backend, but they do NOT cost the same on a
//! discrete GPU, and the split is deliberate:
//!
//! | solver | device work | host read-back |
//! |---|---|---|
//! | `cholesky` | centering, Gram, factorization, solve, intercept — ALL on device | NOTHING (the `α` diagonal write and the intercept dot both moved into kernels) |
//! | `svd` | centering, SVD (or Gram+eig), both GEMMs | the length-`k` spectrum |
//! | `lbfgs` | centering, Gram/`Xᵀy`, AND the whole projected-CD solve — one cube, sweep loop in-kernel | NOTHING (host twin on cpu / `d > 256`, which reads `d² + d`) |
//! | `sparse_cg` | centering, Gram/`Xᵀy` — the whole `O(n·d)` reduction | `d² + d` floats, INDEPENDENT of `n_samples` |
//! | `lsqr`, `sag`/`saga` | centering | the `n×d` design, ONCE |
//!
//! `lsqr` and `sag`/`saga` are the only arms that ship the design matrix back,
//! and they do it once rather than per iteration. That is the point: LSQR needs
//! `X·v` / `Xᵀ·u` every iteration and SAG needs ONE ROW per step, so a
//! device-resident form would be one-to-two kernel launches per iteration over a
//! `d`-element operand — the exact per-iteration-launch pathology that made
//! `sgd_solve`, HDBSCAN's core-distance scan, and UMAP's per-epoch layout
//! host arms in this codebase. At the `~50 µs` launch overhead measured here,
//! thousands of SAG epochs would cost more in launches alone than the entire
//! host solve. A caller who wants the fully device-resident path on a GPU should
//! use the default `cholesky` (or `svd`), which is exactly what `auto` picks.
//!
//! `sample_weight` adds ONE host round-trip of the design (the weighted column
//! means and the `√w` row rescale have no `center_columns` equivalent). The
//! unweighted path — the default — never pays it.
//!
//! ## Measured: `Ridge()` on a Colab T4 (RIDGE-DEFAULT-CUDA, f32, min-of-9,
//! ## upload INSIDE the timer — `results/ridge_default_t4_dd37f93.log`)
//!
//! Whole-fit, device arm against mlrs's own host arm forced on the same VM (a
//! 2-vCPU Xeon @2GHz — see the caveat below):
//!
//! | shape | host arm | device arm | |
//! |---|---|---|---|
//! | 1 000 × 8 | 0.078 ms | 0.400 ms | 0.20× |
//! | 10 000 × 16 | 1.10 ms | 0.96 ms | 1.15× |
//! | 10 000 × 64 | 6.37 ms | 4.43 ms | 1.44× |
//! | 100 000 × 16 | 7.53 ms | 6.01 ms | 1.25× |
//! | 100 000 × 64 | 58.9 ms | 30.7 ms | 1.92× |
//! | 500 000 × 16 | 35.8 ms | 38.9 ms | 0.92× |
//! | 100 000 × 128 | 206 ms | 59.2 ms | **3.49×** |
//! | 100 000 × 256 | 786 ms | 313 ms | **2.51×** |
//!
//! Two things that table does NOT say. First, the two rungs where the device
//! arm loses are the two `host_fit_applicable` already routes to the host
//! (`1 000 × 8`) or calls a wash (`500 000 × 16`, 8%) — a Python caller gets the
//! better arm at every rung but that one. Second, the host column is the SAME
//! CODE the cpu backend runs, on a 2-thread VM; on the 16-thread dev box it is
//! roughly 4× faster (`100 000 × 256` in 77.9 ms, not 786), so against THAT cpu
//! the T4 does not win at any rung on this ladder. The GPU's advantage is
//! `n·d²/2` arithmetic over an `n·d` transfer, so it grows with `d` and the
//! crossover against a 16-thread host sits above `d = 256`.
//!
//! The upload is why. Drained laps at `100 000 × 256`: 308 ms of upload against
//! 4.8 ms of means + 13.3 ms of Gram+solve. The compute is 5.8% of the fit and
//! the transfer is 96% — the same wall the `positive=True` campaign hit
//! (`results/ridge_positive_t4_*.log`: 0.33–0.92 GB/s on a link that should do
//! 6–12, and getting WORSE as the operand grows, which is the signature of a
//! per-call allocation cost rather than a slow link). Nothing here changes it,
//! and on this hardware it is the only lever left.
//!
//! ## Both normal-equations arms have a second, fully-HOST route
//! [`Ridge::fit_from_host_slice`] runs the whole fit — means, centering, Gram,
//! `Xᵀy`, solve — from host memory, uploading only the fitted `coef_` and
//! `intercept_`. [`Ridge::host_fit_applicable`] picks it for `cholesky` and
//! `lbfgs` on the cpu backend (where the device composition is pathological:
//! `center_columns` falls back to the per-column-round-trip `column_reduce`
//! there, measured at 59.6 s of a 60.1 s `1 000 × 8` fit) and, on ANY backend,
//! below the fixed dispatch-cost floor. Everything else keeps the device route
//! above.
//!
//! Tests live in `crates/mlrs-algos/tests/ridge_test.rs` and
//! `crates/mlrs-algos/tests/ridge_params_test.rs` (AGENTS.md §2), never an
//! in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::center::center_columns;
use mlrs_backend::prims::cholesky::cholesky_solve_reg;
use mlrs_backend::prims::eig::eig;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::gram::{
    column_means, fused_centering_available, gram_xty, gram_xty_centered,
};
use mlrs_backend::prims::gram_host::{centered_gram_xty, gram_host_applicable};
use mlrs_backend::prims::linear_predict::{linear_predict, HostMirror, HostPrediction};
use mlrs_backend::prims::nnls::{device_nnls_applicable, ridge_intercept_device, ridge_nnls};
use mlrs_backend::prims::svd::svd;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::linear::elastic_net::predict_linear_from_host;
use crate::linear::ridge_solvers;
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};

/// sklearn's `Ridge` default `tol` (`1e-4`). Only the iterative solvers read it.
const RIDGE_DEFAULT_TOL: f64 = 1e-4;

/// Row cap of the one-sided Jacobi SVD kernel (`mlrs_kernels::MAX_ROWS`).
/// Above it the `svd` solver takes the Gram+eig route (see [`solve_svd`]).
const SVD_JACOBI_MAX_ROWS: usize = 256;

/// Column cap of the same kernel (`mlrs_kernels::MAX_COLS`).
const SVD_JACOBI_MAX_COLS: usize = 64;

/// Order cap of the Jacobi eig kernel (`mlrs_kernels::MAX_DIM`) — the bound on
/// the Gram+eig fallback the `svd` solver uses above the Jacobi SVD caps.
const GRAM_EIG_MAX_FEATURES: usize = 64;

/// sklearn's `_solve_svd` singular-value cutoff (`idx = s > 1e-15`, "same
/// default value as scipy.linalg.pinv"). Below it a direction contributes
/// nothing to `coef`.
const SVD_ZERO_SIGMA: f64 = 1e-15;

/// Seed used for `sag`/`saga` when `random_state` is `None`. sklearn draws from
/// the global numpy RNG there, which makes the SAMPLING ORDER — but not the
/// converged fixed point — irreproducible. mlrs pins a constant instead so an
/// unseeded fit is deterministic; the resulting coefficients agree with
/// sklearn's within the oracle tolerance either way (both converge to the same
/// unique minimizer).
const SAG_DEFAULT_SEED: u64 = 0;

/// The `solver` selector — sklearn's eight `StrOptions` values, one-for-one.
///
/// A non-scalar hyperparameter, so the builder takes this enum directly (the
/// `KernelKind` precedent) and the PyO3 boundary parses the sklearn STRING into
/// it via [`TryFrom<&str>`], surfacing an unknown name as
/// [`BuildError::UnknownSolver`] (D-09).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RidgeSolver {
    /// Pick a solver from the other hyperparameters (sklearn's
    /// `resolve_solver_for_numpy`): `lbfgs` when `positive`, else `cholesky`
    /// for the dense `X` mlrs takes.
    Auto,
    /// `coef = V·diag(σ/(σ²+α))·Uᵀ·y` from the thin SVD.
    Svd,
    /// Cholesky factorization of `(XᵀX + αI)` — the default.
    Cholesky,
    /// Paige–Saunders LSQR on the `√α`-damped least-squares system.
    Lsqr,
    /// Conjugate gradients on the `(XᵀX + αI)` operator.
    SparseCg,
    /// Stochastic average gradient.
    Sag,
    /// SAGA — the unbiased-correction SAG variant.
    Saga,
    /// The bound-constrained arm; valid only with `positive = true`.
    Lbfgs,
}

impl RidgeSolver {
    /// The sklearn `solver` string, for `solver_` and for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            RidgeSolver::Auto => "auto",
            RidgeSolver::Svd => "svd",
            RidgeSolver::Cholesky => "cholesky",
            RidgeSolver::Lsqr => "lsqr",
            RidgeSolver::SparseCg => "sparse_cg",
            RidgeSolver::Sag => "sag",
            RidgeSolver::Saga => "saga",
            RidgeSolver::Lbfgs => "lbfgs",
        }
    }

    /// Resolve `auto` exactly as sklearn's `resolve_solver_for_numpy` does for a
    /// DENSE `X` with `return_intercept=False`: `positive` ⇒ `lbfgs`, else
    /// `cholesky`. Every other value passes through unchanged.
    pub fn resolve(self, positive: bool) -> RidgeSolver {
        match self {
            RidgeSolver::Auto if positive => RidgeSolver::Lbfgs,
            RidgeSolver::Auto => RidgeSolver::Cholesky,
            other => other,
        }
    }

    /// Does this solver consume `sample_weight` DIRECTLY (rather than through
    /// the `√w` row rescale)? sklearn: `if solver not in ["sag", "saga"]`.
    fn takes_sample_weight_directly(self) -> bool {
        matches!(self, RidgeSolver::Sag | RidgeSolver::Saga)
    }

    /// Does this solver read ONLY the normal equations (`XᵀX`, `Xᵀy`) and never
    /// the design matrix itself?
    ///
    /// This is the precondition for FUSED centering. Every solver needs the
    /// design centered, but a Gram-only solver never needs the centered design
    /// to EXIST: the subtraction can happen inside the accumulation kernel, so
    /// the `n × d` centered copy — its allocation, its write and its re-read —
    /// disappears. On a `100 000 × 256` wgpu fit that copy measured 151 ms of
    /// 528. `lsqr` and `sag`/`saga` genuinely consume rows and cannot fuse;
    /// `svd` consumes `X` directly below the Jacobi caps and so cannot either.
    ///
    /// `cholesky` — the DEFAULT, `positive = false` arm — can, and this is where
    /// that route came from: it was written for `lbfgs` and left keyed on that
    /// one solver, so `Ridge()` with no arguments kept paying for a centered
    /// design it never looked at.
    fn consumes_gram_only(self) -> bool {
        matches!(self, RidgeSolver::Cholesky | RidgeSolver::Lbfgs)
    }
}

impl TryFrom<&str> for RidgeSolver {
    type Error = BuildError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "auto" => Ok(RidgeSolver::Auto),
            "svd" => Ok(RidgeSolver::Svd),
            "cholesky" => Ok(RidgeSolver::Cholesky),
            "lsqr" => Ok(RidgeSolver::Lsqr),
            "sparse_cg" => Ok(RidgeSolver::SparseCg),
            "sag" => Ok(RidgeSolver::Sag),
            "saga" => Ok(RidgeSolver::Saga),
            "lbfgs" => Ok(RidgeSolver::Lbfgs),
            other => Err(BuildError::UnknownSolver {
                value: other.to_string(),
            }),
        }
    }
}

/// L2-penalized least squares (LINEAR-02) with sklearn's full parameter set.
///
/// Construct with the zero-arg [`Ridge::new`] (sklearn defaults) or
/// [`Ridge::builder`], then the consuming [`Fit::fit`] (returns the
/// `Fitted`-tagged sibling) and [`Predict::predict`]. Fitted
/// `coef_`/`intercept_` are device-resident (D-03); the host accessors
/// [`coef`](Ridge::coef) / [`intercept`](Ridge::intercept) materialize them on
/// demand and exist ONLY on `Ridge<F, Fitted>` (the compile-time typestate
/// replaces the old runtime `NotFitted` guard, D-03).
pub struct Ridge<F, S = Unfit> {
    /// L2 penalty strength (`alpha ≥ 0`; `alpha = 0` degenerates to OLS).
    /// Added to the Gram diagonal only — never to the intercept (D-05).
    alpha: F,
    /// Whether to center `X`/`y` and recover a bias term (D-05).
    fit_intercept: bool,
    /// sklearn's `copy_X`. Stored for parity; mlrs never writes into the
    /// caller's buffer, so this has no observable effect (module docs).
    copy_x: bool,
    /// Iteration cap for the iterative solvers. `None` takes each solver's
    /// scipy/sklearn default, exactly as sklearn's `max_iter=None` does.
    max_iter: Option<usize>,
    /// Stopping tolerance for the iterative solvers (sklearn default `1e-4`).
    tol: f64,
    /// Which solver to use (`Auto` resolves at `fit`).
    solver: RidgeSolver,
    /// Constrain `coef_ >= 0`. Requires the `lbfgs` solver (or `auto`, which
    /// resolves to it) — the same constraint sklearn enforces.
    positive: bool,
    /// Seed for the `sag`/`saga` sampling order. `None` uses
    /// [`SAG_DEFAULT_SEED`].
    random_state: Option<u64>,
    /// Fitted coefficients (length `n_features`), device-resident, `None` until
    /// `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (length 1), device-resident, `None` until `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
    /// sklearn's `n_iter_`: the iterative solver's iteration count, and `None`
    /// for every solver sklearn leaves unset (see the module-doc table).
    n_iter_: Option<usize>,
    /// sklearn's `solver_`: the solver that ACTUALLY ran, after `auto`
    /// resolution and after any singular-Gram fallback. `None` until `fit`.
    solver_: Option<RidgeSolver>,
    /// Memoized host copy of `(coef_, intercept_)` for the host-ingress
    /// `predict` path (IN-05 `OnceLock` mirror idiom). Empty until the first
    /// `predict_from_host` on the cpu backend, and never filled at all on the
    /// device backends — see
    /// [`HostMirror`](mlrs_backend::prims::linear_predict::HostMirror) for why a
    /// 64-byte read-back is worth caching. Fresh on every `fit`.
    predict_mirror: HostMirror<F>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> Ridge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct a `Ridge` with sklearn's `Ridge` defaults (`alpha = 1.0`,
    /// `fit_intercept = true`, `copy_X = true`, `max_iter = None`,
    /// `tol = 1e-4`, `solver = 'auto'`, `positive = false`,
    /// `random_state = None`) directly in the `Unfit` state. This is the SINGLE
    /// source of truth for the default hyperparameters (D-08): the builder
    /// `Default` re-derives from here via [`Ridge::into_builder`], rather than
    /// re-listing the literals. Defaults are trusted valid, so this bypasses
    /// [`RidgeBuilder::build`]'s validation.
    pub fn new() -> Self {
        Self {
            alpha: F::from_int(1),
            fit_intercept: true,
            copy_x: true,
            max_iter: None,
            tol: RIDGE_DEFAULT_TOL,
            solver: RidgeSolver::Auto,
            positive: false,
            random_state: None,
            coef_: None,
            intercept_: None,
            n_iter_: None,
            solver_: None,
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        }
    }

    /// Start building a `Ridge` from sklearn's defaults (D-08 single source).
    pub fn builder() -> RidgeBuilder {
        RidgeBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`RidgeBuilder::default`] to re-derive the
    /// defaults from [`Ridge::new`] (D-08), and available to callers who want to
    /// tweak a constructed estimator before fitting.
    pub fn into_builder(self) -> RidgeBuilder {
        RidgeBuilder {
            alpha: host_to_f64(self.alpha),
            fit_intercept: self.fit_intercept,
            copy_x: self.copy_x,
            max_iter: self.max_iter,
            tol: self.tol,
            solver: self.solver,
            positive: self.positive,
            random_state: self.random_state,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `coef_`/`intercept_`/`n_iter_`/`solver_` fields are excluded — all are
    /// `None` in any `Unfit` value). Used by the defaults-equality test
    /// (BLDR-01): `Ridge::new().hyperparams_eq(&Ridge::builder().build()?)`.
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        host_to_f64(self.alpha) == host_to_f64(other.alpha)
            && self.fit_intercept == other.fit_intercept
            && self.copy_x == other.copy_x
            && self.max_iter == other.max_iter
            && self.tol == other.tol
            && self.solver == other.solver
            && self.positive == other.positive
            && self.random_state == other.random_state
    }

    /// `fit` with sklearn's `sample_weight` — the full-surface entry point.
    ///
    /// [`Fit::fit`] is this with `sample_weight = None` (the `Fit` trait carries
    /// no weight slot, so the weighted form needs its own name rather than a
    /// widened trait every other estimator would have to absorb).
    ///
    /// `sample_weight` is a length-`n_samples` host slice of NON-NEGATIVE finite
    /// weights. It changes the fit in two places, both sklearn-faithful:
    /// the column means become weighted (`x̄ = Σwᵢxᵢ / Σwᵢ`), and the rows are
    /// rescaled by `√wᵢ` for every solver EXCEPT `sag`/`saga`, which take the
    /// weights directly in their stochastic gradient.
    pub fn fit_with_sample_weight(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<Ridge<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-04-05-03 / ASVS V5: data-DEPENDENT geometry guard BEFORE any
        //     prim launch (the data-INDEPENDENT hyperparameter checks — `alpha`,
        //     `tol`, `max_iter`, and the `positive`/`solver` compatibility pair —
        //     were validated at build(), Pitfall 7). ---
        let alpha64 = host_to_f64(self.alpha);
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "ridge",
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

        // --- sample_weight validation (T-04-05-03): a wrong-length weight
        //     vector is a geometry error; a negative or non-finite weight would
        //     make `√w` NaN in the rescale and silently poison every downstream
        //     reduction, so it is rejected as a typed error instead. ---
        let sw64: Option<Vec<f64>> = validate_sample_weight::<F>("ridge", sample_weight, n_samples)?;

        let resolved = self.solver.resolve(self.positive);

        // RIDGE_PROFILE=1: per-phase wall-clock attribution (the LR_PROFILE
        // precedent in `linear_regression.rs`'s `fit_gram_eig` — attribution
        // only, since kernel launches are async and a lap only completes at the
        // next readback that drains the queue; a tiny forced readback after
        // `gram_xty`/`cholesky_solve` pins each phase's lap to ITS OWN kernels
        // rather than bleeding into the next phase's).
        let profile = std::env::var("RIDGE_PROFILE").is_ok();
        // A lap only means something if it ENDS with a real blocking read-back:
        // `client.sync()` returns a future, so an undrained lap measures enqueue
        // time and silently bleeds into whatever readback comes next
        // (RIDGE-POS-PERF). This one-element array is read after each phase to
        // pin every lap to its own kernels. Allocated only when profiling, and
        // the drains forbid cross-phase overlap, so a profiled TOTAL runs a
        // little high — the split is what this is for, not the total.
        let probe: Option<DeviceArray<ActiveRuntime, F>> = if profile {
            Some(DeviceArray::from_host(pool, &[f64_to_host::<F>(1.0)]))
        } else {
            None
        };
        let lap0 = std::time::Instant::now();

        // --- 1. Centering + (for the non-SAG solvers) the `√w` row rescale.
        //        With NO sample_weight this is the original DEVICE-resident
        //        `center_columns` composition, unchanged: no host round-trip of
        //        the full n×d design. See `preprocess` for the weighted arm. ---
        //
        //        The `positive` device arm takes the FUSED route instead: only
        //        the column means are formed here, and the subtraction happens
        //        inside `gram_xty_centered`'s accumulation kernel. That drops
        //        the `n×d` centered allocation, its write and its re-read — the
        //        second largest device cost of the fit after the design upload
        //        (measured 9 ms of a 25 ms `n=100 000, d=64` wgpu fit, 151 ms of
        //        528 ms at `d=256`). Only this arm can do it: every other solver
        //        consumes the centered DESIGN, not its Gram.
        let rescale = sw64.is_some() && !resolved.takes_sample_weight_directly();
        let fused_center = resolved.consumes_gram_only()
            && sw64.is_none()
            && self.fit_intercept
            && fused_centering_available::<F>(n_features)
            && (resolved != RidgeSolver::Lbfgs || device_nnls_applicable::<F>(n_features));
        let (mut x_mean, mut y_mean, x_owned, y_owned, dev_means) = if fused_center {
            let (xm, ym) = column_means::<F>(pool, x, y, n_samples, n_features)?;
            (Vec::new(), 0.0f64, None, None, Some((xm, ym)))
        } else {
            let (xm, ym, xo, yo) = preprocess::<F>(
                pool,
                x,
                y,
                n_samples,
                n_features,
                self.fit_intercept,
                sw64.as_deref(),
                rescale,
            )?;
            (xm, ym, xo, yo, None)
        };
        let x_ref = x_owned.as_ref().unwrap_or(x);
        let y_ref = y_owned.as_ref().unwrap_or(y);
        drain_profile_probe::<F>(pool, &probe);
        let t_center = if profile { lap0.elapsed().as_secs_f64() } else { 0.0 };

        // --- 2. Solve. Each arm returns the device-resident `coef_`, the
        //        `n_iter_` sklearn would report for it, and the solver actually
        //        used (which differs from `resolved` only on the singular-Gram
        //        Cholesky→SVD fallback). ---
        let lap1 = std::time::Instant::now();
        let (coef, n_iter, solver_used) = match resolved {
            RidgeSolver::Auto => unreachable!("resolve() never returns Auto"),
            RidgeSolver::Cholesky => {
                let means = dev_means.as_ref().map(|(xm, ym)| (xm, ym));
                match solve_cholesky::<F>(
                    pool, x_ref, y_ref, means, n_samples, n_features, alpha64,
                ) {
                    Ok(coef) => (coef, None, RidgeSolver::Cholesky),
                    // sklearn's `except LinAlgError: solver = "svd"` retry.
                    Err(AlgoError::Prim(PrimError::NotPositiveDefinite { .. })) => {
                        // The SVD arm consumes the centered DESIGN, which the
                        // fused route deliberately never materializes. Build it
                        // here rather than penalizing every successful fit for a
                        // branch that only runs after a factorization has
                        // already failed. `center_columns` recomputes the same
                        // means the fused pass has on the device, so the retry
                        // sees exactly the operands the unfused route would have
                        // handed it.
                        let staged = if dev_means.is_some() {
                            let (xc, xm) =
                                center_columns::<F>(pool, x, (n_samples, n_features))?;
                            let (yc, ym) = center_columns::<F>(pool, y, (n_samples, 1))?;
                            xm.release_into(pool);
                            ym.release_into(pool);
                            Some((xc, yc))
                        } else {
                            None
                        };
                        let (sx, sy) = match &staged {
                            Some((xc, yc)) => (xc, yc),
                            None => (x_ref, y_ref),
                        };
                        let coef =
                            solve_svd::<F>(pool, sx, sy, n_samples, n_features, alpha64);
                        if let Some((xc, yc)) = staged {
                            xc.release_into(pool);
                            yc.release_into(pool);
                        }
                        (coef?, None, RidgeSolver::Svd)
                    }
                    Err(e) => return Err(e),
                }
            }
            RidgeSolver::Svd => (
                solve_svd::<F>(pool, x_ref, y_ref, n_samples, n_features, alpha64)?,
                None,
                RidgeSolver::Svd,
            ),
            RidgeSolver::SparseCg => {
                let (gram, xty) = host_gram::<F>(pool, x_ref, y_ref, n_samples, n_features)?;
                let coef = ridge_solvers::sparse_cg(
                    &gram,
                    &xty,
                    n_features,
                    alpha64,
                    self.tol,
                    self.max_iter,
                );
                (upload_coef::<F>(pool, &coef), None, RidgeSolver::SparseCg)
            }
            RidgeSolver::Lbfgs => {
                // sklearn leaves `n_iter_` at None for the lbfgs arm.
                let coef = solve_nonnegative::<F>(
                    pool,
                    x_ref,
                    y_ref,
                    dev_means.as_ref().map(|(xm, ym)| (xm, ym)),
                    n_samples,
                    n_features,
                    alpha64,
                    self.tol,
                    self.max_iter,
                )?;
                (coef, None, RidgeSolver::Lbfgs)
            }
            RidgeSolver::Lsqr => {
                let (xh, yh) = host_design::<F>(pool, x_ref, y_ref);
                let (coef, itn) = ridge_solvers::lsqr(
                    &xh,
                    &yh,
                    n_samples,
                    n_features,
                    alpha64,
                    self.tol,
                    self.max_iter,
                );
                (upload_coef::<F>(pool, &coef), Some(itn), RidgeSolver::Lsqr)
            }
            RidgeSolver::Sag | RidgeSolver::Saga => {
                let (xh, yh) = host_design::<F>(pool, x_ref, y_ref);
                let (coef, epochs) = ridge_solvers::sag(
                    &xh,
                    &yh,
                    sw64.as_deref(),
                    n_samples,
                    n_features,
                    alpha64,
                    self.tol,
                    self.max_iter,
                    self.random_state.unwrap_or(SAG_DEFAULT_SEED),
                    resolved == RidgeSolver::Saga,
                );
                (upload_coef::<F>(pool, &coef), Some(epochs), resolved)
            }
        };
        drain_profile_probe::<F>(pool, &probe);
        let t_solve = if profile { lap1.elapsed().as_secs_f64() } else { 0.0 };
        let lap2 = std::time::Instant::now();

        if let Some(xc) = x_owned {
            xc.release_into(pool);
        }
        if let Some(yc) = y_owned {
            yc.release_into(pool);
        }

        // --- 3. intercept_ = ȳ − x̄·coef_ when fit_intercept, else 0 (D-05). α
        //        is NOT applied here — the intercept is unpenalized — and NEITHER
        //        is the `positive` bound (sklearn constrains only `coef_`). ---
        //
        // TWO ARMS. The fused (`positive`) route leaves `x̄`/`ȳ` on the device
        // and the solve leaves `coef` there, so every operand of this dot is
        // already resident: `ridge_intercept_device` finishes the fit without a
        // single host round-trip. The host arm reads `x̄`, `ȳ` and `coef` back —
        // three BLOCKING read-backs — to do the same dot in `f64` and upload one
        // scalar.
        //
        // The device arm is the default because it is what "device-resident
        // fit" should mean, but the host arm is NOT vestigial: it is the only
        // one available to the non-fused solvers (whose means were computed on
        // the host by `preprocess`), and it accumulates in `f64` where the
        // kernel accumulates in `F`. `MLRS_RIDGE_HOST_INTERCEPT=1` forces it
        // anywhere, which is how the two are A/B'd for both speed and drift.
        let device_intercept = self.fit_intercept
            && dev_means.is_some()
            && !mlrs_backend::abflag::is_on("MLRS_RIDGE_HOST_INTERCEPT");

        let intercept_dev: DeviceArray<ActiveRuntime, F> = if device_intercept {
            let (xm, ym) = dev_means.expect("device_intercept implies dev_means");
            let out = ridge_intercept_device::<F>(pool, &xm, &ym, &coef, n_features)?;
            xm.release_into(pool);
            ym.release_into(pool);
            out
        } else {
            // The fused arm's means still live on the device here; `d + 1`
            // floats cross once, where this arm's dot needs them anyway.
            if let Some((xm, ym)) = dev_means {
                x_mean = xm.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
                y_mean = host_to_f64(ym.to_host(pool)[0]);
                xm.release_into(pool);
                ym.release_into(pool);
            }
            let coef_host = coef.to_host(pool);
            let intercept = if self.fit_intercept {
                let mut dot = 0.0f64;
                for c in 0..n_features {
                    dot += x_mean[c] * host_to_f64(coef_host[c]);
                }
                y_mean - dot
            } else {
                0.0
            };
            DeviceArray::from_host(pool, &[f64_to_host::<F>(intercept)])
        };

        if profile {
            // `tail` is the intercept recovery: three blocking read-backs
            // (x_mean, y_mean, coef) plus the scalar upload. It was noise when
            // the Gram cost 21 ms; it is not noise now that the Gram costs 4.
            let t_tail = lap2.elapsed().as_secs_f64();
            eprintln!(
                "RIDGE_PROFILE n={n_samples} d={n_features} solver={}: \
                 preprocess={t_center:.4}s solve={t_solve:.4}s tail={t_tail:.4}s",
                solver_used.name()
            );
        }
        if let Some(p) = probe {
            p.release_into(pool);
        }

        Ok(Ridge {
            alpha: self.alpha,
            fit_intercept: self.fit_intercept,
            copy_x: self.copy_x,
            max_iter: self.max_iter,
            tol: self.tol,
            solver: self.solver,
            positive: self.positive,
            random_state: self.random_state,
            coef_: Some(coef),
            intercept_: Some(intercept_dev),
            n_iter_: n_iter,
            solver_: Some(solver_used),
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }

    /// Does the fully-HOST fit arm ([`Ridge::fit_from_host_slice`]) apply to
    /// this configuration?
    ///
    /// `true` for the two NORMAL-EQUATIONS solvers — `cholesky` (the
    /// `positive = false` default) and `lbfgs` (the `positive = true` arm) —
    /// and only where the formation belongs on the host
    /// (`prims::gram_host::gram_host_applicable` — the cpu backend, plus the
    /// fixed-dispatch-cost floor on every backend). Both consume only `XᵀX` and
    /// `Xᵀy`, which is what makes a host arm possible at all: the solve after
    /// them is `O(d³)` on a matrix a few hundred wide. Every other solver still
    /// runs the device path.
    ///
    /// The `cholesky` half is the RIDGE-DEFAULT-CUDA addition, and it closes a
    /// live 100 000× regression rather than merely adding a fast path: on the
    /// cpu backend `Ridge()` went through `center_columns`, whose cpu arm walks
    /// the `d` columns one at a time with an upload + launch + blocking readback
    /// each. That was measured at 59.6 s of a 60.1 s `1 000 × 8` fit for the
    /// `positive = true` arm before it got a host route, and `positive = false`
    /// — the DEFAULT — was still paying it.
    ///
    /// `shape` is `(n_samples, n_features)`; the floor is a function of the
    /// problem size, so the caller must know it before deciding.
    ///
    /// Callers branch on this rather than letting `fit_from_host_slice` decide,
    /// because the two entry points take DIFFERENT operand types (host slice vs
    /// [`DeviceArray`]) and the choice therefore has to be made before ingress —
    /// which is the whole point: on the applicable arm the design is never
    /// uploaded at all.
    pub fn host_fit_applicable(&self, shape: (usize, usize)) -> bool {
        self.solver.resolve(self.positive).consumes_gram_only()
            && gram_host_applicable(shape.0, shape.1)
    }

    /// [`Fit::fit`] over HOST slices — the no-upload, no-launch ingress for the
    /// `positive = true` arm on the cpu backend.
    ///
    /// `x` is the `n × d` row-major design and `y` the length-`n` target, both
    /// borrowed from host memory (at the Python boundary, the Arrow values
    /// themselves). Nothing about the FITTED estimator differs from one produced
    /// by [`Fit::fit`] — `coef_`/`intercept_` are still device-resident, so
    /// `predict` has one path — only the route there does:
    ///
    /// | | [`Fit::fit`] on cpu | this |
    /// |---|---|---|
    /// | design upload | `n·d` | none |
    /// | column means | `d` × (upload + launch + blocking readback) | one parallel host pass |
    /// | centering | one launch writing a fresh `n·d` buffer | folded into the tile build, never materialized |
    /// | Gram / `Xᵀy` | `gram_xty` launch | one parallel host pass |
    /// | solve | on the read-back Gram | the same solver, same Gram |
    ///
    /// The solve is [`ridge_solvers::cholesky_ridge`] for `positive = false` —
    /// with [`ridge_solvers::gram_eig_ridge`] as sklearn's singular-Gram retry —
    /// and [`ridge_solvers::nonnegative_cd`] for `positive = true`. Both read
    /// only `XᵀX` / `Xᵀy`, which is the property that makes this arm possible.
    ///
    /// Measured at `1 000 × 8`, `positive=True`, f64: 60.1 s → 0.2 ms.
    ///
    /// Returns [`PrimError::UnsupportedCapability`] when
    /// [`Ridge::host_fit_applicable`] is false, so a caller that forgets to
    /// branch gets a typed error rather than a silently different answer.
    pub fn fit_from_host_slice(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        y: &[F],
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<Ridge<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        if !self.host_fit_applicable(shape) {
            return Err(AlgoError::Prim(PrimError::UnsupportedCapability {
                operand: "ridge.fit_from_host_slice",
                capability: "the host fit arm (a normal-equations solver on a host-Gram backend)",
            }));
        }
        let resolved = self.solver.resolve(self.positive);

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
        let sw64 = validate_sample_weight::<F>("ridge", sample_weight, n_samples)?;

        let profile = std::env::var("RIDGE_PROFILE").is_ok();
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
        let t_center = if profile { lap0.elapsed().as_secs_f64() } else { 0.0 };

        let lap1 = std::time::Instant::now();
        let alpha64 = host_to_f64(self.alpha);
        // Same solver split as the device arm, on the same normal equations.
        // sklearn leaves `n_iter_` at None for BOTH of these, so no sweep count
        // is kept.
        let (coef64, solver_used) = match resolved {
            RidgeSolver::Lbfgs => {
                let (w, _sweeps) = ridge_solvers::nonnegative_cd(
                    &gram,
                    &xty,
                    n_features,
                    alpha64,
                    self.tol,
                    self.max_iter,
                );
                (w, RidgeSolver::Lbfgs)
            }
            // sklearn's `except LinAlgError: solver = "svd"` retry, host-side:
            // a non-positive pivot re-solves through the eigendecomposition and
            // reports `solver_ = "svd"`, exactly as the device arm does.
            _ => match ridge_solvers::cholesky_ridge(&gram, &xty, n_features, alpha64) {
                Some(w) => (w, RidgeSolver::Cholesky),
                None => (
                    ridge_solvers::gram_eig_ridge(&gram, &xty, n_features, alpha64),
                    RidgeSolver::Svd,
                ),
            },
        };
        let t_solve = if profile { lap1.elapsed().as_secs_f64() } else { 0.0 };

        // intercept_ = ȳ − x̄·coef_ when fit_intercept, else 0 (D-05); α is not
        // applied (the intercept is unpenalized) and neither is the `positive`
        // bound (sklearn constrains only `coef_`) — the same arithmetic
        // `fit_with_sample_weight` does, on the means this pass already has.
        let intercept = if self.fit_intercept {
            let dot: f64 = x_mean
                .iter()
                .zip(coef64.iter())
                .map(|(m, c)| m * c)
                .sum();
            y_mean - dot
        } else {
            0.0
        };

        if profile {
            eprintln!(
                "RIDGE_PROFILE n={n_samples} d={n_features} solver={} (host): \
                 preprocess={t_center:.4}s solve={t_solve:.4}s",
                solver_used.name()
            );
        }

        Ok(Ridge {
            alpha: self.alpha,
            fit_intercept: self.fit_intercept,
            copy_x: self.copy_x,
            max_iter: self.max_iter,
            tol: self.tol,
            solver: self.solver,
            positive: self.positive,
            random_state: self.random_state,
            coef_: Some(upload_coef::<F>(pool, &coef64)),
            intercept_: Some(DeviceArray::from_host(
                pool,
                &[f64_to_host::<F>(intercept)],
            )),
            n_iter_: None,
            solver_: Some(solver_used),
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }
}


/// Drain the device queue for [`RIDGE_PROFILE`] lap attribution.
///
/// A no-op when profiling is off. See the call site for why `client.sync()`
/// cannot be used here.
fn drain_profile_probe<F>(
    pool: &BufferPool<ActiveRuntime>,
    probe: &Option<DeviceArray<ActiveRuntime, F>>,
) where
    F: Float + CubeElement + Pod,
{
    if let Some(p) = probe {
        let v = p.to_host(pool);
        debug_assert!(!v.is_empty());
    }
}

/// Validate an optional `sample_weight` and widen it to `f64` (T-04-05-03).
///
/// Shared by [`Ridge::fit_with_sample_weight`], [`Ridge::fit_from_host_slice`]
/// and (`pub(crate)`) `BayesianRidge`'s two ingresses, so no path can drift on
/// weight validation. A wrong-length vector is a geometry error; a negative or
/// non-finite weight would make `√w` NaN in the rescale and silently poison
/// every downstream reduction; an all-zero vector leaves nothing to fit (the
/// rescale zeroes the whole design and the penalized solve would hand back the
/// all-zero coefficient vector as though it were an answer).
pub(crate) fn validate_sample_weight<F>(
    estimator: &'static str,
    sample_weight: Option<&[F]>,
    n_samples: usize,
) -> Result<Option<Vec<f64>>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let Some(sw) = sample_weight else {
        return Ok(None);
    };
    if sw.len() != n_samples {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "sample_weight",
            rows: n_samples,
            cols: 1,
            len: sw.len(),
        }));
    }
    let sw: Vec<f64> = sw.iter().map(|&v| host_to_f64(v)).collect();
    if let Some(bad) = sw.iter().position(|v| !v.is_finite() || *v < 0.0) {
        return Err(AlgoError::InvalidSampleWeight {
            estimator,
            index: bad,
            value: sw[bad],
        });
    }
    if sw.iter().all(|&v| v == 0.0) {
        return Err(AlgoError::ZeroSampleWeightSum { estimator });
    }
    Ok(Some(sw))
}

impl<F> Default for Ridge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`Ridge`] (D-01). Scalar setters are `f64`-typed per the A5
/// convention; `build::<F>()` narrows to the target float `F`. `Default`
/// re-derives the sklearn defaults from [`Ridge::new`] (D-08 single source)
/// rather than holding literals (Pitfall 1: default-drift breaks the oracle gate
/// silently).
#[derive(Debug, Clone, Copy)]
pub struct RidgeBuilder {
    alpha: f64,
    fit_intercept: bool,
    copy_x: bool,
    max_iter: Option<usize>,
    tol: f64,
    solver: RidgeSolver,
    positive: bool,
    random_state: Option<u64>,
}

impl Default for RidgeBuilder {
    /// Re-derive the sklearn defaults from [`Ridge::new`] (D-08 single source).
    /// `f64` is pinned only to read the F-independent scalar defaults — the
    /// builder is non-generic, so the choice of `F` here is irrelevant.
    fn default() -> Self {
        Ridge::<f64, Unfit>::new().into_builder()
    }
}

impl RidgeBuilder {
    /// Set the L2 penalty strength `alpha` (A5: `f64` setter).
    pub fn alpha(mut self, v: f64) -> Self {
        self.alpha = v;
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

    /// Set the iterative solvers' iteration cap. `None` (the sklearn default)
    /// takes each solver's own scipy/sklearn default.
    pub fn max_iter(mut self, v: Option<usize>) -> Self {
        self.max_iter = v;
        self
    }

    /// Set the iterative solvers' stopping tolerance (sklearn default `1e-4`).
    pub fn tol(mut self, v: f64) -> Self {
        self.tol = v;
        self
    }

    /// Set the solver (sklearn's `solver`). Takes the [`RidgeSolver`] enum
    /// directly (non-scalar selector — the `KernelKind` precedent).
    pub fn solver(mut self, v: RidgeSolver) -> Self {
        self.solver = v;
        self
    }

    /// Constrain the coefficients to be non-negative (sklearn's `positive`).
    /// Requires `solver` to be `auto` or `lbfgs`; any other combination is
    /// rejected at [`build`](RidgeBuilder::build), as in sklearn.
    pub fn positive(mut self, v: bool) -> Self {
        self.positive = v;
        self
    }

    /// Seed the `sag`/`saga` sampling order (sklearn's `random_state`).
    pub fn random_state(mut self, v: Option<u64>) -> Self {
        self.random_state = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08; the data-DEPENDENT
    /// geometry check lives in [`Fit::fit`]):
    ///
    /// - `alpha >= 0` ([`BuildError::InvalidAlpha`]) — a negative penalty makes
    ///   `(XᵀX + αI)` indefinite and the Cholesky factorization undefined
    ///   (relocated from the old fit-body check, T-04-05-03 / Pitfall 7).
    /// - `tol >= 0` and finite ([`BuildError::InvalidTol`]) — sklearn's
    ///   `Interval(Real, 0, None, closed="left")`.
    /// - `max_iter >= 1` when given ([`BuildError::InvalidMaxIter`]) — sklearn's
    ///   `Interval(Integral, 1, None, closed="left")`.
    /// - `solver = 'lbfgs'` requires `positive = true`
    ///   ([`BuildError::LbfgsRequiresPositive`]) and `positive = true` requires
    ///   `solver ∈ {auto, lbfgs}` ([`BuildError::PositiveUnsupportedSolver`]) —
    ///   the two `ValueError`s `sklearn.Ridge.fit` raises for this pair. Both
    ///   operands are hyperparameters, so mlrs catches them at `build()`
    ///   instead of at `fit` (the D-08 split).
    ///
    /// The stored `f64` `alpha` is narrowed to the target float `F` via cast
    /// (A5).
    pub fn build<F>(self) -> Result<Ridge<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if !(self.alpha >= 0.0) {
            return Err(BuildError::InvalidAlpha {
                estimator: "ridge",
                alpha: self.alpha,
            });
        }
        if !self.tol.is_finite() || self.tol < 0.0 {
            return Err(BuildError::InvalidTol {
                estimator: "ridge",
                tol: self.tol,
            });
        }
        if self.max_iter == Some(0) {
            return Err(BuildError::InvalidMaxIter {
                estimator: "ridge",
                max_iter: 0,
            });
        }
        if self.solver == RidgeSolver::Lbfgs && !self.positive {
            return Err(BuildError::LbfgsRequiresPositive { estimator: "ridge" });
        }
        if self.positive && !matches!(self.solver, RidgeSolver::Auto | RidgeSolver::Lbfgs) {
            return Err(BuildError::PositiveUnsupportedSolver {
                estimator: "ridge",
                solver: self.solver.name(),
            });
        }
        Ok(Ridge {
            alpha: f64_to_host::<F>(self.alpha),
            fit_intercept: self.fit_intercept,
            copy_x: self.copy_x,
            max_iter: self.max_iter,
            tol: self.tol,
            solver: self.solver,
            positive: self.positive,
            random_state: self.random_state,
            coef_: None,
            intercept_: None,
            n_iter_: None,
            solver_: None,
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }
}

impl<F> Ridge<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `coef_` (length `n_features`). `Some` by
    /// construction on the `Fitted` state, so no `NotFitted` branch is needed
    /// (the compile-time typestate replaces the runtime guard, D-03).
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.coef_
            .as_ref()
            .expect("coef_ is Some by construction on Ridge<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `intercept_` (scalar). `Some` by construction on
    /// the `Fitted` state (D-03).
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> F {
        self.intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on Ridge<F, Fitted>")
            .to_host(pool)[0]
    }

    /// sklearn's `n_iter_`: the iteration count of the solver that ran, or
    /// `None` for the solvers sklearn itself leaves unset (`cholesky`, `svd`,
    /// `sparse_cg`, `lbfgs` — see the module-doc table).
    pub fn n_iter(&self) -> Option<usize> {
        self.n_iter_
    }

    /// sklearn's `solver_`: the solver that ACTUALLY ran — `auto` already
    /// resolved, and reflecting the singular-Gram Cholesky→SVD fallback.
    pub fn solver(&self) -> RidgeSolver {
        self.solver_
            .expect("solver_ is Some by construction on Ridge<F, Fitted>")
    }

    /// `predict` for a test matrix that is still on the HOST — returns the
    /// length-`n_samples` predictions plus the operand-finiteness verdict.
    ///
    /// The host-ingress twin of [`Predict::predict`]: same result, but it reads
    /// the caller's buffer in place on cpu instead of paying an `m × n` upload
    /// that costs more than the prediction. All four dense linear regressors
    /// share ONE implementation — see [`predict_linear_from_host`] for the
    /// backend routing, the measurements, and the finiteness verdict's meaning.
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
            "ridge",
            pool,
            x,
            shape,
        )
    }
}

impl<F> Fit<F> for Ridge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = Ridge<F, Fitted>;

    /// Unweighted `fit` — [`Ridge::fit_with_sample_weight`] with no weights.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Ridge<F, Fitted>, AlgoError> {
        self.fit_with_sample_weight(pool, x, y, shape, None)
    }
}

/// Center (and, when `rescale`, `√w`-scale) the design, returning the column
/// means, the target mean, and the OWNED replacement buffers (`None` ⇒ read the
/// caller's `x`/`y` directly, with no copy at all).
///
/// Two regimes, matching sklearn's `_preprocess_data` + `_rescale_data` split:
///
/// - **No `sample_weight`** — the original DEVICE path: [`center_columns`]
///   composes `column_reduce` + the center kernel with no host round-trip of the
///   full `n×d` design (the pre-LINEAR-02-perf host two-pass form was an
///   `O(n·d)` PCIe-bound cost that dominates at scale). With
///   `!fit_intercept` there is nothing to remove and both buffers stay `None`.
/// - **With `sample_weight`** — a host pass, because neither the WEIGHTED column
///   mean nor the `√w` row scale is expressible through `center_columns`. Only
///   the weighted path pays this.
#[allow(clippy::type_complexity)]
fn preprocess<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    n_samples: usize,
    n_features: usize,
    fit_intercept: bool,
    sw: Option<&[f64]>,
    rescale: bool,
) -> Result<
    (
        Vec<f64>,
        f64,
        Option<DeviceArray<ActiveRuntime, F>>,
        Option<DeviceArray<ActiveRuntime, F>>,
    ),
    AlgoError,
>
where
    F: Float + CubeElement + Pod,
{
    let Some(sw) = sw else {
        // --- Unweighted: the untouched device path. ---
        if !fit_intercept {
            return Ok((vec![0.0f64; n_features], 0.0f64, None, None));
        }
        let (x_c, x_mean_dev) = center_columns::<F>(pool, x, (n_samples, n_features))?;
        let (y_c, y_mean_dev) = center_columns::<F>(pool, y, (n_samples, 1))?;
        let x_mean: Vec<f64> = x_mean_dev
            .to_host(pool)
            .iter()
            .map(|&v| host_to_f64(v))
            .collect();
        let y_mean = host_to_f64(y_mean_dev.to_host(pool)[0]);
        x_mean_dev.release_into(pool);
        y_mean_dev.release_into(pool);
        return Ok((x_mean, y_mean, Some(x_c), Some(y_c)));
    };

    // --- Weighted: host preprocessing. ---
    let xh: Vec<f64> = x.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let yh: Vec<f64> = y.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();

    let mut x_mean = vec![0.0f64; n_features];
    let mut y_mean = 0.0f64;
    if fit_intercept {
        // sklearn's `_preprocess_data`: `np.average(X, axis=0, weights=sw)`.
        let wsum: f64 = sw.iter().sum();
        if wsum > 0.0 {
            for r in 0..n_samples {
                let w = sw[r];
                if w == 0.0 {
                    continue;
                }
                for c in 0..n_features {
                    x_mean[c] += w * xh[r * n_features + c];
                }
                y_mean += w * yh[r];
            }
            for m in x_mean.iter_mut() {
                *m /= wsum;
            }
            y_mean /= wsum;
        }
    }

    let mut xc: Vec<F> = vec![F::from_int(0i64); n_samples * n_features];
    let mut yc: Vec<F> = vec![F::from_int(0i64); n_samples];
    for r in 0..n_samples {
        // `_rescale_data` multiplies BOTH X and y rows by √w, which leaves the
        // weighted least-squares objective `Σ wᵢ(yᵢ − xᵢβ)²` unchanged while
        // letting every unweighted solver run verbatim.
        let scale = if rescale { sw[r].sqrt() } else { 1.0 };
        for c in 0..n_features {
            let v = (xh[r * n_features + c] - x_mean[c]) * scale;
            xc[r * n_features + c] = f64_to_host::<F>(v);
        }
        yc[r] = f64_to_host::<F>((yh[r] - y_mean) * scale);
    }

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &xc);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &yc);
    Ok((x_mean, y_mean, Some(x_dev), Some(y_dev)))
}

/// The default DEVICE solve: `(XᵀX + αI)·coef = Xᵀy` by Cholesky — and, since
/// RIDGE-DEFAULT-CUDA, without a single host round-trip.
///
/// `means` is the `(x̄, ȳ)` pair from [`column_means`] on the FUSED route, in
/// which case `x_ref`/`y_ref` are the caller's RAW design and the centering
/// happens inside the accumulation kernel. `None` means `x_ref`/`y_ref` are
/// already centered (or the fit has no intercept) and the raw Gram is formed
/// directly.
///
/// Two host round-trips used to live in this function and no longer do:
///
/// | | before | now |
/// |---|---|---|
/// | centered design | an `n × d` buffer written by `center_columns`, then re-read by `gram_xty` | never materialized — [`gram_xty_centered`] subtracts as it accumulates |
/// | `α` on the diagonal | `gram.to_host()`, a host loop over `d` of the `d²` entries, `from_host()` | [`cholesky_solve_reg`]'s `alpha`, added as the kernel reads `A[i][i]` |
///
/// The α round-trip is the smaller of the two in bytes and the larger in
/// synchronisation: it drained the queue in the middle of the fit, between the
/// Gram and the factorization, so neither could overlap the other.
///
/// The numerical result is unchanged in kind — `α` still lands on the Gram
/// diagonal only, never on the intercept (D-05 / T-04-05-02) — but it is NOT
/// bit-identical to the pre-fusion path: fusing the centering re-associates the
/// accumulation. The committed `ridge_f32/f64_seed42` fixtures gate that at the
/// 1e-5 oracle tolerance.
#[allow(clippy::too_many_arguments)]
fn solve_cholesky<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_ref: &DeviceArray<ActiveRuntime, F>,
    y_ref: &DeviceArray<ActiveRuntime, F>,
    means: Option<(
        &DeviceArray<ActiveRuntime, F>,
        &DeviceArray<ActiveRuntime, F>,
    )>,
    n_samples: usize,
    n_features: usize,
    alpha64: f64,
) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    // --- Raw Gram G = XᵀX (d×d) and c = Xᵀy (d×1) via the row-blocked
    //     `gram_xty` prim (RESEARCH Open Q1 — NOT the scaled covariance;
    //     LINEAR-01/02 perf lever shared with LinearRegression). ---
    let (gram, xty) = match means {
        Some(m) => gram_xty_centered::<F>(pool, x_ref, y_ref, m, n_samples, n_features)?,
        None => gram_xty::<F>(pool, x_ref, y_ref, n_samples, n_features)?,
    };

    // --- Thread the Gram buffer through `out` so the factor reuses it in place
    //     — no parallel n² allocation (D-11 gate 2). The kernel only READS `out`
    //     as its working input, so the threaded buffer is consumed (released
    //     back to the pool) by the call; we clone the handle for `out` and keep
    //     `gram` as the `a` operand. A non-SPD pivot (near-singular Gram)
    //     surfaces NotPositiveDefinite → the caller's sklearn-faithful SVD retry
    //     (Pitfall 4 / T-04-05-01), never NaN coef_. ---
    let gram_out =
        DeviceArray::<ActiveRuntime, F>::from_raw(gram.handle().clone(), n_features * n_features);
    let coef = cholesky_solve_reg::<F>(pool, &gram, &xty, n_features, 1, alpha64, Some(gram_out));

    // The Gram buffer was consumed (its cloned handle threaded through `out` and
    // released by the Cholesky solve — so we do NOT release `gram` again here,
    // avoiding a double-release of the shared allocation). `xty` is ours either
    // way, INCLUDING on the error path that feeds the SVD retry.
    drop(gram);
    xty.release_into(pool);
    Ok(coef?)
}

/// The `solver='svd'` arm: `coef = V·diag(σ/(σ²+α))·Uᵀ·y`, sklearn's
/// `_solve_svd` including its `σ > 1e-15` cutoff.
///
/// Dual path, mirroring `LinearRegression`'s (D-02): the thin [`svd`] prim
/// inside the one-sided Jacobi kernel's shape caps, and the Gram+[`eig`] form
/// above them. The two are ALGEBRAICALLY identical — with `G = XᵀX = V·diag(λ)·Vᵀ`
/// and `λ = σ²`, `Vᵀ·Xᵀy` has entries `σᵢ·(Uᵀy)ᵢ`, so
/// `V·diag(1/(λ+α))·Vᵀ·Xᵀy = V·diag(σ/(σ²+α))·Uᵀy` term for term (the σ = 0
/// directions drop out of both forms). The eig route squares `X`'s condition
/// number, which is the accepted tradeoff above the Jacobi caps and the same one
/// `fit_gram_eig` documents.
fn solve_svd<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_ref: &DeviceArray<ActiveRuntime, F>,
    y_ref: &DeviceArray<ActiveRuntime, F>,
    n_samples: usize,
    n_features: usize,
    alpha64: f64,
) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let fits_jacobi = n_samples.max(n_features) <= SVD_JACOBI_MAX_ROWS
        && n_samples.min(n_features) <= SVD_JACOBI_MAX_COLS;

    if !fits_jacobi {
        return solve_svd_gram_eig::<F>(pool, x_ref, y_ref, n_samples, n_features, alpha64);
    }

    // --- Thin SVD: X = U·diag(σ)·Vᵀ, U (n×k), σ (k), Vᵀ (k×d), k = min(n, d). ---
    let k = n_samples.min(n_features);
    let (u, s, vt) = svd::<F>(pool, x_ref, (n_samples, n_features))?;

    // t1 = Uᵀ·y (k×1). `u` is (n×k) row-major; transa reads it as Uᵀ (k×n) —
    // no transpose buffer (D-06).
    let t1 = gemm::<F>(
        pool,
        &u,
        (k, n_samples),
        y_ref,
        (n_samples, 1),
        true,
        false,
        None,
    )?;

    // t2 = diag(σ/(σ²+α))·t1 — sklearn's `d[idx] = s_nnz / (s_nnz**2 + alpha)`
    // over the length-k spectrum (a host pass over a TINY vector).
    let s64: Vec<f64> = s.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let t1_host: Vec<f64> = t1.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let mut t2_host: Vec<F> = vec![F::from_int(0i64); k];
    for i in 0..k {
        let sigma = s64[i];
        let scaled = if sigma > SVD_ZERO_SIGMA {
            sigma / (sigma * sigma + alpha64) * t1_host[i]
        } else {
            0.0
        };
        t2_host[i] = f64_to_host::<F>(scaled);
    }
    let t2_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &t2_host);

    // coef = V·t2 (d×1). `vt` is Vᵀ (k×d) row-major; transa reads it as V.
    let coef = gemm::<F>(pool, &vt, (n_features, k), &t2_dev, (k, 1), true, false, None)?;

    u.release_into(pool);
    s.release_into(pool);
    vt.release_into(pool);
    t1.release_into(pool);
    t2_dev.release_into(pool);
    Ok(coef)
}

/// The Gram+eig form of [`solve_svd`], for shapes above the Jacobi SVD caps.
/// `coef = V·diag(1/(λ+α))·Vᵀ·Xᵀy` with `G = XᵀX = V·diag(λ)·Vᵀ`.
fn solve_svd_gram_eig<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_ref: &DeviceArray<ActiveRuntime, F>,
    y_ref: &DeviceArray<ActiveRuntime, F>,
    n_samples: usize,
    n_features: usize,
    alpha64: f64,
) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    if n_features > GRAM_EIG_MAX_FEATURES {
        return Err(AlgoError::NFeaturesExceedsMaxDim {
            estimator: "ridge (solver='svd')",
            n_features,
            max: GRAM_EIG_MAX_FEATURES,
        });
    }
    let d = n_features;
    let (gram, xty) = gram_xty::<F>(pool, x_ref, y_ref, n_samples, d)?;

    // G = V·diag(w)·Vᵀ, w DESCENDING. `v` is (d×d) COLUMN-major, so a ROW-major
    // read of that same buffer is already Vᵀ (the `eig` convention).
    let (w, v) = eig::<F>(pool, &gram, d, None)?;
    gram.release_into(pool);

    // t1 = Vᵀ·(Xᵀy).
    let t1 = gemm::<F>(pool, &v, (d, d), &xty, (d, 1), false, false, None)?;
    xty.release_into(pool);

    let w_host: Vec<f64> = w.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let t1_host: Vec<f64> = t1.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    w.release_into(pool);
    t1.release_into(pool);

    let mut t2_host: Vec<F> = vec![F::from_int(0i64); d];
    for i in 0..d {
        // λ is σ², clamped at 0 against PSD rounding noise; the cutoff is
        // sklearn's on σ itself so the two paths drop the same directions.
        let lambda = w_host[i].max(0.0);
        let sigma = lambda.sqrt();
        let denom = lambda + alpha64;
        let scaled = if sigma > SVD_ZERO_SIGMA && denom > 0.0 {
            t1_host[i] / denom
        } else {
            0.0
        };
        t2_host[i] = f64_to_host::<F>(scaled);
    }
    let t2_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &t2_host);

    // coef = V·t2 (transa reads the row-major Vᵀ buffer as V).
    let coef = gemm::<F>(pool, &v, (d, d), &t2_dev, (d, 1), true, false, None)?;
    v.release_into(pool);
    t2_dev.release_into(pool);
    Ok(coef)
}

/// The `positive=True` arm (`solver='lbfgs'`): the non-negative ridge solve,
/// on-device wherever the device kernel applies and on the host otherwise.
///
/// Both arms run the SAME projected cyclic coordinate descent on the SAME
/// device-formed Gram — ascending coordinate order, the same closed-form
/// projected update off the Gram diagonal, and the same
/// `max|Δw| ≤ tol·max(1, max|w|)` stop. They differ only in where the arithmetic
/// happens (and, on the device arm, in the summation order of the per-sweep
/// gradient rebuild). The objective is strictly convex over a box for `α > 0`,
/// so the constrained minimiser is UNIQUE and the two arms agree to within the
/// oracle tolerance rather than merely being "both plausible".
///
/// The device arm is the point of the split: `gram_xty` has already produced `G`
/// and `Xᵀy` in device memory, and the whole `O(d²)`-per-sweep solve fits in one
/// cube (`prims::nnls`), so nothing crosses the bus — where the host arm must
/// read `d² + d` floats back, solve, and re-upload `coef`. On cpu (where a
/// `d`-unit barrier-synchronised cube is a pathology, not a parallel launch) and
/// for `d` above the kernel's cube-dim cap, the host arm still carries the
/// solve, so no shape loses support.
#[allow(clippy::too_many_arguments)]
fn solve_nonnegative<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_ref: &DeviceArray<ActiveRuntime, F>,
    y_ref: &DeviceArray<ActiveRuntime, F>,
    means: Option<(
        &DeviceArray<ActiveRuntime, F>,
        &DeviceArray<ActiveRuntime, F>,
    )>,
    n_samples: usize,
    n_features: usize,
    alpha64: f64,
    tol: f64,
    max_iter: Option<usize>,
) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    if !device_nnls_applicable::<F>(n_features) {
        // `means` is `None` on this arm by construction (the caller gates the
        // fused route on the same predicate), so `x_ref` is already centered.
        let (gram, xty) = host_gram::<F>(pool, x_ref, y_ref, n_samples, n_features)?;
        let (coef, _sweeps) =
            ridge_solvers::nonnegative_cd(&gram, &xty, n_features, alpha64, tol, max_iter);
        return Ok(upload_coef::<F>(pool, &coef));
    }

    // Fully device-resident: the Gram stays where `gram_xty` wrote it and the
    // solve reads it in place, so `coef_` is produced without a single host
    // round-trip. With `means` present the centering is fused into that same
    // accumulation and `x_ref` is the caller's RAW design.
    let (gram, xty) = match means {
        Some(m) => gram_xty_centered::<F>(pool, x_ref, y_ref, m, n_samples, n_features)?,
        None => gram_xty::<F>(pool, x_ref, y_ref, n_samples, n_features)?,
    };
    let coef = ridge_nnls::<F>(pool, &gram, &xty, n_features, alpha64, tol, max_iter);
    gram.release_into(pool);
    xty.release_into(pool);
    Ok(coef?)
}

/// Form the raw Gram `XᵀX` (`d×d`) and `Xᵀy` (`d`) on-device and read the two
/// SMALL results back as `f64` for a host solver.
///
/// The `O(n·d)` reduction stays on the device; what crosses to the host is
/// `d² + d` floats, independent of `n_samples` — which is why `sparse_cg` (and
/// the `positive` arm's HOST fallback) never touch the design matrix at all.
/// The `positive` arm's device path does not call this: it solves the Gram
/// where `gram_xty` left it, with no read-back (see [`solve_nonnegative`]).
fn host_gram<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x_ref: &DeviceArray<ActiveRuntime, F>,
    y_ref: &DeviceArray<ActiveRuntime, F>,
    n_samples: usize,
    n_features: usize,
) -> Result<(Vec<f64>, Vec<f64>), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let (gram, xty) = gram_xty::<F>(pool, x_ref, y_ref, n_samples, n_features)?;
    let gram_host: Vec<f64> = gram.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let xty_host: Vec<f64> = xty.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    gram.release_into(pool);
    xty.release_into(pool);
    Ok((gram_host, xty_host))
}

/// Read the (already centered / rescaled) design back as `f64` for the two host
/// solvers that genuinely need per-ROW access — `lsqr` (Golub–Kahan
/// bidiagonalization of `X` itself) and `sag`/`saga` (one sample per step).
/// Forming the Gram for them would defeat the point of choosing those solvers.
fn host_design<F>(
    pool: &BufferPool<ActiveRuntime>,
    x_ref: &DeviceArray<ActiveRuntime, F>,
    y_ref: &DeviceArray<ActiveRuntime, F>,
) -> (Vec<f64>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let xh = x_ref.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let yh = y_ref.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    (xh, yh)
}

/// Stage a host `f64` coefficient vector back onto the device as `F` (D-03: the
/// fitted state is device-resident whichever solver produced it, so `predict`
/// has ONE path).
///
/// `pub(crate)` because `BayesianRidge` ends every fit here too — its evidence
/// loop is host-side, but its fitted state is device-resident for the same
/// reason, so the two share the shared `linear_predict` route.
pub(crate) fn upload_coef<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    coef: &[f64],
) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    let host: Vec<F> = coef.iter().map(|&v| f64_to_host::<F>(v)).collect();
    DeviceArray::from_host(pool, &host)
}

impl<F> Predict<F> for Ridge<F, Fitted>
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

        // `coef_`/`intercept_` are `Some` by construction on `Ridge<F, Fitted>`
        // (the compile-time typestate replaces the old runtime `NotFitted`
        // guard, D-03).
        let coef = self
            .coef_
            .as_ref()
            .expect("coef_ is Some by construction on Ridge<F, Fitted>");
        let intercept = self
            .intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on Ridge<F, Fitted>");

        // --- T-04-05-03 / ASVS V5: geometry + fitted-n_features consistency. ---
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

        // y_pred = X_test · coef + intercept via ONE fused device launch
        // (LINEAR-02 predict perf lever): the `linear_predict` prim's GATHER
        // matvec+bias kernel replaces the prior gemm→`intercept.to_host()`→
        // `raw.to_host()`→host bias-loop→`from_host` round-trips (the
        // `center`/`gram` host-sync pathology, same class of fix). The result
        // stays device-resident; the PyO3 boundary's terminal readback is the
        // only host↔device crossing.
        Ok(linear_predict::<F>(
            pool,
            x,
            coef,
            intercept,
            (n_samples, n_features),
        )?)
    }
}
