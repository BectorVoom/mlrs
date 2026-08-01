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
//! | `cholesky` | DEVICE: [`gram_xty`] + [`cholesky_solve`] on `(XᵀX + αI)·coef = Xᵀy` | `None` |
//! | `svd` | DEVICE: [`svd`] then `coef = V·diag(σ/(σ²+α))·Uᵀy`; the Gram+[`eig`] form above the Jacobi caps | `None` |
//! | `sparse_cg` | HOST CG on the device-formed Gram (sklearn's own `Xᵀ(X·x) + αx` operator) | `None` |
//! | `lsqr` | HOST Paige–Saunders LSQR with `damp = √α` | `Some(itn)` |
//! | `sag` / `saga` | HOST stochastic average gradient, sklearn's `get_auto_step_size` | `Some(epochs)` |
//! | `lbfgs` | HOST projected coordinate descent on the Gram (the `positive=True` arm) | `None` |
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
//! | `cholesky` | centering, Gram, factorization, solve — ALL on device | the `d×d` Gram (for the α diagonal write, which cubecl 0.10 cannot do in place) |
//! | `svd` | centering, SVD (or Gram+eig), both GEMMs | the length-`k` spectrum |
//! | `sparse_cg`, `lbfgs` | centering, Gram/`Xᵀy` — the whole `O(n·d)` reduction | `d² + d` floats, INDEPENDENT of `n_samples` |
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
//! Tests live in `crates/mlrs-algos/tests/ridge_test.rs` and
//! `crates/mlrs-algos/tests/ridge_params_test.rs` (AGENTS.md §2), never an
//! in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::center::center_columns;
use mlrs_backend::prims::cholesky::cholesky_solve;
use mlrs_backend::prims::eig::eig;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::gram::gram_xty;
use mlrs_backend::prims::linear_predict::{linear_predict, HostMirror, HostPrediction};
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
        let sw64: Option<Vec<f64>> = match sample_weight {
            Some(sw) => {
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
                        estimator: "ridge",
                        index: bad,
                        value: sw[bad],
                    });
                }
                // All-zero weights leave nothing to fit: the `√w` rescale zeroes
                // the whole design and the penalized solve would hand back the
                // all-zero coefficient vector as though it were an answer.
                if sw.iter().all(|&v| v == 0.0) {
                    return Err(AlgoError::ZeroSampleWeightSum { estimator: "ridge" });
                }
                Some(sw)
            }
            None => None,
        };

        let resolved = self.solver.resolve(self.positive);

        // RIDGE_PROFILE=1: per-phase wall-clock attribution (the LR_PROFILE
        // precedent in `linear_regression.rs`'s `fit_gram_eig` — attribution
        // only, since kernel launches are async and a lap only completes at the
        // next readback that drains the queue; a tiny forced readback after
        // `gram_xty`/`cholesky_solve` pins each phase's lap to ITS OWN kernels
        // rather than bleeding into the next phase's).
        let profile = std::env::var("RIDGE_PROFILE").is_ok();
        let lap0 = std::time::Instant::now();

        // --- 1. Centering + (for the non-SAG solvers) the `√w` row rescale.
        //        With NO sample_weight this is the original DEVICE-resident
        //        `center_columns` composition, unchanged: no host round-trip of
        //        the full n×d design. See `preprocess` for the weighted arm. ---
        let rescale = sw64.is_some() && !resolved.takes_sample_weight_directly();
        let (x_mean, y_mean, x_owned, y_owned) = preprocess::<F>(
            pool,
            x,
            y,
            n_samples,
            n_features,
            self.fit_intercept,
            sw64.as_deref(),
            rescale,
        )?;
        let x_ref = x_owned.as_ref().unwrap_or(x);
        let y_ref = y_owned.as_ref().unwrap_or(y);
        let t_center = if profile { lap0.elapsed().as_secs_f64() } else { 0.0 };

        // --- 2. Solve. Each arm returns the device-resident `coef_`, the
        //        `n_iter_` sklearn would report for it, and the solver actually
        //        used (which differs from `resolved` only on the singular-Gram
        //        Cholesky→SVD fallback). ---
        let lap1 = std::time::Instant::now();
        let (coef, n_iter, solver_used) = match resolved {
            RidgeSolver::Auto => unreachable!("resolve() never returns Auto"),
            RidgeSolver::Cholesky => {
                match solve_cholesky::<F>(pool, x_ref, y_ref, n_samples, n_features, alpha64) {
                    Ok(coef) => (coef, None, RidgeSolver::Cholesky),
                    // sklearn's `except LinAlgError: solver = "svd"` retry.
                    Err(AlgoError::Prim(PrimError::NotPositiveDefinite { .. })) => (
                        solve_svd::<F>(pool, x_ref, y_ref, n_samples, n_features, alpha64)?,
                        None,
                        RidgeSolver::Svd,
                    ),
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
                let (gram, xty) = host_gram::<F>(pool, x_ref, y_ref, n_samples, n_features)?;
                let (coef, _sweeps) = ridge_solvers::nonnegative_cd(
                    &gram,
                    &xty,
                    n_features,
                    alpha64,
                    self.tol,
                    self.max_iter,
                );
                // sklearn leaves `n_iter_` at None for the lbfgs arm.
                (upload_coef::<F>(pool, &coef), None, RidgeSolver::Lbfgs)
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
        let t_solve = if profile { lap1.elapsed().as_secs_f64() } else { 0.0 };

        if let Some(xc) = x_owned {
            xc.release_into(pool);
        }
        if let Some(yc) = y_owned {
            yc.release_into(pool);
        }

        // --- 3. intercept_ = ȳ − x̄·coef_ when fit_intercept, else 0 (D-05). α
        //        is NOT applied here — the intercept is unpenalized — and NEITHER
        //        is the `positive` bound (sklearn constrains only `coef_`). ---
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
        let intercept_dev: DeviceArray<ActiveRuntime, F> =
            DeviceArray::from_host(pool, &[f64_to_host::<F>(intercept)]);

        if profile {
            eprintln!(
                "RIDGE_PROFILE n={n_samples} d={n_features} solver={}: \
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
            coef_: Some(coef),
            intercept_: Some(intercept_dev),
            n_iter_: n_iter,
            solver_: Some(solver_used),
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }
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

/// The default DEVICE solve: `(XᵀX + αI)·coef = Xᵀy` by Cholesky.
///
/// Byte-identical to the pre-parameter-surface implementation — the committed
/// `ridge_f32/f64_seed42` fixtures exercise exactly this path and see NO
/// behavioural change.
fn solve_cholesky<F>(
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
    // --- Raw Gram G = XᵀX (d×d) and c = Xᵀy (d×1) via the row-blocked
    //     `gram_xty` prim (RESEARCH Open Q1 — NOT the scaled covariance;
    //     LINEAR-01/02 perf lever shared with LinearRegression). ---
    let (raw_gram, xty) = gram_xty::<F>(pool, x_ref, y_ref, n_samples, n_features)?;

    // --- alpha on the Gram DIAGONAL only (D-05 / T-04-05-02). Add `alpha` to
    //     element [i·n+i]; NEVER to the intercept (the intercept is recovered
    //     post-solve, outside this penalized system). cubecl 0.10 has no
    //     in-place device write, so we materialize the small n×n Gram, add α on
    //     the diagonal, RELEASE the raw-Gram buffer back to the pool (so no
    //     parallel n² buffer lives), and re-stage the regularized Gram —
    //     `from_host` recycles the just-released n² byte-size from the free-list
    //     (D-11 gate 2: no second live n²). ---
    let mut gram_host = raw_gram.to_host(pool);
    for i in 0..n_features {
        let d = host_to_f64(gram_host[i * n_features + i]) + alpha64;
        gram_host[i * n_features + i] = f64_to_host::<F>(d);
    }
    raw_gram.release_into(pool);
    let gram: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &gram_host);

    // --- Thread the regularized Gram buffer through `out` so the factor reuses
    //     it in place — no parallel n² allocation (D-11 gate 2). The kernel only
    //     READS `out` as its working input, so the threaded buffer is consumed
    //     (released back to the pool) by the call; we clone the handle for `out`
    //     and keep `gram` as the `a` operand. A non-SPD pivot (near-singular
    //     Gram) surfaces NotPositiveDefinite → the caller's sklearn-faithful SVD
    //     retry (Pitfall 4 / T-04-05-01), never NaN coef_. ---
    let gram_out =
        DeviceArray::<ActiveRuntime, F>::from_raw(gram.handle().clone(), n_features * n_features);
    let coef = cholesky_solve::<F>(pool, &gram, &xty, n_features, 1, Some(gram_out))?;

    // The Gram buffer was consumed (its cloned handle threaded through `out` and
    // released by the Cholesky solve — so we do NOT release `gram` again here,
    // avoiding a double-release of the shared allocation).
    drop(gram);
    xty.release_into(pool);
    Ok(coef)
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

/// Form the raw Gram `XᵀX` (`d×d`) and `Xᵀy` (`d`) on-device and read the two
/// SMALL results back as `f64` for a host solver.
///
/// The `O(n·d)` reduction stays on the device; what crosses to the host is
/// `d² + d` floats, independent of `n_samples` — which is why `sparse_cg` and
/// the `positive` arm never touch the design matrix at all.
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
fn upload_coef<F>(pool: &mut BufferPool<ActiveRuntime>, coef: &[f64]) -> DeviceArray<ActiveRuntime, F>
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
