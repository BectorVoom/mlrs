//! `Arima` / `AutoArima` (TSA-01, Phase 22) — ARIMA(p,d,q) via an exact
//! Kalman-filter MLE, sklearn/cuML-adjacent (`endog`-only, no X/y split —
//! cuML's `tsa.arima.ARIMA` takes a single series the same way).
//!
//! ## Scope (read before reaching for `trend`/seasonal support — neither exists)
//! - **Zero-mean only.** statsmodels' `trend='c'` enters via a state-space
//!   regression component under an *approximate-diffuse* initialization —
//!   a genuinely different (and substantially more involved) filtering
//!   branch from the exact-stationary one this module uses. Reproducing it
//!   correctly needs more validation than this pass had time for, so `Arima`
//!   fits a ZERO-MEAN model on the (differenced) series only. Callers who
//!   need a mean/drift term should demean their series before fitting — the
//!   verified oracle-equivalence below is for `trend='n'`.
//! - **No seasonal component (SARIMAX (P,D,Q,s)).** Tracked as a known
//!   follow-up (Phase 22 originally scoped it in; this pass ships the
//!   non-seasonal core only).
//! - **HOST-side, not a device kernel.** The Kalman recursion's per-step
//!   state is `r×r` with `r = max(p, q+1)` — typically 2-6 — and the
//!   recursion is inherently SEQUENTIAL over time. There is no batch axis
//!   here (`Arima` fits ONE series), so there is no useful device
//!   parallelism to extract at this call size; a genuinely GPU-shaped
//!   version would batch the Kalman recursion across MANY independent
//!   series (cuML's actual `tsa.arima` batching axis) — not implemented
//!   here, tracked as a follow-up alongside seasonal support.
//!
//! ## State-space form (Harvey representation, concentrated scale)
//! Verified EXACT (loglikelihood ≤1e-8) against
//! `statsmodels.tsa.statespace.sarimax.SARIMAX(order=(p,0,q), trend='n',
//! concentrate_scale=True)` at fixed parameters, by direct comparison at
//! design time (not merely asserted — reproduced digit-for-digit). Given AR
//! coefficients `φ_1..φ_p` and MA coefficients `θ_1..θ_q`, let
//! `r = max(p, q+1)`:
//!
//! ```text
//! T[i][0] = φ_{i+1} (0 if i+1 > p),  T[i][i+1] = 1 for i < r-1   (r×r)
//! Z = [1, 0, ..., 0]                                              (1×r)
//! R = [1, θ_1, ..., θ_{r-1}]ᵀ (0 beyond q)                        (r×1)
//! ```
//! `α_{t+1} = T α_t + R η_t`, `η_t ~ N(0, 1)` (scale concentrated out),
//! `y_t = Z α_t`. Initial `α_1 = 0`, `P_1` solves the discrete Lyapunov
//! equation `P_1 = T P_1 Tᵀ + R Rᵀ` (the stationary covariance) via a
//! fixed-point iteration ([`stationary_p1`]).
//!
//! Kalman recursion per step: `v_t = y_t - Z α_t`, `F_t = Z P_t Zᵀ`,
//! `K_t = T P_t Zᵀ / F_t`, `α_{t+1} = T α_t + K_t v_t`,
//! `P_{t+1} = T P_t Tᵀ + RRᵀ - K_t F_t K_tᵀ`. The concentrated log-likelihood:
//! `σ̂² = (1/n) Σ v_t²/F_t`,
//! `ℓ = -n/2·log(2π) - n/2·(log σ̂² + 1) - 1/2 Σ log F_t`.
//!
//! MLE: `mlrs_backend::prims::lbfgs::lbfgs_minimize` directly over the raw
//! `(φ, θ)` vector — NOT a stationarity/invertibility transform (no attempt
//! to keep AR/MA roots outside the unit circle, unlike statsmodels' Jones
//! transform). The objective adds a HINGE PENALTY on the fitted residual
//! variance `σ̂²` relative to the series' own scale, guarding against the
//! well-known degenerate "common AR/MA factor" pathology (an AR root and MA
//! root drifting together, driving the concentrated likelihood to a
//! spurious unbounded extreme as `σ̂² → 0` — hit in practice during
//! development, not a hypothetical; see the barrier's construction site for
//! why a smooth coefficient-bound reparameterization does NOT fix this —
//! it saturates and masks the finite-difference gradient instead).
//! Gradient via central finite differences (the parameter count is tiny —
//! `p+q`, typically ≤ 10 — so this is cheap and avoids hand-deriving the
//! Kalman-filter adjoint).
//!
//! Tests live in `crates/mlrs-algos/tests/arima_test.rs` (AGENTS.md §2).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::lbfgs::{lbfgs_minimize, LbfgsStopReason, LBFGS_FTOL, LBFGS_GTOL, LBFGS_MAXLS};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{Fitted, State, Unfit};

/// The `(p, d, q)` component bound (keeps the Kalman state dimension and the
/// differencing pass tractable — a documented mlrs v1 scope cap, not a
/// statistical requirement).
const MAX_PQ: usize = 10;
const MAX_D: usize = 5;
/// Approximate-zero guard for the concentrated variance (a degenerate
/// all-equal series would otherwise divide by zero).
const MIN_SIGMA2: f64 = 1e-12;
/// L-BFGS gradient finite-difference step.
const FD_STEP: f64 = 1e-6;
/// Stationary-`P1` fixed-point iteration cap (r is tiny — this converges in a
/// handful of iterations for any `(φ, θ)` inside the stationary/invertible
/// region; a non-stationary trial point during the UNCONSTRAINED L-BFGS
/// search may not converge, in which case the iteration is simply truncated
/// at the cap — an inexact-but-finite `P1` that keeps the objective defined
/// everywhere, matching the "no explicit stationarity transform" scope).
const LYAPUNOV_MAX_ITER: usize = 200;

/// ARIMA(p,d,q), zero-mean, builder-fronted + typestate
/// (`Arima<F, S = Unfit>`). No `Debug` derive — the family precedent
/// (`DeviceArray` is not `Debug`), though `Arima` itself holds no device
/// state (host-only compute, see module docs).
pub struct Arima<F, S = Unfit>
where
    S: State,
{
    p: usize,
    d: usize,
    q: usize,
    /// Fitted AR coefficients (length `p`).
    ar_: Vec<f64>,
    /// Fitted MA coefficients (length `q`).
    ma_: Vec<f64>,
    /// Fitted concentrated innovation variance.
    sigma2_: f64,
    /// Log-likelihood at the fitted parameters.
    loglik_: f64,
    /// AIC / AICc / BIC at the fitted parameters.
    aic_: f64,
    aicc_: f64,
    bic_: f64,
    /// The differenced-series length actually filtered (`n_obs - d`).
    nobs_: usize,
    /// Whether the L-BFGS MLE reported `gtol` convergence.
    converged_: bool,
    /// The LAST value of the series at every differencing level (length
    /// `d`), needed to reconstruct multi-step forecasts back up to the
    /// original scale: `diff_last_[k]` is the last observation of the
    /// `k`-th difference (`diff_last_[0]` = the original series' last value).
    /// Only the last value per level is retained — the un-differencing
    /// cumulative-sum seed is the last element alone, so storing the full
    /// per-level series would waste `O(d·n)` for nothing.
    diff_last_: Vec<f64>,
    /// The final Kalman state `α_T` and covariance `P_T` at the end of the
    /// fitted (differenced) series — the forecast starting point.
    final_state_: Vec<f64>,
    final_cov_: Vec<f64>,
    _float: PhantomData<F>,
    _state: PhantomData<S>,
}

impl<F> Arima<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// `(p, d, q)` defaults to `(0, 0, 0)` (a degenerate but well-defined
    /// zero-mean white-noise model) — the sklearn-style D-08 single source.
    pub fn new() -> Self {
        Self {
            p: 0,
            d: 0,
            q: 0,
            ar_: Vec::new(),
            ma_: Vec::new(),
            sigma2_: 0.0,
            loglik_: 0.0,
            aic_: 0.0,
            aicc_: 0.0,
            bic_: 0.0,
            nobs_: 0,
            converged_: false,
            diff_last_: Vec::new(),
            final_state_: Vec::new(),
            final_cov_: Vec::new(),
            _float: PhantomData,
            _state: PhantomData,
        }
    }

    /// Start building from the defaults (D-08 single source).
    pub fn builder() -> ArimaBuilder {
        ArimaBuilder::default()
    }

    /// Fold this (unfit) estimator back into a builder.
    pub fn into_builder(self) -> ArimaBuilder {
        ArimaBuilder { p: self.p, d: self.d, q: self.q }
    }

    /// Fit to a single series `y` (device-resident, length `n_obs`).
    /// CONSUMES `self`. `n_obs` must exceed `d + max(p, q + 1)` (enough
    /// observations to difference and seed the Kalman filter) — violated
    /// geometry is a typed [`PrimError::ShapeMismatch`] BEFORE any compute.
    pub fn fit(
        self,
        pool: &BufferPool<ActiveRuntime>,
        y: &DeviceArray<ActiveRuntime, F>,
        n_obs: usize,
    ) -> Result<Arima<F, Fitted>, AlgoError> {
        if y.len() != n_obs {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_obs,
                cols: 1,
                len: y.len(),
            }));
        }
        let y_host: Vec<f64> = y.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
        fit_from_host(self.p, self.d, self.q, &y_host)
    }
}

impl<F> Default for Arima<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`Arima`] (data-INDEPENDENT validation at `build`, D-08).
#[derive(Debug, Clone, Copy, Default)]
pub struct ArimaBuilder {
    p: usize,
    d: usize,
    q: usize,
}

impl ArimaBuilder {
    /// Set the AR order `p`.
    pub fn p(mut self, v: usize) -> Self {
        self.p = v;
        self
    }
    /// Set the differencing order `d`.
    pub fn d(mut self, v: usize) -> Self {
        self.d = v;
        self
    }
    /// Set the MA order `q`.
    pub fn q(mut self, v: usize) -> Self {
        self.q = v;
        self
    }
    /// Set the full `(p, d, q)` order at once.
    pub fn order(mut self, p: usize, d: usize, q: usize) -> Self {
        self.p = p;
        self.d = d;
        self.q = q;
        self
    }

    /// Build the (unfit) estimator, validating the order bound
    /// ([`BuildError::InvalidArimaOrder`], data-INDEPENDENT, D-08).
    pub fn build<F>(self) -> Result<Arima<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if self.p > MAX_PQ || self.q > MAX_PQ || self.d > MAX_D {
            return Err(BuildError::InvalidArimaOrder {
                estimator: "arima",
                p: self.p,
                d: self.d,
                q: self.q,
                max_pq: MAX_PQ,
                max_d: MAX_D,
            });
        }
        Ok(Arima {
            p: self.p,
            d: self.d,
            q: self.q,
            ar_: Vec::new(),
            ma_: Vec::new(),
            sigma2_: 0.0,
            loglik_: 0.0,
            aic_: 0.0,
            aicc_: 0.0,
            bic_: 0.0,
            nobs_: 0,
            converged_: false,
            diff_last_: Vec::new(),
            final_state_: Vec::new(),
            final_cov_: Vec::new(),
            _float: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<F> Arima<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Fitted AR coefficients (length `p`).
    pub fn ar(&self) -> &[f64] {
        &self.ar_
    }
    /// Fitted MA coefficients (length `q`).
    pub fn ma(&self) -> &[f64] {
        &self.ma_
    }
    /// Fitted (concentrated) innovation variance.
    pub fn sigma2(&self) -> f64 {
        self.sigma2_
    }
    /// Log-likelihood at the fitted parameters.
    pub fn loglik(&self) -> f64 {
        self.loglik_
    }
    /// Akaike information criterion.
    pub fn aic(&self) -> f64 {
        self.aic_
    }
    /// Corrected AIC (AutoARIMA's model-selection criterion).
    pub fn aicc(&self) -> f64 {
        self.aicc_
    }
    /// Bayesian information criterion.
    pub fn bic(&self) -> f64 {
        self.bic_
    }
    /// The differenced-series length actually filtered.
    pub fn nobs(&self) -> usize {
        self.nobs_
    }
    /// The fitted `(p, d, q)` order.
    pub fn order(&self) -> (usize, usize, usize) {
        (self.p, self.d, self.q)
    }
    /// Whether the L-BFGS MLE reported `gtol` convergence (`false` means the
    /// fit reached `maxiter` or a line-search stall — the returned
    /// parameters are the best iterate found, not a certified optimum).
    pub fn converged(&self) -> bool {
        self.converged_
    }

    /// The terminal Kalman state covariance `P_T` (row-major `r×r`, `r =
    /// max(p, q+1)`) at the end of the fitted (differenced) series —
    /// [`Self::forecast`]'s starting-point uncertainty (debug/diagnostics;
    /// forecast confidence intervals are not otherwise exposed in v1).
    pub fn final_cov(&self) -> &[f64] {
        &self.final_cov_
    }

    /// Forecast `n_periods` steps ahead (original series scale), CONSUMING
    /// no state (the fitted `self` is reusable). The Kalman state is
    /// propagated forward with zero innovations (`v_t = 0`, the standard
    /// multi-step point-forecast recursion), then the `d` differencing
    /// levels are un-done via cumulative summing from the original series'
    /// tail.
    pub fn forecast(&self, n_periods: usize) -> Vec<f64> {
        let r = self.final_state_.len();
        let (t_mat, _z, _r_vec) = build_state_space(&self.ar_, &self.ma_, r);
        let mut a = self.final_state_.clone();
        let mut diff_forecast = Vec::with_capacity(n_periods);
        for _ in 0..n_periods {
            diff_forecast.push(a[0]);
            a = mat_vec(&t_mat, &a, r);
        }
        // Un-difference: level_k = cumulative sum of level_{k+1} seeded from
        // the ORIGINAL series' tail at that level (standard ARIMA forecast
        // reconstruction, one level at a time from the innermost difference
        // back up to the original scale).
        let mut level = diff_forecast;
        for k in (0..self.d).rev() {
            let mut last = self.diff_last_[k];
            let mut out = Vec::with_capacity(level.len());
            for &v in &level {
                last += v;
                out.push(last);
            }
            level = out;
        }
        level
    }
}

/// Shared fit body (used by both `Arima::fit` and [`AutoArima`]'s grid
/// search): difference `y` `d` times, MLE the zero-mean ARMA(p,q) Kalman
/// filter, and materialize the `Fitted` estimator.
fn fit_from_host<F>(p: usize, d: usize, q: usize, y: &[f64]) -> Result<Arima<F, Fitted>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let r = p.max(q + 1).max(1);
    let n_obs = y.len();
    // Enough points to difference d times and still seed an r-state filter
    // with at least a handful of observations (T-22-01 / ASVS V5).
    if n_obs < d + r + 1 {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "y (too few observations for the requested order)",
            rows: n_obs,
            cols: 1,
            len: d + r + 1,
        }));
    }

    // --- Differencing, keeping only each level's LAST value for forecast
    //     reconstruction (the un-differencing cumulative sum seeds from the
    //     last element alone — no need to retain the full per-level series). ---
    let mut diff_last = Vec::with_capacity(d);
    let mut cur = y.to_vec();
    for _ in 0..d {
        diff_last.push(*cur.last().expect("differencing level is non-empty (n_obs > d guard)"));
        cur = cur.windows(2).map(|w| w[1] - w[0]).collect();
    }
    let y_diff = cur;
    let nobs = y_diff.len();

    // --- MLE via L-BFGS over the raw (φ, θ) vector, gradient by central
    //     finite differences (the STATEMENT-form small-n_params case). ---
    let n_params = p + q;
    let result = if n_params == 0 {
        // A pure zero-mean white-noise model: no free (φ, θ) — sigma2 is the
        // sample variance, no optimization needed.
        None
    } else {
        let x0 = vec![0.0f64; n_params];
        // Unconstrained Gaussian-concentrated-likelihood ARMA MLE has a
        // classic degeneracy: an AR root and MA root can drift toward a
        // common value ("common factor"), driving the fitted residual
        // variance σ̂² → 0 — and since the concentrated loglik has a
        // `-n/2·log(σ̂²)` term, that makes the "likelihood" grow WITHOUT
        // BOUND as σ̂² → 0 (a real failure mode, hit during development on
        // the module's own oracle fixture, not a hypothetical). A `tanh`
        // coefficient bound does NOT fix this: L-BFGS's line search reaches
        // the saturated region in one step (the unpenalized gradient near
        // the origin already points steeply toward larger coefficients),
        // and `tanh`'s vanishing derivative there makes the chain-ruled
        // finite-difference gradient collapse to ~0 regardless of what the
        // true landscape looks like — L-BFGS reports false `gtol`
        // convergence AT the saturation boundary. So the coefficients are
        // optimized DIRECTLY (no reparameterization); the guard is instead
        // an OUTRIGHT REJECTION whenever σ̂² falls below `sigma2_floor` (a
        // small fraction of `Var(y_diff)` — a legitimately-fit series
        // should never need a residual variance many orders of magnitude
        // below its own variance). A same-order-of-growth PENALTY (tried
        // first, at both linear and quadratic order in `-ln(σ̂²)`) does not
        // reliably win: the degenerate reward is ITSELF `-n/2·log σ̂²`, so
        // matching its growth rate only narrows the margin, never
        // guarantees it (empirically confirmed insufficient during
        // development on this module's own oracle fixture — with a linear
        // penalty ~1.6e3 lost to a ~4e4 reward; quadratic still lost).
        // A flat, large, constant rejection sidesteps that arms race
        // entirely: the L-BFGS strong-Wolfe line search backtracks (smaller
        // step, same direction) whenever a trial point fails sufficient
        // decrease, so returning "infeasible" here reliably pushes the
        // search back toward the well-posed region instead of accepting
        // the degenerate jump.
        let mean_y = y_diff.iter().sum::<f64>() / nobs as f64;
        let var_y = y_diff.iter().map(|&v| (v - mean_y).powi(2)).sum::<f64>() / nobs as f64;
        let sigma2_floor = (var_y * 1e-6).max(MIN_SIGMA2 * 100.0);
        let barrier_obj = |phi: &[f64], theta: &[f64]| -> f64 {
            let pass = kalman_pass(phi, theta, &y_diff);
            let raw_sigma2 = pass.sigma2; // already floored at MIN_SIGMA2 by kalman_pass
            if raw_sigma2 < sigma2_floor {
                return 1e12;
            }
            -pass.loglik
        };
        let obj = |x: &[f64]| -> (f64, Vec<f64>) {
            let (phi, theta) = split_params(x, p, q);
            let ll = barrier_obj(&phi, &theta);
            let mut grad = vec![0.0f64; n_params];
            for k in 0..n_params {
                let mut xp = x.to_vec();
                let mut xm = x.to_vec();
                xp[k] += FD_STEP;
                xm[k] -= FD_STEP;
                let (phi_p, theta_p) = split_params(&xp, p, q);
                let (phi_m, theta_m) = split_params(&xm, p, q);
                let lp = barrier_obj(&phi_p, &theta_p);
                let lm = barrier_obj(&phi_m, &theta_m);
                grad[k] = (lp - lm) / (2.0 * FD_STEP);
            }
            (ll, grad)
        };
        Some(
            lbfgs_minimize(x0, obj, LBFGS_GTOL, LBFGS_FTOL, LBFGS_MAXLS, 200)
                .map_err(AlgoError::Prim)?,
        )
    };

    let (params, converged) = match &result {
        Some(r) => (
            r.x.clone(),
            r.converged || matches!(r.stop_reason, LbfgsStopReason::FtolStall),
        ),
        None => (Vec::new(), true),
    };
    let (ar, ma) = split_params(&params, p, q);

    // --- Final filter pass at the fitted params: sigma2, loglik, and the
    //     terminal (state, cov) forecast starting point. ---
    let pass = kalman_pass(&ar, &ma, &y_diff);

    let k = n_params as f64; // free params (sigma2 is concentrated, not counted separately per statsmodels' concentrate_scale convention... but AIC counts it: +1)
    let k_aic = k + 1.0; // + sigma2
    let n = nobs as f64;
    let aic = -2.0 * pass.loglik + 2.0 * k_aic;
    let aicc = if n - k_aic - 1.0 > 0.0 {
        aic + (2.0 * k_aic * (k_aic + 1.0)) / (n - k_aic - 1.0)
    } else {
        f64::INFINITY
    };
    let bic = -2.0 * pass.loglik + k_aic * n.ln();

    Ok(Arima {
        p,
        d,
        q,
        ar_: ar,
        ma_: ma,
        sigma2_: pass.sigma2,
        loglik_: pass.loglik,
        aic_: aic,
        aicc_: aicc,
        bic_: bic,
        nobs_: nobs,
        converged_: converged,
        diff_last_: diff_last,
        final_state_: pass.final_state,
        final_cov_: pass.final_cov,
        _float: PhantomData,
        _state: PhantomData,
    })
}

/// Split a flat `[φ_1..φ_p, θ_1..θ_q]` parameter vector.
fn split_params(x: &[f64], p: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
    (x[..p].to_vec(), x[p..p + q].to_vec())
}

/// Build the Harvey state-space `(T, Z, R)` for AR coefficients `phi`, MA
/// coefficients `theta`, at state dimension `r` (verified exact — module
/// docs). `Z` is always `[1, 0, ..., 0]` so callers rarely need it; returned
/// for completeness/symmetry with the recursion.
fn build_state_space(phi: &[f64], theta: &[f64], r: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut t_mat = vec![0.0f64; r * r];
    for i in 0..r {
        t_mat[i * r] = phi.get(i).copied().unwrap_or(0.0);
        if i + 1 < r {
            t_mat[i * r + i + 1] = 1.0;
        }
    }
    let mut z = vec![0.0f64; r];
    z[0] = 1.0;
    let mut r_vec = vec![0.0f64; r];
    r_vec[0] = 1.0;
    for i in 1..r {
        r_vec[i] = theta.get(i - 1).copied().unwrap_or(0.0);
    }
    (t_mat, z, r_vec)
}

/// Solve the discrete Lyapunov equation `P = T P Tᵀ + Q` via a fixed-point
/// iteration (module docs: `LYAPUNOV_MAX_ITER`, truncated — not asserted —
/// on non-convergence so the objective stays defined off the
/// stationary/invertible region during the unconstrained L-BFGS search).
fn stationary_p1(t_mat: &[f64], q_mat: &[f64], r: usize) -> Vec<f64> {
    let mut p = q_mat.to_vec();
    for _ in 0..LYAPUNOV_MAX_ITER {
        let tp = mat_mat(t_mat, &p, r);
        let tpt = mat_mat_t(&tp, t_mat, r);
        let mut next = vec![0.0f64; r * r];
        let mut max_diff = 0.0f64;
        for i in 0..r * r {
            next[i] = tpt[i] + q_mat[i];
            max_diff = max_diff.max((next[i] - p[i]).abs());
        }
        p = next;
        if max_diff < 1e-13 {
            break;
        }
    }
    p
}

/// `r×r · r×r` matrix product.
fn mat_mat(a: &[f64], b: &[f64], r: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut acc = 0.0f64;
            for k in 0..r {
                acc += a[i * r + k] * b[k * r + j];
            }
            out[i * r + j] = acc;
        }
    }
    out
}

/// `A · Bᵀ` for `r×r` matrices.
fn mat_mat_t(a: &[f64], b: &[f64], r: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut acc = 0.0f64;
            for k in 0..r {
                acc += a[i * r + k] * b[j * r + k];
            }
            out[i * r + j] = acc;
        }
    }
    out
}

/// `r×r · r×1`.
fn mat_vec(a: &[f64], x: &[f64], r: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r];
    for i in 0..r {
        let mut acc = 0.0f64;
        for k in 0..r {
            acc += a[i * r + k] * x[k];
        }
        out[i] = acc;
    }
    out
}

/// The result of one full Kalman-filter pass over a (differenced) series.
struct KalmanPass {
    sigma2: f64,
    loglik: f64,
    final_state: Vec<f64>,
    final_cov: Vec<f64>,
}

/// One full Kalman-filter pass (Harvey representation, concentrated scale —
/// module docs). `phi`/`theta` are raw (unconstrained) coefficient vectors.
fn kalman_pass(phi: &[f64], theta: &[f64], y: &[f64]) -> KalmanPass {
    let p = phi.len();
    let r = p.max(theta.len() + 1).max(1);
    let (t_mat, _z, r_vec) = build_state_space(phi, theta, r);
    let q_mat = outer(&r_vec, r); // R Rᵀ (σ²=1, concentrated)

    let mut a = vec![0.0f64; r];
    let mut cov = stationary_p1(&t_mat, &q_mat, r);

    let n = y.len();
    let mut sum_v2_f = 0.0f64;
    let mut sum_log_f = 0.0f64;

    for &yt in y {
        let v = yt - a[0]; // Z a = a[0]
        let f = cov[0].max(1e-300); // Z P Zᵀ = P[0][0]
        // K = T P Zᵀ / F — Zᵀ selects column 0 of P, so T·P[:,0] / F.
        let p_col0: Vec<f64> = (0..r).map(|i| cov[i * r]).collect();
        let tp_col0 = mat_vec(&t_mat, &p_col0, r);
        let k: Vec<f64> = tp_col0.iter().map(|&v| v / f).collect();

        let ta = mat_vec(&t_mat, &a, r);
        a = (0..r).map(|i| ta[i] + k[i] * v).collect();

        // P_{t+1} = T P Tᵀ + Q - K F Kᵀ.
        let tp = mat_mat(&t_mat, &cov, r);
        let tpt = mat_mat_t(&tp, &t_mat, r);
        let kfk = outer(&k, r).iter().map(|&v| v * f).collect::<Vec<f64>>();
        cov = (0..r * r).map(|i| tpt[i] + q_mat[i] - kfk[i]).collect();

        sum_v2_f += v * v / f;
        sum_log_f += f.ln();
    }

    let sigma2 = (sum_v2_f / n as f64).max(MIN_SIGMA2);
    let loglik =
        -0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln() - 0.5 * n as f64 * (sigma2.ln() + 1.0)
            - 0.5 * sum_log_f;

    KalmanPass { sigma2, loglik, final_state: a, final_cov: cov }
}

/// The concentrated Kalman-filter log-likelihood of `y` (already
/// differenced, if applicable) at FIXED raw `(phi, theta)` — no
/// optimization. A public diagnostic entry point into the same recursion
/// [`Arima::fit`] optimizes over (verified exact vs
/// `statsmodels.tsa.statespace.sarimax.SARIMAX(trend='n',
/// concentrate_scale=True).loglike` — module docs).
pub fn loglik(phi: &[f64], theta: &[f64], y: &[f64]) -> f64 {
    kalman_pass(phi, theta, y).loglik
}

/// `x xᵀ` for a length-`r` vector.
fn outer(x: &[f64], r: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r * r];
    for i in 0..r {
        for j in 0..r {
            out[i * r + j] = x[i] * x[j];
        }
    }
    out
}

// ===========================================================================
// AutoArima — bounded (p, q) grid search over AICc, at a fixed d.
// ===========================================================================

/// `AutoArima` (TSA-01): fit every `(p, q)` in `0..=max_p × 0..=max_q` at a
/// FIXED, caller-supplied `d` (mlrs does not auto-select `d` — a documented
/// scope reduction from pmdarima/`auto_arima`'s KPSS-driven `d` search),
/// keep the CONVERGED fit with the lowest AICc. An EXHAUSTIVE grid, not the
/// Hyndman-Khandakar stepwise heuristic pmdarima uses — simpler, slower,
/// and exactly reproducible, which is what this v1 scope calls for.
pub struct AutoArima;

impl AutoArima {
    /// Search `p ∈ 0..=max_p`, `q ∈ 0..=max_q` at `d`, returning the
    /// lowest-AICc CONVERGED fit. Errors ([`AlgoError::NotConverged`]) if
    /// every candidate in the grid failed to converge.
    pub fn search<F>(
        pool: &BufferPool<ActiveRuntime>,
        y: &DeviceArray<ActiveRuntime, F>,
        n_obs: usize,
        d: usize,
        max_p: usize,
        max_q: usize,
    ) -> Result<Arima<F, Fitted>, AlgoError>
    where
        F: Float + CubeElement + Pod,
    {
        if y.len() != n_obs {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_obs,
                cols: 1,
                len: y.len(),
            }));
        }
        if d > MAX_D || max_p > MAX_PQ || max_q > MAX_PQ {
            return Err(AlgoError::Build(BuildError::InvalidArimaOrder {
                estimator: "auto_arima",
                p: max_p,
                d,
                q: max_q,
                max_pq: MAX_PQ,
                max_d: MAX_D,
            }));
        }
        let y_host: Vec<f64> = y.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();

        let mut best: Option<Arima<F, Fitted>> = None;
        for p in 0..=max_p {
            for q in 0..=max_q {
                let candidate = fit_from_host::<F>(p, d, q, &y_host);
                let Ok(cand) = candidate else { continue };
                if !cand.converged() || !cand.aicc().is_finite() {
                    continue;
                }
                let better = match &best {
                    None => true,
                    Some(b) => cand.aicc() < b.aicc(),
                };
                if better {
                    best = Some(cand);
                }
            }
        }
        best.ok_or(AlgoError::NotConverged { estimator: "auto_arima", max_iter: 200 })
    }
}
