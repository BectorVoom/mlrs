//! `HuberRegressor` (HUBER-01) — L2-regularized robust linear regression,
//! ≈ `sklearn.linear_model.HuberRegressor`.
//!
//! ## The model
//! Huber regression minimizes the squared loss on the samples whose scaled
//! residual `|(yᵢ − xᵢ·w − c)/σ|` is below `epsilon` and the (scaled) ABSOLUTE
//! loss on the ones above it, so a handful of gross outliers can move the fit by
//! a bounded amount instead of dominating it. The scale `σ` is not a
//! hyperparameter: it is FITTED jointly with `(w, c)`, which is what makes the
//! estimator equivariant under rescaling `y` and lets `epsilon` carry a single
//! fixed default (1.35) across problems.
//!
//! ```text
//!   min_{w, c, σ>0}  σ·Σᵢ swᵢ
//!                  + Σ_{|rᵢ| ≤ ε·σ}  swᵢ·rᵢ²/σ
//!                  + Σ_{|rᵢ| >  ε·σ} swᵢ·(2·ε·|rᵢ| − ε²·σ)
//!                  + α·‖w‖²                     with rᵢ = yᵢ − xᵢ·w − c.
//! ```
//!
//! Everything about the parameterization is pinned to sklearn's
//! `_huber_loss_and_gradient`: the penalty is `α·‖w‖²` (NOT `½α‖w‖²`), it never
//! touches the intercept or `σ`, and `σ` multiplies `Σ swᵢ` — sklearn's
//! `n_samples`, which is the sample-weight SUM, not the row count.
//!
//! ## Solver — L-BFGS over `[w, c, σ]`, with a barrier on `σ`
//! The objective is the perspective transform of the Huber loss and so is
//! JOINTLY convex on `σ > 0` with a unique minimizer. sklearn hands it to
//! `scipy.optimize.minimize(method="L-BFGS-B")` with a box that only pins
//! `σ ≥ 10·eps`; mlrs uses the validated 05-06 [`lbfgs_minimize`] primitive,
//! which is UNbounded, and reproduces the one active constraint with a `+∞`
//! barrier ([`SIGMA_FLOOR`]). That is exact rather than an approximation: the
//! bound is never active at the optimum of a non-degenerate problem (the
//! `Σ swᵢrᵢ²/σ` term diverges as `σ → 0⁺`), so it exists only to keep the line
//! search off the half-line where the loss is undefined, which is precisely what
//! rejecting the step does. The strong-Wolfe search's Armijo test reads `+∞` as
//! "no sufficient decrease" and zooms back into the feasible bracket without
//! ever looking at the gradient there.
//!
//! ## Why mlrs solves TIGHTER than sklearn does (and what it means for the oracle)
//! sklearn passes `tol` through as scipy's `gtol` but leaves `factr` at its
//! default `1e7`, so the run actually stops on the **relative-f** criterion
//! `Δf/max(|f|,1) ≤ 1e7·eps ≈ 2.2e-9` long before the gradient test can fire.
//! Measured consequence (sklearn 1.9.0, `n ∈ {200, 500, 2000, 20000}`): the
//! returned parameters sit ~1.4e-6 … 5.6e-6 (absolute) away from the true
//! minimizer, and **`tol` cannot move that** — sweeping `tol` from `1e-5` down
//! to `1e-12` leaves `n_iter_` and every fitted attribute bit-identical.
//!
//! mlrs therefore runs [`lbfgs_minimize`] with the caller's `tol` as `gtol` but
//! a TIGHT `ftol = 64·eps`, which lands within ~4e-9 of the minimizer for ~3
//! extra objective evaluations (13 → 16 at `n = 200`, 18 → 20 at `n = 20 000`).
//! Two things follow, and both are deliberate:
//!
//! - mlrs's answer is a strictly BETTER minimizer of sklearn's own objective, so
//!   the rigorous oracle gate is the objective VALUE (`huber_test.rs` checks
//!   `L(mlrs) ≤ L(sklearn)`), not bit-agreement of the parameters.
//! - the parameter agreement is bounded below by sklearn's own `factr` residual —
//!   ~5.6e-6 absolute — so the fixture band is documented at that scale rather
//!   than at 1e-5, and `gen_oracle.py` ASSERTS the residual's size so a scipy or
//!   sklearn upgrade that changes it is caught at generation.
//!
//! Matching sklearn's `factr` instead would not help: both runs would then stop
//! short of the optimum in unrelated directions and the gap would roughly
//! DOUBLE.
//!
//! ## Per-iteration cost
//! The whole cost of the fit is the objective evaluation, which is
//! [`HuberObjective`] — one fused pass over the design on cpu (against
//! scikit-learn's five passes and two full-size fancy-index copies per
//! evaluation), the two-GEMM shape on the device backends. See that prim's
//! module docs for the measured breakdown.
//!
//! Tests live in `crates/mlrs-algos/tests/huber_test.rs` (AGENTS.md §2), never
//! an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::huber_objective::{HuberDesign, HuberEval, HuberObjective};
use mlrs_backend::prims::lbfgs::{lbfgs_minimize, LbfgsStopReason, LBFGS_MAXLS};
use mlrs_backend::prims::linear_predict::{HostMirror, HostPrediction};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::linear::elastic_net::{predict_linear, predict_linear_from_host};
use crate::linear::ridge::{upload_coef, validate_sample_weight};
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};

/// The lower bound scipy's box puts on `σ`, reproduced verbatim:
/// `np.finfo(np.float64).eps * 10` (`_huber.py`, "setting it to be zero might
/// cause undefined bounds"). Below this the objective is rejected with `+∞`
/// (module docs).
pub const SIGMA_FLOOR: f64 = f64::EPSILON * 10.0;

/// sklearn's initial scale when not warm-starting: `parameters[-1] = 1`.
const SIGMA_INIT: f64 = 1.0;

/// scipy L-BFGS-B `factr·eps` is `2.2e-9`; mlrs uses `64·eps` instead — see the
/// module docs for why solving tighter is the right call and what it costs.
const HUBER_FTOL: f64 = 64.0 * f64::EPSILON;

/// Multiplier on the dtype precision floor `sqrt(eps_F)` below which a
/// line-search breakdown is accepted as "already at the numerical minimum"
/// rather than reported as a non-stationary failure — the `linear_svc.rs`
/// `floor_accept` shape.
///
/// Scaled by `Σ swᵢ` because the Huber gradient is an `O(n)` sum where the SVM's
/// is `O(1)` after its `½‖w‖²` normalization: the resolvable gradient near a
/// flat minimum grows with the number of terms accumulated into it.
const GRAD_FLOOR_K: f64 = 0.5;

/// Robust linear regressor (HUBER-01). Construct via [`HuberRegressor::builder`],
/// then the consuming [`Fit::fit`] (or [`HuberRegressor::fit_with_sample_weight`] /
/// [`HuberRegressor::fit_from_host_slice`]) + [`Predict::predict`].
///
/// Fitted `coef_` (length `n_features`) and `intercept_` (length 1) are
/// device-resident (D-03) so `predict` shares the dense linear regressors' fused
/// matvec path; `scale_` / `n_iter_` / `outliers_` are host scalars the solve
/// produces directly. The host accessors exist ONLY on
/// `HuberRegressor<F, Fitted>` (the compile-time typestate replaces a runtime
/// `NotFitted` guard, D-03).
pub struct HuberRegressor<F, S = Unfit> {
    /// Outlier cut-off on the SCALED residual (sklearn `epsilon`, `>= 1.0`).
    epsilon: f64,
    /// L-BFGS iteration cap (sklearn `max_iter`).
    max_iter: usize,
    /// Strength of the `α·‖w‖²` penalty (sklearn `alpha`).
    alpha: f64,
    /// Whether a subsequent `fit` seeds itself from this fit's parameters
    /// (sklearn `warm_start`). Carried onto the fitted estimator so
    /// [`HuberRegressor::warm_start_params`] can tell a caller whether to.
    warm_start: bool,
    /// Whether to fit an intercept (sklearn `fit_intercept`).
    fit_intercept: bool,
    /// L-BFGS gradient tolerance (sklearn `tol`).
    tol: f64,
    /// Warm-start seed `[coef…, intercept?, σ]`, length `d_aug + 1`. `None` is
    /// sklearn's cold start (all-zero weights, `σ = 1`). Only consulted when
    /// `warm_start` is set.
    init_params: Option<Vec<f64>>,
    /// Fitted coefficients (device-resident), `None` until `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (device-resident, length 1), `None` until `fit`. Exactly
    /// `0.0` when `fit_intercept` is false, mirroring sklearn.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted scale `σ` (sklearn `scale_`).
    scale_: f64,
    /// L-BFGS iterations performed, capped at `max_iter` — sklearn's
    /// `_check_optimize_result` reports `min(nit, max_iter)`.
    n_iter_: usize,
    /// Whether the solve reached its stopping criterion inside `max_iter`.
    /// sklearn only WARNS when this is false (`ConvergenceWarning`), so mlrs
    /// surfaces it rather than erroring — see the convergence policy in
    /// [`HuberRegressor::solve`].
    converged_: bool,
    /// `|yᵢ − xᵢ·coef_ − intercept_| > scale_·epsilon` at the fitted parameters
    /// (sklearn `outliers_`), length `n_samples`.
    outliers_: Vec<bool>,
    /// The packed `[coef…, intercept?, σ]` the next warm start would seed from.
    params_: Vec<f64>,
    /// Memoized host copy of `(coef_, intercept_)` for the host-ingress
    /// `predict` path (the IN-05 `OnceLock` mirror idiom, shared with the other
    /// dense linear regressors). Fresh on every `fit`.
    predict_mirror: HostMirror<F>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> HuberRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `HuberRegressor` with sklearn's defaults (D-03).
    pub fn builder() -> HuberRegressorBuilder {
        HuberRegressorBuilder::default()
    }

    /// The outlier cut-off on the scaled residual.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
    /// The L-BFGS iteration cap.
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }
    /// The `α·‖w‖²` penalty strength.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    /// Whether a `fit` seeds itself from [`HuberRegressorBuilder::init_params`].
    pub fn warm_start(&self) -> bool {
        self.warm_start
    }
    /// Whether an intercept is fitted.
    pub fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }
    /// The L-BFGS gradient tolerance.
    pub fn tol(&self) -> f64 {
        self.tol
    }
}

impl<F> HuberRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `coef_` (length `n_features`). `Some` by
    /// construction on the `Fitted` state (D-03).
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.coef_
            .as_ref()
            .expect("coef_ is Some by construction on HuberRegressor<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `intercept_` (scalar). `Some` by construction on
    /// the `Fitted` state (D-03).
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> F {
        self.intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on HuberRegressor<F, Fitted>")
            .to_host(pool)[0]
    }

    /// The fitted scale `σ` (sklearn `scale_`), always in `f64`: the whole solve
    /// runs in `f64` whatever the design's width, and rounding the scale to `f32`
    /// would hide real drift in the joint `(w, σ)` iteration behind the storage
    /// format (the `bayesian_ridge.rs` precision precedent).
    pub fn scale(&self) -> f64 {
        self.scale_
    }

    /// L-BFGS iterations performed, capped at `max_iter` (sklearn `n_iter_`).
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Whether the solve met its stopping criterion within `max_iter`. False is
    /// sklearn's `ConvergenceWarning` case, not an error (module docs).
    pub fn converged(&self) -> bool {
        self.converged_
    }

    /// `|yᵢ − xᵢ·coef_ − intercept_| > scale_·epsilon` per training sample
    /// (sklearn `outliers_`).
    pub fn outliers(&self) -> &[bool] {
        &self.outliers_
    }

    /// The packed `[coef…, intercept?, σ]` a warm-started refit seeds from —
    /// feed it back through [`HuberRegressorBuilder::init_params`].
    ///
    /// This is how sklearn's `warm_start` is expressed against a CONSUMING
    /// typestate `fit`: sklearn keeps the previous attributes on the same object
    /// and re-reads them (`np.concatenate((self.coef_, [self.intercept_,
    /// self.scale_]))`), which a `fit` that moves `self` cannot do. Handing the
    /// packed vector back makes the same seed explicit — and the layout here is
    /// byte-for-byte the one sklearn concatenates, so the two starting points
    /// agree exactly.
    pub fn warm_start_params(&self) -> &[f64] {
        &self.params_
    }

    /// `predict` for a test matrix still in the CALLER'S memory — the ingress
    /// the Arrow/PyO3 boundary actually has.
    ///
    /// A Huber prediction is the same `X·coef_ + intercept_` matvec the dense
    /// linear regressors compute, so it shares their [`predict_linear_from_host`]
    /// path verbatim (D-03 — one implementation, not another copy) and inherits
    /// its backend routing and fused operand-finiteness verdict.
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
            "huber_regressor",
            pool,
            x,
            shape,
        )
    }
}

/// Builder for [`HuberRegressor`] (D-01). Default field initializers encode the
/// sklearn `HuberRegressor` defaults (D-03): `epsilon=1.35`, `max_iter=100`,
/// `alpha=0.0001`, `warm_start=False`, `fit_intercept=True`, `tol=1e-5`.
#[derive(Debug, Clone, Default)]
pub struct HuberRegressorBuilder {
    epsilon: Option<f64>,
    max_iter: Option<usize>,
    alpha: Option<f64>,
    warm_start: bool,
    fit_intercept: Option<bool>,
    tol: Option<f64>,
    init_params: Option<Vec<f64>>,
}

impl HuberRegressorBuilder {
    /// Set the outlier cut-off on the scaled residual (sklearn `epsilon >= 1`).
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = Some(epsilon);
        self
    }
    /// Set the L-BFGS iteration cap (sklearn `max_iter >= 0`).
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = Some(max_iter);
        self
    }
    /// Set the `α·‖w‖²` penalty strength (sklearn `alpha >= 0`).
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }
    /// Set whether a `fit` seeds from [`Self::init_params`] (sklearn
    /// `warm_start`). Without an `init_params` seed this flag has no effect, the
    /// same way sklearn's has none until the estimator has a `coef_`.
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }
    /// Set whether to fit an intercept.
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = Some(fit_intercept);
        self
    }
    /// Set the L-BFGS gradient tolerance (sklearn `tol >= 0`).
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = Some(tol);
        self
    }
    /// Seed the next `fit` with a previous fit's
    /// [`warm_start_params`](HuberRegressor::warm_start_params) — the packed
    /// `[coef…, intercept?, σ]`. Only consulted when [`Self::warm_start`] is set.
    pub fn init_params(mut self, params: Vec<f64>) -> Self {
        self.init_params = Some(params);
        self
    }

    /// Build the estimator, validating the data-INDEPENDENT hyperparameters
    /// (D-08) against sklearn's own `_parameter_constraints`:
    /// `epsilon ∈ [1, ∞)` ([`BuildError::InvalidEpsilon`]), `alpha ∈ [0, ∞)`
    /// ([`BuildError::InvalidAlpha`]), `tol ∈ [0, ∞)` ([`BuildError::InvalidTol`]).
    ///
    /// `max_iter` is NOT rejected at `0`: sklearn's constraint is
    /// `Interval(Integral, 0, None, closed="left")`, and `max_iter=0` is a real
    /// (if odd) configuration — scipy still performs one line search and then
    /// reports `n_iter_ = 0`. [`HuberRegressor::solve`] reproduces exactly that.
    pub fn build<F>(self) -> Result<HuberRegressor<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        let epsilon = self.epsilon.unwrap_or(1.35);
        let max_iter = self.max_iter.unwrap_or(100);
        let alpha = self.alpha.unwrap_or(1e-4);
        let fit_intercept = self.fit_intercept.unwrap_or(true);
        let tol = self.tol.unwrap_or(1e-5);

        // sklearn: `Interval(Real, 1.0, None, closed="left")`. Below 1 the
        // "quadratic near zero, linear in the tail" shape inverts and the
        // estimator stops being a robustification of least squares at all, so
        // this is a hard reject rather than a documented divergence.
        if !(epsilon >= 1.0) {
            return Err(BuildError::InvalidEpsilon {
                estimator: "huber_regressor",
                epsilon,
            });
        }
        if !(alpha >= 0.0) {
            return Err(BuildError::InvalidAlpha {
                estimator: "huber_regressor",
                alpha,
            });
        }
        if !(tol >= 0.0) {
            return Err(BuildError::InvalidTol {
                estimator: "huber_regressor",
                tol,
            });
        }

        Ok(HuberRegressor {
            epsilon,
            max_iter,
            alpha,
            warm_start: self.warm_start,
            fit_intercept,
            tol,
            init_params: self.init_params,
            coef_: None,
            intercept_: None,
            scale_: SIGMA_INIT,
            n_iter_: 0,
            converged_: false,
            outliers_: Vec::new(),
            params_: Vec::new(),
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for HuberRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = HuberRegressor<F, Fitted>;

    /// Unweighted `fit` — [`HuberRegressor::fit_with_sample_weight`] with no
    /// weights.
    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<HuberRegressor<F, Fitted>, AlgoError> {
        self.fit_with_sample_weight(pool, x, y, shape, None)
    }
}

impl<F> HuberRegressor<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// `fit` with sklearn's `sample_weight`, over a DEVICE-resident design.
    ///
    /// `sample_weight` is a length-`n_samples` host slice of NON-NEGATIVE finite
    /// weights (rejected as [`AlgoError::InvalidSampleWeight`] otherwise), or
    /// `None` for the unweighted fit. The weights enter EVERY term of the
    /// objective — including the `σ·Σ swᵢ` term and the `σ` derivative — so a
    /// weighted fit is not a reweighting of an unweighted solution.
    pub fn fit_with_sample_weight(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<HuberRegressor<F, Fitted>, AlgoError> {
        let (n_samples, _) = shape;

        // --- Data-DEPENDENT geometry guard BEFORE any launch (the
        //     data-INDEPENDENT params were validated at build(), D-08). ---
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "huber_regressor",
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
        let y_host = y.to_host(pool);
        self.fit_core(pool, HuberDesign::Device(x), &y_host, shape, sample_weight)
    }

    /// [`Fit::fit`] over HOST slices — the no-upload ingress the Arrow/PyO3
    /// boundary actually has.
    ///
    /// Same reasoning as `LinearSVC::fit_from_host_slice`: the cpu objective
    /// reads the design from host memory, so uploading it first costs three full
    /// passes over `n·d` elements (`from_host` copies twice, `to_host` once) for
    /// nothing. The solve is bit-identical to [`Fit::fit`]'s.
    pub fn fit_from_host_slice(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        y: &[F],
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<HuberRegressor<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // The slice twin of the D-08 geometry guard: `validate_geometry` reads a
        // DeviceArray's length, which we do not have here.
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
        self.fit_core(pool, HuberDesign::Host(x), y, shape, sample_weight)
    }

    /// The solve itself, shared by both ingresses (D-03).
    fn fit_core(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        design: HuberDesign<'_, F>,
        y_host: &[F],
        shape: (usize, usize),
        sample_weight: Option<&[F]>,
    ) -> Result<HuberRegressor<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        let sw64 = validate_sample_weight::<F>("huber_regressor", sample_weight, n_samples)?;
        let targets: Vec<f64> = y_host.iter().map(|&v| host_to_f64(v)).collect();

        // The evaluator owns the design in whatever form this backend wants and
        // (on cpu) the worker pool, for the LIFETIME of the solve — not per
        // iteration.
        let objective = HuberObjective::<F>::new(
            pool,
            design,
            (n_samples, n_features),
            targets,
            sw64,
            self.fit_intercept,
        )
        .map_err(AlgoError::Prim)?;

        let outcome = match self.solve(&objective, pool) {
            Ok(o) => o,
            Err(e) => {
                objective.release_into(pool);
                return Err(e);
            }
        };

        let d_aug = objective.d_aug();
        let sigma = outcome.params[d_aug];
        let outliers = match objective.outlier_mask(pool, &outcome.params[..d_aug], sigma, self.epsilon)
        {
            Ok(m) => m,
            Err(e) => {
                objective.release_into(pool);
                return Err(AlgoError::Prim(e));
            }
        };
        objective.release_into(pool);

        let coef_dev = upload_coef::<F>(pool, &outcome.params[..n_features]);
        let intercept_dev = upload_coef::<F>(
            pool,
            &[if self.fit_intercept {
                outcome.params[n_features]
            } else {
                0.0
            }],
        );

        Ok(HuberRegressor {
            epsilon: self.epsilon,
            max_iter: self.max_iter,
            alpha: self.alpha,
            warm_start: self.warm_start,
            fit_intercept: self.fit_intercept,
            tol: self.tol,
            init_params: None,
            coef_: Some(coef_dev),
            intercept_: Some(intercept_dev),
            scale_: sigma,
            n_iter_: outcome.n_iter,
            converged_: outcome.converged,
            outliers_: outliers,
            params_: outcome.params,
            predict_mirror: HostMirror::new(),
            _state: PhantomData,
        })
    }

    /// The L-BFGS drive over `p = [w, c?, σ]`, returning the final iterate plus
    /// sklearn's `n_iter_` and the convergence verdict.
    ///
    /// ## The parameter layout is sklearn's, verbatim
    /// `p[..n_features]` are the coefficients, `p[n_features]` is the intercept
    /// when one is fitted, and `p[d_aug]` — the LAST entry either way — is `σ`.
    /// Keeping that order is what lets [`warm_start_params`](HuberRegressor::warm_start_params)
    /// be handed straight back and lets a cross-check against sklearn's
    /// `_huber_loss_and_gradient` compare gradients entry by entry.
    ///
    /// ## Convergence policy
    /// sklearn raises only on scipy's `status == 2` (an abnormal line-search
    /// termination) and merely WARNS when the iteration cap is hit, so mlrs
    /// mirrors that split: a cap or an `ftol` plateau returns the iterate with
    /// `converged = false`, and only a line-search breakdown that is ALSO not at
    /// the dtype's resolvable-gradient floor becomes [`AlgoError::NotConverged`].
    /// The floor accept matters because `ftol = 64·eps` deliberately drives the
    /// solve to the point where the objective stops changing in floating point,
    /// which is exactly where a strong-Wolfe search can no longer find a
    /// decreasing step.
    fn solve(
        &self,
        objective: &HuberObjective<'_, F>,
        pool: &mut BufferPool<ActiveRuntime>,
    ) -> Result<SolveOutcome, AlgoError> {
        let d_aug = objective.d_aug();
        let n_params = d_aug + 1;
        let sw_total = objective.sw_total();
        let epsilon = self.epsilon;
        let alpha = self.alpha;
        // The penalized block is the COEFFICIENTS only, so it stops one short of
        // `d_aug` when an intercept is fitted.
        let n_features = if self.fit_intercept { d_aug - 1 } else { d_aug };

        // sklearn's cold start: all-zero weights with `parameters[-1] = 1`. A
        // warm start replaces it with the previous fit's packed parameters,
        // which is the same vector sklearn concatenates.
        let mut x0 = vec![0.0f64; n_params];
        x0[d_aug] = SIGMA_INIT;
        if self.warm_start {
            if let Some(seed) = self.init_params.as_ref() {
                if seed.len() != n_params {
                    return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                        operand: "warm_start init_params",
                        rows: n_params,
                        cols: 1,
                        len: seed.len(),
                    }));
                }
                x0.copy_from_slice(seed);
                // A seed whose scale is already infeasible would put the very
                // first evaluation on the barrier and abort the solve; clamp it
                // back into the box scipy's `bounds` would have projected it to.
                if !(x0[d_aug] >= SIGMA_FLOOR) {
                    x0[d_aug] = SIGMA_FLOOR;
                }
            }
        }

        // The closure is called once per L-BFGS iteration and once per
        // line-search step. A prim failure is CAPTURED (never panics across the
        // boundary) and surfaced after the solve.
        let mut prim_err: Option<PrimError> = None;
        // `MLRS_HUBER_PROBE=1` prints the solve's SHAPE: objective evaluations,
        // outer iterations, stop reason, residual gradient and wall time.
        //
        // The evaluation count is the number that matters and the one `n_iter_`
        // hides: an outer iteration costs one evaluation plus however many the
        // strong-Wolfe line search needed, and those two failure modes call for
        // opposite fixes ("the pass got slower" vs "the solver took more
        // steps"). Two designs of the SAME geometry that both report 16
        // iterations were measured 3x apart on wall-clock purely because one
        // needed far more line-search steps. Through `abflag` rather than
        // `std::env` so a test can force it without an environ data race (the
        // `MLRS_SVM_PROBE` precedent).
        let mut probe_evals: usize = 0;
        let probe_t0 = std::time::Instant::now();
        let shape = LossShape {
            d_aug,
            n_features,
            sw_total,
            epsilon,
            alpha,
        };

        // `max_iter = 0` is legal in sklearn and does NOT mean "no work":
        // scipy's L-BFGS-B checks the cap AFTER the first iteration, so it still
        // performs one line search and then reports `n_iter_ = 0`. Passing 0 to
        // `lbfgs_minimize` would instead mean "use the 100 default", so the cap
        // is floored at 1 for the solver and the REPORTED count is clamped back
        // to `max_iter` below — which reproduces both halves of that quirk.
        let solver_maxiter = self.max_iter.max(1);
        // `tol = 0` is legal in sklearn (`Interval(Real, 0, …)`) and simply means
        // the gradient test can never fire, leaving the `ftol` plateau as the
        // only stop. `lbfgs_minimize` treats gtol the same way, so it is passed
        // through unclamped.
        let mut result = {
            let joint = |p: &[f64]| -> (f64, Vec<f64>) {
                probe_evals += 1;
                if prim_err.is_some() {
                    return (f64::MAX, vec![0.0f64; n_params]);
                }
                let sigma = p[d_aug];
                // The `σ > 0` barrier (module docs). `+∞` fails the Armijo test,
                // so the line search zooms back into the feasible bracket; the
                // gradient returned here is never read on that path, and
                // returning zeros keeps `dot(g, direction)` finite so no NaN can
                // leak into the bracket bookkeeping.
                if !(sigma >= SIGMA_FLOOR) {
                    return (f64::INFINITY, vec![0.0f64; n_params]);
                }
                let ev = match objective.eval(pool, &p[..d_aug], sigma, epsilon) {
                    Ok(ev) => ev,
                    Err(e) => {
                        prim_err = Some(e);
                        return (f64::MAX, vec![0.0f64; n_params]);
                    }
                };
                let mut parts = assemble(ev, &p[..d_aug], sigma, &shape);
                parts.grad.push(parts.dsigma);
                (parts.loss, parts.grad)
            };
            lbfgs_minimize(x0, joint, self.tol, HUBER_FTOL, LBFGS_MAXLS, solver_maxiter)?
        };
        if let Some(e) = prim_err {
            return Err(AlgoError::Prim(e));
        }

        // The resolvable-gradient floor near a flat minimum: `k·sqrt(eps_F)`
        // scaled by the number of weighted terms summed into each entry.
        let floor_accept = GRAD_FLOOR_K * f_epsilon::<F>().sqrt() * sw_total.max(1.0);

        // --- Fixed-σ refinement: what scipy's BOUND gets for free ------------
        //
        // scipy solves this as an L-BFGS-B problem whose only bound is `σ ≥
        // 10·eps`, so when the search drives σ onto (or numerically into) that
        // bound it PROJECTS and keeps optimizing the remaining free variables
        // along it. `lbfgs_minimize` is unbounded and its strong-Wolfe search
        // instead breaks down there — it cannot find a decreasing step in a
        // direction most of whose length lies outside the feasible set.
        //
        // The fix is the same active-set move, done explicitly: when the joint
        // search breaks down away from the gradient floor, PIN σ at the value it
        // reached and re-solve over `w` alone. That is a strictly smaller
        // problem started from the current iterate, so it can only lower the
        // objective, and it is the block that is still identifiable — the
        // configuration that provokes this (`epsilon = 1`, where σ cancels out
        // of the objective exactly and the whole σ ray is flat) is precisely one
        // where σ carries no information and `w` still does.
        //
        // On a well-conditioned fit the joint search converges or stalls on
        // `ftol`, never on a breakdown, so this never runs.
        let breakdown = result.stop_reason == LbfgsStopReason::LineSearchFailed
            && result.max_grad > floor_accept;
        if breakdown && result.iters > 1 {
            let sigma = result.x[d_aug];
            let remaining = solver_maxiter.saturating_sub(result.iters).max(1);
            let w0 = result.x[..d_aug].to_vec();
            let refined = {
                let fixed = |w: &[f64]| -> (f64, Vec<f64>) {
                    probe_evals += 1;
                    if prim_err.is_some() {
                        return (f64::MAX, vec![0.0f64; d_aug]);
                    }
                    let ev = match objective.eval(pool, w, sigma, epsilon) {
                        Ok(ev) => ev,
                        Err(e) => {
                            prim_err = Some(e);
                            return (f64::MAX, vec![0.0f64; d_aug]);
                        }
                    };
                    let parts = assemble(ev, w, sigma, &shape);
                    (parts.loss, parts.grad)
                };
                lbfgs_minimize(w0, fixed, self.tol, HUBER_FTOL, LBFGS_MAXLS, remaining)?
            };
            if let Some(e) = prim_err {
                return Err(AlgoError::Prim(e));
            }
            // Only adopt a refinement that actually improved: L-BFGS is a
            // descent method, so this holds by construction, but the objective
            // is re-summed here and the guard costs one comparison.
            if refined.loss < result.loss {
                result.x[..d_aug].copy_from_slice(&refined.x);
                result.loss = refined.loss;
                result.max_grad = refined.max_grad;
                result.iters += refined.iters;
                result.converged = refined.converged;
                result.stop_reason = refined.stop_reason;
            }
        }

        let at_floor = result.max_grad <= floor_accept;
        // A line-search breakdown is only an ERROR when NO step was ever
        // accepted: then there is nothing to return but the starting point, and
        // sklearn's scipy `status == 2` (which it raises on) is the same
        // condition. After at least one accepted step — and after the fixed-σ
        // refinement above has had its turn — the breakdown means the search can
        // no longer find a decreasing point, which is what `ftol = 64·eps`
        // deliberately drives towards. Reporting `converged = false` and handing
        // the iterate back is what sklearn does there, so mlrs does too.
        if result.stop_reason == LbfgsStopReason::LineSearchFailed && !at_floor && result.iters <= 1
        {
            return Err(AlgoError::NotConverged {
                estimator: "huber_regressor",
                max_iter: self.max_iter,
            });
        }

        if mlrs_backend::abflag::is_on("MLRS_HUBER_PROBE") {
            eprintln!(
                "[huber probe] n={} d_aug={d_aug} evals={probe_evals} iters={} \
                 stop={:?} max_grad={:.3e} loss={:.9e} solve={:.2}ms",
                objective.n(),
                result.iters,
                result.stop_reason,
                result.max_grad,
                result.loss,
                probe_t0.elapsed().as_secs_f64() * 1e3,
            );
        }

        // What counts as CONVERGED, and why `FtolStall` does.
        //
        // scipy's L-BFGS-B reports `CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`
        // as SUCCESS (`status == 0`) — it is the criterion sklearn's Huber
        // actually stops on for essentially every fit, since it leaves `factr`
        // at `1e7`. mlrs stops on the same criterion, only at `64·eps` instead
        // of `1e7·eps`, which is a STRICTLY tighter place to stop. Treating that
        // as non-convergence would make the shim emit a `ConvergenceWarning` on
        // every default fit while sitting closer to the optimum than the
        // reference — the warning would be exactly backwards.
        //
        // What remains non-converged is the honest set: the `max_iter` cap
        // (sklearn's own `ConvergenceWarning` case) and a line-search breakdown
        // that is not at the resolvable-gradient floor.
        let stopped_cleanly = result.converged
            || at_floor
            || result.stop_reason == LbfgsStopReason::FtolStall;

        Ok(SolveOutcome {
            n_iter: result.iters.min(self.max_iter),
            converged: stopped_cleanly,
            params: result.x,
        })
    }
}

/// The scalars the loss/gradient assembly needs beyond one [`HuberEval`].
struct LossShape {
    /// Augmented weight length (`n_features + 1` when fitting an intercept).
    d_aug: usize,
    /// Penalized block width — the coefficients, never the intercept.
    n_features: usize,
    /// `Σ swᵢ` (sklearn's `n_samples`).
    sw_total: f64,
    epsilon: f64,
    alpha: f64,
}

/// One assembled objective value with its `w` gradient and the separate `σ`
/// derivative.
struct LossParts {
    loss: f64,
    /// `∂L/∂w`, length `d_aug`.
    grad: Vec<f64>,
    /// `∂L/∂σ`. Dropped by the fixed-σ refinement, appended to `grad` by the
    /// joint solve.
    dsigma: f64,
}

/// sklearn's `_huber_loss_and_gradient`, assembled from the five reductions ONE
/// pass over the design produced.
///
/// Written once and shared by both solves so the joint search and the fixed-σ
/// refinement cannot drift apart — the refinement optimizes the SAME objective,
/// only over fewer variables.
fn assemble(ev: HuberEval, w: &[f64], sigma: f64, shape: &LossShape) -> LossParts {
    let LossShape {
        d_aug,
        n_features,
        sw_total,
        epsilon,
        alpha,
    } = *shape;
    let squared_loss = ev.sq_sum / sigma;
    let outlier_loss = 2.0 * epsilon * ev.out_abs_sum - sigma * ev.out_sw_sum * epsilon * epsilon;
    // The penalty covers the COEFFICIENTS only — never the intercept and never σ
    // (sklearn slices `w[:n_features]` before `np.dot(w, w)`).
    let mut penalty = 0.0f64;
    for &wv in &w[..n_features] {
        penalty += wv * wv;
    }
    let loss = sw_total * sigma + squared_loss + outlier_loss + alpha * penalty;

    let mut grad = ev.xtg;
    debug_assert_eq!(grad.len(), d_aug);
    for (gj, &wv) in grad[..n_features].iter_mut().zip(&w[..n_features]) {
        *gj += 2.0 * alpha * wv;
    }
    LossParts {
        loss,
        grad,
        // ∂L/∂σ = Σ swᵢ − ε²·Σ_{outlier} swᵢ − squared_loss/σ.
        dsigma: sw_total - ev.out_sw_sum * epsilon * epsilon - squared_loss / sigma,
    }
}

/// What [`HuberRegressor::solve`] hands back.
struct SolveOutcome {
    /// L-BFGS iterations, already clamped to `max_iter` (sklearn `n_iter_`).
    n_iter: usize,
    /// Whether a stopping criterion fired inside the cap.
    converged: bool,
    /// The final iterate `[w, c?, σ]`.
    params: Vec<f64>,
}

/// Machine epsilon of `F` (f32 / f64) as `f64`, for the flat-minimum residual
/// precision floor (the `linear_svc.rs` helper).
fn f_epsilon<F: Pod>() -> f64 {
    match size_of::<F>() {
        4 => f32::EPSILON as f64,
        8 => f64::EPSILON,
        _ => unreachable!("huber_regressor is f32/f64 only"),
    }
}

impl<F> Predict<F> for HuberRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        predict_linear(
            self.coef_.as_ref(),
            self.intercept_.as_ref(),
            "huber_regressor",
            pool,
            x,
            shape,
        )
    }
}
