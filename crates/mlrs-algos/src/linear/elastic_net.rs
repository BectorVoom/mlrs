//! `ElasticNet` (LINEAR-04) — L1+L2-penalized least squares via the shared
//! coordinate-descent solver (D-03), matching
//! `sklearn.linear_model.ElasticNet`.
//!
//! ## Solver (deliberately coordinate descent — the iterative-solver family)
//! ElasticNet minimizes
//! `½‖y − Xβ‖² + α·l1_ratio·n·‖β‖₁ + ½·α·(1−l1_ratio)·n·‖β‖₂²`
//! by cyclic coordinate descent on the CENTERED design via the validated 05-05
//! [`cd_solve`](mlrs_backend::prims::coordinate_descent::cd_solve) primitive,
//! driven by the shared [`cd_fit`] host helper. It is NOT a Cholesky / SVD
//! normal-equations solve (those are Ridge / LinearRegression) nor the L-BFGS
//! LogReg optimizer (05-10) — the three families use deliberately different
//! solvers and must not be unified (see `linear/mod.rs`).
//!
//! ## Lasso is the `l1_ratio == 1` case (D-03)
//! [`Lasso`](crate::linear::lasso::Lasso) is a thin wrapper that delegates to the
//! same [`cd_fit`] with `l1_ratio = 1.0` (→ `l2_reg = 0`, pure L1). Both
//! estimators share one coordinate-descent implementation.
//!
//! ## Penalty mapping + center-then-solve intercept
//! `cd_fit` maps the user-facing `(alpha, l1_ratio)` to sklearn's un-normalized
//! `(l1_reg = α·l1_ratio·n, l2_reg = α·(1−l1_ratio)·n)` (Pitfall 1), centers
//! `(X, y)` when `fit_intercept` (D-13), and recovers the unpenalized
//! `intercept_ = ȳ − x̄·coef_` — reproducing sklearn's `coef_`/`intercept_` within
//! 1e-5 INCLUDING the exact sparsity (zero) pattern.
//!
//! ## Device residency (D-03)
//! Fitted `coef_` (length `n_features`) and `intercept_` (length 1) are stored as
//! device-resident [`DeviceArray`]s; `predict` runs the `X_test · coef_` GEMM
//! on-device and broadcasts the intercept (the `ridge.rs` predict path),
//! materializing to the host only at a Rust accessor / oracle boundary.
//!
//! Tests live in `crates/mlrs-algos/tests/elastic_net_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::linear_predict::linear_predict;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::linear::coordinate_descent::{cd_fit, CD_DEFAULT_MAX_ITER, CD_DEFAULT_TOL};
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};

/// L1+L2-penalized least squares (LINEAR-04) fitted by the shared
/// coordinate-descent solver.
///
/// Construct with the zero-arg [`ElasticNet::new`] (sklearn defaults:
/// `alpha = 1.0`, `l1_ratio = 0.5`, `fit_intercept = true`, `max_iter = 1000`,
/// `tol = 1e-4`) or [`ElasticNet::builder`] (which subsumes the former
/// `new`/`with_opts` constructors — every hyperparameter is a builder setter),
/// then the consuming [`Fit::fit`] (returns the `Fitted`-tagged sibling) and
/// [`Predict::predict`]. Fitted `coef_`/`intercept_` are device-resident (D-03);
/// the host accessors [`coef`](ElasticNet::coef) /
/// [`intercept`](ElasticNet::intercept) materialize them on demand and exist ONLY
/// on `ElasticNet<F, Fitted>` (the compile-time typestate replaces the old
/// runtime `NotFitted` guard, D-03).
pub struct ElasticNet<F, S = Unfit> {
    /// Overall penalty strength (`alpha ≥ 0`; `alpha = 0` degenerates to OLS).
    /// Validated at `build()` → [`BuildError::InvalidAlpha`] (T-05-09-01).
    alpha: F,
    /// L1/L2 mixing parameter (`0 ≤ l1_ratio ≤ 1`; `1` ⇒ Lasso, `0` ⇒ Ridge-like
    /// pure L2). Validated at `build()` → [`BuildError::InvalidL1Ratio`]
    /// (T-05-09-01).
    l1_ratio: F,
    /// Whether to center `X`/`y` and recover a bias term (D-13).
    fit_intercept: bool,
    /// Coordinate-descent iteration cap (sklearn default 1000).
    max_iter: usize,
    /// Coordinate-descent stopping tolerance (sklearn default 1e-4).
    tol: f64,
    /// Fitted coefficients (length `n_features`), device-resident, `None` until
    /// `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (length 1), device-resident, `None` until `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> ElasticNet<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an `ElasticNet` with sklearn's defaults (`alpha = 1.0`,
    /// `l1_ratio = 0.5`, `fit_intercept = true`, `max_iter = 1000`, `tol = 1e-4`)
    /// directly in the `Unfit` state. This is the SINGLE source of truth for the
    /// default hyperparameters (D-08): the builder `Default` re-derives from here
    /// via [`ElasticNet::into_builder`]. Defaults are trusted valid, so this
    /// bypasses [`ElasticNetBuilder::build`]'s validation.
    pub fn new() -> Self {
        Self {
            alpha: F::from_int(1),
            l1_ratio: f64_to_host::<F>(0.5),
            fit_intercept: true,
            max_iter: CD_DEFAULT_MAX_ITER,
            tol: CD_DEFAULT_TOL,
            coef_: None,
            intercept_: None,
            _state: PhantomData,
        }
    }

    /// Start building an `ElasticNet` from sklearn's defaults (D-08 single source).
    pub fn builder() -> ElasticNetBuilder {
        ElasticNetBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`ElasticNetBuilder::default`] to re-derive the
    /// defaults from [`ElasticNet::new`] (D-08).
    pub fn into_builder(self) -> ElasticNetBuilder {
        ElasticNetBuilder {
            alpha: host_to_f64(self.alpha),
            l1_ratio: host_to_f64(self.l1_ratio),
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `coef_`/`intercept_` fields are excluded — both are `None` in any `Unfit`
    /// value). Used by the defaults-equality test (BLDR-01):
    /// `ElasticNet::new().hyperparams_eq(&ElasticNet::builder().build()?)`.
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        host_to_f64(self.alpha) == host_to_f64(other.alpha)
            && host_to_f64(self.l1_ratio) == host_to_f64(other.l1_ratio)
            && self.fit_intercept == other.fit_intercept
            && self.max_iter == other.max_iter
            && self.tol == other.tol
    }
}

impl<F> Default for ElasticNet<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`ElasticNet`] (D-01). It subsumes BOTH the former `new(alpha,
/// l1_ratio, fit_intercept)` AND `with_opts(alpha, l1_ratio, fit_intercept,
/// max_iter, tol)` constructors — every hyperparameter is a setter. Setters are
/// `f64`/`usize` per the A5 convention; `build::<F>()` narrows `alpha`/`l1_ratio`
/// to the target float `F`. `Default` re-derives the sklearn defaults from
/// [`ElasticNet::new`] (D-08 single source) rather than holding literals
/// (Pitfall 1).
#[derive(Debug, Clone, Copy)]
pub struct ElasticNetBuilder {
    alpha: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
}

impl Default for ElasticNetBuilder {
    /// Re-derive the sklearn defaults from [`ElasticNet::new`] (D-08 single
    /// source).
    fn default() -> Self {
        ElasticNet::<f64, Unfit>::new().into_builder()
    }
}

impl ElasticNetBuilder {
    /// Set the overall penalty strength `alpha` (A5: `f64` setter).
    pub fn alpha(mut self, v: f64) -> Self {
        self.alpha = v;
        self
    }

    /// Set the L1/L2 mixing parameter `l1_ratio` (A5: `f64` setter).
    pub fn l1_ratio(mut self, v: f64) -> Self {
        self.l1_ratio = v;
        self
    }

    /// Set whether to center `X`/`y` and recover a bias term.
    pub fn fit_intercept(mut self, v: bool) -> Self {
        self.fit_intercept = v;
        self
    }

    /// Set the coordinate-descent iteration cap (sklearn `max_iter`).
    pub fn max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }

    /// Set the coordinate-descent stopping tolerance (sklearn `tol`).
    pub fn tol(mut self, v: f64) -> Self {
        self.tol = v;
        self
    }

    /// Build the (unfit) estimator, validating the data-INDEPENDENT hyperparameters
    /// BEFORE any data is seen (relocated from the old `cd_fit` fit-body checks,
    /// Pitfall 7; the data-DEPENDENT geometry check stays in [`Fit::fit`]):
    ///
    /// - `alpha >= 0` ([`BuildError::InvalidAlpha`]).
    /// - `0 <= l1_ratio <= 1` ([`BuildError::InvalidL1Ratio`]).
    ///
    /// The stored `f64` `alpha`/`l1_ratio` are narrowed to the target float `F`
    /// (A5).
    pub fn build<F>(self) -> Result<ElasticNet<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if !(self.alpha >= 0.0) {
            return Err(BuildError::InvalidAlpha {
                estimator: "elastic_net",
                alpha: self.alpha,
            });
        }
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            return Err(BuildError::InvalidL1Ratio {
                estimator: "elastic_net",
                l1_ratio: self.l1_ratio,
            });
        }
        Ok(ElasticNet {
            alpha: f64_to_host::<F>(self.alpha),
            l1_ratio: f64_to_host::<F>(self.l1_ratio),
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
            coef_: None,
            intercept_: None,
            _state: PhantomData,
        })
    }
}

impl<F> ElasticNet<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `coef_` (length `n_features`). `Some` by
    /// construction on the `Fitted` state (D-03).
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.coef_
            .as_ref()
            .expect("coef_ is Some by construction on ElasticNet<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `intercept_` (scalar). `Some` by construction on
    /// the `Fitted` state (D-03).
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> F {
        self.intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on ElasticNet<F, Fitted>")
            .to_host(pool)[0]
    }
}

impl<F> Fit<F> for ElasticNet<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = ElasticNet<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<ElasticNet<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // Data-DEPENDENT geometry guard BEFORE any prim launch (the
        // data-INDEPENDENT `alpha >= 0` / `l1_ratio ∈ [0, 1]` checks were validated
        // at build() — Pitfall 7).
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "elastic_net",
            operation: "fit (requires y)",
        })?;

        // Delegate to the shared CD helper (penalty map + centering + cd_solve +
        // intercept recovery). cd_fit validates alpha/l1_ratio/geometry BEFORE any
        // launch (T-05-09-01).
        let (coef, intercept) = cd_fit::<F>(
            pool,
            x,
            y,
            n_samples,
            n_features,
            host_to_f64(self.alpha),
            host_to_f64(self.l1_ratio),
            self.fit_intercept,
            self.tol,
            self.max_iter,
            "elastic_net",
        )?;

        Ok(ElasticNet {
            alpha: self.alpha,
            l1_ratio: self.l1_ratio,
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
            coef_: Some(coef),
            intercept_: Some(intercept),
            _state: PhantomData,
        })
    }
}

impl<F> Predict<F> for ElasticNet<F, Fitted>
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
            "elastic_net",
            pool,
            x,
            shape,
        )
    }
}

/// Shared `X·coef_ + intercept_` prediction path for the coordinate-descent
/// linear models (the `ridge.rs` GEMM-then-broadcast precedent). Used by both
/// [`ElasticNet`] and [`Lasso`](crate::linear::lasso::Lasso) so the predict
/// surface is implemented once (D-03). Errors with [`AlgoError::NotFitted`] when
/// called before `fit` and [`PrimError::ShapeMismatch`] / [`PrimError::DimMismatch`]
/// on a geometry / `n_features` disagreement (ASVS V5).
pub(crate) fn predict_linear<F>(
    coef_: Option<&DeviceArray<ActiveRuntime, F>>,
    intercept_: Option<&DeviceArray<ActiveRuntime, F>>,
    estimator: &'static str,
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let (n_samples, n_features) = shape;

    let coef = coef_.ok_or(AlgoError::NotFitted {
        estimator,
        operation: "predict",
    })?;
    let intercept = intercept_.ok_or(AlgoError::NotFitted {
        estimator,
        operation: "predict",
    })?;

    // --- ASVS V5: geometry + fitted-n_features consistency. ---
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

    // y_pred = X_test · coef + intercept via ONE fused device launch (the
    // LINEAR-01/02 predict perf lever, shared by ElasticNet + Lasso): the
    // `linear_predict` prim's GATHER matvec+bias kernel replaces the prior
    // gemm→`intercept.to_host()`→`raw.to_host()`→host bias-loop→`from_host`
    // round-trips (the `center`/`gram` host-sync pathology, same class of fix).
    // The result stays device-resident; the PyO3 boundary's terminal readback
    // is the only host↔device crossing.
    Ok(linear_predict::<F>(
        pool,
        x,
        coef,
        intercept,
        (n_samples, n_features),
    )?)
}
