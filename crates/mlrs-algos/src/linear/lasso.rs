//! `Lasso` (LINEAR-03) — L1-penalized least squares, the `l1_ratio == 1` case of
//! [`ElasticNet`](crate::linear::elastic_net::ElasticNet) (D-03), matching
//! `sklearn.linear_model.Lasso`.
//!
//! ## Thin wrapper over the shared coordinate-descent solver (D-03)
//! Lasso is exactly ElasticNet with `l1_ratio = 1.0` (→ `l2_reg = 0`, pure L1):
//! it delegates to the SAME [`cd_fit`](crate::linear::coordinate_descent::cd_fit)
//! host helper with `l1_ratio` pinned to 1, so the coordinate-descent loop, the
//! penalty mapping (`l1_reg = α·n`), the center-then-solve intercept (D-13), and
//! the device-side duality-gap stop are NOT re-implemented here. This is the
//! deliberate shared CD path — it is NOT the L-BFGS `LogisticRegression` solver
//! (05-10) and must not be unified with it (see `linear/mod.rs`).
//!
//! ## Sparse `coef_` (Pitfall 1)
//! With pure L1 the soft-threshold drives sub-threshold coordinates to EXACT
//! zero; Lasso reproduces sklearn's sparse `coef_` within 1e-5 INCLUDING the
//! exact zero/sparsity pattern, and the unpenalized `intercept_ = ȳ − x̄·coef_`.
//!
//! ## Device residency (D-03)
//! Fitted `coef_` / `intercept_` are device-resident; `predict` reuses the shared
//! [`predict_linear`](crate::linear::elastic_net::predict_linear) GEMM path
//! (identical to ElasticNet / Ridge).
//!
//! Tests live in `crates/mlrs-algos/tests/lasso_test.rs` (AGENTS.md §2), never an
//! in-source `#[cfg(test)] mod tests`.

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64};

use crate::error::{AlgoError, BuildError};
use crate::linear::coordinate_descent::{cd_fit, CD_DEFAULT_MAX_ITER, CD_DEFAULT_TOL};
use crate::linear::elastic_net::predict_linear;
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};

/// L1-penalized least squares (LINEAR-03) — the `l1_ratio == 1` case of
/// [`ElasticNet`](crate::linear::elastic_net::ElasticNet), fitted by the shared
/// coordinate-descent solver.
///
/// Construct with the zero-arg [`Lasso::new`] (sklearn defaults: `alpha = 1.0`,
/// `fit_intercept = true`, `max_iter = 1000`, `tol = 1e-4`) or [`Lasso::builder`]
/// (which subsumes the former `new`/`with_opts` constructors — every
/// hyperparameter is a builder setter), then the consuming [`Fit::fit`] (returns
/// the `Fitted`-tagged sibling) and [`Predict::predict`]. Fitted
/// `coef_`/`intercept_` are device-resident (D-03); the host accessors
/// [`coef`](Lasso::coef) / [`intercept`](Lasso::intercept) materialize them on
/// demand and exist ONLY on `Lasso<F, Fitted>` (the compile-time typestate
/// replaces the old runtime `NotFitted` guard, D-03).
pub struct Lasso<F, S = Unfit> {
    /// L1 penalty strength (`alpha ≥ 0`; `alpha = 0` degenerates to OLS).
    /// Validated at `build()` → [`BuildError::InvalidAlpha`] (T-05-09-01).
    alpha: F,
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

impl<F> Lasso<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct a `Lasso` with sklearn's defaults (`alpha = 1.0`,
    /// `fit_intercept = true`, `max_iter = 1000`, `tol = 1e-4`) directly in the
    /// `Unfit` state. This is the SINGLE source of truth for the default
    /// hyperparameters (D-08): the builder `Default` re-derives from here via
    /// [`Lasso::into_builder`]. Defaults are trusted valid, so this bypasses
    /// [`LassoBuilder::build`]'s validation.
    pub fn new() -> Self {
        Self {
            alpha: F::from_int(1),
            fit_intercept: true,
            max_iter: CD_DEFAULT_MAX_ITER,
            tol: CD_DEFAULT_TOL,
            coef_: None,
            intercept_: None,
            _state: PhantomData,
        }
    }

    /// Start building a `Lasso` from sklearn's defaults (D-08 single source).
    pub fn builder() -> LassoBuilder {
        LassoBuilder::default()
    }

    /// Decompose this (unfit) estimator back into its builder, copying every
    /// hyperparameter. Used by [`LassoBuilder::default`] to re-derive the defaults
    /// from [`Lasso::new`] (D-08).
    pub fn into_builder(self) -> LassoBuilder {
        LassoBuilder {
            alpha: host_to_f64(self.alpha),
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `coef_`/`intercept_` fields are excluded — both are `None` in any `Unfit`
    /// value). Used by the defaults-equality test (BLDR-01):
    /// `Lasso::new().hyperparams_eq(&Lasso::builder().build()?)`.
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        host_to_f64(self.alpha) == host_to_f64(other.alpha)
            && self.fit_intercept == other.fit_intercept
            && self.max_iter == other.max_iter
            && self.tol == other.tol
    }
}

impl<F> Default for Lasso<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`Lasso`] (D-01). It subsumes BOTH the former `new(alpha,
/// fit_intercept)` AND `with_opts(alpha, fit_intercept, max_iter, tol)`
/// constructors — every hyperparameter is a setter. Setters are `f64`/`usize`
/// per the A5 convention; `build::<F>()` narrows `alpha` to the target float `F`.
/// `Default` re-derives the sklearn defaults from [`Lasso::new`] (D-08 single
/// source) rather than holding literals (Pitfall 1).
#[derive(Debug, Clone, Copy)]
pub struct LassoBuilder {
    alpha: f64,
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
}

impl Default for LassoBuilder {
    /// Re-derive the sklearn defaults from [`Lasso::new`] (D-08 single source).
    fn default() -> Self {
        Lasso::<f64, Unfit>::new().into_builder()
    }
}

impl LassoBuilder {
    /// Set the L1 penalty strength `alpha` (A5: `f64` setter).
    pub fn alpha(mut self, v: f64) -> Self {
        self.alpha = v;
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

    /// Build the (unfit) estimator, validating the data-INDEPENDENT `alpha >= 0`
    /// BEFORE any data is seen ([`BuildError::InvalidAlpha`]) — relocated from the
    /// old `cd_fit` fit-body check (Pitfall 7; the data-DEPENDENT geometry check
    /// stays in [`Fit::fit`]). The stored `f64` `alpha` is narrowed to the target
    /// float `F` (A5).
    pub fn build<F>(self) -> Result<Lasso<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        if !(self.alpha >= 0.0) {
            return Err(BuildError::InvalidAlpha {
                estimator: "lasso",
                alpha: self.alpha,
            });
        }
        Ok(Lasso {
            alpha: f64_to_host::<F>(self.alpha),
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
            coef_: None,
            intercept_: None,
            _state: PhantomData,
        })
    }
}

impl<F> Lasso<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Host copy of the fitted `coef_` (length `n_features`). `Some` by
    /// construction on the `Fitted` state (D-03).
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<F> {
        self.coef_
            .as_ref()
            .expect("coef_ is Some by construction on Lasso<F, Fitted>")
            .to_host(pool)
    }

    /// Host copy of the fitted `intercept_` (scalar). `Some` by construction on
    /// the `Fitted` state (D-03).
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> F {
        self.intercept_
            .as_ref()
            .expect("intercept_ is Some by construction on Lasso<F, Fitted>")
            .to_host(pool)[0]
    }
}

impl<F> Fit<F> for Lasso<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = Lasso<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Lasso<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // Data-DEPENDENT geometry guard BEFORE any prim launch (the
        // data-INDEPENDENT `alpha >= 0` check was validated at build() — Pitfall 7).
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "lasso",
            operation: "fit (requires y)",
        })?;

        // Lasso = ElasticNet l1_ratio = 1.0 (→ l2_reg = 0, pure L1): delegate to
        // the SAME shared CD helper, NOT a duplicate coordinate-descent loop
        // (D-03). cd_fit validates alpha/l1_ratio/geometry before any launch.
        let (coef, intercept) = cd_fit::<F>(
            pool,
            x,
            y,
            n_samples,
            n_features,
            host_to_f64(self.alpha),
            1.0, // l1_ratio = 1 ⇒ pure L1 (l2_reg = 0)
            self.fit_intercept,
            self.tol,
            self.max_iter,
            "lasso",
        )?;

        Ok(Lasso {
            alpha: self.alpha,
            fit_intercept: self.fit_intercept,
            max_iter: self.max_iter,
            tol: self.tol,
            coef_: Some(coef),
            intercept_: Some(intercept),
            _state: PhantomData,
        })
    }
}

impl<F> Predict<F> for Lasso<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        // Reuse the shared X·coef + intercept GEMM path (D-03), no duplicate.
        predict_linear(
            self.coef_.as_ref(),
            self.intercept_.as_ref(),
            "lasso",
            pool,
            x,
            shape,
        )
    }
}
