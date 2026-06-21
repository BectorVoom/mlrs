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

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::host_to_f64;

use crate::error::AlgoError;
use crate::linear::coordinate_descent::{cd_fit, CD_DEFAULT_MAX_ITER, CD_DEFAULT_TOL};
use crate::linear::elastic_net::predict_linear;
use crate::traits::{Fit, Predict};

/// L1-penalized least squares (LINEAR-03) — the `l1_ratio == 1` case of
/// [`ElasticNet`](crate::linear::elastic_net::ElasticNet), fitted by the shared
/// coordinate-descent solver.
///
/// Construct with [`Lasso::new`] (`alpha`, `fit_intercept`) or [`Lasso::with_opts`]
/// to override `max_iter` / `tol`, then [`Fit::fit`] and [`Predict::predict`].
/// Fitted `coef_`/`intercept_` are device-resident (D-03); the host accessors
/// [`coef`](Self::coef) / [`intercept`](Self::intercept) materialize them on
/// demand.
pub struct Lasso<F> {
    /// L1 penalty strength (`alpha ≥ 0`; `alpha = 0` degenerates to OLS).
    /// Validated at `fit` (T-05-09-01).
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
}

impl<F> Lasso<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `Lasso` with penalty `alpha` and the `fit_intercept`
    /// flag (D-06 minimal surface), using sklearn's default `max_iter = 1000` /
    /// `tol = 1e-4`. A negative `alpha` is rejected at `fit` with
    /// [`AlgoError::InvalidAlpha`] (T-05-09-01).
    pub fn new(alpha: F, fit_intercept: bool) -> Self {
        Self::with_opts(alpha, fit_intercept, CD_DEFAULT_MAX_ITER, CD_DEFAULT_TOL)
    }

    /// Like [`Lasso::new`] but overrides the coordinate-descent `max_iter` and
    /// stopping `tol`.
    pub fn with_opts(alpha: F, fit_intercept: bool, max_iter: usize, tol: f64) -> Self {
        Self {
            alpha,
            fit_intercept,
            max_iter,
            tol,
            coef_: None,
            intercept_: None,
        }
    }

    /// Host copy of the fitted `coef_` (length `n_features`). Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    pub fn coef(&self, pool: &BufferPool<ActiveRuntime>) -> Result<Vec<F>, AlgoError> {
        self.coef_
            .as_ref()
            .map(|c| c.to_host(pool))
            .ok_or(AlgoError::NotFitted {
                estimator: "lasso",
                operation: "coef_",
            })
    }

    /// Host copy of the fitted `intercept_` (scalar). Errors with
    /// [`AlgoError::NotFitted`] before `fit`.
    pub fn intercept(&self, pool: &BufferPool<ActiveRuntime>) -> Result<F, AlgoError> {
        self.intercept_
            .as_ref()
            .map(|i| i.to_host(pool)[0])
            .ok_or(AlgoError::NotFitted {
                estimator: "lasso",
                operation: "intercept_",
            })
    }
}

impl<F> Fit<F> for Lasso<F>
where
    F: Float + CubeElement + Pod,
{
    fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
        let (n_samples, n_features) = shape;
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

        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept);
        Ok(self)
    }
}

impl<F> Predict<F> for Lasso<F>
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
