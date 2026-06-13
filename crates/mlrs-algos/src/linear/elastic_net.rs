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

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;
use crate::linear::coordinate_descent::{cd_fit, CD_DEFAULT_MAX_ITER, CD_DEFAULT_TOL};
use crate::traits::{Fit, Predict};

/// L1+L2-penalized least squares (LINEAR-04) fitted by the shared
/// coordinate-descent solver.
///
/// Construct with [`ElasticNet::new`] (`alpha`, `l1_ratio`, `fit_intercept`) or
/// [`ElasticNet::with_opts`] to override `max_iter` / `tol`, then [`Fit::fit`]
/// and [`Predict::predict`]. Fitted `coef_`/`intercept_` are device-resident
/// (D-03); the host accessors [`coef`](Self::coef) / [`intercept`](Self::intercept)
/// materialize them on demand.
pub struct ElasticNet<F> {
    /// Overall penalty strength (`alpha ≥ 0`; `alpha = 0` degenerates to OLS).
    alpha: F,
    /// L1/L2 mixing parameter (`0 ≤ l1_ratio ≤ 1`; `1` ⇒ Lasso, `0` ⇒ Ridge-like
    /// pure L2). Validated at `fit` (T-05-09-01).
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
}

impl<F> ElasticNet<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `ElasticNet` with penalty `alpha`, mixing `l1_ratio`,
    /// and the `fit_intercept` flag (D-06 minimal surface), using sklearn's
    /// default `max_iter = 1000` / `tol = 1e-4`. A negative `alpha`
    /// ([`AlgoError::InvalidAlpha`]) or an `l1_ratio ∉ [0, 1]`
    /// ([`AlgoError::InvalidL1Ratio`]) is rejected at `fit` (T-05-09-01).
    pub fn new(alpha: F, l1_ratio: F, fit_intercept: bool) -> Self {
        Self::with_opts(
            alpha,
            l1_ratio,
            fit_intercept,
            CD_DEFAULT_MAX_ITER,
            CD_DEFAULT_TOL,
        )
    }

    /// Like [`ElasticNet::new`] but overrides the coordinate-descent `max_iter`
    /// and stopping `tol`.
    pub fn with_opts(
        alpha: F,
        l1_ratio: F,
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            alpha,
            l1_ratio,
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
                estimator: "elastic_net",
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
                estimator: "elastic_net",
                operation: "intercept_",
            })
    }
}

impl<F> Fit<F> for ElasticNet<F>
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

        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept);
        Ok(self)
    }
}

impl<F> Predict<F> for ElasticNet<F>
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

    // y_pred = X_test · coef  (m×1) via the Phase-2 GEMM, on-device (D-03).
    let raw = gemm::<F>(
        pool,
        x,
        (n_samples, n_features),
        coef,
        (n_features, 1),
        false,
        false,
        None,
    )?;

    // Broadcast-add the scalar intercept (tiny length-m host pass; the fitted
    // state itself stays device-resident, materialized only at this terminal).
    let bias = host_to_f64(intercept.to_host(pool)[0]);
    let raw_host = raw.to_host(pool);
    let mut pred_host: Vec<F> = vec![F::from_int(0i64); n_samples];
    for r in 0..n_samples {
        pred_host[r] = f64_to_host::<F>(host_to_f64(raw_host[r]) + bias);
    }
    raw.release_into(pool);
    Ok(DeviceArray::from_host(pool, &pred_host))
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine (mirrors the
/// `ridge.rs` helper).
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("elastic_net is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("elastic_net is f32/f64 only"),
    }
}
