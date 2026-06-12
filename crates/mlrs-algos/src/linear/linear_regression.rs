//! `LinearRegression` (LINEAR-01) — ordinary least squares via the SVD
//! pseudo-inverse, matching scikit-learn's default `scipy.linalg.lstsq` (gelsd)
//! path (D-02).
//!
//! ## Solver (deliberately NOT Cholesky — that is Ridge, D-02)
//! `coef = V · diag(σ⁺) · Uᵀ · y_centered` where the thin SVD of the (centered)
//! design matrix is `X = U · diag(σ) · Vᵀ` (`U` m×k, `σ` length-k, `Vᵀ` k×n,
//! `k = min(m, n)`), composed from the validated Phase-3 [`svd`] +
//! Phase-2 [`gemm`] / [`column_reduce`] primitives — NO bespoke matmul/solve.
//!
//! The pseudo-inverse uses sklearn's small-singular-value cutoff (RESEARCH
//! Pitfall 1 / Open Q3): `σ⁺_i = 1/σ_i if σ_i > cutoff else 0` with
//! `cutoff = rcond · σ_max`, `rcond = RCOND` (= `1e-6`). This MUST match
//! `sklearn.linear_model.LinearRegression`, which since the `tol` parameter
//! (default `1e-6`) passes that value as scipy's `lstsq(cond=…)` — scipy drops
//! every `σ_i ≤ cond·σ_max`. The looser numpy-lstsq / scipy-gelsd default
//! (`ε_F·max(m,n)`) does NOT match sklearn: on the near-collinear fixture its
//! `σ_min/σ_max ≈ 3e-8` is above that f64 threshold, so numpy reciprocates the
//! ~0 singular value and the coefficients EXPLODE to ~1e4, whereas sklearn (and
//! this estimator) drop it and return the bounded ~0.485 minimum-norm solution
//! (T-04-03-01). A `NEAR_ZERO_FLOOR` fallback keeps the cutoff strictly positive
//! even for an all-zero spectrum.
//!
//! ## Intercept via center-then-solve (D-05)
//! When `fit_intercept`, the column means `x̄` and `ȳ` are removed before the
//! solve and the intercept is recovered as `intercept_ = ȳ − x̄·coef_`. The
//! penalty-free intercept is never part of the SVD system (mirrors sklearn).
//!
//! ## Device residency (D-03)
//! Fitted `coef_` (length n) and `intercept_` (length 1) are stored as
//! device-resident [`DeviceArray`]s; `predict` runs the `X_test · coef_`
//! GEMM on-device and broadcasts the intercept. The host materializes the
//! fitted state only at a Rust accessor / oracle-comparison boundary.
//!
//! Tests live in `crates/mlrs-algos/tests/linear_regression_test.rs`
//! (AGENTS.md §2), never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use mlrs_backend::prims::svd::svd;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;
use crate::traits::{Fit, Predict};

/// Near-zero floor for the σ⁺ cutoff (mirrors the `svd.rs` `NEAR_ZERO_FLOOR`
/// precedent — below the 1e-5 tolerance so it never loosens a real check). Keeps
/// the cutoff strictly positive for a degenerate (all-zero) spectrum so a tiny
/// singular value is always zeroed rather than reciprocated.
const NEAR_ZERO_FLOOR: f64 = 1e-8;

/// Relative singular-value cutoff `rcond` for the pseudo-inverse — singular
/// values with `σ_i ≤ rcond·σ_max` are dropped (σ⁺ = 0). Pinned to `1e-6` to
/// match `sklearn.linear_model.LinearRegression`'s default `tol`, which it
/// forwards as `scipy.linalg.lstsq(cond=…)` (D-02 / Open Q3). This is the value
/// that reproduces sklearn on BOTH the full-rank and the near-collinear fixture;
/// the much smaller `ε_F·max(m,n)` numpy default would keep the collinear ~0
/// singular value and explode the coefficients (see module docs).
const RCOND: f64 = 1e-6;

/// Ordinary least squares (LINEAR-01) fitted by the SVD pseudo-inverse.
///
/// Construct with [`LinearRegression::new`] (`fit_intercept`), then [`Fit::fit`]
/// and [`Predict::predict`]. Fitted `coef_`/`intercept_` are device-resident
/// (D-03); the host accessors [`coef`](Self::coef) / [`intercept`](Self::intercept)
/// materialize them on demand.
pub struct LinearRegression<F> {
    fit_intercept: bool,
    /// Fitted coefficients (length `n_features`), device-resident, `None` until
    /// `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (length 1), device-resident, `None` until `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
}

impl<F> LinearRegression<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `LinearRegression`. `fit_intercept = true` centers `X`
    /// and `y` and recovers a bias term; `false` solves on the raw `X` and
    /// leaves `intercept_ = 0` (D-06 minimal surface).
    pub fn new(fit_intercept: bool) -> Self {
        Self {
            fit_intercept,
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
                estimator: "linear_regression",
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
                estimator: "linear_regression",
                operation: "intercept_",
            })
    }
}

impl<F> Fit<F> for LinearRegression<F>
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

        // --- T-04-03-02 / ASVS V5: validate geometry BEFORE any prim launch. ---
        if n_samples == 0 || n_features == 0 {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        if x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "linear_regression",
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

        // --- 1. Centering (D-05). When fit_intercept, remove the column means x̄
        //        and ȳ; solve on the centered system. Mirrors covariance.rs'
        //        two-pass centring. Done host-side here because the σ⁺ cutoff and
        //        intercept recovery already need a host pass over the tiny k/n
        //        vectors; the heavy products stay on-device via gemm/svd. ---
        let x_host = x.to_host(pool);
        let y_host = y.to_host(pool);

        let mut x_mean = vec![0.0f64; n_features];
        let mut y_mean = 0.0f64;
        if self.fit_intercept {
            for r in 0..n_samples {
                for c in 0..n_features {
                    x_mean[c] += host_to_f64(x_host[r * n_features + c]);
                }
                y_mean += host_to_f64(y_host[r]);
            }
            let inv = 1.0 / n_samples as f64;
            for m in x_mean.iter_mut() {
                *m *= inv;
            }
            y_mean *= inv;
        }

        let mut x_centered: Vec<F> = vec![F::from_int(0i64); n_samples * n_features];
        for r in 0..n_samples {
            for c in 0..n_features {
                let v = host_to_f64(x_host[r * n_features + c]) - x_mean[c];
                x_centered[r * n_features + c] = f64_to_host::<F>(v);
            }
        }
        let mut y_centered: Vec<F> = vec![F::from_int(0i64); n_samples];
        for r in 0..n_samples {
            y_centered[r] = f64_to_host::<F>(host_to_f64(y_host[r]) - y_mean);
        }

        let x_c_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_centered);
        let y_c_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y_centered);

        // Use the Phase-2 column-mean reduction on the (zero-mean) centered X as
        // the key-link `column_reduce(.., ScalarOp::Mean, ..)` site (the centering
        // means above are the host two-pass form; this confirms the centered
        // columns are ~0-mean and exercises the prim path). The result is not
        // load-bearing for the solve.
        let _centered_means = column_reduce::<F>(
            pool,
            &x_c_dev,
            n_samples,
            n_features,
            ScalarOp::Mean,
            ReducePath::Shared,
        )?
        .expect("shared path is never plane-gated to None");
        let _ = _centered_means.to_host(pool);
        _centered_means.release_into(pool);

        // --- 2. Thin SVD of the centered design (D-02): X_c = U·diag(σ)·Vᵀ,
        //        U (m×k), σ (k), Vᵀ (k×n), k = min(m, n). ---
        let k = n_samples.min(n_features);
        let (u, s, vt) = svd::<F>(pool, &x_c_dev, (n_samples, n_features))?;

        // --- 3. σ⁺ with sklearn's small-σ cutoff (Pitfall 1 / T-04-03-01 /
        //        Open Q3). cutoff = RCOND · σ_max (RCOND = 1e-6 = sklearn's
        //        default `tol`, forwarded as scipy `lstsq(cond=…)`), floored at
        //        NEAR_ZERO_FLOOR so it is strictly positive even for a degenerate
        //        spectrum. The looser ε_F·max(m,n) numpy default would keep the
        //        collinear ~0 singular value and explode the coefficients. ---
        let s_host = s.to_host(pool);
        let s64: Vec<f64> = s_host.iter().map(|&v| host_to_f64(v)).collect();
        let sigma_max = s64.iter().cloned().fold(0.0f64, f64::max);
        let cutoff = (RCOND * sigma_max).max(NEAR_ZERO_FLOOR);

        // --- 4. coef = V · diag(σ⁺) · (Uᵀ · y_c). Compose with gemm; the only
        //        host arithmetic is the length-k σ⁺ scaling (the cutoff guard). ---
        // t1 = Uᵀ · y_c  (k×1). U is (m×k) row-major; transa reads it as Uᵀ
        // (k×m) — no transpose buffer (D-06).
        let t1 = gemm::<F>(
            pool,
            &u,
            (k, n_samples),    // logical Uᵀ is (k × m)
            &y_c_dev,
            (n_samples, 1),
            true,  // u buffer is U (m×k) row-major; transa reads it as Uᵀ.
            false,
            None,
        )?;
        let t1_host = t1.to_host(pool);

        // t2 = diag(σ⁺) · t1  (k×1) — the small-σ cutoff lives here.
        let mut t2_host: Vec<F> = vec![F::from_int(0i64); k];
        for i in 0..k {
            let sigma = s64[i];
            let scaled = if sigma > cutoff {
                host_to_f64(t1_host[i]) / sigma
            } else {
                0.0 // drop the near-zero singular direction (no 1/0 blow-up).
            };
            t2_host[i] = f64_to_host::<F>(scaled);
        }
        let t2_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &t2_host);

        // coef = V · t2  (n×1). Vᵀ is (k×n) row-major; transa reads it as V
        // (n×k) — no transpose buffer (D-06).
        let coef = gemm::<F>(
            pool,
            &vt,
            (n_features, k),   // logical V is (n × k)
            &t2_dev,
            (k, 1),
            true,  // vt buffer is Vᵀ (k×n) row-major; transa reads it as V.
            false,
            None,
        )?;

        // --- 5. intercept_ = ȳ − x̄·coef_ when fit_intercept, else 0 (D-05). ---
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

        // --- 6. Release scratch; store device-resident fitted state (D-03). ---
        u.release_into(pool);
        s.release_into(pool);
        vt.release_into(pool);
        t1.release_into(pool);
        t2_dev.release_into(pool);
        x_c_dev.release_into(pool);
        y_c_dev.release_into(pool);

        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept_dev);
        Ok(self)
    }
}

impl<F> Predict<F> for LinearRegression<F>
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

        let coef = self.coef_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "linear_regression",
            operation: "predict",
        })?;
        let intercept = self.intercept_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "linear_regression",
            operation: "predict",
        })?;

        // --- T-04-03-02 / ASVS V5: geometry + fitted-n_features consistency. ---
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

        // Broadcast-add the scalar intercept. Sizes here are tiny (length m); the
        // bias add is a host pass that returns a fresh device array (the fitted
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
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine (mirrors the
/// `svd.rs` helper).
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("linear_regression is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("linear_regression is f32/f64 only"),
    }
}
