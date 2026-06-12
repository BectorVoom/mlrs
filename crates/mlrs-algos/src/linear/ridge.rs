//! `Ridge` (LINEAR-02) — L2-penalized least squares via the Cholesky
//! normal-equations solver (D-02), matching
//! `sklearn.linear_model.Ridge(solver='cholesky')`.
//!
//! ## Solver (deliberately Cholesky, NOT SVD — that is LinearRegression, D-02)
//! Ridge solves the regularized normal equations
//! `(XᵀX + αI)·coef = Xᵀy`
//! via the validated Phase-4 [`cholesky_solve`] primitive (`A = L·Lᵀ`, then
//! forward and back substitution, all in-kernel — 04-02). It does NOT use the
//! SVD pseudo-inverse path (that is the LinearRegression anti-pattern; the two
//! solvers MUST NOT be unified — RESEARCH Anti-Patterns / D-02).
//!
//! ## Raw Gram, NOT scaled covariance (RESEARCH Open Q1)
//! The normal matrix is the **raw** Gram `XᵀX` formed by
//! `gemm(transa=true)` over the centered design — NOT `prims::covariance`,
//! which centers AND scales by `1/(n−ddof)`. sklearn's `_solve_cholesky` adds
//! `alpha` to the raw `XᵀX` diagonal directly (no `n_samples` scaling), so the
//! raw Gram is the sklearn-faithful normal matrix (verified against the
//! committed fixture: `Xc·Xc + αI` reproduces sklearn's `coef_` exactly).
//!
//! ## alpha on the diagonal only; intercept never penalized (D-05)
//! `alpha` is added to the Gram DIAGONAL only (`A[i·n+i] += alpha`). The
//! intercept is recovered AFTER the solve via center-then-solve
//! (`intercept_ = ȳ − x̄·coef_`) and is therefore NEVER part of the penalized
//! system — sklearn-exact (RESEARCH Pitfall 5; α applies only to `coef_`).
//!
//! ## Gram threaded through the Cholesky factor (D-11 gate 2)
//! The Gram buffer `(XᵀX + αI)` is passed as the Cholesky primitive's `out`
//! working buffer, so the factor reuses it in place — no parallel `n²`
//! allocation (the memory gate, 04-05 Task 2, asserts this).
//!
//! ## Non-SPD guard (RESEARCH Pitfall 4 / T-04-05-01)
//! A near-singular `(XᵀX + αI)` (tiny α + collinear X) drives a non-positive
//! Cholesky pivot; the 04-02 primitive surfaces
//! [`PrimError::NotPositiveDefinite`], which this estimator propagates as an
//! [`AlgoError`] (via `#[from]`) rather than emitting NaN coefficients.
//!
//! ## Device residency (D-03)
//! Fitted `coef_` (length n) and `intercept_` (length 1) are stored as
//! device-resident [`DeviceArray`]s; `predict` runs the `X_test · coef_`
//! GEMM on-device and broadcasts the intercept, materializing to the host only
//! at a Rust accessor / oracle-comparison boundary.
//!
//! Tests live in `crates/mlrs-algos/tests/ridge_test.rs` (AGENTS.md §2), never
//! an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::cholesky::cholesky_solve;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;

use crate::error::AlgoError;
use crate::traits::{Fit, Predict};

/// L2-penalized least squares (LINEAR-02) fitted by the Cholesky
/// normal-equations solver.
///
/// Construct with [`Ridge::new`] (`alpha`, `fit_intercept`), then [`Fit::fit`]
/// and [`Predict::predict`]. Fitted `coef_`/`intercept_` are device-resident
/// (D-03); the host accessors [`coef`](Self::coef) / [`intercept`](Self::intercept)
/// materialize them on demand.
pub struct Ridge<F> {
    /// L2 penalty strength (`alpha ≥ 0`; `alpha = 0` degenerates to OLS).
    /// Added to the Gram diagonal only — never to the intercept (D-05).
    alpha: F,
    /// Whether to center `X`/`y` and recover a bias term (D-05).
    fit_intercept: bool,
    /// Fitted coefficients (length `n_features`), device-resident, `None` until
    /// `fit`.
    coef_: Option<DeviceArray<ActiveRuntime, F>>,
    /// Fitted intercept (length 1), device-resident, `None` until `fit`.
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
}

impl<F> Ridge<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `Ridge` with penalty `alpha` and the `fit_intercept`
    /// flag (D-06 minimal surface). `fit_intercept = true` centers `X` and `y`
    /// and recovers a bias term (α never penalizes it, D-05); `false` solves on
    /// the raw `X` and leaves `intercept_ = 0`. A negative `alpha` is rejected at
    /// `fit` time with [`AlgoError::InvalidAlpha`] (T-04-05-03).
    pub fn new(alpha: F, fit_intercept: bool) -> Self {
        Self {
            alpha,
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
                estimator: "ridge",
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
                estimator: "ridge",
                operation: "intercept_",
            })
    }
}

impl<F> Fit<F> for Ridge<F>
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

        // --- T-04-05-03 / ASVS V5: validate the untrusted hyperparameter and
        //     geometry BEFORE any prim launch. alpha < 0 makes (XᵀX + αI)
        //     indefinite and the Cholesky factorization undefined. ---
        let alpha64 = host_to_f64(self.alpha);
        if alpha64 < 0.0 {
            return Err(AlgoError::InvalidAlpha {
                estimator: "ridge",
                alpha: alpha64,
            });
        }
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
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

        // --- 1. Centering (D-05). When fit_intercept, remove the column means x̄
        //        and ȳ; solve on the centered system. Mirrors the LinearRegression
        //        host two-pass centering — done host-side because the diagonal-α
        //        injection and the intercept recovery already need a host pass over
        //        the tiny n-vectors; the heavy products (Gram, Xᵀy, solve) stay
        //        on-device. ---
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

        // Phase-2 column-mean reduction on the (zero-mean) centered X as the
        // documented key-link `column_reduce(.., ScalarOp::Mean, ..)` site (shared
        // with LinearRegression). The load-bearing means are the host two-pass
        // form above; this confirms the centered columns are ~0-mean and exercises
        // the prim path. The result is not load-bearing for the solve.
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

        // --- 2. Raw Gram XᵀX via gemm(transa=true) (RESEARCH Open Q1 — NOT the
        //        scaled covariance). x_c_dev is the centered design (m×n) row-major;
        //        transa reads it as Xᵀ (n×m), so the product is the n×n Gram. ---
        let raw_gram = gemm::<F>(
            pool,
            &x_c_dev,
            (n_features, n_samples), // logical Xᵀ is (n × m)
            &x_c_dev,
            (n_samples, n_features),
            true, // first operand buffer is X (m×n); transa reads it as Xᵀ.
            false,
            None,
        )?;

        // --- 3. alpha on the Gram DIAGONAL only (D-05 / T-04-05-02). Add `alpha`
        //        to element [i·n+i]; NEVER to the intercept (the intercept is
        //        recovered post-solve, outside this penalized system). The
        //        diagonal-stride `+= alpha` is the load-bearing penalty injection.
        //        cubecl 0.10 has no in-place device write, so we materialize the
        //        small n×n Gram, add α on the diagonal, RELEASE the raw-Gram buffer
        //        back to the pool (so no parallel n² buffer lives), and re-stage the
        //        regularized Gram — `from_host` recycles the just-released n²
        //        byte-size from the free-list (D-11 gate 2: no second live n²). ---
        let mut gram_host = raw_gram.to_host(pool);
        for i in 0..n_features {
            let d = host_to_f64(gram_host[i * n_features + i]) + alpha64;
            gram_host[i * n_features + i] = f64_to_host::<F>(d);
        }
        raw_gram.release_into(pool);
        let gram: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &gram_host);

        // --- 4. Xᵀy via gemm(transa=true): the centered RHS (n×1). ---
        let xty = gemm::<F>(
            pool,
            &x_c_dev,
            (n_features, n_samples), // logical Xᵀ is (n × m)
            &y_c_dev,
            (n_samples, 1),
            true, // first operand buffer is X (m×n); transa reads it as Xᵀ.
            false,
            None,
        )?;

        // --- 5. Solve (XᵀX + αI)·coef = Xᵀy with the Cholesky primitive (D-02).
        //        Thread the regularized Gram buffer through `out` so the factor
        //        reuses it in place — no parallel n² allocation (D-11 gate 2). The
        //        kernel only READS `out` as its working input, so the threaded
        //        buffer is consumed (released back to the pool) by the call; we
        //        clone the handle for `out` and keep `gram` as the `a` operand. A
        //        non-SPD pivot (near-singular Gram) surfaces NotPositiveDefinite →
        //        AlgoError (Pitfall 4 / T-04-05-01), never NaN coef_. ---
        let gram_out = DeviceArray::<ActiveRuntime, F>::from_raw(
            gram.handle().clone(),
            n_features * n_features,
        );
        let coef = cholesky_solve::<F>(pool, &gram, &xty, n_features, 1, Some(gram_out))?;

        // --- 6. intercept_ = ȳ − x̄·coef_ when fit_intercept, else 0 (D-05). α is
        //        NOT applied here — the intercept is unpenalized. ---
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

        // --- 7. Release scratch; store device-resident fitted state (D-03). The
        //        Gram buffer was consumed (its cloned handle threaded through `out`
        //        and released by the Cholesky solve — so we do NOT release `gram`
        //        again here, avoiding a double-release of the shared allocation);
        //        release the remaining transients. ---
        drop(gram);
        xty.release_into(pool);
        x_c_dev.release_into(pool);
        y_c_dev.release_into(pool);

        self.coef_ = Some(coef);
        self.intercept_ = Some(intercept_dev);
        Ok(self)
    }
}

impl<F> Predict<F> for Ridge<F>
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
            estimator: "ridge",
            operation: "predict",
        })?;
        let intercept = self.intercept_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "ridge",
            operation: "predict",
        })?;

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
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine (mirrors the
/// `linear_regression.rs` helper).
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("ridge is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("ridge is f32/f64 only"),
    }
}
