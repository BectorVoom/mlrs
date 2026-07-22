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

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::center::center_columns;
use mlrs_backend::prims::cholesky::cholesky_solve;
use mlrs_backend::prims::gram::gram_xty;
use mlrs_backend::prims::linear_predict::linear_predict;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::typestate::{validate_geometry, Fit, Fitted, Predict, Unfit};

/// L2-penalized least squares (LINEAR-02) fitted by the Cholesky
/// normal-equations solver.
///
/// Construct with the zero-arg [`Ridge::new`] (sklearn defaults: `alpha = 1.0`,
/// `fit_intercept = true`) or [`Ridge::builder`], then the consuming
/// [`Fit::fit`] (returns the `Fitted`-tagged sibling) and [`Predict::predict`].
/// Fitted `coef_`/`intercept_` are device-resident (D-03); the host accessors
/// [`coef`](Ridge::coef) / [`intercept`](Ridge::intercept) materialize them on
/// demand and exist ONLY on `Ridge<F, Fitted>` (the compile-time typestate
/// replaces the old runtime `NotFitted` guard, D-03).
pub struct Ridge<F, S = Unfit> {
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
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> Ridge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Construct an `Ridge` with sklearn's `Ridge` defaults (`alpha = 1.0`,
    /// `fit_intercept = true`) directly in the `Unfit` state. This is the SINGLE
    /// source of truth for the default hyperparameters (D-08): the builder
    /// `Default` re-derives from here via [`Ridge::into_builder`], rather than
    /// re-listing the literals. Defaults are trusted valid, so this bypasses
    /// [`RidgeBuilder::build`]'s validation.
    pub fn new() -> Self {
        Self {
            alpha: F::from_int(1),
            fit_intercept: true,
            coef_: None,
            intercept_: None,
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
        }
    }

    /// Compare the hyperparameter subset of two `Unfit` estimators (the fitted
    /// `coef_`/`intercept_` fields are excluded — both are `None` in any `Unfit`
    /// value). Used by the defaults-equality test (BLDR-01):
    /// `Ridge::new().hyperparams_eq(&Ridge::builder().build()?)`.
    pub fn hyperparams_eq(&self, other: &Self) -> bool {
        host_to_f64(self.alpha) == host_to_f64(other.alpha)
            && self.fit_intercept == other.fit_intercept
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

/// Builder for [`Ridge`] (D-01). Setters are `f64`-typed per the A5 convention;
/// `build::<F>()` narrows to the target float `F`. `Default` re-derives the
/// sklearn defaults from [`Ridge::new`] (D-08 single source) rather than holding
/// literals (Pitfall 1: default-drift breaks the oracle gate silently).
#[derive(Debug, Clone, Copy)]
pub struct RidgeBuilder {
    alpha: f64,
    fit_intercept: bool,
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

    /// Build the (unfit) estimator, validating the data-INDEPENDENT
    /// hyperparameters BEFORE any data is seen (D-08; the data-DEPENDENT
    /// geometry check lives in [`Fit::fit`]):
    ///
    /// - `alpha >= 0` ([`BuildError::InvalidAlpha`]) — a negative penalty makes
    ///   `(XᵀX + αI)` indefinite and the Cholesky factorization undefined
    ///   (relocated from the old fit-body check, T-04-05-03 / Pitfall 7).
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
        Ok(Ridge {
            alpha: f64_to_host::<F>(self.alpha),
            fit_intercept: self.fit_intercept,
            coef_: None,
            intercept_: None,
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
}

impl<F> Fit<F> for Ridge<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = Ridge<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<Ridge<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-04-05-03 / ASVS V5: data-DEPENDENT geometry guard BEFORE any
        //     prim launch (the data-INDEPENDENT `alpha >= 0` check was validated
        //     at build() — Pitfall 7). ---
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

        // RIDGE_PROFILE=1: per-phase wall-clock attribution (the LR_PROFILE
        // precedent in `linear_regression.rs`'s `fit_gram_eig` — attribution
        // only, since kernel launches are async and a lap only completes at the
        // next readback that drains the queue; a tiny forced readback after
        // `gram_xty`/`cholesky_solve` pins each phase's lap to ITS OWN kernels
        // rather than bleeding into the next phase's).
        let profile = std::env::var("RIDGE_PROFILE").is_ok();
        let lap0 = std::time::Instant::now();

        // --- 1. Centering, DEVICE-resident (D-05 / perf): mirrors
        //        LinearRegression's `fit_gram_eig` (`center_columns` composes
        //        `column_reduce` + the center kernel with no host round-trip of
        //        the full n×d design — the original host two-pass form here was
        //        an O(n·d) PCIe-bound cost that dominates at scale). When
        //        !fit_intercept there is nothing to remove — `x`/`y` are read
        //        directly by `gram_xty` below with NO copy. ---
        let (x_mean, y_mean, x_owned, y_owned) = if self.fit_intercept {
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
            (x_mean, y_mean, Some(x_c), Some(y_c))
        } else {
            (vec![0.0f64; n_features], 0.0f64, None, None)
        };
        let x_ref = x_owned.as_ref().unwrap_or(x);
        let y_ref = y_owned.as_ref().unwrap_or(y);
        let t_center = if profile { lap0.elapsed().as_secs_f64() } else { 0.0 };

        // --- 2. Raw Gram G = XᵀX (d×d) and c = Xᵀy (d×1) via the row-blocked
        //        `gram_xty` prim (RESEARCH Open Q1 — NOT the scaled covariance;
        //        LINEAR-01/02 perf lever shared with LinearRegression): replaces
        //        the skinny-output/huge-K `gemm` pair that starved the GPU of
        //        parallel work regardless of `n_samples` (see `mlrs_kernels::gram`
        //        module docs) with a row-blocked shared-memory accumulation. ---
        let lap1 = std::time::Instant::now();
        let (raw_gram, xty) = gram_xty::<F>(pool, x_ref, y_ref, n_samples, n_features)?;
        if profile {
            // Force a drain so this lap attributes ONLY gram_xty's kernels, not
            // whatever runs next (the LR_PROFILE precedent's readback-boundary
            // caveat) — a tiny d-element readback, not the n-heavy data.
            let _ = xty.to_host(pool);
        }
        let t_gram = if profile { lap1.elapsed().as_secs_f64() } else { 0.0 };
        if let Some(xc) = x_owned {
            xc.release_into(pool);
        }
        if let Some(yc) = y_owned {
            yc.release_into(pool);
        }

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

        // --- 4. Solve (XᵀX + αI)·coef = Xᵀy with the Cholesky primitive (D-02).
        //        `xty` (n×1) was already formed above by `gram_xty` — no separate
        //        gemm needed. Thread the regularized Gram buffer through `out` so
        //        the factor reuses it in place — no parallel n² allocation (D-11
        //        gate 2). The kernel only READS `out` as its working input, so the
        //        threaded buffer is consumed (released back to the pool) by the
        //        call; we clone the handle for `out` and keep `gram` as the `a`
        //        operand. A non-SPD pivot (near-singular Gram) surfaces
        //        NotPositiveDefinite → AlgoError (Pitfall 4 / T-04-05-01), never
        //        NaN coef_. ---
        let lap2 = std::time::Instant::now();
        let gram_out = DeviceArray::<ActiveRuntime, F>::from_raw(
            gram.handle().clone(),
            n_features * n_features,
        );
        let coef = cholesky_solve::<F>(pool, &gram, &xty, n_features, 1, Some(gram_out))?;
        if profile {
            let _ = coef.to_host(pool);
        }
        let t_solve = if profile { lap2.elapsed().as_secs_f64() } else { 0.0 };

        // --- 5. intercept_ = ȳ − x̄·coef_ when fit_intercept, else 0 (D-05). α is
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

        // --- 6. Release scratch; store device-resident fitted state (D-03). The
        //        Gram buffer was consumed (its cloned handle threaded through `out`
        //        and released by the Cholesky solve — so we do NOT release `gram`
        //        again here, avoiding a double-release of the shared allocation);
        //        release the remaining transients. ---
        drop(gram);
        xty.release_into(pool);

        if profile {
            eprintln!(
                "RIDGE_PROFILE n={n_samples} d={n_features}: center={t_center:.4}s gram_xty={t_gram:.4}s solve={t_solve:.4}s"
            );
        }

        Ok(Ridge {
            alpha: self.alpha,
            fit_intercept: self.fit_intercept,
            coef_: Some(coef),
            intercept_: Some(intercept_dev),
            _state: PhantomData,
        })
    }
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
