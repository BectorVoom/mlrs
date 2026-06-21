//! `linear::coordinate_descent` — the SHARED host fit helper for the two
//! iterative-solver linear models, Lasso (LINEAR-03) and ElasticNet (LINEAR-04),
//! built on the validated Phase-5 coordinate-descent primitive
//! ([`mlrs_backend::prims::coordinate_descent::cd_solve`], 05-05, D-03).
//!
//! ## One helper, two estimators (D-03 — Lasso = ElasticNet `l1_ratio == 1`)
//! `ElasticNet<F>` and `Lasso<F>` BOTH delegate to [`cd_fit`]; Lasso simply
//! passes `l1_ratio = 1.0` (→ `l2_reg = 0`, pure L1). This is the deliberate
//! shared coordinate-descent path. It is NOT unified with the L-BFGS LogReg
//! solver (05-10 owns that — a different optimizer for a different objective;
//! see the `linear/mod.rs` "deliberately different solvers" note).
//!
//! ## Penalty mapping (RESEARCH `_coordinate_descent.py:781-782` / Pitfall 1)
//! The user-facing `(alpha, l1_ratio)` are mapped to sklearn's UN-normalized
//! penalties before the solve:
//! `l1_reg = alpha · l1_ratio · n_samples`,
//! `l2_reg = alpha · (1 − l1_ratio) · n_samples`.
//! The `n_samples` scaling is load-bearing (Pitfall 1): omitting it shifts the
//! whole sparsity pattern and the oracle's exact zeros no longer match sklearn.
//!
//! ## Center-then-solve intercept (D-13, reused from `ridge.rs`)
//! sklearn runs `enet_coordinate_descent` on the CENTERED design, then recovers
//! the intercept from the means. `cd_fit` mirrors the `ridge.rs` center-then-solve
//! precedent VERBATIM: when `fit_intercept`, it removes the column means `x̄` and
//! the target mean `ȳ` host-side (two-pass), solves the centered problem with
//! `cd_solve`, and recovers `intercept_ = ȳ − x̄·coef_`. The intercept is NEVER
//! penalized (it is outside the centered penalized system, exactly like Ridge's
//! D-05). With `fit_intercept = false` the raw design is solved and the intercept
//! is 0.
//!
//! ## Validate-before-launch (ASVS V5 / T-05-09-01)
//! `alpha ≥ 0` (`InvalidAlpha`) and `0 ≤ l1_ratio ≤ 1` (`InvalidL1Ratio`) are
//! checked, along with the geometry, BEFORE any prim launch so an untrusted
//! hyperparameter becomes a typed error, not an out-of-bounds device read.
//!
//! Tests live in `crates/mlrs-algos/tests/{lasso,elastic_net}_test.rs`
//! (AGENTS.md §2), never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::coordinate_descent::cd_solve;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::AlgoError;

/// sklearn's default coordinate-descent stopping tolerance (`tol = 1e-4`); the
/// primitive scales it by `‖y‖²` internally (Pitfall 2).
pub const CD_DEFAULT_TOL: f64 = 1e-4;
/// sklearn's default coordinate-descent iteration cap (`max_iter = 1000`).
pub const CD_DEFAULT_MAX_ITER: usize = 1000;

/// Shared host fit helper for Lasso / ElasticNet (D-03). Validates the untrusted
/// `(alpha, l1_ratio)` and geometry, centers `(X, y)` when `fit_intercept`
/// (D-13), maps `(alpha, l1_ratio)` → the un-normalized `(l1_reg, l2_reg)`
/// (Pitfall 1), runs the validated [`cd_solve`] primitive on the centered design,
/// and recovers the unpenalized `intercept_ = ȳ − x̄·coef_`.
///
/// Returns the device-resident `(coef_, intercept_)` (`coef_` length `d`,
/// `intercept_` length 1, D-03). `estimator` names the caller (`"lasso"` /
/// `"elastic_net"`) for the error variants.
///
/// - `alpha < 0` → [`AlgoError::InvalidAlpha`] (T-05-09-01).
/// - `l1_ratio ∉ [0, 1]` → [`AlgoError::InvalidL1Ratio`] (T-05-09-01).
/// - A wrong geometry → [`AlgoError::Prim`]`(`[`PrimError::ShapeMismatch`]`)`.
/// - A non-convergent solve (the `cd_solve` cap) surfaces as
///   [`AlgoError::NotConverged`].
#[allow(clippy::too_many_arguments)]
pub fn cd_fit<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    y: &DeviceArray<ActiveRuntime, F>,
    n_samples: usize,
    n_features: usize,
    alpha: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    tol: f64,
    max_iter: usize,
    estimator: &'static str,
) -> Result<
    (
        DeviceArray<ActiveRuntime, F>,
        DeviceArray<ActiveRuntime, F>,
    ),
    AlgoError,
>
where
    F: Float + CubeElement + Pod,
{
    // --- ASVS V5 / T-05-09-01: validate the untrusted hyperparameters and the
    //     geometry BEFORE any prim launch. ---
    if !(alpha >= 0.0) {
        return Err(AlgoError::InvalidAlpha { estimator, alpha });
    }
    if !(0.0..=1.0).contains(&l1_ratio) {
        return Err(AlgoError::InvalidL1Ratio {
            estimator,
            l1_ratio,
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
    if y.len() != n_samples {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch {
            operand: "y",
            rows: n_samples,
            cols: 1,
            len: y.len(),
        }));
    }

    // --- 1. Center X and y host-side (D-13, the ridge.rs two-pass means copied
    //        verbatim). sklearn runs enet_coordinate_descent on the centered
    //        design; cd_solve solves the centered problem. With fit_intercept the
    //        means are removed; otherwise x̄ = ȳ = 0 and the raw design is solved. ---
    let x_host = x.to_host(pool);
    let y_host = y.to_host(pool);

    let mut x_mean = vec![0.0f64; n_features];
    let mut y_mean = 0.0f64;
    if fit_intercept {
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

    // WR-04: center in the WORKING dtype `F` exactly like sklearn (which centers
    // once in the input dtype), NOT in f64-then-narrow. We round each mean to `F`
    // precision first, so the centered design is `fl_F(x_F − mean_F)` — the SAME
    // single rounding sklearn performs — rather than `fl_F(x_f64 − mean_f64)`,
    // which injects a different round-off into `norm2_cols` (the soft-threshold
    // denominator that drives the exact-zero sparsity pattern). cd_solve then
    // re-reads this `F` design and promotes to f64 LOSSLESSLY, so there is one
    // consistent representation end-to-end.
    let x_mean_f: Vec<f64> = x_mean.iter().map(|&m| narrow_to_f::<F>(m)).collect();
    let y_mean_f = narrow_to_f::<F>(y_mean);
    let mut x_centered: Vec<F> = vec![F::from_int(0i64); n_samples * n_features];
    for r in 0..n_samples {
        for c in 0..n_features {
            let xij = host_to_f64(x_host[r * n_features + c]); // already F-exact
            x_centered[r * n_features + c] = f64_to_host::<F>(xij - x_mean_f[c]);
        }
    }
    let mut y_centered: Vec<F> = vec![F::from_int(0i64); n_samples];
    for r in 0..n_samples {
        let yi = host_to_f64(y_host[r]); // already F-exact
        y_centered[r] = f64_to_host::<F>(yi - y_mean_f);
    }

    let x_c_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_centered);
    let y_c_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y_centered);

    // --- 2. Penalty mapping (_coordinate_descent.py:781-782 / Pitfall 1): the
    //        un-normalized penalties carry the n_samples scaling. l1_reg = α·l1_ratio·n,
    //        l2_reg = α·(1−l1_ratio)·n. (Lasso passes l1_ratio = 1 → l2_reg = 0.) ---
    let l1_reg = alpha * l1_ratio * n_samples as f64;
    let l2_reg = alpha * (1.0 - l1_ratio) * n_samples as f64;

    // --- 3. Solve the centered CD problem with the validated 05-05 primitive. A
    //        non-convergent run surfaces as AlgoError::NotConverged (the primitive
    //        caps at max_iter and never emits silent NaN). ---
    let coef = cd_solve::<F>(
        pool,
        &x_c_dev,
        &y_c_dev,
        n_samples,
        n_features,
        l1_reg,
        l2_reg,
        tol,
        max_iter,
    )
    .map_err(|e| map_cd_error(e, estimator, max_iter))?;

    // --- 4. Recover intercept_ = ȳ − x̄·coef_ (D-13, unpenalized — outside the
    //        centered penalized system, exactly like ridge.rs D-05). 0 when not
    //        fitting an intercept. ---
    let coef_host = coef.to_host(pool);
    let intercept = if fit_intercept {
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

    // --- 5. Release the centered-design scratch; return device-resident state. ---
    x_c_dev.release_into(pool);
    y_c_dev.release_into(pool);

    Ok((coef, intercept_dev))
}

/// Map a `cd_solve` [`PrimError`] to the estimator-facing [`AlgoError`]. A
/// primitive convergence failure becomes [`AlgoError::NotConverged`] (carrying
/// the `max_iter` cap the caller can raise); every other prim failure is wrapped
/// transparently via `#[from]`.
fn map_cd_error(e: PrimError, estimator: &'static str, max_iter: usize) -> AlgoError {
    match e {
        PrimError::NotConverged { .. } => AlgoError::NotConverged {
            estimator,
            max_iter,
        },
        other => AlgoError::Prim(other),
    }
}

/// Round an `f64` to the WORKING precision of `F` (round-trip through f32 when
/// `F = f32`, identity when `F = f64`), returned as an f64 that is exactly
/// representable in `F` (WR-04 — match sklearn's single working-dtype centering).
fn narrow_to_f<F: Pod>(v: f64) -> f64 {
    match size_of::<F>() {
        4 => v as f32 as f64,
        8 => v,
        _ => unreachable!("coordinate-descent estimators are f32/f64 only"),
    }
}
