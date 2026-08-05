//! Shared host-arithmetic helpers for the [`super`] scaler family (D-03 — no
//! `ScalerBase`, mirrors the `naive_bayes::nb_common` free-function precedent).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::AlgoError;

/// Column-wise `(mean, POPULATION variance)` over `x` (`n × d`, row-major), as
/// host `f64` (RESEARCH Pitfall 4 — accumulate in `f64` regardless of `F`).
///
/// TWO passes over ONE host materialization, deliberately NOT the one-pass
/// `E[x²] − mean²` identity over a `column_reduce` pair. Two independent
/// failures compound in that spelling:
///
/// - `column_reduce::<F>` accumulates in the ELEMENT type `F`
///   (`mlrs_kernels::reduce`'s `SharedMemory::<F>`), so an `f32` design's `Σx²`
///   is already rounded to `f32` before anything widens it. `host_to_f64`
///   afterwards preserves that error, it cannot undo it.
/// - The identity subtracts two `O(Σx²)`-sized quantities to leave an
///   `O(var)`-sized one, so a column whose offset is large relative to its
///   spread loses the answer to cancellation even with exact `f64` sums.
///
/// Together they are not academic. 10 000 rows of `N(1000, 1)` at `f32` give
/// `Σx² ≈ 1e10` carried at ~6e-8 relative — ±600 absolute — against a
/// `mean² = 1e6` from which a true variance of `1.0` has to survive. The result
/// comes out wrong by orders of magnitude, and when it clamps to `0` the
/// degenerate-column gate then sets `scale_ = 1` and `transform` silently
/// returns UNSCALED data. A mean-zero fixture (what the committed oracle blobs
/// use) exercises none of this.
///
/// sklearn's `_incremental_mean_and_var` sums with `dtype=np.float64` for the
/// same reason. The extra pass reads a buffer that is already host-resident,
/// and `transform` materializes the same design anyway
/// ([`affine_columns_host`]), so this keeps the module on one host arm rather
/// than adding one.
pub(crate) fn column_mean_var<F>(
    pool: &BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
) -> (Vec<f64>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let host = x.to_host(pool);
    let n64 = n as f64;

    let mut mean = vec![0.0f64; d];
    for row in host.chunks_exact(d) {
        for (m, &v) in mean.iter_mut().zip(row.iter()) {
            *m += host_to_f64(v);
        }
    }
    for m in mean.iter_mut() {
        *m /= n64;
    }

    let mut var = vec![0.0f64; d];
    for row in host.chunks_exact(d) {
        for ((v, &xv), &m) in var.iter_mut().zip(row.iter()).zip(mean.iter()) {
            let dev = host_to_f64(xv) - m;
            *v += dev * dev;
        }
    }
    // `max(0.0)`: an exactly-constant column can leave a `-0.0` here, and
    // sklearn's `var_` is never negative.
    for v in var.iter_mut() {
        *v = (*v / n64).max(0.0);
    }

    (mean, var)
}

/// Column-wise `(min, max)` over `x` (`n × d`, row-major), as host `f64`.
pub(crate) fn column_min_max<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
) -> Result<(Vec<f64>, Vec<f64>), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let min_dev = column_reduce::<F>(pool, x, n, d, ScalarOp::Min, ReducePath::Shared)?
        .ok_or(AlgoError::Prim(PrimError::InternalNone {
            operand: "column_reduce",
            context: "ScalarOp::Min",
        }))?;
    let max_dev = column_reduce::<F>(pool, x, n, d, ScalarOp::Max, ReducePath::Shared)?
        .ok_or(AlgoError::Prim(PrimError::InternalNone {
            operand: "column_reduce",
            context: "ScalarOp::Max",
        }))?;
    let min64: Vec<f64> = min_dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let max64: Vec<f64> = max_dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    min_dev.release_into(pool);
    max_dev.release_into(pool);
    Ok((min64, max64))
}

/// Every column of `x` (`n × d`, row-major), materialized to host `f64`
/// (RobustScaler's median/quantile needs a per-column SORT, which
/// `column_reduce` cannot express — the ARIMA/BayesianRidge "inherently
/// sequential, no useful device parallelism" precedent applies here too).
pub(crate) fn columns_host_f64<F>(pool: &BufferPool<ActiveRuntime>, x: &DeviceArray<ActiveRuntime, F>, n: usize, d: usize) -> Vec<Vec<f64>>
where
    F: Float + CubeElement + Pod,
{
    let host = x.to_host(pool);
    (0..d)
        .map(|c| (0..n).map(|r| host_to_f64(host[r * d + c])).collect())
        .collect()
}

/// `10 * F::EPSILON` (the `sklearn.preprocessing._data._handle_zeros_in_scale`
/// threshold, per-dtype) below which a column scale is treated as a degenerate
/// zero (constant column) and replaced with `1.0` rather than dividing by it.
pub(crate) fn zeros_eps<F: CubeElement>() -> f64 {
    if std::mem::size_of::<F>() == 4 {
        10.0 * f32::EPSILON as f64
    } else {
        10.0 * f64::EPSILON
    }
}

/// `sklearn._handle_zeros_in_scale` with `constant_mask=None`: replace a
/// near-zero scale with `1.0` so a constant column divides by `1`, not `0`
/// (PREP-01's degenerate-column gate).
///
/// This is the DEFAULT sklearn path, which `MinMaxScaler`, `MaxAbsScaler`,
/// `RobustScaler` and `normalize` all take. `StandardScaler` does NOT — it
/// passes an explicit `constant_mask`; see [`handle_zeros_in_scale_masked`].
pub(crate) fn handle_zeros_in_scale(scale: &mut [f64], eps: f64) {
    for s in scale.iter_mut() {
        if s.abs() < eps {
            *s = 1.0;
        }
    }
}

/// `sklearn._handle_zeros_in_scale` with an explicit `constant_mask` — the
/// `StandardScaler` path, where the `10 · eps` test is NOT applied at all.
pub(crate) fn handle_zeros_in_scale_masked(scale: &mut [f64], constant_mask: &[bool]) {
    for (s, &constant) in scale.iter_mut().zip(constant_mask.iter()) {
        if constant {
            *s = 1.0;
        }
    }
}

/// `sklearn.preprocessing._data._is_constant_feature`: is a feature
/// indistinguishable from a constant one, given the round-off its own mean
/// injects into the variance?
///
/// ```text
/// var <= n·eps·var + (n·mean·eps)²
/// ```
///
/// This is NOT the `sqrt(var) < 10·eps` test [`handle_zeros_in_scale`] applies.
/// The difference is the `mean` term, and it is the whole point: variance is
/// computed by summing `n` squared deviations, so the noise floor scales with
/// the magnitude of the values, not with `1.0`. A column of 60 samples centred
/// near `1e8` with a true standard deviation of `1e-7` is constant to sklearn —
/// its bound is `(60·1e8·2.2e-16)² ≈ 1.7e-12`, above the `1e-14` variance —
/// while `sqrt(var) = 1e-7` sails past an absolute `2.2e-15` threshold. Taking
/// the absolute test there divides by `1e-7` and returns values of order `±1`
/// where sklearn returns order `1e-7`: a ~1.0 per-element divergence against a
/// 1e-5 contract, on exactly the near-constant columns the gate exists for.
///
/// `eps` is `f64::EPSILON` on BOTH widths, matching sklearn's comment that "in
/// scikit-learn, variance is always computed using float64 accumulators" — and
/// matching [`column_mean_var`], which does the same here.
pub(crate) fn is_constant_feature(var: f64, mean: f64, n_samples: usize) -> bool {
    let eps = f64::EPSILON;
    let n = n_samples as f64;
    let upper_bound = n * eps * var + (n * mean * eps) * (n * mean * eps);
    var <= upper_bound
}

/// Apply the per-column affine map `out[r, c] = x[r, c] * scale[c] + shift[c]`
/// on the host (RESEARCH Pitfall 4: accumulate in `f64`), re-uploading the
/// result — the same single host-materialize pass `pca.rs`'s column centering
/// uses, generalized to a multiply-then-add (D-05: `Transform` is a one-shot
/// terminal materialize, not a mid-pipeline round-trip).
///
/// `clamp` folds `MinMaxScaler(clip=True)`'s `feature_range` bound into the SAME
/// pass. It is a parameter rather than the caller's own follow-up loop because
/// that loop would run on the buffer this function just UPLOADED: a second
/// `to_host` plus a second `from_host` over the full `n × d` result, i.e. an
/// extra 64 MiB down and 64 MiB up on a 1 000 000 × 16 `f32` transform, to
/// apply an operation that costs nothing on the value already in a register.
pub(crate) fn affine_columns_host<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    scale: &[f64],
    shift: &[f64],
    clamp: Option<(f64, f64)>,
) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    let x_host = x.to_host(pool);
    let mut out = vec![F::from_int(0i64); n * d];
    for r in 0..n {
        for c in 0..d {
            let mut v = host_to_f64(x_host[r * d + c]) * scale[c] + shift[c];
            if let Some((lo, hi)) = clamp {
                v = v.clamp(lo, hi);
            }
            out[r * d + c] = f64_to_host::<F>(v);
        }
    }
    DeviceArray::from_host(pool, &out)
}

/// Inverse-normal CDF (quantile function) via Peter Acklam's rational
/// approximation (`|error| < 1.15e-9` over `(0, 1)`) — RobustScaler's
/// `unit_variance=True` needs `Φ⁻¹(q/100)` for an ARBITRARY `quantile_range`,
/// not just the default `(25, 75)` (whose `Φ⁻¹(0.75) − Φ⁻¹(0.25)` is a fixed
/// literal `1.3489795...`), so a closed-form literal cannot cover it.
pub(crate) fn norm_ppf(p: f64) -> f64 {
    // Coefficients for the rational approximations (Acklam, 2003).
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    let p_high = 1.0 - P_LOW;

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}
