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

/// Column-wise `(sum, sum_of_squares)` over `x` (`n × d`, row-major), as host
/// `f64` (RESEARCH Pitfall 4 — accumulate in `f64` regardless of `F` so an f32
/// caller does not lose the mean/variance to catastrophic cancellation).
///
/// CR-01 precedent (`covariance.rs`/`pca.rs`): the reduction is an INTERNAL
/// implementation detail of `fit`, always run on the always-portable
/// `ReducePath::Shared` path (never plane-gated to `None`).
pub(crate) fn column_sum_sumsq<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
) -> Result<(Vec<f64>, Vec<f64>), AlgoError>
where
    F: Float + CubeElement + Pod,
{
    let sum_dev = column_reduce::<F>(pool, x, n, d, ScalarOp::Sum, ReducePath::Shared)?
        .ok_or(AlgoError::Prim(PrimError::InternalNone {
            operand: "column_reduce",
            context: "ScalarOp::Sum",
        }))?;
    let sumsq_dev = column_reduce::<F>(pool, x, n, d, ScalarOp::SumSq, ReducePath::Shared)?
        .ok_or(AlgoError::Prim(PrimError::InternalNone {
            operand: "column_reduce",
            context: "ScalarOp::SumSq",
        }))?;
    let sum64: Vec<f64> = sum_dev.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    let sumsq64: Vec<f64> = sumsq_dev
        .to_host(pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    sum_dev.release_into(pool);
    sumsq_dev.release_into(pool);
    Ok((sum64, sumsq64))
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

/// `sklearn._handle_zeros_in_scale`: replace a near-zero scale with `1.0` so a
/// constant column divides by `1`, not `0` (PREP-01's degenerate-column gate).
pub(crate) fn handle_zeros_in_scale(scale: &mut [f64], eps: f64) {
    for s in scale.iter_mut() {
        if s.abs() < eps {
            *s = 1.0;
        }
    }
}

/// Apply the per-column affine map `out[r, c] = x[r, c] * scale[c] + shift[c]`
/// on the host (RESEARCH Pitfall 4: accumulate in `f64`), re-uploading the
/// result — the same single host-materialize pass `pca.rs`'s column centering
/// uses, generalized to a multiply-then-add (D-05: `Transform` is a one-shot
/// terminal materialize, not a mid-pipeline round-trip).
pub(crate) fn affine_columns_host<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    d: usize,
    scale: &[f64],
    shift: &[f64],
) -> DeviceArray<ActiveRuntime, F>
where
    F: Float + CubeElement + Pod,
{
    let x_host = x.to_host(pool);
    let mut out = vec![F::from_int(0i64); n * d];
    for r in 0..n {
        for c in 0..d {
            let v = host_to_f64(x_host[r * d + c]) * scale[c] + shift[c];
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
