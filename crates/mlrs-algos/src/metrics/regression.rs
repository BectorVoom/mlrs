//! Regression metrics (METR-REG-01..03, extended to sklearn's FULL parameter
//! surface by METR-PARAM-01). Inputs are ROW-MAJOR `n_samples × n_outputs`
//! (`n_outputs = 1` is the single-output case), matching the C-order `ravel()`
//! the Python shim performs. Generic over the input float `F` (`f32`/`f64`),
//! but every sum accumulates in `f64` regardless of `F` (the
//! `covariance::empirical_covariance` f64-accumulate-then-cast precedent,
//! SPEC §3/§4).
//!
//! The `multioutput` reduction ([`MultiOutput`]) and `r2_score`'s
//! `force_finite` are implemented here rather than in the shim so the Rust
//! surface is usable standalone AND so the per-output loop stays a single pass
//! over the data (the shim would otherwise have to ship `n_outputs` separate
//! calls, each re-walking `sample_weight`).
//!
//! Tests live in `crates/mlrs-algos/tests/metrics_regression_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use mlrs_core::host_to_f64;

use super::{validate_weight, MetricError, MetricOut, MultiOutput};

/// Split a row-major `n_samples × n_outputs` buffer pair into its validated
/// geometry, or fail. Shared by all three metrics so the shape contract is
/// stated exactly once.
fn geometry<F>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
) -> Result<usize, MetricError> {
    if y_true.len() != y_pred.len() {
        return Err(MetricError::LengthMismatch);
    }
    if n_outputs == 0 || y_true.len() % n_outputs != 0 {
        return Err(MetricError::BadShape);
    }
    let n_samples = y_true.len() / n_outputs;
    validate_weight(n_samples, sample_weight)?;
    Ok(n_samples)
}

/// The total sample weight, or `Err(ZeroWeightSum)` when it is zero.
///
/// A zero weight-total (all-zero `sample_weight`, or empty input with unit
/// weights) makes every weighted mean below undefined. sklearn's `np.average`
/// raises `ZeroDivisionError("Weights sum to zero, can't be normalized")`;
/// return a typed error rather than the silent `0.0/0.0 = NaN` an earlier
/// version produced. `validate_weight` already rejected negative/NaN weights,
/// so a non-negative finite total is `0.0` iff every weight is `0.0`.
fn weight_total(n_samples: usize, sample_weight: Option<&[f64]>) -> Result<f64, MetricError> {
    let total = match sample_weight {
        Some(sw) => sw.iter().sum::<f64>(),
        None => n_samples as f64,
    };
    if total == 0.0 {
        return Err(MetricError::ZeroWeightSum);
    }
    Ok(total)
}

/// Per-output weighted mean of `term(sample, output)` — the shared shape of
/// `mean_squared_error` (`term = squared error`) and `mean_absolute_error`
/// (`term = absolute error`), and of `r2_score`'s `mean_true`.
///
/// One pass over the row-major buffer, accumulating `n_outputs` partial sums,
/// so the memory-access order matches the layout (a per-output outer loop would
/// stride by `n_outputs` and re-read the whole buffer once per column).
fn weighted_column_means(
    n_samples: usize,
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    term: impl Fn(usize, usize) -> f64,
) -> Result<Vec<f64>, MetricError> {
    let total = weight_total(n_samples, sample_weight)?;
    let mut sums = vec![0.0f64; n_outputs];
    for i in 0..n_samples {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        for (j, sum) in sums.iter_mut().enumerate() {
            *sum += w * term(i, j);
        }
    }
    for sum in sums.iter_mut() {
        *sum /= total;
    }
    Ok(sums)
}

/// Reduce a per-output score vector over `multioutput`, with an optional
/// per-output weight vector for [`MultiOutput::VarianceWeighted`]
/// (`r2_score`'s denominator).
///
/// `variance_weights` is `None` for the error metrics, which is what makes
/// `MultiOutput::VarianceWeighted` an error there rather than a silently
/// different reduction (sklearn 1.9.0 rejects the string for
/// `mean_squared_error`/`mean_absolute_error`).
fn reduce(
    scores: Vec<f64>,
    multioutput: MultiOutput<'_>,
    variance_weights: Option<&[f64]>,
) -> Result<MetricOut, MetricError> {
    let weighted_mean = |scores: &[f64], weights: Option<&[f64]>| -> Result<f64, MetricError> {
        match weights {
            None => Ok(scores.iter().sum::<f64>() / scores.len() as f64),
            Some(w) => {
                if w.len() != scores.len() {
                    return Err(MetricError::BadMultiOutputWeights);
                }
                let total: f64 = w.iter().sum();
                if total == 0.0 {
                    return Err(MetricError::ZeroWeightSum);
                }
                Ok(scores
                    .iter()
                    .zip(w.iter())
                    .map(|(&s, &wj)| s * wj)
                    .sum::<f64>()
                    / total)
            }
        }
    };

    match multioutput {
        MultiOutput::RawValues => Ok(MetricOut::Raw(scores)),
        MultiOutput::UniformAverage => Ok(MetricOut::Scalar(weighted_mean(&scores, None)?)),
        MultiOutput::Weights(w) => Ok(MetricOut::Scalar(weighted_mean(&scores, Some(w))?)),
        MultiOutput::VarianceWeighted => {
            let denominator = variance_weights.ok_or(MetricError::UnsupportedMultiOutput)?;
            // sklearn: when EVERY denominator is zero (all outputs constant, or
            // a 1-sample input) `np.average` would divide by zero, so it falls
            // back to UNIFORM weights rather than erroring.
            let weights = if denominator.iter().all(|&d| d == 0.0) {
                None
            } else {
                Some(denominator)
            };
            Ok(MetricOut::Scalar(weighted_mean(&scores, weights)?))
        }
    }
}

/// `r2 = 1 - ss_res/ss_tot` per output (`ss_res = Σ w_i*(y_true-y_pred)²`,
/// `ss_tot = Σ w_i*(y_true - weighted_mean(y_true))²`), reduced over
/// `multioutput`.
///
/// `force_finite` (sklearn ≥1.1, METR-PARAM-01) selects the degenerate-case
/// policy, matching `sklearn.metrics._regression._assemble_fraction_of_explained_deviance`
/// exactly:
///
/// | `ss_res` | `ss_tot` | `force_finite = true` | `force_finite = false` |
/// |---|---|---|---|
/// | `0`  | `0`  | `1.0` (perfect prediction of a constant target) | `NaN` (`1 - 0/0`) |
/// | `≠0` | `0`  | `0.0`                                           | `-inf` (`1 - x/0`) |
/// | `0`  | `≠0` | `1.0`                                           | `1.0` |
/// | `≠0` | `≠0` | `1 - ss_res/ss_tot`                             | same |
///
/// An input with fewer than 2 samples returns `Scalar(NaN)` for EVERY
/// `multioutput` (including `raw_values`) — sklearn returns the bare
/// `float("nan")` from its early guard before the reduction runs, with an
/// `UndefinedMetricWarning` the Python shim re-emits.
///
/// Returns `Err(MetricError::LengthMismatch)`/`Err(MetricError::InvalidWeight)`
/// on a bad `sample_weight` and `Err(MetricError::BadShape)` on a buffer whose
/// length is not a multiple of `n_outputs` — no panic.
pub fn r2_score<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    multioutput: MultiOutput<'_>,
    force_finite: bool,
) -> Result<MetricOut, MetricError> {
    let n_samples = geometry(y_true, y_pred, n_outputs, sample_weight)?;
    if n_samples < 2 {
        // Matches sklearn's own early return (an UndefinedMetricWarning + a
        // scalar NaN), which fires BEFORE the multioutput reduction.
        return Ok(MetricOut::Scalar(f64::NAN));
    }

    let mean_true = weighted_column_means(n_samples, n_outputs, sample_weight, |i, j| {
        host_to_f64(y_true[i * n_outputs + j])
    })?;

    let mut ss_res = vec![0.0f64; n_outputs];
    let mut ss_tot = vec![0.0f64; n_outputs];
    for i in 0..n_samples {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        for j in 0..n_outputs {
            let t = host_to_f64(y_true[i * n_outputs + j]);
            let p = host_to_f64(y_pred[i * n_outputs + j]);
            ss_res[j] += w * (t - p) * (t - p);
            ss_tot[j] += w * (t - mean_true[j]) * (t - mean_true[j]);
        }
    }

    let scores: Vec<f64> = (0..n_outputs)
        .map(|j| {
            if !force_finite {
                // The raw formula, NaN/-inf and all (sklearn's
                // `force_finite=False` branch).
                1.0 - ss_res[j] / ss_tot[j]
            } else if ss_res[j] == 0.0 {
                // Perfect prediction — 1.0 even when ss_tot is also zero.
                1.0
            } else if ss_tot[j] == 0.0 {
                // Constant target, imperfect prediction: 0.0 rather than -inf.
                0.0
            } else {
                1.0 - ss_res[j] / ss_tot[j]
            }
        })
        .collect();

    reduce(scores, multioutput, Some(&ss_tot))
}

/// `mse_j = Σ w_i*(y_true_ij-y_pred_ij)² / Σ w_i` per output, reduced over
/// `multioutput`. MSE ONLY — no `squared` parameter (sklearn ≥1.4 removed
/// `squared=False`; RMSE is the separate `root_mean_squared_error`, out of
/// scope here).
///
/// [`MultiOutput::VarianceWeighted`] returns
/// `Err(MetricError::UnsupportedMultiOutput)`, matching sklearn 1.9.0's own
/// rejection of that string for this function.
pub fn mean_squared_error<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    multioutput: MultiOutput<'_>,
) -> Result<MetricOut, MetricError> {
    let n_samples = geometry(y_true, y_pred, n_outputs, sample_weight)?;
    let scores = weighted_column_means(n_samples, n_outputs, sample_weight, |i, j| {
        let t = host_to_f64(y_true[i * n_outputs + j]);
        let p = host_to_f64(y_pred[i * n_outputs + j]);
        (t - p) * (t - p)
    })?;
    reduce(scores, multioutput, None)
}

/// `mae_j = Σ w_i*|y_true_ij-y_pred_ij| / Σ w_i` per output, reduced over
/// `multioutput`.
///
/// [`MultiOutput::VarianceWeighted`] returns
/// `Err(MetricError::UnsupportedMultiOutput)` (same rationale as
/// [`mean_squared_error`]).
pub fn mean_absolute_error<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    multioutput: MultiOutput<'_>,
) -> Result<MetricOut, MetricError> {
    let n_samples = geometry(y_true, y_pred, n_outputs, sample_weight)?;
    let scores = weighted_column_means(n_samples, n_outputs, sample_weight, |i, j| {
        let t = host_to_f64(y_true[i * n_outputs + j]);
        let p = host_to_f64(y_pred[i * n_outputs + j]);
        (t - p).abs()
    })?;
    reduce(scores, multioutput, None)
}
