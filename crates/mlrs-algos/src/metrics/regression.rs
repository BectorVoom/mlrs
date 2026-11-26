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

// ==================== The rest of sklearn's regression surface ====================
//
// Same contract as the three metrics above: row-major `n_samples × n_outputs`
// input, f64 accumulation, one pass per statistic, and the `multioutput`
// reduction in [`reduce`].

/// sklearn's `_assemble_fraction_of_explained_deviance`: `1 - num/den` per
/// output with the `force_finite` table [`r2_score`] documents, reduced over
/// `multioutput` (`VarianceWeighted` weights by `den`, falling back to uniform
/// when every `den` is zero).
fn assemble_explained(
    num: &[f64],
    den: &[f64],
    multioutput: MultiOutput<'_>,
    force_finite: bool,
) -> Result<MetricOut, MetricError> {
    let scores = num
        .iter()
        .zip(den)
        .map(|(&n, &d)| {
            if !force_finite {
                1.0 - n / d
            } else if n == 0.0 {
                1.0
            } else if d == 0.0 {
                0.0
            } else {
                1.0 - n / d
            }
        })
        .collect();
    reduce(scores, multioutput, Some(den))
}

/// `explained_variance_score`: `1 - Var(y_true - y_pred) / Var(y_true)` per
/// output (both weighted variances), with `r2_score`'s `force_finite` policy.
/// Unlike `r2_score`, a bias in the predictions is NOT penalized — the
/// residual's own mean is subtracted.
pub fn explained_variance_score<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    multioutput: MultiOutput<'_>,
    force_finite: bool,
) -> Result<MetricOut, MetricError> {
    let n = geometry(y_true, y_pred, n_outputs, sample_weight)?;
    let t = |i: usize, j: usize| host_to_f64(y_true[i * n_outputs + j]);
    let p = |i: usize, j: usize| host_to_f64(y_pred[i * n_outputs + j]);
    let diff_mean = weighted_column_means(n, n_outputs, sample_weight, |i, j| t(i, j) - p(i, j))?;
    let true_mean = weighted_column_means(n, n_outputs, sample_weight, t)?;
    let num = weighted_column_means(n, n_outputs, sample_weight, |i, j| {
        let d = t(i, j) - p(i, j) - diff_mean[j];
        d * d
    })?;
    let den = weighted_column_means(n, n_outputs, sample_weight, |i, j| {
        let d = t(i, j) - true_mean[j];
        d * d
    })?;
    assemble_explained(&num, &den, multioutput, force_finite)
}

/// `max_error`: the largest absolute residual. Single-output only
/// (`Err(UnsupportedTarget)` otherwise, sklearn's "Multioutput not supported").
pub fn max_error<F: Pod>(y_true: &[F], y_pred: &[F], n_outputs: usize) -> Result<f64, MetricError> {
    geometry(y_true, y_pred, n_outputs, None)?;
    if n_outputs != 1 {
        return Err(MetricError::UnsupportedTarget);
    }
    Ok(y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (host_to_f64(t) - host_to_f64(p)).abs())
        .fold(0.0, f64::max))
}

/// `mean_absolute_percentage_error`: `|y_pred - y_true| / max(|y_true|, eps)`
/// averaged per output (`eps` = f64 machine epsilon, so a zero target gives a
/// huge but finite term, exactly as sklearn).
pub fn mean_absolute_percentage_error<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    multioutput: MultiOutput<'_>,
) -> Result<MetricOut, MetricError> {
    let n = geometry(y_true, y_pred, n_outputs, sample_weight)?;
    let scores = weighted_column_means(n, n_outputs, sample_weight, |i, j| {
        let t = host_to_f64(y_true[i * n_outputs + j]);
        let p = host_to_f64(y_pred[i * n_outputs + j]);
        (p - t).abs() / t.abs().max(f64::EPSILON)
    })?;
    reduce(scores, multioutput, None)
}

/// Per-output mean squared error of `f(y_true)` vs `f(y_pred)`, the shared
/// body of the (root) mean squared (log) errors.
fn column_mse<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    f: impl Fn(f64) -> f64,
) -> Result<Vec<f64>, MetricError> {
    let n = geometry(y_true, y_pred, n_outputs, sample_weight)?;
    weighted_column_means(n, n_outputs, sample_weight, |i, j| {
        let d = f(host_to_f64(y_true[i * n_outputs + j])) - f(host_to_f64(y_pred[i * n_outputs + j]));
        d * d
    })
}

/// `Err(DomainError)` when any target or prediction is `<= -1` (the log1p
/// metrics' domain).
fn check_log1p_domain<F: Pod>(y_true: &[F], y_pred: &[F]) -> Result<(), MetricError> {
    if y_true.iter().chain(y_pred).any(|&v| host_to_f64(v) <= -1.0) {
        return Err(MetricError::DomainError);
    }
    Ok(())
}

/// `root_mean_squared_error`: the per-output square root of the MSE, THEN
/// reduced over `multioutput` (so `uniform_average` is the mean of the RMSEs,
/// not the root of the mean MSE — sklearn's definition).
pub fn root_mean_squared_error<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    multioutput: MultiOutput<'_>,
) -> Result<MetricOut, MetricError> {
    let mse = column_mse(y_true, y_pred, n_outputs, sample_weight, |v| v)?;
    reduce(mse.into_iter().map(f64::sqrt).collect(), multioutput, None)
}

/// `mean_squared_log_error`: the MSE of `log1p(y_true)` vs `log1p(y_pred)`.
/// `Err(DomainError)` for any value `<= -1`.
pub fn mean_squared_log_error<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    multioutput: MultiOutput<'_>,
) -> Result<MetricOut, MetricError> {
    check_log1p_domain(y_true, y_pred)?;
    let mse = column_mse(y_true, y_pred, n_outputs, sample_weight, f64::ln_1p)?;
    reduce(mse, multioutput, None)
}

/// `root_mean_squared_log_error`: the per-output root of
/// [`mean_squared_log_error`], then reduced.
pub fn root_mean_squared_log_error<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    multioutput: MultiOutput<'_>,
) -> Result<MetricOut, MetricError> {
    check_log1p_domain(y_true, y_pred)?;
    let mse = column_mse(y_true, y_pred, n_outputs, sample_weight, f64::ln_1p)?;
    reduce(mse.into_iter().map(f64::sqrt).collect(), multioutput, None)
}

/// sklearn's `_weighted_percentile(values, weights, percentile_rank, average)`
/// for ONE column, reproduced step for step (it is not the textbook weighted
/// quantile, and the differences are observable):
///
/// 1. sort by value; NaN entries (sorted last) get zero weight;
/// 2. `cdf = cumsum(sorted weights)`, target `= rank/100 * cdf[-1]` (nudged off
///    zero with `nextafter` so rank 0 picks the first POSITIVE-weight value);
/// 3. index = `searchsorted(cdf, target, 'left')`, clipped;
/// 4. with `average = true`, if the cdf lands EXACTLY on the target (within
///    f64 epsilon) the result is the mean of that value and the next
///    positive-weight one.
///
/// Returns NaN when every weight is zero (sklearn's own early return).
pub(crate) fn weighted_percentile(values: &[f64], weights: &[f64], rank: f64, average: bool) -> f64 {
    if weights.iter().all(|&w| w == 0.0) {
        return f64::NAN;
    }
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let sorted_w: Vec<f64> = idx
        .iter()
        .map(|&i| if values[i].is_nan() { 0.0 } else { weights[i] })
        .collect();
    let mut cdf = Vec::with_capacity(n);
    let mut run = 0.0f64;
    for &w in &sorted_w {
        run += w;
        cdf.push(run);
    }
    let mut target = rank / 100.0 * cdf[n - 1];
    if target == 0.0 {
        target = f64::from_bits(1); // nextafter(0, 1)
    }
    let max_idx = n - 1;
    let pos = cdf.partition_point(|&c| c < target).min(max_idx);
    let v = values[idx[pos]];
    if !average {
        return v;
    }
    let above = cdf[pos] - target > f64::EPSILON;
    if above {
        return v;
    }
    let mut next = (pos + 1).min(max_idx);
    if sorted_w[next] == 0.0 {
        let cdf_val = cdf[pos];
        let right = cdf.partition_point(|&c| c <= cdf_val);
        next = if right > max_idx { pos } else { right };
    }
    (v + values[idx[next]]) / 2.0
}

/// numpy's `median` (mean of the two middle values for an even count; NaN if
/// any value is NaN). An O(n) selection, not a sort — numpy's `median` is a
/// partition too, and a full sort measured 2x slower than it at n = 1e6.
fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 || values.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mid = n / 2;
    let (lower, upper, _) = values.select_nth_unstable_by(mid, f64::total_cmp);
    let hi = *upper;
    if n % 2 == 1 {
        hi
    } else {
        // The lower middle is the largest element of the left partition.
        let lo = lower.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo + hi) / 2.0
    }
}

/// Column `j` of a row-major buffer, as f64.
fn column<F: Pod>(y: &[F], n: usize, n_outputs: usize, j: usize) -> Vec<f64> {
    (0..n).map(|i| host_to_f64(y[i * n_outputs + j])).collect()
}

/// `median_absolute_error`: the per-output median of `|y_pred - y_true|` —
/// numpy's median unweighted, sklearn's averaged weighted 50th percentile with
/// `sample_weight`.
pub fn median_absolute_error<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    multioutput: MultiOutput<'_>,
) -> Result<MetricOut, MetricError> {
    let n = geometry(y_true, y_pred, n_outputs, sample_weight)?;
    let scores = (0..n_outputs)
        .map(|j| {
            let mut abs: Vec<f64> = (0..n)
                .map(|i| (host_to_f64(y_pred[i * n_outputs + j]) - host_to_f64(y_true[i * n_outputs + j])).abs())
                .collect();
            match sample_weight {
                None => median(&mut abs),
                Some(w) => weighted_percentile(&abs, w, 50.0, true),
            }
        })
        .collect();
    reduce(scores, multioutput, None)
}

/// The pinball (quantile) loss of one residual: `alpha·d` above, `(alpha-1)·d`
/// below.
fn pinball(t: f64, p: f64, alpha: f64) -> f64 {
    let d = t - p;
    let sign = if d >= 0.0 { 1.0 } else { 0.0 };
    alpha * sign * d - (1.0 - alpha) * (1.0 - sign) * d
}

/// `mean_pinball_loss` at quantile `alpha ∈ [0, 1]`.
pub fn mean_pinball_loss<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    alpha: f64,
    multioutput: MultiOutput<'_>,
) -> Result<MetricOut, MetricError> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(MetricError::InvalidParameter);
    }
    let n = geometry(y_true, y_pred, n_outputs, sample_weight)?;
    let scores = weighted_column_means(n, n_outputs, sample_weight, |i, j| {
        pinball(host_to_f64(y_true[i * n_outputs + j]), host_to_f64(y_pred[i * n_outputs + j]), alpha)
    })?;
    reduce(scores, multioutput, None)
}

/// `d2_pinball_score`: `1 - pinball(y, ŷ) / pinball(y, q_alpha(y))` per output,
/// `q_alpha` being the (weighted, averaged) `alpha`-quantile of `y_true` —
/// the best constant model. Always `force_finite`. Fewer than 2 samples give a
/// scalar NaN (sklearn's early return, warned by the shim).
pub fn d2_pinball_score<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    alpha: f64,
    multioutput: MultiOutput<'_>,
) -> Result<MetricOut, MetricError> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(MetricError::InvalidParameter);
    }
    let n = geometry(y_true, y_pred, n_outputs, sample_weight)?;
    if n < 2 {
        return Ok(MetricOut::Scalar(f64::NAN));
    }
    let num = weighted_column_means(n, n_outputs, sample_weight, |i, j| {
        pinball(host_to_f64(y_true[i * n_outputs + j]), host_to_f64(y_pred[i * n_outputs + j]), alpha)
    })?;
    let ones = vec![1.0f64; n];
    let w = sample_weight.unwrap_or(&ones);
    let quantile: Vec<f64> = (0..n_outputs)
        .map(|j| weighted_percentile(&column(y_true, n, n_outputs, j), w, alpha * 100.0, true))
        .collect();
    let den = weighted_column_means(n, n_outputs, sample_weight, |i, j| {
        pinball(host_to_f64(y_true[i * n_outputs + j]), quantile[j], alpha)
    })?;
    assemble_explained(&num, &den, multioutput, true)
}

/// `d2_absolute_error_score`: [`d2_pinball_score`] at `alpha = 0.5`.
pub fn d2_absolute_error_score<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    multioutput: MultiOutput<'_>,
) -> Result<MetricOut, MetricError> {
    d2_pinball_score(y_true, y_pred, n_outputs, sample_weight, 0.5, multioutput)
}

/// Validate the Tweedie deviance domain for `power`, exactly sklearn's rules:
/// `p < 0` needs `y_pred > 0`; `p = 0` anything; `1 <= p < 2` needs `y_true >=
/// 0` and `y_pred > 0`; `p >= 2` needs both `> 0`; `0 < p < 1` has no
/// distribution and is `Err(InvalidParameter)`.
fn check_tweedie_domain(y_true: &[f64], y_pred: &[f64], power: f64) -> Result<(), MetricError> {
    let bad = if power < 0.0 {
        y_pred.iter().any(|&p| p <= 0.0)
    } else if power == 0.0 {
        false
    } else if (1.0..2.0).contains(&power) {
        y_true.iter().any(|&t| t < 0.0) || y_pred.iter().any(|&p| p <= 0.0)
    } else if power >= 2.0 {
        y_true.iter().any(|&t| t <= 0.0) || y_pred.iter().any(|&p| p <= 0.0)
    } else {
        return Err(MetricError::InvalidParameter);
    };
    if bad {
        return Err(MetricError::DomainError);
    }
    Ok(())
}

/// The unit Tweedie deviance of one `(y, mu)` pair (sklearn's
/// `_mean_tweedie_deviance` term, branch for branch: `xlogy` makes the
/// Poisson term `0` at `y = 0`).
fn tweedie_unit(t: f64, mu: f64, p: f64) -> f64 {
    if p < 0.0 {
        let tp = if t > 0.0 { t } else { 0.0 };
        2.0 * (tp.powf(2.0 - p) / ((1.0 - p) * (2.0 - p)) - t * mu.powf(1.0 - p) / (1.0 - p)
            + mu.powf(2.0 - p) / (2.0 - p))
    } else if p == 0.0 {
        (t - mu) * (t - mu)
    } else if p == 1.0 {
        let xlogy = if t == 0.0 { 0.0 } else { t * (t / mu).ln() };
        2.0 * (xlogy - t + mu)
    } else if p == 2.0 {
        2.0 * ((mu / t).ln() + t / mu - 1.0)
    } else {
        2.0 * (t.powf(2.0 - p) / ((1.0 - p) * (2.0 - p)) - t * mu.powf(1.0 - p) / (1.0 - p)
            + mu.powf(2.0 - p) / (2.0 - p))
    }
}

fn mean_tweedie_f64(t: &[f64], mu: &[f64], sample_weight: Option<&[f64]>, power: f64) -> Result<f64, MetricError> {
    let total = weight_total(t.len(), sample_weight)?;
    let mut sum = 0.0f64;
    for i in 0..t.len() {
        sum += sample_weight.map_or(1.0, |sw| sw[i]) * tweedie_unit(t[i], mu[i], power);
    }
    Ok(sum / total)
}

/// `mean_tweedie_deviance` (single-output). `power = 0` is the MSE, `1` the
/// Poisson and `2` the Gamma deviance.
pub fn mean_tweedie_deviance<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    power: f64,
) -> Result<f64, MetricError> {
    geometry(y_true, y_pred, n_outputs, sample_weight)?;
    if n_outputs != 1 {
        return Err(MetricError::UnsupportedTarget);
    }
    let t: Vec<f64> = y_true.iter().map(|&v| host_to_f64(v)).collect();
    let p: Vec<f64> = y_pred.iter().map(|&v| host_to_f64(v)).collect();
    check_tweedie_domain(&t, &p, power)?;
    mean_tweedie_f64(&t, &p, sample_weight, power)
}

/// `d2_tweedie_score`: `1 - D(y, ŷ) / D(y, ȳ)`, `ȳ` the weighted mean target.
/// Fewer than 2 samples give NaN (the shim warns).
pub fn d2_tweedie_score<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    n_outputs: usize,
    sample_weight: Option<&[f64]>,
    power: f64,
) -> Result<f64, MetricError> {
    let n = geometry(y_true, y_pred, n_outputs, sample_weight)?;
    if n_outputs != 1 {
        return Err(MetricError::UnsupportedTarget);
    }
    if n < 2 {
        return Ok(f64::NAN);
    }
    let t: Vec<f64> = y_true.iter().map(|&v| host_to_f64(v)).collect();
    let p: Vec<f64> = y_pred.iter().map(|&v| host_to_f64(v)).collect();
    check_tweedie_domain(&t, &p, power)?;
    let num = mean_tweedie_f64(&t, &p, sample_weight, power)?;
    let total = weight_total(n, sample_weight)?;
    let mean = (0..n).map(|i| sample_weight.map_or(1.0, |sw| sw[i]) * t[i]).sum::<f64>() / total;
    let den = mean_tweedie_f64(&t, &vec![mean; n], sample_weight, power)?;
    Ok(1.0 - num / den)
}
