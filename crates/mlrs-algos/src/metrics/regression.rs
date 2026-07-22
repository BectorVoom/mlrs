//! Regression metrics (METR-REG-01..03). Single-output only (1-D `y_true`/
//! `y_pred`) — no `multioutput` parameter anywhere in this module (SPEC §2
//! non-goal). Generic over the input float `F` (`f32`/`f64`), but every sum
//! accumulates in `f64` regardless of `F` (the `covariance::empirical_covariance`
//! f64-accumulate-then-cast precedent, SPEC §3/§4).
//!
//! Tests live in `crates/mlrs-algos/tests/metrics_regression_test.rs`
//! (AGENTS.md §2 — no in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use mlrs_core::host_to_f64;

use super::{validate_weight, MetricError};

/// Shared weighted-mean helper (TASK-12 Refactor step, code-review fix):
/// `Σ w_i * term(i) / Σ w_i`. Genuinely reused by all three regression
/// metrics — `r2_score`'s `mean_true` (`term = y_true`), `mean_squared_error`
/// (`term = squared error`), and `mean_absolute_error` (`term = absolute
/// error`) — via a per-element closure rather than three parallel
/// hand-rolled accumulation loops (the previous doc-comment claimed this
/// sharing without the code actually doing it; `mean_squared_error`/
/// `mean_absolute_error` each duplicated the same weight-total loop).
fn weighted_mean(
    len: usize,
    sample_weight: Option<&[f64]>,
    term: impl Fn(usize) -> f64,
) -> Result<f64, MetricError> {
    let mut weight_total = 0.0f64;
    let mut weighted_sum = 0.0f64;
    for i in 0..len {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        weight_total += w;
        weighted_sum += w * term(i);
    }
    // A zero weight-total (all-zero `sample_weight`, or empty input with unit
    // weights) makes the weighted mean undefined. sklearn raises
    // `ValueError("Sample weights must contain at least one non-zero
    // number.")`; return a typed error rather than the silent `0.0/0.0 = NaN`
    // the previous version produced (code-review fix). `validate_weight`
    // already rejected negative/NaN weights, so a non-negative finite total
    // is `0.0` iff every weight is `0.0`.
    if weight_total == 0.0 {
        return Err(MetricError::ZeroWeightSum);
    }
    Ok(weighted_sum / weight_total)
}

/// `r2 = 1 - ss_res/ss_tot` (`ss_res = Σ w_i*(y_true_i-y_pred_i)²`, `ss_tot =
/// Σ w_i*(y_true_i - weighted_mean(y_true))²`). Perfect prediction → `1.0`.
/// Constant `y_true` (`ss_tot == 0`) returns sklearn's ACTUAL pinned value
/// (empirically `0.0` for a non-exact-matching constant-target `y_pred` —
/// TASK-02's fixture, read from the real `scikit-learn==1.9.0` output, not
/// hand-derived, SPEC §5 REG note / §9 risk 5); an EXACT `ss_res == 0.0`
/// match (perfect prediction, including the constant-vs-constant case) is
/// handled first and always returns `1.0` before the `ss_tot == 0` branch is
/// consulted, matching sklearn's own precedence.
///
/// Returns `Err(MetricError::LengthMismatch)`/`Err(MetricError::InvalidWeight)`
/// on a bad `sample_weight` — no panic (code-review fix: a too-short
/// `sample_weight` previously indexed out of bounds and panicked, and a
/// too-long one was silently truncated with no error).
pub fn r2_score<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    if y_true.len() != y_pred.len() {
        return Err(MetricError::LengthMismatch);
    }
    validate_weight(y_true.len(), sample_weight)?;

    let mean_true = weighted_mean(y_true.len(), sample_weight, |i| host_to_f64(y_true[i]))?;

    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;
    for i in 0..y_true.len() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        let t = host_to_f64(y_true[i]);
        let p = host_to_f64(y_pred[i]);
        ss_res += w * (t - p) * (t - p);
        ss_tot += w * (t - mean_true) * (t - mean_true);
    }

    Ok(if ss_res == 0.0 {
        // Exact match (perfect prediction, or a constant target predicted
        // exactly) — sklearn returns 1.0 unconditionally here, before
        // consulting the ss_tot==0 branch.
        1.0
    } else if ss_tot == 0.0 {
        // Constant y_true with a NON-exact y_pred: sklearn's documented
        // (and empirically pinned, TASK-02) behavior is 0.0.
        0.0
    } else {
        1.0 - ss_res / ss_tot
    })
}

/// `mse = Σ w_i*(y_true_i-y_pred_i)² / Σ w_i`. MSE ONLY — no `squared`
/// parameter (SPEC §2 non-goal / §9 risk 1; sklearn ≥1.4 removed
/// `squared=False`, RMSE is the separate `root_mean_squared_error`, out of
/// scope here).
///
/// Returns `Err(MetricError::LengthMismatch)`/`Err(MetricError::InvalidWeight)`
/// on a bad `sample_weight` — no panic (code-review fix, same class of bug
/// as `r2_score`).
pub fn mean_squared_error<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    if y_true.len() != y_pred.len() {
        return Err(MetricError::LengthMismatch);
    }
    validate_weight(y_true.len(), sample_weight)?;

    weighted_mean(y_true.len(), sample_weight, |i| {
        let t = host_to_f64(y_true[i]);
        let p = host_to_f64(y_pred[i]);
        (t - p) * (t - p)
    })
}

/// `mae = Σ w_i*|y_true_i-y_pred_i| / Σ w_i`.
///
/// Returns `Err(MetricError::LengthMismatch)`/`Err(MetricError::InvalidWeight)`
/// on a bad `sample_weight` — no panic (code-review fix, same class of bug
/// as `r2_score`).
pub fn mean_absolute_error<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    sample_weight: Option<&[f64]>,
) -> Result<f64, MetricError> {
    if y_true.len() != y_pred.len() {
        return Err(MetricError::LengthMismatch);
    }
    validate_weight(y_true.len(), sample_weight)?;

    weighted_mean(y_true.len(), sample_weight, |i| {
        let t = host_to_f64(y_true[i]);
        let p = host_to_f64(y_pred[i]);
        (t - p).abs()
    })
}
