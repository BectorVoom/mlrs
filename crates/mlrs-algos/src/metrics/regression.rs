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

/// Shared weighted-mean/weight-total helper (TASK-12 Refactor step) reused
/// by `r2_score`/`mean_squared_error`/`mean_absolute_error` — each metric's
/// per-sample term differs (squared error, squared error + variance,
/// absolute error) enough that each writes its own direct f64-accumulation
/// loop below (matching the existing `empirical_covariance.rs` style)
/// rather than threading one generic closure through a shared fold.
fn weighted_mean_and_total<F: Pod>(y: &[F], sample_weight: Option<&[f64]>) -> (f64, f64) {
    let mut weight_total = 0.0f64;
    let mut weighted_sum = 0.0f64;
    for i in 0..y.len() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        weight_total += w;
        weighted_sum += w * host_to_f64(y[i]);
    }
    (weighted_sum / weight_total, weight_total)
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
pub fn r2_score<F: Pod>(y_true: &[F], y_pred: &[F], sample_weight: Option<&[f64]>) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "r2_score: length mismatch");
    let (mean_true, _) = weighted_mean_and_total(y_true, sample_weight);

    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;
    for i in 0..y_true.len() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        let t = host_to_f64(y_true[i]);
        let p = host_to_f64(y_pred[i]);
        ss_res += w * (t - p) * (t - p);
        ss_tot += w * (t - mean_true) * (t - mean_true);
    }

    if ss_res == 0.0 {
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
    }
}

/// `mse = Σ w_i*(y_true_i-y_pred_i)² / Σ w_i`. MSE ONLY — no `squared`
/// parameter (SPEC §2 non-goal / §9 risk 1; sklearn ≥1.4 removed
/// `squared=False`, RMSE is the separate `root_mean_squared_error`, out of
/// scope here).
pub fn mean_squared_error<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    sample_weight: Option<&[f64]>,
) -> f64 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "mean_squared_error: length mismatch"
    );
    let mut weight_total = 0.0f64;
    let mut acc = 0.0f64;
    for i in 0..y_true.len() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        weight_total += w;
        let t = host_to_f64(y_true[i]);
        let p = host_to_f64(y_pred[i]);
        acc += w * (t - p) * (t - p);
    }
    acc / weight_total
}

/// `mae = Σ w_i*|y_true_i-y_pred_i| / Σ w_i`.
pub fn mean_absolute_error<F: Pod>(
    y_true: &[F],
    y_pred: &[F],
    sample_weight: Option<&[f64]>,
) -> f64 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "mean_absolute_error: length mismatch"
    );
    let mut weight_total = 0.0f64;
    let mut acc = 0.0f64;
    for i in 0..y_true.len() {
        let w = sample_weight.map_or(1.0, |sw| sw[i]);
        weight_total += w;
        let t = host_to_f64(y_true[i]);
        let p = host_to_f64(y_pred[i]);
        acc += w * (t - p).abs();
    }
    acc / weight_total
}
