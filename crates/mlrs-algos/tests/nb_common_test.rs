//! Plan 11-01 Wave-0 — `nb_common` free-function standalone validation.
//!
//! Exercises the five NON-device NB free functions on hand-computed examples
//! (D-03 — the shared math is functions, not a base struct) PLUS the
//! `class_grouped_sum` GATHER launch witness under `--features cpu` (the
//! one-owner-per-`(class, feature)` reduce-prim composition — ROADMAP #1, the
//! cpu-launch gate). These tests are NOT `#[ignore]`: the free functions are
//! complete in Wave 0 (the estimators that CALL them are filled in Waves 1/2).
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use mlrs_algos::naive_bayes::nb_common::{
    accuracy_score, argmax_decode, argmin_decode, class_grouped_sum,
    class_grouped_sumsq, empirical_class_log_prior, log_sum_exp_normalize,
};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// `log_sum_exp_normalize` on a hand-computed 3-class row: proba sums to 1 and
/// `log_proba == joint_ll − lse`.
#[test]
fn log_sum_exp_normalize_three_class() {
    let joint_ll = [-1.0f64, -2.0, -3.0];
    let (proba, log_proba) = log_sum_exp_normalize(&joint_ll, 3);

    // Hand reference: lse = log(e^-1 + e^-2 + e^-3).
    let lse = (joint_ll.iter().map(|&v| v.exp()).sum::<f64>()).ln();
    for (i, &ll) in joint_ll.iter().enumerate() {
        assert!(
            (log_proba[i] - (ll - lse)).abs() < 1e-12,
            "log_proba[{i}] = {} != joint_ll - lse = {}",
            log_proba[i],
            ll - lse
        );
        assert!(
            (proba[i] - (ll - lse).exp()).abs() < 1e-12,
            "proba[{i}] mismatch"
        );
    }
    let sum: f64 = proba.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "proba must sum to 1, got {sum}");
}

/// The max-shift keeps large-magnitude joint LLs from overflowing; proba still
/// sums to 1.
#[test]
fn log_sum_exp_normalize_large_magnitude() {
    let joint_ll = [1000.0f64, 999.0, 998.0];
    let (proba, _log_proba) = log_sum_exp_normalize(&joint_ll, 3);
    let sum: f64 = proba.iter().sum();
    assert!(sum.is_finite(), "proba sum must be finite (no overflow)");
    assert!((sum - 1.0).abs() < 1e-12, "proba must sum to 1, got {sum}");
    // The largest LL gets the largest probability.
    assert!(proba[0] > proba[1] && proba[1] > proba[2]);
}

/// `empirical_class_log_prior`: a uniform `[10, 10]` yields `[ln 0.5, ln 0.5]`.
#[test]
fn empirical_class_log_prior_uniform() {
    let lp = empirical_class_log_prior(&[10.0, 10.0]);
    let half = 0.5f64.ln();
    assert!((lp[0] - half).abs() < 1e-12);
    assert!((lp[1] - half).abs() < 1e-12);

    // Skewed [30, 10] → [ln 0.75, ln 0.25].
    let lp2 = empirical_class_log_prior(&[30.0, 10.0]);
    assert!((lp2[0] - 0.75f64.ln()).abs() < 1e-12);
    assert!((lp2[1] - 0.25f64.ln()).abs() < 1e-12);
}

/// `argmax_decode` / `argmin_decode` map the per-row arg through `classes_`.
#[test]
fn argmax_argmin_decode() {
    // 2 rows × 3 classes; classes_ = [10, 20, 30].
    let joint_ll = [
        0.1, 0.9, 0.2, // row 0: argmax=1 (→20), argmin=0 (→10)
        0.5, 0.4, 0.7, // row 1: argmax=2 (→30), argmin=1 (→20)
    ];
    let classes_ = [10i64, 20, 30];
    assert_eq!(argmax_decode(&joint_ll, &classes_), vec![20, 30]);
    assert_eq!(argmin_decode(&joint_ll, &classes_), vec![10, 20]);
}

/// Lowest-index tie-break for argmax/argmin (sklearn / reduce-prim convention).
#[test]
fn decode_tie_break_lowest_index() {
    // Row with a tie at the max: indices 0 and 2 both 0.9.
    let joint_ll = [0.9f64, 0.1, 0.9];
    let classes_ = [10i64, 20, 30];
    // argmax tie → lowest index 0 → 10.
    assert_eq!(argmax_decode(&joint_ll, &classes_), vec![10]);
    // argmin tie (only one min here) → index 1 → 20.
    assert_eq!(argmin_decode(&joint_ll, &classes_), vec![20]);
}

/// `accuracy_score`: `[1,1,0]` vs `[1,0,0]` → 2/3.
#[test]
fn accuracy_score_fraction() {
    let acc = accuracy_score(&[1, 1, 0], &[1, 0, 0]);
    assert!((acc - 2.0 / 3.0).abs() < 1e-12, "acc = {acc}");
    // Perfect match → 1.0; all wrong → 0.0.
    assert!((accuracy_score(&[5, 6], &[5, 6]) - 1.0).abs() < 1e-12);
    assert!(accuracy_score(&[1, 2], &[3, 4]).abs() < 1e-12);
}

/// TASK-03 (Plan-Check Issue 8): `nb_common::accuracy_score`'s delegation to
/// `metrics::classification::accuracy_score` computes empty input as
/// `weighted_correct/weighted_total = 0.0/0.0`, which is IEEE-754 `NaN` —
/// the SAME documented empty-input contract as before the delegation
/// (`nb_common.rs`'s doc-comment), now locked in by a dedicated regression
/// assertion so a future refactor of the shared `accuracy_score` cannot
/// silently change empty-input behavior to `0.0` or a panic.
#[test]
fn nb_common_accuracy_score_empty_input_is_nan() {
    assert!(accuracy_score(&[], &[]).is_nan());
}

/// LAUNCH WITNESS: `class_grouped_sum` GATHER on a 4×2 host example with 2
/// classes, validated against a host reference. Runs under `--features cpu` (the
/// reduce-prim launch is the ROADMAP cpu-launch gate; NO new `#[cube]` kernel).
#[test]
fn class_grouped_sum_launch_witness() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // 4 rows × 2 features, row-major. Rows 0,2 are class 0; rows 1,3 are class 1.
    //   row0 = [1, 2]  (c0)
    //   row1 = [3, 4]  (c1)
    //   row2 = [5, 6]  (c0)
    //   row3 = [7, 8]  (c1)
    let x_host: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let class_of_row = [0usize, 1, 0, 1];
    let n_classes = 2;

    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);
    let got = class_grouped_sum::<f64>(&mut pool, &x_dev, (4, 2), &class_of_row, n_classes)
        .expect("class_grouped_sum GATHER launches on cpu");

    // Host reference: c0 = row0 + row2 = [6, 8]; c1 = row1 + row3 = [10, 12].
    assert_eq!(got.len(), 2, "n_classes rows");
    assert_eq!(got[0].len(), 2, "n_features cols");
    assert!((got[0][0] - 6.0).abs() < 1e-12, "c0 f0 = {}", got[0][0]);
    assert!((got[0][1] - 8.0).abs() < 1e-12, "c0 f1 = {}", got[0][1]);
    assert!((got[1][0] - 10.0).abs() < 1e-12, "c1 f0 = {}", got[1][0]);
    assert!((got[1][1] - 12.0).abs() < 1e-12, "c1 f1 = {}", got[1][1]);
}

/// LAUNCH WITNESS: `class_grouped_sumsq` (the A5 sum-of-squares GATHER) on the
/// same 4×2 example — per-axis `ScalarOp::SumSq` is exposed by the reduce prim
/// (no squared-host-copy needed).
#[test]
fn class_grouped_sumsq_launch_witness() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let class_of_row = [0usize, 1, 0, 1];

    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);
    let got = class_grouped_sumsq::<f64>(&mut pool, &x_dev, (4, 2), &class_of_row, 2)
        .expect("class_grouped_sumsq GATHER launches on cpu");

    // c0 = row0² + row2² = [1+25, 4+36] = [26, 40].
    // c1 = row1² + row3² = [9+49, 16+64] = [58, 80].
    assert!((got[0][0] - 26.0).abs() < 1e-10, "c0 f0 = {}", got[0][0]);
    assert!((got[0][1] - 40.0).abs() < 1e-10, "c0 f1 = {}", got[0][1]);
    assert!((got[1][0] - 58.0).abs() < 1e-10, "c1 f0 = {}", got[1][0]);
    assert!((got[1][1] - 80.0).abs() < 1e-10, "c1 f1 = {}", got[1][1]);
}

/// A class with NO rows contributes an all-zero row (the GATHER tolerates an
/// empty owner — no launch for that class).
#[test]
fn class_grouped_sum_empty_class() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // All 3 rows are class 0; class 1 and 2 are empty.
    let x_host: Vec<f64> = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    let class_of_row = [0usize, 0, 0];
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);
    let got = class_grouped_sum::<f64>(&mut pool, &x_dev, (3, 2), &class_of_row, 3)
        .expect("class_grouped_sum with empty classes");

    assert_eq!(got.len(), 3);
    assert!((got[0][0] - 6.0).abs() < 1e-12 && (got[0][1] - 6.0).abs() < 1e-12);
    assert_eq!(got[1], vec![0.0, 0.0], "empty class 1 → zero row");
    assert_eq!(got[2], vec![0.0, 0.0], "empty class 2 → zero row");
}
