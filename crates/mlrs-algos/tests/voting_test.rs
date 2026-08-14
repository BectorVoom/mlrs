//! `mlrs_algos::ensemble::voting` — the structural rules and the host
//! aggregation behind `mlrs.VotingRegressor` (VOTE-01).
//!
//! Every rule here is observable from Python as an exception message, a
//! `get_feature_names_out()` string, or a predicted value, so the assertions
//! are on exact texts and exact values rather than on shapes.
//!
//! The aggregation half is asserted BIT-EXACTLY against a reference computed
//! here in the order numpy uses (`Σ predⱼ·wⱼ`, left to right, then a DIVISION by
//! `Σ wⱼ`). A tolerance would hide the two regressions this code can actually
//! have — an accumulation reassociated, or a reciprocal-multiply substituted for
//! the division — and both of those are exactly what the Python oracle's
//! `array_equal` against `np.average` would then fail on with no local test to
//! localise it.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use mlrs_algos::ensemble::voting::{
    active_weight_slots, check_weights_len, stack_columns, transform_feature_names,
    weighted_average, VotingError,
};

/// The message text a `VotingError::Value` carries, for comparing against
/// sklearn's verbatim.
fn value_message(err: VotingError) -> String {
    match err {
        VotingError::Value(msg) => msg,
        other => panic!("expected a Value error, got {other:?}"),
    }
}

// --------------------------------------------------------------------------- //
// `weights` length — sklearn `_BaseVoting.fit`
// --------------------------------------------------------------------------- //

#[test]
fn matching_weight_and_estimator_counts_are_accepted() {
    assert!(check_weights_len(3, 3).is_ok());
    // Zero of each is not this rule's business — an empty `estimators` is
    // rejected by `kept_indices`, and reporting it here too would give the same
    // mistake two different messages.
    assert!(check_weights_len(0, 0).is_ok());
}

#[test]
fn a_weight_count_mismatch_reports_sklearns_message_verbatim() {
    let msg = value_message(check_weights_len(2, 3).unwrap_err());
    assert_eq!(
        msg,
        "Number of `estimators` and weights must be equal; got 2 weights, 3 estimators"
    );
    // Both directions, because sklearn interpolates the counts in a fixed order
    // and a transposed format string would still read plausibly.
    let msg = value_message(check_weights_len(4, 1).unwrap_err());
    assert_eq!(
        msg,
        "Number of `estimators` and weights must be equal; got 4 weights, 1 estimators"
    );
}

// --------------------------------------------------------------------------- //
// `_weights_not_none` — the `'drop'` filter over `weights`
// --------------------------------------------------------------------------- //
//
// The rule answers with POSITIONS, not values: the shim's weights are arbitrary
// Python objects whose dtype numpy propagates into `predict`'s result, and
// carrying them through here as `f64` would erase a `float32` weight array's
// dtype. See `active_weight_slots`.

#[test]
fn dropped_entries_lose_their_weight_slots_and_the_rest_keep_order() {
    let got = active_weight_slots(4, &[false, true, false, true]).unwrap();
    assert_eq!(got, vec![0, 2]);
}

#[test]
fn nothing_dropped_keeps_every_slot() {
    let got = active_weight_slots(3, &[false, false, false]).unwrap();
    assert_eq!(got, vec![0, 1, 2]);
}

#[test]
fn weights_are_checked_against_the_full_list_not_the_kept_one() {
    // THE rule that makes `set_params(name='drop')` work on a weighted
    // ensemble: three weights and three entries is legal even though only one
    // entry survives. Filtering before checking would reject this.
    assert_eq!(
        active_weight_slots(3, &[true, true, false]).unwrap(),
        vec![2]
    );
    // …and a vector sized to the KEPT count is the error, not the accepted form.
    let msg = value_message(active_weight_slots(1, &[true, true, false]).unwrap_err());
    assert_eq!(
        msg,
        "Number of `estimators` and weights must be equal; got 1 weights, 3 estimators"
    );
}

// --------------------------------------------------------------------------- //
// `get_feature_names_out`
// --------------------------------------------------------------------------- //

#[test]
fn transform_names_are_class_underscore_name_per_kept_member() {
    let names = vec!["lr".to_string(), "rf".to_string()];
    assert_eq!(
        transform_feature_names("votingregressor", &names),
        vec!["votingregressor_lr", "votingregressor_rf"]
    );
}

#[test]
fn transform_names_carry_no_within_block_index() {
    // The difference from stacking's `meta_feature_names`, which appends an
    // index (and no separator) for a multi-column block. A regressor member
    // contributes exactly one column, so an index here would be wrong on every
    // member rather than only on some.
    let names = vec!["only".to_string()];
    assert_eq!(
        transform_feature_names("votingregressor", &names),
        vec!["votingregressor_only"]
    );
}

#[test]
fn no_kept_members_names_nothing() {
    assert!(transform_feature_names("votingregressor", &[]).is_empty());
}

// --------------------------------------------------------------------------- //
// `stack_columns` — sklearn `_BaseVoting._predict`
// --------------------------------------------------------------------------- //

#[test]
fn columns_are_stacked_into_a_row_major_n_by_k_matrix() {
    let a = [1.0f64, 2.0, 3.0];
    let b = [10.0f64, 20.0, 30.0];
    let c = [100.0f64, 200.0, 300.0];
    let got = stack_columns(&[&a[..], &b[..], &c[..]], 3).unwrap();
    assert_eq!(
        got,
        vec![1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0]
    );
}

#[test]
fn a_single_member_stacks_to_one_column() {
    let a = [1.0f32, 2.0];
    assert_eq!(stack_columns(&[&a[..]], 2).unwrap(), vec![1.0, 2.0]);
}

#[test]
fn a_short_column_is_rejected_by_name_and_position() {
    let a = [1.0f64, 2.0, 3.0];
    let b = [1.0f64, 2.0];
    let msg = value_message(stack_columns(&[&a[..], &b[..]], 3).unwrap_err());
    assert_eq!(
        msg,
        "prediction column 1 has 2 elements, expected n_rows = 3"
    );
}

#[test]
fn no_columns_at_all_reports_sklearns_all_dropped_message() {
    let empty: [&[f64]; 0] = [];
    let msg = value_message(stack_columns(&empty, 4).unwrap_err());
    assert_eq!(
        msg,
        "All estimators are dropped. At least one is required to be an estimator."
    );
}

// --------------------------------------------------------------------------- //
// `weighted_average` — sklearn `VotingRegressor.predict`'s `np.average`
// --------------------------------------------------------------------------- //

/// numpy's own evaluation order, written out: form each product, sum left to
/// right, then DIVIDE by the (also left-to-right) weight sum.
fn numpy_order_average(cols: &[&[f64]], weights: &[f64], r: usize) -> f64 {
    let mut acc = cols[0][r] * weights[0];
    for j in 1..cols.len() {
        acc += cols[j][r] * weights[j];
    }
    let mut denom = weights[0];
    for &w in &weights[1..] {
        denom += w;
    }
    acc / denom
}

#[test]
fn the_uniform_case_is_the_plain_mean() {
    let a = [1.0f64, 2.0];
    let b = [3.0f64, 6.0];
    let got = weighted_average(&[&a[..], &b[..]], None, 2).unwrap();
    assert_eq!(got, vec![2.0, 4.0]);
}

#[test]
fn weights_are_applied_in_member_order() {
    let a = [1.0f64, 1.0];
    let b = [3.0f64, 3.0];
    // (1·1 + 3·3) / 4 = 2.5, and the transposed weighting would give 1.5 — so
    // this catches a zip that pairs the weights with the wrong columns.
    let got = weighted_average(&[&a[..], &b[..]], Some(&[1.0, 3.0]), 2).unwrap();
    assert_eq!(got, vec![2.5, 2.5]);
    let got = weighted_average(&[&a[..], &b[..]], Some(&[3.0, 1.0]), 2).unwrap();
    assert_eq!(got, vec![1.5, 1.5]);
}

#[test]
fn the_result_is_bit_identical_to_numpys_evaluation_order() {
    // Values chosen so the sum is NOT exactly representable: reassociating the
    // accumulation, or replacing the division by a reciprocal-multiply, changes
    // the last bit and this assertion catches it.
    let a: Vec<f64> = (0..64).map(|i| 0.1 + i as f64 * 0.7).collect();
    let b: Vec<f64> = (0..64).map(|i| 1.0 / (i as f64 + 3.0)).collect();
    let c: Vec<f64> = (0..64).map(|i| (i as f64).sqrt() * 0.3).collect();
    let cols: Vec<&[f64]> = vec![&a, &b, &c];
    let weights = [0.3f64, 0.7, 1.9];

    let got = weighted_average(&cols, Some(&weights), 64).unwrap();
    for (r, &v) in got.iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            numpy_order_average(&cols, &weights, r).to_bits(),
            "row {r} diverged from numpy's evaluation order"
        );
    }
}

#[test]
fn f32_stays_in_f32_rather_than_accumulating_wider() {
    // A wider accumulator would be MORE accurate and still WRONG: numpy reduces
    // an f32 row in f32, and the Python oracle compares the two exactly.
    let a: Vec<f32> = (0..32).map(|i| 0.1 + i as f32 * 0.7).collect();
    let b: Vec<f32> = (0..32).map(|i| 1.0 / (i as f32 + 3.0)).collect();
    let weights = [0.3f32, 1.7];
    let got = weighted_average(&[&a[..], &b[..]], Some(&weights), 32).unwrap();
    for (r, &v) in got.iter().enumerate() {
        let expected = (a[r] * weights[0] + b[r] * weights[1]) / (weights[0] + weights[1]);
        assert_eq!(v.to_bits(), expected.to_bits(), "row {r}");
    }
}

#[test]
fn a_zero_weight_sum_is_numpys_zero_division_not_an_infinity() {
    let a = [1.0f64, 2.0];
    let b = [3.0f64, 4.0];
    let err = weighted_average(&[&a[..], &b[..]], Some(&[1.0, -1.0]), 2).unwrap_err();
    assert_eq!(err, VotingError::ZeroWeightSum);
    assert_eq!(err.to_string(), "Weights sum to zero, can't be normalized");
}

#[test]
fn negative_weights_that_do_not_cancel_are_allowed() {
    // numpy permits them, so parity does too; only the SUM being zero is an
    // error. A guard on individual weights would reject fits sklearn completes.
    let a = [2.0f64];
    let b = [4.0f64];
    let got = weighted_average(&[&a[..], &b[..]], Some(&[3.0, -1.0]), 1).unwrap();
    assert_eq!(got, vec![(2.0 * 3.0 + 4.0 * -1.0) / 2.0]);
}

#[test]
fn a_weight_vector_that_does_not_match_the_column_count_is_rejected() {
    let a = [1.0f64];
    let b = [2.0f64];
    let msg = value_message(weighted_average(&[&a[..], &b[..]], Some(&[1.0]), 1).unwrap_err());
    assert_eq!(msg, "have 2 prediction columns but 1 weights");
}

#[test]
fn zero_rows_aggregate_to_nothing_rather_than_failing() {
    let empty: [f64; 0] = [];
    assert!(weighted_average(&[&empty[..]], None, 0).unwrap().is_empty());
    assert!(stack_columns(&[&empty[..]], 0).unwrap().is_empty());
}
