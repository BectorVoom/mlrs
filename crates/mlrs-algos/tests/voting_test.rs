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

// --------------------------------------------------------------------------- //
// VotingClassifier (VOTE-CLF-01)
// --------------------------------------------------------------------------- //

use mlrs_algos::ensemble::voting::{
    argmax_rows, check_feature_names_supported, classifier_feature_names, hard_vote_labels,
    hstack_blocks, voting_mode, Voting,
};

#[test]
fn the_two_voting_spellings_round_trip() {
    assert_eq!(voting_mode("hard").unwrap(), Voting::Hard);
    assert_eq!(voting_mode("soft").unwrap(), Voting::Soft);
    assert_eq!(voting_mode("hard").unwrap().as_str(), "hard");
    assert_eq!(voting_mode("soft").unwrap().as_str(), "soft");
}

#[test]
fn an_unrecognized_voting_value_reports_sklearns_constraint_message() {
    let msg = value_message(voting_mode("majority").unwrap_err());
    // The OPTION SET is asserted, not the whole string's ordering: sklearn
    // renders it from a Python `set` whose order moves with `PYTHONHASHSEED`
    // (the python oracle parses both sides for exactly this reason).
    assert!(msg.starts_with("The 'voting' parameter of VotingClassifier must be a str among "));
    assert!(msg.contains("'hard'"));
    assert!(msg.contains("'soft'"));
    assert!(msg.ends_with("Got 'majority' instead."));
    // Case matters — sklearn's `StrOptions` is a plain set membership test.
    assert!(voting_mode("Hard").is_err());
    assert!(voting_mode("").is_err());
}

#[test]
fn hard_voting_feature_names_are_one_per_member() {
    let kept = vec!["lr".to_string(), "nb".to_string()];
    assert_eq!(
        classifier_feature_names("votingclassifier", &kept, Voting::Hard, 3),
        vec!["votingclassifier_lr", "votingclassifier_nb"]
    );
}

#[test]
fn soft_voting_feature_names_append_the_class_index_without_a_separator() {
    let kept = vec!["lr".to_string(), "nb".to_string()];
    // Member-major, matching `np.hstack(probas)`, and `lr0` not `lr_0` — sklearn
    // writes `f"{class_name}_{name}{i}"`, and a caller reading these back into a
    // DataFrame is matching on the exact string.
    assert_eq!(
        classifier_feature_names("votingclassifier", &kept, Voting::Soft, 3),
        vec![
            "votingclassifier_lr0",
            "votingclassifier_lr1",
            "votingclassifier_lr2",
            "votingclassifier_nb0",
            "votingclassifier_nb1",
            "votingclassifier_nb2",
        ]
    );
}

#[test]
fn only_soft_voting_without_a_flattened_transform_has_no_feature_names() {
    assert!(check_feature_names_supported(Voting::Hard, true).is_ok());
    assert!(check_feature_names_supported(Voting::Hard, false).is_ok());
    assert!(check_feature_names_supported(Voting::Soft, true).is_ok());
    let msg = value_message(check_feature_names_supported(Voting::Soft, false).unwrap_err());
    assert_eq!(
        msg,
        "get_feature_names_out is not supported when `voting='soft'` and \
         `flatten_transform=False`"
    );
}

#[test]
fn uniform_hard_voting_is_a_plain_majority_with_the_lowest_index_winning_a_tie() {
    // Rows: [0,0,1] -> 0 wins 2-1; [0,1,2] -> a three-way tie, first index wins;
    // [2,2,2] -> unanimous; [1,2,1] -> 1 wins.
    let a = [0u32, 0, 2, 1];
    let b = [0u32, 1, 2, 2];
    let c = [1u32, 2, 2, 1];
    let got = hard_vote_labels(&[&a, &b, &c], None, 4, 3).unwrap();
    assert_eq!(got, vec![0, 0, 2, 1]);
}

#[test]
fn weights_change_which_label_wins_a_hard_vote() {
    // One member votes 1, two vote 0 — but the dissenter carries weight 5.
    let a = [1u32];
    let b = [0u32];
    let c = [0u32];
    assert_eq!(
        hard_vote_labels(&[&a, &b, &c], None, 1, 2).unwrap(),
        vec![0]
    );
    assert_eq!(
        hard_vote_labels(&[&a, &b, &c], Some(&[5.0, 1.0, 1.0]), 1, 2).unwrap(),
        vec![1]
    );
}

#[test]
fn a_hard_vote_never_looks_above_the_rows_own_largest_label() {
    // `np.bincount(x, weights=w)` is `x.max() + 1` long, so class 2 is not a
    // candidate for a row whose members all voted 0 — even though its implicit
    // count of 0.0 would beat these NEGATIVE weights. Reproducing that is the
    // whole reason the argmax is bounded per row; a full-width tally answers 1.
    let a = [0u32];
    let b = [0u32];
    let got = hard_vote_labels(&[&a, &b], Some(&[-1.0, -2.0]), 1, 3).unwrap();
    assert_eq!(got, vec![0]);
}

#[test]
fn the_hard_vote_tally_is_reset_between_rows() {
    // A regression guard for the scratch tally: row 0 votes entirely for class
    // 2, row 1 entirely for class 0. If the tally leaked, row 1 would still see
    // class 2's three votes and answer 2.
    let a = [2u32, 0];
    let b = [2u32, 0];
    let c = [2u32, 0];
    assert_eq!(
        hard_vote_labels(&[&a, &b, &c], None, 2, 3).unwrap(),
        vec![2, 0]
    );
}

#[test]
fn a_label_outside_the_declared_class_count_is_rejected_before_any_launch() {
    let a = [0u32, 7];
    let msg = value_message(hard_vote_labels(&[&a], None, 2, 3).unwrap_err());
    assert!(msg.contains("label 7"), "{msg}");
    // The device arm cannot check this for itself — `counts[r * n_bins + label]`
    // would land in the next row's tally — so the host validator is the only
    // gate, and it must run on both arms.
    assert!(hard_vote_labels(&[&a], None, 2, 8).is_ok());
}

#[test]
fn argmax_rows_takes_the_first_maximum() {
    let mat = [0.1f64, 0.5, 0.4, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0];
    assert_eq!(argmax_rows(&mat, 3, 3).unwrap(), vec![1, 0, 2]);
}

#[test]
fn hstack_blocks_lays_the_members_out_side_by_side() {
    // Two members, two rows, three classes.
    let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0];
    let got = hstack_blocks(&[&a, &b], 2, 3).unwrap();
    assert_eq!(
        got,
        vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0]
    );
}

#[test]
fn soft_voting_is_the_regressors_reduction_over_a_flattened_block() {
    // The claim `mlrs_algos::ensemble::voting::vote_soft_proba` rests on: an
    // `(n, C)` average over `k` members IS `weighted_average` with `n * C`
    // elements per column. Asserted bit-exactly, because that identity is what
    // lets soft voting inherit the regressor's numpy-parity guarantee.
    let a = [0.1f64, 0.9, 0.8, 0.2];
    let b = [0.5f64, 0.5, 0.4, 0.6];
    let w = [3.0f64, 1.0];
    let flat = weighted_average(&[&a, &b], Some(&w), 4).unwrap();
    let expected: Vec<f64> = (0..4)
        .map(|i| (a[i] * w[0] + b[i] * w[1]) / (w[0] + w[1]))
        .collect();
    assert_eq!(flat, expected);
    assert_eq!(argmax_rows(&flat, 2, 2).unwrap(), vec![1, 0]);
}
