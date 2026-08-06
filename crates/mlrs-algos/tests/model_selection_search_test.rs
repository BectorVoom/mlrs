//! `ParameterGrid` / `ParameterSampler` / search-schedule / aggregation /
//! threshold-tuning gate (MODSEL-RS-04..07).
//!
//! These are the parts of the surface that carry no `.npz` fixture, because
//! what they produce is a *schedule* or a *ranking* rather than a data array.
//! The expected values are literals taken from a live scikit-learn 1.9 — each
//! test names the call that produced them so a future sklearn change is a
//! one-line re-derivation rather than an archaeology exercise.

use mlrs_algos::model_selection::param::{sample_parameter_grid, GridSpec, ParameterGrid};
use mlrs_algos::model_selection::rng::NumpyRandomState;
use mlrs_algos::model_selection::search::{
    best_index, evaluate_candidates, exhaust_n_candidates, halving_schedule, run_halving, top_k,
    HalvingParams, MinResources,
};
use mlrs_algos::model_selection::threshold::{
    apply_threshold, interp, linspace, tune_threshold, FoldCurve, ThresholdGrid,
};
use mlrs_algos::model_selection::validate::{
    check_is_partition, partition_inverse, permutation_pvalue, rank_scores, summarize_scores,
    translate_train_sizes, TrainSizes,
};

// ==================== ParameterGrid ====================

/// The reference enumeration, from
/// `list(ParameterGrid([{"a":[1,2,3],"b":["x","y"]},{"c":[0.1,0.2]}]))`:
/// `a=1 b=x, a=1 b=y, a=2 b=x, a=2 b=y, a=3 b=x, a=3 b=y, c=0.1, c=0.2`.
fn two_grid() -> ParameterGrid {
    ParameterGrid::new(vec![
        GridSpec::new([("a", 3), ("b", 2)]).expect("non-empty"),
        GridSpec::new([("c", 2)]).expect("non-empty"),
    ])
}

#[test]
fn parameter_grid_length_sums_the_sub_grids() {
    assert_eq!(two_grid().len(), 8);
}

#[test]
fn parameter_grid_enumeration_matches_sklearn_order() {
    // The LAST key alphabetically varies fastest, so `b` cycles inside `a`.
    let grid = two_grid();
    let got: Vec<(usize, Vec<usize>)> = grid.iter().map(|c| (c.grid, c.value_indices)).collect();
    assert_eq!(
        got,
        vec![
            (0, vec![0, 0]),
            (0, vec![0, 1]),
            (0, vec![1, 0]),
            (0, vec![1, 1]),
            (0, vec![2, 0]),
            (0, vec![2, 1]),
            (1, vec![0]),
            (1, vec![1]),
        ]
    );
}

#[test]
fn parameter_grid_nth_agrees_with_iteration() {
    // sklearn's `__getitem__` peels the keys in DESCENDING order while
    // `__iter__` uses `product` over ascending keys; the two must land on the
    // same candidate, and `ParameterSampler` depends on it (it samples indices
    // and looks them up with `__getitem__`).
    let grid = two_grid();
    for (i, want) in grid.iter().enumerate() {
        assert_eq!(grid.nth(i).expect("in range"), want, "candidate {i}");
    }
    assert!(grid.nth(8).is_none(), "past the end must be None");
}

#[test]
fn parameter_grid_keyless_sub_grid_yields_one_candidate() {
    // `ParameterGrid([{}])` has length 1 — the empty parameter dict — not 0.
    let grid = ParameterGrid::new(vec![
        GridSpec::new(Vec::<(String, usize)>::new()).expect("ok")
    ]);
    assert_eq!(grid.len(), 1);
    assert_eq!(
        grid.nth(0).expect("in range").value_indices,
        Vec::<usize>::new()
    );
}

#[test]
fn parameter_grid_rejects_an_empty_value_list() {
    let err = GridSpec::new([("a", 0usize)]).expect_err("empty value lists are rejected");
    assert!(err.to_string().contains("non-empty sequence"));
}

// ==================== ParameterSampler ====================

#[test]
fn parameter_sampler_matches_sklearn_draw() {
    // list(ParameterSampler({"a":[1,2,3],"b":["x","y"],"c":[0,1,2,3]},
    //                       n_iter=5, random_state=42))
    // maps to grid indices [8, 16, 0, 18, 11].
    let grid = ParameterGrid::new(vec![
        GridSpec::new([("a", 3), ("b", 2), ("c", 4)]).expect("non-empty")
    ]);
    assert_eq!(grid.len(), 24);
    let mut rng = NumpyRandomState::from_seed(42);
    let sampled = sample_parameter_grid(&grid, 5, &mut rng);
    assert_eq!(sampled.indices, vec![8, 16, 0, 18, 11]);
    assert!(sampled.warning.is_none());
}

#[test]
fn parameter_sampler_clamps_to_the_grid_size_with_a_warning() {
    let grid = ParameterGrid::new(vec![GridSpec::new([("a", 3)]).expect("non-empty")]);
    let mut rng = NumpyRandomState::from_seed(0);
    let sampled = sample_parameter_grid(&grid, 10, &mut rng);
    assert_eq!(sampled.indices.len(), 3);
    assert!(sampled
        .warning
        .expect("sklearn warns here")
        .contains("smaller than n_iter=10"));
}

// ==================== score aggregation ====================

#[test]
fn rank_scores_matches_scipy_rankdata_min() {
    // scipy.stats.rankdata(-[0.5, 0.9, 0.9, 0.1], method="min") -> [3, 1, 1, 4]
    assert_eq!(rank_scores(&[0.5, 0.9, 0.9, 0.1]), vec![3, 1, 1, 4]);
}

#[test]
fn rank_scores_puts_nan_last_without_dropping_it() {
    // A failed fold must still produce a rank — sklearn replaces the NaN with
    // `nanmin - 1` rather than removing the candidate from `cv_results_`.
    let ranks = rank_scores(&[0.5, f64::NAN, 0.9]);
    assert_eq!(ranks, vec![2, 3, 1]);
}

#[test]
fn summarize_scores_uses_a_population_std() {
    // rows: candidate 0 = [1, 2, 3] -> mean 2, population std sqrt(2/3);
    //       candidate 1 = [4, 4, 4] -> mean 4, std 0.
    let summary = summarize_scores(&[1.0, 2.0, 3.0, 4.0, 4.0, 4.0], 2, 3).expect("shape ok");
    assert_eq!(summary.mean, vec![2.0, 4.0]);
    assert!((summary.std[0] - (2.0f64 / 3.0).sqrt()).abs() < 1e-15);
    assert_eq!(summary.std[1], 0.0);
    assert_eq!(summary.rank, vec![2, 1]);
}

#[test]
fn summarize_scores_rejects_a_mis_shaped_matrix() {
    assert!(summarize_scores(&[1.0, 2.0, 3.0], 2, 3).is_err());
}

// ==================== search driver ====================

#[test]
fn evaluate_candidates_visits_candidate_major() {
    let mut order = Vec::new();
    let out = evaluate_candidates(&[0, 1, 2], 2, |c, s| {
        order.push((c, s));
        c as f64 + s as f64 * 0.1
    })
    .expect("valid");
    assert_eq!(
        order,
        vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        "sklearn iterates candidates outermost"
    );
    assert_eq!(out.best, 2);
}

#[test]
fn best_index_ignores_nan_and_keeps_the_first_maximum() {
    assert_eq!(best_index(&[0.1, f64::NAN, 0.9, 0.9]), 2);
    assert_eq!(best_index(&[f64::NAN, f64::NAN]), 0);
}

#[test]
fn top_k_keeps_the_highest_scores_and_drops_nan_first() {
    // np.roll(np.argsort([0.1, nan, 0.9, 0.5]), 1)[-2:] -> the 0.9 and 0.5 rows
    let survivors = top_k(&[0.1, f64::NAN, 0.9, 0.5], 2);
    let mut sorted = survivors.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![2, 3]);
    assert_eq!(
        *survivors.last().expect("non-empty"),
        2,
        "argsort order is worst-to-best, so the winner is last"
    );
}

// ==================== successive halving ====================

/// The four reference schedules, each from a live
/// `HalvingGridSearchCV(Ridge(), {"alpha": [7 values]}, cv=3).fit(X, y)` on a
/// 1000-row regression problem (so `smallest_resources = n_splits * 2 = 6`).
#[test]
fn halving_schedule_matches_sklearn() {
    let base = HalvingParams {
        factor: 3,
        min_resources: MinResources::Exhaust,
        max_resources: 1000,
        aggressive_elimination: false,
        smallest_resources: 6,
    };

    // factor=3: min 333, required 2, possible 2, resources [333, 999]
    let s = halving_schedule(base, 7).expect("valid");
    assert_eq!(s.min_resources, 333);
    assert_eq!((s.n_required_iterations, s.n_possible_iterations), (2, 2));
    assert_eq!(
        s.iterations
            .iter()
            .map(|i| i.n_resources)
            .collect::<Vec<_>>(),
        vec![333, 999]
    );
    assert_eq!(
        s.iterations
            .iter()
            .map(|i| i.n_candidates)
            .collect::<Vec<_>>(),
        vec![7, 3]
    );

    // factor=2: min 250, required 3, possible 3, resources [250, 500, 1000]
    let s = halving_schedule(HalvingParams { factor: 2, ..base }, 7).expect("valid");
    assert_eq!(s.min_resources, 250);
    assert_eq!((s.n_required_iterations, s.n_possible_iterations), (3, 3));
    assert_eq!(
        s.iterations
            .iter()
            .map(|i| i.n_resources)
            .collect::<Vec<_>>(),
        vec![250, 500, 1000]
    );
    assert_eq!(
        s.iterations
            .iter()
            .map(|i| i.n_candidates)
            .collect::<Vec<_>>(),
        vec![7, 4, 2]
    );

    // aggressive_elimination + max_resources=200: min 66, resources [66, 198]
    let s = halving_schedule(
        HalvingParams {
            max_resources: 200,
            aggressive_elimination: true,
            ..base
        },
        7,
    )
    .expect("valid");
    assert_eq!(s.min_resources, 66);
    assert_eq!(
        s.iterations
            .iter()
            .map(|i| i.n_resources)
            .collect::<Vec<_>>(),
        vec![66, 198]
    );

    // explicit min_resources=20: possible 4 but required 2, so 2 rounds run
    let s = halving_schedule(
        HalvingParams {
            min_resources: MinResources::Fixed(20),
            ..base
        },
        7,
    )
    .expect("valid");
    assert_eq!(s.min_resources, 20);
    assert_eq!((s.n_required_iterations, s.n_possible_iterations), (2, 4));
    assert_eq!(
        s.iterations
            .iter()
            .map(|i| i.n_resources)
            .collect::<Vec<_>>(),
        vec![20, 60]
    );
}

#[test]
fn halving_rejects_a_min_above_max() {
    let err = halving_schedule(
        HalvingParams {
            factor: 3,
            min_resources: MinResources::Fixed(5000),
            max_resources: 1000,
            aggressive_elimination: false,
            smallest_resources: 6,
        },
        7,
    )
    .expect_err("min above max is a sklearn ValueError");
    assert!(err.to_string().contains("is greater than max_resources_"));
}

#[test]
fn exhaust_n_candidates_fills_the_budget() {
    // HalvingRandomSearchCV(n_candidates="exhaust") draws max // min.
    assert_eq!(exhaust_n_candidates(1000, 6), 166);
    assert_eq!(exhaust_n_candidates(5, 10), 1, "never zero candidates");
}

#[test]
fn run_halving_narrows_to_the_best_candidate() {
    // A synthetic evaluator where candidate 4 is strictly best at every
    // resource level: the run must end holding it.
    let params = HalvingParams {
        factor: 3,
        min_resources: MinResources::Fixed(10),
        max_resources: 270,
        aggressive_elimination: false,
        smallest_resources: 6,
    };
    let rounds = run_halving(params, 9, 3, |candidate, _split, _n| {
        if candidate == 4 {
            1.0
        } else {
            candidate as f64 * 0.01
        }
    })
    .expect("valid");
    assert_eq!(rounds.len(), 3);
    assert_eq!(
        rounds
            .iter()
            .map(|r| r.candidates.len())
            .collect::<Vec<_>>(),
        vec![9, 3, 1]
    );
    let last = rounds.last().expect("non-empty");
    assert_eq!(last.candidates, vec![4]);
}

// ==================== curve schedules ====================

#[test]
fn translate_train_sizes_fractions_match_sklearn() {
    // _translate_train_sizes([0.1, 0.325, 0.55, 0.775, 1.0], 100) — the
    // `np.linspace(0.1, 1.0, 5)` default learning_curve ticks.
    let (sizes, warning) = translate_train_sizes(
        &TrainSizes::Fractions(vec![0.1, 0.325, 0.55, 0.775, 1.0]),
        100,
    )
    .expect("valid");
    assert_eq!(sizes, vec![10, 32, 55, 77, 100]);
    assert!(warning.is_none());
}

#[test]
fn translate_train_sizes_clips_a_tiny_fraction_up_to_one_row() {
    let (sizes, _) = translate_train_sizes(&TrainSizes::Fractions(vec![0.001]), 10).expect("valid");
    assert_eq!(sizes, vec![1], "0.01 rows truncates to 0, then clips to 1");
}

#[test]
fn translate_train_sizes_warns_when_ticks_collapse() {
    // 0.11 and 0.12 of 10 rows both truncate to 1 -> one tick, not two.
    let (sizes, warning) =
        translate_train_sizes(&TrainSizes::Fractions(vec![0.11, 0.12]), 10).expect("valid");
    assert_eq!(sizes, vec![1]);
    assert!(warning
        .expect("sklearn warns")
        .contains("Removed duplicate entries"));
}

#[test]
fn translate_train_sizes_rejects_out_of_range() {
    assert!(translate_train_sizes(&TrainSizes::Fractions(vec![0.5, 1.5]), 100).is_err());
    assert!(translate_train_sizes(&TrainSizes::Absolute(vec![50, 200]), 100).is_err());
    assert!(translate_train_sizes(&TrainSizes::Absolute(vec![0, 50]), 100).is_err());
}

// ==================== permutation test ====================

#[test]
fn permutation_pvalue_matches_the_documented_formula() {
    // 2 of 9 permutations beat the true score -> (2 + 1) / (9 + 1)
    let perms = vec![0.9, 0.95, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
    assert!((permutation_pvalue(0.9, &perms) - 0.3).abs() < 1e-12);
}

#[test]
fn permutation_pvalue_is_never_zero() {
    let perms = vec![0.0; 99];
    assert!((permutation_pvalue(1.0, &perms) - 0.01).abs() < 1e-12);
}

// ==================== cross_val_predict partition ====================

#[test]
fn partition_check_accepts_a_kfold_and_rejects_a_shuffle_split() {
    assert!(check_is_partition(&[vec![0, 1], vec![2, 3]], 4).is_ok());
    // overlapping test sets — a ShuffleSplit
    let err = check_is_partition(&[vec![0, 1], vec![1, 2]], 4).expect_err("not a partition");
    assert!(err.to_string().contains("only works for partitions"));
    // an uncovered row
    assert!(check_is_partition(&[vec![0, 1]], 4).is_err());
}

#[test]
fn partition_inverse_scatters_fold_order_back_to_row_order() {
    // Folds test rows [2, 0] then [1]; the concatenated prediction buffer is
    // therefore [row2, row0, row1], so row 0 is at position 1.
    let inverse = partition_inverse(&[vec![2, 0], vec![1]], 3).expect("a partition");
    assert_eq!(inverse, vec![1, 2, 0]);
}

// ==================== decision-threshold tuning ====================

#[test]
fn apply_threshold_treats_an_exact_tie_as_positive() {
    assert_eq!(apply_threshold(&[0.4, 0.5, 0.6], 0.5), vec![0, 1, 1]);
}

#[test]
fn interp_clamps_outside_the_known_range() {
    // np.interp([-1, 0.5, 5], [0, 1, 2], [10, 20, 30]) -> [10, 15, 30]
    let got = interp(&[-1.0, 0.5, 5.0], &[0.0, 1.0, 2.0], &[10.0, 20.0, 30.0]).expect("valid");
    assert_eq!(got, vec![10.0, 15.0, 30.0]);
}

#[test]
fn linspace_hits_both_endpoints_exactly() {
    let got = linspace(0.0, 1.0, 5);
    assert_eq!(got, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    assert_eq!(linspace(2.0, 3.0, 1), vec![2.0]);
}

#[test]
fn tune_threshold_picks_the_best_interpolated_mean() {
    // Two folds whose objective peaks at 0.5 and 0.6; interpolated onto a
    // common grid the mean peaks between them.
    let folds = vec![
        FoldCurve {
            thresholds: vec![0.0, 0.5, 1.0],
            scores: vec![0.0, 1.0, 0.0],
        },
        FoldCurve {
            thresholds: vec![0.0, 0.6, 1.0],
            scores: vec![0.0, 1.0, 0.0],
        },
    ];
    let tuned = tune_threshold(&folds, &ThresholdGrid::Count(11)).expect("valid");
    assert_eq!(tuned.thresholds.len(), 11);
    assert!(
        (tuned.best_threshold - 0.5).abs() < 1e-12 || (tuned.best_threshold - 0.6).abs() < 1e-12,
        "peak landed at {}",
        tuned.best_threshold
    );
    assert!(tuned.best_score > 0.9);
}

#[test]
fn tune_threshold_rejects_a_constant_classifier() {
    let folds = vec![FoldCurve {
        thresholds: vec![0.5, 0.5],
        scores: vec![1.0, 1.0],
    }];
    let err = tune_threshold(&folds, &ThresholdGrid::Count(5))
        .expect_err("a constant score curve cannot be optimized");
    assert!(err.to_string().contains("constant predictions"));
}

#[test]
fn tune_threshold_honors_an_explicit_grid() {
    let folds = vec![FoldCurve {
        thresholds: vec![0.0, 1.0],
        scores: vec![0.0, 1.0],
    }];
    let tuned = tune_threshold(&folds, &ThresholdGrid::Explicit(vec![0.2, 0.8])).expect("valid");
    assert_eq!(tuned.thresholds, vec![0.2, 0.8]);
    assert_eq!(tuned.best_threshold, 0.8);
}
