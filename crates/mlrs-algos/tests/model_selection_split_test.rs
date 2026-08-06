//! Splitter oracle gate (MODSEL-RS-02) — every splitter against sklearn's own
//! indices, with no Python in the loop.
//!
//! The fixture (`tests/fixtures/model_selection_splits_seed42.npz`, written by
//! `scripts/gen_oracle.py::gen_model_selection_splits`) stores the literal
//! train/test index vectors a live scikit-learn produced for 28 splitter
//! configurations. The assertion is **exact index equality, in order** — not
//! set equality, not size equality:
//!
//! * sorting before comparing would hide the mask-based/permutation-based
//!   ordering distinction that `split.rs` documents, and a caller who zips a
//!   split against another array does observe that order;
//! * comparing sizes only would pass against any generator that consumes
//!   numpy's MT19937 stream differently, which is the single most likely way
//!   for this port to be subtly wrong.
//!
//! Regenerate with `python scripts/gen_oracle.py` (needs numpy + scikit-learn).

use mlrs_algos::model_selection::split::*;
use mlrs_algos::model_selection::{factorize, RandomStateSpec, SizeSpec, Split};
use mlrs_core::oracle::{load_npz, OracleCase};

#[test]
fn factorize_produces_the_codes_the_splitters_expect() {
    // The Rust-native entry point for a caller holding real labels. It must
    // agree with `np.unique(y, return_inverse=True)` — SORTED distinct values,
    // codes into them — because that is the encoding every splitter's class and
    // group logic assumes, and a first-appearance encoding here would quietly
    // change which rows land in which fold.
    let labels = ["gamma", "alpha", "beta", "alpha", "gamma"];
    let (classes, codes) = factorize(&labels);
    assert_eq!(classes, vec!["alpha", "beta", "gamma"]);
    assert_eq!(codes, vec![2, 0, 1, 0, 2]);

    // ...and the codes drive a splitter directly, which is the documented
    // Rust-native usage.
    let out = StratifiedKFold::new(2)
        .split(&codes)
        .expect("2 folds over classes of size 2/1/2");
    assert_eq!(out.splits.len(), 2);
}

fn fixture() -> OracleCase {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/model_selection_splits_seed42.npz"
    );
    load_npz(path).expect("model_selection oracle fixture is committed")
}

fn ints(case: &OracleCase, name: &str) -> Vec<i64> {
    case.expect_f64(name).iter().map(|&v| v as i64).collect()
}

/// Re-cut the flat `<case>__train` / `<case>__test` buffers into splits.
fn expected(case: &OracleCase, name: &str) -> Vec<Split> {
    let train = ints(case, &format!("{name}__train"));
    let train_len = ints(case, &format!("{name}__train_len"));
    let test = ints(case, &format!("{name}__test"));
    let test_len = ints(case, &format!("{name}__test_len"));
    assert_eq!(
        train_len.len(),
        test_len.len(),
        "{name}: train/test split counts disagree in the fixture"
    );
    let (mut ti, mut si) = (0usize, 0usize);
    train_len
        .iter()
        .zip(&test_len)
        .map(|(&tl, &sl)| {
            let split = Split {
                train: train[ti..ti + tl as usize].to_vec(),
                test: test[si..si + sl as usize].to_vec(),
            };
            ti += tl as usize;
            si += sl as usize;
            split
        })
        .collect()
}

/// Compare a splitter's output against the fixture, split by split.
fn assert_matches(case: &OracleCase, name: &str, got: &Splits) {
    let want = expected(case, name);
    assert_eq!(
        got.splits.len(),
        want.len(),
        "{name}: produced {} splits, sklearn produced {}",
        got.splits.len(),
        want.len()
    );
    for (i, (g, w)) in got.splits.iter().zip(&want).enumerate() {
        assert_eq!(g.test, w.test, "{name}: split {i} test indices differ");
        assert_eq!(g.train, w.train, "{name}: split {i} train indices differ");
    }
}

fn n_samples(case: &OracleCase) -> usize {
    case.expect_f64("n_samples")[0] as usize
}

// ==================== KFold ====================

#[test]
fn kfold_matches_sklearn() {
    let case = fixture();
    let n = n_samples(&case);

    assert_matches(&case, "kfold_5", &KFold::new(5).split(n).expect("valid"));
    assert_matches(&case, "kfold_7", &KFold::new(7).split(n).expect("valid"));

    let shuffled = KFold {
        n_splits: 5,
        shuffle: true,
        random_state: RandomStateSpec::Seed(42),
    };
    assert_matches(
        &case,
        "kfold_5_shuffle_42",
        &shuffled.split(n).expect("valid"),
    );

    let shuffled = KFold {
        n_splits: 3,
        shuffle: true,
        random_state: RandomStateSpec::Seed(0),
    };
    assert_matches(
        &case,
        "kfold_3_shuffle_0",
        &shuffled.split(n).expect("valid"),
    );
}

#[test]
fn kfold_shuffle_reports_ascending_indices() {
    // The mask-based family's defining property, asserted directly rather than
    // only implied by the fixture: shuffling changes WHICH rows are in a fold,
    // never the order they are reported in.
    let out = KFold {
        n_splits: 4,
        shuffle: true,
        random_state: RandomStateSpec::Seed(3),
    }
    .split(20)
    .expect("valid");
    for split in &out.splits {
        assert!(split.test.windows(2).all(|w| w[0] < w[1]));
        assert!(split.train.windows(2).all(|w| w[0] < w[1]));
    }
}

// ==================== GroupKFold ====================

#[test]
fn group_kfold_matches_sklearn() {
    let case = fixture();
    let groups = ints(&case, "groups");

    assert_matches(
        &case,
        "groupkfold_3",
        &GroupKFold::new(3).split(&groups).expect("valid"),
    );

    let shuffled = GroupKFold {
        n_splits: 3,
        shuffle: true,
        random_state: RandomStateSpec::Seed(42),
    };
    assert_matches(
        &case,
        "groupkfold_3_shuffle_42",
        &shuffled.split(&groups).expect("valid"),
    );
}

#[test]
fn group_kfold_never_splits_a_group() {
    let case = fixture();
    let groups = ints(&case, "groups");
    let out = GroupKFold::new(3).split(&groups).expect("valid");
    for split in &out.splits {
        let test_groups: std::collections::HashSet<i64> =
            split.test.iter().map(|&i| groups[i as usize]).collect();
        for &t in &split.train {
            assert!(
                !test_groups.contains(&groups[t as usize]),
                "a group leaked across the train/test boundary"
            );
        }
    }
}

// ==================== StratifiedKFold ====================

#[test]
fn stratified_kfold_matches_sklearn() {
    let case = fixture();
    let y = ints(&case, "y");

    assert_matches(
        &case,
        "stratkfold_3",
        &StratifiedKFold::new(3).split(&y).expect("valid"),
    );

    for (name, n_splits, seed) in [
        ("stratkfold_3_shuffle_42", 3usize, 42u32),
        ("stratkfold_4_shuffle_7", 4, 7),
    ] {
        let splitter = StratifiedKFold {
            n_splits,
            shuffle: true,
            random_state: RandomStateSpec::Seed(seed),
        };
        assert_matches(&case, name, &splitter.split(&y).expect("valid"));
    }
}

#[test]
fn stratified_kfold_warns_on_a_too_small_class() {
    // 3 members of class 1 against n_splits=4: sklearn warns rather than
    // failing, and the Python layer has to be able to re-emit that warning.
    let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1];
    let out = StratifiedKFold::new(4).split(&y).expect("valid");
    assert_eq!(out.splits.len(), 4);
    assert!(
        out.warnings
            .iter()
            .any(|w| w.contains("least populated class")),
        "expected the least-populated-class warning, got {:?}",
        out.warnings
    );
}

// ==================== StratifiedGroupKFold ====================

#[test]
fn stratified_group_kfold_matches_sklearn() {
    let case = fixture();
    let y = ints(&case, "y");
    let groups = ints(&case, "groups");

    assert_matches(
        &case,
        "stratgroupkfold_3",
        &StratifiedGroupKFold::new(3)
            .split(&y, &groups)
            .expect("valid"),
    );

    let shuffled = StratifiedGroupKFold {
        n_splits: 3,
        shuffle: true,
        random_state: RandomStateSpec::Seed(42),
    };
    assert_matches(
        &case,
        "stratgroupkfold_3_shuffle_42",
        &shuffled.split(&y, &groups).expect("valid"),
    );
}

// ==================== TimeSeriesSplit ====================

#[test]
fn time_series_split_matches_sklearn() {
    let case = fixture();
    let n = n_samples(&case);

    assert_matches(
        &case,
        "timeseries_5",
        &TimeSeriesSplit::new(5).split(n).expect("valid"),
    );
    assert_matches(
        &case,
        "timeseries_3_gap2",
        &TimeSeriesSplit {
            n_splits: 3,
            gap: 2,
            ..Default::default()
        }
        .split(n)
        .expect("valid"),
    );
    assert_matches(
        &case,
        "timeseries_3_max10_test5",
        &TimeSeriesSplit {
            n_splits: 3,
            max_train_size: Some(10),
            test_size: Some(5),
            ..Default::default()
        }
        .split(n)
        .expect("valid"),
    );
}

#[test]
fn time_series_split_never_trains_on_the_future() {
    let out = TimeSeriesSplit::new(4).split(50).expect("valid");
    for split in &out.splits {
        let last_train = *split.train.last().expect("non-empty train");
        let first_test = split.test[0];
        assert!(
            last_train < first_test,
            "train index {last_train} is not strictly before test index {first_test}"
        );
    }
}

// ==================== Leave-out family ====================

#[test]
fn leave_one_out_matches_sklearn() {
    let case = fixture();
    let n = n_samples(&case);
    assert_matches(&case, "loo", &LeaveOneOut.split(n).expect("valid"));
}

#[test]
fn leave_p_out_matches_sklearn() {
    let case = fixture();
    let n = n_samples(&case);
    assert_matches(&case, "lpo_2", &LeavePOut::new(2).split(n).expect("valid"));
}

#[test]
fn leave_p_out_streams_the_same_splits_it_materializes() {
    // `split_at` is what the Python layer drives (materializing C(n, p) splits
    // is not an option at realistic n), so it must agree with the eager path
    // split for split — including the lexicographic combination ORDER, which
    // is what makes an mlrs `LeavePOut` interchangeable with sklearn's inside
    // a `cross_validate`.
    let lpo = LeavePOut::new(3);
    let eager = lpo.split(9).expect("valid");
    let total = lpo.get_n_splits(9).expect("valid");
    assert_eq!(total as usize, eager.splits.len());
    for (i, want) in eager.splits.iter().enumerate() {
        assert_eq!(&lpo.split_at(9, i as u128).expect("valid"), want);
    }
}

#[test]
fn leave_one_group_out_matches_sklearn() {
    let case = fixture();
    let groups = ints(&case, "groups");
    assert_matches(
        &case,
        "logo",
        &LeaveOneGroupOut.split(&groups).expect("valid"),
    );
}

#[test]
fn leave_p_groups_out_matches_sklearn() {
    let case = fixture();
    let groups = ints(&case, "groups");
    assert_matches(
        &case,
        "lpgo_2",
        &LeavePGroupsOut::new(2).split(&groups).expect("valid"),
    );
}

// ==================== PredefinedSplit ====================

#[test]
fn predefined_split_matches_sklearn() {
    let case = fixture();
    let test_fold = ints(&case, "test_fold");
    assert_matches(
        &case,
        "predefined",
        &PredefinedSplit::new(test_fold).split().expect("valid"),
    );
}

#[test]
fn predefined_split_never_tests_the_minus_one_rows() {
    let split = PredefinedSplit::new(vec![-1, 0, 1, -1, 0]);
    assert_eq!(split.get_n_splits(), 2);
    for s in &split.split().expect("valid").splits {
        assert!(!s.test.contains(&0) && !s.test.contains(&3));
    }
}

// ==================== ShuffleSplit family ====================

#[test]
fn shuffle_split_matches_sklearn() {
    let case = fixture();
    let n = n_samples(&case);

    assert_matches(
        &case,
        "shufflesplit_42",
        &ShuffleSplit {
            random_state: RandomStateSpec::Seed(42),
            ..Default::default()
        }
        .split(n)
        .expect("valid"),
    );
    assert_matches(
        &case,
        "shufflesplit_5_t03_1",
        &ShuffleSplit {
            n_splits: 5,
            test_size: SizeSpec::Float(0.3),
            random_state: RandomStateSpec::Seed(1),
            ..Default::default()
        }
        .split(n)
        .expect("valid"),
    );
    assert_matches(
        &case,
        "shufflesplit_train_int",
        &ShuffleSplit {
            n_splits: 3,
            train_size: SizeSpec::Int(20),
            test_size: SizeSpec::Int(10),
            random_state: RandomStateSpec::Seed(5),
        }
        .split(n)
        .expect("valid"),
    );
}

#[test]
fn group_shuffle_split_matches_sklearn() {
    let case = fixture();
    let groups = ints(&case, "groups");

    assert_matches(
        &case,
        "groupshuffle_42",
        &GroupShuffleSplit {
            random_state: RandomStateSpec::Seed(42),
            ..Default::default()
        }
        .split(&groups)
        .expect("valid"),
    );
    assert_matches(
        &case,
        "groupshuffle_3_t04_2",
        &GroupShuffleSplit {
            n_splits: 3,
            test_size: SizeSpec::Float(0.4),
            random_state: RandomStateSpec::Seed(2),
            ..Default::default()
        }
        .split(&groups)
        .expect("valid"),
    );
}

#[test]
fn stratified_shuffle_split_matches_sklearn() {
    let case = fixture();
    let y = ints(&case, "y");

    assert_matches(
        &case,
        "stratshuffle_42",
        &StratifiedShuffleSplit {
            random_state: RandomStateSpec::Seed(42),
            ..Default::default()
        }
        .split(&y)
        .expect("valid"),
    );
    assert_matches(
        &case,
        "stratshuffle_5_t025_3",
        &StratifiedShuffleSplit {
            n_splits: 5,
            test_size: SizeSpec::Float(0.25),
            random_state: RandomStateSpec::Seed(3),
            ..Default::default()
        }
        .split(&y)
        .expect("valid"),
    );
}

// ==================== Repeated splitters ====================

#[test]
fn repeated_kfold_matches_sklearn() {
    let case = fixture();
    let n = n_samples(&case);
    assert_matches(
        &case,
        "repeatedkfold_3x2_42",
        &RepeatedKFold {
            n_splits: 3,
            n_repeats: 2,
            random_state: RandomStateSpec::Seed(42),
        }
        .split(n)
        .expect("valid"),
    );
}

#[test]
fn repeated_kfold_repeats_are_not_identical() {
    // The shared-generator semantics, asserted structurally: re-seeding per
    // repeat would make repeat 2 a copy of repeat 1 while still matching every
    // per-repeat invariant.
    let out = RepeatedKFold {
        n_splits: 3,
        n_repeats: 2,
        random_state: RandomStateSpec::Seed(42),
    }
    .split(30)
    .expect("valid");
    assert_eq!(out.splits.len(), 6);
    assert_ne!(out.splits[0].test, out.splits[3].test);
}

#[test]
fn repeated_stratified_kfold_matches_sklearn() {
    let case = fixture();
    let y = ints(&case, "y");
    assert_matches(
        &case,
        "repeatedstratkfold_3x2_42",
        &RepeatedStratifiedKFold {
            n_splits: 3,
            n_repeats: 2,
            random_state: RandomStateSpec::Seed(42),
        }
        .split(&y)
        .expect("valid"),
    );
}

// ==================== structural invariants (independent of sklearn) ====================

#[test]
fn every_fold_family_covers_each_row_exactly_once() {
    // A gate that a SHARED misunderstanding between mlrs and sklearn would
    // still fail: across the folds of a k-fold splitter every row is tested
    // exactly once, and no row is ever in both sides of one split.
    let case = fixture();
    let n = n_samples(&case);
    let y = ints(&case, "y");
    let groups = ints(&case, "groups");

    let families: Vec<(&str, Splits)> = vec![
        ("KFold", KFold::new(5).split(n).expect("valid")),
        (
            "GroupKFold",
            GroupKFold::new(3).split(&groups).expect("valid"),
        ),
        (
            "StratifiedKFold",
            StratifiedKFold::new(3).split(&y).expect("valid"),
        ),
        (
            "StratifiedGroupKFold",
            StratifiedGroupKFold::new(3)
                .split(&y, &groups)
                .expect("valid"),
        ),
        ("LeaveOneOut", LeaveOneOut.split(n).expect("valid")),
        (
            "LeaveOneGroupOut",
            LeaveOneGroupOut.split(&groups).expect("valid"),
        ),
    ];

    for (name, out) in families {
        let mut tested = vec![0usize; n];
        for split in &out.splits {
            let test: std::collections::HashSet<i64> = split.test.iter().copied().collect();
            for &t in &split.train {
                assert!(!test.contains(&t), "{name}: row {t} is in both sides");
            }
            assert_eq!(
                split.train.len() + split.test.len(),
                n,
                "{name}: a split does not cover every row"
            );
            for &t in &split.test {
                tested[t as usize] += 1;
            }
        }
        assert!(
            tested.iter().all(|&c| c == 1),
            "{name}: some row was tested {:?} times",
            tested.iter().max()
        );
    }
}

#[test]
fn stratified_splitters_preserve_the_class_balance() {
    let case = fixture();
    let y = ints(&case, "y");
    let overall = class_fractions(&y, &(0..y.len() as i64).collect::<Vec<_>>());
    for split in &StratifiedKFold::new(3).split(&y).expect("valid").splits {
        let test_frac = class_fractions(&y, &split.test);
        for (a, b) in overall.iter().zip(&test_frac) {
            assert!(
                (a - b).abs() < 0.12,
                "class balance drifted: {overall:?} vs {test_frac:?}"
            );
        }
    }
}

fn class_fractions(y: &[i64], rows: &[i64]) -> Vec<f64> {
    let n_classes = y.iter().max().copied().unwrap_or(0) as usize + 1;
    let mut counts = vec![0f64; n_classes];
    for &r in rows {
        counts[y[r as usize] as usize] += 1.0;
    }
    counts.iter().map(|c| c / rows.len() as f64).collect()
}

// ==================== train_test_split ====================

#[test]
fn train_test_split_unshuffled_is_a_prefix_suffix_cut() {
    let mut rng = mlrs_algos::model_selection::NumpyRandomState::from_seed(0);
    let split = train_test_split_indices(
        100,
        SizeSpec::Float(0.25),
        SizeSpec::None,
        false,
        None,
        &mut rng,
    )
    .expect("valid");
    assert_eq!(split.train, (0..75).collect::<Vec<i64>>());
    assert_eq!(split.test, (75..100).collect::<Vec<i64>>());
}

#[test]
fn train_test_split_rejects_stratify_without_shuffle() {
    let mut rng = mlrs_algos::model_selection::NumpyRandomState::from_seed(0);
    let y: Vec<i64> = (0..100).map(|i| i % 2).collect();
    let err = train_test_split_indices(
        100,
        SizeSpec::Float(0.25),
        SizeSpec::None,
        false,
        Some(&y),
        &mut rng,
    )
    .expect_err("stratify + shuffle=False is not implemented in sklearn either");
    assert!(err
        .to_string()
        .contains("not implemented for shuffle=False"));
}

#[test]
fn train_test_split_uses_its_own_default_test_size() {
    // 0.25, NOT `ShuffleSplit`'s 0.1 — `train_test_split` resolves the sizes
    // itself and passes absolute counts down, so the inner splitter's default
    // never applies.
    let mut rng = mlrs_algos::model_selection::NumpyRandomState::from_seed(0);
    let split = train_test_split_indices(80, SizeSpec::None, SizeSpec::None, true, None, &mut rng)
        .expect("valid");
    assert_eq!(split.test.len(), 20);
    assert_eq!(split.train.len(), 60);
}
