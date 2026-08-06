//! The `docs/model-selection.md` Rust-native walkthrough, compiled.
//!
//! Documentation that does not compile is worse than none: a reader who copies
//! a snippet and hits a type error concludes the whole API is stale. This file
//! is that document's §4 end to end — factorize, split, gather, search — so the
//! examples cannot rot silently.
//!
//! Run the container half with the optional adapters enabled:
//!
//! ```text
//! cargo test -p mlrs-algos --features cpu,ndarray --test model_selection_doc_example_test
//! ```

use mlrs_algos::model_selection::container::{take_split, RowMajor};
use mlrs_algos::model_selection::search::{
    evaluate_candidates, run_halving, HalvingParams, MinResources,
};
use mlrs_algos::model_selection::split::{train_test_split_indices, StratifiedKFold};
use mlrs_algos::model_selection::{factorize, NumpyRandomState, RandomStateSpec, SizeSpec};

/// 60 rows of 3 features plus a 3-class string target.
fn design() -> (Vec<f64>, Vec<&'static str>) {
    let labels = ["setosa", "versicolor", "virginica"];
    let y: Vec<&str> = (0..60).map(|i| labels[i % 3]).collect();
    let x: Vec<f64> = (0..60 * 3).map(|i| (i % 7) as f64 * 0.5).collect();
    (x, y)
}

#[test]
fn doc_walkthrough_splits_gathers_and_searches() {
    let (values, labels) = design();
    let n_rows = labels.len();

    // --- factorize + split ------------------------------------------------
    let (classes, y_codes) = factorize(&labels);
    assert_eq!(classes.len(), 3);

    let folds = StratifiedKFold {
        n_splits: 5,
        shuffle: true,
        random_state: RandomStateSpec::Seed(42),
    }
    .split(&y_codes)
    .expect("5 folds over 20 members per class");
    assert_eq!(folds.splits.len(), 5);
    assert!(folds.warnings.is_empty());

    let mut rng = NumpyRandomState::from_seed(0);
    let holdout = train_test_split_indices(
        n_rows,
        SizeSpec::Float(0.25),
        SizeSpec::None,
        true,
        Some(&y_codes),
        &mut rng,
    )
    .expect("a 25% stratified holdout");
    assert_eq!(holdout.test.len(), 15);
    assert_eq!(holdout.train.len(), 45);

    // --- gather -----------------------------------------------------------
    let matrix = RowMajor {
        data: &values,
        n_cols: 3,
    };
    let (x_train, x_test) = take_split(&matrix, &holdout);
    assert_eq!(x_train.len(), 45 * 3);
    assert_eq!(x_test.len(), 15 * 3);

    // --- search -----------------------------------------------------------
    // A stand-in evaluator: candidate 2 is best on every split. A real caller
    // fits an estimator here; the driver only needs a number back.
    let candidates: Vec<usize> = (0..5).collect();
    let results = evaluate_candidates(&candidates, folds.splits.len(), |candidate, _split| {
        if candidate == 2 {
            1.0
        } else {
            0.1 * candidate as f64
        }
    })
    .expect("valid");
    assert_eq!(results.candidates[results.best], 2);
    assert_eq!(results.summary.rank[2], 1);

    // --- successive halving ------------------------------------------------
    let rounds = run_halving(
        HalvingParams {
            factor: 3,
            min_resources: MinResources::Exhaust,
            max_resources: n_rows,
            aggressive_elimination: false,
            smallest_resources: folds.splits.len() * 2,
        },
        9,
        folds.splits.len(),
        |candidate, _split, _n_resources| if candidate == 4 { 1.0 } else { 0.0 },
    )
    .expect("valid");
    let last = rounds.last().expect("at least one round");
    assert!(last.candidates.contains(&4), "the winner survived to the end");
}

#[cfg(feature = "ndarray")]
#[test]
fn doc_walkthrough_ndarray_gather() {
    use ndarray::Array2;

    let (values, labels) = design();
    let x = Array2::from_shape_vec((labels.len(), 3), values).expect("60x3");
    let (_, y_codes) = factorize(&labels);

    let mut rng = NumpyRandomState::from_seed(0);
    let holdout = train_test_split_indices(
        labels.len(),
        SizeSpec::Float(0.25),
        SizeSpec::None,
        true,
        Some(&y_codes),
        &mut rng,
    )
    .expect("valid");

    let (x_train, x_test) = take_split(&x, &holdout);
    assert_eq!(x_train.nrows(), 45);
    assert_eq!(x_test.nrows(), 15);
    assert_eq!(x_train.ncols(), 3);
}
